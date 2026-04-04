#!/usr/bin/env python3
# inference_hpu.py — FastFold AlphaFold inference on Intel Gaudi (Synapse 1.23.0)
#
# Modernized from the original habana/inference.py (R1.7.1) to work with:
#   - Synapse 1.23.0 / PyTorch 2.9.0
#   - Standard torch.softmax (replaces custom TPC fused_softmax)
#   - Modern habana_frameworks imports
#   - Lazy mode via PT_HPU_LAZY_MODE=1 (set automatically)
#
# Usage:
#   # With precomputed alignments (recommended for K8s):
#   python inference_hpu.py /data/input/target.fasta /data/mmcif/ \
#       --use_precomputed_alignments /data/alignments \
#       --output_dir /data/output \
#       --param_path /data/params/params_model_1.npz \
#       --model_name model_1
#
#   # Full pipeline with database search (needs ~2TB databases mounted):
#   python inference_hpu.py /data/input/target.fasta /data/mmcif/ \
#       --output_dir /data/output \
#       --uniref90_database_path /data/uniref90/uniref90.fasta \
#       --mgnify_database_path /data/mgnify/mgy_clusters.fa \
#       --bfd_database_path /data/bfd/bfd_metaclust.sorted_opt \
#       --pdb70_database_path /data/pdb70/pdb70 \
#       --uniclust30_database_path /data/uniref30/UniRef30_2021_03

import argparse
import contextlib
import os
import random
import shutil
import sys
import tempfile
import time
from datetime import date

import numpy as np
import torch

# ---- Gaudi HPU initialization ----
# Must happen before any model code.
# Lazy mode is required for compatibility with FastFold's mark_step() patterns.
os.environ.setdefault("PT_HPU_LAZY_MODE", "1")

import habana_frameworks.torch.core as htcore  # noqa: E402

# ---- FastFold imports ----
import fastfold.habana as habana  # noqa: E402
from fastfold.common import protein, residue_constants  # noqa: E402
from fastfold.config import model_config  # noqa: E402
from fastfold.data import data_pipeline, feature_pipeline, templates  # noqa: E402
from fastfold.data.parsers import parse_fasta  # noqa: E402
from fastfold.habana.distributed import init_dist  # noqa: E402
from fastfold.habana.fastnn.ops import set_chunk_size  # noqa: E402
from fastfold.habana.inject_habana import inject_habana  # noqa: E402
from fastfold.model.hub import AlphaFold  # noqa: E402
from fastfold.model.nn.triangular_multiplicative_update import (  # noqa: E402
    set_fused_triangle_multiplication,
)
from fastfold.utils.import_weights import import_jax_weights_  # noqa: E402
from fastfold.utils.tensor_utils import tensor_tree_map  # noqa: E402

# Optional: ray workflow for parallel data processing
try:
    from fastfold.workflow.template import (
        FastFoldDataWorkFlow,
        FastFoldMultimerDataWorkFlow,
    )

    HAS_WORKFLOW = True
except (ImportError, RuntimeError):
    HAS_WORKFLOW = False

# Optional: Amber relaxation
try:
    import fastfold.relax.relax as relax

    HAS_RELAX = True
except ImportError:
    HAS_RELAX = False


@contextlib.contextmanager
def temp_fasta_file(fasta_str: str):
    with tempfile.NamedTemporaryFile("w", suffix=".fasta") as fasta_file:
        fasta_file.write(fasta_str)
        fasta_file.seek(0)
        yield fasta_file.name


def add_data_args(parser: argparse.ArgumentParser):
    parser.add_argument("--uniref90_database_path", type=str, default=None)
    parser.add_argument("--mgnify_database_path", type=str, default=None)
    parser.add_argument("--pdb70_database_path", type=str, default=None)
    parser.add_argument("--uniclust30_database_path", type=str, default=None)
    parser.add_argument("--bfd_database_path", type=str, default=None)
    parser.add_argument("--pdb_seqres_database_path", type=str, default=None)
    parser.add_argument("--uniprot_database_path", type=str, default=None)
    parser.add_argument(
        "--jackhmmer_binary_path", type=str, default="/usr/bin/jackhmmer"
    )
    parser.add_argument("--hhblits_binary_path", type=str, default="/usr/bin/hhblits")
    parser.add_argument(
        "--hhsearch_binary_path", type=str, default="/usr/bin/hhsearch"
    )
    parser.add_argument("--kalign_binary_path", type=str, default="/usr/bin/kalign")
    parser.add_argument("--hmmsearch_binary_path", type=str, default="hmmsearch")
    parser.add_argument("--hmmbuild_binary_path", type=str, default="hmmbuild")
    parser.add_argument(
        "--max_template_date",
        type=str,
        default=date.today().strftime("%Y-%m-%d"),
    )
    parser.add_argument("--obsolete_pdbs_path", type=str, default=None)
    parser.add_argument("--release_dates_path", type=str, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument(
        "--enable_workflow",
        default=False,
        action="store_true",
        help="Use ray workflow for parallel MSA processing",
    )
    parser.add_argument(
        "--inplace",
        default=False,
        action="store_true",
        help="Use inplace ops to reduce memory",
    )


def inference_model(rank, world_size, result_q, batch, args):
    """Run AlphaFold inference on a single HPU."""
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Initialize Habana HPU + distributed
    habana.enable_habana()
    init_dist()

    device = torch.device("hpu")

    config = model_config(args.model_name)
    if args.chunk_size:
        config.globals.chunk_size = args.chunk_size

    if "v3" in args.param_path:
        set_fused_triangle_multiplication()

    config.globals.inplace = False
    config.globals.is_multimer = args.model_preset == "multimer"

    print(f"[HPU rank {rank}] Loading model {args.model_name}...")
    model = AlphaFold(config)
    import_jax_weights_(model, args.param_path, version=args.model_name)

    print(f"[HPU rank {rank}] Injecting Habana-optimized Evoformer...")
    model = inject_habana(model)
    model = model.eval()
    model = model.to(device=device)

    set_chunk_size(model.globals.chunk_size)

    print(f"[HPU rank {rank}] Running inference...")
    with torch.no_grad():
        batch = {k: torch.as_tensor(v).to(device=device) for k, v in batch.items()}

        t = time.perf_counter()
        out = model(batch)
        htcore.mark_step()
        elapsed = time.perf_counter() - t
        print(f"[HPU rank {rank}] Inference time: {elapsed:.2f}s")

    out = tensor_tree_map(lambda x: np.array(x.cpu()), out)
    result_q.put(out)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def save_prediction(batch, out, tag, args, config):
    """Save unrelaxed (and optionally relaxed) PDB prediction."""
    # Toss out recycling dimensions
    batch = tensor_tree_map(lambda x: np.array(x[..., -1].cpu()), batch)

    plddt = out["plddt"]
    mean_plddt = np.mean(plddt)
    print(f"Mean pLDDT: {mean_plddt:.2f}")

    plddt_b_factors = np.repeat(
        plddt[..., None], residue_constants.atom_type_num, axis=-1
    )

    unrelaxed_protein = protein.from_prediction(
        features=batch, result=out, b_factors=plddt_b_factors
    )

    # Save unrelaxed PDB
    unrelaxed_path = os.path.join(
        args.output_dir, f"{tag}_{args.model_name}_unrelaxed.pdb"
    )
    with open(unrelaxed_path, "w") as f:
        f.write(protein.to_pdb(unrelaxed_protein))
    print(f"Saved unrelaxed prediction to {unrelaxed_path}")

    # Optional: Amber relaxation
    if HAS_RELAX and not args.skip_relaxation:
        try:
            amber_relaxer = relax.AmberRelaxation(use_gpu=False, **config.relax)
            t = time.perf_counter()
            relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
            print(f"Relaxation time: {time.perf_counter() - t:.2f}s")

            relaxed_path = os.path.join(
                args.output_dir, f"{tag}_{args.model_name}_relaxed.pdb"
            )
            with open(relaxed_path, "w") as f:
                f.write(relaxed_pdb_str)
            print(f"Saved relaxed prediction to {relaxed_path}")
        except Exception as e:
            print(f"WARNING: Amber relaxation failed: {e}")
            print("Unrelaxed prediction is still available.")
    else:
        if not HAS_RELAX:
            print("NOTE: OpenMM not installed; skipping Amber relaxation.")


def inference_monomer(args):
    """Run monomer inference."""
    print("Running in monomer mode...")
    config = model_config(args.model_name)

    template_featurizer = templates.TemplateHitFeaturizer(
        mmcif_dir=args.template_mmcif_dir,
        max_template_date=args.max_template_date,
        max_hits=config.data.predict.max_templates,
        kalign_binary_path=args.kalign_binary_path,
        release_dates_path=args.release_dates_path,
        obsolete_pdbs_path=args.obsolete_pdbs_path,
    )

    use_small_bfd = args.preset == "reduced_dbs"

    data_processor = data_pipeline.DataPipeline(
        template_featurizer=template_featurizer,
    )

    output_dir_base = args.output_dir
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)

    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(sys.maxsize)

    feature_processor = feature_pipeline.FeaturePipeline(config.data)

    if args.use_precomputed_alignments is None:
        alignment_dir = os.path.join(output_dir_base, "alignments")
    else:
        alignment_dir = args.use_precomputed_alignments

    # Read input FASTA
    with open(args.fasta_path, "r") as fp:
        fasta = fp.read()
    seqs, tags = parse_fasta(fasta)
    seq, tag = seqs[0], tags[0]

    print(f"Target: {tag} (length {len(seq)})")

    fasta_path = os.path.join(args.output_dir, "tmp.fasta")
    with open(fasta_path, "w") as fp:
        fp.write(f">{tag}\n{seq}")

    # Run MSA alignment (or use precomputed)
    if args.use_precomputed_alignments is None:
        local_alignment_dir = os.path.join(alignment_dir, tag)
        if not os.path.exists(local_alignment_dir):
            os.makedirs(local_alignment_dir)

        if args.enable_workflow and HAS_WORKFLOW:
            print("Running alignment with ray workflow...")
            alignment_runner = FastFoldDataWorkFlow(
                jackhmmer_binary_path=args.jackhmmer_binary_path,
                hhblits_binary_path=args.hhblits_binary_path,
                hhsearch_binary_path=args.hhsearch_binary_path,
                uniref90_database_path=args.uniref90_database_path,
                mgnify_database_path=args.mgnify_database_path,
                bfd_database_path=args.bfd_database_path,
                uniclust30_database_path=args.uniclust30_database_path,
                pdb70_database_path=args.pdb70_database_path,
                use_small_bfd=use_small_bfd,
                no_cpus=args.cpus,
            )
            t = time.perf_counter()
            alignment_runner.run(fasta_path, alignment_dir=local_alignment_dir)
            print(f"Alignment workflow time: {time.perf_counter() - t:.2f}s")
        else:
            alignment_runner = data_pipeline.AlignmentRunner(
                jackhmmer_binary_path=args.jackhmmer_binary_path,
                hhblits_binary_path=args.hhblits_binary_path,
                hhsearch_binary_path=args.hhsearch_binary_path,
                uniref90_database_path=args.uniref90_database_path,
                mgnify_database_path=args.mgnify_database_path,
                bfd_database_path=args.bfd_database_path,
                uniclust30_database_path=args.uniclust30_database_path,
                pdb70_database_path=args.pdb70_database_path,
                use_small_bfd=use_small_bfd,
                no_cpus=args.cpus,
            )
            alignment_runner.run(fasta_path, local_alignment_dir)

    print("Generating features...")
    feature_dict = data_processor.process_fasta(
        fasta_path=fasta_path,
        alignment_dir=(
            alignment_dir
            if args.use_precomputed_alignments
            else os.path.join(alignment_dir, tag)
        ),
    )

    # Clean up temporary FASTA
    if os.path.exists(fasta_path):
        os.remove(fasta_path)

    import numpy as np
    for k,v in feature_dict.items():
        if hasattr(v, "shape"): print(f"  {k}: {v.shape} {v.dtype}")
    processed_feature_dict = feature_processor.process_features(
        feature_dict, mode="predict"
    )

    batch = processed_feature_dict

    # Launch inference on HPU(s)
    import torch.multiprocessing as mp

    manager = mp.Manager()
    result_q = manager.Queue()

    torch.multiprocessing.spawn(
        inference_model,
        nprocs=args.hpus,
        args=(args.hpus, result_q, batch, args),
    )

    out = result_q.get()
    save_prediction(batch, out, tag, args, config)


def inference_multimer(args):
    """Run multimer inference."""
    print("Running in multimer mode...")
    config = model_config(args.model_name)

    predict_max_templates = 4
    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=args.template_mmcif_dir,
        max_template_date=args.max_template_date,
        max_hits=predict_max_templates,
        kalign_binary_path=args.kalign_binary_path,
        release_dates_path=args.release_dates_path,
        obsolete_pdbs_path=args.obsolete_pdbs_path,
    )

    monomer_data_processor = data_pipeline.DataPipeline(
        template_featurizer=template_featurizer,
    )
    data_processor = data_pipeline.DataPipelineMultimer(
        monomer_data_pipeline=monomer_data_processor,
    )

    output_dir_base = args.output_dir
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)

    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(sys.maxsize)

    feature_processor = feature_pipeline.FeaturePipeline(config.data)

    if args.use_precomputed_alignments is None:
        alignment_dir = os.path.join(output_dir_base, "alignments")
    else:
        alignment_dir = args.use_precomputed_alignments

    # Read input FASTA
    fasta_path = args.fasta_path
    with open(fasta_path, "r") as fp:
        data = fp.read()

    lines = [
        l.replace("\n", "")
        for prot in data.split(">")
        for l in prot.strip().split("\n", 1)
    ][1:]
    tags, seqs = lines[::2], lines[1::2]

    # Run alignment for each chain
    if args.use_precomputed_alignments is None:
        for tag, seq in zip(tags, seqs):
            local_alignment_dir = os.path.join(alignment_dir, tag)
            if not os.path.exists(local_alignment_dir):
                os.makedirs(local_alignment_dir)
            else:
                shutil.rmtree(local_alignment_dir)
                os.makedirs(local_alignment_dir)

            chain_fasta_str = f">chain_{tag}\n{seq}\n"
            with temp_fasta_file(chain_fasta_str) as chain_fasta_path:
                if args.enable_workflow and HAS_WORKFLOW:
                    alignment_runner = FastFoldMultimerDataWorkFlow(
                        jackhmmer_binary_path=args.jackhmmer_binary_path,
                        hhblits_binary_path=args.hhblits_binary_path,
                        hmmsearch_binary_path=args.hmmsearch_binary_path,
                        hmmbuild_binary_path=args.hmmbuild_binary_path,
                        uniref90_database_path=args.uniref90_database_path,
                        mgnify_database_path=args.mgnify_database_path,
                        bfd_database_path=args.bfd_database_path,
                        uniclust30_database_path=args.uniclust30_database_path,
                        uniprot_database_path=args.uniprot_database_path,
                        pdb_seqres_database_path=args.pdb_seqres_database_path,
                        use_small_bfd=(args.bfd_database_path is None),
                        no_cpus=args.cpus,
                    )
                    t = time.perf_counter()
                    alignment_runner.run(
                        chain_fasta_path, alignment_dir=local_alignment_dir
                    )
                    print(
                        f"Alignment for {tag}: {time.perf_counter() - t:.2f}s"
                    )
                else:
                    alignment_runner = data_pipeline.AlignmentRunnerMultimer(
                        jackhmmer_binary_path=args.jackhmmer_binary_path,
                        hhblits_binary_path=args.hhblits_binary_path,
                        hmmsearch_binary_path=args.hmmsearch_binary_path,
                        hmmbuild_binary_path=args.hmmbuild_binary_path,
                        uniref90_database_path=args.uniref90_database_path,
                        mgnify_database_path=args.mgnify_database_path,
                        bfd_database_path=args.bfd_database_path,
                        uniclust30_database_path=args.uniclust30_database_path,
                        uniprot_database_path=args.uniprot_database_path,
                        pdb_seqres_database_path=args.pdb_seqres_database_path,
                        use_small_bfd=(args.bfd_database_path is None),
                        no_cpus=args.cpus,
                    )
                    alignment_runner.run(chain_fasta_path, local_alignment_dir)
            print(f"Finished alignment for {tag}")

    local_alignment_dir = alignment_dir

    feature_dict = data_processor.process_fasta(
        fasta_path=fasta_path, alignment_dir=local_alignment_dir
    )

    import numpy as np
    for k,v in feature_dict.items():
        if hasattr(v, "shape"): print(f"  {k}: {v.shape} {v.dtype}")
    processed_feature_dict = feature_processor.process_features(
        feature_dict, mode="predict", is_multimer=True
    )

    batch = processed_feature_dict

    # Launch inference on HPU(s)
    import torch.multiprocessing as mp

    manager = mp.Manager()
    result_q = manager.Queue()

    torch.multiprocessing.spawn(
        inference_model,
        nprocs=args.hpus,
        args=(args.hpus, result_q, batch, args),
    )

    out = result_q.get()
    tag = "_".join(tags)
    save_prediction(batch, out, tag, args, config)


def main():
    parser = argparse.ArgumentParser(
        description="FastFold AlphaFold inference on Intel Gaudi HPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inference with precomputed alignments (recommended for K8s):
  python inference_hpu.py /data/input/target.fasta /data/mmcif/ \\
      --use_precomputed_alignments /data/alignments/target \\
      --output_dir /data/output \\
      --param_path /data/params/params_model_1.npz

  # Full pipeline with database search:
  python inference_hpu.py /data/input/target.fasta /data/mmcif/ \\
      --output_dir /data/output \\
      --uniref90_database_path /data/uniref90/uniref90.fasta \\
      --mgnify_database_path /data/mgnify/mgy_clusters.fa \\
      --bfd_database_path /data/bfd/bfd_metaclust.sorted_opt \\
      --pdb70_database_path /data/pdb70/pdb70 \\
      --uniclust30_database_path /data/uniref30/UniRef30_2021_03

  # Multi-HPU inference (Dynamic Axial Parallelism):
  python inference_hpu.py target.fasta mmcif/ --hpus 4 --output_dir output/
""",
    )

    parser.add_argument("fasta_path", type=str, help="Path to input FASTA file")
    parser.add_argument(
        "template_mmcif_dir", type=str, help="Path to mmCIF template directory"
    )
    parser.add_argument(
        "--use_precomputed_alignments",
        type=str,
        default=None,
        help="Path to precomputed alignment directory (skips database search)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getcwd(),
        help="Output directory for PDB predictions",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="model_1",
        help="Model config: model_{1-5}, model_{1-5}_ptm, model_{1-5}_multimer",
    )
    parser.add_argument(
        "--param_path",
        type=str,
        default=None,
        help="Path to model parameters (.npz). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--cpus", type=int, default=12, help="Number of CPUs for alignment tools"
    )
    parser.add_argument(
        "--hpus",
        type=int,
        default=1,
        help="Number of Gaudi HPUs for inference (DAP parallelism)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="full_dbs",
        choices=("reduced_dbs", "full_dbs"),
    )
    parser.add_argument("--data_random_seed", type=str, default=None)
    parser.add_argument(
        "--model_preset",
        type=str,
        default="monomer",
        choices=["monomer", "multimer"],
        help="Monomer or multimer prediction mode",
    )
    parser.add_argument(
        "--skip_relaxation",
        default=False,
        action="store_true",
        help="Skip Amber relaxation (outputs unrelaxed PDB only)",
    )

    add_data_args(parser)
    args = parser.parse_args()

    if args.param_path is None:
        args.param_path = os.path.join(
            "data", "params", "params_" + args.model_name + ".npz"
        )

    print("FastFold AlphaFold inference on Gaudi HPU")
    print(f"  Model: {args.model_name}")
    print(f"  Preset: {args.model_preset}")
    print(f"  HPUs: {args.hpus}")
    print(f"  Parameters: {args.param_path}")
    print(f"  Output: {args.output_dir}")
    if args.use_precomputed_alignments:
        print(f"  Alignments: {args.use_precomputed_alignments} (precomputed)")
    print()

    if args.model_preset == "multimer":
        inference_multimer(args)
    else:
        inference_monomer(args)


if __name__ == "__main__":
    main()
