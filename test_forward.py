"""
FastFold Habana - Synthetic Forward Pass Test
Requires: PT_HPU_LAZY_MODE=1
Run: PT_HPU_LAZY_MODE=1 python test_forward.py
"""

import os
import torch
import habana_frameworks.torch.core as htcore
import fastfold.habana as habana
from fastfold.config import model_config
from fastfold.model.hub import AlphaFold
from fastfold.utils.import_weights import import_jax_weights_
from fastfold.habana.inject_habana import inject_habana
from fastfold.habana.distributed import init_dist

# Init distributed (required even for single-HPU)
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
habana.enable_habana()
init_dist()

# ── Dimensions ────────────────────────────────────────────────────────────────
N_res     = 63
N_seq     = 128
N_extra   = 128
N_templ   = 4
num_iters = 1      # recycling iterations
PARAMS    = "/workspace/params/params_model_1.npz"
device    = torch.device("hpu")

# ── Helpers ───────────────────────────────────────────────────────────────────
def r(*shape):
    return torch.rand(*shape)

def ri(*shape, high=20):
    return torch.randint(0, high, shape)

def ones(*shape):
    return torch.ones(*shape)

def zeros(*shape):
    return torch.zeros(*shape)

# ── Synthetic batch ───────────────────────────────────────────────────────────
# Final dimension of every tensor is the recycling dimension (num_iters)
batch = {
    # Sequence features
    "aatype":                           ri(N_res,                   num_iters),
    "target_feat":                      r(N_res, 22,                num_iters),
    "residue_index":                    torch.arange(N_res).unsqueeze(-1).expand(N_res, num_iters),
    "seq_mask":                         ones(N_res,                  num_iters),
    "seq_length":                       torch.tensor([N_res] * num_iters),
    "no_recycling_iters":               torch.tensor([num_iters - 1] * num_iters),
    "resolution":                       torch.tensor([1.0] * num_iters),

    # MSA features
    "msa_feat":                         r(N_seq, N_res, 49,          num_iters),
    "msa_mask":                         ones(N_seq, N_res,            num_iters),
    "msa_row_mask":                     ones(N_seq,                   num_iters),
    "bert_mask":                        ones(N_seq, N_res,            num_iters),
    "true_msa":                         ri(N_seq, N_res,             num_iters, high=23),

    # Pair features
    "pair_mask":                        ones(N_res, N_res,            num_iters),
    "pseudo_beta":                      r(N_res, 3,                   num_iters),
    "pseudo_beta_mask":                 ones(N_res,                   num_iters),
    "backbone_rigid_tensor":            r(N_res, 4, 4,                num_iters),
    "backbone_rigid_mask":              ones(N_res,                   num_iters),
    "residx_atom37_to_atom14":          ri(N_res, 37,                num_iters, high=14),
    "atom37_atom_exists":               ones(N_res, 37,               num_iters),

    # Extra MSA features
    "extra_msa":                        ri(N_extra, N_res,           num_iters, high=23),
    "extra_msa_mask":                   ones(N_extra, N_res,          num_iters),
    "extra_msa_row_mask":               ones(N_extra,                 num_iters),
    "extra_has_deletion":               zeros(N_extra, N_res,         num_iters),
    "extra_deletion_value":             zeros(N_extra, N_res,         num_iters),

    # Template features
    "template_mask":                    ones(N_templ,                 num_iters),
    "template_aatype":                  ri(N_templ, N_res,           num_iters, high=20),
    "template_all_atom_positions":      r(N_templ, N_res, 37, 3,     num_iters),
    "template_all_atom_mask":           ones(N_templ, N_res, 37,      num_iters),
    "template_pseudo_beta":             r(N_templ, N_res, 3,          num_iters),
    "template_pseudo_beta_mask":        ones(N_templ, N_res,          num_iters),
    "template_torsion_angles_sin_cos":  r(N_templ, N_res, 7, 2,      num_iters),
    "template_alt_torsion_angles_sin_cos": r(N_templ, N_res, 7, 2,   num_iters),
    "template_torsion_angles_mask":     ones(N_templ, N_res, 7,       num_iters),
}

# ── Move to HPU ───────────────────────────────────────────────────────────────
batch = {k: v.to(device) for k, v in batch.items()}

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
config = model_config("model_1")
config.globals.inplace = False
model = AlphaFold(config)
import_jax_weights_(model, PARAMS, version="model_1")
model = inject_habana(model)
model = model.eval().to(device)
print("Model loaded.")

# ── Forward pass ──────────────────────────────────────────────────────────────
print("Running forward pass...")
with torch.no_grad():
    out = model(batch)
    htcore.mark_step()

print("Forward pass complete!")
print("Output keys:", list(out.keys()))
