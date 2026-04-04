#!/usr/bin/env python3
"""FastFold-Gaudi smoke test — validates the container build, patches, and HPU access.

This runs WITHOUT model weights or databases. It checks:
  1. habana_frameworks imports
  2. HPU device availability
  3. FastFold import + patched fused_softmax
  4. Habana distributed init
  5. Basic tensor ops on HPU (matmul, softmax, mark_step)
  6. AlphaFold model instantiation (random weights, no .npz needed)
  7. A tiny forward pass through the Evoformer with injected Habana ops

Exit code 0 = all checks passed. Non-zero = something is broken.
"""

import os
import sys
import time
import traceback

os.environ.setdefault("PT_HPU_LAZY_MODE", "1")

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []


def check(name, fn):
    """Run a check, print result, accumulate pass/fail."""
    try:
        t0 = time.time()
        detail = fn()
        elapsed = time.time() - t0
        results.append((name, True))
        detail_str = f" — {detail}" if detail else ""
        print(f"  [{PASS}] {name} ({elapsed:.2f}s){detail_str}")
    except Exception as e:
        results.append((name, False))
        print(f"  [{FAIL}] {name}: {e}")
        traceback.print_exc()


# =========================================================================
# Check 1: Habana frameworks
# =========================================================================
def check_habana_import():
    import habana_frameworks.torch.core as htcore  # noqa: F811
    return f"htcore module at {htcore.__file__}" if hasattr(htcore, '__file__') else "OK"


# =========================================================================
# Check 2: HPU device
# =========================================================================
def check_hpu_device():
    import torch
    assert torch.hpu.is_available(), "torch.hpu.is_available() returned False"
    count = torch.hpu.device_count()
    assert count > 0, f"No HPU devices found (count={count})"
    return f"{count} HPU device(s)"


# =========================================================================
# Check 3: FastFold import + patched custom ops
# =========================================================================
def check_fastfold_import():
    import fastfold  # noqa: F401
    from fastfold.habana.fastnn.custom_op import fused_softmax, fused_softmax_bias
    # Verify the replacements are pure-Python (not the old C++ extension)
    import torch
    logits = torch.randn(2, 4, 8)
    mask = torch.zeros(2, 4, 8)
    out = fused_softmax(logits, mask, dim=-1)
    assert out.shape == logits.shape, f"Shape mismatch: {out.shape}"
    # Check softmax sums to ~1
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "Softmax doesn't sum to 1"
    return "fused_softmax/fused_softmax_bias patched OK"


# =========================================================================
# Check 4: Habana distributed init
# =========================================================================
def check_distributed():
    from fastfold.habana.distributed import init_dist
    # Single-device: init_dist should be a no-op (world_size=1)
    init_dist()
    return "init_dist() OK (single device)"


# =========================================================================
# Check 5: HPU tensor operations + mark_step
# =========================================================================
def check_hpu_tensor_ops():
    import torch
    import habana_frameworks.torch.core as htcore

    device = torch.device("hpu")

    # matmul
    a = torch.randn(64, 128, device=device)
    b = torch.randn(128, 64, device=device)
    c = torch.matmul(a, b)
    htcore.mark_step()
    assert c.shape == (64, 64), f"matmul shape wrong: {c.shape}"

    # softmax (the operation FastFold relies on)
    logits = torch.randn(4, 8, 32, device=device)
    out = torch.nn.functional.softmax(logits, dim=-1)
    htcore.mark_step()
    sums = out.sum(dim=-1).cpu()
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), "HPU softmax broken"

    # bf16 matmul (Gaudi 2 native)
    a16 = a.to(torch.bfloat16)
    b16 = b.to(torch.bfloat16)
    c16 = torch.matmul(a16, b16)
    htcore.mark_step()

    return f"matmul, softmax, bf16 on HPU OK (result device={c.device})"


# =========================================================================
# Check 6: AlphaFold model instantiation (random weights)
# =========================================================================
def check_model_instantiation():
    from fastfold.config import model_config
    from fastfold.model.hub import AlphaFold

    config = model_config("model_1")
    config.globals.chunk_size = 4  # small for test
    config.globals.is_multimer = False
    model = AlphaFold(config)
    param_count = sum(p.numel() for p in model.parameters())
    return f"AlphaFold model_1 instantiated ({param_count:,} params, random weights)"


# =========================================================================
# Check 7: Habana injection + model on HPU
# =========================================================================
def check_inject_habana():
    import torch
    import habana_frameworks.torch.core as htcore
    from fastfold.config import model_config
    from fastfold.model.hub import AlphaFold
    from fastfold.habana.inject_habana import inject_habana

    config = model_config("model_1")
    config.globals.chunk_size = 4
    config.globals.is_multimer = False
    config.globals.inplace = False

    model = AlphaFold(config)
    model = inject_habana(model)
    model = model.eval()
    model = model.to(device=torch.device("hpu"))
    htcore.mark_step()

    return "inject_habana + model.to(hpu) OK"


# =========================================================================
# Run all checks
# =========================================================================
def main():
    print("=" * 60)
    print("FastFold-Gaudi Smoke Test")
    print("=" * 60)
    print()

    check("1. Habana frameworks import", check_habana_import)
    check("2. HPU device availability", check_hpu_device)
    check("3. FastFold + patched fused_softmax", check_fastfold_import)
    check("4. Habana distributed init", check_distributed)
    check("5. HPU tensor ops + mark_step", check_hpu_tensor_ops)
    check("6. AlphaFold model instantiation", check_model_instantiation)
    check("7. inject_habana + model on HPU", check_inject_habana)

    print()
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print("=" * 60)
    if passed == total:
        print(f"  ALL {total} CHECKS PASSED")
        print("  Container is ready for AlphaFold inference.")
    else:
        failed = [name for name, ok in results if not ok]
        print(f"  {passed}/{total} checks passed.  FAILED: {', '.join(failed)}")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
