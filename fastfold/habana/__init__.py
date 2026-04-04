import habana_frameworks.torch.core as htcore  # noqa: F401

from fastfold.habana.fastnn import EvoformerStack, ExtraMSAStack  # noqa: F401

ENABLE_HABANA = False
ENABLE_HMP = False
ENABLE_LAZY_MODE = False


def enable_habana():
    """Initialize HPU backend. On Synapse 1.23.0+, the import above is sufficient."""
    global ENABLE_HABANA, ENABLE_LAZY_MODE
    ENABLE_HABANA = True
    ENABLE_LAZY_MODE = True


def is_habana():
    global ENABLE_HABANA
    return ENABLE_HABANA


def enable_hmp():
    global ENABLE_HMP
    ENABLE_HMP = True


def is_hmp():
    global ENABLE_HMP
    return ENABLE_HMP
