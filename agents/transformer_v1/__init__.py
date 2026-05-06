# transformer_v1 — action-decoder agent (freeze-trained imitation policy
# + physical_v4 launch helper). Design lives in DESIGN.md. Importing
# ``runner`` triggers the ``@register("transformer_v1", ...)`` decorator
# so ``Agent("transformer_v1")`` resolves once this package is imported.

# Lazy import so the package can still be imported in environments
# without torch (e.g. doc generation). The actual registration happens
# inside ``runner``.
try:
    from . import runner  # noqa: F401
except ImportError:
    pass

try:
    from . import ppo  # noqa: F401
except ImportError:
    pass
