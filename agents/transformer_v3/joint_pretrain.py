"""v3 joint pretrain entry — the v2 driver with ``--arch v3`` forced.

    python -m agents.transformer_v3.joint_pretrain \
        --out-dir ... --pair-cache-path ... (same flags as the v2 driver)

Identical to ``python -m agents.transformer_v2.pretrain.joint_pretrain
--arch v3 ...``: dual-rate L2 model, 18-frame union restack of both
caches, v2 warm-start key-mapped onto both branches (fusion layers stay
zero-init so epoch 0 starts exactly at the v2 model's behavior).
"""

from __future__ import annotations

import sys


def main() -> None:
    from ..transformer_v2.pretrain.joint_pretrain import main as v2_main

    if "--arch" in sys.argv:
        i = sys.argv.index("--arch")
        if sys.argv[i + 1 : i + 2] != ["v3"]:
            raise SystemExit(
                "agents.transformer_v3.joint_pretrain is the --arch v3 "
                "entry; drop the flag or pass --arch v3"
            )
    else:
        sys.argv.append("--arch")
        sys.argv.append("v3")
    v2_main()


if __name__ == "__main__":
    main()
