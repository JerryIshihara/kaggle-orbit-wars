from __future__ import annotations

import ast
import inspect
import tarfile
from datetime import datetime
from pathlib import Path

import agents

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = _REPO_ROOT / "logs" / "submissions"


def pack_agent(
    agent_id: str,
    out_dir: Path | str | None = None,
    extra_files: list[Path | str] | None = None,
    bundle: bool = False,
) -> dict[str, Path | None]:
    """Pack a registered agent into a Kaggle-submittable `main.py`.

    Rewrites the agent's source to (1) drop the `from .registry import register`
    import, (2) strip the `@register(...)` decorator, and (3) append an
    `agent = <fn_name>` alias at the end so Kaggle can pick up the callable.

    Args:
      agent_id: registered agent id (e.g. "sniper").
      out_dir: where to write `main.py`. Defaults to `submissions/<agent_id>/`.
      extra_files: paths of additional files to copy alongside `main.py` (e.g.,
        model weights, helper modules). They're copied verbatim, filename preserved.
      bundle: also emit a `.tar.gz` at `out_dir/../<agent_id>.tar.gz`
        containing every file in `out_dir`.

    Returns:
      {"main": Path to main.py, "bundle": Path or None, "dir": out_dir}.
    """
    spec = agents.Agent(id=agent_id)
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUT / agent_id
    out_dir.mkdir(parents=True, exist_ok=True)

    module = inspect.getmodule(spec.fn)
    if module is None:
        raise RuntimeError(f"could not resolve source module for agent {agent_id!r}")
    source = inspect.getsource(module)

    main_code, sibling_modules = _transform_source(
        source,
        fn_name=spec.fn.__name__,
        agent_id=agent_id,
        description=spec.description,
    )
    main_path = out_dir / "main.py"
    main_path.write_text(main_code)

    # Copy any sibling helper modules (e.g. agents/physics_utils.py) that the
    # transformed main.py now imports as top-level, so they sit next to main.py
    # in the bundle and resolve at submission time.
    for mod_name in sibling_modules:
        sibling_path = _REPO_ROOT / "agents" / f"{mod_name}.py"
        if sibling_path.exists():
            (out_dir / f"{mod_name}.py").write_bytes(sibling_path.read_bytes())

    if extra_files:
        for f in extra_files:
            p = Path(f)
            (out_dir / p.name).write_bytes(p.read_bytes())

    bundle_path: Path | None = None
    if bundle:
        bundle_path = out_dir.parent / f"{agent_id}.tar.gz"
        with tarfile.open(bundle_path, "w:gz") as tar:
            for p in sorted(out_dir.iterdir()):
                tar.add(p, arcname=p.name)

    return {"main": main_path, "bundle": bundle_path, "dir": out_dir}


def _transform_source(source: str, fn_name: str, agent_id: str, description: str) -> tuple[str, set[str]]:
    """Return (transformed_source, set_of_sibling_module_names_to_bundle).

    Transforms applied:
      - drop ``from ..registry import register``
      - drop ``@register(...)`` decorators
      - rewrite cross-agent relative imports like ``from ..physics_utils import X``
        into top-level ``from physics_utils import X`` (so the bundled module
        next to ``main.py`` resolves) and record the module name so the
        packer can copy the file alongside.
      - append ``agent = <fn_name>`` alias.
    """
    tree = ast.parse(source)
    new_body: list[ast.stmt] = []
    sibling_modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[-1] == "registry":
            continue
        if isinstance(node, ast.ImportFrom) and node.level and node.level >= 2 and node.module:
            # `from ..foo import X`  →  `from foo import X` (foo is bundled).
            sibling_modules.add(node.module)
            node.level = 0
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = [
                d for d in node.decorator_list if not _is_register_decorator(d)
            ]
        new_body.append(node)

    new_body.append(
        ast.Assign(
            targets=[ast.Name(id="agent", ctx=ast.Store())],
            value=ast.Name(id=fn_name, ctx=ast.Load()),
        )
    )
    tree.body = new_body
    ast.fix_missing_locations(tree)

    header = (
        "# Auto-packed by utils.pack_agent\n"
        f"# agent id   : {agent_id}\n"
        f"# description: {description}\n"
        f"# packed at  : {datetime.now().isoformat(timespec='seconds')}\n\n"
    )
    return header + ast.unparse(tree) + "\n", sibling_modules


def _is_register_decorator(d: ast.expr) -> bool:
    if isinstance(d, ast.Call):
        return _is_register_decorator(d.func)
    if isinstance(d, ast.Name):
        return d.id == "register"
    if isinstance(d, ast.Attribute):
        return d.attr == "register"
    return False
