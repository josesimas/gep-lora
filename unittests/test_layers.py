"""
The layering rule, held: apps -> assistant -> service -> core, never upward.

    gep_lora.core       imports nothing else of gep_lora but gep_lora.paths
    gep_lora.service    core
    gep_lora.assistant  service, core
    gep_lora.apps       assistant, service, core
    gep_lora.tools      core (dev aids over the engine)

Read from the source rather than by importing it, so an import tucked inside a
function -- the kind that crept in when the catalogue needed the API's
database path -- is caught as surely as one at the top of a file.
"""

import ast
import os
import unittest

from gep_lora import paths

PACKAGE = os.path.join(paths.ROOT, "gep_lora")

# What each layer may import from gep_lora, besides itself.
ALLOWED = {
    "core": {"paths"},
    "service": {"paths", "core"},
    "assistant": {"paths", "core", "service"},
    "apps": {"paths", "core", "service", "assistant"},
    "tools": {"paths", "core"},
}


def layer_of(module):
    """'gep_lora.service.drawn' -> 'service'; None outside gep_lora."""
    parts = module.split(".")
    if parts[0] != "gep_lora":
        return None
    return parts[1] if len(parts) > 1 else ""


def imported(tree):
    """Every module an AST imports, absolute, with the line it is on."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            if node.module == "gep_lora":
                for alias in node.names:            # from gep_lora import paths
                    yield "gep_lora." + alias.name, node.lineno
            else:
                yield node.module, node.lineno


def sources():
    for folder, dirs, files in os.walk(PACKAGE):
        dirs[:] = [name for name in dirs if name not in ("__pycache__", "templates", "static")]
        for name in files:
            if name.endswith(".py"):
                yield os.path.join(folder, name)


class LayerTests(unittest.TestCase):

    def test_no_layer_imports_one_above_it(self):
        broken = []
        for path in sources():
            relative = os.path.relpath(path, PACKAGE).replace(os.sep, "/")
            own = relative.split("/")[0]
            if own not in ALLOWED:                   # __init__.py, paths.py
                continue
            with open(path, encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), path)
            for module, line in imported(tree):
                layer = layer_of(module)
                if layer is None or layer == own or layer in ALLOWED[own]:
                    continue
                broken.append("gep_lora/%s:%d imports %s" % (relative, line, module))
        self.assertEqual([], broken, "an import against the layering:\n" + "\n".join(broken))

    def test_every_layer_is_listed(self):
        """A new folder under gep_lora/ is a new layer, and needs a rule here."""
        folders = {name for name in os.listdir(PACKAGE)
                   if os.path.isdir(os.path.join(PACKAGE, name)) and name != "__pycache__"}
        self.assertEqual(set(ALLOWED), folders)

    def test_nothing_imports_the_old_top_level_packages(self):
        """The folders gep_lora/ replaced; an import of one would only work
        from a stale checkout with the old folders still beside it."""
        old = {"config", "search", "blends", "storage", "metrics", "evaluators", "testing",
               "reporting", "adapters", "tools", "async_api", "async_api_agent",
               "start_run", "continue_run", "main"}
        found = []
        for path in sources():
            with open(path, encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), path)
            for module, line in imported(tree):
                if module.split(".")[0] in old:
                    found.append("%s:%d imports %s" % (os.path.relpath(path, paths.ROOT),
                                                       line, module))
        self.assertEqual([], found)


if __name__ == "__main__":
    unittest.main()
