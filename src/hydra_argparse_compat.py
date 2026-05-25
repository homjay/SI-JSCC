"""Compatibility patches for Hydra + argparse.

Background:
- `hydra-core==1.3.2` constructs an argparse parser with a non-string `help` object
  for `--shell-completion`.
- Python 3.14's argparse validates help strings more strictly and can crash during
  parser construction.

This module provides a small monkeypatch that converts non-string `help` values
into strings.
"""

from __future__ import annotations

import sys


def patch_hydra_argparse_help_py314() -> None:
    """Patch argparse to accept Hydra's non-string help on Python 3.14+.

    Idempotent: safe to call multiple times.
    """

    if sys.version_info < (3, 14):
        return

    import argparse

    if getattr(argparse.ArgumentParser.add_argument, "__hydra_help_patched__", False):
        return

    original_add_argument = argparse.ArgumentParser.add_argument

    def patched_add_argument(self, *args, **kwargs):
        help_value = kwargs.get("help")
        if help_value is not None and not isinstance(help_value, str):
            help_type = type(help_value)
            # Hydra defines LazyCompletionHelp as a local class inside get_args_parser.
            # Avoid evaluating its expensive repr; use a plain string.
            if help_type.__name__ == "LazyCompletionHelp" and help_type.__module__.startswith(
                "hydra"
            ):
                kwargs["help"] = "Install or uninstall shell completion"
            else:
                kwargs["help"] = str(help_value)

        return original_add_argument(self, *args, **kwargs)

    setattr(patched_add_argument, "__hydra_help_patched__", True)
    argparse.ArgumentParser.add_argument = patched_add_argument  # type: ignore[assignment]
