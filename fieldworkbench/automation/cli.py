"""Thin CLI. Run python -m fieldworkbench.automation --help (no GUI imports)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .commands import COMMANDS
from .connection import CommandClient, discover
from .protocol import CommandError, dumps, failure, loads, validate


def parser():
    root = argparse.ArgumentParser(
        prog="fwb", description="Control a running Field Workbench instance."
    )
    root.add_argument(
        "--instance",
        help="Explicit instance ID; required when multiple instances are running",
    )
    root.add_argument(
        "--runtime-dir",
        help="Private discovery directory, matching app.py --automation-runtime-dir",
    )
    root.add_argument("--timeout", type=float, default=30.0)
    root.add_argument(
        "--request-id",
        help="Optional identity for replaying an identical recent request",
    )
    actions = root.add_subparsers(dest="action", required=True)
    actions.add_parser(
        "instances", help="List live instances (never displays authentication tokens)"
    )
    actions.add_parser(
        "commands", help="Show the shared command catalog without connecting to FWB"
    )
    call = actions.add_parser("call", help="Execute any command from the catalog")
    call.add_argument("command", choices=COMMANDS)
    arguments = call.add_mutually_exclusive_group()
    arguments.add_argument("--params", default="{}", help="JSON object")
    arguments.add_argument("--params-file", help="UTF-8 JSON file, or - for stdin")
    actions.add_parser(
        "mcp",
        help="Serve the same commands over MCP stdio (requires requirements-agent.txt)",
    )
    return root


def main(argv=None):
    options = parser().parse_args(argv)
    try:
        if not 0 < options.timeout <= 300:
            raise CommandError(
                "invalid_params",
                "Timeout must be greater than zero and at most 300 seconds.",
            )
        if options.action == "instances":
            result = [
                {k: v for k, v in d.items() if k != "token"}
                for d in discover(options.runtime_dir)
            ]
        elif options.action == "commands":
            from .commands import OPERATION_SCHEMAS

            result = {
                "commands": [c.catalog() for c in COMMANDS.values()],
                "operation_schemas": OPERATION_SCHEMAS,
            }
        elif options.action == "mcp":
            from .mcp import run

            run(CommandClient(options.instance, options.runtime_dir, options.timeout))
            return 0
        else:
            raw = options.params
            if options.params_file:
                raw = (
                    sys.stdin.read()
                    if options.params_file == "-"
                    else Path(options.params_file).read_text(encoding="utf-8")
                )
            parameters = loads(raw)
            validate(parameters, COMMANDS[options.command].input_schema)
            result = CommandClient(
                options.instance, options.runtime_dir, options.timeout
            ).call(options.command, parameters, request_id=options.request_id)
            print(dumps(result))
            return 0 if result["ok"] else 1
        print(dumps(result))
        return 0
    except (CommandError, OSError, ValueError) as error:
        if not isinstance(error, CommandError):
            error = CommandError("client_error", str(error))
        if options.action == "mcp":
            print(f"FWB MCP: {error}", file=sys.stderr)
        else:
            print(dumps(failure(error)))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
