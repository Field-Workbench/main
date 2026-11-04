"""MCP stdio adapter: tools and schemas are generated from the shared registry."""

from __future__ import annotations

import asyncio

from .commands import COMMANDS
from .protocol import CommandError, dumps, failure

INSTRUCTIONS = (
    "Control the already-running Field Workbench instance. List projects first and use explicit project IDs. "
    "Inspect project.get before writes and pass its revision as expected_revision. All model positions are SI metres; "
    "orientation is a rotation vector in degrees. Inspect object parameters for their units. Calculations run on frozen "
    "copies: poll jobs and explicitly apply selected results. Private editor drafts are excluded. Never blindly retry edits "
    "after a connection failure. Commands and editing operation schemas are available through commands.list. "
    "Use exports for large data. No command saves the project unless explicitly requested."
)


def create_server(client):
    try:
        from mcp import types
        from mcp.server.lowlevel import Server
    except ImportError as error:
        raise CommandError(
            "missing_dependency",
            "Install requirements-agent.txt in the MCP client's Python environment.",
        ) from error
    server = Server("Field Workbench", instructions=INSTRUCTIONS)
    mapping = {command.tool_name: command for command in COMMANDS.values()}

    @server.list_tools()
    async def list_tools():
        return [
            types.Tool(
                name=c.tool_name,
                description=c.description,
                inputSchema=c.input_schema,
                annotations=types.ToolAnnotations(
                    readOnlyHint=not c.writes,
                    destructiveHint=c.writes,
                    openWorldHint=False,
                ),
            )
            for c in COMMANDS.values()
        ]

    @server.call_tool(validate_input=False)
    async def call_tool(name, arguments):
        # Validation/errors belong to the FWB service, identical to CLI calls.
        command = mapping.get(name)
        try:
            if command is None:
                raise CommandError("unknown_command", f"Unknown tool: {name}")
            response = await asyncio.to_thread(
                client.call, command.name, arguments or {}
            )
        except CommandError as error:
            response = failure(error)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=dumps(response))],
            structuredContent=response,
            isError=not response["ok"],
        )

    return server


def run(client):
    server = create_server(client)
    from mcp.server.stdio import stdio_server

    async def serve():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(serve())
