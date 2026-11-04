"""Real MCP SDK stdio round trips; optional only when the MCP extra is absent."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest

from test_automation_transport import TransportHarness

from fieldworkbench.automation.commands import COMMANDS


@unittest.skipUnless(
    importlib.util.find_spec("mcp"), "Optional requirements-agent.txt is not installed"
)
class MCPTests(TransportHarness):
    def test_stdio_tools_match_catalog_and_share_live_cli_state(self):
        async def exercise():
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            parameters = StdioServerParameters(
                command=sys.executable,
                args=[
                    "fwb.py",
                    "--instance",
                    self.host.instance_id,
                    "--runtime-dir",
                    self.directory.name,
                    "mcp",
                ],
            )
            async with (
                stdio_client(parameters) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                tools = (await session.list_tools()).tools
                self.assertEqual(
                    {t.name for t in tools},
                    {c.tool_name for c in COMMANDS.values()},
                )
                for tool in tools:
                    command = next(
                        c for c in COMMANDS.values() if c.tool_name == tool.name
                    )
                    self.assertEqual(tool.inputSchema, command.input_schema)
                    self.assertEqual(tool.annotations.readOnlyHint, not command.writes)
                projects = (
                    await session.call_tool("fwb_projects_list", {})
                ).structuredContent
                project = projects["result"][0]
                added = await session.call_tool(
                    "fwb_object_add",
                    {
                        "project_id": project["id"],
                        "expected_revision": project["revision"],
                        "template": "circle",
                    },
                )
                self.assertFalse(added.isError, added)
                stale = await session.call_tool(
                    "fwb_notes_set",
                    {
                        "project_id": project["id"],
                        "expected_revision": project["revision"],
                        "notes": "stale",
                    },
                )
                self.assertTrue(stale.isError)
                self.assertEqual(
                    stale.structuredContent["error"]["code"], "revision_conflict"
                )
                invalid = await session.call_tool(
                    "fwb_objects_list", {"project_id": project["id"], "typo": True}
                )
                self.assertTrue(invalid.isError)
                self.assertEqual(
                    invalid.structuredContent["error"]["code"], "invalid_params"
                )
                return added.structuredContent["result"]["object_id"]

        oid = self.background(lambda: asyncio.run(exercise()))
        response = self.background(
            lambda: self.client.call(
                "objects.list", {"project_id": self.context.project_id}
            )
        )
        self.assertEqual(response["result"][0]["id"], oid)


if __name__ == "__main__":
    unittest.main()
