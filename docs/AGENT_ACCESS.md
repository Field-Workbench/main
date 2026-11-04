# Field Workbench agent access

## Implementation plan and scope

The system has one command layer shared by a command-line client and an MCP stdio
server. Both attach to the same running desktop instance and address explicit
project IDs. Live, committed project state includes unsaved changes; private drafts
in individual editors are not part of that state until those editors commit them.

Implementation sequence:

1. **Command contract:** one registry of command names, descriptions, JSON input
   schemas, write annotations, revision requirements, and structured errors.
2. **Live host:** local discovery, authenticated loopback connection, queued Qt
   delivery, project lifecycle/save integration, and ordinary undo/view updates.
3. **Clients:** lightweight CLI and SDK-based MCP stdio adapter, both using the
   registry and the same service. Neither imports the numerical engine or Qt.
4. **Calculations:** bounded background queue on frozen models, source provenance,
   cooperative cancellation, retained results, exports, explicit candidate/snapshot
   application, and source revision checks.
5. **Verification/documentation:** real adapter tests, socket/Qt/CLI tests, real MCP
   client tests, regression checks, and setup/extension documentation.

These components are implemented together. User acceptance testing is deferred;
passing automated tests is not a claim of manual or packaged-release acceptance.
The initial command set covers project creation/open/save/export, model object
inspection and editing, notes, resource discovery, field/measurement/DRC calculations,
and both existing-coil and generative optimization. It does not yet expose every
specialist editor, map renderer, waveform/exposure authoring action, or export format.
Those can extend the same command layer without adding a separate transport API.

## Start a controllable instance

From the source directory, using the normal FWB environment:

```bash
python app.py --automation
# Or open a project immediately:
python app.py --automation "examples/Maxwell constant-field.magpy.json"
```

Automation is opt-in for each process. Ordinary launches create no listener.
The listener uses an ephemeral port on `127.0.0.1`, a random instance ID, and a
random token. Its private discovery file is removed on clean shutdown. Stale files
left by a crash are ignored when their endpoint no longer identifies the instance.
A discovery file cannot change the destination host away from loopback.

On POSIX, discovery lives in `$XDG_RUNTIME_DIR/field-workbench-<uid>` or the equivalent
per-user directory in the OS temporary directory. The directory must be mode 0700
and descriptors mode 0600, owned by the current user. On Windows, use the user's
private temporary directory (normal OS account/ACL isolation); POSIX mode checks do
not apply. Do not share the discovery directory: descriptors contain credentials.

For an explicit location, use `--automation-runtime-dir /private/directory` when
launching the GUI and `--runtime-dir /private/directory` for CLI/MCP. Paths are local
to the computer running FWB. A cloud agent needs an executor on that computer or
another deliberately configured connection; local discovery alone provides no
remote network access.

## Windows executable and separate client

The current Windows build freezes only the GUI. Its desktop entry point accepts
`--automation`; successful packaged operation must still be verified on Windows.
From Command Prompt, launch the extracted release candidate:

```bat
"C:\FWB\FieldWorkbench\FieldWorkbench.exe" --automation
```

Keep `_internal` and all release resources together. A normal double-click launch
does not enable automation. Use matching source from the same release revision for
the separate client; do not pass `fwb.py` to the GUI executable.

```bat
cd /d C:\FWB\source\field_workbench
py -3.12 fwb.py instances
py -3.12 fwb.py commands
py -3.12 fwb.py --instance INSTANCE_ID call projects.list
```

The CLI uses the standard library and needs neither Qt nor the numerical stack.
For MCP, create a separate client environment and install its optional dependency:

```bat
py -3.12 -m venv C:\FWB\agent-env
C:\FWB\agent-env\Scripts\python.exe -m pip install -r requirements-agent.txt
C:\FWB\agent-env\Scripts\python.exe fwb.py --instance INSTANCE_ID mcp
```

Example Codex configuration (TOML literal strings preserve Windows backslashes):

```toml
[mcp_servers.field_workbench]
command = 'C:\FWB\agent-env\Scripts\python.exe'
args = ['C:\FWB\source\field_workbench\fwb.py', '--instance', 'INSTANCE_ID', 'mcp']
```

Replace paths and the discovered instance ID. These paths must be on the same
Windows computer as FWB. The MCP bridge does not launch a GUI and remains pinned
to the chosen instance. Restart/reconfigure it after the GUI restarts with a new ID.
Follow the packaged acceptance checklist in [TESTING.md](../TESTING.md); source
integration tests alone do not establish Windows executable support.

## CLI

The source entry points are interchangeable:

```bash
python fwb.py --help
python -m fieldworkbench.automation --help
python fwb.py instances
python fwb.py commands
python fwb.py --instance INSTANCE_ID call projects.list
python fwb.py --instance INSTANCE_ID call project.get \
  --params '{"project_id":"PROJECT_ID"}'
```

`commands` works offline and includes editing operation schemas. It is also available
from the live instance as `commands.list`. The generic `call` command is intentional:
there is no independently maintained CLI command hierarchy or conversion logic.
JSON parameters preserve booleans, vectors, arrays, and numerical settings exactly.
Use `--params-file path.json` for complex inputs, or `--params-file -` for stdin.
Global options precede `call` or `mcp`.

After inspecting the project, pass its returned revision to every project write:

```bash
python fwb.py --instance INSTANCE_ID call object.add --params \
  '{"project_id":"PROJECT_ID","expected_revision":"REVISION","template":"circle"}'
```

Inspect the new object with `object.get`; use its ID in subsequent calls. For example,
change the current and position in one undoable transaction:

```json
{
  "project_id": "PROJECT_ID",
  "expected_revision": "LATEST_REVISION",
  "operations": [
    {"method": "set_coil_physical", "params": {"object_id": "COIL_ID", "drive_current_a": 2}},
    {"method": "set_transform", "params": {"object_id": "COIL_ID", "position": [0, 0, 0.05]}}
  ]
}
```

Submit this file to `call objects.edit --params-file edit.json`. There is no implicit
save. `project.save` saves the committed project, while `project.export` writes a
copy without changing the project path or dirty baseline. Saving to a new existing
path requires `overwrite: true`; saving the project's current path replaces its
own previous save. Save/export writes are atomic and reject destinations belonging
to another open project where applicable.

CLI output is JSON. Calls return a versioned envelope:

```json
{"protocol_version":1,"request_id":"...","ok":true,"result":{},"project_id":"...","revision":"..."}
```

Errors use `ok: false` and `error: {code, message, details}`. Call exit codes are 0
for success, 1 for a server command error, and 2 for client/configuration/transport
errors. Parser usage errors also exit 2. Discovery/catalog commands return their
JSON list/catalog directly. They never print credentials.

## MCP

Only the separate MCP bridge requires the optional dependency:

```bash
python -m pip install -r requirements-agent.txt
python fwb.py --instance INSTANCE_ID mcp
```

This starts an MCP stdio server attached to the existing GUI, not another FWB GUI.
Use a Python environment containing the pinned MCP SDK, and an absolute path to
`fwb.py` in agent configuration. It need not be the GUI's Python environment.
For example, the equivalent Codex configuration is:

```toml
[mcp_servers.field_workbench]
command = "/absolute/path/to/mcp-environment/bin/python"
args = ["/absolute/path/to/field_workbench/fwb.py", "--instance", "INSTANCE_ID", "mcp"]
```

Use Windows Python paths on Windows. Add `--runtime-dir` before `mcp` if the GUI
uses a custom discovery location. An explicit instance is recommended; otherwise
exactly one responsive local instance must exist at first tool use. An MCP bridge
pins its chosen instance for its lifetime and never silently switches to another.

Each command becomes a tool, e.g. `project.get` becomes `fwb_project_get`.
Input schemas and write annotations come from the same registry used by the CLI
and live service. Command errors are MCP tool errors (`isError: true`) with the
same structured envelope and a text representation. stdout contains only MCP
protocol traffic; startup diagnostics go to stderr. No OpenAI API key is involved.

## Units and edits

- Native model parameters and positions use SI metres unless the parameter metadata
  returned by `object.get` specifies another unit.
- Transform `orientation` is a rotation vector in degrees, not Euler angles. Object
  inspection also returns the adapter's Euler representation for reference.
- Coil physical fields keep their explicit units, e.g. `drive_current_a`,
  `winding_axial_width_mm`, and `lead_length_m`. Use the physical current field for
  drive current; the underlying Magpylib `current` parameter may encode ampere-turns.
- `objects.edit` validates the exposed operation names and parameters before applying
  an atomic batch through the native adapter. A failed batch restores the original
  scene/history. A successful batch is one undo step. Unknown coil physical fields
  are rejected. `apply_edit` currently exposes the label path only.
- Finish modal interactive dialogs before external writes. Other editors can retain
  private drafts; agent edits do not commit those drafts automatically.
- A project revision hashes committed document state and runtime solver policy.
  Changed inputs cause `revision_conflict`; re-inspect before constructing a new
  edit. A job additionally checks its frozen source revision before applying results.

## Calculations and retained results

`calculation.start` takes `kind`, `settings`, `project_id`, and `expected_revision`.
Supported settings:

| Kind | Settings |
| --- | --- |
| `field` | `points_m`: array of 3-vectors; optional `field`: `B` or `H` |
| `measurement` | `object_id`: eligible Measurement volume |
| `drc` | `{}` |
| `optimizer` | Native generative optimizer settings; seed with `optimizer.inspect` |
| `optimizer_tuning` | Native existing-coil tuning settings, validated by the optimizer |

Jobs transition `queued → running → succeeded/failed/cancelled`. There is one
worker per instance, at most eight pending jobs, and at most 32 retained jobs.
Each job owns its launch-time document, solver preferences, and numerical adapter;
no live adapter or widgets are accessed from the worker. Closing a project requests
cancellation of its jobs and releases their retained data. Quitting requests cancellation; the GUI does not join the daemon numerical
worker. Cancellation inside numerical libraries remains cooperative.

Poll with `job.get`; inspect `job.result` for a bounded summary or pass `full: true`
for arrays. Use `job.export` for a complete JSON result including source document,
settings, runtime policy, and source revision. Undefined/nonfinite numerical outputs
serialize as JSON null. Exports retain the full result even when summaries omit
large arrays. Messages are limited to 8 MiB; use exports for larger data.

`job.cancel` is cooperative: callbacks and optimizer cancellation checkpoints observe
it; an uninterruptible numerical call may finish first. Cancelled results are never
published as successful results. `job.forget` releases a terminal job's retained data.

`job.apply` stores a measurement snapshot or applies an explicitly selected optimizer
`candidate_id` from `finalists`, through the ordinary undo path. The live model must
still match the job's source revision. Field and DRC jobs have no model edit to apply.
Results and job IDs are session-local, not persisted queue entries. Export results
before exiting if they must be kept outside the native saved-result workflow.

## Request lifetime and retries

Live reads/writes execute on the Qt main thread. The listener limits concurrent
connections and queued deliveries; the numerical queue has separate limits.
A request waiting more than 25 seconds to execute expires without applying later.
An operation that already started may finish after the caller disconnects; a timeout
can therefore report `outcome_unknown: true`. Never blindly retry a write with a
new request ID after an ambiguous connection failure.

The instance retains the last 128 completed request responses, bounded to 32 MiB.
The CLI's `--request-id` can replay an identical recent request without repeating its
effects. Reusing that ID with different parameters returns `request_id_conflict`.
This is bounded session-local replay protection, not durable exactly-once delivery:
after eviction or restart, inspect actual project state before deciding what to do.

## Extension points and automated checks

- `commands.py`: public contract, schemas, descriptions, and tool/write metadata.
- `service.py`: transport-independent command behavior and revision enforcement.
- `qt_host.py`: the sole adapter to project widgets, main-thread delivery, and replay.
- `connection.py` / `server.py`: instance discovery and authenticated loopback framing.
- `jobs.py`: frozen numerical work, queue/cancellation/provenance/result retention.
- `cli.py` / `mcp.py`: presentation and protocol adapters, no duplicated model logic.

Add a command/schema and service behavior together; CLI help and MCP tools follow
automatically. Prefer explicit domain actions over arbitrary Python execution or
reflection-based exposure of adapter methods. Extend job kinds for expensive work.

Run focused tests in the normal environment:

```bash
python -m unittest discover -s tests -p 'test_automation_*.py' -v
```

With `requirements-agent.txt` installed, this includes a real SDK client launching
the MCP stdio process and verifying shared live state with the CLI. Without it,
that optional test is skipped. Local-connection tests need loopback socket access;
Qt tests use the existing offscreen test convention. Full release verification
continues to follow `TESTING.md`. Windows packaging and manual user acceptance
remain deferred and are not inferred from source-level integration tests.
