# Field Workbench changes

## Unreleased — Measurement playback and local agent access

The source version remains 0.14.0.313. The following changes are not a claim of
Windows packaged or manual release acceptance.

- Added optional projected sensor placements and well rims to newly generated static and scripted 2D maps. Well rims use built-in plate geometry and follow the plate transform; they are independent of object outlines.
- Added optional Live measurement analysis to Scripted 2D/3D maps. The panel follows playback and scrubbing, using defined well points or configured measurement sampling.
- Added expandable individual-point rows for explicitly defined samples, including well labels, selection highlighting, and draggable metric columns. Point child rows are limited to 2,000 per object; aggregate statistics retain all numerical samples.
- Updated the existing 4D-box example.

### Shared local CLI/MCP command system

- Added one command registry and live dispatcher for project/model operations, resource discovery, notes, calculations, and optimizer jobs. CLI help and MCP tool schemas are derived from that registry.
- Added opt-in `--automation` startup, authenticated per-user instance discovery, main-thread command dispatch, bounded request replay protection, atomic exports, and explicit project revision checks.
- Added frozen-model background jobs with progress, cooperative cancellation, retained numerical results, provenance exports, and undoable measurement-snapshot/optimizer-finalist application.
- Added `fwb.py`, a lightweight source CLI, an optional pinned MCP stdio bridge dependency, automated end-to-end coverage, and `docs/AGENT_ACCESS.md`. User acceptance and packaged-release testing are deferred.

## 0.14.0.313 — Physical-refinement coordinate propagation and report cleanup

- Fixed capable-physics candidate reconstruction so every structural coordinate supplied by a candidate is restored, including scout-invisible freedoms such as axial winding length. Previously `parent_row_from_values()` copied only Centreline-visible structural variables, which could reset a physically optimized axial winding length to the source value between directional axes and before authoritative finalist vetting.
- Kept the fix generic rather than special-casing axial width: future structural freedoms omitted from the Centreline scout now survive physical directional/coupled refinement whenever the capable-physics candidate carries them.
- Corrected Optimizer 2.4 Balanced/Full report wording. Centreline is described as basin reconnaissance for these efforts; required directional closure is established under capable physics for the activated coupling family (Balanced) or every selected structural freedom (Full).
- Added an effort-specific note above the Centreline directional-closure audit so an intentionally open diagnostic scout table in Balanced/Full is not misrepresented as a failed effort contract. Fast continues to identify Centreline closure as required correctness evidence for scout-visible freedoms.
- Added regression guards for scout-invisible physical-coordinate preservation and the corrected Balanced reporting semantics. Dependency-light optimizer/candidate/policy/packaging gate passes 133/133 tests in this environment.

## 0.14.0.312 — Optimizer 2.4 unit-test and source-capability cleanup

- Fixed Auto model-policy handling for turn-count freedoms on legacy/coils without enabled physical winding geometry. The generic turns descriptor remains conservative, but source-specific capability context now recognizes that no winding-cross-section magnetic effect exists when physical winding geometry is disabled; Centreline can therefore represent the active magnetic turns effect instead of incorrectly requiring a Bundled distributed-winding floor.
- Preserved strict model promises: turns on coils with enabled physical winding geometry still require distributed-winding capable refinement where the effort contract demands it, and explicit Current Sheet/Bundled/Exact authority continues to reject unsupported sources rather than silently falling back.
- Updated stale Optimizer 2.2 report assertions to Optimizer 2.4 staged-pipeline wording.
- Reworked brittle source-string budget tests to validate the current `optimizer_search_plan` behavior: effort-scaled reconnaissance ceilings, protected correctness/discretionary partitioning, and authoritative validation ceilings.
- Added pure policy regressions for turns with and without enabled physical winding geometry. Dependency-light optimizer/candidate/policy/packaging gate passes 131/131 tests in this environment; full StudioAdapter/UI coverage still requires the pinned Magpylib/PySide6 runtime on the normal development machine.

## 0.14.0.311 — Optimizer 2.4 directional-search completion (Phase 8)

- Completed the Optimizer 2.4 user-facing transparency pass. Optimization reports now show protected correctness versus discretionary stage consumption and add separate **Directional Closure — Centreline Reconnaissance** and **Directional Closure — Capable Physics** audits with effective bounds, endpoint state, best scalar value, closure reason, trend, reopen count, and measured feasibility-boundary pairs.
- Updated Search Trace to present the actual 2.4 strategy: mandatory bounds/directional closure and coupled feasibility recovery are correctness work; relation/symmetry extras, Sobol/global skeptics, adaptive basin challenges, and alternate physical anchors are discretionary. Physical refinement now displays bounded/coupled directional closure separately from the global skeptic.
- Updated Optimizer Help and TESTING guidance around the correctness-first invariant: Fast saves time by reducing discretionary exploration, not by leaving an improving scout-visible direction open; a best value at a user bound is explicitly bound-limited rather than implied to be an unconstrained optimum.
- Updated deferred Optimizer 2.3 policy tests for the 2.4 Balanced contract: once an axial winding capability gap activates the racetrack geometry/spacing family under capable physics, every active member must close directionally before Balanced may report completion.
- Added `test_optimizer_24_directional.py` with regressions for monotonic improvement to both bounds, interior turnover/bracketing, source-anchored increment lattices, DRC feasibility-boundary recovery, rejection of solver failures as physical boundaries, racetrack shape/spacing coupling, protected budget partitioning, reporting/UI integration, and a dependency-light large-range microbenchmark.
- Dependency-light optimizer/candidate/packaging gate passes 129/129 tests. Full discovery reaches 393 tests with no assertion failures; 16 modules remain unavailable in this container solely because the pinned runtime dependencies Magpylib/Magpylib Studio, PySide6, and PyGeomag are not installed (one unrelated test is skipped). The complete source tree compiles successfully.
- Directional microbenchmarks in this environment: monotonic 1–1000 closes at the upper bound in 12 evaluations, an interior optimum at 823 closes in 21, monotonic 70–85 mm racetrack geometry closes at 85 mm in 7, and a DRC-limited 83.0/83.5 mm safe/blocked boundary closes in 14 total evaluations. These are algorithmic sanity measurements, not hardware/runtime SLAs.

## 0.14.0.310 — Optimizer 2.4 protected correctness budgets (Phase 7)

- Split reconnaissance and capable-physics stage budgets into a protected correctness reserve plus a separate discretionary exploration allowance. Optional relation-atlas, Sobol, adaptive-basin, alternate-anchor, and skeptic work can no longer consume capacity reserved for mandatory bound/directional/coupled/feasibility closure.
- Added stage-plan schema v3 fields for `correctness_reserve`, `discretionary_allowance`, correctness/discretionary scopes, and the `protected_correctness_plus_discretionary` contract.
- Centreline recursive scouts now charge generic atlas/global/adaptive work to the discretionary lane while raw bound evidence and bounded directional/coupled/feasibility recovery consume the correctness lane. Cache reuse remains free.
- Fast directional/coupled search stops with unresolved closure rather than borrowing from discretionary capacity when its protected correctness reserve is exhausted. Fast may still leave scout-invisible physical freedoms explicitly limited, but it can no longer report convergence with an open scout-visible direction it promised to close.
- Capable-physics refinement now uses the same protected split: one mandatory physical anchor plus directional/coupled/feasibility closure are correctness work; alternate basin anchors and the post-closure global skeptic are discretionary.
- A skeptic-discovered physical basin may still become the best candidate, but it must have remaining correctness capacity for its own closure pass; otherwise the physical effort contract remains unresolved rather than inheriting closure evidence from another basin.
- Search-stage accounting now records planned and consumed correctness/discretionary capacity independently, including exhaustion flags and `correctness_incomplete` execution status when mandatory closure is left open.
- Balanced physical budgeting reserves against the full structural family when a Centreline capability gap may activate coupled freedoms, rather than sizing the protected reserve from the gap coordinate alone.
- Phase 7 intentionally does not run the comprehensive optimizer regression/integration suite; that remains deferred to the final planned testing/reporting phase. Source compilation/package sanity only.

## 0.14.0.309 — Optimizer 2.4 capable-physics directional closure (Phase 6)

- Replaced the capable-physics stage's min/mid/max + Sobol + local-trust-region refinement as the primary search strategy with the same bounded directional-closure engine used by the Fast Centreline correctness floor.
- Balanced now treats every activated capable-physics coupling coordinate (the Centreline capability gap plus its declared coupled freedoms) as a directional-closure obligation; Full applies that obligation to every selected structural freedom. Distinct physical values alone are no longer sufficient completion evidence.
- Physical refinement now evaluates several retained scout basins at capable physics, chooses the best physical anchor, and performs mandatory legal-endpoint/bracket search on each active coordinate using the optimizer's existing lexicographic objective.
- Added capable-physics coupled cycling: when one physical coordinate moves, earlier conditional closures in the same coupling component are archived/reopened and re-searched until a complete cycle is stable or the stage budget is exhausted.
- Reused Phase-5 feasibility-boundary recovery under capable physics. Geometry-only DRC is probed before an expensive field solve, measured DRC/electrical boundaries can close a direction, and model/evaluation failures cannot masquerade as physical limits.
- Reframed the physical Sobol cloud as a post-closure global skeptic. If it discovers a better basin, that challenger must undergo its own directional-closure pass; if budget exhaustion prevents that pass, the winning basin is deliberately marked directionally unresolved instead of inheriting closure evidence from another basin.
- Added separate `physical_directional_closure`, pass history, directional evaluation counts, global-skeptic counts, and feasibility-boundary counts to multi-fidelity optimizer telemetry while preserving the flat candidate registry and authoritative queue.
- Phase 6 intentionally leaves correctness-budget formalization, user-facing closure reporting, and the comprehensive integration/regression suite to the remaining planned phases. Source compile/package sanity only; no full test run was performed.

## 0.14.0.308 — Optimizer 2.4 feasibility-boundary closure (Phase 5)

- Integrated the existing prepared geometry-only DRC probe directly into Fast bounded directional search, so physical feasibility is part of directional closure rather than a later post-search veto.
- Added `optimizer_feasibility_boundary.py`: when a legal user endpoint is rejected by a measured hard constraint, the optimizer bisects on the actual increment lattice (or continuous interval) to locate the nearest safe-side feasible boundary and scores that real boundary candidate.
- Added explicit constraint classification to directional observations. DRC/electrical failures may define a feasibility boundary; solver/model/evaluation failures never masquerade as physical boundary evidence.
- A direction whose best legal point lies on a measured hard boundary now closes as `feasibility_boundary` instead of being labelled an arbitrary interior optimum or left silently unresolved. Interior optima can also close normally once a measured feasibility boundary supplies the previously missing side of the bracket.
- Feasibility-boundary evidence is stored per lower/upper side in the directional-closure record and survives into reopen history when a coupled coordinate later invalidates the conditional slice.
- Coupled Phase-4 cycles naturally repolish the other dimensions around a recovered feasibility-boundary incumbent, then reopen/recheck the constrained axis if the coupled geometry moves.
- Added rejection-payload caching so active electrical-limit failures can be distinguished from generic evaluation failures when directional search is running with electrical enforcement enabled.
- Added Fast/scout telemetry for feasibility-boundary axes, boundary evaluations, safe/infeasible values, and recovery use. Boundary recovery is treated as correctness-floor work; Phase 7 will formalize its separate non-discretionary budget reserve.
- Integration/regression testing remains intentionally deferred until the final Optimizer 2.4 phase; this phase received compile/package and dependency-light helper sanity only.

## 0.14.0.307 — Optimizer 2.4 coupled directional closure (Phase 4)

- Added coupling topology derived from optimizer capability metadata instead of geometry-kind whitelists.
- Fast directional closure now revisits connected structural freedoms cyclically until the whole coupled family is stable or its current search budget is exhausted.
- Conditional 1-D closure is explicitly reopened when a coupled coordinate moves; prior slice evidence is archived instead of being treated as globally valid.
- Same-family geometry dimensions are coupled automatically, so racetrack end diameter and straight length are revisited together with pair spacing.
- Added coupled-cycle telemetry to the design-space map and propagated it through the multi-fidelity scout result for later reporting/convergence policy.
- This phase intentionally does not add the final regression/integration suite; testing remains deferred to the planned final phase.

## 0.14.0.306 — Optimizer 2.4 Fast directional-closure integration (Phase 3)

- Wired the Phase-2 bounded directional-search engine into the recognized Fast linked-coil engineering funnel. Scout-visible scalar geometry freedoms are now derived from the actual structural-variable list rather than a hard-coded relation whitelist.
- Fixed the racetrack omission that motivated Optimizer 2.4: `racetrack_end_diameter` and `racetrack_straight_length` now receive the same mandatory bounded directional search as other scout-visible dimensions, with path shape searched before pair spacing.
- Restored unconditional raw lower/upper endpoint insurance in the Fast design map. Recognizing a two-coil assembly no longer suppresses generic endpoint evidence; cached duplicates are naturally reused by the later directional pass.
- Each directional axis now evaluates the real optimizer objective through the existing candidate cache, carries the best candidate payload forward as the next funnel incumbent, records bound/turnover/trend evidence in `multi_fidelity.directional_closure`, and exposes per-axis diagnostics in the design-space map.
- Directional axes are capability/variable driven rather than topology-whitelisted. Orientation and shared translation retain their existing target-informed Fast stages in this phase; Phase 4 will add coupled revisit cycles so later shape/spacing changes cannot leave earlier closure evidence stale.
- Recursive Centreline scout directional evidence is propagated back into the parent optimizer contract, while scout-invisible freedoms remain unresolved for later capable-physics stages.
- Added Fast directional accounting fields (axis count, closed/unresolved count, evaluation count and per-axis summaries) and synchronized the latest directional-closure plan before optimizer returns.
- Integration/regression testing remains intentionally deferred until the final Optimizer 2.4 implementation phase; this phase received compile/package sanity only.

## 0.14.0.305 — Optimizer 2.4 bounded directional search primitive (Phase 2)

- Added a reusable objective-agnostic bounded 1-D search engine for structural freedoms.
- Legal lower/upper endpoints, source value, and incumbent are mandatory observations before directional closure is possible.
- Interior optima are adaptively bracketed/subdivided to the increment lattice (or continuous tolerance) instead of being accepted from a local polish.
- Bound-winning searches close explicitly as lower/upper bound limited; infeasible endpoints remain unresolved for the later feasibility-boundary phase.
- Directional-closure records now retain scalar observation history, tested values, best observation, and the Phase-2 search policy.
- Search execution remains intentionally unwired from the Fast/Balanced/Full funnel until Phase 3.

## 0.14.0.304 — Optimizer 2.4 directional-closure foundation (Phase 1)

- Added `optimizer_directional_closure.py`, a dependency-light Optimizer 2.4 bookkeeping layer that separates **search correctness/closure** from the existing model-fidelity/observability contract. A freedom can now be fully visible to Centreline while still remaining explicitly unresolved because its improving direction has not been closed.
- Every bounded structural optimizer freedom now receives a directional-closure record containing requested/effective bounds, increment lattice, source value, lower/upper endpoint evidence, observation summary, best-observation placeholder, trend state, bracket state, and an explicit closure reason.
- New structural freedoms default to `unresolved`; local polish, Sobol sampling, or an interior incumbent do not imply closure. Accepted future closure reasons are interior turnover, lower/upper bound best, measured feasibility boundary, objective saturation, proven redundancy, or a genuinely fixed range.
- Added lattice-aware bound handling matching the optimizer's source-anchored increment convention, so future bound closure refers to the actual legal endpoint on a 0.5 mm / integer / other requested search lattice rather than an unreachable numeric limit.
- Added pure helpers for recording tested endpoints, storing directional trend evidence, explicitly closing a coordinate, and recomputing plan-level unresolved/bound-limited/feasibility-limited summaries. These are intentionally not yet wired into the search algorithm; Phase 2 will provide the real bounded one-dimensional directional search.
- `StudioAdapter` now initializes and carries `multi_fidelity.directional_closure` beside `freedom_coverage`. No Fast/Balanced/Full search behavior or convergence decision is changed in this phase; integration/regression testing is deferred until the final implementation phase as planned.

## 0.14.0.303 — Optimizer authoritative ranking/early-stop fix

- Fixed a Fast-search regression where the Current Sheet promotion stage could correctly rank a better candidate first, then the protected source-baseline/natural-clear insurance pass moved those protected rows back ahead of the physical leader before authoritative validation. With a decisive promotion rank, Fast could therefore validate only the baseline and stop early even when a better physically screened candidate was already known.
- Protected authoritative candidates are now treated as insurance rather than ranking overrides: after a capable physical ranking exists, its leader remains in authoritative slot 0, protected/source/boundary champions are inserted immediately behind it, and the remaining ranked queue keeps its established order. The historical protected-first behavior is retained when no higher-fidelity ranking exists.
- Tightened progressive authoritative stopping so a decisive middle-fidelity result may stop early only after the actual higher-fidelity leader survives authoritative validation. A surviving insurance candidate cannot end the vet if the ranked leader failed or was not tested.
- Added `optimizer_authoritative_policy.py` with dependency-light queue-ordering and progressive-stop helpers, plus regressions reproducing the observed 75%-coverage promoted candidate versus 66.7%-coverage protected baseline case.
- Validation in the packaging environment: the optimizer/candidate-registry Optimizer 2.3 gate and packaging/static gate pass (117 dependency-light tests total), and the complete Python source tree compiles. Live Current Sheet execution still requires the pinned Magpylib/PySide6 runtime.

## 0.14.0.302 — Optimizer explicit-model telemetry fix

- Fixed a Phase-5 fidelity-guard regression that falsely rejected valid Current Sheet/Bundled/Exact optimizer candidates when current was solved through temporary zero/basis adapters. The parent candidate adapter had no direct field-model report, so valid physical evaluations were misdiagnosed as silent Centreline fallback.
- Current-basis construction now preserves the `coil_modeling` reports emitted by the actual zero-current and per-current-group basis field solves and carries that evidence with the solved candidate.
- Strict optimizer model validation now checks the aggregated reports from the field solves that actually contributed to the candidate instead of relying on the parent adapter's last report. A missing report remains a hard failure, and any genuine requested-model fallback in any contributing solve is still rejected.
- Added dependency-light regressions for successful multi-report Current Sheet evidence, true fallback rejection, and missing-evidence rejection, plus static integration guards confirming solved candidates pass their collected telemetry into the strict fidelity check.
- Validation in the packaging environment: 112 dependency-light optimizer/candidate-registry/packaging tests pass and the complete Python source tree compiles. Full runtime optimizer execution still requires the pinned Magpylib/PySide6 environment.

## 0.14.0.301 — Optimizer 2.3 regression/validation pass (Phase 7)

- Added `tests/test_optimizer_23_policy.py`, a dependency-light Optimizer 2.3 regression suite covering the capability table, effort contracts, model-resolution policy, independent stage budgets, candidate-registry fidelity ordering, report/help integration, and the exact racetrack freedom pattern that exposed the Optimizer 2.2 axial-width blind spot.
- Locked the racetrack contract: Centreline must report axial winding length as unobservable; Balanced expands that capability gap to pair spacing and racetrack path geometry while requiring real physical variation of axial width; Full reopens axial width, pair spacing, end diameter and straight length across their permitted physical-search spans.
- Added completion-invariant regressions proving Balanced cannot complete with an unvaried capability-gap freedom and Full cannot complete while any structural freedom (for example pair spacing) has fewer than two distinct capable-physics values. Fast remains allowed to finish with physical-only freedoms explicitly marked limited.
- Added model-policy regressions for Current Sheet racetrack refinement/authority, Exact→Bundled→Exact staging, Auto's stable Bundled distributed-winding floor, Centreline rejection for Balanced/Full winding-distribution gaps, and Current Sheet rejection of unsupported Circle participants instead of silent Centreline fallback.
- Added search-plan and registry regressions proving Fast reserves no movable physical-search budget, Balanced/Full reserve independent physical budgets, Full's completion set contains every structural freedom, and `physical_refinement` fidelity ranks above fixed promotion but below authoritative validation.
- Updated the deferred Qt Search Trace progress regression from the retired Optimizer 2.2 `reality.*` node names to the Optimizer 2.3 `drc.*` / `authority.*` stages.
- Expanded `TESTING.md` with a focused Optimizer 2.3 gate and a release-candidate racetrack acceptance check that verifies Full/current-sheet runs actually vary axial width and spacing and report a satisfied freedom-coverage contract.
- Validation in the packaging environment: the dependency-light optimizer/candidate-registry/packaging gates pass and the complete source tree compiles. The full application suite still requires the pinned Magpylib/PySide6/PyGeomag runtime described in `requirements.txt`; environments without those packages will report import errors rather than application-test failures.

## 0.14.0.300 — Optimizer 2.3 user-facing fidelity transparency (Phase 6)

- Reworked the Optimizer Limits-page **Fast / Balanced / Full** descriptions to match the new effort contracts. The selected Preferences coil model is now named directly in the effort hint, and the Review page states the staged fidelity route before a run begins.
- Rebuilt **Search Trace** around the actual Optimizer 2.3 pipeline: freeze/validate → Centreline reconnaissance → geometry/DRC feasibility → capable-physics refinement when required → authoritative validation. Physical and authoritative stages display their resolved coil models instead of the old combined “promotion/final vet” stage.
- Search Trace now derives its pre-run stage ceilings from the same capability, effort, model-policy and search-plan helpers used by the optimizer itself, and completed-run telemetry is reconciled back into the new DRC/physical/authority trace nodes.
- Physical-refinement progress now distinguishes retained-basin anchors, multi-basin physical exploration and local physical adaptation/convergence rather than reporting the entire movable search as one opaque stage.
- Added a prominent **SEARCH / FIDELITY CONTRACT** section to copied optimizer reports showing the planned/executed reconnaissance, physical-refinement and authoritative stages, their models, ceilings and effort scope.
- Added a per-parameter **FREEDOM COVERAGE** report section. Every selected freedom now reports Centreline scout visibility/status, physical-search model/status and distinct-value count, authoritative-model visibility, and final completion state. This makes limited or unresolved freedoms visible without inferring them from scout counts.
- Updated Fidelity Audit wording so the middle-fidelity row is labelled **Physical** and may represent movable physical refinement or the fallback fixed-coordinate promotion screen.
- Updated convergence/failure wording so runs that actually performed capable-physics refinement no longer describe themselves as a cheap map followed only by a final reality vet, and incomplete effort contracts are identified explicitly.
- Rewrote the Optimizer Help topic for the three-stage fidelity architecture, effort-mode guarantees, Preferences model resolution, winding-distribution capability gaps, Search Trace, and report audits. Added an explicit note that **Skip wizard / Full control** is separate from **Full search effort**.
- Phase 6 remains behavior-focused UI/report/help work; the full optimizer regression, deterministic-worker checks and benchmark tuning remain deferred to Phase 7. All Python modules were compile-checked after the changes.

## 0.14.0.299 — Optimizer 2.3 Preferences/model fidelity policy (Phase 5)

- Added `optimizer_model_policy.py` to resolve the Preferences coil-model choice into explicit **Centreline reconnaissance / capable physical refinement / authoritative validation** models before expensive search work begins.
- **Current Sheet** now remains Current Sheet for physical refinement and final authority when all optimizer-participating sources support the sheet representation; source support is checked explicitly because the current implementation supports closed Polyline coils rather than Circle sources.
- **Exact turns** now uses **Bundled** as the bounded-cost multidimensional physical-search surrogate while preserving Exact turns as final authority. Candidate-specific Exact fallback is treated as rejection rather than silently accepting Centreline physics.
- **Auto** remains dynamic for authoritative validation, but movable physical refinement receives a deterministic capability floor: Bundled whenever winding-distribution effects must be represented, Bundled when available for Full physical refinement of ordinary geometry, otherwise Centreline only when Centreline fully represents the active freedoms.
- **Centreline** remains valid when it actually represents all freedoms being searched, but Balanced/Full now reject the run up front when a required capability-gap freedom (for example axial winding length) needs winding-distribution physics that Centreline cannot represent. Fast may still intentionally leave such freedoms limited under its effort contract.
- Added source-aware optimizer model support metadata covering physical-winding availability, Current Sheet path eligibility, Exact turn-source limits, and per-model reasons for unsupported coils. Explicit requested models may no longer silently fall back to Centreline on participating optimizer coils and still be described as authoritative.
- Added strict candidate-level fidelity checks for physical-refinement, fixed-coordinate promotion, and authoritative validation. If candidate geometry causes a requested explicit model to fall back, that candidate is rejected instead of being scored with different physics.
- Search-stage plans now carry the resolved physical model/reason and authoritative mode, and optimizer reports include a concise model-fidelity-policy line alongside the effort contract.
- Phase 5 keeps UI/help wording cleanup for Phase 6 and the full optimizer regression/benchmark suite for Phase 7. All Python modules were compile-checked and the pure model-policy scenarios were smoke-checked; the container does not provide Magpylib, so runtime StudioAdapter smoke execution was not available here.

## 0.14.0.298 — Optimizer 2.3 effort contracts and Full-search completeness (Phase 4)

- Added `optimizer_effort_policy.py` to make **Fast / Balanced / Full** materially different search contracts instead of only different cheap-map budgets.
- **Fast** now explicitly remains reconnaissance + authoritative verification: it reserves no movable physical-refinement budget and may finish with Centreline-invisible freedoms marked **limited**, rather than implying they were fully optimized.
- **Balanced** now requires every Centreline capability-gap freedom to be actively varied under capable physical physics before the run may report convergence. Declared coupled coordinates are allowed to move with those gaps, while fully Centreline-represented freedoms do not independently require physical reopening.
- **Full** now reopens **every selected structural freedom** in the capable-physics stage. All structural freedoms receive min/mid/max insurance and full-bound low-discrepancy physical exploration across retained scout basins instead of treating the Centreline optimum as fixed context.
- Added a hard effort-completion invariant based on actual per-freedom physical distinct-value counts. Full cannot report `converged=True` unless every selected freedom is physically explored, analytically inner-solved, or explicitly proven redundant; Balanced applies the same rule to its capability-gap freedoms.
- Authoritative finalists are still returned when an effort contract is incomplete, but the run is marked non-converged and names the unresolved freedom IDs instead of silently claiming completion.
- Added effort-scope/completion metadata to `multi_fidelity`, `search_stage_plan`, and the returned `convergence` payload, including active/full-span/required physical freedoms, limited freedoms, unresolved freedoms, and completion status.
- Fixed a Phase-3 accounting bug where the authoritative-vet cleanup block reset `physical_refinement_seed_count` and `physical_refinement_evaluations` to zero after movable physical refinement had already run.
- Updated optimizer report/search-strategy wording to describe the new Fast reconnaissance, Balanced hybrid, and Full dimensional policies and to surface whether the effort contract was satisfied.
- Phase 4 intentionally leaves final Preferences/model-resolution policy for Phase 5; only syntax/import/helper smoke checks were run, with the full regression suite still deferred until the implementation phases are complete.

## 0.14.0.297 — Optimizer 2.3 capable-physics refinement engine (Phase 3)

- Added a real **movable physical-refinement stage** between Centreline reconnaissance/DRC mapping and authoritative validation. Unlike the old promotion screen, this stage generates new structural settings instead of only re-ranking fixed scout coordinates.
- Added `optimizer_physical_refinement.py` to resolve a provisional capable search model, expand scout-invisible/partial freedoms into their declared coupled structural coordinates, and size the retained multi-basin seed set.
- Physical refinement now evaluates several DRC-feasible scout basins with the existing fixed-candidate worker pool, preserving the bounded inner current solve, physical coil model, DRC, electrical gates, cancellation/pause behavior, and flat candidate registry.
- Capability-gap freedoms receive explicit min/mid/max physical coverage across retained basins, followed by a low-discrepancy Sobol cloud. Missing-physics coordinates span their full user bounds while coupled scout-visible coordinates are allowed to move in a local trust region.
- Added multi-basin local coordinate adaptation under physical physics after the initial cloud. The best physically refined candidates are merged ahead of the original DRC-survivor queue for authoritative validation while the original candidates remain as fallback insurance.
- Physical-refinement observations now use the dedicated `physical_refinement` candidate-registry fidelity tier and record per-freedom **distinct physical value counts**, so Phase 4 can distinguish genuine variation from a fixed-value physical check when enforcing Full-search completeness.
- The older fixed-coordinate intermediate promotion stage remains as a fallback when movable physical refinement is unavailable; it is skipped when the new physical stage succeeds.
- Added stage accounting/report fields for physical model, active/required freedoms, seeds, generated/evaluated/feasible settings, local-adaptation work, sample count, workers, elapsed time, and execution status.
- Updated optimizer convergence/report wording so a run that used movable physical refinement no longer claims that the expensive physics served only as a final verifier.
- Phase 3 intentionally does **not** yet impose the final Fast/Balanced/Full completion semantics; Phase 4 will use the new coverage evidence to make those guarantees explicit.

## 0.14.0.296 — Optimizer 2.3 staged search architecture (Phase 2)

- Split optimizer work into explicit **reconnaissance**, **physical refinement**, and **authoritative validation** stages with independent evaluation ceilings instead of overloading one parent `budget` value.
- Added `optimizer_search_plan.py` to centralize effort-dependent scout budgets, authoritative validation ceilings, and a reserved capable-physics refinement budget for the Phase-3 executor.
- Rewired the automatic cheap-first path to obtain its Centreline reconnaissance budget from the shared stage plan after capability filtering determines the actual scout dimensionality.
- Preserved saved/programmatic `evaluation_budget` compatibility by treating it as the authoritative/legacy exact ceiling; reconnaissance capacity no longer shares or consumes that allowance.
- Added `multi_fidelity.search_stage_plan` and `search_stage_accounting` so optimizer results can distinguish planned/evaluated work by stage. Phase 2 intentionally reserves but does not yet execute physical refinement.
- Kept the older exact-search `budget` alias temporarily for legacy branches so the architectural split can land before Phase 3 extracts/reuses the physical search machinery.

## 0.14.0.295 — Optimizer 2.3 capability foundation (Phase 1)

- Added an explicit optimizer freedom/fidelity capability layer describing which magnetic effects each freedom can change, whether it affects DRC/electrical constraints, and which coil field models can represent it.
- Replaced the Centreline scout's hard-coded axial-winding exclusion with capability-driven observability. Existing Optimizer 2.2 search behavior is intentionally preserved in this phase, but every omitted freedom now carries an explicit reason.
- Classify **axial winding length** as invisible to Centreline and **turn count** as only partially represented: Centreline sees ampere-turn scaling but not turn-dependent winding-pack redistribution. Existing turns/current redundancy detection is retained as a dynamic run-specific annotation rather than being confused with full physical observability.
- Added per-freedom `multi_fidelity.freedom_coverage` and a compact fidelity-capability summary to optimizer results, providing the bookkeeping later phases will use to require real physical refinement before Full can report convergence.
- Reserved `physical_search` / `physical_refinement` candidate-registry fidelity tiers above fixed-coordinate physical promotion and below authoritative validation.

## 0.14.0.294 — Exposure Summary sortable-column spacing

- Widened the default **Exposure summary** columns slightly so every header has room for the active sort arrow without clipping or crowding the label.

## 0.14.0.293 — Duplicate preparation confirmation

- Added a confirmation prompt when **Create new analysis** is pressed and a **Current** cached analysis already exists for the same exposure and target. Choosing **No** leaves the existing cache untouched; choosing **Yes** proceeds with a fresh preparation as before.
- Stale/historical analyses do not trigger the confirmation, since they are not reusable current preparations.
- Updated Exposure Analysis help and regression coverage for the duplicate-preparation safeguard.

## 0.14.0.292 — Exposure Analysis header control row

- Moved **Hide/Show analysis info** and **Open region viewer…** into the shared header's top row, immediately to the right of the page title, giving every Exposure Analysis page the same `<navigation>  Title  [analysis info]  [region viewer]` layout.
- Refined the collapsed state so the complete top row stays visible—including the page title and Region Viewer launcher—while only the detailed target/exposure information and compact preview are hidden.
- Updated Exposure Analysis Help and source-contract coverage for the shared top-row layout.

## 0.14.0.291 — Collapsible Exposure Analysis banner

- Added a shared **Hide analysis info** / **Show analysis info** control to `ExposureAnalysisHeaderPanel`, placed immediately to the left of **Open region viewer…** while expanded.
- Collapsing the shared header now hides the page heading/explanation, target/exposure details, preparation controls/statistics, Region Viewer launcher, and the entire target preview, leaving only the applicable Analysis targets/Home navigation plus the restore control. The outer vertical splitter is resized so the current analysis page receives the reclaimed space.
- Hidden compact previews are now paused/released through the shared header state instead of continuing to hold an active WebGL render; expanding restores the owner's previous preview-active state. The same behavior is available in target selection and every prepared-analysis page.

## 0.14.0.290 — Exposure Summary text and tooltip cleanup

- Removed the remaining **Rows combine…** explanatory paragraph from **Exposure summary**, leaving the exposure/source summary line and table controls to carry the page context.
- Replaced the statistics table's native metric-header tooltip handling with an explicitly wrapped tooltip popup so long descriptions remain fully readable instead of being clipped to a narrow header section on some Linux Qt styles.

## 0.14.0.289 — Exposure Summary layout cleanup

- Reworked **Exposure summary** so the regional statistics table now uses the full lower workspace width instead of sharing a side-by-side layout with filters.
- Added a **Show/hide filters** button beside the tree-expansion controls; the **Filters / highlights** editor now appears beneath the main table and starts hidden by default.
- Removed the redundant explanatory note that said the regional statistics table lived in its own panel on the right, and updated the Help/source-contract text to match the new arrangement.

## 0.14.0.288 — Exposure Analysis source-test cleanup

- Updated the static Exposure Analysis navigation contract to follow the shared `ExposureAnalysisHeaderPanel`, which now owns the **← Analysis targets** button, while retaining the prepared-analysis busy-state check in `exposure_actions.py`.

## 0.14.0.287 — Exposure Analysis header text cleanup

- Reduced the **Exposure target selection** surface-region summary by removing the duplicate sampling-depth text; the editable depth control remains directly below it.
- Removed the explanatory geometric-sampling note and the redundant saved-analysis count/status line from the chooser header, leaving the saved-analysis table as the single source of that information.
- Removed the routine live-geometry note beneath the prepared-analysis preview for current/in-memory analyses while retaining the warning when a saved preparation is stale.

## 0.14.0.286 — Shared Exposure Analysis context header

- Added a reusable `ExposureAnalysisHeaderPanel` used by both **Exposure target selection** and the prepared **Exposure analysis home**/method views. The shared template owns page navigation/headline, explanation text, prominent Target/Exposure identity, compact statistics/controls, Region Viewer launcher, and the whole-target 3-D preview.
- Changed target selection to start at roughly one-third context header / two-thirds target-and-saved-analysis browsing while retaining draggable splitters. The shared header itself starts at the established two-thirds information / one-third preview proportion.
- Prepared analysis views now use the same upper-third layout and visual hierarchy as target selection. Home starts with **← Analysis targets** plus a short “choose an analysis” explanation; method pages reuse the same header with page-specific titles/explanations and **← Analysis home** navigation.
- Kept the existing preparation metadata available inside the shared information area, including current/stale state, region/sample counts, surface depth where applicable, and preparation time, while preserving the persistent live target preview and Region Viewer behavior.

## 0.14.0.285 — Exposure Analysis landing-page wording

- Rename the target-selection heading from **Create exposure analysis** to **Exposure analysis** and replace the introductory copy with a concise explanation of target selection, base preparation, statistical calculations, and automatic caching.
- Promote the selected exposure name to the same heading style as the target name and remove the smaller duplicate exposure/duration summary line.

## 0.14.0.284 — Region Viewer turntable default

- Make Plotly Turntable the default 3-D rotation mode in the full Exposure Analysis Region Viewer as well as the compact target previews. Orbit remains available from the native in-viewer Plotly toolbar.

## 0.14.0.283 — Exposure Analysis selection and preview controls

- Selected LPBA40 brain regions now render at full opacity so the active anatomy is easier to distinguish in both Brain View and the Exposure Analysis Region Viewer.
- Compact Exposure Analysis previews now include Plotly's native reset-camera/Home control alongside Orbit and Turntable.
- Compact previews now start in Plotly Turntable rotation mode while the full Region Viewer retains its existing 3-D interaction defaults.
- Fullscreen PlotView clones preserve the configured default 3-D drag mode.

## 0.14.0.282 — Exposure Analysis 3D controls (Phase 2)

- Enabled Plotly's native in-viewer 3-D rotation controls on compact Exposure Analysis previews. Compact previews use the reduced `rotation` modebar profile, exposing Orbit and Turntable without adding external Qt camera buttons or PNG/export clutter.
- Gave the modeless Region Viewer the full native 3-D Plotly modebar plus the shared Workbench `SCENE VIEW` command strip: Export image, Home, Fullscreen, and Front/Right/Top. Fullscreen preserves the same native control profile and rendering contract.
- Removed the Region Viewer's separate Fit object/Fit region buttons and double-click-to-fit behavior so the viewer now uses the same camera/control vocabulary as Model Editor and other 3-D Workbench views.
- Added a shared `PlotView.save_png_with_dialog()` path so `SceneViewControls` can provide Export image without viewer-specific file-save plumbing. Updated Exposure Analysis Help text and regression contracts for the unified controls.

## 0.14.0.281 — Shared 3D viewer controls (Phase 1)

- Added named native Plotly 3-D modebar profiles to `PlotView`. The existing `standard` profile remains the default, `full_3d` explicitly requests the normal 3-D control set, and the new `rotation` profile is ready for compact anatomy previews that should expose only Plotly's native Orbit/Turntable interaction modes. Native PNG modebar export can now be enabled/disabled independently per viewer.
- Fullscreen `PlotView` clones now inherit the source viewer's modebar visibility/profile, native image-save choice, selection-event policy, live-view publishing policy, and WebGL pixel ratio so fullscreen never silently changes the interaction contract.
- Extracted the Model Editor's `SCENE VIEW` command strip into reusable `SceneViewControls`, preserving Export image, Home, Fullscreen, and Front/Right/Top behaviors while keeping those Workbench commands separate from Plotly's in-viewer native tools. Model Editor now consumes the shared control directly while preserving its existing public button attributes.
- Added source-contract coverage for the shared command bar and native modebar profiles. Phase 2 will apply these shared controls to Exposure Analysis compact previews and the modeless Region Viewer.

## 0.14.0.280 — Region display architecture cleanup (Phase 3)

- Completed the Exposure Analysis display-path cleanup so compact previews and regional anatomy viewers now have explicit, non-overlapping responsibilities. Compact previews require the canonical Model Editor target figure from `StudioAdapter.analysis_target_preview_figure()`; the obsolete custom object-mesh fallback and anatomy-specific vertex-welding/shading path were removed entirely.
- Built-in Region Viewer targets now preserve complete display topology. Authored Surface-region busts and ordinary Measurement targets render every labelled source face regardless of any supplied generic budget, while LPBA40 continues to render all 206,080 atlas-shell triangles through the shared Brain View renderer. Optional deterministic face sampling remains available only for explicit future custom/unknown region providers.
- Removed the old embedded/Region Viewer triangle-budget constants from the normal UI path. The modeless Region Viewer now requests full regional geometry directly; no built-in anatomy target is perforated by individual-face sampling.
- Strengthened renderer regression contracts: Model Editor and Exposure Analysis compact previews share the same `_geometry_mesh_traces()` source for colour, opacity and flat-shading; LPBA40 Maps/Exposure views share the same trace/style helper; Bust and Measurement Region Viewer tests assert full triangle preservation; built-in display branches no longer depend on the generic face sampler.

## 0.14.0.279 — LPBA40 Region Viewer renderer unification (Phase 2)

- Extracted the established LPBA40 atlas-shell presentation from Maps → Brain into a shared dependency-light `lpba40_display` module. Brain View and Exposure Analysis now use the same region trace builder, major-division colour palette, normal/selected opacity values, yellow selection colour, smooth-shading mode, and lighting settings.
- Exposure Analysis no longer applies its generic 60,000-triangle face budget to SRI24/LPBA40 Region Viewer anatomy. The Region Viewer now renders the complete bundled atlas shell set (206,080 triangles across 56 labels), matching the topology used by the Maps Brain selector instead of dropping individual faces.
- LPBA40 Region Viewer selection now follows the Maps Brain selector semantics: selected labels use the shared yellow highlight while non-selected shells retain their normal major-division colours and 0.34 opacity rather than being heavily faded.
- Added regression coverage ensuring LPBA40 bypasses generic face sampling, retains every source triangle, and shares display constants/trace construction with Brain View. Updated the packaging contract to follow the shared LPBA40 display module.

## 0.14.0.278 — Exposure Analysis preview renderer unification (Phase 1)

- Replaced the compact Exposure Analysis target preview's custom anatomical Mesh3d reconstruction with the same canonical Workbench mesh-trace builder used by Model Editor Scene View. Compact previews now inherit the exact scene-object colour, opacity, world transform, flat-shading mode, and object-specific display overlays from the normal scene renderer.
- Added `StudioAdapter.analysis_target_preview_figure()` to produce a target-only Scene View figure and apply only preview presentation cleanup: axes/grid and legend hidden, hover/picking disabled, tight margins, and target-focused camera fitting. Scene visibility does not suppress an analysis-target preview.
- Refactored Model Editor mesh rendering through a shared `_geometry_mesh_traces()` helper so Scene View and Exposure Analysis cannot silently drift into separate mesh presentation implementations.
- Updated both Create Exposure Analysis and prepared-analysis header previews to request the canonical target figure alongside the numerical region scene. The previous custom object renderer remains only as a defensive fallback for external callers and is no longer used by the normal Exposure Analysis workflow.

## 0.14.0.277 — Region viewer presentation cleanup

- Render authored bust Surface regions fully opaque in the Region Viewer during the normal whole-object view; explicit region selection still subdues the non-selected regions for visual focus.
- Replace technical WebGL/resource-release wording in paused Exposure Analysis previews with user-facing status text, including "Preview is disabled while the Region Viewer is active." when appropriate.
- Update the Exposure Analysis Help smoke expectation to its promoted **Maps / Analysis** Help category.

## 0.14.0.276 — Exposure Analysis launcher artwork

- Replaced the temporary Project Manager Exposure Analysis bar-chart placeholder with the supplied final `exposure_analysis.svg` artwork. The existing launcher resource name and Project Manager wiring are unchanged.

## 0.14.0.275 — Exposure Analysis promoted in Project Manager

- Promoted **Exposure Analysis** to a first-class Project Manager application under **Maps / Analysis**, beside Map Creator, with the description **Analyze field exposure over time and across spatial regions.** The existing **Analyze exposure…** action in Exposure Editor remains a contextual shortcut to the same workspace.
- Added direct Project Manager launch behavior: projects with one saved exposure open it immediately; projects with several exposures ask which one to analyze; projects with none direct the user to create/save an exposure first. Project-launched Exposure Analysis windows remain project-scoped and close with their project.
- Renamed the former **Datasets** dashboard component to **Data / Results** and moved **Saved Results** there beside Dataset Manager. Maps / Analysis now summarizes saved maps and Exposure Analysis preparations, while Data / Results summarizes datasets/calibrations plus the total saved-result catalogue.
- Added Exposure Analysis to the Project menu and updated the suite Help, README, testing/build documentation, component shortcuts, accessibility labels, and dashboard regression contracts for the revised application hierarchy.
- Added a temporary bundled `exposure_analysis.svg` launcher icon (simple bar-chart placeholder) so the new application card is visually complete until final project artwork replaces it.

## 0.14.0.274

- Reworked the **Create exposure analysis** chooser to the balanced dashboard layout established during workflow testing. The upper and lower work areas start at an approximately 50/50 vertical split while remaining user-resizable.
- The upper section now starts at a 2:1 split: target/exposure information, sampling-depth controls and saved-analysis status stay entirely in the left panel, while the unsegmented interactive target preview receives the full right panel height. The **Open region viewer…** action moved to the bottom-right of the information panel so it no longer consumes preview space.
- The lower section now starts at a 1:2 split between **Available analysis targets** and **Saved analyses for selected target**, matching the relative amount of information in those tables while retaining their existing action buttons.
- The chooser preview hides its built-in Region Viewer button because the chooser now owns that action explicitly; prepared-analysis header previews retain their existing button and behavior.

## 0.14.0.273

- Fixed the remaining patchy/faceted appearance of compact Exposure Analysis previews. The bundled SRI24 brain and Generic bust STL files repeat vertices independently per triangle, so smooth Plotly shading could not interpolate normals even when the complete surface was rendered. Compact anatomical previews now weld coincident display vertices before rendering, preserving every triangle while creating shared normals and reducing the WebGL vertex payload. Ordinary Measurement objects retain hard-edged scene shading.
- Restored the Create Exposure Analysis target preview to a compact top-banner footprint. The chooser now caps the target overview/preview height (with accessibility-font scaling) so the 3D WebEngine size hint cannot consume most of the window and push the target/saved-analysis tables to the bottom.

## 0.14.0.272

- Fixed the faceted/perforated appearance of compact Exposure Analysis target previews. The preview had been enforcing the 20,000-triangle regional display budget by selecting individual faces from the object's connected outer shell; omitted faces therefore appeared as holes and made the brain/bust look noisy even though smooth Mesh3d shading was enabled.
- Compact previews now render the complete ordinary object surface, matching the geometry shown by the Model Editor. The display-only triangle budget remains in place for the dedicated Region Viewer, where the much heavier per-region anatomy scene benefits from decimation. Numerical Exposure Analysis geometry is unchanged.

## 0.14.0.271

- Simplified Exposure Analysis anatomy previews: compact previews now show the target's ordinary unsegmented outer scene surface using its normal object colour/opacity. Authored Surface-region colours and LPBA40 region shells are reserved for the dedicated Region Viewer, where they are useful for browsing/highlighting.
- Removed the Plotly 3D grid/axes from both compact anatomy previews and the Region Viewer, including Home/Fit operations that previously re-enabled axis visibility after the initial axes-free figure was created.
- Extended `AnalysisRegionScene` with a separate object-surface representation. Surface-region and Measurement targets reuse their normal object mesh; SRI24/LPBA40 targets use the actual transformed SRI24 brain outer shell for compact previews while retaining the per-region atlas shells for Region Viewer selection.
- Fixed the Saved Results live-refresh regression introduced by the polling removal. Discrete saved-result catalogue changes now refresh an already-open Saved Results manager immediately, while broader model/exposure/waveform/dataset edits remain event-driven and debounced; periodic idle polling is still absent.

## 0.14.0.270

- Fixed high idle CPU usage and rhythmic UI stalls in **Saved results**. The manager no longer polls every two seconds and recomputes live Exposure Analysis dependency fingerprints on the GUI thread.
- Saved Results refresh is now event-driven: model, exposure, waveform, dataset, map, snapshot, and analysis-cache changes schedule one coalesced refresh; the explicit **Refresh** button and refresh-on-open remain available.
- Rapid project edits are debounced into a single Saved Results refresh, avoiding repeated tree rebuilds and expensive current/stale checks.

## 0.14.0.269 — Region Viewer performance and WebGL resource cleanup

- Optimized Exposure Analysis anatomy previews for sustained interactive use. The renderer no longer maintains Plotly Mesh3d hover/pick buffers for region geometry because browsing is tree-driven; per-triangle hover and 3D click picking are disabled while region highlighting remains available through the existing tree/analysis-selection path.
- Disabled unnecessary live-camera publication and selection event plumbing in `AnalysisRegionPreview`. Orbit/zoom remains fully interactive inside Plotly, but camera motion is no longer serialized through QWebChannel/Python when no Workbench consumer uses it.
- Added a preview-specific 1× WebGL pixel ratio to avoid rendering anatomy scenes at high-DPI framebuffer resolutions that multiply fill-rate cost without materially improving these small/contextual views. Existing PlotView users retain their previous defaults.
- Reduced display-only triangle budgets from 30,000 to 20,000 for embedded previews and from 100,000 to 60,000 for the modeless Region Viewer. Deterministic per-region allocation still guarantees that every visible atomic region is represented; numerical Exposure Analysis geometry is unchanged.
- Added explicit WebGL suspension for inactive embedded previews. Hidden chooser/saved-analysis pages now purge their Plotly scene instead of retaining GPU buffers, and the active compact preview is temporarily released while the larger Region Viewer is open. Closing the Region Viewer restores only the currently visible embedded preview.
- Added `PlotView.release_rendered_figure()` for reusable browser surfaces that need to free GL resources without destroying their QWebEngineView, plus source-contract coverage for the lightweight interaction path and preview lifetime/resource rules.

## 0.14.0.268 — Region Viewer synchronization and polish

- Completed the final Region Viewer/preview synchronization pass. Exposure Analysis method-page region changes now flow one way into the persistent header preview and any already-open Region Viewer, while Region Viewer tree browsing remains visual-only and never changes the numerical analysis selection or starts calculations. Returning to Exposure Analysis Home restores the whole-object visual presentation.
- Added explicit reusable selection control to the modeless Region Viewer. Reopening the same unchanged target updates the selected/highlighted region without rebuilding the WebGL scene or discarding the current camera; a changed mesh/transform/region scene is detected through the deterministic display-scene identity and is refreshed normally.
- Made explicit Region Viewer opens resolve current live model geometry rather than blindly trusting an older compact-preview scene. If the target was edited, the viewer refreshes; if it was deleted or is otherwise unavailable, the viewer clears stale display geometry and reports that the saved numerical analysis remains readable even though live display geometry is unavailable.
- Clarified live-versus-frozen provenance for saved preparations. Prepared-analysis headers now identify the preview as live model geometry, and stale saved preparations visibly warn that the current display may differ from the geometry used for their frozen numerical results. The same context appears in the modeless Region Viewer.
- Kept viewer exploration and analysis control intentionally separate: analysis selection can synchronize the viewer, but viewer tree clicks have no callback into Exposure Analysis. This prevents anatomical browsing from accidentally queuing analyses or changing the result being examined.
- Tightened compact-preview sizing for enlarged accessibility fonts while retaining Qt 6 device-independent scaling, and retained application-wide Light/Dark theme forwarding through the shared `AnalysisRegionPreview`/`PlotView` path. Updated Exposure Analysis Help for synchronization, stale/deleted-target behavior, live-geometry provenance, and unchanged-target camera reuse.
- Added Phase 6 source-contract coverage and updated the earlier Region Viewer contract test for the extended context-aware open API.

## 0.14.0.267 — Modeless Exposure Analysis Region Viewer

- Added a modeless **Region viewer** shared by Exposure Analysis previews. It presents the analysis target/region hierarchy in a tree on the left and a larger interactive 3D view on the right; selecting a hierarchy row highlights the exact same atomic region labels used by the numerical regional-analysis backend while leaving the rest of the object visible but subdued.
- Reused the existing `AnalysisRegionPreview`/`analysis_region_render` stack at the dedicated 100,000-triangle Region Viewer budget rather than introducing a second rendering path. Brain, authored Surface-region, and ordinary Measurement targets therefore keep the same colours, region ownership, decimation and whole-object geometry in compact previews and the larger viewer.
- Added **Fit object** and **Fit region** controls. `PlotView.fit_3d_traces()` fits axis bounds to the selected region traces while preserving the user's current camera direction and leaving the full target rendered; double-clicking a tree row performs the same region fit.
- Added display-only region metadata for the selected tree row: region kind, atomic-region count where applicable, display triangle count, display surface area, display bounds, atlas name, and Surface-region sampling depth. These values explicitly describe visible geometry and remain separate from authoritative exposure sample/volume statistics.
- Wired both the Create Analysis target preview and the persistent prepared-analysis header preview to the Region Viewer. One viewer instance is reused for an entire Exposure Analysis workspace, so switching targets does not spawn duplicate windows; standalone reopened saved analyses likewise own only one viewer. Closing the viewer hides it for session reuse, while closing its owning Exposure Analysis workspace explicitly retires the WebEngine renderer.

## 0.14.0.266 — Persistent Exposure Analysis target preview

- Embedded one reusable `AnalysisRegionPreview` in the prepared-analysis information banner that sits above the Exposure Analysis page stack. The whole Brain, Surface-region object, or Measurement object therefore remains visible on Analysis home and on Summary, Spectrum/FFT, Timing, Threshold, and frequency-weighted pages without creating page-specific renderers.
- Resolved the prepared target scene once from the saved preparation's `target_id` through `StudioAdapter.analysis_region_scene()`. Scene loading is deferred by one event-loop turn so the banner can paint its loading state first; preview failures stay visual-only and cannot block cached/saved analysis browsing.
- Kept numerical and visual lifetimes separate: the persistent preview renders only display geometry, does not regenerate surface shells or exposure field samples, and survives normal page navigation while the page stack changes underneath it.
- Added explicit cleanup/invalidation for the browser-backed header preview when the prepared target view closes, updated Exposure Analysis Help for the persistent visual context, and added source-contract coverage for the Phase 4 integration. The **Open region viewer…** request remains reserved for the Phase 5 modeless Region Viewer.

## 0.14.0.265 — Exposure Analysis target-preview integration

- Embedded the reusable `AnalysisRegionPreview` on the **Create exposure analysis** target-selection page. The selected Brain, Surface-region, or Measurement target is shown as an interactive whole-object 3D preview beside its identity and sampling details before any field preparation is started.
- The target preview resolves through `StudioAdapter.analysis_region_scene()`, so chooser visualization uses the exact same renderer-agnostic region-scene provider already reserved for the later modeless Region Viewer; no target-specific duplicate rendering path was introduced.
- Added a deferred one-event-loop preview refresh with a serial selection guard. This lets the loading state paint before scene construction and prevents a queued older target selection from overwriting a newer selection when users move quickly through the target tree. Preview failures are contained to the visual panel and never block saved-analysis browsing or creation.
- Kept display rendering independent of numerical preparation. Changing the near-surface sampling depth updates analysis provenance but does not rebuild the unchanged authored-surface preview, while selecting another target resolves and auto-fits that target's display scene.
- Added cleanup of the chooser's browser-backed preview when the Exposure Analysis workspace closes, updated Help for the new visual confirmation step, and added source-contract coverage for target-selection integration. The **Open region viewer…** signal remains intentionally unwired until the modeless Region Viewer phase.

## 0.14.0.264 — Reusable Exposure Analysis region preview

- Added a shared dependency-light `analysis_region_render` layer that converts `AnalysisRegionScene` geometry into deterministic Plotly Mesh3d traces. The renderer is intentionally shared by the compact preview and the future modeless Region Viewer so region colours, ownership, hover names, highlight semantics, and display decimation cannot drift between the two UIs.
- Added deterministic per-region display decimation with a 30,000-triangle compact-preview budget and a 100,000-triangle Region Viewer default budget. Every visible atomic region retains geometry even under decimation; source analysis geometry and numerical sampling are never modified. This keeps the approximately 206k-triangle LPBA40 visual scene practical in embedded previews.
- Added `AnalysisRegionPreview`, a reusable QWidget backed by one lifetime `PlotView`. It supports whole-object rendering, automatic Home fitting, orbit/zoom interaction, loading/empty states, theme forwarding, optional region highlighting through incremental scalar restyles, and an **Open region viewer…** request signal without creating the Phase 5 window prematurely.
- Extended `PlotView` with an opt-in `display_modebar` setting while retaining its existing default behavior. Compact analysis previews disable the Plotly modebar and fullscreen controls so the small surface stays visually quiet.
- Added dependency-light renderer regression tests and source-contract coverage for the preview widget. This phase deliberately does not place the preview on Exposure Analysis pages; Create Analysis integration remains Phase 3.

## 0.14.0.263 — Exposure Analysis region-viewer foundation

- Added a dependency-light `AnalysisRegionScene` contract for Exposure Analysis visualization. It carries one world-space triangle mesh, per-face atomic analysis-region labels, the existing `AnalysisRegionNode` hierarchy, object display metadata, bounds, and deterministic geometry identity without depending on Qt, Plotly, WebGL, or the numerical exposure result pages.
- Added a generic scene resolver for all current Exposure Analysis target providers. LPBA40 brains use the bundled placed per-label display shells; authored Surface-region objects use their real transformed object mesh and authored face ownership; ordinary Measurement volumes use the transformed object surface as a single region.
- Region-scene hierarchy keys and integer label ids intentionally match the numerical Exposure Analysis providers, so future compact previews and the modeless Region Viewer can highlight the exact same Face/Scalp/brain/Measurement selection without a second region-mapping system. Authored surface geometry is used for display rather than pretending the inward voxel shell is a literal anatomical solid.
- Added per-label presentation colours for LPBA40 and authored Surface regions, plus region lookup and triangle-mask helpers that the shared preview/viewer renderer can consume directly.
- Added `StudioAdapter.analysis_region_scene()` as the UI-facing provider boundary and dependency-light regression coverage for hierarchy selection, display-geometry identity, Measurement surfaces, LPBA40 shells, and Surface-region precedence. No preview widget or Region Viewer UI is introduced in this phase.

## 0.14.0.262 — Surface-region target semantics

- Made authored Surface regions intrinsic Exposure Analysis region sets regardless of the object's Model Editor scene role. Assigning the Generic bust (or another eligible region-aware mesh) the Measurement role no longer replaces its Face/Scalp/etc. Exposure Analysis target with whole-object Measurement-volume sampling.
- Removed scene role from the Surface-region Exposure Analysis geometry identity, so changing only the object's role does not make an otherwise-current prepared Surface-region analysis stale. Ordinary Measurement-role objects without authored eligible regions continue to use the established whole-volume provider.
- Updated Exposure Analysis Help text to make the distinction between scene Measurement role and authored Surface-region analysis geometry explicit.

## 0.14.0.261 — Surface-region Exposure Analysis validation and cleanup

- Completed the Phase 7 validation/cleanup pass for inward Surface-region Exposure Analysis. The surface-volume lattice now snaps dimension/spacing ratios that differ from an exact integer only by floating-point roundoff before applying `ceil`, preventing rigid object translations from spuriously adding an extra lattice cell and changing represented shell volume.
- Advanced the surface-volume provider identity to version 2 so geometry caches produced by the earlier translation-sensitive lattice rule are not silently reused. The generic regional cache schema remains unchanged; affected saved preparations become stale through their normal geometry/preparation identity checks.
- Bounded the user-facing inward sampling depth to **0.01–30 mm** in the numerical provider, Studio adapter, Model Editor inspector, and Exposure Analysis chooser. The default remains 2 mm. This keeps the feature in its intended near-surface/deeper-head use range while avoiding pathological very-deep cold builds.
- Benchmarked the packaged Generic bust cold shell build in the dependency-light environment: approximately 12 s at 2 mm, 22.5 s at 20 mm, and 29.5 s at the new 30 mm maximum. Existing progress/cancellation reporting remains active during those cache-miss builds; matching shared geometry caches avoid rebuilding the shell.
- Expanded numerical validation with rigid-translation invariance, rotation tolerance, uniform scaling, deterministic repeatability, root/child volume conservation, real progress/cancellation interruption, an analytic curved-shell sphere check, a uniform synthetic field check, and a known spatial-gradient check across opposite surface regions.
- Updated surface-region source documentation and Help to describe the implemented provider rather than the old phased/future wording, and documented the supported depth range.

## 0.14.0.260 — Exposure Analysis persistence test cleanup

- Updated the core Exposure Analysis round-trip regression to reflect the v5 preparation contract introduced by the layered regional cache work. Saved preparations intentionally enrich `exposure_analysis_context` with the generic regional engine/provider description plus geometry, exposure-data, and preparation identities; the test now verifies those fields and confirms the original caller-supplied payload remains otherwise unchanged.
- No runtime Exposure Analysis or cache behavior changed in this patch.

## 0.14.0.259 — Surface-region Exposure Analysis UI and workflow

- Finished the Phase 6 user-facing workflow for near-surface Exposure Analysis. Region-aware geometry now exposes a persistent **Exposure sampling depth** control directly in the Model Editor Surface regions inspector, with explicit wording that the inward layer is a geometric field-sampling volume rather than a tissue-thickness or biological-dose model.
- Added the same near-surface depth control to the Exposure Analysis target chooser so users can change the selected bust/surface target without leaving the analysis workflow. Applying a new depth updates the geometry object through normal undoable project operations; earlier saved preparations remain readable but become Stale because their geometry identity no longer matches.
- Reorganized the Exposure Analysis chooser into **Region sets** and **Measurement volumes**, with Brain regions and Surface regions identified separately. Brain and bust/surface targets continue to use the same regional analysis tools rather than separate numerical workflows.
- Added **← Analysis targets** navigation inside analyses opened from Exposure Editor. Users can move directly between complementary near-surface and SRI24 brain preparations in the same Exposure Analysis workspace without closing/reopening the workflow; target switching is disabled while an analysis job is active.
- Exposed prepared near-surface geometry provenance in the region browser. Surface-region tooltips and the active-region summary now report inward depth, sampled volume, surface area, source-voxel population, and retained field-sample count where available.
- Added a **Volume** column to Surface-region Exposure Summary while leaving the column hidden for the existing Brain and Measurement summary layouts. The volume comes from the authoritative weighted shell geometry and is descriptive geometry metadata, not an exposure/dose metric.
- Expanded the surface-region root metadata to carry total sampled volume, total authored surface area, source-voxel count, and retained-sample count so whole-shell rows have the same provenance as individual Face/Scalp/Ear/etc. rows.
- Updated Generic bust, custom surface-region, and Exposure Analysis Help topics for the depth setting, stale-preparation behavior, target switching, region provenance, and surface-summary volume display. Added dependency-light Phase 6 UI contract coverage.

## 0.14.0.258 — Surface-region Exposure Analysis integration

- Integrated the Phase 3 inward surface-region volume provider as a first-class Exposure Analysis target. Closed, unambiguous meshes with authored surface regions (including the Generic bust in its normal Visual role) now appear alongside SRI24 Brain Areas and explicit Measurement-role volumes.
- Preserved the established whole-object Measurement workflow: assigning a region-aware mesh the Measurement role deliberately analyses the complete object volume, while a non-Measurement region-aware mesh uses its authored inward surface regions.
- Routed surface-region targets through the same generic prepared-field basis and downstream Summary, Spectrum/FFT, Timing & transitions, Threshold & duration, and Frequency-weighted screening machinery used by the other regional providers. No parallel bust-specific numerical analysis path was introduced.
- Added a generic hierarchy snapshot to prepared Exposure Analysis context so saved/reopened analyses can reconstruct authored surface-region trees without LPBA40 assumptions. Surface rows use the shared weighted regional metrics and retain their source-shell voxel / retained-field-sample provenance.
- Connected Phase 5 cache provenance to the native Phase 3 surface-volume identity. Surface depth, transformed mesh, authored ownership, provider version, and sampling controls now identify the shared geometry layer before it is combined with the physical exposure-data identity.
- Added persistent per-object surface-analysis depth metadata (default 2 mm) and an adapter operation for changing it. The full user-facing depth editor remains Phase 6 UI work; changing the stored depth already produces distinct geometry/preparation identities.
- Threaded Exposure Analysis progress/cancellation through surface-shell generation so a geometry-cache miss reports work instead of leaving the preparation apparently stalled.
- Updated target browsing, saved-analysis reconstruction, regional summary presentation, and Saved Results sample wording for near-surface regions. Added dependency-light hierarchy/integration regression coverage.

## 0.14.0.257 — Layered regional-analysis cache identities

- Added a project-level shared regional-geometry cache for resolved `AnalysisRegionSet` data. Geometry/sample membership is persisted independently of any exposure or derived analysis page, uses compact binary ndarray storage, and can be explicitly listed or removed without invalidating already prepared Exposure Analysis field bases.
- Added cache serialization/round-trip validation for generic region hierarchies, weighted sample sets, provider metadata, authoritative per-label volumes, and source/retained sample counts. Cached geometry carries both a request identity and a content identity so stale/corrupt membership cannot be reused silently.
- Surface-region inward volumes now compute a pre-build cache identity from transformed geometry, authored ownership, sampling depth, provider versions, and lattice/sample controls. `StudioAdapter.surface_region_volume_set()` reuses a matching shared geometry cache before rebuilding the expensive shell lattice.
- Existing LPBA40 and Measurement regional providers now use the same shared geometry-cache layer, allowing their resolved spatial sample sets to be reused across different exposures while keeping target display metadata refreshable.
- Split Exposure Analysis provenance into independent geometry, physical exposure-data, and combined preparation identities. Derived Summary/Spectrum/FFT/Timing/Threshold/frequency-weighted cache keys remain scoped beneath the immutable preparation identity.
- Advanced saved Exposure Analysis to format v5 (`FWBEA5`). Derived-result cache blobs now carry the preparation identity they belong to and are rejected if attached to a different preparation. Previous v4 Exposure Analysis caches are intentionally discarded on load; project/model data is unaffected.
- Added explicit adapter APIs for listing/deleting shared regional-geometry caches as future cache-manager integration points, and advanced Workbench metadata to v26 for the new project-level cache collection.

## 0.14.0.256 — Inward surface-region volume sampling

- Added a generic `surface_region_volume` provider that converts any closed, unambiguous authored surface-region mesh into deterministic inward volumetric analysis regions at a configurable sampling depth. The normal end-user control is the depth; source-lattice density and retained field-sample limits are bounded internal sampling controls.
- Near-surface membership is evaluated with an interior-only voxel-centre integration lattice. Each accepted source voxel must lie inside/on the closed mesh and within the requested distance of the surface; ownership comes from the exact nearest authored surface triangle, so neighbouring surface regions form one non-overlapping volume partition rather than independent extrusions.
- Added deterministic per-region downsampling for field calculations while preserving source-voxel physical volume through `AnalysisSampleSet` weights and authoritative label volumes, mirroring the existing LPBA40 weighted-sampling strategy. Unclassified surface ownership remains explicitly excluded and its represented volume is reported separately.
- Added per-region metadata for surface area, sampling depth, represented volume, source voxel count and retained field-sample count, plus whole-provider geometry/sampling metadata for later UI and cache inspection.
- Added a transformed-geometry + authored-region fingerprint and a derived-volume identity that includes sampling depth and resolved lattice settings. Changing depth, object placement/scale, region ownership, or sampling resolution therefore produces a distinct derived geometry before Phase 4 cache integration.
- Added `StudioAdapter.surface_region_volume_set()` as the provider boundary for region-aware triangle objects. Exposure Analysis does not consume this provider yet; that integration remains Phase 4.
- Added dependency-light regression coverage for analytic shell volume, inward-only membership, volume-preserving downsampling, depth identity changes, unclassified-surface exclusion, and region metadata.

## 0.14.0.255 — Region-aware surface geometry provider

- Added a dependency-light `surface_region_geometry` provider that converts authored named-face mesh regions into deterministic geometric queries suitable for future derived volumetric analyses. The provider is generic and works with arbitrary region-aware triangle meshes, not only the bundled Generic bust.
- Added exact nearest-triangle distance/closest-point lookup accelerated by a bounding-volume hierarchy. Equidistant seam cases use the lowest original triangle index as a stable tie-break, preserving deterministic surface-region ownership.
- Added certified inside/outside classification for closed manifold meshes using orientation-independent ray parity, plus signed surface depth (positive inward, negative outward) so the next phase can form inward-only sampling shells without depending on triangle-normal direction.
- Surface ownership now keeps unassigned triangles explicitly unclassified, while overlapping authored assignments are marked ambiguous and block derived-volume generation instead of silently choosing a region. Open/non-manifold meshes remain usable for nearest-surface queries but cannot claim interior volume membership.
- Added per-region surface-area metadata, compact readiness diagnostics, and a transient `StudioAdapter.surface_region_geometry()` boundary that resolves the same transformed world-space mesh used by the rest of Field Workbench.
- Added dependency-light regressions covering analytic cube distance/containment, seam tie-breaking, unclassified and overlapping faces, open meshes, on-surface behavior, and the packaged 38k-face Generic bust. The bundled bust validates as a closed, complete five-region partition ready for Phase 3 volume generation.

## 0.14.0.254 — Generic spatial-region analysis core

- Refactored regional field/exposure calculations into a source-agnostic `spatial_analysis` core shared by atlas regions, measurement volumes, and future derived-region providers such as near-surface bust shells. LPBA40/Brain View keeps compatibility wrappers while the numerical engine no longer depends on brain-specific types or function names.
- Added explicit `AnalysisRegionNode`, `AnalysisSampleSet`, and `AnalysisRegionSet` contracts so region providers resolve geometry/sampling once and downstream frame/playback statistics consume the same weighted representation regardless of source.
- Measurement-role objects now use the generic regional types directly and carry authoritative physical region-volume metadata independently of sample density, while preserving the existing arbitrary-object measurement workflow.
- Exposure Analysis preparations now record the regional engine/schema and region-provider identity in their context and dependency fingerprint, separating target geometry from the provider that turns it into analysis regions.
- Deliberately advanced the saved Exposure Analysis preparation/cache format to v4 (`FWBEA4`) so pre-refactor regional caches are discarded rather than interpreted under the new contract; project/model data remains unaffected.
- Added dependency-light regression coverage for generic-region volume handling, LPBA40 compatibility wrappers, and region-provider identity, and updated packaging/cache-version checks for the new schema.

## 0.14.0.253 — Exposure Analysis cache/queue consolidation

- Finished the four-phase saved-analysis cache update with a cleanup/regression pass across selective cache management, frequency-weighted reference queueing, and compact weighted-result persistence.
- Manage cache now reports the exact compressed derived-cache storage reclaimed by the most recent selective/all-derived removal. Per-row byte estimates are deliberately not shown because binary arrays can be shared/deduplicated across configurations; the before/after derived-cache size is authoritative.
- Expanded regressions for selective-delete storage accounting and Undo, queue cancellation/deduplication, cached weighted browsing while other references run, and reopening compact weighted results after their shared Spectrum/FFT cache has been removed.
- Audited Exposure Analysis Help so the cache manager, single-worker reference queue, compact weighted-cache format, and storage-recovery behavior are described together.

## 0.14.0.252 — Compact frequency-weighted derived cache

- Slimmed persisted Frequency-weighted screening results so they no longer duplicate the matching full Spectrum/FFT frequency, bin-RMS, reference-curve, spectral-ratio, or weighted-µT arrays for every region/reference profile.
- Frequency-weighted cache rows now retain authoritative scalar results at full calculation precision, one bounded normalized display trace, and only the 30 strongest spectral rows needed by the results table.
- Stored weighted display traces use float32 display precision while peak search, spectral summation, and all reported scalar values continue to be calculated before compaction at full numerical precision.
- The matching rectangular/detrended Spectrum/FFT result remains a separate shared cache entry and continues to be reusable by later weighting profiles and the Spectrum/FFT viewer. Weighted cached results remain independently viewable even if that shared Spectrum cache is later removed through Manage cache because their strongest-frequency table rows are retained locally.
- Updated Frequency-weighted Help and regression coverage for compact-cache storage, table rendering, and separation of authoritative calculation outputs from display-only cached data.

## 0.14.0.251 — Frequency-weighted reference queue

- Reworked Frequency-weighted screening so each reference profile has an independent action/state for the current sampling configuration: uncached references show **Calculate**/**Queue**, active references show **Running** or **Queued**, and completed references show **Open**.
- Added **Calculate all uncached references** to queue every missing ICNIRP/EU reference for the selected Auto/Custom sampling configuration without running calculations concurrently.
- Cached frequency-weighted results can now be opened and their result-region tree browsed while another reference is running or queued; opening cached data no longer overwrites the active calculation progress state.
- Auto/Custom sampling controls stay locked until the page's pending reference queue is exhausted, duplicate submissions are suppressed, and Cancel clears both the active weighting request and the page's pending reference requests.
- Preserved the target-wide single-worker FIFO: an already-waiting Summary/Spectrum/Timing/Threshold job keeps its place ahead of a frequency-weighting follow-up reference. Matching rectangular/detrended spectrum caches continue to be reused between profiles.
- Updated Exposure Analysis Help and regression coverage for multi-reference queueing, cached browsing during calculation, deduplication, queue advancement, and the weighted result-browser exception to setup-region locking.

## 0.14.0.250 — Selective Exposure Analysis cache management

- Added **Manage cache…** to Saved Results → Exposure analyses for selectively removing individual derived analysis configurations while preserving the expensive prepared field basis and authoritative exposure source.
- The cache manager lists each stored Summary, Spectrum/FFT, Timing, Threshold, and frequency-weighted configuration with human-readable settings plus cached/processing metadata when available.
- Added **Select all**, **Clear selection**, **Remove selected**, and **Remove all derived results** controls. Explicit removals are undoable and automatically remove matching cache metadata without disturbing the parent preparation.
- Saved Results immediately refreshes cached-work counts and preparation/derived/total storage after selective removal.
- Added adapter and UI regression coverage for selective removal, metadata cleanup, preparation preservation, storage/count refresh, and Undo restoration.

## 0.14.0.249 — Final Exposure Analysis test cleanup

- Updated the last stale Exposure Analysis workspace regression to patch the current all-region spectrum calculation entry point (`analyse_regions_spectrum`) instead of the removed single-region helper.
- No production calculation or UI behaviour changed.

## 0.14.0.248 — Exposure Analysis regression follow-up

- Fixed the target-level busy-state refresh when an analysis page starts its automatic first calculation during page construction. The page is now registered before the shared state is refreshed, so region-selection controls lock immediately instead of waiting for the worker's first progress signal.
- Updated the Timing configuration-cache regression to verify the requested threshold through the configuration key and cached result rather than expecting an obsolete threshold-percentage line in the rendered timing summary.

## 0.14.0.247 — Exposure Analysis regression cleanup

- Updated direct analysis-page regressions for the asynchronous first-calculation behavior introduced by the automatic/queued Exposure Analysis workflow. Spectrum, Threshold, and FFT tests now wait for their first background calculation before asserting result-state controls or plot output.
- Made the shared Exposure Analysis navigation wait helper require a stable idle state across event-loop passes so a queued `QTimer.singleShot` follow-up configuration cannot be mistaken for completion.
- Updated the FFT configuration-cache regression to finish the Custom-mode transition before changing the sample count, matching the real enabled-control workflow and avoiding an assertion in the hand-off between two automatic calculations.
- Updated the background-failure regression to mock the low-level worker launch instead of bypassing `_start_analysis_job`, preserving the 0.14.0.245 single-worker scheduler/status path and its region-selection lock.
- Updated Saved Results snapshot-window regressions for the current `_result_windows` lifecycle registry used by `SnapshotManagerDialog`.
- Removed one duplicate `_analysis_busy` assignment in Threshold & duration while reviewing the affected path.

## 0.14.0.246 — Release-readiness documentation audit

- Performed a release-facing documentation pass after the Exposure Analysis workflow/window/queue redesign without changing numerical behavior or project data formats.
- Updated built-in Exposure Analysis Help to match the current side-by-side Exposure Summary layout; the Filters / highlights panel is now described on the left of the regional statistics table rather than above it.
- Updated the durable release-candidate checklist in `TESTING.md` for the explicit saved-analysis chooser, fresh preparation flow, **Calculate all uncached analyses**, single-worker queue, taskbar-visible analysis windows, automatic first calculation, cache reopening, and current configuration-key behavior. Removed the obsolete **Recalculate** and **Exposure analysis hub** wording.
- Updated `BUILDING_WINDOWS.md` with the same current frozen-build acceptance path, including queue controls and shutdown behavior.
- Rewrote the public README around the current Project Manager suite (Model Editor, Map Creator, Saved Results, Exposure Editor, Waveform Editor, Dataset Manager), replaced the empty Windows-download link with Releases-page guidance, and noted that older videos can predate the suite layout.
- Added packaging regressions that keep these release-facing documents aligned with the current suite and Exposure Analysis UI.

## 0.14.0.245 — Single-worker Exposure Analysis queue

- Added **Calculate all uncached analyses** to Exposure Analysis home. It submits one current/default configuration for Summary, Spectrum/FFT, Timing, Threshold, and the currently selected frequency-weighted profile without expanding every alternate setting.
- Replaced parallel derived-analysis execution inside one Exposure Analysis window with a simple FIFO single-worker queue. Opening another uncached method while one calculation is active now queues it instead of competing for CPU and memory.
- Extended the stacked calculation monitor to show queued jobs as well as the active job. Queued jobs can be cancelled before starting; the active job retains Pause/Run and Cancel controls.
- A paused calculation deliberately keeps the one worker slot, so queued work waits until it is resumed, cancelled, fails, or completes. Processing time still excludes time spent paused.
- Updated analysis-home card activity, frequency-weighting status, Help, and lifecycle regression coverage for Running/Queued transitions and Calculate all behavior.

## 0.14.0.244 — Concurrent Exposure Analysis progress controls

- Replaced the single shared Exposure Analysis progress bar with a stacked progress area that keeps one independent row for every active calculation, preventing simultaneous Timing, Threshold, FFT, or other jobs from flickering over the same controls.
- Added per-calculation **Pause / Run** and **Cancel** controls. Pause is cooperative at numerical progress boundaries, just like cancellation, and one paused analysis does not pause other active analyses.
- Added pause-aware processing timing so time spent actually paused is excluded from the calculation processing time stored with the cache.
- Kept progress rows visible while navigating between an analysis page and the Analysis home, and automatically removes each row from the active stack when that calculation completes, fails, or is cancelled.
- Updated analysis-home activity text, Help, and lifecycle/regression checks for concurrent running, pausing, paused, and cancelling states.

## 0.14.0.243 — Taskbar-visible Exposure Analysis windows

- Changed the Exposure Analysis workspace launched from Exposure Editor from a native transient child into a normal parentless `Qt.Window`, so KDE/other desktop environments can show it in the Field Workbench taskbar/window group like Model Editor, Maps, and Exposure Editor.
- Applied the same taskbar-visible top-level behavior to saved Exposure Analysis resources reopened from Saved Results. The launching surface still retains and closes those windows explicitly, so the ownership/lifecycle behavior from 0.14.0.240 is preserved without using native transient parenting.
- Added a shared `configure_application_window()` helper for top-level application windows and reused it from the existing application-window launcher, keeping window flags/modality/persistent settings consistent.
- Updated Exposure Analysis Help and regression checks to distinguish logical lifecycle ownership from native window parenting.

## 0.14.0.242 — Exposure Analysis result workspace cleanup

- Reworked **Exposure Summary** into a side-by-side workspace: the shared Filters / highlights controls occupy a compact left panel while the sortable regional statistics table gets its own full-height panel on the right.
- Removed explicit **Run / Recalculate / Re-run** controls from the Exposure Analysis result pages. Opening an uncached method, or changing a calculation-setting cache key, now calculates that configuration once automatically; exact cached configurations are reused.
- Simplified Frequency-weighted screening to one **Open** action per reference profile. Open uses the cached result when present or performs the first calculation when needed; there is no separate rerun action.
- Removed redundant result-navigation controls from the analysis workflow: no bottom **Back**, explicit **Close**, or **Back to objects** button is shown after an analysis is opened. The persistent **← Analysis home** control handles method navigation and normal window controls handle window lifetime.
- Kept cancellation controls only while real work is in progress, and updated Help/regression expectations for the immutable saved-analysis workflow.

## 0.14.0.241 — Explicit Exposure Analysis chooser

- Reworked the **Create exposure analysis** landing page so selected target/exposure information sits above the chooser instead of consuming the right side of the window.
- Added a second side-by-side **Saved analyses for selected object** table with Name, freshness Status, Cached work, Updated time, and Stored size. The table is filtered to the already-selected exposure and object, so redundant Exposure and Target columns are omitted.
- Split the ambiguous old **Create analysis** action into explicit choices: **Open selected analysis** opens the exact saved cache (including stale historical analyses), while **Create new analysis** always performs a fresh preparation. Double-clicking a saved row opens it as well.
- Fresh preparations are now saved as new analysis resources instead of silently reopening or overwriting a matching current cache. Existing analyses remain available in the chooser and Saved Results.
- Keyed live target views by saved analysis ID so multiple saved analyses for the same exposure/object can be opened and revisited without conflating their caches.
- Removed the now-obsolete current-cache lookup/upsert/full-preparation-replacement adapter pathways; prepared analysis resources are immutable apart from their derived-result cache, rename, and deletion.
- Updated Exposure Analysis Help and regression checks for the explicit object → saved analysis/new preparation workflow.

## 0.14.0.240 — Exposure Analysis window hierarchy cleanup

- Converted Exposure Summary, Spectrum, FFT, Timing & transitions, Threshold & duration, and Frequency-weighted screening from `QDialog` subclasses masquerading as embedded widgets into true `QWidget` analysis pages owned by `ExposureAnalysisTargetView`. Removed their redundant window titles, Close button boxes, dialog rejection paths, and the old `setWindowFlags(Qt.Widget)` embedding shim.
- Made Exposure Editor launch `ExposureAnalysisWorkspace` as a modeless owned auxiliary window instead of calling `exec()`. One analysis workspace is retained per exposure, registered with the project context, and closes with its owning Exposure Editor/project.
- Made Exposure Analysis resources reopened from Saved Results use an owned modeless `ExposureAnalysisWindow` rather than being detached into a parentless top-level window. Other Saved Results viewers keep their existing independent-window behavior.
- Replaced close-veto lifecycle code with cooperative cancellation that does not block owner/application shutdown. Analysis and preparation workers are retained independently until their threads finish, so closing an Exposure Analysis window requests cancellation and closes immediately without destroying a running `QThread`.
- Kept the Frequency-weighted result browser as the one intentional child auxiliary window, now represented explicitly by `FrequencyWeightedResultsWindow`; it is modeless and owned by the Frequency-weighted page. Spectrum-to-FFT navigation now switches pages through a signal instead of launching a nested modal FFT dialog.
- Updated Help and lifecycle/regression checks for the new window/page hierarchy and removed stale dialog-era lifecycle assumptions and close behavior.

## 0.14.0.239 — Exposure analysis regression cleanup

- Updated the Exposure Analysis navigation regressions for the new all-region Exposure Summary table. Tests now validate cached/calculated summary values through the shared regional table instead of the removed one-region text summary widget.
- Removed the stale single-region Exposure Summary mock expectation; Exposure Summary now uses the all-region batch calculation path exclusively.
- Removed the obsolete `WaveformRegionAnalysisDialog` compatibility alias and updated its remaining test to use `SpectrumAnalysisDialog` directly.

## 0.14.0.238 — Exposure analysis home cleanup

- Removed the target-level region tree and selected-region note from the Exposure Analysis home page. The browser now appears only on individual region-specific analysis pages; Exposure Summary continues to use its own full-width regional statistics table.
- Renamed the guided landing page to **Exposure analysis home** and moved the analysis identity directly under the title: **Target** first, then **Exposure**, followed by compact preparation, atlas, duration, region/sample-count, and preparation-time details.
- Kept the 0.14.0.237 compact-footer behavior so completed and cached results do not reserve space for an inactive progress bar.

## 0.14.0.237 — Compact completed-analysis footer

- Hide the Exposure Analysis footer progress bar as soon as a run reaches a terminal state or a saved result is loaded. Completed/cached status and processing-time text remain visible, while the bar is reserved for work that is actively running.

## 0.14.0.236 — Exposure summary regional table

- Rebuilt Exposure Summary around one sortable all-region table instead of a separate region browser plus one-region text block. Mean regional spatial RMS, exposure RMS, peak RMS/time, B-time, and B²-time now sit directly beside each brain/measurement row.
- Added the same compact Brain Areas-style filter/highlight rule grid above the summary table, including numeric conditions, Left/Right rules for LPBA40 targets, Filter versus Highlight actions, and selectable highlight colours.
- Summary runs and saved summary caches are now consumed as the all-region result they already are; changing the selected row only synchronizes the region used by the other analysis pages and never triggers calculation.
- Exposure Summary owns its regional browser, so the target-level sidebar is hidden on that page and restored automatically on the analysis home and other region-specific tools. Brain summaries retain Collapse / Expand / Expand major hierarchy controls.
- Updated Exposure Analysis Help and packaging regressions for the combined summary table and removed the obsolete one-region summary-page expectation.

## 0.14.0.235 — Exposure analysis wizard, Phase 6

- Finished the Exposure Analysis cleanup around one authoritative full-physical-exposure identity per exposure/target. Preview/Full scope fields are no longer part of saved Exposure Analysis records, lookup, Saved Results, or target-workspace state.
- Bumped the derived Exposure Analysis cache/dependency format to version 3. Older derived analysis records are intentionally discarded when loaded while the underlying scene, exposure definitions, models, and other basic project content continue to load normally.
- Removed the obsolete saved-basis reuse/rebuild pathway and its `basis_source` / reused-analysis metadata. A Current preparation opens directly; when preparation is genuinely required it is built cleanly from the current scene rather than adapting an older derived cache.
- Removed the old playback-summary fallback from Exposure Summary so prepared Exposure Analysis data is its sole authoritative source.
- Simplified Saved Results by removing the obsolete Scope column and reporting one full physical exposure duration for every Exposure Analysis record.
- Cleaned Exposure Analysis Help and regression expectations to match the completed object → preparation → fixed-target analysis workflow and the new cache lifecycle.

## 0.14.0.234 — Exposure analysis wizard, Phase 5

- Added a compact **Prepared analysis** overview above every target workspace showing preparation freshness, target/atlas, full physical exposure duration, prepared-region count, sample-point count, and original preparation time when available.
- Separated each method card's persistent **Cache** state from transient run activity. Cached work now remains visibly cached while a rerun is running, cancelled, or fails; the activity line explains what happened without implying that earlier results were lost.
- Simplified home cache ownership to the prepared target/configuration. Region selection only chooses which member of an all-region result set is displayed and no longer participates in cache-status calculation.
- Made cache-persistence failures explicit per affected method. A calculated result remains usable in memory, the card shows **Save failed**, and a local **Retry save** action retries the project-cache write without rerunning the analysis.
- Removed the old global retry-button/status presentation and updated Exposure Analysis Help and regressions for the clearer preparation/cache lifecycle.

## 0.14.0.233 — Exposure analysis wizard, Phase 4

- Replaced the prepared-target analysis table with a guided, explanation-first analysis home based on the pre-rebuild design. Each method now states the question it answers, summarizes the quantities it calculates, and links directly to its literature or guideline basis.
- Restored separate **Characterize the exposure** and **Compare against a reference framework** sections so frequency-weighted screening is visually and conceptually distinct from descriptive exposure statistics.
- Added a compact cache line to every method card. Cache state is now summarized for the prepared target/configuration rather than pretending an all-region calculation belongs independently to whichever brain region is currently selected.
- Kept the Phase-3 fixed-target architecture: the region tree chooses which member of an all-region result set is displayed, while opening a method or changing regions remains calculation-free.
- Removed the old target-view launcher-table plumbing and updated Help/regressions to describe the guided analysis home. Also removed leftover Help text claiming Preview/Full Exposure Analysis records coexist.

## 0.14.0.232 — Exposure analysis wizard, Phase 3

- Split the prepared-object analysis UI into a dedicated `ExposureAnalysisTargetView`. The wizard now embeds that QWidget directly instead of coercing a QDialog into a stacked widget and hiding its window controls.
- Kept object selection/preparation in `ExposureAnalysisWorkspace`; once created, one target has a fixed analysis view with its own regional selection and derived-analysis pages. Returning to **Back to objects** leaves that target view intact for reuse.
- Brain targets keep their LPBA40 region browser inside the target view. Single-result Measurement targets now omit the pointless one-item region sidebar and use the full analysis width.
- Added a thin `ExposureAnalysisWindow` shell for opening prepared resources from Saved Results, reusing the exact same target view and cache-persistence owner as the wizard.
- Removed the obsolete Scripted-playback Analysis button/path and the old `PlaybackAnalysisDialog` / `PlaybackSummaryDialog` naming. Exposure analysis is no longer represented as a playback feature anywhere in the active UI architecture.
- Renamed the Help topic to **Exposure analysis** and updated structural regressions for the target-view architecture.

## 0.14.0.231 — Exposure analysis wizard, Phase 2

- Removed the legacy `ExposureAnalysisSetupDialog` completely. **Create analysis** now transitions directly from the object chooser to an embedded preparation page in the Exposure Analysis workspace.
- Preparation now freezes the current model, starts the existing cancellable field/timeline worker, reports progress in-place, saves the prepared analysis, and opens that target automatically when the worker finishes.
- Current preparations still bypass calculation and open immediately. New preparations no longer construct, lock, auto-click, or dispose a second configuration dialog behind the wizard.
- Added explicit preparation cancellation and close-during-preparation handling to the workspace, including cancellation/completion race protection before publishing a prepared cache.
- Reduced `exposure_analysis.py` to the preparation engine and target helpers by removing obsolete setup-window UI code and all Qt dependencies; the workspace now owns its small thread worker.
- Updated regression/package checks for the new single-window preparation architecture.

## 0.14.0.230 — Exposure analysis wizard, Phase 1

- Replaced the combined object/region/analysis landing page with a dedicated object chooser. It lists only eligible SRI24 brains and Measurement-role volumes; selecting one shows target metadata without starting magnetic-field calculations.
- Removed Preview-duration controls from the active Exposure Analysis workflow. New analyses are full physical-exposure analyses; the retained short display trace remains an internal implementation detail.
- Removed the obsolete Preview-scope preparation path and the old brain-specific preparation alias; the active Exposure Analysis preparation API now has one full-exposure path for both brain and Measurement targets.
- Added target-specific chooser details: LPBA40 atlas/region information for brains and geometry, volume, sampling mode, and sample-point count for Measurement volumes, plus Current/Stale/Not created preparation state.
- Renamed the primary action to **Create analysis**. Current preparations open immediately; otherwise the existing cancellable preparation worker runs and then opens the selected target's analysis workspace.
- Stopped mirroring prepared brain regions into the object chooser. Once an analysis is opened, its own region tree is used, with **Back to objects** returning to the flat target list. Removed the obsolete duplicated-region and Preview-scope workspace plumbing.
- Updated Exposure Analysis Help and regressions for the new wizard entry flow.

## 0.14.0.229 — Exposure analysis rebuild, Phase 5

- Added Cancel run for background analyses, retaining completed results and rejecting cancelled or partially completed batches, including a cancellation/completion race.
- Closing a running workspace cancels safely before closing. Preparation waits for its worker thread to finish before disposing of its dialog; auxiliary weighted-result windows close with their parent.
- Kept progress visible on the analysis table, throttled worker progress updates, and bounded progress-bar ranges for very large sample counts. Single-volume weighting in the workspace now runs in the background too.
- Merge independently calculated configurations from multiple windows of the same saved analysis. Save failures keep the in-memory result, show a persistent message and offer Retry saving results.
- Recheck changed, cleared and deleted cached records when revisiting a target or region, retaining stale historical records in Saved Results.
- Updated Help and simplified preparation text. Completed the exposure-analysis rebuild checklist and removed it from the source package.
- Added worker cancellation, queued completion, close-during-run, retry, concurrent-window persistence and stale/deleted-cache lifecycle tests.

## 0.14.0.228 — Exposure analysis rebuild, Phase 4

- Analyze exposure now opens an object tree for SRI24 brains and eligible Measurement-role volumes. Prepare selected objects explicitly and browse several targets in one workspace, with separate tables, settings and saved records.
- Added dedicated preparation for generated volume samples and custom XYZ points through the existing volume sampler. All five analysis types use the same regional engine; custom-point RMS is labelled separately from a whole-volume estimate.
- Preview and Full scopes coexist per target, reusing current field calculations where possible. Runs cover the selected object's regions; overlapping objects are not combined.
- Measurement cache dependencies include geometry, world placement and active sampling settings. Saved names include the target; measurement records no longer display an LPBA40 label.
- Added direct-field parity, generated/custom sampling, multi-object and mixed-brain navigation, saved-record roundtrip and target invalidation tests. Updated Help for the new workflow.

## 0.14.0.227 — Exposure analysis rebuild, Phase 3

- Threshold runs now cover all prepared regions in one shared source pass, retaining bounded excursion previews and exact repeated-cycle summaries.
- Threshold values and frequency-weighting profile/sampling choices each save one all-region configuration. Region changes browse saved results; different settings coexist and failed runs preserve earlier results.
- Frequency weighting reuses an exact matching saved spectrum, or calculates and saves it for reuse by other profiles and Spectrum/FFT. Profile-specific filtering and peak searches still run separately.
- The weighted results window labels its own Auto/Custom sampling choice and keeps that choice while setup controls change. Updated all-region guidance and Help.
- Added parity, repeated-boundary, spectrum-reuse, settings-isolation and grouped persistence regressions. No cache migration is required.

## 0.14.0.226 — Exposure analysis rebuild, Phase 2

- Summary, Spectrum/FFT and Timing now calculate every prepared region in one explicit Run. The tree then browses those results without recalculating.
- Share source replay for summary, current FFTs for spectra, and current peaks, pulse timing and derivatives for timing. Regional spatial-RMS calculations retain their own regional moments.
- Save each complete all-region configuration together within its prepared exposure/target/scope record. Different FFT and timing settings coexist; failed runs do not replace previous results.
- Updated cache status, Saved Results configuration counts, Run tooltips and Help. All-region runs use background workers.
- Added numerical parity, shared-processing, packet-boundary, grouped save/reopen and workspace tests.
- Threshold grouping and frequency-weighting lifecycle consolidation remain Phase 3; measurement volumes/multiple targets remain Phase 4.

## 0.14.0.225 — Exposure analysis rebuild, Phase 1

- Replaced landing cards and page-specific region dropdowns with a persistent region tree, selection breadcrumb, and analysis table.
- Embedded analysis pages now wait for explicit Run; opening pages and changing regions/settings only browse saved results or show Not calculated.
- Added Ready/Running/Failed table status, synchronized frequency-weighted result browsing, and a selection lock during background work.
- Removed implementation commentary from summary results and simplified preparation and analysis text; updated Help.
- Numerical engines and calculation scope are unchanged: summary/FFT/timing/threshold remain per-region until Phase 2; weighting already covers all regions.

## 0.14.0.224 — Shared frequency weighting and regional browsing

- Replaced the frequency-weighted region dropdown with a collapsible brain-area tree, also available in the results window.
- Run/Re-run now calculates all prepared regions for one profile and sampling configuration, sharing spectral transforms and weighted current blocks.
- Region selection browses cached results without recalculating; the profile table reports regional coverage and total processing time.
- Save all regional results in one cache update after successful completion; retain bounded processing for long exposures.
- Updated Help and added numerical equivalence, shared-transform, and tree-navigation regression checks.

## 0.14.0.223 — On-demand frequency-weighted analysis

- Opening frequency-weighted screening and changing its options no longer starts calculations.
- Added a reference-profile table with exact-setting cache status, processing time, and Run/Re-run/View controls.
- Cached results open in a separate window with statistics, spectral table, graph, and Back navigation.
- Shortened the introduction and documented the workflow and technical limitations in Help.

## 0.14.0.222 — Release-candidate documentation and cleanup

- Completed a release-readiness documentation pass after the Exposure Analysis/cache refactor. Built-in Help now consistently separates Scripted playback visualization from authoritative Exposure Analysis, removes obsolete Playback-summary/numerical-analysis-timeline guidance for newly generated playbacks, and renames the research-basis topic to **Exposure statistics and research basis**.
- Updated the project overview/workflow, README, Windows release checklist, and testing guide for empty new projects, automatic Exposure Analysis caching, Saved Results cache management, project-to-project waveform import, and the current Scripted/Brain workflow.
- Consolidated the remaining manual release gates into `TESTING.md` so the temporary development `RELEASE_REVIEW.md` can be removed from release sources. The source-release builder now excludes that review file even if it exists locally, and the obsolete-file cleaner removes it from overlay-updated source trees.
- Removed stale documentation implying that a packaged HM/default demonstration scene or bundled `examples/` directory is part of the current release. Focused test fixtures remain explicitly test-only.
- Added packaging/help regressions for the release-review exclusion and current Exposure Analysis terminology. No numerical, solver, exposure, cache-format, or project-format behavior changed in this cleanup.

## 0.14.0.221 — Exposure Analysis storage/serialization cleanup (Phase 12)

- Split each Exposure Analysis resource into an immutable compressed **preparation blob** and a separate compressed **derived-result cache**. Completing another Summary/FFT/Timing/Threshold/frequency-weighted calculation now updates only the derived cache instead of repacking the LPBA40 field basis and authoritative replay source.
- Added packed binary storage for NumPy analysis arrays. Large FFT/frequency-weighted arrays are no longer serialized as decimal JSON, and byte-identical arrays are deduplicated across derived configurations inside the same parent cache before compression.
- Automatic preparation/cache writes remain part of the project dirty state but no longer consume Undo history. Explicit Rename/Delete cache operations remain undoable.
- Added preparation/cache SHA-256 checksums and changed project dirty-state signatures to reference those fixed-size hashes rather than repeatedly embedding potentially multi-megabyte cache blobs. Normal project saves still write the complete cache data.
- Saved Results now reports total compressed storage as separate preparation and derived-cache contributions, plus the packed pre-compression size. Exposure Analysis project format is now v2 and Workbench metadata v25; no migration layer was added for the unreleased earlier cache format.
- Added regressions for split-cache persistence, immutable preparation blobs during derived-cache updates, Undo-history isolation, binary-array deduplication, and catalogue blob hiding.

## 0.14.0.220 — Exposure Analysis cache invalidation audit (Phase 11)

- Reworked the Exposure Analysis dependency fingerprint around canonical physical inputs rather than the complete serialized scene. Expensive caches now stay Current across presentation-only edits such as labels/colours, Sensor changes, display-unit changes, unrelated visual meshes, waveform/exposure display names, notes, and construction-only bobbin/flange changes.
- The fingerprint still invalidates on numerical changes to field-source pose/visibility/parameters, active background fields, driven-coil field/electrical properties, source/load routing and timing, referenced waveform samples/electrical metadata, coil modeling method, or the resolved SRI24/LPBA40 target placement.
- Driven-coil catalogue current is excluded from the dependency key because authoritative exposure preparation explicitly zeros then unit-drives those coils; static non-driven coil current remains a dependency.
- Added dependency-policy version 2 so earlier unreleased caches become Stale rather than being silently reused under the new rules.
- Updated Help to document Current/Stale semantics and repaired the Analysis Hub UI smoke assertion for the current help copy.

## 0.14.0.219 — Exposure Analysis cache management (Phase 10)

- Refined **Saved Results → Exposure analyses** around the automatic-cache workflow. The catalogue now shows a **Cached work** summary alongside exposure, target, scope, freshness, update time, and stored size so users can see at a glance whether a resource contains preparation only or completed derived configurations.
- Added a selectable **Selected analysis cache** detail panel with the exact cached tool/configuration breakdown, number of regions represented, original preparation time, LPBA40 sample count, authoritative/physical duration, and compressed-versus-raw project storage. Multi-selection reports aggregate cached configurations and storage.
- Exposure Analysis records now keep compact cache-management metadata (`cached_result_count`, per-tool counts, cached-region count, preparation time and basis provenance) beside the compressed payload. Saved Results can therefore summarize large analyses without inflating/decompressing every authoritative payload during its periodic catalogue refresh.
- Opening a cache still restores the existing Exposure Analysis workspace without recalculation; optional Rename remains an organizational label only. **Delete cache** now states explicitly that it removes the prepared field basis/source plus every derived configuration under that parent resource, while Undo continues to restore the deletion. Current/Stale historical behavior is unchanged.
- Added/updated source and adapter regressions for the management metadata and cache-oriented Saved Results controls.

## 0.14.0.218 — Cache-aware Exposure Analysis home (Phase 9)

- Made the Exposure Analysis home page reflect the automatic cache directly. Each analysis card now reports the selected region's state as **Not calculated**, **Cached**, or a count of cached parameter configurations; Spectrum distinguishes its default cached spectrum from additional FFT configurations.
- Cache indicators follow the home Region selector and refresh immediately after a derived calculation is written to the parent Exposure Analysis resource or when the user returns from an analysis page. Reserved cache-metadata rows are excluded from the counts.
- Reworked the analysis context banner around the new cache model. It now reports **Preparation cache: Current** or **Stale historical result**, preserves the original preparation processing time, and identifies when the LPBA40 field basis was reused from another current analysis. Preview/Full duration and verified-cycle reuse remain visible on the adjacent scope line.
- Updated Exposure Analysis Help and added UI/source regressions for per-region cache-state counts, preparation provenance, and live home-page refresh. No project-format migration was added; there are no released Exposure Analysis caches requiring compatibility handling.

## 0.14.0.217 — Automatic Exposure Analysis caches (Phases 7–8)

- Replaced the manual **Save analysis…** lifecycle with automatic project caching. **Analyze exposure…** now looks for an exact Current preparation cache using the exposure/model/Brain-target dependency fingerprint plus Preview/Full scope and analysis duration. Exact hits open the Exposure Analysis workspace without rebuilding the LPBA40 basis or authoritative timeline.
- Fresh preparation is automatically upserted as an **Exposure Analysis** project resource as soon as it completes, then the preparation dialog opens the analysis workspace directly. The normal project dirty/save lifecycle is preserved: calculations update the in-memory project and mark it changed, but FWB does not silently write the `.magpy.json` file to disk.
- Exposure summary, Spectrum/FFT, Timing & transitions, Threshold & duration, and Frequency-weighted screening now automatically persist every completed result into their parent Exposure Analysis resource. Settings that change the numerical result remain part of the cache key, allowing multiple regions, thresholds, FFT configurations, reference profiles and sample-count configurations to coexist.
- Derived cache entries now retain calculation metadata including measured processing time. Exact cache hits report **Loaded from cache** and, for newly generated entries, the original processing time instead of rerunning the calculation. Existing pre-0.14.0.217 cache entries remain readable and report that no recalculation was required when timing metadata is unavailable.
- **Recalculate** is now an explicit cache bypass for the selected analysis-page configuration: it recomputes the result and replaces that exact cache entry. Ordinary page opening and parameter navigation continue to prefer matching cached results.
- **Saved Results → Exposure analyses** is now the management surface for these automatically maintained caches. Its timestamp column reflects the latest cache update, and deleting the parent Exposure Analysis discards its preparation payload and all associated derived-result caches.
- Added adapter/update regressions and source checks for exact preparation lookup/upsert, automatic derived-cache persistence, original-processing-time provenance, and removal of the manual Save-analysis action.

## 0.14.0.216 — Frequency-weighted analysis side-by-side viewer

- Moved the **Frequency-weighted regional index** Plotly viewer into a dedicated right-side pane instead of stacking it between the textual result and strongest-frequency table. The result summary and spectral table now share the left pane, giving both the plot and tabular results substantially more vertical room.
- Made the frequency-weighted result area horizontally resizable with a non-collapsible splitter and a 3:2 default text/viewer balance. Narrower spectral-result columns keep the left-side table usable when the Exposure Analysis workspace is resized down.
- Added UI regression coverage for the new left-results/right-viewer layout while preserving the existing calculation, cache, progress-footer, and recalculation behavior.

## 0.14.0.215 — Responsive summary analysis and determinate time-domain progress

- Moved Exposure summary calculation off the GUI thread whenever an authoritative exposure source is present. Opening the summary page now returns immediately and mirrors live progress/timing into the shared Exposure Analysis footer instead of blocking for the full cumulative replay.
- Added deterministic sample-work progress to Exposure summary, Threshold & duration, and Timing & transitions. Verified-cycle analyses report progress against the retained packet samples actually evaluated rather than the potentially huge logical repeat count.
- Timing progress now accounts for its peak-current pass plus regional-slew/drive-threshold work, while Threshold progress tracks regional field evaluation and excursion detection. Replay-recipe analyses use the existing conservative event-aware sample estimate and finish with an explicit 100% completion update.
- Preserved derived-result caching: reopening a completed page still reports **Loaded from cache — No recalculation** instead of rerunning the analysis.

## 0.14.0.214 — Exposure Analysis footer progress and timings

- Moved derived Exposure Analysis page progress out of the analysis body and into the shared bottom-left footer, aligned with the page-level **Back** button. Exposure summary, Spectrum/FFT, Timing & transitions, Threshold & duration, and frequency-weighted screening all use the same footer status area.
- Added a live elapsed-time counter while an analysis is running. The footer now remains visible after completion and reports the measured processing time instead of disappearing as soon as the result arrives.
- Cached derived results now explicitly report **Loaded from cache — No recalculation** in the same footer, including caches restored from a saved Exposure Analysis. Failed calculations retain a stopped-after elapsed time for troubleshooting.
- Kept standalone analysis dialogs compatible with the same status/timing helper while the embedded Exposure Analysis workspace mirrors their state into its persistent footer.

## 0.14.0.213 — Exposure Analysis engine optimization (Phase 6)

- Extended verified physical-cycle reuse to **Preview** analysis as well as Full exposure. Long preview intervals can now use the same transient + repeated cycle + remainder packet model instead of replaying every physical cycle; retained display traces also map repeated timestamps back onto the verified steady-state window.
- Added persistent **regional-basis reuse**: when a Current saved Exposure Analysis has the same exposure/model/Brain-target dependency fingerprint and drive order, a new Preview/Full run reuses its frozen LPBA40 second moments and skips the fixed-scene + per-driven-coil magnetic-field solves. Stale/mismatched analyses are never reused.
- Reworked LPBA40 regional-basis assembly to accumulate exact unnormalised second moments once per atomic atlas label and build parent/child hierarchy rows from those accumulators. This removes repeated full-atlas rescans for overlapping hierarchy rows while preserving the same weighted RMS quadratic coefficients.
- Regional time-domain RMS evaluation now collapses exact repeated current plateaus/states before applying the quadratic field expression, then expands the results back to the original sample timeline. Continuously varying records retain the direct vectorized path.
- Field-basis generation now reuses one isolated zeroed scene adapter while stepping the unit-current coil between basis solves, avoiding a full scene deep-copy/reconstruction for every driven coil. Numerical reconstruction semantics are unchanged.
- Added Phase-6 regression coverage for hierarchy-moment equivalence, exact repeated-current-state reuse, and Preview-scope verified-cycle compression; updated Help with the new conservative reuse rules.

## 0.14.0.212 — Persistent Exposure Analysis results (Phase 5)

- Added first-class **Exposure analyses** project resources. The Exposure Analysis workspace now has **Save analysis…**, which stores the frozen LPBA40 regional basis, authoritative physical exposure source, scope/target context, and any completed derived-analysis cache entries in the `.magpy.json` project.
- Added **Saved Results → Exposure analyses** with Current/Stale status, Open analysis, Rename, multi-delete, compressed-size reporting, and historical reopening. Reopening restores the analysis hub directly without another field-basis solve; already-computed summary/FFT/timing/threshold/frequency-weighted cache entries are restored when they were present at save time.
- Added conservative dependency fingerprints for saved Exposure Analysis results. Exposure protocol/waveform changes, electromagnetic scene or coil-property changes, background-field changes, Brain target changes, or a different coil-field representation mark the result **Stale** without changing its frozen numbers.
- Exposure Analysis persistence is deliberately separate from Static Measurement-volume snapshots and cached Maps viewers. Saved Results and Project Manager counts now include the new resource type, and project resource registries expose it as `exposure_analysis`.
- Updated Help and release notes for the persistent-analysis lifecycle and bumped Workbench metadata to version 24.

## 0.14.0.211 — Exposure Analysis workspace (Phase 4)

- Added **Analyze exposure…** to the Exposure Editor catalogue for the selected saved exposure. Authoritative time-domain analysis is now launched from the exposure workflow rather than Scripted playback.
- Added a dedicated preparation window that freezes the current project model, identifies an SRI24/LPBA40 Brain Areas target, and makes **Preview duration** versus **Full exposure** scope explicit before calculation begins.
- Exposure-analysis preparation runs on a worker thread with visible stage/detail progress and cooperative cancellation. It builds the physical exposure timeline plus fixed/per-driven-coil magnetic-field bases over the same volume-weighted LPBA40 sampling used by Brain View; it does not build animation frames or WebGL playback data.
- Migrated the existing guided analysis interface into the new **Exposure analysis** workspace, retaining Exposure summary, Spectrum/FFT, Timing & transitions, Threshold & duration, and ICNIRP 2010 / EU 2013/35/EU frequency-weighted screening. Region selection and derived-analysis caching remain shared across its persistent stacked pages.
- Added exposure/model/target/scope context to the migrated workspace and cleaned its user-facing wording so the tools describe the physical exposure rather than a paused or looping playback.
- Added an **Exposures** Help category and documented the new separation between Scripted visualization and authoritative exposure analysis. Phase 4 does not yet persist completed analysis sessions; saved first-class Exposure Analysis results remain Phase 5 work.
- Added source/UI regression coverage for the new Exposure Editor action, preparation engine wiring, Help placement, and dedicated Exposure Analysis hub context.

## 0.14.0.210 — Playback warning regression cleanup

- Updated the large-playback warning regression to match the Phase 1–3 scripted-playback cleanup: it now requires the current **Brain Areas metrics** wording and explicitly verifies that the removed **measurement-volume** playback wording does not return.

## 0.14.0.209 — Brain playback metric cleanup

- Kept Brain Areas playback metrics precomputed rather than moving them to a lazy/on-demand model after the Phase 1–2 separation reduced the 2 s / 120 FPS SAM360 Brain build from several minutes to about one minute in desktop testing.
- Optimized the remaining Brain Areas frame pass without changing metric definitions: exact repeated coil-current states are evaluated once and reused across matching frames, and P5/P95 now share one weighted sort instead of sorting each region twice. Near-equal current states are deliberately not merged.
- Cleaned the large-playback warning so it no longer refers to removed measurement-volume playback statistics and instead identifies Brain Areas metrics as the remaining Brain-mode cost.
- Hardened UI smoke-test source assertions with a checked-in-file/AST fallback when `inspect.getsource()` cannot recover source under a particular runner/import environment, fixing the three reported `OSError: could not get source code` failures without weakening the assertions.
- Added Brain-analysis regression coverage for exact repeated-current-state reuse while preserving one output row per rendered frame and normal progress completion.

## 0.14.0.208 — Separate scripted playback from exposure analysis

- Measurement volumes and snapshot actions are now Static 2D/3D features only. Scripted 2D/3D hides the measurement panel/divider entirely, while Brain mode keeps its separate Brain Areas panel and uses Brain-specific show/hide wording.
- Saved-exposure Create playback no longer offers Preview/Full statistics or performs the authoritative full-exposure replay, measurement-volume playback summaries, or frozen analysis-source work as a prerequisite for opening the viewer. The scripted timeline is now prepared only for the requested visual preview.
- Brain scripted playback still computes current-frame Brain Areas metrics for its actual playback frames, but no longer computes or attaches the cumulative playback summary/full-exposure Brain analysis. Progress now distinguishes playback-frame metrics from exposure-analysis samples.
- The existing ICNIRP/spectrum/timing/threshold/frequency-weighting analysis backend and interface are retained for migration into the planned Exposure Analysis workflow, but new scripted playback no longer launches that cumulative analysis path.
- Added regression coverage for playback-only exposure timelines, scripted-viewer panel ownership, render-task plans, and Brain payloads without cumulative exposure summaries.

## 0.14.0.207 — Import waveforms from projects

- Waveform Editor → catalogue Import can now choose a <code>.magpy.json</code> project while the standard Field Workbench waveform JSON encoding is selected. Saved project waveforms fill the existing Waveforms to import list, where any subset can be checked and imported directly.
- Project-waveform import reads only the waveform metadata instead of opening the source model, and preserves each waveform's catalogue name, physical samples, units, electrical source metadata, notes, and authoring recipe. Malformed optional waveform records are skipped consistently with normal project loading.
- Native <code>.fwbwave</code>/<code>.fwbwave.json</code> import keeps its existing behaviour.

## 0.14.0.206 — Recreate dataset point sensors

- Fixed repeated “(saved coordinates)” labels: the missing-sensor suffix is display-only, and older duplicated suffixes are cleaned when datasets are loaded or saved.
- Dataset Manager now offers Create point sensors for the selected dataset. It creates a new named group with one point sensor per saved row, preserving world coordinates and local measurement orientations without changing the dataset or its calibration links.
- Sensor creation is one undoable operation; invalid orientations are rejected before editing and failures roll back the whole addition.

## 0.14.0.205 — Open all analyzed measurement results

- Analyze all in the unified viewer now opens the existing View / save snapshot result window for each successfully analyzed measurement volume.
- Fixed the missing QTreeWidget import in the raw measurement dataset editor smoke test.
- Source archive omits examples as requested.

## 0.14.0.204 — Snapshot model freshness

- New snapshots compare resolved source geometry, poses and excitation rather than the scene's edit-event history. Saving maps/snapshots and reapplying unchanged transforms do not stale them.
- Freshness retains coil specifications, solver choice, measurement settings/sample positions, effective background field, and the content of linked calibrations and their datasets. Unlinked datasets and descriptive notes/save timestamps are excluded.
- Older snapshots retain their original fingerprint checks: they did not store the dataset state needed to safely convert their fingerprints. Recalculate/save a snapshot to adopt the new check; historical results are never relabelled as fresh without evidence.
- Includes the .203 exposure notes changes and preserves the examples from the uploaded source archive, including the updated 4D-box project.

## 0.14.0.203 — Exposure notes browser

- Exposure manager now shows Name / Steps / Duration / Updated, with a resizable notes panel on the right. Publication URLs in notes are clickable; the Updated column uses UTC.
- Edit exposure now has compact name/duration inputs on the left and a multiline Notes editor on the right.
- Existing descriptions become notes automatically. The existing description storage field is retained, so notes travel with project saves, duplicate, import/export, and playback metadata without a format migration.

## 0.14.0.202 — Scripted source audio monitor

- Each Drive panel has an Audio checkbox; selecting one turns the others off. Audio selection is session-only and always starts off.
- Web Audio streams short 48 kHz buffers from the original authored source samples, independent of reduced plots and statistics scope. Continuous phase offsets and once-per-step routing are respected; only the visible playback interval is heard.
- Audio follows Play, Stop/Reset, seeking, Loop and playback speed/pitch. Switching sources fades the old audio; hiding or closing the viewer stops it. DC is removed and AC normalized with a conservative output gain.
- This is an authored command monitor, before electrical slew/load response, not a pickup or calibrated sound simulation. Video exports remain silent. Older saved playback payloads need rebuilding to supply the original audio source.
- Fixed seeking during playback retaining the old slider position, and the ordinary playback loop retaining its previous time anchor.

## 0.14.0.201 — Safer Playback analysis navigation

- Playback analysis detail pages now replace the dialog-level **Close** button with **Back**, returning to the analysis home page instead of closing the whole workspace. The normal **Close** button is shown only on the analysis home page.
- Kept the existing **← Analysis home** control at the top as a second quick route back, and updated Playback-analysis help text to describe both navigation options.

## 0.14.0.200 — Responsive long-exposure analysis

- Simplified the FFT page so the large explanatory block is gone. The page now keeps only a compact two-line result summary with the analysis mode/record size, Δf, Nyquist, and DC level, leaving substantially more room for the plot.
- Long/reduced playback analyses now open their page immediately and perform expensive authoritative replay off the GUI thread. Spectrum/FFT, Timing & transitions, Threshold & duration, and Frequency-weighted screening show an in-page progress bar/status instead of making the Playback analysis window look frozen. FFT progress advances by spectral record; timing reports its two analysis passes.
- Added a Playback-analysis session cache. Reopening a page or switching from the Spectrum summary to the default FFT view reuses the already-calculated result instead of replaying the same 30-minute exposure again. Frequency-weighted screening can also reuse the session's matching FFT result.
- Derived analysis results remain session-only rather than being written into saved maps. Saved maps continue to store the authoritative compact field basis/source replay needed to reproduce analyses, avoiding bulky/stale FFT and threshold-dependent results in project data.
- Removed a duplicate initial FFT recalculation that could run the same expensive automatic FFT twice when the FFT page was first created.

## 0.14.0.199 — Roomier coil correction controls

- Split the Model Editor **Magnetics** correction controls onto separate form rows: **Correction**, **Source**, and **Factor**. Calibration names and resolved factors now get the full inspector field width instead of competing for one narrow horizontal row.
- The **Source** row remains hidden in Custom mode and appears only when Calibration mode is selected.

## 0.14.0.198 — Raw measurement datasets only

- Simplified the Dataset Manager workflow so step 1 is now strictly raw measurement evidence. The Add/Edit dataset window no longer contains fit-coil selection, a per-dataset calibration button, fit summaries, or model/corrected/residual result columns.
- The Measurement datasets table now shows only **Name, Sensor, Quantity, Samples, and Updated (UTC)**; the redundant **Own fit** and **Fit coils** columns were removed.
- Saving a dataset now clears legacy per-dataset fit/coil metadata and fit-derived sample values, leaving fitting exclusively to step 2, **Calibration / Composite fits**. Existing legacy datasets can still be opened; re-saving them converts them to the raw-only workflow.
- Updated Dataset Manager help/copy so the three stages are explicit: collect raw data, derive a calibration, then link that calibration to whichever coils should use it.

## 0.14.0.197 — Separate calibration fit basis from application

- Calibration / Composite fits now treat their selected coils strictly as the **fit basis** used to derive the correction factor. A ready fitted factor can be linked to **any coil in the scene**, so one measured channel can supply the same correction to equivalent unmeasured coils when appropriate.
- Removed the adapter and UI restrictions that limited coil links to members of the fitted coil group. Recalculating a calibration continues to propagate its updated factor to every coil linked to that relation, regardless of which coils were used to fit it.
- Renamed Dataset Manager wording from **Target coils** to **Fit coils / Fit coil group** and updated help text to make the provenance-versus-application split explicit. The Model Editor coil inspector now exposes the same unrestricted set of ready calibrations.

## 0.14.0.196 — Calibration-link row binding

- Fixed Coil correction links incorrectly claiming that no Calibration / Composite fit existed when switching an earlier coil row from Custom to Calibration. Each row now keeps its own calibration-filter helper instead of accidentally reusing the helper created for the final coil row.
- The calibration selector therefore lists the ready fits whose target group actually contains the coil being edited, and selecting Calibration can immediately apply the existing fitted factor.

## 0.14.0.195 — Analysis scope summary

- Replaced the long explanatory paragraph at the top of Playback analysis with a compact statement of the statistics scope and the actual duration analysed, for example **Preview statistics • 6 s analysed** or **Full exposure statistics • 1,800 s analysed**.
- Regional waveform-analysis payloads now retain `statistics_scope` and `statistics_duration_s` alongside the frozen basis so the analysis hub reports the interval its calculations actually describe, including saved/reopened results.

## 0.14.0.194 — Readable video-export trace preview

- Kept the export-window trace preview from collapsing the source waveform when a sequencer is also shown. The live preview now reserves enough vertical space for the sequencer lane plus a useful signal graph, while capping the trace area so the camera preview still keeps most of the window.
- The exported frame composition is unchanged: the existing export trace fraction still determines the final video layout. The extra sizing only affects the on-screen preview and is recomputed when the export window is resized.

## 0.14.0.193 — Compact sequencer trace labels

- Shortened sequencer-lane coil labels to stable **Coil 1**, **Coil 2**, etc. display names while retaining the original scene names in playback metadata/statistics. This prevents names such as “Circular coil 1” collapsing to an unhelpful “Circul…” inside narrow routing steps.
- Reduced sequencer-lane text slightly and tightened its row spacing. Step headers now keep the full wording when space permits and automatically fall back to compact forms such as **S1 500ms** in narrow steps, preserving the step number and duration instead of clipping away the useful part.

## 0.14.0.192 — Scripted exposure summary breathing room

- Gave the wrapped Saved exposure summary a small explicit height-for-width safety margin so tight sidebar sizing cannot clip antialiased pixels from the first or last text line. The existing paragraph gap is retained without making the Scripted source section materially taller.

## 0.14.0.191 — 3D heatmap control sizing

- Split the 3D-only intensity/opacity controls from the colour-limit rows so the latter use the same two-column label/editor layout as 2D. Lower/Upper limit text now keeps its natural width instead of being squeezed by the four-column opacity grid.
- Kept the opacity endpoint controls in their own balanced grid, so their editors share only the space left after their labels.

## 0.14.0.190 — Scripted sidebar sizing and persistence

- Made the Scripted source switcher measure the visible form at the actual sidebar width, so wrapped saved-exposure guidance and summary text get enough vertical space instead of clipping at narrow window sizes. Added a small bottom breathing margin without bringing back the old oversized source panel.
- Scripted source and Traces disclosure states now persist between Maps / Analysis sessions.
- Viewer-local trace choices are now saved whenever Maps closes, including when Saved exposure is the active scripted source; previously that save path only ran while Built-in waveform was selected.

## 0.14.0.189 — Compact saved-exposure source form

- Removed hidden Built-in waveform page sizing from the Scripted source switcher. The visible source form now determines its height, with top-aligned fields and normal spacing.
- Moved Preview / Full exposure statistics guidance directly below the radio buttons. Exposure details follow with a small paragraph gap.

## 0.14.0.188 — Compact transport and individual trace panels

- Replaced the resizable/scrolling playback-controls pane with a compact bottom transport: Play, Reset/Stop, Loop, speed, and timeline. Slice-mode controls remain available where relevant.
- Gave each signal and the sequencer a separate draggable panel. Signals redraw their complete vertical range to fit panel height, with individual invert controls; sequencer lanes scale to fit instead of scrolling or cropping. Double-click a divider to restore balanced heights.
- Moved Export traces beside Export video in 2D, 3D, and Brain result actions, retaining the CSV trace-selection dialog.
- Removed the GPU-status and axis/navigation text overlays from the view, along with the transport's FPS readout.
- Kept video export's existing composition independent of interactive panel sizes.

## 0.14.0.187 — Scrollable measurement panels

- Made the shared Measurement volumes / Brain areas form scroll within short result and Selector panels, retaining the existing filter/results divider.
- Gave filter and results tables useful minimum heights and protected each section's layout minimum, keeping Add/Remove buttons below the filter grid instead of overlapping it.
- Selector analysis/save actions remain accessible below the scrolling measurement form.

## 0.14.0.186 — Resizable playback panes

- Added draggable horizontal dividers between the map, optional traces, and playback controls. Divider keyboard controls and double-click reset are also available.
- Refit and redraw playback panes as the window changes size. Controls wrap and scroll within their own pane; long traces and sequencer lanes scroll independently.
- Keep static maps and video-export composition independent of playback divider positions.

## 0.14.0.185 — View-specific measurement-panel defaults

- Changed the unified Maps / Analysis workspace so ordinary **2D** and **3D** views start with the right-hand Measurement volumes panel closed, giving the map/viewer the available width by default. **Brain** view still starts with its analysis panel open.
- Measurement-panel visibility is now remembered separately for 2D, 3D, and Brain while the Maps window is open. Switching Static/Scripted therefore preserves the user's choice for the active visual view without forcing one global panel state across all views.
- Replaced the ambiguous shared **Show / hide panel** caption with an action-specific **Show measurements** / **Hide measurements** label and matching tooltip. Dragging the panel splitter fully closed/open updates the same state and label.
- Updated Maps/Brain help and UI/source regressions for the new per-view defaults and explicit toggle text.

## 0.14.0.184 — Responsive Maps / Analysis layout

- Made the unified Maps / Analysis window respond much better to smaller window sizes. The suite workspace now has a modest 820×560 minimum instead of relying on the much wider minimum-size hints inherited from its standalone map dialogs.
- Removed the standalone 300–310 px control-pane width floor when 2D/3D/Brain controls are adopted into Maps. The sidebar can shrink to 240 px and its scrolling forms wrap long rows instead of pushing controls outside the window.
- Added responsive splitter clamping: the left sidebar and right Measurement/Brain panel shrink before the central viewer is crushed. The right analysis panel now uses a 220 px minimum and remembers its preferred width when the window is enlarged again.
- Reduced the default right-panel allocation and applied the same narrower minimum to rendered Exposure/Brain result panels, while preserving the existing drag-to-resize and show/hide behavior.
- Added source/UI regressions for the unified workspace minimum geometry and removal of the inherited control-scroll width floor.

## 0.14.0.183 — Scripted statistics scope and preview-only traces

- Added a **Statistics** radio selector to Maps → Scripted source → Saved exposure. **Preview** is the fast default and limits precomputed playback/regional statistics to the selected preview duration; **Full exposure** preserves complete physical-duration statistics when explicitly requested.
- Saved regional analysis now freezes the authoritative source for the selected statistics scope, so summary, timing/threshold, FFT and frequency-weighting tools agree with the scope chosen when the map was built.
- Saved-exposure Drive and Sequencer traces now use an event-aware preview-only display timeline instead of the bounded full-exposure retained trace. A 2 s preview of a 30 minute exposure therefore plots 2 s of waveform/routing detail instead of compressing 30 minutes into a washed-out lane.
- Preview statistics no longer resolve the long full-exposure retained display trace. Full-exposure electrical/statistical replay is performed only when **Full exposure** is selected, while viewer frame construction remains limited by Preview Duration either way.
- Measurement/Brain playback metadata and summary labels now report whether statistics describe the preview or full exposure, and the render-job task label changes accordingly.

## 0.14.0.182 — Exposure analysis clock seam fix

- Fixed a pathological long-exposure analysis clock caused by machine-precision timestamp seams at authored waveform/component boundaries. Adjacent IEEE-754 timestamps are still retained as event knots so sharp edges are sampled, but sub-tolerance intervals no longer define the uniform authoritative sampling clock.
- A 30-minute waveform with a ~3.3 ms physical point delay can therefore no longer be misreported as roughly 4.05 quintillion logical samples because of a ~4.44e-16 s floating-point seam.
- Added a regression using an exact next-representable-float component boundary and verifies the resulting 30-minute analysis estimate remains in the expected sub-million/million-scale range rather than overflowing into astronomical work.

## 0.14.0.181 — Long shared-source exposure acceleration and measurement progress

- Reworked the shared-source parallel RL resolver so long piecewise-linear source histories no longer recompute a general SciPy matrix exponential at every waveform knot. Voltage-source branch networks now use an exact symmetric modal solution for arbitrary sample times, while the current-source branch solver reuses exact cached state transitions for repeated intervals. This is especially important for long voltage-source exposures with several parallel drive groups, such as the 4D-box three-series-string topology.
- Preserved the same continuous-time RL equations and source-output-resistance coupling; deterministic parity tests cover the existing series/parallel source cases, and a new regression verifies that thousands of identical 3.3 ms point-delay intervals collapse to a small number of matrix-exponential evaluations.
- Measurement-volume playback now emits progress immediately while measurement sample sets are being prepared, before its field-basis solve starts. The render page therefore no longer remains on the completed playback-timeline message if sample preparation or the first measurement solve is the next expensive operation.
- Added measurement sample-count detail before the measurement-volume basis calculation so unusually large/custom measurement regions are visible in the processing information rather than looking like an unexplained stall.

## 0.14.0.180 — Frozen replay fix and long-exposure timeline progress

- Fixed saved non-periodic authoritative replay after the shared-source electrical-network change in 0.14.0.177. The frozen replay adapter now exposes the same shared-source network solver used by the live StudioAdapter, restoring save/reopen analysis for multiple same-source drive groups.
- Made 2D scripted playback expose **Resolve playback timeline** as a real render task instead of leaving the completed field-basis row active while a long saved exposure is being electrically replayed.
- Long non-periodic retained display traces are now resolved in chronological chunks with visible progress (`chunk / total` plus physical exposure time reached). The retained sample positions and electrical result are unchanged; this only makes the bounded full-duration preparation observable.
- Added a separate **Calculate measurement-volume field bases** task before measurement frame metrics and full-exposure statistics. Measurement-volume spatial solves are no longer hidden behind either the main field-basis row or frame-metric work.
- Added full-environment regressions for frozen replay through the shared-source network, chunked retained-trace parity/progress, and the expanded 2D/3D measured-playback task plan.

## 0.14.0.179 — Large-playback warning and live playback-build progress

- Added a preflight confirmation for scripted playback requests of 2,000 frames or more. The warning reports the exact frame count and approximate amount of frame data at the selected FPS before the render job is queued.
- Split the 3D scripted render plan so playback timeline/scene preparation, measurement-volume frame metrics, full-exposure measurement statistics, final WebGL packaging, and viewer opening can report as distinct work instead of hiding long measurement work under a nearly frozen **Build playback and WebGL data** row.
- Measurement-volume frame metrics now report incremental frame counts, while authoritative full-exposure summaries report their own processed analysis-sample progress.
- Added render-request metadata for the number of active measurement volumes so the extra progress rows only appear when that work is actually required.

## 0.14.0.178 — Duplicate exposure drive groups

- Made the Exposure routing-table **Duplicate** action contextual. Selecting a Step still deep-clones the complete step; selecting a Drive now clones only that drive group immediately after itself inside the same step.
- Drive-group duplication preserves source assignment, selected coils, Series/Parallel wiring, and per-coil R/L overrides while generating a fresh drive-group ID and independent override dictionaries.
- The cloned drive is selected automatically for immediate editing. Because an exact clone initially references the same physical coils, the existing save-time one-drive-per-coil validation remains in force until the duplicate is reassigned as needed.

## 0.14.0.177 — Shared-source parallel drive groups

- Removed the one-drive-group-per-source restriction from exposure routing. A source may now feed several drive groups in the same step.
- Groups that share one source are solved as **parallel routes at that source's terminals**, while each group keeps its own internal **Series** or **Parallel** coil wiring. This directly supports layouts such as three parallel channels containing two series coils each.
- Voltage-source output resistance is applied **once to the combined shared-source network**, so same-source groups are not treated as isolated copies of the source. Current-source commands likewise represent the total source current and divide among the parallel routes according to the uncoupled RL network.
- Updated bounded/long-exposure streaming and frozen replay to solve same-source groups together, preserving the shared-source electrical semantics in playback, statistics, FFT/weighting replay, and saved regional analysis.
- Drive info and Add/Edit drive-group previews now include same-source peer groups in the step, report the combined source-terminal load, and explain the parallel-route model. Mutual inductance remains outside the electrical model.
- Exposure save validation now resolves complete step source networks rather than validating each route as an isolated source/load pair.

## 0.14.0.176 — Duplicate exposure routing steps

- Added **Duplicate** to the Exposure routing table. It inserts a deep clone immediately after the selected step, preserving duration, source assignments, coil selections, Series/Parallel wiring, and per-coil R/L overrides.
- Regenerates the duplicated step and drive-group internal IDs so the copied route remains an independent canonical protocol entry.
- The action is enabled only for a selected step row, avoiding ambiguity with individual drive-group editing.

## 0.14.0.175 — Inline exposure step durations

- Made routing-step duration directly editable in the Exposure editor table with an in-place seconds spin box; step timing no longer opens a separate configuration dialog.
- **Add step** now copies the duration of the current last step, making repeated routing intervals quick to author; the new duration remains editable immediately in the table.
- Restricted **Edit…** to drive-group rows because step rows no longer need a separate editor, and refreshed the window-local wheel guard for dynamically rebuilt duration controls.

## 0.14.0.174 — Roomier drive-group load overrides

- Removed the redundant **Edit waveform…** button from Add/Edit drive group; waveform choice and editing remain owned by the Exposure source table / Waveform Editor workflow.
- Gave the per-coil R/L override table a larger flexible vertical allocation (155–220 px) so several selected coils and their measured load overrides can be edited without the table feeling cramped.
- Kept the coil selector and Electrical preview visible around the expanded override area.

## 0.14.0.173 — Exposure drive-group R/L overrides

- Added optional **per-coil resistance and self-inductance overrides** directly to Add/Edit drive group. Selected coils appear in a compact override table; blank cells continue using the model-side electrical load, while entered resistance (Ω) and inductance (mH) apply only to that exposure route.
- Extended the canonical exposure schema and portable exposure file format so drive-group overrides survive project save/reopen, portable export/import, long-exposure streaming, verified-cycle compression, and frozen authoritative-analysis replay. Series/Parallel resolution uses the overridden branch values before calculating equivalent load and current sharing.
- Electrical diagnostics now distinguish **drive override**, **model override**, calculated resistance, and estimated inductance. Geometry-estimated-inductance warnings are suppressed for branches with either kind of explicit inductance override.
- Removed the voltage-source diagnostic footer saying that current/power limits are not modeled, as requested. Current-source compliance-voltage diagnostics remain.
- Clarified that the coil magnetic field correction factor is deliberately **not** applied to the air-core inductance estimate: it calibrates magnetic output/effective ampere-turns and is not generally a physically valid inductance multiplier.
- Exposure schema is now v6 and portable exposure files v7; v5/v6 data remains readable and normalizes without drive-local overrides.

## 0.14.0.172 — Wheel-safe focus behavior

- Removed the remaining hover-wheel focus jump from combo boxes and spin-box-style numeric editors. The shared window-local wheel guard now converts Qt `WheelFocus` to `StrongFocus`, preserving normal click/Tab/keyboard focus while preventing a passive wheel gesture from moving keyboard focus onto the editor.
- Kept the 0.14.0.171 window-local event-filter architecture; no QApplication-wide Python event filter was reintroduced, so the Model Editor / Map Creator WebEngine teardown fix remains intact.
- Extended the Qt wheel regression to force `WheelFocus`, verify that installation normalizes it to `StrongFocus`, and assert that scrolling over combo/spin editors leaves keyboard focus on the previously focused widget while still routing the scroll gesture to the enclosing panel.

## 0.14.0.171 — Window-local wheel guard / Qt teardown fix

- Removed the 0.14.0.169 QApplication-wide Python event filter after full PySide6/WebEngine testing showed native segmentation faults during UI-smoke rendering and when closing Model Editor / Map Creator windows. The crash occurred below Python exception handling during Qt/WebEngine teardown.
- Kept the wheel-safety behavior but moved it to each Workbench top-level window. The common persistent-window helper installs/refreshes a local guard when a window is shown, so normal combo/spin editors remain protected without observing unrelated Qt/WebEngine objects application-wide.
- Made the local guard idempotent and safe to rescan, including internal line editors. Exposure source rows explicitly refresh it after their dynamic table cells are rebuilt, so newly-added waveform/phase/mode controls remain wheel-safe.
- Added a normal-dependency lifecycle regression that creates Model Editor + 2D Map Editor windows with wheel guards, closes them, flushes deferred deletes, and processes the event loop.

## 0.14.0.170 — Clickable project-component shortcuts

- Made the Project Manager's large component labels clickable shortcuts to their primary applications: **Model → Model Editor**, **Maps / Analysis → Map Creator**, **Exposures → Exposure Editor**, and **Datasets → Dataset Manager**.
- Kept the existing populated-state green underline intact; clickable labels now use a pointing-hand cursor, keyboard activation with Enter/Space, accessible names, and tooltips identifying the application they open.
- Updated Project Manager help and added UI/source regressions for the shortcut mapping.

## 0.14.0.169 — Global wheel-safe value editors

- Applied the existing mouse-wheel safety rule at the QApplication level so combo boxes and spin-box-style numeric editors across **all Field Workbench windows** no longer change value merely because the pointer is over them while scrolling. This also covers dynamically-created editors such as Exposure source-table cells without per-dialog wiring.
- Wheel gestures over protected editors are routed to the nearest scrollable table/panel when one exists; otherwise they are consumed. Normal clicking, keyboard editing, arrow controls, and intentional wheel-scrolling inside an opened combo popup remain available.
- Retained the existing local wheel guards for isolated/test-hosted dialogs, and added a UI regression for dynamically-created combo/spin editors plus packaging assertions for the application-wide installation.

## 0.14.0.168 — Reuse untouched startup project tab

- Opening a project from Project Manager now replaces the original untouched startup **Untitled** placeholder in place instead of leaving an unnecessary empty tab beside the opened file.
- Only the automatic startup placeholder is disposable. A startup tab with project changes, a startup tab with a project application window open, and blank projects deliberately created with New/+ are preserved as normal project tabs.
- Added Project Manager regressions covering both replacement of the pristine startup placeholder and preservation of changed/user-created empty tabs.

## 0.14.0.167 — Exposure source-table test cleanup

- Updated the Phase-6 waveform-catalogue regression to inspect the Exposure editor's new inline source table instead of the removed `source_list` widget. The rename-refresh/cancel behavior under test is unchanged; this is test-only cleanup after the 0.14.0.166 UI refactor.

## 0.14.0.166 — Inline exposure sources and trigger mode

- Reworked the Exposure editor's source section into an in-place table. Source name, saved waveform, phase offset, and playback mode are edited directly; the separate Add/Edit source configuration dialog and Edit source button are gone.
- Added source playback choices **Continuous** and **Once per step**. Continuous preserves the absolute exposure-time waveform clock. Once per step starts one phase-shifted waveform cycle whenever a routing step using that source begins, truncates at the step boundary when necessary, and commands zero after the cycle when the step is longer. A later step using the source fires a new shot.
- Kept electrical load state separate from waveform triggering: compatible Series/Parallel coil groups may retain their RL current across routing boundaries even when a Once-per-step source waveform restarts. Source slew still applies to the command-to-zero tail.
- Made the event-aware full-exposure analysis clock aware of Once-per-step source events, and made verified periodic compression depend only on the routing period plus genuinely Continuous source periods. Frozen authoritative replay therefore preserves the same source mode after save/reopen.
- Added seconds/minutes/hours selection to Exposure duration while retaining seconds as the canonical stored unit. Exposure schema is now v5 and portable exposure files v6; v4 exposure data and v5 portable files normalize into the new Continuous source mode.
- Updated Exposure help and added schema/runtime regressions for playback-mode normalization, single-shot restart, phase offset, idle steps, and command-to-zero behavior.

## 0.14.0.165 — Test cleanup

- Updated the reduced-trace regional-analysis regression so all five guarded tools (summary, timing, threshold, FFT and frequency weighting) match the stable **authoritative analysis source** portion of the current fail-closed message rather than obsolete per-tool wording. Application behavior is unchanged.

## 0.14.0.164 — Exposure routing workspace cleanup

- Reclaimed the Exposure editor's vertical workspace for the routing schedule by removing the always-visible electrical-diagnostics block and its explanatory paragraph beneath the table.
- Added an on-demand **Drive info…** button for the selected drive row. It opens the same detailed source/load/branch diagnostic information in a separate selectable-text dialog, including the selected step duration and the zero-initial-current preview caveat.
- Kept the live electrical preview in the Add/Edit drive-group dialog, where it remains useful while choosing source, coils, and Series/Parallel wiring. Missing-coil warnings remain visible in the main editor because they require immediate attention.
- Updated Exposure help and the editor regression to reflect the table-first layout.

## 0.14.0.163 — DAC import inherits waveform encoding

- Fixed confirmed Create/Edit imports of the bundled **Unsigned 8-bit DAC** encoding incorrectly rejecting Voltage waveforms because the generic profile still carried a historical Current/mA default. The bundled DAC file is now treated as raw 0..255 codes at this boundary, not as an inherently current-based signal.
- Active DAC import now inherits the waveform's already-confirmed Current/Voltage quantity and engineering unit, asks only for **Point delay**, **Code 0**, and **Code 255** mappings, and stores that inherited physical encoding in the editable authored component. Quantity-specific user format profiles keep their existing mismatch protection.
- Updated Waveform Editor help and packaging regressions so the fixed waveform-encoding rule from 0.14.0.124 is explicit for DAC imports as well as normal waveform imports.

## 0.14.0.162 — Long-exposure release hardening

- Completed Phase 5 of the long-exposure regional-analysis cleanup with a release-hardening pass rather than new analysis features. The authoritative frozen source now has one validation boundary shared by time-domain replay, verified-cycle spectral analysis, replay-recipe FFT sampling and segmented frequency weighting. Unsupported schema/version/mode data therefore fails consistently before numerical replay begins.
- Added an explicit saved-result integrity check tying frozen current-column order to the regional field-basis drive order. A corrupted or internally inconsistent result can no longer silently apply one coil's replayed current to another coil's quadratic field basis; all regional numerical tools fail closed and ask for the map to be regenerated. Duplicate/malformed authoritative drive metadata is rejected at the same boundary.
- Added a dependency-light regression proving summary, threshold, timing, FFT and frequency weighting all reject a frozen drive-order mismatch, plus a replay-recipe spectral regression proving unsupported authoritative versions are rejected before the frozen electrical resolver is entered.
- Added a full-environment save/reopen regression for reduced non-periodic exposure maps. It crosses the saved-result JSON boundary, deliberately destroys the retained display trace, then exercises summary, threshold, timing, segmented FFT and segmented frequency weighting from the self-contained replay recipe alone.
- Added a focused long-exposure regional-analysis release-gate command to `TESTING.md`, clarified the uncompressed-stream code comment, and refreshed the release review. No legacy migration layer or new user-facing sampling options were introduced.

## 0.14.0.161 — Authoritative long-exposure frequency weighting

- Completed Phase 4 of the long-exposure regional-tool cleanup. Frequency-weighted screening no longer consumes the bounded display trace when a saved exposure is reduced; it now uses the same authoritative spectral source and source-honest sampling plan introduced for FFT.
- Preserved the historical whole-record weighted-peak calculation for short/unreduced records. Verified periodic long exposures apply the published gain/phase weighting directly to one already-verified steady-state physical cycle, so thousands of repeats are never expanded and the physical source bandwidth is preserved.
- Added bounded full-duration weighting for long non-periodic/unsettled exposures. Equal authoritative records span the physical duration; each scored record is filtered as the centre of a three-record FFT context block, keeping circular block boundaries one record away from the reported samples while memory remains bounded. The whole exposure uses a circular outer boundary consistent with the historical DFT convention.
- Kept the spectral-summation cross-check on the Phase-3 full-duration spectrum. Segmented results therefore pair a phase-aware block-weighted peak with the power-averaged regional vector spectrum rather than deriving either result from the thinned display trace.
- Frequency-weighted results now report whole-record / verified-cycle / segmented-full-duration mode, physical and record duration, samples per record, spectral and weighting frequency resolution, source/analysis Nyquist, the actually resolved frequency range, and whether the bounded spectral-work budget reduced sampling. Full-band reporting now requires both the 1 Hz low end and 100 kHz high end to be resolved.
- Bounded the plotted weighted-index trace independently of the peak calculation. Long segmented analyses retain at most the normal 65,536-point presentation budget while the peak search still evaluates every analysed weighted sample. Added Auto/Custom sampling controls so reduced long exposures use the source-honest automatic record plan by default.
- Updated Exposure and Frequency-weighted Help plus the release review. Added dependency-light regressions for a 20 Hz × 30 minute verified exposure, segmented full-duration weighted-peak equivalence on an integer-cycle tone, bounded weighted-trace presentation, and continued short-record profile math.

## 0.14.0.160 — Authoritative long-exposure FFT

- Completed Phase 3 of the long-exposure regional-tool cleanup. Spectrum analysis and the configurable FFT viewer no longer consume the bounded 65,536-point display trace when a saved exposure is reduced; they use the Phase-1 authoritative analysis source.
- Preserved the historical whole-record FFT exactly for short/unreduced records. Verified periodic long exposures now analyze one already-verified steady-state physical cycle, so exposure duration no longer collapses the FFT sample clock. The result explicitly identifies the verified-cycle method, cycle duration, physical exposure duration, frequency resolution, Nyquist frequency, and whether the cycle itself had to be spectrally downsampled.
- Added a bounded full-duration spectral method for long non-periodic/unsettled exposures. The complete physical duration is divided into equal non-overlapping FFT records and regional XYZ vector spectral power is averaged across those records. Uniform spectral sampling never exceeds the authoritative source clock; a bounded total spectral-work budget can only reduce the reported Nyquist rather than inventing bandwidth.
- Auto FFT record sizing now uses nearly all available source-supported intervals around the 65,536-sample boundary instead of discarding a large fraction of the source clock merely because one whole record would exceed the per-record cap. Custom FFT samples remain samples per record. Rectangular/Hann/Hamming/Blackman windows and mean-removal controls continue to use the same energy-normalized vector-field math.
- Updated Spectrum/FFT UI and Help to state which method is active (whole record, verified steady-state cycle, or segmented full-duration), the record/segment duration, and source-versus-analysis Nyquist when a spectral-work bound reduces sampling. The 65,536-point saved trace is now presentation data for FFT as well as the Phase-2 time-domain tools.
- Frequency-weighted screening remains the sole intentional reduced-long-exposure gap for Phase 4. Its safety guard now describes only frequency weighting; short unreduced weighting behavior is unchanged.
- Added dependency-light regressions for a 20 Hz × 30 minute verified exposure, segmented full-duration spectra, source-bandwidth honesty, automatic record sizing across the 65,536 boundary, and unchanged short-record FFT math.

## 0.14.0.159 — Authoritative time-domain regional analysis

- Completed Phase 2 of the long-exposure regional-tool cleanup. Exposure-summary fallback, Timing & transitions, and Threshold & duration now consume the Phase-1 authoritative packet source instead of the bounded 65,536-point display trace. Saved exposures therefore retain full physical-time behavior after save/reopen even when their presentation trace is reduced.
- Split timing/transitions from FFT. Regional vector slew and drive-current pulse/gap statistics now stream independently of spectral analysis, so a long exposure no longer loses time-domain analysis merely because a whole-record FFT is not yet available. Drive thresholds are established from the full authoritative peak current before crossings are accumulated.
- Added exact repeat-aware interval aggregation for verified steady-state packets. Pulse counts, active/inactive time, pulse-width/gap min/max/mean, threshold time-above, excursion count and longest uninterrupted interval are calculated from cycle weights without expanding thousands of repeated cycles. Cycle-boundary excursions are merged correctly.
- Bounded Threshold & duration presentation to the first 2,000 excursion rows while retaining complete-exposure totals and counts. Timing output likewise reports exact aggregate width/gap statistics without requiring one UI item per repeated pulse.
- Kept ordinary non-exposure waveform maps on their complete retained timeline. Reduced saved traces without an authoritative source fail clearly and should simply be regenerated; no legacy saved-map migration layer was added.
- Whole-record FFT and frequency-weighted screening remain the two deliberate long-exposure gaps for Phases 3 and 4. Their reduced-trace guard now describes only those spectral limitations.
- Added lightweight regressions for repeated authoritative summaries, threshold/timing results, cycle-boundary excursion merging, 100,000-cycle bounded interval presentation, short unreduced waveform behavior, and continued spectral guarding.

## 0.14.0.158 — Frozen authoritative long-exposure analysis source

- Added the Phase-1 foundation for completing long-exposure regional tools: every saved-exposure regional waveform payload now carries a versioned, self-contained authoritative analysis source alongside the bounded display trace. The frozen source records the normalized exposure protocol, exact source waveform command traces and electrical metadata, driven-coil R/L load data, drive order, physical analysis spacing, compression metadata, and logical sample-count estimate.
- Verified periodic exposures additionally freeze the already-verified transient / representative-cycle / remainder packets with repeat weights. Reopened results can therefore reuse the exact compressed physical analysis without rerunning steady-state verification or expanding thousands of repeated cycles.
- Non-periodic or unsettled exposures retain only the compact source/load recipe and regenerate their event-aware chronological current packets through the same bounded-memory electrical resolver when an authoritative consumer asks for them. Replay does not consult the live project, so later project edits cannot silently change a saved map's physical analysis source.
- Kept the 65,536-point retained trace as presentation data only. The separate waveform-summary, timing, threshold, FFT and frequency-weighting dialogs still keep the 0.14.0.157 reduced-trace guard in this phase; later phases can now migrate them onto the new authoritative packet factory one family at a time. Saved maps created by 0.14.0.157 have the safety marker but not this new replay source, so regenerate any long-exposure regional results that should gain future full-resolution tool support.
- Added regressions for JSON and saved-map-codec round-trip of the frozen regional payload, live-project-independent replay of uncompressed exposure packets, direct replay of verified weighted packets, and bounded storage for the 20 Hz × 30 minute compression case.

## 0.14.0.157 — Release-review cleanup and long-exposure analysis guards

- Bound candidate periodic-cycle construction before allocating a complete time grid, and stop collecting event-enriched chunks at the verification budget. Oversized repeat cycles now fall back to ordinary streamed analysis without first materializing the oversized cycle.
- Calculate saved-exposure Drive trace Mean/DC, RMS and peak-to-peak from the full source timeline, with verified cycle weights where available. Narrow pulses no longer disappear from these readouts merely because the retained display trace misses them; disconnected sources still advance and contribute their own source statistics.
- Preserve reduced-trace provenance, analysis spacing and compression metadata in saved regional waveform payloads. Separate waveform-summary, FFT, frequency-weighting, pulse-timing and threshold-duration tools now reject reduced display traces explicitly instead of reporting potentially aliased or incomplete results. Main exposure-table and sensor statistics continue using the full streamed/compressed analysis. Full-resolution long-exposure support in the separate tools remains follow-up work; older saved maps should be regenerated to record the new provenance.
- Updated Exposure Editor Help to describe streaming, verified periodic compression and retained-trace limits; removed the obsolete million-sample limits from that page. Restored the missing type-only viewer import in the shared result panel.
- Added regressions for early oversized-cycle fallback, incremental event-grid limits, pulses missed by the display trace in both compressed and uncompressed exposures, and saved reduced-trace rejection across the separate analysis tools.

## 0.14.0.156 — Verified periodic long-exposure compression

- Added conservative repeated-cycle compression on top of the streamed long-exposure engine. Field Workbench now finds a common repeat period across the looping routing schedule and every continuous waveform source, then compares consecutive fully resolved source/current cycles before reusing anything.
- Periodicity is verified from the actual electrical result rather than waveform phase alone. Source slew and voltage/RL transients therefore run explicitly until consecutive cycles match at strict numerical tolerances; exposures that do not settle within the bounded verification window automatically remain on the ordinary streamed path.
- Authoritative scalar, Brain-area, and measurement-volume statistics now accept weighted analysis packets. A verified complete cycle is integrated once and multiplied across the remaining physical repeats; extrema are evaluated from the representative cycle and cumulative B-time/B²-time/mean/RMS retain the full exposure duration. Initial transients and final partial cycles are still evaluated explicitly.
- The bounded retained trace used by plots, drive traces and FFT payloads now phase-maps late samples onto the same verified physical cycle. This prevents presentation sampling from walking every repeated waveform event after the statistics themselves have already been compressed.
- Hardened long-run switching clocks: source command events now use an explicit chronological knot/cycle cursor, exact source-segment boundaries are right-continuous within floating-point tolerance, routing-boundary ownership uses the physical switch tolerance, and tiny internal interval roundoff is clipped before electrical resolution. These changes prevent mathematically identical repeated boundaries from drifting onto adjacent waveform samples or routing steps.
- Added regression coverage for mismatched source/routing periods (1.0 s waveform with a 0.6 s routing cycle), compressed-vs-explicit scalar and regional statistics, fast RL settling, slow-RL fallback, and a 20 Hz × 30 minute exposure representing more than 27 million logical event-aware samples while retaining only the bounded presentation trace. No exposure schema or user-facing sampling controls changed in this phase.

## 0.14.0.155 — Streamed long-exposure statistics

- Reworked saved-exposure analysis around a stateful, chronological current resolver. Continuous source phase/slew and series/parallel RL state now advance once along the physical exposure clock instead of expanding every repeated source command and routing occurrence into one full-duration event array.
- Replaced the monolithic saved-exposure analysis timestamp/current matrix with event-aware analysis chunks. Authoritative observation, Brain-area, and measurement-volume statistics now consume those chunks directly while preserving source command events, just-before-event samples, routing boundaries, and trapezoidal integration across chunk boundaries.
- Kept presentation data separate from authoritative statistics: plots, FFT/source payloads, and saved analysis traces retain at most 65,536 evenly distributed full-duration samples, while statistics continue over the complete event-aware exposure timeline. This removes the previous 1,000,000-analysis-sample and 2,000,000-expanded-source-event hard stops from the saved-exposure analysis path.
- Preserved continuous-source switching semantics at exact routing boundaries, including a preview that ends exactly on a switch, and added regressions comparing the new streaming resolver/statistics with the former array calculations on short runs plus a long exposure beyond the old analysis-sample ceiling.
- This is the first long-duration phase: memory use is now bounded with exposure duration, but calculation time still scales with the number of source/routing events and analysis samples. Repeated-cycle/steady-state compression and user-facing sampling controls remain follow-up work.

## 0.14.0.154 — Continuous exposure sources and separate preview duration

- Added physical exposure duration and an editable source catalogue (stable IDs, waveform references, repeat mode and phase offsets). Routing steps now connect those sources to coil loads instead of restarting a waveform. The existing internal `sequence` / `drive_groups` container names are retained; groups now reference `source_id`.
- Sources advance on the absolute exposure clock, including waveform wraps, source slew and disconnected intervals. Routing repeats independently until the physical exposure ends; a nonrepeating schedule disconnects its loads after its final step. Compatible coil groups retain RL current across routing boundaries. Rewired/disconnected coils still use the documented ideal-switching assumption.
- Added **Duration** to the Saved exposure Scripted source controls to limit the preview from time zero. Rendering uses only that preview; Brain, measurement-volume, sensor and drive statistics cover the full physical exposure. Saved map payloads retain physical and preview durations. Drive traces now show continuous sources independently of routing.
- Full-exposure analysis uses source/routing-based sampling independent of preview FPS and duration, includes switching boundaries, and calculates regional means/peaks in bounded-memory batches. Long runs that exceed 1,000,000 analysis samples, 2,000,000 command events per source, or 1,000,000 routing steps fail explicitly. Compressed/cycle-averaged long-duration analysis remains a follow-up phase; sampled peaks are not guaranteed continuous-time extrema.
- Bumped exposure schema to v4 and portable exposure files to v5. Version-3 drafts without a source catalogue are upgraded with one source per waveform and one routing-pass physical duration. Review migrated durations and any repeated use of the same source within a step; at this version each source was limited to one series/parallel load group per step (that restriction is removed in 0.14.0.177).
- Updated source-reference import/export, editor diagnostics, help, and regression tests covering clock continuity, independent phase, idle slew, RL state, separate preview/analysis timelines, and full-duration regional statistics.

## 0.14.0.153 — Scripted trace and Maps tab-strip cleanup

- Renamed the Scripted viewer's **Base excitation trace** to **Drive trace** and **Drive state trace** to **Sequencer trace**. Saved exposures now populate Drive trace from the same resolved exposure-drive groups used by playback, including several simultaneous source-command traces when a protocol contains independent drive groups.
- Kept Drive-trace export/video handling multi-source aware while preserving the single quick-waveform drive trace for built-in Scripted sources.
- Made the compact **Scripted source** section size to its active page so selecting **Saved exposure** no longer reserves the taller built-in-waveform editor height.
- Moved the Maps result tabs into a full-width strip above the preview/right-panel row. The measurement/region panel now begins below the tabs at the same height as the preview, giving result tabs substantially more horizontal room.
- Moved **Load saved map…** to the left edge of that full-width strip immediately before the permanent **Selector** tab.

## 0.14.0.152 — Scripted sensor-trace selector

- Replaced the old observer-source/sample/signal controls in Scripted Maps with a dedicated **Sensor trace** checkbox directly beneath the drive trace controls. The selector table is shown only while Sensor trace is enabled.
- Added a per-source table of eligible sensors and physical probes with independent trace checkboxes, per-sensor sample selection, and an independent field-signal selector for each row (`|B|`, `Bx`, `By`, `Bz`, and absolute-axis variants; `H` equivalents are used for H-field maps).
- Sensor-trace settings now preserve each selected source's signal and sample independently, so several synchronized traces can capture different components in one playback run.

## 0.14.0.151 — Correct line-oriented DAC encoding

- Corrected the bundled **Unsigned 8-bit DAC** declarative profile to match the actual file format: UTF-8 text, one decimal 0..255 sample per line, with the first line storing the point count.
- The built-in DAC profile now expresses that layout entirely through ordinary Format Builder fields: text storage, newline records, first-record sample count, one skipped header record, runtime sample interval, and runtime two-point engineering-value mapping.
- Added regression coverage for point-count decoding and count mismatch errors while keeping DAC import on the same generic profile/runtime-settings path as user-created encodings.

## 0.14.0.150 — Declarative runtime-timed waveform encodings

- Generalized import-time timing into a normal waveform-format property. Any built-in or user-created encoding can choose **User-entered sample interval at import** instead of hard-coding timing or storing timestamps in the file.
- Added the matching generic **User-entered two-point mapping at import** conversion mode so raw-code-to-engineering-value translation is profile-driven as well.
- Runtime-timed encoded imports now retain their raw source samples, sample interval, value-mapping settings, and complete declarative format profile as one editable authored component. Reopening the component uses the same generic editor regardless of which encoding produced it.
- Rebuilt **Unsigned 8-bit DAC** as an ordinary bundled format profile using the same binary-record, runtime-timing, and runtime-mapping settings exposed by Format Builder. There is no separate DAC import decoder or DAC-specific authoring component.
- Duplicating the bundled DAC profile now uses the ordinary profile-copy mechanism, so the duplicate can be freely edited and any user profile can opt into the same runtime-timing behavior.
- Removed the obsolete legacy DAC-specific authoring recipe path; waveform playback continues to use normalized physical samples while encoded source details remain authoring metadata.

## 0.14.0.149 — Editable copies of the built-in DAC encoding

- Enabled **Duplicate…** for the protected **Unsigned 8-bit DAC** encoding in the catalogue waveform importer.
- Duplicating the DAC encoding now opens the normal waveform format builder with a user-owned profile preconfigured as binary unsigned 8-bit data, one byte per sample, EOF termination, a 1 ms fixed interval, and the built-in default 0..255 → 0..1 mA linear mapping.
- The built-in DAC entry remains protected and retains its special per-import timing/mapping dialog and editable-source-code workflow; only the duplicate becomes a normal editable format profile.

## 0.14.0.148 — Catalogue importer defaults

- Simplified the catalogue waveform-import guidance by removing the repeated encoding auto-detection warnings from the visible importer text.
- Moved **Field Workbench waveform JSON** to the top of the Available encodings list and select it by default when the importer opens, while still allowing the user to explicitly choose any other encoding before browsing.

## 0.14.0.147 — Format-first catalogue waveform import

- Reorganized Waveform Editor catalogue import around explicit encoding selection: choose an encoding first, then choose a file; Field Workbench does not guess encodings from filename or contents.
- Moved unsigned 8-bit DAC into the **Available encodings** list as a built-in editable encoding alongside native Field Workbench and reusable format profiles.
- Replaced the old text/CSV file shortcut with **Manual input…**, an editable raw-data table that supports direct cell entry and clipboard table paste.
- Removed the raw sample preview table and Preview button from the main encoded-file importer. Selecting a file immediately decodes it with the chosen encoding and lists the waveform(s) available to import.

## 0.14.0.146 — Reliable endpoint tangent proxies

- Fixed a Free-form editor edge case where a tangent proxy could be placed almost directly beneath an endpoint anchor sitting on the graph boundary, making the proxy appear to disappear.
- Proxy geometry now ignores near-zero anchor-edge contacts and uses the next real plot-edge crossing; a separated edge fallback keeps even outward-pointing boundary tangents recoverable.

## 0.14.0.145 — Waveform time-axis label spacing

- Increased the waveform viewer bottom plot margin and moved the **Time (s)** axis title slightly upward so it remains fully visible instead of clipping against the bottom edge when the viewer panel is short.

## 0.14.0.144 — Waveform modebar tooltip scoping

- Removed the viewer-wide Waveform Editor tooltip that could span most of the application when hovering anywhere over the plot.
- Modebar help is now limited to the individual icon under the pointer, matching the behaviour of the map-viewer controls.

## 0.14.0.143 — Consistent free-form tangent proxies

- Fixed Free-form tangent-handle recovery so an edge proxy appears as soon as the real handle leaves the actual waveform plot rectangle, rather than only after it leaves the wider viewer widget.
- Proxy behaviour is now consistent at the top, bottom, left, and right plot edges, including handles that move into axis-label or toolbar margins.

## 0.14.0.142 — Context-sensitive waveform modebar and box zoom

- Hid the waveform viewer's **Select components** modebar tool in passive navigator/encoding modes; it now appears only during Create/Edit authoring, where component manipulation is relevant.
- Fixed rectangle zoom so click-drag draws a visible Plotly-style translucent selection box while dragging and applies the selected axes range on release.
- Box-zoom release is now clamped to the plot area, so dragging slightly beyond an axis edge still completes the zoom instead of silently doing nothing.

## 0.14.0.141 — Map-style waveform viewer controls

- Reworked the Waveform Editor viewer controls into a compact icon-only in-viewer modebar matching the Plotly map viewers, with component selection, box zoom, pan, zoom in/out, reset axes, and PNG export.
- Kept the native waveform canvas and its direct component/free-form editing behavior while adopting the same unobtrusive viewer-control layout used elsewhere in Field Workbench.
- Added intermediate labelled ticks on both waveform axes instead of showing only the vertical minimum/maximum, and separated the axis titles into **Time (s)** and the configured waveform quantity/unit for a map-viewer-like presentation.

## 0.14.0.140 — Periodic cycle-count authoring and AP validation

- Added an editable **Cycle count** control to the Sine and Square waveform tools. Duration and cycle count stay synchronized with frequency, so either physical duration or number of cycles can drive the component length.
- Reflowed periodic generator controls across two rows so the added cycle-count field does not compress magnitude, offset, duty-cycle, or phase inputs.
- Action Potential timing inputs now accept freely typed values and validate their relationships live instead of enforcing ordering by clamping. Invalid timing combinations are highlighted red until they satisfy `0 ≤ depolarization < peak < trough < recovered ≤ 100%`.
- Action Potential signal levels receive the same live validation treatment for `peak > resting > trough`; invalid combinations remain editable while preview/apply are disabled.

## 0.14.0.139 — Action-potential editor preview and layout

- Action-potential components now get an automatic focused preview in the persistent main waveform viewer when the tool opens, so millisecond-scale events remain visible even when appended to a multi-second waveform. **Home** still returns to the full waveform view.
- Removed the redundant Action Potential explanatory paragraph and stacked **Event timing** and **Signal levels** as separate full-width rows to prevent their controls from being compressed side-by-side.
- Changed the Action Potential duration editor to **milliseconds** while preserving seconds internally in the authored waveform recipe and runtime signal.

## 0.14.0.138 — Waveform catalogue action cleanup

- Added a full-width **Delete** button to the bottom of the Waveform catalogue controls.
- Expanded the waveform catalogue right-click menu to include **Edit**, **Rename**, **Duplicate**, **Export…**, and **Delete** for the selected waveform.
- Context-menu commands now follow the same selection/task-state enable rules as the visible catalogue buttons.

## 0.14.0.137 — Free-form selected-point value editing

- Simplified the Free-form tool header to the concise direct-manipulation hint and moved selected-point details into the **Free-form drawing** group beside the drawing controls.
- Added an editable **Vertical position** field for the selected Free-form control point, expressed directly in the waveform's configured physical units (for example mA or V). Numeric changes update the persistent viewer and authored curve immediately.
- Kept graphical dragging and numeric editing synchronized so selecting or moving a point in the viewer refreshes its point number, time position, and physical vertical value.

## 0.14.0.136 — Free-form point repair and component reordering

- Fixed **Add point** in the graphical Free-form editor by routing insertion through the persistent viewer's direct-manipulation model, keeping the visible control points, selection, authored recipe, and live preview synchronized.
- Added a confirmation dialog before **Reset curve** replaces the current free-form control points and tangent adjustments.
- Added **Move up** and **Move down** to **Component actions** so selected waveform components can be reordered one position at a time without having to drag very short components in the viewer. The buttons automatically disable at the first/last component boundaries.

## 0.14.0.135 — Recoverable free-form tangent handles

- Waveform Editor: free-form tangent handles that leave the visible viewer now leave a filled proxy marker pinned to the plot edge in the same direction. The proxy can be grabbed and dragged inward to recover the true off-screen handle without changing the waveform's ordinary plot scale.
- Added **Reset selected tangent** to restore the selected anchor's mirrored tangent to a neutral local default.
- Added **Fit handles** to temporarily expand the viewer limits enough to show every authored free-form tangent handle; **Home** returns to the ordinary waveform view.

## 0.14.0.134 — Unbounded free-form tangent handles

- Waveform Editor: Free-form Bézier tangent handles can now be dragged above or below the current graph bounds, allowing stronger overshoot/undershoot without zooming first. Anchor points remain constrained to the component time domain.
- Free-form tangent guides and handles are no longer clipped to the axes rectangle, so controls remain visible in the surrounding viewer margin while editing beyond the plotted range.

## 0.14.0.133 — Scrollable imports and graphical free-form editing

- Waveform Editor: the inline Text/CSV import mapping form now scrolls vertically instead of compressing its controls when the Waveform-tools workspace is short.
- Waveform Editor: Free form is once again a direct graphical editor in the persistent main viewer. The complete composed waveform remains visible for scale; double-click adds control points, anchors and mirrored tangent handles can be dragged directly, and interior points can be deleted.
- Removed the temporary user-facing Bézier point table while retaining the same exact editable Bézier authoring recipe underneath.

## 0.14.0.132 — Waveform import joins the tool palette

- Moved **Import waveform…** out of the left **Component actions** card and into the main **Waveform tools** grid so all ways of adding waveform content live together beneath the viewer.
- Added a matching compact import icon: an outlined box/tray with a downward arrow entering it, using the same blue line treatment as the waveform-generator icons.
- Kept **Component actions** focused on operations on the selected component only: **Edit**, **Clone**, and **Delete**.
- Updated Waveform Editor help/About text and packaging regressions for the revised layout.

## 0.14.0.131 — Waveform components and responsive editor layout

- Renamed user-facing waveform **chunk** terminology to **component** throughout the editor, including the component browser/actions, plot-selection toolbar, inline tool/import messages, and help text. Internal recipe structures remain unchanged.
- Made the waveform preview vertically responsive by removing the old tall plot minimum and lowering the reusable preview minimum so the graph follows the viewer panel instead of being clipped when the tool area needs more height.
- Moved **Save waveform** and **Cancel** from the top workflow strip to the bottom of the left authoring panel. The recessed top strip now stays focused on the current mode and the Notes visibility control.

## 0.14.0.130 — Compact waveform tool buttons

- Tightened the embedded waveform-generator buttons so the five-tool grid remains legible in narrow editor layouts.
- Reduced generator icon dimensions and icon-to-label footprint, lowered the hard minimum button size, and tightened grid spacing to prevent labels or button outlines from overlapping.

## 0.14.0.129 — Waveform navigator state and shell polish

- Disabled **Rename** and **Duplicate** in Waveform navigator whenever no catalogue waveform is selected, matching Edit/Export/Delete selection behavior.
- Recessed the persistent Waveform Editor workflow/status strip with a sunken frame beneath the menu bar.
- Changed the Waveform Editor workspace backing surface to the same card/group surface used by its panels, eliminating the black gaps between panes in Dark mode while retaining theme-aware Light/Dark styling.

## 0.14.0.128 — Unified waveform import entry point

- Consolidated editable DAC import into the general **Import waveform…** workflow. The separate **Import DAC…** button in Chunk actions and the separate File-menu DAC action are removed.
- Added **8-bit DAC (editable)** as a supported in-editor import source. Selecting a DAC source switches directly into the existing inline DAC timing/value-mapping editor while preserving the original source codes for later chunk editing.
- Added the same DAC path to catalogue-level **Import waveform…**, alongside text/CSV, native Field Workbench waveform files, structured readers, and reusable format profiles.
- Updated Waveform Editor help/About text and packaging regressions to describe one import entry point instead of separate waveform/DAC commands.

## 0.14.0.127 — Waveform chunk browser and dedicated tool workspace

- Added a persistent **Chunks** list beneath the read-only Waveform encoding summary while authoring. The list mirrors authored generator, sampled-import, and editable DAC chunks in playback order and stays synchronized with chunk selection in the main waveform viewer.
- Added a **Chunk actions** card with **Edit**, **Clone**, **Delete**, **Import waveform…**, and **Import DAC…**. Generated/DAC chunks reopen through Edit, while any exact chunk can be cloned or deleted and the physical waveform is rebuilt from the resulting recipe.
- Moved waveform and DAC import entry points out of the generator selector. The **Waveform tools** panel beneath the viewer is now reserved entirely for Square, Sine, DC / pause, Action potential, and Free form, giving those controls the full lower workspace.
- Clarified **Import waveform…** as the general one-chunk importer for native Field Workbench waveform files, text/CSV tables, and supported reusable format profiles; inline import controls still preview in the lower workspace before Apply.
- Updated Waveform Editor help/About text and regression coverage for the chunk-centric authoring workflow.

## 0.14.0.126 — Editable DAC authoring chunks

- Changed DAC imports from generic translated sample chunks into first-class editable authoring segments. Each DAC chunk now stores the original unsigned 8-bit code sequence, sample interval, code-0/code-255 engineering mapping, and source provenance alongside the native physical playback trace.
- Added inline DAC-point editing to the embedded DAC tool. **Edit DAC points…** exposes the stored codes directly; timing, mapping, and point changes all preview the complete resulting waveform in the persistent viewer before Apply.
- Made saved DAC chunks reopenable just like generated chunks. Double-clicking a DAC segment, or using **Edit authored segment…**, restores its original codes and translation controls; Apply rebuilds the full waveform while Cancel leaves the working copy unchanged.
- Updated catalogue-level DAC import to save the same editable authoring recipe, so DAC files imported as new project waveforms remain reopenable after entering Edit mode. Generic text/CSV/native sampled imports remain lossless/reorderable sampled chunks.
- Updated Waveform Editor help/About text and regression coverage for editable DAC recipes and direct source-code preservation.

## 0.14.0.125 — Waveform Editor workflow strip and compact panels

- Added a persistent workflow/status strip directly beneath the Waveform Editor menu bar. It identifies the current state as **Waveform navigator**, **Waveform encoding**, or **Waveform editor**.
- Moved authoring **Save waveform / Cancel** controls out of the left Waveform encoding card and into the left side of the new strip. Removed the redundant “Creating waveform”/“Editing waveform” text and made the encoding **Edit** button span the card width.
- Moved **Show / hide notes** to the right side of the workflow strip. Notes can now collapse completely so the viewer/tool workspace reclaims the full former Notes width, while the toggle remains reachable.
- Added hard minimum dimensions to the waveform-generator and import buttons so icons/text cannot be squeezed into unreadable layouts when the window is narrowed.
- Updated Waveform Editor help and packaging regressions for the new workflow strip and panel behavior.

## 0.14.0.124 — Waveform definition and fixed encoding workflow

- Split Waveform Editor creation into a dedicated definition step before authoring. **Create new** now places Name, Signal type, Units, slew, and voltage-source output resistance in the centre **Waveform information** card beneath the persistent viewer; Waveform tools stay hidden until **Create waveform** confirms the definition.
- After confirmation, the waveform definition moves to a read-only **Waveform encoding** card on the left with an explicit **Edit** button plus the existing **Save waveform / Cancel** controls. Editing an existing catalogue waveform jumps directly to this confirmed authoring state instead of reopening definition controls automatically.
- Made Current/Voltage and engineering units waveform-level state for the whole authoring session. Generator tools, text/CSV imports, structured waveform imports, and DAC imports can no longer redefine the first chunk's encoding; changing it requires the explicit encoding Edit workflow.
- Simplified active-edit imports accordingly: DAC translation now asks only for timing and code-0/code-255 engineering mappings, while text/CSV mapping no longer asks for signal type/units. Compatible structured A↔mA and V↔mV sources are converted automatically to the fixed waveform unit; opposite electrical quantities are rejected with a clear message.
- Tightened Cancel/Esc behavior so cancelling an encoding edit restores the prior definition without mutating the working waveform, while cancelling the initial unconfirmed definition returns to navigation. Updated help and regression coverage for the new three-state navigation → definition → authoring workflow.

## 0.14.0.123 — Waveform Editor embedded-authoring cleanup phase 4

- Finished the Waveform Editor conversion to the embedded authoring model by removing the retired modal generator framework and dedicated free-form builder canvas. Square, Sine, DC / pause, Action Potential, Free Form, generated-segment reopening, and active-edit imports now use only the persistent editor/viewer workflow.
- Reduced `native_dialogs.py` to true external-format boundaries: catalogue DAC translation plus DAC/Rigol export. The shared time-span formatter moved to a small presentation helper used by the embedded tools.
- Tightened authoring state handling: `Esc` cancels the innermost active tool preview first, then Create/Edit; direct Save is blocked while an unapplied preview is active; cross-window waveform requests cannot discard a preview; project close explicitly asks before dropping an unapplied preview.
- Kept File-menu waveform/DAC import available during Create/Edit so those commands enter the same inline import panels as the Waveform-tools buttons, while preventing a second inline tool from being started before Apply/Cancel.
- Updated Waveform Editor help/About text and packaging regressions to describe and enforce the single-viewer embedded workflow and absence of the old builder-window files.

## 0.14.0.122 — Embedded waveform imports phase 3

- Moved in-editor **Import waveform…** into the Waveform tools card beneath the persistent viewer. Text/CSV, native readers, and reusable format profiles now decode one prospective signal inline and preview the complete appended waveform before anything is committed.
- Added focused inline text/CSV mapping for active waveform edits: delimiter/header/encoding, time column or fixed interval, signal column/name, Current/Voltage units, and scale/offset all update the main waveform preview directly. Catalogue-level Import keeps the larger multi-signal/profile-management workflow.
- Reworked in-editor **Import DAC…** so the file picker only chooses the source. Sample interval, Current/Voltage units, and code-0/code-255 engineering mappings now appear inline under the viewer and update the appended preview live; **Apply** commits the translated lossless chunk and **Cancel** restores the working waveform.
- Added a generic non-destructive imported-document preview path to the native editor. Existing waveforms require imported chunks to match their signal type/units; the first imported chunk establishes those settings automatically.
- Kept catalogue-level DAC import as a one-step saved-resource workflow while removing the extra translation/preview dialog from active Create/Edit authoring.

## 0.14.0.121 — Embedded waveform tools phase 2

- Replaced the modal Square, Sine, DC / pause, Action-potential, and Free-form authoring workflow with controls embedded directly in the **Waveform tools** card beneath the persistent main viewer. Selecting a tool now swaps the chooser for that tool's parameters instead of opening a second window and second preview.
- Added non-destructive full-waveform live previews to the main waveform viewer. New generator chunks preview exactly where they will append; editing an authored chunk previews the regenerated complete waveform without changing the working copy until **Apply**. **Cancel** restores the committed working waveform and returns to the tool chooser.
- Routed generated-segment reopening through the same embedded controls, including generated chunks inside exact mixed generated/imported recipes. Double-clicking a generated chunk in Edit mode now opens that chunk inline; imported/sample chunks remain lossless/reorderable but do not expose generator parameters.
- Added an embedded Free-form Bézier anchor table (time, value, tangent time, tangent value) so Free-form creation/editing remains in the single-viewer workflow rather than spawning the old dedicated graph window.
- Disabled waveform-level Save and signal type/unit changes while a generator preview is active so an unapplied preview cannot be mistaken for committed waveform data. Import/DAC embedding remains Phase 3.

## 0.14.0.120 — Waveform editing workspace phase 1

- Fixed the unified Maps footer teardown race that could leave a queued geometry callback touching an already-deleted shared tab widget during UI smoke-test cleanup.
- Reworked Waveform Editor Create/Edit mode so the left Waveform catalogue is replaced by a dedicated **Waveform options** panel with Name, Signal type, Units, slew/output settings, plus **Save waveform** and **Cancel** at the bottom.
- Kept Notes permanently on the right side of the editor, added a centered **Show / hide notes** control beneath it, and made the Notes content editable while authoring.
- Moved the **Waveform tools** selector below the main waveform viewer into the same lower-card position occupied by Waveform information during browse mode. Restored the DC / pause tool to that selector and the Waveform menu.
- Stopped automatically opening a generated-segment dialog as soon as Edit mode begins; the editing workspace now opens first and lets the user choose the next authoring action.
- Restored append semantics for in-editor waveform and DAC imports: active Create/Edit imports no longer ask to replace the working waveform and instead append a lossless chunk. The next phases will embed generator/import controls and non-destructive previews directly into the editor.

## 0.14.0.119 — Waveform catalogue navigation layout

- Moved browse-mode **Edit**, **Export…**, and **Close** out of the Waveform information card and into the left Waveform catalogue, giving navigation mode one consistent action area.
- Reorganized the catalogue controls as **Create new / Edit**, **Rename / Duplicate**, **Import… / Export…**, followed by a full-width **Close** button. Delete remains available from the catalogue context menu or Delete key without occupying another permanent button slot.
- Replaced the fixed Notes/tool side column with a draggable splitter so the Notes panel can be resized against the waveform viewer/information area; the same divider position is used by Waveform tools while creating or editing.

## 0.14.0.118 — Unified Maps footer alignment

- Moved **Save map…** and **Fullscreen viewer** into a shared footer centered beneath the plot area in Maps / Analysis.
- Moved **Show / hide panel** into its own centered right-hand footer zone beneath the Measurement volumes / Brain areas panel.
- Kept the footer aligned with whichever right-hand panel is active, including Scripted result panels and Brain views, while preserving the existing collapsible-panel behavior.

## 0.14.0.117 — Interactive waveform composition

- Added Home, pan, rectangle/wheel zoom, chunk selection, and PNG export controls to the waveform plot.
- Made exact recipe chunks selectable and drag-reorderable in the plot; generated chunks can also be reopened by double-clicking, while imported chunks retain their lossless samples as reorderable recipe sections.
- Added the DC / pause generator for constant zero or non-zero levels over a chosen duration.
- Simplified every waveform builder to one Apply action: new chunks always append, the first chunk initializes an empty waveform, and the builder closes after a successful Apply.
- Changed in-editor waveform and DAC imports to append lossless chunks instead of replacing the working waveform.

## 0.14.0.116 — Pre-release transition audit

- Updated the README and remaining suite-window comments to describe the completed single-signal waveform and timed source-to-coil drive-group architecture rather than the retired channel-based transition model.
- Updated the project-resource test fixture to use waveform quantity/unit metadata instead of the removed channel-count summary.
- Made the three Qt WebEngine-specific UI checks report a clear skip when the host cannot load WebEngine's native system libraries; all non-WebEngine UI and workflow tests continue to run normally.

## 0.14.0.115 — Simpler populated-component underline

- Replaced the SVG background artwork used for populated Project Component labels with a simple `#77ff00` underline beneath the label text. Empty components remain unaccented.
- Kept the existing resource-driven `populated` state unchanged, so the visual simplification does not alter how Model, Maps / Analysis, Exposures, or Datasets determine whether they contain project content.
- Removed the custom cap/centre/cap SVG rendering path and its now-unused highlight assets from Project Manager; component text stays entirely under the normal light/dark theme styling with only the separate green underline painted for populated sections.

## 0.14.0.114 — Overlapped fixed-cap project highlights

- Reworked the populated Project Component highlight around the supplied three-piece SVG geometry: the left/right caps now keep their native aspect ratios and are never horizontally stretched.
- Made only the centre section horizontally stretchable and paint it last, substantially overlapping both caps using the overlap encoded in the source artwork plus a small seam allowance so the cap ends are hidden rather than visibly butted together.
- Split the supplied artwork into internal left/centre/right SVG render assets with tight source view boxes so Qt preserves the intended cap shapes and gradients while the label width changes.

## 0.14.0.113 — Corrected project-component highlight artwork

- Replaced the project-component populated-state highlight with the revised supplied SVG whose left/right spacer paths are fully transparent.
- Kept the simple `.112` stretch-to-fit renderer unchanged so this build isolates the artwork change: normal theme text colour, text-sized highlight width, and existing padding are preserved.

## 0.14.0.112 — Simpler project-component highlight

- Replaced the experimental multi-part project-component highlighter with the new simpler supplied SVG artwork. The label renderer now trims the SVG to its visible painted bounds and stretches that single mark directly behind the component title.
- Removed the three-slice/cap composition logic and its spacer offset. Highlight width is still derived from the actual title text, while normal theme text colour remains untouched.

## 0.14.0.111 — Project component highlight experiment

- Replaced the Project Manager's populated-component green text treatment with the supplied stretchable green highlighter artwork, keeping the component title text in the normal light/dark theme colour.
- Added a reusable three-slice SVG-backed label renderer. The complete SVG is composited first so its overlapping cap/centre effects stay intact; fixed end regions preserve the hand-painted caps while the middle absorbs width changes for labels such as Model, Maps / Analysis, Exposures, and Datasets.
- Sized the highlight from the actual title text rather than the fixed component column, so shorter and longer component names receive appropriately sized marks. The existing `populated` state still controls when the effect appears.

## 0.14.0.110 — Waveform/exposure workflow closure

- Completed Phase 6 with an end-to-end reference-integrity pass across Waveform Editor, Exposure Editor, Project Manager, and project reopen. Waveform edits and renames keep the stable waveform resource ID used by exposure drive groups; duplication creates a new independent ID and does not retarget existing exposures; deleting a referenced waveform remains blocked until the exposure reference is changed or removed.
- Protected Waveform Editor working copies from cross-window refreshes. Model/Exposure resource changes may refresh the catalogue while Create/Edit is active without replacing the private working waveform, and an Exposure Editor request to open another source now offers Save/Discard/Cancel instead of silently switching documents.
- Integrated waveform drafts with project lifetime. Closing Waveform Editor with unsaved changes offers Save/Discard/Cancel, and Project Manager resolves private waveform edits before it calculates project dirtiness so a just-saved waveform draft participates in the normal project-file Save/Discard/Cancel decision.
- Protected Exposure Editor drafts at project close. Because an open exposure dialog is intentionally private until **Save exposure**, Project Manager now keeps the project open and asks the user to finish that Save/Cancel decision rather than silently destroying an uncommitted sequence.
- Propagated live resource changes between suite applications. Waveform catalogue changes refresh open Exposure Editor source names and electrical diagnostics; model-side coil/load changes and shared-project Undo/Redo refresh exposure coil/source references without rewriting the protocol. Sibling waveform/exposure commits also refresh Model Editor's shared Undo/Redo state and project-dirty title.
- Refreshed the remaining Maps/help wording around the final ownership boundaries: Waveform Editor creates reusable electrical sources, Exposure Editor owns timed source-to-load wiring/sequencing, model coils own R/L, and Maps consumes the resolved per-coil current timeline. Updated stale regression fixtures/assertions and added Phase 6 lifecycle/Cancel/reference tests.

## 0.14.0.109 — Waveform import/export cleanup

- Completed Phase 5 by adding a focused **Text / CSV table** importer for ordinary waveform data. The importer previews the source table, supports comma/tab/semicolon/whitespace layouts plus header/encoding choices, maps an explicit time column (or a fixed interval only when the file has no time column), and lets each selected data column define its own waveform name, Current/Voltage type, compatible engineering units, and explicit `output = source × scale + offset` translation.
- Catalogue-level import now treats multi-signal source files correctly: every checked signal is saved as an independent one-signal waveform resource in one operation. Import invoked while Create/Edit is active switches to a deliberate single-signal replacement workflow instead of silently choosing one channel. Advanced format-profile imports use the same checkable selection model; native Field Workbench waveform files are already canonical one-signal resources.
- Normalized import provenance is retained with the waveform. Text/CSV records remember column/timing/conversion choices, advanced format-profile imports snapshot the profile used, and legacy DAC translation records its interval and code-to-engineering mapping. Raw source rows remain visible only inside the import preview.
- Standardized DAC import with the rest of the catalogue workflow: importing from browse mode creates the translated waveform resource directly; importing during an active waveform task replaces only that working copy. Device-specific timing and encoding remain confined to the explicit DAC boundary.
- Made export resource-oriented. **Export…**, FWB JSON, CSV, DAC, and Rigol exports now operate only on the selected saved catalogue waveform, so unsaved working edits can never leak into an exported file. Added a simple physical two-column **Export CSV…** path (time in seconds + waveform value); FWB JSON remains the lossless metadata/recipe-preserving interchange format.
- Simplified the native `.fwbwave` interchange format to the same forward-only model as the catalogue: version 2 contains exactly one Current/Voltage signal plus waveform-level source metadata and optional authoring recipe. The never-released version-1 multi-channel/arbitrary-signal path and its serializer were removed rather than carried as migration code.
- Updated Waveform Editor and playback help around the new import/export boundary and added headless text-table parsing/export regression coverage plus Qt catalogue/import checks for multi-signal creation.

## 0.14.0.108 — Phase 4 source/load cleanup

- Completed Phase 4.7 and closed the electrical source/load overhaul. Saved exposures now have one strict canonical schema only: timed steps contain source-to-coil drive groups, every group requires a real waveform source and at least one coil load, and unknown/retired exposure fields are rejected at every schema level instead of being migrated or silently ignored. Drive groups store stable resource IDs plus Series/Parallel topology; waveform names are resolved from the live waveform resource rather than cached in exposure data.
- Simplified Maps' built-in scripted source into an explicitly lightweight current generator. The hidden playback controller no longer carries the retired voltage/RL controls, old coil-sequencer editor, exposure-shaped quick-source state, or direct project-waveform runtime path. Voltage sources, coil load physics, concurrent drive groups, and timed sequencing now enter playback only through saved Exposures.
- Tightened waveform source metadata to the same forward-only model. Electrical metadata has one authoritative waveform-level location: slew for Current/Voltage and output resistance for Voltage only. Old waveform-side resistance/inductance fields and signal-level duplicate electrical metadata now fail validation rather than being stripped as compatibility data.
- Made portable exposure files strictly self-contained by requiring every referenced waveform resource to be bundled and refusing save/export/import when a source is missing. Scene-to-scene exposure import likewise requires the referenced waveform resources instead of preserving unresolved source IDs.
- Renamed the viewer-facing sequencer trace to **Drive state trace**, refreshed camera/playback terminology and help around the source → load → exposure → resolved-current workflow, and expanded regression guards for the canonical exposure/source boundaries. Base model-file compatibility remains unchanged.

## 0.14.0.107 — Exposure electrical diagnostics

- Completed Phase 4.6 by adding live electrical diagnostics to Exposure Editor. Selecting a drive group now reports its effective uncoupled R/L load, predicted source peak current/voltage/power, load-terminal peak voltage, and per-coil branch peak current/voltage; selecting a step gives a compact diagnostic summary for all concurrent groups.
- Added the same live preview to the Add/Edit drive-group dialog so source, coil selection, and Series/Parallel changes immediately show their electrical consequences before the group is committed. Exposure save now resolves every drive group once, catching missing/nonphysical model-side electrical load data before playback.
- Added explicit diagnostic cautions for geometry-estimated inductance, the uncoupled mutual-inductance assumption, near-zero effective load resistance, and ideal current-source steps whose required voltage includes an impulse. Source-limit notes make clear that current-source compliance voltage and voltage-source current/power ratings are not yet modeled.
- Kept diagnostics separate from exposure data: they are derived from waveform source metadata and model-side coil R/L rather than stored as another set of electrical properties. The editor also explains waveform repeat/truncate behavior for the selected step and labels the compact preview as a one-waveform, zero-initial-current engineering preview rather than full scheduled playback.

## 0.14.0.106 — Exposure playback integration

- Completed Phase 4.5 by feeding canonical saved exposures directly into Scripted Maps playback. The playback worker now samples each exposure step into resolved per-coil current timelines and drives the cached magnetic-field bases with those physical coil currents rather than projecting the exposure back through the retired single-source controls.
- Step duration is now the scheduling authority: shorter saved waveforms repeat as signal-generator programs to fill the step, while longer waveforms are truncated. Repeats within a step share continuous RL state, and unchanged coil-set/topology groups carry their final inductor current into the next step even when the waveform source changes. Rewiring/removing a group remains an explicit ideal-switch boundary until a discharge/flyback path model is added.
- Preserved concurrent drive groups, voltage-source output resistance, and series/parallel branch-current differences all the way through 2D, 3D, and Brain playback, observation traces, measurement-volume statistics, and Brain-region analysis. Exposure sequencing metadata remains available to the viewer without re-gating the already-resolved electrical timeline.
- Saved exposures marked to loop now open playback with the viewer Loop control enabled by default. Maps' quick built-in waveform source remains a separate one-off current-source path.

## 0.14.0.105 — Exposure electrical resolver

- Completed Phase 4.4 by adding a source/load electrical resolver for canonical exposure drive groups. Current and voltage waveform sources are combined with each selected coil's resolved model-side resistance and self-inductance, with voltage-source output resistance included as a series source resistance.
- Added uncoupled series and parallel RL resolution. Series groups share one branch current; parallel voltage groups solve the coupled branch currents (including finite source resistance), while parallel current groups enforce the commanded total current and solve its distribution across the RL branches. Mutual inductance remains explicitly outside this first resolver.
- Added adapter-level `resolve_exposure_drive_group()` and `resolve_exposure_step_electrical()` boundaries that return per-coil current/voltage traces plus source/load diagnostics for the upcoming playback integration. A coil cannot be assigned to two independent drive groups in the same step.
- Removed the temporary pre-drive-group exposure compatibility path. Saved exposures now use only the step/drive-group schema; the old global drive/waveform/sequencer projection, editor migration, and single-waveform runtime resolver are gone. Portable exposure files now use the canonical multi-waveform bundle only. Base model-file compatibility remains unchanged.
- Kept Maps' built-in quick waveform controls as internal playback state rather than treating them as saved exposure data. Saved exposure playback now validates/resolves the electrical groups directly and waits for Phase 4.5 to place those per-coil currents on the playback timeline.

## 0.14.0.104 — Phase 4.3 regression cleanup

- Updated the waveform catalogue Qt regressions to match the Phase 3.8 cleanup: the removed raw `editor_panel` is now asserted absent, and empty-preview state is checked through the shared preview canvas `signal` property rather than the retired channel-oriented API.
- Fixed legacy exposure-to-drive-group migration so a saved waveform source is preserved as an incomplete drive group even when the legacy exposure or sequencer step has no active coil ids. This prevents the source assignment from disappearing when the Phase 4.3 editor opens older/incomplete exposure data.

## 0.14.0.103 — Exposure drive groups

- Completed Phase 4.3 by replacing Exposure Editor's global “waveform + driven coils + sequencer” authoring surface with an ordered step model. Each step can contain zero or more concurrent **drive groups**.
- Each drive group now references one saved project waveform source, selects one or more physical coil loads, and records an explicit **Series** or **Parallel** connection. Source characteristics remain with the waveform; coil resistance/self-inductance remain with the model.
- Added a nested step/drive-group editor with per-step duration, sequence looping, add/edit/remove actions, waveform editing shortcuts, and source summaries. The exposure library now summarizes source count, drive-group count, coil count, step count, and total sequence duration.
- Added schema helpers for all waveform/coil references in a drive-group protocol. Waveform deletion checks now see references from every group; portable exposure export/import bundles and remaps all referenced waveform resources rather than assuming a single source. Scene-to-scene exposure import performs the same multi-source remapping.
- Kept a deliberately limited legacy runtime projection only as a transition to Phases 4.4–4.5. Maps blocks voltage-source, parallel multi-coil, and concurrent/multi-source drive-group playback rather than silently simulating them with the old single-drive model.

## 0.14.0.102 — Coil electrical load properties

- Completed Phase 4.2 by moving load-side electrical properties into the coil model. Each coil now stores independent resistance and self-inductance source modes: use the winding/geometry estimate by default or select an explicit measured/manual override.
- Kept the existing winding resistance calculation as the default resolved load resistance and added an approximate air-core self-inductance estimate based on the mean current-path perimeter, turn count, and winding-bundle size. The estimate is deliberately presented as an engineering default rather than a precision coupled-inductor solve.
- Added an **Electrical Load** section to the coil inspector with resistance and inductance estimate/override controls plus a live resolved R/L summary. Read-only winding estimates now show the resolved load values and identify whether each came from the model estimate or an override.
- Added `coil_electrical_load()` as the model-side source/load boundary for the upcoming exposure resolver. Project/model serialization, portable-scene records, Scene Recipe import/context, and help now preserve/explain coil R/L provenance. Mutual inductance between separate coils remains outside the model for now.

## 0.14.0.101 — Voltage-source output resistance

- Started Phase 4.1 by formalizing waveform records as electrical source definitions. Voltage waveforms now expose an **Output resistance** property in ohms in the Waveform information card; it defaults to 0 Ω for an ideal voltage source and remains editable alongside the other source characteristics.
- Kept current-source semantics intentionally simple: Current waveforms have no output-resistance field in the UI or canonical project/native metadata. Switching a waveform from Voltage to Current removes any stale voltage-source resistance.
- Canonicalized source electrical metadata at project/native serialization boundaries. Source slew remains valid for both signal types; voltage output resistance must be finite and non-negative; coil/load resistance and inductance keys are stripped from waveform resources.
- Propagate voltage-source output resistance through project-waveform runtime metadata so the later exposure source/load resolver can consume it without re-reading editor state. Updated Waveform Editor help and added source-model regression coverage.

## 0.14.0.100 — Waveform catalogue regression cleanup

- Updated the catalogue integration regression to exercise a real supported Sine authoring recipe rather than the removed synthetic `test` generator alias, keeping the Phase 3.8 cleanup strict while restoring the full Qt test suite.

## 0.14.0.99 — Waveform builder cleanup

- Finished Phase 3.8 by removing the dormant raw time/value editor from the Waveform Editor entirely. The native workspace now owns only one electrical signal, its persistent preview, metadata, and the unified builder/apply pipeline; Add/Remove/Paste sample-table authoring paths no longer exist.
- Cleaned the single-signal UI/backend vocabulary left over from the old channel-oriented editor and removed the temporary generator compatibility class. Square, Sine, Action Potential, and Free Form now all derive directly from the unified builder framework.
- Removed the short-lived `.91`–`.95` `generation` metadata migration path. The versioned `authoring_recipe` introduced in Phase 3.5 is now the only generator-authoring model; imported/compiled sections remain explicit provenance rather than reconstructed generator settings.
- Kept digitization/timing controls only at file-format boundaries that require them, such as legacy DAC translation and Rigol export. Updated help/UI wording so normal waveform construction consistently talks in physical signal terms and Free Form consistently uses Bézier anchors.
- Added packaging guards to prevent the raw sample table, legacy generator wrapper, and transitional generation-metadata path from returning.

## 0.14.0.98 — Reopen generated waveforms in their builders

- Finished Phase 3.7 by making exact recipe-backed waveforms editable through the same generator controls that originally authored them. Selecting **Edit** on a single generated waveform reopens Square, Sine, Action Potential, or Free Form with its saved physical settings restored.
- Added generated-segment selection for composed waveforms. Multi-chunk recipes list each authored segment by type/time (and frequency or Bézier-anchor count where useful), and **Edit generated segment…** remains available in the Waveform tools panel while editing.
- Segment edits preview the complete resulting waveform and regenerate the full all-generated recipe transactionally on Apply. Changing one segment's duration shifts/recomposes following generated chunks without exposing or reconciling sample intervals.
- Restored Free-form Bézier anchors and mirrored tangent geometry directly into the interactive canvas, including correct editing coordinates when the selected segment sits in the middle of a longer waveform.
- Detached recipes and compositions containing compiled/imported baselines remain provenance only rather than being presented as exactly reconstructable generator controls. Added headless regression coverage for recipe-to-generator reconstruction and generated-sequence recomposition.

## 0.14.0.97 — Waveform chunk composition

- Finished Phase 3.6 by moving Replace/Append composition into one shared physical-trace routine used by both the live builder preview and the actual Apply operation, so the graph now shows the exact trace that will be committed.
- Appended chunks retain their own automatically chosen internal time grids; Field Workbench does not resample an existing waveform or require the next generator to match its point spacing. Builder status now reports the append boundary and resulting waveform duration.
- Preserve the preceding chunk's final sample at discontinuous joins instead of overwriting it with the first value of the new chunk. The new command begins immediately after the authored boundary using floating-point ordering rather than an invented/user-visible sample interval.
- Made chunk application transactional: composition and the updated authoring recipe are validated before the working waveform is mutated. Recipe segment positions stay waveform-relative even when an imported source carried a non-zero time origin.
- Added headless regression coverage for Replace normalization, continuous joins, discontinuous joins, and independent chunk time grids.

## 0.14.0.96 — Structured waveform authoring recipes

- Finished Phase 3.5 by separating waveform authoring intent from the sampled playback representation. Builder-created waveforms now store a versioned `authoring_recipe` with one ordered recipe segment per generated chunk.
- Preserved meaningful generator controls rather than digitization details: Square/Sine/Action-potential parameters are stored in physical terms, while Free-form recipes retain the complete cubic-Bézier anchor and mirrored-tangent geometry. Internal sample count/spacing are deliberately absent.
- Composition recipes retain generated segments across Append operations; when the pre-existing section has no recoverable recipe it is represented explicitly as a compiled baseline instead of inventing generator settings. Short-lived .91-.95 `generation` metadata is converted opportunistically when another segment is appended.
- Added recipe integrity tracking. Direct sample edits keep the original recipe only as **detached provenance**, preventing stale generator controls from being treated as authoritative. Signal type/unit changes keep the recipe's signal context synchronized.
- Validate/canonicalize authoring recipes at native/project serialization boundaries and use the structured recipe for the Waveform Editor Source summary.

## 0.14.0.95 — Free-form Bézier builder phase

- Finished Phase 3.4 by replacing the temporary Free-form control-point table with a direct-manipulation Bézier editor inside the unified full-preview builder.
- Free-form authoring now works directly on the graph in physical time and electrical units: double-click adds anchors, dragging anchors reshapes the signal, and mirrored tangent handles bend smooth cubic Bézier segments. Interior anchors can be removed with Delete/Backspace or right-click.
- Kept Replace/Append context in the interactive graph, including the existing waveform and append boundary when applicable, while the authored Bézier path remains directly editable.
- Stored Bézier anchor/tangent geometry in generator provenance and moved curve sampling completely behind the authoring layer; Free form exposes no output-point count, sample interval, bit depth, or encoding controls.

## 0.14.0.94 — Action-potential builder phase

- Finished Phase 3.3 by giving Action Potential a dedicated polished panel inside the unified full-preview builder instead of leaving it as a generic collection of numeric fields.
- Split authoring into clear **Event timing** and **Signal levels** groups. Timing remains relative to the chosen physical segment duration, while the builder derives the actual start, peak, trough, and recovery times automatically.
- Added live readouts for the rise, peak-to-trough, and recovery intervals plus peak/trough excursions from the resting level, all in the waveform's physical electrical units.
- Kept the generated waveform model free of user-facing point counts, sample intervals, bit depth, or encoding controls.

## 0.14.0.93 — Square and sine builder phase

- Finished Phase 3.2 by moving Square and Sine onto a dedicated periodic-signal layer within the unified builder framework. Both builders now share the same frequency, peak-to-peak, offset, duration, placement, live-preview, and apply behavior instead of duplicating those controls.
- Kept shape-specific controls focused: Square adds duty cycle; Sine adds phase. All authoring remains in physical time and electrical units with no point count, sample interval, bit depth, or encoding controls.
- Added live derived readouts for period, cycle count over the chosen duration, and the resulting low/high signal range so users do not need to calculate those values themselves.
- Updated Waveform Editor help to describe the full-preview Square/Sine workflow and the derived readouts.

## 0.14.0.92 — Unified waveform builder framework

- Added a reusable waveform-builder shell with a large live preview above the type-specific controls, matching the WaveBuilder-style authoring flow while keeping all controls in physical signal terms.
- Square, Sine, Action potential, and Free form now use the same builder window framework. Numeric edits, duration, placement, and Free-form control-point changes regenerate the preview without changing the working waveform until **Apply** is pressed.
- Append mode previews the complete resulting waveform and marks the boundary where the new segment begins; Replace mode previews only the generated result.
- Extracted the theme-aware waveform preview canvas so the main Waveform Editor and builder windows share the same graph rendering. Added an overridable preview-canvas hook for the later interactive Bézier Free-form builder.

## 0.14.0.91 — Signal-generator waveform authoring

- Replaced point-count/sample-interval waveform construction with physical signal-generator controls. Square wave now uses frequency, duration, peak-to-peak magnitude, offset, and duty cycle; sine uses frequency, duration, peak-to-peak magnitude, offset, and phase.
- Reworked Action potential and Free form around physical duration and shape controls. Internal curve representation is automatic and generated segments can be appended without matching an existing sample interval.
- Removed raw-data editing and sample-count presentation from the normal Waveform Editor UI. Generation settings are retained as source/provenance metadata instead of making digitization part of the authoring model.
- Kept timing/encoding controls only at file boundaries that actually require them, notably legacy DAC import/export and vendor-format translation.

## 0.14.0.90 — Single-waveform catalogue, phase 2

- Consolidated project waveform storage around one electrical signal per entry; project saves reject multi-signal documents and invalid signal units. Single-waveform native exports use the same version 2 signal record.
- Removed waveform-channel controls and channel references from Waveform Editor and Exposure Editor. Multi-signal file imports offer a signal choice before opening a reviewable working copy.
- Exposures reference waveforms by stable ID, retain coil assignments/sequencing and physical drive settings, and use the selected waveform’s signal type and slew rate. Rename, duplicate, project reload, and portable exposure import/export use the same source mechanism.
- Removed coil resistance/inductance metadata from saved generator signals and updated help. No legacy project-waveform migration is included.

## 0.14.0.89 — Waveform generator properties

- Removed inductance from Waveform Editor browsing and editing. Waveforms describe the generator signal; coil resistance and inductance belong with the model and exposure setup.
- Saving waveform edits clears the former waveform inductance field while retaining the signal slew setting. Updated help accordingly.

## 0.14.0.88 — Waveform information grouping

- Swapped **Signal type** and **Samples** in the Waveform information card so editable electrical choices (Signal type and Units) stay together on the left, while derived waveform statistics (Duration, Samples, and Peak-to-peak) stay together on the right.

## 0.14.0.87 — Electrical waveform characteristics

- Restricted Waveform Editor authoring/import choices to playback-relevant electrical signals: **Current** or **Voltage**, with compatible **mA/A** or **mV/V** units. New blank waveforms now start as Current in mA.
- Replaced the free-text electrical/conversion field with explicit waveform characteristics. Peak-to-peak magnitude is calculated directly from samples; slew rate is saved in SI units and displayed in the selected engineering unit per millisecond.
- Current waveforms expose a current slew rate. Voltage waveforms expose a voltage slew rate plus optional inductance in mH. The same fields remain read-only while browsing and become editable in Edit/Create mode.
- Kept Source/provenance separate from user Notes, and retained raw sample access only behind **Edit data…**. Updated DAC/general import format controls and Waveform Editor help to use the Current/Voltage model.

## 0.14.0.86 — Waveform Editor information/data separation

- Kept the waveform graph and information card as the normal editor surface; entering Edit now turns the same information fields into editable controls instead of exposing the raw sample table.
- Moved waveform notes into the full-height right-hand panel while browsing. The same panel becomes **Waveform tools** only while creating or editing a waveform.
- Hid raw time/value samples by default and added an explicit **Edit data…** tool that temporarily swaps the lower information card for the raw table; legacy multi-signal resources retain only a selector there until Phase 2 splits them.
- Added editable waveform name, signal name, quantity, units, conversion/origin descriptions, and notes to the working copy while keeping duration and sample count derived/read-only.

## 0.14.0.85 — Waveform Editor workbench layout

- Reworked the Waveform Editor right side around the WaveBuilder-style layout: a persistent large waveform plot above one mode-dependent lower panel, with the waveform catalogue unchanged on the left.
- The plot now remains visible even when the project contains no waveforms; the lower information panel shows an empty-state hint instead of replacing the whole workspace.
- Moved Square pulse, Sine wave, Action potential, and Free form into a tall waveform-tools strip on the far right and added Import waveform / Import DAC there so construction tools span the full plot-and-information height.
- Browsing keeps waveform information read-only. Entering Edit/Create swaps the lower information area for the existing editable waveform/sample controls with Save to project / Cancel, while generator tools can enter an explicit working-copy edit automatically.
- Changed ordinary waveform import to load a reviewable working copy before project save, matching the new task-oriented layout.

## 0.14.0.84 — Waveform Editor browsing layout, phase 1

- Reorganized Waveform Editor around a persistent left-hand waveform catalogue, a large selected-waveform preview, and a compact information panel below the plot.
- Removed the separate Load step: selecting a catalogue entry now previews it immediately. Consolidated Create new, Import, Duplicate, Rename, and Delete around the catalogue and removed the redundant top toolbar/document button rows.
- Hid channel controls, raw sample tables, and waveform generators during normal browsing. Existing detailed editing remains available behind Edit, while Create new exposes the existing generators only for the active creation task with Save to project / Cancel controls.
- Added a clear empty-project state and browsing metadata for name, duration, sample count, quantity, units, signal count, recorded conversion information, origin, and notes. Existing multi-signal resources remain intact for the Phase 2 single-waveform migration.

## 0.14.0.83 — Project Manager banner alignment

- Replaced the Project Manager title banner with the trimmed supplied artwork.
- Aligned the banner's visible right edge with the dashboard header/cards by using the Project Manager layout margin rather than transparent SVG padding.

## 0.14.0.82 — Project Manager banner spacing

- Replaced the Project Manager title banner with the latest supplied artwork, adding transparent space on the right so the wordmark no longer sits directly against the window edge.

## 0.14.0.81 — Banner revision and test cleanup

- Replaced the Project Manager title banner with the latest supplied artwork and updated its display aspect ratio.
- Updated 3D map UI smoke expectations to the current Heatmap / Heatmap + fluxlines wording.
- Made the 2D contours-with-fluxlines test validate display-layer composition independently of whether the symmetric test fixture produces traceable in-plane fluxlines.

## 0.14.0.80 — Project Manager banner orientation

- Replaced the Project Manager title banner with the revised artwork, keeping the banner right-aligned while placing the accent stripe to the left of the wordmark.

## 0.14.0.79 — Project Manager banner refresh

- Updated the Project Manager Save button with the revised supplied SVG icon.
- Replaced the title banner with the new wide design and anchored it to the right side of the toolbar.

## 0.14.0.78 — Project Manager icons

- Updated the Project Manager New, Open, and Save buttons with the supplied SVG icons.

## 0.14.0.77 — Exposure Editor wording cleanup

- Simplified Exposure Editor guidance around the user-facing concept: a waveform driving one or more coils, with optional coil sequencing over time.
- Removed stable-ID, portable-resource, and reference-storage jargon from the main Exposure Editor and New/Edit exposure guidance.

## 0.14.0.76 — Measurement panel copy cleanup

- Simplified the unified viewer Measurement volumes panel around the user-facing volume → analyze → snapshot workflow.
- Removed frozen-launch implementation terminology from the panel status text, tooltips, and analysis messages.

## 0.14.0.75 — Scripted field-view constraint

- Limited the unified Maps **Field view** control to Heatmap in Scripted mode, since scripted playback does not render fluxlines.
- Kept the full Heatmap / Fluxlines / Heatmap + fluxlines choices in Static mode.
- Added a defensive 3D playback setting so scripted requests cannot retain a fluxline display mode internally.

## 0.14.0.74 — Unified Maps option naming

- Renamed 2D Map plane and 3D/Brain Map volume sections to **Map options** for consistent terminology across views.
- Renamed **Sample config** to **Sampling options**.
- Ordered **Heatmap options** before **Fluxline options** in the 2D setup pane to match 3D.

## 0.14.0.73 — 2D map control cleanup

- Moved 2D Scene options directly below Sample config to match the 3D control order.
- Renamed the 2D Field view options section to Contour lines.
- Allowed contour lines to be enabled with Heatmap, Fluxlines, or Heatmap + fluxlines; Fluxlines-only maps now perform the scalar sampling needed for the contour overlay without displaying a heatmap.

## 0.14.0.72 — Unified 3D field-view controls

- Made the 3D **Field view** selector match 2D: Heatmap, Fluxlines, or Heatmap + fluxlines, and kept it at the top of the setup pane.
- Renamed the 3D Slices / Full Volume selector to **Render mode** and moved it into Map volume so field representation and 3D rendering method are separate concepts.

## 0.14.0.71 — Maps control organization

- Moved the applicable Field view selector to the top of the 2D/3D setup pane, immediately before Map plane/Map volume.
- Moved Flux density out of Sample config and into Fluxline options alongside the other fluxline-specific controls.

## 0.14.0.70 — Green text outline refinement

- Reduced the black outline on active green Project Manager text from 0.5 px to 0.3 px for a subtler light-theme edge.

## 0.14.0.69 — Refined Project Manager text outline

- Reduced the black outline on active `#75ff00` Project Manager text from 1.4 px to 0.5 px for a crisper, lighter appearance.

## 0.14.0.68 — Outlined active Project Manager labels

- Added native Qt path-stroked text for the Project Manager's active green project title and populated Project Component labels.
- Active `#75ff00` text now has a thin black outline for stronger contrast in the light theme; inactive/empty labels continue using normal theme text rendering.

## 0.14.0.67 — Project Manager banner update

- Replaced the Project Manager title banner with the latest revised SVG artwork.
- Preserved the centered toolbar banner layout while sanitizing the SVG for reliable Qt rendering.

## 0.14.0.66 — Project Manager content highlighting

- Saved/named project titles now remain `#75ff00`; Untitled projects continue using the active theme text colour.
- Project Component labels turn `#75ff00` whenever their section contains project resources, while empty sections retain the normal theme colour.

## 0.14.0.65 — Loaded-project title cue

- Newly opened project files briefly highlight their Project Manager title in `#75ff00`, then fade back to the active theme text colour.
- The cue runs only when a project file is newly opened; selecting an already-open tab or creating an empty project does not retrigger it.

## 0.14.0.64 — Scripted trace cleanup

- Removed the redundant Scripted-source hint about trace controls being below Camera.
- Sequencer trace can now visualize the ordinary Scene-options active-coil set even when Advanced sequencer is off.
- Advanced sequencer traces continue to show timed step/active-coil changes as before.

## 0.14.0.63 — Persistent Maps measurement panel

- The Maps/Analysis right-hand measurement/region panel now preserves the user-selected visibility and width while switching Static/Scripted and 2D/3D/Brain views.
- Renamed the shared panel toggle to **Show / hide panel**.
- Moved the frozen-field status beneath the Measurement volumes heading and arranged Static snapshot actions in two rows for narrow panel widths.

## 0.14.0.62 — Project Manager banner refresh

- Replaced the Project Manager toolbar banner with the revised larger-text SVG banner.
- Updated the banner aspect ratio so the centered wordmark fills the available toolbar space better.
- Shipped the banner as sanitized path-based SVG content for reliable Qt rendering.

Release history is listed newest first. Add future changes under **Unreleased**,
then move them into a versioned entry when preparing that release. Record user-visible
changes, fixes, and relevant build or compatibility notes. Keep usage instructions
in Help Topics.

This history consolidates the available development notes; it is not a complete
record of every earlier release. Older entries describe the behavior at that time
and may be superseded by newer entries.

## Unreleased

No changes recorded yet.

## 0.14.0.61 — Unified Maps display cleanup

- Moved Show grid into Scene options for 2D, 3D, and Brain Maps views.
- Renamed Display Options to Heatmap options; 2D/3D Heatmap options now hide automatically when the selected field display is fluxlines-only, while the field-display selector remains available.
- Reduced the effective minimum width of the unified right-hand Measurement/Brain Areas panels so they can be dragged substantially narrower before collapsing.

## 0.14.0.60 — Unified Maps measurement panel

- Moved Static Measurement-volume analysis/snapshot controls out of the left setup pane and into a persistent right-hand measurement area shared by the unified Maps Selector and ordinary Static 2D/3D result views.
- The shared Measurement-volumes panel now lists every eligible frozen-scene Measurement object and can analyze one or all rows, displaying RMS, mean, P95, peak, uniformity, direction, and volume with the same table/filter presentation used by Scripted playback.
- Added cached View / save snapshot access so an analyzed frozen result can be reopened and stored without repeating its field solve, plus direct Saved Results access from the measurement panel.
- Scripted and Brain tabs continue to use their existing per-result Measurement/Brain Areas panels; the unified shell yields the right side to those panels instead of showing a duplicate.
- Updated Maps help and regression guards for the always-present measurement/region workflow.

## 0.14.0.59 — Centered Project Manager banner

- Replaced the Project Manager wordmark with the revised supplied SVG artwork.
- Kept the converted FIELD WORKBENCH lettering as vector paths while removing stale empty Inkscape text/flow metadata for reliable Qt rendering.
- Centered the banner horizontally across the full Project Manager toolbar/window instead of attaching it to either edge or centering it only in the space remaining after toolbar controls.

## 0.14.0.58 — Reliable Project Manager banner

- Moved the Field Workbench banner to sit immediately after the New/Open/Save project controls on the left side of the Project Manager toolbar.
- Converted the banner wordmark from Inkscape live/flowed SVG text into vector path outlines so Qt and other SVG renderers no longer depend on font availability or support for CSS `shape-inside`.
- Added packaging guards that require the shipped banner to remain raster-free and live-text-free.

## 0.14.0.57 — Project Manager toolbar banner

- Added the supplied Field Workbench SVG wordmark to the Project Manager project-file toolbar, right-aligned on the same row as New/Open/Save.
- The banner preserves its native aspect ratio and renders at the current toolbar icon height, so it stays compact and sharp instead of consuming extra vertical space.

## 0.14.0.56 — Viewport multi-selection and clear-selection clicks

- Fixed Ctrl/Command-click and Shift-click selection in the Model Editor 3D viewport by carrying keyboard modifiers from the actual mouse gesture instead of relying on Plotly's point-click event to preserve them.
- Additive viewport clicks now toggle objects into/out of the same multi-selection used by the scene tree, while ordinary object clicks continue to replace the selection.
- A stationary click on empty 3D scene space now clears the current selection; camera drags and clicks on Plotly toolbar/legend controls do not clear it.

## 0.14.0.55 — Model Editor drag-over-object orbiting

- Changed 3D viewport picking so a selectable object is chosen only after a genuine stationary click rather than immediately when Plotly reports a GL3D click hit.
- Click-drag gestures that begin directly on coils, magnets, meshes, probes, or other selectable scene geometry now remain camera-orbit gestures instead of being interrupted by a selection-overlay refresh.
- Added a small five-pixel click/drag threshold plus late-click suppression so Ctrl/Shift multi-selection keeps normal click behavior without stealing an in-progress orbit.

## 0.14.0.54 — Idle Project Manager performance

- Removed the Project Manager's 1.2-second background polling loop. Project/model/resource dashboard updates are now driven by the existing suite change signals plus the explicit Refresh command.
- Eliminated redundant registry rebuilds and duplicate model signature serialization during Project Manager refreshes, reducing background CPU work while Model Editor is open and returning CPU time to interactive 3D rotation.
- Updated project-resource comments and packaging regression guards so the suite cannot quietly reintroduce aggressive Project Manager polling.

## 0.14.0.53 — Open-project icon refinement

- Replaced the Project Manager Open-project toolbar SVG with the revised custom icon supplied for the release candidate.
- Kept the existing Qt fallback behavior and all other Project Manager toolbar/application icons unchanged.

## 0.14.0.52 — Project toolbar icons

- Replaced the Project Manager Open and Save toolbar glyphs with the new bundled custom SVG artwork.
- Kept themed Qt standard icons as fallbacks if either toolbar SVG cannot be loaded.
- Extended packaging checks to verify both toolbar assets are true vector SVGs and are included with the existing icon bundle.

## 0.14.0.51 — Complete Project Manager icon set

- Replaced the remaining temporary Saved Results and Dataset Manager launcher icons with the supplied custom 128×128 SVG artwork.
- Project Manager now uses bundled custom SVG icons for all six application launchers, with the existing Qt standard-icon fallback retained for missing or invalid assets.
- Extended packaging regression coverage to verify all six launcher SVGs are present, vector-only, and bundled through the shared icon asset directory.

## 0.14.0.50 — Project dashboard typography tuning

- Reduced the Project Components category labels from 22 pt to 18 pt for a more balanced dashboard hierarchy.
- Increased the active project name in the Project Manager header to 16 pt so the current project is more visually prominent.
- Moved both sizes into explicit Project Manager theme rules so the global widget font size cannot override them.

## 0.14.0.49 — Project launcher icon set

- Replaced the Project Manager launcher artwork for Model Editor, Map Creator, Exposure Editor, and Waveform Editor with the new bundled 128×128 SVG icon set; Saved Results and Dataset Manager keep their existing disk/grid icons.
- Made application icon buttons expand vertically with their dashboard rows and keep a square footprint, so the SVG artwork grows and shrinks with the Project Manager window instead of remaining at a mostly fixed size.
- Kept the existing Qt standard-icon fallbacks for missing or invalid custom SVG assets and expanded packaging regression coverage for all four bundled vector icons.

## 0.14.0.48 — Model Editor launcher artwork

- Replaced the temporary standard Model Editor launcher glyph with the supplied custom Helmholtz-coil SVG on the Project Manager dashboard.
- Bundled the new SVG as a first-class Field Workbench icon asset with a standard Qt icon fallback if the custom asset cannot be loaded.
- Added the icon asset directory to Linux and Windows PyInstaller data collection so packaged releases use the same launcher artwork as source runs.

## 0.14.0.47 — Project summary line breaks

- Split the Project Manager Exposures summary into explicit exposure and waveform lines so Qt cannot wrap a number away from its descriptor.
- Split the Datasets summary the same way for datasets and calibrations.

## 0.14.0.46 — Saved-result counts and dirty-state emphasis

- Added live result counts directly to the Saved Results tabs: Static maps, Scripted maps, and Snapshots now show their current project counts and update whenever the result catalog refreshes.
- Split Project Manager save status into its own compact status label and added a red highlighted dirty state for **Unsaved changes**, while clean and never-saved states remain neutral.
- Added regression coverage for live Saved Results tab counts and the Project Manager dirty-status styling contract.

## 0.14.0.45 — Project dashboard fit and centering

- Increased the Project Manager minimum height slightly so the complete four-section dashboard fits without triggering a vertical scrollbar at the smallest supported window size.
- Centered each Project Components summary block within the space between its large category label and connector arrow, while preserving wrapping and shrinkage at narrower widths.

## 0.14.0.44 — Project dashboard title tuning

- Reduced the Project Components category headings from 28 pt to 22 pt while keeping the stronger dashboard hierarchy introduced in 0.14.0.43.

## 0.14.0.43 — Project dashboard title scale

- Fixed Project Components typography being visually stuck at the global 10 pt theme size; the global QWidget stylesheet was overriding the per-widget font changes from 0.14.0.42.
- Moved Project Manager dashboard typography into explicit theme selectors so the category titles now render at a deliberately prominent scale.
- Set Model, Maps / Analysis, Exposures, and Datasets to 28 pt bold display text with wrapping and a wider title column, closely matching the supplied 36 pt Gimp mockup at the application UI scale.
- Kept component summaries and application copy at smaller tiers so the category labels remain the dominant visual hierarchy.

## 0.14.0.42 — Project dashboard typography

- Increased the Project Components category labels substantially so Model, Maps / Analysis, Exposures, and Datasets become the dominant visual anchors of the dashboard.
- Increased the left-side project summary text modestly to use the available card space without making the landing page text-dense.
- Gave application names a smaller typography bump while keeping descriptions secondary, preserving a clear component → application hierarchy.
- Added regression checks for the dashboard typography hierarchy.

## 0.14.0.41 — Compact project launcher layout

- Flipped Project Manager application launchers so the large icon button sits on the left and the application name/description sits on the right.
- Reduced dashboard row minimums and launcher padding so all four Project Components bands fit more comfortably at the minimum window size.
- Increased the Project Manager minimum height to 780 px so the complete Model, Maps / Analysis, Exposures, and Datasets dashboard remains visible instead of clipping the final group.
- Tightened responsive launcher icon sizing for the more compact row geometry and updated regression tests for the new ordering and minimum-size contract.

## 0.14.0.40 — Compact project launcher dashboard

- Removed the decorative Project Components glyph panels so non-interactive artwork no longer competes with the real launch controls.
- Reworked each Applications row into a compact text column plus a large icon-only launcher button on the right; launcher icons resize with the available row height and remain accessible by tooltip/name.
- Shortened application descriptions and Project Components summaries, including collapsing Static maps / Scripted maps / Snapshots into a single saved-results count on the landing page.
- Reduced dashboard row minimums and raised the Project Manager minimum-height contract so Model, Maps / Analysis, Exposures, and Datasets remain visible together at the minimum supported window size while still expanding naturally on larger displays.
- Simplified the project-header contents summary into four high-level totals: objects, saved results, exposure items, and data items.

## 0.14.0.39 — Unified dashboard group frames

- Reworked each Project Manager dashboard band into one shared sunken frame spanning Project Components, the connector-arrow gutter, and Applications.
- Removed the independent application-card frames inside Maps / Analysis and Exposures; Map Creator + Saved Results and Exposure Editor + Waveform Editor now live inside one uninterrupted group boundary.
- Made connector arrows part of the same recessed group surface so the middle column no longer reads as a separate pane.
- Extended the **PROJECT COMPONENTS** heading across the connector column and updated dashboard regression tests for the unified grouping.

## 0.14.0.38 — Locked project/application dashboard grid

- Rebuilt the Project Manager component/application dashboard as one shared six-row grid instead of independent left/right layouts. Model and Dataset rows now share their exact height with their single application cards, while Maps / Analysis and Exposures span the same two physical rows occupied by their paired application cards.
- Moved the connector arrows into the shared center grid column, so row alignment is governed by one layout and cannot drift between Project Components and Applications.
- Removed the remaining visible center strip by eliminating separate pane containers and making the connector gutter use the same card surface; the dashboard now reads as one continuous aligned structure.
- Updated dashboard regression tests to assert the exact shared-grid row/column relationships.

## 0.14.0.37 — Project dashboard icon space

- Removed the visible Project-components/Applications splitter bar and rebuilt the dashboard as one shared surface, while keeping the aligned left/right structure and connector arrows.
- Added dedicated large glyph/icon panels to each Project Components card (Model, Maps / Analysis, Exposures, and Datasets) so the dashboard can carry stronger visual identification instead of relying on text alone.
- Tightened the Project Components card layout so summary text stays centered between the section title and the new glyph area; the current release uses clean placeholder standard icons that can be replaced with custom section artwork later.
- Updated Project Manager tests and theme styling for the no-bar dashboard layout and the new glyph panels.

## 0.14.0.36 — Project dashboard relationships

- Renamed the left Project Manager dashboard heading to **Project components** and replaced the transitional architecture note with live project metadata: save status, last-saved time, and a compact contents summary.
- Removed the partial Model Editor / Map Creator quick-launch pair from the top toolbar; application launching now lives consistently in the dashboard and Project menu.
- Added a narrow, subdued connector-arrow gutter from each Project component row to its corresponding application row, making the Model, Maps / Analysis, Exposures, and Datasets ownership relationships visually explicit.
- Updated Project Manager Help and dashboard regression tests for the new relationship-focused layout.

## 0.14.0.35 — Project dashboard polish

- Removed the redundant Model, Maps / Analysis, Exposures, and Datasets category labels from individual application launcher cards; the aligned project summaries already provide that context.
- Expanded each application button/description across the full launcher-card width for a cleaner paired-dashboard layout.
- Gave Project Manager project-summary and application cards a slightly lighter inset surface in both light and dark themes so the four aligned dashboard sections read more clearly against the surrounding workspace.

## 0.14.0.34 — Project dashboard layout

- Reworked the Project Manager landing page into four large project-summary sections aligned directly with their related application launchers: Model, Maps / Analysis, Exposures, and Datasets.
- Maps / Analysis and Exposures now span two launcher rows; Waveform Editor sits with Exposure Editor, while Dataset Manager occupies the final row.
- Replaced the redundant miniature project resource tree with live summary information for scene objects, saved maps and snapshots, exposure protocols and waveforms, measurement datasets, and calibrations.
- Project and Application panes now share one vertically scrolling dashboard surface so their row alignment is preserved at smaller window sizes.
- Updated the Project menu and Help wording to match the dashboard organization.

## 0.14.0.33 — UI smoke-test teardown fix

- Fixed the Project Manager UI smoke-test teardown so synthetic dirty projects are closed without opening an interactive save prompt during unattended test runs.
- Removed duplicate close/delete cleanup callbacks from the project-owned Maps smoke test; Qt resources are now closed once through a shared non-interactive test helper, so genuine Maps failures produce a single traceback instead of cascading cleanup errors.

## 0.14.0.32 — Empty Scripted Maps feedback

- Unified Maps now reports **Scene is empty — nothing to render.** consistently in Scripted 2D, 3D, and Brain modes instead of silently ignoring Create Map when the frozen launch scene contains no objects.
- The background render boundary also applies the empty-scene no-op guard to waveform jobs as a defensive fallback.

## 0.14.0.31 — Scripted Maps initialization fix

- Restored the missing Qt table-item import used by the playback sequencer, fixing the `QTableWidgetItem` `NameError` raised when the unified Maps workspace switches into Scripted mode.
- Restored the missing `QInputDialog` import used by Exposure Editor rename, found during the same undefined-global audit.
- Added a dependency-free static test that scans Field Workbench modules for referenced global names that are neither defined, imported, built in, nor implicit module globals, so import-cleanup regressions can be caught even when Qt-dependent smoke tests cannot run.

## 0.14.0.30 — Source-less Maps rendering

- Made Static Maps tolerate projects that contain scene geometry but no active magnetic-field source. The viewer now uses a viewer-only zero-field fallback while preserving the stricter solver behavior used by analysis, calibration, optimization, probes, and other non-viewer calculations.
- Geometry-only 3D renders suppress the synthetic zero-field layer and keep the selected scene objects/grid visible; 2D renders omit the heatmap/fluxlines and retain available object cross-sections.
- Clicking Static map on a truly empty project now performs a cheap no-op with a status message instead of launching a render job that can only fail.
- Made no-field renders independent of irrelevant manual colour limits and fluxline settings, preventing display-only controls from turning a valid geometry render into a calculation failure.

## 0.14.0.29 — Phase 7 project ownership cleanup

- Made **Project Manager** the sole owner of project lifetime, save state, and multiple open projects. **Model Editor** now edits one project model and no longer carries its retired internal scene-tab system or suite-application launch ownership.
- Routed optimizer finalists and AI-created derivatives into new Project Manager projects instead of Model Editor tabs. Unsaved derivative projects retain a descriptive project-tab label.
- Removed direct raw **Dataset → coil** correction links. Coils now use either a custom factor or a derived **Calibration / Composite Calibration**, preserving measurement provenance behind the calibration layer.
- Simplified Exposure authoring to built-in sine/square/triangle generators or stable references to project waveform resources. Removed the playback **Custom / Wave points** editor and its CSV/DAC/general waveform import paths.
- Project waveform samples are now resolved into a runtime SI trace only when Scripted Maps executes an exposure; exposure protocols no longer materialize those samples into an embedded/custom waveform representation.
- Portable exposure files keep the exposure protocol reference-based and bundle the referenced waveform as a sibling resource; import creates a local waveform resource and rewrites the reference.
- Kept DAC file translation in **Waveform Editor** as an explicit import/export boundary while removing the retired raw DAC-code native waveform quantity and migration paths.
- Updated Help, README, packaging checks, exposure/project tests, and UI smoke expectations to enforce the new suite ownership boundaries.

## 0.14.0.28 — Suite smoke-test cleanup

- Updated the unified Maps / Analysis UI smoke test to expect the current suite-owned window title (`Maps / Analysis — <project>`) instead of the retired snapshot-era title. No runtime behavior changed.

## 0.14.0.27 — Unified Saved Results manager

- Broadened the former Snapshot Manager into a single **Saved Results** manager for all persistent Maps / Analysis products. The window now has **Static maps**, **Scripted maps**, and **Snapshots** tabs.
- Added saved-map management actions for **Open**, **Rename**, and multi-select **Delete** without inflating the cached payloads merely to list or organize them. Static/Scripted map tables also show view type, save time, raw payload size, compressed stored size, and renderer.
- Opening a cached map from Saved Results routes it back into the project-owned Maps / Analysis workspace and reconstructs the stored result without recalculation.
- Project Manager now labels the application **Saved Results**. Activating the Static maps or Scripted maps resource category opens the corresponding Saved Results tab, while double-clicking an individual map still opens that result directly in Maps / Analysis.
- Kept the existing snapshot workflow intact inside the Snapshots tab, including multi-selection comparison, import-from-scene, rename, delete, and independent result viewers.


## 0.14.0.26 — Saved Static and Scripted maps

- Added persistent **Saved map** resources under **Maps / Analysis**. Already-rendered Static and Scripted 2D/3D/Brain maps can now be stored with the project and reopened without repeating the field calculation.
- Added **Save map…** beside **Fullscreen viewer** on rendered map tabs. The saved resource stores the calculated Plotly/WebGL payload plus Brain regional metrics needed to reconstruct the existing result UI.
- Added **Load saved map…** at the Maps result-tab strip and direct project-tree opening of individual saved maps. Reopened maps are ordinary interactive result tabs and do not start a solver job.
- Project Manager now organizes Maps / Analysis into **Static maps**, **Scripted maps**, and **Snapshots**, so precalculated visualizations and quantitative snapshot results remain distinct project resources.
- Saved map payloads are compressed JSON embedded in the current `.magpy.json` project container. Large map data stays out of frozen solver-scene copies so keeping cached results does not inflate subsequent render-job IPC.
- Added safe lazy decoding: project open and Project Manager enumeration do not decompress saved maps; the payload is decoded only when that map is reopened.

## 0.14.0.25 — Maps / Analysis extraction, phase 6

- Promoted **Maps / Analysis** into a project-context-owned suite application. Project Manager now opens the unified Static / Scripted 2D / 3D / Brain workspace directly from the project adapter instead of opening Model Editor as a hidden owner first.
- Routed the suite-managed Model Editor **Maps**, **Static map**, and **Scripted map** entry points back to the project-owned Maps / Analysis window, preventing duplicate viewer ownership paths while retaining standalone compatibility mode.
- Promoted **Snapshot Manager / Saved results** into a project-context-owned suite application. Project-tree saved-result entries can open their stored result directly, and the Model Editor Saved results action routes to the same suite manager.
- Added a **Measurement snapshots** section to Static Maps. It lists Measurement volumes from the frozen Maps launch scene, runs the established volume analysis/result viewer, and saves exact sample coordinates, B vectors, settings, statistics, and provenance into the live project.
- Added frozen-analysis snapshot persistence so a result saved from an older open Maps window keeps the launch-scene fingerprint and field apparatus. If the Model Editor changed afterward, the saved result is correctly reported as stale instead of being misidentified as current.
- Kept Maps launch geometry frozen while applying suite-global solver/theme preferences to live viewer adapters. If Model Editor is already open, its current camera can still seed Maps; Model Editor is no longer required for the launch.
- Reordered the Project Manager launcher cards so **Map Creator** and **Snapshot Manager** are adjacent under Maps / Analysis, matching the project tree where Saved results are the persistent data product of Maps.
- Help now opens directly from Project Manager rather than creating Model Editor solely to host Help.
- Updated Help, README, and About text for the Phase-6 suite ownership boundary.

## 0.14.0.24 — Composite measurement calibrations

- Reworked **Dataset Manager** into three explicit layers: raw **Measurement datasets**, derived **Calibration / Composite fits**, and **Coil correction links**. Raw measurements remain independent provenance and are never merged or overwritten by a derived calibration.
- Added project-level composite calibration resources that select whole datasets or individual measured samples from any number of raw datasets and fit one shared scale-only correction against a chosen coil group.
- Added **equal-point** and **equal-source** weighting. Equal-point weighting treats every selected measurement equally; equal-source weighting gives each selected dataset portion equal total influence so a dense scan does not automatically dominate a smaller source.
- Composite fits retain source selections, target coils, weighting, fit residual/R² statistics, dataset update versions, and a model fingerprint. Editing source data or relevant model state marks the saved fit **Needs recalculation** instead of silently changing its result.
- Coil correction links now use fitted Calibration / Composite fits as the normal path. A calibration can be shared by all coils in its fitted target group; the UI only offers a calibration to coils that were actually part of that fit. Existing direct Dataset links remain readable as **Dataset (legacy)** for compatibility.
- Prevented deletion of a raw dataset while a composite calibration references it. Deleting a linked calibration can preserve each affected coil's current resolved factor as a Custom value.
- Added composite-calibration resources to the Project Manager Datasets tree and direct project-tree opening of individual calibrations.
- Updated Model Editor's coil correction controls, Help, and project documentation for the derived-calibration workflow.

## 0.14.0.23 — Dataset Manager extraction, phase 5

- Promoted **Dataset Manager** into an independent project-aware suite application backed directly by the active project `ProjectContext`; Model Editor no longer needs to be open to create, edit, fit, link, or delete measurement datasets.
- Reused the established dataset manager and Create/Edit dataset controls inside the suite window, preserving measurement entry/import, scale fitting, coil correction links, and existing `.magpy.json` storage without changing dataset representation.
- Project-tree dataset resources now open directly in Dataset Manager, while activating the Datasets category opens the manager without forcing an editor.
- Routed the suite-managed Model Editor **Measurements** action to the same project Dataset Manager instead of creating a second Model-Editor-owned dataset window. Standalone/compatibility Model Editor launches retain the legacy local manager path.
- Synchronized Dataset Manager changes back to an already-open Model Editor so fitted/link-derived correction factors and project dirty state refresh while ownership remains with the suite project context.
- Registered Dataset Manager as a project-scoped suite window so several projects can keep independent dataset windows open alongside mixed Model, Exposure, and Waveform editors.
- Updated Help and suite documentation for the Phase-5 ownership boundary.

## 0.14.0.22 — Native waveform editor cleanup

- Removed WaveBuilder-style 8-bit/16-bit authoring modes, the dual WaveBuilder/Native workspaces, point-code document state, bit-depth preferences, and the old DAC-oriented generator UI. Waveform Editor now edits the Field Workbench native time/value/channel model directly at all times.
- Reimplemented square, sine, action-potential, and free-form construction as floating-point native generators. Generated segments inherit the selected channel's quantity/unit and can replace or append at an explicit physical sample interval.
- Reworked legacy DAC import into an explicit translation boundary: users define sample interval, quantity/unit, and the engineering values corresponding to code 0 and code 255 before samples enter the editor.
- Reworked DAC and Rigol export so integer/vendor encodings are created only in the exported copy. DAC export explicitly maps native values to unsigned 8-bit codes; Rigol export maps native values to the generator's ±5 V range and uses its 16-bit transport encoding internally.
- Removed obsolete WaveBuilder authoring dialogs, bit-depth preference state, DAC-oriented plot/document code, and random-sequence authoring modules from the Waveform Editor runtime. Retained small integer-code helpers only inside legacy DAC/Rigol import/export codecs.
- Existing Phase-3 <code>dac_code</code> waveform resources remain readable; the editor treats their numeric samples as ordinary native/code-valued data without guessing a physical calibration, and the legacy quantity/editor metadata is retired on save.
- Updated the waveform format builder so new profiles describe engineering-value output rather than a DAC-code editor quantity; old DAC-code profiles are migrated to arbitrary/code-valued output when edited.

## 0.14.0.21 — Exposure Editor extraction, phase 4

- Promoted **Exposure Editor** into an independent project-aware suite application. Opening Exposures from Project Manager or the suite-managed Model Editor now routes to the same project Exposure Editor instead of creating a Model-Editor-owned library window.
- Project-tree exposure resources now open directly in Exposure Editor, and mixed application windows from multiple project tabs retain their owning `ProjectContext`.
- Added stable project-waveform references to saved exposure protocols. New arbitrary/imported drive signals select a waveform resource and channel rather than embedding a duplicate sample array in each exposure; generated sine/square/triangle waveforms remain self-contained.
- Added runtime resolution of project waveform references for Scripted Maps playback, including physical current/voltage unit conversion and validation. Raw DAC-code or other non-physical channels are not silently treated as current or voltage.
- Preserved legacy custom-point exposures. Existing inline custom waveforms remain editable and can be promoted explicitly into a reusable project waveform with **Save embedded points as project waveform…**.
- Kept portable `.fwbexposure.json` files self-contained: exporting a project-referenced exposure resolves the selected waveform channel into inline points only in the exported copy. Scene-to-scene exposure import also copies referenced project waveform resources when available and rewrites their local IDs.
- Prevented deletion of project waveform resources that are still referenced by saved exposures, avoiding dangling references during normal editing.
- Updated Help and suite documentation for the Phase 4 ownership boundary: Waveform Editor owns arbitrary waveform construction/import/translation; Exposure Editor owns drive assignment, waveform selection, and sequencing.

## 0.14.0.20 — Native waveform editing, phase 3 completion

- Added a second **Native channels** workspace to Waveform Editor for direct editing of native/imported Field Workbench waveform resources without converting them through WaveBuilder's 8/16-bit DAC representation.
- Native editing preserves each channel's physical time axis, floating-point samples, quantity, unit, source metadata, and independent timeline. Multi-channel resources can add, remove, rename, and edit channels directly.
- Replaced the native sample grid with a model-backed table so large imported waveforms do not require creating one Qt widget item per cell; the graph down-samples only its drawing path while leaving stored samples untouched.
- Added native time/value paste, row insertion/removal, channel creation/removal, quantity/unit editing, and native `.fwbwave` export from the active Waveform Editor workspace.
- Loading an imported/native project waveform now opens it directly in the Native channels workspace. WaveBuilder-authored DAC resources still reopen losslessly in WaveBuilder and are also inspectable/editable as native channels.
- Saving native edits intentionally retires stale WaveBuilder authoring metadata; merely inspecting and re-saving an unchanged WaveBuilder resource preserves that metadata.
- WaveBuilder-only generators, DAC import/export, Rigol export, and 8/16-bit authoring controls are disabled while the Native channels workspace is active so the two editing models are not silently mixed.
- Phase 3 now has the intended project-level waveform ownership: waveform creation/import/translation and native editing live in Waveform Editor. Existing exposure-embedded custom points remain compatible until Exposure Editor is migrated to waveform-resource references in Phase 4.

## 0.14.0.19 — Waveform Editor integration, phase 3 begins

- Promoted WaveBuilder 2.0.0.16 into a native Field Workbench **Waveform Editor** suite window instead of launching or embedding a separate application. The authoring/generator, legacy DAC, random-sequence, free-form, and Rigol export engines are integrated while project/window ownership stays with Field Workbench.
- Kept Field Workbench's global Light/Dark theme authoritative. The integrated editor has no private theme preference and its hand-painted waveform/free-form canvases read the Workbench plot palette directly.
- Reimplemented WaveBuilder's main graph with a lightweight Qt painter canvas so the Waveform Editor does not carry its own Matplotlib/QtAgg plotting path and follows Field Workbench's plot palette directly.
- Added first-class standalone waveform resources to project metadata. Native waveform resources preserve sampled times, channels, quantities, units, and metadata and appear under **Waveforms** in the Project Manager tree.
- Added project-waveform create/load/save/rename/duplicate/delete controls. WaveBuilder-authored single-channel DAC-code resources retain editor metadata (8/16-bit depth and point delay) so they can be reopened for editing.
- Moved general waveform-format import into the Waveform Editor as a project-level ingestion path using the existing native reader/profile system; imported physical/native channels are stored without sample/unit conversion. Legacy exposure-embedded custom waveforms remain readable and continue to be edited through Exposure Editor during the migration.
- Added native `.fwbwave`/`.fwbwave.json` export from Waveform Editor while retaining legacy `.dac` and Rigol RAF/CSV export.

## 0.14.0.18 — Suite project context, phase 2

- Added a project-scoped `ProjectContext` so project identity, native adapter, path/save baseline, and live suite-window associations no longer conceptually belong to the Model Editor.
- Added a central resource registry with stable namespaced identities for model objects, snapshots, exposures, datasets, and transitional embedded custom waveforms, without changing `.magpy.json` serialization.
- Rebuilt the Project Manager tree from the shared registry and exposed resource keys in tooltips for traceable cross-application references.
- Passed the shared project context into suite-managed Model Editors and tagged Maps / Analysis, Exposure Editor, Dataset Manager, and Snapshot Manager windows with their owning project and application role, laying the foundation for extracting them as independent suite applications in later phases.
- Kept all existing editors and file formats behaviorally compatible; Waveform Editor remains the next extraction phase.

## 0.14.0.17 — Preferences UI smoke-test deadlock fix

- Fixed the UI smoke suite hanging in `test_preferences_theme_choice_applies_and_persists_wavebuilder_dark_palette` after global Preferences moved into the shared suite helper.
- Preserved injectable dialog construction for the standalone Model Editor compatibility path, allowing the existing non-blocking Preferences test double to intercept the dialog instead of entering a real modal `exec()` loop in Qt offscreen mode.

## 0.14.0.16 — Project tree and section-header polish

- Changed the Project Manager's **PROJECT** and **APPLICATIONS** title bars to use a raised bevel while retaining their dark header background.
- Simplified **Maps / Analysis** in the project tree by removing the non-persistent Static maps and Scripted playback placeholder entries.
- Moved saved **Snapshots** under **Maps / Analysis**, reflecting that snapshots are the persistent data product created from static map analysis.
- Kept **Exposures** and **Datasets** as independent top-level project resource categories rather than nesting them under Maps / Analysis.

## 0.14.0.15 — Suite preferences and viewer-local fluxline cleanup

- Moved application-wide Preferences into the Project Manager (`Edit → Preferences…`) so theme, coil field model, solver temporary-memory budget, worker concurrency, and solver diagnostics are controlled once for all open projects and live suite windows.
- Removed the suite-managed Model Editor's duplicate Preferences entry; standalone compatibility launches still expose the same global dialog.
- Moved the minimum-field and conductor-distance fluxline cleanup controls out of global Preferences and into 2D/3D Maps as a viewer-local **Fluxline options** section. The section is shown only when the selected display includes fluxlines.
- Existing Maps windows receive changed global solver/resource settings without altering their viewer-local fluxline cleanup choices.

## 0.14.0.14 — Project workspace border and save icon contrast

- Added a continuous outer border around the Project Manager workspace below the menu/toolbar, including the status-bar edge, so the suite work area reads as one bounded region.
- Replaced the low-contrast platform save icon in the Project Manager toolbar with a theme-aware accent-colour save glyph for reliable visibility in both Dark and Light modes.

## 0.14.0.13 — Project Manager recessed section layout

- Split the Project Manager chrome into separate recessed visual sections for the project-tab line, project information band, and each application launcher row.
- Grouped each application category label with its button and description inside one sunken launcher frame.
- Added dark title bars for the PROJECT and APPLICATIONS headings while keeping the surrounding manager workspace on the normal menu/toolbar surface.

## 0.14.0.12 — Project tab chrome polish

- Moved the new-project `+` control to the left side of the project tabs so project creation stays beside the tab list.
- Reframed the complete project-tab workspace with the same chrome surface used by the menu/toolbar and a subtle sunken bevel, replacing the darker document-style backing surface.

## 0.14.0.11 — Blank projects and resilient suite launcher layout

- New and startup projects now begin with an empty model instead of the former packaged demonstration scene.
  Example projects remain available under <code>examples</code> for users who want a populated model to explore.
- Removed the user-facing default/demo-scene creation actions from Project Manager and Model Editor workflows.
- Renamed the former packaged starter scene as an internal optimizer benchmark fixture so it is no longer part of
  the project-creation model.
- Made the Project Manager application launcher pane vertically scrollable and gave application cards/descriptions
  protected minimum heights so descriptions do not collapse into their buttons at smaller window sizes.

## 0.14.0.10 — Suite project manager, phase 1

- Added a KiCad-style Project Manager as the application launch window, with a project/resource
  tree on the left and suite application launchers on the right.
- Added closable, movable project tabs so several independent projects can stay open in one
  manager session. Existing Maps, Exposure, Dataset, Snapshot, and Model Editor windows can be
  opened concurrently from different project tabs.
- Kept `.magpy.json` as the complete project container for this first migration phase; no scene,
  exposure, dataset, snapshot, or waveform serialization was changed.
- Project trees show scene objects, saved exposures, measurement datasets, embedded custom
  waveforms, and saved snapshots. Waveform Editor is reserved for the later WaveBuilder phase.
- In suite-managed Model Editors, New/Open hand projects back to the Project Manager and the old
  scene-tab strip is hidden, avoiding a second competing project-tab system. Closing only a Model
  Editor no longer closes its sibling analysis/editor windows or the project itself.

## 0.14.0.9 — Scripted playback layout polish

- Moved the playback timeline onto its own full-width row so Play/Reset/Loop no
  longer force the timeline beyond the actual viewer width beside the regions panel.
- Made the playback control grid shrink to the WebGL viewer's real available width.
- Made Scripted source sizing follow only the selected source page, so Saved exposure
  no longer reserves the taller Built-in waveform form's empty space.

## 0.14.0.8 — Scripted viewer trace controls

- Moved the complete Traces section out of Exposure authoring and into Maps → Scripted,
  below Camera, where it applies to the current viewer regardless of scripted source.
- Exposure definitions no longer store excitation, sequencer, observer, sample/component,
  or Brain-region trace selections; legacy trace blocks are ignored when normalized.
- Loading a saved exposure or quick built-in waveform now leaves the viewer's trace
  choices untouched.

## 0.14.0.7 — Technical-note help audit

- Documented waveform preview/channel selection, time and unit validation, and
  the distinction between format profiles, waveform data, and saved exposures.
- Expanded saved-result freshness/status explanations and signed comparison conventions.
- Clarified mesh volume/centroid versus sampling and bounded regional statistics.
- Retained the consolidated legacy changelog without a technical-notes appendix.
- Help/documentation update only; calculation and UI behavior are unchanged.

## 0.14.0.6 — Consolidated change history

- Combined `WAVEFORM_IO.md` and `PLAYBACK_REGION_CHANGES.md` into this file.
- Consolidated the recorded version history into one changelog.
- Added a README link and taught the overlay-update cleaner to remove the two
  retired documentation files.
- Documentation and source-maintenance update only; application behavior is unchanged.

## 0.14.0.5 — Release review and cleanup

- Removed obsolete hidden dataset selection, deletion, and coil-link controls from
  the measurement editor, together with their unused handlers.
- Removed unused imports and the superseded region-table wheel filter.
- Updated Help Topics for datasets, exposure authoring/importing, and scrolling.
- Corrected the Windows build-size documentation and added release smoke checks.

## 0.14.0.0–0.14.0.4 — Measurement workflow and editor polish

- Measurements manages scene-saved datasets and coil correction links.
- Create/Edit dataset owns measurement entry, paste/import, and calibration fitting.
- Enlarged the measurement editor and provided adjustable section heights.
- Protected measurement/exposure editors and child dialogs against accidental
  value changes from mouse-wheel scrolling.

## 0.13.0.387 — Pre-release cleanup

- Removed the retired Results launcher/dialog and its scene-switch callback path; Sensor plot, Volume analysis, and Saved results are now direct top-level tools.
- Renamed the internal toolbar key from the abandoned `gallery` wording to `saved_results`.
- Removed the now-unused hidden shared-controls placeholder from Maps; scripted-source controls are owned by the active scripted page and advanced protocol editing lives in Exposures.
- Removed stale waveform-I/O references to automatic/content-based detection; importers remain explicitly selected.
- Updated internal workspace comments to the current unified Maps terminology while retaining Static/Scripted compatibility entry points.

## 0.13.0.385–0.13.0.386 — Scripted-source simplification

- Maps scripted mode now offers quick built-in Sine/Square/Triangle sources or a named Saved exposure instead of embedding the advanced authoring controls.
- Advanced Drive/Waveform/import/Sequencer/Trace editing lives in Exposures; Maps selects and renders the saved protocol.
- Missing saved-exposure coil references stop playback instead of being silently discarded.
- Result tabs use `<view>\<Static|waveform|exposure> - <n>` names with independent numbering per view/source.

## 0.13.0.384 — Exposure library

- Added scene-owned named Exposures with stable coil references, portable `.fwbexposure.json` import/export, duplication/rename/delete, and editor support for drive, waveform, sequencer, and trace state.
- Exposure definitions are stored with the model and travel with `.magpy.json` scene files.

## 0.13.0.381–0.13.0.383 — Unified Maps workspace

- Static and Scripted modes now live in one Maps window above the shared 2D/3D/Brain selector.
- All calculated results share one tab strip; changing mode/view returns to Selector while prior result tabs remain available.
- Result-tab ownership restores the correct mode/view when an older result is selected.

## 0.13.0.379 — Saved-results wording

- Replaced the short-lived Gallery wording with Saved results throughout the user interface and Help while retaining the on-disk `snapshot_*` schema for compatibility.

## 0.13.0.378 — Test/wording cleanup

- Updated stale UI smoke expectations for the renamed Static map and Scripted map workspaces.
- Relaxed the Scene Recipe capability check to the current singular “frozen launch-time scene copy” wording and aligned the guide with Static map / Scripted map terminology.

## 0.13.0.376 — Playback sidebar polish

- Removed the explanatory Playback protocol heading/note from the unified sidebar.
- Drive, Waveform, Coil sequencer, and Traces now sit below Display Options and above Camera.
- The legacy WaveformPlaybackDialog shell is explicitly hidden when used as the shared protocol controller, preventing it from appearing over the preview.

## 0.13.0.375 — Shared playback protocol

- Playback no longer opens a transitional waveform-setup dialog from each 2D/3D/Brain view.
- One persistent Playback protocol block now lives in the unified left pane and carries Drive, Waveform/custom points/import, Coil sequencer, and Traces across view switches.
- The active view contributes only its map/camera settings when Create playback is pressed; the protocol itself is shared.

## 0.13.0.374 — Shared Static / Playback sidebar

- The unified field workspace now owns one persistent left/right splitter instead of embedding a complete splitter for every 2D, 3D, and Brain page.
- `2D | 3D | Brain` is a single shared selector at the top of the left pane.
- View-specific setup/action controls are re-parented into one stacked control area, so the left-pane width and selector no longer jump when views are changed.
- A hidden fixed shared-controls host remains available for future workspace-wide controls; playback protocol controls now use the active view's existing scroll so the left pane avoids nested scroll areas.
- Existing view implementations, calculation paths, result tabs, and renderer cleanup remain intact underneath the consolidated shell.

## 0.13.0.373 — Unified Static / Playback field workspace

- Removed the intermediate 2D / 3D / Brain chooser window from the new task-first Static and Playback toolbar flow.
- Static and Playback now open one cached workspace directly, with a synchronized **2D | 3D | Brain** selector pinned above the left-hand setup pane.
- Existing 2D, 3D and Brain implementations are embedded as pages in one top-level window for Phase 1b, preserving their proven calculation/render paths while eliminating extra top-level viewer windows.
- All embedded pages are made from one launch-time detached scene snapshot; later-created views cannot silently pick up source-scene edits made after the workspace opened.
- Brain remains visible but disabled when the launch snapshot has no unambiguous SRI24 brain.
- Each view is created lazily and cached, so switching away and back preserves that view's setup/results.
- The common selector insertion hooks are exposed on the view control columns for the next consolidation phases.

## 0.13.0.370 — Waveform format builder

- Reworked the waveform-I/O foundation around **explicit importer selection**; automatic probing/detection is deliberately deferred.
- Added a persistent **Import waveform…** chooser that lists coded built-in readers and user-created format profiles in one place.
- Added the **Waveform format builder** for declarative text or binary layouts: numeric type/sign/width, byte order, record/field separators, headers/offsets/stride, timing source, point-count source, output units, and linear scale/offset conversion.
- Added sample-file decode preview plus portable `.fwbformat.json` profile save/import/export/duplicate/edit/delete support.
- Added a general fixed-record parser used by the builder and importer, while keeping the old CSV/DAC shortcuts temporarily available until they are recreated as profiles in the next phase.
- Kept the normalized `WaveformDocument` model and native FWB JSON interchange format, but registered readers are now selected explicitly rather than probed.

## 0.13.0.368 — NumPy/help regression cleanup

- Regional B-time and B²-time integration now prefers `numpy.trapezoid`, with a fallback to `numpy.trapz` only on older NumPy versions. This restores compatibility with newer NumPy releases where `np.trapz` is no longer available.
- The frequency-weighted Help topic now consistently uses the published **weighted-peak** terminology, matching the UI regression expectation.

## 0.13.0.367 — Startup inspector state

- Dynamic scene-inspector editors now opt out of application UI-value persistence.
- A prior object's Name (and other document-backed inspector values) can no longer overwrite the selected object's values when the main window first restores its UI state.
- The unittest release checks now explicitly guard against the obsolete pytest imports and stale video-export bridge assertion.

## 0.13.0.366 — FFT plot controls

The FFT page now places a blue **Recalculate FFT** action directly below the
analysis settings. Recalculation explicitly re-homes both plot axes to the
current frequency limit and the newly calculated spectrum. Display-only changes
(Plot max frequency, Show DC and Log Y) also request a fresh 2D axis fit
immediately, so an older manual pan/zoom cannot hide a newly selected range.
The FFT page also exposes **Fullscreen viewer**, using the same independent
true-fullscreen PlotView renderer as the other FWB plot views. The current 2D
zoom/range is carried into fullscreen and Esc/F11 returns to the analysis page.

## 0.11.4.39 — User-defined measurement points

- Added exact local XYZ measurement points that follow object transforms and work with statistics, snapshots, comparisons, and optimizer targets.
- Distinguished user-defined samples from generated voxel sampling in result windows.

## 0.11.4.38 — Windows splash startup

- Added the Windows bootloader splash and kept it visible until the main window was shown.
- Restored and activated the main window after the splash closed to address foreground-focus problems.

## 0.11.4.37 — Windows release dependencies

- Switched Windows builds to console-free windowed mode and explicitly collected Magpylib display backends and Plotly dependencies.

## 0.11.4.36 — Light-theme regression compatibility

- Restored the stylesheet rule expected by UI tests while retaining the complete application light palette.

## 0.11.4.35 — Consistent light theme

- Applied a complete light palette and shared styling across the main application and independent windows, avoiding inherited dark desktop colours.

## 0.11.4.34 — Centred single 3D slice

- Fixed single-slice 3D maps to place the slice at the selected map centre instead of the volume boundary.

## 0.11.4.33 — Return axes in 3D maps

- Added a Return axes control using the Workbench fitted engineering camera and axis ranges.

## 0.11.4.32 — Multiple measurement-volume analyses

- Added sequential analysis of multiple selected Measurement volumes, with independent results and cancellable progress.
- Improved invalid-selection guidance and batch completion/error reporting.

## 0.11.4.31 — Sensor-plot styling

- Applied the shared analysis-window styling to Sensor Path Plot and highlighted Calculate as its primary action.

## 0.11.4.30 — Shared analysis-window styling

- Centralized application styling so detached measurement, snapshot, comparison, and manager windows retained consistent controls and result formatting.
- Highlighted snapshot-saving and comparison actions.

## 0.11.4.29 — Live snapshot manager

- Reused one Snapshot Manager window and added manual and periodic catalogue refresh.
- Selected newly saved snapshots while preserving existing selections during background refreshes.

## 0.11.4.28 — Independent windows and sensor progress

- Made snapshot, measurement, comparison, and sensor windows ordinary independent windows without forced stacking above the main application.
- Deferred sensor calculations until Calculate and added cancellable progress, solver details, and elapsed-time reporting.

## 0.11.4.27 — Volume progress and non-modal results

- Added shared determinate progress and cancellation to measurement-volume calculations across solver representations.
- Made measurement, snapshot, and comparison results non-modal, with independent lifetimes and reliable renderer/overlay cleanup.

## 0.11.4.26 — Racetrack labels

- Corrected racetrack coils appearing as Square coils in the scene object list.

## 0.11.4.25 — Error-aware automatic coil modeling

- Added sparse Exact-turn reference checks to choose the least expensive coil representation meeting empirical error tolerances.
- Cached validation by geometry and observer region and reported measured errors, selection reasons, and unmet tolerances.
- Retained the cheaper first-stage policy for optimizer exploration.

## 0.11.4.24 — Compact winding-pack classification

- Corrected Auto compactness checks so square winding cross-sections did not unnecessarily cross the Centreline threshold.

## 0.11.4.23 — Automatic coil modeling

- Added Auto as the default field model, choosing Exact, Bundled, Current sheet, or Centreline per coil and observer set.
- Used construction geometry, observer proximity, workload, and memory budget to select a representation and record fallback decisions.

## 0.11.4.22 — Built-in Help Topics

- Added searchable, categorized help covering calculations, measurements, comparisons, DRC, optimization, and limitations, with F1 access.
- Replaced the lengthy About history with a compact version and credits display.

## 0.11.4.21 — Cancellable field-map progress

- Centred map progress dialogs and made Close/Escape request cancellation at the next safe solver-batch boundary.
- Restored controls and reported elapsed time after cancellation.

## 0.11.4.20 — Configure maps before calculating

- Opened 2D and 3D map windows without an automatic calculation, allowing settings to be chosen before Calculate.

## 0.11.4.19 — Map timing and solver status

- Added elapsed time to map progress and persistent status bars showing the actual solver representation, fallback, and total time.
- Clarified solver preferences and diagnostics wording.

## 0.11.4.18 — Bounded solver memory and diagnostics

- Added a configurable temporary-memory budget and batched both observers and detailed sources, accounting for polyline segment counts.
- Preallocated result arrays and exposed memory/workload details in progress reports.
- Added flushed JSONL diagnostics for investigating interrupted calculations.

## 0.11.4.17 — Determinate map progress

- Added completed/total samples, percentages, and solver-stage reporting for batched 2D and 3D calculations, with separate rendering and grid phases.

## 0.11.4.16 — Initial map progress display

- Added immediate activity indicators and disabled Calculate during map computation and rendering.

## 0.11.4.15 — Default sensor pixels

- Treated an omitted sensor pixel definition as one local-origin sample, fixing point/path sensor evaluation through the consolidated backend.

## 0.11.4.14 — Consolidated field solver

- Routed maps, probes, sensors, measurements, snapshots, exports, and generated coil copies through one backend honoring the global coil-model preference.
- Added consistent Exact turns, Bundled, Current sheet, and Centreline representations with recorded per-coil fallbacks.

## 0.11.4.13 — Consistent physical assemblies

- Unified winding, barrel, and flange geometry for circular, racetrack, and closed planar coils so displayed assemblies matched DRC envelopes.
- Reported unsupported open/non-planar coil assemblies instead of generating misleading bobbins.

## 0.11.4.12 — Racetrack inspector cleanup

- Moved parametric racetrack dimensions beneath position/rotation and hid redundant generated-vertex controls.
- Connected geometry changes to the normal Apply action and live winding-length estimates.

## 0.11.3.5 — HM starter scene and point-sensor scale

- Packaged the HM two-coil example with physical construction, measurement sphere, point sensor, and saved snapshot.
- Fixed single-point plot scaling and displayed the exact reading while retaining one physical CSV sample.

## 0.11.3.4 — Field-map colours and point sensors

- Applied a full blue-to-red colour range to map data, with one shared range across 3D slices.
- Fixed isolated point-sensor plots to report real field values without requiring an axis sensor.

## 0.11.3.3 — Layered 3D field maps

- Added configurable cubic field sampling with layered X/Y/Z surfaces, shared colours, centre-outward slice visibility, and optional scene geometry.
- Reordered analysis tools around probes, sensors, and maps and standardized Point probe naming.

## 0.11.3.2 — Import, copying, export, and recent files

- Added optional bounding-box centering for STL/OBJ imports and selectable/copyable measurement, snapshot, and probe results.
- Added complete sensor-trace CSV export and persistent Open Recent history with missing-file handling.

## 0.11.3.1 — Mesh containment verdict

- Restored the explicit Inside/outside DRC enabled inspector verdict for containment-reliable meshes.

## 0.11.3 — Shared geometry capabilities

- Unified geometry eligibility across the inspector, DRC, measurements, snapshots, and optimizer targets/placement surfaces.
- Extended the measurement and optimization workflow to eligible primitives and healthy closed meshes with deterministic local-volume sampling.
- Retained surface uses for damaged meshes, marked invalidated snapshots stale, and migrated the unused Test geometry role to Visual.

## 0.11.2.1 — Optimizer scrolling and turn counts

- Routed mouse-wheel gestures to Search setup scrolling instead of changing editor values.
- Added actual per-coil turn counts to finalist tables and comparisons.

## 0.11.2 — Physically constrained optimizer

- Added background optimization of one to four circular coils around a spherical target, with signed-current fitting and optional anatomy-surface placement.
- Searched geometry and winding choices under hard electrical, construction, coverage, mass, and clearance limits.
- Added coarse/final validation, Pareto filtering, reproducible seeds, Pause/Cancel, finalist visualization/comparison, and undoable candidate application.

## 0.11.1 — Imported-mesh design checks

- Extended DRC to STL/OBJ exclusions with mesh-health diagnostics, transformed surface checks, and containment for reliable closed meshes.
- Distinguished surface-only clearance from certified containment and exposed a separate numerical refinement tolerance.

## 0.11.0.1 — Compact scene legend

- Moved the 3D legend below the scene to recover viewport width and allow horizontal wrapping.

## 0.11.0 — Physical assemblies and design-rule checks

- Rendered electrically inert circular-coil winding packs, barrels, and flanges from saved construction dimensions.
- Added physical clearance/collision checks against primitive exclusions and other coils, with saved global/per-zone rules, clickable results, and stale-result tracking.
- Used transformed assembly geometry for close checks rather than treating overlapping world bounding boxes as collisions.

## 0.10.16 — Turn counts in estimates

- Added winding turn counts to the estimates panel and copied estimate reports.

## 0.10.15 — Automatic winding radial build

- Derived radial build from turns, finished-wire size, packing factor, and axial width, updating assembly dimensions live.
- Retained manual overrides and preserved existing hand-entered builds during migration.

## 0.10.14 — Mesh scaling and assembly dimensions

- Added independent local X/Y/Z scaling and original/scaled dimensions for imported meshes, preserving legacy uniform scales.
- Added optional circular-coil winding/bobbin envelope metadata, derived dimensions, and construction warnings without changing field or electrical results.

## 0.10.13 — Selection after mesh import

- Used CSS-safe stable Plotly trace identities to prevent selection failures with imported filenames containing punctuation or Unicode.

## 0.10.12 — Reliable 3D scene replacement

- Rendered structural changes into a fresh surface and swapped it in only after success, preventing blank scenes and out-of-order updates.
- Retained the last valid scene if replacement rendering failed.

## 0.10.11 — Imported-mesh visibility

- Rebuilt the WebGL scene after structural changes to prevent stale geometry, colours, opacity, and visibility.
- Kept faster selection-only updates and preserved the fitted camera.

## 0.10.10 — Stable trace styling

- Added persistent trace identities so imported and primitive meshes could not inherit another object's face colours after scene changes.

## 0.10.9 — Inspector and complete-scene fitting

- Widened the inspector and included imported meshes in camera fitting and equal-scale scene bounds.
- Added convenient coil Apply actions and protected inspector values from accidental wheel changes.

## 0.10.8 — Layered calculation verification

- Expanded tests for independent physics, direct Magpylib parity, and saved reference scenes, including vectors, transforms, units, current conventions, and cancellation.
- Verified pinned calculation-engine versions and documented optional isolated upstream testing.

## 0.10.7 — Mean-diameter clarity

- Clarified circular-coil size as mean diameter, displayed mean radius, and added selectable/copyable estimates.

## 0.10.6 — Empirical field correction

- Added saved per-coil correction factors that scale magnetic output without changing physical drive, heating, resistance, or mass estimates.
- Added analytical field/scaling checks and improved GUI-test WebEngine cleanup.

## 0.10.5 — Correct 3D display units

- Corrected normalization of Magpylib display coordinates so rendered coil sizes matched millimetre axes and selection outlines.

## 0.10.4 — Wire resistance and current convention

- Added editable resistance per kilometre initialized from AWG, allowing measured/manufacturer resistance in electrical estimates.
- Defaulted new coils to RMS or DC current while retaining saved peak-sine settings.

## 0.10.3 — Reliable scene-tree dragging

- Prevented unchanged or cancelled drags from deleting tree rows by keeping scene-model hierarchy changes authoritative.

## 0.10.2 — Camera transition cleanup

- Applied the final fitted or preserved camera before painting, removing the intermediate startup/selection camera jump.

## 0.10 — Camera consistency and snapshot import

- Unified startup, scene-load, and Home camera fitting while preserving the view during selection and overlay changes.
- Added undoable import of frozen snapshots from other scenes, retaining source identity without importing geometry or settings.

## 0.9 — Snapshot comparisons

- Added actual-strength and mean-normalized shape comparisons for matching local sample grids, including full-vector and angular statistics.
- Added local-frame previews, signed-difference slices/clouds, overlaid distributions, and full comparison results.
- Aligned the starter camera with Home and restored New default scene under File.

## 0.8 — Saved field snapshots

- Added scene-embedded snapshots with exact sample coordinates, field vectors, settings, statistics, and timestamps.
- Added management, freshness/status tracking, and undoable snapshot changes while preserving stale results without recalculation.
- Improved fitted engineering camera presets and retained compatibility with older scenes.

## 0.7 — Measurement-volume analysis

- Added saved measurement sampling, target, and tolerance settings with configurable sampling quality and transform-aware field evaluation.
- Added field-strength, uniformity, coverage, direction, and full statistical results, plus distribution plots and sampled-field overlays.
- Preserved measurement settings through cloning, undo/redo, and scene save/reopen.
