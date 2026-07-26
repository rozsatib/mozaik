# Eccentricity-dependent LGN and retinocortical mode

## Document status

This document is the implementation anchor for an optional
eccentricity-dependent visual topography in Mozaik.

The design has been reviewed at the architectural and scientific-assumption
level. Implementation has not started. The work is split by
[Two-phase implementation boundary](#two-phase-implementation-boundary), with
the smaller stages listed in [Implementation sequence](#implementation-sequence).
This document prevents later stages or Phase 2 from silently changing
decisions made in Phase 1.

The new mode must coexist with the existing uniform Cartesian mode. Legacy
configurations and legacy numerical results are a hard compatibility boundary.

## How to use this document

This document is normative. Words such as "must", "require", and "do not"
define implementation requirements. Names declared under
[Required production API](#required-production-api) are final for this
implementation unless an accepted design change updates this document and the
affected implementation within the same maintainer-reviewed change or commit
series.

An implementation session must not rely on the previous design conversation
or on files under `/home/tibor/cuni/dev/log_polar_retina`. All required
formulas, constants, coordinate conventions, and decisions are reproduced
here.

When implementation reveals that a requirement is incompatible with the
current code:

1. stop that implementation stage;
2. record the exact incompatibility in this document;
3. do not substitute an approximation or a different scientific convention;
4. obtain a design decision before continuing.

The intentionally failing full-neuron spatial-frequency test is the only
planned test failure. All other tests introduced by this work must pass.

## Design changes during implementation

Implementation may reveal that an approved requirement is feasible but no
longer preferred. Such preference changes are permitted, but they must not be
implemented as undocumented deviations from this document.

### Unconstrained implementation details

An implementation detail may change without a design amendment only when the
change does not alter any requirement stated by this document, including:

- public or required internal APIs;
- configuration schemas or validation behavior;
- numerical or scientific behavior;
- coordinate conventions or units;
- random-number consumption or seeded draw order;
- MPI reproducibility;
- legacy compatibility;
- phase or stage ownership;
- performance or memory requirements explicitly stated here;
- tests, regression anchors, or acceptance criteria.

Examples include private local-variable names, equivalent internal control
flow, and private helper extraction that leaves all specified behavior
unchanged.

When it is unclear whether a detail is constrained, treat it as constrained
until the impact has been reviewed.

### Changes to approved requirements

A change to an approved requirement must follow this procedure:

1. Stop implementation of the affected requirement before implementing the
   alternative.
2. Record the proposed change under
   [Implementation design change log](#implementation-design-change-log).
3. Identify every affected normative section, production API, configuration
   key, test, regression anchor, stage, and later-stage dependency.
4. Decide whether to accept, reject, or defer the proposal.
5. If accepted, update all affected normative sections so that the main body
   describes one coherent current design.
6. Update the relevant tests and acceptance criteria.
7. Rerun every earlier stage or phase gate whose contract or observable
   behavior may have changed.
8. Resume dependent implementation only after the document no longer contains
   unresolved contradictions.

The implementation design change log records history and rationale. It does
not override the normative body of this document. When a logged decision and
the normative body disagree, implementation remains blocked until the
normative text is reconciled.

An unresolved proposal may remain open without blocking unrelated work only
when its affected stages and dependencies are explicitly identified and none
of them includes the current implementation work.

### Impact analysis

Every accepted or open design-change entry must explicitly check the following
categories:

- scientific assumptions and formulas;
- API and configuration contracts;
- coordinate systems, units, and domains;
- cap semantics;
- random streams and deterministic draw order;
- MPI and cross-rank reproducibility;
- legacy numerical compatibility;
- Phase 1 contracts consumed by Phase 2;
- test expectations and regression anchors;
- memory and runtime behavior;
- documentation elsewhere in this file.

Writing only that downstream conflicts "should be checked" is insufficient.
The entry must list the specific sections, stages, and tests examined,
including an explicit statement when no impact was found.

## Implementation design change log

This log is append-only. Superseded requirements must still be removed or
rewritten in the normative sections; retaining history here is not a reason to
leave contradictory requirements in the main document.

Use the following template for each proposed change:

```text
Change ID:
Status: Proposed | Accepted | Rejected | Deferred
Date:
Discovered during stage:
Requested by:
Original requirement:
Proposed change:
Reason:
Alternatives considered:
Affected normative sections:
Affected production APIs or configuration:
Affected stages:
Affected tests and regression anchors:
Phase 1 / Phase 2 compatibility impact:
Randomness or MPI impact:
Scientific or numerical impact:
Required acceptance-gate reruns:
Normative sections reconciled: Yes | No | Not applicable
Implementation may continue: Yes | No
Resolution:
```

A proposal is accepted only after explicit maintainer approval. Codex may
identify a discrepancy, propose alternatives, and edit this document when
instructed, but it must not infer approval from implementation convenience.

```text
Change ID: ECC-P1-S0-001
Status: Accepted
Date: 2026-07-26
Discovered during stage: Phase 1 Stage 0
Requested by: Human maintainer
Original requirement: Record exact fixed-seed legacy LGN behavior for the
single-process and two-rank acceptance matrix.
Proposed change: Document that the separately recorded one-rank and two-rank
post-presentation contrast-tail hashes reflect a known bug in the PyNN version
used by the current Mozaik. If a later PyNN version fixes the bug, retain both
anchors until the human maintainer explicitly identifies which legacy
recording is valid.
Reason: A backend fix may make the rank-count-specific hashes converge between
implementation stages, but that does not itself determine which historical
recording is the required compatibility target.
Alternatives considered: Silently choose the result produced by the newer
backend; accept either hash; remove or normalize the contrast-tail anchor.
All were rejected because they would relax or redefine the legacy boundary
without maintainer approval.
Affected normative sections: Acceptance criteria and tolerances / Gabor,
connectivity, and legacy behavior; Implementation sequence / Stage 0.
Affected production APIs or configuration: None.
Affected stages: Stage 0 and every later Phase 1 or Phase 2 stage that reruns
the legacy LGN acceptance anchors.
Affected tests and regression anchors:
tests/models/vision/test_legacy_lgn_regression.py one-rank and two-rank
post-presentation contrast-response SHA-256 fixtures.
Phase 1 / Phase 2 compatibility impact: Both phases must retain the separately
recorded anchors unless the maintainer explicitly selects the valid legacy
recording after a PyNN fix.
Randomness or MPI impact: The known difference is rank-count-specific. Seeded
positions, kernels, luminance responses, and injected-current traces remain
separately covered and are not relaxed by this clarification.
Scientific or numerical impact: None; this records an existing backend bug and
does not change LGN formulas, currents, or scientific behavior.
Required acceptance-gate reruns: After a PyNN change affecting the bug, rerun
the Stage 0 one-rank and two-rank legacy LGN tests. After the maintainer selects
the valid recording, rerun every already completed later-stage gate that
checks legacy LGN behavior.
Normative sections reconciled: Yes
Implementation may continue: Yes
Resolution: Accepted by explicit maintainer instruction. A future agent must
stop for an explicit user decision if the PyNN fix changes these anchors and
must not relax, replace, normalize, or delete them independently.
```

## Version-control ownership

The human maintainer owns all version-control operations.

Codex must not:

- create, amend, squash, or delete commits;
- create, switch, reset, rebase, or merge branches;
- modify tags;
- push, force-push, fetch, pull, or otherwise change a remote repository;
- discard or overwrite unrelated working-tree changes.

Codex may modify requested working-tree files, run tests, inspect diffs, and
recommend logical commit boundaries. At each stage stopping point it must
report:

- files modified;
- tests and commands run;
- exact test outcomes;
- unresolved design issues;
- deviations proposed or accepted;
- earlier acceptance gates that must be rerun;
- a suggested division of the changes into human-created commits.

A reference in this document to the "same change" or "same implementation
chunk" means the same maintainer-reviewed working-tree change or commit series.
It does not authorize Codex to create or publish commits.

## Two-phase implementation boundary

Implementation is delivered in two independently reviewable phases. The
complete design remains the final contract, but a fresh implementation session
must be able to stop after Phase 1 without constructing or modifying cortex.

### Phase 1: LGN only

Phase 1 owns:

- legacy LGN regression anchors;
- the visual-domain, density, RF-size, cap, and DoG analytical functions;
- `RadiallySymmetricLGNTopography` properties and methods needed by LGN;
- fixed-count ON/OFF position sampling and `ExplicitPositions`;
- `RetinalInhomogeneousDiskSheet`;
- per-cell RF scaling and support;
- derived stimulus resolution and the common input-resolution property;
- the eccentricity-only luminance correction;
- LGN current injection, spiking, MPI reproducibility, performance reporting,
  and analytical-versus-simulated LGN preferred-SF characterization.

The Phase 1 integration model ends at the ON/OFF LGN sheets. It does not
construct a cortical sheet, LGN-to-cortex connector, or recurrent connector.
Eccentric LGN must not be paired temporarily with
`VisualCorticalUniformSheet` or legacy `GaborConnector`; such a mixed model
remains an error. Phase 1 therefore changes no cortical positions,
connectivity, annotations, maps, stimulators, selectors, visualization, or LFP
analysis.

Phase 1 may keep the final `topography` configuration subsection, including
mapping calibration parameters, so parameter files do not need a second
migration. It validates the unified explicit-cap policy, including the
empirical lower bound, but it need not expose or call cortical coordinate
transforms yet. Those methods are added to the same provider in Phase 2; do
not introduce a temporary provider class or duplicate cap implementation.

Phase 1 is complete when all LGN-scoped ordinary tests pass in legacy and
eccentric modes, including one- and two-rank reproducibility. Its
analytical-versus-spiking SF characterization then writes its plot and fails
with the separately documented `TODO Finish` assertion. No cortical test is
part of the Phase 1 gate.

### Phase 2: cortical correspondence

Phase 2 owns:

- the provider's forward and inverse retinotopic coordinate methods;
- `RetinotopicCorticalSheet` and its corner-origin position generation;
- explicit physical-coordinate interfaces and all affected generic callers;
- `EccentricityDependentGaborConnector`;
- position-dependent layer-4 carrier frequency, jitter rejection, cortical
  map interpretation, and replacement annotations;
- LGN-to-cortex and resulting recurrent-connectivity behavior;
- Gabor-profile plots, cortical compatibility errors, and the complete
  LGN-to-layer-4 integration test.

Phase 2 consumes the Phase 1 LGN API without changing its position
distribution, RF construction, resolution, luminance behavior, seeded draw
order, or passing tests. If Phase 2 reveals that an LGN contract must change,
update this document and rerun the entire Phase 1 acceptance gate before
continuing.

References to the "complete mode" below mean Phase 1 plus Phase 2. Requirements
about cortex do not become Phase 1 deliverables merely because the shared
mathematics or final API is described earlier in the document.

## Repository and dependency assumptions

The implementation uses existing repository dependencies:

- NumPy for arrays and seeded `RandomState` streams;
- SciPy for `brentq`, numerical integration in tests, and nearest-neighbour map
  interpolation;
- PyNN for populations, structures, cells, and connectors;
- NEST through the existing Mozaik simulator integration;
- Matplotlib for validation plots;
- mpi4py through Mozaik's existing MPI setup when available.

Do not add a new runtime dependency for this work.

The implementation assumes model constructors follow the existing visual-model
pattern:

1. call `Model.__init__`;
2. create `model.visual_field`;
3. construct and assign `model.input_layer`;
4. construct cortical sheets;
5. construct feedforward connectors;
6. construct recurrent connectors.

Eccentricity mode must validate this order. A retinotopic cortical sheet
constructed before `model.input_layer` or an eccentricity input constructed
before `model.visual_field` raises a clear configuration-order error.

## Authoritative configuration sources

`model.visual_field` is the sole source of visual-domain centre, width, and
height in eccentricity mode:

- centre: `model.visual_field.location_x/location_y`;
- width: `model.visual_field.size_x`;
- height: `model.visual_field.size_y`.

The eccentricity LGN component does not accept a second `size` parameter.
This avoids the legacy situation in which `model.visual_field.size` and
`retina_lgn.params.size` can differ. Legacy mode retains its existing separate
`size` parameter and behavior.

The eccentricity input component constructs the single
`RadiallySymmetricLGNTopography` instance and exposes it as
`model.input_layer.topography`. In Phase 1, its ON/OFF sheets store the same
object reference. Phase 2 additionally gives that reference to every
`RetinotopicCorticalSheet`; the eccentricity-specific Gabor connector then
verifies object identity across both LGN source sheets and the cortical
target. Do not duplicate topography parameters in the cortical-sheet or
connector configuration.

Before constructing the provider, the eccentricity input component must
validate that `model.visual_field` exists and supplies finite
`location_x`, `location_y`, `size_x`, and `size_y` values. The centre is
accepted only when:

```text
abs(location_x) <= 1e-12 degrees
abs(location_y) <= 1e-12 degrees
```

This tolerance admits harmless parsed floating-point zero; it is not support
for an off-centre field. Both sizes must be positive and finite. Errors must
name the invalid field and explain that eccentricity mode currently requires a
fixation-centred visual rectangle.

The shared provider is a runtime object, not serialized inside a
`ParameterSet`. Its constructor contract is:

```text
RadiallySymmetricLGNTopography(visual_field, parameters)
```

where `parameters` is the input component's `topography` subsection. The
provider copies validated scalar configuration; it does not retain mutable
configuration aliases. The input, both LGN sheets, cortical sheets, and
connectors nevertheless share the one provider object so independently
resolved cap or map constants cannot diverge.

## Goals

The eccentricity-dependent mode must provide:

1. A fixed-size ON LGN population and a fixed-size OFF LGN population whose
   receptive-field centres follow an eccentricity-dependent density.
2. Per-cell LGN receptive-field sizes determined by eccentricity.
3. A nonlinear visual-field-to-cortical correspondence for LGN-to-cortex
   connectivity.
4. Layer-4 Gabor carrier frequencies determined from the theoretical local LGN
   preferred spatial frequency.
5. A user-selected eccentricity cap that, when present, is applied
   consistently to cortical magnification, LGN density, and LGN RF size.
6. A correction of the known luminance-response scaling bug in the new mode
   only.

LGN positions remain ordinary Cartesian visual-field coordinates in degrees.
The LGN population is not represented in log-polar coordinates.

## Non-goals

The first implementation must not add:

- homotypic exclusion;
- a regular anatomical retinal mosaic;
- Poisson-disc sampling;
- Lloyd relaxation;
- density-dependent fan-in normalization;
- incoming-weight normalization;
- image pyramids;
- scale-space convolution;
- kernel scale bins;
- log-polar stimulus resampling;
- a biologically derived cortical spatial-frequency map;
- changes to temporal LGN RF parameters;
- changes to LGN gain parameters;
- changes to spiking-neuron parameters;
- automatic recalibration when theoretical and measured spatial-frequency
  preferences differ.

The generated LGN positions form an inhomogeneous random set of functional
receptive-field centres. They are not a complete anatomical retinal mosaic.

## Approved decisions

The following decisions are requirements, not open implementation choices:

- The visual field remains centred at the global origin `(0, 0)`.
- Off-centre visual-field subsections are not supported in this phase.
- The maximum modelled eccentricity is half the smaller visual-field edge.
- Neuron RF centres outside that eccentricity are not modelled.
- Rectangular stimulus pixels outside the maximum eccentricity remain
  available; no circular stimulus mask is applied.
- Only configurations with `E_max >= 90 degrees` are rejected. Unused
  rectangle corners may lie beyond 90 degrees.
- The cortical sheet has a corner origin in eccentricity mode.
- One cortical direction represents eccentricity and the other represents
  polar angle.
- The cortical `u=0` boundary is excluded from generated neuron positions.
- The angular seam is intentionally discontinuous.
- A smaller user-specified cortical extent crops the represented visual field;
  it does not rescale the retinotopic mapping.
- Full per-cell 3D LGN kernels are used initially.
- Eccentricity mode is incompatible with `original_2024_lgn_mode`.
- Eccentricity mode initially supports only the Cai97 DoG RF with
  `subtract_mean=False`.
- The new Gabor RF annotations replace the old annotation values, including
  for recurrent layer-4 correlation-based connectivity.
- Gabor preferred frequency is evaluated at the unjittered mapped position.
- RF-centre jitter is resampled until it remains inside the LGN centre domain.
- Jitter sampling fails with a clear error after 10,000 unsuccessful attempts.
- Orientation and phase map values are defined in cortical coordinates.
- The known luminance scaling bug is corrected only in eccentricity mode.
- The first full-neuron spatial-frequency validation intentionally ends in a
  failing test after writing a plot and a `TODO Finish` marker.

## Current Mozaik implementation

This section records the legacy behavior that the implementer must understand
and preserve.

### LGN component and population construction

The current input component is
`mozaik.models.vision.spatiotemporalfilter.SpatioTemporalFilterRetinaLGN`.

Its relevant parameters are:

- `density`: documented and used as neurons per square visual degree;
- `size`: visual-field width and height in degrees;
- `receptive_field.func`;
- `receptive_field.func_params`;
- `receptive_field.width`;
- `receptive_field.height`;
- `receptive_field.spatial_resolution`;
- `receptive_field.temporal_resolution`;
- `receptive_field.duration`;
- `original_2024_lgn_mode`;
- cell, noise, gain-control, recorder, and recording parameters.

For each of `X_ON` and `X_OFF`, the component creates a
`mozaik.sheets.vision.RetinalUniformSheet`. That sheet creates

```text
int(sx * sy * density)
```

neurons using a PyNN `RandomStructure` inside a `Cuboid` centred at `(0, 0)`.
ON and OFF positions are independent sequential draws from
`mozaik.pynn_rng`.

The legacy `density` parameter is therefore not a population count and must
not be reused with a different meaning.

### Random numbers and MPI

Mozaik initializes `mozaik.pynn_rng` and `mozaik.rng` through its MPI setup.
`mozaik.get_seeds` is used when independent NumPy random states are required.
All ranks must request the same number of seeds in the same order.

The current sheets force evaluation of `Population.positions` during
construction so PyNN position generation is reproducible under supported MPI
execution.

LGN noise seeds are generated for global cell identifiers and instantiated
only for locally owned cells. The eccentricity implementation must retain this
global-seed/local-construction pattern.

### Receptive fields

The legacy component constructs one shared
`SpatioTemporalReceptiveField` for ON cells and one negated shared RF for OFF
cells.

The standard RF function is `mozaik.models.vision.cai97.stRF_2d`:

```text
centre_spatial(x, y) * centre_temporal(t)
- surround_spatial(x, y) * delayed_surround_temporal(t)
```

The spatial Gaussian convention in `cai97.F_2d` is:

$$
F(x,y;A,\sigma)
=
A\exp\left(-\frac{x^2+y^2}{2\sigma^2}\right).
$$

Thus `sigma_c` and `sigma_s` are conventional Gaussian standard deviations in
visual degrees. The configured `Ac` and `As` are peak amplitudes.

`SpatioTemporalReceptiveField.quantize`:

1. computes `nx = ceil(width/dx)`, `ny = ceil(height/dy)`, and
   `nt = ceil(duration/dt)`;
2. expands the realised width, height, and duration to those sample counts;
3. samples at spatial pixel centres and temporal bin starts;
4. divides the resulting kernel by `nx * ny * nt`.

The standard reference values observed during design inspection were:

- `Ac = 1`;
- `As = 0.032473`;
- `sigma_c = 0.103253 degrees`;
- `sigma_s = 0.461558 degrees`;
- `width = height = 3 degrees`;
- `spatial_resolution = 0.05 degrees/pixel`;
- `duration = 200 ms`;
- `temporal_resolution = 7 ms`;
- `subtract_mean = False`.

These values are model configuration, not constants to hard-code.

### Per-cell stimulus extraction

`CellWithReceptiveField` creates a `VisualRegion` centred on the cell's
Cartesian RF position. Its size equals that cell's RF width and height.

For every frame it calls `VisualSpace.view` with the RF spatial resolution.
Existing `VisualSpace` semantics fill requested regions outside the stimulus
object with background. The RF is not clipped at the edge of the LGN sheet.

The new mode must preserve those boundary semantics. In particular, an LGN
centre near `E_max` may have RF support extending beyond `E_max`, and it may
sample rectangular stimulus pixels whose eccentricity exceeds `E_max`.

### Luminance, contrast, current, and spikes

The current cell code separates the quantized kernel into:

$$
\bar K(t)=\operatorname{mean}_{x,y}K(x,y,t)
$$

and

$$
K_{\mathrm{contrast}}(x,y,t)=K(x,y,t)-\bar K(t).
$$

The contrast image is divided by background luminance and spatially multiplied
and summed against `K_contrast`. The luminance path multiplies `bar K(t)` by
mean image luminance.

The separated responses pass through the configured contrast and luminance
gain functions and become injected currents. Currents are delivered through
step/noise current sources or through the integrated current-source neuron
path. LGN spiking is performed by the configured PyNN/NEST cell model.

The eccentricity work must not alter temporal parameters, delay, gain
parameters, noise, or neuron parameters.

### Stimulus resolution

`mozaik.experiments.vision.VisualExperiment` currently derives stimulus pixel
density from:

```text
1 / model.input_layer.parameters.receptive_field.spatial_resolution
```

This direct parameter access assumes one fixed RF resolution. It must become a
common input-component property while preserving the legacy return value.

### Cortical sheets and magnification

`mozaik.sheets.vision.VisualCorticalUniformSheet` accepts:

- `sx`, `sy` in micrometres;
- `density` in neurons per square millimetre;
- `magnification_factor` in micrometres per degree.

It stores neuron positions in visual-degree-equivalent coordinates by dividing
physical cortical dimensions by `magnification_factor`. Its random structure
is centred at `(0, 0)`.

`SheetWithMagnificationFactor` supplies:

- `vf_2_cs`;
- `cs_2_vf`;
- `dvf_2_dcs`;
- `size_in_degrees`.

These methods assume a single linear magnification factor. That assumption is
invalid in eccentricity mode. Existing distance connectors, direct
stimulators, and population selectors call these methods and must be routed
through explicit coordinate interfaces for the new sheet.

All legacy models relevant to this project use
`magnification_factor = 1000 um/degree`. At that value, numerical positions in
stored visual-degree equivalents equal positions in cortical millimetres,
which masked the fact that orientation maps are cortical-space maps. Leave the
legacy orientation-map path unchanged. The eccentricity connector gets the
scientifically correct cortical-space map path independently.

### LGN-to-cortex and recurrent connectivity

`mozaik.connectors.meta_connectors.GaborConnector` draws or loads:

- orientation;
- phase;
- aspect ratio;
- envelope size;
- carrier frequency;
- RF-centre jitter.

It writes:

- `LGNAfferentOrientation`;
- `LGNAfferentPhase`;
- `LGNAfferentAspectRatio`;
- `LGNAfferentSize`;
- `LGNAfferentFrequency`;
- `LGNAfferentX`;
- `LGNAfferentY`.

`mozaik.connectors.vision.GaborArborization` evaluates the Gabor over
Cartesian LGN positions. Position and envelope units are visual degrees and
frequency is cycles per visual degree.

The modular sampling connector samples the configured number of contacts from
the candidate probability distribution. Sampling may select a source more
than once, which becomes greater aggregate weight. The new mode must not
change this behavior or normalize fan-in.

`mozaik.connectors.vision.V1CorrelationBasedConnectivity` reuses the same
Gabor annotations for recurrent layer-4 connectivity. Replacing mapped
positions and frequencies will therefore intentionally change recurrent
connectivity in eccentricity mode.

## Affected repository files

The smallest expected production change set is:

| Phase | File | Required change |
|---|---|---|
| 1 | `mozaik/models/vision/topography.py` | Add the analytical provider's density, RF-size, cap, and DoG-supporting LGN contract. Phase 2 extends this same module with coordinate transforms. |
| 1 | `mozaik/models/vision/cai97.py` | Add the convention-specific analytical DoG optimum used for LGN validation and later cortical frequency. |
| 1 | `mozaik/models/vision/spatiotemporalfilter.py` | Add the separate eccentricity input/cell path, per-cell RF construction, dynamic resolution, and new-mode luminance correction; add the common resolution property to legacy input. |
| 1, then 2 | `mozaik/sheets/vision.py` | Phase 1 adds `ExplicitPositions` and the retinal sheet. Phase 2 adds the retinotopic cortical sheet and explicit physical-coordinate methods without changing Phase 1 behavior. |
| 1 | `mozaik/experiments/vision.py` | Read `input_layer.visual_space_resolution_deg`. |
| 1 | `devtools/dummy_model.py` | Expose the common resolution property for experiment tests and utilities. |
| 2 | `mozaik/connectors/meta_connectors.py` | Add the independent eccentricity Gabor connector. |
| 2 | `mozaik/connectors/modular_connector_functions.py` | Route cortical sheet-coordinate distances through the explicit physical-distance interface. |
| 2 | `mozaik/connectors/fast.py` | Use the same physical-distance interface instead of a concrete legacy-sheet type check. |
| 2 | `mozaik/connectors/vision.py` | Make cortical map lookup coordinate-aware and reject co-circular geometry on the retinotopic sheet until its scientific meaning is specified. |
| 2 | `mozaik/sheets/direct_stimulator.py` | Convert sheet positions to physical cortical micrometres through the explicit interface. |
| 2 | `mozaik/sheets/population_selector.py` | Convert physical grid coordinates and sheet positions through the explicit interface. |
| 2 | `mozaik/visualization/misc.py` | Plot retinotopic sheets through explicit cortical or visual coordinate conversion instead of scalar magnification. |
| 2 | `mozaik/analysis/lfp.py` | Raise a clear unsupported-analysis error for the scalar-magnification propagation analysis on retinotopic sheets. |

Add focused tests under:

```text
tests/models/vision/test_topography.py
tests/models/vision/test_eccentricity_spatiotemporalfilter.py
tests/models/vision/test_eccentricity_lgn_spatial_frequency.py
tests/sheets/test_retinotopic_vision.py
tests/connectors/test_eccentricity_gabor.py
tests/integration/test_eccentricity_mode.py
```

The first three files are Phase 1 test locations. The remaining sheet,
connector, and full integration files are Phase 2. If an integration test
module contains both scopes, name or group tests so the Phase 1 LGN-only subset
can be selected without deselecting any LGN assertions.

Existing nearby test modules may be extended instead where that better matches
repository organization. Do not store new regression anchors in
`tests/full_model/reference_data`; use small literal arrays, deterministic
hashes documented with their generating configuration, or assertions against
existing test behavior.

Mozaik does not have one generic visual-model builder: consuming model
constructors directly instantiate their sheet and Gabor classes. Therefore the
library change supplies the new components but cannot silently switch an
arbitrary model. Each consuming model that opts in must explicitly instantiate
`RetinotopicCorticalSheet` and
`EccentricityDependentGaborConnector` after constructing the eccentricity
input. The repository integration test supplies a minimal such model. No
legacy example or model constructor is migrated merely by adding these
classes.

## Incorporated topography mathematics

The production implementation must not import
`/home/tibor/cuni/dev/log_polar_retina/retina_lut.py`. The functions are
computational formulas, not lookup tables.

Add the production module:

```text
mozaik/models/vision/topography.py
```

Use these module-level analytical helper names:

- `relative_lgn_cell_density_at_eccentricity`;
- `relative_lgn_cell_density`;
- `capped_relative_lgn_cell_density_at_eccentricity`;
- `capped_relative_lgn_cell_density`;
- `rf_center_sigma`;
- `retinotopic_map_parameters`;
- `_visual_polar_to_cortical_centered_mm`;
- `_cortical_centered_mm_to_visual_polar`.

The four density helpers and `rf_center_sigma` are Phase 1.
`retinotopic_map_parameters` and the two coordinate helpers are Phase 2.
Phase 1 may use a narrowly scoped empirical-cap calculation needed to validate
the unified cap, but must not expose a second temporary mapping API.

The two underscore-prefixed helpers implement the analytical `(r, theta)`
to `(u, v)` formulas. The provider owns the public Cartesian `(x, y)` to
corner-origin `(u, a)` wrappers, so callers cannot accidentally confuse `v`
with stored sheet coordinate `a`.

Place the convention-specific DoG optimum calculation in `cai97.py` as
`dog_optimal_spatial_frequency`.

### LGN density

The fitted radial LGN density is:

$$
\rho(E)
=
200.54919\exp(-E/2.0738426)
+
55.730403\exp(-E/16.692502).
$$

Eccentricity is in visual degrees and density is nominally cells per square
degree. Because the population count is fixed, only the relative density
shape matters.

For Cartesian coordinates:

$$
E=\sqrt{x_{\mathrm{vf}}^2+y_{\mathrm{vf}}^2}
$$

and the sampling weight is:

$$
w(x_{\mathrm{vf}},y_{\mathrm{vf}})=\rho(E).
$$

This is an area density with respect to `dx dy`. It contains neither the polar
Jacobian `E` nor normalization for a particular domain.

For an explicit cap:

$$
\rho_{\mathrm{cap}}(E)=\rho(\max(E,E_{\mathrm{cap}})).
$$

The density is finite and strictly positive at fixation. The fit is
mathematically finite throughout the supported range. It is empirically poorly
constrained above approximately 60 degrees, so the model should log one
extrapolation warning rather than clip values.

### RF centre size

The fitted centre sigma is:

$$
\sigma_c(E)
=
\frac{10^{0.014E-0.758}}{\sqrt{2}}
\quad\text{degrees}.
$$

This is a conventional Gaussian standard deviation matching `cai97.F_2d`.
It is nonzero at fixation:

$$
\sigma_c(0)\mathrel{\mathop{\simeq}}0.123448\ \text{degrees}.
$$

For an explicit cap:

$$
\sigma_{c,\mathrm{cap}}(E)
=
\sigma_c(\max(E,E_{\mathrm{cap}})).
$$

The underlying regression was reported below 25 degrees. If `E_max > 25
degrees`, log one initialization warning that larger-eccentricity RF sizes are
extrapolated and may not be accurate. Extrapolation is approved.

### DoG optimum

For the conventional Gaussian definition used by Cai97, the nonzero
stationary spatial frequency is:

$$
f_{\mathrm{opt}}^2
=
\frac{
\log(A_s)-\log(A_c)+4[\log(\sigma_s)-\log(\sigma_c)]
}{
2\pi^2(\sigma_s^2-\sigma_c^2)
}.
$$

The result is cycles per spatial unit. With sigmas in degrees, it is cycles per
degree.

Validate positive finite amplitudes and sigmas, unequal sigmas, and a positive
real `f_opt^2`. The reference Mozaik parameters produce approximately:

```text
0.800897 cycles/degree
```

which agrees with the historical configured value `0.8`.

### Retinotopic mapping

The mapping functions operate on:

- visual eccentricity `r` in degrees;
- visual polar angle `theta` in radians;
- cortical coordinates `u`, `v` in millimetres.

The fitted full map uses:

- full bilateral cortical area `760 mm^2`;
- central areal magnification `3.6 mm^2/degree^2`;
- full calibration eccentricity `90 degrees`;
- peripheral exponent `beta = 1.59`.

The mapping derives:

- empirical cap eccentricity;
- peripheral coefficient;
- cap magnification;
- simulated cortical area;
- equal maximum cortical axis extent `L_max`;
- angular scale.

Let:

$$
A_{\mathrm{full}}=760\ \mathrm{mm}^2,
\qquad
M_0=3.6\ \mathrm{mm}^2/\mathrm{degree}^2,
$$

$$
\Phi=2\pi,
\qquad
R_{\mathrm{full}}=90\ \mathrm{degrees},
\qquad
\beta=1.59.
$$

The empirical cap is obtained by solving for `q` in `(0, 1]`:

$$
\frac{q^2}{2}
+
\frac{q^\beta}{2-\beta}
\left(1-q^{2-\beta}\right)
=
\frac{A_{\mathrm{full}}}
{\Phi M_0 R_{\mathrm{full}}^2}.
$$

Then:

$$
r_{\mathrm{cap,emp}}=qR_{\mathrm{full}},
$$

which is approximately `1.8210964699 degrees` for the constants above, and:

$$
C=M_0r_{\mathrm{cap,emp}}^\beta.
$$

For mapping purposes, let `r_cap` be the empirical cap when the configured cap
is `None`, and the validated user cap otherwise. The capped central areal
magnification is:

$$
M_{\mathrm{cap}}=Cr_{\mathrm{cap}}^{-\beta}.
$$

For simulated maximum eccentricity `R = E_max`, cortical area is:

$$
A_{\mathrm{sim}}
=
\Phi
\left[
\frac{M_{\mathrm{cap}}r_{\mathrm{cap}}^2}{2}
+
\frac{C}{2-\beta}
\left(
R^{2-\beta}-r_{\mathrm{cap}}^{2-\beta}
\right)
\right]
$$

when `r_cap < R`, and:

$$
A_{\mathrm{sim}}
=
\Phi\frac{M_{\mathrm{cap}}R^2}{2}
$$

when `r_cap >= R`.

Define:

$$
L_{\max}=\sqrt{A_{\mathrm{sim}}},
\qquad
\alpha=\frac{L_{\max}}{\Phi},
$$

where `alpha` is the cortical angular scale in millimetres per radian. The
cortical location of the cap is:

$$
u_{\mathrm{cap}}
=
\frac{M_{\mathrm{cap}}r_{\mathrm{cap}}^2}{2\alpha}.
$$

The forward radial map is:

$$
u(r)
=
\frac{M_{\mathrm{cap}}r^2}{2\alpha}
$$

for `r <= r_cap`, and:

$$
u(r)
=
u_{\mathrm{cap}}
+
\frac{C}{\alpha(2-\beta)}
\left(
r^{2-\beta}-r_{\mathrm{cap}}^{2-\beta}
\right)
$$

for `r > r_cap`.

The angular map is:

$$
v(\theta)=\alpha\theta.
$$

The inverse radial map is:

$$
r(u)
=
\sqrt{\frac{2\alpha u}{M_{\mathrm{cap}}}
}
$$

for `u <= u_cap`, and:

$$
r(u)
=
\left[
r_{\mathrm{cap}}^{2-\beta}
+
\frac{\alpha(2-\beta)}{C}
(u-u_{\mathrm{cap}})
\right]^{1/(2-\beta)}
$$

for `u > u_cap`.

The inverse angle is:

$$
\theta(v)=v/\alpha,
$$

wrapped to `[-pi, pi)`.

`u` increases with eccentricity. `v` represents unwrapped polar angle over:

$$
-L_{\max}/2\leq v<L_{\max}/2.
$$

The cortical sheet uses a corner-origin angular coordinate:

$$
a=v+L_{\max}/2,
$$

so:

$$
0\leq a<L_{\max}.
$$

Cartesian wrappers use:

$$
r=\operatorname{hypot}(x,y),\qquad
\theta=\operatorname{atan2}(y,x)
$$

and:

$$
x=r\cos(\theta),\qquad
y=r\sin(\theta).
$$

All scalar and array inputs to the production provider must be finite.
Eccentricities must be nonnegative and visual positions must satisfy
`hypot(x, y) <= E_max` when validation is requested. Mapping inputs must be
inside their resolved full-map domain: `0 <= r <= E_max`,
`0 <= u <= L_max`, and an angular coordinate in the half-open interval
represented by the map. Values outside those domains raise `ValueError`;
they are not clipped. Scalar methods return Python `float`; broadcast array
inputs return NumPy arrays of the broadcast shape.

Validate `0 < beta < 2`, positive finite calibration constants, and positive
finite `full_max_eccentricity`. This implementation fixes the calibration
constants shown above and exposes only `full_max_eccentricity` and `beta` in
configuration. A configured explicit cap must be finite, must be at least the
derived empirical cap, and must not exceed `E_max`. `E_max >= 90 degrees` is
the only visual-rectangle size rejection based on the intended mathematical
range.

## Required production API

The following modules and public class names are required:

```text
mozaik/models/vision/topography.py
    RadiallySymmetricLGNTopography

mozaik/models/vision/spatiotemporalfilter.py
    EccentricityDependentCellWithReceptiveField
    EccentricityDependentSpatioTemporalFilterRetinaLGN

mozaik/sheets/vision.py
    ExplicitPositions
    RetinalInhomogeneousDiskSheet
    RetinotopicCorticalSheet

mozaik/connectors/meta_connectors.py
    EccentricityDependentGaborConnector

mozaik/models/vision/cai97.py
    dog_optimal_spatial_frequency
```

Phase 1 delivers `RadiallySymmetricLGNTopography`,
`dog_optimal_spatial_frequency`, both eccentricity LGN classes,
`ExplicitPositions`, and `RetinalInhomogeneousDiskSheet`. Phase 2 delivers
`RetinotopicCorticalSheet` and
`EccentricityDependentGaborConnector`, then extends the provider and cortical
sheet APIs with the coordinate methods listed below. The final public names
are fixed from Phase 1 onward; Phase 2 must extend rather than replace the
provider or retinal components.

`ExplicitPositions` is a PyNN `space.BaseStructure` implementation, even
though it lives in `mozaik.sheets.vision` for this narrowly scoped change.
Its constructor accepts one finite NumPy-compatible array with shape `(3, N)`.
The first two rows are sheet coordinates and the third row is zero.
`generate_positions(n)` requires `n == N` and returns a copy, so PyNN cannot
mutate the provider's canonical position array.

The production inheritance and constructor contracts are:

```text
EccentricityDependentSpatioTemporalFilterRetinaLGN(SensoryInputComponent)
    __init__(model, parameters)

RetinalInhomogeneousDiskSheet(Sheet)
    __init__(model, parameters, positions_deg, topography)

RetinotopicCorticalSheet(Sheet)
    __init__(model, parameters)

EccentricityDependentGaborConnector(BaseComponent)
    __init__(network, lgn_on, lgn_off, target, parameters, name)
```

The new input component must not subclass
`SpatioTemporalFilterRetinaLGN`: inherited `required_parameters` would force
legacy `density`, `size`, and fixed spatial resolution because Mozaik merges
parameter schemas over the complete MRO. Likewise,
`RetinotopicCorticalSheet` must not subclass
`SheetWithMagnificationFactor`, and the new Gabor connector must not subclass
legacy `GaborConnector`. Share behavior through narrow helper functions or a
mixin with no `required_parameters`; do not broaden the global parameter
validator.

The required topography provider interface is:

```text
topography.max_eccentricity_deg
topography.user_cap_eccentricity_deg
topography.mapping_cap_eccentricity_deg
topography.mapping_axis_extent_mm
topography.relative_density_at_eccentricity(eccentricity_deg)
topography.relative_density_xy(x_deg, y_deg)
topography.center_sigma_deg(eccentricity_deg)
topography.visual_to_cortical_mm(x_deg, y_deg)
topography.cortical_to_visual_deg(u_mm, a_mm)
topography.validate_visual_position(x_deg, y_deg)
```

The Phase 1 provider subset is:

```text
max_eccentricity_deg
user_cap_eccentricity_deg
relative_density_at_eccentricity
relative_density_xy
center_sigma_deg
validate_visual_position
```

Phase 2 adds `mapping_cap_eccentricity_deg`,
`mapping_axis_extent_mm`, `visual_to_cortical_mm`, and
`cortical_to_visual_deg`. The constructor still validates and stores all final
configuration keys in Phase 1, including those used only when Phase 2 resolves
the map.

The provider is immutable after construction. Array-returning methods accept
NumPy-broadcast-compatible inputs and return a Python float for scalar inputs
and an ndarray for array inputs.

`user_cap_eccentricity_deg` is the configured value, including `None`.
`mapping_cap_eccentricity_deg` is always the resolved finite cap used by the
cortical map.

The provider's public Cartesian mapping methods use corner-origin cortical
coordinates: `visual_to_cortical_mm` returns `(u_mm, a_mm)`, and
`cortical_to_visual_deg` accepts `(u_mm, a_mm)`. The centred angular coordinate
`v = a - L_max/2` is an internal analytical coordinate only. At Cartesian
fixation `(0, 0)`, the forward wrapper uses the canonical `atan2(0, 0) == 0`
and therefore returns `(u=0, a=L_max/2)`; callers must not infer a unique
cortical angle at fixation.

The required input-component property is:

```text
input_layer.visual_space_resolution_deg
```

Both the legacy and eccentricity input components implement it.

The required cortical coordinate interface is:

```text
sheet.sheet_to_cortical_um(x_sheet, y_sheet)
sheet.cortical_um_to_sheet(x_um, y_um)
sheet.sheet_distance_to_cortical_um(distance_sheet)
sheet.sheet_coordinate_bounds()
sheet.visual_to_cortical_mm(x_deg, y_deg)
sheet.cortical_to_visual_deg(x_sheet, y_sheet)
```

`RetinotopicCorticalSheet` positions are `(u_mm, a_mm)` in millimetres.
`sheet_to_cortical_um` therefore multiplies both coordinates by 1000, and
`sheet_distance_to_cortical_um` multiplies distances by 1000. The visual
conversion methods delegate to the shared topography.

`sheet_coordinate_bounds()` returns `(x_min, x_max, y_min, y_max)` in stored
sheet coordinates. It returns centred bounds for legacy sheets and
`(0, L_u, 0, L_a)` for the retinotopic sheet. Border-dependent connector
functions must use this method instead of unconditionally adding
`size_x/2` and `size_y/2`.

`RetinotopicCorticalSheet.size_in_degrees()` must raise
`NotImplementedError`, because no rectangular visual-degree size exists for
the nonlinear sheet. Every supported eccentricity-mode caller must use one of
the explicit interfaces above.

`RetinalInhomogeneousDiskSheet` stores Cartesian degrees and its
`size_in_degrees()` reports the enclosing visual rectangle `(2*E_max,
2*E_max)` only for APIs that require a source bounding box. This value does
not mean that the corners contain neurons. It must not provide cortical
distance conversion methods.

`RetinotopicCorticalSheet` calls `Sheet.__init__` with `size_x=L_u` and
`size_y=L_a`, both in millimetres, then constructs its population from an
`ExplicitPositions` structure. It obtains `topography` only from
`model.input_layer`; absence of an eccentricity input is a configuration
error. A population count of zero is an error rather than a silent empty
sheet.

## Mode separation and configuration

### Legacy mode

Do not change:

- `SpatioTemporalFilterRetinaLGN`;
- `RetinalUniformSheet`;
- `VisualCorticalUniformSheet`;
- legacy `GaborConnector` execution;
- legacy `density` meaning;
- fixed RF sharing;
- fixed stimulus resolution;
- legacy luminance calculation;
- linear magnification methods;
- legacy Gabor frequency draws;
- seeded draw order.

The configured input-component class is the canonical mode switch:

```text
SpatioTemporalFilterRetinaLGN
    -> legacy mode

EccentricityDependentSpatioTemporalFilterRetinaLGN
    -> eccentricity mode
```

Model construction must then select matching cortical sheets and the matching
Gabor connector. Mixed combinations are invalid:

- eccentricity input plus `VisualCorticalUniformSheet`: error;
- eccentricity input plus legacy `GaborConnector`: error in model wiring or
  connector validation;
- legacy input plus `RetinotopicCorticalSheet`: error;
- legacy input plus `EccentricityDependentGaborConnector`: error.

Phase 1 avoids every mixed combination by constructing no cortex. Phase 2
adds validation at consuming-model and connector boundaries; Phase 1 must not
modify `VisualCorticalUniformSheet` or legacy `GaborConnector` merely to
support a temporary hybrid.

Existing configurations select existing classes and acquire no new required
parameters. Because current model constructors invoke `GaborConnector`
directly, each model supporting the new mode must make the connector-class
selection explicit after constructing `input_layer`; adding the new classes
alone is not sufficient.

### Eccentricity mode components

Introduce these exact components:

- `EccentricityDependentSpatioTemporalFilterRetinaLGN`;
- `RetinalInhomogeneousDiskSheet`;
- `RetinotopicCorticalSheet`;
- `EccentricityDependentGaborConnector`;
- `RadiallySymmetricLGNTopography`.

The topography provider is a small immutable configured object, not a large
framework. Only the eccentricity input component constructs it. The provider
is then shared by object reference with ON/OFF LGN sheets and cortical sheets,
as specified under [Authoritative configuration sources](#authoritative-configuration-sources).

### New LGN parameters

The eccentricity LGN component replaces legacy `density` with:

- `number_per_polarity`: positive integer;
- `minimum_samples_per_center_sigma`: positive finite float;
- `topography.cap_eccentricity`: `None` or positive finite degrees;
- `topography.full_max_eccentricity`: default `90.0`;
- `topography.beta`: default `1.59`.

It retains:

- reference RF function and parameters;
- reference RF width and height;
- temporal resolution and duration;
- cell configuration;
- gain control;
- noise;
- recording configuration.

It does not accept legacy `size`; the domain comes from `model.visual_field`.
It does not accept `receptive_field.spatial_resolution`; resolution comes from
the minimum sampling requirement. Mozaik parameter validation requires exact
key equality, so supplying either legacy-only key to the new component raises
`KeyError`.

Use these exact `required_parameters` types for optional values:

```text
"cap_eccentricity": (float, type(None))
```

The existing `ParametrizedObject.check_parameters` passes such a tuple to
`isinstance`, so no global parameter-checker change is required.

The complete eccentricity input schema is the legacy schema with only the
following deliberate substitutions:

```text
remove:
    density
    size
    receptive_field.spatial_resolution

add:
    number_per_polarity: int
    minimum_samples_per_center_sigma: float
    topography:
        cap_eccentricity: (float, type(None))
        full_max_eccentricity: float
        beta: float

retain unchanged:
    linear_scaler: float
    mpi_reproducible_noise: bool
    recorders: ParameterSet
    recording_interval: float
    receptive_field:
        func: str
        func_params: ParameterSet
        width: float
        height: float
        temporal_resolution: float
        duration: float
    original_2024_lgn_mode: bool
    cell:
        model: str
        params: ParameterSet
        receptors: ParameterSet
        native_nest: bool
        initial_values: ParameterSet
    gain_control:
        gain: float
        non_linear_gain:
            luminance_gain: float
            luminance_scaler: float
            contrast_gain: float
            contrast_scaler: float
    noise:
        mean: float
        stdev: float
```

`number_per_polarity` must be an integer greater than zero; do not accept a
float that happens to be integral despite the parameter framework's historic
`int`/`float` leniency. The sampling requirement, temporal dimensions, RF
dimensions, and `full_max_eccentricity` must be positive and finite. The
component explicitly rejects `original_2024_lgn_mode=True` before allocating
populations or kernels.

The internally created `RetinalInhomogeneousDiskSheet` receives only the
inherited `Sheet` parameter keys (`cell`, `mpi_safe`,
`artificial_stimulators`, `name`, `recorders`, and `recording_interval`).
Count and positions are constructor data supplied by the owning input
component, not additional sheet configuration.

### New cortical parameters

The retinotopic cortical sheet retains:

- `density` in neurons per square millimetre;
- cell and recording configuration.

It accepts:

- `sx`: positive micrometres or `None`;
- `sy`: positive micrometres or `None`;

It does not use a linear visual-field `magnification_factor`.

Declare:

```text
"sx": (float, type(None))
"sy": (float, type(None))
```

in `required_parameters`. Do not add optional-value support globally.

The cortical sheet takes no duplicate topography parameters. During
construction it requires an eccentricity input layer and obtains:

```text
self.topography = model.input_layer.topography
```

Its complete schema is the inherited `Sheet` schema plus:

```text
density: float
sx: (float, type(None))
sy: (float, type(None))
```

The new Gabor connector has the same schema as legacy `GaborConnector` except
that `frequency` and `topological` are absent. Topological placement and
position-derived frequency are mandatory in this class:

```text
target_synapses, aspect_ratio, size, orientation_preference, phase,
rf_jitter, off_bias, delay_functions, delay_expression, local_module,
short_term_plasticity, base_weight, num_samples_functions,
num_samples_expression, num_samples, or_map, or_map_location,
or_map_stretch, phase_map, phase_map_location, gauss_coefficient
```

Use the exact types from `GaborConnector.required_parameters` for every
retained key. Supplying `frequency` or `topological` to the new connector is a
configuration error; omitting them from legacy configuration is still an
error.

The words "default" above describe the recommended values that must appear in
a model parameter file. Mozaik's current exact `ParameterSet` validation does
not inject defaults. A minimal mode-specific configuration shape is:

```text
Phase 1 input component:
    number_per_polarity = N
    minimum_samples_per_center_sigma = S
    topography:
        cap_eccentricity = None
        full_max_eccentricity = 90.0
        beta = 1.59
    receptive_field:
        # legacy RF keys except spatial_resolution

Phase 2 cortical sheet:
    sx = None
    sy = None
    density = <neurons/mm^2>
    # inherited Sheet keys

Phase 2 Gabor connector:
    # legacy keys except frequency and topological
```

Existing legacy parameter files are not migrated and retain all old keys.

## Initialization and runtime sequence

The required initialization order in eccentricity mode is:

1. The consuming model constructs a centred `VisualRegion` and `VisualSpace`.
2. The eccentricity input validates the visual field and its own exact
   parameter schema.
3. It constructs one `RadiallySymmetricLGNTopography`, resolves `E_max`, cap
   policy, LGN density/RF behavior, and warning conditions. It stores and
   validates mapping calibration configuration without constructing cortex.
4. It obtains two position seeds, samples complete ON/OFF arrays, and creates
   exactly two `RetinalInhomogeneousDiskSheet` populations.
5. It allocates noise/current-source infrastructure using the legacy global
   cell-order/local ownership rules.
6. It derives every cell's RF scale, determines the one global image
   resolution from the smallest actual centre sigma, then builds complete
   per-local-cell ON/OFF kernels and input cells.
7. The input constructor returns and the consuming model assigns it to
   `model.input_layer`.
8. In Phase 2, the shared provider resolves its mapping constants. Each
   `RetinotopicCorticalSheet` retrieves it, resolves `sx/sy`, samples explicit
   physical cortical positions, and constructs its unchanged PyNN cell
   population.
9. The eccentricity Gabor connector maps every global cortical position to an
   unjittered visual position, derives local carrier frequency, samples the
   retained parameters and accepted jitter in deterministic global order,
   writes the existing annotation names, and invokes the unchanged ON/OFF
   modular sampling connectors.
10. Recurrent connectors run normally and consume the replaced annotations.

Steps 1 through 7 are the complete Phase 1 initialization. Steps 8 through 10
are added in Phase 2. A Phase 1 model returns after step 7 and is fully usable
for LGN stimulus-response experiments; it does not install placeholders for
steps 8 through 10.

At stimulus runtime:

1. `VisualExperiment` requests the common input resolution and renders the
   same rectangular image resolution for all cells.
2. Each local LGN cell requests its own rectangular RF region, which may
   extend beyond the RF-centre disk, and convolves it with its own kernel.
3. Contrast follows the existing zero-mean path; luminance follows the
   eccentricity-only spatial-sum path.
4. Existing gain, temporal-state, current injection, noise, and PyNN/NEST
   spiking machinery runs unchanged.
5. The input's stored full stimulus frames use the same derived pixel size;
   caching semantics remain unchanged.

### Cap semantics

`cap_eccentricity=None` deliberately has different retinal and cortical
consequences:

- cortical mapping uses its empirically derived cap to avoid singular central
  magnification;
- LGN density is not capped;
- LGN RF size is not capped;
- cortical Gabor frequency uses the uncapped local RF size.

Phase 1 implements and tests the two LGN bullets and stores the cap metadata.
Phase 2 implements and tests the cortical mapping and Gabor bullets. An
explicit cap's LGN density/RF effects belong to Phase 1; using that identical
cap in mapping and Gabor frequency is a Phase 2 regression against the Phase 1
resolved value.

The uncapped RF is well-defined because `sigma_c(0)` is nonzero.

For an explicit cap:

- require the map's empirical lower bound;
- require `cap_eccentricity <= E_max`;
- apply the exact same explicit cap to mapping, density, RF size, surround
  size, and theoretical Gabor frequency.

Equality with `E_max` is allowed and makes the entire modelled retinal domain
capped.

## Visual and cortical domains

### Coordinate convention summary

| Space | Stored coordinates | Units | Origin and positive directions |
|---|---|---|---|
| LGN / visual field | `(x, y, 0)` | degrees | Fixation is `(0, 0)`; positive `x` is rightward and positive `y` is upward in visual coordinates. Mathematically, `theta=atan2(y,x)` increases counter-clockwise. |
| Analytical cortical map | `(u, v)` | millimetres | `u=0` is fixation and increases with eccentricity; `v=0` corresponds to visual angle zero and increases with `theta`. This centred form is internal. |
| Retinotopic cortical sheet | `(u, a, 0)` | millimetres | Lower-left corner is `(0, 0)`; `a=v+L_max/2`. Thus `a=0` is the `theta=-pi` seam, `a=L_max/2` is `theta=0`, and `a` increases with visual polar angle. |
| Physical cortical APIs | `(x_um, y_um)` | micrometres | Same origin and directions as the owning cortical sheet, with unit conversion only for the retinotopic sheet. |

Array image indices are not coordinate axes: current RF quantization reverses
the meshgrid inputs to match row/column storage. Preserve that established
rendering convention and test asymmetric coordinate probes so the new mapping
does not introduce an x/y transpose or sign reversal.

### Visual domain

Require the configured visual-field centre to equal `(0, 0)`.

For visual width `W` and height `H`, define:

$$
E_{\max}=\frac{\min(W,H)}{2}.
$$

Require:

$$
0<E_{\max}<90\ \text{degrees}.
$$

LGN RF centres occupy:

$$
D_{\mathrm{LGN}}
=
\{(x,y):x^2+y^2<E_{\max}^2\}.
$$

The visual rectangle remains the stimulus rendering domain. Its corners and
any excess band along the longer dimension contain no RF centres, but pixels
are not masked.

### Full cortical-map extent

Compute mapping parameters from `E_max` and the configured cap policy. Let:

$$
L_{\max}
=
\sqrt{\text{simulated cortical area}}
$$

in millimetres.

This full mapping is fixed by `E_max`, even when the simulated cortical sheet
is smaller.

### Configured cortical extent

Resolve `sx` and `sy` independently:

- `sx=None` gives `sx = 1000 * L_max` micrometres;
- `sy=None` gives `sy = 1000 * L_max` micrometres;
- an explicit companion dimension remains explicit.

Reject either value when it is nonpositive or greater than
`1000 * L_max`.

Convert configured dimensions to:

$$
L_u=s_x/1000,\qquad L_a=s_y/1000
$$

in millimetres. Generate cortical neurons over:

$$
0<u\leq L_u,\qquad 0\leq a<L_a.
$$

Mapping to visual space always uses:

$$
v=a-L_{\max}/2,
$$

not `a - L_a/2`. Therefore:

- reducing `L_u` omits peripheral eccentricities;
- reducing `L_a` retains an angular interval beginning at the `-pi` seam;
- neither crop changes angular scale or radial magnification.

This lower-left anchoring is intentional.

### Cortical population count and density

Keep cortical density semantics unchanged:

$$
N_{\mathrm{cortex}}
=
\operatorname{int}(L_uL_a\,\rho_{\mathrm{cortex}}),
$$

where density is neurons per square millimetre.

Changing `E_max`, the cap, or explicit `sx/sy` can therefore change total
cortical neuron count while preserving density.

## Coordinate interfaces

The current names `vf_2_cs`, `cs_2_vf`, and `dvf_2_dcs` combine coordinate
conversion with the assumption of a scalar magnification. Do not give them
misleading nonlinear meanings.

Introduce these exact sheet interfaces:

- `sheet_to_cortical_um(x_sheet, y_sheet)`;
- `cortical_um_to_sheet(x_um, y_um)`;
- `sheet_distance_to_cortical_um(distance_sheet)`;
- `visual_to_cortical_mm(x_deg, y_deg)`;
- `cortical_to_visual_deg(u_mm, a_mm)`.

`SheetWithMagnificationFactor` implements the first three methods as aliases
with exactly the same arithmetic as its existing conversion methods:
sheet coordinates are visual-degree equivalents and physical coordinates are
micrometres. Its old public methods remain untouched. The retinotopic sheet
stores physical millimetre coordinates, so the first three methods only
multiply or divide by 1000. Retinal sheets do not implement cortical
conversion.

Update the known generic callers in:

- `mozaik/connectors/modular_connector_functions.py`;
- `mozaik/connectors/fast.py`;
- `mozaik/sheets/direct_stimulator.py`;
- `mozaik/sheets/population_selector.py`.

Distance-dependent cortical connectors must continue receiving physical
micrometre distances. Pointwise retinotopic conversion must never be used as a
scalar distance conversion.

Apply the interfaces to existing callers as follows:

- `DistanceDependentModularConnectorFunction` computes Euclidean distance in
  stored sheet coordinates and calls `source.sheet_distance_to_cortical_um`.
  It raises a descriptive error when used with a source that has no cortical
  metric.
- `DistanceDependentProbabilisticArborization` checks for the explicit
  distance method, not `isinstance(SheetWithMagnificationFactor)`.
- Threshold-based modular contact-count functions use
  `sheet_coordinate_bounds`; this preserves their legacy centred result and
  gives the new sheet correct corner-origin border distances.
- `OpticalStimulatorArray`, cortical stimulation-pattern helpers, and
  region/grid selectors with micrometre inputs use
  `sheet_to_cortical_um` or `cortical_um_to_sheet` as appropriate.
- `MapDependentModularConnectorFunction` interprets map arrays over the
  configured cortical rectangle when its source and target are retinotopic
  cortical sheets. Its legacy branch is byte-for-byte behaviorally unchanged.
- `CoCircularModularConnectorFunction` raises `NotImplementedError` for
  `RetinotopicCorticalSheet`. Its current `atan2` treats stored axes as a
  Euclidean visual plane; applying it to eccentricity and unwrapped-angle axes
  would silently change the scientific model. Available future choices are to
  define co-circularity in mapped Cartesian visual coordinates or to define a
  cortical-manifold rule. Neither is part of this phase.
- Any other caller of `size_in_degrees` on a retinotopic cortical sheet must
  be converted only if it has an unambiguous cortical-coordinate
  interpretation; otherwise it must fail explicitly.

All cortical-coordinate stimulator and selector parameters are corner-origin
in eccentricity mode. Existing utilities that generate a grid around a
configured offset keep doing so, but a user who wants the centre of the new
sheet must provide `(500*L_u, 500*L_a)` in micrometres. Do not silently add a
half-sheet translation.

The modular connector's existing `local_module` calculation remains anchored
at stored sheet `(0, 0)`. In eccentricity mode that is the cortical corner,
not the sheet centre. This is a direct consequence of the approved origin and
is not silently recentered. Models using a nonempty `local_module` must accept
that consequence or request a separate scientific redesign.

`plot_layer_activity(..., cortical_coordinates=True)` converts either cortical
sheet through `sheet_to_cortical_um`. With
`cortical_coordinates=False`, a retinotopic sheet is mapped pointwise through
`cortical_to_visual_deg`; legacy plotting remains unchanged.
`NauhausAnalysis` in `analysis/lfp.py` divides physical distances by one
scalar `magnification_factor`, which has no nonlinear equivalent. It must
raise a named unsupported-operation error when sheet parameters lack that
legacy scalar. Implementing a retinotopic version requires a separate choice
between local visual distance and cortical physical distance.

Audit every use of:

- `magnification_factor`;
- `vf_2_cs`;
- `cs_2_vf`;
- `dvf_2_dcs`;
- `size_in_degrees`.

The eccentricity path must either use a semantically valid replacement or
raise a clear unsupported-operation error.

## LGN position generation

### Required distribution

The density is radially symmetric and the approved RF-centre domain is a full
disk. Sampling can therefore use a uniform-area disk proposal without
rectangle angular corrections.

The target Cartesian probability density is:

$$
p(x,y)=
\frac{\rho_{\mathrm{effective}}(\operatorname{hypot}(x,y))}
{2\pi\int_0^{E_{\max}}r\rho_{\mathrm{effective}}(r)\,dr}
$$

inside the disk and zero outside it. The production sampler need not evaluate
the denominator, because rejection from a uniform-area proposal produces this
normalized distribution exactly. Tests must evaluate the denominator
independently.

For independent uniform variates `U` and `V`:

$$
r=E_{\max}\sqrt{U},\qquad
\theta=2\pi V-\pi.
$$

The candidate Cartesian position is:

$$
x=r\cos(\theta),\qquad y=r\sin(\theta).
$$

For no user cap, accept with:

$$
P_{\mathrm{accept}}(r)=\frac{\rho(r)}{\rho(0)}.
$$

For an explicit cap, accept with:

$$
P_{\mathrm{accept}}(r)
=
\frac{\rho(\max(r,E_{\mathrm{cap}}))}
{\rho(E_{\mathrm{cap}})}.
$$

The uniform-area proposal already contains the radial area factor. Do not
multiply density by `r` again.

Continue drawing until exactly `number_per_polarity` positions have been
accepted.

### ON/OFF and MPI reproducibility

Request a fixed number of position seeds through Mozaik's seeded
infrastructure on every MPI rank. Use separate deterministic streams for ON
and OFF.

Concretely, call `mozaik.get_seeds(2)` exactly once, on every rank and before
any position draws. Construct two
`numpy.random.RandomState` objects from the returned seeds in `X_ON`, `X_OFF`
order. Draw every proposal and rejection for a polarity from its private
stream. Do not draw these positions from global `mozaik.rng` or
`mozaik.pynn_rng`, because rejection count must not perturb unrelated
component streams.

Generate the full global ON position array and full global OFF position array
identically on every rank. Construct PyNN populations from explicit global
positions, then let PyNN assign local cells.

Requirements:

- exactly `N` ON cells;
- exactly `N` OFF cells;
- ON and OFF draws are independent;
- the same configuration and seed give identical positions;
- different seeds give different valid positions;
- results do not depend on MPI rank count.

The new sheet should force position realization during initialization as the
legacy sheet does.

After population construction, retain a read-only canonical `(2, N)` position
array on each retinal sheet for test/debug comparison and assert that PyNN's
realized first two position rows are exactly equal to it. Noise seeding then
follows the existing global-cell loop: request one seed per global cell in a
fixed polarity order and instantiate current-source RNGs only for locally
owned cells.

## Cortical position generation

Generate cortical neurons uniformly in the configured physical rectangle
while preserving the existing density/count calculation.

Request exactly one seed with `mozaik.get_seeds(1)` for each cortical sheet,
on every rank and in model construction order. Use a private
`RandomState` to generate the full global `(u, a)` array and construct the
PyNN population through `ExplicitPositions`. This makes cortical positions
independent of MPI partitioning while retaining deterministic dependence on
the normal Mozaik seed and sheet construction order.

Use:

$$
u\in[0,L_u],\qquad a\in[0,L_a)
$$

as the proposal domain.

An exactly generated `u == 0` is invalid because the entire `u=0` boundary
maps to fixation and is not invertible. Replace only exact boundary values
with:

$$
u_{\epsilon}=10^{-12}L_{\max}.
$$

Do not exclude a finite central strip. If later numerical tests show that this
epsilon is insufficient for stable round trips, change it only with a
documented numerical justification.

The upper angular edge remains half-open. Values on opposite sides of the
angular seam are not treated as neighbours in this phase.

## Per-cell receptive fields

### Supported RF type

At initialization, require:

- RF function is the Cai97 `stRF_2d` implementation;
- `subtract_mean` is false;
- required Cai97 parameters are present and finite;
- `original_2024_lgn_mode` is false.

Raise clear configuration errors rather than falling back to generic shared
kernels or old response shortcuts.

### Sigma assignment

For each LGN position:

$$
E_i=\sqrt{x_i^2+y_i^2}.
$$

If no user cap:

$$
\sigma_{c,i}=\sigma_c(E_i).
$$

With an explicit cap:

$$
\sigma_{c,i}=\sigma_c(\max(E_i,E_{\mathrm{cap}})).
$$

Let reference parameters be:

$$
\sigma_{c,\mathrm{ref}},\qquad
\sigma_{s,\mathrm{ref}}.
$$

Set:

$$
\mathrm{scale}_i
=
\frac{\sigma_{c,i}}{\sigma_{c,\mathrm{ref}}},
$$

$$
\sigma_{s,i}
=
\mathrm{scale}_i\sigma_{s,\mathrm{ref}}.
$$

Thus:

$$
\frac{\sigma_{s,i}}{\sigma_{c,i}}
=
\frac{\sigma_{s,\mathrm{ref}}}{\sigma_{c,\mathrm{ref}}}.
$$

Keep `Ac`, `As`, and all temporal parameters unchanged.

### Spatial support

Scale support with the same factor:

$$
W_i=\mathrm{scale}_iW_{\mathrm{ref}},
\qquad
H_i=\mathrm{scale}_iH_{\mathrm{ref}}.
$$

This preserves truncation in centre and surround sigma units. A larger
surround must not be forced into the reference support.

Quantization may expand realised support by less than one pixel because it
uses `ceil`. Tests must measure this separately from biological support
scaling.

### Full 3D implementation

Construct one complete `SpatioTemporalReceptiveField` for each locally owned
LGN cell. ON cells use the Cai97 function and OFF cells use its negation.

Do not introduce factorization, lazy regeneration, scale bins, or
interpolation in the first implementation. Full kernels provide the direct
reference behavior needed to discover the actual bottleneck.

## Stimulus spatial resolution

Use minimum samples per centre sigma as the numerical sampling criterion. It
corresponds directly to the value returned by `rf_center_sigma` and does not
depend on an arbitrary support multiplier.

After both global LGN populations have been generated:

1. Calculate every cell's effective centre sigma.
2. Find the smallest sigma actually present.
3. Calculate:

   $$
   dx_{\mathrm{raw}}
   =
   \frac{\min_i\sigma_{c,i}}
   {\text{minimum_samples_per_center_sigma}}.
   $$

4. Round downward to two significant digits.
5. Use that `dx` for all RFs and all stimulus rendering.

For a positive value `x`, two-significant-digit downward rounding can be
defined deterministically by:

$$
k=\lfloor\log_{10}x\rfloor,
$$

$$
q=10^{k-1},
$$

$$
\operatorname{round\_down}_2(x)=q\lfloor x/q\rfloor.
$$

For example, `0.151218...` becomes `0.15`, never a larger value. Implement the
quantizer using `decimal.Decimal(str(dx_raw))` and `ROUND_FLOOR`, with the
power-of-ten exponent determined from the decimal value rather than a binary
floating-point logarithm. Convert the final two-significant-digit decimal to
`float` once. This makes values at decimal boundaries deterministic. Test
values exactly on and immediately to either side of powers of ten.

Verify after rounding and RF quantization that:

$$
\sigma_{c,i}/dx
\geq
\text{minimum_samples_per_center_sigma}
$$

for every cell.

Expose the common input-component property
`visual_space_resolution_deg`:

- legacy input returns configured `receptive_field.spatial_resolution`;
- eccentricity input returns the derived resolution.

Change `VisualExperiment` to use the property. Do not change the resolution of
legacy stimuli.

The eccentricity component stores this value as an ordinary attribute and
uses it everywhere a shared `self.rf["X_ON"].spatial_resolution` was
previously used, including the stored full-stimulus image in
`calculate_kernel_responses`. There is no representative shared RF in the new
component. `input_cells[rf_type]` is authoritative for per-cell kernels.
Update `devtools.DummyModel` to expose
`input_layer.visual_space_resolution_deg` while retaining its legacy nested
parameter fixture if existing tests use it.

## Eccentricity-mode luminance correction

### Problem

The quantized kernel is already divided by `nx * ny * nt`. The legacy
luminance path then takes its spatial mean and multiplies that by mean image
luminance. This loses a factor `nx * ny`.

When RF width and height scale by `s` at fixed pixel size, the number of
spatial samples grows approximately as `s^2`. The legacy luminance response
therefore falls approximately as `1/s^2`, even though complete convolution and
the contrast path remain nearly invariant.

This is scientifically unacceptable for an LGN population whose RF size
varies systematically with eccentricity.

### Preliminary characterization

Before this design was finalized, an exploratory calculation scaled the RF,
its support, and representative stimuli together by factors:

```text
0.5, 1, 2, 4
```

at fixed pixel size. It found:

- direct complete-kernel response varied by less than approximately `9e-6`
  relative;
- contrast-path response varied by less than approximately `2.3e-4`;
- luminance-path response followed the expected approximate `1/s^2`
  dependence;
- final injected current varied by approximately 3.5 to 4 percent under the
  tested default gain conditions.

These are exploratory measurements, not regression tolerances. They indicate
that the scale dependence is specifically associated with the luminance path,
while its effect on final current depends on stimulus and gain operating
point.

### New-mode calculation

Implement this without duplicating the cell response pipeline. Refactor
`CellWithReceptiveField` to call one protected hook while preparing kernel
components:

```text
_luminance_kernel_from_spatial_mean(spatial_mean, kernel_shape)
```

The legacy implementation returns `spatial_mean` exactly. The eccentricity
cell override returns `spatial_mean * kernel_shape[0] * kernel_shape[1]`.
Contrast construction always subtracts the unchanged spatial mean. All
subsequent initialization, state carry-over, `view`, null response, and
current-generation code continues to consume
`rf.kernel_luminance_component`. This single hook prevents one state path from
being corrected while another remains mean-normalized. Because legacy returns
the pre-refactor value exactly, its arrays must be bitwise unchanged.

For the eccentricity-specific cell path, calculate:

$$
\bar K(t)=\operatorname{mean}_{x,y}K(x,y,t),
$$

$$
K_{\mathrm{contrast}}(x,y,t)=K(x,y,t)-\bar K(t),
$$

and:

$$
K_{\mathrm{luminance}}(t)
=
\sum_{x,y}K(x,y,t)
=
n_xn_y\bar K(t).
$$

Continue multiplying `K_luminance` by mean image luminance. This is equivalent
to the mean-image contribution of a full spatial sum.

Use the summed luminance kernel consistently for:

- initial background state;
- luminance step response;
- blank/null input;
- explicit stimulus response;
- temporal state carried between frames or presentations;
- injected-current calculation.

The contrast calculation remains unchanged.

### Compatibility boundary

Do not change the legacy `CellWithReceptiveField` calculation. Implement a
separate eccentricity cell subclass or an explicit response-normalization
strategy selected only by the eccentricity component.

Do not recalibrate luminance gain, contrast gain, amplitudes, or neuron
parameters as part of this change. The corrected current may differ
substantially from historically calibrated currents. Measure and report that
effect before making a later scientific calibration decision.

Supplementary problem description:

```text
/home/tibor/luminance_scaling_bug.md
```

The implementation must not depend on that external file; the necessary
design is included here.

## Retinocortical mapping

### Direction used by the connector

The natural direction is cortical-to-visual:

1. Read the cortical neuron's unchanged local `(u, a)` position.
2. Convert `a` to mapping coordinate `v = a - L_max/2`.
3. Call the inverse retinotopic mapping to obtain `(r, theta)`.
4. Convert to Cartesian visual degrees `(x, y)`.
5. Use `(x, y)` as the cortical neuron's theoretical visual RF centre.

Do not map every LGN position forward and do not transform coordinates back
and forth unnecessarily.

Using inverse mapping avoids having to transform a Cartesian Gabor envelope
through a nonlinear anisotropic Jacobian.

### Fixation singularity

All points on `u=0` map to fixation, so conceptually:

```text
visual_to_cortical_mm(cortical_to_visual_deg(0, a))
```

cannot reproduce every `a`. Excluding generated `u=0` neurons resolves this
for simulated cells but does not make the mathematical mapping bijective at
the boundary.

Mapping tests must:

- test both round trips for `u > 0`;
- test fixation separately;
- not demand cortical round-trip identity along `u=0`.

### Angular seam

The inverse map returns angle in `[-pi, pi)`. The forward map may accept `pi`,
which is the same visual direction as `-pi` but the opposite end of the
unwrapped cortical sheet.

Tests must compare angles modulo `2*pi`, avoid treating the upper endpoint as
an independent point, and document the expected discontinuity. Do not add
periodic cortical connectivity in this phase.

Add a TODO near seam-sensitive connectivity indicating that major behavioral
effects may require future treatment.

## Gabor receptive fields

### Annotation generation

The eccentricity-specific Gabor connector must require topological mapping.
It does not draw carrier frequency from the legacy `frequency` distribution.

For each cortical neuron:

1. Obtain its unjittered visual RF centre through `cortical_to_visual`.
2. Calculate unjittered visual eccentricity.
3. Calculate local centre sigma using the cap policy.
4. Calculate local surround sigma with the LGN reference ratio.
5. Use the exact LGN reference `Ac` and `As`.
6. Calculate theoretical `dog_optimal_spatial_frequency`.
7. Draw orientation, phase, aspect ratio, and envelope size as before.
8. Load orientation or phase map values in cortical coordinates when enabled.
9. Draw RF-centre jitter and resample until the jittered centre is inside the
   LGN centre disk.
10. Write the standard `LGNAfferent*` annotations.

`LGNAfferentFrequency` is based on the unjittered position.
`LGNAfferentX/Y` contain the accepted jittered visual position.

### Jitter rejection

Draw x and y jitter together. Accept when:

$$
\sqrt{x_{\mathrm{jittered}}^2+y_{\mathrm{jittered}}^2}
\leq E_{\max}.
$$

Do not clip or project onto the boundary. After 10,000 rejected pairs, raise an
error containing:

- cortical neuron identifier;
- cortical position;
- unjittered visual position;
- `E_max`;
- jitter distribution description.

All ranks must consume identical draws. The resampling loop must therefore run
over global cortical neurons in a fixed order on every rank.

### Gabor units and unchanged parameters

`GaborArborization` evaluates LGN source positions and Gabor centres in visual
degrees. Therefore `LGNAfferentFrequency` remains cycles per degree; no
cortical-distance frequency conversion is needed.

Do not change:

- Gabor envelope size;
- aspect ratio;
- orientation;
- phase;
- Gaussian coefficient;
- contact count;
- base weight;
- delays.

Changing carrier frequency without changing envelope size intentionally
changes the number of carrier cycles under the envelope.

### Recurrent layer-4 consequences

Write the new position and frequency into the existing annotation names.
`V1CorrelationBasedConnectivity` will then calculate recurrent RF correlation
using the new visual positions and frequencies.

This change is expected and approved. Do not preserve parallel legacy
annotations in eccentricity mode.

## Orientation and phase maps

Orientation-map arrays represent cortical space, not visual space.

For the retinotopic sheet:

- first array axis corresponds to cortical `u`;
- second array axis corresponds to cortical `a`;
- coordinates use the configured physical cortical rectangle;
- the base grid for `or_map_stretch == 1` spans `[0, L_u] x [0, L_a]`;
- map lookup uses the unjittered cortical position.

Preserve `or_map_stretch` by scaling the map grid about the configured cortical
rectangle's centre. At stretch one, the map exactly spans the sheet.

Phase maps are not currently used, but if enabled they follow the same
cortical coordinate rule.

The user is responsible for supplying a map appropriate for the configured
cortical size. Do not silently stretch a map based on metadata that does not
exist, and do not attempt to infer whether a supplied file represents a
different cortical extent.

Legacy models all use `1000 um/degree`, so their historical degree-equivalent
orientation-map interpolation is numerically identical to millimetre
coordinates. Leave the legacy path unchanged.

## Connectivity sampling

Preserve the existing Gabor candidate weights and modular contact-sampling
mechanism.

Requirements:

- every contact selected by the existing sampler remains connected;
- duplicate selections retain their current aggregate-weight effect;
- no fan-in normalization is added;
- no total incoming-weight normalization is added;
- no density-dependent correction is added.

At the feedforward connection-construction stage add:

```text
TODO: Nonuniform LGN density may create eccentricity-dependent unique fan-in
and aggregate input. Consider afferent-count or weight normalization in a
future, separately validated model change.
```

## Theoretical and measured LGN preferred spatial frequency

### Theoretical value

For each representative eccentricity:

1. construct the same effective centre and surround parameters used by the LGN
   cell;
2. calculate the analytical DoG optimum;
3. record the theoretical cycles-per-degree value.

Also verify the analytical formula against a dense numerical Fourier-domain
evaluation of the continuous DoG.

### Full-neuron characterization

Use drifting sinusoidal gratings and define measured preferred spatial
frequency as the frequency maximizing spike-output F1.

Approved initial settings:

- contrast: 100 percent;
- duration: 2 seconds;
- temporal frequency: 1 Hz;
- spatial coordinate window: large enough to contain the entire RF support.

Record optima or response curves at:

- linear spatial/spatiotemporal filter response;
- injected-current response;
- spike-output F1.

The first implementation deliberately leaves the following for manual
selection:

- mean luminance;
- trial count;
- transient-discard interval;
- spatial-frequency grid;
- refinement grid;
- peak-fitting estimator;
- final agreement tolerance.

The test must:

1. generate a plot containing theoretical and measured curves;
2. make the output path visible;
3. contain a `TODO Finish` comment at the unfinished validation point;
4. fail unconditionally after writing the plot.

This failing test is an expected deliverable, not a passing acceptance test.
It prevents incomplete scientific criteria from being mistaken for completed
validation.

Do not recalibrate the model if analytical and spike-output optima differ.
Report which processing stage introduces the difference.

## Gabor-profile validation

Generate representative layer-4 Gabor profiles at several unjittered
eccentricities, including:

- the central uncapped or capped region;
- an intermediate eccentricity;
- near 25 degrees when represented;
- a larger extrapolated eccentricity when represented.

Use identical envelope, aspect ratio, orientation, and phase. Plot:

- a two-dimensional profile;
- a one-dimensional carrier-axis cross-section;
- the numerical carrier frequency;
- the number of carrier cycles under a fixed envelope interval.

The plots must visibly demonstrate changing carrier frequency and unchanged
envelope.

## Test plan

Phase ownership within this section is:

- **Phase 1:** LGN parts of A; density, RF-size, and DoG parts of B; retinal
  cap behavior in C; all of D, G, H, I, and M; and an LGN-only integration
  subset of L.
- **Phase 2:** cortical/connectivity parts of A; mapping parts of B and C; all
  of E, F, J, and K; Gabor-profile validation; and the complete
  LGN-to-layer-4 form of L.

Tests must be named or marked by directory/class organization so these groups
can be invoked independently without skipping assertions. Phase 2 reruns all
Phase 1 ordinary tests as regressions. The intentional failure in M is run
after the passing Phase 1 suite and is not part of Phase 2's ordinary
pass/fail gate.

### A. Legacy regression

- Existing LGN configurations produce identical ON/OFF counts.
- Existing seeded positions are exactly unchanged.
- Shared legacy kernels are exactly unchanged.
- Legacy kernel responses and injected currents are unchanged.
- Legacy stimulus resolution is unchanged.
- Legacy LGN-to-cortex annotations and connection lists are unchanged.
- Legacy recurrent correlation-based connectivity is unchanged.
- Legacy orientation-map behavior is unchanged.
- Run existing single-process and supported MPI regression configurations.

### B. Topography mathematics

- Density at fixation is finite, positive, and equals the fitted amplitudes'
  sum.
- Equal eccentricities give equal density.
- Cartesian density contains no radial Jacobian.
- Full-disc sampling produces radial mass proportional to `E * rho(E)`, not
  `E^2 * rho(E)`.
- Nonfinite inputs raise clear errors.
- Density extrapolation is finite and warns above its empirical range.
- Known RF eccentricities give expected conventional sigmas.
- RF extrapolation warns once at component initialization.
- DoG reference parameters give approximately `0.800897 cycles/degree`.
- Invalid DoG parameter combinations raise errors.

### C. Cap policy

- `cap=None` uses empirical cortical cap.
- `cap=None` leaves retinal density uncapped.
- `cap=None` leaves RF size uncapped.
- `cap=None` uses uncapped sigmas for theoretical Gabor frequency.
- Explicit cap plateaus cortical magnification, density, and RF size at the
  identical eccentricity.
- Explicit cap propagates to surround size and Gabor frequency.
- Cap below the mapping's empirical minimum is rejected.
- Cap above `E_max` is rejected.
- Cap equal to `E_max` is accepted.

### D. Domain and position sampling

- `E_max` equals half the smaller visual-field edge.
- Off-centre fields are rejected.
- `E_max >= 90 degrees` is rejected.
- Rectangle corners beyond 90 degrees are accepted when `E_max < 90`.
- Exactly `N` ON and `N` OFF positions are generated.
- Every LGN centre satisfies `hypot(x, y) < E_max`.
- No centres fill rectangle corners or excess bands.
- Rectangular stimulus pixels outside `E_max` remain renderable.
- Empirical radial and two-dimensional histograms match integrated capped or
  uncapped density.
- Fixed seeds reproduce exact positions.
- Different seeds produce different valid samples.
- ON and OFF samples are independent.
- Position hashes match across supported MPI rank counts.

### E. Cortical sheet

- Both `sx=None` and `sy=None` derive independently.
- Explicit companion dimensions remain unchanged.
- Nonpositive sizes are rejected.
- Sizes above `L_max` are rejected.
- Smaller `sx` crops eccentricity without rescaling the mapping.
- Smaller `sy` crops angle from the lower seam without rescaling.
- Cortical density and count retain current units and rounding.
- Positions use a lower-left origin.
- Exact `u=0` draws move to the deterministic epsilon.
- No finite-width central strip is removed.

### F. Coordinate transformations

- Cartesian visual and polar visual wrappers agree.
- Both round trips agree away from `u=0` and the seam.
- Visual angles are compared modulo `2*pi`.
- Fixation behavior is tested separately.
- Cortical units are millimetres and visual units are degrees.
- Axis direction and corner origin are explicit.
- Cropped cortical sheets still use full-map `L_max` and angular scale.
- Generic distance connectors receive physical micrometre distances.
- The nonlinear map does not alter LGN Cartesian coordinates.

### G. RF assignment and support

- Each cell receives its position-derived centre sigma.
- Eccentricity is measured from global `(0, 0)`.
- Explicit caps and no-cap behavior are correct.
- Surround-to-centre sigma ratio is constant.
- Amplitudes and temporal parameters remain unchanged.
- Width and height scale with sigma.
- Largest surrounds have the same nominal truncation in sigma units.
- Quantized support expansion is measured.
- RFs near `E_max` use existing rectangular stimulus-boundary semantics.
- Unsupported RF functions, `subtract_mean=True`, and
  `original_2024_lgn_mode=True` raise errors.

### H. Stimulus resolution

- The smallest actual centre sigma controls global resolution.
- The configured samples-per-sigma criterion is satisfied.
- Downward two-significant-digit rounding is deterministic.
- Decimal-boundary cases are stable.
- Higher sampling requirements produce converging responses.
- Legacy resolution remains fixed and unchanged.

### I. Luminance scaling

- New-mode luminance kernel equals the spatial kernel sum.
- New contrast kernel remains zero-mean at every time point.
- New starting and blank luminance states use the summed kernel.
- Geometrically scaled uniform-luminance stimuli give comparable luminance
  responses.
- Geometrically scaled spots and gratings give comparable linear responses.
- Tests cover ON/OFF cells and several scale factors.
- Direct full-kernel, contrast, luminance, combined response, injected current,
  and spikes are reported separately.
- Kernel truncation, quantization, and boundary cases are isolated.
- Legacy mean-based luminance behavior remains exactly unchanged.
- New-mode current changes are quantified without gain compensation.

### J. Gabor annotations and maps

- Mapped visual centre comes from the cortical neuron's position.
- Preferred frequency uses the unjittered centre.
- Jittered centre remains inside `E_max`.
- Jitter rejection is deterministic.
- The 10,000-attempt error contains diagnostic context.
- Centre/surround parameters passed to the DoG formula match the LGN model.
- Frequency is cycles per degree.
- Envelope, aspect ratio, orientation, and phase remain unchanged.
- Orientation maps are interpolated in cortical coordinates.
- Orientation-map stretch is applied about the cortical rectangle centre.
- User-supplied maps are not silently remapped to another cortical extent.
- Gabor profile plots show frequency change and fixed envelope.

### K. Feedforward and recurrent connectivity

- Existing candidate weight calculation is preserved.
- Existing contact sampling is preserved.
- All selected contacts remain.
- Duplicate-contact behavior remains.
- No incoming-weight or fan-in normalization is introduced.
- Mapped positions change feedforward retinotopy as expected.
- New annotations replace legacy annotation values.
- Recurrent RF-correlation connectivity changes consistently with the new
  annotations.
- The future fan-in-normalization TODO is present.
- Legacy connection lists remain unchanged.

### L. End-to-end integration

- A small centred visual field constructs LGN and cortical populations.
- The LGN disk and cortical rectangle have expected dimensions.
- Explicit and no-cap configurations both initialize.
- A cropped cortical sheet initializes and represents only its map subset.
- A stimulus presentation completes through LGN current injection and
  layer-4 input.
- RFs at the disk rim can sample unmasked rectangular stimulus pixels.
- Single-process and supported MPI runs complete reproducibly.
- Memory and timing measurements are emitted.

### M. Analytical versus simulated SF

- Continuous analytical DoG optimum is tested numerically.
- Full LGN neurons receive drifting gratings.
- Linear, current, and spike F1 curves are plotted.
- Representative eccentricities and RF scales are covered.
- Approved initial contrast, duration, temporal frequency, and RF-fitting
  window are used.
- The test writes the requested plot and then fails intentionally with its
  unfinished-validation TODO.

## Acceptance criteria and tolerances

These values make the test plan executable without recovering decisions from
the design conversation.

### Numerical functions

- Compare closed-form density, RF sigma, DoG, and scalar mapping reference
  values with `rtol=1e-12`, `atol=1e-12` in their documented units.
- Compare Cartesian/polar and forward/inverse mapping round trips with
  `rtol=1e-10`, `atol=1e-10` for points at least `1e-9` map units from
  `u=0` and the angular endpoint. Compare angle using modular angular
  distance.
- Test fixation separately: every visual angle at `r=0` maps to `u=0`, while
  no cortical angular round-trip is asserted there.
- Validate array broadcasting against elementwise scalar calls with exact
  shape equality and the same numerical tolerances.

### Sampling and MPI

- For each of uncapped and explicitly capped density, draw `N=50,000` points
  from one fixed test seed.
- Construct the expected radial CDF independently with
  `scipy.integrate.quad` over `r * rho_effective(r)`, using absolute and
  relative integration tolerances `1e-12`. A one-sample Kolmogorov-Smirnov
  test against that CDF must have `p >= 1e-3`.
- Transform angle to a uniform variate on `[0, 1)` and require its
  Kolmogorov-Smirnov `p >= 1e-3`.
- Require every angular quadrant fraction to differ from `0.25` by at most
  `0.015`. Also compare a fixed 8-radial-bin by 12-angular-bin histogram with
  independently integrated expected probabilities and require Pearson
  chi-square `p >= 1e-3`; merge expected-count bins if any expected count is
  below five.
- Exact count, domain membership, fixed-seed equality, and different-seed
  inequality are deterministic assertions independent of the statistical
  tests.
- The mandatory MPI matrix is one and two ranks. Phase 1 compares canonical
  global ON/OFF positions and per-global-ID RF parameters with
  `numpy.array_equal`. Phase 2 additionally compares cortical positions and
  Gabor annotations exactly, then compares connection tuples after sorting by
  projection, source ID, target ID, delay, and weight. A four-rank run is
  recommended when CI capacity permits.

### RFs, resolution, and responses

- Centre and surround parameter assignments use `rtol=1e-12`,
  `atol=1e-12`. Quantized dimensions must equal the documented `ceil`
  calculation exactly.
- The rounded pixel size must equal the expected `Decimal` result exactly as a
  float, and every cell must satisfy the sampling lower bound with
  `atol=1e-12`.
- RF rescaling uses scale factors `0.5, 1, 2, 4`, both polarities, at least
  eight samples per centre sigma, identical temporal sampling, and stimulus
  windows with one pixel of background beyond the largest support. Test a
  uniform field, a Gaussian blob scaled with the RF, and a sinusoidal grating
  whose wavelength is scaled with the RF.
- For boundary-free scaled stimuli, direct full-kernel response,
  contrast-path response, corrected luminance response, and injected current
  must each differ from the scale-one value by no more than one percent of
  `max(abs(reference), 1e-12)`. Report every relative error even when it
  passes. Spike counts are characterized but do not receive this invariance
  assertion because threshold crossings are discontinuous.
- A separate truncation test uses identical RF parameters at different
  supports and a separate boundary test moves the same RF toward the stimulus
  rectangle edge. Their deviations are reported independently and are not
  attributed to normalization.
- At each temporal sample, the new contrast kernel spatial sum must be within
  `atol=1e-12` of zero, and its luminance kernel must equal the direct spatial
  sum with `rtol=1e-12`, `atol=1e-12`.

### Gabor, connectivity, and legacy behavior

- Derived `LGNAfferentFrequency` must match a direct provider/DoG calculation
  at the unjittered position with `rtol=1e-12`, `atol=1e-12`.
- Unchanged envelope, aspect-ratio, orientation, and phase behavior is checked
  with fixed deterministic distributions and exact equality. Map lookup is
  checked at corners, centre, and cell positions against known small arrays.
- Exercise the feedforward sampler independently with identical supplied
  candidate weights and RNG state in legacy and eccentricity connector helper
  paths; selected global IDs and duplicate aggregation must be identical. In a
  separate mapping test, assert the analytically expected candidate weights
  and accept the corresponding intended change in sampled contacts.
- Before each phase's production edits, Stage 0 must record that phase's
  fixed-seed legacy arrays. Phase 1 covers LGN arrays, positions, kernels, and
  currents; Phase 2 adds cortical positions, annotations, and sorted
  connection tuples. Use `numpy.array_equal` or a checked deterministic
  SHA-256 fixture. Simulator traces use the tolerance already declared by the
  nearest existing regression test; if none exists, record and justify backend
  precision before choosing one. No legacy tolerance may be loosened as part
  of this work.
- The Phase 1 Stage 0 post-presentation contrast-response tail has separate
  one-rank and two-rank SHA-256 anchors because of a known bug in the PyNN
  version used by the current Mozaik. If a later PyNN version fixes that bug,
  do not infer a new compatibility target from the changed output. Stop and
  obtain explicit user instruction identifying which recorded legacy result
  is valid. Until then, do not relax, replace, normalize, or delete either
  anchor.
- All existing tests must pass unchanged. The new passing eccentricity suite
  must pass in full before running the deliberately unfinished SF
  characterization.

### Validation artifacts and the intentional failure

Gabor-profile tests write plots under pytest's `tmp_path` and assert that each
file exists and is nonempty; a command-line validation runner may instead use
`MOZAIK_TEST_ARTIFACT_DIR`. It must log the absolute output path.

The full-neuron SF characterization must be isolated in a plainly named test
module or test class and run only after the passing suite has been reported.
After successfully writing its plot it ends with:

```text
assert False, "TODO Finish: define and approve LGN spike-output SF acceptance criteria"
```

Do not mark it skipped or expected-failure. The documented result of the
initial implementation is all ordinary tests passing plus this one explicit
failure. A missing plot, simulation exception, or any different failing test
is not an accepted result.

## Performance and memory

### Expected scaling

For cell `i`:

$$
n_{x,i}\mathrel{\mathop{\simeq}}\lceil W_i/dx\rceil,
\qquad
n_{y,i}\mathrel{\mathop{\simeq}}\lceil H_i/dx\rceil,
\qquad
n_t=\lceil T/dt\rceil.
$$

One float64 3D kernel requires approximately:

$$
8n_{x,i}n_{y,i}n_t\ \text{bytes}.
$$

The current cell path also stores contrast-derived arrays and response state,
so practical per-cell memory is greater than the raw kernel.

With approximately four samples per centre sigma and reference temporal
sampling, earlier design estimates were:

- near fixation: roughly `6.4 MiB` per cell for kernel plus contrast data;
- near 25 degrees: roughly `31.8 MiB`;
- near 90 degrees: potentially about `2 GiB`.

These are order-of-magnitude estimates, not capacity guarantees. Since
`E_max < 90`, the last case is approached but not reached.

### Required reporting

At initialization report:

- `E_max`;
- cap policy and resolved empirical map cap;
- minimum and maximum LGN eccentricity;
- minimum and maximum centre/surround sigma;
- derived degrees per pixel;
- minimum and maximum kernel shapes;
- estimated total local kernel memory;
- ON/OFF global and local cell counts.

Those are Phase 1 reports. Phase 2 additionally reports configured cortical
dimensions, `L_max`, cortical global/local counts, and represented
eccentricity/angular ranges.

Benchmark:

- position-generation time;
- RF quantization time;
- resident memory before and after RF construction;
- per-frame stimulus rendering time;
- per-cell convolution time;
- total input-generation time.

Those are Phase 1 benchmarks. Phase 2 additionally benchmarks cortical
position generation and connector construction.

Report scaling with:

- LGN count;
- `E_max`;
- cap;
- minimum samples per sigma;
- MPI rank count.

Phase 2 adds cortical-count scaling and separates input-layer cost from
feedforward/recurrent connection cost.

### Deferred optimization TODOs

Do not implement these now, but record them near the full-kernel allocation:

- exact centre/surround temporal-spatial factorization;
- kernel banks at eccentricity tiers;
- lazy or streaming kernels;
- recursive Gaussian filtering;
- image pyramids;
- scale-space convolution;
- log-polar-like image pretransformation;
- scale-dependent stimulus resampling.

Choose a future optimization only after benchmarks identify the dominant
bottleneck.

## Implementation sequence

Each stage should be a reviewable chunk with its own focused tests.
Stages 0 through 4 constitute Phase 1. Stages 5 through 7 constitute Phase 2.

### Stage 0: legacy characterization

- Before Phase 1 production changes, capture deterministic legacy LGN counts,
  positions, kernels, currents, stimulus resolution, and relevant
  single-process/MPI behavior.
- At the start of Phase 2, before cortical production changes, add legacy
  cortical positions, annotations, orientation-map behavior, and connection
  lists to the regression anchors.
- Do not make a phase's production changes until that phase's regression
  anchors exist.
- The Stage 0 one-rank and two-rank post-presentation contrast-tail hashes are
  intentionally distinct records of a known PyNN bug. A future PyNN fix does
  not authorize an agent to select one, make the assertion accept both, or
  regenerate the expected value. The agent must stop and obtain explicit user
  instruction defining the valid legacy recording before continuing with an
  affected stage.

Phase 1 exit criterion: legacy LGN behavior can be compared exactly after
every later stage. Phase 2 extends that criterion to cortical behavior.

### Stage 1: mathematical provider

- Add the production topography module.
- Port density, cap, and RF-size formulas.
- Add the corrected convention-specific DoG optimum.
- Store and validate the final mapping calibration configuration without
  implementing cortical coordinate methods yet.
- Document LGN units, domains, provenance, and extrapolation.
- Add Phase 1 pure numerical tests.

Exit criterion: production LGN functions match the source formulas without
importing the external development directory, and the provider can be
extended in place by Phase 2.

### Stage 2: LGN disk positions

- Add eccentricity LGN configuration and validation.
- Add fixed count per polarity.
- Add disk rejection sampling and explicit-position sheets.
- Establish MPI reproducibility.
- Add cap-policy and statistical sampling tests.

Exit criterion: exact counts and statistically correct positions are
reproducible without constructing variable RFs.

### Stage 3: per-cell RFs and resolution

- Assign per-cell sigmas and support.
- Derive global stimulus resolution.
- Construct complete per-cell 3D kernels.
- Expose the common resolution property.
- Add extrapolation logging and memory estimates.
- Add RF/support/resolution tests.

Exit criterion: variable RFs render and convolve at the required shared
resolution.

### Stage 4: new-mode luminance correction

- Add the eccentricity-specific summed-luminance path.
- Cover starting, blank, explicit, and carried state.
- Add RF-rescaling characterization.
- Add theoretical DoG versus complete LGN preferred-SF characterization,
  including the required plot and intentional `TODO Finish` failure.
- Confirm legacy behavior is unchanged.

Exit criterion: the new luminance response no longer has the unintended
`1/s^2` dependence, current changes are reported, all ordinary Phase 1 tests
pass, and the separately run SF characterization produces its plot before its
expected failure.

### Stage 5: retinotopic cortical sheet

- Add the provider's forward/inverse mapping formulas and Cartesian wrappers.
- Document mapping units, fixation singularity, and angular seam.
- Add corner-origin physical cortical coordinates.
- Derive or validate `sx/sy`.
- Exclude exact `u=0`.
- Add explicit sheet/physical/visual coordinate methods.
- Update generic distance/stimulator/selector callers.
- Add cropped-sheet and round-trip tests.

Exit criterion: full and cropped cortical sheets map consistently without a
scalar magnification assumption.

### Stage 6: Gabor and connectivity integration

- Add the eccentricity-specific Gabor connector.
- Calculate theoretical frequency from unjittered position.
- Add deterministic jitter rejection.
- Interpret maps in cortical coordinates.
- Replace standard annotations.
- Verify intended recurrent-connectivity changes.
- Add the fan-in TODO.

Exit criterion: feedforward and recurrent connectivity use the new RF
description while legacy connection lists remain unchanged.

### Stage 7: functional validation

- Generate representative Gabor plots.
- Run the centred disk-to-cortex end-to-end model.
- Add Phase 2 cortical and connector performance reporting.
- Rerun the complete passing Phase 1 suite and separately confirm that its
  intentionally failing SF characterization still writes the expected plot.

Exit criterion: the model runs end to end, limitations are quantified, and
Phase 2 has not changed any Phase 1 contract. The unfinished LGN scientific
acceptance criterion remains visibly separate from the passing Phase 2 suite.

## Known limitations and future work

The following are accepted limitations, not blockers:

- Density and RF-size fits use cat-derived constraints and are extrapolated in
  parts of the supported range.
- The density is radially averaged and lacks visual-streak and nasotemporal
  anisotropy.
- Independent sampling lacks retinal mosaic regularity.
- The angular cortical seam may have behavioral consequences.
- A cropped angular sheet retains the interval from the lower seam rather than
  a centred interval.
- Fixed Gabor envelope size with varying carrier frequency changes cycles per
  envelope.
- Recurrent connectivity changes with the new Gabor annotations.
- Nearby LGN inputs can have slightly different RF scales and preferred
  frequencies from the cortical neuron's theoretical assignment.
- Nonuniform LGN density can change unique fan-in and aggregate input.
- Full per-cell 3D kernels may be prohibitively expensive.
- Corrected luminance responses may make historical gain values unsuitable.
- Orientation-map correctness for a selected cortical size is the user's
  responsibility.
- The full-neuron spatial-frequency agreement criterion is intentionally
  unfinished.

## Remaining decisions

There are no remaining decisions that block the initial implementation.

The following previously implicit compatibility limits are now explicit and
do not require an implementation guess:

- a consuming model must opt into all three new component families in its own
  constructor; the Mozaik library cannot infer and rewrite direct constructor
  calls;
- off-centre visual rectangles remain unsupported;
- `CoCircularModularConnectorFunction` is unsupported on the retinotopic
  sheet because its present Euclidean-plane interpretation is scientifically
  ambiguous;
- `NauhausAnalysis` is unsupported for retinotopic sheets because it requires
  one scalar visual/cortical magnification;
- nonempty `local_module` configuration is corner-anchored in the new sheet,
  exactly as stored coordinate `(0, 0)` dictates;
- cortical stimulators and selectors receive corner-origin physical
  coordinates in eccentricity mode;
- no automatic orientation-map resizing or validation against an unknown
  source extent is possible; correctness of the supplied cortical map remains
  the user's responsibility;
- memory feasibility for a particular count, eccentricity range, and sampling
  requirement cannot be guaranteed in advance. Initialization must report the
  estimate before the expensive kernel-allocation loop, but this phase adds no
  arbitrary memory cutoff.

The following later scientific decisions are deliberately deferred:

- whether and how to recalibrate LGN luminance/contrast gains after measuring
  corrected new-mode currents;
- whether the angular seam needs periodic or anatomical treatment;
- whether eccentricity-dependent fan-in or incoming weight should eventually
  be normalized;
- which kernel/stimulus optimization should replace full 3D kernels;
- final protocol and tolerance for theoretical-versus-spiking preferred
  spatial frequency;
- whether the corrected luminance normalization should eventually become a
  separate general mode or replace legacy behavior in a future migration.

Any change to these deferred decisions must follow
[Design changes during implementation](#design-changes-during-implementation).
Dependent implementation must not begin until the accepted decision has been
incorporated into the normative sections of this document.
