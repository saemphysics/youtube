# Request-driven educational media pipeline

The production path starts with an unrestricted user request. No lesson ID,
topic-to-script mapping, authored narration bank, or preselected sound/style
template chooses the output.

## Processing model

```text
free-form request + optional preferences
    -> request_pipeline.py
    -> knowledge/source/problem retrieval
    -> provider analysis and proposed engine calls
    -> engine_router.py evaluates validated math/physics operations
    -> provider composes original artifacts from the results
    -> generated plan JSON
    -> output_pipeline.py validates and adapts each artifact
    -> batch_runner.py -> Manim renderer + selected audio backend
```

The model provider is a plug-in rather than a hard-coded service. Implement a
Python module exposing `create_provider()`, whose provider has `analyze()` and
`compose()` methods implementing the protocol in `request_pipeline.py`.
Configure it as `SAEM_GENERATION_PROVIDER=package.module:create_provider`.
Configure credentials only in that provider's environment. If no provider is
configured, generation stops with an explicit error; it never falls back to
authored example scripts.

```sh
export SAEM_GENERATION_PROVIDER=my_provider:create_provider
python request_pipeline.py \
  --request "Explain why a narrow pipe changes water speed, with a numerical example." \
  --preferences '{"orientation":"vertical","audio_direction":"calm, no music"}' \
  --output generated/pipe-flow.plan.json
python output_pipeline.py generated/pipe-flow.plan.json --csv-output generated/pipe-flow.csv --dry-run
python output_pipeline.py generated/pipe-flow.plan.json --csv-output generated/pipe-flow.csv
```

The request is free text. Preferences are an extensible JSON object, not a
catalog of lesson or narration choices. The provider returns its own intent,
engine calls, narration, equations, visuals, storyboard beats, voice/audio
direction, visual direction, language, orientation, quality, and platform.
Each output is distinct and remains available as a plan JSON artifact.

## Matrices as evidence, not scripts

- `knowledge_matrix.csv` relates concepts to source references and registered
  mathematical/physical engines. It has no lesson IDs or prewritten output
  assignments.
- `problem_matrix.csv` holds reviewed, source-linked problem summaries and
  solution checks for retrieval. These records are evidence for a newly
  composed explanation; they are not selected as finished output episodes.
- `book_sources.csv` records provenance and review limits. Title-level mappings
  to unreviewed sources can route a request to relevant material, but those
  sources cannot support a verified problem claim. Verified problems require
  page-level review and a readable source.
- For a matched, text-readable PDF, the request context includes a small number
  of query-matched excerpts with page numbers and review status. Excerpts are
  generated at runtime, capped in length, and are not stored as a text corpus.
- Matched concepts and problems are supplied as context. No match is not
  replaced by a default concept, script, solver, or style.
- The five recently added PDFs contribute retrieval context, not a trained
  model. Four have extractable text; the parallel-hardware PDF does not and
  needs OCR before it can provide passages. Their ledger entries disclose
  review limits, and none of these new sources is eligible as a verified
  problem until reviewed.

The local Feynman computation book's contents cover instruction sets, finite
state machines, computability, information, and reversible computation. This
architecture applies those general ideas as explicit stages, inspectable data,
and bounded transformations. It does not copy the book's text.

Strang's *Computational Science and Engineering I* presents matrices as
operators on meaningful quantities, connects structure to efficient numerical
methods, and emphasizes interpreting the computed result. The calculus
references build from functions and rates of change. Accordingly, proposed
operations go through the registered calculus/physics engines; results are
passed back to the provider before it writes the explanation. Calculations
accept data and a restricted arithmetic-expression language, never executable
provider code.

## Python modules

| Module | Responsibility |
|---|---|
| `request_pipeline.py` | Free-text request, provider protocol, matrix retrieval, orchestration, and plan JSON |
| `matrix_engine.py` | Source-aware concept retrieval and engine-evidence ranking |
| `problem_matrix.py` | Validation and retrieval of reviewed problem evidence |
| `engine_router.py` | Discover/execute registered operations; safe expressions; JSON-safe results |
| `output_pipeline.py` | Validate generated artifacts, adapt them to the existing renderer/audio batch interface, and invoke the render backend |
| `physics_engine.py`, `math_engine.py` | Numerical methods and registered domain engines |
| `render_universal.py`, `generate_audio.py`, `batch_runner.py` | Existing media/render backends behind the generated-plan adapter |

Generated CSV is a transient renderer adapter, not the authoring format or
source of truth. The plan JSON retains dynamic audio and visual directions
alongside render fields. Voice identity is selected by the request/provider
using a permitted reference file; no repository voice sample is silently
chosen. Where the current synthesis backend cannot implement a requested
audio feature, surface that limitation rather than substituting a preset.

## Design limits and extension

The repository supplies deterministic numerical engines and render primitives,
not a natural-language model. Open-ended interpretation and original
explanation therefore require an installed provider plug-in. The provider
must use source notes with their review limits, ground numerical claims in
returned engine results or cited verified evidence, and request only supported
operations and visual primitives. Unsupported requests should be identified,
not answered with a canned fallback.

To extend the system, add a reviewed knowledge/source mapping, a tested engine
operation, a renderer primitive, or a provider/audio adapter. Avoid adding
topic-specific narration banks or creator-specific signature scenes. All
visual and audio directions are generated for the request; the system aims for
clear, original mathematical storytelling without imitating any creator's
distinctive style.
