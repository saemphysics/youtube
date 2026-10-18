# Getting the most from the request-driven video pipeline

This pipeline is designed for open-ended requests: describe what you want a
viewer to understand, and the configured generation provider proposes the
explanation, calculations, storyboard, visuals, and audio direction. There are
no required lesson IDs, fixed scripts, or preset narration/style selections.

## Before generating

The repository supplies source/problem retrieval, verified math and physics
engines, renderer primitives, and the output adapter. It does not include a
natural-language generation model. Configure a compatible provider plug-in
before requesting a video:

```sh
export SAEM_GENERATION_PROVIDER=my_provider:create_provider
```

The module and factory name above are examples. Your provider must implement
`analyze()` and `compose()` as described in
[REQUEST_ARCHITECTURE.md](./REQUEST_ARCHITECTURE.md). Configure any provider
credentials in your local environment; do not put secrets in the request,
repository, or preferences JSON.

## Writing a useful request

Write naturally. The request can be a question, a desired learning outcome, a
problem to solve, or a combination. Useful details include:

- **Goal:** What should the viewer understand or be able to do afterward?
- **Audience:** What should be assumed, and what needs a brief introduction?
- **Scope:** Which ideas are essential, and what should be left out?
- **Reasoning:** Ask for definitions, assumptions, units, intermediate steps,
  checks, or comparisons when they matter.
- **Visual reasoning:** Describe what viewers should see change, compare, or
  measure. This is guidance, not a demand to use a particular preset scene.
- **Evidence:** Name a source or problem when relevant. Ask for page-aware
  support and a clear distinction between verified evidence and explanation.
- **Output intent:** Say whether you want one focused video or several
  complementary artifacts, and describe the desired orientation, language, or
  audio character in ordinary language.

For example, instead of a topic label such as “calculus,” give the pipeline a
goal and context:

> Make an intuitive explanation of why the derivative of position is velocity.
> Start with a moving object, compare average speed over a long interval with
> speed over shrinking intervals, then connect the limiting idea to the slope
> of a position-time graph. Use units, one simple numerical example, and avoid
> assuming the viewer already knows limit notation.

Other requests can focus on a different kind of learning:

> Explain how a matrix transforms a vector in two dimensions. Show the original
> basis vectors and vector, apply a concrete matrix, and interpret what happens
> to lengths, directions, and area. Include a small numerical example and
> distinguish a rotation from a general linear transformation.

> Create a source-grounded explanation of a reviewed projectile-motion problem.
> State the assumptions, show the component equations and units, check the
> final result, and make clear which details come from the source versus the
> newly composed explanation.

> Explain interference by comparing two waves before and after their phases
> align. Show where the resulting amplitude increases or cancels, and explain
> the result without treating a diagram as proof.

These are examples of free-form input, not required templates. Change the
order, wording, detail, language, or subject to suit your goal.

## Optional preferences

The CLI accepts a JSON object for extra constraints. It is optional; use it
only for concise machine-readable preferences. You can also put preferences
directly in the natural-language request.

```sh
python request_pipeline.py \
  --request "Explain why a narrow pipe changes water speed. Use continuity and a numerical example, state assumptions, and show the area and speed changing together." \
  --preferences '{"orientation":"horizontal","language":"en","visual_direction":"Use uncluttered diagrams with readable labels.","audio":{"direction":"clear and conversational","ambient":{"kind":"none"}}}' \
  --output generated/pipe-flow.plan.json
```

Preferences are not a fixed menu and are passed to the provider as user
context. The provider and render backends still have to support the requested
feature. The current sound-bed synthesizer supports `none`, seeded filtered
`noise`, and `tones` plans; it does not synthesize arbitrary instruments or
music from descriptive words. A voice reference may be requested only if the
provider/audio backend is configured to use that available file. Ask for an
audio direction, but do not assume every style or voice request is technically
available.

## Generate, inspect, and render

First create a plan. Then validate it through the renderer adapter with a
dry-run before rendering:

```sh
python request_pipeline.py \
  --request "Explain how resonance builds in a driven oscillator. Show the driving frequency, natural frequency, and amplitude response, and explain what damping changes." \
  --output generated/resonance.plan.json

python output_pipeline.py generated/resonance.plan.json --dry-run
python output_pipeline.py generated/resonance.plan.json
```

The generated plan JSON is the durable, inspectable output. The CSV used by the
existing renderer is a temporary adapter format, not an authoring template.
Inspect the plan and dry-run output before spending time on a full render.
Use `--csv-output path/to/adapter.csv` on `output_pipeline.py` if you want to
keep a copy of the adapter for troubleshooting.

## Getting better results through iteration

Treat the first plan as a draft to review, not a guaranteed final video:

1. Check that the provider understood the learning goal and audience.
2. Check every numerical claim against the returned engine results, units,
   assumptions, and any cited evidence.
3. Check that visuals are supported by the renderer and that the storyboard
   explains rather than merely decorates the calculation.
4. If something is missing, make a follow-up request more specific: identify
   the gap and the change you want, while preserving what already works.
5. Regenerate and dry-run before rendering again.

For a multi-part learning sequence, ask for complementary outputs with
different purposes (for example, intuition first and a worked example next)
rather than asking for repeated versions of the same explanation. Whether one
request produces multiple artifacts depends on the provider.

## Sources, problems, and calculation limits

The knowledge and problem matrices guide retrieval; they do not select a
finished lesson or script. Ask for source-grounded work when the distinction
matters. A source may be present for topic discovery but still be marked
unreviewed; uploaded PDFs are not automatically verified or used to train a
model. Image-only and otherwise unreadable PDFs cannot provide reliable
textual passages until reviewed or OCR'd.

The numerical engines perform registered deterministic calculations. They do
not independently prove that a model, assumption, or equation is appropriate
for a real-world situation. State relevant conditions in the request (such as
neglecting air resistance, using a constant acceleration, or assuming
incompressible flow), and ask the provider to explain them. Unsupported
operations or renderer features should be surfaced rather than silently
replaced with a preset.

The pipeline does not include a built-in provider. If none is configured,
generation stops with an explicit error; there is no canned narration or
success-shaped fallback.

## Generate a ten-video YouTube batch

Use `generate_video_batch.py` when you want a set of videos from your own
physics requests. Create a UTF-8 text file with exactly ten non-empty lines,
one complete free-form request per line. Include both concept explanations
and worked-problem requests in the mix you want. The first five lines become
horizontal 3-minute videos; the last five become vertical 1-minute videos.
The renderer adds a five-second closing card, so narration is planned for
2:55 in each horizontal video and 0:55 in each vertical video.
The batch tool does not invent or substitute topics for you.

Example request-file shape (replace each line with your own request):

```text
Explain the physics concept or problem I want for video 1, including the audience and goal.
Explain the physics concept or problem I want for video 2, including assumptions and units.
Explain the physics concept or problem I want for video 3, including a visual reasoning idea.
Explain the physics concept or problem I want for video 4, including what the viewer should learn.
Explain the physics concept or problem I want for video 5, including a worked example if useful.
Explain the physics concept or problem I want for video 6 in a focused one-minute format.
Explain the physics concept or problem I want for video 7 in a focused one-minute format.
Explain the physics concept or problem I want for video 8 in a focused one-minute format.
Explain the physics concept or problem I want for video 9 in a focused one-minute format.
Explain the physics concept or problem I want for video 10 in a focused one-minute format.
```

Run plan generation and validate the renderer CSV without rendering videos:

```sh
python generate_video_batch.py \
  --input my_physics_requests.txt \
  --output-dir generated/my-physics-series \
  --dry-run
```

This still calls your configured provider ten times to generate and validate
the plans. It writes one inspectable JSON plan per request under
`generated/my-physics-series/plans/`, plus a combined renderer CSV. Review
those plans and the dry-run output before rendering. To render the batch:

```sh
python generate_video_batch.py \
  --input my_physics_requests.txt \
  --output-dir generated/my-physics-series \
  --render
```

Rendering requires the repository's media dependencies, including
`espeak-ng`, `ffmpeg`, and the configured Manim environment. The batch uses
the offline eSpeak backend to fit narration to the requested duration; if the
narration is much too short or too long for that timing, synthesis reports an
error rather than silently changing the target. You can regenerate after
asking for narration paced to the requested duration. The five landscape and
five portrait format constraints are checked against every provider-generated
plan before the CSV is written.

The batch's editorial direction asks for original first-principles
explanations, intuitive physical analogies, and elegant, readable visuals.
It does not reproduce a named educator's distinctive style, voice, or
signature scenes. Make each request your own: specify the concept/problem,
audience, assumptions, numerical checks, language, and what the viewer
should understand.
