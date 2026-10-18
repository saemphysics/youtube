# Paired lecture and short CSV architecture

## Editorial source of truth

`lecture_series_plan.csv` is the curriculum and story-design file. Each of its
10 rows is one lesson, not one video. `lesson_id` is the stable join key for
the two video versions. Each lesson intentionally combines:

1. A question that creates curiosity.
2. A first-principles model and a connection to neighboring concepts.
3. One worked numerical problem with a checked answer.
4. A misconception to correct.
5. A visual progression that makes the model observable.

Keep equations and worked answers dimensionally consistent. Edit the plan
before updating the authored, paired narration and visual definitions in
`generate_lecture_series.py`, so the lecture and its Short teach the same core
idea and use the same numerical example.

## Render-ready exports

Generate two separate files from the plan and authored scripts:

```sh
python generate_lecture_series.py
```

| Output | Rows | Per-row orientation | Runtime target | Frame |
|---|---:|---|---:|---|
| `content_batch_horizontal.csv` | 10 | `horizontal` | 180 seconds | 16:9 |
| `content_batch_vertical.csv` | 10 | `vertical` | 60 seconds | 9:16 |

Each output uses the existing `content_batch.csv` render schema. The
`video_id` must be unique per export, for example `mix01_horizontal` and
`mix01_vertical`. Include `lesson_id` as an extra column in both files so the
pair can be joined; the batch runner ignores extra columns. Keep one output
row per video so either orientation can be rendered independently.

Map the editorial plan to render columns as follows:

| Plan data | Render CSV field |
|---|---|
| `lesson_id`, sequence | Extra metadata columns |
| `lesson_title` | `header_title` |
| Central question | `tagline` or `question_text` |
| `concept_stack`, `first_principles_model` | `audio_script`; concise equation labels in `equations_json` |
| `worked_problem`, `worked_solution` | `question_text`, `audio_script`; for quiz rows also populate all four options and `correct_answer` |
| `visual_story` | `visual_data_json`, `beats_json`, `bindings_json` |
| Format | `orientation`, `platform`, and unique format-specific `video_id` |
| Production language and resolution | `language`, `quality` |

Use the exact columns in the current CSV header for renderer compatibility.
JSON-valued fields must be serialized JSON arrays, not Python representations.
Keep visual primitives in `schema_validator.KNOWN_PRIMITIVES` and keep positions
inside the selected aspect-ratio frame.

## Story timing and script budget

These targets assume clear narration around 130-145 words per minute. They are
guides, not a reason to pad a script; leave room for visual pauses and equation
reads.

### Horizontal lecture — about 3 minutes

- 0:00-0:15: surprising observation and the question.
- 0:15-0:35: intuition, physical setup, and prediction.
- 0:35-1:15: build the model and connect concepts.
- 1:15-2:10: work the numerical problem on screen, including units.
- 2:10-2:40: test the result, correct the misconception, and show a limit.
- 2:40-2:55: concise synthesis and transfer question.
- 2:55-3:00: quiet visual resolution / end card.
- Aim for roughly 350-410 spoken words (the authored batch targets 350-395);
  omit an extended intro and avoid
  repeating the same takeaway in several different phrasings.

Use the horizontal frame for comparisons, coordinate systems, full derivations,
and a legible worked solution. Give each beat enough time for the viewer to
read labels and predict the next step.

### Vertical Short — about 1 minute

- 0:00-0:04: concrete puzzle or surprising result.
- 0:04-0:12: one intuitive picture and one governing relationship.
- 0:12-0:42: one worked step-by-step example with units.
- 0:42-0:54: result, misconception correction, and one-sentence takeaway.
- 0:54-1:00: visual hold / end card.
- Aim for roughly 120-145 spoken words. Preserve the same lesson ID and
  numerical example, but do not shrink the lecture script mechanically.

Use one dominant object or diagram, large labels, and a single visual change at
a time. Keep equations short enough to read on a phone.

## Cross-format content rules

- Both rows for a lesson answer the same central question and share the core
  equation, assumptions, and worked-example result.
- The Short is a complete mini-explanation, not a trailer or an excerpt that
  depends on the lecture.
- The lecture adds derivation, a second conceptual connection, assumptions,
  and a transfer question rather than simply repeating the Short.
- Every numerical example states units and uses explicit assumptions.
- Mix concepts through the causal chain in the lesson, not by listing unrelated
  chapter names. Each episode has one central question and one worked problem.
- Aim for intuitive, original, visually precise teaching and conversational
  first-principles explanations. Do not reproduce any creator's scripts,
  signature phrasing, or distinctive scene designs.

## Production sequence

1. Review `lecture_series_plan.csv` and verify each calculation.
2. Keep one horizontal and one vertical script for every `lesson_id` in
   `generate_lecture_series.py`; the paired scripts are written independently.
3. Run the exporter to produce exactly 10 rows in each render-ready CSV,
   preserving their separate orientation values and unique IDs.
4. Run `python schema_validator.py content_batch_horizontal.csv` and
   `python schema_validator.py content_batch_vertical.csv`.
5. Copy the desired export to `content_batch.csv` before running
   `python batch_runner.py --dry-run`; then render with the desired quality.
6. Check generated durations and audio pacing before the full-quality render.
   The batch runner adds a 5-second closing card after voice duration, so
   narration word count is only an estimate; confirm the final MP4 duration
   after synthesis and adjust the script or TTS speed if needed.

## Matrix-guided requested batches

`book_sources.csv` records which local references could be inspected and
their review scope. `knowledge_matrix.csv` links original concept labels and
keywords to those sources, the authored lesson IDs, and every domain in
`EngineCatalog`. The matrix is a curated index, not a full-text copy of the
books or a claim that every topic has a working solver.

Request an original, already-authored lesson by topic:

```sh
python generate_requested_csv.py --topic "eigenvectors" --orientation both --seed 2026
python schema_validator.py content_batch_vertical.csv
```

The generator matches topic words to reviewed matrix vocabulary and samples
only among linked, authored lessons. `--seed` makes that choice repeatable.
The normalized match weights are recommendation weights, not calibrated
probabilities; this index is not a trained language model and does not invent
scripts or silently run every solver. If a concept has no authored lesson, the
request fails and needs a new reviewed lesson first. Outputs retain this
project's render-ready CSV schema; select the intended orientation and copy it
to `content_batch.csv` before rendering, as in the production sequence above.

The matrix source ledger flags the supplied Volume I DOCX as incomplete (2
bytes); no knowledge is attributed to that file. Matrix links describe
high-level concept relationships using source headings and course-level
references, not copied book text. The local Strang Calculus PDF identifies the
Third Edition; its chapter 12.3 and 12.4 headings support the reviewed
curvature, circular-motion, and polar-coordinate links. The incomplete
Feynman Volume I file is not used to claim mechanics coverage.

## Exam-problem matrix and explanation batches

`problem_matrix.csv` indexes individually reviewed Physics questions by
source, printed question number, PDF page, and `knowledge_matrix.csv` concept
ID. It stores paraphrased prompts and checked solution summaries rather than
copied exam text. `problem_matrix.py` rejects stale source/concept links, and
`generate_problem_csv.py` emits render-ready explanation rows in the existing
schema:

```sh
python generate_problem_csv.py --orientation both
python schema_validator.py problem_batch_vertical.csv
python schema_validator.py problem_batch_horizontal.csv
```

The curated set currently includes verified circular-motion, instantaneous
power, lens-power, and magnetic-flux problems. The two optics/electromagnetism
rows link to reviewed Feynman concept sources through the knowledge matrix;
the calculus relationships link to Strang Calculus where applicable. PDFs
that are image-only, have an unreliable text layer, or have not yet had their
Physics sections reviewed remain catalogued but are not used for generated
content. Video scripts and visualizations are original and use this project's
primitives; they are not copies of any creator's script or signature scenes.
