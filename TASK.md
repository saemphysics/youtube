# TASK.md: Universal Physics Video Renderer

## 🌌 Overview
This repository is an automated, AI-driven media pipeline designed to generate high-quality, educational physics and mathematics videos. It functions as a "virtual studio," combining Large Language Models (LLMs) for scriptwriting with deterministic math engines for accuracy, and cinematic rendering (Manim) for visuals. 

It is specifically engineered to produce content in the style of **Feynman** (intuitive, analogy-first explanations) and **3Blue1Brown** (smooth, geometric visual proofs).


# TASK.md: Universal Physics Video Renderer

## 🌌 Overview
This repository is an automated, AI-driven media pipeline designed to generate high-quality, educational physics and mathematics videos. It functions as a "virtual studio," combining Large Language Models (LLMs) for scriptwriting with deterministic math engines for accuracy, and cinematic rendering (Manim) for visuals. 

It is specifically engineered to produce content in the style of **Feynman** (intuitive, analogy-first explanations) and **3Blue1Brown** (smooth, geometric visual proofs).

---

## 🚀 Core Capabilities

1. **Deterministic Physics & Math Engines:** 
   The system does not "guess" equations. It uses internal engines (`math_engine.py`, `physics_engine.py`) to calculate trajectories, wave interference, vector fields, and kinematics. The LLM proposes the operations, but the deterministic engines execute and validate them.
2. **Knowledge Matrix Retrieval:** 
   Cross-references prompts against verified textbooks and exam problems (`knowledge_matrix.csv`, `problem_matrix.csv`, `book_sources.pdf`) to ensure pedagogical accuracy.
3. **Cinematic Manim Rendering:** 
   Supports both 16:9 (Horizontal/YouTube) and 9:16 (Vertical/Shorts) orientations. Includes 50+ custom visual primitives (e.g., `wave_packet_tunneling`, `pendulum`, `vector_field`, `bohr_orbit_photon`).
4. **Neural Voice Synthesis (TTS):** 
   Uses Coqui XTTS-v2 for high-fidelity voice cloning based on a reference audio sample, complete with automatically generated ambient soundscapes (`soundscape.py`).
5. **Automated Audio/Video Stitching:** 
   Automatically syncs voiceover pacing with Manim scene durations and applies broadcast-standard audio limiting via FFmpeg.

---

## 🏗️ System Architecture

The pipeline is split into two distinct phases to optimize compute resources:

### Phase 1: The Brains (Lightweight)
- **`request_pipeline.py`**: Ingests the prompt, queries the knowledge matrix, executes math engines, and asks the LLM to generate a structured `plan.json`.
- **`output_pipeline.py`**: Validates the `plan.json` against the schema and converts it into a renderer-adapter `render.csv`.
- *Dependencies:* `numpy`, `scipy`, `pypdf`, `google-genai`. (No GPU or heavy system libraries required).

### Phase 2: The Muscle (Heavy)
- **`batch_runner.py`**: Reads the CSV row-by-row.
- **`generate_audio.py`**: Synthesizes the voiceover and ambient pad.
- **`render_universal.py`**: Executes the Manim scenes to generate raw MP4s.
- **FFmpeg**: Stitches the audio and video together, adding a 5-second closing card.
- *Dependencies:* `manim`, `torch`, `TTS`, `texlive`, `libcairo2-dev`, `ffmpeg`.

---

## 📥 Inputs & 📤 Outputs

### Inputs
| File / Variable | Purpose |
| :--- | :--- |
| **`request.txt`** | A plain text file containing **one video prompt per line**. (No headers, no blank lines). |
| **`gemini_provider.py`** | The LLM integration module. Requires `GEMINI_API_KEY`. |
| **`local_provider.py`** | An offline, deterministic stub for testing the pipeline without API costs. |
| **`knowledge_matrix.csv`** | The database of physics concepts, keywords, and engine weights. |
| **`saem_voice_sample.wav`** | (Optional) A 6-10 second WAV file used by Coqui TTS to clone the narrator's voice. |

### Outputs
| Path | Purpose |
| :--- | :--- |
| **`generated/plan-*.json`** | The raw LLM/Engine generated script, visual beats, and math data. |
| **`generated/render-*.csv`** | The validated adapter CSV that the Manim renderer understands. |
| **`output_shorts/*_final.mp4`** | The final, fully stitched video ready for upload. |

---

## 🛠️ How to Use (Tasks & Workflows)

### Task 1: The Offline Smoke Test (No API Key Required)
Use this to verify your local environment, math engines, and Manim installation are working before burning API tokens or GPU time.

```bash
# 1. Set the provider to the local offline stub
export SAEM_GENERATION_PROVIDER="local_provider:create_provider"

# 2. Generate a dummy plan
python request_pipeline.py \
  --request "Test the renderer offline." \
  --output generated/test.plan.json

# 3. Validate and convert to CSV (dry-run skips actual rendering)
python output_pipeline.py generated/test.plan.json \
  --csv-output generated/test.csv --dry-run



------------------------------



# 1. Set your API key and provider
export GEMINI_API_KEY="your_api_key_here"
export SAEM_GENERATION_PROVIDER="gemini_provider:create_provider"

# 2. Generate the plan
python request_pipeline.py \
  --request "Explain the Principle of Least Action using a Feynman-style analogy of a hiker finding the easiest path, followed by the Euler-Lagrange derivation." \
  --preferences '{"orientation":"horizontal", "quality":"1080p"}' \
  --output generated/least_action.plan.json

# 3. Convert to CSV
python output_pipeline.py generated/least_action.plan.json \
  --csv-output generated/least_action.csv

# 4. Render and Stitch (Use --draft for a fast 480p preview)
python batch_runner.py --csv generated/least_action.csv --draft



-------------------------------



   Explain Fourier series using rotating epicycles.
   Derive the Navier-Stokes equations for a 2D incompressible fluid.
   Visualize Quantum Superposition using Feynman path integrals.




---

## 🚀 Core Capabilities

1. **Deterministic Physics & Math Engines:** 
   The system does not "guess" equations. It uses internal engines (`math_engine.py`, `physics_engine.py`) to calculate trajectories, wave interference, vector fields, and kinematics. The LLM proposes the operations, but the deterministic engines execute and validate them.
2. **Knowledge Matrix Retrieval:** 
   Cross-references prompts against verified textbooks and exam problems (`knowledge_matrix.csv`, `problem_matrix.csv`, `book_sources.pdf`) to ensure pedagogical accuracy.
3. **Cinematic Manim Rendering:** 
   Supports both 16:9 (Horizontal/YouTube) and 9:16 (Vertical/Shorts) orientations. Includes 50+ custom visual primitives (e.g., `wave_packet_tunneling`, `pendulum`, `vector_field`, `bohr_orbit_photon`).
4. **Neural Voice Synthesis (TTS):** 
   Uses Coqui XTTS-v2 for high-fidelity voice cloning based on a reference audio sample, complete with automatically generated ambient soundscapes (`soundscape.py`).
5. **Automated Audio/Video Stitching:** 
   Automatically syncs voiceover pacing with Manim scene durations and applies broadcast-standard audio limiting via FFmpeg.

---

## 🏗️ System Architecture

The pipeline is split into two distinct phases to optimize compute resources:

### Phase 1: The Brains (Lightweight)
- **`request_pipeline.py`**: Ingests the prompt, queries the knowledge matrix, executes math engines, and asks the LLM to generate a structured `plan.json`.
- **`output_pipeline.py`**: Validates the `plan.json` against the schema and converts it into a renderer-adapter `render.csv`.
- *Dependencies:* `numpy`, `scipy`, `pypdf`, `google-genai`. (No GPU or heavy system libraries required).

### Phase 2: The Muscle (Heavy)
- **`batch_runner.py`**: Reads the CSV row-by-row.
- **`generate_audio.py`**: Synthesizes the voiceover and ambient pad.
- **`render_universal.py`**: Executes the Manim scenes to generate raw MP4s.
- **FFmpeg**: Stitches the audio and video together, adding a 5-second closing card.
- *Dependencies:* `manim`, `torch`, `TTS`, `texlive`, `libcairo2-dev`, `ffmpeg`.

---

## 📥 Inputs & 📤 Outputs

### Inputs
| File / Variable | Purpose |
| :--- | :--- |
| **`request.txt`** | A plain text file containing **one video prompt per line**. (No headers, no blank lines). |
| **`gemini_provider.py`** | The LLM integration module. Requires `GEMINI_API_KEY`. |
| **`local_provider.py`** | An offline, deterministic stub for testing the pipeline without API costs. |
| **`knowledge_matrix.csv`** | The database of physics concepts, keywords, and engine weights. |
| **`saem_voice_sample.wav`** | (Optional) A 6-10 second WAV file used by Coqui TTS to clone the narrator's voice. |

### Outputs
| Path | Purpose |
| :--- | :--- |
| **`generated/plan-*.json`** | The raw LLM/Engine generated script, visual beats, and math data. |
| **`generated/render-*.csv`** | The validated adapter CSV that the Manim renderer understands. |
| **`output_shorts/*_final.mp4`** | The final, fully stitched video ready for upload. |

---

## 🛠️ How to Use (Tasks & Workflows)

### Task 1: The Offline Smoke Test (No API Key Required)
Use this to verify your local environment, math engines, and Manim installation are working before burning API tokens or GPU time.

```bash
# 1. Set the provider to the local offline stub
export SAEM_GENERATION_PROVIDER="local_provider:create_provider"

# 2. Generate a dummy plan
python request_pipeline.py \
  --request "Test the renderer offline." \
  --output generated/test.plan.json

# 3. Validate and convert to CSV (dry-run skips actual rendering)
python output_pipeline.py generated/test.plan.json \
  --csv-output generated/test.csv --dry-run
```

### Task 2: Single Video Generation (Local Terminal / Codespaces)
Use this to generate a specific, high-quality video using Gemini.

```bash
# 1. Set your API key and provider
export GEMINI_API_KEY="your_api_key_here"
export SAEM_GENERATION_PROVIDER="gemini_provider:create_provider"

# 2. Generate the plan
python request_pipeline.py \
  --request "Explain the Principle of Least Action using a Feynman-style analogy of a hiker finding the easiest path, followed by the Euler-Lagrange derivation." \
  --preferences '{"orientation":"horizontal", "quality":"1080p"}' \
  --output generated/least_action.plan.json

# 3. Convert to CSV
python output_pipeline.py generated/least_action.plan.json \
  --csv-output generated/least_action.csv

# 4. Render and Stitch (Use --draft for a fast 480p preview)
python batch_runner.py --csv generated/least_action.csv --draft
```

### Task 3: Batch Production via GitHub Actions (Cloud Rendering)
This is the recommended workflow for rendering multiple videos without tying up your local machine.

1. **Edit `request.txt`**: Add up to 10 prompts, one per line.
   ```text
   Explain Fourier series using rotating epicycles.
   Derive the Navier-Stokes equations for a 2D incompressible fluid.
   Visualize Quantum Superposition using Feynman path integrals.
   ```
2. **Commit and Push**: Push `request.txt` to the `main` branch.
3. **Monitor Actions**: Go to the **Actions** tab in GitHub. The workflow will automatically:
   - Split `request.txt` into parallel jobs.
   - Generate the JSON plans (Phase 1).
   - Install heavy system dependencies (TeX, Cairo, FFmpeg).
   - Render the videos and upload them as Artifacts (Phase 2).
4. **Download**: Click into the completed workflow run and download the `video-*` artifacts.

---

## 🎨 Prompt Engineering Guide (Feynman & 3b1b Styles)

To get the best results from the LLM, structure your prompts in `request.txt` to explicitly trigger the system's capabilities:

* **The Feynman Style (Analogy First):**
  > *"Explain [Concept] using a simple, everyday analogy first (e.g., water flowing through pipes for electrical current). Then transition to the rigorous mathematical law."*
* **The 3b1b Style (Visual Morphing):**
  > *"Use minimalist geometry. Explicitly describe how the visual elements morph (e.g., 'Watch how the derivative vector aligns with the slope of the curve as it moves along the x-axis')." Use domain '[domain]' and visual primitive '[primitive]'.*
* **Forcing Engine Execution:**
  > *"Use domain 'mechanics' and operation 'projectile_trajectory' to calculate the exact apex of the motion before showing the visual."*

### Allowed Visual Primitives (Reference)
When writing prompts, you can request these specific primitives (defined in `schema_validator.py`):
`vector_field`, `pendulum`, `projectile_motion`, `wave_packet_tunneling`, `bohr_orbit_photon`, `accretion_disk`, `pipe_with_gauge`, `two_source_waves`, `pv_diagram_tracker`, `compass_grid`.

---

## 🧑‍🏫 Human-in-the-Loop (Teacher Approval)

Because the LLM generates the narration and analogies, you may want a human to review the script before rendering.

**Local Review:**
Open `generated/plan-*.json` in any text editor. You can safely edit the `"narration"` text, reorder the `"beats"` array, or fix equations. Once saved, run `output_pipeline.py` to validate your edits, then proceed to `batch_runner.py`.

**GitHub Actions Review:**
The workflow includes a `require-approval` toggle. If enabled in the GitHub UI, the pipeline will pause at the **"teacher-approval"** Environment gate. A repository admin must manually click "Approve" in the Actions UI before the heavy Manim rendering begins.
```
