Here is a comprehensive, deeply researched `README.md` that synthesizes the repository's architecture, requirements, and execution pipeline. You can copy and paste this directly into the root of your repository to replace the missing documentation.

***

# 🎬 SaemPhysics Video Automation Pipeline

**An autonomous, request-driven pipeline for generating high-quality educational physics and mathematics videos.**

This repository serves as a "virtual studio" for creating 3Blue1Brown and Feynman-style educational content. It bridges Large Language Models (LLMs) with deterministic physics/math engines and **Manim** (the mathematical animation engine) to turn a simple text prompt into a fully rendered, voiced, and mastered `.mp4` video.

---

## 🌟 Key Features

*   **Request-Driven Generation:** No hard-coded scripts. Provide a natural language request (e.g., *"Explain the Principle of Least Action using a Feynman analogy"*), and the pipeline autonomously plans the lesson.
*   **Deterministic Engines:** LLM hallucinations are prevented by routing calculations through verified `math_engine.py` and `physics_engine.py` before animation.
*   **High-Fidelity Neural TTS:** Utilizes **Coqui TTS (XTTS-v2)** for expressive, high-quality voiceovers, paired with `soundscape.py` for ambient background audio mastering (LUFS targeting).
*   **Automated FFmpeg Stitching:** Automatically synchronizes Manim renders with generated audio, applies audio limiting, and attaches a closing card.
*   **Pluggable AI Providers:** The LLM layer is decoupled. Use the included Google Gemini provider or write your own adapter for OpenAI, Anthropic, or local models.

---

## 🏗️ Architecture: The Pipeline

The production path follows a strict sequence to ensure educational accuracy and visual coherence:

1.  **Request Parsing:** Free-form text + JSON preferences (orientation, style) are passed to `request_pipeline.py`.
2.  **Knowledge Retrieval:** The system queries `knowledge_matrix.csv` and `problem_matrix.csv` for verified academic sources.
3.  **LLM Analysis:** The configured provider (e.g., Gemini) analyzes the request and proposes a sequence of `engine_calls`.
4.  **Engine Execution:** The `engine_router.py` executes validated math/physics operations to guarantee correct data.
5.  **Artifact Composition:** The LLM generates the final script, visual beats, and Manim scene instructions (`GeneratedPlan`).
6.  **Rendering & Stitching:** `batch_runner.py` synthesizes the voiceover, renders the Manim scenes via `render_universal.py`, and uses FFmpeg to mux the final video.

---

## ⚙️ System Prerequisites

Before installing Python dependencies, you **must** install these OS-level tools. Manim and TTS rely heavily on them.

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install -y ffmpeg texlive-full libcairo2-dev libpango1.0-dev pkg-config libsndfile1
```

**macOS:**
```bash
brew install ffmpeg cairo pango pkg-config libsndfile
# You will also need a LaTeX distribution (e.g., MacTeX or BasicTeX)
```

**Windows:**
It is highly recommended to use **WSL2** (Windows Subsystem for Linux) and follow the Linux instructions, as Manim and TeX Live are notoriously difficult to configure natively on Windows.

---

## 🚀 Installation

Create a virtual environment and install dependencies. **Note:** PyTorch must be installed from the CPU index to ensure Coqui TTS stability.

```bash
# 1. Clone and enter the repository
git clone https://github.com/saemphysics/youtube.git
cd youtube

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install PyTorch CPU-only (CRITICAL for TTS)
pip install torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# 4. Install project requirements
pip install -r requirements.txt
```

---

## 🔑 Configuration

This pipeline requires a Generation Provider to write scripts and plan visuals. The repository includes a Google Gemini adapter.

1. Obtain a **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/).
2. Set the required environment variables in your terminal:

```bash
export GEMINI_API_KEY="your_actual_api_key_here"
export SAEM_GENERATION_PROVIDER="gemini_provider:create_provider"
```

---

## 📖 Quick Start: Generating Your First Video

To create a video, you must run the pipeline in three sequential steps: **Plan**, **Adapt**, and **Render**.

### Step 1: Generate the Lesson Plan
Provide a detailed prompt to get the best "Feynman/3b1b" results.

```bash
python request_pipeline.py \
  --request "Explain the Principle of Least Action. Use a Feynman-style intuitive analogy of a thrown ball, followed by a step-by-step visual derivation of the Euler-Lagrange equation." \
  --preferences '{"orientation":"horizontal", "style":"Feynman lectures, minimalist Manim geometry, deep analytical voice"}' \
  --output generated/least-action.plan.json
```

### Step 2: Adapt and Validate
Convert the JSON plan into a CSV format that the batch renderer understands.

```bash
python output_pipeline.py generated/least-action.plan.json --csv-output generated/render.csv
```
*(Tip: Add `--dry-run` to validate the CSV without triggering the render).*

### Step 3: Render and Stitch
This command will trigger the Coqui TTS model (downloading XTTS-v2 on first run), render the Manim scenes, and stitch the audio/video via FFmpeg.

```bash
python batch_runner.py --csv generated/render.csv
```
*The final `.mp4` files will be available in the `output/` directory.*

---

## 🧩 Advanced: Creating Custom Providers

The LLM layer is entirely modular. If you want to use GPT-4, Claude, or a local Llama 3 model, you can write a custom provider.

Your provider module must expose a `create_provider()` factory that returns a class implementing two methods:
1.  `analyze(request, context, operations)`: Returns a structured list of engine calls and source references.
2.  `compose(request, context, analysis, engine_results)`: Returns a list of visual artifacts and scripts.

**Example Usage:**
```bash
export SAEM_GENERATION_PROVIDER="my_openai_module:create_provider"
python request_pipeline.py ...
```

---

## 📂 Project Structure

| File | Purpose |
| :--- | :--- |
| `request_pipeline.py` | Entry point. Orchestrates the LLM and Engine routing to generate the `.json` plan. |
| `gemini_provider.py` | Default Google Gemini adapter implementing the `analyze`/`compose` protocol. |
| `output_pipeline.py` | Validates the JSON plan and exports a renderer-adapter `.csv`. |
| `batch_runner.py` | Subprocess orchestrator. Handles TTS synthesis, Manim execution, and FFmpeg stitching. |
| `render_universal.py` | The universal Manim bridge that interprets JSON visual data into Python scenes. |
| `generate_audio.py` | Interface for Coqui XTTS-v2 neural voice generation. |
| `soundscape.py` | Audio mastering, LUFS normalization, and ambient pad generation. |
| `math_engine.py` / `physics_engine.py` | Deterministic calculators to verify LLM logic before animation. |

---

## 🐛 Troubleshooting & Known Issues

*   **`pypdf` Version Error:** If you encounter issues with `pypdf`, ensure you are using `>=4.2.0`. Version `6.21.1` was a typo in older package indexes and has been patched in `requirements.txt`.
*   **TTS Crashes / CUDA Errors:** If Coqui TTS crashes during audio generation, ensure you installed the **CPU-only** version of PyTorch as specified in the Installation steps. XTTS-v2 behaves unpredictably with mismatched CUDA/Torch versions.
*   **Missing LaTeX:** If Manim throws a `FileNotFoundError` for `.tex` files, your OS-level LaTeX installation (TeX Live / MacTeX) is incomplete or not in your system `PATH`.
*   **Request Length Limit:** The `request_pipeline.py` enforces a maximum character limit (`MAX_REQUEST_CHARS`) for prompts. Break extremely long syllabus requests into smaller, focused topics.
