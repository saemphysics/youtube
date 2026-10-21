"""Gemini generation provider for the request-driven pipeline."""
import json
import os
import re

from google import genai
from google.genai import types

# ============================================================================
# STRICT INSTRUCTION CONSTANTS
# These ensure the LLM output perfectly matches request_pipeline.py validation
# ============================================================================
ANALYZE_INSTRUCTIONS = (
    "Return ONE JSON object with exactly: "
    '"intent" (non-empty sentence), '
    '"concept_ids" (only IDs present in matched_concepts), '
    '"source_refs" (only IDs present in source_notes), '
    '"engine_calls" (list of {call_id, domain, operation, parameters} '
    "with unique non-empty call_id and JSON-object parameters). Never return a list."
)

COMPOSE_INSTRUCTIONS = (
    "Return a JSON list of artifact objects. Each MUST contain non-empty: "
    "artifact_id, title, narration, language, quality, platform; "
    'orientation exactly "horizontal" or "vertical"; '
    'visuals: non-empty list of {"type": <primitive from the provided list>, ...}; '
    'beats: non-empty list of {"type": one of hook/promise/analogy/law/emphasis/'
    "experiment/midpoint/punchline, \"narration\": ...}; "
    "equations: list of non-empty strings; bindings: list of objects; "
    "audio: object with script/voice_reference/ambient/speed; "
    "plus tagline, question, visual_direction."
)


def _extract_json(text: str):
    """Extract JSON from Gemini's response, handling markdown fences."""
    if not text or not text.strip():
        raise ValueError("Gemini returned an empty response (safety filter or quota).")
    text = text.strip()
    
    # Try to extract from markdown code fence
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    
    # If still not JSON, find the first { or [
    if text and text[0] not in "{[":
        start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
        if start == -1:
            raise ValueError(f"Gemini response contains no JSON: {text[:200]!r}")
        text = text[start:]
    
    return json.loads(text)


class GeminiProvider:
    """Provider implementation using Google Gemini."""
    
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.0-flash-exp"
        self._json_config = types.GenerateContentConfig(
            temperature=0.65,
            response_mime_type="application/json",
        )
        self._pending_artifacts = None

    def analyze(self, request, knowledge, engine_operations):
        """Analyze the request and propose engine operations."""
        prompt = (
            f"You are the ANALYSIS stage of a physics video generation pipeline.\n"
            f"{ANALYZE_INSTRUCTIONS}\n"
            f"Available engine operations: {json.dumps(engine_operations)}\n"
            f"Request: {request.text}\n"
            f"Knowledge context: {json.dumps(knowledge, default=str)[:2000]}"
        )
        
        response = self.client.models.generate_content(
            model=self.model_name, 
            contents=prompt, 
            config=self._json_config,
        )
        
        data = _extract_json(response.text)
        
        # Fallback: If model returned a list instead of dict, cache it and return empty analysis
        if isinstance(data, list):
            self._pending_artifacts = data
            return {
                "intent": f"Demonstration of: {request.text.strip()[:100]}",
                "concept_ids": [],
                "source_refs": [],
                "engine_calls": [],
            }
        
        if not isinstance(data, dict):
            raise ValueError("provider analysis must return AnalysisPlan or a JSON object")
        
        return data

    def compose(self, request, knowledge, analysis, engine_results):
        """Compose the final video artifacts from analysis and engine results."""
        # If we have pending artifacts from analyze(), return them
        if self._pending_artifacts:
            artifacts, self._pending_artifacts = self._pending_artifacts, None
            return artifacts
        
        prompt = (
            f"You are the COMPOSE stage of a physics video generation pipeline.\n"
            f"{COMPOSE_INSTRUCTIONS}\n"
            f"Original request: {request.text}\n"
            f"Analysis: {json.dumps(analysis, default=str)}\n"
            f"Engine results: {json.dumps(engine_results, default=str)}"
        )
        
        response = self.client.models.generate_content(
            model=self.model_name, 
            contents=prompt, 
            config=self._json_config,
        )
        
        data = _extract_json(response.text)
        
        # Handle nested structure if the model wrapped the list in a dict
        if isinstance(data, dict):
            data = data.get("artifacts", [])
        
        if not isinstance(data, list) or not data:
            raise ValueError("provider must return at least one generated artifact")
        
        return data


# ============================================================================
# CRITICAL: This function MUST be at column 0 (no indentation)
# The name MUST be exactly "create_provider"
# ============================================================================
def create_provider():
    """Factory function that returns a GeminiProvider instance.
    
    This is called by request_pipeline.py via:
        factory = getattr(importlib.import_module("gemini_provider"), "create_provider")
        provider = factory()
    """
    return GeminiProvider()
