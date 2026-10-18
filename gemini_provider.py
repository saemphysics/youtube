import os
import json
import re
import google.generativeai as genai

from request_pipeline import (
    UserRequest, 
    AnalysisPlan, 
    EngineCall, 
    GeneratedArtifact
)

class GeminiProvider:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")
        
        genai.configure(api_key=api_key)
        # Using gemini-1.5-flash for speed and high token limit to handle massive JSONs
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def _parse_json(self, text):
        """Cleans markdown code blocks from Gemini's response and parses JSON."""
        text = text.strip()
        # Remove ```json and ``` if they exist
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        
        if text.endswith("```"):
            text = text[:-3]
            
        return json.loads(text.strip())

    def analyze(self, request: UserRequest, knowledge: dict, engine_operations: dict) -> AnalysisPlan:
        """Phase 1: Understand the prompt and decide which physics engines to call."""
        
        prompt = f"""
You are an expert physics curriculum designer and AI planner.
Read the user's request and output ONLY a valid JSON object matching the AnalysisPlan schema below. No markdown, no explanations.

User Request:
{request.text}

Available Knowledge (Use these exact concept_ids and source_refs if relevant):
{json.dumps(knowledge.get('matched_concepts', []), indent=2)}

Available Engines (Use these exact domains and operations):
{json.dumps(engine_operations, indent=2)}

Required JSON Output Format:
{{
  "intent": "Brief summary of what the user wants",
  "concept_ids": ["concept_id_1", "concept_id_2"],
  "source_refs": ["source_ref_1"],
  "engine_calls": [
    {{
      "call_id": "unique_call_1",
      "domain": "mechanics",
      "operation": "projectile_trajectory",
      "parameters": {{"speed": 10, "angle_deg": 45, "dt": 0.1, "gravity": 9.81, "max_time": 2.0}}
    }}
  ]
}}
"""
        response = self.model.generate_content(prompt)
        data = self._parse_json(response.text)
        
        calls = tuple(
            EngineCall(
                call_id=str(c["call_id"]),
                domain=str(c["domain"]),
                operation=str(c["operation"]),
                parameters=c["parameters"]
            ) for c in data.get("engine_calls", [])
        )
        
        return AnalysisPlan(
            intent=str(data.get("intent", "Physics video generation")),
            concept_ids=tuple(data.get("concept_ids", [])),
            source_refs=tuple(data.get("source_refs", [])),
            engine_calls=calls
        )

    def compose(self, request: UserRequest, knowledge: dict, analysis: AnalysisPlan, engine_results: dict) -> list[GeneratedArtifact]:
        """Phase 2: Create the actual video script, visuals, and beats using the math results."""
        
        prompt = f"""
You are an expert 3Blue1Brown and Richard Feynman style physics animator.
Create the video artifacts based on the User Request and the Math Engine Results.
Output ONLY a valid JSON array of objects. No markdown, no explanations.

User Request:
{request.text}

Analysis Intent:
{analysis.intent}

Math Engine Results (Use these exact numbers in your narration):
{json.dumps(engine_results, indent=2)}

CRITICAL RULES:
1. You MUST strictly use the exact visual primitives and beat types requested by the user.
2. Every artifact must have a unique 'artifact_id' (lowercase, underscores).
3. Ensure the 'audio' object includes: "voice_reference": "saem_voice_sample.wav", "ambient": {{"kind": "none"}}, "speed": 1.0, "temperature": 0.65.

Required JSON Output Format (Array of Artifacts):
[
  {{
    "artifact_id": "derivative_concept_1",
    "title": "The Derivative as Velocity",
    "tagline": "Why position changes into speed",
    "question": "",
    "narration": "Feynman style narration here...",
    "equations": ["v = \\frac{{dx}}{{dt}}"],
    "visuals": [
      {{"type": "graph", "id": "graph_1"}}
    ],
    "beats": [
      {{"type": "hook", "text": "Imagine driving a car...", "focus": "graph_1"}},
      {{"type": "analogy", "text": "The speedometer shows...", "focus": "graph_1"}},
      {{"type": "law", "text": "Mathematically, this is the derivative...", "focus": "graph_1"}},
      {{"type": "punchline", "text": "Velocity is just the slope.", "focus": "graph_1"}}
    ],
    "bindings": [],
    "orientation": "horizontal",
    "quality": "1080p",
    "platform": "youtube",
    "language": "en",
    "audio": {{
      "voice_reference": "saem_voice_sample.wav",
      "ambient": {{"kind": "none"}},
      "speed": 1.0,
      "temperature": 0.65
    }},
    "visual_direction": "3b1b style, black background"
  }}
]
"""
        response = self.model.generate_content(prompt)
        data = self._parse_json(response.text)
        
        artifacts = []
        for item in data:
            artifacts.append(
                GeneratedArtifact(
                    artifact_id=str(item["artifact_id"]),
                    title=str(item["title"]),
                    tagline=str(item.get("tagline", "")),
                    question=str(item.get("question", "")),
                    narration=str(item["narration"]),
                    equations=tuple(item.get("equations", [])),
                    visuals=tuple(item.get("visuals", [])),
                    beats=tuple(item.get("beats", [])),
                    bindings=tuple(item.get("bindings", [])),
                    orientation=str(item.get("orientation", "horizontal")),
                    quality=str(item.get("quality", "1080p")),
                    platform=str(item.get("platform", "youtube")),
                    language=str(item.get("language", "en")),
                    audio=item.get("audio", {}),
                    visual_direction=str(item.get("visual_direction", "3b1b style"))
                )
            )
        return artifacts

def create_provider():
    """Factory function required by request_pipeline.py"""
    return GeminiProvider()
