"""local_provider.py — Deterministic offline provider (no LLM, no API key)."""

class LocalProvider:
    def analyze(self, request, knowledge, engine_operations):
        # Returns empty lists for concepts/sources to pass matrix validation checks
        return {
            "intent": f"Offline deterministic demonstration of: {request.text.strip()[:300]}",
            "concept_ids": [],      
            "source_refs": [],
            "engine_calls": [],
        }

    def compose(self, request, knowledge, analysis, engine_results):
        narration = (
            "Watch this curve. The slope of the curve is the derivative. "
            "That single idea is the whole lesson."
        )
        return [{
            "artifact_id": "manual_demo",
            "title": "Manual Pipeline Test",
            "tagline": "Offline run, no LLM",
            "question": request.text.strip()[:500],
            "narration": narration,
            "equations": ["v = dx/dt"],
            "visuals": [
                {"type": "graph", "fn": "sin(t)", "t_min": 0.0, "t_max": 6.283},
            ],
            "beats": [
                {"type": "hook",     "narration": "Watch this curve."},
                {"type": "analogy",  "narration": "Think of a speedometer in a car."},
                {"type": "law",      "narration": "The slope of position is velocity."},
                {"type": "punchline","narration": "That single idea is the whole lesson."},
            ],
            "bindings": [{"beat": "law", "visual": "graph"}],
            "orientation": "horizontal",
            "quality": "720p",
            "platform": "youtube",
            "language": "en",
            "audio": {
                "script": narration,
                "voice_reference": "saem_voice_sample.wav",
                "ambient": {"kind": "none"},
                "speed": 1.0,
                "temperature": 0.65,
            },
            "visual_direction": "minimalist blackboard, 3b1b-style clean geometry",
        }]


def create_provider():   # column 0, exact name
    return LocalProvider()
