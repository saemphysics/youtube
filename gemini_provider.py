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
