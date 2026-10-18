#!/usr/bin/env python3
"""schema_validator.py — Pre-flight CSV & JSON validator (stdlib only).

Supports both 9:16 (vertical) and 16:9 (horizontal). Validates:
  * CSV structure (required + optional columns)
  * JSON parsing for all *_json columns
  * Primitive names against KNOWN_PRIMITIVES
  * Beat types against SUPPORTED_BEAT_TYPES
  * Focus targets against visual element IDs
  * Frame-bounds for element positions
  * Edge-proximity warnings
  * PYQ structural integrity
  * Duplicate video_id and element id detection
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import sys
from typing import Any

# ────────────────────────────────────────────────────────────────────────
# Orientation
# ────────────────────────────────────────────────────────────────────────
ORIENTATION = os.environ.get("SAEM_ORIENTATION", "vertical").lower()


def _bounds(orientation: str):
    if orientation == "vertical":       # 9:16
        return -4.5, 4.5, -8.0, 8.0
    return -8.0, 8.0, -4.5, 4.5         # 16:9


# ────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────
EDGE_MARGIN = 0.5

REQUIRED_COLS = [
    "video_id", "concept_type", "header_title", "tagline",
    "question_text",
    "option_a", "option_b", "option_c", "option_d",
    "correct_answer",
    "equations_json", "visual_data_json", "audio_script",
]

OPTIONAL_COLS = [
    "beats_json", "bindings_json", "formula_sheet_json",
    "manifest_path", "language", "quality", "orientation",
]

KNOWN_PRIMITIVES = {
    # Static
    "field", "vector_field",
    "charge", "particle", "dot",
    "vector", "arrow", "force",
    "line", "rod", "segment",
    "arc", "angle", "angle_arc",
    "brace",
    "graph", "function", "curve",
    "shape", "lens", "circle", "rectangle", "ellipse", "polygon",
    "text", "group",
    # Motion
    "orbit", "phasor", "ray", "particle_gas", "heat_flow",
    "energy_bar", "wave", "number_line",
    "pendulum", "spring",
    # Self-animated
    "equation_morph", "live_label", "numerical_chain",
    "warning_box", "quiz_frame", "concept_map",
    # Physics-engineered
    "polarized_wave", "dipole_radiation", "two_path_interference",
    "driven_oscillator", "wave_pulse", "harmonic_decomposition",
    "exponential_atmosphere", "energy_landscape",
    "angular_measurement", "crystal_lattice", "projectile_motion",
    "damped_oscillator", "radiating_wave", "two_source_waves",
    "driven_electron", "color_triangle", "thermionic_emission",
    "beat_pattern", "mirror_scene", "medium_atoms",
    # Day 5 advanced
    "field_by_opacity", "polar_pattern", "phase_portrait",
    "light_wave_slice", "sliced_wave", "spectral_sum", "stream_field",
    # v2 primitives
    "taylor_expand", "grid_warp",
    # Artifact catalog v2.2
    "torsion_rod", "rolling_wheel_cycloid", "interference_pattern",
    "two_beam_splitter", "potential_well", "induction_coil",
    "compass_grid", "pipe_with_gauge", "pv_diagram_tracker",
    "wave_packet_tunneling", "bohr_orbit_photon", "accretion_disk",
}

SUPPORTED_BEAT_TYPES = {
    "hook", "promise", "law", "emphasis", "analogy", "midpoint",
    "experiment", "punchline", "cta", "silent", "annotate", "shake",
}

SUPPORTED_LANGS = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
    "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi",
    "hinglish",
}

SUPPORTED_QUALITY = {
    "l", "m", "h", "h60", "k",
    "draft", "720p", "1080p", "1080p60", "4k", "4k60",
}

SUPPORTED_CONCEPT_TYPES = {"GENERIC", "PYQ", "3D", "3D_HELIX"}

_PLACEHOLDER = "\x00\x01BS\x01\x00"


# ────────────────────────────────────────────────────────────────────────
# Tolerant JSON
# ────────────────────────────────────────────────────────────────────────
def safe_json_loads(val_str: str) -> Any:
    """Idempotent, multi-layer tolerant JSON loader."""
    if not val_str or val_str.strip() in ('""', ""):
        return []
    s = val_str
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    s = s.replace("\\\\", _PLACEHOLDER)
    s = s.replace("\\", "\\\\")
    s = s.replace(_PLACEHOLDER, "\\\\")
    return json.loads(s)


# ────────────────────────────────────────────────────────────────────────
# Bounds check (shared, no duplication)
# ────────────────────────────────────────────────────────────────────────
def _check_bounds(item: dict, vid: str, row_idx: int,
                  orientation: str) -> list[str]:
    """Return a list of warning strings for any out-of-bounds coordinates.
    Also warns when an element is very near the frame edge."""
    x_min, x_max, y_min, y_max = _bounds(orientation)
    out = []
    for k in ("pos", "start", "end", "center", "pivot", "anchor"):
        if k not in item:
            continue
        c = item[k]
        if not isinstance(c, (list, tuple)) or len(c) < 2:
            continue
        try:
            x, y = float(c[0]), float(c[1])
        except (TypeError, ValueError):
            continue
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            out.append(
                f"[WARN] Row {row_idx} ({vid}) '{item.get('type')}' "
                f"{k}=({x:.2f},{y:.2f}) outside {orientation} bounds"
            )
        elif not (x_min + EDGE_MARGIN <= x <= x_max - EDGE_MARGIN
                  and y_min + EDGE_MARGIN <= y <= y_max - EDGE_MARGIN):
            out.append(
                f"[WARN] Row {row_idx} ({vid}) '{item.get('type')}' "
                f"{k}=({x:.2f},{y:.2f}) near frame edge — may clip"
            )
    return out


# ────────────────────────────────────────────────────────────────────────
# Validator
# ────────────────────────────────────────────────────────────────────────
def validate_csv(path: str) -> bool:
    has_errors = False
    seen_vids: set[str] = set()

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print("[ERROR] CSV is empty.")
            return False

        missing = [c for c in REQUIRED_COLS if c not in header]
        if missing:
            print(f"[ERROR] Missing required columns: {missing}")
            return False

        def col_index(name: str) -> int:
            try:
                return header.index(name)
            except ValueError:
                return -1

        idx_beats     = col_index("beats_json")
        idx_bindings  = col_index("bindings_json")
        idx_language  = col_index("language")
        idx_quality   = col_index("quality")
        idx_formula   = col_index("formula_sheet_json")
        idx_manifest  = col_index("manifest_path")
        idx_orient    = col_index("orientation")

        required_col_count = len(header)

        for row_idx, row in enumerate(reader, start=2):
            if not row or not any(cell.strip() for cell in row):
                continue
            if len(row) != required_col_count:
                print(f"[ERROR] Row {row_idx}: {len(row)} cols, "
                      f"header has {required_col_count}.")
                has_errors = True
                continue

            vid = row[0].strip() or f"row_{row_idx}"

            # --- duplicate video_id
            if vid in seen_vids:
                print(f"[ERROR] Row {row_idx} ({vid}): duplicate video_id")
                has_errors = True
            seen_vids.add(vid)

            # --- concept_type
            concept = row[1].strip().upper()
            if concept and concept not in SUPPORTED_CONCEPT_TYPES:
                print(f"[WARN] Row {row_idx} ({vid}): unknown concept_type "
                      f"'{concept}'")

            # --- per-row orientation
            row_orientation = ORIENTATION
            if 0 <= idx_orient < len(row) and row[idx_orient].strip():
                ro = row[idx_orient].strip().lower()
                if ro in ("vertical", "horizontal"):
                    row_orientation = ro
                else:
                    print(f"[WARN] Row {row_idx} ({vid}): bad orientation "
                          f"'{ro}'; using {ORIENTATION}.")

            # --- PYQ structural integrity
            if concept == "PYQ":
                if not row[4].strip():
                    print(f"[ERROR] Row {row_idx} ({vid}): PYQ missing "
                          f"question_text")
                    has_errors = True
                n_opts = sum(
                    1 for i in (5, 6, 7, 8)
                    if i < len(row) and row[i].strip()
                )
                if n_opts < 2:
                    print(f"[ERROR] Row {row_idx} ({vid}): PYQ needs "
                          f"≥2 options (has {n_opts})")
                    has_errors = True
                if not row[9].strip():
                    print(f"[ERROR] Row {row_idx} ({vid}): PYQ missing "
                          f"correct_answer")
                    has_errors = True

            # --- JSON columns: equations + visual_data
            visual_ids: set[str] = set()
            for col_name, col_idx in (("equations_json", 10),
                                      ("visual_data_json", 11)):
                try:
                    parsed = safe_json_loads(row[col_idx].strip())
                    if not isinstance(parsed, list):
                        print(f"[ERROR] Row {row_idx} ({vid}): "
                              f"{col_name} not a list.")
                        has_errors = True
                    elif col_name == "visual_data_json":
                        for elem_idx, it in enumerate(parsed):
                            if not isinstance(it, dict):
                                continue
                            t = str(it.get("type", "")).lower()
                            if t and t not in KNOWN_PRIMITIVES:
                                print(f"[WARN] Row {row_idx} ({vid}): "
                                      f"unknown primitive '{t}'")
                            # element id uniqueness
                            eid = str(it.get("id") or f"elem_{elem_idx}")
                            if eid in visual_ids:
                                print(f"[WARN] Row {row_idx} ({vid}): "
                                      f"duplicate element id '{eid}'")
                            visual_ids.add(eid)
                            # bounds
                            for w in _check_bounds(it, vid, row_idx,
                                                   row_orientation):
                                print(w)
                except Exception as e:
                    print(f"[ERROR] Row {row_idx} ({vid}): "
                          f"{col_name} — {e}")
                    has_errors = True

            # --- optional JSON columns
            def _check_optional(idx: int, name: str) -> Any:
                nonlocal has_errors
                if idx < 0 or idx >= len(row) or not row[idx].strip():
                    return None
                try:
                    parsed = safe_json_loads(row[idx].strip())
                    if not isinstance(parsed, list):
                        print(f"[ERROR] Row {row_idx} ({vid}): "
                              f"{name} not a list.")
                        has_errors = True
                        return None
                    return parsed
                except Exception as e:
                    print(f"[ERROR] Row {row_idx} ({vid}): {name} — {e}")
                    has_errors = True
                    return None

            _check_optional(idx_bindings, "bindings_json")
            _check_optional(idx_formula, "formula_sheet_json")
            beats = _check_optional(idx_beats, "beats_json")

            # --- beat validation (needs visual_ids from above)
            if isinstance(beats, list):
                for b_i, b in enumerate(beats):
                    if not isinstance(b, dict):
                        print(f"[WARN] Row {row_idx} ({vid}): beat {b_i} "
                              f"not a dict")
                        continue
                    bt = str(b.get("type", "")).lower()
                    if bt and bt not in SUPPORTED_BEAT_TYPES:
                        print(f"[WARN] Row {row_idx} ({vid}): beat {b_i} "
                              f"unknown type '{bt}'")
                    focus = str(b.get("focus", "")).strip()
                    if focus and focus not in visual_ids:
                        print(f"[WARN] Row {row_idx} ({vid}): beat {b_i} "
                              f"focus='{focus}' not among visual ids "
                              f"{sorted(visual_ids)}")

            # --- language
            if 0 <= idx_language < len(row) and row[idx_language].strip():
                lang = row[idx_language].strip().lower()
                if lang not in SUPPORTED_LANGS:
                    print(f"[WARN] Row {row_idx} ({vid}): unsupported "
                          f"language '{lang}'")

            # --- quality
            if 0 <= idx_quality < len(row) and row[idx_quality].strip():
                q = row[idx_quality].strip().lower()
                if q not in SUPPORTED_QUALITY:
                    print(f"[WARN] Row {row_idx} ({vid}): unknown quality "
                          f"'{q}'")

            # --- manifest_path
            if 0 <= idx_manifest < len(row) and row[idx_manifest].strip():
                mp = row[idx_manifest].strip()
                if not os.path.exists(mp):
                    print(f"[WARN] Row {row_idx} ({vid}): manifest_path "
                          f"'{mp}' does not exist.")

    if seen_vids:
        print(f"[INFO] Unique video_ids: {len(seen_vids)}")
    return not has_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="generated renderer-adapter CSV")
    target = parser.parse_args().csv
    print(f"[INFO] Validating {target}")
    if validate_csv(target):
        print("[SUCCESS] All checks passed.")
        return 0
    print("[FATAL] Validation failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
