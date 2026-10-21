#!/usr/bin/env python3
"""
render_universal.py — Cinematic universal renderer (v2.3).

Supports both 9:16 (vertical, YouTube Shorts) and 16:9 (horizontal).

v2.3 changes (visual kit integration):
  * 8 cinematic additions plugged in via visual_kit module:
    annotate, spotlight, ambient, kinetic_title, term_bind,
    camera_shake, vector_decomposition, cinematic_transition.
  * _do_transition now supports wipe / push / iris / zoom_blur.
  * _beat_emphasis uses spotlight (dims everything except the focus).
  * _reveal_visuals auto-installs ambient motion on all visuals.
  * Fixed: 12 artifact primitives now correctly defined as class
    methods (previously at module level — router silently failed).

v2.2 changes (retention + motion):
  * Cold-open: no intro bumper. Content starts on frame 1.
  * Branding moved to a richer 5s CLOSING CARD (logo + tagline + CTA).
  * 3b1b-style motion kit: staggered reveals, camera focus, morph_tex.
  * New beat types: "promise", "midpoint", "cta".
  * Vertical zone table retuned so the primary visual sits at y ≈ +1.0,
    clear of YouTube's bottom UI overlays.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import textwrap

import numpy as np
from manim import *
from manim import rate_functions as rf
from physics_engine import (
    ElectrostaticsEngine,
    MechanicsEngine,
    WavesEngine,
    integrate_rk4,
)

# ============================================================ voiceover
try:
    from manim_voiceover import VoiceoverScene
    from manim_voiceover.services.gtts import GTTSService
    HAS_VOICEOVER = True
except ImportError:
    HAS_VOICEOVER = False
    VoiceoverScene = Scene
    GTTSService = None

 # ✅ NEW - Import only when needed
HAS_NX = False
HAS_STREAMLINES = False

def _get_networkx():
    global HAS_NX
    if not HAS_NX:
        try:
            import networkx as nx
            HAS_NX = True
            return nx
        except ImportError:
            return None
    import networkx as nx
    return nx

def _get_streamlines():
    global HAS_STREAMLINES
    if not HAS_STREAMLINES:
        try:
            from manim.mobject.vector_field import StreamLines
            HAS_STREAMLINES = True
            return StreamLines
        except ImportError:
            return None
    from manim.mobject.vector_field import StreamLines
    return StreamLines

# ============================================================ sound design
try:
    from sound_design import add as _play_sound
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False

    def _play_sound(scene, name, gain_db=0.0):
        return

# ============================================================ visual kit
try:
    import visual_kit
    HAS_VISUAL_KIT = True
except ImportError:
    HAS_VISUAL_KIT = False
    visual_kit = None

# ============================================================ orientation
ORIENTATION = os.environ.get("SAEM_ORIENTATION", "vertical").lower()

# ============================================================ quality
QUALITY_PRESETS = {
    "draft":   dict(pixel_width=854,  pixel_height=480,  frame_rate=15),
    "720p":    dict(pixel_width=1280, pixel_height=720,  frame_rate=30),
    "1080p":   dict(pixel_width=1920, pixel_height=1080, frame_rate=30),
    "1080p60": dict(pixel_width=1920, pixel_height=1080, frame_rate=60),
    "4k":      dict(pixel_width=3840, pixel_height=2160, frame_rate=30),
    "4k60":    dict(pixel_width=3840, pixel_height=2160, frame_rate=60),
    "l":       dict(pixel_width=854,  pixel_height=480,  frame_rate=15),
    "m":       dict(pixel_width=1280, pixel_height=720,  frame_rate=30),
    "h":       dict(pixel_width=1920, pixel_height=1080, frame_rate=30),
    "h60":     dict(pixel_width=1920, pixel_height=1080, frame_rate=60),
    "k":       dict(pixel_width=3840, pixel_height=2160, frame_rate=30),
}


def apply_quality(preset: str, orientation: str | None = None):
    global ORIENTATION
    if orientation is not None:
        ORIENTATION = orientation.lower()
    q = QUALITY_PRESETS.get(preset, QUALITY_PRESETS["1080p"])
    if ORIENTATION == "vertical":
        config.pixel_width  = q["pixel_height"]
        config.pixel_height = q["pixel_width"]
        config.frame_width  = 9.0
        config.frame_height = 16.0
    else:
        config.pixel_width  = q["pixel_width"]
        config.pixel_height = q["pixel_height"]
        config.frame_width  = 16.0
        config.frame_height = 9.0
    config.frame_rate = q["frame_rate"]


apply_quality("1080p", ORIENTATION)
config.background_color = "#0B0C10"

# Enable scene caching for faster re-renders
config.cache_dir = "media/cache"
config.disable_caching = False
config.cache_flush = False


def is_opengl() -> bool:
    return getattr(config, "renderer", "cairo") == "opengl"


def frame_bounds():
    if ORIENTATION == "vertical":
        return -4.5, 4.5, -8.0, 8.0
    return -8.0, 8.0, -4.5, 4.5


# ============================================================ color
COLOR = {
    "bg": "#0B0C10", "card": "#15161D", "border": "#2A2C36",
    "white": "#F2F3F5", "muted": "#8A8F9C",
    "field": "#3498DB", "pos": "#FF4B4B", "neg": "#00D2FF",
    "accent": "#F1C40F", "force": "#2ECC71", "momentum": "#E67E22",
    "hot": "#FF4B4B", "cold": "#3498DB", "warn": "#E67E22",
}
SEMANTIC = {
    "force": COLOR["field"], "field": COLOR["field"],
    "energy": COLOR["accent"], "accent": COLOR["accent"],
    "charge": COLOR["pos"], "positive": COLOR["pos"],
    "negative": COLOR["neg"], "velocity": COLOR["force"],
    "momentum": COLOR["momentum"], "muted": COLOR["muted"],
    "hot": COLOR["hot"], "cold": COLOR["cold"], "warn": COLOR["warn"],
}

# ============================================================ zones
ZONES_VERTICAL = {
    "title":     6.3,
    "hook":      4.5,
    "visual":    1.0,
    "chain":    -2.0,
    "equations": -4.5,
    "answer":   -6.5,
    "cta":       0.0,
}
ZONES_HORIZONTAL = {
    "title":     3.80,
    "hook":      2.60,
    "visual":    0.20,
    "chain":    -1.80,
    "equations": -3.00,
    "answer":    -4.00,
    "cta":       0.0,
}


def zone(name: str) -> float:
    table = (ZONES_VERTICAL if ORIENTATION == "vertical"
             else ZONES_HORIZONTAL)
    return table.get(name, 0.0)


# ============================================================ ease curves
RF_SOFT   = rf.ease_in_out_sine
RF_SNAP   = rf.ease_out_cubic
RF_SPRING = rf.ease_out_back

EASE = {
    "soft":   rf.ease_in_out_sine,
    "land":   rf.ease_out_cubic,
    "launch": rf.ease_in_cubic,
    "snap":   rf.ease_out_back,
    "linear": lambda t: t,
}


def ease_for(name: str):
    return EASE.get(str(name or "soft").lower(), EASE["soft"])


# ============================================================ typography
TYPO = {
    "hero":  {"font_size": 42, "weight": BOLD},
    "label": {"font_size": 24, "weight": NORMAL},
    "note":  {"font_size": 18, "weight": NORMAL, "slant": ITALIC},
}


def typo(kind: str, text: str, color=WHITE):
    style = TYPO.get(kind, TYPO["label"])
    try:
        return Text(text,
                    font_size=style["font_size"],
                    weight=style.get("weight", NORMAL),
                    color=color,
                    slant=style.get("slant", NORMAL))
    except Exception:
        return Text(text, font_size=style["font_size"], color=color)


# ============================================================ categories
SELF_ANIMATED = {
    "equation_morph", "numerical_chain", "warning_box",
    "quiz_frame", "concept_map", "driven_oscillator",
    "wave_pulse", "harmonic_decomposition",
}


def _log(m):
    print(f"[render] {m}", file=sys.stderr, flush=True)


# ============================================================ animation kit
def stagger_reveal(scene, mobs, lag_ratio=0.10, run_time=1.5,
                   shift=UP * 0.20):
    """Cascade a list of mobjects onto screen, one at a time."""
    if not mobs:
        return
    anims = []
    for m in mobs:
        if has_updater(m):
            anims.append(FadeIn(m, run_time=0.5))
        elif isinstance(m, Arrow):
            anims.append(GrowArrow(m, run_time=0.6))
        elif isinstance(m, (Line, DashedLine, Arc)):
            anims.append(Create(m, rate_func=rf.ease_out_cubic,
                                run_time=0.7))
        elif isinstance(m, (Circle, Ellipse)):
            anims.append(GrowFromCenter(m, run_time=0.5))
        else:
            anims.append(FadeIn(m, shift=shift))
    scene.play(
        LaggedStart(*anims, lag_ratio=lag_ratio),
        run_time=run_time,
        rate_func=rf.ease_out_cubic,
    )


def morph_tex(scene, old, new, run_time=1.3):
    """Transform one MathTex into another, matching substrings."""
    try:
        scene.play(
            TransformMatchingTex(old, new),
            run_time=run_time,
            rate_func=rf.ease_in_out_sine,
        )
    except Exception:
        scene.play(
            Transform(old, new),
            run_time=run_time,
            rate_func=rf.ease_in_out_sine,
        )


def focus_on(scene, mob, width_frac=0.55, run_time=1.2, ease="land"):
    """Zoom the camera onto a mobject."""
    fw = config.frame_width
    scene.play(
        scene.camera.frame.animate
            .set(width=fw * width_frac)
            .move_to(mob.get_center()),
        run_time=run_time,
        rate_func=ease_for(ease),
    )


def pulse(scene, mob, scale=1.08, run_time=0.45):
    """Quick scale pulse for emphasis. Returns to original size."""
    scene.play(
        mob.animate(rate_func=there_and_back, run_time=run_time).scale(scale),
    )


# ============================================================ helpers
_PLACEHOLDER = "\x00\x01BS\x01\x00"


 # ✅ NEW - Simpler fallback
def safe_json_loads(val_str):
    if not val_str or val_str.strip() in ('""', ""):
        return []
    try:
        return json.loads(val_str)
    except json.JSONDecodeError:
        # Try removing escape characters
        try:
            return json.loads(val_str.replace("\\\\", "\\"))
        except json.JSONDecodeError:
            return []

def to_3d(p):
    if p is None:
        return np.zeros(3)
    if not isinstance(p, (list, tuple, np.ndarray)):
        return np.zeros(3)
    if len(p) == 2:
        return np.array([float(p[0]), float(p[1]), 0.0])
    if len(p) >= 3:
        return np.array([float(p[0]), float(p[1]), float(p[2])])
    return np.zeros(3)


def resolve_color(item):
    if not isinstance(item, dict):
        return COLOR["accent"]
    if "color" in item:
        return item["color"]
    return SEMANTIC.get(str(item.get("semantic", "")).lower().strip(),
                        COLOR["field"])


def safe_latex(s, font_size=26, color=WHITE):
    if s is None:
        return Text("", font_size=font_size, color=color)
    s = str(s).strip()
    if not s:
        return Text("", font_size=font_size, color=color)
    if "\\" in s or "^" in s or "_" in s or "{" in s:
        try:
            return MathTex(s, font_size=font_size, color=color)
        except Exception:
            pass
    try:
        return Text(s, font_size=font_size, color=color)
    except Exception:
        return Text("?", font_size=font_size, color=color)


def math_or_text(s, font_size=26, color=WHITE, force_math=False):
    s = str(s).strip()
    if not s:
        return Text("", font_size=font_size, color=color)
    if force_math or "\\" in s or "^" in s or "_" in s:
        try:
            return MathTex(s, font_size=font_size, color=color)
        except Exception:
            pass
    try:
        return Text(s, font_size=font_size, color=color)
    except Exception:
        return Text("?", font_size=font_size, color=color)


_ALLOWED_MATH = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs,
    "log": np.log, "pi": np.pi, "e": np.e,
}


def eval_expr(expr, x):
    expr = str(expr).replace("^", "**")
    try:
        return float(eval(expr, {"__builtins__": None},
                          {**_ALLOWED_MATH, "x": x}))
    except Exception:
        return 0.0


def has_updater(m):
    if getattr(m, "updaters", None):
        return True
    if hasattr(m, "submobjects"):
        return any(has_updater(s) for s in m.submobjects)
    return False


def fit_to_frame(mob, margin=0.85):
    """Uniform-scale a mobject so it fits inside the usable frame."""
    if mob is None:
        return None
    try:
        fw = config.frame_width * margin
        fh = config.frame_height * margin
        w = float(mob.width)
        h = float(mob.height)
        if w <= fw and h <= fh:
            return mob
        scale = min(fw / max(w, 1e-6), fh / max(h, 1e-6))
        mob.scale(scale)
    except Exception:
        pass
    return mob


def live_wait(scene, duration, mobs=None, breathe_scale=0.008):
    if duration <= 0.05:
        return
    candidates = mobs if mobs is not None else \
                 [m for m in scene.mobjects if isinstance(m, VMobject)]
    targets = []
    for m in candidates:
        if not isinstance(m, VMobject):
            continue
        if has_updater(m):
            continue
        try:
            if m.get_opacity() < 0.1:
                continue
        except Exception:
            pass
        targets.append(m)
    targets = targets[:5]
    if not targets:
        scene.wait(duration)
        return
    cycles = max(1, int(duration / 3.0))
    per = duration / cycles
    for _ in range(cycles):
        anims = []
        for m in targets:
            try:
                anims.append(
                    m.animate(rate_func=there_and_back, run_time=per)
                     .scale(1 + breathe_scale)
                )
            except Exception:
                continue
        if anims:
            try:
                scene.play(*anims, run_time=per)
            except Exception:
                scene.wait(per)
        else:
            scene.wait(per)


def _triangle_wave(x, lo, hi):
    period = 2 * (hi - lo)
    if period <= 0:
        return lo
    y = (x - lo) % period
    if y > (hi - lo):
        y = period - y
    return lo + y


def _dipole_field(charges):
    chs = [(np.array(c["pos"], dtype=float), float(c.get("charge", 1)))
           for c in charges]

    def _field(p):
        total = np.zeros(3)
        for q_pos, q_val in chs:
            r = p - q_pos
            r_mag = float(np.linalg.norm(r))
            if r_mag < 0.20:
                r_mag = 0.20
            total += q_val * r / (r_mag ** 3)
        return total

    return _field


# ============================================================ camera
class CameraDirector:
    def __init__(self, scene):
        self.s = scene
        self.default_width = config.frame_width

    def zoom_to(self, target_width, center=None, run_time=1.2, ease="soft"):
        frame = self.s.camera.frame
        builder = frame.animate.set(width=target_width)
        if center is not None:
            builder = builder.move_to(center)
        self.s.play(builder, run_time=run_time, rate_func=ease_for(ease))

    def reset(self, run_time=0.8, ease="soft"):
        self.s.play(
            self.s.camera.frame.animate.set(
                width=self.default_width).move_to(ORIGIN),
            run_time=run_time, rate_func=ease_for(ease),
        )

    def pan_to(self, target, run_time=1.0, ease="soft"):
        self.s.play(
            self.s.camera.frame.animate.move_to(target),
            run_time=run_time, rate_func=ease_for(ease),
        )

    def focus(self, mobject, padding=1.4, run_time=1.2, ease="land"):
        target_width = max(
            mobject.width * padding,
            mobject.height * padding *
                (config.frame_width / config.frame_height),
            2.0,
        )
        self.s.play(
            self.s.camera.frame.animate
                .set(width=target_width)
                .move_to(mobject.get_center()),
            run_time=run_time, rate_func=ease_for(ease),
        )


def make_card(content, width=None):
    if width is None:
        width = config.frame_width * 0.92
    pad = 0.35
    bg = RoundedRectangle(
        corner_radius=0.22, width=width,
        height=content.height + pad * 2,
        fill_color=COLOR["card"], fill_opacity=0.96,
        stroke_color=COLOR["border"], stroke_width=1.5,
    )
    content.move_to(bg.get_center())
    return VGroup(bg, content)


# ============================================================ branding
class SaemLabsBranding:
    GOLD, BLUE, WHITE = "#F1C40F", "#3498DB", "#F2F3F5"

    @classmethod
    def _emblem(cls, radius=0.50, color=None):
        color = color or cls.GOLD
        ring = Circle(radius=radius, color=color, stroke_width=3)
        span = radius * 1.3
        amp = radius * 0.40
        sine = ParametricFunction(
            lambda t: np.array([t, amp * np.sin(2 * np.pi * t / span), 0.0]),
            t_range=[-span / 2.0, span / 2.0],
            color=color, stroke_width=3,
        )
        return VGroup(ring, sine)

    @classmethod
    def _build_full_logo(cls, scale=1.0):
        em = cls._emblem(radius=0.50)
        wm = Text("SaemLabs", font_size=60, weight=BOLD, color=cls.WHITE)
        rule = Line(LEFT, RIGHT, color=cls.GOLD,
                    stroke_width=1.5).set_width(3.4)
        tl = Text("NEET  ·  JEE  ·  MIT", font_size=22, color=cls.BLUE)
        group = VGroup(em, wm, rule, tl).arrange(DOWN, buff=0.30)
        group.scale(scale)
        return group

    @classmethod
    def _build_watermark(cls):
        ring = Circle(radius=0.13, color=cls.GOLD, stroke_width=2.2)
        span = 0.13 * 1.3
        amp = 0.13 * 0.40
        sine = ParametricFunction(
            lambda t: np.array([t, amp * np.sin(2 * np.pi * t / span), 0.0]),
            t_range=[-span / 2.0, span / 2.0],
            color=cls.GOLD, stroke_width=2.2,
        )
        emblem = VGroup(ring, sine)
        wm = Text("SaemLabs", font_size=18, weight=BOLD, color=cls.WHITE)
        group = VGroup(emblem, wm).arrange(RIGHT, buff=0.16)
        group.set_opacity(0.72)
        return group

    @classmethod
    def pin_watermark(cls, scene, corner="DR", width_frac=0.085):
        wm = cls._build_watermark()
        scene.add(wm)

        def _pin(m):
            frame = scene.camera.frame
            fw = frame.get_width()
            fh = frame.get_height()
            m.set_width(fw * width_frac)
            pad = fw * 0.025
            c = frame.get_center()
            if corner == "DR":
                pos = c + np.array([fw / 2 - pad - m.width / 2,
                                    -fh / 2 + pad + m.height / 2, 0])
            elif corner == "UR":
                pos = c + np.array([fw / 2 - pad - m.width / 2,
                                    fh / 2 - pad - m.height / 2, 0])
            elif corner == "DL":
                pos = c + np.array([-fw / 2 + pad + m.width / 2,
                                    -fh / 2 + pad + m.height / 2, 0])
            else:
                pos = c + np.array([-fw / 2 + pad + m.width / 2,
                                    fh / 2 - pad - m.height / 2, 0])
            m.move_to(pos)

        wm.add_updater(_pin)
        _pin(wm)
        return wm

    @classmethod
    def closing_card(cls, scene, duration=5.0):
        frame = scene.camera.frame
        fade_targets = [m for m in scene.mobjects if m is not frame]
        if fade_targets:
            scene.play(*[FadeOut(m) for m in fade_targets],
                       run_time=0.5,
                       rate_func=rf.ease_in_cubic)

        logo = cls._build_full_logo()
        logo.move_to(UP * 1.8)
        for part in logo:
            part.set_opacity(0)

        cta = Text("Subscribe for daily physics",
                   font_size=28, color=cls.WHITE)
        cta.move_to(DOWN * 1.8)
        cta.set_opacity(0)

        scene.add(logo, cta)

        scene.play(logo[0].animate.set_opacity(1),
                   logo[1].animate.set_opacity(1),
                   run_time=0.65,
                   rate_func=rf.ease_out_cubic)
        scene.play(logo[2].animate.set_opacity(1),
                   logo[3].animate.set_opacity(1),
                   cta.animate.set_opacity(1),
                   run_time=0.5,
                   rate_func=rf.ease_out_cubic)

        try:
            scene.play(logo[0].animate(rate_func=there_and_back,
                                        run_time=0.9).scale(1.06))
        except Exception:
            scene.wait(0.9)

        scene.wait(max(0.3, duration - 3.0))

        scene.play(FadeOut(VGroup(logo, cta)),
                   run_time=0.5,
                   rate_func=rf.ease_in_cubic)

    @classmethod
    def intro_bumper(cls, scene, duration=3.0):
        """Deprecated. Do not call — content should open cold."""
        pass

    @classmethod
    def outro_card(cls, scene, duration=4.0):
        """Alias for closing_card, kept for older callers."""
        cls.closing_card(scene, duration=duration)


# ============================================================ 2D scene
class UniversalPhysicsScene(MovingCameraScene, VoiceoverScene):
    def __init__(self, scene_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.sk = scene_kwargs or {}
        self.eq_mobs = []
        self.visuals_by_id = {}
        self._law_fired = False
        self._self_animated = []
        self.global_time = ValueTracker(0.0)
        self._watermark = None

    # -------------------------------------------------- construct
    def construct(self):
        sk = self.sk
        self.director = CameraDirector(self)

        no_voiceover = bool(sk.get("no_voiceover", False))
        render_only = bool(sk.get("render_only", False))
        use_voiceover = (HAS_VOICEOVER and GTTSService
                         and not no_voiceover and not render_only)
        if use_voiceover:
            try:
                self.set_speech_service(GTTSService(lang="en", tld="com"))
            except Exception as e:
                _log(f"voiceover service init failed: {e}")
        else:
            _log("Voiceover disabled — using manifest durations.")

        branding_on = sk.get("branding", "on") != "off"
        closing_dur = float(sk.get("outro_duration", 5.0))
        duration = float(sk.get("duration", 180.0))

        # ---- COLD OPEN: no intro bumper. Content begins on frame 1.
        if branding_on:
            self._watermark = SaemLabsBranding.pin_watermark(self,
                                                             corner="DR")

        self.global_time.add_updater(lambda m, dt: m.increment_value(dt))
        self.add(self.global_time)

        eq_card = self._build_equation_card(
            safe_json_loads(sk.get("equations_json", "[]"))
        )
        visual_group = self._build_visuals(
            safe_json_loads(sk.get("visual_data_json", "[]"))
        )

        beats = safe_json_loads(sk.get("beats_json", "[]"))
        use_beats = bool(beats) and any(
            isinstance(b, dict) and "type" in b for b in beats
        )
        main_target = max(4.0, duration - closing_dur)

        if use_beats:
            self._run_narrated_timeline(
                beats, eq_card, main_target, use_voiceover,
                visual_group=visual_group,
            )
        else:
            # Legacy fallback: title -> visuals -> equations -> wait.
            self._build_title(sk.get("header_title", ""),
                              sk.get("tagline", ""))
            if len(visual_group) > 0:
                self._reveal_visuals(visual_group)
            if len(eq_card) > 0:
                target_eq_time = max(3.0, main_target / 3.0)
                gap = target_eq_time - self.renderer.time
                if gap > 0.1:
                    live_wait(self, gap)
                self._reveal_equation_card(eq_card)
            remaining = main_target - self.renderer.time
            if remaining > 0.4:
                live_wait(self, remaining)

        options = [o for o in (sk.get("options") or []) if o and o.strip()]
        correct = sk.get("correct_answer", "")
        if options or correct:
            ans_box = self._build_answers(options, correct)
            if ans_box is not None:
                self.play(FadeIn(ans_box, shift=UP * 0.15), run_time=0.5)
                self.play(Circumscribe(ans_box, color=COLOR["accent"],
                                       buff=0.15, run_time=0.7))

        if sk.get("formula_sheet"):
            self._render_formula_sheet(sk["formula_sheet"], eq_card)

        if self._watermark is not None:
            try:
                self._watermark.clear_updaters()
                self.play(FadeOut(self._watermark), run_time=0.4)
            except Exception:
                pass
            self._watermark = None

        # ---- CLOSING CARD
        if branding_on:
            SaemLabsBranding.closing_card(self, duration=closing_dur)
        else:
            live_wait(self, 1.5)

    # -------------------------------------------------- title (legacy)
    def _build_title(self, title, tagline):
        parts = []
        if title:
            try:
                parts.append(Text(title, weight=BOLD, font_size=44,
                                  color=WHITE))
            except Exception:
                parts.append(Text(title, font_size=44, color=WHITE))
        if tagline:
            parts.append(Text(tagline, font_size=24,
                              color=COLOR["muted"], slant=ITALIC))
        if not parts:
            return
        header = VGroup(*parts).arrange(DOWN, buff=0.18)
        max_w = config.frame_width * 0.88
        if header.width > max_w:
            header.scale_to_fit_width(max_w)
        header.move_to([0, zone("hook"), 0])
        self.play(FadeIn(header, shift=DOWN * 0.3,
                         rate_func=rf.ease_out_cubic), run_time=0.6)
        self.wait(0.3)
        self.play(FadeOut(header, shift=UP * 0.3,
                          rate_func=rf.ease_in_cubic), run_time=0.4)

    # -------------------------------------------------- visuals
    def _build_visuals(self, visual_data):
        group = VGroup()
        self._self_animated = []
        for idx, item in enumerate(visual_data):
            if not isinstance(item, dict):
                continue
            itype = str(item.get("type", "")).lower()
            mob = self._build_primitive(item, idx)
            if mob is None:
                continue
            if itype not in SELF_ANIMATED:
                fit_to_frame(mob)
            if itype in SELF_ANIMATED:
                self._self_animated.append(mob)
            else:
                group.add(mob)
            key = str(item.get("id") or f"elem_{idx}")
            self.visuals_by_id[key] = mob
        return group
     
  # ✅ NEW - Batch small groups, stagger large ones
    def _reveal_visuals(self, group):
        if len(group) == 0:
            return
        if len(group) <= 3:
        # Small group: reveal all at once
        self.play(*[FadeIn(m, shift=UP*0.2) for m in group], run_time=0.8)
    else:
        # Large group: stagger
        stagger_reveal(self, list(group), lag_ratio=0.08, run_time=1.4)
        if HAS_VISUAL_KIT:
        visual_kit.ambient(self, list(group))
     
        if HAS_VISUAL_KIT:
            visual_kit.ambient(self, list(group))
        self.director.zoom_to(config.frame_width * 0.92,
                              run_time=0.9, ease="launch")
        self.director.reset(run_time=0.6, ease="soft")

    # -------------------------------------------------- equations
    def _build_equation_card(self, equations):
        if not equations:
            return VGroup()
        eq_mobs = [safe_latex(eq, font_size=36, color=WHITE)
                   for eq in equations if eq]
        if not eq_mobs:
            return VGroup()
        self.eq_mobs = eq_mobs
        eq_group = VGroup(*eq_mobs).arrange(DOWN, buff=0.35)
        max_w = config.frame_width * 0.88
        if eq_group.width > max_w:
            eq_group.scale_to_fit_width(max_w)
        return make_card(eq_group,
                         width=config.frame_width * 0.94
                         ).move_to([0, zone("equations"), 0])

    def _reveal_equation_card(self, card):
        bg, content = card[0], card[1]
        bg.save_state()
        bg.set_opacity(0).scale(0.95)

        self.play(Restore(bg, rate_func=rf.ease_out_back), run_time=0.6)
        focus_on(self, card, width_frac=0.55, run_time=0.9, ease="land")

        if isinstance(content, VGroup) and len(content) > 0:
            self.play(Write(content[0], rate_func=smooth), run_time=1.2)
            self.wait(0.2)
            for i in range(1, len(content)):
                try:
                    self.play(
                        TransformMatchingTex(
                            content[i - 1].copy(),
                            content[i],
                        ),
                        run_time=1.3,
                        rate_func=rf.ease_in_out_sine,
                    )
                except Exception:
                    self.play(Write(content[i], rate_func=smooth),
                              run_time=1.0)
                self.wait(0.2)

        self.director.reset(run_time=0.7, ease="soft")
        self._law_fired = True

    # -------------------------------------------------- timeline
    def _run_narrated_timeline(self, beats, eq_card, main_target,
                                use_voiceover, visual_group=None):
        manifest = self._load_manifest()
        valid_beats = [b for b in beats if isinstance(b, dict)]
        n = max(1, len(valid_beats))
        beat_durations = self._distribute_durations(n, manifest, main_target)
        _log(f"Timeline: {n} beats, voiceover="
             f"{'on' if use_voiceover else 'off'}")

        for i, beat in enumerate(valid_beats):
            kind  = str(beat.get("type", "")).lower()
            text  = str(beat.get("text", "")).strip()
            focus = str(beat.get("focus", "")).strip()
            ease_name = str(beat.get("ease", "soft"))
            transition = str(beat.get("transition", "")).lower()

            if transition and i > 0:
                self._do_transition(transition)

            if kind == "silent":
                dur = float(beat.get("duration", beat_durations[i]))
                _play_sound(self, "drone", gain_db=-22)
                live_wait(self, dur)
                continue

            try:
                if use_voiceover and text:
                    with self.voiceover(text=text) as tracker:
                        self._dispatch_beat(kind, focus, eq_card,
                                            tracker.duration, ease_name,
                                            visual_group=visual_group)
                else:
                    self._dispatch_beat(kind, focus, eq_card,
                                        beat_durations[i], ease_name,
                                        visual_group=visual_group)
            except Exception as e:
                _log(f"beat '{kind}' failed: {type(e).__name__}: {e}")
                self.wait(2.0)

        remaining = main_target - self.renderer.time
        if remaining > 0.4:
            live_wait(self, remaining)

    def _do_transition(self, name: str):
        _play_sound(self, "whoosh", gain_db=-12)
        try:
            if HAS_VISUAL_KIT and name in ("wipe", "push",
                                            "iris", "zoom_blur"):
                visual_kit.cinematic_transition(self, kind=name)
                return
            if name in ("fade_black", "black"):
                cover = Rectangle(width=config.frame_width + 2,
                                  height=config.frame_height + 2,
                                  color="#000000", fill_opacity=1.0)
                self.play(FadeIn(cover), run_time=0.35,
                          rate_func=rf.ease_in_cubic)
                self.play(FadeOut(cover), run_time=0.35,
                          rate_func=rf.ease_out_cubic)
            elif name in ("dissolve", "cross"):
                cover = Rectangle(width=config.frame_width + 2,
                                  height=config.frame_height + 2,
                                  color="#0B0C10", fill_opacity=0.0)
                self.add(cover)
                self.play(cover.animate.set_opacity(0.85), run_time=0.25)
                self.play(cover.animate.set_opacity(0.0), run_time=0.25)
                self.remove(cover)
            elif name in ("pan_left", "pan_right"):
                dx = 3.0 if name == "pan_right" else -3.0
                self.play(
                    self.camera.frame.animate.shift(RIGHT * dx),
                    run_time=0.5, rate_func=rf.ease_in_cubic,
                )
                self.play(
                    self.camera.frame.animate.shift(LEFT * dx),
                    run_time=0.5, rate_func=rf.ease_in_out_sine,
                )
        except Exception as e:
            _log(f"transition '{name}' failed: {e}")

    _manifest_cache = {}

    def _load_manifest(self):
    path = self.sk.get("manifest_path", "")
    if not path or not os.path.exists(path):
        return None
    if path in self._manifest_cache:
        return self._manifest_cache[path]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._manifest_cache[path] = data
        return data
    except Exception as e:
        _log(f"manifest load failed: {e}")
        return None

    def _distribute_durations(self, n_beats, manifest, main_target):
        if manifest and manifest.get("timeline"):
            timeline = manifest["timeline"]
            total = len(timeline)
            if total > 0:
                per = max(1, total // n_beats)
                out = []
                for i in range(n_beats):
                    s = i * per
                    e = (i + 1) * per if i < n_beats - 1 else total
                    seg = sum(t.get("duration", 0.0)
                              for t in timeline[s:e])
                    out.append(max(1.5, seg))
                return out
        per = max(1.5, main_target / n_beats)
        return [per] * n_beats

    # -------------------------------------------------- beat dispatch
BEAT_DISPATCH = {
    "hook": "_beat_hook",
    "promise": "_beat_promise",
    "law": "_beat_law",
    "emphasis": "_beat_emphasis",
    "analogy": "_beat_analogy",
    "midpoint": "_beat_midpoint",
    "experiment": "_beat_experiment",
    "punchline": "_beat_punchline",
    "cta": "_beat_cta",
}

    def _dispatch_beat(self, kind, focus, eq_card, duration, ease_name="soft", visual_group=None):
        method_name = self.BEAT_DISPATCH.get(kind)
        if method_name:
        method = getattr(self, method_name)
        if kind in ("hook", "promise", "analogy", "midpoint", "experiment"):
            method(focus, duration, ease_name, visual_group=visual_group)
        elif kind in ("law", "punchline"):
            method(eq_card, duration)
        else:
            method(duration)
    else:
        live_wait(self, duration)

    def _beat_hook(self, focus, duration, ease_name="soft",
                    visual_group=None):
        _play_sound(self, "tick", gain_db=-18)
        if (visual_group is not None and len(visual_group) > 0
                and not any(m in self.mobjects for m in visual_group)):
            stagger_reveal(self, list(visual_group),
                           lag_ratio=0.10, run_time=1.4)
        mob = self.visuals_by_id.get(focus)
        if mob is not None:
            self.play(Indicate(mob, color=COLOR["accent"],
                               scale_factor=1.10),
                      run_time=min(duration, 1.5),
                      rate_func=ease_for(ease_name))
            live_wait(self, max(0, duration - 1.5))
        else:
            live_wait(self, duration)

    def _beat_promise(self, focus, duration, ease_name="soft",
                       visual_group=None):
        if (visual_group is not None and len(visual_group) > 0
                and not any(m in self.mobjects for m in visual_group)):
            stagger_reveal(self, list(visual_group),
                           lag_ratio=0.10, run_time=1.6)
            self.director.zoom_to(config.frame_width * 0.94,
                                  run_time=0.8, ease="launch")
            self.director.reset(run_time=0.6, ease="soft")
        mob = self.visuals_by_id.get(focus)
        if mob is not None:
            pulse(self, mob, scale=1.05, run_time=0.5)
        live_wait(self, max(0, duration - 2.4))

    def _beat_law(self, eq_card, duration):
        if len(eq_card) == 0:
            live_wait(self, duration)
            return
        if not self._law_fired:
            self._reveal_equation_card(eq_card)
        live_wait(self, max(0, duration - 2.0))

    def _beat_emphasis(self, focus, duration, ease_name="soft"):
        """Emphasis beat with spotlight dim of non-focused elements."""
        mob = self.visuals_by_id.get(focus)
        if mob is None:
            live_wait(self, duration)
            return
        dimmed = None
        if HAS_VISUAL_KIT:
            dimmed = visual_kit.spotlight(self, mob, dim=0.25,
                                           run_time=0.4)
        try:
            self.play(Indicate(mob, color=COLOR["accent"],
                               scale_factor=1.15),
                      run_time=min(max(0.4, duration - 0.9), 1.2),
                      rate_func=ease_for(ease_name))
            live_wait(self, max(0, duration - 1.7))
        finally:
            if dimmed and HAS_VISUAL_KIT:
                visual_kit.unspotlight(self, dimmed, run_time=0.4)

    def _beat_analogy(self, focus, duration, ease_name="soft"):
        mob = self.visuals_by_id.get(focus)
        if mob is not None:
            self.play(
                mob.animate(rate_func=ease_for(ease_name),
                            run_time=min(duration, 1.0)).scale(1.08)
            )
            live_wait(self, max(0, duration - 1.0))
        else:
            live_wait(self, duration)

    def _beat_midpoint(self, focus, duration, ease_name="soft"):
        mob = self.visuals_by_id.get(focus)
        if mob is not None:
            focus_on(self, mob, width_frac=0.65, run_time=0.9,
                     ease="land")
            pulse(self, mob, scale=1.10, run_time=0.5)
            self.director.reset(run_time=0.7, ease="soft")
            live_wait(self, max(0, duration - 2.1))
        else:
            live_wait(self, duration)

    def _beat_experiment(self, focus, duration, ease_name="soft"):
        _play_sound(self, "tick", gain_db=-20)
        mob = self.visuals_by_id.get(focus)
        if mob is not None:
            self.play(Indicate(mob, color=COLOR["accent"],
                               scale_factor=1.10),
                      run_time=min(duration, 1.2),
                      rate_func=ease_for(ease_name))
            live_wait(self, max(0, duration - 1.2))
        else:
            live_wait(self, duration)

    def _beat_punchline(self, eq_card, duration):
        _play_sound(self, "chime", gain_db=-16)
        if len(eq_card) > 0:
            self.play(Circumscribe(eq_card, color=COLOR["accent"],
                                   buff=0.25, run_time=1.2))
            live_wait(self, max(0, duration - 1.2))
        else:
            live_wait(self, duration)

    def _beat_cta(self, duration):
        msg = Text("Subscribe for daily physics",
                   font_size=32, color=COLOR["accent"], weight=BOLD)
        msg.move_to([0, zone("cta"), 0])
        fit_to_frame(msg, margin=0.80)
        self.play(FadeIn(msg, scale=0.92,
                         rate_func=rf.ease_out_back), run_time=0.5)
        live_wait(self, max(0, duration - 1.2))
        self.play(FadeOut(msg, shift=UP * 0.20,
                          rate_func=rf.ease_in_cubic), run_time=0.5)

    # -------------------------------------------------- formula sheet
    def _render_formula_sheet(self, formulas, eq_card=None):
        if not isinstance(formulas, list) or not formulas:
            return
        if eq_card is not None and len(eq_card) > 0:
            try:
                self.play(eq_card.animate.set_opacity(0), run_time=0.5)
            except Exception:
                pass
        title = Text("Key formulas", font_size=30,
                     color=COLOR["accent"], weight=BOLD)
        lines = VGroup()
        for f in formulas[:8]:
            lines.add(safe_latex(f, font_size=28, color=WHITE))
        lines.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        content = VGroup(title, lines).arrange(DOWN, aligned_edge=LEFT,
                                                buff=0.35)
        max_w = config.frame_width * 0.88
        if content.width > max_w:
            content.scale_to_fit_width(max_w)
        card = make_card(content, width=config.frame_width * 0.94)
        card.move_to([0, zone("visual"), 0])
        card.set_opacity(0)
        self.play(card.animate.set_opacity(1), run_time=0.8,
                  rate_func=rf.ease_out_cubic)

    # -------------------------------------------------- answers
    def _build_answers(self, options, correct):
        parts = []
        if options:
            opt_fs = 24 if len(options) >= 4 else 26
            opts = VGroup(*[safe_latex(o, opt_fs, WHITE) for o in options])
            if len(options) == 4:
                opts.arrange_in_grid(rows=2, cols=2, buff=(1.0, 0.22))
            else:
                opts.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
            max_w = config.frame_width * 0.88
            if opts.width > max_w:
                opts.scale_to_fit_width(max_w)
            parts.append(opts)
        if correct:
            ans = safe_latex(correct, 26, color=COLOR["force"])
            ans_box = VGroup(
                RoundedRectangle(
                    corner_radius=0.18,
                    width=min(max(ans.width + 0.6, 6.0),
                              config.frame_width * 0.88),
                    height=ans.height + 0.45,
                    fill_color=COLOR["card"], fill_opacity=0.95,
                    stroke_color=COLOR["force"], stroke_width=2.5,
                ),
                ans,
            )
            parts.append(ans_box)
        if not parts:
            return None
        group = VGroup(*parts).arrange(DOWN, buff=0.35)
        max_w = config.frame_width * 0.92
        if group.width > max_w:
            group.scale_to_fit_width(max_w)
        group.move_to([0, zone("visual") - 1.8, 0])
        return group

    # -------------------------------------------------- router
    def _build_primitive(self, item, idx):
        itype = str(item.get("type", "")).lower()
        try:
            if itype in ("field", "vector_field"):
                return self._prim_field(item)
            if itype in ("charge", "particle", "dot"):
                return self._prim_charge(item)
            if itype in ("vector", "arrow", "force"):
                return self._prim_vector(item)
            if itype in ("line", "rod", "segment"):
                return self._prim_line(item)
            if itype == "brace":
                return self._prim_brace(item)
            if itype in ("arc", "angle", "angle_arc"):
                return self._prim_arc(item)
            if itype in ("graph", "function", "curve"):
                return self._prim_graph(item)
            if itype in ("shape", "lens", "circle", "rectangle",
                         "ellipse", "polygon"):
                return self._prim_shape(item)
            if itype == "text":
                return self._prim_text(item)
            if itype == "group":
                sub = VGroup()
                for j, child in enumerate(item.get("children", [])):
                    m = self._build_primitive(child, j)
                    if m is not None:
                        sub.add(m)
                return sub if len(sub) > 0 else None
            if itype == "orbit":
                return self._prim_orbit(item, trail=item.get("trail", False))
            if itype == "phasor":
                return self._prim_phasor(item)
            if itype == "ray":
                return self._prim_ray(item, trail=item.get("trail", False))
            if itype == "particle_gas":
                return self._prim_particle_gas(item)
            if itype == "heat_flow":
                return self._prim_heat_flow(item)
            if itype == "energy_bar":
                return self._prim_energy_bar(item)
            if itype == "wave":
                return self._prim_wave(item)
            if itype == "number_line":
                return self._prim_number_line(item)
            if itype == "pendulum":
                return self._prim_pendulum(item)
            if itype == "spring":
                return self._prim_spring(item)
            if itype == "equation_morph":
                return self._prim_equation_morph(item)
            if itype == "live_label":
                return self._prim_live_label(item)
            if itype == "numerical_chain":
                return self._prim_numerical_chain(item)
            if itype == "warning_box":
                return self._prim_warning_box(item)
            if itype == "quiz_frame":
                return self._prim_quiz_frame(item)
            if itype == "concept_map":
                return self._prim_concept_map(item)
            if itype == "polarized_wave":
                return self._prim_polarized_wave(item)
            if itype == "dipole_radiation":
                return self._prim_dipole_radiation(item)
            if itype == "two_path_interference":
                return self._prim_two_path_interference(item)
            if itype == "driven_oscillator":
                return self._prim_driven_oscillator(item)
            if itype == "wave_pulse":
                return self._prim_wave_pulse(item)
            if itype == "harmonic_decomposition":
                return self._prim_harmonic_decomposition(item)
            if itype == "exponential_atmosphere":
                return self._prim_exponential_atmosphere(item)
            if itype == "energy_landscape":
                return self._prim_energy_landscape(item)
            if itype == "angular_measurement":
                return self._prim_angular_measurement(item)
            if itype == "crystal_lattice":
                return self._prim_crystal_lattice(item)
            if itype == "projectile_motion":
                return self._prim_projectile_motion(item)
            if itype == "damped_oscillator":
                return self._prim_damped_oscillator(item)
            if itype == "radiating_wave":
                return self._prim_radiating_wave(item)
            if itype == "two_source_waves":
                return self._prim_two_source_waves(item)
            if itype == "driven_electron":
                return self._prim_driven_electron(item)
            if itype == "color_triangle":
                return self._prim_color_triangle(item)
            if itype == "thermionic_emission":
                return self._prim_thermionic_emission(item)
            if itype == "beat_pattern":
                return self._prim_beat_pattern(item)
            if itype == "mirror_scene":
                return self._prim_mirror_scene(item)
            if itype == "medium_atoms":
                return self._prim_medium_atoms(item)
            if itype == "field_by_opacity":
                return self._prim_field_by_opacity(item)
            if itype == "polar_pattern":
                return self._prim_polar_pattern(item)
            if itype == "phase_portrait":
                return self._prim_phase_portrait(item)
            if itype == "light_wave_slice":
                return self._prim_light_wave_slice(item)
            if itype == "sliced_wave":
                return self._prim_sliced_wave(item)
            if itype == "spectral_sum":
                return self._prim_spectral_sum(item)
            if itype == "stream_field":
                return self._prim_stream_field(item)
            if itype == "taylor_expand":
                return self._prim_taylor_expand(item)
            if itype == "grid_warp":
                return self._prim_grid_warp(item)
            # ---- artifact catalog v2.2 (12 new primitives)
            if itype == "torsion_rod":
                return self._prim_torsion_rod(item)
            if itype == "rolling_wheel_cycloid":
                return self._prim_rolling_wheel_cycloid(item)
            if itype == "interference_pattern":
                return self._prim_interference_pattern(item)
            if itype == "two_beam_splitter":
                return self._prim_two_beam_splitter(item)
            if itype == "potential_well":
                return self._prim_potential_well(item)
            if itype == "induction_coil":
                return self._prim_induction_coil(item)
            if itype == "compass_grid":
                return self._prim_compass_grid(item)
            if itype == "pipe_with_gauge":
                return self._prim_pipe_with_gauge(item)
            if itype == "pv_diagram_tracker":
                return self._prim_pv_diagram_tracker(item)
            if itype == "wave_packet_tunneling":
                return self._prim_wave_packet_tunneling(item)
            if itype == "bohr_orbit_photon":
                return self._prim_bohr_orbit_photon(item)
            if itype == "accretion_disk":
                return self._prim_accretion_disk(item)
        except Exception as e:
            _log(f"primitive '{itype}' failed: {type(e).__name__}: {e}")
            return None
        return None

    # ============================================================ static
    def _prim_field(self, item):
        direction = str(item.get("direction", "RIGHT")).upper()
        rows = int(item.get("rows", 7))
        op = float(item.get("opacity", 0.4))
        color = resolve_color(item)
        label = item.get("label", "")
        dv = {"RIGHT": RIGHT, "LEFT": LEFT,
              "UP": UP, "DOWN": DOWN}.get(direction, RIGHT)
        field = VGroup()
        fw = config.frame_width * 0.75
        fh = config.frame_height * 0.55
        for y in np.linspace(-fh / 2, fh / 2, rows):
            for x in np.linspace(-fw / 2, fw / 2, 9):
                s = np.array([x, y, 0.0])
                e = s + dv * 0.7
                arr = Arrow(start=s, end=e, buff=0, color=color,
                            stroke_width=2,
                            max_tip_length_to_length_ratio=0.32)
                arr.set_opacity(op)
                field.add(arr)
        if label:
            field.add(safe_latex(label, 30, color)
                      .next_to(field, UP, buff=0.3))
        return field

    def _prim_charge(self, item):
        color = resolve_color(item)
        label = item.get("label", "")
        pos = to_3d(item.get("pos", [0, 0, 0]))
        r = float(item.get("radius", 0.22))
        dot = Dot(point=pos, radius=r, color=color)
        if label:
            return VGroup(dot, safe_latex(label, 24, WHITE)
                          .next_to(dot, UP, buff=0.15))
        return dot

    def _prim_vector(self, item):
        color = resolve_color(item)
        label = item.get("label", "")
        s = to_3d(item.get("start", [0, 0, 0]))
        e = to_3d(item.get("end", [1, 0, 0]))
        arr = Arrow(start=s, end=e, buff=0, color=color, stroke_width=5)
        if label:
            return VGroup(arr, safe_latex(label, 26, color)
                          .next_to(arr.get_end(), RIGHT, buff=0.15))
        return arr

    def _prim_line(self, item):
        color = resolve_color(item)
        label = item.get("label", "")
        s = to_3d(item.get("start", [-1, 0, 0]))
        e = to_3d(item.get("end", [1, 0, 0]))
        if item.get("dashed", False):
            mob = DashedLine(start=s, end=e, color=color)
        else:
            mob = Line(start=s, end=e, color=color, stroke_width=3)
        if label:
            return VGroup(mob, safe_latex(label, 24, color)
                          .next_to(mob, RIGHT, buff=0.15))
        return mob

    def _prim_brace(self, item):
        s = to_3d(item.get("start", [-1.0, -1.0, 0]))
        e = to_3d(item.get("end",   [ 1.0, -1.0, 0]))
        direction = str(item.get("direction", "DOWN")).upper()
        label = item.get("label", "")
        color = resolve_color(item)

        dir_map = {"UP": UP, "DOWN": DOWN, "LEFT": LEFT, "RIGHT": RIGHT}
        dir_vec = dir_map.get(direction, DOWN)

        try:
            brace = BraceBetweenPoints(s, e, direction=dir_vec)
            brace.set_color(color)
        except Exception as exc:
            _log(f"brace failed: {exc}")
            return None

        if label:
            lbl = safe_latex(label, 22, color)
            try:
                lbl.next_to(brace, dir_vec, buff=0.15)
            except Exception:
                lbl.next_to(brace, DOWN, buff=0.15)
            return VGroup(brace, lbl)
        return brace

    def _prim_arc(self, item):
        color = resolve_color(item)
        label = item.get("label", "")
        c = to_3d(item.get("center", [0, 0, 0]))
        arc = Arc(
            radius=float(item.get("radius", 0.8)),
            start_angle=np.radians(float(item.get("start_angle", 0))),
            angle=np.radians(float(item.get("angle", 60))),
            arc_center=c, color=color,
        )
        if label:
            return VGroup(arc, safe_latex(label, 22, color)
                          .next_to(arc, RIGHT, buff=0.15))
        return arc

    def _prim_graph(self, item):
        color = resolve_color(item)
        xr = item.get("x_range", [0, 5])
        yr = item.get("y_range", [-2, 2])
        x_len = float(item.get("x_length", config.frame_width * 0.85))
        y_len = float(item.get("y_length", config.frame_height * 0.30))
        axes = Axes(
            x_range=[xr[0], xr[1], 1], y_range=[yr[0], yr[1], 1],
            x_length=x_len, y_length=y_len,
            axis_config={"color": COLOR["muted"], "stroke_width": 2},
        )
        expr = item.get("expression", "x")
        g = axes.plot(lambda x: eval_expr(expr, x),
                      x_range=[xr[0], xr[1]], color=color, stroke_width=4)
        return VGroup(axes, g)

    def _prim_shape(self, item):
        color = resolve_color(item)
        kind = str(item.get("kind", "rectangle")).lower()
        pos = to_3d(item.get("pos", [0, 0, 0]))
        fo = float(item.get("fill_opacity", 0.2))
        if kind == "rectangle":
            d = item.get("dims", [2.0, 1.0])
            return Rectangle(width=d[0], height=d[1], color=color,
                             fill_opacity=fo).move_to(pos)
        if kind == "ellipse":
            return Ellipse(width=float(item.get("width", 3.0)),
                           height=float(item.get("height", 1.8)),
                           color=color, fill_opacity=fo).move_to(pos)
        if kind == "lens":
            return Ellipse(width=0.8, height=3.6, color=color,
                           fill_color=color, fill_opacity=0.3).move_to(pos)
        if kind == "polygon":
            pts = [to_3d(p) for p in item.get("points", [])]
            if len(pts) >= 3:
                return Polygon(*pts, color=color, fill_opacity=fo)
            return None
        if kind == "circle":
            return Circle(radius=float(item.get("radius", 1.0)),
                          color=color, fill_opacity=fo).move_to(pos)
        return Circle(radius=float(item.get("radius", 1.0)),
                      color=color, stroke_width=3).move_to(pos)

    def _prim_text(self, item):
        color = resolve_color(item)
        txt = str(item.get("text", item.get("label", "")))
        pos = to_3d(item.get("pos", [0, 0, 0]))
        return safe_latex(txt, int(item.get("font_size", 26)),
                          color).move_to(pos)

    # ============================================================ motion
    def _prim_orbit(self, item, trail=False):
        center = to_3d(item.get("center", [0, 0, 0]))
        w = float(item.get("width", config.frame_width * 0.75))
        h = float(item.get("height", config.frame_height * 0.40))
        period = max(2.0, float(item.get("period", 12.0)))
        start_angle = float(item.get("start_angle", 0)) * DEGREES
        body_color = item.get("body_color", COLOR["pos"])
        body_radius = float(item.get("body_radius", 0.18))
        body_label = item.get("body_label", "")
        orbit_color = item.get("orbit_color", COLOR["field"])
        ellipse = Ellipse(width=w, height=h, color=orbit_color,
                          stroke_width=2,
                          stroke_opacity=0.55).move_to(center)
        a, b = w / 2.0, h / 2.0
        gt = self.global_time

        def _pos_fn():
            t = (gt.get_value() / period) * TAU + start_angle
            return center + np.array([a * np.cos(t), b * np.sin(t), 0.0])

        body = Dot(radius=body_radius, color=body_color)
        body.add_updater(lambda m: m.move_to(_pos_fn()))
        group = VGroup(ellipse, body)
        if trail:
            trail_path = TracedPath(
                lambda: _pos_fn(),
                dissipating_time=period * 0.6,
                stroke_color=body_color, stroke_width=3,
                stroke_opacity=[0.9, 0.0],
            )
            group.add(trail_path)
        if body_label:
            lbl = always_redraw(
                lambda: safe_latex(body_label, 24, body_color)
                        .next_to(_pos_fn(), UR, buff=0.12)
            )
            group.add(lbl)
        return group

    def _prim_phasor(self, item):
        radius = float(item.get("radius", 2.5))
        period = max(2.0, float(item.get("period", 10.0)))
        start_angle = float(item.get("start_angle", 0)) * DEGREES
        color = resolve_color(item)
        show_proj = bool(item.get("show_projection", False))
        show_trail = bool(item.get("trail", False))
        circle = Circle(radius=radius, color=COLOR["muted"],
                        stroke_width=1.5, stroke_opacity=0.4)
        origin_dot = Dot(ORIGIN, radius=0.05, color=COLOR["white"])
        gt = self.global_time

        def _angle():
            return (gt.get_value() / period) * TAU + start_angle

        def _end():
            t = _angle()
            return np.array([radius * np.cos(t), radius * np.sin(t), 0.0])

        vector = Dot(radius=body_radius, color=body_color)
        vector.add_updater(lambda m: m.move_to(_pos_fn()))
        group = VGroup(circle, vector, origin_dot)
        if show_proj:
            def _proj_x():
                t = _angle()
                x = radius * np.cos(t)
                y = radius * np.sin(t)
                return DashedLine(start=np.array([x, 0, 0]),
                                  end=np.array([x, y, 0]),
                                  color=color, stroke_width=1.5,
                                  stroke_opacity=0.6)

            def _proj_y():
                t = _angle()
                x = radius * np.cos(t)
                y = radius * np.sin(t)
                return DashedLine(start=np.array([0, y, 0]),
                                  end=np.array([x, y, 0]),
                                  color=color, stroke_width=1.5,
                                  stroke_opacity=0.6)
            group.add(always_redraw(_proj_x), always_redraw(_proj_y))
        if show_trail:
            trail = TracedPath(
                _end, dissipating_time=period * 1.2,
                stroke_color=color, stroke_width=2,
                stroke_opacity=[0.7, 0.0],
            )
            group.add(trail)
        return group

    def _prim_ray(self, item, trail=False):
        s = to_3d(item.get("start", [-5.0, 0, 0]))
        e = to_3d(item.get("end", [5.0, 0, 0]))
        color = resolve_color(item)
        travel_time = max(0.5, float(item.get("travel_time", 3.0)))
        loop = bool(item.get("loop", True))
        line = Line(s, e, color=color, stroke_width=2, stroke_opacity=0.55)
        gt = self.global_time

        def _pos():
            t_raw = gt.get_value() / travel_time
            t = (t_raw % 1.0) if loop else min(1.0, t_raw)
            return s + t * (e - s)
         
        photon = Dot(radius=body_radius, color=body_color)
        photon.add_updater(lambda m: m.move_to(_pos_fn()))
        group = VGroup(line, photon)
        if trail:
            trail_path = TracedPath(
                _pos, dissipating_time=travel_time * 0.4,
                stroke_color=color, stroke_width=2,
                stroke_opacity=[0.9, 0.0],
            )
            group.add(trail_path)
        return group

    def _prim_particle_gas(self, item):
        dims = item.get("container_dims", [10.0, 5.0])
        num = int(item.get("num_particles", 20))
        speed = float(item.get("speed", 1.2))
        color = resolve_color(item)
        w, h = float(dims[0]), float(dims[1])
        half_w, half_h = w / 2.0 - 0.25, h / 2.0 - 0.25
        container = Rectangle(width=w, height=h, color=COLOR["field"],
                              stroke_width=2, fill_opacity=0.04)
        rng = np.random.default_rng(42)
        particles = []
        for _ in range(num):
            pos = np.array([rng.uniform(-half_w, half_w),
                            rng.uniform(-half_h, half_h), 0.0])
            angle = rng.uniform(0, TAU)
            vel = np.array([np.cos(angle), np.sin(angle), 0.0]) * speed
            dot = Dot(pos, radius=0.08, color=color)
            particles.append({"mob": dot, "pos": pos, "vel": vel})
        gt = self.global_time
     
    from manim import DotCloud

    def _prim_particle_gas(self, item):
       dims = item.get("container_dims", [10.0, 5.0])
       num = int(item.get("num_particles", 20))
       speed = float(item.get("speed", 1.2))
       color = resolve_color(item)
       w, h = float(dims[0]), float(dims[1])
       half_w, half_h = w / 2.0 - 0.25, h / 2.0 - 0.25
       container = Rectangle(width=w, height=h, color=COLOR["field"],
                            stroke_width=2, fill_opacity=0.04)
    
    rng = np.random.default_rng(42)
    positions = np.array([
        [rng.uniform(-half_w, half_w), rng.uniform(-half_h, half_h), 0.0]
        for _ in range(num)
    ])
    velocities = np.array([
        [np.cos(rng.uniform(0, TAU)), np.sin(rng.uniform(0, TAU)), 0.0] * speed
        for _ in range(num)
    ])
    
    cloud = DotCloud(positions, color=color, radius=0.08)
    gt = self.global_time
    
    def update_cloud(m):
        t = gt.get_value()
        new_positions = positions + velocities * t
        # Triangle wave bounce
        new_positions[:, 0] = _triangle_wave(new_positions[:, 0], -half_w, half_w)
        new_positions[:, 1] = _triangle_wave(new_positions[:, 1], -half_h, half_h)
        m.set_points(new_positions)
    
    cloud.add_updater(update_cloud)
    return VGroup(container, cloud)

        def _make_updater(pdata, hw=half_w, hh=half_h):
            def upd(m):
                t = gt.get_value()
                x = pdata["pos"][0] + pdata["vel"][0] * t
                y = pdata["pos"][1] + pdata["vel"][1] * t
                x = _triangle_wave(x, -hw, hw)
                y = _triangle_wave(y, -hh, hh)
                m.move_to(np.array([x, y, 0.0]))
            return upd

        for pdata in particles:
            pdata["mob"].add_updater(_make_updater(pdata))
        dots_group = VGroup(*[p["mob"] for p in particles])
        return VGroup(container, dots_group)

    def _prim_heat_flow(self, item):
        s = to_3d(item.get("start", [-3, 1.5, 0]))
        e = to_3d(item.get("end", [3, -1.5, 0]))
        num = int(item.get("num_particles", 10))
        travel_time = max(1.0, float(item.get("travel_time", 3.0)))
        color = resolve_color(item)
        gt = self.global_time
        dots = []
        for i in range(num):
            phase = i / max(1, num)

            def _make_dot(phase=phase):
                t = ((gt.get_value() / travel_time) + phase) % 1.0
                pos = s + t * (e - s)
                return Dot(pos, radius=0.12, color=color) \
                    .set_opacity(0.3 + 0.7 * np.sin(np.pi * t))
            dots.append(always_redraw(_make_dot))
        return VGroup(*dots)

    def _prim_energy_bar(self, item):
        pos = to_3d(item.get("pos", [0, 0, 0]))
        width = float(item.get("width",
                               config.frame_width * 0.7))
        fill = float(np.clip(item.get("fill", 1.0), 0.0, 1.0))
        label = str(item.get("label", ""))
        color = resolve_color(item)
        bg = Rectangle(width=width, height=0.45,
                       stroke_color=COLOR["border"], stroke_width=1.5,
                       fill_opacity=0.10).move_to(pos)
        bar = Rectangle(width=max(0.01, width * fill), height=0.45,
                        stroke_width=0, fill_color=color, fill_opacity=0.85)
        bar.move_to(np.array([bg.get_left()[0] + width * fill / 2.0,
                              pos[1], 0.0]))
        group = VGroup(bg, bar)
        if label:
            group.add(safe_latex(label, 24, WHITE)
                      .next_to(bg, LEFT, buff=0.25))
        return group

    def _prim_wave(self, item):
        xr = item.get("x_range", [-6.0, 6.0])
        amp = float(item.get("amplitude", 0.8))
        freq = float(item.get("frequency", 1.5))
        speed = float(item.get("speed", 2.0))
        color = resolve_color(item)
        x0, x1 = float(xr[0]), float(xr[1])
        y_max = amp * 1.4
        x_len = float(item.get("x_length", config.frame_width * 0.88))
        y_len = float(item.get("y_length", config.frame_height * 0.25))
        axes = Axes(x_range=[x0, x1, 1], y_range=[-y_max, y_max, 1],
                    x_length=x_len, y_length=y_len,
                    axis_config={"color": COLOR["muted"],
                                 "stroke_width": 1.5})
        gt = self.global_time

        def _curve():
            t = gt.get_value()
            return axes.plot(
                lambda x: WavesEngine.traveling_wave(
                    amplitude=amp,
                    wave_number=freq,
                    angular_frequency=speed,
                    phase=0.0,
                    x=x,
                    t=t,
                ),
                x_range=[x0, x1], color=color, stroke_width=3.5,
            )
        return VGroup(axes, always_redraw(_curve))

    def _prim_number_line(self, item):
        x_range = item.get("x_range", [-5, 5])
        length = float(item.get("length", config.frame_width * 0.88))
        nl = NumberLine(x_range=[x_range[0], x_range[1], 1], length=length,
                        color=COLOR["muted"], include_numbers=True,
                        font_size=20)
        gt = self.global_time
        period = max(2.0, float(item.get("period", 10.0)))
        marker = always_redraw(
            lambda: Dot(nl.n2p(x_range[0] + (x_range[1] - x_range[0]) *
                       ((gt.get_value() / period) % 1.0)),
                        radius=0.12, color=COLOR["accent"])
        )
        return VGroup(nl, marker)

    _trajectory_cache = {}

    def _prim_pendulum(self, item):
        pivot = to_3d(item.get("pivot", [0, 2.5, 0]))
        length = float(item.get("length", 3.0))
        start_angle = float(item.get("angle", 30)) * DEGREES
        gravity = float(item.get("gravity", 9.81))
        color = resolve_color(item)
        bob_r = float(item.get("bob_radius", 0.25))
        gt = self.global_time
        duration = max(0.0, float(self.sk.get("duration", 30.0)))
    
    # Cache key
    cache_key = (length, start_angle, gravity, duration)
    if cache_key not in self._trajectory_cache:
        def _derivative(_t, state):
            theta, angular_velocity = state
            return np.array([angular_velocity, -(gravity/length)*np.sin(theta)])
        times, states = integrate_rk4(_derivative, [start_angle, 0.0], duration, 1/120)
        self._trajectory_cache[cache_key] = (times, states)
    
    times, states = self._trajectory_cache[cache_key]
    
    # ... rest of the method

        def _bob_pos():
            t = float(np.clip(gt.get_value(), times[0], times[-1]))
            right = min(np.searchsorted(times, t, side="right"), len(times) - 1)
            left = max(0, right - 1)
            span = times[right] - times[left]
            alpha = (t - times[left]) / span if span else 0.0
            theta = states[left, 0] + alpha * (states[right, 0] - states[left, 0])
            return pivot + length * np.array([np.sin(theta),
                                              -np.cos(theta), 0.0])
        pivot_dot = Dot(pivot, radius=0.08, color=COLOR["white"])
        rod = Dot(radius=body_radius, color=body_color)
        rod.add_updater(lambda m: m.move_to(_pos_fn()))
        bob = Dot(radius=body_radius, color=body_color)
        bob.add_updater(lambda m: m.move_to(_pos_fn()))

        arc = Arc(radius=length,
                  start_angle=-PI / 2 - abs(start_angle),
                  angle=2 * abs(start_angle),
                  arc_center=pivot, color=COLOR["border"],
                  stroke_width=1.2, stroke_opacity=0.4)
        return VGroup(arc, rod, pivot_dot, bob)

    def _prim_spring(self, item):
        anchor = to_3d(item.get("anchor", [0, 2.0, 0]))
        rest = float(item.get("rest_length", 2.0))
        amp = float(item.get("amplitude", 0.6))
        period = max(1.0, float(item.get("period", 2.5)))
        color = resolve_color(item)
        gt = self.global_time

        def _bob_pos():
            t = gt.get_value() * (TAU / period)
            return anchor + DOWN * (rest + amp * np.sin(t))

        def _spring():
            b = _bob_pos()
            zigs = 10
            pts = [anchor]
            dy = (b[1] - anchor[1]) / (zigs + 1)
            for i in range(1, zigs + 1):
                sign = 1 if i % 2 else -1
                pts.append(np.array([anchor[0] + 0.22 * sign,
                                     anchor[1] + dy * i, 0.0]))
            pts.append(b)
            return VMobject().set_points_as_corners(pts) \
                .set_stroke(COLOR["muted"], width=2.5,
        spring = Dot(radius=body_radius, color=body_color)
        spring.add_updater(lambda m: m.move_to(_pos_fn()))
        bob = Dot(radius=body_radius, color=body_color)
        bob.add_updater(lambda m: m.move_to(_pos_fn()))
        return VGroup(spring, bob)

    # ============================================================ self-animated
    def _prim_equation_morph(self, item):
        steps = item.get("steps", [])
        if not isinstance(steps, list) or len(steps) < 2:
            return None
        font_size = int(item.get("font_size", 36))
        color = resolve_color(item)
        pos = to_3d(item.get("pos", [0, zone("equations"), 0]))
        try:
            current = MathTex(steps[0], font_size=font_size, color=color)
        except Exception:
            current = safe_latex(steps[0], font_size, color)
        current.move_to(pos)
        self.play(Write(current, rate_func=smooth), run_time=1.2)
        self.wait(0.3)
        for next_s in steps[1:]:
            try:
                target = MathTex(next_s, font_size=font_size, color=color)
            except Exception:
                target = safe_latex(next_s, font_size, color)
            target.move_to(pos)
            try:
                self.play(TransformMatchingTex(current, target), run_time=1.4)
            except Exception:
                self.play(Transform(current, target), run_time=1.4)
            self.wait(0.3)
        return VGroup(current)

    def _prim_live_label(self, item):
        pos = to_3d(item.get("pos", [0, 0, 0]))
        fmt = str(item.get("format", "t = {v:.1f}"))
        fn = str(item.get("var_fn", "linear(t)"))
        color = resolve_color(item)
        font_size = int(item.get("font_size", 26))
        gt = self.global_time

        def _value():
            t = gt.get_value()
            expr = re.sub(r"\bt\b", f"({t:.6f})", fn)
            try:
                return float(eval(expr, {"__builtins__": None},
                                  {"sin": np.sin, "cos": np.cos,
                                   "tan": np.tan, "pi": np.pi, "e": np.e}))
            except Exception:
                return 0.0

        def _make():
            try:
                txt = fmt.format(v=_value())
            except Exception:
                txt = f"{_value():.2f}"
            return Text(txt, font_size=font_size, color=color).move_to(pos)
        return always_redraw(_make)

    def _prim_numerical_chain(self, item):
        steps = item.get("steps", [])
        if not isinstance(steps, list) or not steps:
            return None
        color = resolve_color(item)
        font_size = int(item.get("font_size", 30))
        pos = to_3d(item.get("pos", [0, zone("chain"), 0]))
        group = VGroup()
        for s in steps:
            group.add(math_or_text(s, font_size, color, force_math=True))
        group.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        max_w = config.frame_width * 0.88
        if group.width > max_w:
            group.scale_to_fit_width(max_w)
        group.move_to(pos)
        for line in group:
            self.play(FadeIn(line, shift=RIGHT * 0.20), run_time=0.7)
            self.wait(0.1)
        return group

    def _prim_warning_box(self, item):
        txt = str(item.get("text", ""))
        pos = to_3d(item.get("pos", [0, zone("hook"), 0]))
        if not txt:
            return None
        lines = textwrap.wrap(txt, width=60) or [txt]
        body = Paragraph(*lines, alignment="LEFT",
                         font_size=26, line_spacing=0.75, color=WHITE)
        icon = Text("!", font_size=32, color=COLOR["bg"], weight=BOLD)
        icon_circle = Circle(radius=0.28, color=COLOR["warn"],
                             fill_color=COLOR["warn"],
                             fill_opacity=1).move_to(icon.get_center())
        icon_glyph = VGroup(icon_circle, icon)
        content = VGroup(icon_glyph, body).arrange(RIGHT, buff=0.35)
        box_w = min(max(content.width + 0.7, 9.0),
                    config.frame_width * 0.92)
        box = VGroup(
            RoundedRectangle(
                corner_radius=0.20,
                width=box_w,
                height=content.height + 0.55,
                fill_color=COLOR["card"], fill_opacity=0.96,
                stroke_color=COLOR["warn"], stroke_width=2.5,
            ),
            content,
        ).move_to(pos)
        if box.width > config.frame_width * 0.94:
            box.scale_to_fit_width(config.frame_width * 0.94)
        self.play(FadeIn(box, shift=UP * 0.20), run_time=0.8)
        return box

    def _prim_quiz_frame(self, item):
        q = str(item.get("question", ""))
        opts = item.get("options", [])
        correct = int(item.get("correct", 0))
        pause = float(item.get("pause", 4.0))
        if not q or len(opts) < 2:
            return None
        q_lines = textwrap.wrap(q, width=56) or [q]
        q_mob = Paragraph(*q_lines, alignment="center", font_size=28,
                          line_spacing=0.8, color=WHITE)
        max_w = config.frame_width * 0.88
        if q_mob.width > max_w:
            q_mob.scale_to_fit_width(max_w)
        q_mob.move_to([0, zone("hook"), 0])
        opt_mobs = []
        for i, o in enumerate(opts[:4]):
            opt_mobs.append(Text(f"({chr(65+i)}) {o}",
                                 font_size=24, color=WHITE))
        opts_vg = VGroup(*opt_mobs).arrange(DOWN, aligned_edge=LEFT,
                                             buff=0.22)
        if opts_vg.width > max_w:
            opts_vg.scale_to_fit_width(max_w)
        opts_vg.move_to([0, zone("visual") - 1.5, 0])
        self.play(FadeIn(q_mob, shift=DOWN * 0.20),
                  FadeIn(opts_vg, shift=UP * 0.15), run_time=1.0)
        self.wait(pause)
        target = opt_mobs[min(correct, len(opt_mobs) - 1)]
        self.play(target.animate.set_color(COLOR["force"]).scale(1.05),
                  run_time=0.6)
        self.play(Circumscribe(target, color=COLOR["force"],
                               buff=0.15, run_time=1.0))
        return VGroup(q_mob, opts_vg)

    def _prim_concept_map(self, item):
        nx = _get_networkx()
        if nx is None:
        _log("networkx not installed; skipping concept_map")
        return None
    # ... rest of method
        nodes = item.get("nodes", [])
        edges = item.get("edges", [])
        if not nodes:
            return None
        pos = to_3d(item.get("pos", [0, zone("visual"), 0]))
        G = nx.DiGraph()
        for n in nodes:
            G.add_node(n["id"])
        for e in edges:
            G.add_edge(e["from"], e["to"], label=e.get("label", ""))
        try:
            layout = nx.circular_layout(G)
        except Exception:
            layout = nx.spring_layout(G, seed=42)
        positions = {
            nid: pos[:2] + np.array([layout[nid][0] * 3.5,
                                     layout[nid][1] * 2.5])
            for nid in G.nodes
        }
        node_mobs = {}
        for n in nodes:
            nid = n["id"]
            col = n.get("color", COLOR["accent"])
            mob = VGroup(
                Circle(radius=0.55, color=col, stroke_width=2.5,
                       fill_color=col, fill_opacity=0.15),
                safe_latex(n.get("label", nid), 24, col),
            ).move_to(np.array([positions[nid][0],
                                positions[nid][1], 0.0]))
            node_mobs[nid] = mob
        edge_mobs = VGroup()
        for e in edges:
            try:
                a = node_mobs[e["from"]].get_center()
                b = node_mobs[e["to"]].get_center()
                arr = Arrow(a, b, buff=0.60, color=COLOR["muted"],
                            stroke_width=2,
                            max_tip_length_to_length_ratio=0.20)
                edge_mobs.add(arr)
                if e.get("label"):
                    lbl = safe_latex(e["label"], 18, COLOR["muted"]) \
                        .move_to((a + b) / 2 + UP * 0.20)
                    edge_mobs.add(lbl)
            except Exception:
                continue
        group = VGroup(edge_mobs, *node_mobs.values())
        self.play(LaggedStart(
            Create(edge_mobs),
            *[FadeIn(m, scale=0.7) for m in node_mobs.values()],
            lag_ratio=0.10), run_time=2.0)
        return group

    # ============================================================ day 4+
    def _prim_polarized_wave(self, item):
        length = float(item.get("length", config.frame_width * 0.85))
        amp = float(item.get("amplitude", 1.0))
        freq = float(item.get("frequency", 1.5))
        color = resolve_color(item)
        angle = float(item.get("angle", 0))
        axes = Axes(
            x_range=[-length/2, length/2, 1],
            y_range=[-amp*1.5, amp*1.5, 1],
            x_length=length, y_length=amp*3,
            axis_config={"color": COLOR["muted"], "stroke_width": 1.5},
        )
        wave1 = axes.plot(lambda x: amp * np.cos(freq * x),
                          x_range=[-length/2, length/2],
                          color=color, stroke_width=3)
        wave2 = axes.plot(lambda x: 0.6 * amp * np.sin(freq * x),
                          x_range=[-length/2, length/2],
                          color=COLOR["pos"], stroke_width=2.5)
        if angle:
            wave2.rotate(angle * DEGREES, about_point=axes.c2p(0, 0))
        return VGroup(axes, wave1, wave2)

    def _prim_dipole_radiation(self, item):
        pos = to_3d(item.get("pos", [0, 0, 0]))
        color = resolve_color(item)
        q1 = Dot(pos + LEFT * 0.3, color=COLOR["pos"])
        q2 = Dot(pos + RIGHT * 0.3, color=COLOR["neg"])
        theta = np.linspace(0, 2 * np.pi, 200)
        r = np.abs(np.sin(theta)) * 2.0
        pts = [pos + np.array([r[i] * np.cos(theta[i]),
                                r[i] * np.sin(theta[i]), 0])
               for i in range(len(theta))]
        lobe = VMobject().set_points_as_corners(pts) \
            .set_stroke(color, width=2)
        return VGroup(q1, q2, lobe)

    def _prim_two_path_interference(self, item):
        s = to_3d(item.get("start", [-3, 1, 0]))
        a = to_3d(item.get("end_a", [3, 1, 0]))
        b = to_3d(item.get("end_b", [3, -1, 0]))
        color = resolve_color(item)
        path_a = Line(s, a, color=color, stroke_width=3)
        path_b = Line(s, b, color=COLOR["accent"], stroke_width=3)
        screen = Line([3.2, -2, 0], [3.2, 2, 0],
                      color=COLOR["white"], stroke_width=2)
        return VGroup(path_a, path_b, screen)

    def _prim_driven_oscillator(self, item):
        anchor = to_3d(item.get("anchor", [0, 2.0, 0]))
        rest = float(item.get("rest_length", 2.0))
        amp = float(item.get("amplitude", 0.6))
        period = max(1.0, float(item.get("period", 2.5)))
        color = resolve_color(item)
        gt = self.global_time

        def _bob_pos():
            t = gt.get_value() * (TAU / period)
            return anchor + DOWN * (rest + amp * np.sin(t))

        def _spring():
            b = _bob_pos()
            zigs = 10
            pts = [anchor]
            dy = (b[1] - anchor[1]) / (zigs + 1)
            for i in range(1, zigs + 1):
                sign = 1 if i % 2 else -1
                pts.append(np.array([anchor[0] + 0.22 * sign,
                                     anchor[1] + dy * i, 0.0]))
            pts.append(b)
            return VMobject().set_points_as_corners(pts) \
                .set_stroke(COLOR["muted"], width=2.5)
        spring = always_redraw(_spring)
        bob = always_redraw(
            lambda: Circle(radius=0.25, color=color,
                           fill_color=color, fill_opacity=0.85)
                    .move_to(_bob_pos()))
        drive = always_redraw(
            lambda: Arrow(_bob_pos() + DOWN * 0.4,
                          _bob_pos() + DOWN * 0.4 + RIGHT * 0.5,
                          color=COLOR["accent"], stroke_width=3,
                          max_tip_length_to_length_ratio=0.3))
        return VGroup(spring, bob, drive)

    def _prim_wave_pulse(self, item):
        length = float(item.get("length", config.frame_width * 0.85))
        speed = float(item.get("speed", 2.0))
        width = float(item.get("width", 1.0))
        color = resolve_color(item)
        gt = self.global_time

        def _pulse():
            t = gt.get_value()
            x = np.linspace(-length/2, length/2, 200)
            y = np.exp(-((x - speed * t) / width) ** 2) \
                * np.sin(5 * (x - speed * t))
            return VMobject().set_points_as_corners(
                [np.array([x[i], y[i] * 0.8, 0]) for i in range(len(x))]
            ).set_stroke(color, width=3)
        return always_redraw(_pulse)

    def _prim_harmonic_decomposition(self, item):
        length = float(item.get("length", config.frame_width * 0.85))
        n_terms = int(item.get("n_terms", 5))
        color = resolve_color(item)
        x_len = length
        y_len = config.frame_height * 0.30
        axes = Axes(
            x_range=[-length/2, length/2, 1], y_range=[-2, 2, 1],
            x_length=x_len, y_length=y_len,
            axis_config={"color": COLOR["muted"], "stroke_width": 1.5},
        )
        gt = self.global_time

        def _wave():
            t = gt.get_value()
            x = np.linspace(-length/2, length/2, 300)
            y = np.zeros_like(x)
            for n in range(1, n_terms + 1, 2):
                y += (4 / (np.pi * n)) * np.sin(n * x)
            y *= np.cos(t * 0.5)
            return VMobject().set_points_as_corners(
                [axes.c2p(x[i], y[i]) for i in range(len(x))]
            ).set_stroke(color, width=3)
        return VGroup(axes, always_redraw(_wave))

    def _prim_exponential_atmosphere(self, item):
        n_particles = int(item.get("num_particles", 40))
        height = float(item.get("height", 4.0))
        scale_h = float(item.get("scale_height", 1.2))
        color = resolve_color(item)
        box = Rectangle(width=6, height=height, color=COLOR["border"],
                        stroke_width=1.5, fill_opacity=0.05)
        rng = np.random.default_rng(42)
        dots = VGroup()
        for _ in range(n_particles):
            y = min(rng.exponential(scale_h), height / 2)
            x = rng.uniform(-2.8, 2.8)
            dots.add(Dot([x, y - height/2 + 0.2, 0],
                         radius=0.06, color=color))
        return VGroup(box, dots)

    def _prim_energy_landscape(self, item):
        width = float(item.get("width", 6.0))
        depth = float(item.get("depth", 2.0))
        color = resolve_color(item)
        axes = Axes(
            x_range=[-width/2, width/2, 1],
            y_range=[-0.5, depth + 0.5, 1],
            x_length=width, y_length=depth,
            axis_config={"color": COLOR["muted"], "stroke_width": 1.5},
        )
        curve = axes.plot(lambda x: x ** 2,
                          x_range=[-width/2, width/2],
                          color=color, stroke_width=4)
        ball = Dot(axes.c2p(0, 0.1), radius=0.15, color=COLOR["accent"])
        return VGroup(axes, curve, ball)

    def _prim_angular_measurement(self, item):
        center = to_3d(item.get("center", [0, 0, 0]))
        radius = float(item.get("radius", 1.5))
        start = float(item.get("start_angle", 0))
        angle = float(item.get("angle", 45))
        color = resolve_color(item)
        label = item.get("label", "")
        arc = Arc(radius=radius, start_angle=start * DEGREES,
                  angle=angle * DEGREES, arc_center=center, color=color)
        end1 = center + radius * np.array([np.cos(start * DEGREES),
                                            np.sin(start * DEGREES), 0])
        end2 = center + radius * np.array(
            [np.cos((start + angle) * DEGREES),
             np.sin((start + angle) * DEGREES), 0])
        r1 = Line(center, end1, color=COLOR["muted"], stroke_width=1.5)
        r2 = Line(center, end2, color=COLOR["muted"], stroke_width=1.5)
        group = VGroup(r1, r2, arc)
        if label:
            mid = (start + angle / 2) * DEGREES
            lbl_pos = center + (radius * 1.15) * np.array(
                [np.cos(mid), np.sin(mid), 0])
            group.add(safe_latex(label, 24, color).move_to(lbl_pos))
        return group

    def _prim_crystal_lattice(self, item):
        rows = int(item.get("rows", 5))
        cols = int(item.get("cols", 5))
        spacing = float(item.get("spacing", 0.8))
        color = resolve_color(item)
        dots = VGroup()
        for i in range(rows):
            for j in range(cols):
                x = (j - (cols - 1) / 2) * spacing
                y = (i - (rows - 1) / 2) * spacing
                dots.add(Dot([x, y, 0], radius=0.08, color=color))
        return dots

    def _prim_projectile_motion(self, item):
        v0 = float(item.get("v0", 5.0))
        angle_deg = float(item.get("angle", 45))
        g = float(item.get("g", 9.8))
        color = resolve_color(item)
        gt = self.global_time
        angle = np.deg2rad(angle_deg)
        t_max = 2 * v0 * np.sin(angle) / g
        flight = MechanicsEngine.projectile_trajectory(v0, angle_deg, dt=0.02, gravity=g, max_time=t_max + 1.0)
        pts = [np.array([x, y, 0.0]) for x, y in flight]
        path = VMobject().set_points_as_corners(pts).set_stroke(color, width=3)

        def _ball_pos():
            t = min(gt.get_value() % (t_max + 1.0), t_max)
            x = v0 * np.cos(angle) * t
            y = v0 * np.sin(angle) * t - 0.5 * g * t ** 2
            if y < 0.0:
                y = 0.0
            return np.array([x, y, 0.0])
        ball = always_redraw(lambda: Dot(_ball_pos(), radius=0.15,
                                          color=COLOR["accent"]))
        return VGroup(path, ball)

    # ============================================================ day 5+
    def _prim_damped_oscillator(self, item):
        anchor = to_3d(item.get("anchor", [0, 2.0, 0]))
        rest = float(item.get("rest_length", 2.0))
        amplitude = float(item.get("amplitude", 0.8))
        omega = float(item.get("omega", 3.0))
        gamma = float(item.get("gamma", 0.4))
        color = resolve_color(item)
        gt = self.global_time

        def _bob_pos():
            t = gt.get_value()
            env = amplitude * np.exp(-gamma * t / 2.0)
            return anchor + DOWN * (rest + env * np.sin(omega * t))

        def _spring():
            b = _bob_pos()
            zigs = 10
            pts = [anchor]
            dy = (b[1] - anchor[1]) / (zigs + 1)
            for i in range(1, zigs + 1):
                sign = 1 if i % 2 else -1
                pts.append(np.array([anchor[0] + 0.22 * sign,
                                     anchor[1] + dy * i, 0.0]))
            pts.append(b)
            return VMobject().set_points_as_corners(pts) \
                .set_stroke(COLOR["muted"], width=2.5)
        spring = always_redraw(_spring)
        bob = always_redraw(
            lambda: Circle(radius=0.25, color=color,
                           fill_color=color, fill_opacity=0.85)
                    .move_to(_bob_pos()))
        return VGroup(spring, bob)

    def _prim_radiating_wave(self, item):
        pos = to_3d(item.get("pos", [-3.0, 0, 0]))
        num_rings = int(item.get("num_rings", 6))
        wavelength = float(item.get("wavelength", 0.8))
        speed = float(item.get("speed", 2.0))
        max_radius = float(item.get("max_radius", 4.0))
        color = resolve_color(item)
        gt = self.global_time
        rings = VGroup()
        for i in range(num_rings):
            phase = i / max(1, num_rings)

            def _make_ring(phase=phase):
                t = gt.get_value()
                r = (t * speed + phase * wavelength * num_rings) % max_radius
                if r < 0.05:
                    return VMobject()
                opacity = max(0.05, 0.85 * (1.0 - r / max_radius))
                return Circle(radius=r, color=color, stroke_width=2,
                              stroke_opacity=opacity).move_to(pos)
            rings.add(always_redraw(_make_ring))
        source_dot = Dot(pos, radius=0.15, color=color)
        return VGroup(source_dot, rings)

    def _prim_two_source_waves(self, item):
        src_a = to_3d(item.get("source_a", [-2.0, 0.0, 0]))
        src_b = to_3d(item.get("source_b", [-2.0, -1.0, 0]))
        num_rings = int(item.get("num_rings", 5))
        wavelength = float(item.get("wavelength", 0.8))
        speed = float(item.get("speed", 1.5))
        max_radius = float(item.get("max_radius", 5.0))
        color_a = item.get("color_a", COLOR["pos"])
        color_b = item.get("color_b", COLOR["field"])
        gt = self.global_time

        def _make_rings(source, color):
            rings = VGroup()
            for i in range(num_rings):
                phase = i / max(1, num_rings)

                def _make_ring(phase=phase, source=source, color=color):
                    t = gt.get_value()
                    r = (t * speed +
                         phase * wavelength * num_rings) % max_radius
                    if r < 0.05:
                        return VMobject()
                    opacity = max(0.05, 0.7 * (1.0 - r / max_radius))
                    return Circle(radius=r, color=color, stroke_width=1.8,
                                  stroke_opacity=opacity).move_to(source)
                rings.add(always_redraw(_make_ring))
            return rings

        rings_a = _make_rings(src_a, color_a)
        rings_b = _make_rings(src_b, color_b)
        dot_a = Dot(src_a, radius=0.12, color=color_a)
        dot_b = Dot(src_b, radius=0.12, color=color_b)
        return VGroup(rings_a, rings_b, dot_a, dot_b)

    def _prim_driven_electron(self, item):
        pos = to_3d(item.get("pos", [-1.0, 0.3, 0]))
        amplitude = float(item.get("amplitude", 0.35))
        period = max(0.5, float(item.get("period", 1.5)))
        phase_lag_deg = float(item.get("phase_lag_deg", 90.0))
        electron_color = item.get("electron_color", COLOR["pos"])
        field_color = item.get("drive_field_color", COLOR["accent"])
        gt = self.global_time

        def _electron_pos():
            t = gt.get_value()
            x = pos[0] + amplitude * np.cos(2 * np.pi * t / period)
            return np.array([x, pos[1], 0])

        def _field_arrow():
            t = gt.get_value()
            phase = np.deg2rad(phase_lag_deg)
            fx = amplitude * 1.2 * np.cos(2 * np.pi * t / period + phase)
            tail = _electron_pos()
            head = tail + np.array([fx, 0.0, 0])
            return Arrow(tail, head, color=field_color, stroke_width=3,
                         buff=0, max_tip_length_to_length_ratio=0.35)

        electron = always_redraw(
            lambda: Dot(_electron_pos(), radius=0.14, color=electron_color))
        field = always_redraw(_field_arrow)
        return VGroup(electron, field)

    def _prim_color_triangle(self, item):
        points = item.get("triangle_points",
                          [[-2.5, -1.5, 0], [2.5, -1.5, 0], [0.5, 2.0, 0]])
        labels = item.get("labels", ["R", "G", "B"])
        colors = item.get("colors", ["#FF4B4B", "#2ECC71", "#3498DB"])
        marker_pos = to_3d(item.get("marker_pos", [0.1, 0.3, 0]))
        marker_color = item.get("marker_color", COLOR["accent"])
        pts = [to_3d(p) for p in points[:3]]
        triangle = Polygon(*pts, color=COLOR["muted"], stroke_width=1.5,
                           fill_opacity=0.08)
        corner_dots = VGroup()
        labels_mobs = VGroup()
        for i, (p, lbl, col) in enumerate(zip(pts, labels, colors)):
            corner_dots.add(Dot(p, radius=0.15, color=col))
            direction = UP if i == 2 else DOWN
            labels_mobs.add(safe_latex(lbl, 30, col)
                            .next_to(p, direction, buff=0.25))
        marker = Dot(marker_pos, radius=0.14, color=marker_color)
        return VGroup(triangle, corner_dots, labels_mobs, marker)

    def _prim_thermionic_emission(self, item):
        pos = to_3d(item.get("pos", [0, -0.5, 0]))
        dims = item.get("box_dims", [4.5, 3.5])
        num_inside = int(item.get("num_electrons", 8))
        num_escaping = int(item.get("num_escaping", 3))
        color = resolve_color(item)
        escaped_color = item.get("escaped_color", COLOR["force"])
        w, h = float(dims[0]), float(dims[1])
        box = Rectangle(width=w, height=h, color=COLOR["field"],
                        stroke_width=2, fill_opacity=0.08).move_to(pos)
        rng = np.random.default_rng(42)
        inside = VGroup()
        for _ in range(num_inside):
            p = np.array([pos[0] + rng.uniform(-w / 2 + 0.4, w / 2 - 0.4),
                          pos[1] + rng.uniform(-h / 2 + 0.4, h / 2 - 0.4),
                          0.0])
            inside.add(Dot(p, radius=0.08, color=color))
        escaping = VGroup()
        arrows = VGroup()
        for _ in range(num_escaping):
            x0 = pos[0] + rng.uniform(-w / 2 + 0.4, w / 2 - 0.4)
            p1 = np.array([x0, pos[1] + h / 2 - 0.05, 0.0])
            p2 = np.array([x0, pos[1] + h / 2 + rng.uniform(0.7, 1.3), 0.0])
            arrows.add(DashedLine(p1, p2, color=escaped_color,
                                  stroke_width=1.5, dash_length=0.08))
            escaping.add(Dot(p2, radius=0.08, color=escaped_color))
        return VGroup(box, inside, arrows, escaping)

    def _prim_beat_pattern(self, item):
        xr = item.get("x_range", [0, 10])
        f1 = float(item.get("f1", 5.0))
        f2 = float(item.get("f2", 4.0))
        amp = float(item.get("amplitude", 1.0))
        color_sum = item.get("color_sum", COLOR["accent"])
        color_a = item.get("color_a", COLOR["field"])
        color_b = item.get("color_b", COLOR["momentum"])
        show_components = bool(item.get("show_components", True))
        x0, x1 = float(xr[0]), float(xr[1])
        x_len = config.frame_width * 0.80
        y_len = config.frame_height * 0.28
        axes = Axes(
            x_range=[x0, x1, 1],
            y_range=[-2.2 * amp, 2.2 * amp, 1],
            x_length=x_len, y_length=y_len,
            axis_config={"color": COLOR["muted"], "stroke_width": 1.5},
        )
        group = VGroup(axes)
        if show_components:
            group.add(axes.plot(lambda x: amp * np.cos(f1 * x),
                                x_range=[x0, x1],
                                color=color_a, stroke_width=1.5))
            group.add(axes.plot(lambda x: amp * np.cos(f2 * x),
                                x_range=[x0, x1],
                                color=color_b, stroke_width=1.5))
        group.add(axes.plot(
            lambda x: amp * (np.cos(f1 * x) + np.cos(f2 * x)),
            x_range=[x0, x1], color=color_sum, stroke_width=3.5))
        return group

    def _prim_mirror_scene(self, item):
        axis_x = float(item.get("axis_x", 0.0))
        left_pos = to_3d(item.get("left_marker", [-1.8, 0.5, 0]))
        right_pos = to_3d(item.get("right_marker", [1.8, 0.5, 0]))
        left_label = str(item.get("left_label", "real"))
        right_label = str(item.get("right_label", "mirror"))
        marker_color = item.get("marker_color", COLOR["pos"])
        mirror_color = item.get("mirror_color", COLOR["field"])
        axis_color = item.get("axis_color", COLOR["accent"])
        axis = DashedLine([axis_x, -3.0, 0], [axis_x, 3.0, 0],
                          color=axis_color, stroke_width=2)
        left_dot = VGroup(
            Dot(left_pos, radius=0.15, color=marker_color),
            safe_latex(left_label, 22, marker_color)
                .next_to(left_pos, UL, buff=0.15))
        right_dot = VGroup(
            Dot(right_pos, radius=0.15, color=mirror_color),
            safe_latex(right_label, 22, mirror_color)
                .next_to(right_pos, UR, buff=0.15))
        return VGroup(axis, left_dot, right_dot)

    def _prim_medium_atoms(self, item):
        num = int(item.get("num_atoms", 4))
        spacing = float(item.get("spacing", 1.6))
        amplitude = float(item.get("amplitude", 0.12))
        period = max(0.5, float(item.get("period", 1.5)))
        pos_color = item.get("pos_color", COLOR["force"])
        neg_color = item.get("neg_color", COLOR["pos"])
        drive_color = item.get("drive_color", COLOR["accent"])
        gt = self.global_time
        atoms = VGroup()
        for i in range(num):
            base_x = (i - (num - 1) / 2.0) * spacing

            def _make_pair(base_x=base_x):
                t = gt.get_value()
                offset = amplitude * np.cos(2 * np.pi * t / period)
                pos_charge = Dot([base_x + offset, 0.22, 0],
                                 radius=0.09, color=pos_color)
                neg_charge = Dot([base_x - offset, -0.22, 0],
                                 radius=0.09, color=neg_color)
                return VGroup(pos_charge, neg_charge)
            atoms.add(always_redraw(_make_pair))
        field_arrow = always_redraw(
            lambda: Arrow([-num * spacing / 2.0 - 1.0, 0, 0],
                          [-num * spacing / 2.0 - 0.3, 0, 0],
                          color=drive_color, stroke_width=3))
        field_label = safe_latex(r"\vec{E}", 24, drive_color) \
            .next_to(field_arrow, LEFT, buff=0.15)
        return VGroup(atoms, field_arrow, field_label)

    # ============================================================ day 5 advanced
    def _prim_field_by_opacity(self, item):
        from manim import DotCloud

        sources = item.get("sources",
                           [[-2.0, 0.0, 0], [2.0, 0.0, 0]])
        wavelength = float(item.get("wavelength", 1.0))
        k = TAU / wavelength
        omega = float(item.get("omega", 3.0))
        density = int(item.get("density", 12))
        width = float(item.get("width", config.frame_width * 0.85))
        height = float(item.get("height", config.frame_height * 0.45))
        color = resolve_color(item)
        show_intensity = bool(item.get("show_intensity", False))
        gt = self.global_time

        step = 1.0 / density
        points = np.array([
            [x, y, 0.0]
            for x in np.arange(-width / 2, width / 2 + step, step)
            for y in np.arange(-height / 2, height / 2 + step, step)
        ])
        radius = step / 2.0
        cloud = DotCloud(points, color=color, radius=radius)
        src_arr = np.array([to_3d(s) for s in sources])

        def _update(dots):
            t = gt.get_value()
            pts = dots.get_points()
            amp = np.zeros(len(pts), dtype=complex)
            for src in src_arr:
                r = np.linalg.norm(pts - src, axis=1) + 1e-6
                amp += np.exp(1j * (k * r - omega * t)) / np.sqrt(r)
            vals = np.abs(amp) if show_intensity else np.real(amp)
            lo, hi = vals.min(), vals.max()
            norm = (vals - lo) / max(hi - lo, 1e-6)
            dots.set_opacity(0.05 + 0.9 * norm)

        cloud.add_updater(_update)
        _update(cloud)
        return cloud

    def _prim_polar_pattern(self, item):
        radius_max = float(item.get("radius_max", 3.0))
        pattern_type = str(item.get("pattern", "dipole"))
        sweep = bool(item.get("sweep", True))
        color = resolve_color(item)

        plane = PolarPlane(radius_max=radius_max, size=5.8,
                           background_line_style={"stroke_opacity": 0.28})
        try:
            plane.add_coordinates()
        except Exception:
            pass

        if pattern_type == "dipole":
            def _fn(theta, phase=0.0):
                return 1.0 + 1.1 * max(
                    0.0, np.cos(theta - phase)) ** 4 \
                    + 0.25 * np.cos(4 * theta) ** 2
        elif pattern_type == "single":
            def _fn(theta, phase=0.0):
                return 2.0 * abs(np.sin(theta - phase))
        else:
            def _fn(theta, phase=0.0):
                return 1.0

        if sweep:
            phase = ValueTracker(-0.35)

            def _pattern():
                return plane.plot_polar_graph(
                    lambda t: _fn(t, phase.get_value()),
                    theta_range=[0, TAU], color=color, stroke_width=5,
                )
            pattern = always_redraw(_pattern)

            def _beam():
                return Arrow(
                    plane.polar_to_point(0.0, 0.0),
                    plane.polar_to_point(phase.get_value(),
                                         radius_max * 0.92),
                    buff=0, color=COLOR["force"], stroke_width=4,
                )
            beam = always_redraw(_beam)
            phase.add_updater(lambda m, dt: m.increment_value(0.35 * dt))
            return VGroup(plane, pattern, beam, phase)
        else:
            pattern = plane.plot_polar_graph(_fn, theta_range=[0, TAU],
                                             color=color, stroke_width=5)
            return VGroup(plane, pattern)

    def _prim_phase_portrait(self, item):
        amplitude = float(item.get("amplitude", 1.7))
        gamma = float(item.get("gamma", 0.28))
        omega = float(item.get("omega", 2.6))
        color_spring = item.get("spring_color", "#3498DB")
        color_phase = item.get("phase_color", "#F1C40F")
        gt = self.global_time

        def _x(t):
            return amplitude * np.exp(-gamma * t) * np.cos(omega * t)

        def _v(t):
            return amplitude * np.exp(-gamma * t) * (
                -gamma * np.cos(omega * t) - omega * np.sin(omega * t))

        anchor = np.array([-4.5, 0.0, 0.0])
        rest = 1.6

        def _bob_pos():
            return np.array([-4.5 + _x(gt.get_value()),
                             -rest - 0.5, 0.0])

        def _spring():
            b = _bob_pos()
            pts = [anchor]
            zigs = 10
            for i in range(1, zigs + 1):
                s = i / (zigs + 1)
                y = anchor[1] + s * (b[1] - anchor[1])
                offset = 0.18 * (1 if i % 2 else -1)
                pts.append(np.array([anchor[0] + offset, y, 0]))
            pts.append(b)
            return VMobject().set_points_as_corners(pts) \
                .set_stroke(COLOR["muted"], width=2.5)

        spring = always_redraw(_spring)
        bob = always_redraw(
            lambda: Circle(radius=0.22, color=color_spring,
                           fill_color=color_spring, fill_opacity=0.85)
                    .move_to(_bob_pos()))
        wall = Rectangle(width=0.3, height=2.6, color=COLOR["muted"],
                         fill_opacity=0.6).move_to(anchor + LEFT * 0.15)
        pivot_dot = Dot(anchor, radius=0.08, color=COLOR["white"])

        axes = Axes(
            x_range=[-amplitude * 1.2, amplitude * 1.2, 0.5],
            y_range=[-amplitude * omega * 1.2,
                     amplitude * omega * 1.2, 1.0],
            x_length=5.5, y_length=5.5,
            axis_config={"color": COLOR["muted"], "stroke_width": 2},
        ).shift(np.array([4.0, 0.0, 0.0]))
        x_label = safe_latex("x", 22, color_spring).next_to(
            axes.x_axis.get_end(), DOWN, buff=0.2)
        v_label = safe_latex(r"\dot{x}", 22, color_phase).next_to(
            axes.y_axis.get_end(), LEFT, buff=0.2)

        def _phase_point():
            t = gt.get_value()
            return axes.c2p(_x(t), _v(t))

        phase_dot = always_redraw(
            lambda: Dot(_phase_point(), radius=0.10, color=color_phase))
        phase_trail = TracedPath(
            lambda: _phase_point(),
            stroke_color=color_phase, stroke_width=2.5,
            stroke_opacity=[0.9, 0.2],
        )
        return VGroup(wall, spring, pivot_dot, bob,
                      axes, x_label, v_label, phase_trail, phase_dot)

    def _prim_light_wave_slice(self, item):
        if not is_opengl():
            _log("light_wave_slice requires OpenGL — falling back to "
                 "field_by_opacity.")
            fallback = dict(item)
            fallback["type"] = "field_by_opacity"
            fallback.setdefault(
                "sources",
                item.get("point_sources", [[-2.0, 0, 0], [2.0, 0, 0]]))
            fallback.setdefault("wavelength",
                                1.0 / max(float(item.get("wave_number", 3.0)),
                                          0.01))
            return self._prim_field_by_opacity(fallback)
        return self._prim_field_by_opacity({
            "type": "field_by_opacity",
            "sources": item.get("point_sources",
                                [[-2.0, 0, 0], [2.0, 0, 0]])[:32],
            "wavelength": 1.0 / max(
                float(item.get("wave_number", 3.0)), 0.01),
            "omega": float(item.get("frequency", 0.5)) * TAU,
            "density": int(item.get("density", 16)),
            "show_intensity": bool(item.get("show_intensity", False)),
            "color": resolve_color(item),
        })

    def _prim_sliced_wave(self, item):
        origin = to_3d(item.get("interface_origin", [0, 0, 0]))
        index = float(item.get("index", 1.5))
        c_vac = float(item.get("c", 2.0))
        freq = float(item.get("frequency", 0.25))
        amp = float(item.get("amplitude", 1.0))
        color = resolve_color(item)
        gt = self.global_time

        x_range = item.get("x_range", [-5.0, 5.0])
        x0, x1 = float(x_range[0]), float(x_range[1])
        k_vac = TAU * freq / max(c_vac, 0.01)
        omega = TAU * freq
        int_x = float(origin[0])

        axes = Axes(
            x_range=[x0, x1, 1], y_range=[-amp * 1.5, amp * 1.5, 1],
            x_length=config.frame_width * 0.85,
            y_length=config.frame_height * 0.22,
            axis_config={"color": COLOR["muted"], "stroke_width": 1.5},
        )
        interface_line = DashedLine(
            axes.c2p(int_x, -amp * 1.4),
            axes.c2p(int_x, amp * 1.4),
            color=COLOR["accent"], stroke_width=2,
        )
        label = safe_latex(f"n = {index:.2f}", 20, COLOR["accent"]) \
            .next_to(interface_line, UP, buff=0.15)

        def _wave():
            t = gt.get_value()
            phase_at_int = k_vac * int_x - omega * t

            def _y(x):
                if x < int_x:
                    return amp * np.sin(k_vac * x - omega * t)
                else:
                    k_med = k_vac * index
                    return amp * np.sin(phase_at_int + k_med * (x - int_x))

            return axes.plot(_y, x_range=[x0, x1],
                             color=color, stroke_width=3.5)

        return VGroup(axes, interface_line, label, always_redraw(_wave))

    def _prim_spectral_sum(self, item):
        components = item.get("components", [
            {"amplitude": 1.0, "frequency": 5.0, "phase": 0.0,
             "color": "#3498DB"},
            {"amplitude": 0.5, "frequency": 4.0, "phase": 0.0,
             "color": "#F1C40F"},
        ])
        show_components = bool(item.get("show_components", True))
        show_sum = bool(item.get("show_sum", True))
        x_max = float(item.get("x_max", 6.0))

        x_len = config.frame_width * 0.80
        y_len = config.frame_height * 0.35
        axes = Axes(
            x_range=[0, x_max, 1], y_range=[-2.5, 2.5, 1],
            x_length=x_len, y_length=y_len,
            axis_config={"color": COLOR["muted"], "stroke_width": 1.5},
        )
        group = VGroup(axes)

        if show_components:
            for c in components:
                a = float(c.get("amplitude", 1.0))
                f = float(c.get("frequency", 1.0))
                ph = float(c.get("phase", 0.0))
                col = c.get("color", COLOR["field"])
                group.add(axes.plot(
                    lambda x, a=a, f=f, ph=ph: a * np.sin(f * x + ph),
                    x_range=[0, x_max], color=col, stroke_width=1.8,
                ))

        if show_sum:
            def _sum(x):
                total = 0.0
                for c in components:
                    total += float(c.get("amplitude", 1.0)) * np.sin(
                        float(c.get("frequency", 1.0)) * x
                        + float(c.get("phase", 0.0)))
                return total

            group.add(axes.plot(_sum, x_range=[0, x_max],
                                color=COLOR["accent"], stroke_width=4))
        return group

    def _prim_stream_field(self, item):
        if not HAS_STREAMLINES:
            _log("StreamLines unavailable — falling back to field_by_opacity")
            return self._prim_field_by_opacity(item)

        charges = item.get("charges", [
            {"pos": [-1.0, 0.0, 0], "charge": 1},
            {"pos": [1.0, 0.0, 0], "charge": -1},
        ])
        xr = item.get("x_range", [-4.0, 4.0])
        yr = item.get("y_range", [-3.0, 3.0])
        anchors = int(item.get("max_anchors_per_line", 30))
        color = resolve_color(item)

        field = _dipole_field(charges)
        stream = StreamLines(
            field, color=color, stroke_width=2,
            x_range=[xr[0], xr[1], 0.6],
            y_range=[yr[0], yr[1], 0.6],
            padding=1.0,
            max_anchors_per_line=anchors,
        )
        charge_dots = VGroup()
        for c in charges:
            p = to_3d(c["pos"])
            col = COLOR["pos"] if float(c.get("charge", 1)) > 0 \
                  else COLOR["neg"]
            lbl = "+" if float(c.get("charge", 1)) > 0 else "−"
            charge_dots.add(VGroup(
                Dot(p, radius=0.15, color=col),
                safe_latex(lbl, 22, WHITE).move_to(p),
            ))
        return VGroup(stream, charge_dots)

    # ============================================================ v2 primitives
    def _prim_taylor_expand(self, item):
        from math import comb

        expr = str(item.get("expr", "sin(x)"))
        xr = item.get("x_range", [-3.0, 3.0])
        max_terms = min(int(item.get("max_terms", 6)), 6)
        term_time = float(item.get("term_time", 1.5))
        color = resolve_color(item)
        x0, x1 = float(xr[0]), float(xr[1])

        def _f(x):
            return eval_expr(expr, x)

        coeffs = []
        h = 1e-3
        for k in range(0, max_terms + 1):
            if k == 0:
                coeffs.append(_f(0.0))
                continue
            total = 0.0
            for j in range(k + 1):
                total += ((-1) ** j) * comb(k, j) * _f((k / 2.0 - j) * h)
            deriv = total / (h ** k)
            coeffs.append(deriv / math.factorial(k))

        def _approx(n_terms):
            return lambda x: sum(c * (x ** k)
                                 for k, c in enumerate(coeffs[:n_terms]))

        x_len = config.frame_width * 0.85
        y_len = config.frame_height * 0.32
        axes = Axes(
            x_range=[x0, x1, 0.5], y_range=[-2.5, 2.5, 1],
            x_length=x_len, y_length=y_len,
            axis_config={"color": COLOR["muted"], "stroke_width": 1.5},
        )
        group = VGroup(axes)
        target = axes.plot(_f, x_range=[x0, x1],
                           color=COLOR["muted"], stroke_width=3,
                           stroke_opacity=0.5)
        group.add(target)

        for k in range(1, max_terms + 1):
            try:
                approx = axes.plot(_approx(k),
                                   x_range=[x0, x1],
                                   color=color, stroke_width=4)
                self.play(Create(approx), run_time=term_time,
                          rate_func=rf.ease_out_cubic)
                group.add(approx)
            except Exception as e:
                _log(f"taylor term {k} failed: {e}")
                break
        return group

    def _prim_grid_warp(self, item):
        mass_pos = to_3d(item.get("mass_pos", [0.0, 0.0, 0.0]))
        mass_str = float(item.get("mass", 1.5))
        grid_size = float(item.get("grid_size",
                                   config.frame_width * 0.75))
        divisions = int(item.get("divisions", 20))
        color = resolve_color(item)

        def _warp_point(p):
            r = float(np.sqrt((p[0] - mass_pos[0]) ** 2
                              + (p[1] - mass_pos[1]) ** 2)) + 1e-3
            warp = mass_str / (r + 0.4)
            return np.array([p[0], p[1] - warp, 0.0])

        lines = VGroup()
        half = grid_size / 2.0
        step = grid_size / max(1, divisions)

        for y in np.arange(-half, half + step / 2, step):
            pts = [_warp_point(np.array([x, y, 0.0]))
                   for x in np.linspace(-half, half, 60)]
            lines.add(VMobject().set_points_as_corners(pts)
                      .set_stroke(color, width=1.2, opacity=0.55))

        for x in np.arange(-half, half + step / 2, step):
            pts = [_warp_point(np.array([x, y, 0.0]))
                   for y in np.linspace(-half, half, 60)]
            lines.add(VMobject().set_points_as_corners(pts)
                      .set_stroke(color, width=1.2, opacity=0.55))

        mass_dot = Dot(mass_pos, radius=0.28, color=COLOR["accent"])
        mass_label = safe_latex("M", 26, COLOR["accent"]) \
            .next_to(mass_dot, UR, buff=0.15)
        return VGroup(lines, mass_dot, mass_label)

    # ============================================================ artifact catalog v2.2
    def _prim_torsion_rod(self, item):
        """Bifilar suspension: two parallel strings hold a rod; as the rod
        rotates by angle theta, the strings twist inward, lifting it."""
        pos = to_3d(item.get("pos", [0, 0, 0]))
        length = float(item.get("length", 2.4))
        height = float(item.get("height", 3.0))
        start_angle = float(item.get("angle", 30)) * DEGREES
        period = max(1.5, float(item.get("period", 4.0)))
        color = resolve_color(item)
        str_color = item.get("string_color", COLOR["muted"])
        gt = self.global_time

        top_left = pos + np.array([-length / 2, height / 2, 0])
        top_right = pos + np.array([length / 2, height / 2, 0])

        def _angle():
            return start_angle * np.cos(gt.get_value() * TAU / period)

        def _bottom(side):
            a = _angle()
            x_off = side * (length / 2) * np.cos(a)
            y_off = -height / 2 + 0.35 * abs(np.sin(a))
            return pos + np.array([x_off, y_off, 0])

        def _string_l():
            return Line(top_left, _bottom(-1),
                        color=str_color, stroke_width=2)

        def _string_r():
            return Line(top_right, _bottom(1),
                        color=str_color, stroke_width=2)

        def _rod():
            return Line(_bottom(-1), _bottom(1),
                        color=color, stroke_width=8)

        ceiling = Line(pos + np.array([-length, height / 2, 0]),
                       pos + np.array([length, height / 2, 0]),
                       color=COLOR["border"], stroke_width=2)

        return VGroup(ceiling,
                      always_redraw(_string_l),
                      always_redraw(_string_r),
                      always_redraw(_rod))

    def _prim_rolling_wheel_cycloid(self, item):
        """Wheel rolling without slipping; a point on the rim traces a cycloid."""
        radius = float(item.get("radius", 1.2))
        speed = float(item.get("speed", 2.0))
        period = max(4.0, float(item.get("period", 12.0)))
        color = resolve_color(item)
        dot_color = item.get("dot_color", COLOR["pos"])
        track_color = item.get("track_color", COLOR["muted"])
        gt = self.global_time

        fw = config.frame_width * 0.85
        x0 = -fw / 2 + 0.4
        x1 = fw / 2 - 0.4
        span = x1 - x0

        ground = Line([x0 - 0.4, -radius, 0],
                      [x1 + 0.4, -radius, 0],
                      color=track_color, stroke_width=2)

        def _cx():
            t = gt.get_value() / period
            return x0 + (t * speed) % span

        def _wheel():
            return Circle(radius=radius, color=color, stroke_width=3) \
                .move_to(np.array([_cx(), 0.0, 0.0]))

        def _rim_pos():
            xc = _cx()
            theta = -xc / radius
            return np.array([xc + radius * np.sin(theta),
                             -radius + radius * np.cos(theta),
                             0.0])

        wheel = always_redraw(_wheel)
        dot = always_redraw(
            lambda: Dot(_rim_pos(), radius=0.14, color=dot_color))
        trail = TracedPath(
            lambda: _rim_pos(),
            stroke_color=dot_color,
            stroke_width=2.5,
            dissipating_time=period * 4.0,
        )
        return VGroup(ground, trail, wheel, dot)

    def _prim_interference_pattern(self, item):
        """Alternating bright/dark vertical bands (thin film / N-slit fringe)."""
        width = float(item.get("width", config.frame_width * 0.85))
        height = float(item.get("height", config.frame_height * 0.35))
        num_bright = int(item.get("num_bright", 6))
        bright_color = item.get("bright_color", COLOR["accent"])
        dark_color = item.get("dark_color", "#050510")
        pos = to_3d(item.get("pos", [0, 0, 0]))

        total_bands = num_bright * 2
        band_w = width / total_bands
        bands = VGroup()
        for i in range(total_bands):
            x = -width / 2 + band_w * (i + 0.5)
            is_bright = (i % 2 == 0)
            col = bright_color if is_bright else dark_color
            op = 0.95 if is_bright else 0.85
            band = Rectangle(width=band_w, height=height,
                             stroke_width=0,
                             fill_color=col, fill_opacity=op)
            band.move_to(pos + np.array([x, 0, 0]))
            bands.add(band)
        return bands

    def _prim_two_beam_splitter(self, item):
        """Michelson interferometer: beam splits at half-silvered mirror."""
        pos = to_3d(item.get("pos", [0, 0, 0]))
        arm = float(item.get("arm_length", 2.0))
        speed = float(item.get("speed", 1.6))
        color = item.get("beam_color", COLOR["accent"])
        mirror_color = item.get("mirror_color", COLOR["muted"])
        splitter_color = item.get("splitter_color", COLOR["field"])
        gt = self.global_time

        splitter = Line(pos + np.array([-0.55, -0.55, 0]),
                        pos + np.array([0.55, 0.55, 0]),
                        color=splitter_color, stroke_width=3)
        m_top = Line(pos + np.array([-0.55, arm, 0]),
                     pos + np.array([0.55, arm, 0]),
                     color=mirror_color, stroke_width=5)
        m_right = Line(pos + np.array([arm, -0.55, 0]),
                       pos + np.array([arm, 0.55, 0]),
                       color=mirror_color, stroke_width=5)

        incoming = Arrow(pos + np.array([-arm * 0.75, -arm * 0.75, 0]),
                         pos, buff=0, color=color, stroke_width=3)
        outgoing = Arrow(pos, pos + np.array([-arm * 0.75, arm * 0.75, 0]),
                         buff=0, color=color, stroke_width=3)

        path_up = Line(pos, pos + np.array([0, arm, 0]),
                       color=color, stroke_width=1.5, stroke_opacity=0.5)
        path_right = Line(pos, pos + np.array([arm, 0, 0]),
                          color=COLOR["pos"], stroke_width=1.5,
                          stroke_opacity=0.5)

        def _photon_up():
            t = (gt.get_value() * speed) % 4.0
            if t < 1:
                p = pos + np.array([0, arm * t, 0])
            elif t < 2:
                p = pos + np.array([0, arm * (2 - t), 0])
            else:
                p = pos + np.array([0, arm, 0])
            return Dot(p, radius=0.09, color=color)

        def _photon_right():
            t = (gt.get_value() * speed + 1.5) % 4.0
            if t < 1:
                p = pos + np.array([arm * t, 0, 0])
            elif t < 2:
                p = pos + np.array([arm * (2 - t), 0, 0])
            else:
                p = pos + np.array([arm, 0, 0])
            return Dot(p, radius=0.09, color=COLOR["pos"])

        return VGroup(splitter, m_top, m_right,
                      incoming, outgoing, path_up, path_right,
                      always_redraw(_photon_up),
                      always_redraw(_photon_right))

    def _prim_potential_well(self, item):
        """Infinite potential well with standing-wave eigenfunction psi_n."""
        width = float(item.get("width", 4.0))
        height = float(item.get("height", 3.5))
        n_state = max(1, int(item.get("n_state", 1)))
        color = resolve_color(item)
        wall_color = item.get("wall_color", COLOR["muted"])
        pos = to_3d(item.get("pos", [0, 0, 0]))

        left = Line(pos + np.array([-width / 2, -height / 2, 0]),
                    pos + np.array([-width / 2, height / 2, 0]),
                    color=wall_color, stroke_width=4)
        right = Line(pos + np.array([width / 2, -height / 2, 0]),
                     pos + np.array([width / 2, height / 2, 0]),
                     color=wall_color, stroke_width=4)
        floor = Line(pos + np.array([-width / 2, -height / 2, 0]),
                     pos + np.array([width / 2, -height / 2, 0]),
                     color=wall_color, stroke_width=1.5)

        amp = height * 0.32
        N = 120
        pts = []
        for i in range(N + 1):
            xl = width * i / N
            yl = amp * np.sin(n_state * np.pi * xl / width)
            pts.append(pos + np.array([-width / 2 + xl, -height / 2 + yl, 0]))
        wave = VMobject().set_points_as_corners(pts).set_stroke(color, width=3.5)

        lbl = safe_latex(f"n = {n_state}", 22, color).move_to(
            pos + np.array([width / 2 + 0.7, height / 2 - 0.3, 0]))
        return VGroup(left, right, floor, wave, lbl)

    def _prim_induction_coil(self, item):
        """Bar magnet sliding through a stationary coil."""
        pos = to_3d(item.get("pos", [0, 0, 0]))
        coil_r = float(item.get("coil_radius", 1.4))
        magnet_len = float(item.get("magnet_length", 1.6))
        speed = float(item.get("speed", 1.1))
        coil_color = item.get("coil_color", COLOR["field"])
        n_color = item.get("n_color", COLOR["pos"])
        s_color = item.get("s_color", COLOR["neg"])
        cur_color = item.get("current_color", COLOR["accent"])
        gt = self.global_time

        num_turns = 6
        coil = VGroup()
        for i in range(num_turns):
            dx = (i - (num_turns - 1) / 2) * 0.13
            e = Ellipse(width=0.45, height=coil_r * 2,
                        color=coil_color, stroke_width=2.5)
            e.move_to(pos + np.array([dx, 0, 0]))
            coil.add(e)

        def _mx():
            span = coil_r * 4
            t = gt.get_value() * speed
            return -span / 2 + (t % span)

        def _magnet():
            c = pos + np.array([_mx(), 0, 0])
            half = magnet_len / 2
            north = Rectangle(width=magnet_len / 2, height=0.5,
                              color=n_color, fill_color=n_color,
                              fill_opacity=0.8, stroke_width=0)
            south = Rectangle(width=magnet_len / 2, height=0.5,
                              color=s_color, fill_color=s_color,
                              fill_opacity=0.8, stroke_width=0)
            north.move_to(c + np.array([half / 2, 0, 0]))
            south.move_to(c + np.array([-half / 2, 0, 0]))
            nl = safe_latex("N", 20, WHITE).move_to(north.get_center())
            sl = safe_latex("S", 20, WHITE).move_to(south.get_center())
            return VGroup(north, south, nl, sl)

        def _cur():
            x_mag = _mx()
            sign = 1 if x_mag < 0 else -1
            tail = pos + np.array([0, coil_r + 0.6, 0])
            head = tail + np.array([sign * 1.3, 0, 0])
            return Arrow(tail, head, buff=0, color=cur_color, stroke_width=4)

        return VGroup(coil, always_redraw(_magnet), always_redraw(_cur))

    def _prim_compass_grid(self, item):
        """Grid of compass needles that align with a magnetic dipole field."""
        magnet_pos = to_3d(item.get("magnet_pos", [0, 0, 0]))
        rows = int(item.get("rows", 6))
        cols = int(item.get("cols", 9))
        spacing = float(item.get("spacing", 0.9))
        needle_color = item.get("needle_color", COLOR["field"])
        magnet_color = item.get("magnet_color", COLOR["warn"])
        pos = to_3d(item.get("pos", [0, 0, 0]))

        m_vec = np.array([1.0, 0.0, 0.0])

        def _field_at(p):
            r = p - magnet_pos
            r_mag = float(np.linalg.norm(r)) + 1e-3
            r_hat = r / r_mag
            return (3.0 * np.dot(m_vec, r_hat) * r_hat - m_vec) / (r_mag ** 3)

        needles = VGroup()
        for i in range(rows):
            for j in range(cols):
                x = (j - (cols - 1) / 2) * spacing
                y = (i - (rows - 1) / 2) * spacing
                p = pos + np.array([x, y, 0])
                if np.linalg.norm(p - magnet_pos) < 0.8:
                    continue
                f = _field_at(p)
                fm = float(np.linalg.norm(f))
                if fm < 1e-3:
                    continue
                fh = f / fm
                scale = 0.30
                arr = Arrow(p - fh * scale / 2, p + fh * scale / 2, buff=0,
                            color=needle_color, stroke_width=2,
                            max_tip_length_to_length_ratio=0.45)
                arr.set_opacity(min(1.0, fm * 0.25 + 0.20))
                needles.add(arr)

        magnet = Rectangle(width=1.2, height=0.35,
                           color=magnet_color, fill_color=magnet_color,
                           fill_opacity=0.85, stroke_width=0)
        magnet.move_to(magnet_pos)
        nl = safe_latex("N", 22, WHITE).move_to(
            magnet.get_left() + np.array([0.20, 0, 0]))
        sl = safe_latex("S", 22, WHITE).move_to(
            magnet.get_right() + np.array([-0.20, 0, 0]))
        return VGroup(needles, magnet, nl, sl)

    def _prim_pipe_with_gauge(self, item):
        """Horizontal pipe with a Venturi constriction."""
        length = float(item.get("length", config.frame_width * 0.72))
        inlet_h = float(item.get("inlet_height", 1.8))
        throat_h = float(item.get("throat_height", 0.9))
        pipe_color = item.get("pipe_color", COLOR["field"])
        fluid_color = item.get("fluid_color", COLOR["cold"])
        gauge_color = item.get("gauge_color", COLOR["accent"])
        pos = to_3d(item.get("pos", [0, -0.8, 0]))
        gt = self.global_time

        x0 = pos[0] - length / 2
        x1 = pos[0] + length / 2
        x_mid = pos[0]
        throat_w = length * 0.22

        def _half_h(x):
            dx = abs(x - x_mid)
            if dx <= throat_w * 0.5:
                return throat_h / 2.0
            if dx >= throat_w * 1.5:
                return inlet_h / 2.0
            t = (dx - throat_w * 0.5) / throat_w
            return (throat_h / 2.0) + (inlet_h - throat_h) / 2.0 * t

        N = 100
        xs = np.linspace(x0, x1, N + 1)
        tops = [np.array([x, pos[1] + _half_h(x), 0]) for x in xs]
        bots = [np.array([x, pos[1] - _half_h(x), 0]) for x in xs]
        top_curve = VMobject().set_points_as_corners(tops) \
            .set_stroke(pipe_color, width=4)
        bot_curve = VMobject().set_points_as_corners(bots) \
            .set_stroke(pipe_color, width=4)

        def _flow(phase):
            def _mk():
                t = (gt.get_value() * 1.3 + phase) % 1.0
                x = x0 + (x1 - x0) * t
                return Dot(np.array([x, pos[1], 0]),
                           radius=0.08, color=fluid_color)
            return always_redraw(_mk)

        flow = VGroup(*[_flow(i / 16.0) for i in range(16)])

        gauge_top = pos[1] + inlet_h / 2 + 1.0

        def _gauge(x, h, label):
            box = Rectangle(width=0.35, height=1.2,
                            color=gauge_color, stroke_width=1.5)
            box.move_to(np.array([x, gauge_top + 0.6, 0]))
            liquid = Rectangle(width=0.28, height=h,
                               color=gauge_color, fill_color=gauge_color,
                               fill_opacity=0.8, stroke_width=0)
            liquid.move_to(np.array([x, gauge_top + 0.6 - 0.6 + h / 2, 0]))
            stem = Line(np.array([x, pos[1] + inlet_h / 2, 0]),
                        np.array([x, gauge_top, 0]),
                        color=gauge_color, stroke_width=1.5)
            lbl = safe_latex(label, 18, gauge_color).move_to(
                np.array([x, gauge_top + 1.35, 0]))
            return VGroup(box, liquid, stem, lbl)

        gauges = VGroup(
            _gauge(x0 + length * 0.15, 1.0, "P_1"),
            _gauge(x_mid, 0.3, "P_2"),
        )
        return VGroup(top_curve, bot_curve, flow, gauges)

    def _prim_pv_diagram_tracker(self, item):
        """P-V diagram of a Carnot-like cycle."""
        width = float(item.get("width", 4.5))
        height = float(item.get("height", 3.6))
        period = max(6.0, float(item.get("period", 12.0)))
        color = resolve_color(item)
        dot_color = item.get("dot_color", COLOR["accent"])
        pos = to_3d(item.get("pos", [0, 0, 0]))
        gt = self.global_time

        axes = Axes(
            x_range=[0, 4, 1], y_range=[0, 4, 1],
            x_length=width, y_length=height,
            axis_config={"color": COLOR["muted"], "stroke_width": 2},
        ).move_to(pos)
        x_lbl = safe_latex("V", 22, COLOR["muted"]).next_to(
            axes.x_axis.get_end(), DOWN, buff=0.15)
        y_lbl = safe_latex("P", 22, COLOR["muted"]).next_to(
            axes.y_axis.get_end(), LEFT, buff=0.15)

        def _point(t):
            seg = int(t * 4)
            u = t * 4 - seg
            if seg == 0:
                V = 1.0 + u * 1.5
                P = 3.0 / V
            elif seg == 1:
                V = 2.5 + u * 0.7
                P = 1.2 * (2.5 / V) ** 1.4
            elif seg == 2:
                V = 3.2 - u * 1.9
                P = 1.92 / V
            else:
                V = 1.3 - u * 0.3
                P = 1.5 * (1.3 / V) ** 1.4
            return V, P

        N = 200
        path_pts = [axes.c2p(*_point(i / N)) for i in range(N + 1)]
        path = VMobject().set_points_as_corners(path_pts) \
            .set_stroke(color, width=3, opacity=0.75)

        def _dot_pos():
            t = (gt.get_value() / period) % 1.0
            return axes.c2p(*_point(t))

        dot = always_redraw(lambda: Dot(_dot_pos(), radius=0.12,
                                          color=dot_color))
        trail = TracedPath(
            lambda: _dot_pos(),
            stroke_color=dot_color, stroke_width=2,
            dissipating_time=period * 2.0,
        )
        return VGroup(axes, x_lbl, y_lbl, path, trail, dot)

    def _prim_wave_packet_tunneling(self, item):
        """Gaussian wave packet incident on a barrier."""
        length = float(item.get("length", config.frame_width * 0.85))
        barrier_w = float(item.get("barrier_width", 0.4))
        barrier_h = float(item.get("barrier_height", 1.5))
        sigma = float(item.get("packet_width", 0.6))
        period = max(6.0, float(item.get("period", 14.0)))
        color = resolve_color(item)
        barrier_color = item.get("barrier_color", COLOR["warn"])
        gt = self.global_time

        x0, x1 = -length / 2, length / 2
        axes = Axes(
            x_range=[x0, x1, 1], y_range=[-1.5, 2.0, 1],
            x_length=length,
            y_length=config.frame_height * 0.28,
            axis_config={"color": COLOR["muted"], "stroke_width": 1.5},
        )
        baseline = Line(axes.c2p(x0, 0), axes.c2p(x1, 0),
                        color=COLOR["muted"], stroke_width=1)
        barrier = Rectangle(width=barrier_w, height=barrier_h,
                            color=barrier_color, fill_color=barrier_color,
                            fill_opacity=0.45, stroke_width=2)
        barrier.move_to(axes.c2p(0, barrier_h / 2))

        def _waveform():
            t = (gt.get_value() / period) % 1.0
            xa = np.linspace(x0, x1, 300)
            y = np.zeros_like(xa)
            if t < 0.5:
                c = x0 + (0 - x0) * (t / 0.5)
                g = np.exp(-((xa - c) ** 2) / (2 * sigma ** 2))
                y += g * np.sin(8 * (xa - c))
            else:
                u = (t - 0.5) / 0.5
                cr = 0 - (0 - x0) * u
                gr = np.exp(-((xa - cr) ** 2) / (2 * sigma ** 2))
                y += gr * np.sin(8 * (xa - cr))
                ct = 0 + (x1 - 0) * u
                gt_ = 0.25 * np.exp(-((xa - ct) ** 2)
                                    / (2 * (sigma * 0.8) ** 2))
                y += gt_ * np.sin(8 * (xa - ct))
            pts = [axes.c2p(xa[i], y[i]) for i in range(len(xa))]
            return VMobject().set_points_as_corners(pts) \
                .set_stroke(color, width=2.5)

        return VGroup(axes, baseline, barrier, always_redraw(_waveform))

    def _prim_bohr_orbit_photon(self, item):
        """Bohr atom: electron jumps inward, emits a photon, returns."""
        pos = to_3d(item.get("pos", [0, 0, 0]))
        r_in = float(item.get("r_inner", 0.8))
        r_out = float(item.get("r_outer", 1.8))
        period = max(4.0, float(item.get("period", 10.0)))
        nuc_color = item.get("nucleus_color", COLOR["pos"])
        e_color = item.get("electron_color", COLOR["field"])
        orb_color = item.get("orbit_color", COLOR["muted"])
        ph_color = item.get("photon_color", COLOR["accent"])
        gt = self.global_time

        nucleus = Dot(pos, radius=0.18, color=nuc_color)
        orbit_o = Circle(radius=r_out, color=orb_color,
                         stroke_width=1.5, stroke_opacity=0.5).move_to(pos)
        orbit_i = Circle(radius=r_in, color=orb_color,
                         stroke_width=1.5, stroke_opacity=0.5).move_to(pos)

        def _state(t):
            if t < 0.35:
                u = t / 0.35
                return r_out, u * 1.5 * TAU
            if t < 0.45:
                u = (t - 0.35) / 0.10
                r = r_out * (1 - u) + r_in * u
                a = 1.5 * TAU + u * PI * 0.5
                return r, a
            if t < 0.65:
                u = (t - 0.45) / 0.20
                return r_in, 1.5 * TAU + PI * 0.5 + u * TAU
            if t < 0.75:
                u = (t - 0.65) / 0.10
                r = r_in * (1 - u) + r_out * u
                a = 1.5 * TAU + PI * 0.5 + TAU + u * PI * 0.5
                return r, a
            u = (t - 0.75) / 0.25
            return r_out, 2.5 * TAU + PI + u * 0.5 * TAU

        def _e_pos():
            t = (gt.get_value() / period) % 1.0
            r, a = _state(t)
            return pos + np.array([r * np.cos(a), r * np.sin(a), 0])

        def _photon():
            t = (gt.get_value() / period) % 1.0
            if not (0.36 <= t <= 0.60):
                return VMobject()
            u = (t - 0.36) / 0.24
            _, a0 = _state(0.40)
            d = np.array([np.cos(a0), np.sin(a0), 0])
            dist = r_out + u * 3.5
            p = pos + d * dist
            return Dot(p, radius=0.11, color=ph_color)

        electron = always_redraw(
            lambda: Dot(_e_pos(), radius=0.14, color=e_color))
        photon = always_redraw(_photon)
        return VGroup(orbit_o, orbit_i, nucleus, electron, photon)

    def _prim_accretion_disk(self, item):
        """Black hole with a warped accretion disk."""
        pos = to_3d(item.get("pos", [0, 0, 0]))
        r_h = float(item.get("r_horizon", 0.9))
        r_d = float(item.get("r_disk", 3.2))
        disk_h = float(item.get("disk_h", 0.5))
        disk_color = item.get("disk_color", COLOR["accent"])
        bright_color = item.get("bright_color", "#FFD700")
        dim_color = item.get("dim_color", "#8B0000")
        bh_color = item.get("bh_color", "#000000")
        period = max(3.0, float(item.get("period", 8.0)))
        gt = self.global_time

        lens = Arc(radius=r_h + 0.45, start_angle=PI / 6,
                   angle=2 * PI / 3, color=dim_color,
                   stroke_width=4, stroke_opacity=0.75).move_to(pos)

        top_disk = Ellipse(width=r_d * 2.0, height=disk_h * 2.0,
                           color=disk_color, fill_color=disk_color,
                           fill_opacity=0.55, stroke_width=2)
        top_disk.move_to(pos + np.array([0, disk_h * 0.35, 0]))

        horizon = Circle(radius=r_h, color=bh_color,
                         fill_color=bh_color, fill_opacity=1.0,
                         stroke_width=0).move_to(pos)
        ring = Circle(radius=r_h + 0.08, color=disk_color,
                      stroke_width=2, stroke_opacity=0.9).move_to(pos)

        bright_side = Arc(radius=r_h + 0.20, start_angle=-PI / 3,
                          angle=2 * PI / 3,
                          color=bright_color, stroke_width=10,
                          stroke_opacity=0.95).move_to(pos)

        def _orbit_dot(phase):
            def _mk():
                t = (gt.get_value() / period + phase) % 1.0
                a = t * TAU
                r = r_h + (r_d - r_h) * (0.55 + 0.35 * np.cos(a))
                x = r * np.cos(a)
                y = r * np.sin(a) * 0.32
                op = 0.30 if y < -0.10 else 0.95
                d = Dot(pos + np.array([x, y, 0]),
                        radius=0.06, color=bright_color)
                d.set_opacity(op)
                return d
            return always_redraw(_mk)

        dots = VGroup(*[_orbit_dot(i / 8.0) for i in range(8)])
        return VGroup(lens, top_disk, horizon, ring, bright_side, dots)


# ============================================================ 3D scenes
class Universal3DPhysicsScene(ThreeDScene, VoiceoverScene):
 
    def __init__(self, scene_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.sk = scene_kwargs or {}
        self._watermark = None

    def construct(self):
        sk = self.sk
        no_voiceover = bool(sk.get("no_voiceover", False))
        render_only = bool(sk.get("render_only", False))
        use_voiceover = (HAS_VOICEOVER and GTTSService
                         and not no_voiceover and not render_only)
        if use_voiceover:
            try:
                self.set_speech_service(GTTSService(lang="en", tld="com"))
            except Exception:
                pass

        branding_on = sk.get("branding", "on") != "off"
        outro_dur = float(sk.get("outro_duration", 5.0))

        self.set_camera_orientation(phi=70 * DEGREES, theta=-50 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.15)

        if branding_on:
            wm = SaemLabsBranding._build_watermark()
            wm.to_corner(DR, buff=0.5)
            self.add_fixed_in_frame_mobjects(wm)
            self._watermark = wm

        title = sk.get("header_title", "")
        if title:
            t = Text(title, font_size=36, weight=BOLD,
                     color=WHITE).to_corner(UL, buff=0.8)
            self.add_fixed_in_frame_mobjects(t)

        axes = ThreeDAxes(x_range=[-5, 5], y_range=[-5, 5], z_range=[-5, 5])
        self.add(axes)
        helix = ParametricFunction(
            lambda t: np.array([np.cos(2 * t) * 2, np.sin(2 * t) * 2,
                                t / 3 - 2]),
            t_range=[-4, 4], color=COLOR["accent"], stroke_width=4,
        )
        self.play(Create(helix), run_time=4.0)

        equations = safe_json_loads(sk.get("equations_json", "[]"))
        if equations:
            eq_group = VGroup(
                *[safe_latex(e, 28, WHITE) for e in equations[:3]]
            ).arrange(DOWN, buff=0.3)
            eq_group.to_corner(DL, buff=1.0)
            self.add_fixed_in_frame_mobjects(eq_group)
            for eq in eq_group:
                self.play(Write(eq), run_time=0.8)

        self.wait(8.0)
        self.stop_ambient_camera_rotation()
        if self._watermark is not None:
            try:
                self.remove(self._watermark)
            except Exception:
                pass
        if branding_on:
            SaemLabsBranding.closing_card(self, duration=outro_dur)


class SaemLabs3DScene(ThreeDScene, VoiceoverScene):
    def __init__(self, scene_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.sk = scene_kwargs or {}
        self._watermark = None

    def construct(self):
        sk = self.sk
        no_voiceover = bool(sk.get("no_voiceover", False))
        render_only = bool(sk.get("render_only", False))
        use_voiceover = (HAS_VOICEOVER and GTTSService
                         and not no_voiceover and not render_only)
        if use_voiceover:
            try:
                self.set_speech_service(GTTSService(lang="en", tld="com"))
            except Exception:
                pass

        branding_on = sk.get("branding", "on") != "off"
        outro_dur = float(sk.get("outro_duration", 5.0))

        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.10)

        if branding_on:
            wm = SaemLabsBranding._build_watermark()
            wm.to_corner(DR, buff=0.5)
            self.add_fixed_in_frame_mobjects(wm)
            self._watermark = wm

        title = sk.get("header_title", "")
        if title:
            t = Text(title, font_size=36, weight=BOLD,
                     color=WHITE).to_corner(UL, buff=0.8)
            self.add_fixed_in_frame_mobjects(t)

        sphere_colors = [COLOR["pos"], COLOR["field"], COLOR["accent"]]
        sphere_labels = ["physics", "chemistry", "biology"]
        spheres = VGroup()
        for i, (col, lbl) in enumerate(zip(sphere_colors, sphere_labels)):
            s = Sphere(radius=0.6, color=col, resolution=(16, 16))
            s.move_to([(i - 1) * 3.0, 0.5, 0])
            label = Text(lbl, font_size=24, color=col)
            label.next_to(s, DOWN, buff=0.4)
            spheres.add(VGroup(s, label))

        self.play(FadeIn(spheres, shift=UP * 0.5), run_time=2.0)
        self.wait(2.0)

        lines = VGroup()
        for i in range(3):
            for j in range(i + 1, 3):
                a = spheres[i][0].get_center()
                b = spheres[j][0].get_center()
                lines.add(Line(a, b, color=COLOR["white"], stroke_width=1.5))
        self.play(Create(lines), run_time=1.5)

        equations = safe_json_loads(sk.get("equations_json", "[]"))
        if equations:
            eq_group = VGroup(
                *[safe_latex(e, 28, WHITE) for e in equations[:2]]
            ).arrange(DOWN, buff=0.3)
            eq_group.to_corner(DL, buff=1.0)
            self.add_fixed_in_frame_mobjects(eq_group)
            for eq in eq_group:
                self.play(Write(eq), run_time=0.8)

        self.wait(8.0)
        self.stop_ambient_camera_rotation()
        if self._watermark is not None:
            try:
                self.remove(self._watermark)
            except Exception:
                pass
        if branding_on:
            SaemLabsBranding.closing_card(self, duration=outro_dur)


# ============================================================ CLI
def main():
    p = argparse.ArgumentParser(
        description="Manim universal cinematic renderer (vertical default)."
    )
    p.add_argument("--video_id", required=True)
    p.add_argument("--header_title", default="")
    p.add_argument("--tagline", default="")
    p.add_argument("--question_text", default="")
    p.add_argument("--options_json", default="[]")
    p.add_argument("--correct_answer", default="")
    p.add_argument("--equations_json", default="[]")
    p.add_argument("--visual_data_json", default="[]")
    p.add_argument("--beats_json", default="[]")
    p.add_argument("--bindings_json", default="[]")
    p.add_argument("--formula_sheet_json", default="[]")
    p.add_argument("--manifest_path", default="")
    p.add_argument("--concept_type", default="GENERIC")
    p.add_argument("--quality", default="1080p",
                   choices=list(QUALITY_PRESETS.keys()))
    p.add_argument("--orientation", default=ORIENTATION,
                   choices=["vertical", "horizontal"])
    p.add_argument("--duration", type=float, default=180.0)
    p.add_argument("--intro_duration", type=float, default=0.0)
    p.add_argument("--outro_duration", type=float, default=5.0)
    p.add_argument("--branding", default="on", choices=["on", "off"])
    p.add_argument("--no_voiceover", action="store_true")
    p.add_argument("--draft", action="store_true")
    p.add_argument("--render_only", action="store_true")
    p.add_argument("--custom_scene", default="")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    apply_quality("draft" if args.draft else args.quality, args.orientation)

    try:
        options = safe_json_loads(args.options_json)
    except Exception:
        options = []
    try:
        formula_sheet = safe_json_loads(args.formula_sheet_json)
    except Exception:
        formula_sheet = []

    scene_kwargs = {
        "header_title":     args.header_title,
        "tagline":          args.tagline,
        "question_text":    args.question_text,
        "options":          options,
        "correct_answer":   args.correct_answer,
        "equations_json":   args.equations_json,
        "visual_data_json": args.visual_data_json,
        "beats_json":       args.beats_json,
        "bindings_json":    args.bindings_json,
        "formula_sheet":    formula_sheet,
        "manifest_path":    args.manifest_path,
        "concept_type":     args.concept_type.upper(),
        "duration":         args.duration,
        "intro_duration":   args.intro_duration,
        "outro_duration":   args.outro_duration,
        "branding":         args.branding,
        "no_voiceover":     args.no_voiceover,
        "render_only":      args.render_only,
        "orientation":      args.orientation,
    }

    scene = None

    if args.custom_scene and os.path.exists(args.custom_scene):
        _log(f"Custom scene override: {args.custom_scene}")
        try:
            spec = importlib.util.spec_from_file_location(
                "custom_scene_mod", args.custom_scene)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "build_scene"):
                scene = mod.build_scene(scene_kwargs)
            elif hasattr(mod, "Scene"):
                scene = mod.Scene(scene_kwargs=scene_kwargs)
            else:
                _log("Custom scene has no build_scene or Scene; "
                     "falling back to universal renderer.")
                scene = None
        except Exception as e:
            _log(f"Custom scene load failed: {e}; falling back.")
            scene = None

    if scene is None:
        ct = args.concept_type.upper()
        if ct == "3D":
            scene = SaemLabs3DScene(scene_kwargs=scene_kwargs)
        elif ct == "3D_HELIX":
            scene = Universal3DPhysicsScene(scene_kwargs=scene_kwargs)
        else:
            scene = UniversalPhysicsScene(scene_kwargs=scene_kwargs)

    scene.render()

    out_path = None
    try:
        out_path = str(scene.renderer.file_writer.movie_file_path)
    except Exception:
        pass
    if out_path is None or not os.path.exists(out_path):
        import glob
        candidates = glob.glob(
            os.path.join("media", "videos", "**", "*Scene.mp4"),
            recursive=True,
        )
        if candidates:
            out_path = max(candidates, key=os.path.getmtime)
    if out_path is None or not os.path.exists(out_path):
        _log("FATAL could not locate rendered output")
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    shutil.move(out_path, args.output)
    _log(f"SUCCESS {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
