#!/usr/bin/env python3
"""Generate render-ready explanation batches from the reviewed problem matrix."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from generate_lecture_series import FIELDS
from problem_matrix import Problem, load_problem_matrix
from schema_validator import audio_word_bounds

ROOT = Path(__file__).resolve().parent
PROBLEM_FIELDS = (
    *FIELDS,
    "problem_id",
    "concept_id",
    "source_ref",
    "source_pages",
    "question_number",
    "feynman_sources",
    "strang_sources",
    "related_lesson_ids",
)

SCRIPTS = {
    "neet2024_physics_q01": {
        "vertical": """A bead moving around a circle keeps changing direction, even when its speed stays constant. That change requires an inward force. Here the mass and radius stay fixed, and the speed doubles. For circular motion, the inward force is mass times speed squared divided by radius. Since only speed changes, tension is proportional to speed squared. Replace v with 2v: the new force is m times (2v) squared divided by r, which is four times the original. So the string tension becomes 4T. Doubling speed does not merely double the force; the square matters. The orbit's curved path is the clue: velocity changes direction continuously, and an inward force keeps bending it. If the radius changed too, compare both ratios. Same mass, same radius, twice the speed: four times the tension.""",
        "horizontal": """Picture a bead traveling around a circular track. Its speed can stay steady while its velocity changes, because velocity includes direction. At every point on the circle, the direction turns. A change in velocity requires acceleration, and for circular motion that acceleration points toward the center. The inward force that produces it is called centripetal force. It is not a new kind of force; it is the inward component of tension, gravity, friction, or another ordinary interaction.

For a mass m moving at speed v on a circle of radius r, the required inward force is m v squared divided by r. In this exam problem, the string supplies that inward force. The mass and radius are unchanged, so tension varies only with v squared. Doubling the speed changes the expression to m times (2v) squared divided by r. The four from squaring the two gives four times the original tension: T_new equals 4T.

Now connect the formula to the geometry. In a short time interval, the bead's velocity arrow rotates through a small angle while keeping almost the same length. The difference between the old and new velocity arrows points inward. Strang's calculus treatment of curvature and normal direction makes this turning rate visible: a tighter curve, or a faster trip around the same curve, demands more inward acceleration. Here the radius is fixed, so the speed change alone sets the ratio.

Check the result with units and proportionality. Both forces have the same units, so the ratio is dimensionless. If speed were tripled, tension would be nine times larger, not three times. If instead the radius doubled at fixed speed, the required force would be halved. These comparisons are useful before substituting any numbers.

The model assumes the string remains taut and the motion stays circular. A real setup could also include gravity or change shape, but the question holds the radius and mass fixed and asks for the tension ratio. The visual rule is simple: same circle, faster motion, stronger inward pull. The square law explains why a modest speed increase can sharply raise the load on a string.""",
    },
    "neet2024_physics_q10": {
        "vertical": """A position rule can reveal speed when we differentiate with respect to time. Here x(t)=2t-1 in SI units, so dx/dt is 2 meters per second. For a force along the motion, instantaneous power is force times velocity. Multiply 5 newtons by 2 meters per second: the power is 10 joules per second, or 10 watts. The constant minus one shifts the particle's starting position, but its derivative is zero, so it does not change the speed. Keep the steps in order: position gives velocity by differentiation; velocity combines with force to give power. Units provide a check: newtons times meters per second equals watts. If force pointed opposite the motion, power would be negative; here we use the direction implied by the answer key. The result is a constant power because the force and velocity are constant in this model.""",
        "horizontal": """A position function answers where a particle is. Its derivative answers how quickly that position changes. This distinction turns a short algebraic expression into a power calculation. The exam gives x(t)=2t-1 in SI units and a force of 5 newtons. We take the force along the positive motion, matching the keyed result.

Differentiate position with respect to time: dx/dt=2 meters per second. The constant minus one disappears because it shifts every position by the same amount without changing how position evolves. The velocity is therefore constant. Instantaneous power is the rate at which a force transfers energy. For a force parallel to velocity, P=Fv. Substituting gives 5 newtons times 2 meters per second, or 10 joules per second. Since one joule per second is one watt, the answer is 10 watts.

The calculus connection is more than a trick. Strang introduces velocity as the derivative of distance: the slope of a position-versus-time graph is speed along that line. Power then combines this rate of motion with the applied force. In vector form, power is the dot product of force and velocity. If the force makes an angle, only the component along the motion contributes. The geometry of the force and the velocity matters, but here the question says the force acts along the motion, so the dot product simplifies to a plain product.

Notice what did not matter. The starting position changes if the constant term changes, but the derivative remains 2. The power stays the same as long as the force and velocity stay the same. If the position instead contained a term such as t squared, the velocity would vary with time, and so would the instantaneous power. A linear position law implies a constant slope and constant speed, which makes the power calculation especially clean.

Check the dimensions: a newton is a kilogram meter per second squared. Multiplying by meters per second gives kilogram meters squared per second cubed, which is a watt. Also check the sign: a force aligned with motion transfers energy into the motion; a force opposing it removes energy, giving negative power. The source question does not spell out a force angle, so this worked explanation states the aligned-force assumption explicitly. Differentiate first, then multiply by the force. The units and the sign are both valid checks before accepting the final number.""",
    },
    "neet2022_physics_q02": {
        "vertical": """A symmetric convex lens converges parallel rays, and its power tells us how strongly. Use the lensmaker relation P=(n-1)(1/R1-1/R2). Here n is 1.5, and each curvature radius has magnitude 20 centimeters, or 0.20 meters. With the usual sign convention, R1 is positive and R2 is negative. The reciprocal terms add: five plus five inverse meters. Multiply by n-1, which is one-half, and the result is positive five inverse meters, or +5 diopters. The positive sign identifies a converging lens. Assign the surface signs before calculating; making both radii positive would cancel the curvature incorrectly. The important visual is two surfaces bending rays toward one shared focus. The lens power is not arbitrary; it is a direct measurement of how sharply the lens bends light toward that common point.""",
        "horizontal": """A lens can turn parallel incoming rays toward a focus. Its optical power measures the strength of that convergence, in inverse meters called diopters. For a thin lens in air, the lensmaker relation is P=(n-1)(1/R1-1/R2), where n is the lens refractive index and each radius carries a sign set by the surface curvature.

The problem gives a symmetric biconvex lens with n=1.5 and radius magnitude 20 centimeters for each surface. Convert first: 20 centimeters is 0.20 meters. Under the standard left-to-right sign convention, the first surface has R1=+0.20 meters and the second has R2=-0.20 meters. This opposite sign is essential; it reflects that the two surfaces curve in opposite directions as seen by the traveling light.

Now evaluate the reciprocal difference. One over R1 is +5 inverse meters; one over R2 is -5 inverse meters. Subtracting the second gives 5-(-5)=10 inverse meters. The refractive-index factor is n-1=0.5, so the power is 0.5 times 10, or +5 inverse meters. Since one diopter equals one inverse meter, the final power is +5 D.

The sign and size both make sense. A biconvex lens with index greater than air is converging, so its power is positive. Two curved surfaces contribute to the total bending, which is why their signed reciprocal radii add in this symmetric case. If the lens material had the same refractive index as the surrounding air, n-1 would be zero and the lens would have no optical power. That is the key idea: bending depends on the difference between the lens and the surrounding medium, not on the surfaces alone.

A quick unit check catches a common error: the radii must be converted to meters if the answer is requested in diopters. Using 20 as though it were meters would make the power one hundred times too small. Another check is the curvature sign; using two positive radii would erase the effect. In a visual explanation, trace two parallel rays through the lens and show them meeting at a real focus. That picture explains why positive power means convergence, while the signed formula supplies the numerical answer. The same relationship is used in eyeglass prescriptions and camera optics, where more curvature or higher index means stronger focusing.""",
    },
    "neet2022_physics_q06": {
        "vertical": """Magnetic flux measures how much magnetic field passes through a loop. It is B times area times the cosine of the angle between the field and the area's normal. This square loop has side one meter, so its area is one square meter. Its plane is perpendicular to the field, which means the normal points along the field and the angle is zero. With B=0.5 tesla, flux is 0.5 times 1 times cos zero: 0.5 weber. The loop's one-ohm resistance is irrelevant because the question asks for flux, not induced current. Measure the angle from the normal, not from the surface. That distinction separates full flux from zero flux. It also explains why a loop tipped sideways can have much less flux even when the field magnitude is unchanged.""",
        "horizontal": """Magnetic flux is a geometric count of field passing through a surface. For a uniform magnetic field and a flat loop, flux is Phi=B A cos(theta). The angle theta is measured between the field and the loop's area normal, an arrow perpendicular to the plane. This angle convention is the key to the problem.

The loop is a square with side one meter, so its area is one meter squared. The plane of the loop is perpendicular to the field. Therefore the field points along the normal: theta is zero degrees, not ninety. Cosine of zero is one, so the flux is 0.5 tesla times 1 square meter times 1, giving 0.5 weber.

You can visualize the orientation by tilting a flat card in a field. When the field passes straight through the card, the area normal and field line up and the flux is largest. Turn the card edge-on to the field and the field runs parallel to its surface; the normal is perpendicular to the field and flux falls to zero. The cosine captures this projected area. If the loop were rotated ninety degrees, the flux would vanish even though the magnetic field remained the same size.

This question also points toward Faraday's law, which connects changing magnetic flux to induced voltage. But here the field, area, and orientation are given at one instant, and the requested quantity is flux itself. The loop resistance does not change Phi; resistance would matter later if we asked how much current a changing flux drives. The geometry is the primary message here, not the circuit response.

Check the units: tesla times square meter is a weber. The numerical value is half a weber because half a tesla crosses one square meter. A frequent mistake is measuring theta from the loop's plane. If one uses ninety degrees for this setup, the computed zero would contradict the picture of field lines passing directly through the loop. Always draw the normal first, then set the angle. That one geometric step makes the equation almost automatic, and it is the same habit that helps with induced emf and induction problems later.""",
    },
}

BEATS = [
    {"type": "hook", "text": "Start from the physical puzzle", "focus": ""},
    {"type": "law", "text": "Build the governing model", "focus": ""},
    {"type": "experiment", "text": "Work the values and units", "focus": ""},
    {"type": "emphasis", "text": "Check the sign and assumptions", "focus": ""},
    {"type": "punchline", "text": "Keep the transferable idea", "focus": ""},
]


def _select_problems(
    problems: tuple[Problem, ...],
    problem_ids: list[str] | tuple[str, ...] | None,
) -> tuple[Problem, ...]:
    if problem_ids is None:
        selected = problems
    else:
        if not problem_ids or len(set(problem_ids)) != len(problem_ids):
            raise ValueError("selected problem IDs must be a non-empty unique list")
        by_id = {problem.problem_id: problem for problem in problems}
        missing = set(problem_ids) - set(by_id)
        if missing:
            raise ValueError(f"unknown problem IDs: {', '.join(sorted(missing))}")
        selected = tuple(by_id[problem_id] for problem_id in problem_ids)
    if set(SCRIPTS) != {problem.problem_id for problem in problems}:
        raise ValueError("problem matrix IDs must match the authored script IDs")
    return selected


def _build_rows(problems: tuple[Problem, ...], orientation: str) -> list[dict[str, str]]:
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError("orientation must be horizontal or vertical")
    rows = []
    for problem in problems:
        script = SCRIPTS[problem.problem_id][orientation]
        minimum, maximum = audio_word_bounds(orientation)
        word_count = len(script.split())
        if not minimum <= word_count <= maximum:
            raise ValueError(
                f"{problem.problem_id} {orientation} script has {word_count} words; "
                f"expected {minimum}-{maximum}"
            )
        sources = problem.concept.source_refs
        lessons = problem.concept.lesson_ids
        rows.append({
            "video_id": f"{problem.problem_id}_{orientation}",
            "concept_type": "GENERIC",
            "header_title": problem.concept.concept,
            "tagline": problem.problem_summary,
            "question_text": problem.problem_summary,
            "option_a": "",
            "option_b": "",
            "option_c": "",
            "option_d": "",
            "correct_answer": problem.worked_solution,
            "equations_json": json.dumps(problem.equations, ensure_ascii=False),
            "visual_data_json": json.dumps(problem.visual_data, ensure_ascii=False),
            "audio_script": script,
            "orientation": orientation,
            "beats_json": json.dumps(BEATS, ensure_ascii=False),
            "bindings_json": json.dumps([
                {
                    "equation_idx": 0,
                    "visual_id": problem.visual_data[0]["type"],
                    "color": "#F1C40F",
                }
            ]),
            "language": "en",
            "quality": "1080p",
            "platform": "youtube" if orientation == "horizontal" else "shorts",
            "lesson_id": ";".join(lessons),
            "problem_id": problem.problem_id,
            "concept_id": problem.concept.concept_id,
            "source_ref": problem.source_ref,
            "source_pages": problem.source_pages,
            "question_number": str(problem.question_number),
            "feynman_sources": ";".join(
                source for source in sources if source.startswith("feynman_")
            ),
            "strang_sources": ";".join(
                source for source in sources if source.startswith("strang_")
            ),
            "related_lesson_ids": ";".join(lessons),
        })
    return rows


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=PROBLEM_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def generate(
    *,
    orientation: str = "vertical",
    output_dir: Path = ROOT,
    problem_ids: list[str] | tuple[str, ...] | None = None,
) -> tuple[Path, ...]:
    problems = _select_problems(load_problem_matrix(), problem_ids)
    orientations = ("horizontal", "vertical") if orientation == "both" else (orientation,)
    if any(item not in {"horizontal", "vertical"} for item in orientations):
        raise ValueError("orientation must be horizontal, vertical, or both")
    outputs = []
    for item in orientations:
        path = output_dir / f"problem_batch_{item}.csv"
        _write_rows(path, _build_rows(problems, item))
        outputs.append(path)
    return tuple(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orientation",
        choices=("vertical", "horizontal", "both"),
        default="both",
    )
    parser.add_argument(
        "--problem-id",
        action="append",
        dest="problem_ids",
        help="problem ID to include; repeat to select multiple; default is all reviewed problems",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        outputs = generate(
            orientation=args.orientation,
            output_dir=args.output_dir,
            problem_ids=args.problem_ids,
        )
    except ValueError as exc:
        parser.error(str(exc))
    for path in outputs:
        print(f"CSV: {path}")


if __name__ == "__main__":
    main()
