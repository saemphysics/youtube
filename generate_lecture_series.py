#!/usr/bin/env python3
"""Export the authored 10-lesson plan as paired horizontal and vertical CSVs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / "lecture_series_plan.csv"
FIELDS = [
    "video_id", "concept_type", "header_title", "tagline",
    "question_text", "option_a", "option_b", "option_c", "option_d",
    "correct_answer", "equations_json", "visual_data_json", "audio_script",
    "orientation", "beats_json", "bindings_json", "language", "quality",
    "platform", "lesson_id",
]

# Scripts are authored independently for each format; the Short is not cut from
# the lecture. Keep the shared example and conclusion synchronized by lesson ID.
SCRIPTS = {
    "mix01": {
        "short": """Two speakers play the same note. In one spot, the sound gets louder; a step away, it nearly disappears. The waves met with different timing.

At a fixed point, wave displacements add. If each wave has amplitude A and they arrive in phase, their crests line up: the total amplitude is 2A. For the same kind of wave in the same medium, intensity scales as amplitude squared. So compared with one speaker, the intensity is (2A) squared over A squared: four times.

Now shift one wave by half a cycle. A crest meets a trough, and the displacement cancels at that point. The quiet spot is part of a pattern: energy is redistributed to places where waves reinforce.

Add signed displacements first; square the resulting amplitude when comparing intensity. Two equal, in-phase waves give four times the intensity, not twice.""",
        "long": """Two speakers play the same note. Stand in one place and the sound is strong; move a little and it can become surprisingly quiet. The speakers have not changed their power. What changed is the relative timing of the waves that reached you.

To understand this, do not add loudness labels. Track the displacement of the air at one point. A wave may push that air slightly forward while another pulls it slightly backward. The signed displacements add: y_total equals y_one plus y_two. This is the superposition principle.

Suppose both speakers send the same tone with amplitude A, and their crests arrive together. Then the sum has amplitude 2A. For waves of the same kind traveling through the same medium, intensity is proportional to amplitude squared. Relative to one speaker, the intensity is (2A)^2 divided by A^2, which is four. The displacement doubled; the intensity quadrupled. Those are different quantities.

Now imagine delaying one speaker by half a cycle. A crest from one arrives with a trough from the other. At that location, equal displacements cancel. But cancellation at one point is not energy being destroyed. Nearby, the timing is different, and other places have constructive interference. The spatial pattern has bright regions, quiet regions, nodes, and antinodes.

This model assumes coherent sources with the same frequency and comparable amplitudes. If the sources drift out of step, the pattern moves or blurs. If one speaker is much louder, cancellation is incomplete. Those limitations are useful: they tell us what to measure.

So keep two operations in the right order. First add wave displacements, including their signs and phases. Then relate the resulting amplitude to intensity. In-phase equal waves give four times the intensity of one wave at that point, not twice. If you walk around a room with two speakers, your ears are measuring a map of phase difference.""",
        "long_extension": """A quick prediction makes the idea testable. If we lower the volume knob on both speakers by the same factor, every displacement becomes smaller, but the locations of constructive and destructive interference stay put as long as their phase relationship is unchanged. If we reverse the polarity of one speaker, we shift the phase by half a cycle and exchange the bright and quiet locations. That is a change in the pattern, not a failure of superposition.""",
        "visual": [
            {"type": "wave", "amplitude": 0.8, "frequency": 1.0, "speed": 1.1,
             "x_range": [-5.5, 5.5], "color": "#3498DB"},
            {"type": "graph", "expression": "sin(x)+sin(x)", "x_range": [-6, 6],
             "y_range": [-2.5, 2.5], "color": "#F1C40F"},
        ],
        "equations": ["y_total=y1+y2", "I ∝ A²", "A_total=2A ⇒ I_total=4I"],
        "beats_short": ["hook", "law", "emphasis", "punchline"],
        "beats_long": ["hook", "promise", "analogy", "law", "experiment", "emphasis", "punchline", "cta"],
    },
    "mix02": {
        "short": """A ball launched at an angle follows a curve, though nothing pushes it sideways after release. The curve combines two simpler motions.

Launch at 20 meters per second, 30 degrees above horizontal. Split the velocity: horizontal speed is 20 cosine 30, or 10 root 3 meters per second. Vertical speed is 20 sine 30, or 10 meters per second. Ignore air resistance and take gravity as 10 meters per second squared.

The upward speed decreases by 10 meters per second each second. It takes one second to reach the top, then one more to return, so flight time is 2 seconds. Horizontal speed stays constant. Range is 10 root 3 times 2, about 34.6 meters.

Resolve velocity before using the equations: initial speed is not all horizontal. Forty-five degrees maximizes range only for equal launch and landing heights with negligible air resistance.""",
        "long": """A thrown ball draws a parabola, but after it leaves your hand there is no forward push keeping it moving. The shape is easier to understand by separating the motion into horizontal and vertical parts.

Take a launch speed of 20 meters per second at 30 degrees above level ground. Resolve that velocity. The horizontal component is 20 cosine 30, which is 10 root 3, about 17.3 meters per second. The vertical component is 20 sine 30, or 10 meters per second. These components describe the same initial velocity; they are not two extra motions added afterward.

Neglect air resistance and use g equals 10 meters per second squared. Horizontally, acceleration is zero, so the horizontal speed stays 17.3 meters per second. Vertically, acceleration is downward. Starting upward at 10 meters per second, the ball loses 10 meters per second of vertical speed each second. It reaches zero vertical speed after one second. Symmetry about the top gives another second to return to its launch height, so total flight time is 2 seconds.

Now calculate the range. Horizontal distance is horizontal speed times time: 10 root 3 meters per second times 2 seconds, or 20 root 3 meters, about 34.6 meters. Notice the units: meters per second times seconds gives meters.

The path is curved because horizontal position keeps increasing while vertical position first rises and then falls. Each component follows a simple rule; the curve is their combination. This separation fails if air drag is important, because then the horizontal speed also changes.

One popular statement says 45 degrees gives maximum range. That is true for an ideal projectile launched and landing at the same height, with fixed speed and no air resistance. It is not a universal rule. For this 30-degree example we can calculate the range directly, rather than trusting a memorized angle.""",
        "long_extension": """As a sanity check, double the flight time while keeping horizontal speed fixed and the range doubles. Raise the launch angle and the horizontal component shrinks while vertical flight time grows. Those competing changes explain why the best angle is a balance, not simply the largest upward component. Always sketch the axes and initial velocity before substituting numbers.""",
        "visual": [
            {"type": "projectile_motion", "v0": 20.0, "angle": 30, "g": 10.0,
             "color": "#3498DB"},
            {"type": "vector", "start": [0, 0, 0], "end": [2.1, 1.2, 0],
             "label": "v", "color": "#F1C40F"},
        ],
        "equations": ["vx=v cos θ", "vy=v sin θ", "x=vx t", "y=vy t−gt²/2"],
        "beats_short": ["hook", "law", "emphasis", "experiment", "punchline"],
        "beats_long": ["hook", "promise", "analogy", "law", "emphasis", "experiment", "punchline", "cta"],
    },
    "mix03": {
        "short": """Drop a ball from five meters: where does its height go? In the ideal model, gravitational potential energy becomes kinetic energy.

Set the lost height energy, m g h, equal to the gained motion energy, one-half m v squared. The mass appears on both sides, so it cancels. With g equal to 10 meters per second squared and h equal to 5 meters, v squared is 2 times 10 times 5, or 100. The speed just before impact is 10 meters per second.

Check the units: meters squared per second squared under the square root becomes meters per second. A heavier ball has more energy at the same height, but needs more energy for the same speed; that is why mass cancels. This assumes negligible air resistance. With drag, some mechanical energy becomes heat and sound, while total energy accounting still holds.""",
        "long": """A ball held above the floor is not moving, yet lowering it can produce motion. The useful question is not where a little packet of energy is hiding. It is how the energy accounting changes as the ball falls.

Near Earth's surface, a mass m at height h has gravitational potential energy m g h relative to a chosen zero height. If we release it from rest and ignore air resistance, that energy becomes kinetic energy, one-half m v squared. Set them equal: m g h equals one-half m v squared. The mass cancels, leaving v equals the square root of 2 g h.

Use the example: a drop of 5 meters and g approximately 10 meters per second squared. Then v equals the square root of 2 times 10 times 5, which is the square root of 100, or 10 meters per second just before impact. Check the dimensions: g h has units meters squared per second squared, so its square root is meters per second.

Why did mass disappear? At the same height, a heavier object has proportionally more gravitational energy. But for a given speed, it also needs proportionally more kinetic energy. The two factors of mass cancel in the ideal prediction. This does not say the energy amounts are identical for different masses; it says the predicted speed is the same.

The model has a boundary. Air resistance transfers some organized mechanical energy into thermal motion of the air and object, and into sound. Mechanical energy alone then decreases, but total energy accounting still includes those forms. The result also depends on the height difference, not the absolute height label on a ruler.

So height is not a substance that leaks into the ball. It is a way to calculate the energy available under gravity. The 5-meter example predicts 10 meters per second, provided drag is negligible. If you want to test the assumption, compare objects with different shapes and watch where the simple prediction begins to fail.""",
        "long_extension": """Energy is especially useful when the force changes along a path, because we can compare the initial and final states without tracking every instant. Here gravity is nearly constant and the path is simple, so kinematics would reach the same speed. Agreement between two independent methods is a strong check; the energy method also makes the mass cancellation visible immediately.""",
        "visual": [
            {"type": "energy_bar", "pos": [-2, 0, 0], "height": 2.5,
             "label": "P.E.", "color": "#F1C40F"},
            {"type": "energy_bar", "pos": [1, 0, 0], "height": 0.5,
             "label": "K.E.", "color": "#2ECC71"},
            {"type": "projectile_motion", "v0": 0.1, "angle": 80, "g": 9.81,
             "color": "#3498DB"},
        ],
        "equations": ["U=mgh", "K=½mv²", "v=√(2gh)=10 m/s"],
        "beats_short": ["hook", "law", "emphasis", "experiment", "punchline"],
        "beats_long": ["hook", "promise", "analogy", "law", "emphasis", "experiment", "punchline", "cta"],
    },
    "mix04": {
        "short": """Why is a door easier to turn at its handle than beside its hinge? Turning effect depends on force and lever arm.

For a perpendicular push, torque is force times distance from the axis. Put a 2 kilogram point mass 0.5 meters from an axis. Its rotational inertia is m r squared: 2 times 0.5 squared, or 0.5 kilogram meter squared. A 4 newton perpendicular force gives torque 0.5 times 4, or 2 newton meters.

Angular acceleration is torque divided by rotational inertia: 2 divided by 0.5, or 4 radians per second squared. The same force farther from the hinge makes more torque. A force directed along the lever arm produces no turning effect, because its perpendicular component is zero.

The axis, lever arm, and force direction matter. Mass distribution sets rotational inertia; torque sets how quickly angular velocity changes.""",
        "long": """A door can be opened with a small push at the handle, yet the same push beside the hinge barely turns it. The force did not become stronger. Its distance from the rotation axis changed.

For a force perpendicular to a lever arm, torque magnitude is r times F. More generally it is r F sine theta, where theta is the angle between the position vector and force. Only the perpendicular part of the force turns the object. A push directly along the lever arm has zero torque.

Now calculate a compact model. Place a 2 kilogram point mass 0.5 meters from an axis. Its moment of inertia is I equals m r squared: 2 times 0.5 squared equals 0.5 kilogram meter squared. Apply a 4 newton force perpendicular to the radius. Torque is 0.5 meters times 4 newtons, or 2 newton meters. Using tau equals I alpha, angular acceleration is 2 divided by 0.5, or 4 radians per second squared.

The units check: a newton meter divided by a kilogram meter squared becomes inverse seconds squared, the unit of angular acceleration when radians are dimensionless. The direction follows the right-hand rule; the calculation above gives the magnitude.

This point-mass model also teaches a second idea. Rotational inertia depends on how mass is distributed relative to the chosen axis. Move the same mass twice as far away and its contribution to I becomes four times larger. For a door, mass is spread across a surface, but the same principle applies.

So a turning push is not described by force alone. We need the axis, the perpendicular lever arm, and the object's inertia. A handle is effective because it gives the force a long lever arm. The same geometry explains why changing the axis or moving mass changes how readily an object spins.""",
        "long_extension": """There is a direct translational parallel. Newton's law says force divided by mass sets linear acceleration; the rotational version says torque divided by moment of inertia sets angular acceleration. The analogy is helpful only when we match the quantities carefully: mass corresponds to rotational inertia, and force corresponds to torque. A large force through the pivot can still produce zero torque.""",
        "visual": [
            {"type": "angular_measurement", "pos": [0, 0, 0], "radius": 1.8,
             "color": "#F1C40F"},
            {"type": "vector", "start": [0, 0, 0], "end": [2, 0, 0],
             "label": "r", "color": "#3498DB"},
            {"type": "vector", "start": [2, 0, 0], "end": [2, 1.5, 0],
             "label": "F", "color": "#FF4B4B"},
        ],
        "equations": ["τ=rF sin θ", "I=mr²", "α=τ/I=4 rad/s²"],
        "beats_short": ["hook", "law", "emphasis", "experiment", "punchline"],
        "beats_long": ["hook", "promise", "analogy", "law", "emphasis", "experiment", "punchline", "cta"],
    },
    "mix05": {
        "short": """Put your thumb over part of a hose outlet and the jet speeds up. Why? In steady incompressible flow, the same volume must pass each section every second.

Flow rate is area times speed: Q equals A v. A circular pipe's area is pi r squared. If the radius shrinks from 2 centimeters to 1 centimeter, the area becomes one quarter, not one half.

The wide section carries water at 0.5 meters per second. Continuity says A1 v1 equals A2 v2. So v2 is A1 over A2 times v1: four times 0.5, or 2 meters per second.

The radius ratio is squared, a step people often miss. This assumes steady, incompressible flow with no leaks. Continuity predicts the speed change, not pressure losses in a real hose.""",
        "long": """Partly cover a hose outlet and the water jet often moves faster. The first principle is not “small pipes create speed.” It is conservation of volume for steady, incompressible flow.

Imagine marking a slice of water in a wide pipe. In one second, a volume equal to cross-sectional area times speed passes that slice. Call the volume flow rate Q equals A v. If there is no leak or accumulation between two sections, the same volume per second passes both. Thus A1 v1 equals A2 v2.

Take radii of 2 centimeters in the wide section and 1 centimeter in the narrow section. A circle's area is pi r squared, so the area ratio wide to narrow is (2/1)^2, or 4. If the water speed in the wide section is 0.5 meters per second, the narrow-section speed is 4 times 0.5, or 2 meters per second.

Notice why halving the radius does not merely double speed. The cross-sectional area falls to one quarter, so speed must rise by a factor of four to carry the same volume each second. The units also fit: area times speed is square meters times meters per second, or cubic meters per second.

The assumptions matter. We treated water density as constant and the flow as steady. A real hose can have friction, turbulence, and pressure changes; continuity still expresses mass conservation, but this simple calculation alone does not predict every pressure or jet detail. For compressible flow, density must be included.

The general habit is to draw a control volume and count what enters and leaves. That same logic applies to traffic through a doorway, charge in a wire, or people moving through a corridor. In the pipe example, the radius is halved, area becomes one quarter, and speed rises from 0.5 to 2 meters per second.""",
        "long_extension": """Pressure and speed are connected by a different statement, Bernoulli's equation, under additional ideal-flow assumptions. Continuity alone does not say why the pressure changed or how much energy a pump must supply. Keeping those laws separate prevents a common leap: this area calculation gives a speed ratio, not a complete pressure analysis.""",
        "visual": [
            {"type": "pipe_with_gauge", "pos": [0, 0, 0], "color": "#3498DB"},
            {"type": "vector", "start": [-2.8, 0, 0], "end": [-1.3, 0, 0],
             "label": "v₁", "color": "#F1C40F"},
            {"type": "vector", "start": [1.0, 0, 0], "end": [2.8, 0, 0],
             "label": "v₂", "color": "#2ECC71"},
        ],
        "equations": ["Q=Av", "A1v1=A2v2", "v2=4v1=2 m/s"],
        "beats_short": ["hook", "law", "emphasis", "experiment", "punchline"],
        "beats_long": ["hook", "promise", "analogy", "law", "emphasis", "experiment", "punchline", "cta"],
    },
    "mix06": {
        "short": """At the midpoint between a positive and negative charge, do their electric fields cancel? No. Both point from the positive charge toward the negative one there.

Put plus and minus 2 microcoulombs 20 centimeters apart. The midpoint is 10 centimeters, or 0.10 meters, from each. Coulomb's field magnitude is k times absolute q divided by r squared. With k equal to 9 times 10 to the ninth, each field is 9e9 times 2e-6 divided by 0.10 squared: 1.8 million newtons per coulomb.

The directions match, so the fields add: 3.6 million newtons per coulomb toward the negative charge. We add vectors, not charge labels. If the charges had the same sign, the midpoint directions would oppose.

Electric field is force per unit positive test charge. A probe then feels F=qE. Symmetry gives direction; the inverse-square law gives magnitude.""",
        "long": """Place a positive and a negative charge on a line. At their midpoint, it is tempting to say their effects cancel because the charges have equal magnitude. But fields are vectors, and direction decides whether contributions add or subtract.

Electric field is force per unit positive test charge. A positive source creates a field pointing away from itself; a negative source creates a field pointing toward itself. At the midpoint between opposite charges, both arrows point from positive to negative. So they reinforce.

Let's calculate. The charges are plus and minus 2 microcoulombs, separated by 0.20 meters. The midpoint is 0.10 meters from each. For one point charge, field magnitude is k absolute q over r squared. Use k equals 9 times 10 to the ninth newton meter squared per coulomb squared. One contribution is 9e9 times 2e-6 divided by 0.10 squared, or 1.8e6 newtons per coulomb. The second has the same magnitude and same direction, so total field is 3.6e6 newtons per coulomb toward the negative charge.

Check units: k q over r squared gives newtons per coulomb. If a positive test charge of 1 nanocoulomb were placed there, its force would be qE, about 3.6 millinewtons in that direction. The test charge is a probe; we assume it is small enough not to rearrange the source charges.

Now compare equal charges with the same sign. At their midpoint the field arrows point in opposite directions and cancel. The source charges have not changed in magnitude; only the geometry of the vectors changed. Away from the midpoint, the distances and directions must both be included.

This is why field diagrams are more than decoration. Draw a contribution from every source at the observation point, then add tip-to-tail. The midpoint result is large because both arrows agree, not small because the source labels look opposite.""",
        "long_extension": """A final check is to imagine changing the probe charge. The electric field at that location remains the same in this ideal model, while the force scales with the probe charge. That distinction is why field is a property of the sources and position, whereas force describes what a particular test charge experiences.""",
        "visual": [
            {"type": "charge", "pos": [-2, 0, 0], "label": "+q", "color": "#FF4B4B"},
            {"type": "charge", "pos": [2, 0, 0], "label": "−q", "color": "#00D2FF"},
            {"type": "vector", "start": [0, 0, 0], "end": [1.7, 0, 0],
             "label": "E", "color": "#F1C40F"},
        ],
        "equations": ["E=k|q|/r²", "E_total=E1+E2", "E_mid=3.6×10⁶ N/C"],
        "beats_short": ["hook", "law", "emphasis", "experiment", "punchline"],
        "beats_long": ["hook", "promise", "analogy", "law", "emphasis", "experiment", "punchline", "cta"],
    },
    "mix07": {
        "short": """A magnet moving near a coil can make a meter twitch. The important quantity is not magnetic field alone; it is magnetic flux and how quickly it changes.

For a uniform field through a flat loop, flux is B A cosine theta. Faraday's law says induced voltage equals minus the rate of change of flux. Suppose one turn's flux changes from 0.01 to 0.03 webers in 0.10 seconds. The average voltage magnitude is change in flux divided by time: 0.02 over 0.10, or 0.20 volts.

The minus sign gives the direction: the induced current's magnetic effect opposes the change in flux. It does not mean the voltage magnitude is negative. A steady field through a stationary loop has constant flux, so this simple setup induces no voltage.

Change field, area, or angle and flux changes. A generator rotates its coil; voltage measures flux-change rate.""",
        "long": """Move a magnet toward a coil and a connected meter deflects. Hold the magnet still and the deflection settles. The difference is a changing magnetic flux through the loop.

For a uniform magnetic field and a flat loop, flux is Phi equals B A cosine theta. Here theta is the angle between the field and the loop's area direction, perpendicular to its surface. Flux can change if the field changes, the loop area changes, or the loop rotates.

Faraday's law says the induced electromotive force is minus the rate of change of flux. For one turn, suppose flux rises from 0.01 webers to 0.03 webers over 0.10 seconds. The average change is 0.02 webers. Divide by 0.10 seconds to get a voltage magnitude of 0.20 volts. The units work because one weber per second is one volt.

Why the minus sign? It gives the direction described by Lenz's law: the induced current creates a magnetic effect opposing the change in flux. If the external flux into the page is increasing, the induced field points out of the page. The loop resists the change; it does not simply oppose the original field in every situation.

This distinction makes a useful prediction. A large constant magnetic field through a stationary loop has constant flux and produces no induced voltage in this ideal law. A smaller field that changes quickly can produce a larger voltage. The rate of change matters.

In a generator, a coil rotates in a magnetic field, so its angle changes and the flux rises and falls. The numerical loop here is one turn; with N identical turns, the induced voltage scales by N. We found 0.20 volts in magnitude. The sign and current direction require knowing whether flux is increasing or decreasing and how the loop is oriented.""",
        "long_extension": """This model also links geometry to calculus. Area and orientation determine how much field passes through the loop; the time derivative measures how quickly that amount changes. A graph of flux against time makes the voltage visible as a slope. A steeper slope means a larger induced emf, while reversing the slope reverses its sign.""",
        "visual": [
            {"type": "induction_coil", "pos": [0, 0, 0], "color": "#F1C40F"},
            {"type": "vector", "start": [-3, 0, 0], "end": [-1.5, 0, 0],
             "label": "B", "color": "#3498DB"},
            {"type": "graph", "expression": "0.01+0.02*x", "x_range": [0, 1],
             "y_range": [0, 0.04], "color": "#2ECC71"},
        ],
        "equations": ["Φ=BA cos θ", "ε=−dΦ/dt", "|εavg|=0.20 V"],
        "beats_short": ["hook", "law", "emphasis", "experiment", "punchline"],
        "beats_long": ["hook", "promise", "analogy", "law", "emphasis", "experiment", "punchline", "cta"],
    },
    "mix08": {
        "short": """A matrix can stretch some directions more than others. An eigenvector is a direction that comes out pointing the same way, only scaled.

Take the matrix with rows 2, negative 1 and negative 1, 2. Try the vector [1,1]. The result is [1,1], so its eigenvalue is 1. Try [1, negative 1]. The result is [3, negative 3], which is three times the original, so its eigenvalue is 3.

Why care about these arrows? In coupled oscillators, they are normal-mode shapes: the masses move in coordinated patterns that keep their shape while oscillating. A complicated motion can be split into these simpler modes.

The rule is matrix times vector equals eigenvalue times vector. In dynamics, frequencies can depend on square roots of related eigenvalues, so the model must define that connection.""",
        "long": """Imagine a transformation that acts on every arrow in a plane. Most arrows turn and stretch. But a few special directions come out pointing along themselves. Those are eigenvectors; their scale factors are eigenvalues.

Use the symmetric matrix whose rows are [2, negative 1] and [negative 1, 2]. Multiply it by [1,1]. The first component is 2 minus 1, or 1; the second is negative 1 plus 2, also 1. The result is [1,1]. This vector is unchanged, so its eigenvalue is 1.

Now multiply by [1, negative 1]. The result is [3, negative 3], which is three times the original vector. So [1, negative 1] is another eigenvector and its eigenvalue is 3. These two directions are independent, so they form a basis for the plane.

This connects directly to coupled motion. Picture two identical masses linked so each affects the other. If they move together in the [1,1] pattern, that shape may have one natural response. If they move oppositely in the [1, negative 1] pattern, the coupling changes the restoring effect, giving a different response. These patterns are called normal modes.

Why is that useful? A general initial displacement can be expressed as a sum of mode shapes. Instead of solving one tangled motion, solve each mode and add them. In a mass-spring model, the exact relationship between matrix eigenvalues and squared angular frequencies comes from the equation M x double-dot plus K x equals zero; the stiffness and mass matrices both matter. The small example alone does not specify a frequency.

So the eigenvector is the shape of a coordinated motion, and its eigenvalue records how the chosen matrix scales that shape. First verify by multiplication, then interpret the matrix in the physical model. The interpretation is earned by the equations, not by the word “eigen.”""",
        "long_extension": """For our symmetric example, the two perpendicular mode directions also make the geometry easy to see. A displacement can be decomposed into a sum along [1,1] and [1, negative 1]. The matrix acts on each component separately, scaling one by 1 and the other by 3. That is the visual meaning of diagonalization: choose coordinates aligned with the natural directions.""",
        "visual": [
            {"type": "vector", "start": [0, 0, 0], "end": [1.5, 1.5, 0],
             "label": "[1,1]", "color": "#3498DB"},
            {"type": "vector", "start": [0, 0, 0], "end": [1.5, -1.5, 0],
             "label": "[1,−1]", "color": "#F1C40F"},
            {"type": "grid_warp", "pos": [0, 0, 0], "color": "#2ECC71"},
        ],
        "equations": ["A v=λv", "A[1,1]=1[1,1]", "A[1,−1]=3[1,−1]"],
        "beats_short": ["hook", "law", "emphasis", "experiment", "punchline"],
        "beats_long": ["hook", "promise", "analogy", "law", "emphasis", "experiment", "punchline", "cta"],
    },
    "mix09": {
        "short": """Why does a light ray bend when it enters glass? Light travels more slowly in glass, so a path that spends less distance there can take less time.

Snell's law is n1 sine theta1 equals n2 sine theta2, with angles measured from the normal. From air into glass, take n1 equal to 1, n2 equal to 1.5, and incidence angle 30 degrees.

Then sine theta2 equals (1 over 1.5) times sine 30, which is one third. The refracted angle is arcsine one third, about 19.5 degrees. It is smaller than 30 degrees, so the ray bends toward the normal.

The shortest path is not always fastest when speeds differ. Fermat's principle makes travel time stationary; Snell's law follows at a flat boundary between uniform media.""",
        "long": """A ray entering glass changes direction at the boundary. If light simply chose the shortest distance, why would it bend? Because the speed of light differs in the two materials. The relevant comparison is travel time, not distance alone.

For a uniform medium, refractive index n is the speed of light in vacuum divided by its speed in that medium. Fermat's principle says the physical ray makes travel time stationary under small changes of path. At a flat boundary, comparing nearby crossing points gives Snell's law: n1 sine theta1 equals n2 sine theta2. The angles are measured from the normal, the line perpendicular to the surface.

Let's calculate air to glass. Use n1 equals 1.0, n2 equals 1.5, and incidence angle 30 degrees. Then sine theta2 equals (n1/n2) sine 30. That is (1/1.5) times 0.5, or one third. Taking the inverse sine gives theta2 about 19.5 degrees. Since the angle from the normal got smaller, the ray bent toward the normal.

There is a helpful running analogy. Suppose you move quickly along a path on pavement but slowly through sand. The fastest route to a target may be longer on the pavement and shorter in sand, because crossing at a different point changes how much time is spent at each speed. Light's path balances the same kind of tradeoff, though its exact rule is wave physics.

The phrase “least time” can be misleading. More generally the path is stationary in travel time; in some setups that need not be a global minimum. This simple calculation also assumes homogeneous materials and a planar interface. The refracted direction follows from the boundary condition, not from a ray independently deciding what it wants.

For the given indices and 30-degree incidence, the answer is about 19.5 degrees. A useful visual test is to draw the normal first. Then compare both angles from that line, not from the glass surface.""",
        "long_extension": """The sine factors are not arbitrary decorations. They arise because moving the crossing point a little changes the path lengths along each side by the horizontal projection of that shift. At the stationary-time path, those changes balance after each region's travel speed is included. The boundary law is a compact trace of that geometric balance.""",
        "visual": [
            {"type": "ray", "start": [-3, 1.5, 0], "end": [0, 0, 0],
             "travel_time": 3, "color": "#F1C40F"},
            {"type": "ray", "start": [0, 0, 0], "end": [2, -2.5, 0],
             "travel_time": 3, "color": "#3498DB"},
            {"type": "line", "start": [0, -3, 0], "end": [0, 3, 0],
             "label": "normal", "color": "#8A8F9C"},
        ],
        "equations": ["n1 sin θ1=n2 sin θ2", "sin θ2=1/3", "θ2≈19.5°"],
        "beats_short": ["hook", "law", "emphasis", "experiment", "punchline"],
        "beats_long": ["hook", "promise", "analogy", "law", "emphasis", "experiment", "punchline", "cta"],
    },
    "mix10": {
        "short": """A random walk looks unpredictable, but the crowd's spread follows a simple rule. If independent steps have zero mean, their variances add.

Suppose each of 100 steps has a standard deviation of 1 centimeter. Variance per step is 1 centimeter squared. After 100 independent steps, variance is 100 centimeter squared, so standard deviation is the square root: 10 centimeters.

It grows like square root of N, not N. A single walker might end far from the start or close to it; the standard deviation describes the ensemble's typical spread, not a guaranteed distance for every person.

This is a bridge from random microscopic motion to diffusion. Many irregular steps create a predictable widening cloud. Random does not mean without structure; it means we predict distributions rather than the exact path.""",
        "long": """Watch one person taking random steps: their position is hard to predict. Now watch a thousand people taking independent steps. Their individual paths differ, but the distribution of positions develops a predictable width.

Suppose each one-dimensional step has mean zero and standard deviation 1 centimeter. For independent steps, variances add. One step has variance 1 centimeter squared. After N steps, variance is N centimeter squared, and standard deviation is square root of N centimeters. At N equals 100, the standard deviation is 10 centimeters.

Why not 100 centimeters? That would be the distance if every step pointed in the same direction. Random positive and negative steps partly cancel. The expected displacement can remain zero, while the spread grows. Standard deviation measures the typical scale of that spread; it does not promise every walker is exactly 10 centimeters away.

This is a powerful change in description. For a single person, we may need the entire sequence of steps. For a large group, we can often predict a probability distribution. As time passes, the distribution broadens. In physical systems, this connection leads from random molecular motion to diffusion of particles, heat, or dye.

There are assumptions hidden in the square-root rule: steps are independent, have finite variance, and share the same scale. If steps are correlated, or if some rare jumps are enormous, the growth can be different. Also, centimeters are units of position; variance carries squared units, which is why we take the square root at the end.

For 100 independent steps with standard deviation 1 centimeter, the final spread scale is about 10 centimeters, not 100. One path may surprise us. The distribution of many paths can still be measured and modeled. That is how statistical physics finds order without pretending to predict every collision.""",
        "long_extension": """The distribution carries more information than its center alone. If its mean stays at zero but its width grows, saying “the walkers went nowhere” misses the spreading. Measuring both average and spread separates systematic drift from random fluctuations, a distinction used in experiments from particle tracking to stock-price models. It also reminds us that a prediction about a population is not a promise about one member of it.""",
        "visual": [
            {"type": "particle_gas", "num_particles": 32, "box": [6, 3.8],
             "color": "#3498DB"},
            {"type": "graph", "expression": "exp(-0.5*x*x)", "x_range": [-3, 3],
             "y_range": [0, 1], "color": "#F1C40F"},
        ],
        "equations": ["Var(sum)=sum Var", "σN=σ√N", "σ100=10 cm"],
        "beats_short": ["hook", "law", "emphasis", "experiment", "punchline"],
        "beats_long": ["hook", "promise", "analogy", "law", "emphasis", "experiment", "punchline", "cta"],
    },
}

BEAT_TEXT = {
    "hook": "A question worth seeing",
    "promise": "Build the idea from a simple model",
    "analogy": "Make the invisible mechanism visible",
    "law": "The governing relationship",
    "emphasis": "Pause on the key step",
    "experiment": "Test the prediction with numbers",
    "punchline": "What the result teaches us",
    "cta": "Carry this model to the next question",
}


def _beats(types: list[str]) -> list[dict[str, str]]:
    return [
        {"type": beat_type, "text": BEAT_TEXT[beat_type], "focus": ""}
        for beat_type in types
    ]


def _load_plan() -> list[dict[str, str]]:
    with PLAN.open(encoding="utf-8", newline="") as stream:
        plan = list(csv.DictReader(stream))
    if len(plan) != 10 or len({row["lesson_id"] for row in plan}) != 10:
        raise ValueError("lesson plan must contain exactly 10 unique lesson IDs")
    if set(row["lesson_id"] for row in plan) != set(SCRIPTS):
        raise ValueError("plan lesson IDs must match the authored script IDs")
    return plan


def _select_lessons(
    plan: list[dict[str, str]],
    lesson_ids: list[str] | tuple[str, ...] | None,
) -> list[dict[str, str]]:
    if lesson_ids is None:
        return plan
    if not lesson_ids or len(set(lesson_ids)) != len(lesson_ids):
        raise ValueError("selected lesson IDs must be a non-empty unique list")
    lessons_by_id = {lesson["lesson_id"]: lesson for lesson in plan}
    missing = set(lesson_ids) - set(lessons_by_id)
    if missing:
        raise ValueError(f"unknown lesson IDs: {', '.join(sorted(missing))}")
    return [lessons_by_id[lesson_id] for lesson_id in lesson_ids]


def _build_rows(
    lessons: list[dict[str, str]],
    orientation: str,
) -> list[dict[str, str]]:
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError("orientation must be horizontal or vertical")
    rows = []
    for lesson in lessons:
        lesson_id = lesson["lesson_id"]
        content = SCRIPTS[lesson_id]
        visual = content["visual"]
        rows.append({
                "video_id": f"{lesson_id}_{orientation}",
                "concept_type": "GENERIC",
                "header_title": lesson["lesson_title"],
                "tagline": lesson["central_question"],
                "question_text": lesson["worked_problem"],
                "option_a": "",
                "option_b": "",
                "option_c": "",
                "option_d": "",
                "correct_answer": "",
                "equations_json": json.dumps(content["equations"], ensure_ascii=False),
                "visual_data_json": json.dumps(visual, ensure_ascii=False),
                "audio_script": (
                    content["long"] + "\n\n" + content["long_extension"]
                    if orientation == "horizontal" else content["short"]
                ),
                "orientation": orientation,
                "beats_json": json.dumps(
                    _beats(content["beats_long" if orientation == "horizontal" else "beats_short"]),
                    ensure_ascii=False,
                ),
                "bindings_json": json.dumps([
                    {"equation_idx": 0, "visual_id": visual[0]["type"], "color": "#F1C40F"}
                ]),
                "language": "en",
                "quality": "1080p",
                "platform": "youtube" if orientation == "horizontal" else "shorts",
                "lesson_id": lesson_id,
            })
    return rows


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def export(
    output_dir: Path,
    lesson_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Path]:
    """Export all lessons, or a requested selection, in both orientations."""
    plan = _select_lessons(_load_plan(), lesson_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for orientation in ("horizontal", "vertical"):
        path = output_dir / f"content_batch_{orientation}.csv"
        _write_rows(path, _build_rows(plan, orientation))
        outputs[orientation] = path
    return outputs


def export_one(
    output_path: Path,
    orientation: str,
    lesson_ids: list[str] | tuple[str, ...],
) -> Path:
    """Write selected authored lessons to one existing render-schema CSV."""
    plan = _select_lessons(_load_plan(), lesson_ids)
    _write_rows(output_path, _build_rows(plan, orientation))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    outputs = export(args.output_dir)
    for orientation, path in outputs.items():
        with path.open(encoding="utf-8", newline="") as stream:
            count = sum(1 for _ in csv.DictReader(stream))
        print(f"{orientation}: {path} ({count} lessons)")


if __name__ == "__main__":
    main()
