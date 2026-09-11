"""
Day 11 — Fuzzy risk inference engine (v1).

Combines three inputs (probability, confidence, traffic_density) via a
Mamdani fuzzy inference system into a graded risk level (Low / Medium /
High / Critical), rather than trusting a single classifier's raw argmax
prediction — directly motivated by Day 10's finding that GaussianNB's hard
classification decisions are unreliable (severe BENIGN/attack confusion)
even where feature selection and tuning have already been applied.

IInputs (all normalized to [0, 1]):
  - probability:      P(attack) = 1 - P(BENIGN), i.e. the model's estimated
                       likelihood that this traffic is malicious (any attack
                       class), NOT simply the top predicted class's raw
                       probability. Design note: an earlier version used
                       P(top class), which produced backwards results for
                       confident-BENIGN predictions (scored as High risk,
                       since a confident BENIGN call has high P(top class)
                       and high confidence but is not actually risky).
                       Redefining as P(attack) fixes this by construction.
  - confidence:       margin between top-1 and top-2 class probabilities
                       across all 15 classes (decisiveness of the model's
                       specific call, independent of whether that call is
                       BENIGN or an attack type)
  - traffic_density:  normalized flow intensity (e.g. Flow Bytes/s, clipped
                       to the 1st-99th percentile per Day 5's EDA findings)
Output: risk_score (0-100, crisp) and risk_level (categorical).

This module defines the engine in isolation with synthetic test inputs.
Day 12 wires it up to real model predictions.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ---------------------------------------------------------------------------
# Antecedents (inputs) and consequent (output)
# ---------------------------------------------------------------------------

probability = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "probability")
confidence = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "confidence")
traffic_density = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "traffic_density")
risk = ctrl.Consequent(np.arange(0, 101, 1), "risk")

for var in (probability, confidence, traffic_density):
    var["low"] = fuzz.trapmf(var.universe, [0, 0, 0.25, 0.45])
    var["medium"] = fuzz.trimf(var.universe, [0.30, 0.50, 0.70])
    var["high"] = fuzz.trapmf(var.universe, [0.55, 0.75, 1.0, 1.0])

risk["low"] = fuzz.trapmf(risk.universe, [0, 0, 15, 35])
risk["medium"] = fuzz.trapmf(risk.universe, [20, 35, 50, 65])
risk["high"] = fuzz.trapmf(risk.universe, [50, 65, 80, 90])
risk["critical"] = fuzz.trapmf(risk.universe, [80, 90, 100, 100])

LEVEL_SCORE = {"low": 0, "medium": 1, "high": 2}


def score_to_risk_level(total_score: int) -> str:
    if total_score <= 1:
        return "low"
    elif total_score <= 3:
        return "medium"
    elif total_score == 4:
        return "high"
    else:  # 5 or 6
        return "critical"


rules = []
for p_level in ("low", "medium", "high"):
    for c_level in ("low", "medium", "high"):
        for d_level in ("low", "medium", "high"):
            total = LEVEL_SCORE[p_level] + LEVEL_SCORE[c_level] + LEVEL_SCORE[d_level]
            output_level = score_to_risk_level(total)
            rules.append(
                ctrl.Rule(
                    probability[p_level] & confidence[c_level] & traffic_density[d_level],
                    risk[output_level],
                )
            )

print(f"Generated {len(rules)} rules (exhaustive 3x3x3 coverage)")


risk_ctrl = ctrl.ControlSystem(rules)


def compute_risk(prob: float, conf: float, density: float) -> tuple[float, str]:
    """Run the fuzzy inference system on one set of inputs.

    Returns (risk_score, risk_level): risk_score is the crisp centroid-
    defuzzified value (0-100); risk_level is whichever output membership
    function has the highest degree of membership at that score.
    """
    sim = ctrl.ControlSystemSimulation(risk_ctrl)
    sim.input["probability"] = float(np.clip(prob, 0, 1))
    sim.input["confidence"] = float(np.clip(conf, 0, 1))
    sim.input["traffic_density"] = float(np.clip(density, 0, 1))
    sim.compute()

    risk_score = sim.output["risk"]

    memberships = {
        "Low": fuzz.interp_membership(risk.universe, risk["low"].mf, risk_score),
        "Medium": fuzz.interp_membership(risk.universe, risk["medium"].mf, risk_score),
        "High": fuzz.interp_membership(risk.universe, risk["high"].mf, risk_score),
        "Critical": fuzz.interp_membership(risk.universe, risk["critical"].mf, risk_score),
    }
    risk_level = max(memberships, key=memberships.get)

    return risk_score, risk_level


if __name__ == "__main__":
    scenarios = [
        ("Confident attack, heavy traffic", 0.95, 0.90, 0.90),
        ("Confident attack, light traffic", 0.95, 0.90, 0.10),
        ("Ambiguous prediction, heavy traffic", 0.55, 0.20, 0.85),
        ("Ambiguous prediction, light traffic", 0.55, 0.20, 0.15),
        ("Low-confidence guess, moderate traffic", 0.35, 0.15, 0.50),
        ("Confident benign-looking, light traffic", 0.90, 0.85, 0.05),
        ("Everything moderate", 0.50, 0.50, 0.50),
    ]

    print(f"{'Scenario':<42} {'Prob':>5} {'Conf':>5} {'Dens':>5}  ->  {'Score':>6}  {'Level'}")
    for name, p, c, d in scenarios:
        score, level = compute_risk(p, c, d)
        print(f"{name:<42} {p:>5.2f} {c:>5.2f} {d:>5.2f}  ->  {score:>6.2f}  {level}")
