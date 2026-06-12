"""
World Cup 2026 prediction model.

Two layers, as designed:
1. Elo layer  - live World Football Elo ratings (eloratings.net), updated
   match-by-match with the standard Elo formula (K=60 for World Cup,
   goal-difference multiplier).
2. Adaptive ML calibration layer - a Bayesian logistic regression that
   re-fits on the accumulated 2026 tournament results every time you update.
   It learns corrections to the raw Elo win expectancy (e.g. if favorites
   are over/under-performing this tournament) and a goal-rate multiplier.
   With few matches it stays close to its prior; as results accumulate the
   data takes over (MAP estimation with an L2 prior).

Match scores are simulated with independent Poisson goal models whose
rates are driven by the calibrated win expectancy.
"""

import math
import random

HOME_BONUS = 100.0          # Elo bonus for host nations (US/MEX/CAN play at home)
K_WORLDCUP = 60.0           # Elo K-factor for World Cup matches
BASE_GOALS = 2.55           # prior expected total goals per WC match
PRIOR_STRENGTH = 25.0       # pseudo-match count anchoring the ML prior


# ----------------------------------------------------------------------
# Elo layer
# ----------------------------------------------------------------------

def win_expectancy(elo_a, elo_b):
    """Classic Elo win expectancy for side A."""
    return 1.0 / (1.0 + 10.0 ** (-(elo_a - elo_b) / 400.0))


def elo_update(elo_h, elo_a, hs, as_, k=K_WORLDCUP):
    """Return new (elo_h, elo_a) after a result hs:as_."""
    we = win_expectancy(elo_h, elo_a)
    if hs > as_:
        res = 1.0
    elif hs < as_:
        res = 0.0
    else:
        res = 0.5
    gd = abs(hs - as_)
    if gd <= 1:
        g = 1.0
    elif gd == 2:
        g = 1.5
    else:
        g = (11.0 + gd) / 8.0
    delta = k * g * (res - we)
    return elo_h + delta, elo_a - delta


# ----------------------------------------------------------------------
# Adaptive ML calibration layer
# ----------------------------------------------------------------------

class Calibrator:
    """
    Logistic regression  P(home beats away | not draw) = sigmoid(a + b * logit(We))
    with Gaussian prior a~N(0, .) , b~N(1, .)  (i.e. "trust raw Elo" prior).
    Also learns a goal-rate multiplier shrunk toward 1.0.
    Fit by MAP gradient descent on this tournament's matches only -
    this is the dynamic, self-updating part of the model.
    """

    def __init__(self):
        self.a = 0.0
        self.b = 1.0
        self.goal_mult = 1.0
        self.n_obs = 0

    @staticmethod
    def _logit(p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    def calibrated_we(self, raw_we):
        return self._sigmoid(self.a + self.b * self._logit(raw_we))

    def fit(self, samples, total_goals_obs=None, n_matches=0):
        """
        samples: list of (raw_we, outcome) for decisive comparisons,
                 outcome 1 if the Elo-favored computation's home side won,
                 0 if it lost, 0.5 for draws (draws get half weight each way).
        """
        self.n_obs = len(samples)
        if samples:
            lr = 0.05
            a, b = 0.0, 1.0
            for _ in range(800):
                # Gaussian prior a~N(0,1/PRIOR_STRENGTH), b~N(1,1/PRIOR_STRENGTH):
                # acts like PRIOR_STRENGTH pseudo-observations anchoring Elo.
                ga = PRIOR_STRENGTH * (a - 0.0)
                gb = PRIOR_STRENGTH * (b - 1.0)
                for we, y in samples:
                    x = self._logit(we)
                    p = self._sigmoid(a + b * x)
                    ga += (p - y)
                    gb += (p - y) * x
                n = len(samples) + PRIOR_STRENGTH
                a -= lr * ga / n
                b -= lr * gb / n
            self.a, self.b = a, b
        # goal-rate multiplier, shrunk toward 1.0
        if total_goals_obs is not None and n_matches > 0:
            expected = BASE_GOALS * n_matches
            raw = total_goals_obs / expected if expected > 0 else 1.0
            w = n_matches / (n_matches + PRIOR_STRENGTH)
            self.goal_mult = (1 - w) * 1.0 + w * raw
            self.goal_mult = min(max(self.goal_mult, 0.7), 1.4)


# ----------------------------------------------------------------------
# Match simulation
# ----------------------------------------------------------------------

def goal_rates(we, goal_mult=1.0):
    """Split expected total goals between the sides based on win expectancy."""
    total = BASE_GOALS * goal_mult
    lh = total * (we ** 1.1) / (we ** 1.1 + (1 - we) ** 1.1)
    la = total - lh
    return max(lh, 0.15), max(la, 0.15)


def poisson(lam):
    """Knuth's algorithm - keeps the engine dependency-free."""
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        p *= random.random()
        if p <= L:
            return k
        k += 1


def simulate_match(elo_h, elo_a, cal, knockout=False):
    """Return (home_goals, away_goals, home_advances_if_knockout)."""
    we = cal.calibrated_we(win_expectancy(elo_h, elo_a))
    lh, la = goal_rates(we, cal.goal_mult)
    hs, as_ = poisson(lh), poisson(la)
    if not knockout or hs != as_:
        return hs, as_, hs > as_
    # extra time (~1/3 of regulation rates)
    eh, ea = poisson(lh / 3.0), poisson(la / 3.0)
    hs += eh
    as_ += ea
    if hs != as_:
        return hs, as_, hs > as_
    # penalty shootout - mild quality edge
    p = 0.5 + (we - 0.5) * 0.3
    return hs, as_, random.random() < min(max(p, 0.35), 0.65)
