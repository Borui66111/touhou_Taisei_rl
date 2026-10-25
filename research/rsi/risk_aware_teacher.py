"""E2 native-Ta​​isei risk-aware *baseline* controller, not a trained policy.

Uses only observations returned by the real Taisei native environment.
Prediction is locally linear, therefore not a collision oracle. We must compare
it to the existing geometric heuristic on identical *native* game seeds before
using it for imitation learning or calling it an improvement.
"""
from __future__ import annotations

import math

# Same movement index contract as native_rl.env.DIRECTIONS: still, U, D, L, R,
# UL, UR, DL, DR; indices 9-17 repeat those directions in Focus mode.
DIRECTIONS = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0),
              (-1, -1), (1, -1), (-1, 1), (1, 1))
WIDTH, HEIGHT = 480.0, 560.0
LEFT, RIGHT, TOP, BOTTOM = 16.0, 464.0, 16.0, 544.0


def _risk_for_bullet(b, vx, vy, horizon):
    if len(b) < 7 or not b[6]:
        return 0.0
    dx, dy = float(b[0]) * WIDTH, float(b[1]) * HEIGHT
    rvx, rvy = float(b[2]) * 20.0 - vx, float(b[3]) * 20.0 - vy
    relative_speed2 = rvx * rvx + rvy * rvy
    if relative_speed2 > 1e-10:
        t = max(0.0, min(float(horizon), -(dx * rvx + dy * rvy) / relative_speed2))
    else:
        t = 0.0
    sep = math.hypot(dx + rvx * t, dy + rvy * t)
    effective_radius = 6.0 + 30.0 * max(abs(float(b[4])), abs(float(b[5])))
    clearance = max(0.0, sep - effective_radius)
    # Earlier risk is weighted more heavily. This proxy does not handle
    # non-linear bullets, complex laser geometry, or invulnerability.
    return 110.0 * math.exp(-clearance / 14.0) / (1.0 + t / 18.0)


def _wall_penalty(px, py, nx, ny, vx, vy):
    # Penalize projected border proximity as well as repeatedly commanding
    # motion *toward* an already reached edge. Raw projected position is not
    # clamped when scoring, to avoid hiding movement saturation.
    margin = 36.0
    proximity = (
        max(0.0, LEFT + margin - nx) ** 2
        + max(0.0, nx - (RIGHT - margin)) ** 2
        + max(0.0, TOP + margin - ny) ** 2
        + max(0.0, ny - (BOTTOM - margin)) ** 2
    )
    penalty = 35.0 * proximity / (margin * margin)
    if px < LEFT + 12.0 and vx < 0:
        penalty += 65.0
    if px > RIGHT - 12.0 and vx > 0:
        penalty += 65.0
    if py < TOP + 12.0 and vy < 0:
        penalty += 65.0
    if py > BOTTOM - 12.0 and vy > 0:
        penalty += 65.0
    return penalty


def risk_aware_action(obs, horizon=18):
    """Choose 0..17 movement action; shooting remains automatic in TaiseiEnv.

    Deterministic and stateless, so it is a transparent teacher candidate,
    not a learned or provably safe shield. This version is fixed before native
    E2 evaluation; do not tune against the final sealed seeds.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 60:
        raise ValueError("Horizon must be an integer in 1..60")
    p = obs["player"]
    px, py = float(p[0]) * WIDTH, float(p[1]) * HEIGHT
    boss = obs["boss"]
    target_x = max(LEFT, min(RIGHT, px + float(boss[1]) * WIDTH)) if boss[0] else WIDTH / 2.0
    target_y = 480.0
    best_action, best_score = 0, float("-inf")
    for action in range(18):
        dx, dy = DIRECTIONS[action % 9]
        norm = math.hypot(dx, dy) or 1.0
        speed = 2.5 if action >= 9 else 5.5
        vx, vy = speed * dx / norm, speed * dy / norm
        nx, ny = px + vx * horizon, py + vy * horizon
        wall_cost = _wall_penalty(px, py, nx, ny, vx, vy)
        position_cost = .022 * abs(max(LEFT, min(RIGHT, nx)) - target_x) + .016 * abs(
            max(TOP, min(BOTTOM, ny)) - target_y)
        risk = sum(_risk_for_bullet(b, vx, vy, horizon) for b in obs["bullets"])
        # No simulated trajectories are invented here. Laser point samples in
        # the actual observation have no velocity, so penalize conservatively.
        for z in obs["lasers"]:
            if len(z) < 4 or not z[3]:
                continue
            lx, ly = float(z[0]) * WIDTH, float(z[1]) * HEIGHT
            t = min(horizon, 8)
            sep = math.hypot(lx - vx * t, ly - vy * t)
            clearance = max(0.0, sep - 30.0 * abs(float(z[2])) - 6.0)
            risk += 60.0 * math.exp(-clearance / 16.0)
        # Tiny preference for staying/focused maneuvering if all else equal.
        motion_cost = .04 * (abs(dx) + abs(dy))
        score = -risk - wall_cost - position_cost - motion_cost
        if score > best_score:
            best_score = score
            best_action = action
    return best_action
