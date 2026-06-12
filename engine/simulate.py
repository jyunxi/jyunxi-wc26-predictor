"""
Monte Carlo simulator for the 2026 World Cup.

Usage:  python engine/simulate.py [--sims 10000]

Reads  data/teams.json + data/matches.json
Writes data/predictions.json and index.html (from dashboard_template.html)

Implements the official tournament format:
- 12 groups of 4, top 2 + 8 best third-placed teams advance (32 teams)
- Official Round-of-32 slot map (FIFA matches 73-88) with third-place
  slot constraints; allocation found by constraint matching
- Full knockout bracket through the final (matches 89-104)
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (Calibrator, elo_update, simulate_match, win_expectancy,
                   HOME_BONUS)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Official R32 structure. Each entry: (slot_a, slot_b) where
# "1A" = winner of group A, "2A" = runner-up, "3:CEFHI" = a third-placed
# team drawn from one of those groups (constraint set from FIFA Annex C).
R32 = [
    ("M73", "2A", "2B"),
    ("M74", "1E", "3:ABCDF"),
    ("M75", "1F", "2C"),
    ("M76", "1C", "2F"),
    ("M77", "1I", "3:CDFGH"),
    ("M78", "2E", "2I"),
    ("M79", "1A", "3:CEFHI"),
    ("M80", "1L", "3:EHIJK"),
    ("M81", "1D", "3:BEFIJ"),
    ("M82", "1G", "3:AEHIJ"),
    ("M83", "2K", "2L"),
    ("M84", "1H", "2J"),
    ("M85", "1B", "3:EFGIJ"),
    ("M86", "1J", "2H"),
    ("M87", "1K", "3:DEIJL"),
    ("M88", "2D", "2G"),
]
R16 = [("M89", "M74", "M77"), ("M90", "M73", "M75"), ("M91", "M76", "M78"),
       ("M92", "M79", "M80"), ("M93", "M83", "M84"), ("M94", "M81", "M82"),
       ("M95", "M86", "M88"), ("M96", "M85", "M87")]
QF = [("M97", "M89", "M90"), ("M98", "M93", "M94"),
      ("M99", "M91", "M92"), ("M100", "M95", "M96")]
SF = [("M101", "M97", "M98"), ("M102", "M99", "M100")]

THIRD_SLOTS = [(m, set(spec.split(":")[1])) for m, _, spec in
               [(m, a, b) for m, a, b in R32 if b.startswith("3:")]]


def load_data():
    with open(os.path.join(ROOT, "data", "teams.json")) as f:
        tj = json.load(f)
    with open(os.path.join(ROOT, "data", "matches.json")) as f:
        mj = json.load(f)
    fj, xj = {"teams": {}, "weights": {}}, {"fixtures": []}
    p = os.path.join(ROOT, "data", "factors.json")
    if os.path.exists(p):
        with open(p) as f:
            fj = json.load(f)
    p = os.path.join(ROOT, "data", "fixtures.json")
    if os.path.exists(p):
        with open(p) as f:
            xj = json.load(f)
    pj = {"teams": {}, "pos_baselines": {}}
    p = os.path.join(ROOT, "data", "players.json")
    if os.path.exists(p):
        with open(p) as f:
            pj = json.load(f)
    sj = {"referees": {}, "assignments": {}, "market": {}, "baseline": {}}
    p = os.path.join(ROOT, "data", "signals.json")
    if os.path.exists(p):
        with open(p) as f:
            sj = json.load(f)
    return tj, mj, fj, xj, pj, sj


def player_form(p, baselines):
    s = p.get("stats") or {}
    if s.get("min", 0) < 45:
        return 0.0
    b = baselines.get(p["pos"], {"pass": 82, "drib": 50, "g90": 0.2})
    f = 0.0
    g90 = s.get("g", 0) / (s["min"] / 90.0)
    if p["pos"] != "GK":
        f += max(-1.0, min(1.0, (g90 - b["g90"]) * 2.2))
    if s.get("pass") is not None:
        f += max(-0.7, min(0.7, (s["pass"] - b["pass"]) / 7.0))
    if s.get("drib") is not None and b["drib"] > 0:
        f += max(-0.4, min(0.4, (s["drib"] - b["drib"]) / 18.0))
    if s.get("a"):
        f += min(0.5, s["a"] * 0.25)
    return max(-2.0, min(2.0, f))


def age_coef(p, pj, w_age=1.0):
    """Position-specific age curve (mirrors JS engine)."""
    ac = pj.get("age_curve", {})
    lo, hi = ac.get(p["pos"], [24, 30])
    age = p.get("age")
    if age is None:
        return 1.0
    c = 1.0
    if age < lo:
        c = max(ac.get("min_young", 0.85), 1 - ac.get("pre_decline", 0.015) * (lo - age))
    elif age > hi:
        c = max(ac.get("min_old", 0.72), 1 - ac.get("post_decline", 0.02) * (age - hi))
    return 1 + (c - 1) * w_age


def q_eff(p, pj, w_league=1.0, w_age=1.0):
    """Quality scaled by club-league strength and age curve (mirrors JS)."""
    coef = pj.get("leagues", {}).get(p.get("lg", "OTHER"), 0.7)
    return (p["q"] * (1 + (coef - 1) * w_league)) * age_coef(p, pj, w_age)


def player_layer(team, pj, weight=1.0, w_league=1.0, w_age=1.0, w_exp=1.0):
    """(elo_adj, morale) - availability by importance, injury-proneness risk,
    in-tournament form, and positional-combination balance (mirrors JS)."""
    adj, morale = 0.0, 0.0
    pos_tot, pos_avail = {}, {}
    pos_w = pj.get("pos_weights", {"GK": 0.8, "DEF": 1.0, "MID": 1.1, "FWD": 1.0})
    for p in pj.get("teams", {}).get(team, []):
        q = q_eff(p, pj, w_league, w_age)
        pos_tot[p["pos"]] = pos_tot.get(p["pos"], 0) + q
        out = 0.0
        if p.get("st") in ("injured", "suspended") or p.get("ban", 0) > 0:
            out = 1.0
        elif p.get("st") == "doubtful":
            out = 0.5
        if out:
            adj -= max(0, q - 5) * 9 * out
            pos_avail[p["pos"]] = pos_avail.get(p["pos"], 0) + q * (1 - out)
            continue
        pos_avail[p["pos"]] = pos_avail.get(p["pos"], 0) + q
        risk = (p.get("inj", 1) + (0.5 if p.get("age", 27) > 32 else 0))
        adj -= max(0, q - 5) * 9 * risk * 0.05      # expected-absence discount
        adj += min(p.get("wc", 0), 3) / 3.0 * (q / 10.0) * 4 * w_exp  # WC experience
        adj += p.get("mental", 0) * (q / 10.0) * 2 * w_exp            # documented temperament
        f = player_form(p, pj.get("pos_baselines", {}))
        adj += f * q * 0.9
        morale += f * q / 10.0 * 0.25
    # star-dependence: quality concentration is easier to scheme against
    avail_q = [q_eff(p, pj, w_league, w_age)
               for p in pj.get("teams", {}).get(team, [])
               if p.get("st") not in ("injured", "suspended", "doubtful")
               and p.get("ban", 0) == 0]
    if len(avail_q) > 1:
        adj -= min(8.0, max(0.0, max(avail_q) / sum(avail_q) - 0.32) * 50)
    for pos, tot in pos_tot.items():                # positional balance penalty
        loss = 1 - pos_avail.get(pos, 0) / tot
        adj -= pos_w.get(pos, 1.0) * loss * loss * pj.get("balance_scale", 55)
    return (max(-150.0, min(120.0, adj)) * weight,
            max(-1.5, min(1.5, morale)))


def factor_adj(team, fj, pj):
    """Locker-room (news x credibility, rumor-discounted, + player-form morale)
    + WC pedigree + player availability/form layer."""
    f = fj.get("teams", {}).get(team)
    w = fj.get("weights", {})
    p_adj, p_morale = player_layer(team, pj, w.get("players", 1.0),
                                   w.get("league", 1.0), w.get("age", 1.0),
                                   w.get("experience", 1.0))
    if not f:
        return p_adj
    rumor_trust = w.get("rumor_trust", 0.5)
    morale = p_morale
    for n in f.get("news", []):
        cred = n["credibility"] * (rumor_trust if n["credibility"] < 0.7 else 1.0)
        morale += n["impact"] * cred
    morale = max(-5.0, min(5.0, morale))
    locker = morale / 5.0 * fj.get("locker_scale", 50) * w.get("locker_room", 1.0)
    ped = (f.get("pedigree", 10) - 30) / 70.0 * fj.get("pedigree_scale", 60) \
        * w.get("pedigree", 1.0)
    return locker + ped + p_adj


def current_elos(tj, matches):
    """Start from snapshot, apply any results not already baked in."""
    elo = {}
    group_of = {}
    for g, teams in tj["groups"].items():
        for t in teams:
            elo[t["name"]] = float(t["elo"])
            group_of[t["name"]] = g
    for m in matches:
        if not m.get("applied", False):
            h, a = m["home"], m["away"]
            elo[h], elo[a] = elo_update(elo[h], elo[a], m["hs"], m["as"])
    return elo, group_of


FACTORS = {"teams": {}, "weights": {}}   # set in main()
PLAYERS = {"teams": {}, "pos_baselines": {}}  # set in main()


def effective(elo, team, hosts):
    return (elo[team] + (HOME_BONUS if team in hosts else 0.0)
            + factor_adj(team, FACTORS, PLAYERS))


def fit_calibrator(elo_snapshot, matches, hosts):
    """Re-fit the adaptive layer on this tournament's played matches."""
    cal = Calibrator()
    samples, goals, n = [], 0, 0
    for m in matches:
        h, a = m["home"], m["away"]
        we = win_expectancy(effective(elo_snapshot, h, hosts),
                            effective(elo_snapshot, a, hosts))
        if m["hs"] > m["as"]:
            y = 1.0
        elif m["hs"] < m["as"]:
            y = 0.0
        else:
            y = 0.5
        samples.append((we, y))
        goals += m["hs"] + m["as"]
        n += 1
    cal.fit(samples, total_goals_obs=goals, n_matches=n)
    return cal


def group_table(teams, results):
    """results: list of (home, away, hs, as). Returns ordered team list."""
    pts = defaultdict(int)
    gd = defaultdict(int)
    gf = defaultdict(int)
    for h, a, hs, as_ in results:
        gf[h] += hs
        gf[a] += as_
        gd[h] += hs - as_
        gd[a] += as_ - hs
        if hs > as_:
            pts[h] += 3
        elif hs < as_:
            pts[a] += 3
        else:
            pts[h] += 1
            pts[a] += 1
    # head-to-head for exact two-way ties on (pts, gd, gf)
    h2h = {(h, a): (hs, as_) for h, a, hs, as_ in results}

    def sort_key(t):
        return (pts[t], gd[t], gf[t], random.random())

    order = sorted(teams, key=sort_key, reverse=True)
    # apply head-to-head correction for adjacent exact ties
    for i in range(len(order) - 1):
        t1, t2 = order[i], order[i + 1]
        if (pts[t1], gd[t1], gf[t1]) == (pts[t2], gd[t2], gf[t2]):
            r = h2h.get((t2, t1))
            if r and r[0] > r[1]:
                order[i], order[i + 1] = t2, t1
            r = h2h.get((t1, t2))
            if r and r[0] < r[1]:
                order[i], order[i + 1] = t2, t1
    return order, pts, gd, gf


def allocate_thirds(qualified_groups):
    """Constraint matching of 8 qualified third-place groups to the 8 slots."""
    slots = THIRD_SLOTS
    groups = list(qualified_groups)
    random.shuffle(groups)
    assignment = {}

    def backtrack(i):
        if i == len(slots):
            return True
        slot_match, allowed = slots[i]
        for g in groups:
            if g in allowed and g not in assignment.values():
                assignment[slot_match] = g
                if backtrack(i + 1):
                    return True
                del assignment[slot_match]
        return False

    # order slots by fewest options first for fast matching
    return assignment if backtrack(0) else None


def run_simulations(n_sims, tj, mj):
    hosts = set(tj["hosts"])
    played = mj["matches"]
    elo, group_of = current_elos(tj, played)
    cal = fit_calibrator(elo, played, hosts)

    groups = {g: [t["name"] for t in teams] for g, teams in tj["groups"].items()}
    played_group = defaultdict(list)
    for m in played:
        if m["stage"] == "group":
            played_group[m["group"]].append((m["home"], m["away"], m["hs"], m["as"]))

    # fixtures still to play in each group (round robin minus played)
    remaining = {}
    for g, ts in groups.items():
        done = {frozenset((h, a)) for h, a, _, _ in played_group[g]}
        rem = []
        for i in range(4):
            for j in range(i + 1, 4):
                if frozenset((ts[i], ts[j])) not in done:
                    rem.append((ts[i], ts[j]))
        remaining[g] = rem

    stats = {t: defaultdict(int) for t in elo}

    for _ in range(n_sims):
        winners, runners, thirds = {}, {}, {}
        third_info = {}
        for g, ts in groups.items():
            results = list(played_group[g])
            for h, a in remaining[g]:
                hs, as_, _ = simulate_match(effective(elo, h, hosts),
                                            effective(elo, a, hosts), cal)
                results.append((h, a, hs, as_))
            order, pts, gd, gf = group_table(ts, results)
            winners[g], runners[g] = order[0], order[1]
            thirds[g] = order[2]
            third_info[g] = (pts[order[2]], gd[order[2]], gf[order[2]],
                             random.random())
            stats[order[0]]["group_win"] += 1
            stats[order[0]]["advance"] += 1
            stats[order[1]]["advance"] += 1

        # best 8 thirds
        ranked = sorted(third_info, key=lambda g: third_info[g], reverse=True)
        qual_thirds = ranked[:8]
        for g in qual_thirds:
            stats[thirds[g]]["advance"] += 1

        assignment = allocate_thirds(qual_thirds)
        if assignment is None:  # extremely rare; re-allocate greedily
            assignment = {m: g for (m, _), g in zip(THIRD_SLOTS, qual_thirds)}

        def resolve(slot, match_id):
            if slot.startswith("1"):
                return winners[slot[1]]
            if slot.startswith("2"):
                return runners[slot[1]]
            return thirds[assignment[match_id]]

        alive = {}
        for mid, sa, sb in R32:
            ta, tb = resolve(sa, mid), resolve(sb, mid)
            stats[ta]["r32"] += 1
            stats[tb]["r32"] += 1
            _, _, a_wins = simulate_match(effective(elo, ta, hosts),
                                          effective(elo, tb, hosts), cal,
                                          knockout=True)
            alive[mid] = ta if a_wins else tb
        for rnd, key in ((R16, "r16"), (QF, "qf"), (SF, "sf")):
            for mid, ma, mb in rnd:
                ta, tb = alive[ma], alive[mb]
                stats[ta][key] += 1
                stats[tb][key] += 1
                _, _, a_wins = simulate_match(effective(elo, ta, hosts),
                                              effective(elo, tb, hosts), cal,
                                              knockout=True)
                alive[mid] = ta if a_wins else tb
        ta, tb = alive["M101"], alive["M102"]
        stats[ta]["final"] += 1
        stats[tb]["final"] += 1
        _, _, a_wins = simulate_match(effective(elo, ta, hosts),
                                      effective(elo, tb, hosts), cal,
                                      knockout=True)
        stats[ta if a_wins else tb]["champion"] += 1

    return elo, group_of, cal, stats, played


def build_output(n_sims, elo, group_of, cal, stats, played, tj, fj, xj, pj, sj):
    teams_out = []
    snapshot = {t["name"]: t["elo"] for g in tj["groups"].values() for t in g}
    for t, s in stats.items():
        teams_out.append({
            "name": t,
            "group": group_of[t],
            "elo": round(elo[t], 1),
            "elo_change": round(elo[t] - snapshot[t], 1),
            "p_advance": s["advance"] / n_sims,
            "p_group_win": s["group_win"] / n_sims,
            "p_r32": s["r32"] / n_sims,
            "p_r16": s["r16"] / n_sims,
            "p_qf": s["qf"] / n_sims,
            "p_sf": s["sf"] / n_sims,
            "p_final": s["final"] / n_sims,
            "p_champion": s["champion"] / n_sims,
        })
    teams_out.sort(key=lambda x: -x["p_champion"])
    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_sims": n_sims,
        "matches_played": len(played),
        "elo_snapshot_date": tj["elo_snapshot_date"],
        "calibration": {"a": round(cal.a, 4), "b": round(cal.b, 4),
                        "goal_mult": round(cal.goal_mult, 3),
                        "n_obs": cal.n_obs},
        "hosts": tj["hosts"],
        "groups": {g: [t["name"] for t in ts] for g, ts in tj["groups"].items()},
        "matches": played,
        "fixtures": xj.get("fixtures", []),
        "factors": fj,
        "players": pj,
        "signals": sj,
        "results": [
            {"date": m["date"], "home": m["home"], "away": m["away"],
             "score": f'{m["hs"]}-{m["as"]}', "group": m.get("group", "")}
            for m in played
        ],
        "teams": teams_out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=10000)
    args = ap.parse_args()

    tj, mj, fj, xj, pj, sj = load_data()
    FACTORS.clear()
    FACTORS.update(fj)
    PLAYERS.clear()
    PLAYERS.update(pj)
    elo, group_of, cal, stats, played = run_simulations(args.sims, tj, mj)
    out = build_output(args.sims, elo, group_of, cal, stats, played,
                       tj, fj, xj, pj, sj)

    with open(os.path.join(ROOT, "data", "predictions.json"), "w") as f:
        json.dump(out, f, indent=1)

    tpl_path = os.path.join(ROOT, "dashboard_template.html")
    if os.path.exists(tpl_path):
        with open(tpl_path) as f:
            html = f.read()
        html = html.replace("/*__DATA__*/null", json.dumps(out))
        with open(os.path.join(ROOT, "index.html"), "w") as f:
            f.write(html)

    top = out["teams"][:5]
    print(f"Simulated {args.sims} tournaments | {len(played)} real matches | "
          f"calibration a={cal.a:+.3f} b={cal.b:.3f} goals x{cal.goal_mult:.2f}")
    for t in top:
        print(f'  {t["name"]:<14} champion {t["p_champion"]*100:5.1f}%  '
              f'final {t["p_final"]*100:5.1f}%')


if __name__ == "__main__":
    main()
