"""
One-click data updater (run on your own computer with internet access):

    python fetch_data.py        # update Elo ratings + match results
    python engine/simulate.py   # re-run predictions & rebuild dashboard

or simply:  python fetch_data.py --simulate

Data sources:
- Elo ratings:  https://www.eloratings.net/World.tsv  (no key needed)
- Match results: ESPN public scoreboard API (no key needed)
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from datetime import date, datetime, timedelta

ROOT = os.path.dirname(os.path.abspath(__file__))

# eloratings.net team codes used in teams.json
ELO_URL = "https://www.eloratings.net/World.tsv"
ESPN_URL = ("https://site.api.espn.com/apis/site/v2/sports/soccer/"
            "fifa.world/scoreboard?dates={d}")

UA = {"User-Agent": "Mozilla/5.0 (WC26-predictor; personal use)"}


def http_get(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def update_elo():
    path = os.path.join(ROOT, "data", "teams.json")
    with open(path) as f:
        tj = json.load(f)
    print("Fetching Elo ratings from eloratings.net ...")
    tsv = http_get(ELO_URL)
    ratings = {}
    for line in tsv.strip().splitlines():
        parts = line.split("\t")
        if len(parts) > 3:
            ratings[parts[2]] = float(parts[3])
    n = 0
    for teams in tj["groups"].values():
        for t in teams:
            if t["code"] in ratings:
                t["elo"] = ratings[t["code"]]
                n += 1
    tj["elo_snapshot_date"] = date.today().isoformat()
    with open(path, "w") as f:
        json.dump(tj, f, indent=2)
    # ratings now include everything played so far -> mark matches applied
    mpath = os.path.join(ROOT, "data", "matches.json")
    with open(mpath) as f:
        mj = json.load(f)
    for m in mj["matches"]:
        m["applied"] = True
    with open(mpath, "w") as f:
        json.dump(mj, f, indent=2)
    print(f"  updated {n}/48 team ratings (snapshot {tj['elo_snapshot_date']})")


def update_results(days_back=3):
    """Pull finished World Cup matches from ESPN's public scoreboard."""
    mpath = os.path.join(ROOT, "data", "matches.json")
    with open(mpath) as f:
        mj = json.load(f)
    known = {(m["date"], m["home"], m["away"]) for m in mj["matches"]}
    added = 0
    for k in range(days_back + 1):
        d = date.today() - timedelta(days=k)
        try:
            data = json.loads(http_get(ESPN_URL.format(d=d.strftime("%Y%m%d"))))
        except Exception as e:
            print(f"  ESPN fetch failed for {d}: {e}")
            continue
        for ev in data.get("events", []):
            comp = ev.get("competitions", [{}])[0]
            if comp.get("status", {}).get("type", {}).get("completed") is not True:
                continue
            sides = comp.get("competitors", [])
            if len(sides) != 2:
                continue
            home = next((s for s in sides if s.get("homeAway") == "home"), None)
            away = next((s for s in sides if s.get("homeAway") == "away"), None)
            if not home or not away:
                continue
            rec = (d.isoformat(), home["team"]["displayName"],
                   away["team"]["displayName"])
            if rec in known:
                continue
            note = (comp.get("notes") or [{}])[0].get("headline", "")
            stage = "group" if "Group" in note else "knockout"
            group = note.split("Group ")[-1][:1] if "Group" in note else ""
            mj["matches"].append({
                "date": rec[0], "stage": stage, "group": group,
                "home": rec[1], "away": rec[2],
                "hs": int(home.get("score", 0)), "as": int(away.get("score", 0)),
                "applied": False,
            })
            known.add(rec)
            added += 1
    with open(mpath, "w") as f:
        json.dump(mj, f, indent=2)
    print(f"  added {added} new finished match(es)")


NAME_FIX = {"USA": "United States", "Turkey": "Turkiye",
            "Czech Republic": "Czechia",
            "Bosnia & Herzegovina": "Bosnia and Herzegovina",
            "Curaçao": "Curacao", "Korea Republic": "South Korea"}


def update_odds():
    """Pull Pinnacle odds via The Odds API -> de-vigged market signals,
    with day-over-day sudden-move detection."""
    spath = os.path.join(ROOT, "data", "signals.json")
    with open(spath) as f:
        sj = json.load(f)
    # key resolution: env var (GitHub Actions secret) > data/secrets.json (local)
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        sec = os.path.join(ROOT, "data", "secrets.json")
        if os.path.exists(sec):
            with open(sec) as f:
                key = json.load(f).get("odds_api_key")
    tpl = sj.get("odds_api", {}).get("url_template")
    if not (key and tpl):
        print("  no odds API key/template configured; skipping odds")
        return
    url = tpl.replace("{KEY}", key)
    print("Fetching Pinnacle odds (The Odds API) ...")
    events = json.loads(http_get(url))
    today = date.today().isoformat()
    n, moves = 0, 0
    for ev in events:
        bks = ev.get("bookmakers") or []
        if not bks:
            continue
        outcomes = bks[0]["markets"][0]["outcomes"]
        h = NAME_FIX.get(ev["home_team"], ev["home_team"])
        a = NAME_FIX.get(ev["away_team"], ev["away_team"])
        o = {x["name"]: x["price"] for x in outcomes}
        oh = o.get(ev["home_team"]); oa = o.get(ev["away_team"]); od = o.get("Draw")
        if not (oh and oa and od):
            continue
        inv = [1/oh, 1/oa, 1/od]; tot = sum(inv)
        ph, pa, pd = [x/tot for x in inv]
        we = round(ph + 0.5*pd, 4)
        key = f"{h}|{a}"
        old = sj["market"].get(key, {}).get("we_home")
        note = "Pinnacle盘口(The Odds API),去水隐含概率"
        if old is not None and abs(we - old) > 0.08:
            note = f"⚠ 24小时盘口大幅异动: {old*100:.0f}% → {we*100:.0f}%"
            moves += 1
        sj["market"][key] = {"we_home": we, "ph": round(ph,4), "pa": round(pa,4),
                             "pd": round(pd,4), "odds": [oh, oa, od],
                             "date": today, "note": note}
        n += 1
    with open(spath, "w") as f:
        json.dump(sj, f, indent=1, ensure_ascii=False)
    print(f"  odds updated for {n} matches; {moves} sudden move(s) flagged")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true",
                    help="run the simulation after updating data")
    ap.add_argument("--sims", type=int, default=10000)
    args = ap.parse_args()

    try:
        update_results()   # results first (before they get baked into Elo)
    except Exception as e:
        print(f"Result update failed (will still try Elo): {e}")
    try:
        update_elo()
    except Exception as e:
        print(f"Elo update failed: {e}")
    try:
        update_odds()
    except Exception as e:
        print(f"Odds update failed: {e}")

    if args.simulate:
        subprocess.run([sys.executable,
                        os.path.join(ROOT, "engine", "simulate.py"),
                        "--sims", str(args.sims)], check=True)


if __name__ == "__main__":
    main()
