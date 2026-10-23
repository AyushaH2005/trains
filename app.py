"""Harbour-line periodic timetable (PESP) as a MILP  +  Flask API.
Run:  pip install flask pulp  &&  python app.py   ->  http://localhost:5000
"""
import itertools, time
import pulp
from flask import Flask, jsonify, request, send_from_directory

T = 60                                   # period (minutes)
CV, VP, VA = 17, 63, 25                  # CSMT-VDLR, VDLR-PNVL, VDLR-ADH running times
RUN = {"L1": 2 * (CV + VP), "L2": 2 * (CV + VA), "L3": 2 * (VP + VA)}
DEFAULTS = dict(n1=5, n2=3, n3=1, headway=3, gap_lo=10, gap_hi=14,
                pn_min=5, pn_max=10, ad_min=5, ad_max=10,
                halt_min=2, halt_max=4, tlimit=60)

app = Flask(__name__, static_folder="static")


def solve_model(P):
    n1, n2, n3 = P["n1"], P["n2"], P["n3"]
    H = P["headway"]
    m = pulp.LpProblem("PESP_harbour", pulp.LpMinimize)
    k = itertools.count()

    def I(lb, ub, nm):  return pulp.LpVariable(f"{nm}{next(k)}", lb, ub, cat="Integer")
    def B():            return pulp.LpVariable(f"y{next(k)}", 0, 1, cat="Integer")

    # ---- decision variables: departure minute (mod 60) of every trip ----
    a = [I(0, T - 1, "a") for _ in range(n1)]   # L1 CSMT->PNVL, dep CSMT
    b = [I(0, T - 1, "b") for _ in range(n1)]   # L1 PNVL->CSMT, dep PNVL
    c = [I(0, T - 1, "c") for _ in range(n2)]   # L2 CSMT->ADH,  dep CSMT
    e = [I(0, T - 1, "e") for _ in range(n2)]   # L2 ADH->CSMT,  dep ADH
    f = [I(0, T - 1, "f") for _ in range(n3)]   # L3 PNVL->ADH,  dep PNVL
    g = [I(0, T - 1, "g") for _ in range(n3)]   # L3 ADH->PNVL,  dep ADH
    h1 = [I(P["halt_min"], P["halt_max"], "h") for _ in range(n3)]  # halt VDLR (f trips)
    h2 = [I(P["halt_min"], P["halt_max"], "h") for _ in range(n3)]  # halt VDLR (g trips)

    # ---- frequency / symmetry breaking: sorted departures with cyclic gaps ----
    def ordered(v, glo, ghi):
        for x, y in zip(v, v[1:]):
            m.__iadd__(y - x >= glo); m.__iadd__(y - x <= ghi)
        if len(v) > 1:
            m.__iadd__(T + v[0] - v[-1] >= glo); m.__iadd__(T + v[0] - v[-1] <= ghi)
    ordered(a, P["gap_lo"], P["gap_hi"]); ordered(b, P["gap_lo"], P["gap_hi"])
    for v in (c, e, f, g): ordered(v, H, T)

    # ---- track headway (PESP): 3 <= (ev - eu + 60p) <= 57 for every pair on a track-direction
    def headway(events):
        for u, v in itertools.combinations(events, 2):
            p = I(-6, 6, "p"); d = v - u + T * p
            m.__iadd__(d >= H); m.__iadd__(d <= T - H)
    headway([*a, *c])                                              # T1 CSMT->VDLR
    headway([*[x + VP for x in b], *[x + VA for x in e]])          # T1 VDLR->CSMT
    headway([*[x + CV for x in a], *[g[q] + VA + h2[q] for q in range(n3)]])   # T2 VDLR->PNVL
    headway([*b, *f])                                              # T2 PNVL->VDLR
    headway([*[x + CV for x in c], *[f[q] + VP + h1[q] for q in range(n3)]])   # T3 VDLR->ADH
    headway([*e, *g])                                              # T3 ADH->VDLR

    # ---- turnaround matching: every arrival is linked to one departure ----
    W = []
    def match(arr, dep, dmin, dmax):
        y = {}
        for (an, ae) in arr:
            for (dn, de) in dep:
                yy, w, q = B(), pulp.LpVariable(f"w{next(k)}", 0, dmax), I(0, 3, "q")
                y[an, dn] = (yy, w)
                m.__iadd__(w >= dmin * yy); m.__iadd__(w <= dmax * yy)
                M = 400
                m.__iadd__(de - ae + T * q - w <= M * (1 - yy))
                m.__iadd__(de - ae + T * q - w >= -M * (1 - yy))
                W.append(w)
        for an, _ in arr: m.__iadd__(pulp.lpSum(y[an, dn][0] for dn, _ in dep) == 1)
        for dn, _ in dep: m.__iadd__(pulp.lpSum(y[an, dn][0] for an, _ in arr) == 1)
        return y
    csmt_a = [(f"L1i{i}", b[i] + 80) for i in range(n1)] + [(f"L2i{i}", e[i] + 42) for i in range(n2)]
    csmt_d = [(f"L1o{i}", a[i]) for i in range(n1)] + [(f"L2o{i}", c[i]) for i in range(n2)]
    pn_a = [(f"L1o{i}", a[i] + 80) for i in range(n1)] + [(f"L3a{i}", g[i] + VA + VP + h2[i]) for i in range(n3)]
    pn_d = [(f"L1i{i}", b[i]) for i in range(n1)] + [(f"L3p{i}", f[i]) for i in range(n3)]
    ad_a = [(f"L2o{i}", c[i] + 42) for i in range(n2)] + [(f"L3p{i}", f[i] + VP + VA + h1[i]) for i in range(n3)]
    ad_d = [(f"L2i{i}", e[i]) for i in range(n2)] + [(f"L3a{i}", g[i]) for i in range(n3)]
    mc = match(csmt_a, csmt_d, 3, 3)
    mp = match(pn_a, pn_d, P["pn_min"], P["pn_max"])
    ma = match(ad_a, ad_d, P["ad_min"], P["ad_max"])

    # ---- fleet size: vehicle-minutes per period = 60 x rakes ----
    N = I(1, 200, "N")
    run = n1 * RUN["L1"] + n2 * RUN["L2"] + n3 * RUN["L3"]
    m.__iadd__(run + pulp.lpSum(W) + pulp.lpSum(h1) + pulp.lpSum(h2) == T * N)
    m.__iadd__(N)

    t0 = time.time()
    m.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=P["tlimit"]))
    st = pulp.LpStatus[m.status]
    if st not in ("Optimal",) and not (m.status == 1):
        return dict(status=st, error="No feasible timetable found for these parameters (or time limit hit).")
    V = lambda x: int(round(pulp.value(x)))

    # ---- extract trips ----
    trips = {}
    def add(id_, line, fr, to, dep, legs, dur):
        trips[id_] = dict(id=id_, line=line, frm=fr, to=to, dep=dep, legs=legs, dur=dur)
    for i in range(n1):
        add(f"L1o{i}", 1, "CSMT", "PNVL", V(a[i]), [("CSMT", "VDLR", 0, CV), ("VDLR", "PNVL", CV, CV + VP)], 80)
        add(f"L1i{i}", 1, "PNVL", "CSMT", V(b[i]), [("PNVL", "VDLR", 0, VP), ("VDLR", "CSMT", VP, 80)], 80)
    for i in range(n2):
        add(f"L2o{i}", 2, "CSMT", "ADH", V(c[i]), [("CSMT", "VDLR", 0, CV), ("VDLR", "ADH", CV, CV + VA)], 42)
        add(f"L2i{i}", 2, "ADH", "CSMT", V(e[i]), [("ADH", "VDLR", 0, VA), ("VDLR", "CSMT", VA, 42)], 42)
    for i in range(n3):
        hh = V(h1[i]); add(f"L3p{i}", 3, "PNVL", "ADH", V(f[i]),
            [("PNVL", "VDLR", 0, VP), ("VDLR", "VDLR", VP, VP + hh), ("VDLR", "ADH", VP + hh, VP + hh + VA)], VP + hh + VA)
        hh = V(h2[i]); add(f"L3a{i}", 3, "ADH", "PNVL", V(g[i]),
            [("ADH", "VDLR", 0, VA), ("VDLR", "VDLR", VA, VA + hh), ("VDLR", "PNVL", VA + hh, VA + hh + VP)], VA + hh + VP)

    succ, dwell = {}, {}
    for y in (mc, mp, ma):
        for (an, dn), (yy, w) in y.items():
            if V(yy) == 1: succ[an] = dn; dwell[an] = V(w)

    # ---- rake rosters (cycles of the arrival->departure links) ----
    seen, cycles, rakes = set(), [], []
    for t in sorted(trips, key=lambda x: (trips[x]["dep"], x)):
        if t in seen: continue
        seq, cur = [], t
        while cur not in seen:
            seen.add(cur); seq.append(cur); cur = succ[cur]
        s, steps, segs = 0, [], []
        for tid in seq:
            tr = trips[tid]; arr = s + tr["dur"]
            steps.append(dict(trip=tid, line=tr["line"], frm=tr["frm"], to=tr["to"], dep=s, arr=arr, turn=dwell[tid]))
            for (x, y_, o1, o2) in tr["legs"]:
                segs.append(dict(s=s + o1, e=s + o2, a=x, b=y_, line=tr["line"]))
            segs.append(dict(s=arr, e=arr + dwell[tid], a=tr["to"], b=tr["to"], line=tr["line"]))
            s = arr + dwell[tid]
        L = s; assert L % T == 0, "cycle not multiple of period"
        cid = len(cycles); start0 = trips[seq[0]]["dep"]
        names = []
        for j in range(L // T):
            nm = f"R{len(rakes) + 1}"; names.append(nm)
            rakes.append(dict(name=nm, cycle=cid, shift=start0 + T * j))
        cycles.append(dict(id=cid, L=L, start0=start0, steps=steps, segs=segs, rakes=names))
    return dict(status="Optimal" if st == "Optimal" else st, rakes=V(N), solve_time=round(time.time() - t0, 2),
                n_vars=len(m.variables()), n_cons=len(m.constraints),
                trips=sorted(trips.values(), key=lambda x: x["dep"]), cycles=cycles, fleet=rakes,
                dwell_total=sum(dwell.values()))


@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/api/solve", methods=["OPTIONS"])
def preflight(): return ("", 204)


from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

@app.get("/")
def index():
    return send_from_directory(str(BASE_DIR), "index.html")


@app.post("/api/solve")
def api_solve():
    P = {**DEFAULTS, **{kk: int(v) for kk, v in (request.get_json(silent=True) or {}).items() if kk in DEFAULTS}}
    P["tlimit"] = max(5, min(P["tlimit"], 300))
    try:
        return jsonify(solve_model(P))
    except Exception as ex:
        return jsonify(status="Error", error=str(ex)), 500


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
