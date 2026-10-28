"""Benchmark: BFS vs Manhattan-A* vs Threat-aware A*, plus forecaster accuracy and
hidden-enemy failure analysis.   python benchmark.py [--quick]"""
import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from zeus import *

QUICK = "--quick" in sys.argv
SIZES = {50: 30, 100: 20, 250: 6}
if QUICK:
    SIZES = {50: 6, 100: 3, 250: 1}
TOPOS = ["open", "urban", "bridge"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2026)


def trial(n, topo):
    cover, blocked = make_map(n, topo, rng)
    threat, _ = make_threat(blocked, rng)
    c0, r0 = (n // 2, n // 2), 0.48 * n
    r1 = 0.6 * r0
    # ---- Stage 1: forecast vs hidden truth (true sigma differs from prior sigma)
    fc = ZoneForecaster(cover, blocked)
    s_true = 0.12 * n
    hist = []
    prev = np.array(c0, float)
    for _ in range(3):                      # 3 past observed shifts
        d = rng.normal(0, s_true, 2)
        hist.append(float((d ** 2).sum()))
    d2 = ((fc.centers - np.array(c0)) ** 2).sum(-1)
    p_true = fc.prior * np.exp(-d2 / (2 * s_true ** 2)) * (np.sqrt(d2) <= r0 - r1)
    p_true /= p_true.sum()
    ti = rng.choice(p_true.size, p=p_true.ravel())
    truth = fc.centers.reshape(-1, 2)[ti]
    post = fc.posterior(c0, r0, r1, hist)
    goal = fc.best_goal(post)
    fc_err = float(np.linalg.norm(np.array(goal) - truth))
    fc_naive = float(np.linalg.norm(np.array(c0) - truth))          # "stay at centre" guess
    mass_near = float(post[np.linalg.norm(fc.centers - truth, axis=-1) <= 0.25 * n].sum())
    # ---- Stage 3: route to forecast goal
    mask = circle_mask(n, c0, r0) & ~blocked
    goal = nearest_free(mask, goal)
    for _ in range(50):
        s = nearest_free(mask, tuple(rng.integers(0, n, 2)))
        if np.hypot(s[0] - goal[0], s[1] - goal[1]) > 0.35 * n:
            break
    runs = {"BFS": bfs(mask, s, goal), "A*-Manhattan": baseline_astar(mask, s, goal),
            "Threat-A*": threat_astar(mask, threat, cover, s, goal)}
    if not all(r["found"] for r in runs.values()):
        return None
    # hidden enemy: appears at midpoint of the threat-aware route (unobserved at planning time)
    mid = runs["Threat-A*"]["path"][len(runs["Threat-A*"]["path"]) // 2]
    t_hid, _ = make_threat(blocked, np.random.default_rng(1), k=1, extra=[mid])
    t_true = np.maximum(threat, t_hid)
    rows = []
    for name, r in runs.items():
        rows.append(dict(size=n, topo=topo, algo=name, expanded=r["expanded"],
                         generated=r["generated"], peak_open=r["peak_open"], ms=r["ms"],
                         steps=len(r["path"]) - 1, survival=survival(r["path"], threat, cover),
                         survival_hidden=survival(r["path"], t_true, cover),
                         threat_exposure=float(np.mean([threat[p] for p in r["path"]])),
                         fc_err=fc_err, fc_naive=fc_naive, fc_mass=mass_near))
    return rows


rows = []
for n, k in SIZES.items():
    for topo in TOPOS:
        done = 0
        while done < k:
            r = trial(n, topo)
            if r:
                rows += r; done += 1
        print(f"done {n}x{n} {topo}", flush=True)
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/benchmark_raw.csv", index=False)

agg = df.groupby(["size", "algo"]).agg(
    survival=("survival", "mean"), survival_hidden=("survival_hidden", "mean"),
    expanded=("expanded", "mean"), generated=("generated", "mean"),
    peak_open=("peak_open", "mean"), ms=("ms", "mean"), steps=("steps", "mean"),
    exposure=("threat_exposure", "mean")).round(3)
agg.to_csv(f"{OUT}/benchmark_summary.csv")
by_topo = df.groupby(["topo", "size", "algo"])[["survival", "expanded", "ms"]].mean().round(3)
by_topo.to_csv(f"{OUT}/benchmark_by_topology.csv")
print("\n=== SUMMARY (means) ===\n", agg.to_string())
algos = ["BFS", "A*-Manhattan", "Threat-A*"]

# forecaster accuracy (one row per trial -> take Threat-A* rows)
f = df[df.algo == "Threat-A*"]
fc_tab = f.groupby("size")[["fc_err", "fc_naive", "fc_mass"]].mean().round(3)
fc_tab.to_csv(f"{OUT}/forecaster_accuracy.csv")
print("\n=== FORECASTER (error in cells, lower better; mass = posterior within 0.25n of truth) ===\n",
      fc_tab.to_string())

# ---------------- plots
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a in algos:
    d = agg.xs(a, level="algo")
    ax[0].plot(d.index ** 2, d.ms, "o-", label=a)
    ax[1].plot(d.index ** 2, d.expanded, "o-", label=a)
for x, t in zip(ax, ["Execution time (ms)", "Nodes expanded"]):
    x.set_xscale("log"); x.set_yscale("log"); x.set_xlabel("State space size (cells)")
    x.set_ylabel(t); x.legend(); x.grid(alpha=.3)
ax[0].set_title("State space vs runtime"); ax[1].set_title("Instance size vs expanded nodes")
plt.tight_layout(); plt.savefig(f"{OUT}/plot_runtime_nodes.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
w = 0.25
sizes = sorted(SIZES)
for i, a in enumerate(algos):
    d = agg.xs(a, level="algo")
    ax[0].bar(np.arange(3) + i * w, d.survival.values, w, label=a)
    ax[1].bar(np.arange(3) + i * w, d.survival_hidden.values, w, label=a)
for x, t in zip(ax, ["Survival probability (known threats)", "Survival (with unobserved hidden enemy)"]):
    x.set_xticks(np.arange(3) + w); x.set_xticklabels([f"{s}x{s}" for s in sizes])
    x.set_title(t); x.set_ylim(0, 1); x.legend(); x.grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig(f"{OUT}/plot_survival.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(5.5, 4))
for a in algos:
    d = agg.xs(a, level="algo")
    ax.plot(d.index ** 2, d.peak_open, "o-", label=a)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("State space size")
ax.set_ylabel("Peak open-list size"); ax.set_title("Memory (frontier) growth"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/plot_memory.png", dpi=150); plt.close()

# ---------------- demo run + log + route picture
r = np.random.default_rng(5)
n = 60
cover, blocked = make_map(n, "bridge", r)
threat, enemies = make_threat(blocked, r)
res = igl_decision(cover, blocked, threat, (30, 5), (30, 30), 28, 17, time_left=40, history_d2=[40, 90])
res["facts"].pop("ready", None)
log = {"decision": res["decision"], "goal": res["goal"], "rules_fired": res["rules_fired"],
       "facts": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["facts"].items()}}
json.dump(log, open(f"{OUT}/demo_log.json", "w"), indent=2, default=str)
print("\n=== DEMO DECISION LOG ===\n", json.dumps(log, indent=2, default=str))
fig = plt.figure(figsize=(16, 5), facecolor="#0b0f19")
ax0 = fig.add_subplot(1, 3, 1, facecolor="#080c14")
im0 = ax0.imshow(res["posterior"], cmap="viridis")
plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
ax0.set_title("1. Bayesian Zone Posterior", color="#ffffff", fontsize=11, fontweight="bold")
ax0.tick_params(colors="#8da2c0")

ax1 = fig.add_subplot(1, 3, 2, facecolor="#080c14")
im1 = ax1.imshow(threat, cmap="magma", alpha=0.85)
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
if blocked.any():
    ax1.imshow(np.ma.masked_where(~blocked, np.ones_like(blocked)), cmap="Blues", alpha=0.6)
ax1.add_patch(plt.Circle((30, 30), 28, fill=False, edgecolor="#00e5ff", linewidth=2, linestyle="-"))
ax1.add_patch(plt.Circle(res["goal"][::-1], 17, fill=False, edgecolor="#ffd700", linewidth=1.5, linestyle="--"))
for key, col in [("baseline_route", "#ff2a55"), ("route", "#00ff88")]:
    rt = res[key]
    if rt and rt["found"]:
        p = np.array(rt["path"])
        ax1.plot(p[:, 1], p[:, 0], col, lw=2.5,
                 label=f"{'Baseline A*' if key == 'baseline_route' else 'Threat-A*'} ({len(p)-1} stp)")
ax1.scatter(*res["goal"][::-1], c="#ffd700", marker="*", s=180, edgecolors="#000000", label="Forecast Goal")
ax1.scatter(30, 5, c="#00e5ff", marker="o", s=100, edgecolors="#ffffff", label="Start Deploy")
ax1.legend(loc="upper right", facecolor="#0b0f19", edgecolor="#1b2538", labelcolor="#ffffff", fontsize=8)
ax1.set_title(f"2. Tactical 2D Radar HUD ({res['decision']})", color="#ffffff", fontsize=11, fontweight="bold")
ax1.tick_params(colors="#8da2c0")

ax2 = fig.add_subplot(1, 3, 3, projection="3d", facecolor="#080c14")
X, Y = np.meshgrid(np.arange(n), np.arange(n))
ax2.plot_surface(X, Y, threat, cmap="magma", alpha=0.8, edgecolor="none")
ax2.contour(X, Y, threat, zdir="z", offset=0, cmap="coolwarm", alpha=0.4)
for key, col in [("baseline_route", "#ff2a55"), ("route", "#00ff88")]:
    rt = res[key]
    if rt and rt["found"]:
        p = np.array(rt["path"])
        ax2.plot(p[:, 1], p[:, 0], [threat[y, x] + 0.05 for y, x in p], col, lw=2.5,
                 label="Baseline A*" if key == "baseline_route" else "Threat-A*")
ax2.scatter([5], [30], [threat[30, 5] + 0.08], color="#00e5ff", s=70, marker="o")
ax2.scatter([res["goal"][1]], [res["goal"][0]], [threat[res["goal"][0], res["goal"][1]] + 0.08], color="#ffd700", s=120, marker="*")
ax2.view_init(elev=38, azim=-55)
ax2.set_title("3. 3D Threat Landscape Perspective", color="#ffffff", fontsize=11, fontweight="bold")
ax2.tick_params(colors="#8da2c0", labelsize=7)
for axis in [ax2.xaxis, ax2.yaxis, ax2.zaxis]:
    axis.set_pane_color((0.04, 0.06, 0.1, 1.0))
    axis._axinfo["grid"]["color"] = (0.11, 0.15, 0.23, 1.0)
ax2.legend(facecolor="#0b0f19", edgecolor="#1b2538", labelcolor="#ffffff", fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT}/plot_demo_route.png", dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
plt.savefig("plot_demo_route.png", dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print("saved to", OUT)
