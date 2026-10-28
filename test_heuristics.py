"""Empirical verification of admissibility / consistency (+ counterexample for naive h)."""
import numpy as np
from zeus import *

rng = np.random.default_rng(7)
viol_adm = viol_con = checked = 0
naive_over = 0
for topo in ("open", "urban", "bridge"):
    for _ in range(5):
        n = 30
        cover, blocked = make_map(n, topo, rng)
        threat, _ = make_threat(blocked, rng)
        mask = circle_mask(n, (n // 2, n // 2), 14) & ~blocked
        g = nearest_free(mask, (n // 2, n // 2))
        cost = threat_cost(mask, threat, cover)
        hstar = true_cost_to_goal(cost, g)
        for y in range(n):
            for x in range(n):
                if hstar[y, x] == INF:
                    continue
                h = CMIN * max(abs(y - g[0]), abs(x - g[1]))
                checked += 1
                viol_adm += h > hstar[y, x] + 1e-9
                # naive: Manhattan with scale 1 on the 8-connected threat-aware problem
                naive_over += (abs(y - g[0]) + abs(x - g[1])) > hstar[y, x] + 1e-9
                for dy, dx in M8:
                    v = (y + dy, x + dx)
                    if 0 <= v[0] < n and 0 <= v[1] < n and cost[v] != INF:
                        c = cost[v] * (SQ2 if dy and dx else 1)
                        hv = CMIN * max(abs(v[0] - g[0]), abs(v[1] - g[1]))
                        viol_con += h > c + hv + 1e-9
print(f"nodes checked           : {checked}")
print(f"admissibility violations: {viol_adm}   (h <= h*)")
print(f"consistency violations  : {viol_con}   (h(n) <= c + h(n'))")
print(f"COUNTEREXAMPLES: naive Manhattan(scale 1) over-estimates at {naive_over} nodes "
      f"-> inadmissible for 8-connected, threat-weighted costs")
assert viol_adm == 0 and viol_con == 0

# optimality: threat-aware A* must equal Dijkstra (h=0) cost
for topo in ("open", "urban", "bridge"):
    n = 40
    cover, blocked = make_map(n, topo, rng)
    threat, _ = make_threat(blocked, rng)
    mask = circle_mask(n, (20, 20), 19) & ~blocked
    s, g = nearest_free(mask, (3, 3)), nearest_free(mask, (36, 36))
    a = threat_astar(mask, threat, cover, s, g)
    d = astar(threat_cost(mask, threat, cover), s, g, hs=0.0)
    assert abs(a["cost"] - d["cost"]) < 1e-9, (topo, a["cost"], d["cost"])
    print(f"{topo:7s} A* cost {a['cost']:.3f} == Dijkstra {d['cost']:.3f} | "
          f"expanded {a['expanded']} vs {d['expanded']}")
print("ALL TESTS PASSED")
