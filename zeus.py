"""
Z.E.U.S. - Zone Evaluation and Utility System  (core engine)
Stage 1: Bayesian zone forecaster   (Unit 3)
Stage 2: Forward-chaining expert system for macro-strategy   (Unit 4)
Stage 3: Threat-aware A* + baselines (BFS, Manhattan A*)   (Units 1-2)
"""
import heapq
import math
import time
from collections import deque

import numpy as np

# ---------------------------------------------------------------- constants
CMIN, WT, WC = 0.4, 3.0, 0.6      # min step cost, threat weight, cover weight
SQ2 = math.sqrt(2)
M8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
M4 = M8[:4]
INF = float("inf")


def make_erangel_map(n, rng):
    """Return (cover, blocked) for the iconic BGMI Erangel battle royale map."""
    cover = np.zeros((n, n), dtype=float)
    blocked = np.zeros((n, n), dtype=bool)
    yy, xx = np.indices((n, n))
    dist_from_center = np.hypot(yy - n / 2, xx - n / 2)
    # Ocean boundary
    blocked[dist_from_center > 0.48 * n] = True

    # Sosnovka River Channel separating Mainland (North) and Sosnovka Island (South)
    river_y = int(0.70 * n)
    river_w = max(2, int(0.06 * n))
    river_mask = (yy >= river_y) & (yy < river_y + river_w) & (~blocked)
    blocked[river_mask] = True

    # Iconic Sosnovka Bridges: West Bridge & East Bridge
    b1_x, b2_x = int(0.35 * n), int(0.65 * n)
    bw = max(1, int(0.03 * n))
    blocked[river_y:river_y + river_w, max(0, b1_x - bw):min(n, b1_x + bw + 1)] = False
    blocked[river_y:river_y + river_w, max(0, b2_x - bw):min(n, b2_x + bw + 1)] = False

    # Sparse grassland cover (trees, dips, haystacks)
    land_mask = ~blocked
    cover[land_mask] = (rng.random((n, n))[land_mask] < 0.12) * 0.3

    # Key BGMI Erangel POIs with compound buildings (cover = 0.75)
    pois = [
        ("Pochinki", int(0.48 * n), int(0.46 * n), max(3, int(0.07 * n))),
        ("Military Base", int(0.85 * n), int(0.50 * n), max(3, int(0.08 * n))),
        ("School/Rozhok", int(0.36 * n), int(0.52 * n), max(3, int(0.06 * n))),
        ("Georgopol", int(0.24 * n), int(0.22 * n), max(3, int(0.07 * n))),
        ("Yasnaya", int(0.32 * n), int(0.74 * n), max(3, int(0.07 * n))),
        ("Gatka", int(0.50 * n), int(0.26 * n), max(2, int(0.05 * n))),
        ("Mylta", int(0.62 * n), int(0.72 * n), max(2, int(0.05 * n)))
    ]
    for name, py, px, rad in pois:
        y0, y1 = max(0, py - rad), min(n, py + rad + 1)
        x0, x1 = max(0, px - rad), min(n, px + rad + 1)
        sub_mask = ~blocked[y0:y1, x0:x1]
        cover[y0:y1, x0:x1][sub_mask] = 0.75

    return cover, blocked


def get_poi_labels(n, topo):
    """Return landmark names and coordinates (y, x) for tactical HUD annotations."""
    if topo in ("erangel", "erangel_full"):
        return [
            ("Pochinki", (int(0.48 * n), int(0.46 * n))),
            ("Sosnovka Base", (int(0.85 * n), int(0.50 * n))),
            ("West Bridge", (int(0.70 * n), int(0.35 * n))),
            ("East Bridge", (int(0.70 * n), int(0.65 * n))),
            ("School / Rozhok", (int(0.36 * n), int(0.52 * n))),
            ("Georgopol", (int(0.24 * n), int(0.22 * n))),
            ("Yasnaya Polyana", (int(0.32 * n), int(0.74 * n))),
            ("Gatka", (int(0.50 * n), int(0.26 * n))),
            ("Mylta", (int(0.62 * n), int(0.72 * n)))
        ]
    elif topo in ("bridge", "erangel_bridge"):
        return [
            ("Mainland Shore", (int(0.30 * n), int(0.50 * n))),
            ("Sosnovka Shore", (int(0.70 * n), int(0.50 * n))),
            ("Bridge 1", (int(0.50 * n), int(0.20 * n))),
            ("Bridge 2", (int(0.50 * n), int(0.50 * n))),
            ("Bridge 3", (int(0.50 * n), int(0.80 * n)))
        ]
    elif topo in ("urban", "erangel_pochinki"):
        return [
            ("Central Compound", (int(0.50 * n), int(0.50 * n))),
            ("Church Hill", (int(0.25 * n), int(0.30 * n))),
            ("Warehouse Row", (int(0.75 * n), int(0.65 * n)))
        ]
    elif topo in ("open", "erangel_gatka"):
        return [
            ("Gatka Trenches", (int(0.45 * n), int(0.40 * n))),
            ("Haystack Ridge", (int(0.55 * n), int(0.60 * n)))
        ]
    return []


# ------------------------------------------------------------ environment
def make_map(n, topo, rng):
    """Return (cover[0..1], blocked[bool]) for topo in erangel/open/urban/bridge."""
    cover = np.zeros((n, n))
    blocked = np.zeros((n, n), bool)
    if topo in ("erangel", "erangel_full"):
        return make_erangel_map(n, rng)
    elif topo in ("open", "erangel_gatka"):
        cover = (rng.random((n, n)) < 0.08) * 0.3          # sparse trees / wheat fields
    elif topo in ("urban", "erangel_pochinki"):
        for _ in range(max(4, n * n // 150)):
            h, w = rng.integers(3, 9, 2)
            y, x = rng.integers(0, n - h), rng.integers(0, n - w)
            cover[y:y + h, x:x + w] = 0.75                  # buildings = cover
        blocked = rng.random((n, n)) < 0.02                 # rubble / walls
    elif topo in ("bridge", "erangel_bridge"):
        mid, half = n // 2, max(2, n // 25)
        blocked[:, mid - half:mid + half] = True            # river
        for b in np.linspace(n * 0.2, n * 0.8, 3).astype(int):
            blocked[b - 1:b + 2, mid - half:mid + half] = False   # bridges
    else:
        raise ValueError(topo)
    return cover, blocked


def make_threat(blocked, rng, k=None, extra=()):
    """Gaussian threat field from k enemy squads, values in [0,1]."""
    n = blocked.shape[0]
    enemies = list(extra)
    if k is None:
        k = max(3, n // 8)
    if k > 0:
        free = np.argwhere(~blocked)
        if len(free) > 0:
            chosen = rng.choice(len(free), min(k, len(free)), replace=False)
            enemies += [tuple(free[i]) for i in chosen]
    yy, xx = np.indices((n, n))
    t = np.zeros((n, n))
    if enemies:
        sigma = n * 0.04 + 2
        for ey, ex in enemies:
            t = np.maximum(t, np.exp(-((yy - ey) ** 2 + (xx - ex) ** 2) / (2 * sigma ** 2)))
    return t, enemies


def circle_mask(n, c, r):
    yy, xx = np.indices((n, n))
    return (yy - c[0]) ** 2 + (xx - c[1]) ** 2 <= r * r


def step_risk(threat, cover):
    """Per-step death probability used to *evaluate* (not plan) a route."""
    return 0.15 * threat * (1 - cover) + 0.002


def survival(path, threat, cover):
    if not path:
        return 0.0
    p = 1.0
    for y, x in path[1:]:
        p *= 1 - step_risk(threat[y, x], cover[y, x])
    return p


def nearest_free(mask, p):
    pts = np.argwhere(mask)
    d = ((pts - np.array(p)) ** 2).sum(1)
    return tuple(int(v) for v in pts[d.argmin()])


# ------------------------------------------- Stage 1: Bayesian forecaster
class ZoneForecaster:
    """
    Tiny Bayesian network   Landmass L -> Z_{t+1} <- Z_t
      P(Z_{t+1}=c | L, Z_t) ∝ P(c | L) * N(c; Z_t, sigma^2) * 1[|c-Z_t| <= R_t - R_{t+1}]
    sigma is learnt from historic shifts (conjugate inverse-gamma style update).
    """

    def __init__(self, cover, blocked, cell=5):
        n = cover.shape[0]
        m = n // cell
        self.n, self.cell, self.m = n, cell, m
        land = cover[:m * cell, :m * cell].reshape(m, cell, m, cell).mean((1, 3))
        free = 1 - blocked[:m * cell, :m * cell].reshape(m, cell, m, cell).mean((1, 3))
        w = (0.3 + land) * free                       # POIs / buildings attract zones
        self.prior = w / w.sum()
        ys, xs = np.mgrid[0:m, 0:m]
        self.centers = np.stack([ys * cell + cell // 2, xs * cell + cell // 2], -1)
        self.a0, self.s0 = 2.0, n * 0.08

    def sigma(self, shifts_d2=()):
        m = len(shifts_d2)
        return math.sqrt((self.a0 * self.s0 ** 2 + sum(shifts_d2) / 2) / (self.a0 + m))

    def posterior(self, prev_c, r_prev, r_next, shifts_d2=()):
        s = self.sigma(shifts_d2)
        d2 = ((self.centers - np.array(prev_c)) ** 2).sum(-1)
        like = np.exp(-d2 / (2 * s * s)) * (np.sqrt(d2) <= r_prev - r_next)
        post = self.prior * like
        if post.sum() == 0:                            # fall back to containment only
            post = self.prior * (np.sqrt(d2) <= r_prev)
        return post / post.sum()

    def best_goal(self, post):
        i = np.unravel_index(post.argmax(), post.shape)
        return tuple(int(v) for v in self.centers[i])


# ----------------------------------- Stage 2: forward-chaining expert system
class Rule:
    def __init__(self, name, cond, act):
        self.name, self.cond, self.act = name, cond, act


RULES = [
    Rule("R1 urgent", lambda f: f["time_left"] < 1.2 * f["eta"], lambda f: {"urgent": True}),
    Rule("R2 defensible", lambda f: (f.get("in_zone") or f.get("in_next_zone")) and f["cover_here"] >= 0.35,
         lambda f: {"defensible": True}),
    Rule("R3 chokepoint", lambda f: f["choke_near"] and (f.get("in_zone") or f.get("in_next_zone")),
         lambda f: {"gate_ok": True}),
    Rule("R4 risky route", lambda f: f["route_risk"] > 0.25, lambda f: {"risky": True}),
    Rule("R5 ready", lambda f: True, lambda f: {"ready": True}),
    Rule("R6 utilities", lambda f: f.get("ready"), lambda f: {
        "U_HOLD": 2.5 * f.get("defensible", 0) + 1.5 * f["cover_here"] + 1.2 * f.get("in_zone", 0)
                  - 3.0 * f.get("urgent", 0),
        "U_GATEKEEP": 3.0 * f.get("gate_ok", 0) + 1.2 * f.get("in_zone", 0) + 0.8 * f.get("defensible", 0)
                      - 1.5 * f.get("urgent", 0),
        "U_ROTATE": 2.2 * (not f.get("in_zone", False)) + 2.0 * f.get("urgent", 0)
                    - 1.5 * f.get("risky", 0) + 0.2}),
    Rule("R7 decide", lambda f: "U_HOLD" in f, lambda f: {
        "decision": max(("HOLD", "GATEKEEP", "ROTATE"), key=lambda a: f["U_" + a])}),
]


def forward_chain(facts, rules=RULES):
    facts, fired, changed = dict(facts), [], True
    while changed:
        changed = False
        for r in rules:
            if r.name not in fired and r.cond(facts):
                facts.update(r.act(facts))
                fired.append(r.name)
                changed = True
    return facts, fired


def choke_near(blocked, pos, radius=6):
    y0, y1 = max(1, pos[0] - radius), min(blocked.shape[0] - 1, pos[0] + radius + 1)
    x0, x1 = max(1, pos[1] - radius), min(blocked.shape[1] - 1, pos[1] + radius + 1)
    for y in range(y0, y1):
        for x in range(x0, x1):
            if not blocked[y, x] and blocked[y - 1:y + 2, x - 1:x + 2].sum() >= 4:
                return True
    return False


# --------------------------------------------- Stage 3: search algorithms
def _result(path, cost, expanded, generated, peak, t0):
    return dict(path=path, cost=cost, expanded=expanded, generated=generated,
                peak_open=peak, ms=(time.perf_counter() - t0) * 1000, found=bool(path))


def _rebuild(parent, g):
    p = [g]
    while p[-1] in parent:
        p.append(parent[p[-1]])
    return p[::-1]


def bfs(passable, s, g, return_explored=False):
    """Baseline 1: uninformed 4-connected BFS (unit cost)."""
    t0, n = time.perf_counter(), passable.shape[0]
    pas = passable.tolist()
    q, parent, seen, exp, gen, peak = deque([s]), {}, {s}, 0, 1, 1
    while q:
        u = q.popleft()
        exp += 1
        if u == g:
            path = _rebuild(parent, g)
            res = _result(path, len(path) - 1, exp, gen, peak, t0)
            if return_explored:
                res["explored"] = list(seen)
            return res
        for dy, dx in M4:
            v = (u[0] + dy, u[1] + dx)
            if 0 <= v[0] < n and 0 <= v[1] < n and pas[v[0]][v[1]] and v not in seen:
                seen.add(v); parent[v] = u; q.append(v); gen += 1
        peak = max(peak, len(q))
    res = _result([], INF, exp, gen, peak, t0)
    if return_explored:
        res["explored"] = list(seen)
    return res


def astar(cost, s, g, moves=M8, hs=CMIN, mode="cheb", diag_mult=True, return_explored=False):
    """
    Generic A*. cost[y][x] = cost of ENTERING a cell (inf = forbidden).
    mode 'cheb' : h = hs * Chebyshev   (threat-aware agent, hs = CMIN)
    mode 'manh' : h = hs * Manhattan   (baseline; 4-connected, unit cost)
    """
    t0, n = time.perf_counter(), cost.shape[0]
    C = cost.tolist()

    def h(p):
        dy, dx = abs(p[0] - g[0]), abs(p[1] - g[1])
        return hs * (max(dy, dx) if mode == "cheb" else dy + dx)

    gs, parent, closed = {s: 0.0}, {}, set()
    heap, exp, gen, peak = [(h(s), 0.0, s)], 0, 1, 1
    while heap:
        _, gu, u = heapq.heappop(heap)
        if u in closed:
            continue
        closed.add(u)
        exp += 1
        if u == g:
            res = _result(_rebuild(parent, g), gu, exp, gen, peak, t0)
            if return_explored:
                res["explored"] = list(closed)
            return res
        for dy, dx in moves:
            v = (u[0] + dy, u[1] + dx)
            if not (0 <= v[0] < n and 0 <= v[1] < n):
                continue
            c = C[v[0]][v[1]]
            if c == INF or v in closed:
                continue
            ng = gu + c * (SQ2 if (diag_mult and dy and dx) else 1)
            if ng < gs.get(v, INF):
                gs[v] = ng; parent[v] = u
                heapq.heappush(heap, (ng + h(v), ng, v)); gen += 1
        peak = max(peak, len(heap))
    res = _result([], INF, exp, gen, peak, t0)
    if return_explored:
        res["explored"] = list(closed)
    return res


def baseline_astar(mask, s, g, return_explored=False):
    """Baseline 2: 4-connected, unit-cost, Manhattan A* (ignores threat/cover)."""
    cost = np.where(mask, 1.0, INF)
    return astar(cost, s, g, moves=M4, hs=1.0, mode="manh", diag_mult=False, return_explored=return_explored)


def threat_cost(mask, threat, cover):
    # Pro esports threat weighting: steep penalty on direct line-of-sight to keep safe distance from campers
    penalty = WT * (threat + 3.0 * (threat ** 2))
    c = np.maximum(CMIN, 1 + penalty - WC * cover)      # step = 1 + w_threat - w_cover
    return np.where(mask, c, INF)                           # outside blue zone => inf


def threat_astar(mask, threat, cover, s, g, return_explored=False):
    """Advanced agent: 8-connected threat-weighted A*, h = CMIN * Chebyshev."""
    return astar(threat_cost(mask, threat, cover), s, g, moves=M8, hs=CMIN, mode="cheb", return_explored=return_explored)


def true_cost_to_goal(cost, g, moves=M8):
    """Backward Dijkstra -> exact h*(n) (used to verify admissibility/consistency)."""
    n = cost.shape[0]
    d = np.full((n, n), INF)
    d[g] = 0.0
    pq = [(0.0, g)]
    while pq:
        du, u = heapq.heappop(pq)
        if du > d[u]:
            continue
        for dy, dx in moves:
            v = (u[0] - dy, u[1] - dx)        # predecessor v --(dy,dx)--> u
            if 0 <= v[0] < n and 0 <= v[1] < n and cost[v] != INF:
                w = cost[u] * (SQ2 if dy and dx else 1) if cost[u] != INF else INF
                if du + w < d[v]:
                    d[v] = du + w
                    heapq.heappush(pq, (d[v], v))
    return d


def compute_gatekeep_context(pos, zone_c, enemies=(), n=80, enemies_inside=()):
    """
    Computes esports field-of-fire angle cone and enemy approach sides for gatekeeping.
    Only targets enemy squads trapped outside the safe zone who must rotate in.
    """
    def to_compass(dy, dx):
        deg = math.degrees(math.atan2(dx, -dy))
        return (deg + 360) % 360

    def compass_name(deg):
        sectors = ['North', 'Northeast', 'East', 'Southeast', 'South', 'Southwest', 'West', 'Northwest']
        idx = int((deg + 22.5) // 45) % 8
        return sectors[idx]

    enemy_sides = []
    enemy_bearings = []
    for ey, ex in enemies:
        dy, dx = ey - pos[0], ex - pos[1]
        dist_m = int(math.hypot(dy, dx) * (8000 / n))
        b = to_compass(dy, dx)
        enemy_bearings.append(b)
        enemy_sides.append({
            'bearing': round(b, 1),
            'sector': compass_name(b),
            'distance_m': dist_m,
            'grid_pos': (ey, ex),
            'label': f"{compass_name(b)} ({int(b)}° • {dist_m}m)"
        })

    enemy_sides.sort(key=lambda s: s['distance_m'])

    # Outward direction facing away from zone center (towards blue zone)
    dy_out, dx_out = pos[0] - zone_c[0], pos[1] - zone_c[1]
    outward_bearing = to_compass(dy_out, dx_out)

    if enemy_bearings:
        close_b = [e['bearing'] for e in enemy_sides if e['distance_m'] <= 3500] or enemy_bearings
        sin_sum = sum(math.sin(math.radians(b)) for b in close_b)
        cos_sum = sum(math.cos(math.radians(b)) for b in close_b)
        mean_b = (math.degrees(math.atan2(sin_sum, cos_sum)) + 360) % 360
        span = 85.0
    else:
        mean_b = outward_bearing
        span = 90.0

    b_start = (mean_b - span / 2) % 360
    b_end = (mean_b + span / 2) % 360

    distinct_sectors = []
    seen = set()
    for es in enemy_sides:
        if es['sector'] not in seen:
            seen.add(es['sector'])
            distinct_sectors.append(es)

    inside_sides = []
    for ey, ex in enemies_inside:
        dy, dx = ey - pos[0], ex - pos[1]
        dist_m = int(math.hypot(dy, dx) * (8000 / n))
        b = to_compass(dy, dx)
        inside_sides.append({
            'bearing': round(b, 1),
            'sector': compass_name(b),
            'distance_m': dist_m,
            'grid_pos': (ey, ex),
            'label': f"{compass_name(b)} ({int(b)}° • {dist_m}m)"
        })

    return {
        'mean_bearing': round(mean_b, 1),
        'span_deg': span,
        'b_start': round(b_start, 1),
        'b_end': round(b_end, 1),
        'primary_sector': compass_name(mean_b),
        'enemy_sides': enemy_sides,
        'distinct_sectors': distinct_sectors,
        'outside_count': len(enemies),
        'inside_count': len(enemies_inside),
        'inside_sides': inside_sides
    }


# -------------------------------------------------------- full IGL pipeline
def igl_decision(cover, blocked, threat, pos, zone_c, r_now, r_next, time_left,
                 history_d2=(), strategy_mode="AUTO", enemies=(), next_c=None):
    """Runs Stage 1 -> 2 -> 3 once and returns a comprehensive tactical log dict."""
    n = cover.shape[0]
    fc = ZoneForecaster(cover, blocked)
    post = fc.posterior(zone_c, r_now, r_next, history_d2)
    if next_c is not None:
        goal = next_c
    else:
        goal = fc.best_goal(post)

    # Navigation mask for pathfinding across dry land (respects ocean/rivers, crosses bridges)
    nav_mask = ~blocked
    goal = nearest_free(nav_mask, goal)

    # Safe zone mask for candidate targets inside the current playzone (Blue circle)
    zone_mask = circle_mask(n, zone_c, r_now) & ~blocked
    if not zone_mask.any():
        zone_mask = ~blocked

    # Next zone mask for candidate targets inside the next playzone (Yellow circle)
    next_circ_mask = circle_mask(n, goal, r_next) & ~blocked
    if not next_circ_mask.any():
        next_circ_mask = zone_mask

    in_current_zone = bool(math.dist(pos, zone_c) <= r_now)
    in_next_zone = bool(math.dist(pos, goal) <= r_next)

    # Separate enemy squads: those trapped outside the yellow zone vs already secured inside
    enemies_outside_next = [e for e in enemies if math.hypot(e[0] - goal[0], e[1] - goal[1]) > r_next]
    enemies_inside_next = [e for e in enemies if math.hypot(e[0] - goal[0], e[1] - goal[1]) <= r_next]

    yy, xx = np.indices((n, n))

    # Candidate Target 1: Best Fortified Compound INSIDE the Yellow Next Zone
    valid_next_compounds = next_circ_mask & (cover >= 0.25)
    if valid_next_compounds.any():
        comp_scores_next = np.where(
            valid_next_compounds,
            cover * 8.0 - threat * 12.0 - np.hypot(yy - pos[0], xx - pos[1]) * 0.04,
            -999.0
        )
        compound_in_yellow = np.unravel_index(np.argmax(comp_scores_next), comp_scores_next.shape)
        compound_in_yellow = (int(compound_in_yellow[0]), int(compound_in_yellow[1]))
    else:
        compound_in_yellow = goal

    # Candidate Target 2: Center Ridge / Safe Terrain inside the Yellow Next Zone
    if threat[goal] < 0.45:
        center_ridge_pos = goal
    else:
        safe_ridge_scores = np.where(
            next_circ_mask,
            cover * 4.0 - threat * 15.0 - np.hypot(yy - goal[0], xx - goal[1]) * 0.08,
            -999.0
        )
        safe_ridge_idx = np.unravel_index(np.argmax(safe_ridge_scores), safe_ridge_scores.shape)
        center_ridge_pos = (int(safe_ridge_idx[0]), int(safe_ridge_idx[1]))
    is_center_open = bool(cover[center_ridge_pos] < 0.40)

    # Candidate Target 3: Gatekeep Anchor on the Yellow Zone perimeter facing outside enemies
    if enemies_outside_next:
        d_goal_center = np.hypot(yy - goal[0], xx - goal[1])
        d_squad = np.hypot(yy - pos[0], xx - pos[1])
        edge_dist = np.abs(d_goal_center - (r_next * 0.90))
        edge_score = np.maximum(0.0, 1.0 - edge_dist / max(1.0, r_next * 0.25))
        gate_scores = np.where(
            next_circ_mask,
            edge_score * 12.0 + cover * 3.5 - threat * 8.0 - d_squad * 0.40,
            -999.0
        )
        gatekeep_pos = np.unravel_index(np.argmax(gate_scores), gate_scores.shape)
        gatekeep_pos = (int(gatekeep_pos[0]), int(gatekeep_pos[1]))
    else:
        gatekeep_pos = pos

    # Baseline probe for Expert System rules
    base_probe = baseline_astar(nav_mask, pos, goal)
    eta = base_probe["cost"] if base_probe["found"] else 99
    risk_probe = 1 - survival(base_probe["path"], threat, cover) if base_probe["found"] else 1.0

    facts = dict(time_left=time_left, eta=eta, in_zone=in_current_zone, in_next_zone=in_next_zone,
                 cover_here=float(cover[pos]), choke_near=choke_near(blocked, pos),
                 route_risk=risk_probe, threat_here=float(threat[pos]))
    out, fired = forward_chain(facts)
    decision = out["decision"]

    # Resolve target_pos and active tactical strategy
    if strategy_mode == "COMPOUND":
        target_pos = compound_in_yellow if not in_next_zone else (pos if cover[pos] >= 0.35 else compound_in_yellow)
        active_strategy = "FORTIFIED COMPOUND"
    elif strategy_mode == "RIDGE_CENTER":
        target_pos = center_ridge_pos
        active_strategy = "CENTER RIDGE & CAR BUNKER" if is_center_open else "CENTRAL COMPOUND"
    elif strategy_mode == "GATEKEEP":
        target_pos = pos if in_next_zone else gatekeep_pos
        active_strategy = "GATEKEEP ZONE PERIMETER"
    else:  # AUTO
        if not in_next_zone:
            # USER RULE 1: SQUAD IS NOT IN YELLOW ZONE -> PRIORITY IS TO GO INTO THAT ZONE
            decision = "ROTATE"
            if threat[compound_in_yellow] < 0.60 and cover[compound_in_yellow] >= 0.25:
                target_pos = compound_in_yellow
                active_strategy = "FORTIFIED COMPOUND"
            else:
                target_pos = center_ridge_pos
                active_strategy = "CENTER RIDGE & CAR BUNKER" if is_center_open else "CENTRAL COMPOUND"
        else:
            # USER RULE 2: SQUAD IS ALREADY IN YELLOW ZONE -> GATEKEEP TEAMS NOT IN YELLOW ZONE
            if enemies_outside_next:
                decision = "GATEKEEP"
                target_pos = pos
                active_strategy = "GATEKEEP ZONE PERIMETER"
            else:
                decision = "HOLD"
                target_pos = pos
                active_strategy = "FORTIFIED COMPOUND" if cover[pos] >= 0.35 else "CENTER RIDGE & CAR BUNKER"

    # Compute Threat-Aware route using dry-land navigation mask
    if target_pos == pos:
        res = dict(path=[pos], cost=0.0, expanded=1, generated=1, peak_open=1, ms=0.1, found=True)
        base = dict(path=[pos], cost=0.0, expanded=1, generated=1, peak_open=1, ms=0.1, found=True)
    else:
        res = threat_astar(nav_mask, threat, cover, pos, target_pos)
        base = baseline_astar(nav_mask, pos, target_pos)

    # Compute Gatekeep Field-of-Fire Angle & Enemy Approach Sides
    gatekeep_info = compute_gatekeep_context(target_pos, goal, enemies_outside_next, n, enemies_inside=enemies_inside_next)

    # Generate Professional Esports Briefing, Protocols, and Risk Factors
    target_cover = float(cover[target_pos])
    is_target_open = bool(target_cover < 0.40)
    protocols = []
    risks = []
    tactical_advice = ""

    dist_to_yellow_m = int(abs(math.dist(pos, goal) - r_next) * (8000 / n))
    dist_to_edge_m = int(abs(math.dist(pos, zone_c) - r_now) * (8000 / n))

    if in_next_zone:
        # Squad is firmly inside the Yellow Safe Zone
        if active_strategy == "GATEKEEP ZONE PERIMETER":
            primary_sec = gatekeep_info['primary_sector']
            b_mid = int(gatekeep_info['mean_bearing'])
            b_s, b_e = int(gatekeep_info['b_start']), int(gatekeep_info['b_end'])
            n_out = len(enemies_outside_next)
            if gatekeep_info['distinct_sectors']:
                sides_desc = ", ".join([f"{s['sector']} ({int(s['bearing'])}° • {s['distance_m']}m)" for s in gatekeep_info['distinct_sectors'][:3]])
                tactical_advice = (
                    f"Squad is securely positioned inside the Yellow Safe Zone. "
                    f"Active Tactical Mission: Gatekeep the {n_out} hostile squad(s) trapped outside in the dying blue zone. "
                    f"Intercept incoming rotators from these sectors: {sides_desc}. "
                    "Pre-aim DMR sightlines and hold trigger discipline until enemies commit to open kill-zones."
                )
                for s in gatekeep_info['distinct_sectors'][:3]:
                    protocols.append({
                        "title": f"Gatekeep {s['sector']} Approach Corridor ({int(s['bearing'])}°)",
                        "desc": f"Hostile squad at {s['distance_m']}m is outside the safe zone and must rotate into your line. Pre-aim long-range DMR crossfire."
                    })
            else:
                tactical_advice = (
                    f"Squad is securely positioned inside the Yellow Safe Zone. "
                    f"Established perimeter gatekeep overlooking the {primary_sec.upper()} corridor ({b_s}°–{b_e}°). "
                    "Punish late-rotating vehicles or sprinters under blue zone pressure."
                )
                protocols.append({
                    "title": f"Gatekeep {primary_sec} Perimeter Sector ({b_mid}°)",
                    "desc": "Anchor behind rock or ridge crests with backs to the safe zone, pre-aiming DMR sightlines into the blue zone."
                })

            protocols.append({
                "title": "Ignore Internal Secured Squads",
                "desc": "Focus strictly on external teams. Do not compromise perimeter cover to duel squads already established in compounds."
            })
            protocols.append({
                "title": "Split 2-2 Angle Formation",
                "desc": f"Assign 2 DMR players to the active gatekeep angle ({b_s}°–{b_e}°), while 2 players watch compound flank angles."
            })
            risks.append({
                "hazard": "Desperate Full-Speed Vehicle Crash",
                "mitigation": "Hold vehicle spray and frag grenades ready; fall back to interior building walls if enemies crash."
            })
            if enemies_inside_next:
                risks.append({
                    "hazard": f"Internal Compound Threats ({len(enemies_inside_next)} squads inside)",
                    "mitigation": "Maintain courtyard wall cover; do not expose your back to nearby compounds already occupied by enemies."
                })
        elif active_strategy == "FORTIFIED COMPOUND":
            tactical_advice = (
                "Squad is firmly secured inside a fortified compound within the Yellow Safe Zone. "
                f"{'All spotted enemies are already inside the zone perimeter. ' if enemies_inside_next else ''}"
                "Hold interior compound defense, maintaining 360-degree observation while conserving utility and armor."
            )
            protocols.append({
                "title": "Courtyard Vehicle Parking",
                "desc": "Park vehicles inside the compound courtyard or against interior walls to prevent snipers from puncturing tires."
            })
            protocols.append({
                "title": "Window Crossfire Setup",
                "desc": "Position two players on front-facing windows for perimeter oversight and one player watching the rear entrance."
            })
            risks.append({
                "hazard": "Grenade Clearance by Attackers",
                "mitigation": "Do not stack multiple players in the same room; hold separate window angles."
            })
        else:  # CENTER RIDGE & CAR BUNKER
            tactical_advice = (
                f"Squad is holding high-ground ridge at Grid ({target_pos[1]}, {target_pos[0]}) inside the Yellow Safe Zone. "
                "Center terrain gives prime position for future circle shifts; execute Ridge & Car Bunker fortification."
            )
            protocols.append({
                "title": "Hull-Down Ridge Positioning",
                "desc": "Hold the concave reverse slope of the ridge. Stay below the crest line to prevent silhouette spotting from surrounding compound roofs."
            })
            protocols.append({
                "title": "Vehicle Fortification (Car Bunker)",
                "desc": "Park 2–3 vehicles in a V-formation. Puncture exposed tires immediately to lower the vehicle chassis to the ground."
            })
            risks.append({
                "hazard": "Compound Window Sightlines",
                "mitigation": "Stay strictly below the crest lip and avoid static peeking against compound windows."
            })
    else:
        # Squad is OUTSIDE the Yellow Safe Zone -> PRIORITY IS ROTATION INTO YELLOW ZONE
        if active_strategy == "FORTIFIED COMPOUND":
            tactical_advice = (
                f"Squad is outside the Yellow Safe Zone ({dist_to_yellow_m}m to safe line). "
                f"Top Priority: Execute tactical flank rotation into Fortified Compound at Grid ({target_pos[1]}, {target_pos[0]}) inside the Yellow Zone. "
                "Flanking via the un-camped weak side to secure concrete hard cover for upcoming phases."
            )
            protocols.append({
                "title": "Weak-Side Flank Transit",
                "desc": "Follow the planned green flank path around camper fire lines to safely reach the yellow safe zone."
            })
            protocols.append({
                "title": "Courtyard Vehicle Crash",
                "desc": "Drive directly into the compound courtyard. Deploy one smoke grenade upon dismount to cover room-clearing entries."
            })
            protocols.append({
                "title": "Compound Room Clearance",
                "desc": "Clear lower floors with utility before anchoring upper windows; check corners for lurking campers."
            })
            risks.append({
                "hazard": "Pre-occupied Compound",
                "mitigation": "Scout upper windows for player models before vehicle commit; throw utility into entry doors."
            })
            risks.append({
                "hazard": "Late Gatekeepers on Approach",
                "mitigation": "Follow the planned green flank path to bypass camper sniper ranges."
            })
        else:
            tactical_advice = (
                f"Squad is outside the Yellow Safe Zone ({dist_to_yellow_m}m to safe line). "
                f"Top Priority: Execute tactical rotation into Open Ridge / Field at Grid ({target_pos[1]}, {target_pos[0]}) inside the Yellow Zone. "
                "Because destination is open terrain without hard buildings, squad must execute Ridge & Car Bunker tactics upon arrival."
            )
            protocols.append({
                "title": "Vehicle Fortification (Car Bunker)",
                "desc": "Park 2–3 vehicles in a V-formation or triangle. Puncture exposed tires immediately to eliminate undercarriage gaps and create solid metal cover."
            })
            protocols.append({
                "title": "Hull-Down Ridge Play (Reverse Slope)",
                "desc": "Drop into the lowest concave dip of the ridge. Stay strictly below the crest lip to break line-of-sight from surrounding compound snipers."
            })
            protocols.append({
                "title": "Smoke Screen Readiness",
                "desc": "Carry 4–6 smoke grenades per player to deploy visual smoke walls for healing or 360-degree revives."
            })
            protocols.append({
                "title": "Vehicle Momentum & Serpentine Transit",
                "desc": "Maintain 80 km/h vehicle speed; serpentine while crossing open grass to avoid compound sniper lead shots."
            })
            risks.append({
                "hazard": "Blue Zone Damage Tick",
                "mitigation": "Commit to vehicle transit rather than foot sprinting to arrive before blue zone pressure increases."
            })
            risks.append({
                "hazard": "Open Field Crossfire",
                "mitigation": "Avoid stopping in open grass; crash directly into the reverse slope of the ridge."
            })

    # Compatibility list of warnings
    warnings = [f"{p['title']}: {p['desc']}" for p in protocols]

    return dict(
        posterior=post,
        goal=goal,
        target_pos=target_pos,
        compound_pos=compound_in_yellow,
        center_ridge_pos=center_ridge_pos,
        gatekeep_pos=gatekeep_pos,
        active_strategy=active_strategy,
        target_terrain="Open Ridge / Field" if is_target_open else "Fortified Compound",
        facts=out,
        rules_fired=fired,
        decision=decision,
        route=res,
        baseline_route=base,
        mask=zone_mask,
        next_mask=next_circ_mask,
        nav_mask=nav_mask,
        in_zone=in_current_zone,
        in_next_zone=in_next_zone,
        enemies_outside_next=enemies_outside_next,
        enemies_inside_next=enemies_inside_next,
        tactical_advice=tactical_advice,
        protocols=protocols,
        risks=risks,
        warnings=warnings,
        gatekeep_info=gatekeep_info
    )



