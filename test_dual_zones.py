import numpy as np
import math
from zeus import make_erangel_map, make_threat, igl_decision, compute_gatekeep_context

# 1. Initialize grid
n = 80
rng = np.random.default_rng(42)
cover, blocked = make_erangel_map(n, rng)

# Zone settings
# Blue Zone at center (40, 40) with radius 25
zone_c = (40, 40)
r_now = 25

# Yellow Zone shifted to (48, 45) with radius 14
next_c = (48, 45)
r_next = 14

print(f"Blue Zone: Center={zone_c}, Radius={r_now}")
print(f"Yellow Zone: Center={next_c}, Radius={r_next}")

# Scenario A: Squad is NOT in yellow zone (pos at 25, 25 - inside blue, outside yellow)
pos_outside_yellow = (25, 25)
d_blue = math.hypot(pos_outside_yellow[0] - zone_c[0], pos_outside_yellow[1] - zone_c[1])
d_yellow = math.hypot(pos_outside_yellow[0] - next_c[0], pos_outside_yellow[1] - next_c[1])
print(f"\n--- Scenario A (Outside Yellow) ---")
print(f"Squad pos: {pos_outside_yellow}, dist to blue={d_blue:.1f} (r_now={r_now}), dist to yellow={d_yellow:.1f} (r_next={r_next})")
assert d_yellow > r_next, "Squad should be outside yellow zone"

# Place 2 enemies
enemies_a = [(22, 28), (55, 55)]
threat_a, enemies_list_a = make_threat(blocked, np.random.default_rng(42), k=0, extra=enemies_a)

res_a = igl_decision(
    cover=cover,
    blocked=blocked,
    threat=threat_a,
    pos=pos_outside_yellow,
    zone_c=zone_c,
    r_now=r_now,
    r_next=r_next,
    time_left=45,
    history_d2=[40, 90],
    strategy_mode="AUTO",
    enemies=enemies_list_a,
    next_c=next_c
)

print(f"Decision: {res_a['decision']}")
print(f"Active Strategy: {res_a['active_strategy']}")
print(f"In Zone: {res_a['in_zone']}, In Next Zone: {res_a['in_next_zone']}")
print(f"Target pos: {res_a['target_pos']}")
d_tgt_to_yellow = math.hypot(res_a['target_pos'][0] - next_c[0], res_a['target_pos'][1] - next_c[1])
print(f"Target distance to yellow center: {d_tgt_to_yellow:.1f} (r_next={r_next})")
assert res_a['decision'] == "ROTATE", "Priority MUST be ROTATE when outside yellow zone"
assert d_tgt_to_yellow <= r_next + 1.0, "Target MUST be inside the yellow zone"
print(">>> Scenario A PASSED! Priority is to rotate into the yellow zone.")

# Scenario B: Squad IS in yellow zone (pos at 46, 44), 
# 2 enemies outside yellow zone (at 25, 25 and 30, 55)
# 1 enemy already inside yellow zone (at 49, 46)
pos_inside_yellow = (46, 44)
d_yellow_b = math.hypot(pos_inside_yellow[0] - next_c[0], pos_inside_yellow[1] - next_c[1])
print(f"\n--- Scenario B (Inside Yellow, Mixed Enemies) ---")
print(f"Squad pos: {pos_inside_yellow}, dist to yellow={d_yellow_b:.1f} (r_next={r_next})")
assert d_yellow_b <= r_next, "Squad should be inside yellow zone"

enemy_outside_1 = (25, 25)
enemy_outside_2 = (30, 55)
enemy_inside = (49, 46)

campers_b = [enemy_outside_1, enemy_outside_2, enemy_inside]
threat_b, enemies_list_b = make_threat(blocked, np.random.default_rng(42), k=0, extra=campers_b)

res_b = igl_decision(
    cover=cover,
    blocked=blocked,
    threat=threat_b,
    pos=pos_inside_yellow,
    zone_c=zone_c,
    r_now=r_now,
    r_next=r_next,
    time_left=45,
    history_d2=[40, 90],
    strategy_mode="AUTO",
    enemies=enemies_list_b,
    next_c=next_c
)

print(f"Decision: {res_b['decision']}")
print(f"Active Strategy: {res_b['active_strategy']}")
print(f"In Next Zone: {res_b['in_next_zone']}")
print(f"Enemies Outside Next: {res_b['enemies_outside_next']}")
print(f"Enemies Inside Next: {res_b['enemies_inside_next']}")
print(f"Gatekeep Sectors: {[s['sector'] for s in res_b['gatekeep_info']['distinct_sectors']]}")

assert res_b['decision'] == "GATEKEEP", "Priority MUST be GATEKEEP when inside yellow zone with outside enemies"
assert len(res_b['enemies_outside_next']) == 2, "Must find exactly 2 enemies outside yellow"
assert len(res_b['enemies_inside_next']) == 1, "Must find exactly 1 enemy inside yellow"

# Check that the inside enemy is NOT in the gatekeep targets
outside_coords = set(res_b['enemies_outside_next'])
assert enemy_inside not in outside_coords, "Enemy already inside must NOT be gatekept"
print(">>> Scenario B PASSED! Gatekeeps strictly the enemies outside the yellow zone, ignoring inside teams.")
print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")
