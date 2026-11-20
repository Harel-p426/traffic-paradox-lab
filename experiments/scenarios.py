"""Experiment 2: the four scenarios on the classic network (4,000 drivers).

Run from the project root:   python -m experiments.scenarios
"""

from experiments.style import AQUA, BLUE, INK, ORANGE, setup
from trafficlab import (BRAESS_DEMAND, braess, equilibrium, equilibrium_with_tolls,
                        marginal_cost_tolls, optimum)

plt = setup()
D = BRAESS_DEMAND
net = braess(True)

rows = [
    ("No new road\n(selfish drivers)", equilibrium(braess(False), D).avg_time, BLUE),
    ("New road\n(selfish drivers)", equilibrium(net, D).avg_time, ORANGE),
    ("New road +\ncongestion tolls", equilibrium_with_tolls(net, D, marginal_cost_tolls(net, D)).avg_time, AQUA),
    ("New road,\nplanner assigns routes", optimum(net, D).avg_time, AQUA),
]
for name, value, _ in rows:
    print(f"{name.replace(chr(10), ' '):35s} {value:6.2f} min")

fig, ax = plt.subplots(figsize=(9, 5.2))
xs = range(len(rows))
bars = ax.bar(xs, [r[1] for r in rows], color=[r[2] for r in rows], width=0.6)
for x, (_, value, _) in zip(xs, rows):
    ax.text(x, value + 1.2, f"{value:.1f}", ha="center", va="bottom", color=INK, fontweight="bold")
ax.set_xticks(list(xs), [r[0] for r in rows])
ax.set_ylabel("average travel time (minutes)")
ax.set_title("Same 4,000 drivers: the new road makes everyone slower, tolls fix it")
ax.set_ylim(0, 92)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig("images/scenarios.png", dpi=160)
print("saved images/scenarios.png")
