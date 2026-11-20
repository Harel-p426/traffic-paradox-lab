"""Experiment 1: for how many drivers does the new road make everybody slower?

Run from the project root:   python -m experiments.demand_sweep
"""

import numpy as np

from experiments.style import AQUA, BLUE, INK2, ORANGE, setup
from trafficlab import braess, equilibrium, optimum

plt = setup()

demands = np.arange(50, 12001, 50, dtype=float)   # start at 50: zero drivers has no travel time
without = np.array([equilibrium(braess(False), d).avg_time for d in demands])
with_road = np.array([equilibrium(braess(True), d).avg_time for d in demands])
best_possible = np.array([optimum(braess(True), d).avg_time for d in demands])

# Where does the new road make drivers strictly slower?
hurts = with_road > without + 1e-3     # ignore differences below 0.001 minutes (numerical noise)
lo, hi = demands[hurts].min(), demands[hurts].max()
print(f"The new road makes every driver slower for demands between about "
      f"{lo - 50:,.0f} and {hi + 50:,.0f} drivers.")
print(f"  at 4,000 drivers: without {without[demands == 4000][0]:.1f} min, "
      f"with {with_road[demands == 4000][0]:.1f} min")

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.axvspan(lo - 25, hi + 25, color=ORANGE, alpha=0.08, lw=0)
ax.plot(demands, without, color=BLUE, label="no new road")
ax.plot(demands, with_road, color=ORANGE, label="new road, everyone selfish")
ax.plot(demands, best_possible, color=AQUA, linestyle=(0, (5, 3)), label="new road, best possible assignment")

# direct labels where the lines are clearly apart
ax.text(300, without[demands == 300][0] + 2.5, "no new road", color=BLUE, ha="left", va="bottom",
        fontweight="bold")
ax.text(6700, 92.5, "new road, everyone selfish", color=ORANGE, ha="center", va="bottom",
        fontweight="bold")
ax.text(5200, 12, "the paradox:\nthe new road hurts everyone", color=INK2, ha="center",
        va="bottom", fontsize=10)
ax.legend(loc="lower right", frameon=False, labelcolor=INK2)

ax.set_xlabel("number of drivers")
ax.set_ylabel("average travel time (minutes)")
ax.set_title("When does adding a road make everyone slower?")
ax.set_xlim(0, 12000)
ax.set_ylim(0, 112)
fig.tight_layout()
fig.savefig("images/demand_sweep.png", dpi=160)
print("saved images/demand_sweep.png")
