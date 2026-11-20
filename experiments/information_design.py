"""Experiment 4: what if the app controls what drivers know?

Part 1  The lottery: the app tells each driver a route at random (with the optimal shares).
        Would drivers obey?
Part 2  A share q of the drivers knows about the new road, the rest do not.
        (a) everybody is selfish, (b) an app assigns routes to the drivers who do not know.

Run from the project root:   python -m experiments.information_design
"""

import numpy as np

from experiments.style import AQUA, BLUE, INK, INK2, ORANGE, setup
from trafficlab import (BRAESS_DEMAND, BRAESS_SHORTCUT, braess, equilibrium, equilibrium_by_knowledge,
                        guided_outcome, lottery, optimum)

plt = setup()
D, H = BRAESS_DEMAND, BRAESS_SHORTCUT
net = braess(True)

# ------------------------------------------------------------------ Part 1: lottery
rep = lottery(net, D, H)
no_road = equilibrium(braess(False), D).avg_time
everyone_knows = equilibrium(net, D).avg_time
print("=== Part 1: the lottery (4,000 drivers) ===")
print(f"no new road:                 {no_road:.2f} min")
print(f"new road, everybody knows:   {everyone_knows:.2f} min")
print(f"new road, lottery:           {rep.avg_time:.4f} min on average\n")
print(rep.table())

idx = [r for r in np.flatnonzero(rep.recommended)]
idx.sort(key=lambda r: -rep.times[r])
labels = [f"told {rep.labels[r]}\n({rep.shares[r] * 100:.4g}% of drivers)" for r in idx]
values = [rep.times[r] for r in idx]

fig, ax = plt.subplots(figsize=(9, 5.2))
bars = ax.bar(range(len(idx)), values, color=AQUA, width=0.55, zorder=3)
for x, v in enumerate(values):
    ax.text(x, v + 1.2, f"{v:.1f}", ha="center", va="bottom", color=INK, fontweight="bold")
for y, color in [(everyone_knows, ORANGE), (no_road, BLUE)]:
    ax.axhline(y, color=color, linestyle=(0, (5, 3)), linewidth=1.6, zorder=2)
ax.text(2.45, everyone_knows + 1.2, "80: new road, everybody knows", color=ORANGE, ha="right",
        va="bottom", fontweight="bold")
ax.text(2.45, no_road + 1.2, "65: no new road", color=BLUE, ha="right", va="bottom", fontweight="bold")
ax.set_xticks(range(len(idx)), labels)
ax.set_ylabel("travel time (minutes)")
ax.set_title(f"The lottery is fair only on average ({rep.avg_time:.2f}): most drivers get more than 65")
ax.set_ylim(0, 92)
ax.set_xlim(-0.6, 2.6)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig("images/lottery.png", dpi=160)

# ----------------------------------------------------------- Part 2: share who knows
qs = np.unique(np.round(np.append(np.arange(0, 1.0001, 0.01), 0.125), 4))   # includes 1/8 exactly
selfish = np.array([equilibrium_by_knowledge(net, D, q, H).avg_time for q in qs])
guided = np.array([guided_outcome(net, D, q).avg_time for q in qs])
closed_form = 65 - 5 * qs + 20 * qs ** 2
optimum_time = optimum(net, D).avg_time

print("\n=== Part 2: a share q of the drivers knows about the new road ===")
print(f"{'q':>5} {'everybody selfish':>18} {'app guides the rest':>20}")
for q in (0, 0.05, 0.1, 0.125, 0.25, 0.5, 0.75, 1.0):
    i = int(np.argmin(abs(qs - q)))
    print(f"{q:>5.3f} {selfish[i]:>18.4f} {guided[i]:>20.4f}")
print(f"\nmax |selfish - (65 - 5q + 20q^2)| over the grid: {abs(selfish - closed_form).max():.2e}")
print(f"best share of informed drivers (selfish): q = {qs[int(np.argmin(selfish))]:.3f}, "
      f"time {selfish.min():.4f}  (optimum {optimum_time:.4f})")
crossing = qs[selfish <= 65 + 1e-9].max()
print(f"selfish time is at or below 65 up to q = {crossing:.3f}, and above 65 after that")
gain = selfish - guided
helped = qs[gain > 1e-6]
print(f"app helps (gain > 0) for q up to {helped.max():.3f} (next grid point 0.125 has no gain); "
      f"largest gain {gain.max():.4f} min at q = {qs[int(np.argmax(gain))]:.2f}")
print(f"guided time equals the optimum for q <= {qs[abs(guided - optimum_time) < 1e-4].max():.3f}")

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.axhline(no_road, color=BLUE, linestyle=(0, (5, 3)), linewidth=1.6)
ax.plot(qs * 100, selfish, color=ORANGE, label="everybody selfish")
ax.plot(qs * 100, guided, color=AQUA, linestyle=(0, (4, 3)), label="app assigns routes to drivers who do not know")
best_q = qs[int(np.argmin(selfish))]
ax.plot([best_q * 100], [selfish.min()], marker="o", markersize=8, color=ORANGE,
        markeredgecolor="#fcfcfb", markeredgewidth=2, zorder=5)
ax.annotate(f"{best_q * 100:.4g}% informed:\n{selfish.min():.2f} min (the optimum)", xy=(best_q * 100, selfish.min()),
            xytext=(23, 68.5), color=INK2, fontsize=10, ha="left",
            arrowprops=dict(arrowstyle="-", color=INK2, linewidth=1))
ax.text(99, no_road + 0.9, "65: no new road (or nobody knows)", color=BLUE, ha="right", va="bottom",
        fontweight="bold")
ax.text(99, selfish[-1] + 0.6, "80: everybody knows", color=ORANGE, ha="right", va="bottom", fontweight="bold")
ax.text(32, 63.2, "above 12.5% the two lines coincide: the app can no longer help", color=INK2, fontsize=10)
ax.set_xlabel("share of drivers who know about the new road (%)")
ax.set_ylabel("average travel time (minutes)")
ax.set_title("A few informed drivers help; too many hurt")
ax.set_xlim(0, 100)
ax.set_ylim(62, 82)
ax.legend(loc="upper left", frameon=False, labelcolor=INK2)
fig.tight_layout()
fig.savefig("images/information_design.png", dpi=160)
print("\nsaved images/lottery.png and images/information_design.png")
