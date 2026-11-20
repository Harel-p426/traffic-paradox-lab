"""Experiment 3: how common is the paradox in random road networks?

For each random network we add one cheap new road between two nodes that are
not yet connected, and compare the average travel time at equilibrium before
and after.

Run from the project root:   python -m experiments.random_search
"""

import numpy as np

from experiments.style import BLUE, INK2, ORANGE, setup
from trafficlab import Edge, Network, equilibrium

N_NETWORKS = 3000
N_NODES = 6
SEED = 0
TOL = 1e-4          # changes smaller than 0.01% count as "no change"

rng = np.random.default_rng(SEED)


def random_network(n=N_NODES):
    """Random acyclic network on nodes 0..n-1 (origin 0, destination n-1).

    Each possible road i->j (i<j) exists with probability 1/2. Half of the roads
    congest (time = a*x + b), half have a fixed travel time.
    """
    while True:
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    if rng.random() < 0.5:
                        edges.append(Edge(str(i), str(j), a=rng.uniform(0.2, 2), b=rng.uniform(0, 1)))
                    else:
                        edges.append(Edge(str(i), str(j), a=0.0, b=rng.uniform(0.5, 3)))
        try:
            return Network(edges, "0", str(n - 1))
        except ValueError:     # no path from origin to destination: draw again
            continue


changes, examples = [], []
networks_with_a_harmful_road = 0     # networks where AT LEAST ONE possible new road hurts everyone
networks_tested = 0
for _ in range(N_NETWORKS):
    net = random_network()
    existing = {(e.u, e.v) for e in net.edges}
    missing = [(str(i), str(j)) for i in range(N_NODES) for j in range(i + 1, N_NODES)
               if (str(i), str(j)) not in existing]
    if not missing:
        continue
    demand = rng.uniform(0.5, 3.0)
    before = equilibrium(net, demand).avg_time

    # (a) Question 1: if we add ONE random new road, how often does it hurt?
    u, v = missing[rng.integers(len(missing))]
    new_edge = Edge(u, v, a=0.0, b=rng.uniform(0.0, 0.3))      # a cheap, uncongestable road
    after = equilibrium(net.with_edge(new_edge), demand).avg_time
    change = after / before - 1.0
    changes.append(change)
    if change > TOL:
        examples.append((change, net, new_edge, demand))

    # (b) Question 2: is there ANY possible new road (between unconnected nodes) that hurts?
    networks_tested += 1
    for (u, v) in missing:
        candidate = Edge(u, v, a=0.0, b=0.1)
        if equilibrium(net.with_edge(candidate), demand).avg_time > before * (1 + TOL):
            networks_with_a_harmful_road += 1
            break

changes = np.array(changes)
worse = changes > TOL
better = changes < -TOL
print(f"{len(changes)} random networks ({N_NODES} nodes), one new cheap road each")
print(f"  new road makes EVERY driver slower (Braess paradox): {worse.mean():6.1%}")
print(f"  new road makes every driver faster:                  {better.mean():6.1%}")
print(f"  no change:                                           {(~worse & ~better).mean():6.1%}")
print(f"  worst slowdown found: +{changes.max():.1%}")
print(f"\nNetworks where at least one missing road slows every driver "
      f"(each missing road tried with fixed time 0.1): {networks_with_a_harmful_road / networks_tested:.1%}")

plt = setup()
fig, ax = plt.subplots(figsize=(9, 5.2))
pct = changes * 100
changed = pct[np.abs(changes) > TOL]                 # networks where something happened
bins = np.linspace(min(changed.min(), -40), max(changed.max(), 10), 61)
ax.hist(changed[changed < 0], bins=bins, color=BLUE, edgecolor="#fcfcfb", linewidth=1.2)
ax.hist(changed[changed > 0], bins=bins, color=ORANGE, edgecolor="#fcfcfb", linewidth=1.2)
ax.axvline(0, color=INK2, linewidth=1)
ax.set_yscale("log")
ax.set_ylim(0.8, 600)          # headroom for the labels
ax.set_xlabel("change in average travel time after adding one road (%)")
ax.set_ylabel("number of networks (log scale)")
ax.set_title(f"Braess paradox in random networks: {worse.mean():.1%} get slower")
ax.text(0.40, 0.90, "faster", color=BLUE, fontweight="bold", ha="center", transform=ax.transAxes)
ax.text(0.93, 0.50, "slower for\neveryone", color=ORANGE, fontweight="bold", ha="center",
        transform=ax.transAxes)
ax.text(0.02, 0.96, f"(not shown: {(~worse & ~better).mean():.0%} of networks, no change)",
        color=INK2, ha="left", va="top", fontsize=10, transform=ax.transAxes)
fig.tight_layout()
fig.savefig("images/random_search.png", dpi=160)
print("saved images/random_search.png")

if examples:
    change, net, edge, demand = max(examples, key=lambda t: t[0])
    print(f"\nWorst example: demand {demand:.2f}, new road {edge.u}->{edge.v} (time {edge.b:.2f}), "
          f"average time goes up {change:.1%}")
