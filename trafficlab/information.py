"""Information design: what if the app controls what drivers know?

In the basic model every driver knows every road and picks the fastest route for
themselves. Here we ask a different question: can an app that *recommends routes*
(and does not necessarily show every road) bring the traffic close to the optimum
without tolls?

Three tools:

1. lottery()                    The app draws a route for each driver at random, with the
                                probabilities of the optimal flow. Would drivers obey?
2. equilibrium_by_knowledge()   A share of the drivers knows about a "hidden" road and picks
                                routes selfishly; the others only know the other roads.
3. guided_outcome()             The same share knows the hidden road, but the remaining drivers
                                follow the app, which assigns their routes as well as possible.

Assumptions (these are the limits of the model, and they are strong):

* Drivers who follow the app do exactly what it says, as long as they do not know about a
  better route. lottery() checks when this assumption is reasonable.
* The "informed" drivers know every road and always choose the fastest route for themselves.
* Traffic is a continuum: one driver switching routes does not change anybody's travel time.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .model import Network
from .solver import _solve, optimum


# ----------------------------------------------------------------- helpers
def paths_using(net: Network, road: tuple[str, str]) -> np.ndarray:
    """Boolean array: which paths use the road (u, v)?"""
    matches = [i for i, e in enumerate(net.edges) if (e.u, e.v) == tuple(road)]
    if not matches:
        raise ValueError(f"the network has no road {road[0]}->{road[1]}")
    return net.incidence[:, matches[0]] > 0.5


def path_times(net: Network, edge_flows: np.ndarray) -> np.ndarray:
    """Travel time of every path, given the flow on every road."""
    return net.incidence @ net.edge_times(edge_flows)


def _check_share(q: float) -> None:
    if not 0.0 <= q <= 1.0:
        raise ValueError("informed_share must be between 0 and 1")


# ------------------------------------------------------------------ lottery
@dataclass
class LotteryReport:
    """The optimal flow turned into a lottery, and whether drivers would obey it."""
    network: Network
    demand: float
    hidden: tuple[str, str]
    labels: list[str]
    shares: np.ndarray             # probability that a driver is told each path
    times: np.ndarray              # travel time of each path under the lottery
    uses_hidden: np.ndarray        # does the path use the hidden road?
    avg_time: float                # expected travel time of one driver (= optimum)
    gain_if_informed: np.ndarray   # minutes saved by switching, for a driver who knows every road
    gain_if_uninformed: np.ndarray  # same, for a driver who knows only the roads he was shown
    tolerance: float

    @property
    def recommended(self) -> np.ndarray:
        """Paths that the lottery actually recommends to somebody."""
        return self.shares > 1e-9

    @property
    def obedient_if_informed(self) -> np.ndarray:
        return self.gain_if_informed <= self.tolerance

    @property
    def obedient_if_uninformed(self) -> np.ndarray:
        return self.gain_if_uninformed <= self.tolerance

    def table(self) -> str:
        rows = [f"{'told to take':<12} {'share':>7} {'time':>7}  "
                f"{'obeys if informed':>18} {'obeys if uninformed':>20}"]
        for r in np.flatnonzero(self.recommended):
            rows.append(f"{self.labels[r]:<12} {self.shares[r] * 100:>6.4g}% {self.times[r]:>7.2f}  "
                        f"{'yes' if self.obedient_if_informed[r] else 'NO (saves %.1f)' % self.gain_if_informed[r]:>18} "
                        f"{'yes' if self.obedient_if_uninformed[r] else 'NO':>20}")
        rows.append(f"expected time of a random driver: {self.avg_time:.4f}")
        return "\n".join(rows)


def lottery(net: Network, demand: float, hidden: tuple[str, str], tolerance: float = 1e-4) -> LotteryReport:
    """Turn the optimal flow into route probabilities and test whether drivers would obey.

    A driver told to take path r compares the time of r with the other paths he knows.
    * An INFORMED driver knows every path.
    * An UNINFORMED driver knows only the paths that do not use the hidden road. If he is
      told to take a path that uses the hidden road, he learns about it, so he then knows everything.
    He obeys if no path he knows is faster than the path he was told to take (a gain of 0 means
    he is indifferent, and we assume an indifferent driver obeys).
    """
    opt = optimum(net, demand)
    times = path_times(net, opt.edge_flows)
    uses = paths_using(net, hidden)
    old = np.flatnonzero(~uses)
    if len(old) == 0:
        raise ValueError("every path uses the hidden road")

    gain_informed = np.maximum(0.0, times - times.min())
    gain_uninformed = np.where(uses, gain_informed, np.maximum(0.0, times - times[old].min()))
    return LotteryReport(net, demand, tuple(hidden), net.path_labels, opt.path_flows / demand,
                         times, uses, opt.avg_time, gain_informed, gain_uninformed, tolerance)


# ----------------------------------------------- selfish drivers, mixed knowledge
@dataclass
class KnowledgeOutcome:
    informed_share: float
    avg_time: float
    total_time: float
    informed_avg_time: float       # nan if nobody is informed
    uninformed_avg_time: float     # nan if everybody is informed
    informed_flows: np.ndarray
    uninformed_flows: np.ndarray


def equilibrium_by_knowledge(net: Network, demand: float, informed_share: float,
                             hidden: tuple[str, str]) -> KnowledgeOutcome:
    """Everybody is selfish, but only `informed_share` of the drivers know the hidden road.

    The other drivers choose the fastest route among the paths that avoid it.
    This is an equilibrium with two classes of drivers (each class has its own set of allowed paths).
    """
    _check_share(informed_share)
    if demand <= 0:
        raise ValueError("demand must be positive")
    n = len(net.paths)
    old = np.flatnonzero(~paths_using(net, hidden))
    if len(old) == 0:
        raise ValueError("every path uses the hidden road")

    f, l = informed_share * demand, (1 - informed_share) * demand
    classes, which = [], []
    if f > 0:
        classes.append((f, np.arange(n))); which.append("informed")
    if l > 0:
        classes.append((l, old)); which.append("uninformed")
    hs, _, _ = _solve(net.incidence, net.a, net.b, classes)
    flows = dict(zip(which, hs))
    h_inf = flows.get("informed", np.zeros(n))
    h_un = flows.get("uninformed", np.zeros(n))

    x = net.edge_flows(h_inf + h_un)
    times = path_times(net, x)
    total = net.total_time(x)
    return KnowledgeOutcome(
        informed_share, total / demand, total,
        float(h_inf @ times / f) if f > 0 else float("nan"),
        float(h_un @ times / l) if l > 0 else float("nan"),
        h_inf, h_un)


# ------------------------------------------------- an app guides some drivers
@dataclass
class GuidedOutcome:
    informed_share: float
    avg_time: float
    total_time: float
    guided_flows: np.ndarray       # drivers the app sends on each path
    informed_flows: np.ndarray     # drivers who choose for themselves on each path
    guided_avg_time: float         # nan if nobody is guided
    informed_avg_time: float       # nan if nobody is informed


def _compositions(total: int, parts: int):
    """All ways to write `total` as an ordered sum of `parts` non-negative integers."""
    if parts == 1:
        yield (total,)
        return
    for i in range(total + 1):
        for rest in _compositions(total - i, parts - 1):
            yield (i,) + rest


def _informed_response(net: Network, informed_demand: float, guided_flows: np.ndarray):
    """Equilibrium of the informed drivers when `guided_flows` are already on the roads."""
    n = len(net.paths)
    offset = net.edge_flows(guided_flows)
    if informed_demand <= 0:
        return np.zeros(n), offset
    hs, _, _ = _solve(net.incidence, net.a, net.b, [(informed_demand, np.arange(n))], offset=offset)
    return hs[0], offset + net.edge_flows(hs[0])


def evaluate_guidance(net: Network, demand: float, informed_share: float,
                      guided_flows: np.ndarray) -> GuidedOutcome:
    """Outcome when the app sends `guided_flows` drivers on each path and the rest choose selfishly."""
    _check_share(informed_share)
    guided_flows = np.asarray(guided_flows, dtype=float)
    f, l = informed_share * demand, (1 - informed_share) * demand
    if guided_flows.min() < -1e-9 or abs(guided_flows.sum() - l) > 1e-6 * max(1.0, demand):
        raise ValueError("guided flows must be non-negative and add up to the guided demand")
    h, x = _informed_response(net, f, guided_flows)
    times = path_times(net, x)
    total = net.total_time(x)
    return GuidedOutcome(
        informed_share, total / demand, total, guided_flows, h,
        float(guided_flows @ times / l) if l > 0 else float("nan"),
        float(h @ times / f) if f > 0 else float("nan"))


def guided_outcome(net: Network, demand: float, informed_share: float,
                   resolution: int = 40) -> GuidedOutcome:
    """The best the app can do: assign the guided drivers to paths so total travel time is smallest,
    knowing that the informed drivers will then respond selfishly.

    This is a "Stackelberg" problem (the app moves first, the informed drivers react).
    It is searched, not solved in closed form: a grid over all assignments with `resolution`
    steps, followed by local refinement. Meant for small networks (at most 4 paths).
    """
    _check_share(informed_share)
    if demand <= 0:
        raise ValueError("demand must be positive")
    n = len(net.paths)
    if n > 4:
        raise ValueError("guided_outcome searches over assignments and supports at most 4 paths")
    f, l = informed_share * demand, (1 - informed_share) * demand

    def total(g: np.ndarray) -> float:
        _, x = _informed_response(net, f, g)
        return net.total_time(x)

    if l <= 1e-12:
        return evaluate_guidance(net, demand, informed_share, np.zeros(n))

    # 1. coarse grid over every way to split the guided drivers between the paths
    best_g, best_val = None, np.inf
    for comp in _compositions(resolution, n):
        g = l * np.array(comp, dtype=float) / resolution
        val = total(g)
        if val < best_val:
            best_g, best_val = g, val

    # 2. local refinement: move drivers between pairs of paths, halving the step when stuck
    step, evals = l / resolution, 0
    while step > l * 1e-9 and evals < 500_000:
        improved = True
        while improved and evals < 500_000:
            improved = False
            for i in range(n):
                for j in range(n):
                    if i == j or best_g[i] <= 0:
                        continue
                    d = min(step, best_g[i])
                    cand = best_g.copy()
                    cand[i] -= d
                    cand[j] += d
                    val = total(cand)
                    evals += 1
                    if val < best_val - 1e-13 * abs(best_val):
                        best_g, best_val, improved = cand, val, True
        step /= 2

    return evaluate_guidance(net, demand, informed_share, best_g)
