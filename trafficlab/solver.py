"""Equilibrium and optimum of a selfish-routing game.

Two different questions about the same network and the same number of drivers:

1. EQUILIBRIUM (Wardrop / Nash): every driver picks the route that is fastest
   *for them*. At equilibrium all routes that are used take exactly the same
   time, and no unused route is faster. Nobody can gain by switching alone.

2. OPTIMUM: a central planner assigns routes to minimise the *total* travel
   time of everybody.

Both are convex optimisation problems.  For a road with time a*x + b:

    equilibrium = minimise  sum_e ( a_e/2 * x_e^2 + b_e * x_e )     (Beckmann potential)
    optimum     = minimise  sum_e ( a_e   * x_e^2 + b_e * x_e )     (total time)

The second is the first with the slope doubled: a driver who enters a road
slows down everybody already on it, so the "social" cost of one more driver is
the *marginal* cost  d/dx [ x * (a*x + b) ] = 2*a*x + b.

Solver: pairwise Frank-Wolfe on path flows.  Repeatedly take the most expensive
route that still carries traffic and move as many drivers as is useful to the
cheapest route (exact step size, because the objective is quadratic).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Network


@dataclass
class Solution:
    network: Network
    demand: float
    path_flows: np.ndarray    # drivers on each path
    edge_flows: np.ndarray    # drivers on each edge
    total_time: float         # sum of everyone's travel time (true road times)
    avg_time: float           # total_time / demand
    gap: float                # optimality gap at the end (should be ~0)
    iterations: int
    toll_revenue: float = 0.0  # total tolls paid (only when tolls were used)

    def describe(self) -> str:
        lines = [f"average time per driver: {self.avg_time:.3f}"]
        for label, flow in zip(self.network.path_labels, self.path_flows):
            if flow > 1e-6 * max(self.demand, 1e-12):
                lines.append(f"  {label}: {flow:,.1f} drivers")
        return "\n".join(lines)


def _solve(M: np.ndarray, a: np.ndarray, b: np.ndarray, classes, offset=None,
           tol: float = 1e-10, max_iter: int = 50000):
    """Minimise  sum_e (a_e/2 x_e^2 + b_e x_e)  where  x = offset + M.T @ (sum of class flows).

    `classes` is a list of (demand, allowed_paths): a group of drivers of that size who may
    only use the paths whose indices are in `allowed_paths`. Each class has its own
    constraint  sum(h_class) = demand,  h_class >= 0.

    With one class that may use every path and no offset this is the ordinary problem.
    More classes model drivers who know about different roads. `offset` is traffic that is
    already on the roads and is not being chosen here (for example, drivers who follow an app).

    M is the path-edge incidence matrix.  Returns (list of class flows, gap, iterations).
    """
    n_paths, n_edges = M.shape
    x0 = np.zeros(n_edges) if offset is None else np.asarray(offset, dtype=float)
    classes = [(float(d), np.asarray(allowed, dtype=int)) for d, allowed in classes]

    hs = []
    start_cost = M @ (a * x0 + b)              # cost of each path before anybody from the classes drives
    for demand, allowed in classes:
        h = np.zeros(n_paths)
        h[allowed[int(np.argmin(start_cost[allowed]))]] = demand
        hs.append(h)

    gap = 0.0
    for it in range(1, max_iter + 1):
        x = x0 + M.T @ sum(hs)
        cost = M @ (a * x + b)                 # marginal cost of each path = its "price" right now

        # Duality gap: how much cheaper things would be if every class were on its best path.
        # It is >= 0 and equals 0 exactly at the solution.
        gap, scale = 0.0, 0.0
        for (demand, allowed), h in zip(classes, hs):
            c = cost[allowed]
            gap += float(h[allowed] @ c - demand * c.min())
            scale += demand * float(c.min())
        if gap <= tol * max(1.0, scale):
            return hs, gap, it

        moved = False
        for (demand, allowed), h in zip(classes, hs):
            x = x0 + M.T @ sum(hs)             # refresh: a previous class may just have moved
            cost = M @ (a * x + b)
            sub = cost[allowed]
            best = int(allowed[int(np.argmin(sub))])
            used = h[allowed] > 1e-14
            worst = int(allowed[int(np.argmax(np.where(used, sub, -np.inf)))])
            if worst == best:
                continue

            # Move t drivers from `worst` to `best`.  Along this direction the objective is a
            # quadratic in t, so the best t has a closed form (clipped to what `worst` carries).
            dx = M[best] - M[worst]                              # change of edge flows per unit moved
            slope = float(cost[best] - cost[worst])              # <= 0: moving helps
            curvature = float(np.sum(a * dx * dx))
            t = h[worst] if curvature <= 1e-15 else min(h[worst], -slope / curvature)
            h[best] += t
            h[worst] -= t
            moved = True
        if not moved:
            return hs, gap, it

    return hs, gap, max_iter


def _minimize(M: np.ndarray, a: np.ndarray, b: np.ndarray, demand: float,
              tol: float = 1e-10, max_iter: int = 50000):
    """The ordinary problem: one class of drivers who may use every path. Returns (h, gap, it)."""
    hs, gap, it = _solve(M, a, b, [(demand, np.arange(M.shape[0]))], tol=tol, max_iter=max_iter)
    return hs[0], gap, it


def _package(net: Network, demand: float, h: np.ndarray, gap: float, it: int,
             toll_revenue: float = 0.0) -> Solution:
    h = np.where(h < 1e-9 * max(demand, 1.0), 0.0, h)
    x = net.edge_flows(h)
    total = net.total_time(x)               # always measured with the TRUE road times
    return Solution(net, demand, h, x, total, total / demand if demand > 0 else 0.0,
                    gap, it, toll_revenue)


def equilibrium(net: Network, demand: float) -> Solution:
    """What actually happens when every driver chooses selfishly."""
    if demand <= 0:
        return _package(net, 0.0, np.zeros(len(net.paths)), 0.0, 0)
    h, gap, it = _minimize(net.incidence, net.a, net.b, demand)
    return _package(net, demand, h, gap, it)


def optimum(net: Network, demand: float) -> Solution:
    """The assignment that minimises total travel time (what a planner would choose)."""
    if demand <= 0:
        return _package(net, 0.0, np.zeros(len(net.paths)), 0.0, 0)
    h, gap, it = _minimize(net.incidence, 2.0 * net.a, net.b, demand)
    return _package(net, demand, h, gap, it)


def price_of_anarchy(net: Network, demand: float) -> float:
    """Total time at equilibrium divided by total time at the optimum (>= 1)."""
    return equilibrium(net, demand).total_time / optimum(net, demand).total_time


def marginal_cost_tolls(net: Network, demand: float) -> np.ndarray:
    """Toll per edge that makes the selfish equilibrium coincide with the optimum.

    At the optimum, one extra driver on edge e adds  a_e * x_e  to the time of the
    drivers already there. Charging exactly that amount (the 'externality') makes
    each driver pay the true cost of their choice. This is a Pigouvian toll.
    """
    opt = optimum(net, demand)
    return net.a * opt.edge_flows


def equilibrium_with_tolls(net: Network, demand: float, tolls: np.ndarray) -> Solution:
    """Selfish equilibrium when drivers also have to pay `tolls` (in time units).

    Drivers choose using time + toll. The returned times are the true travel
    times (tolls are a transfer, not wasted time); revenue is reported separately.
    """
    tolled = net.with_extra_cost(tolls)
    if demand <= 0:
        return _package(net, 0.0, np.zeros(len(net.paths)), 0.0, 0)
    h, gap, it = _minimize(tolled.incidence, tolled.a, tolled.b, demand)
    x = net.edge_flows(h)
    revenue = float(np.sum(np.asarray(tolls) * x))
    return _package(net, demand, h, gap, it, toll_revenue=revenue)
