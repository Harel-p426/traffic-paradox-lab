"""Road-network model.

A network is a directed graph with one origin (source) and one destination
(target). Every road (edge) has a travel time that depends on how many
drivers use it:

    time(x) = a * x + b

    b = free-flow time (what you pay even on an empty road)
    a = congestion slope (extra time per additional driver)

    a = 0           -> a road that never gets congested (e.g. a long highway)
    b = 0, a > 0    -> a narrow road: fast when empty, slow when crowded
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Edge:
    u: str            # where the road starts
    v: str            # where the road ends
    a: float = 0.0    # congestion slope
    b: float = 0.0    # free-flow time

    def time(self, x: float) -> float:
        return self.a * x + self.b


class Network:
    """A directed network with a single origin-destination pair."""

    def __init__(self, edges, source: str, target: str):
        self.edges = tuple(edges)
        keys = [(e.u, e.v) for e in self.edges]
        if len(set(keys)) != len(keys):
            raise ValueError("parallel edges (same u->v twice) are not supported")
        self.source = source
        self.target = target

        # Vectors of coefficients, one entry per edge (same order as self.edges).
        self.a = np.array([e.a for e in self.edges], dtype=float)
        self.b = np.array([e.b for e in self.edges], dtype=float)

        # All simple paths (no repeated nodes) from source to target.
        self.paths = self._enumerate_paths()
        if not self.paths:
            raise ValueError(f"no path from {source} to {target}")

        # incidence[r, e] = 1 if path r uses edge e, else 0.
        # With path flows h (vector), the flow on each edge is  incidence.T @ h
        # and the time of each path is  incidence @ (edge times).
        self.incidence = np.zeros((len(self.paths), len(self.edges)))
        for r, path in enumerate(self.paths):
            self.incidence[r, path] = 1.0

    # ------------------------------------------------------------------ paths
    def _enumerate_paths(self) -> list[list[int]]:
        outgoing: dict[str, list[tuple[str, int]]] = {}
        for i, e in enumerate(self.edges):
            outgoing.setdefault(e.u, []).append((e.v, i))

        found: list[list[int]] = []

        def dfs(node: str, visited: set[str], path: list[int]) -> None:
            if node == self.target:
                found.append(list(path))
                return
            for nxt, i in outgoing.get(node, []):
                if nxt not in visited:
                    visited.add(nxt)
                    path.append(i)
                    dfs(nxt, visited, path)
                    path.pop()
                    visited.remove(nxt)

        dfs(self.source, {self.source}, [])
        return found

    def path_label(self, r: int) -> str:
        path = self.paths[r]
        nodes = [self.edges[path[0]].u] + [self.edges[i].v for i in path]
        return "→".join(nodes)

    @property
    def path_labels(self) -> list[str]:
        return [self.path_label(r) for r in range(len(self.paths))]

    # --------------------------------------------------------------- helpers
    def edge_flows(self, path_flows: np.ndarray) -> np.ndarray:
        """Flow on every edge, given the flow on every path."""
        return self.incidence.T @ path_flows

    def edge_times(self, edge_flows: np.ndarray) -> np.ndarray:
        return self.a * edge_flows + self.b

    def total_time(self, edge_flows: np.ndarray) -> float:
        """Sum over all drivers of their travel time (driver-minutes)."""
        return float(np.sum(edge_flows * self.edge_times(edge_flows)))

    # ------------------------------------------------------ build variations
    def with_edge(self, edge: Edge) -> "Network":
        """The same network plus one extra road."""
        return Network(self.edges + (edge,), self.source, self.target)

    def with_extra_cost(self, extra: np.ndarray) -> "Network":
        """The same network with a fixed extra cost (toll) added on each edge."""
        new_edges = [Edge(e.u, e.v, e.a, e.b + float(t)) for e, t in zip(self.edges, extra)]
        return Network(new_edges, self.source, self.target)
