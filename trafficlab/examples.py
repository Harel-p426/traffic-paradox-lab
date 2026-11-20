"""Ready-made networks used in the tests, experiments and README."""

from __future__ import annotations

from .model import Edge, Network


def braess(shortcut: bool = True) -> Network:
    """The classic example: 4,000 drivers go from S to T.

        S -> A : x / 100 minutes   (narrow road, congests)
        A -> T : 45 minutes        (long highway, never congests)
        S -> B : 45 minutes        (long highway)
        B -> T : x / 100 minutes   (narrow road)
        A -> B : 0 minutes         (the new shortcut)
    """
    edges = [
        Edge("S", "A", a=0.01, b=0.0),
        Edge("A", "T", a=0.0, b=45.0),
        Edge("S", "B", a=0.0, b=45.0),
        Edge("B", "T", a=0.01, b=0.0),
    ]
    if shortcut:
        edges.append(Edge("A", "B", a=0.0, b=0.0))
    return Network(edges, "S", "T")


BRAESS_DEMAND = 4000.0
BRAESS_SHORTCUT = ("A", "B")     # the new road, which some drivers may not know about


def braess_normalized(shortcut: bool = True) -> Network:
    """Same shape with times x, 1, 1, x, 0 and a total flow of 1.

    Here the equilibrium time is 2 and the optimum is 1.5, so the Price of
    Anarchy is exactly 4/3, the known worst case for linear road times.
    """
    edges = [
        Edge("S", "A", a=1.0, b=0.0),
        Edge("A", "T", a=0.0, b=1.0),
        Edge("S", "B", a=0.0, b=1.0),
        Edge("B", "T", a=1.0, b=0.0),
    ]
    if shortcut:
        edges.append(Edge("A", "B", a=0.0, b=0.0))
    return Network(edges, "S", "T")
