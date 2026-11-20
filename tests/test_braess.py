"""Tests: every expected number here can also be derived by hand (see docs/)."""

import numpy as np
import pytest

from trafficlab import (
    BRAESS_DEMAND, Edge, Network, braess, braess_normalized,
    equilibrium, equilibrium_with_tolls, marginal_cost_tolls, optimum, price_of_anarchy,
)

D = BRAESS_DEMAND


# ---------------------------------------------------------------- the paradox
def test_without_shortcut_everyone_needs_65_minutes():
    sol = equilibrium(braess(shortcut=False), D)
    assert sol.avg_time == pytest.approx(65.0)
    assert sorted(sol.path_flows) == pytest.approx([2000.0, 2000.0])


def test_with_shortcut_everyone_needs_80_minutes():
    """The paradox: adding a road makes every driver slower."""
    net = braess(shortcut=True)
    sol = equilibrium(net, D)
    assert sol.avg_time == pytest.approx(80.0)
    assert sol.path_flows[net.path_labels.index("S→A→B→T")] == pytest.approx(D)


# -------------------------------------------------------------------- optimum
def test_optimum_uses_the_shortcut_a_little():
    net = braess(shortcut=True)
    opt = optimum(net, D)
    assert opt.avg_time == pytest.approx(64.6875, abs=1e-4)
    flows = dict(zip(net.path_labels, opt.path_flows))
    assert flows["S→A→T"] == pytest.approx(1750.0, abs=0.5)
    assert flows["S→B→T"] == pytest.approx(1750.0, abs=0.5)
    assert flows["S→A→B→T"] == pytest.approx(500.0, abs=0.5)


def test_price_of_anarchy_of_the_classic_example():
    assert price_of_anarchy(braess(), D) == pytest.approx(80.0 / 64.6875, rel=1e-5)


def test_normalized_network_hits_the_four_thirds_bound():
    net = braess_normalized()
    assert equilibrium(net, 1.0).total_time == pytest.approx(2.0)
    assert optimum(net, 1.0).total_time == pytest.approx(1.5, abs=1e-6)
    assert price_of_anarchy(net, 1.0) == pytest.approx(4 / 3, rel=1e-6)


def test_pigou_example():
    """Two roads, times x and 1, one unit of traffic: equilibrium 1, optimum 0.75."""
    net = Network([
        Edge("S", "A", a=1.0), Edge("A", "T"),     # congestible road
        Edge("S", "B", b=1.0), Edge("B", "T"),     # constant road
    ], "S", "T")
    assert equilibrium(net, 1.0).avg_time == pytest.approx(1.0)
    assert optimum(net, 1.0).avg_time == pytest.approx(0.75, abs=1e-6)


# ---------------------------------------------------------------------- tolls
def test_marginal_cost_tolls_are_22_5_on_the_two_narrow_roads():
    net = braess()
    tolls = dict(zip([(e.u, e.v) for e in net.edges], marginal_cost_tolls(net, D)))
    assert tolls[("S", "A")] == pytest.approx(22.5, abs=1e-3)
    assert tolls[("B", "T")] == pytest.approx(22.5, abs=1e-3)
    assert tolls[("A", "T")] == pytest.approx(0.0)
    assert tolls[("A", "B")] == pytest.approx(0.0)


def test_tolls_restore_the_optimum():
    net = braess()
    tolls = marginal_cost_tolls(net, D)
    fixed = equilibrium_with_tolls(net, D, tolls)
    assert fixed.avg_time == pytest.approx(optimum(net, D).avg_time, abs=1e-3)
    assert fixed.avg_time < equilibrium(net, D).avg_time


# --------------------------------------------------- general mathematical facts
def _random_network(rng, n=6):
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
        except ValueError:
            continue


def test_equilibrium_condition_on_random_networks():
    """All used paths take the same time and no unused path is faster (Wardrop)."""
    rng = np.random.default_rng(0)
    for _ in range(30):
        net = _random_network(rng)
        demand = rng.uniform(0.5, 3)
        sol = equilibrium(net, demand)
        times = net.incidence @ net.edge_times(sol.edge_flows)
        used = sol.path_flows > 1e-6
        assert sol.path_flows.sum() == pytest.approx(demand)
        assert times[used].max() - times.min() < 1e-4


def test_optimum_never_worse_than_equilibrium():
    rng = np.random.default_rng(1)
    for _ in range(30):
        net = _random_network(rng)
        demand = rng.uniform(0.5, 3)
        assert optimum(net, demand).total_time <= equilibrium(net, demand).total_time + 1e-6


def test_poa_never_exceeds_four_thirds_for_linear_times():
    rng = np.random.default_rng(2)
    for _ in range(30):
        net = _random_network(rng)
        assert price_of_anarchy(net, rng.uniform(0.5, 3)) <= 4 / 3 + 1e-6


def test_adding_a_road_never_hurts_the_optimum():
    """A planner can always ignore a new road, so the optimum cannot get worse."""
    assert optimum(braess(True), D).total_time <= optimum(braess(False), D).total_time + 1e-6


# ------------------------------------------------------- range of the paradox
@pytest.mark.parametrize("demand,expected", [
    (2000, "helps"),       # little traffic: the shortcut saves time
    (3500, "hurts"),
    (4000, "hurts"),
    (6000, "hurts"),
    (8500, "hurts"),
    (10000, "neutral"),    # heavy traffic: nobody uses the shortcut
])
def test_when_the_shortcut_hurts(demand, expected):
    before = equilibrium(braess(False), demand).avg_time
    after = equilibrium(braess(True), demand).avg_time
    if expected == "helps":
        assert after < before - 1e-6
    elif expected == "hurts":
        assert after > before + 1e-6
    else:
        assert after == pytest.approx(before, abs=1e-3)
