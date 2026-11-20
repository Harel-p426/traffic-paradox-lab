"""Tests for the information-design part (lottery, mixed knowledge, guided drivers).

The derivation is in the README (Results, item 5).  The key formula is the average
travel time when a share q of the drivers knows about the shortcut and everybody is selfish:

    avg(q) = 65 - 5 q + 20 q^2        (it is smallest at q = 1/8, where it equals the optimum 64.6875)
"""

import numpy as np
import pytest

from trafficlab import (
    BRAESS_DEMAND, BRAESS_SHORTCUT, Edge, Network, braess,
    equilibrium, equilibrium_by_knowledge, evaluate_guidance, guided_outcome,
    lottery, optimum, path_times, paths_using,
)
from tests.test_braess import _random_network

D = BRAESS_DEMAND
H = BRAESS_SHORTCUT
NET = braess(True)
SHORT = "S→A→B→T"


def _by_label(net, values):
    return dict(zip(net.path_labels, values))


# -------------------------------------------------------------------- lottery
def test_lottery_probabilities_and_times():
    rep = lottery(NET, D, H)
    shares, times = _by_label(NET, rep.shares), _by_label(NET, rep.times)
    assert shares["S→A→T"] == pytest.approx(0.4375, abs=1e-6)
    assert shares["S→B→T"] == pytest.approx(0.4375, abs=1e-6)
    assert shares[SHORT] == pytest.approx(0.125, abs=1e-6)
    assert times["S→A→T"] == pytest.approx(67.5, abs=1e-4)
    assert times["S→B→T"] == pytest.approx(67.5, abs=1e-4)
    assert times[SHORT] == pytest.approx(45.0, abs=1e-4)
    assert rep.avg_time == pytest.approx(64.6875, abs=1e-6)


def test_lottery_expected_time_is_the_weighted_average_of_path_times():
    rep = lottery(NET, D, H)
    assert float(rep.shares @ rep.times) == pytest.approx(rep.avg_time, abs=1e-6)


def test_lottery_is_unfair_to_most_drivers_but_fair_in_expectation():
    """Most drivers (87.5%) are told a route that takes MORE than the 65 minutes without the road."""
    rep = lottery(NET, D, H)
    times = _by_label(NET, rep.times)
    assert times["S→A→T"] > 65 and times["S→B→T"] > 65
    assert rep.avg_time < 65


def test_drivers_obey_only_if_they_do_not_know_the_shortcut():
    rep = lottery(NET, D, H)
    informed = _by_label(NET, rep.gain_if_informed)
    uninformed = _by_label(NET, rep.gain_if_uninformed)
    assert informed["S→A→T"] == pytest.approx(22.5, abs=1e-3)       # would switch and save 22.5 minutes
    assert informed["S→B→T"] == pytest.approx(22.5, abs=1e-3)
    assert informed[SHORT] == pytest.approx(0.0, abs=1e-3)
    assert all(v <= 1e-3 for v in uninformed.values())              # nothing they know is better
    obeys_informed = _by_label(NET, rep.obedient_if_informed)
    assert obeys_informed[SHORT] and not obeys_informed["S→A→T"] and not obeys_informed["S→B→T"]
    assert rep.obedient_if_uninformed.all()


# ------------------------------------------------ selfish drivers, mixed knowledge
@pytest.mark.parametrize("q", [0.0, 0.05, 0.1, 0.125, 0.25, 0.5, 0.75, 1.0])
def test_mixed_knowledge_matches_the_closed_form(q):
    out = equilibrium_by_knowledge(NET, D, q, H)
    assert out.avg_time == pytest.approx(65 - 5 * q + 20 * q * q, abs=1e-4)


def test_endpoints_match_the_basic_model():
    assert equilibrium_by_knowledge(NET, D, 1.0, H).avg_time == pytest.approx(equilibrium(NET, D).avg_time)
    assert equilibrium_by_knowledge(NET, D, 0.0, H).avg_time == pytest.approx(
        equilibrium(braess(False), D).avg_time)


def test_one_eighth_informed_drivers_is_exactly_optimal():
    """With 500 of 4,000 drivers on the shortcut, selfish choices already give the optimum."""
    out = equilibrium_by_knowledge(NET, D, 0.125, H)
    assert out.avg_time == pytest.approx(optimum(NET, D).avg_time, abs=1e-4)
    qs = np.linspace(0, 1, 81)
    values = [equilibrium_by_knowledge(NET, D, q, H).avg_time for q in qs]
    assert qs[int(np.argmin(values))] == pytest.approx(0.125)


def test_more_than_a_quarter_informed_is_worse_than_nobody_knowing():
    assert equilibrium_by_knowledge(NET, D, 0.2, H).avg_time < 65 < equilibrium_by_knowledge(NET, D, 0.3, H).avg_time


def test_knowledge_flows_are_consistent():
    out = equilibrium_by_knowledge(NET, D, 0.4, H)
    assert out.informed_flows.sum() == pytest.approx(0.4 * D)
    assert out.uninformed_flows.sum() == pytest.approx(0.6 * D)
    uses = paths_using(NET, H)
    assert out.uninformed_flows[uses].sum() == pytest.approx(0.0)     # they never use a road they do not know
    x = NET.edge_flows(out.informed_flows + out.uninformed_flows)
    times = path_times(NET, x)
    used_i = out.informed_flows > 1e-6
    used_u = out.uninformed_flows > 1e-6
    assert times[used_i].max() <= times.min() + 1e-4                   # informed: no path is faster
    assert times[used_u].max() <= times[~uses].min() + 1e-4           # uninformed: no known path is faster


def test_multi_class_solver_on_random_networks():
    """q = 0 must equal the equilibrium of the network without the hidden road; q = 1 the full one."""
    rng = np.random.default_rng(7)
    checked = 0
    while checked < 25:
        net = _random_network(rng)
        demand = rng.uniform(0.5, 3)
        hidden = net.edges[int(rng.integers(len(net.edges)))]
        key = (hidden.u, hidden.v)
        keep = [e for e in net.edges if (e.u, e.v) != key]
        try:
            smaller = Network(keep, net.source, net.target)
        except ValueError:
            continue                          # removing the road disconnects the network
        checked += 1
        assert equilibrium_by_knowledge(net, demand, 0.0, key).avg_time == pytest.approx(
            equilibrium(smaller, demand).avg_time, rel=1e-6, abs=1e-6)
        assert equilibrium_by_knowledge(net, demand, 1.0, key).avg_time == pytest.approx(
            equilibrium(net, demand).avg_time, rel=1e-6, abs=1e-6)
        mixed = equilibrium_by_knowledge(net, demand, 0.5, key)
        x = net.edge_flows(mixed.informed_flows + mixed.uninformed_flows)
        times = path_times(net, x)
        uses = paths_using(net, key)
        assert times[mixed.informed_flows > 1e-6].max() <= times.min() + 1e-4
        assert times[mixed.uninformed_flows > 1e-6].max() <= times[~uses].min() + 1e-4


# ------------------------------------------------------------ app guides drivers
@pytest.mark.parametrize("q", [0.0, 0.05, 0.1, 0.125])
def test_app_reaches_the_optimum_when_at_most_one_eighth_know_the_shortcut(q):
    out = guided_outcome(NET, D, q)
    assert out.avg_time == pytest.approx(64.6875, abs=1e-4)


def test_app_has_no_power_when_everybody_knows():
    assert guided_outcome(NET, D, 1.0).avg_time == pytest.approx(80.0)


def test_app_never_beats_the_optimum_and_is_never_worse_than_selfish():
    opt = optimum(NET, D).avg_time
    for q in np.linspace(0, 1, 11):
        guided = guided_outcome(NET, D, q).avg_time
        selfish = equilibrium_by_knowledge(NET, D, q, H).avg_time
        assert opt - 1e-6 <= guided <= selfish + 1e-6


def test_guided_assignment_is_not_beaten_by_random_assignments():
    """Independent check of the search: no random assignment of the guided drivers does better."""
    rng = np.random.default_rng(3)
    for q in (0.05, 0.3, 0.6):
        best = guided_outcome(NET, D, q)
        guided_demand = (1 - q) * D
        for _ in range(1500):
            g = guided_demand * rng.dirichlet(np.ones(len(NET.paths)) * rng.choice([0.3, 1, 3]))
            assert evaluate_guidance(NET, D, q, g).avg_time >= best.avg_time - 1e-6


def test_guided_flows_add_up():
    out = guided_outcome(NET, D, 0.1)
    assert out.guided_flows.sum() == pytest.approx(0.9 * D)
    assert out.informed_flows.sum() == pytest.approx(0.1 * D)


# ----------------------------------------------------------------- bad input
def test_bad_input_is_rejected():
    with pytest.raises(ValueError):
        equilibrium_by_knowledge(NET, D, 1.5, H)
    with pytest.raises(ValueError):
        equilibrium_by_knowledge(NET, D, 0.5, ("X", "Y"))         # no such road
    single_road = Network([Edge("S", "T", a=0.01, b=1.0)], "S", "T")
    with pytest.raises(ValueError):
        lottery(single_road, 100.0, ("S", "T"))                    # every path uses the hidden road
