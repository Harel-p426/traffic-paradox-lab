"""Interactive demo.   Run:  streamlit run app.py"""

import pandas as pd
import streamlit as st

from trafficlab import (BRAESS_SHORTCUT, braess, equilibrium, equilibrium_by_knowledge,
                        equilibrium_with_tolls, guided_outcome, lottery, marginal_cost_tolls,
                        optimum, price_of_anarchy)

st.set_page_config(page_title="Traffic Paradox Lab", layout="wide")
st.title("Traffic Paradox Lab")
st.caption("Selfish drivers, one new road, and Braess's paradox. Move the sliders and watch "
           "who gets faster and who gets slower.")

# ---------------------------------------------------------------- controls
demand = st.sidebar.slider("Number of drivers", 100, 12000, 4000, step=100)
shortcut = st.sidebar.checkbox("Build the new road A→B (takes 0 minutes)", value=True)
tolls_on = st.sidebar.checkbox("Charge congestion tolls", value=False)

# ---------------------------------------------------------------- compute
net = braess(shortcut)
without_road = equilibrium(braess(False), demand)
selfish = equilibrium(net, demand)
best = optimum(net, demand)
tolls = marginal_cost_tolls(net, demand)
shown = equilibrium_with_tolls(net, demand, tolls) if tolls_on else selfish

# ---------------------------------------------------------------- headline numbers
c1, c2, c3 = st.columns(3)
c1.metric("Average travel time", f"{shown.avg_time:.1f} min",
          delta=f"{shown.avg_time - without_road.avg_time:+.1f} vs. no new road",
          delta_color="inverse")
c2.metric("Best possible (planner assigns routes)", f"{best.avg_time:.1f} min")
c3.metric("Price of anarchy", f"{price_of_anarchy(net, demand):.3f}",
          help="Total time with selfish drivers divided by total time at the optimum.")

if shortcut and not tolls_on and selfish.avg_time > without_road.avg_time + 1e-3:
    st.error(f"Braess's paradox: the new road makes EVERY driver "
             f"{selfish.avg_time - without_road.avg_time:.1f} minutes slower.")
elif shortcut and tolls_on:
    st.success("With congestion tolls, selfish drivers end up exactly at the optimum.")

# ---------------------------------------------------------------- where do drivers go
st.subheader("Where the drivers go")
routes = pd.DataFrame({"Route": net.path_labels, "Drivers": shown.path_flows.round(1)})
st.bar_chart(routes.set_index("Route"))

if tolls_on:
    st.subheader("Toll on each road (in minutes-equivalent)")
    st.dataframe(pd.DataFrame({
        "Road": [f"{e.u}→{e.v}" for e in net.edges],
        "Toll": tolls.round(2),
    }), hide_index=True)

# ---------------------------------------------------------------- information design
st.divider()
st.header("Information design: what if the app controls what drivers know?")
if not shortcut:
    st.info("Build the new road in the sidebar to see this section.")
else:
    st.caption("A share of the drivers knows about the new road and picks routes selfishly. "
               "The rest only know the old roads. Either they choose selfishly too, or an app "
               "assigns their routes as well as possible (they obey as long as they do not know "
               "of a better route).")
    share = st.slider("Share of drivers who know about the new road (%)", 0, 100, 10) / 100
    mixed = equilibrium_by_knowledge(net, demand, share, BRAESS_SHORTCUT)
    guided = guided_outcome(net, demand, share)

    d1, d2, d3 = st.columns(3)
    d1.metric("Everybody selfish", f"{mixed.avg_time:.2f} min")
    d2.metric("App assigns routes to drivers who do not know", f"{guided.avg_time:.2f} min",
              delta=f"{guided.avg_time - mixed.avg_time:+.2f} vs. selfish", delta_color="inverse")
    d3.metric("Optimum", f"{best.avg_time:.2f} min")

    st.subheader("The lottery that reaches the optimum")
    report = lottery(net, demand, BRAESS_SHORTCUT)
    rows = [r for r in range(len(report.labels)) if report.recommended[r]]
    st.dataframe(pd.DataFrame({
        "Told to take": [report.labels[r] for r in rows],
        "Share of drivers (%)": [round(100 * report.shares[r], 2) for r in rows],
        "Travel time (min)": [round(report.times[r], 2) for r in rows],
        "Obeys if he knows every road": ["yes" if report.obedient_if_informed[r] else
                                         f"no (would save {report.gain_if_informed[r]:.1f} min)" for r in rows],
        "Obeys if he knows only the old roads": ["yes" if report.obedient_if_uninformed[r] else "no"
                                                 for r in rows],
    }), hide_index=True)
    st.caption(f"Expected travel time of a random driver: {report.avg_time:.2f} min. "
               "The lottery is fair only on average: drivers who draw a slow route lose.")
