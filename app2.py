import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(layout="wide")
st.title("AI-Driven Inventory Decision Lab")

tab1, tab2, tab3 = st.tabs([
    "Simulation",
    "Cost vs Service Level",
    "Scenario Comparison"
])

# Common Inputs
st.sidebar.header("Input Parameters")

mean_demand = st.sidebar.number_input("Mean Daily Demand", value=100)
std_demand = st.sidebar.number_input("Demand Std Dev", value=20)
lead_time = st.sidebar.number_input("Lead Time (Days)", value=5)
holding_cost = st.sidebar.number_input("Holding Cost per Unit", value=2.0)
stockout_cost = st.sidebar.number_input("Stockout Cost per Unit", value=10.0)

service_level = st.sidebar.slider("Service Level", 0.80, 0.99, 0.95)

mean_lt = mean_demand * lead_time
std_lt = std_demand * np.sqrt(lead_time)
z_value = norm.ppf(service_level)

safety_stock = z_value * std_lt
reorder_point = mean_lt + safety_stock

# Monte Carlo Simulation
num_sim = 10000
sim_demand = np.random.normal(mean_lt, std_lt, num_sim)

stockouts = sim_demand > reorder_point
stockout_prob = np.mean(stockouts)

avg_inventory = safety_stock / 2
expected_stockout_units = np.mean(
    np.maximum(sim_demand - reorder_point, 0)
)

total_cost = (
    holding_cost * avg_inventory +
    stockout_cost * expected_stockout_units
)

# ---------------- TAB 1 ---------------- #
with tab1:

    st.header("Probabilistic Inventory Model")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Safety Stock", round(safety_stock,2))
        st.metric("Reorder Point", round(reorder_point,2))
        st.metric("Stockout Probability (%)", round(stockout_prob*100,2))
        st.metric("Expected Total Cost", round(total_cost,2))

    with col2:
        fig, ax = plt.subplots()
        ax.hist(sim_demand, bins=50)
        ax.axvline(reorder_point)
        ax.set_title("Lead Time Demand Distribution")
        st.pyplot(fig)

    # Deterministic comparison
    st.subheader("Deterministic Comparison")

    deterministic_rop = mean_lt
    det_stockouts = np.mean(sim_demand > deterministic_rop)

    st.write(f"Deterministic ROP: {round(deterministic_rop,2)}")
    st.write(f"Stockout Probability (Deterministic): {round(det_stockouts*100,2)} %")

# ---------------- TAB 2 ---------------- #
with tab2:

    st.header("Cost vs Service Level Sensitivity")

    service_range = np.linspace(0.80, 0.99, 50)
    cost_list = []

    for sl in service_range:
        z = norm.ppf(sl)
        ss = z * std_lt
        rop = mean_lt + ss

        sim = np.random.normal(mean_lt, std_lt, num_sim)
        stock_units = np.mean(np.maximum(sim - rop, 0))

        cost = holding_cost * (ss/2) + stockout_cost * stock_units
        cost_list.append(cost)

    fig2, ax2 = plt.subplots()
    ax2.plot(service_range, cost_list)
    ax2.set_xlabel("Service Level")
    ax2.set_ylabel("Total Cost")
    ax2.set_title("Cost vs Service Level")

    st.pyplot(fig2)

# ---------------- TAB 3 ---------------- #
with tab3:

    st.header("Scenario Comparison")

    high_service = 0.98
    low_service = 0.85

    z_high = norm.ppf(high_service)
    z_low = norm.ppf(low_service)

    ss_high = z_high * std_lt
    ss_low = z_low * std_lt

    st.write("High Service Level (98%) Safety Stock:", round(ss_high,2))
    st.write("Low Service Level (85%) Safety Stock:", round(ss_low,2))

    st.write("Increase in Inventory Due to Higher Service:",
             round(ss_high - ss_low,2))
