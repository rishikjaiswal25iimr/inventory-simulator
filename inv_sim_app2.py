import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(layout="wide")
st.title("AI-Driven Inventory Decision Lab")

# ---------------- SIDEBAR INPUTS ---------------- #
st.sidebar.header("Base Parameters")

mean_demand = st.sidebar.number_input("Mean Daily Demand", value=100)
std_demand = st.sidebar.number_input("Demand Std Dev", value=20)
lead_time = st.sidebar.number_input("Lead Time (Days)", value=5)
holding_cost = st.sidebar.number_input("Holding Cost per Unit", value=2.0)
stockout_cost = st.sidebar.number_input("Stockout Cost per Unit", value=10.0)
ordering_cost = st.sidebar.number_input("Ordering Cost (for EOQ)", value=500.0)

service_level = st.sidebar.slider("Service Level", 0.80, 0.99, 0.95)

annual_demand = mean_demand * 365

tabs = st.tabs([
    "1️⃣ Simulation",
    "2️⃣ Cost vs Service",
    "3️⃣ Scenario Compare",
    "4️⃣ AI Forecast",
    "5️⃣ EOQ Model",
    "6️⃣ Auto Optimization",
    "7️⃣ Risk Heatmap"
])

# Common calculations
mean_lt = mean_demand * lead_time
std_lt = std_demand * np.sqrt(lead_time)
z_value = norm.ppf(service_level)

safety_stock = z_value * std_lt
reorder_point = mean_lt + safety_stock

num_sim = 10000
sim_demand = np.random.normal(mean_lt, std_lt, num_sim)

stockouts = sim_demand > reorder_point
stockout_prob = np.mean(stockouts)

avg_inventory = safety_stock / 2
expected_stockout_units = np.mean(np.maximum(sim_demand - reorder_point, 0))

total_cost = holding_cost * avg_inventory + stockout_cost * expected_stockout_units

# ---------------- TAB 1 ---------------- #
with tabs[0]:
    st.header("Probabilistic Simulation")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Safety Stock", round(safety_stock,2))
        st.metric("Reorder Point", round(reorder_point,2))
        st.metric("Stockout Probability (%)", round(stockout_prob*100,2))
        st.metric("Expected Cost", round(total_cost,2))

    with col2:
        fig, ax = plt.subplots()
        ax.hist(sim_demand, bins=50)
        ax.axvline(reorder_point)
        st.pyplot(fig)

# ---------------- TAB 2 ---------------- #
with tabs[1]:
    st.header("Cost vs Service Level")

    service_range = np.linspace(0.80, 0.99, 40)
    cost_list = []

    for sl in service_range:
        z = norm.ppf(sl)
        ss = z * std_lt
        rop = mean_lt + ss
        sim = np.random.normal(mean_lt, std_lt, num_sim)
        stock_units = np.mean(np.maximum(sim - rop, 0))
        cost = holding_cost*(ss/2) + stockout_cost*stock_units
        cost_list.append(cost)

    fig2, ax2 = plt.subplots()
    ax2.plot(service_range, cost_list)
    ax2.set_xlabel("Service Level")
    ax2.set_ylabel("Total Cost")
    st.pyplot(fig2)

# ---------------- TAB 3 ---------------- #
with tabs[2]:
    st.header("Scenario Comparison")

    high = 0.98
    low = 0.85

    ss_high = norm.ppf(high) * std_lt
    ss_low = norm.ppf(low) * std_lt

    st.write("High Service (98%) Safety Stock:", round(ss_high,2))
    st.write("Low Service (85%) Safety Stock:", round(ss_low,2))
    st.write("Inventory Increase:", round(ss_high - ss_low,2))

# ---------------- TAB 4 ---------------- #
with tabs[3]:
    st.header("AI Forecast Toggle (Trend-Based Demand)")

    trend_rate = st.slider("Annual Demand Growth (%)", 0, 20, 5)

    forecast_days = 365
    trend_factor = 1 + (trend_rate/100)
    future_demand = mean_demand * trend_factor

    forecast = np.random.normal(future_demand, std_demand, forecast_days)

    fig3, ax3 = plt.subplots()
    ax3.plot(forecast[:100])
    ax3.set_title("Forecasted Demand Trend (First 100 Days)")
    st.pyplot(fig3)

    st.write("New Projected Daily Demand:", round(future_demand,2))

# ---------------- TAB 5 ---------------- #
with tabs[4]:
    st.header("EOQ Model Integration")

    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)

    total_eoq_cost = (
        (annual_demand / eoq) * ordering_cost +
        (eoq / 2) * holding_cost
    )

    st.metric("Economic Order Quantity (EOQ)", round(eoq,2))
    st.metric("Annual EOQ Cost", round(total_eoq_cost,2))

# ---------------- TAB 6 ---------------- #
with tabs[5]:
    st.header("Automatic Service-Level Optimization")

    service_range = np.linspace(0.80, 0.99, 60)
    best_cost = 999999
    best_service = 0

    for sl in service_range:
        z = norm.ppf(sl)
        ss = z * std_lt
        rop = mean_lt + ss
        sim = np.random.normal(mean_lt, std_lt, num_sim)
        stock_units = np.mean(np.maximum(sim - rop, 0))
        cost = holding_cost*(ss/2) + stockout_cost*stock_units

        if cost < best_cost:
            best_cost = cost
            best_service = sl

    st.metric("Optimal Service Level", round(best_service,3))
    st.metric("Minimum Cost", round(best_cost,2))

# ---------------- TAB 7 ---------------- #
with tabs[6]:
    st.header("Risk Heatmap")

    lt_range = np.linspace(1, 15, 10)
    std_range = np.linspace(10, 50, 10)

    risk_matrix = np.zeros((len(std_range), len(lt_range)))

    for i, sd in enumerate(std_range):
        for j, lt in enumerate(lt_range):
            mean_lt_temp = mean_demand * lt
            std_lt_temp = sd * np.sqrt(lt)
            z = norm.ppf(service_level)
            ss = z * std_lt_temp
            rop = mean_lt_temp + ss

            sim = np.random.normal(mean_lt_temp, std_lt_temp, num_sim)
            stock_units = np.mean(np.maximum(sim - rop, 0))

            cost = holding_cost*(ss/2) + stockout_cost*stock_units
            risk_matrix[i,j] = cost

    fig4, ax4 = plt.subplots()
    im = ax4.imshow(risk_matrix, aspect="auto", origin="lower")
    ax4.set_xlabel("Lead Time")
    ax4.set_ylabel("Demand Variability")
    fig4.colorbar(im)
    st.pyplot(fig4)
