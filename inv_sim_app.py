import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

st.title("AI-Supported Probabilistic Inventory Simulator")

st.markdown("### Input Parameters")

mean_demand = st.number_input("Mean Daily Demand", value=100)
std_demand = st.number_input("Demand Standard Deviation", value=20)
lead_time = st.number_input("Lead Time (Days)", value=5)
holding_cost = st.number_input("Holding Cost per Unit", value=2.0)
stockout_cost = st.number_input("Stockout Cost per Unit", value=10.0)
service_level = st.slider("Target Service Level", 0.80, 0.99, 0.95)

mean_lt = mean_demand * lead_time
std_lt = std_demand * np.sqrt(lead_time)

z_value = norm.ppf(service_level)

safety_stock = z_value * std_lt
reorder_point = mean_lt + safety_stock

st.markdown("### Monte Carlo Simulation")

num_simulations = 10000

simulated_demand = np.random.normal(mean_lt, std_lt, num_simulations)

stockouts = simulated_demand > reorder_point
stockout_probability = np.mean(stockouts)

average_inventory = safety_stock / 2
expected_stockout_units = np.mean(
    np.maximum(simulated_demand - reorder_point, 0)
)

total_cost = (
    holding_cost * average_inventory +
    stockout_cost * expected_stockout_units
)

st.markdown("### Results")

st.write(f"Safety Stock: {round(safety_stock,2)} units")
st.write(f"Reorder Point: {round(reorder_point,2)} units")
st.write(f"Stockout Probability: {round(stockout_probability*100,2)} %")
st.write(f"Expected Total Cost: ₹ {round(total_cost,2)}")

st.markdown("### Demand Distribution")

fig, ax = plt.subplots()
ax.hist(simulated_demand, bins=50)
ax.axvline(reorder_point)
ax.set_title("Lead Time Demand Distribution")
ax.set_xlabel("Demand")
ax.set_ylabel("Frequency")

st.pyplot(fig)
