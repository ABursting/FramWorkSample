import streamlit as st
import pandas as pd

from cfModel import run_solar_model, PROJECT_YEARS

st.set_page_config(page_title="Solar Project Model", layout="wide")

st.title("Solar Project Financial Model")
st.caption("25 year cash flows, IRR, and payback period.")

# Load states for drodpown
@st.cache_data
def load_states() -> list[str]:
    df = pd.read_csv("energy_prices.csv")
    if "state" not in df.columns:
        raise ValueError("energy_prices.csv must include a 'state' column")
    # duplicates
    states = sorted(df["state"].astype(str).unique().tolist())
    return states

states = load_states()

# Sidebar inputs 
st.sidebar.header("Inputs")
state = st.sidebar.selectbox("State", states, index=states.index("California") if "California" in states else 0)
system_size_kw = st.sidebar.number_input("System size (kW DC)", min_value=0.1, max_value=1000.0, value=10.0, step=0.5)

years = st.sidebar.slider("Project horizon (years)", min_value=5, max_value=PROJECT_YEARS, value=PROJECT_YEARS, step=1)

# run the model 
try:
    results = run_solar_model(state=state, system_size_kw=system_size_kw, years=years)
    table = results["cashflow_table"]

    # Headline metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Upfront cost", f"${results['upfront_cost']:,.2f}")
    col2.metric("Annual generation", f"{results['annual_generation_kwh']:,.0f} kWh")

    irr = results["irr"]
    if irr is None:
        col3.metric("Project IRR", "N/A")
    else:
        col3.metric("Project IRR", f"{irr*100:.2f}%")

    payback = results["payback_years"]
    col4.metric("Payback period", "N/A" if payback is None else f"{payback} years")

    st.divider()

    # Table 
    st.subheader("25-year cash flow table")
    st.dataframe(table, use_container_width=True)

    # Quick chart 
    st.subheader("Cumulative cash flow over time")
    chart_df = table[["year", "cumulative_cashflow"]].copy()
    st.line_chart(chart_df, x="year", y="cumulative_cashflow")

except Exception as e:
    st.error(f"Error running model: {e}")
