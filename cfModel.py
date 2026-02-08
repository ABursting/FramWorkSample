# model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import pandas as pd
import numpy_financial as npf


# Configuration 

_ELECTRICITY_PRICE_DF = pd.read_csv("energy_prices.csv")
_ELECTRICITY_PRICE_DF["state"] = _ELECTRICITY_PRICE_DF["state"].astype(str).str.lower()
PROJECT_YEARS: int = 25
CAPEX_PER_WATT: float = 2.50        
ITC_RATE: float = 0.30               
GEN_KWH_PER_KW_YEAR: float = 1400.0  
PRICE_ESCALATION: float = 0.025      
OM_PER_KW_YEAR: float = 15.0         


# Data 

def get_electricity_price(state: str) -> float:
    """
    Returns electricity price ($/kWh) for a given state
    """
    required_cols = {"state", "cost_per_kWh"}
    if not required_cols.issubset(_ELECTRICITY_PRICE_DF.columns):
        raise ValueError("CSV must contain 'state' and 'cost_per_kWh' columns")

    state_clean = state.lower()
    match = _ELECTRICITY_PRICE_DF.loc[_ELECTRICITY_PRICE_DF["state"] == state_clean, "cost_per_kWh"]

    if match.empty:
        raise ValueError(f"State '{state}' not found in price table")

    return float(match.iloc[0])

# Core calculations

def calc_upfront_capex(system_size_kw: float, capex_per_watt: float = CAPEX_PER_WATT) -> float:
    """Capex in $ given kW system size and $/W installed cost."""

    if capex_per_watt <= 0:
        raise ValueError("capex_per_watt must be > 0.")
    watts = system_size_kw * 1000.0
    return capex_per_watt * watts


def calc_itc_value(capex: float, itc_rate: float = ITC_RATE) -> float:
    """ITC value in $"""
    if capex < 0:
        raise ValueError("capex must be >= 0.")
    if not (0 <= itc_rate <= 1):
        raise ValueError("itc_rate must be between 0 and 1.")
    return capex * itc_rate


def calc_annual_generation_kwh(system_size_kw: float, gen_kwh_per_kw_year: float = GEN_KWH_PER_KW_YEAR) -> float:
    """Annual generation in kWh."""
    if gen_kwh_per_kw_year <= 0:
        raise ValueError("gen_kwh_per_kw_year must be > 0.")
    return system_size_kw * gen_kwh_per_kw_year


def electricity_price_for_year(base_price: float, year: int, escalation: float = PRICE_ESCALATION) -> float:
    """
    Price in $/kWh for a give year
      - Year t uses base_price * (1+escalation)^(t-1)
    """
    if base_price <= 0:
        raise ValueError("base_price must be > 0.")
    if year < 1:
        raise ValueError("year must be >= 1 for price escalation calculation.")
    if escalation < 0:
        raise ValueError("escalation must be >= 0.")
    return base_price * ((1.0 + escalation) ** (year - 1))


def annual_om_cost(system_size_kw: float, om_per_kw_year: float = OM_PER_KW_YEAR) -> float:
    """Annual O&M cost in $."""
    if om_per_kw_year < 0:
        raise ValueError("om_per_kw_year must be >= 0.")
    return system_size_kw * om_per_kw_year


# Cash flow schedule and metrics

def build_cashflow_schedule(
    system_size_kw: float,
    base_price_per_kwh: float,
    years: int = PROJECT_YEARS,
    capex_per_watt: float = CAPEX_PER_WATT,
    itc_rate: float = ITC_RATE,
    gen_kwh_per_kw_year: float = GEN_KWH_PER_KW_YEAR,
    escalation: float = PRICE_ESCALATION,
    om_per_kw_year: float = OM_PER_KW_YEAR,
) -> Tuple[pd.DataFrame, list[float]]:
    """
    Returns:
      - cashflow_table: DataFrame with rows 
      - cashflows: list of cash flow value for IRR calculation
    """
    if years <= 0:
        raise ValueError("years must be > 0.")
    if base_price_per_kwh <= 0:
        raise ValueError("base_price_per_kwh must be > 0.")

    capex = calc_upfront_capex(system_size_kw, capex_per_watt)
    itc = calc_itc_value(capex, itc_rate)
    annual_gen = calc_annual_generation_kwh(system_size_kw, gen_kwh_per_kw_year)
    om = annual_om_cost(system_size_kw, om_per_kw_year)

    rows: list[Dict[str, Any]] = []
    cashflows: list[float] = []

    cumulative = 0.0

    # Year 0
    cf0 = -capex + itc
    cumulative += cf0
    rows.append({
        "year": 0,
        "price_per_kwh": None,
        "generation_kwh": 0.0,
        "om_cost": 0.0,
        "net_cashflow": cf0,
        "cumulative_cashflow": cumulative,
    })
    cashflows.append(cf0)

    # Years 1+
    for yr in range(1, years + 1):
        price = electricity_price_for_year(base_price_per_kwh, yr, escalation)
        gross_savings = annual_gen * price
        net_cf = gross_savings - om
        cumulative += net_cf

        rows.append({
            "year": yr,
            "price_per_kwh": price,
            "generation_kwh": annual_gen,
            "om_cost": om,
            "net_cashflow": net_cf,
            "cumulative_cashflow": cumulative,
        })
        cashflows.append(net_cf)

    table = pd.DataFrame(rows)
    return table, cashflows


def calc_project_irr(cashflows: list[float]) -> Optional[float]:
    """
    Returns IRR as a decimal 
    """
    if not cashflows or len(cashflows) < 2:
        return None

    irr = npf.irr(cashflows)

    try:
        irr = float(irr)
    except Exception:
        return None

    if irr != irr:  # NaN check
        return None

    return irr


def calc_payback_period_years(cashflow_table: pd.DataFrame) -> Optional[int]:
    """
    Returns the first year where cumulative cash flow >= 0
    Returns None if never pays back within the horizon.
    """

    cum = cashflow_table["cumulative_cashflow"].tolist()
    yrs = cashflow_table["year"].tolist()

    if cum[0] >= 0:
        return int(yrs[0])

    # Find first non-negative cumulative cashflow
    for i in range(1, len(cum)):
        if cum[i] >= 0:
            return int(yrs[i])

    return None


# Function for frontend  

def run_solar_model(
    state: str,
    system_size_kw: float,
    years: int = PROJECT_YEARS,
) -> Dict[str, Any]:
    """
    Returns:
      - cashflow_table (25-year table)
      - upfront_cost
      - annual_generation_kwh
      - irr (decimal form)
      - payback_years
    """
    base_price = get_electricity_price(state) 
    upfront = calc_upfront_capex(system_size_kw)
    annual_gen = calc_annual_generation_kwh(system_size_kw)

    table, cashflows = build_cashflow_schedule(
        system_size_kw=system_size_kw,
        base_price_per_kwh=base_price,
        years=years,
    )

    irr = calc_project_irr(cashflows)
    payback = calc_payback_period_years(table)

    display_table = table.copy()

    display_table["price_per_kwh"] = display_table["price_per_kwh"].round(4)
    display_table["net_cashflow"] = display_table["net_cashflow"].round(2)
    display_table["cumulative_cashflow"] = display_table["cumulative_cashflow"].round(2)

    return {
        "cashflow_table": display_table,
        "upfront_cost": float(upfront),
        "annual_generation_kwh": float(annual_gen),
        "irr": irr,
        "payback_years": payback,
    }

