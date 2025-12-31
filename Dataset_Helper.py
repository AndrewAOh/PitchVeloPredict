# Load the datasets for rankings and reviews

import streamlit as st
import pandas as pd
from scipy import stats
import sklearn


# ----------------------------
# File paths to the datasets
# ----------------------------
PITCHER_METRIC_PERCENTILES_FP = "Dataset/NCAA_Pitcher_Metric_Percentiles.csv"


# ----------------------------
# Function: load metric percentiles dataset to compare velos
# ----------------------------
@st.cache_data
def load_metric_percentiles_dataset():
    pitcher_metric_percentiles_dataset = pd.read_csv(PITCHER_METRIC_PERCENTILES_FP)
    return pitcher_metric_percentiles_dataset


# ----------------------------
# Function: process metric percentiles dataset to only contain velo percentiles
# ----------------------------
@st.cache_data
def preprocesss_metric_percentiles_dataset(pitcher_metric_percentiles_dataset):
    # 1) Keep only rows related to Power 4 Conferences
    pitcher_velo_percentiles_dataset = pitcher_metric_percentiles_dataset[
        pitcher_metric_percentiles_dataset["Dataset"] == "POWER 4"
    ]

    # 2) Keep only desired final columns
    columns_to_keep = [
        'TaggedPitchType', 'PitcherThrows',
        'RelSpeed_Mean', 'RelSpeed_Std',
    ]
    pitcher_velo_percentiles_dataset = pitcher_velo_percentiles_dataset[columns_to_keep]

    # 3) Return final dataset
    return pitcher_velo_percentiles_dataset


# ----------------------------
# Function: calculate a pitcher's percentile in terms of velo
# ----------------------------
@st.cache_data
def calculate_velo_percentile(velo, PITCHER_VELO_PERCENTILES_DS):
    # Get Pitch Type, Throw Hand from session data
    pitch_type = st.session_state.get("pitch_type", 0)
    throw_hand = st.session_state.get("pitcher_throw", 0)
    
    match = PITCHER_VELO_PERCENTILES_DS[
        (PITCHER_VELO_PERCENTILES_DS['TaggedPitchType'] == pitch_type) &
        (PITCHER_VELO_PERCENTILES_DS['PitcherThrows'] == throw_hand)
    ]

    if not match.empty:
        mean_val = match['RelSpeed_Mean'].mean()
        std_val = match['RelSpeed_Std'].mean()

        if pd.notnull(std_val) and std_val > 0:
            z_score = (velo - mean_val) / std_val
            pct = stats.norm.cdf(z_score) * 100
        else:
            pct = None
    else:
        pct = None

    return round(pct, 1)