import streamlit as st
import pandas as pd

# ----------------------------
# Reset app variables
# ----------------------------
def reset_app():
    """Reset all session state variables"""
    st.session_state.step = 'pitcher_info'
    st.session_state.target_year = None
    st.session_state.pitcher_height = None
    st.session_state.pitcher_throw = None
    st.session_state.pitch_type = None
    st.session_state.available_data = {}
    st.session_state.pitcher_data = {}
    st.session_state.predictions = None