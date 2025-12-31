import streamlit as st
import pandas as pd
import joblib
from scipy import stats


# Import functions from other files
import App_Helper
from App_Helper import (
    reset_app
)

import Model_Helper
from Model_Helper import (
    load_model, 
    load_models,
    determine_model_name,
    prepare_features,
    make_prediction,
    predict_future_years
)

import Dataset_Helper
from Dataset_Helper import (
    load_metric_percentiles_dataset, 
    preprocesss_metric_percentiles_dataset,
    calculate_velo_percentile
)



st.set_page_config(
    page_title="Pitcher Velocity Projector",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>

/* ---------- Layout ---------- */
.block-container {
    max-width: 1200px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}

.main {
    background-color: #f5f7fb;
}

/* ---------- Typography ---------- */
h1 {
    font-size: 2.3rem;
    font-weight: 700;
    color: #020617;
}

h2, h3 {
    color: #0f172a;
    font-weight: 600;
}

p, label {
    color: #475569;
    font-size: 0.95rem;
}

/* ---------- Section Card ---------- */
.section-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 2rem;
    margin: 1.8rem 0;
    border: 1px solid #e5e7eb;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
}

/* ---------- Buttons ---------- */
.stButton>button,
.stButton>button * {
    color: #ffffff !important;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    font-weight: 600;
    padding: 0.8rem 1rem;
    border-radius: 12px;
    border: none;
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.25);
    transition: all 0.25s ease;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(37, 99, 235, 0.35);
}

/* ---------- Prediction Card ---------- */
.prediction-card {
    background: linear-gradient(135deg, #020617, #0f172a);
    padding: 2.2rem;
    border-radius: 18px;
    color: #f8fafc;
    text-align: center;
    margin: 2rem 0;
    box-shadow: 0 18px 40px rgba(2, 6, 23, 0.45);
}

.prediction-card h2 {
    font-size: 3rem;
    margin: 0;
    letter-spacing: -1px;
}

.prediction-card p {
    color: #cbd5f5
}
            
/* ---------- Future Prediction Card ---------- */
.future-prediction-card {
    background: linear-gradient(135deg, #0b1026, #312e81);
    padding: 2.2rem;
    border-radius: 18px;
    color: #f8fafc;
    text-align: center;
    margin: 2rem 0;
    box-shadow: 0 18px 40px rgba(2, 6, 23, 0.45);
}
            
.future-prediction-card p {
    color: #f8fafc;
    font-size: 1.75rem;
    font-weight:500;
}

.future-prediction-card h3 {
    color: #cbd5f5;
    font-size: 2rem;
    margin: 0;
    letter-spacing: -1px;
}

/* ---------- Metric Cards ---------- */
.metric-card {
    background: linear-gradient(135deg, #eef2ff, #e0e7ff) !important;
    padding: 1.6rem;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    box-shadow:
        0 8px 24px rgba(15, 23, 42, 0.06),
        inset 0 1px 0 rgba(255, 255, 255, 0.6);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow:
        0 14px 32px rgba(15, 23, 42, 0.10),
        inset 0 1px 0 rgba(255, 255, 255, 0.7);
}


/* ---------- Info Box ---------- */
.info-box {
    background: linear-gradient(135deg, #dbeafe, #eff6ff);
    padding: 1.2rem 1.4rem;
    border-radius: 12px;
    border-left: 5px solid #2563eb;
    margin-bottom: 1.2rem;
    color: #0f172a;
    font-weight: 500;
}

.info-box p {
    color: #0f172a;
}

/* ---------- Percentile Badge ---------- */
.percentile-badge {
    display: inline-block;
    padding: 0.45rem 1.1rem;
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: white;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.85rem;
    margin-top: 0.6rem;
    box-shadow: 0 4px 12px rgba(34, 197, 94, 0.35);
}

</style>
""", unsafe_allow_html=True)



# --- Session State Initialization ---
if "page" not in st.session_state:
    st.session_state.page = 1
if 'step' not in st.session_state:
    st.session_state.step = 'pitcher_info'
if 'target_year' not in st.session_state:
    st.session_state.target_year = None
if 'pitcher_height' not in st.session_state:
    st.session_state.pitcher_height = None
if 'pitcher_throw' not in st.session_state:
    st.session_state.pitcher_throw = None
if 'pitch_type' not in st.session_state:
    st.session_state.pitch_type = None
if 'available_data' not in st.session_state:
    st.session_state.available_data = {}
if 'pitcher_data' not in st.session_state:
    st.session_state.pitcher_data = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = None



# st.title("VeloProject: Projecting Future Pitcher Velo")
# movie_ranking_tab1, sentiment_model_tab2, project_background_tab3 = st.tabs(["📊 Movie Rankings", "🧠 Model Inference", "📋 Project Background"])

# Load Model
VELO_MODELS_DICT = load_models()

# Load Datasets
PITCHER_METRIC_PERCENTILES_DS = load_metric_percentiles_dataset()
PITCHER_VELO_PERCENTILES_DS = preprocesss_metric_percentiles_dataset(PITCHER_METRIC_PERCENTILES_DS)


# _______________________
# MAIN APP
# -----------------------
# Main App Header
st.markdown("<h1 style='text-align: center;'>⚾ Pitcher Velocity Projector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>Advanced ML-powered velocity predictions for baseball pitchers</p>", unsafe_allow_html=True)
st.markdown("---")

# Main App Logic
if st.session_state.step == 'pitcher_info':
    st.markdown("### Pitcher Information")
    st.markdown("<div class='info-box'>Let's start by gathering some basic information about the pitcher.</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        pitcher_height = st.number_input(
            "Pitcher Height (inches)", 
            min_value=60, 
            max_value=90, 
            value=72, 
            step=1,
            help="Enter the pitcher's height in inches"
        )
        
        pitcher_throw = st.selectbox(
            "Pitcher Throws",
            options=["Right", "Left"],
            index=0,
            help="Select which hand the pitcher throws with"
        )
    
    with col2:
        pitch_type = st.selectbox(
            "Pitch Type",
            options=["Fastball", "Slider", "ChangeUp", "Curveball", "Sinker", "Cutter"],
            index=0,
            help="Select the primary pitch type to predict"
        )
        
        target_year = st.selectbox(
            "Target Prediction Year",
            options=["Sophomore", "Junior", "Senior"],
            index=0,
            help="Select which year you want to predict"
        )
    
    st.markdown("---")
    
    if st.button("Continue to Data Collection ➡️", use_container_width=True):
        st.session_state.pitcher_height = pitcher_height
        st.session_state.pitcher_throw = pitcher_throw
        st.session_state.pitch_type = pitch_type
        st.session_state.target_year = target_year.lower()
        st.session_state.step = 'check_freshman'
        st.rerun()

elif st.session_state.step == 'select_year':
    st.markdown("### Select Target Prediction Year")
    st.markdown("<div class='info-box'>Choose which year you want to predict the pitcher's velocity for.</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Sophomore Year", use_container_width=True):
            st.session_state.target_year = 'sophomore'
            st.session_state.step = 'check_freshman'
            st.rerun()
    
    with col2:
        if st.button("Junior Year", use_container_width=True):
            st.session_state.target_year = 'junior'
            st.session_state.step = 'check_freshman'
            st.rerun()
    
    with col3:
        if st.button("Senior Year", use_container_width=True):
            st.session_state.target_year = 'senior'
            st.session_state.step = 'check_freshman'
            st.rerun()

elif st.session_state.step == 'check_freshman':
    st.markdown(f"### Data Collection for {st.session_state.target_year.title()} Prediction")
    st.markdown("<div class='info-box'>Let's determine what historical data is available.</div>", unsafe_allow_html=True)
    
    st.markdown("#### Do you have **Freshman** year data?")
    st.markdown("*Required: Release Height, Extension, Mean Velocity, 90th Percentile Velocity*")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, I have freshman data", use_container_width=True):
            st.session_state.available_data['freshman'] = True
            st.session_state.step = 'input_freshman'
            st.rerun()
    
    with col2:
        if st.button("❌ No freshman data", use_container_width=True):
            st.session_state.available_data['freshman'] = False
            if st.session_state.target_year == 'sophomore':
                st.session_state.step = 'predict'
            else:
                st.session_state.step = 'check_sophomore'
            st.rerun()

elif st.session_state.step == 'input_freshman':
    st.markdown("### Enter Freshman Year Data")
    
    col1, col2 = st.columns(2)
    with col1:
        relheight = st.number_input("Release Height (feet)", min_value=0.0, max_value=8.0, value=5.9, step=0.1, key="f_relheight")
        extension = st.number_input("Extension (feet)", min_value=3.0, max_value=9.0, value=6.2, step=0.1, key="f_extension")
    
    with col2:
        mean_velo = st.number_input("Mean Velocity (mph)", min_value=50.0, max_value=110.0, value=92.4, step=0.1, key="f_mean_velo")
        percentile_90_velo = st.number_input("90th Percentile Velocity (mph)", min_value=50.0, max_value=110.0, value=94.3, step=0.1, key="f_p90_velo")
    
    if st.button("Continue ➡️", use_container_width=True):
        st.session_state.pitcher_data['freshman'] = {
            'relheight': relheight,
            'extension': extension,
            'mean_velo': mean_velo,
            'p90_velo': percentile_90_velo
        }
        if st.session_state.target_year == 'sophomore':
            st.session_state.step = 'predict'
        else:
            st.session_state.step = 'check_sophomore'
        st.rerun()

elif st.session_state.step == 'check_sophomore':
    st.markdown(f"### Continuing Data Collection")
    
    st.markdown("#### Do you have **Sophomore** year data?")
    st.markdown("*Required: Release Height, Extension, Mean Velocity, 90th Percentile Velocity*")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, I have sophomore data", use_container_width=True):
            st.session_state.available_data['sophomore'] = True
            st.session_state.step = 'input_sophomore'
            st.rerun()
    
    with col2:
        if st.button("❌ No sophomore data", use_container_width=True):
            st.session_state.available_data['sophomore'] = False
            if st.session_state.target_year == 'junior':
                st.session_state.step = 'predict'
            else:
                st.session_state.step = 'check_junior'
            st.rerun()

elif st.session_state.step == 'input_sophomore':
    st.markdown("### Enter Sophomore Year Data")
    
    col1, col2 = st.columns(2)
    with col1:
        relheight = st.number_input("Release Height (feet)", min_value=0.0, max_value=8.0, value=5.9, step=0.1, key="s_relheight")
        extension = st.number_input("Extension (feet)", min_value=3.0, max_value=9.0, value=6.2, step=0.1, key="s_extension")
        
    
    with col2:
        mean_velo = st.number_input("Mean Velocity (mph)", min_value=50.0, max_value=110.0, value=92.4, step=0.1, key="s_mean_velo")
        percentile_90_velo = st.number_input("90th Percentile Velocity (mph)", min_value=50.0, max_value=110.0, value=94.3, step=0.1, key="s_p90_velo")
    
    if st.button("Continue ➡️", use_container_width=True):
        st.session_state.pitcher_data['sophomore'] = {
            'relheight': relheight,
            'extension': extension,
            'mean_velo': mean_velo,
            'p90_velo': percentile_90_velo
        }
        if st.session_state.target_year == 'junior':
            st.session_state.step = 'predict'
        else:
            st.session_state.step = 'check_junior'
        st.rerun()

elif st.session_state.step == 'check_junior':
    st.markdown(f"### Final Data Check")
    
    st.markdown("#### Do you have **Junior** year data?")
    st.markdown("*Required: Release Height, Extension, Mean Velocity, 90th Percentile Velocity*")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, I have junior data", use_container_width=True):
            st.session_state.available_data['junior'] = True
            st.session_state.step = 'input_junior'
            st.rerun()
    
    with col2:
        if st.button("❌ No junior data", use_container_width=True):
            st.session_state.available_data['junior'] = False
            st.session_state.step = 'predict'
            st.rerun()

elif st.session_state.step == 'input_junior':
    st.markdown("### Enter Junior Year Data")
    
    col1, col2 = st.columns(2)
    with col1:
        relheight = st.number_input("Release Height (feet)", min_value=0.0, max_value=8.0, value=5.9, step=0.1, key="j_relheight")
        extension = st.number_input("Extension (feet)", min_value=3.0, max_value=9.0, value=6.2, step=0.1, key="j_extension")
        
    
    with col2:
        mean_velo = st.number_input("Mean Velocity (mph)", min_value=50.0, max_value=110.0, value=92.4, step=0.1, key="j_mean_velo")
        percentile_90_velo = st.number_input("90th Percentile Velocity (mph)", min_value=50.0, max_value=110.0, value=94.3, step=0.1, key="j_p90_velo")
    
    if st.button("Continue to Prediction ➡️", use_container_width=True):
        st.session_state.pitcher_data['junior'] = {
            'relheight': relheight,
            'extension': extension,
            'mean_velo': mean_velo,
            'p90_velo': percentile_90_velo
        }
        st.session_state.step = 'predict'
        st.rerun()

elif st.session_state.step == 'predict':
    if st.session_state.predictions is None:
        st.markdown("### Generating Predictions...")
        
        with st.spinner("Running ML models..."):
            # Determine model and make prediction
            model_name = determine_model_name()
            features = prepare_features()
            
            target_velo = make_prediction(VELO_MODELS_DICT[model_name], features)
            target_percentile = calculate_velo_percentile(target_velo, PITCHER_VELO_PERCENTILES_DS)
            
            # Store primary prediction
            st.session_state.predictions = {
                st.session_state.target_year: {
                    'velocity': target_velo,
                    'percentile': target_percentile
                }
            }
            
            # Predict future years if applicable
            if st.session_state.target_year in ['sophomore', 'junior']:
                future_predictions = predict_future_years(st.session_state.target_year, PITCHER_VELO_PERCENTILES_DS, VELO_MODELS_DICT)
                st.session_state.predictions.update(future_predictions)
        
        st.rerun()
    
    # Display Results
    st.markdown("### Velocity Predictions")
    
    # Display pitcher info summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div class='metric-card'>
                <p style='margin: 0; color: #666; font-size: 0.9rem;'>Height</p>
                <p style='margin: 0; font-size: 1.5rem; font-weight: bold;'>{st.session_state.pitcher_height}"</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class='metric-card'>
                <p style='margin: 0; color: #666; font-size: 0.9rem;'>Throws</p>
                <p style='margin: 0; font-size: 1.5rem; font-weight: bold;'>{st.session_state.pitcher_throw}</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class='metric-card'>
                <p style='margin: 0; color: #666; font-size: 0.9rem;'>Pitch Type</p>
                <p style='margin: 0; font-size: 1.5rem; font-weight: bold;'>{st.session_state.pitch_type}</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

    target = st.session_state.target_year
    primary = st.session_state.predictions[target]

    st.markdown(f"""
    <div class="prediction-card">
        <h2>{primary['velocity']} mph</h2>
        <p style="font-size:1.25rem;margin:0.4rem 0;">
            Predicted {target.title()} Year Velocity
        </p>
        <div class="percentile-badge">
            {primary['percentile']}th Percentile
        </div>
    </div>
    """, unsafe_allow_html=True)

    future_preds = {
        k: v for k, v in st.session_state.predictions.items() if k != target
    }

    if future_preds:
        st.markdown("#### Future Projections")
        cols = st.columns(len(future_preds))

        for col, (year, pred) in zip(cols, future_preds.items()):
            with col:
                st.markdown(f"""
                <div class="future-prediction-card">
                    <h3>{year.title()} Year</h3>
                    <p style="font-size:2rem;font-weight:700;">
                        {pred['velocity']} mph
                    </p>
                    <div class="percentile-badge">
                        {pred['percentile']}th Percentile
                    </div>
                </div>
                """, unsafe_allow_html=True)


    st.divider()
    
    # Model information
    with st.expander("ℹ️ Model Information"):
        model_name = determine_model_name()
        st.write(f"**Model Used:** `{model_name}`")
        # st.write("**Input Features:**")
        # for year, has_data in st.session_state.available_data.items():
        #     if has_data:
        #         data = st.session_state.pitcher_data.get(year, {})
        #         st.write(f"- **{year.title()}:** RelHeight={data.get('relheight', 'N/A')}, Extension={data.get('extension', 'N/A')}")
    
    # Action buttons
    st.markdown("---")
    if st.button("🔄 New Prediction", use_container_width=True):
        reset_app()
        st.rerun()
    

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #999;'>Powered by Machine Learning | Built with Streamlit</p>", unsafe_allow_html=True)
