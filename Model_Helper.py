# Load the hugging face model

import streamlit as st
import joblib

import Dataset_Helper
from Dataset_Helper import calculate_velo_percentile

# ----------------------------
# All the models
# ----------------------------
# Sophomore models
SO_VELO_MODEL_FP = "Models/Sophomore_Avg_Velo_Model.joblib"
# Junior models
JR_VELO_MODEL_FP = "Models/Junior_Avg_Velo_Model.joblib"
JR_VELO_MODEL_FR_FP = "Models/Junior_Avg_Velo_Model_From_FR.joblib"
JR_VELO_MODEL_SO_FP = "Models/Junior_Avg_Velo_Model_From_SO.joblib"
# Senior Models
SR_VELO_MODEL_FP = "Models/Senior_Avg_Velo_Model.joblib"
SR_VELO_MODEL_FR_FP = "Models/Senior_Avg_Velo_Model_From_FR.joblib"
SR_VELO_MODEL_SO_FP = "Models/Senior_Avg_Velo_Model_From_SO.joblib"
SR_VELO_MODEL_JR_FP = "Models/Senior_Avg_Velo_Model_From_JR.joblib"
SR_VELO_MODEL_FR_SO_FP = "Models/Senior_Avg_Velo_Model_From_FR_SO.joblib"
SR_VELO_MODEL_FR_JR_FP = "Models/Senior_Avg_Velo_Model_From_SO_JR.joblib"
SR_VELO_MODEL_SO_JR_FP = "Models/Senior_Avg_Velo_Model_From_SO_JR.joblib"


# ----------------------------
# Function: Load each model into a dictionary
# ----------------------------
@st.cache_resource
def load_models():
    VELO_MODELS_DICT = {
        'SO_VELO_MODEL': load_model(SO_VELO_MODEL_FP),
        'JR_VELO_MODEL': load_model(JR_VELO_MODEL_FP),
        'JR_VELO_MODEL_FR': load_model(JR_VELO_MODEL_FR_FP),
        'JR_VELO_MODEL_SO': load_model(JR_VELO_MODEL_SO_FP),
        'SR_VELO_MODEL': load_model(SR_VELO_MODEL_FP),
        'SR_VELO_MODEL_FR': load_model(SR_VELO_MODEL_FR_FP),
        'SR_VELO_MODEL_SO': load_model(SR_VELO_MODEL_SO_FP),
        'SR_VELO_MODEL_JR': load_model(SR_VELO_MODEL_JR_FP),
        'SR_VELO_MODEL_FR_SO': load_model(SR_VELO_MODEL_FR_SO_FP),
        'SR_VELO_MODEL_FR_JR': load_model(SR_VELO_MODEL_FR_JR_FP),
        'SR_VELO_MODEL_SO_JR': load_model(SR_VELO_MODEL_SO_JR_FP),

    }
    return VELO_MODELS_DICT


# ----------------------------
# Function: Load each model from model_name
# ----------------------------
@st.cache_resource
def load_model(model_name):
    try:
        return joblib.load(model_name)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# ----------------------------
# Function: Determine which model to use
# ----------------------------
def determine_model_name():
    # Determine which model to use based on available data
    target = st.session_state.target_year
    avail = st.session_state.available_data
    
    has_f = avail.get('freshman', False)
    has_s = avail.get('sophomore', False)
    has_j = avail.get('junior', False)
    
    if target == 'sophomore':
        if has_f:
            return 'SO_VELO_MODEL'
        else:
            return None
    elif target == 'junior':
        if has_f and has_s:
            return 'JR_VELO_MODEL'
        elif has_f:
            return 'JR_VELO_MODEL_FR'
        elif has_s:
            return 'JR_VELO_MODEL_SO'
        else:
            return None
    else:  # senior
        if has_f and has_s and has_j:
            return 'SR_VELO_MODEL'
        elif has_f and has_s:
            return 'SR_VELO_MODEL_FR_SO'
        elif has_f and has_j:
            return 'SR_VELO_MODEL_FR_JR'
        elif has_s and has_j:
            return 'SR_VELO_MODEL_SO_JR'
        elif has_f:
            return 'SR_VELO_MODEL_FR'
        elif has_s:
            return 'SR_VELO_MODEL_SO'
        elif has_j:
            return 'SR_VELO_MODEL_JR'
        else:
            return None
        

# ----------------------------
# Function: Encode Pitch Type for Model
# ----------------------------
def encode_pitch_type(pitch_type):
    if pitch_type == 'Fastball':
        return 0
    elif pitch_type == 'Slider':
        return 1
    elif pitch_type == 'ChangeUp':
        return 2
    elif pitch_type == 'Curveball':
        return 3
    elif pitch_type == 'Sinker':
        return 4
    elif pitch_type == 'Cutter':
        return 5
    

# ----------------------------
# Function: Encode Pitch Handness for Model
# ----------------------------
def encode_pitcher_throws(pitch_throws):
    if pitch_throws == 'Right':
        return 0
    elif pitch_throws == 'Left':
        return 1
    

# ----------------------------
# Function: Prepare model's features
# ----------------------------

def prepare_features():
    data = st.session_state.pitcher_data
    features = []

    # Add features in order based on available data
    # Features
    # 'PitchTypeCoded', 'PitcherHandCoded', 'PlayerHeight',
    # 'Jr_mean_extension', 'Jr_mean_rel_height', 
    # 'Fr_mean_velo', 'So_mean_velo', 'Jr_mean_velo', 
    # 'Fr_percentile90_velo', 'So_percentile90_velo', 'Jr_percentile90_velo'

    # -----------------------------
    # Global / categorical features
    # -----------------------------
    if "pitcher_height" in st.session_state:
        features.append(st.session_state["pitcher_height"])

    if "pitch_type" in st.session_state:
        features.append(encode_pitch_type(st.session_state["pitch_type"]))

    if "pitcher_throw" in st.session_state:
        features.append(encode_pitcher_throws(st.session_state["pitcher_throw"]))

    # ------------------------------------------------
    # Mechanical features (latest available year only)
    # Priority: junior → sophomore → freshman
    # ------------------------------------------------
    for year in ["junior", "sophomore", "freshman"]:
        if st.session_state.available_data.get(year, False):
            year_data = data.get(year, {})
            if "extension" in year_data:
                features.append(year_data["extension"])
            if "relheight" in year_data:
                features.append(year_data["relheight"])
            break  # stop after latest available year

    # -----------------------------
    # Mean velocity (Fr → So → Jr)
    # -----------------------------
    if st.session_state.available_data.get("freshman", False):
        fr_data = data.get("freshman", {})
        if "mean_velo" in fr_data:
            features.append(fr_data["mean_velo"])

    if st.session_state.available_data.get("sophomore", False):
        so_data = data.get("sophomore", {})
        if "mean_velo" in so_data:
            features.append(so_data["mean_velo"])

    if st.session_state.available_data.get("junior", False):
        jr_data = data.get("junior", {})
        if "mean_velo" in jr_data:
            features.append(jr_data["mean_velo"])

    # -----------------------------
    # 90th percentile velocity (Fr → So → Jr)
    # -----------------------------
    if st.session_state.available_data.get("freshman", False):
        fr_data = data.get("freshman", {})
        if "p90_velo" in fr_data:
            features.append(fr_data["p90_velo"])

    if st.session_state.available_data.get("sophomore", False):
        so_data = data.get("sophomore", {})
        if "p90_velo" in so_data:
            features.append(so_data["p90_velo"])

    if st.session_state.available_data.get("junior", False):
        jr_data = data.get("junior", {})
        if "p90_velo" in jr_data:
            features.append(jr_data["p90_velo"])

    # # Debug
    # st.write("Final feature vector:", features)
    # st.write("Feature count:", len(features))

    return features



# ----------------------------
# Function: Predict velo
# ----------------------------
def make_prediction(model, features):
    """Make prediction using the appropriate model"""
    if not hasattr(model, "predict"):
        raise TypeError(f"Expected model with predict(), got {type(model)}")
    
    if model is not None:
        try:
            prediction = model.predict([features])[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
    
    return round(prediction, 1)


# ----------------------------
# Function: Predict future velo with current data
# ----------------------------
def predict_future_years(current_year, PITCHER_VELO_PERCENTILES_DS, VELO_MODELS_DICT):
    """Predict subsequent years using the current prediction"""
    predictions = {}
    year_order = ["freshman", "sophomore", "junior", "senior"]
    current_year_holder = current_year

    # Determine which years come next
    start_idx = year_order.index(current_year)
    future_years = year_order[start_idx + 1:]

    for year in future_years:
        # Explicitly determine model + features for this year
        st.session_state.target_year = year
        model_name = determine_model_name()
        features = prepare_features()

        velo = make_prediction(VELO_MODELS_DICT[model_name], features)

        predictions[year] = {
            "velocity": velo,
            "percentile": calculate_velo_percentile(
                velo, PITCHER_VELO_PERCENTILES_DS
            )
        }

    st.session_state.target_year = current_year_holder

    return predictions

    
    # if current_year == 'sophomore':
    #     # Update pitcher data with sophomore prediction
    #     # st.session_state.pitcher_data['sophomore'] = {
    #     #     'relheight': st.session_state.pitcher_data.get('freshman', {}).get('relheight', 6),
    #     #     'extension': st.session_state.pitcher_data.get('freshman', {}).get('extension', 6),
    #     #     'mean_velo': current_velo,
    #     #     'p90_velo': st.session_state.pitcher_data.get('freshman', {}).get('p90_velo', 85)
    #     # }
    #     # st.session_state.available_data['sophomore'] = True
        
    #     # Predict junior
    #     st.session_state.target_year = 'junior'
    #     model_name = determine_model_name()
    #     features = prepare_features()
    #     junior_velo = make_prediction(VELO_MODELS_DICT[model_name], features)
    #     predictions['junior'] = {
    #         'velocity': junior_velo,
    #         'percentile': calculate_velo_percentile(junior_velo, PITCHER_VELO_PERCENTILES_DS)
    #     }
        
    #     # Predict senior
    #     # st.session_state.pitcher_data['junior'] = {
    #     #     'relheight': st.session_state.pitcher_data.get('freshman', {}).get('relheight', 6),
    #     #     'extension': st.session_state.pitcher_data.get('freshman', {}).get('extension', 6),
    #     #     'mean_velo': junior_velo,
    #     #     'p90_velo': st.session_state.pitcher_data.get('sophomore', {}).get('p90_velo', 87)
    #     # }
    #     # st.session_state.available_data['junior'] = True
    #     # st.session_state.target_year = 'senior'
        
    #     model_name = determine_model_name()
    #     features = prepare_features()
    #     senior_velo = make_prediction(VELO_MODELS_DICT[model_name], features)
    #     predictions['senior'] = {
    #         'velocity': senior_velo,
    #         'percentile': calculate_velo_percentile(senior_velo, PITCHER_VELO_PERCENTILES_DS)
    #     }
        
    # elif current_year == 'junior':
    #     # Update pitcher data with junior prediction
    #     # st.session_state.pitcher_data['junior'] = {
    #     #     'relheight': st.session_state.pitcher_data.get('sophomore', {}).get('relheight',
    #     #                 st.session_state.pitcher_data.get('freshman', {}).get('relheight', 6)),
    #     #     'extension': st.session_state.pitcher_data.get('sophomore', {}).get('extension',
    #     #                 st.session_state.pitcher_data.get('freshman', {}).get('extension', 6)),
    #     #     'mean_velo': current_velo,
    #     #     'p90_velo': st.session_state.pitcher_data.get('sophomore', {}).get('p90_velo',
    #     #                st.session_state.pitcher_data.get('freshman', {}).get('p90_velo', 85))
    #     # }
    #     # st.session_state.available_data['junior'] = True
        
    #     # Predict senior
    #     st.session_state.target_year = 'senior'
    #     model_name = determine_model_name()
    #     features = prepare_features()
    #     senior_velo = make_prediction(VELO_MODELS_DICT[model_name], features)
    #     predictions['senior'] = {
    #         'velocity': senior_velo,
    #         'percentile': calculate_velo_percentile(senior_velo, PITCHER_VELO_PERCENTILES_DS)
    #     }
    
    # return predictions