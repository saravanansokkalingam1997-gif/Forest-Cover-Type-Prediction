import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import xgboost


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Forest Cover Type Prediction",
    page_icon="🌲",
    layout="wide"
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_trained_model():
    return joblib.load("best_model.pkl")


@st.cache_resource
def load_label_encoder():
    return joblib.load("label_encoder.pkl")


# ============================================================
# LOAD MODEL
# ============================================================

try:
    trained_model = load_trained_model()
    label_encoder = load_label_encoder()

    model_expected_features = list(
        trained_model.feature_names_in_
    )

except Exception as e:
    st.error("Error loading model files.")
    st.exception(e)
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🌲 Forest Cover Type Prediction")

st.write(
    "Predict the forest cover type using environmental "
    "and geographical features."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Input Features")


# ============================================================
# ENVIRONMENTAL FEATURES
# ============================================================

st.sidebar.subheader("Environmental Features")


elevation = st.sidebar.number_input(
    "Elevation",
    min_value=0,
    max_value=5000,
    value=2500
)


aspect = st.sidebar.number_input(
    "Aspect",
    min_value=0,
    max_value=360,
    value=180
)


slope = st.sidebar.number_input(
    "Slope",
    min_value=0,
    max_value=90,
    value=15
)


horizontal_distance_to_hydrology = st.sidebar.number_input(
    "Horizontal Distance To Hydrology",
    min_value=0,
    max_value=1000,
    value=200
)


vertical_distance_to_hydrology = st.sidebar.number_input(
    "Vertical Distance To Hydrology",
    min_value=-500,
    max_value=500,
    value=50
)


horizontal_distance_to_roadways = st.sidebar.number_input(
    "Horizontal Distance To Roadways",
    min_value=0,
    max_value=10000,
    value=1000
)


hillshade_9am = st.sidebar.number_input(
    "Hillshade 9am",
    min_value=0,
    max_value=255,
    value=200
)


hillshade_noon = st.sidebar.number_input(
    "Hillshade Noon",
    min_value=0,
    max_value=255,
    value=200
)


hillshade_3pm = st.sidebar.number_input(
    "Hillshade 3pm",
    min_value=0,
    max_value=255,
    value=150
)


horizontal_distance_to_fire_points = st.sidebar.number_input(
    "Horizontal Distance To Fire Points",
    min_value=0,
    max_value=10000,
    value=1500
)


# ============================================================
# WILDERNESS AREA
# ============================================================

st.sidebar.subheader("Wilderness Area")

wilderness_area = st.sidebar.selectbox(
    "Wilderness Area",
    [1, 2, 3, 4]
)


# ============================================================
# SOIL TYPE
# ============================================================

st.sidebar.subheader("Soil Type")

soil_type = st.sidebar.selectbox(
    "Soil Type",
    list(range(1, 41))
)


# ============================================================
# CREATE ORIGINAL FEATURES
# ============================================================

input_features = {

    "Elevation": elevation,

    "Aspect": aspect,

    "Slope": slope,

    "Horizontal_Distance_To_Hydrology":
        horizontal_distance_to_hydrology,

    "Vertical_Distance_To_Hydrology":
        vertical_distance_to_hydrology,

    "Horizontal_Distance_To_Roadways":
        horizontal_distance_to_roadways,

    "Hillshade_9am":
        hillshade_9am,

    "Hillshade_Noon":
        hillshade_noon,

    "Hillshade_3pm":
        hillshade_3pm,

    "Horizontal_Distance_To_Fire_Points":
        horizontal_distance_to_fire_points
}


# ============================================================
# ONE-HOT ENCODING
# ============================================================

# Wilderness Area

for i in range(1, 5):

    input_features[f"Wilderness_Area_{i}"] = (
        1 if wilderness_area == i else 0
    )


# Soil Type

for i in range(1, 41):

    input_features[f"Soil_Type_{i}"] = (
        1 if soil_type == i else 0
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

# ------------------------------------------------------------
# Hydrology Features
# ------------------------------------------------------------

input_features["Hydrology_Ratio"] = (
    horizontal_distance_to_hydrology
    /
    (abs(vertical_distance_to_hydrology) + 1)
)


input_features["elev_hydrology"] = (
    elevation - horizontal_distance_to_hydrology
)


input_features["Elevation_VDH"] = (
    elevation - vertical_distance_to_hydrology
)


input_features["Elevation_VDH_abs"] = abs(
    vertical_distance_to_hydrology
)


input_features["Hydrology_Distance"] = np.sqrt(
    horizontal_distance_to_hydrology ** 2
    +
    vertical_distance_to_hydrology ** 2
)


# ------------------------------------------------------------
# Hillshade Features
# ------------------------------------------------------------

input_features["Hillshade_Diff"] = (
    hillshade_9am - hillshade_3pm
)


input_features["Hillshade_Avg"] = (
    hillshade_9am
    +
    hillshade_noon
    +
    hillshade_3pm
) / 3


# ------------------------------------------------------------
# Interaction Features
# ------------------------------------------------------------

input_features["Slope_Elevation"] = (
    slope * elevation
)


input_features["Roads_to_Hydrology"] = (
    horizontal_distance_to_roadways
    /
    (horizontal_distance_to_hydrology + 1)
)


input_features["Hydro_Fire_Sum"] = (
    horizontal_distance_to_hydrology
    +
    horizontal_distance_to_fire_points
)


# ------------------------------------------------------------
# Elevation Bands
# ------------------------------------------------------------

input_features["elev_low"] = int(
    elevation < 2000
)


input_features["elev_mid"] = int(
    2000 <= elevation <= 3000
)


input_features["elev_high"] = int(
    elevation > 3000
)


# ------------------------------------------------------------
# Aspect Circular Features
# ------------------------------------------------------------

input_features["north_south"] = np.cos(
    np.radians(aspect)
)


input_features["east_west"] = np.sin(
    np.radians(aspect)
)


# ------------------------------------------------------------
# Distance Features
# ------------------------------------------------------------

input_features["Mean_Distance"] = (
    horizontal_distance_to_roadways
    +
    horizontal_distance_to_fire_points
    +
    horizontal_distance_to_hydrology
) / 3


input_features["Sum_Distance"] = (
    horizontal_distance_to_roadways
    +
    horizontal_distance_to_fire_points
    +
    horizontal_distance_to_hydrology
)


# ============================================================
# CREATE DATAFRAME
# ============================================================

input_df = pd.DataFrame(
    [input_features]
)


# ============================================================
# ALIGN WITH MODEL FEATURES
# ============================================================

for feature in model_expected_features:

    if feature not in input_df.columns:

        input_df[feature] = 0


input_df = input_df[
    model_expected_features
]


# ============================================================
# PREDICTION
# ============================================================

st.subheader("Prediction")


if st.button(
    "🌲 Predict Forest Cover Type",
    use_container_width=True
):

    try:

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = trained_model.predict(
            input_df
        )[0]


        # ----------------------------------------------------
        # Decode Prediction
        # ----------------------------------------------------

        predicted_class = (
            label_encoder.inverse_transform(
                [prediction]
            )[0]
        )


        # ----------------------------------------------------
        # Display Prediction
        # ----------------------------------------------------

        st.success(
            f"Predicted Forest Cover Type: "
            f"**{predicted_class}**"
        )


        # ----------------------------------------------------
        # Prediction Probabilities
        # ----------------------------------------------------

        if hasattr(
            trained_model,
            "predict_proba"
        ):

            probabilities = (
                trained_model.predict_proba(
                    input_df
                )[0]
            )


            class_labels = (
                label_encoder.inverse_transform(
                    np.arange(
                        len(probabilities)
                    )
                )
            )


            probability_df = pd.DataFrame({

                "Forest Cover Type":
                    class_labels,

                "Probability (%)":
                    probabilities * 100

            })


            probability_df = (
                probability_df
                .sort_values(
                    "Probability (%)",
                    ascending=False
                )
                .reset_index(
                    drop=True
                )
            )


            st.subheader(
                "Prediction Probabilities"
            )


            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )


            st.bar_chart(
                probability_df.set_index(
                    "Forest Cover Type"
                )["Probability (%)"]
            )


        # ----------------------------------------------------
        # Input Summary
        # ----------------------------------------------------

        with st.expander(
            "View Input Summary"
        ):

            summary_df = pd.DataFrame({

                "Feature": [

                    "Elevation",

                    "Aspect",

                    "Slope",

                    "Horizontal Distance To Hydrology",

                    "Vertical Distance To Hydrology",

                    "Horizontal Distance To Roadways",

                    "Hillshade 9am",

                    "Hillshade Noon",

                    "Hillshade 3pm",

                    "Horizontal Distance To Fire Points",

                    "Wilderness Area",

                    "Soil Type"

                ],

                "Value": [

                    elevation,

                    aspect,

                    slope,

                    horizontal_distance_to_hydrology,

                    vertical_distance_to_hydrology,

                    horizontal_distance_to_roadways,

                    hillshade_9am,

                    hillshade_noon,

                    hillshade_3pm,

                    horizontal_distance_to_fire_points,

                    wilderness_area,

                    soil_type

                ]

            })


            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True
            )


    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader(
    "Model Information"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Model",
        "XGBoost"
    )


with col2:

    st.metric(
        "Test Accuracy",
        "95.12%"
    )


with col3:

    st.metric(
        "Macro F1",
        "89.41%"
    )


st.caption(
    "Forest Cover Type Classification • "
    "Machine Learning Deployment"
)
