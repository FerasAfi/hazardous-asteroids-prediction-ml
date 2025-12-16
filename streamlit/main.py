import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


model_data = joblib.load('model.pkl')


model = model_data['model']
feature_names = model_data['feature_names']

numeric_features = ['a', 'e', 'i', 'H', 'Num_obs', 'Num_opps']
categorical_features = ['Orbit_type']


st.set_page_config(
    page_title="Hazardous Asteroid Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #0e1117;
    }

    [data-testid="stSelectbox"] label {
        color: white !important;
        font-weight: 600;
    }

    .sidebar-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        padding: 10px;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        border-radius: 5px;
    }

    .sidebar-divider {
        border-top: 1px solid #2a2d34;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


st.sidebar.markdown('<div class="sidebar-title">🌌 Asteroid Hazard Predictor</div>', unsafe_allow_html=True)


st.sidebar.markdown("### 🔮 **Select Mode**")
option = st.sidebar.selectbox(
    "",
    ["Single Prediction", "Bulk Prediction", "EDA", "3D Orbital Visualization"],
    label_visibility="collapsed"
)


st.sidebar.markdown("")
st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)


st.sidebar.markdown("### ℹ️ **About This App**")
st.sidebar.info(
    """
    **Model:** Random Forest Classifier  
    **Training Data:** NASA/JPL NEO data  
    **Purpose:** Predict potentially hazardous asteroids  
    **Last Updated:** November 2024
    """
)

st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)


st.sidebar.markdown("### 📋 **Features**")
st.sidebar.markdown("""
- **Single Prediction:** Input asteroid parameters
- **Bulk Prediction:** Upload CSV files  
- **EDA:** Data visualization  
- **3D Visualization:** Interactive orbit plots
""")

st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)


st.sidebar.markdown("### 💡 **How to Use**")
st.sidebar.markdown("""
1. Select a mode from above
2. Enter asteroid parameters or upload data
3. View predictions and analysis
4. Download results if needed
""")


st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.8rem;'>
    🚀 Powered by Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)

#Single Prediction
if option == "Single Prediction":
        st.header("🌠 Predict Hazardous Asteroid")
        input_data = {}

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 📊 Basic Parameters")

            with st.container():
                st.markdown("**Absolute Magnitude & Slope**")
                c1, c2 = st.columns(2)
                with c1:
                    input_data['H'] = st.number_input("H (Absolute Magnitude)",
                                                      value=18.0,
                                                      min_value=0.0,
                                                      max_value=30.0,
                                                      help="Lower values indicate brighter/larger asteroids")
                with c2:
                    input_data['G'] = st.number_input("G (Slope Parameter)",
                                                      value=0.15,
                                                      min_value=0.0,
                                                      max_value=1.0,
                                                      help="Photometric slope parameter")

            with st.container():
                st.markdown("**Observation Data**")
                input_data['Num_obs'] = st.number_input("Number of Observations",
                                                        value=100.0,
                                                        min_value=1.0,
                                                        help="Total number of observations")
                input_data['rms'] = st.number_input("RMS Residual",
                                                    value=0.5,
                                                    min_value=0.0,
                                                    help="Root mean square residual of observations")
                input_data['U'] = st.number_input("Uncertainty Parameter (U)",
                                                  value=1.0,
                                                  min_value=0.0,
                                                  help="Uncertainty parameter")

            with st.container():
                st.markdown("**Observation Arc**")
                input_data['Arc_years'] = st.number_input("Observation Arc (Years)",
                                                          value=10.0,
                                                          min_value=0.0,
                                                          help="Time span of observations in years")

        with col2:
            st.markdown("### 🪐 Orbital Elements")

            with st.container():
                st.markdown("**Angular Elements (°)**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    input_data['M'] = st.number_input("M (Mean Anomaly)",
                                                      value=180.0,
                                                      min_value=0.0,
                                                      max_value=360.0,
                                                      help="Mean anomaly in degrees")
                with c2:
                    input_data['Peri'] = st.number_input("ω (Argument of Perihelion)",
                                                         value=100.0,
                                                         min_value=0.0,
                                                         max_value=360.0,
                                                         help="Argument of perihelion in degrees")
                with c3:
                    input_data['Node'] = st.number_input("Ω (Longitude of Node)",
                                                         value=200.0,
                                                         min_value=0.0,
                                                         max_value=360.0,
                                                         help="Longitude of ascending node in degrees")

            with st.container():
                st.markdown("**Keplerian Elements**")
                c1, c2 = st.columns(2)
                with c1:
                    input_data['i'] = st.number_input("i (Inclination)",
                                                      value=5.0,
                                                      min_value=0.0,
                                                      max_value=180.0,
                                                      help="Orbital inclination in degrees")
                with c2:
                    input_data['e'] = st.number_input("e (Eccentricity)",
                                                      value=0.2,
                                                      min_value=0.0,
                                                      max_value=1.0,
                                                      help="Orbital eccentricity")

            with st.container():
                st.markdown("**Semi-major Axis**")
                input_data['a'] = st.number_input("a (Semi-major Axis, AU)",
                                                  value=2.0,
                                                  min_value=0.1,
                                                  help="Semi-major axis in astronomical units")


                if st.button("🔄 Calculate Derived Features", type="secondary", use_container_width=True):
                    if input_data['a'] > 0:
                        input_data['n'] = 360 / (input_data['a'] ** 1.5)
                        input_data['Synodic_period'] = 1 / np.abs(1 - 1 / (input_data['a'] ** 1.5))
                        st.success("Derived features calculated!")

            with st.container():
                st.markdown("**Opposition Data**")
                input_data['Num_opps'] = st.number_input("Number of Oppositions",
                                                         value=5.0,
                                                         min_value=0.0,
                                                         help="Number of oppositions observed")


        with st.expander("📐 Derived Orbital Parameters", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                input_data['n'] = st.number_input("n (Mean Motion, °/day)",
                                                  value=360 / (input_data['a'] ** 1.5) if input_data['a'] > 0 else 0.2,
                                                  help="Mean daily motion = 360 / a^(3/2)")
            with col_b:
                input_data['Synodic_period'] = st.number_input("Synodic Period (years)",
                                                               value=1 / np.abs(1 - 1 / (input_data['a'] ** 1.5)) if
                                                               input_data['a'] > 0 else 400.0,
                                                               help="Time between successive oppositions")

            st.caption("ℹ️ These values are automatically calculated from semi-major axis (a)")


        st.markdown("### 🛰️ Orbit Classification")
        orbit_types = ['MBA', 'Jupiter Trojan', 'Hungaria', 'Phocaea', 'Distant Object',
                       'Object with perihelion distance < 1.665 AU', 'Aten', 'Amor', 'Apollo',
                       'Hilda', 'Atira']

        col_type, col_info = st.columns([1, 2])
        with col_type:
            input_data['Orbit_type'] = st.selectbox("Orbit Type", orbit_types, index=0)

        with col_info:
            orbit_descriptions = {
                'MBA': 'Main Belt Asteroid - between Mars and Jupiter',
                'Jupiter Trojan': 'Shares orbit with Jupiter at L4/L5 points',
                'Hungaria': 'Inner main belt, high inclination',
                'Phocaea': 'Inner main belt, moderate eccentricity',
                'Distant Object': 'Beyond Jupiter\'s orbit',
                'Object with perihelion distance < 1.665 AU': 'Close approach objects',
                'Aten': 'Earth-crossing, semi-major axis < 1 AU',
                'Amor': 'Earth-approaching, perihelion 1.017-1.3 AU',
                'Apollo': 'Earth-crossing, semi-major axis > 1 AU',
                'Hilda': '3:2 resonance with Jupiter',
                'Atira': 'Interior to Earth\'s orbit'
            }
            st.info(
                f"**{input_data['Orbit_type']}**: {orbit_descriptions.get(input_data['Orbit_type'], 'No description available')}")


        st.markdown("---")

        input_df = pd.DataFrame([input_data])

        if st.button("🚀 Predict Hazard Level", type="primary", use_container_width=True):
            input_encoded = pd.get_dummies(input_df, columns=['Orbit_type'])

            missing_cols = set(feature_names) - set(input_encoded.columns)
            for col in missing_cols:
                input_encoded[col] = 0

            input_encoded = input_encoded[feature_names]

            pred = model.predict(input_encoded)[0]
            pred_proba = model.predict_proba(input_encoded)[0][1]

            st.markdown("---")
            st.markdown("## 📊 Prediction Results")

            result_container = st.container()
            with result_container:
                col_result, col_prob = st.columns([2, 1])

                with col_result:
                    if pred == 1:
                        st.error(f"## ⚠️ **HAZARDOUS ASTEROID**")
                        st.write(f"**Probability:** {pred_proba:.2%}")
                        st.warning("""
                            ⚠️ **Potential Threat Detected**

                            This asteroid has characteristics suggesting it may pose a hazard to Earth.
                            Recommended actions:
                            - Monitor for orbital updates
                            - Track for future observations
                            - Report to planetary defense authorities if probability > 80%
                            """)
                    else:
                        st.success(f"## ✅ **NON-HAZARDOUS / UNCERTAIN**")
                        st.write(f"**Probability:** {pred_proba:.2%}")
                        st.info("""
                            ✅ **Low Risk Assessment**

                            This asteroid is not currently classified as hazardous.
                            However, continue monitoring as orbital parameters may change with new observations.
                            """)

                with col_prob:
                    st.metric("Hazard Probability", f"{pred_proba:.1%}")
                    st.progress(pred_proba)

                    if pred_proba > 0.8:
                        confidence = "High"
                    elif pred_proba > 0.5:
                        confidence = "Medium"
                    else:
                        confidence = "Low"
                    st.caption(f"Confidence: **{confidence}**")

#EDA
elif option == "EDA":
    st.header("🔍 Exploratory Data Analysis")

    with st.container():
        st.markdown("### 📁 Upload Your Dataset")
        eda_file = st.file_uploader(
            "Drag and drop or click to browse",
            type="csv",
            key="eda",
            help="Upload a CSV file containing asteroid data with 'hazerdous' column"
        )

    if eda_file is not None:
        df = pd.read_csv(eda_file)

        st.markdown("---")

        with st.container():
            st.markdown("### 📊 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Asteroids", f"{len(df):,}")
            with col2:
                hazardous_count = df['hazerdous'].sum() if 'hazerdous' in df.columns else 0
                st.metric("Hazardous Asteroids", f"{hazardous_count:,}")
            with col3:
                if 'hazerdous' in df.columns:
                    hazard_rate = (df['hazerdous'].mean() * 100)
                    st.metric("Hazard Rate", f"{hazard_rate:.1f}%")
            with col4:
                st.metric("Features", len(df.columns))

        required_cols = ['hazerdous'] + numeric_features
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"⚠️ Dataset missing required columns: {missing_cols}")
            st.info("Please upload a dataset containing all required features")
        else:
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Feature Analysis",
                "🔥 Hazard Distribution",
                "📊 Correlations",
                "🎯 Target Insights",
                "📋 Data Preview"
            ])

            with tab1:
                st.markdown("### 📈 Feature Distribution Analysis")

                col_select, col_info = st.columns([1, 2])
                with col_select:
                    selected_feature = st.selectbox(
                        "Select Feature to Analyze",
                        numeric_features,
                        key="feature_select"
                    )

                with col_info:
                    feature_descriptions = {
                        'a': 'Semi-major Axis (AU): Average distance from the Sun',
                        'e': 'Eccentricity: How elliptical the orbit is (0=circle, 1=parabola)',
                        'i': 'Inclination (°): Tilt of orbit relative to Earth\'s orbital plane',
                        'H': 'Absolute Magnitude: Intrinsic brightness, indicates size',
                        'Num_obs': 'Number of Observations: More observations = better orbit determination',
                        'Num_opps': 'Number of Oppositions: Times asteroid was opposite Sun from Earth'
                    }
                    st.info(
                        f"**{selected_feature}**: {feature_descriptions.get(selected_feature, 'No description available')}")

                viz_col1, viz_col2 = st.columns(2)

                with viz_col1:
                    st.markdown("#### Distribution by Hazard Class")
                    fig1, ax1 = plt.subplots(figsize=(10, 6))

                    sns.histplot(
                        data=df[df['hazerdous'] == 1],
                        x=selected_feature,
                        color='#ff6b6b',
                        label='Hazardous',
                        kde=True,
                        stat='density',
                        alpha=0.6,
                        edgecolor='black',
                        linewidth=0.5
                    )
                    sns.histplot(
                        data=df[df['hazerdous'] == 0],
                        x=selected_feature,
                        color='#4ecdc4',
                        label='Non-Hazardous',
                        kde=True,
                        stat='density',
                        alpha=0.6,
                        edgecolor='black',
                        linewidth=0.5
                    )

                    ax1.set_xlabel(selected_feature, fontsize=12)
                    ax1.set_ylabel('Density', fontsize=12)
                    ax1.set_title(f'Distribution of {selected_feature}', fontsize=14, fontweight='bold')
                    ax1.legend(title='Hazard Class', title_fontsize=12)
                    ax1.grid(True, alpha=0.3)

                    st.pyplot(fig1)

                with viz_col2:
                    st.markdown("#### Statistical Comparison")
                    fig2, ax2 = plt.subplots(figsize=(10, 6))

                    box_data = [
                        df[df['hazerdous'] == 0][selected_feature].dropna(),
                        df[df['hazerdous'] == 1][selected_feature].dropna()
                    ]

                    box = ax2.boxplot(
                        box_data,
                        labels=['Non-Hazardous', 'Hazardous'],
                        patch_artist=True,
                        widths=0.6
                    )

                    colors = ['#4ecdc4', '#ff6b6b']
                    for patch, color in zip(box['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)

                    ax2.set_ylabel(selected_feature, fontsize=12)
                    ax2.set_title(f'{selected_feature} Distribution by Class', fontsize=14, fontweight='bold')
                    ax2.grid(True, alpha=0.3, axis='y')

                    st.pyplot(fig2)

                    st.markdown("##### 📊 Summary Statistics")
                    stats_col1, stats_col2 = st.columns(2)
                    with stats_col1:
                        mean_hazard = df[df['hazerdous'] == 1][selected_feature].mean()
                        st.metric("Hazardous Mean", f"{mean_hazard:.3f}")
                    with stats_col2:
                        mean_safe = df[df['hazerdous'] == 0][selected_feature].mean()
                        st.metric("Non-Hazardous Mean", f"{mean_safe:.3f}")

            with tab2:
                st.markdown("### 🔥 Hazard Class Distribution")

                fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))

                hazard_counts = df['hazerdous'].value_counts()
                labels = ['Non-Hazardous', 'Hazardous']
                colors = ['#4ecdc4', '#ff6b6b']
                explode = (0, 0.1)

                wedges, texts, autotexts = ax3.pie(
                    hazard_counts,
                    labels=labels,
                    colors=colors,
                    autopct='%1.1f%%',
                    startangle=90,
                    explode=explode,
                    wedgeprops=dict(width=0.3, edgecolor='white')
                )

                ax3.set_title('Hazard Class Distribution', fontsize=14, fontweight='bold')

                comparison_features = ['a', 'e', 'i']
                comparison_data = []

                for feature in comparison_features:
                    if feature in df.columns:
                        mean_hazard = df[df['hazerdous'] == 1][feature].mean()
                        mean_safe = df[df['hazerdous'] == 0][feature].mean()
                        comparison_data.append({
                            'feature': feature,
                            'hazardous': mean_hazard,
                            'non_hazardous': mean_safe
                        })

                if comparison_data:
                    df_compare = pd.DataFrame(comparison_data)
                    x = range(len(comparison_data))
                    width = 0.35

                    ax4.bar([i - width / 2 for i in x], df_compare['non_hazardous'], width,
                            label='Non-Hazardous', color='#4ecdc4', alpha=0.8)
                    ax4.bar([i + width / 2 for i in x], df_compare['hazardous'], width,
                            label='Hazardous', color='#ff6b6b', alpha=0.8)

                    ax4.set_xlabel('Feature', fontsize=12)
                    ax4.set_ylabel('Average Value', fontsize=12)
                    ax4.set_title('Feature Comparison by Class', fontsize=14, fontweight='bold')
                    ax4.set_xticks(x)
                    ax4.set_xticklabels(df_compare['feature'])
                    ax4.legend()
                    ax4.grid(True, alpha=0.3, axis='y')

                st.pyplot(fig3)

            with tab3:
                st.markdown("### 📊 Correlation Analysis")

                corr_matrix = df[numeric_features + ['hazerdous']].corr()

                fig4, ax4 = plt.subplots(figsize=(12, 8))
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='coolwarm',
                    center=0,
                    square=True,
                    linewidths=1,
                    cbar_kws={'shrink': 0.8},
                    ax=ax4
                )
                ax4.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
                st.pyplot(fig4)

                st.markdown("#### 🎯 Top Correlations with Hazardous")
                corr_with_target = corr_matrix['hazerdous'].drop('hazerdous').sort_values(key=abs, ascending=False)

                col_top, col_bottom = st.columns(2)
                with col_top:
                    st.markdown("**Positive Correlations**")
                    positive_corr = corr_with_target[corr_with_target > 0]
                    for feature, value in positive_corr.head(5).items():
                        st.progress(float(value), text=f"{feature}: {value:.3f}")

                with col_bottom:
                    st.markdown("**Negative Correlations**")
                    negative_corr = corr_with_target[corr_with_target < 0]
                    for feature, value in negative_corr.head(5).items():
                        st.progress(abs(float(value)), text=f"{feature}: {value:.3f}")

            with tab4:
                st.markdown("### 🎯 Hazard Prediction Insights")

                st.markdown("#### 📈 Feature Importance for Hazard Prediction")

                feature_importance = corr_matrix['hazerdous'].abs().drop('hazerdous').sort_values(ascending=True)

                fig5, ax5 = plt.subplots(figsize=(10, 8))
                y_pos = range(len(feature_importance))
                bars = ax5.barh(y_pos, feature_importance.values, color='#1f77b4')

                for i, (feature, value) in enumerate(feature_importance.items()):
                    color = '#ff6b6b' if corr_matrix.loc[feature, 'hazerdous'] > 0 else '#4ecdc4'
                    bars[i].set_color(color)

                ax5.set_yticks(y_pos)
                ax5.set_yticklabels(feature_importance.index)
                ax5.set_xlabel('Absolute Correlation with Hazardous', fontsize=12)
                ax5.set_title('Feature Importance for Hazard Prediction', fontsize=14, fontweight='bold')
                ax5.grid(True, alpha=0.3, axis='x')

                st.pyplot(fig5)

                st.markdown("#### 💡 Key Insights")

                top_positive = corr_with_target[corr_with_target > 0].head(3)
                top_negative = corr_with_target[corr_with_target < 0].head(3)

                insights_col1, insights_col2 = st.columns(2)

                with insights_col1:
                    st.markdown("**Features that INCREASE hazard probability:**")
                    for feature, corr in top_positive.items():
                        st.write(
                            f"• **{feature}**: Higher values correlate with higher hazard probability (r={corr:.3f})")

                with insights_col2:
                    st.markdown("**Features that DECREASE hazard probability:**")
                    for feature, corr in top_negative.items():
                        st.write(
                            f"• **{feature}**: Higher values correlate with lower hazard probability (r={corr:.3f})")

            with tab5:
                st.markdown("### 📋 Data Preview & Statistics")

                st.dataframe(
                    df.head(20).style.background_gradient(
                        subset=numeric_features,
                        cmap='viridis'
                    ).format("{:.3f}", subset=numeric_features)
                )

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Dataset",
                    data=csv,
                    file_name="asteroid_eda_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                st.markdown("#### 📊 Statistical Summary")
                st.dataframe(df[numeric_features + ['hazerdous']].describe().round(3))

    else:
        st.markdown("---")
        col_info, col_placeholder = st.columns([2, 1])

        with col_info:
            st.markdown("""
            ### 📋 Expected Data Format

            Upload a CSV file containing asteroid data with the following columns:

            **Required Columns:**
            - `hazerdous`: Binary target variable (1 = hazardous, 0 = non-hazardous)
            - `a`: Semi-major axis (AU)
            - `e`: Eccentricity
            - `i`: Inclination (degrees)
            - `H`: Absolute magnitude
            - `Num_obs`: Number of observations
            - `Num_opps`: Number of oppositions

            **Optional Columns:**
            - Any additional orbital parameters or observation data

            **Example Dataset Structure:**
            ```csv
            hazerdous,a,e,i,H,Num_obs,Num_opps,Orbit_type
            0,2.365,0.145,5.231,17.8,156,3,MBA
            1,1.456,0.324,12.567,16.2,89,2,Apollo
            ```
            """)

        with col_placeholder:
            st.image("https://cdn-icons-png.flaticon.com/512/2733/2733830.png", width=150)
            st.markdown("""
            <div style='text-align: center; color: #666; margin-top: 20px;'>
            <h3>⬆️ Upload to Begin</h3>
            <p>Drag and drop your CSV file above</p>
            </div>
            """, unsafe_allow_html=True)

# Bulk CSV Prediction
elif option == "Bulk Prediction":
    st.header("Upload CSV of Asteroids")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df_csv = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df_csv.head())

        if 'Orbit_type' not in df_csv.columns:
            st.error("CSV file must contain an 'Orbit_type' column")
            st.stop()

        df_original = df_csv.copy()
        df_encoded = pd.get_dummies(df_csv, columns=['Orbit_type'])

        missing_cols = set(feature_names) - set(df_encoded.columns)
        for col in missing_cols:
            df_encoded[col] = 0

        df_encoded = df_encoded[feature_names]

        predictions = model.predict(df_encoded)
        probabilities = model.predict_proba(df_encoded)[:, 1]

        df_original['hazardous_pred'] = predictions
        df_original['hazardous_probability'] = probabilities

        st.write("**Predictions:**")
        st.dataframe(df_original.head())

        csv_download = df_original.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions CSV",
            data=csv_download,
            file_name="asteroid_predictions.csv",
            mime="text/csv"
        )

# 3D Visualisation
elif option == "3D Orbital Visualization":
    st.header("3D Scatter of Orbital Parameters")

    viz_file = st.file_uploader("Upload dataset for Visualization", type="csv", key="viz")
    if viz_file is not None:
        df = pd.read_csv(viz_file)
        fig = px.scatter_3d(df, x='a', y='e', z='i',
                            color=df['hazerdous'].map({1: 'Hazardous', 0: 'Uncertain'}),
                            title="3D Orbital Parameters (a, e, i)")
        st.plotly_chart(fig)