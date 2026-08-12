import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Kitwe Solar Irradiance Predictor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 15px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #f6c90e, #e8a000);
        color: #0e1117;
        font-weight: 700;
        font-size: 16px;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        width: 100%;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #e8a000, #f6c90e);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(246, 201, 14, 0.3);
    }

    /* Cards */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }

    .metric-card h2 {
        color: #f6c90e;
        font-size: 2rem;
        margin: 0;
    }

    .metric-card p {
        color: #8b949e;
        margin: 5px 0 0 0;
        font-size: 0.85rem;
    }

    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #1c2128, #161b22);
        border: 2px solid #f6c90e;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }

    .result-box h1 {
        color: #f6c90e;
        font-size: 3rem;
        margin: 0;
    }

    .result-box p {
        color: #8b949e;
        font-size: 1rem;
        margin: 8px 0 0 0;
    }

    /* Section headers */
    .section-header {
        border-left: 4px solid #f6c90e;
        padding-left: 12px;
        margin: 20px 0 10px 0;
    }

    /* Divider */
    hr {
        border-color: #30363d;
    }

    /* Input labels */
    label {
        color: #c9d1d9 !important;
        font-weight: 600 !important;
    }

    /* Number inputs */
    input[type="number"] {
        background-color: #1c2128 !important;
        color: #ffffff !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(base_dir, 'models', 'mlr_model.joblib'))
    scaler = joblib.load(os.path.join(base_dir, 'models', 'scaler.joblib'))
    return model, scaler

@st.cache_data
def load_data():
    df = pd.read_csv(
        os.path.join(base_dir, 'data', 'nasa_power_kitwe_raw.csv'),
        skiprows=12, na_values=-999
    )
    df.columns = ['YEAR', 'MON', 'DAY', 'GHI', 'RH2M', 'T2M', 'CLOUD_AMT']
    df = df.dropna()
    return df

model, scaler = load_model()
df = load_data()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: #f6c90e; font-size: 1.8rem;'>☀️ SolarIQ</h1>
        <p style='color: #8b949e; font-size: 0.8rem;'>Kitwe Solar Irradiance<br>Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔮 Prediction", "📊 EDA Plots"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("""
    <div style='padding: 10px; background: #1c2128; border-radius: 8px;'>
        <p style='color: #8b949e; font-size: 0.75rem; margin: 0;'>
        📍 <b style='color: #c9d1d9'>Location:</b> Kitwe, Zambia<br>
        📅 <b style='color: #c9d1d9'>Data:</b> 2020 – 2025<br>
        🛰️ <b style='color: #c9d1d9'>Source:</b> NASA POWER API<br>
        🤖 <b style='color: #c9d1d9'>Model:</b> Multiple Linear Regression<br>
        📐 <b style='color: #c9d1d9'>R² Score:</b> 0.5647
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <p style='color: #8b949e; font-size: 0.7rem; text-align: center;'>
    CS 301 Group Project<br>
    Copperbelt University<br>
    School of ICT — 2025
    </p>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════════════════════════════════
if page == "🏠 Home":

    st.markdown("""
    <div style='text-align: center; padding: 40px 0 20px 0;'>
        <h1 style='color: #f6c90e; font-size: 3rem; margin: 0;'>☀️ Solar Irradiance Predictor</h1>
        <h3 style='color: #8b949e; font-weight: 400; margin: 10px 0;'>
            Kitwe, Copperbelt Province — Republic of Zambia
        </h3>
        <p style='color: #c9d1d9; max-width: 700px; margin: 0 auto; font-size: 1rem; line-height: 1.7;'>
            A machine learning system for predicting <b style='color: #f6c90e'>Global Horizontal Irradiance (GHI)</b>
            using Multiple Linear Regression trained on six years of NASA POWER satellite climate data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Dataset stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h2>2,190</h2>
            <p>Daily Observations</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h2>6</h2>
            <p>Years of Data (2020–2025)</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h2>0.5647</h2>
            <p>Model R² Score</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class='metric-card'>
            <h2>0.6877</h2>
            <p>RMSE (kWh/m²/day)</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # About section
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("<div class='section-header'><h3 style='color:#ffffff; margin:0'>About This System</h3></div>", unsafe_allow_html=True)
        st.markdown("""
        <p style='color: #c9d1d9; line-height: 1.8;'>
        This system was developed as part of the <b style='color:#f6c90e'>CS 301 Group Project</b>
        at the Copperbelt University, School of Information and Communication Technology.
        <br><br>
        The system uses <b style='color:#f6c90e'>Multiple Linear Regression (MLR)</b> to predict
        Global Horizontal Irradiance (GHI) — the total solar radiation received per unit area —
        using three meteorological predictor variables acquired from the
        <b style='color:#f6c90e'>NASA POWER Renewable Energy API</b>.
        <br><br>
        The model was trained on <b style='color:#f6c90e'>1,533 daily observations</b>
        and evaluated on <b style='color:#f6c90e'>657 held-out test samples</b>,
        representing a 70/30 train-test split of the complete dataset.
        </p>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='section-header'><h3 style='color:#ffffff; margin:0'>Predictor Variables</h3></div>", unsafe_allow_html=True)

        predictors = {
            "🌡️ Temperature (T2M)": ("Temperature at 2 metres above surface", "+0.7494", "°C"),
            "💧 Relative Humidity (RH2M)": ("Relative humidity at 2 metres", "+0.2820", "%"),
            "☁️ Cloud Cover (CLOUD_AMT)": ("Total cloud amount percentage", "-0.9405", "%"),
        }

        for name, (desc, coef, unit) in predictors.items():
            color = "#4ade80" if "+" in coef else "#f87171"
            st.markdown(f"""
            <div style='background:#1c2128; border:1px solid #30363d; border-radius:10px;
                        padding:12px 16px; margin:8px 0;'>
                <b style='color:#f6c90e'>{name}</b><br>
                <span style='color:#8b949e; font-size:0.8rem'>{desc} ({unit})</span><br>
                <span style='color:{color}; font-size:0.85rem'>Coefficient: {coef}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # MLR equation
    st.markdown("<div class='section-header'><h3 style='color:#ffffff; margin:0'>Regression Equation</h3></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#1c2128; border:1px solid #f6c90e; border-radius:12px;
                padding:20px; text-align:center; margin:10px 0;'>
        <p style='color:#8b949e; margin:0 0 8px 0; font-size:0.85rem'>Multiple Linear Regression Model</p>
        <h3 style='color:#f6c90e; margin:0; font-family: monospace;'>
            GHI = 5.6756 + (0.7494 × T2M) + (0.2820 × RH2M) + (-0.9405 × CLOUD_AMT)
        </h3>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 2 — PREDICTION
# ════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":

    st.markdown("""
    <h1 style='color:#f6c90e; margin-bottom:5px;'>🔮 GHI Prediction</h1>
    <p style='color:#8b949e;'>Enter current or forecasted climate values for Kitwe to predict Global Horizontal Irradiance.</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col_inputs, col_result = st.columns([1, 1], gap="large")

    with col_inputs:
        st.markdown("<div class='section-header'><h3 style='color:#ffffff; margin:0'>Climate Input Variables</h3></div>", unsafe_allow_html=True)

        t2m = st.slider(
            "🌡️ Temperature — T2M (°C)",
            min_value=10.0, max_value=30.0,
            value=22.0, step=0.1,
            help="Temperature at 2 metres above the surface"
        )

        rh2m = st.slider(
            "💧 Relative Humidity — RH2M (%)",
            min_value=10.0, max_value=100.0,
            value=65.0, step=0.1,
            help="Relative humidity at 2 metres above the surface"
        )

        cloud_amt = st.slider(
            "☁️ Cloud Cover — CLOUD_AMT (%)",
            min_value=0.0, max_value=100.0,
            value=40.0, step=0.1,
            help="Total cloud amount as a percentage"
        )

        st.markdown("---")
        predict_btn = st.button("☀️ Predict Solar Irradiance", use_container_width=True)

    with col_result:
        st.markdown("<div class='section-header'><h3 style='color:#ffffff; margin:0'>Prediction Output</h3></div>", unsafe_allow_html=True)

        if predict_btn:
            input_data = np.array([[t2m, rh2m, cloud_amt]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

            # Classification
            # Classification — calibrated to Kitwe's GHI distribution
            # Thresholds derived from 6-year NASA POWER dataset percentiles
            # 33rd percentile = 5.3325, 67th percentile = 6.1238
        if prediction < 5.3325:
           level = "Low"
           icon = "🌧️"
           desc = "Below average irradiance — cloudy or rainy conditions likely"
           border_color = "#f87171"
        elif prediction < 6.1238:
           level = "Moderate"
           icon = "⛅"
           desc = "Average irradiance — typical Kitwe conditions"
           border_color = "#fbbf24"
        else:
           level = "High"
           icon = "🌞"
           desc = "Above average irradiance — clear sunny conditions"
           border_color = "#4ade80"

    st.markdown(f"""
            <div style='background:#1c2128; border:2px solid {border_color};
                        border-radius:16px; padding:30px; text-align:center; margin:10px 0;'>
                <p style='color:#8b949e; margin:0; font-size:0.85rem'>Predicted GHI</p>
                <h1 style='color:#f6c90e; font-size:3.5rem; margin:10px 0;'>
                    {prediction:.4f}
                </h1>
                <p style='color:#8b949e; margin:0; font-size:0.85rem'>kWh/m²/day</p>
                <hr style='border-color:#30363d; margin:15px 0;'>
                <h3 style='color:{border_color}; margin:0;'>{icon} {level} Solar Irradiance</h3>
                <p style='color:#8b949e; margin:5px 0 0 0; font-size:0.85rem'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

            # Input summary chart
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'><h3 style='color:#ffffff; margin:0'>Input Summary</h3></div>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('#1c2128')
    ax.set_facecolor('#1c2128')

    variables = ['Temperature\n(°C)', 'Humidity\n(%)', 'Cloud Cover\n(%)']
    values = [t2m, rh2m, cloud_amt]
    colors = ['#f87171', '#60a5fa', '#94a3b8']
    bars = ax.bar(variables, values, color=colors, edgecolor='#30363d', linewidth=0.5)

    for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{val:.1f}', ha='center', va='bottom',
                       color='white', fontsize=9, fontweight='bold')

    ax.set_ylim(0, 110)
    ax.tick_params(colors='#8b949e')
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['left'].set_color('#30363d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.label.set_color('#8b949e')
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.markdown("""
            <div style='background:#1c2128; border:1px dashed #30363d;
                        border-radius:16px; padding:60px 30px; text-align:center;'>
                <h2 style='color:#30363d; margin:0;'>☀️</h2>
                <p style='color:#8b949e; margin:10px 0 0 0;'>
                    Adjust the sliders and click<br>
                    <b style='color:#f6c90e'>Predict Solar Irradiance</b><br>
                    to see the GHI prediction
                </p>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 3 — EDA PLOTS
# ════════════════════════════════════════════════════════════

#elif page == "📊 EDA Plots":

    st.markdown("""
    <h1 style='color:#f6c90e; margin-bottom:5px;'>📊 Exploratory Data Analysis</h1>
    <p style='color:#8b949e;'>Statistical analysis and visualizations of the Kitwe solar irradiance dataset (2020–2025).</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Dataset statistics
    st.markdown("<div class='section-header'><h3 style='color:#ffffff; margin:0'>Dataset Statistics</h3></div>", unsafe_allow_html=True)

    stats = df[['GHI', 'T2M', 'RH2M', 'CLOUD_AMT']].describe().round(4)
    st.dataframe(
        stats.style.background_gradient(cmap='YlOrRd', axis=1),
        use_container_width=True
    )

    st.markdown("---")

    # Correlation Heatmap
    st.markdown("<div class='section-header'><h3 style='color:#ffffff; margin:0'>Correlation Heatmap</h3></div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e; font-size:0.85rem'>Pearson correlation coefficients between GHI and predictor variables.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('#1c2128')
        ax.set_facecolor('#1c2128')
        corr = df[['GHI', 'T2M', 'RH2M', 'CLOUD_AMT']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax,
                   linewidths=0.5, linecolor='#30363d',
                   annot_kws={'color': 'white', 'fontsize': 11})
        ax.tick_params(colors='#c9d1d9')
        plt.title('Correlation Heatmap — Kitwe Solar Irradiance',
                 color='#f6c90e', pad=15)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        correlations = [
            ("GHI ↔ T2M", 0.33, "#4ade80"),
            ("GHI ↔ RH2M", -0.54, "#f87171"),
            ("GHI ↔ CLOUD_AMT", -0.51, "#f87171"),
            ("RH2M ↔ CLOUD_AMT", 0.61, "#fbbf24"),
        ]
        for name, val, color in correlations:
            st.markdown(f"""
            <div style='background:#1c2128; border:1px solid #30363d;
                        border-radius:8px; padding:10px 14px; margin:6px 0;'>
                <span style='color:#c9d1d9; font-size:0.8rem'>{name}</span><br>
                <span style='color:{color}; font-size:1.1rem; font-weight:700'>{val}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Distribution Plots
    st.markdown("<div class='section-header'><h3 style='color:#ffffff; margin:0'>Variable Distributions</h3></div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e; font-size:0.85rem'>Frequency distributions for all variables across the 2,190-day dataset.</p>", unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.patch.set_facecolor('#1c2128')

    plot_config = [
        ('GHI', 'orange', 'GHI Distribution (kWh/m²/day)', axes[0, 0]),
        ('T2M', '#f87171', 'Temperature Distribution (°C)', axes[0, 1]),
        ('RH2M', '#60a5fa', 'Humidity Distribution (%)', axes[1, 0]),
        ('CLOUD_AMT', '#94a3b8', 'Cloud Cover Distribution (%)', axes[1, 1]),
    ]

    for col, color, title, ax in plot_config:
        ax.set_facecolor('#1c2128')
        ax.hist(df[col], bins=30, color=color, edgecolor='#0e1117', linewidth=0.5)
        ax.set_title(title, color='#c9d1d9', fontsize=10)
        ax.tick_params(colors='#8b949e', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    plt.suptitle('Variable Distributions — Kitwe Solar Dataset (2020–2025)',
                color='#f6c90e', fontsize=11, y=1.02)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # Scatter Plots
    st.markdown("<div class='section-header'><h3 style='color:#ffffff; margin:0'>Predictor vs GHI Relationships</h3></div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e; font-size:0.85rem'>Scatter plots showing the relationship between each predictor and GHI.</p>", unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor('#1c2128')

    scatter_config = [
        ('T2M', 'Temperature (T2M) °C', '#f87171', axes[0]),
        ('RH2M', 'Relative Humidity (RH2M) %', '#60a5fa', axes[1]),
        ('CLOUD_AMT', 'Cloud Cover (CLOUD_AMT) %', '#94a3b8', axes[2]),
    ]

    for col, xlabel, color, ax in scatter_config:
        ax.set_facecolor('#1c2128')
        ax.scatter(df[col], df['GHI'], alpha=0.2, color=color, s=10)
        ax.set_xlabel(xlabel, color='#8b949e', fontsize=9)
        ax.set_ylabel('GHI (kWh/m²/day)', color='#8b949e', fontsize=9)
        ax.set_title(f'{col} vs GHI', color='#c9d1d9', fontsize=10)
        ax.tick_params(colors='#8b949e', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    plt.suptitle('Predictor Variables vs GHI — Kitwe Solar Dataset (2020–2025)',
                color='#f6c90e', fontsize=11, y=1.02)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # Monthly GHI trend
    st.markdown("<div class='section-header'><h3 style='color:#ffffff; margin:0'>Monthly Average GHI Trend</h3></div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e; font-size:0.85rem'>Average GHI by month showing seasonal variation across the study period.</p>", unsafe_allow_html=True)

    monthly = df.groupby('MON')['GHI'].mean().reset_index()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#1c2128')
    ax.set_facecolor('#1c2128')

    ax.plot(months, monthly['GHI'], color='#f6c90e',
            linewidth=2.5, marker='o', markersize=7,
            markerfacecolor='#f6c90e', markeredgecolor='#0e1117')
    ax.fill_between(range(12), monthly['GHI'],
                   alpha=0.15, color='#f6c90e')

    ax.set_xticks(range(12))
    ax.set_xticklabels(months, color='#8b949e')
    ax.set_ylabel('Average GHI (kWh/m²/day)', color='#8b949e')
    ax.set_title('Monthly Average GHI — Kitwe (2020–2025)',
                color='#f6c90e', pad=15)
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.grid(axis='y', color='#30363d', linestyle='--', alpha=0.5)

    plt.tight_layout()
    st.pyplot(fig)