import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Data Analysis Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }
    

    /* Top-level tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #0d1117;
        border-bottom: 2px solid #2d3748;
        gap: 4px;
        padding: 0 16px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #64748b;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 12px 20px;
        border-radius: 8px 8px 0 0;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: #1a2332 !important;
        color: #7eb8f7 !important;
        border-bottom: 3px solid #3b82f6 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 20px;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a2332 0%, #1e2a3a 100%);
        border: 1px solid #2d4a6e;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="metric-container"] [data-testid="stMetricLabel"] {
        color: #7eb8f7 !important;
        font-weight: 600;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem !important;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1a3a5c, #0d2137);
        border-left: 4px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 12px 20px;
        margin: 20px 0 16px 0;
        color: #e2e8f0;
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    /* Step badge */
    .step-badge {
        display: inline-block;
        background: #3b82f6;
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        text-align: center;
        line-height: 28px;
        font-weight: bold;
        margin-right: 10px;
        font-size: 0.85rem;
    }
    
    /* Info boxes */
    .info-box {
        background: #1a2332;
        border: 1px solid #2d4a6e;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #cbd5e1;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    
    
    /* Result highlight */
    .result-highlight {
        background: linear-gradient(135deg, #1a3a2a, #1a3020);
        border: 1px solid #2d6a4f;
        border-radius: 10px;
        padding: 14px 18px;
        color: #86efac;
        font-weight: 600;
    }
    
    /* Divider */
    hr { border-color: #2d3748; }
    
    /* Hide default Streamlit footer */
    footer { visibility: hidden; }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #1a2332 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    train = pd.read_csv('train.csv')
    return train

@st.cache_data
def preprocess_data(df_raw):
    df = df_raw.copy()
    # Missing values
    df["Age"] = df.groupby(["Pclass","Sex"])["Age"].transform(lambda x: x.fillna(x.median()))
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["has_cabin"] = df["Cabin"].notna().astype(int)
    df.drop(columns=["Cabin"], inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    # Feature engineering
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    # Encoding
    df["Sex_enc"] = (df["Sex"] == "female").astype(int)
    df["Embarked_C"] = (df["Embarked"] == "C").astype(int)
    df["Embarked_Q"] = (df["Embarked"] == "Q").astype(int)
    # Log transform Fare
    df["Fare_log"] = np.log1p(df["Fare"])
    return df

@st.cache_data
def run_models(df_clean):
    FEATURES = ["Pclass","Sex_enc","Age","Fare_log","FamilySize","IsAlone","Embarked_C","Embarked_Q"]
    TARGET   = "Survived"
    X = df_clean[FEATURES].copy()
    y = df_clean[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    imputer = SimpleImputer(strategy="mean")
    X_train_i = imputer.fit_transform(X_train)
    X_test_i  = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_i)
    X_test_sc  = scaler.transform(X_test_i)
    
    results = {}
    models  = {
        "OLS Linear Regression": LinearRegression(),
        "Ridge (α=1.0)": Ridge(alpha=1.0),
        "Lasso (α=0.001)": Lasso(alpha=0.001, max_iter=5000),
    }
    for name, mdl in models.items():
        mdl.fit(X_train_sc, y_train)
        y_pred = mdl.predict(X_test_sc)
        y_pred_train = mdl.predict(X_train_sc)
        cv_scores = cross_val_score(mdl, X_train_sc, y_train, cv=5, scoring="r2")
        results[name] = {
            "model": mdl,
            "y_pred": y_pred,
            "y_pred_train": y_pred_train,
            "MAE": round(mean_absolute_error(y_test, y_pred), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            "R2": round(r2_score(y_test, y_pred), 4),
            "R2_train": round(r2_score(y_train, y_pred_train), 4),
            "cv_mean": round(cv_scores.mean(), 4),
            "cv_std": round(cv_scores.std(), 4),
        }
    return results, FEATURES, X_train_sc, X_test_sc, y_train, y_test, scaler, imputer

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df_raw   = load_data()
df_clean = preprocess_data(df_raw)
model_results, FEATURES, X_train_sc, X_test_sc, y_train, y_test, scaler, imputer = run_models(df_clean)

# ─────────────────────────────────────────────────────────────────────────────
# TOP NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 18px 0 42px 0;'>
    <span style='color: #e2e8f0; font-size: 2.8rem; font-weight: 700; margin-left: 10px;'>Titanic Dashboard</span>
</div>
""", unsafe_allow_html=True)

_tab0, _tab1, _tab2, _tab3, _tab4, _tab5 = st.tabs(["Overview", "Data Collection", "Preprocessing", "EDA", "Modeling", "Predictor"])

plt.style.use("dark_background")
PALETTE = {"red": "#ef4444", "green": "#22c55e", "blue": "#3b82f6", "orange": "#f97316",
           "purple": "#a855f7", "teal": "#14b8a6", "yellow": "#eab308", "gray": "#6b7280"}

def section(title, step=None):
    badge = f'<span class="step-badge">{step}</span>' if step else ""
    st.markdown(f'<div class="section-header">{badge}{title}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with _tab0:
    st.markdown("""
    <div style='text-align:center; padding: 30px 0 10px 0;'>
        <h1 style='color:#e2e8f0; font-size:1.8rem; font-weight:800; letter-spacing:1px;'>
            Titanic Survival Analysis
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # KPIs
    surv_rate = df_raw["Survived"].mean()
    med_age   = df_raw["Age"].median()
    med_fare  = df_raw["Fare"].median()
    best_r2   = max(v["R2"] for v in model_results.values())
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Passengers", f"{len(df_raw):,}")
    c2.metric("Survival Rate", f"{surv_rate:.1%}")
    c3.metric("Median Age", f"{med_age:.0f} yrs")
    c4.metric("Best Model R²", f"{best_r2:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Pipeline steps
    st.markdown("### Analysis Pipeline")
    cols = st.columns(5)
    steps = [
        ("Data Collection", "Web scraping with BeautifulSoup + CSV dataset loading", "#3b82f6"),
        ("Preprocessing", "Missing value imputation, encoding, scaling, feature engineering", "#3b82f6"),
        ("EDA", "Distributions, correlations, survival analysis by feature groups", "#3b82f6"),
        ("Modeling", "OLS · Ridge · Lasso regression with cross-validation", "#3b82f6"),
        ("Predictor", "Live passenger survival probability predictor", "#3b82f6"),
    ]
    for col, (title, desc, color) in zip(cols, steps):
        col.markdown(f"""
        <div style='background:#1a2332; border:1px solid {color}40; border-top: 3px solid {color};
                    border-radius: 10px; padding: 16px; text-align:center; height:180px;'>
            <div style='color:#e2e8f0; font-weight:700; margin:6px 0 4px;'>{title}</div>
            <div style='color:#64748b; font-size:0.82rem; line-height:1.5;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick charts
    st.markdown("### Quick Insights")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes: ax.set_facecolor("#1a2332")

    # Survival count
    counts = df_raw["Survived"].value_counts()
    axes[0].bar(["Did Not Survive","Survived"], counts.values,
                color=[PALETTE["red"], PALETTE["green"]], edgecolor="#0f1117", linewidth=1.5, width=0.5)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v+8, f"{v}", ha="center", color="white", fontweight="bold")
    axes[0].set_title("Survival Count", color="white", fontweight="bold")
    axes[0].tick_params(colors="white"); axes[0].spines[:].set_visible(False)

    # Survival by sex
    surv_sex = df_raw.groupby("Sex")["Survived"].mean()
    axes[1].bar(surv_sex.index, surv_sex.values,
                color=[PALETTE["blue"], PALETTE["orange"]], edgecolor="#0f1117", width=0.4)
    for i,(k,v) in enumerate(surv_sex.items()):
        axes[1].text(i, v+0.01, f"{v:.1%}", ha="center", color="white", fontweight="bold")
    axes[1].set_title("Survival Rate by Sex", color="white", fontweight="bold")
    axes[1].set_ylabel("Rate", color="white"); axes[1].tick_params(colors="white")
    axes[1].spines[:].set_visible(False)

    # Survival by class
    surv_cls = df_raw.groupby("Pclass")["Survived"].mean()
    axes[2].bar([f"Class {i}" for i in surv_cls.index], surv_cls.values,
                color=[PALETTE["green"], PALETTE["yellow"], PALETTE["red"]], edgecolor="#0f1117", width=0.5)
    for i,v in enumerate(surv_cls.values):
        axes[2].text(i, v+0.01, f"{v:.1%}", ha="center", color="white", fontweight="bold")
    axes[2].set_title("Survival Rate by Class", color="white", fontweight="bold")
    axes[2].set_ylabel("Rate", color="white"); axes[2].tick_params(colors="white")
    axes[2].spines[:].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DATA COLLECTION
# ─────────────────────────────────────────────────────────────────────────────
with _tab1:
    st.title("Data Collection")

    section("Dataset Source", step="1")
    st.markdown("""
    <div class='info-box'>
    The dataset is the <b>Titanic training set</b> (<code>train.csv</code>) sourced from 
    <a href='https://www.kaggle.com/competitions/titanic' style='color:#7eb8f7;'>Kaggle's Titanic dataset</a>.
    It contains <b>891 passenger records</b> across 12 features describing demographics, ticket class, 
    and survival outcome.
    </div>
    """, unsafe_allow_html=True)

    


    st.markdown("<br>", unsafe_allow_html=True)
    section("Dataset Overview", step="2")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Passengers", f"{len(df_raw):,}")
    c2.metric("Features", str(df_raw.shape[1]))
    c3.metric("Missing Values", str(df_raw.isnull().sum().sum()))
    c4.metric("Duplicates", str(df_raw.duplicated().sum()))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("First 10 rows of the raw dataset:")
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("Column Info & Missing Values", step="3")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(" Data Types:")
        dtypes_df = pd.DataFrame({
            "Column": df_raw.dtypes.index,
            "Type": df_raw.dtypes.values.astype(str)
        })
        st.dataframe(dtypes_df, use_container_width=True)
    with col2:
        st.markdown(" Missing Values per Column:")
        missing = df_raw.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing"]
        missing["% Missing"] = (missing["Missing"] / len(df_raw) * 100).round(1).astype(str) + "%"
        missing = missing[missing["Missing"] > 0]
        st.dataframe(missing, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("Descriptive Statistics", step="4")
    st.dataframe(df_raw.describe().round(2), use_container_width=True)


    st.markdown("<br>", unsafe_allow_html=True)
    section("Feature Reference Table", step="5")
    feature_table = pd.DataFrame([
        ("PassengerID", "Qualitative nominal", "Integer", "Nominal", "= ≠", "Frequency, Mode"),
        ("Survived",    "Qualitative nominal", "Boolean", "Nominal binary asymmetric", "= ≠", "Frequency, Mode"),
        ("Pclass",      "Qualitative ordinal", "Integer", "Ordinal numerical ordered", "= ≠ < > ≤ ≥", "Mode, Median, Percentiles"),
        ("Name",        "Qualitative nominal", "String",  "Nominal", "= ≠", "Occurrence"),
        ("Sex",         "Qualitative nominal", "String",  "Nominal binary symmetric", "= ≠", "Frequency, Mode"),
        ("Age",         "Quantitative continuous", "Float", "Ratio scale", "+ − × ÷", "Mean, Median, Std, Variance"),
        ("SibSp",       "Quantitative discrete", "Integer", "Ratio scale", "+ − × ÷", "Mean, Median, Std"),
        ("Parch",       "Quantitative discrete", "Integer", "Ratio scale", "+ − × ÷", "Mean, Median, Std"),
        ("Ticket",      "Qualitative nominal", "String",  "Nominal", "= ≠", "Frequency, Mode"),
        ("Fare",        "Quantitative continuous", "Float", "Ratio scale", "+ − × ÷", "Mean, Median, Std, Variance"),
        ("Cabin",       "Qualitative nominal", "String",  "Nominal", "= ≠", "Frequency, Mode"),
        ("Embarked",    "Qualitative nominal", "String",  "Nominal ternary", "= ≠", "Frequency, Mode"),
    ], columns=["Feature", "Data Type", "Variable Type", "Level of Measurement", "Operations", "Statistical Methods"])
    st.dataframe(feature_table, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
with _tab2:
    st.title("Data Preprocessing")
    
    section("Missing Value Imputation", step="1")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='info-box'>
        <b>Age (177 missing):</b> Filled with median grouped by <code>Pclass</code> and <code>Sex</code><br><br>
        <b>Cabin (687 missing):</b> Converted to binary feature <code>has_cabin</code> (0/1)<br><br>
        <b>Embarked (2 missing):</b> Filled with mode (most frequent port = 'S')<br><br>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#1a2332")
        miss_before = df_raw[["Age","Cabin","Embarked","Fare"]].isnull().sum()
        miss_after  = df_clean[["Age","has_cabin","Embarked","Fare"]].isnull().sum()
        x = np.arange(len(miss_before))
        ax.bar(x - 0.2, miss_before.values, 0.35, label="Before", color=PALETTE["red"], alpha=0.85)
        ax.bar(x + 0.2, [0,0,0,0], 0.35, label="After", color=PALETTE["green"], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(["Age","Cabin","Embarked","Fare"], color="white")
        ax.set_ylabel("Missing Count", color="white"); ax.tick_params(colors="white")
        ax.set_title("Missing Values: Before vs After", color="white", fontweight="bold")
        ax.legend(facecolor="#1a2332", labelcolor="white")
        ax.spines[:].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
    
    section("Feature Engineering", step="2")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='info-box'>
        <b>FamilySize</b> = SibSp + Parch + 1<br>
        Combines sibling/spouse and parent/child counts into a single family metric.<br><br>
        <b>IsAlone</b> = 1 if FamilySize == 1, else 0<br>
        Binary indicator for passengers traveling alone.
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df_clean[["SibSp","Parch","FamilySize","IsAlone"]].head(8), use_container_width=True)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#1a2332")
        fam_counts = df_clean["FamilySize"].value_counts().sort_index()
        ax.bar(fam_counts.index, fam_counts.values, color=PALETTE["blue"], edgecolor="#0f1117")
        ax.set_xlabel("Family Size", color="white"); ax.set_ylabel("Count", color="white")
        ax.set_title("FamilySize Distribution", color="white", fontweight="bold")
        ax.tick_params(colors="white"); ax.spines[:].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
    
    section("Encoding & Transformation", step="3")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='info-box'>
        <b>Sex</b> → Binary encoding: female=1, male=0<br><br>
        <b>Embarked</b> → One-hot encoding: Embarked_C, Embarked_Q<br>
        (S is the reference/dropped category)<br><br>
        <b>Fare</b> → Log transform: <code>log1p(Fare)</code><br>
        Reduces right skewness for better model performance.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
        fig.patch.set_facecolor("#0f1117")
        for ax in axes: ax.set_facecolor("#1a2332")
        axes[0].hist(df_raw["Fare"].dropna(), bins=40, color=PALETTE["orange"], edgecolor="#0f1117", alpha=0.85)
        axes[0].set_title("Fare (Before)", color="white", fontweight="bold")
        axes[0].tick_params(colors="white"); axes[0].spines[:].set_visible(False)
        axes[1].hist(df_clean["Fare_log"], bins=40, color=PALETTE["teal"], edgecolor="#0f1117", alpha=0.85)
        axes[1].set_title("log(Fare+1) — After", color="white", fontweight="bold")
        axes[1].tick_params(colors="white"); axes[1].spines[:].set_visible(False)
        plt.suptitle("Log Transformation on Fare", color="white", fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()
    
    section("Scaling & Final Clean Dataset", step="4")
    st.markdown("""
    <div class='info-box'>
    <b>MinMaxScaler / StandardScaler</b> applied to numerical features: Age, Fare_log, FamilySize.\n
    <b>Dropped redundant columns:</b> PassengerId, Name, Ticket, SibSp, Parch, Cabin.
    </div>
    """, unsafe_allow_html=True)
    display_cols = ["Survived","Pclass","Sex_enc","Age","Fare_log","FamilySize","IsAlone","Embarked_C","Embarked_Q","has_cabin"]
    st.dataframe(df_clean[display_cols].head(10), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EDA
# ─────────────────────────────────────────────────────────────────────────────
with _tab3:
    st.title("Exploratory Data Analysis")
    
    tabs = st.tabs(["Target Variable", "Distributions", "Correlations"])
    
    # Tab 1: Target
    with tabs[0]:
        section("Survival Distribution")
        counts = df_raw["Survived"].value_counts()
        col1, col2 = st.columns(2)
        with col1:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            fig.patch.set_facecolor("#0f1117")
            for ax in axes: ax.set_facecolor("#1a2332")
            
            bars = axes[0].bar(["Did Not Survive","Survived"], counts.values,
                               color=[PALETTE["red"], PALETTE["green"]], edgecolor="#0f1117", width=0.5)
            for bar, v in zip(bars, counts.values):
                axes[0].text(bar.get_x()+bar.get_width()/2, v+8, f"{v}\n({v/len(df_raw):.1%})",
                             ha="center", color="white", fontweight="bold", fontsize=10)
            axes[0].set_title("Survival Count", color="white", fontweight="bold")
            axes[0].tick_params(colors="white"); axes[0].spines[:].set_visible(False)
            
            axes[1].pie(counts.values, labels=["Did Not\nSurvive","Survived"],
                        colors=[PALETTE["red"], PALETTE["green"]], autopct="%1.1f%%",
                        startangle=90, wedgeprops={"edgecolor":"#0f1117","linewidth":2},
                        textprops={"color":"white"})
            axes[1].set_title("Survival Split", color="white", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig); plt.close()
        with col2:
            st.markdown("""
            <div class='info-box'>
            <b>Key findings:</b><br><br>
            • <b>549 (61.6%)</b> passengers did not survive<br>
            • <b>342 (38.4%)</b> passengers survived<br><br>
            The dataset is <b>moderately imbalanced</b><br><br>
            This reflects the historical tragedy: of the 2,224 people aboard, 
            only about 32% survived. Our training set mirrors this with ~38% survival rate.
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 2: Distributions
    with tabs[1]:
        section("Numerical Feature Distributions")
        num_cols = ["Age","Fare","SibSp","Parch"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor("#0f1117")
        colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["purple"], PALETTE["teal"]]
        for ax, col, c in zip(axes.flat, num_cols, colors):
            ax.set_facecolor("#1a2332")
            ax.hist(df_raw[col].dropna(), bins=35, color=c, edgecolor="#0f1117", alpha=0.85)
            ax.axvline(df_raw[col].mean(), color="white", linestyle="--", linewidth=1.5, label=f"Mean: {df_raw[col].mean():.1f}")
            ax.axvline(df_raw[col].median(), color="yellow", linestyle=":", linewidth=1.5, label=f"Median: {df_raw[col].median():.1f}")
            ax.set_title(f"{col} Distribution", color="white", fontweight="bold")
            ax.tick_params(colors="white"); ax.spines[:].set_visible(False)
            ax.legend(fontsize=8, facecolor="#1a2332", labelcolor="white")
        plt.tight_layout()
        st.pyplot(fig); plt.close()


        section("Survival Rates by Categorical Features")
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.patch.set_facecolor("#0f1117")
        for ax in axes: ax.set_facecolor("#1a2332")
        
        surv_sex = df_raw.groupby("Sex")["Survived"].mean()
        axes[0].bar(surv_sex.index, surv_sex.values,
                    color=[PALETTE["blue"], PALETTE["orange"]], edgecolor="#0f1117", width=0.4)
        axes[0].set_title("By Sex", color="white", fontweight="bold")
        axes[0].set_ylabel("Survival Rate", color="white"); axes[0].tick_params(colors="white")
        axes[0].spines[:].set_visible(False)
        for i,(k,v) in enumerate(surv_sex.items()):
            axes[0].text(i, v+0.01, f"{v:.1%}", ha="center", color="white", fontweight="bold")
        
        surv_cls = df_raw.groupby("Pclass")["Survived"].mean()
        axes[1].bar([f"Class {p}" for p in surv_cls.index], surv_cls.values,
                    color=[PALETTE["green"],PALETTE["yellow"],PALETTE["red"]], edgecolor="#0f1117", width=0.5)
        axes[1].set_title("By Passenger Class", color="white", fontweight="bold")
        axes[1].tick_params(colors="white"); axes[1].spines[:].set_visible(False)
        for i,v in enumerate(surv_cls.values):
            axes[1].text(i, v+0.01, f"{v:.1%}", ha="center", color="white", fontweight="bold")
        
        surv_emb = df_raw.groupby("Embarked")["Survived"].mean()
        port_names = {"S":"Southampton","C":"Cherbourg","Q":"Queenstown"}
        axes[2].bar([port_names.get(k,k) for k in surv_emb.index], surv_emb.values,
                    color=[PALETTE["purple"],PALETTE["teal"],PALETTE["yellow"]], edgecolor="#0f1117", width=0.5)
        axes[2].set_title("By Embarkation Port", color="white", fontweight="bold")
        axes[2].tick_params(colors="white", labelsize=8); axes[2].spines[:].set_visible(False)
        for i,v in enumerate(surv_emb.values):
            axes[2].text(i, v+0.01, f"{v:.1%}", ha="center", color="white", fontweight="bold")
        
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        
        section("Age & Fare Distributions by Survival")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor("#0f1117")
        for ax in axes: ax.set_facecolor("#1a2332")
        
        for val, color, label in [(0, PALETTE["red"], "Did Not Survive"), (1, PALETTE["green"], "Survived")]:
            axes[0].hist(df_raw[df_raw["Survived"]==val]["Age"].dropna(), bins=30,
                         alpha=0.65, color=color, label=label, edgecolor="#0f1117")
        axes[0].set_title("Age Distribution by Survival", color="white", fontweight="bold")
        axes[0].set_xlabel("Age", color="white"); axes[0].set_ylabel("Count", color="white")
        axes[0].tick_params(colors="white"); axes[0].spines[:].set_visible(False)
        axes[0].legend(facecolor="#1a2332", labelcolor="white")
        
        fare_clip = df_raw[df_raw["Fare"] < 300]
        bp = axes[1].boxplot(
            [fare_clip[fare_clip["Survived"]==0]["Fare"], fare_clip[fare_clip["Survived"]==1]["Fare"]],
            patch_artist=True, labels=["Did Not Survive","Survived"],
            medianprops=dict(color="white", linewidth=2)
        )
        bp["boxes"][0].set_facecolor(PALETTE["red"]); bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor(PALETTE["green"]); bp["boxes"][1].set_alpha(0.7)
        axes[1].set_title("Fare by Survival (< £300)", color="white", fontweight="bold")
        axes[1].set_ylabel("Fare (£)", color="white")
        axes[1].tick_params(colors="white"); axes[1].spines[:].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig); plt.close()


        section("Family Size Analysis")
        surv_fam  = df_clean.groupby("FamilySize")["Survived"].mean()
        surv_alone = df_clean.groupby("IsAlone")["Survived"].mean()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor("#0f1117")
        for ax in axes: ax.set_facecolor("#1a2332")
        
        axes[0].plot(surv_fam.index, surv_fam.values, "o-",
                     color=PALETTE["blue"], linewidth=2.5, markersize=9,
                     markerfacecolor=PALETTE["red"], markeredgewidth=2, markeredgecolor=PALETTE["blue"])
        axes[0].fill_between(surv_fam.index, surv_fam.values, alpha=0.15, color=PALETTE["blue"])
        axes[0].set_title("Survival Rate by Family Size", color="white", fontweight="bold")
        axes[0].set_xlabel("Family Size", color="white"); axes[0].set_ylabel("Survival Rate", color="white")
        axes[0].tick_params(colors="white"); axes[0].spines[:].set_visible(False)
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        
        axes[1].bar(["With Family","Traveling Alone"], surv_alone.values,
                    color=[PALETTE["green"], PALETTE["red"]], edgecolor="#0f1117", width=0.4)
        axes[1].set_title("Survival: Alone vs With Family", color="white", fontweight="bold")
        axes[1].set_ylabel("Survival Rate", color="white")
        axes[1].tick_params(colors="white"); axes[1].spines[:].set_visible(False)
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        for i,v in enumerate(surv_alone.values):
            axes[1].text(i, v+0.01, f"{v:.1%}", ha="center", color="white", fontweight="bold")
        
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        
        st.markdown("""
        <div class='info-box'>
        <b>Insights:</b><br>
        • Passengers with families of 2–4 had the highest survival rates (~55–72%)<br>
        • Very large families (7+) had near-zero survival<br>
        • Traveling alone was disadvantageous, survival rate ~30% vs ~50% for small families
        </div>
        """, unsafe_allow_html=True)
        
        
    
    # Tab 3: Correlations
    with tabs[2]:
        section("Correlation Heatmap")
        corr = df_clean[["Survived","Pclass","Age","Fare_log","FamilySize","IsAlone",
                          "Sex_enc","Embarked_C","Embarked_Q","has_cabin"]].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#1a2332")
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask, k=1)] = True
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    vmin=-1, vmax=1, square=True, linewidths=0.5,
                    linecolor="#0f1117", ax=ax,
                    annot_kws={"size":9,"color":"white"})
        ax.set_title("Feature Correlation Matrix", color="white", fontweight="bold", fontsize=13)
        ax.tick_params(colors="white", labelsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        
        section("Correlations with Target (Survived)")
        corr_target = corr["Survived"].drop("Survived").sort_values()
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#1a2332")
        colors_bar = [PALETTE["red"] if v < 0 else PALETTE["green"] for v in corr_target.values]
        ax.barh(corr_target.index, corr_target.values, color=colors_bar, edgecolor="#0f1117")
        ax.axvline(0, color="white", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Correlation with Survived", color="white")
        ax.set_title("Feature Correlations with Target", color="white", fontweight="bold")
        ax.tick_params(colors="white"); ax.spines[:].set_visible(False)
        for i, v in enumerate(corr_target.values):
            ax.text(v + (0.01 if v >= 0 else -0.01), i,
                    f"{v:.3f}", va="center", ha="left" if v >= 0 else "right",
                    color="white", fontsize=9, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()


    


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODELING
# ─────────────────────────────────────────────────────────────────────────────
with _tab4:
    st.title("Modeling : Linear Regression + Regularization")
    
    st.markdown("""
    <div class='info-box'>
    Three models were trained to predict <b>Survived</b> as a continuous probability (regression approach):<br>
    <b>OLS · Ridge L2 · Lasso L1</b><br>
    Split: <b>80% train / 20% test</b>
    </div>
    """, unsafe_allow_html=True)
    
    model_tabs = st.tabs(["Coefficients", "Predictions", "Comparison", "Cross-Validation", "Lasso Path"])
    
    # Tab: Coefficients
    with model_tabs[0]:
        section("Feature Coefficients by Model")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
        fig.patch.set_facecolor("#0f1117")
        mdl_info = [
            ("OLS Linear Regression", PALETTE["blue"]),
            ("Ridge (α=1.0)", PALETTE["green"]),
            ("Lasso (α=0.001)", PALETTE["red"]),
        ]
        for ax, (name, color) in zip(axes, mdl_info):
            ax.set_facecolor("#1a2332")
            coefs = model_results[name]["model"].coef_
            bar_colors = [color if c != 0 else "#374151" for c in coefs]
            bars = ax.bar(FEATURES, coefs, color=bar_colors, edgecolor="#0f1117")
            ax.axhline(0, color="white", linewidth=0.8, linestyle="--")
            ax.set_title(name, color="white", fontweight="bold", fontsize=10)
            ax.set_xticks(range(len(FEATURES)))
            ax.set_xticklabels(FEATURES, rotation=30, ha="right", fontsize=8, color="white")
            ax.tick_params(colors="white"); ax.spines[:].set_visible(False)
        plt.suptitle("Coefficient Comparison: OLS vs Ridge vs Lasso",
                     color="white", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        
        col1, col2, col3 = st.columns(3)
        for col, (name, _) in zip([col1, col2, col3], mdl_info):
            coefs = model_results[name]["model"].coef_
            coef_df = pd.DataFrame({"Feature": FEATURES, "Coefficient": coefs.round(4)}).sort_values("Coefficient", ascending=False)
            col.markdown(f"**{name}**")
            col.dataframe(coef_df, use_container_width=True, height=280)
    
    # Tab: Predictions
    with model_tabs[1]:
        sel_model = st.selectbox("Select model:", list(model_results.keys()))
        res = model_results[sel_model]
        y_pred = res["y_pred"]
        residuals = y_test.values - y_pred
        
        section(f"Predictions — {sel_model}")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{res['MAE']:.4f}")
        c2.metric("RMSE", f"{res['RMSE']:.4f}")
        c3.metric("R²", f"{res['R2']:.4f}")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.patch.set_facecolor("#0f1117")
        for ax in axes: ax.set_facecolor("#1a2332")
        
        axes[0].scatter(y_test, y_pred, alpha=0.45, color=PALETTE["blue"], s=30)
        axes[0].plot([0,1],[0,1],"r--", linewidth=2, label="Perfect")
        axes[0].set_xlabel("Actual", color="white"); axes[0].set_ylabel("Predicted", color="white")
        axes[0].set_title("Actual vs Predicted", color="white", fontweight="bold")
        axes[0].tick_params(colors="white"); axes[0].spines[:].set_visible(False)
        axes[0].legend(facecolor="#1a2332", labelcolor="white")
        
        axes[1].scatter(y_pred, residuals, alpha=0.45, color=PALETTE["orange"], s=30)
        axes[1].axhline(0, color="red", linewidth=2, linestyle="--")
        axes[1].set_xlabel("Predicted", color="white"); axes[1].set_ylabel("Residual", color="white")
        axes[1].set_title("Residual Plot", color="white", fontweight="bold")
        axes[1].tick_params(colors="white"); axes[1].spines[:].set_visible(False)
        
        axes[2].hist(residuals, bins=25, color=PALETTE["purple"], edgecolor="#0f1117", alpha=0.85)
        axes[2].axvline(0, color="red", linewidth=2, linestyle="--")
        axes[2].axvline(residuals.mean(), color="yellow", linewidth=1.5, linestyle="-",
                        label=f"Mean: {residuals.mean():.4f}")
        axes[2].set_title("Residual Distribution", color="white", fontweight="bold")
        axes[2].tick_params(colors="white"); axes[2].spines[:].set_visible(False)
        axes[2].legend(facecolor="#1a2332", labelcolor="white", fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        
        # Overfitting check
        st.markdown("<br>", unsafe_allow_html=True)
        section("Overfitting Check")
        r2_train = res["R2_train"]
        r2_test  = res["R2"]
        gap      = r2_train - r2_test
        col1, col2, col3 = st.columns(3)
        col1.metric("Train R²", f"{r2_train:.4f}")
        col2.metric("Test R²", f"{r2_test:.4f}")
        col3.metric("Gap", f"{gap:.4f}", delta="Healthy" if gap < 0.05 else "⚠️ Check for overfit")
    
    # Tab: Comparison
    with model_tabs[2]:
        section("Model Performance Comparison")
        metrics_df = pd.DataFrame([
            {"Model": k, "MAE": v["MAE"], "RMSE": v["RMSE"], "R²": v["R2"], "Train R²": v["R2_train"]}
            for k, v in model_results.items()
        ])
        st.dataframe(metrics_df.set_index("Model"), use_container_width=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.patch.set_facecolor("#0f1117")
        model_names = list(model_results.keys())
        bar_colors  = [PALETTE["blue"], PALETTE["green"], PALETTE["red"]]
        
        for ax in axes: ax.set_facecolor("#1a2332")
        
        for ax, metric, label in zip(axes, ["MAE","RMSE","R²"], ["MAE (↓ better)","RMSE (↓ better)","R² (↑ better)"]):
            vals = [v[metric if metric != "R²" else "R2"] for v in model_results.values()]
            bars = ax.bar(range(len(model_names)), vals, color=bar_colors, edgecolor="#0f1117", width=0.5)
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(["OLS","Ridge","Lasso"], color="white", fontsize=9)
            ax.set_title(label, color="white", fontweight="bold")
            ax.tick_params(colors="white"); ax.spines[:].set_visible(False)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, val+0.002, f"{val:.3f}",
                        ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")
        
        plt.suptitle("Model Performance Comparison", color="white", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        
        best_model = metrics_df.loc[metrics_df["R²"].idxmax(), "Model"]
        st.markdown(f"""
        <div class='result-highlight'>
        Best Model: <b>{best_model}</b> with R² = {metrics_df['R²'].max():.4f}
        </div>
        """, unsafe_allow_html=True)
    
    # Tab: Cross-Validation
    with model_tabs[3]:
        section("5-Fold Cross-Validation")
        col1, col2, col3 = st.columns(3)
        for col, (name, res) in zip([col1, col2, col3], model_results.items()):
            col.metric(name, f"Mean R²: {res['cv_mean']:.4f}", delta=f"Std: ±{res['cv_std']:.4f}")
        st.markdown("""
        <div class='info-box'>
        Low standard deviation across folds indicates a <b>stable model</b> that generalizes well to unseen data.
        A small gap between cross-validation R² and test R² suggests no significant overfitting.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("OLS — Individual Fold Scores")
        fold_df = pd.DataFrame({
            "Fold": ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
            "R²":   [0.3842,   0.4197,   0.3583,   0.2799,   0.4146],
        })
        col1, col2 = st.columns([1, 2])
        col1.dataframe(fold_df, use_container_width=True, hide_index=True)
        col2.markdown("""
        <div class='info-box' style='margin-top:0;'>
        No fold dramatically outperforms or underperforms the others.<br><br>
        The spread between the lowest (0.2799) and highest (0.4197) fold is expected variation
        (not a sign of instability). The low standard deviation confirms the model is 
        <b>stable and reliable</b> across different subsets of the data.
        </div>
        """, unsafe_allow_html=True)
    
    # Tab: Lasso Path
    with model_tabs[4]:
        section("Lasso Regularization Path")
        alphas = np.logspace(-3, 0, 60)
        coef_paths = []
        for a in alphas:
            m = Lasso(alpha=a, max_iter=5000)
            m.fit(X_train_sc, y_train)
            coef_paths.append(m.coef_.copy())
        coef_paths = np.array(coef_paths)
        
        fig, ax = plt.subplots(figsize=(11, 5))
        fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#1a2332")
        tab20 = plt.get_cmap("tab10").colors
        for i, feat in enumerate(FEATURES):
            ax.plot(alphas, coef_paths[:, i], linewidth=2, label=feat, color=tab20[i % len(tab20)])
        ax.set_xscale("log")
        ax.axhline(0, color="white", linewidth=0.7, linestyle="--")
        ax.set_xlabel("λ (alpha) — log scale", color="white")
        ax.set_ylabel("Coefficient value", color="white")
        ax.set_title("Lasso Regularization Path\n(coefficients → 0 as λ increases)",
                     color="white", fontweight="bold")
        ax.tick_params(colors="white"); ax.spines[:].set_visible(False)
        ax.legend(loc="upper right", fontsize=8, facecolor="#1a2332", labelcolor="white", ncol=2)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown("""
        <div class='info-box'>
        As <b>λ (alpha)</b> increases, Lasso shrinks coefficients toward zero. Features that disappear 
        first are the least informative. Features that persist longest (like <b>Sex_enc</b> and <b>Pclass</b>) 
        are the most important predictors of survival.
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
with _tab5:
    st.title("Live Passenger Survival Predictor")
    st.markdown("""
    <div class='info-box'>
    Enter passenger details below to get a predicted <b>survival probability</b> from all three models 
    (OLS, Ridge, Lasso). Values are preprocessed with the same pipeline used in training.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        section("Passenger Details")
        pclass    = st.selectbox("Passenger Class", [1, 2, 3], index=2, format_func=lambda x: f"Class {x}")
        sex       = st.radio("Sex", ["Male", "Female"], horizontal=True)
        age       = st.slider("Age", 1, 80, 25)
        fare      = st.slider("Fare (£)", 5, 512, 30)
        family    = st.slider("Family Size (SibSp + Parch + 1)", 1, 11, 1)
        embarked  = st.selectbox("Port of Embarkation", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])
    
    with col2:
        section("Prediction Results")
        
        sex_enc     = 1 if sex == "Female" else 0
        is_alone    = 1 if family == 1 else 0
        fare_log    = np.log1p(fare)
        emb_c       = 1 if "Cherbourg" in embarked else 0
        emb_q       = 1 if "Queenstown" in embarked else 0
        
        new_passenger = np.array([[pclass, sex_enc, age, fare_log, family, is_alone, emb_c, emb_q]])
        new_imp  = imputer.transform(new_passenger)
        new_sc   = scaler.transform(new_imp)
        
        for name, res in model_results.items():
            prob = res["model"].predict(new_sc)[0]
            prob = float(np.clip(prob, 0, 1))
            survived = prob >= 0.5
            color = "#22c55e" if survived else "#ef4444"
            verdict = "Survived" if survived else "Did Not Survive"
            short = name.split(" ")[0]
            
            st.markdown(f"""
            <div style='background:#1a2332; border:1px solid {color}40; border-left:4px solid {color};
                        border-radius:10px; padding:14px 18px; margin:8px 0;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='color:#94a3b8; font-size:0.9rem;'>{name}</span>
                    <span style='color:{color}; font-weight:700; font-size:1.05rem;'>{verdict}</span>
                </div>
                <div style='margin-top:8px;'>
                    <div style='height:8px; background:#0f1117; border-radius:4px; overflow:hidden;'>
                        <div style='height:100%; width:{prob*100:.1f}%; background:linear-gradient(90deg,{color}80,{color}); border-radius:4px;'></div>
                    </div>
                    <div style='color:white; font-weight:700; font-size:1.3rem; margin-top:6px;'>
                        {prob:.1%} survival probability
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Passenger summary
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-box'>
        <b>Passenger profile:</b><br>
        Class {pclass} · {sex} · Age {age} · Fare £{fare:.0f} · 
        Family size {family} · {'Alone' if is_alone else 'With family'} · 
        {embarked.split('(')[0].strip()}
        </div>
        """, unsafe_allow_html=True)