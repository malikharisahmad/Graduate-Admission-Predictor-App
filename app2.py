# admission_app.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="🎓 Graduate Admission Predictor", layout="wide")

# ------------------------- STYLING ----------------------------
st.markdown("""
    <style>
    .main { background-color: #f2f2f2; }
    h1, h2, h3 { color: #2C3E50; font-weight: 700; }
    .stButton>button { background-color: #2ECC71; color: white; font-weight: bold; padding: 0.75em 2em; }
    .stTabs [data-baseweb="tab"] { font-size: 1.1rem; font-weight: bold; padding: 10px; }
    .css-1aumxhk { background-color: white; border-radius: 15px; padding: 2rem; box-shadow: 0 0 15px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 Graduate Admission Predictor Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("admission_data.csv")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

@st.cache_data
def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return scaler, X_scaled

def train_models(X_scaled, y):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Support Vector Machine": SVR()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        cv_score = cross_val_score(model, X_scaled, y, cv=5).mean()
        results[name] = {
            "MSE": mse,
            "R^2 Score": r2,
            "Cross-Val Score": cv_score,
            "Model": model
        }

    return results

# Tabs: Trainer and Predictor
tab1, tab2 = st.tabs(["📊 Model Trainer", "🎯 Predictor"])

# ------------------------ TAB 1: TRAINER ------------------------
with tab1:
    st.header("📊 Train and Visualize Admission Prediction Models")

    df = load_data()
    X = df.drop("Chance of Admit", axis=1)
    y = df["Chance of Admit"]
    scaler, X_scaled = scale_features(X)

    st.subheader("🔍 Exploratory Data Analysis")
    st.write("Understanding the dataset before training models:")

    # Correlation Heatmap
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

    # Plotly Scatter Matrix (Pairplot Alternative)
    st.subheader("🔗 Feature Relationships (Scatter Matrix)")
    fig_scatter = px.scatter_matrix(
        df,
        dimensions=["GRE Score", "TOEFL Score", "CGPA", "Chance of Admit"],
        color="Chance of Admit",
        title="Feature Relationships"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Feature Distributions
    st.subheader("📊 Feature Distributions")
    fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
    sns.histplot(df["Chance of Admit"], kde=True, bins=20, ax=ax_dist, color='mediumseagreen')
    plt.title("Distribution of Chance of Admit")
    plt.xlabel("Chance of Admit")
    st.pyplot(fig_dist)

    if st.button("🚀 Train Models", use_container_width=True):
        results = train_models(X_scaled, y)
        results_df = pd.DataFrame({k: v for k, v in results.items()}).T.drop(columns='Model')
        best_model_name = results_df["R^2 Score"].idxmax()
        best_model = results[best_model_name]["Model"]

        with open("admission_model.pkl", "wb") as f:
            pickle.dump((scaler, best_model), f)

        st.success(f"✅ Best Performing Model: **{best_model_name}**")

        st.subheader("📋 Evaluation Metrics")
        st.dataframe(results_df.style.background_gradient(cmap="coolwarm").format("{:.3f}"))

        st.subheader("📈 Model Performance Comparison")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        results_df[["MSE", "R^2 Score", "Cross-Val Score"]].plot(kind='bar', ax=ax1)
        plt.title("Model Comparison", fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig1)

        if best_model_name == "Random Forest":
            st.subheader("🧠 Feature Importance (Random Forest)")
            feat_series = pd.Series(best_model.feature_importances_, index=X.columns).sort_values()
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            feat_series.plot(kind='barh', color='dodgerblue', ax=ax2)
            plt.title("Feature Importance")
            plt.xlabel("Importance")
            plt.tight_layout()
            st.pyplot(fig2)

# ------------------------ TAB 2: PREDICTOR ------------------------
with tab2:
    st.header("🎯 Predict Your Admission Probability")

    try:
        with open("admission_model.pkl", "rb") as f:
            scaler, model = pickle.load(f)

        st.markdown("### 🔧 Input Your Profile")

        col1, col2 = st.columns(2)
        with col1:
            gre = st.slider("GRE Score (260 - 340)", 260, 340, 300)
            toefl = st.slider("TOEFL Score (0 - 120)", 0, 120, 100)
            univ_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])
        with col2:
            sop = st.slider("SOP Strength (1.0 - 5.0)", 1.0, 5.0, 3.0, step=0.5)
            lor = st.slider("LOR Strength (1.0 - 5.0)", 1.0, 5.0, 3.0, step=0.5)
            cgpa = st.slider("CGPA (out of 10)", 0.0, 10.0, 8.0, step=0.1)

        research = st.radio("Do you have Research Experience?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        input_data = np.array([[gre, toefl, univ_rating, sop, lor, cgpa, research]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prediction = np.clip(prediction, 0, 1)
        percent = round(prediction * 100, 2)

        if st.button("🎯 Predict Admission Chance", use_container_width=True):
            st.subheader(f"🔍 Estimated Admission Chance: **{percent}%**")
            st.progress(min(int(percent), 100))

            if percent > 80:
                st.success("💡 Excellent chances! Highly likely to get admitted.")
            elif percent > 60:
                st.info("🙂 Good chances, but not guaranteed.")
            else:
                st.warning("⚠️ Consider improving your profile for better chances.")

    except FileNotFoundError:
        st.error("🛑 Model not found. Please train it first in the 'Model Trainer' tab.")

st.markdown("---")
st.caption("📌 Project by Haris Ahmad, Muhammad Rehan, Saad Jamil")
