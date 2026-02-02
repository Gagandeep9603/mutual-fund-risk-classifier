import streamlit as st
import pandas as pd
import joblib
import json
import seaborn as sns
import matplotlib.pyplot as plt

st.write("✅ App started loading...")

# Safe startup loading of required files
try:
    metrics_df = pd.read_csv("model_metrics.csv")

    with open("evaluation_details.json", "r") as f:
        eval_details = json.load(f)

    feature_order = pd.read_csv("feature_order.csv").iloc[:, 0].tolist()
except Exception as e:
    st.error("Startup failed: Missing or ugit add app.pynreadable required file")
    st.error(e)
    st.stop()

# -------------------------
# Load models and scaler
# -------------------------
try:
    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "KNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl"),
        "XGBoost": joblib.load("model/xgboost.pkl")
    }
    scaler = joblib.load("model/scaler.pkl")
except Exception as e:
    st.error(f"Error loading models or scaler: {e}")
    st.stop()

# Risk label mapping
risk_labels = {0: "High", 1: "Low", 2: "Moderate", 3: "Very High"}

# -------------------------
# Streamlit UI
# -------------------------
st.title("Mutual Fund Risk Classification")

st.write(
    "This app classifies mutual funds into risk categories "
    "(Low, Moderate, High, Very High) using machine learning models."
)

page = st.sidebar.radio("Choose Page", ["Model Evaluation", "Predict New Data"])

if page == "Model Evaluation":
    st.subheader("Evaluation Metrics (All Models)")
    st.dataframe(metrics_df)

    st.markdown("---")
    st.subheader("Select Model for Detailed Evaluation")
    model_name = st.selectbox("Choose a model", ["Select a model"] + list(models.keys()))

    if model_name != "Select a model":
        model_metrics = metrics_df[metrics_df["Model"] == model_name].iloc[0]

        st.markdown("### Model Performance Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", round(model_metrics["Accuracy"], 3))
            st.metric("Precision", round(model_metrics["Precision"], 3))
        with col2:
            st.metric("Recall", round(model_metrics["Recall"], 3))
            st.metric("F1 Score", round(model_metrics["F1 Score"], 3))
        with col3:
            st.metric("AUC", round(model_metrics["AUC"], 3))
            st.metric("MCC", round(model_metrics["MCC"], 3))

        st.subheader("Confusion Matrix")
        cm = pd.DataFrame(eval_details[model_name]["confusion_matrix"])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        st.subheader("Classification Report")
        report_dict = eval_details[model_name]["classification_report"]
        report_df = pd.DataFrame(report_dict).transpose().round(3)
        st.dataframe(report_df)

if page == "Predict New Data":
    st.subheader("Upload Test Data for Prediction")
    # Provide sample test dataset for users
    with open("data/test_data.csv", "rb") as f:
        st.download_button(
            label="Download Sample Test Data (CSV)",
            data=f,
            file_name="test_data.csv",
            mime="text/csv"
        )
    uploaded_file = st.file_uploader("Upload test CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        if "Risk_encoded" in data.columns:
            y_true = data["Risk_encoded"]
            X_input = data.drop(columns=["Risk_encoded"])
        else:
            y_true = None
            X_input = data

        # Enforce same feature order as training
        try:
            X_input = X_input[feature_order]
        except KeyError as e:
            st.error(f"Uploaded data is missing required feature(s): {e}")
            st.stop()

        # Scale ONLY feature columns
        try:
            X_scaled = scaler.transform(X_input)
        except Exception as e:
            st.error(f"Feature mismatch or scaling error: {e}")
            st.stop()

        model_name = st.selectbox("Select Model for Prediction", list(models.keys()))
        model = models[model_name]

        # Predictions
        y_pred = model.predict(X_scaled)

        predictions = pd.DataFrame({
            "Predicted Risk": [risk_labels[p] for p in y_pred]
        })

        st.subheader("Predictions")
        st.dataframe(predictions)

        # Download predictions as CSV
        csv = predictions.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )