import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from models.train_models import load_and_train_models
from utils.gemini_helper import generate_ai_analysis
from utils.shap_analysis import generate_shap_plot
from utils.plots import generate_roc_curve
from utils.preprocessing import prepare_input_data
import warnings

warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Heart Disease Detector",
    page_icon="❤️",
    layout="wide"
)


st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.7rem;
        color: #e74c3c;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .risk-high {
        background: #FCEBEB;
        border: 2px solid #e74c3c;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }
    .risk-low {
        background: #EAF3DE;
        border: 2px solid #27ae60;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


(
    df,
    X,
    y,
    scaler,
    trained_models,
    results,
    best_model,
    X_test,
    y_test
) = load_and_train_models()


st.markdown(
    '<div class="main-header">❤️ Heart Disease Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-header">AI-powered heart disease prediction and explainability dashboard</div>',
    unsafe_allow_html=True
)


predict_tab, comparison_tab, analysis_tab, evaluation_tab = st.tabs(
    [
        "🔍 Predict + AI Analysis",
        "📊 Model Comparison",
        "📈 Data Analysis",
        "🧠 Model Evaluation"
    ]
)


with predict_tab:

    st.subheader("Enter Patient Details")

    st.info(
        f"🏆 Best Model: {best_model} | Accuracy: {results[best_model]['accuracy']}%"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 20, 80, 45)

        sex = st.selectbox(
            "Sex",
            [0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male"
        )

        cp = st.selectbox(
            "Chest Pain Type",
            [0, 1, 2, 3],
            format_func=lambda x: [
                "Typical Angina",
                "Atypical Angina",
                "Non-Anginal",
                "Asymptomatic"
            ][x]
        )

        trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
        chol = st.slider("Cholesterol", 100, 600, 200)

    with col2:
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        restecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2],
            format_func=lambda x: [
                "Normal",
                "ST-T Abnormality",
                "LV Hypertrophy"
            ][x]
        )

        thalach = st.slider("Maximum Heart Rate", 60, 220, 150)

        exang = st.selectbox(
            "Exercise Induced Angina",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

    with col3:
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, step=0.1)

        slope = st.selectbox(
            "Slope of ST Segment",
            [0, 1, 2],
            format_func=lambda x: [
                "Upsloping",
                "Flat",
                "Downsloping"
            ][x]
        )

        ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])

        thal = st.selectbox(
            "Thal",
            [0, 1, 2, 3],
            format_func=lambda x: [
                "Normal",
                "Fixed Defect",
                "Reversible Defect",
                "Unknown"
            ][x]
        )

    selected_model_name = st.selectbox(
        "Choose Model",
        list(trained_models.keys()),
        index=list(trained_models.keys()).index(best_model)
    )

    selected_model = trained_models[selected_model_name]

    if st.button(
        "🔍 Predict & Get AI Analysis",
        use_container_width=True,
        type="primary"
    ):

        input_data = prepare_input_data(
            age, sex, cp, trestbps, chol,
            fbs, restecg, thalach, exang,
            oldpeak, slope, ca, thal
        )

        input_scaled = scaler.transform(input_data)
        prediction = selected_model.predict(input_scaled)[0]
        probability = selected_model.predict_proba(input_scaled)[0]
        risk_score = probability[1]

        st.markdown("---")
        st.subheader("Heart Disease Risk Score")
        st.progress(float(risk_score))
        st.write(f"Risk Probability: {risk_score * 100:.2f}%")

        # ✅ FIX 1: HTML ek hi line mein — newlines ki wajah se raw text aa raha tha
        if prediction == 1:
            st.markdown(
                f'<div class="risk-high"><h2 style="color:#e74c3c;">⚠️ HIGH RISK of Heart Disease</h2><h3>Confidence: {round(probability[1] * 100, 1)}%</h3></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="risk-low"><h2 style="color:#27ae60;">✅ LOW RISK of Heart Disease</h2><h3>Confidence: {round(probability[0] * 100, 1)}%</h3></div>',
                unsafe_allow_html=True
            )

        # Risk Factor Analysis
        st.markdown("### Risk Factor Analysis")

        risk_factors = []

        if age > 55:
            risk_factors.append("Age above 55 may increase cardiovascular risk.")
        if chol > 240:
            risk_factors.append("High cholesterol level detected.")
        if trestbps > 140:
            risk_factors.append("Elevated blood pressure observed.")
        if fbs == 1:
            risk_factors.append("High fasting blood sugar detected.")
        if exang == 1:
            risk_factors.append("Exercise-induced angina present.")
        if oldpeak > 2.0:
            risk_factors.append("High ST depression value observed.")
        if ca > 0:
            risk_factors.append(f"{ca} major vessel(s) with narrowing detected.")
        if thal in [1, 2]:
            risk_factors.append("Thalassemia defect detected.")

        if not risk_factors:
            risk_factors.append("No major risk factors detected.")

        for factor in risk_factors:
            st.write(f"• {factor}")

        # Gemini AI Medical Insight
        st.markdown("### Gemini AI Medical Insight")

        with st.spinner("🤖 Generating AI analysis..."):
            try:
                ai_response = generate_ai_analysis(
                    age,
                    sex,
                    trestbps,
                    chol,
                    thalach,
                    prediction,
                    round(risk_score * 100, 2)
                )
                # ✅ FIX 2: st.info → st.markdown so AI response renders properly
                st.markdown(ai_response)

            except Exception as e:
                st.error("⚠️ AI Analysis failed. Please check your Gemini API key.")
                st.caption(f"Technical error: {e}")


with comparison_tab:

    st.subheader("Model Performance Comparison")

    result_df = pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy (%)": [results[m]["accuracy"] for m in results],
        "Precision (%)": [results[m]["precision"] for m in results],
        "Recall (%)": [results[m]["recall"] for m in results],
        "F1 Score (%)": [results[m]["f1"] for m in results],
        "ROC AUC (%)": [results[m]["roc_auc"] for m in results],
        "CV Mean (%)": [results[m]["cv_mean"] for m in results],
    })

    st.dataframe(result_df, use_container_width=True)


with analysis_tab:

    st.subheader("Dataset Analysis")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", len(df))
    col2.metric("Heart Disease Cases", int(df["target"].sum()))
    col3.metric("Healthy Patients", int(len(df) - df["target"].sum()))
    col4.metric("Features", len(df.columns) - 1)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        df.corr(),
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        linewidths=0.5,
        ax=ax
    )
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)


with evaluation_tab:

    st.subheader("Detailed Model Evaluation")

    selected_eval_model = st.selectbox(
        "Select Model to Evaluate",
        list(trained_models.keys())
    )

    selected_model_eval = trained_models[selected_eval_model]
    y_pred = results[selected_eval_model]["y_pred"]

    st.markdown("### Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=["No Disease", "Heart Disease"],
        yticklabels=["No Disease", "Heart Disease"],
        ax=ax
    )
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("ROC Curve")
    roc_fig = generate_roc_curve(selected_model_eval, X_test, y_test)
    st.pyplot(roc_fig)

    st.markdown("---")
    st.subheader("SHAP Feature Importance")

    try:
        rf_model = trained_models["Random Forest"]
        shap_fig = generate_shap_plot(rf_model, X_test)
        st.pyplot(shap_fig)
    except Exception as e:
        st.warning(f"SHAP visualization unavailable: {e}")


st.markdown("---")
st.markdown(
    "<center><h4>❤️ Heart Disease Detection System</h4></center>",
    unsafe_allow_html=True
)
st.markdown(
    "<center>Built with Python, Streamlit, Scikit-learn and Gemini AI</center>",
    unsafe_allow_html=True
)
st.markdown(
    "<center>Developed by Shelly Verma</center>",
    unsafe_allow_html=True
)