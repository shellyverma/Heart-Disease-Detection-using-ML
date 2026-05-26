# ❤️ Heart Disease Detection System

An AI-powered Heart Disease Prediction Web Application built using Python, Streamlit, Scikit-learn, and Gemini AI.

The application predicts the possibility of heart disease using multiple Machine Learning models and provides AI-generated medical insights with interactive visualizations.

---

#  Features

- Heart Disease Prediction using Machine Learning
- Multiple ML Models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine (SVM)
- Gemini AI Medical Analysis
- SHAP Feature Importance Visualization
- ROC Curve Visualization
- Confusion Matrix
- Interactive Streamlit Dashboard
- Dataset Analysis & Correlation Heatmap
- Model Comparison Dashboard
- Risk Probability Score

---

# Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SHAP
- Google Gemini AI
- Python Dotenv

---

# Project Structure

```bash
Heart-Disease-Detection-using-ML/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── data/
│   └── heart_disease_data.csv
│
├── models/
│   └── train_models.py
│
├── utils/
│   ├── gemini_helper.py
│   ├── preprocessing.py
│   ├── plots.py
│   └── shap_analysis.py
```

---

# Dashboard Sections

## 🔍 Predict + AI Analysis
- Patient heart disease prediction
- Risk probability score
- AI-generated medical insight

## Data Analysis
- Correlation heatmap
- Dataset overview
- Patient statistics

## Model Comparison
- Accuracy comparison
- Precision, Recall, F1 Score
- ROC AUC comparison

## Model Evaluation
- Confusion Matrix
- ROC Curve
- SHAP Feature Importance

---

# Run Locally

Clone the repository:

```bash
git clone https://github.com/shellyverma/Heart-Disease-Detection-using-ML.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

# Deployment

Deployed using Streamlit Community Cloud.

---

---

# Acknowledgement

The initial idea and base implementation of this project were inspired by machine learning tutorials and educational resources from GeeksForGeeks and YouTube.

The project was later redesigned and enhanced with:
- Streamlit Web Application
- Multiple ML Models
- Gemini AI Integration
- SHAP Explainability
- ROC Curve Visualization
- Modular Project Structure
- Interactive Dashboards and UI Improvements