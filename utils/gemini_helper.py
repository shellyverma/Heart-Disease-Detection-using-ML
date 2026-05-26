import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def generate_ai_analysis(
    age,
    sex,
    trestbps,
    chol,
    thalach,
    prediction,
    confidence
):
    if not GEMINI_API_KEY:
        return "⚠️ Gemini API key not found. Please add GEMINI_API_KEY in your .env file."

    prompt = f"""
    Analyze the following heart disease prediction result and provide medical insights.

    Patient Information:
    - Age: {age}
    - Sex: {'Male' if sex == 1 else 'Female'}
    - Blood Pressure: {trestbps} mmHg
    - Cholesterol: {chol} mg/dL
    - Maximum Heart Rate: {thalach} bpm
    - Prediction: {'HIGH RISK of Heart Disease' if prediction == 1 else 'LOW RISK of Heart Disease'}
    - Confidence: {confidence}%

    Please provide:
    1. Main health concerns based on these values
    2. Important observations about the patient's condition
    3. Lifestyle recommendations (diet, exercise, habits)
    4. When doctor consultation is needed

    Keep the response simple, clear, and beginner friendly.
    Use bullet points where possible.
    """

    # Try models one by one until one works
    models_to_try = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]

    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception:
            continue

    return "⚠️ AI analysis unavailable. All Gemini models failed. Please check your API key."