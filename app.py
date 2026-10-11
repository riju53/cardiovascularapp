import os
import uuid
import requests
import streamlit as st

from typing import TypedDict, Optional

from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import streamlit as st
from langchain_groq import ChatGroq
#GROQ_API_KEY = "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"
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
model = ChatGroq(
    model = "openai/gpt-oss-120b",
    groq_api_key = GROQ_API_KEY,
    temperature=0.2

)

from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

conn = sqlite3.connect("my-db",check_same_thread=False)
checkpointer = SqliteSaver(conn)

from typing import TypedDict
class ChatState(TypedDict, total=False):
    question: str
    location: str

    age: int
    sex: str
    height_cm: int
    weight_kg: int
    systolic_bp: int
    cholesterol: int
    glucose: int
    smoking: str
    family_history: str
    physical_activity: str
    diabetes: str
    hypertension_history: str

    prediction: Optional[int]
    api_error: Optional[str]

    bmi: float
    bmi_category: str
    research: str
    advice: str
    doctor_results: str
    final_response: str
  
def extract_input(state: ChatState):

    question = state["question"]

    prompt = f"""
You are a cardiovascular health information extraction system.

Extract the following information from the user's message:

1. age
2. sex
3. height_cm
4. weight_kg
5. systolic_bp
6. cholesterol
7. glucose
8. smoking
9. family_history
10. physical_activity
11. diabetes
12. hypertension_history

IMPORTANT RULES:

- Return exactly 12 values.
- Separate values using commas.
- If information is missing, return "Not value".
- NEVER use 0 for missing information.
- Do not add explanations.
- Do not add labels.
- Do not use bullet points.

Allowed categorical values:

sex:
Male or Female

smoking:
Yes or No

family_history:
Yes or No

physical_activity:
High, Medium or Low

diabetes:
Yes or No

hypertension_history:
Yes or No

Example:

45, Male, 172, 82, 140, 220, 110, Yes, Yes, Low, No, Yes

User input:
{question}
"""

    response = model.invoke(prompt)

    raw_output = response.content.strip()

    print("\n========== RAW LLM RESPONSE ==========")
    print(raw_output)
    print("======================================\n")

    values = [v.strip() for v in raw_output.split(",")]

    print("Extracted values:", values)
    print("Number of values:", len(values))

    if len(values) != 12:
        raise ValueError(
            f"Expected 12 values, but received {len(values)}: {values}"
        )

    def safe_int(value):
        value = value.strip()

        if value.lower() in {
            "not value",
            "unknown",
            "none",
            "null",
            ""
        }:
            return None

        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None

    def safe_text(value):
        value = value.strip()

        if value.lower() in {
            "not value",
            "unknown",
            "none",
            "null",
            ""
        }:
            return None

        return value

    return {
        "age": safe_int(values[0]),
        "sex": safe_text(values[1]),
        "height_cm": safe_int(values[2]),
        "weight_kg": safe_int(values[3]),
        "systolic_bp": safe_int(values[4]),
        "cholesterol": safe_int(values[5]),
        "glucose": safe_int(values[6]),
        "smoking": safe_text(values[7]),
        "family_history": safe_text(values[8]),
        "physical_activity": safe_text(values[9]),
        "diabetes": safe_text(values[10]),
        "hypertension_history": safe_text(values[11])
    }


import requests

def safe_predict_input(state: ChatState):

    input_data = {
        "age": state["age"],
        "sex": state["sex"],
        "height_cm": state["height_cm"],
        "weight_kg": state["weight_kg"],
        "systolic_bp": state["systolic_bp"],
        "cholesterol": state["cholesterol"],
        "glucose": state["glucose"],
        "smoking": state["smoking"],
        "family_history": state["family_history"],
        "physical_activity": state["physical_activity"],
        "diabetes": state["diabetes"],
        "hypertension_history": state["hypertension_history"]
    }

    api_url = "https://cardiovascular-fastapi.onrender.com/predict"

    try:
        response = requests.post(
            api_url,
            json=input_data,
            timeout=60
        )

        print("Status code:", response.status_code)
        print("Response body:", response.text)

        response.raise_for_status()
        result = response.json()

        # Match the key returned by your FastAPI endpoint
        if "predicted_category" not in result:
            return {
                "prediction": None,
                "api_error": f"Missing predicted_category: {result}"
            }

        prediction = int(result["predicted_category"])

        if prediction not in (0, 1):
            return {
                "prediction": None,
                "api_error": f"Unexpected model output: {prediction}"
            }

        return {
            "prediction": prediction,
            "api_error": None
        }

    except requests.exceptions.RequestException as exc:
        print("API request error:", exc)
        return {
            "prediction": None,
            "api_error": str(exc)
        }

    except (ValueError, TypeError) as exc:
        print("Prediction response error:", exc)
        return {
            "prediction": None,
            "api_error": str(exc)
        }
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


search = DuckDuckGoSearchRun()


def search_tool(state: ChatState):

    # 1. Extract information from LangGraph state
    age = state["age"]
    sex = state["sex"]
    height_cm = state["height_cm"]
    weight_kg = state["weight_kg"]
    systolic_bp = state["systolic_bp"]
    cholesterol = state["cholesterol"]
    glucose = state["glucose"]
    smoking = state["smoking"]
    family_history = state["family_history"]
    physical_activity = state["physical_activity"]
    diabetes = state["diabetes"]
    hypertension_history = state["hypertension_history"]
    prediction = state.get("prediction")

    # 2. Calculate BMI safely
    if height_cm is None or weight_kg is None or height_cm <= 0:
        bmi = None
        bmi_category = "Unknown"
    else:
        bmi = weight_kg / (height_cm / 100) ** 2

        if bmi < 18.5:
            bmi_category = "Underweight"
        elif bmi < 25:
            bmi_category = "Healthy weight range"
        elif bmi < 30:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obesity range"

    # 3. Search for relevant health information
    query = """
    Find reliable cardiovascular health guidance from authoritative
    sources such as WHO, CDC, and recognized medical organizations.
    Cover blood pressure, cholesterol, blood glucose, smoking,
    physical activity, diabetes, and family history.
    Include relevant source names and links where available.
    """

    try:
        research = search.invoke(query)
    except Exception as e:
        research = (
            "Web research is temporarily unavailable. "
            "Do not invent research findings."
        )

from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()


def advice_tool(state: ChatState):

    # 1. Extract user information
    age = state["age"]
    sex = state["sex"]
    height_cm = state["height_cm"]
    weight_kg = state["weight_kg"]
    systolic_bp = state["systolic_bp"]
    cholesterol = state["cholesterol"]
    glucose = state["glucose"]
    smoking = state["smoking"]
    family_history = state["family_history"]
    physical_activity = state["physical_activity"]
    diabetes = state["diabetes"]
    hypertension_history = state["hypertension_history"]

    prediction = state.get("prediction")
    bmi = state.get("bmi")
    bmi_category = state.get("bmi_category")
    research = state.get("research", "No health research provided.")
    location = state.get("location")

    # 2. Find doctors near the user's supplied location
    if location and str(location).strip():

        doctor_query = (
            f"Cardiologists and qualified heart doctors near {location}. "
            "Find clinic or hospital names, addresses, contact details, "
            "and official websites or appointment links."
        )

        try:
            doctor_results = search.invoke(doctor_query)
        except Exception:
            doctor_results = (
                "Doctor search is currently unavailable. "
                "Do not invent doctors or contact details."
            )

    else:
        doctor_results = (
            "Location not provided. Ask the user for their city "
            "or locality to find nearby doctors."
        )

    # 3. Prepare the advice prompt
    prompt = f"""
You are a cardiovascular health education assistant.

USER INFORMATION:
Age: {age}
Sex: {sex}
Height: {height_cm} cm
Weight: {weight_kg} kg
BMI: {bmi}
BMI category: {bmi_category}
Systolic blood pressure: {systolic_bp}
Cholesterol: {cholesterol}
Glucose: {glucose}
Smoking: {smoking}
Family history: {family_history}
Physical activity: {physical_activity}
Diabetes: {diabetes}
Hypertension history: {hypertension_history}
ML prediction: {prediction}
Location: {location}

HEALTH RESEARCH:
{research}

NEARBY DOCTOR SEARCH RESULTS:
{doctor_results}

TASK:
Write a concise report with these sections:

1. BMI and its limitations
2. Systolic blood pressure
3. Cholesterol and glucose
4. Smoking and family history
5. Physical activity
6. Diabetes and hypertension history
7. ML prediction and its limitations
8. Recommended next steps and when to seek medical care
9. Doctors and healthcare facilities near the user's location

DOCTOR SEARCH RULES:
- Use the supplied search results rather than inventing providers.
- Include names, specialties, addresses, and contact or official
  appointment links when available.
- Clearly state that search results may need verification.
- Do not claim that a doctor is nearby, available, or qualified
  unless the available evidence supports that claim.
- If the location is missing or results are unavailable, explain
  what information or search is needed.

MEDICAL SAFETY:
- Do not diagnose cardiovascular disease.
- Do not present the ML output as a confirmed diagnosis or
  clinically validated individual risk estimate.
- Do not invent missing measurements or interpret them as normal.
- Do not prescribe medication or recommend changing medication.
- Consider measurement units and clinical context.
- Recommend professional assessment when appropriate.
- For possible emergency symptoms such as severe chest pain
  or difficulty breathing, advise immediate emergency care.

Use clear headings and practical, evidence-aware language.
"""

    # 4. Generate the final report
    response = model.invoke(prompt)
    advice = response.content

    # 5. Return updated LangGraph state
    return {
        "advice": advice,

        "doctor_results": doctor_results
    }


def calculate_bmi(state: ChatState):

    height_cm = state.get("height_cm")
    weight_kg = state.get("weight_kg")

    if height_cm is None or weight_kg is None:
        return {
            "bmi": None,
            "bmi_category": None
        }

    if height_cm <= 0 or weight_kg <= 0:
        return {
            "bmi": None,
            "bmi_category": None
        }

    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m ** 2), 1)

    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Healthy weight range"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obesity range"

    print("Calculated BMI:", bmi)
    print("BMI category:", category)

    return {
        "bmi": bmi,
        "bmi_category": category
    }


def final_output(state: ChatState):

    prediction = state.get("prediction")
    bmi = state.get("bmi")
    bmi_category = state.get("bmi_category")
    advice = state.get("advice", "Health advice is unavailable.")
    doctor_results = state.get(
        "doctor_results",
        "No doctor search results available."
    )

    # 1. Format the prediction result
    if prediction == 0:
        prediction_message = (
            "The ML model predicts that cardiovascular disease "
            "is not indicated."
        )
    elif prediction == 1:
        prediction_message = (
            "The ML model predicts that cardiovascular disease "
            "is indicated."
        )
    else:
        prediction_message = "No valid prediction is available."

    # 2. Build the final response
    final_response = f"""
#  Cardiovascular Health Report

## 1. ML Prediction

{prediction_message}

**Important:** This is a machine-learning model output, not a
confirmed diagnosis or a clinically validated individual risk estimate.

## 2. BMI Assessment

- **BMI:** {bmi if bmi is not None else "Not available"}
- **BMI category:** {bmi_category or "Not available"}

BMI is a screening measure and should not be used alone to assess
an individual's cardiovascular health.

## 3. Personalized Health Advice

{advice}

## 4. Nearby Doctors and Healthcare Facilities

{doctor_results}

## 5. Medical Safety

If you experience severe chest pain, difficulty breathing, fainting,
or other possible emergency symptoms, seek emergency medical care
immediately. Do not delay care while using this application.
"""

    # 3. Return the final state
    return {
        "final_response": final_response
    }

from langgraph.graph import StateGraph, START, END

graph = StateGraph(ChatState)

graph.add_node("predict_input", safe_predict_input)
graph.add_node("calculate_bmi", calculate_bmi)
graph.add_node("search_tool", search_tool)
graph.add_node("advice_tool", advice_tool)
graph.add_node("final_output", final_output)

graph.add_edge(START, "predict_input")
graph.add_edge("predict_input", "calculate_bmi")
graph.add_edge("calculate_bmi", "search_tool")
graph.add_edge("search_tool", "advice_tool")
graph.add_edge("advice_tool", "final_output")
graph.add_edge("final_output", END)

graph_app = graph.compile(checkpointer=checkpointer)

# Compile ONCE
graph_app = graph.compile(checkpointer=checkpointer)
# Streamlit portion

st.sidebar.title("❤️ Patient Information")

age = st.sidebar.number_input(
    "Age (years)", min_value=18, max_value=100, value=35
)

sex = st.sidebar.selectbox(
    "Sex", ["Female", "Male"]
)

height_cm = st.sidebar.number_input(
    "Height (cm)", min_value=140, max_value=210, value=174
)

weight_kg = st.sidebar.number_input(
    "Weight (kg)", min_value=30, max_value=200, value=70
)

systolic_bp = st.sidebar.number_input(
    "Systolic BP (mmHg)", min_value=70, max_value=210, value=120
)

cholesterol = st.sidebar.number_input(
    "Cholesterol (mg/dL)", min_value=50, max_value=600, value=180
)

glucose = st.sidebar.number_input(
    "Glucose (mg/dL)", min_value=40, max_value=600, value=95
)

smoking = st.sidebar.selectbox(
    "Do you smoke?", ["No", "Yes"]
)

family_history = st.sidebar.selectbox(
    "Family history of cardiovascular disease?", ["No", "Yes"]
)

physical_activity = st.sidebar.selectbox(
    "Physical activity", ["Low", "Medium", "High"]
)

diabetes = st.sidebar.selectbox(
    "Do you have diabetes?", ["No", "Yes"]
)

hypertension_history = st.sidebar.selectbox(
    "History of hypertension?", ["No", "Yes"]
)

location = st.sidebar.text_input(
    "City or locality",
    placeholder="e.g. Durgapur, West Bengal"
)

# BMI preview
bmi_preview = round(
    weight_kg / ((height_cm / 100) ** 2), 1
)

st.sidebar.metric("Calculated BMI", f"{bmi_preview} kg/m²")


# =====================================================
# 6. GENERATE REPORT
# =====================================================

if st.sidebar.button(
    "Generate Health Report",
    type="primary",
    use_container_width=True
):

    patient_state = {
        "age": int(age),
        "sex": sex,
        "height_cm": int(height_cm),
        "weight_kg": int(weight_kg),
        "systolic_bp": int(systolic_bp),
        "cholesterol": int(cholesterol),
        "glucose": int(glucose),
        "smoking": smoking,
        "family_history": family_history,
        "physical_activity": physical_activity,
        "diabetes": diabetes,
        "hypertension_history": hypertension_history,
        "location": location.strip()
    }

    try:
        # DON'T call build_graph() here
        # graph_app is already compiled above

        thread_id = str(uuid.uuid4())

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        with st.spinner(
            "Generating prediction and preparing your report..."
        ):
            result = graph_app.invoke(
                patient_state,
                config=config
            )

        st.success("Health report generated.")

        st.subheader("Machine Learning Prediction")

        prediction = result.get("prediction")

        if prediction is None:
            st.warning("Prediction unavailable.")

            if result.get("api_error"):
                st.code(result["api_error"])

        else:
            st.metric("Model class", prediction)

            st.caption(
                "Check the training dataset's label mapping to "
                "determine what class 0 or class 1 represents."
            )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "BMI",
                f"{result['bmi']} kg/m²"
            )

        with col2:
            st.metric(
                "BMI category",
                result["bmi_category"]
            )

        st.markdown(result["final_response"])

        with st.expander("Submitted patient information"):
            st.json(patient_state)

        with st.expander("Medical research search results"):
            st.write(
                result.get(
                    "research",
                    "Unavailable"
                )
            )

        with st.expander("Doctor search results"):
            st.write(
                result.get(
                    "doctor_results",
                    "Unavailable"
                )
            )

        st.download_button(
            "Download Health Report",
            data=result["final_response"],
            file_name="cardiovascular_report.md",
            mime="text/markdown"
        )

    except Exception as exc:
        st.error(
            "The report could not be generated."
        )
        st.exception(exc)
