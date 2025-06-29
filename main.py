import streamlit as st
import pandas as pd
import joblib
import json
import re
from datetime import datetime
import io
import os
import base64

# --- Configuration ---
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# Load model, label encoder, and training features
model = joblib.load("best_disease_prediction_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
with open("training_features_list.txt", "r") as f:
    all_symptoms = [line.strip() for line in f.readlines()]

# Load master dataset for descriptions and precautions
df_info = pd.read_csv("master_dataset.csv")

# Load food recommendation dataset
df_food = pd.read_csv("Food_and_Nutrition_TEMPLATE.csv")
df_food["Disease"] = df_food["Disease"].str.lower()

# Prediction history
prediction_history = []

# File path for saving history
HISTORY_FILE = "prediction_history.csv"

# Load existing history if file exists and not empty
if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:
    try:
        prediction_history = pd.read_csv(HISTORY_FILE).to_dict(orient="records")
    except Exception:
        prediction_history = []

# NLP symptom extractor (improved match for symptoms with underscores)
def extract_symptoms_from_text(text):
    text = text.lower()
    text = re.sub(r"[\-_]", " ", text)  # Normalize dashes and underscores to space
    text_words = set(re.findall(r'\w+', text))  # Tokenize words from text

    extracted = []
    for symptom in all_symptoms:
        normalized_symptom = symptom.replace("_", " ")
        symptom_words = set(normalized_symptom.split())

        # Check if all words in the symptom are present in the input text
        if symptom_words.issubset(text_words):
            extracted.append(symptom)
    return extracted

# Function to save to CSV file
def append_to_history_csv(entry):
    df_entry = pd.DataFrame([entry])
    if os.path.exists(HISTORY_FILE):
        df_entry.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
    else:
        df_entry.to_csv(HISTORY_FILE, index=False)

# Function to predict disease and return all info
def predict_disease(symptoms):
    input_vector = [1 if symptom in symptoms else 0 for symptom in all_symptoms]
    input_df = pd.DataFrame([input_vector], columns=all_symptoms)
    predicted_label = model.predict(input_df)[0]
    predicted_disease = label_encoder.inverse_transform([predicted_label])[0].lower()

    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symptoms": ", ".join(symptoms),
        "disease": predicted_disease.capitalize()
    }

    info_row = df_info[df_info['Disease'].str.lower() == predicted_disease]
    if not info_row.empty:
        row = info_row.iloc[0]
        result["description"] = row['Symptom_Description']
        result["precaution"] = row['Precautions']
    else:
        result["description"] = "Not found"
        result["precaution"] = "Not found"

    food_row = df_food[df_food['Disease'] == predicted_disease]
    if not food_row.empty:
        food_row = food_row.iloc[0]
        result["food_eat"] = food_row["Food to Eat"]
        result["food_avoid"] = food_row["Food to Avoid"]
    else:
        result["food_eat"] = "Not found"
        result["food_avoid"] = "Not found"

    prediction_history.append(result)
    append_to_history_csv(result)

    return result

st.set_page_config(page_title="Smart Disease Predictor", layout="centered")

def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode()

    background_css = f"""
    <style>
    /* Global Background */
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{encoded_image}");
        background-repeat: no-repeat;
        background-position: center center;
        background-size: cover;
        background-attachment: fixed;
    }}

    [data-testid="stHeader"], [data-testid="stToolbar"] {{
        background-color: rgba(0, 0, 0, 0.6);
    }}

    .block-container {{
        background-color: rgba(0, 0, 0, 0.85);
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem auto;
        max-width: 95%;
        font-size: 1.1rem;
    }}

    /* Improve mobile scaling */
    @media screen and (max-width: 768px) {{
        [data-testid="stAppViewContainer"] {{
            background-size: cover;
            background-attachment: scroll;
            background-position: center top;
        }}

        .block-container {{
            padding: 1rem;
            margin: 0.5rem auto;
            width: 100%;
            font-size: 1.2rem;
        }}

        .stTextInput input, .stTextArea textarea, .stSelectbox div, .stButton button {{
            font-size: 1.1rem !important;
        }}

        .stRadio label, .stMarkdown p {{
            font-size: 1.1rem !important;
        }}
    }}
    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)

set_background("background.jpg")

st.title("🩺 Smart Disease Prediction and Food Recommendation System")

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False
if "admin_username" not in st.session_state:
    st.session_state.admin_username = ADMIN_USERNAME
if "admin_password" not in st.session_state:
    st.session_state.admin_password = ADMIN_PASSWORD

menu = st.sidebar.radio("Navigate", [
    "🔍 Predict", "📄 CSV Upload", "📜 History", "⬇️ Download History", "🥗 Food Summary",
    "🔐 Admin Login", "🛠️ Admin Dashboard"
])

if menu == "🔍 Predict":
    st.subheader("Symptom Input")
    mode = st.radio("Choose input method:", ["Manual Selection", "Free Text (NLP)"])
    if mode == "Manual Selection":
        selected_symptoms = st.multiselect("Select your symptoms:", options=all_symptoms)
    else:
        free_text = st.text_area("Describe your symptoms in plain English:")
        if free_text:
            selected_symptoms = extract_symptoms_from_text(free_text)
            st.info(f"Extracted Symptoms: {selected_symptoms}")
        else:
            selected_symptoms = []

    if st.button("🔍 Predict Disease") and selected_symptoms:
        result = predict_disease(selected_symptoms)
        st.subheader(f"🦠 Predicted Disease: {result['disease']}")
        st.markdown(f"**📝 Description:** {result['description']}")
        st.markdown(f"**💊 Precautions:** {result['precaution']}")
        st.markdown(f"**✅ Foods to Eat:** {result['food_eat']}")
        st.markdown(f"**❌ Foods to Avoid:** {result['food_avoid']}")

elif menu == "📄 CSV Upload":
    st.subheader("Batch Prediction from CSV")
    csv_file = st.file_uploader("Upload a CSV file with a 'Symptoms' column:", type=["csv"])
    if csv_file is not None:
        df_csv = pd.read_csv(csv_file)
        if 'Symptoms' not in df_csv.columns:
            st.error("CSV must contain a 'Symptoms' column.")
        else:
            batch_results = []
            for index, row in df_csv.iterrows():
                raw_symptoms = row['Symptoms']
                symptoms_list = [sym.strip().lower() for sym in raw_symptoms.split(',')]
                result = predict_disease(symptoms_list)
                batch_results.append({"Input Symptoms": raw_symptoms, **result})

            df_results = pd.DataFrame(batch_results)
            st.dataframe(df_results)

elif menu == "📜 History":
    st.subheader("Prediction History")

    if prediction_history:
        df_history = pd.DataFrame(prediction_history)
        st.dataframe(df_history)

        search = st.text_input("🔍 Search by disease or symptom:")
        if search:
            filtered = df_history[df_history.apply(lambda row: search.lower() in row.to_string().lower(), axis=1)]
            st.dataframe(filtered)

        st.markdown("### 🗑️ Delete an Entry")
        row_to_delete = st.number_input("Enter the row index to delete:", min_value=0, max_value=len(df_history)-1, step=1)

        if st.button("❌ Delete Selected Entry"):
            try:
                prediction_history.pop(row_to_delete)
                pd.DataFrame(prediction_history).to_csv(HISTORY_FILE, index=False)
                st.success("✅ Entry deleted successfully. Please refresh to update.")
            except:
                st.error("⚠️ Could not delete. Invalid index or error occurred.")

        if st.button("🧹 Clear All History"):
            prediction_history.clear()
            open(HISTORY_FILE, 'w').close()
            st.success("✅ All prediction history cleared.")

    else:
        st.info("No predictions made yet.")

elif menu == "⬇️ Download History":
    st.subheader("Download Prediction History")

    if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:
        try:
            df_download = pd.read_csv(HISTORY_FILE)
            if not df_download.empty:
                csv_data = df_download.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download CSV", data=csv_data, file_name="prediction_history.csv", mime="text/csv")
            else:
                st.info("Prediction history is empty.")
        except pd.errors.EmptyDataError:
            st.info("Prediction history is empty.")
        except Exception as e:
            st.error(f"⚠️ Error loading history: {e}")
    else:
        st.info("No prediction history available to download.")

elif menu == "🥗 Food Summary":
    st.subheader("Disease-Based Food Guidance")
    disease_list = sorted(df_food["Disease"].str.capitalize().unique())
    selected_disease = st.selectbox("Select a disease:", disease_list)
    food_row = df_food[df_food["Disease"] == selected_disease.lower()]

    if not food_row.empty:
        food_row = food_row.iloc[0]
        st.markdown(f"**✅ Foods to Eat:** {food_row['Food to Eat']}")
        st.markdown(f"**❌ Foods to Avoid:** {food_row['Food to Avoid']}")
    else:
        st.warning("No food info found.")

elif menu == "🔐 Admin Login":
    st.subheader("Admin Access")
    if st.session_state.admin_logged_in:
        st.success("✅ You are already logged in as admin.")
    else:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("🔓 Login"):
            if username == st.session_state.admin_username and password == st.session_state.admin_password:
                st.session_state.admin_logged_in = True
                st.success("✅ Login successful. Admin access granted.")
            else:
                st.error("❌ Incorrect username or password")

elif menu == "🛠️ Admin Dashboard":
    st.subheader("Admin Tools")
    if st.session_state.admin_logged_in:
        st.success("Welcome, Admin!")

        st.markdown("---")
        st.subheader("📤 Upload/Replace Datasets")
        dataset_file = st.file_uploader("Upload master_dataset.csv:", type=["csv"], key="upload1")
        food_file = st.file_uploader("Upload Food_and_Nutrition_TEMPLATE.csv:", type=["csv"], key="upload2")

        if dataset_file:
            pd.read_csv(dataset_file).to_csv("master_dataset.csv", index=False)
            st.success("✅ master_dataset.csv updated successfully.")
        if food_file:
            pd.read_csv(food_file).to_csv("Food_and_Nutrition_TEMPLATE.csv", index=False)
            st.success("✅ Food_and_Nutrition_TEMPLATE.csv updated successfully.")

        st.markdown("---")
        st.subheader("🔄 Update Model and Feature List")
        model_file = st.file_uploader("Upload new model (.pkl):", type=["pkl"], key="upload3")
        features_file = st.file_uploader("Upload training_features_list.txt:", type=["txt"], key="upload4")

        if model_file:
            with open("best_disease_prediction_model.pkl", "wb") as f:
                f.write(model_file.read())
            st.success("✅ Model updated.")
        if features_file:
            with open("training_features_list.txt", "wb") as f:
                f.write(features_file.read())
            st.success("✅ Feature list updated.")

        st.markdown("---")
        st.subheader("🔑 Change Admin Password")
        new_username = st.text_input("New Username", value=st.session_state.admin_username)
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")

        if st.button("Update Password"):
            if new_password == confirm_password and new_password:
                st.session_state.admin_username = new_username
                st.session_state.admin_password = new_password
                st.success("✅ Username and password updated successfully.")
            else:
                st.error("❌ Passwords do not match or are empty.")
    else:
        st.warning("🔒 Admin login required.")
