# Smart Disease Prediction and Food Recommendation System

**Project Overview**

This project is a machine learning-powered web application that predicts possible diseases based on user-inputted symptoms and provides relevant food recommendations and medical precautions. It is designed to assist users in understanding potential health conditions early and promote healthy dietary choices tailored to those conditions.

Built using Python, scikit-learn, Streamlit, and pandas, the system supports manual input, natural language processing (NLP) for symptom description, batch predictions via CSV upload, and includes an admin dashboard for data and model management.

## Problem Statement
In many parts of the world, especially in low-resource settings, people delay visiting medical professionals due to lack of access, awareness, or cost. This leads to undiagnosed or misdiagnosed illnesses, increasing the risk of severe health outcomes. Additionally, patients often struggle with knowing the right kind of food to consume or avoid during illness, which can delay recovery or worsen symptoms.

There is a need for an accessible, intelligent, and easy-to-use system that can:

- Predict possible diseases based on user-reported symptoms.

- Provide medical advice such as disease descriptions and precautions.

- Recommend foods to eat or avoid based on the predicted illness.

This project aims to solve that problem by building a machine learning-powered system that empowers users with health-related insights, using a smart, interactive interface.

## Features Implemented

•	**Disease Prediction from Symptoms:**

-	Manual selection of symptoms
-	Natural language input (NLP-based keyword extraction)
-	Machine learning model predicts disease based on symptom vectorization
  
•	**Food Recommendation System:**

-	Returns recommended foods to eat and avoid for the predicted disease
  
•	**Batch Prediction via CSV Upload:**

-	Users can upload CSV files containing a 'Symptoms' column
-	The app performs disease predictions for each record
  
•	**Prediction History Logging:**

-	Automatically logs every prediction with timestamp, symptoms, and results
-	Users can view and download history in CSV format
-	Searchable history table
  
•	**Food Summary Viewer:**

-	Users can browse food guidance by disease from the nutrition dataset
  
•	**Admin Login System:**

-	Secured login for admin panel
-	Default username: admin
-	Default password: admin123
-	Ability to update username and password securely
  
•	**Admin Dashboard:**

-	Upload/replace:
-	Disease descriptions dataset (master_dataset.csv)
-	Food recommendation dataset (Food_and_Nutrition_TEMPLATE.csv)
-	ML model (best_disease_prediction_model.pkl)
-	Feature list (training_features_list.txt)
-	Admin-only access control ensures regular users cannot access sensitive features
  
•	**Custom Styling:**

-	Background image support
-	Emoji-based icons for navigation
-	Organized layout using Streamlit UI elements

## Project Structure

project-folder/

app.py                            				_Streamlit application_

best_disease_prediction_model.pkl 				_Trained ML model_

label_encoder.pkl                 				_Label encoder for disease names_

training_features_list.txt        				_List of all symptom features_

master_dataset.csv                				_Disease description and precaution info_

Food_and_Nutrition_TEMPLATE.csv   				_Disease-food mapping_

prediction_history.csv            				_Automatically generated prediction history_

background.jpg                    				_Optional background image_

## System Requirements

Python 3.8+

•	Required Libraries:

`pip install streamlit pandas scikit-learn joblib`

## Accessibility & Usage

•	**User Access**

- Predict diseases via form or free-text
- Upload batch symptom files (CSV)
- View/download prediction history
- Browse disease-specific food guides
  
•	**Admin Access**

- Accessible only after login (Admin Login)
- Upload new data files or model
- Change login credentials securely
- View upload confirmation messages
  
•	**How to Run the App**
1. Place all files in the project directory.
2. From terminal or command prompt, run: `streamlit run main.py`
3. Access the app in your browser at: http://localhost:8501
   
## Future Enhancements (Optional Ideas)

•	Add multi-language support

•	Integrate AI chatbot assistant

•	Enable user registration & session storage

•	Add real-time symptom ranking

•	Connect to real-time food databases via API

