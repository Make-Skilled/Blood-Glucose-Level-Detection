
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# App Title
st.title("Blood Glucose Prediction")

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)

    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Drop non-relevant or ID-like columns
    data = data.drop(columns=['Patient_ID'])

    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['Gender', 'Meal_Timing', 'Activity_Level', 'Diabetes_Type', 'Blood_Glucose_Status']
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Separate features and target variable
    X = data.drop(columns=['Blood_Glucose_Status'])
    y = data['Blood_Glucose_Status']

    # Standardize numerical features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train a Random Forest Classifier
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Evaluate the model
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.write("Model Evaluation:")
    st.write(f"Test Accuracy: {accuracy * 100:.2f}%")
    st.write("Classification Report:")
    st.dataframe(pd.DataFrame(report).transpose())

    # User Input
    st.write("### Predict Blood Glucose Status")
    user_input = {}
    for col in X.columns:
        user_input[col] = st.number_input(f"Enter {col}:", value=0.0)

    if st.button("Predict"):
        # Prepare input for prediction
        input_df = pd.DataFrame([user_input])
        input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

        # Prediction
        prediction = rf_classifier.predict(input_df)
        prediction_label = label_encoders['Blood_Glucose_Status'].inverse_transform(prediction)
        st.write(f"Predicted Blood Glucose Status: {prediction_label[0]}")
