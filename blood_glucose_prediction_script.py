
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
file_path = 'Blood_Glucose_Prediction_Updated_Dataset_100.csv'
data = pd.read_csv(file_path)

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
report = classification_report(y_test, y_pred)

# Cross-validation
cv_scores = cross_val_score(rf_classifier, X, y, cv=5, scoring='accuracy')

# Output results
print("Test Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())
