import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Page title
# -----------------------------
st.set_page_config(page_title="Titanic Survival Prediction")

st.title("🚢 Titanic Survival Prediction")
st.write("Logistic Regression based survival prediction")

# -----------------------------
# Dataset inside code
# -----------------------------
@st.cache_data
def load_data():
    data = {
        "Pclass": [3, 1, 3, 1, 3, 2, 3, 1, 2, 3],
        "Age": [22, 38, 26, 35, 35, 28, 2, 54, 14, 4],
        "Fare": [7.25, 71.28, 7.92, 53.10, 8.05, 13.00, 21.07, 51.86, 11.13, 16.70],
        "Sex": ["male", "female", "female", "female", "male", "male", "female", "male", "female", "male"],
        "Embarked": ["S", "C", "S", "S", "S", "S", "S", "S", "C", "S"],
        "Survived": [0, 1, 1, 1, 0, 0, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)

df = load_data()

# -----------------------------
# Preprocessing
# -----------------------------
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Fare"].fillna(df["Fare"].median(), inplace=True)

df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

features = ["Pclass", "Age", "Fare", "Sex_male"]
X = df[features]
y = df["Survived"]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Feature scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Logistic Regression Model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -----------------------------
# Model performance
# -----------------------------
st.subheader("📊 Model Performance")

accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Accuracy:** {accuracy:.2f}")

if st.checkbox("Show Classification Report"):
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

if st.checkbox("Show Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Survived", "Survived"],
        yticklabels=["Not Survived", "Survived"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# -----------------------------
# User Prediction
# -----------------------------
st.subheader("🧍 Predict Survival")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.number_input("Age", min_value=0.0, value=25.0)
fare = st.number_input("Fare", min_value=0.0, value=10.0)
sex_male = st.selectbox("Gender", ["Female", "Male"])

sex_male = 1 if sex_male == "Male" else 0

if st.button("Predict Survival"):
    input_data = [[pclass, age, fare, sex_male]]
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    result = "Survived ✅" if prediction[0] == 1 else "Did Not Survive ❌"
    st.success(f"**Prediction:** {result}")
