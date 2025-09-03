import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎓 Career Prediction Using Machine Learning")
st.write("Upload student dataset, train a model, and predict career paths.")

# -----------------------------
# 1. Upload Dataset
# -----------------------------
#uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.write("### 📊 Dataset Preview")
#     st.dataframe(df.head())

    # Assume last column is "career" (target)
df = pd.read_csv("student_dummy_data.csv")
target_col = st.selectbox("Select target (career) column", df.columns, index=len(df.columns)-1)

X = df.drop(target_col, axis=1)
y = df[target_col]

# -----------------------------
# 2. Preprocessing
# -----------------------------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

if target_col in categorical_features:
    categorical_features.remove(target_col)

numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -----------------------------
# 3. Models
# -----------------------------
dtree = DecisionTreeClassifier(random_state=42)
svm = SVC(probability=True, random_state=42)
ensemble = VotingClassifier(estimators=[("dt", dtree), ("svm", svm)], voting="soft")

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", ensemble)])

# -----------------------------
# 4. Train Model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
pipeline.fit(X_train, y_train)

st.success("✅ Model trained successfully!")

# -----------------------------
# 5. Prediction Form
# -----------------------------
st.write("### 🎯 Enter Student Details for Prediction")

input_data = {}
for col in X.columns:
    if col in numeric_features:
        input_data[col] = st.number_input(f"{col}", value=float(X[col].mean()))
    else:
        input_data[col] = st.selectbox(f"{col}", options=X[col].unique())

if st.button("Predict Career"):
    new_student = pd.DataFrame([input_data])
    prediction = pipeline.predict(new_student)
    st.success(f"Predicted Career Path: **{prediction[0]}**")
