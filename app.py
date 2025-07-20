import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import seaborn as sns
# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/churn_new_dataset.csv")
    return df

df = load_data()

with open("models/model_scores.pkl", "rb") as f:
    model_scores = joblib.load(f)

# Feature columns
selected_columns = ['Contract', 'tenure', 'InternetService', 'OnlineSecurity',
                    'TechSupport', 'MonthlyCharges', 'PaymentMethod', 'PaperlessBilling', 'Churn']

df = df[selected_columns]

X = df.drop("Churn", axis=1)
y = df["Churn"]

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
])

# Model Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
])

model_pipeline.fit(X, y)

# Layout
st.set_page_config(page_title="Customer Churn App", layout="wide")

st.markdown("<h1 style='text-align:center;'>📊 Customer Churn Prediction Dashboard</h1>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📈 Analytics", "🔍 Predict", "🧠 Model Metrics"])

# -------------------------------- HOME TAB --------------------------------
with tab1:
    st.markdown("## 📊 Customer Churn Overview")

    # === Metrics Row ===
    total_customers = len(df)
    churn_rate = df['Churn'].value_counts(normalize=True).get('Yes', 0)
    churn_count = df['Churn'].value_counts().get('Yes', 0)
    no_churn_count = df['Churn'].value_counts().get('No', 0)

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("👥 Total Customers", total_customers)
    metric2.metric("❌ Churned", churn_count)
    metric3.metric("📉 Churn Rate", f"{churn_rate:.2%}")

    st.markdown("---")

    # === Pie Chart Centered ===
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("### Churn Distribution")
        churn_data = df['Churn'].value_counts()
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.pie(
            churn_data,
            labels=["No Churn", "Churn"],
            autopct='%1.1f%%',
            startangle=90,
            colors=['#66bb6a', '#ef5350']
        )
        ax.axis('equal')
        st.pyplot(fig)

    with col_right:
        st.markdown("### 📌 Churn Distribution")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.countplot(data=df, x='Churn', palette=['#66bb6a', '#ef5350'], ax=ax)
        ax.set_title("Customer Churn Count")
        ax.set_xlabel("Churn")
        ax.set_ylabel("Count")
        st.pyplot(fig)



# -------------------------------- ANALYTICS TAB --------------------------------
with tab2:
    st.subheader("Explore Feature Distributions")

    feat = st.selectbox("Select Feature to Explore", df.columns[:-1])
    fig = px.histogram(df, x=feat, color="Churn", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    if df[feat].nunique() < 10:
        st.markdown("### Value Counts")
        st.dataframe(df[feat].value_counts(), use_container_width=True)

# -------------------------------- PREDICTION TAB --------------------------------
with tab3:
    st.subheader("Make a Prediction")

    # Input form
    with st.form("prediction_form"):
        contract = st.selectbox("Contract", df["Contract"].unique())
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        internet = st.selectbox("Internet Service", df["InternetService"].unique())
        online_sec = st.selectbox("Online Security", df["OnlineSecurity"].unique())
        tech_support = st.selectbox("Tech Support", df["TechSupport"].unique())
        monthly_charges = st.slider("Monthly Charges", 0, 120, 50)
        payment_method = st.selectbox("Payment Method", df["PaymentMethod"].unique())
        paperless = st.selectbox("Paperless Billing", df["PaperlessBilling"].unique())

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame({
            'Contract': [contract],
            'tenure': [tenure],
            'InternetService': [internet],
            'OnlineSecurity': [online_sec],
            'TechSupport': [tech_support],
            'MonthlyCharges': [monthly_charges],
            'PaymentMethod': [payment_method],
            'PaperlessBilling': [paperless]
        })

        pred = model_pipeline.predict(input_df)[0]
        proba = model_pipeline.predict_proba(input_df)[0][1]

        st.markdown("---")
        st.write("### Prediction Result")
        st.success(f"Churn Probability: {proba:.2%}")
        st.info(f"Prediction: {'Will Churn' if pred == 'Yes' else 'Will Not Churn'}")

# -------------------------------- METRICS TAB --------------------------------
with tab4:
    st.markdown("Below is the comparison of model performance:")

    rows = []
    for model, metrics in model_scores.items():
        rows.append({
            "Model": model,
            "Accuracy": f"{metrics['Accuracy'] * 100:.1f}%",
            "Precision (0/1)": f"{metrics['Precision']['Class 0']} / {metrics['Precision']['Class 1']}",
            "Recall (0/1)": f"{metrics['Recall']['Class 0']} / {metrics['Recall']['Class 1']}",
            "F1-score (0/1)": f"{metrics['F1-Score']['Class 0']} / {metrics['F1-Score']['Class 1']}",
            "Weighted F1": round(metrics["Weighted F1-Score"], 2)
        })

    df_scores = pd.DataFrame(rows)
    st.dataframe(df_scores, use_container_width=True)

