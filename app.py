import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 🎨 Custom header with color
st.markdown("<h1 style='color:#ff69b4;'>🌸 HR Attrition Dashboard</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
st.sidebar.header("Upload your HR data")
uploaded_file = st.sidebar.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])

# Main logic
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith("xlsx") else pd.read_csv(uploaded_file)
    
    st.success("✅ File uploaded successfully!")

    # Layout: 2 columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👁️ Preview")
        st.dataframe(df.head())

    with col2:
        st.subheader("📊 Attrition Count")
        fig = px.histogram(df, x="Attrition", color="Attrition", template="plotly_dark", color_discrete_sequence=["#ff1493", "#87CEFA"])
        st.plotly_chart(fig, use_container_width=True)

    # Data prep
    data = df.copy()
    data.drop(['EmployeeNumber', 'EmployeeCount', 'StandardHours', 'Over18'], axis=1, inplace=True, errors='ignore')
    label_encoders = {}
    for col in data.select_dtypes(include='object'):
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    X = data.drop("Attrition", axis=1)
    y = data["Attrition"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Prediction section
    st.markdown("---")
    st.subheader("🔮 Predict Employee Attrition")
    user_input = {}
    for col in X.columns:
        if df[col].dtype == "object":
            user_input[col] = st.selectbox(f"{col}", df[col].unique())
        else:
            user_input[col] = st.slider(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    if st.button("Predict"):
        for col in user_input:
            if isinstance(user_input[col], str):
                user_input[col] = label_encoders[col].transform([user_input[col]])[0]
        user_df = pd.DataFrame([user_input])
        prediction = model.predict(user_df)

        if prediction[0] == 1:
            st.error("⚠️ This employee is likely to resign.")
        else:
            st.success("✅ This employee is likely to stay.")

    # Model accuracy
    st.markdown("---")
    st.subheader("📈 Model Accuracy")
    acc = accuracy_score(y_test, model.predict(X_test))
    st.metric("Accuracy", f"{acc * 100:.2f}%")
else:
    st.info("👈 Upload a file from the sidebar to begin.")
