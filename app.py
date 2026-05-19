import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

h1, h2, h3, h4 {
    color: #FAFAFA;
}

.block-container {
    padding-top: 2rem;
}

.metric-card {
    background-color: #1E222A;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.5);
}

.section-card {
    background-color: #1E222A;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("student-mat.csv", sep=";")

# Convert absences → Attendance %
df["Attendance"] = 100 - (df["absences"] * 2)
df["Attendance"] = df["Attendance"].clip(lower=0)

# ==============================
# UI
# ==============================

st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>
🎓 Student Performance & Attendance Dashboard
</h1>
""", unsafe_allow_html=True)

st.markdown("---")

st.sidebar.header("Enter Student Details")

total_classes = st.sidebar.number_input("Total Classes Conducted", min_value=1)
attended_classes = st.sidebar.number_input("Classes Attended", min_value=0)

required_percentage = st.sidebar.slider("Required Attendance %", 60, 95, 75)

studytime = st.sidebar.slider("Study Time (1-4)", 1, 4, 2)
failures = st.sidebar.slider("Past Class Failures (0-3)", 0, 3, 0)
g1 = st.sidebar.slider("First Period Grade (0-20)", 0, 20, 10)
g2 = st.sidebar.slider("Second Period Grade (0-20)", 0, 20, 10)

# ==============================
# Bunk Calculator
# ==============================

def max_bunks_allowed(total_classes, attended_classes, required_percentage):
    required_fraction = required_percentage / 100
    numerator = attended_classes - (required_fraction * total_classes)

    if numerator <= 0:
        return 0

    bunks = numerator / required_fraction
    return int(bunks)

# ==============================
# Analyze Button
# ==============================

if st.sidebar.button("Analyze"):

    # Current attendance
    current_attendance = (attended_classes / total_classes) * 100

    st.subheader("📊 Attendance Overview")

    st.metric("Required Attendance", f"{required_percentage}%")
    st.metric("Your Attendance", f"{current_attendance:.2f}%")

    st.progress(min(current_attendance / 100, 1.0))

    # ==============================
    # Create Target
    # ==============================

    df["At_Risk"] = (
        (df["Attendance"] < required_percentage) |
        (df["G3"] < 10)
    ).astype(int)

    # Safety check
    if df["At_Risk"].nunique() < 2:
        st.error("⚠️ Dataset has only one class. Adjust slider.")
        st.stop()

    # Features
    features = ["Attendance", "studytime", "failures", "G1", "G2"]

    X = df[features]
    y = df["At_Risk"]
    g3 = df["G3"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==============================
    # Classification Model
    # ==============================

    clf_model = LogisticRegression()
    clf_model.fit(X_train_scaled, y_train)

    y_pred = clf_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    st.sidebar.markdown("### 📊 Model Performance")
    st.sidebar.write(f"Accuracy: {accuracy*100:.2f}%")

    # ==============================
    # Regression Model (G3 Prediction)
    # ==============================

    reg_model = LinearRegression()
    reg_model.fit(X_train_scaled, g3.loc[y_train.index])

    # ==============================
    # Prediction
    # ==============================

    input_df = pd.DataFrame(
        [[current_attendance, studytime, failures, g1, g2]],
        columns=features
    )

    input_scaled = scaler.transform(input_df)
    # Risk prediction
    prediction = clf_model.predict(input_scaled)

    # G3 prediction
    predicted_g3 = reg_model.predict(input_scaled)[0]

    # ==============================
    # Output
    # ==============================

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 Risk Prediction")
    
        if prediction[0] == 1:
            st.error("⚠️ AT RISK")
        else:
            st.success("✅ SAFE")

    with col2:
        st.subheader("📊 Performance")
    
        st.write(f"G3: **{predicted_g3:.2f} / 20**")

        if predicted_g3 < 10:
            st.error("⚠️ FAIL")
        else:
            st.success("✅ PASS")

    # Explain model
    st.markdown("### 🔍 Model Factors")
    st.info("""
    - Attendance  
    - Study Time  
    - Past Failures  
    - Internal Scores (G1, G2)  
    """)

    # ==============================
    # Bunk Calculator
    # ==============================

    bunks_left = max_bunks_allowed(total_classes, attended_classes, required_percentage)

    st.markdown("## 📌 Smart Bunk Calculator")

    if bunks_left > 0:
        st.success(f"🟢 You can bunk **{bunks_left}** classes safely")
        st.progress(min(bunks_left / 50, 1.0))
    else:
        st.error("🔴 No more bunks allowed")

    # ==============================
    # Visualization
    # ==============================

    st.markdown("## 📈 Performance Trend")
    st.caption("Higher attendance generally leads to better performance")

    avg_data = df.groupby("Attendance")["G3"].mean().reset_index()

    fig = plt.figure()
    plt.plot(avg_data["Attendance"], avg_data["G3"])
    plt.xlabel("Attendance (%)")
    plt.ylabel("Final Grade (G3)")
    plt.grid(True)
    st.pyplot(fig)