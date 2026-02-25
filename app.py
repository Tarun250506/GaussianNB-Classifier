import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

st.title("🧠 Naive Bayes Classifier App")

st.write("Upload a CSV dataset to train a Naive Bayes model.")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Select target column
    target_column = st.selectbox("Select Target Column", df.columns)

    # Train-test split slider
    test_size = st.slider("Select Test Size (%)", 10, 50, 20) / 100

    if st.button("Train Model"):

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Handle categorical variables automatically
        X = pd.get_dummies(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = GaussianNB()
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Accuracy
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        st.subheader("📊 Accuracy Results")
        st.write(f"Training Accuracy: **{train_acc:.4f}**")
        st.write(f"Testing Accuracy: **{test_acc:.4f}**")

        # Confusion Matrix
        st.subheader("📌 Confusion Matrix")

        cm = confusion_matrix(y_test, y_test_pred)

        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)
        st.pyplot(fig)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Naive Bayes Classifier", layout="wide")

st.title("🧠 Naive Bayes Classification Dashboard")
st.markdown("Upload dataset → Select target → Train model → View performance")

# -----------------------------
# Sidebar (Inputs)
# -----------------------------
st.sidebar.header("⚙️ Model Configuration")

uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head())

    target_column = st.sidebar.selectbox("Select Target Column", df.columns)

    test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100

    train_button = st.sidebar.button("🚀 Train Model")

    if train_button:

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Convert categorical features
        X = pd.get_dummies(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = GaussianNB()
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        # -----------------------------
        # Results Section
        # -----------------------------
        st.subheader("📊 Model Performance")

        col1, col2 = st.columns(2)

        col1.metric("Training Accuracy", f"{train_acc:.4f}")
        col2.metric("Testing Accuracy", f"{test_acc:.4f}")

        st.subheader("📌 Confusion Matrix")

        cm = confusion_matrix(y_test, y_test_pred)

        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)
        st.pyplot(fig)

else:
    st.info("👈 Upload a dataset from the sidebar to begin.")