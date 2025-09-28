# app.py
import streamlit as st
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import sklearn.tree as tr
import sklearn.ensemble as es
import sklearn.linear_model as lm
import sklearn.neural_network as nn
import sklearn.metrics as m
import numpy as np

st.set_page_config(page_title="AI Student Marks Predictor", layout="wide")

st.title("🎓 AI Student Marks Prediction App")

# ---- Upload CSV ----
uploaded_file = st.file_uploader("Upload your dataset (AI-Data.csv)", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Data Loaded Successfully!")

    # ---- Correlation Heatmap ----
    st.subheader("📊 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sb.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # ---- Sidebar Menu for Graphs ----
    st.sidebar.header("Graph Options")
    graph_choice = st.sidebar.selectbox("Select Graph", [
        "None", "Marks Class Count", "Semester-wise", "Gender-wise",
        "Nationality-wise", "Grade-wise", "Section-wise", "Topic-wise",
        "Stage-wise", "Absent Days-wise"
    ])

    if graph_choice != "None":
        st.subheader(f"📈 {graph_choice} Graph")
        fig, ax = plt.subplots(figsize=(10, 6))
        if graph_choice == "Marks Class Count":
            sb.countplot(x='Class', data=data, order=['L', 'M', 'H'], ax=ax)
        elif graph_choice == "Semester-wise":
            sb.countplot(x='Semester', hue='Class', data=data, hue_order=['L','M','H'], ax=ax)
        elif graph_choice == "Gender-wise":
            sb.countplot(x='gender', hue='Class', data=data, order=['M','F'], hue_order=['L','M','H'], ax=ax)
        elif graph_choice == "Nationality-wise":
            sb.countplot(x='NationalITy', hue='Class', data=data, hue_order=['L','M','H'], ax=ax)
        elif graph_choice == "Grade-wise":
            sb.countplot(x='GradeID', hue='Class', data=data, hue_order=['L','M','H'], ax=ax)
        elif graph_choice == "Section-wise":
            sb.countplot(x='SectionID', hue='Class', data=data, hue_order=['L','M','H'], ax=ax)
        elif graph_choice == "Topic-wise":
            sb.countplot(x='Topic', hue='Class', data=data, hue_order=['L','M','H'], ax=ax)
        elif graph_choice == "Stage-wise":
            sb.countplot(x='StageID', hue='Class', data=data, hue_order=['L','M','H'], ax=ax)
        elif graph_choice == "Absent Days-wise":
            sb.countplot(x='StudentAbsenceDays', hue='Class', data=data, hue_order=['L','M','H'], ax=ax)
        st.pyplot(fig)

    # ---- Data Preprocessing ----
    drop_cols = ["gender","StageID","GradeID","NationalITy","PlaceofBirth",
                 "SectionID","Topic","Semester","Relation","ParentschoolSatisfaction",
                 "ParentAnsweringSurvey","AnnouncementsView"]
    data = data.drop(columns=drop_cols, errors="ignore")

    for column in data.columns:
        if data[column].dtype == object:
            le = pp.LabelEncoder()
            data[column] = le.fit_transform(data[column])

    # Train/Test split
    ind = int(len(data) * 0.70)
    feats = data.values[:, 0:4]
    lbls = data.values[:, 4]
    feats_train, feats_test = feats[:ind], feats[ind:]
    lbls_train, lbls_test = lbls[:ind], lbls[ind:]

    # ---- Train Models ----
    models = {
        "Decision Tree": tr.DecisionTreeClassifier(),
        "Random Forest": es.RandomForestClassifier(),
        "Perceptron": lm.Perceptron(),
        "Logistic Regression": lm.LogisticRegression(),
        "Neural Network (MLP)": nn.MLPClassifier(activation="logistic")
    }

    results = {}
    for name, model in models.items():
        model.fit(feats_train, lbls_train)
        preds = model.predict(feats_test)
        acc = m.accuracy_score(lbls_test, preds)
        results[name] = {"accuracy": acc, "report": m.classification_report(lbls_test, preds, output_dict=True)}

    # ---- Show Accuracy Table ----
    st.subheader("📌 Model Accuracy Comparison")
    acc_df = pd.DataFrame({k: [v["accuracy"]] for k, v in results.items()})
    st.dataframe(acc_df.T.rename(columns={0: "Accuracy"}))

    # ---- Test Input ----
    st.subheader("📝 Test Your Own Input")
    rai = st.number_input("Raised Hands", 0, 100, 10)
    res = st.number_input("Visited Resources", 0, 100, 20)
    dis = st.number_input("Discussions", 0, 50, 5)
    absc = st.selectbox("Absence Days", ["Under-7", "Above-7"])
    absc = 1 if absc == "Under-7" else 0

    arr = np.array([rai, res, dis, absc]).reshape(1, -1)

    if st.button("Predict"):
        for name, model in models.items():
            pred = model.predict(arr)[0]
            if pred == 0:
                pred = "H"
            elif pred == 1:
                pred = "M"
            elif pred == 2:
                pred = "L"
            st.write(f"**{name}:** {pred}")
