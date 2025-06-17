import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc

# Fungsi untuk mengonversi gambar ke base64
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Set Background
bg_image = get_base64("/mnt/data/162f4035-3117-4d15-b8f7-af3fa7e088d1.png")
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load data
url = 'https://raw.githubusercontent.com/Sandi-10/Personality/main/personality_dataset.csv'
df = pd.read_csv(url)
target_encoder = LabelEncoder()
df['Personality'] = target_encoder.fit_transform(df['Personality'])

# Session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_columns' not in st.session_state:
    st.session_state.X_columns = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'last_predictions' not in st.session_state:
    st.session_state.last_predictions = None

# Navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Informasi", "Pemodelan Data", "Prediksi", "Tuning Model", "Anggota Kelompok"])

# Informasi
if page == "Informasi":
    st.title("📘 Informasi Dataset")
    st.dataframe(df.head())
    st.subheader("📊 Deskripsi")
    st.write(df.describe(include='all'))
    st.subheader("📌 Distribusi Kelas")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Personality', ax=ax)
    ax.set_xticklabels(target_encoder.inverse_transform(sorted(df['Personality'].unique())))
    st.pyplot(fig)

# Pemodelan
elif page == "Pemodelan Data":
    st.title("📊 Pemodelan Data")
    model_choice = st.selectbox("Pilih Model", ["RandomForest", "KNN", "SVM"])
    df_model = df.copy()
    X = df_model.drop('Personality', axis=1)
    y = df_model['Personality']
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if st.button("🚀 Latih Model"):
        if model_choice == "RandomForest":
            model = RandomForestClassifier(random_state=42)
        elif model_choice == "KNN":
            model = KNeighborsClassifier()
        else:
            model = SVC(probability=True)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.session_state.model = model
        st.session_state.X_columns = X.columns.tolist()
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.last_predictions = y_pred

        st.metric("🎯 Akurasi", f"{accuracy_score(y_test, y_pred):.2f}")
        st.subheader("📋 Classification Report")
        report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, target_names=target_encoder.classes_)).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))

        st.subheader("📉 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_, ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        st.pyplot(fig)

        if hasattr(model, "feature_importances_"):
            st.subheader("🔍 Pentingnya Fitur")
            importances = model.feature_importances_
            fig_imp, ax_imp = plt.subplots()
            sns.barplot(x=importances, y=X.columns, ax=ax_imp)
            st.pyplot(fig_imp)

# Prediksi
elif page == "Prediksi":
    st.title("🔮 Prediksi Kepribadian")
    if st.session_state.model is None:
        st.warning("⚠️ Latih model terlebih dahulu di halaman Pemodelan Data.")
    else:
        input_data = {}
        for col in df.columns:
            if col != 'Personality':
                if df[col].dtype in [np.int64, np.float64]:
                    input_data[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
                else:
                    input_data[col] = st.selectbox(f"{col}", sorted(df[col].dropna().unique()))
        input_df = pd.DataFrame([input_data])

        if st.button("Prediksi"):
            for col in input_df.columns:
                if input_df[col].dtype == 'object':
                    le = LabelEncoder()
                    le.fit(df[col])
                    input_df[col] = le.transform(input_df[col])

            input_df = input_df[st.session_state.X_columns]
            prediction = st.session_state.model.predict(input_df)[0]
            predicted_label = target_encoder.inverse_transform([prediction])[0]
            st.success(f"Tipe Kepribadian: {predicted_label}")

            st.subheader("📈 Probabilitas Prediksi")
            probs = st.session_state.model.predict_proba(input_df)[0]
            st.bar_chart(pd.Series(probs, index=target_encoder.classes_))

# Tuning Model
elif page == "Tuning Model":
    st.title("🛠️ Tuning Hyperparameter")
    model_type = st.selectbox("Model", ["RandomForest", "KNN", "SVM"])
    df_model = df.copy()
    X = df_model.drop('Personality', axis=1)
    y = df_model['Personality']
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {}
    if model_type == "RandomForest":
        params = {
            'n_estimators': st.slider("n_estimators", 10, 200, 100, 10),
            'max_depth': st.slider("max_depth", 1, 20, 5)
        }
        model = RandomForestClassifier(**params, random_state=42)
    elif model_type == "KNN":
        params = {'n_neighbors': st.slider("n_neighbors", 1, 20, 5)}
        model = KNeighborsClassifier(**params)
    else:
        params = {'C': st.slider("C", 0.1, 10.0, 1.0), 'kernel': st.selectbox("kernel", ['linear', 'rbf'])}
        model = SVC(**params, probability=True)

    if st.button("🔍 Lakukan Tuning"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.metric("🎯 Akurasi Setelah Tuning", f"{acc:.2f}")
        st.session_state.model = model
        st.session_state.X_columns = X.columns.tolist()
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.last_predictions = y_pred

# Anggota
elif page == "Anggota Kelompok":
    st.title("👥 Anggota Kelompok")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("/mnt/data/162f4035-3117-4d15-b8f7-af3fa7e088d1.png", width=180)
    with col2:
        st.markdown("""
        ### 👩‍🏫 Diva Auliya Pusparini  
        🆔 NIM: 2304030041  
        ### 👩‍🎓 Paskalia Kanicha Mardian  
        🆔 NIM: 2304030062  
        ### 👨‍💻 Sandi Krisna Mukti  
        🆔 NIM: 2304030074  
        ### 👩‍⚕ Siti Maisyaroh  
        🆔 NIM: 2304030079
        """)

# Tambahan: Unduh Prediksi
if st.session_state.last_predictions is not None:
    if st.button("📥 Unduh Hasil Prediksi"):
        results = pd.DataFrame({
            'Actual': target_encoder.inverse_transform(st.session_state.y_test),
            'Predicted': target_encoder.inverse_transform(st.session_state.last_predictions)
        })
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("Unduh CSV", csv, "hasil_prediksi.csv", "text/csv")
