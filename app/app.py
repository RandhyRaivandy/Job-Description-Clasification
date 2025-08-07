import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import re
import nltk
import base64
import plotly.express as px

# --- Import NLTK yang diperlukan ---
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# ------------------------------------

# --- NLTK Data Initialization ---
try:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
except LookupError:
    st.error("❌ Data NLTK (punkt, wordnet, omw-1.4, stopwords) tidak ditemukan.")
    st.info("Mohon unduh secara manual melalui terminal dengan perintah berikut:")
    st.code("python -m nltk.downloader punkt wordnet omw-1.4 stopwords")
    st.stop()
except NameError:
    st.error("❌ `WordNetLemmatizer` tidak diimpor. Pastikan Anda memiliki `from nltk.stem import WordNetLemmatizer`.")
    st.stop()

st.set_page_config(
    page_title="CV Content Classifier",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- Load Models and Transformers ---
@st.cache_resource
def load_artifacts(model_path: str, vectorizer_path: str, mapping_path: str, selector_path: str | None = None):
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"❌ Error: File model tidak ditemukan di '{model_path}'.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error memuat model dari '{model_path}': {e}")
        st.stop()

    try:
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
    except FileNotFoundError:
        st.error(f"❌ Error: File vectorizer tidak ditemukan di '{vectorizer_path}'.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error memuat vectorizer dari '{vectorizer_path}': {e}")
        st.stop()

    type_mapping_final = {}
    try:
        with open(mapping_path, "rb") as f:
            loaded_mapping = pickle.load(f)
        
        if isinstance(loaded_mapping, dict):
            if all(isinstance(k, str) and isinstance(v, int) for k, v in loaded_mapping.items()):
                type_mapping_final = {v: k for k, v in loaded_mapping.items()}
            elif all(isinstance(k, int) and isinstance(v, str) for k, v in loaded_mapping.items()):
                type_mapping_final = loaded_mapping
            else:
                st.warning(f"⚠️ Format type_mapping.pkl tidak sepenuhnya dikenali. Ditemukan: {loaded_mapping}. Akan mencoba melanjutkan.")
        else:
            st.error(f"❌ Error: type_mapping.pkl tidak berisi dictionary. Tipe ditemukan: {type(loaded_mapping)}.")
            st.stop()

        if not type_mapping_final:
            st.error("❌ Error: type_mapping.pkl berhasil dimuat tetapi hasilnya kosong atau tidak valid.")
            st.stop()

    except FileNotFoundError:
        st.error(f"❌ Error: File type_mapping tidak ditemukan di '{mapping_path}'.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error memuat type_mapping dari '{mapping_path}': {e}")
        st.stop()

    selector = None
    if selector_path and selector_path.strip():
        try:
            with open(selector_path, "rb") as f:
                selector = pickle.load(f)
        except FileNotFoundError:
            st.warning(f"⚠️ Peringatan: File Chi-Square selector tidak ditemukan di '{selector_path}'. Lanjut tanpa seleksi fitur Chi-Square.")
        except Exception as e:
            st.warning(f"⚠️ Peringatan: Gagal memuat Chi-Square selector dari '{selector_path}': {e}. Lanjut tanpa seleksi fitur Chi-Square.")

    return model, vectorizer, type_mapping_final, selector

# --- Initial Model Load ---
with st.status("Memuat model dan komponen aplikasi...", expanded=True) as status:
    try:
        model_path_default = "models/best_svm_model_chi2.pkl"
        vectorizer_path_default = "models/tfidf_vectorizer_no_stopwords.pkl"
        type_mapping_path_default = "models/type_mapping.pkl"
        chi2_selector_path_default = "models/chi2_selector.pkl"

        model, vectorizer, type_mapping, chi2_selector = load_artifacts(
            model_path_default, vectorizer_path_default,
            type_mapping_path_default, chi2_selector_path_default
        )
        
        # Perbaikan error "show_spinner" dan menyembunyikan status
        status.update(label="Model siap!", state="complete", expanded=False)
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan fatal selama startup: {e}")
        status.update(label=f"Startup gagal: {e}", state="error", expanded=True)
        st.stop()

# --- Preprocessing Function ---
def preprocess_text(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    sentences = sent_tokenize(text)
    processed_sentences = []
    for sent in sentences:
        sent = sent.lower()
        sent = re.sub(r"[^a-z\s]", " ", sent)
        sent = re.sub(r"\s+", " ", sent).strip()
        tokens = word_tokenize(sent)
        tokens = [t for t in tokens if t.isascii()]
        tokens = [t for t in tokens if t not in stop_words]
        lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
        processed = " ".join(lemmatized)
        if processed:
            processed_sentences.append(processed)
    return processed_sentences

# --- Prediction Function for Single Text Input (CV) ---
def predict_text(text_input: str) -> pd.DataFrame:
    processed_sentences = preprocess_text(text_input)
    if not processed_sentences:
        return pd.DataFrame(columns=["Class", "Percentage"])
    
    X = vectorizer.transform(processed_sentences)
    if chi2_selector is not None:
        X = chi2_selector.transform(X)
    
    preds = model.predict(X)
    unique, counts = np.unique(preds, return_counts=True)
    total = sum(counts)
    
    result_dict = {}
    for idx, i in enumerate(unique):
        # Perbaikan KeyError menggunakan .get()
        class_name = type_mapping.get(i, f"Unknown_ID_{i}")
        result_dict[class_name] = (counts[idx] / total) * 100
        
    all_class_names = list(type_mapping.values()) 
    for cls_name in all_class_names:
        if cls_name not in result_dict:
            result_dict[cls_name] = 0.0
            
    df_result = pd.DataFrame(result_dict.items(), columns=["Class", "Percentage"])
    df_result = df_result.sort_values(by="Percentage", ascending=False).reset_index(drop=True)
    return df_result

# --- Prediction Function for CSV Input ---
def predict_csv(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    all_texts_for_processing = []
    cv_sentence_map = {}
    current_sentence_index = 0
    total_cvs = len(df)
    
    with st.status("Memproses file CSV...", expanded=True) as status_csv:
        status_csv.write("Langkah 1/2: Melakukan preprocessing teks...")
        preprocess_bar = st.progress(0)
        for idx, row in df.iterrows():
            cv_id = row["id"]
            text_content = str(row["text"])
            processed_sentences_for_cv = preprocess_text(text_content)
            cv_sentence_map[cv_id] = {
                'start_idx': current_sentence_index,
                'end_idx': current_sentence_index + len(processed_sentences_for_cv),
                'sentences': processed_sentences_for_cv
            }
            all_texts_for_processing.extend(processed_sentences_for_cv)
            current_sentence_index += len(processed_sentences_for_cv)
            preprocess_bar.progress((idx + 1) / total_cvs)
        
        if not all_texts_for_processing:
            status_csv.update(label="Tidak ada teks valid yang ditemukan di CSV setelah preprocessing.", state="warning", expanded=False)
            return pd.DataFrame()

        status_csv.write("Langkah 2/2: Melakukan prediksi dan agregasi hasil...")
        prediction_bar = st.progress(0)
        X_all = vectorizer.transform(all_texts_for_processing)
        if chi2_selector is not None:
            X_all_selected = chi2_selector.transform(X_all)
        else:
            X_all_selected = X_all
        all_preds = model.predict(X_all_selected)

        processed_cv_count = 0
        for cv_id, info in cv_sentence_map.items():
            start = info['start_idx']
            end = info['end_idx']
            cv_preds = all_preds[start:end]
            result_row = {"id": cv_id}
            if len(cv_preds) > 0:
                unique, counts = np.unique(cv_preds, return_counts=True)
                total = sum(counts)
                cv_percentages = {type_mapping.get(i, f"Unknown_ID_{i}"): (counts[idx] / total) * 100 for idx, i in enumerate(unique)}
                for cls_name in list(type_mapping.values()):
                    result_row[cls_name] = cv_percentages.get(cls_name, 0.0)
            else:
                for cls_name in list(type_mapping.values()):
                    result_row[cls_name] = 0.0
            results.append(result_row)
            processed_cv_count += 1
            prediction_bar.progress(processed_cv_count / total_cvs)
        
        status_csv.update(label="Proses selesai!", state="complete", expanded=False)

    all_class_names_sorted = sorted(list(type_mapping.values()))
    final_df_cols = ["id"] + all_class_names_sorted
    df_result = pd.DataFrame(results)
    df_result = df_result.reindex(columns=final_df_cols, fill_value=0.0)
    return df_result

# --- Helper Function for Download Button ---
@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

# --- Function to Display Class Metrics as Cards ---
def display_class_metrics(df_result: pd.DataFrame):
    st.subheader("📊 Distribusi Konten CV")
    df_result = df_result.sort_values(by="Percentage", ascending=False)
    
    num_cols = 3
    if df_result['Percentage'].sum() > 0:
        df_result['Percentage_Normalized'] = (df_result['Percentage'] / df_result['Percentage'].sum()) * 100
    else:
        df_result['Percentage_Normalized'] = 0.0
        
    if 'Class' in df_result.columns and 'Percentage' in df_result.columns:
        fig = px.bar(
            df_result,
            x='Percentage',
            y='Class',
            orientation='h',
            labels={'Percentage': 'Persentase (%)', 'Class': 'Kategori'},
            title='Distribusi Kategori CV',
            color='Class'
        )
        # Menghilangkan legenda
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(num_cols)
    for i, (index, row) in enumerate(df_result.iterrows()):
        with cols[i % num_cols]:
            class_name = row["Class"]
            percentage = row["Percentage_Normalized"] if 'Percentage_Normalized' in df_result.columns else row["Percentage"]
            st.metric(label=class_name, value=f"{percentage:.1f}%")
    st.markdown(f"**Catatan:** Persentase menunjukkan proporsi kalimat dalam CV yang terklasifikasi ke setiap kategori.")

# ============== UI ==============
st.title("🌟 CV Content Classifier")
st.markdown("""
Aplikasi ini membantu menganalisis dan mengklasifikasikan bagian-bagian dari CV Anda 
ke dalam kategori konten yang relevan seperti Pendidikan, Pengalaman, Keterampilan, dll.
""")

st.markdown("""
<style>
div.stDownloadButton > button {
    background-color: #4CAF50; /* Green */
    color: white;
    padding: 10px 24px;
    border-radius: 8px;
    border: none;
    font-size: 16px;
    cursor: pointer;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    transition: 0.3s;
}
div.stDownloadButton > button:hover {
    background-color: #45a049; /* Darker green on hover */
    box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)


st.markdown("---")

input_type = st.selectbox(
    "**Pilih Metode Input:**",
    options=["Teks Langsung 📝", "Unggah File CSV 📁"],
    index=0,
    help="Pilih apakah Anda ingin memasukkan teks CV secara langsung atau mengunggah file CSV."
)

st.markdown("---")

if input_type == "Teks Langsung 📝":
    user_input = st.text_area("Masukkan teks CV Anda di sini:", height=250, 
                              placeholder="Contoh: 'Sales Associate at Retail Giant (2016-2017). Exceeded sales targets by 15%. Dean's List (2019, 2020). Highly motivated and results-oriented professional with 5+ years of experience. Fluent in English and Bahasa Indonesia. Associate Degree in Electrical Engineering, Technical College, 2018.'")
    
    if st.button("Analisis Teks CV", type="primary"):
        if user_input:
            with st.status("Menganalisis teks CV...", expanded=True) as status:
                status.write("Melakukan preprocessing dan prediksi...")
                df_result = predict_text(user_input)
                status.update(label="Analisis Selesai!", state="complete", expanded=False)
            
            if not df_result.empty:
                display_class_metrics(df_result)
            else:
                st.warning("Tidak ada kelas yang dapat diprediksi dari teks yang diberikan. Pastikan teks relevan dan memiliki konten yang cukup.")
        else:
            st.warning("Mohon masukkan teks CV untuk memulai analisis.")

elif input_type == "Unggah File CSV 📁":
    uploaded_file = st.file_uploader("Unggah file CSV Anda (harus memiliki kolom 'id' dan 'text')", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.lower() 

        if "id" in df.columns and "text" in df.columns:
            st.subheader("Data CSV yang Diunggah")
            st.dataframe(
                df.head().style.set_properties(**{'width': '50px'}, subset=pd.IndexSlice[:, ['id']])
                               .set_properties(**{'width': '600px'}, subset=pd.IndexSlice[:, ['text']]),
                use_container_width=False
            )
            st.caption(f"Menampilkan {min(5, len(df))} baris pertama dari {len(df)} baris data.")
            
            if st.button("Proses CSV", type="primary"):
                df_result = predict_csv(df)
                
                if not df_result.empty:
                    st.success("Proses selesai! Hasil prediksi telah ditampilkan dan siap diunduh.")
                    st.subheader("📊 Hasil Prediksi Distribusi Kelas per CV")
                    st.dataframe(df_result, use_container_width=True)
                    
                    csv_data = convert_df_to_csv(df_result)
                    st.download_button(
                        label="📥 Unduh Hasil Prediksi CSV",
                        data=csv_data,
                        file_name="hasil_prediksi_cv.csv",
                        mime="text/csv",
                        key="download_csv_button"
                    )
                else:
                    st.warning("File CSV diproses, tetapi tidak ada prediksi yang dihasilkan. Pastikan kolom 'text' berisi konten yang relevan.")
        else:
            st.error("Format CSV tidak sesuai. File CSV harus memiliki kolom **'id'** dan **'text'**.")
            st.info("Contoh format CSV:\n\n```csv\nid,text\n1,\"Ini adalah pengalaman saya...\"\n2,\"Pendidikan terakhir saya...\"\n```")
    else:
        st.info("Silakan unggah file CSV Anda di sini untuk memulai proses klasifikasi.")
