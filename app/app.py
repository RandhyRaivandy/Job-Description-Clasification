import streamlit as st
import pandas as pd
import joblib
import pickle
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px

# ===== NLTK =====
# import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# nltk.download('punkt', quiet=True)
# nltk.download('wordnet', quiet=True)
# nltk.download('omw-1.4', quiet=True)
# nltk.download('stopwords', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ============== CONFIG ==============
st.set_page_config(page_title="Job Description Clasification", layout="wide")

# ============== HELPERS ==============
@st.cache_resource
def load_artifacts(model_path: str, vectorizer_path: str, mapping_path: str, selector_path: str | None = None):
    model = joblib.load(model_path)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    with open(mapping_path, "rb") as f:
        type_mapping = pickle.load(f)

    try:
        if all(isinstance(k, str) for k in type_mapping.keys()) and all(isinstance(v, int) for v in type_mapping.values()):
            type_mapping = {v: k for k, v in type_mapping.items()}
    except Exception:
        pass

    selector = None
    if selector_path:
        try:
            with open(selector_path, "rb") as f:
                selector = pickle.load(f)
        except Exception as e:
            st.warning(f"Gagal memuat Chi-Square selector: {e}. Lanjut tanpa selector.")

    return model, vectorizer, type_mapping, selector

@st.cache_data
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def preprocess_text(text: str) -> str:
    """Tanpa stopword removal, hanya:
       lower -> regex non-huruf -> tokenisasi -> lemmatize -> join
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isascii()]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens).strip()

def build_top_words_per_class(df: pd.DataFrame, text_col: str, label_col: str, topn: int):
    result = {}
    for cls, subdf in df.groupby(label_col):
        tokens = " ".join(subdf[text_col].astype(str)).split()
        tokens = [t for t in tokens if t not in stop_words]
        counter = Counter(tokens)
        result[cls] = counter.most_common(topn)
    return result

def generate_wordcloud(tokens):
    if not tokens:
        return None
    wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(tokens))
    return wc

def safe_progress(bar, current, total):
    if total == 0:
        bar.progress(0.0)
    else:
        val = min(max(current / total, 0.0), 1.0)
        bar.progress(val)

# ============== UI ==============
st.title("Job Description Clasification")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Model & Artefak")
    model_path = st.text_input("Path model (.pkl)", "models/best_svm_model_chi2.pkl")
    vec_path   = st.text_input("Path TF-IDF vectorizer (.pkl)", "models/tfidf_vectorizer_no_stopwords.pkl")
    map_path   = st.text_input("Path type mapping (.pkl)", "models/type_mapping.pkl")
    selector_path = st.text_input("Path Chi-Square selector (.pkl)", "models/chi2_selector.pkl")

    try:
        model, vectorizer, type_mapping, selector = load_artifacts(
            model_path, vec_path, map_path, selector_path if selector_path.strip() else None
        )
    except Exception as e:
        st.error(f"Gagal memuat artefak: {e}")
        st.stop()

    st.markdown("---")
    st.header("Pengaturan")
    topn_words = st.number_input("Top-N kata / kelas", min_value=5, max_value=50, value=10, step=1)

# ---------- Upload ----------
uploaded = st.file_uploader("Unggah CSV dengan kolom `id` dan `text`", type=["csv"])
if uploaded is None:
    st.info("Silakan unggah file untuk mulai.")
    st.stop()

df_raw = pd.read_csv(uploaded, encoding="utf-8-sig")
df_raw.columns = [c.strip().lower() for c in df_raw.columns]

required = {"id", "text"}
missing = required - set(df_raw.columns)
if missing:
    st.error(f"Kolom wajib hilang: {missing}. Pastikan ada kolom: {required}")
    st.stop()

# ---------- Preprocess ----------
st.subheader("1) Preprocessing")
progress_pre = st.progress(0.0)
df = df_raw.copy()
total_rows = len(df)

for i, text in enumerate(df["text"]):
    df.at[i, "clean_text"] = preprocess_text(text)
    if (i + 1) % 10 == 0 or (i + 1) == total_rows:
        safe_progress(progress_pre, i + 1, total_rows)

df = df[df["clean_text"].str.strip() != ""]
st.success(f"Preprocessing selesai. Total data: {len(df)} baris.")

# ---------- Predict ----------
st.subheader("2) Prediksi Kelas")
n = len(df)
if n == 0:
    st.warning("Tidak ada baris setelah preprocessing.")
    st.stop()

progress_pred = st.progress(0.0)
batch_size = 100 if n > 1000 else n
predictions = []

for start in range(0, n, batch_size):
    end = min(start + batch_size, n)
    X_batch = vectorizer.transform(df['clean_text'].iloc[start:end])
    if selector is not None:
        try:
            X_batch = selector.transform(X_batch)
        except Exception as e:
            st.warning(f"Gagal menerapkan Chi-Square selector pada batch {start}:{end} → {e}")

    y_pred_idx = model.predict(X_batch)
    predictions.extend(y_pred_idx)
    safe_progress(progress_pred, end, n)

df["predicted_class"] = [type_mapping.get(i, str(i)) for i in predictions]
st.success("Prediksi selesai.")

# ---------- Tabel + Download ----------
st.subheader("3) Tabel Hasil Klasifikasi")
show_cols = ["id", "text", "predicted_class"]
st.dataframe(df[show_cols], height=420)

csv_bytes = to_csv_bytes(df[show_cols])
st.download_button(
    "Download hasil (CSV)",
    data=csv_bytes,
    file_name="hasil_klasifikasi.csv",
    mime="text/csv"
)

# ---------- Distribusi Kelas ----------
st.subheader("4) Distribusi Kelas")
dist = df["predicted_class"].value_counts().reset_index()
dist.columns = ["kelas", "jumlah"]
fig_bar = px.bar(dist, x="kelas", y="jumlah", text="jumlah", title="Jumlah Data per Kelas")
fig_bar.update_traces(textposition="outside")
st.plotly_chart(fig_bar, use_container_width=True)

# ---------- WordCloud per Kelas ----------
st.subheader("5) WordCloud per Kelas")
classes = sorted(df["predicted_class"].unique())
for cls in classes:
    tokens = " ".join(df.loc[df["predicted_class"] == cls, "clean_text"]).split()
    wc = generate_wordcloud(tokens)
    st.markdown(f"**Kelas: {cls}**")
    if wc is None:
        st.write("_Tidak ada token setelah preprocessing._")
        continue
    fig_wc, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig_wc)

# ---------- Top-N Words ----------
st.subheader(f"6) Top {topn_words} Kata per Kelas")
top_words = build_top_words_per_class(df, text_col="clean_text", label_col="predicted_class", topn=topn_words)
tabs = st.tabs(classes)
for tab, cls in zip(tabs, classes):
    with tab:
        pairs = top_words.get(cls, [])
        if not pairs:
            st.write("Tidak ada kata.")
            continue
        top_df = pd.DataFrame(pairs, columns=["kata", "freq"])
        st.dataframe(top_df, height=300)
        fig_top = px.bar(top_df, x="kata", y="freq", title=f"Top {topn_words} Kata - {cls}")
        st.plotly_chart(fig_top, use_container_width=True)
