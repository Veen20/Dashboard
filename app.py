# app.py — Dashboard Sentimen Ulasan Google Maps Samsat (Dark/Light + Tabs + Insight)
# ===================================================================================
# Fitur:
# - Dark/Light toggle (CSS)
# - Crawl manual (limit slider) + filter tanggal/sentimen/kata kunci
# - Analisis sentimen otomatis (IndoBERTweet fine-tuned, 3 kelas)
# - Tabs: Overview, Analisis, Visualisasi, Komentar, Saran, Ekspor
# - Visual: Pie, Bar, Line, Heatmap jam×hari, Top kata per sentimen, Trending complaints
# - Insight & Rekomendasi otomatis
# - Ekspor CSV (data terfilter) & TXT (insight + saran)
# - Cache model & data untuk performa (ramah deploy)
# - Menggunakan review_time sebagai waktu utama analisis; created_at untuk monitoring crawl

import calendar
import io
import re
from typing import List
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from supabase import create_client
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

import numpy as np
import torch
# st.write(f"Numpy version: {np.__version__}")
# st.write(f"Torch version: {torch.__version__}")

from crawling_gmaps import crawl_gmaps_reviews
# Ambil secrets dari Streamlit Cloud
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Dashboard Sentimen Samsat",
    layout="wide",
    page_icon="🛰️",
)

# -----------------------------
# Theme Toggle (Light/Dark)
# -----------------------------
mode = st.sidebar.selectbox("Tampilan", ["🌙 Dark", "☀️ Light"], index=0)

DARK_CSS = """
<style>
:root {
  --bg: #0f172a;         /* slate-900 */
  --panel: #0b1220;      /* custom dark */
  --muted: #94a3b8;      /* slate-400 */
  --text: #e5e7eb;       /* gray-200 */
  --pos: #16a34a;        /* green-600 */
  --neg: #dc2626;        /* red-600 */
  --neu: #64748b;        /* slate-500 */
  --accent: #2563eb;     /* blue-600 */
}
html, body, [data-testid="stAppViewContainer"] { background: var(--bg); color: var(--text); }
[data-testid="stSidebar"] { background: #0b1220; border-right: 1px solid #1f2937; }
h1,h2,h3,h4,h5,h6 { color: #f8fafc !important; }
hr { border-color: #1f2937; }
.block-container { padding-top: 1.2rem; }
.card { border: 1px solid #1f2937; background: var(--panel); border-radius: 18px; padding: 14px; }
.badge { padding: 3px 10px; border-radius: 999px; font-weight: 600; font-size: 12px; }
.badge-pos { background: rgba(22,163,74,.15); color: var(--pos); }
.badge-neg { background: rgba(220,38,38,.15); color: var(--neg); }
.badge-neu { background: rgba(100,116,139,.18); color: var(--neu); }
.small { color: var(--muted); font-size: 12px; }
.kpi { background: var(--panel); border: 1px solid #1f2937; border-radius: 16px; padding: 14px; }
.kpi .label { color: var(--muted); font-size: 13px; }
.kpi .value { font-weight: 700; font-size: 22px; }
a { color: #93c5fd; }
</style>
"""

LIGHT_CSS = """
<style>
:root {
  --bg: #ffffff;
  --panel: #ffffff;
  --muted: #6b7280;
  --text: #111827;
  --pos: #16a34a;
  --neg: #dc2626;
  --neu: #64748b;
  --accent: #2563eb;
}
html, body, [data-testid="stAppViewContainer"] { background: var(--bg); color: var(--text); }
[data-testid="stSidebar"] { background: #f9fafb; border-right: 1px solid #e5e7eb; }
h1,h2,h3,h4,h5,h6 { color: #111827 !important; }
hr { border-color: #e5e7eb; }
.block-container { padding-top: 1.2rem; }
.card { border: 1px solid #e5e7eb; background: var(--panel); border-radius: 12px; padding: 14px; }
.badge { padding: 3px 10px; border-radius: 999px; font-weight: 600; font-size: 12px; }
.badge-pos { background: rgba(22,163,74,.12); color: var(--pos); }
.badge-neg { background: rgba(220,38,38,.12); color: var(--neg); }
.badge-neu { background: rgba(100,116,139,.12); color: var(--neu); }
.small { color: var(--muted); font-size: 12px; }
.kpi { background: var(--panel); border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px; }
.kpi .label { color: var(--muted); font-size: 13px; }
.kpi .value { font-weight: 700; font-size: 22px; }
a { color: #1d4ed8; }
</style>
"""

st.markdown(DARK_CSS if mode.startswith("🌙") else LIGHT_CSS, unsafe_allow_html=True)

# -----------------------------
# Supabase Client
# -----------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# Utils
# -----------------------------
ID_STOPWORDS = list(set("""
yang dan di ke dari ada untuk dengan pada adalah itu ini saya aku kami kita mereka kalian dia ia tidak bukan
ya gak nggak tdk ga yg aja nya dong deh lah nih sih pun atau serta karena sehingga agar supaya kalau kalo
tentang dalam sudah belum akan telah masih saat ketika lalu jadi maka pun namun tapi tapii banget bgt sangat
lebih kurang bisa dapat dapatkah kah toh kok punlah punnya trus klw klo be mau semua dg orang mengerti pake
""".split()))

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"http\\S+|www\\.\\S+", " ", s)
    s = re.sub(r"[@#]\\w+", " ", s)
    s = re.sub(r"[^a-zA-Z0-9À-ÿ\\s]", " ", s)
    s = re.sub(r"\\s+", " ", s)
    return s.strip().lower()

def extract_top_words(series: pd.Series, top_n: int = 20, exclude: set = None):
    exclude = exclude or set()
    counter = {}
    for txt in series.fillna(""):
        for w in clean_text(txt).split():
            if len(w) <= 2 or w in ID_STOPWORDS or w in exclude:
                continue
            counter[w] = counter.get(w, 0) + 1
    if not counter:
        return pd.DataFrame(columns=["word", "count"])
    dfw = (pd.DataFrame(counter.items(), columns=["word", "count"])
             .sort_values("count", ascending=False)
             .head(top_n))
    return dfw

@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    model_name = "Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)  # labels: negative, neutral, positive
    return TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False)

@st.cache_data(ttl=60, show_spinner=False)
def fetch_comments() -> pd.DataFrame:
    # Ambil semua kolom (termasuk review_time/review_time_raw bila ada)
    res = supabase.table("comments").select("*").order("created_at", desc=True).execute()
    df = pd.DataFrame(res.data or [])

    # Normalisasi kolom waktu
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    else:
        df["created_at"] = pd.NaT

    if "review_time" in df.columns:
        df["review_time"] = pd.to_datetime(df["review_time"], errors="coerce")
    else:
        # kalau belum ada kolomnya di DB, tetap buat agar downstream aman
        df["review_time"] = pd.NaT

    if "relative_time" not in df.columns:
        df["relative_time"] = None

    # Kolom lain wajib ada
    for col, default in [("comment",""), ("username","Anonim"), ("platform","gmaps")]:
        if col not in df.columns:
            df[col] = default

    if "sentiment" not in df.columns:
        # buat kolom kosong agar downstream (filter/plot) tidak error
        df["sentiment"] = pd.Series(dtype="object")

    # Kolom waktu yang dipakai utama untuk analisis/filter
    # Prioritas: review_time (asli dari Gmaps) -> fallback ke created_at
    df["time_used"] = df["review_time"].fillna(df["created_at"])

    return df

def map_label(label: str) -> str:
    m = {"positive":"Positif","neutral":"Netral","negative":"Negatif"}
    return m.get((label or "").lower(), "Netral")
    

def predict_sentiment(pipe, texts: List[str]) -> List[str]:
    preds = pipe(texts, truncation=True)
    return [map_label(p["label"]) for p in preds] 
    

def badge_html(sent: str) -> str:
    s = (sent or "Netral").capitalize()
    css = "badge-neu"
    if s == "Positif": css = "badge-pos"
    elif s == "Negatif": css = "badge-neg"
    return f"<span class='badge {css}'>{s}</span>"

def auto_insights(df_f: pd.DataFrame) -> list:
    if df_f.empty: return ["Data kosong untuk periode/filter saat ini."]
    total = len(df_f)
    pos_r = (df_f["sentiment"]=="Positif").mean() if "sentiment" in df_f.columns else 0.0
    neg_r = (df_f["sentiment"]=="Negatif").mean() if "sentiment" in df_f.columns else 0.0
    neu_r = (df_f["sentiment"]=="Netral").mean() if "sentiment" in df_f.columns else 0.0
    insights = [
        f"Total ulasan: **{total:,}** (Positif {pos_r*100:.1f}%, Netral {neu_r*100:.1f}%, Negatif {neg_r*100:.1f}%)."
    ]
    if "time_used" in df_f.columns:
        ts = (df_f.assign(date=df_f["time_used"].dt.date)
                 .groupby(["date","sentiment"]).size().reset_index(name="count"))
        if not ts.empty:
            neg_ts = ts[ts["sentiment"]=="Negatif"].sort_values("count", ascending=False)
            if not neg_ts.empty and neg_ts.iloc[0]["count"] >= max(3, np.percentile(neg_ts["count"], 90) if len(neg_ts)>5 else 3):
                d = neg_ts.iloc[0]["date"]
                c = int(neg_ts.iloc[0]["count"])
                insights.append(f"📈 Puncak keluhan terjadi pada **{d}**: **{c}** ulasan negatif.")
    neg_words = extract_top_words(df_f.query("sentiment=='Negatif'")["comment"], top_n=5) if "sentiment" in df_f.columns else pd.DataFrame()
    if isinstance(neg_words, pd.DataFrame) and not neg_words.empty:
        topics = ", ".join(neg_words["word"].tolist())
        insights.append(f"Top isu negatif: **{topics}**.")
    if pos_r >= 0.7:
        insights.append("Mayoritas pengguna **puas** — pertahankan keramahan & kecepatan layanan.")
    elif neg_r >= 0.3:
        insights.append("Proporsi keluhan cukup tinggi — **evaluasi proses antrian & informasi loket**.")
    else:
        insights.append("Positif + Netral mendominasi — jaga konsistensi di jam sibuk.")
    return insights

def auto_recommendations(df_f: pd.DataFrame) -> list:
    if df_f.empty: return ["Belum ada rekomendasi — tidak ada data pada rentang/filter ini."]
    text_neg = " ".join(df_f.query("sentiment=='Negatif'")["comment"].str.lower().tolist()) if "sentiment" in df_f.columns else ""
    recs = []
    if any(k in text_neg for k in ["parkir","parking"]):
        recs.append("🔧 **Parkir**: Tegaskan kebijakan parkir gratis & awasi petugas. Pasang papan informasi besar di pintu masuk.")
    if any(k in text_neg for k in ["antri","antre","antrean","antrian","queue"]):
        recs.append("⏱️ **Antrian**: Terapkan nomor antrian digital & tambah loket pada jam sibuk (pagi & Senin). Umumkan estimasi waktu tunggu.")
    if "pungli" in text_neg or ("uang" in text_neg and "cek fisik" in text_neg):
        recs.append("🛡️ **Integritas**: Kampanye anti pungli di area cek fisik & loket; sediakan kanal pengaduan QR anonim.")
    if any(k in text_neg for k in ["cek fisik","gesek rangka","cek mesin"]):
        recs.append("🔎 **Cek Fisik**: Standarisasi prosedur & publikasikan biaya resmi; sediakan contoh formulir & petugas bantuan.")
    if any(k in text_neg for k in ["plat","tnkb"]):
        recs.append("🚚 **TNKB**: Atur shift istirahat bergantian agar layanan tidak terhenti saat istirahat.")
    if any(k in text_neg for k in ["informasi","pengeras suara","speaker","panggil"]):
        recs.append("📣 **Informasi Publik**: Tambah pengeras suara & layar antrian; perbarui papan info syarat & biaya.")
    if not recs:
        recs.append("✅ Lanjutkan pemantauan berkala; fokus di jam sibuk & respons cepat terhadap tren keluhan baru.")
    return recs


from sklearn.feature_extraction.text import CountVectorizer

def trending_phrases(df, recent_days=7, prev_days=7, top_n=10, ngram_range=(2,3)):
    """Trending frasa (bigram/trigram) dari semua komentar"""
    if df.empty or "time_used" not in df.columns:
        return pd.DataFrame(columns=["phrase","recent_count","prev_count","lift"])

    df = df.copy()
    df["date"] = df["time_used"].dt.date

    comments_recent, comments_prev = [], []
    max_date = df["date"].max()
    if pd.isna(max_date): 
        return pd.DataFrame(columns=["phrase","recent_count","prev_count","lift"])

    recent_start = max_date - timedelta(days=recent_days)
    prev_start   = recent_start - timedelta(days=prev_days)

    comments_recent = df[df["date"] > recent_start]["comment"].fillna("").tolist()
    comments_prev   = df[(df["date"] > prev_start) & (df["date"] <= recent_start)]["comment"].fillna("").tolist()

    vectorizer = CountVectorizer(lowercase=True, stop_words=ID_STOPWORDS, ngram_range=ngram_range)

    # Recent
    X_recent = vectorizer.fit_transform(comments_recent) if comments_recent else None
    recent_counts = pd.DataFrame({
        "phrase": vectorizer.get_feature_names_out() if X_recent is not None else [],
        "recent_count": X_recent.sum(axis=0).A1 if X_recent is not None else []
    })

    # Previous
    if comments_prev:
        X_prev = vectorizer.transform(comments_prev)
        prev_counts = pd.DataFrame({
            "phrase": vectorizer.get_feature_names_out(),
            "prev_count": X_prev.sum(axis=0).A1
        })
    else:
        prev_counts = pd.DataFrame(columns=["phrase","prev_count"])

    merged = recent_counts.merge(prev_counts, on="phrase", how="left").fillna(0)
    merged["lift"] = (merged["recent_count"] + 1) / (merged["prev_count"] + 1)
    merged = merged.sort_values(["lift","recent_count"], ascending=[False, False])

    return merged.head(top_n)



# -----------------------------
# Sidebar: Kontrol
# -----------------------------
st.sidebar.title("⚙️ Kontrol")
st.sidebar.caption("Atur crawling & filter data")

# Crawl controls
st.sidebar.subheader("Crawling")
crawl_limit = st.sidebar.slider("Limit review per crawl", 5, 50, 10, step=5)
if st.sidebar.button("🚀 Crawl Ulasan Terbaru"):
    with st.spinner("Mengambil ulasan dari Google Maps..."):
        added = crawl_gmaps_reviews(limit=crawl_limit)
        fetch_comments.clear()  # refresh cache
    if added == 0:
        st.sidebar.info("ℹ️ Belum ada ulasan terbaru. Data saat ini sudah paling update.")
    else:
        st.sidebar.success(f"{added} komentar baru dimasukkan.")
    

st.sidebar.markdown("---")

df = fetch_comments()
if df.empty:
    st.info("Belum ada data komentar di Supabase. Klik **Crawl Ulasan Terbaru** di sidebar.")
    st.stop()

# Filters
# Ambil tanggal dari dataframe
df_dates = df["time_used"].dropna()
if not df_dates.empty:
    min_date = df_dates.min().date()
    max_date = df_dates.max().date()
else:
    today = datetime.now().date()
    min_date = today - timedelta(days=90)
    max_date = today

# Sidebar filter
date_range = st.sidebar.date_input(
    "Rentang tanggal (ulasan asli)",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    date_from, date_to = date_range
else:
    date_from, date_to = min_date, max_date


sentiment_options = ["Positif", "Netral", "Negatif"]
pick_sentiments = st.sidebar.multiselect("Filter sentimen", sentiment_options, default=sentiment_options)
keyword = st.sidebar.text_input("Cari kata kunci", placeholder="cth: parkir, pungli, cepat")

st.sidebar.markdown("---")
st.sidebar.caption("Model: IndoBERTweet (fine-tuned) • Cache refresh tiap 60 detik")
st.sidebar.subheader("📈 Top Trending Phrases")
recent_days = st.sidebar.slider("Periode Terbaru (hari)", 3, 30, 7, step=1)
prev_days   = st.sidebar.slider("Periode Sebelumnya (hari)", 3, 30, 7, step=1)
top_n      = st.sidebar.slider("Jumlah kata/frasa teratas", 5, 20, 10, step=1)

# -----------------------------
# Title
# -----------------------------
st.title("Dashboard Analisis Sentimen Ulasan Publik Samsat UPTB Palembang 1")
st.caption("Sumber: Google Maps (via SerpAPI) • Database: Supabase • Tampilan: Streamlit")

# -----------------------------
# Load data & model
# -----------------------------


# Info update terakhir berdasar created_at
if "created_at" in df.columns and pd.notna(df["created_at"]).any():
    last_update = df["created_at"].max()
    st.caption(f"🗓️ Data terakhir dicrawl: {last_update.strftime('%d %b %Y %H:%M')}")

with st.spinner("Memuat model sentimen…"):
    pipe = load_sentiment_pipeline()

# Prediksi & simpan ke Supabase bila kolom sentiment kosong / masih null
if "sentiment" not in df.columns or df["sentiment"].isna().any() or (df["sentiment"] == "").any():
    st.info("🔄 Menjalankan analisis sentimen untuk komentar baru...")

    # Pastikan kolom 'sentiment' ada
    if "sentiment" not in df.columns:
        df["sentiment"] = pd.Series(dtype="object")

    mask_new = df["sentiment"].isna() | (df["sentiment"] == "")
    texts = df.loc[mask_new, "comment"].fillna("").astype(str).tolist()

    if texts:
        sentiments = []
        batch = 8  # lebih aman untuk server
        # Filter teks kosong/None
        texts = [t for t in texts if t and t.strip()]
        for i in range(0, len(texts), batch):
            try:
                sentiments.extend(predict_sentiment(pipe, texts[i:i+batch]))
            except Exception as e:
                st.warning(f"Gagal proses batch {i}-{i+batch}: {e}")
                continue


        # Update dataframe lokal
        df.loc[mask_new, "sentiment"] = sentiments

        # Update ke Supabase (kolom sentiment) berdasar primary key 'id'
        if "id" in df.columns:
            for idx, sent in zip(df.loc[mask_new, "id"], sentiments):
                supabase.table("comments").update({"sentiment": sent}).eq("id", idx).execute()

        st.success(f"✅ {len(sentiments)} komentar baru berhasil dianalisis & disimpan ke Supabase.")

# -----------------------------
# Apply filters (berdasar review_time -> fallback ke created_at)
# -----------------------------
# time_used sudah disiapkan di fetch_comments()
mask_date = (df["time_used"].dt.date >= date_from) & (df["time_used"].dt.date <= date_to)
mask_sent = df["sentiment"].isin(pick_sentiments) if "sentiment" in df.columns and pick_sentiments else True
mask_kw = df["comment"].str.contains(keyword, case=False, na=False) if keyword else True
df_f = df[mask_date & mask_sent & mask_kw].copy()

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_analisis, tab_visual, tab_comments, tab_saran, tab_export = st.tabs(
    ["📊 Overview", "🔎 Analisis", "📈 Visualisasi", "💬 Komentar", "💡 Saran", "⬇️ Ekspor"]
)

# ==== OVERVIEW ====
with tab_overview:
    st.subheader("Ringkasan")
    c1, c2, c3, c4 = st.columns(4)
    total_reviews = int(len(df_f))
    pos_rate = float((df_f["sentiment"] == "Positif").mean() * 100) if total_reviews and "sentiment" in df_f.columns else 0.0
    neu_rate = float((df_f["sentiment"] == "Netral").mean() * 100) if total_reviews and "sentiment" in df_f.columns else 0.0
    neg_rate = float((df_f["sentiment"] == "Negatif").mean() * 100) if total_reviews and "sentiment" in df_f.columns else 0.0

    with c1:
        st.markdown(f"<div class='kpi'><div class='label'>Total Ulasan</div><div class='value'>{total_reviews:,}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi'><div class='label'>Positif</div><div class='value' style='color:var(--pos)'>{pos_rate:.1f}%</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi'><div class='label'>Netral</div><div class='value' style='color:var(--neu)'>{neu_rate:.1f}%</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='kpi'><div class='label'>Negatif</div><div class='value' style='color:var(--neg)'>{neg_rate:.1f}%</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    # 👉 Tambahkan Insight Otomatis di sini
    st.subheader("📊 Insight Otomatis")
    insights = auto_insights(df_f)
    for ins in insights:
        st.markdown(f"- {ins}")

    st.markdown("---")
    st.subheader(f"Top Trending Phrases ({recent_days} hari terakhir vs {prev_days} hari sebelumnya)")

   # Gunakan semua komentar, tidak hanya negatif
    tw = trending_phrases(df_f, recent_days=recent_days, prev_days=prev_days, top_n=top_n)

    if tw.empty:
        st.info("Belum ada tren frasa menonjol pada periode ini.")
    else:
        st.dataframe(tw, use_container_width=True)
        fig_tr = px.bar(
            tw, x="phrase", y="recent_count", text="lift",
            labels={"recent_count":"Ulasan Terbaru", "phrase":"Frasa"},
            title="Top Frasa - Perbandingan Recent vs Previous"
        )
        st.plotly_chart(fig_tr, use_container_width=True)



   

# ==== ANALISIS ====
with tab_analisis:
    st.subheader("Distribusi Sentimen")
    if df_f.empty or "sentiment" not in df_f.columns:
        st.info("Data kosong setelah filter.")
    else:
        pie_df = df_f["sentiment"].value_counts().reset_index()
        pie_df.columns = ["sentiment", "count"]
        fig = px.pie(
            pie_df, names="sentiment", values="count", hole=0.35, color="sentiment",
            color_discrete_map={"Positif":"#16a34a","Negatif":"#dc2626","Netral":"#64748b"}
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        bar_df = pie_df.sort_values("sentiment")
        figb = px.bar(bar_df, x="sentiment", y="count",
                      color="sentiment",
                      color_discrete_map={"Positif":"#16a34a","Negatif":"#dc2626","Netral":"#64748b"})
        figb.update_layout(xaxis_title="", yaxis_title="Jumlah Ulasan")
        st.plotly_chart(figb, use_container_width=True)

    st.markdown("---")
    st.subheader("Top Kata per Sentimen")
    colp, coln, colu = st.columns(3)
    if df_f.empty or "sentiment" not in df_f.columns:
        st.info("Data kosong untuk analisis kata.")
    else:
        with colp:
            st.markdown("**Positif**")
            pos_top = extract_top_words(df_f.query("sentiment=='Positif'")["comment"], top_n=12)
            if pos_top.empty: st.write("_Tidak ada kata dominan._")
            else:
                figp = px.bar(pos_top, x="word", y="count"); figp.update_layout(xaxis_title="", yaxis_title="Frekuensi")
                st.plotly_chart(figp, use_container_width=True)
        with coln:
            st.markdown("**Negatif**")
            neg_top = extract_top_words(df_f.query("sentiment=='Negatif'")["comment"], top_n=12)
            if neg_top.empty: st.write("_Tidak ada kata dominan._")
            else:
                fign = px.bar(neg_top, x="word", y="count"); fign.update_layout(xaxis_title="", yaxis_title="Frekuensi")
                st.plotly_chart(fign, use_container_width=True)
        with colu:
            st.markdown("**Netral**")
            neu_top = extract_top_words(df_f.query("sentiment=='Netral'")["comment"], top_n=12)
            if neu_top.empty: st.write("_Tidak ada kata dominan._")
            else:
                figu = px.bar(neu_top, x="word", y="count"); figu.update_layout(xaxis_title="", yaxis_title="Frekuensi")
                st.plotly_chart(figu, use_container_width=True)

# ==== VISUALISASI ====
with tab_visual:
    st.subheader("Tren Ulasan per Hari (dibagi Sentimen)")
    if df_f.empty:
        st.info("Data kosong setelah filter.")
    else:
        ts = (
            df_f.assign(date=df_f["time_used"].dt.date)
                .groupby(["date", "sentiment"]).size()
                .reset_index(name="count")
                .sort_values("date")
        )
        fig2 = px.line(ts, x="date", y="count", color="sentiment", markers=True,
                       color_discrete_map={"Positif":"#16a34a","Negatif":"#dc2626","Netral":"#64748b"})
        fig2.update_layout(xaxis_title="", yaxis_title="Jumlah ulasan")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Heatmap Waktu (Hari × Jam)")
    if df_f.empty or df_f["time_used"].isna().all():
        st.info("Belum bisa membuat heatmap — kolom waktu tidak lengkap.")
    else:
        tmp = df_f.dropna(subset=["time_used"]).copy()
        tmp["weekday"] = tmp["time_used"].dt.weekday.map(lambda x: calendar.day_name[x])# konsisten
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        tmp["weekday"] = pd.Categorical(tmp["weekday"], categories=order, ordered=True)
        tmp["hour"] = tmp["time_used"].dt.hour
        piv = tmp.pivot_table(index="weekday", columns="hour", values="comment", aggfunc="count").fillna(0)
        fig_hm = px.imshow(piv, aspect="auto", labels=dict(x="Jam", y="Hari", color="Jumlah ulasan"))
        st.plotly_chart(fig_hm, use_container_width=True)

# ==== KOMENTAR ====
with tab_comments:
    st.subheader("Komentar Terbaru")
    if df_f.empty:
        st.info("Tidak ada komentar untuk ditampilkan.")
    else:
        view = df_f.sort_values("time_used", ascending=False).head(15)[["time_used","username","platform","sentiment","comment","relative_time"]]
        for _, row in view.iterrows():
            dt = row["time_used"]
            dt_s = dt.strftime("%d %b %Y %H:%M") if pd.notna(dt) else "-"
            badge = badge_html(row["sentiment"])
            raw_hint = f"<span class='small'> • {row['relative_time']}</span>" if isinstance(row.get("relative_time", None), str) and row["relative_time"] else ""
            st.markdown(
                f"""
                <div class="card" style="margin-bottom:10px">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                    <div><b>{(row['username'] or 'Anonim')}</b> <span class="small">({row['platform']})</span></div>
                    <div style="display:flex;gap:10px;align-items:center">
                      {badge}
                      <span class="small">{dt_s}</span>{raw_hint}
                    </div>
                  </div>
                  <div style="line-height:1.55">{row['comment']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ==== SARAN ====
with tab_saran:
    st.subheader("Rekomendasi Otomatis untuk Samsat UPTB Palembang 1")
    for rec in auto_recommendations(df_f):
        st.markdown(f"- {rec}")

# ==== EKSPOR ====
with tab_export:
    st.subheader("Ekspor Data & Insight")
    if df_f.empty:
        st.info("Tidak ada data pada filter saat ini.")
    else:
        # Ekspor CSV (urut berdasar waktu analisis)
        csv = df_f.sort_values("time_used").to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Unduh CSV (Data Terfilter)", data=csv, file_name="komentar_terfilter.csv", mime="text/csv")
        # Ekspor Insight & Saran
        insights_txt = "\n".join(["INSIGHT OTOMATIS:"] + auto_insights(df_f) + ["", "REKOMENDASI:"] + auto_recommendations(df_f))
        st.download_button("⬇️ Unduh Insight & Saran (TXT)", data=insights_txt.encode("utf-8"),
                           file_name="insight_dan_saran.txt", mime="text/plain")

st.caption("💻 Created by **Peni Ilhami**, Computer Systems Student | Real-time Sentiment Analysis Dashboard for Samsat UPTB Palembang 1.")
