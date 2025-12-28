import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime, date

# ======================================================
# 1. KONFIGURASI HALAMAN & STYLE (BIRU-KUNING)
# ======================================================
st.set_page_config(
    page_title="MeteoForecaster Palembang — LSTM",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .header-style {
        text-align: center; color: #0B3C5D; font-size: 38px;
        font-weight: bold; font-family: 'Times New Roman', serif;
    }
    .subheader-style {
        text-align: center; color: #555; font-size: 18px; margin-bottom: 30px;
    }
    .result-card-blue {
        background-color: #E3F2FD; padding: 20px; border-radius: 10px;
        border-left: 10px solid #0B3C5D; margin-bottom: 15px;
        color: #0B3C5D; font-size: 20px; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ======================================================
# 2. FUNGSI LOAD DATA (MENGGUNAKAN 6 FILE LAMA)
# ======================================================
@st.cache_data
def load_all_files():
    try:
        # Membaca data historis (1981-2024) dan peramalan (20 tahun)
        hist = pd.read_csv("data processed_data monthly.csv", index_col=0, parse_dates=True)
        fore = pd.read_csv("data forecast_peramalan 20 tahun semua parameter.csv", index_col=0, parse_dates=True)
        metr = pd.read_csv("evaluation model_metrics.csv", index_col=0)
        meta = pd.read_csv("metadata_model metadata.csv", header=None, index_col=0)
        act_t = pd.read_csv("data dashboard_data aktual test.csv", index_col=0, parse_dates=True)
        pre_t = pd.read_csv("data dashboard_data prediksi test.csv", index_col=0, parse_dates=True)
        return hist, fore, metr, meta, act_t, pre_t
    except Exception as e:
        return None, None, None, None, None, None

df, future_df, metrics_df, metadata_df, actual_test, pred_test = load_all_files()

if df is None:
    st.error("⚠️ File CSV tidak ditemukan. Pastikan 6 file data lama Anda ada di folder yang sama.")
    st.stop()

# ======================================================
# 3. HEADER UTAMA (TAHUN 2025)
# ======================================================
st.markdown('<div class="header-style">DASHBOARD PERAMALAN IKLIM KOTA PALEMBANG</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader-style">Analisis Temporal Jangka Panjang Berbasis Long Short-Term Memory (LSTM) - 2025</div>', unsafe_allow_html=True)
st.markdown("---")

# ======================================================
# 4. SIDEBAR (8 PARAMETER TERPILIH)
# ======================================================
label_map = {
    "TN": "Temperatur Minimum (TN)", "TX": "Temperatur Maksimum (TX)",
    "RH_AVG": "Kelembapan Relatif Rata-rata (RH_AVG)", "RR": "Curah Hujan (RR)",
    "SS": "Lama Penyinaran Matahari (SS)", "FF_X": "Kecepatan Angin Maksimum (FF_X)",
    "FF_AVG": "Kecepatan Angin Rata-rata (FF_AVG)", "DDD_X_sin": "Komponen Arah Angin Maksimum (DDD_X_sin)"
}

st.sidebar.title("📌 Menu Navigasi")
menu = st.sidebar.radio("Pilih Tampilan:", ["Halaman Dashboard", "Uji Validitas (Residual)", "Profil Peneliti"])

st.sidebar.markdown("---")
var_name = st.sidebar.selectbox("Pilih Parameter Iklim:", list(label_map.keys()), format_func=lambda x: label_map[x])

# Filter grafik menggunakan data historis dan peramalan
st.sidebar.subheader("📅 Filter Grafik")
start_d = st.sidebar.date_input("Mulai", df.index.min().date())
end_d = st.sidebar.date_input("Selesai", future_df.index.max().date())

# ======================================================
# 5. HALAMAN 1: DASHBOARD UTAMA
# ======================================================
if menu == "Halaman Dashboard":
    st.subheader(f"🔍 Metrik Evaluasi: {label_map[var_name]}")
    c1, c2, c3 = st.columns(3)
    if var_name in metrics_df.index:
        c1.metric("RMSE", f"{metrics_df.loc[var_name, 'RMSE']:.4f}")
        c2.metric("MAE", f"{metrics_df.loc[var_name, 'MAE']:.4f}")
        c3.metric("R-Squared (R2)", f"{metrics_df.loc[var_name, 'R2']:.4f}")
    
    st.markdown("---")
    st.subheader("📅 Pencarian Prediksi Spesifik (Harian/Bulanan)")
    
    # Fitur pencarian harian yang disesuaikan ke data bulanan
    search_date = st.date_input("Pilih Tanggal Prediksi:", value=date(2025, 12, 1),
                                min_value=date(2025, 1, 1), max_value=date(2044, 12, 31))
    
    search_month_start = search_date.replace(day=1) # Logika cerdas untuk data bulanan
    
    try:
        val_daily = future_df.loc[search_month_start.strftime('%Y-%m-%d'), var_name]
        st.markdown(f"""<div class="result-card-blue">Hasil Prediksi {label_map[var_name]}<br>
                    Periode: {search_date.strftime("%d %B %Y")}<br>Nilai: {val_daily:.2f}</div>""", unsafe_allow_html=True)
    except:
        st.warning("Data untuk periode tersebut tidak ditemukan.")

    st.markdown("---")
    st.subheader("📈 Grafik Tren Historis dan Peramalan")
    combined = pd.concat([df[[var_name]].assign(S="Historis"), future_df[[var_name]].assign(S="Prediksi")])
    mask = (combined.index.date >= start_d) & (combined.index.date <= end_d)
    
    fig = px.line(combined.loc[mask], x=combined.loc[mask].index, y=var_name, color="S",
                  color_discrete_map={"Historis": "#0B3C5D", "Prediksi": "#F2C94C"}, template="plotly_white")
    
    fig.update_traces(hovertemplate="<b>Tanggal:</b> %{x|%d %B %Y}<br><b>Nilai:</b> %{y}")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    csv_bytes = future_df[[var_name]].to_csv().encode('utf-8')
    st.download_button("📥 Download Data Prediksi (CSV)", csv_bytes, 
                       file_name=f"Hasil_Prediksi_{var_name}.csv", mime="text/csv")

# ======================================================
# 6. HALAMAN 2: ANALISIS RESIDUAL
# ======================================================
elif menu == "Uji Validitas (Residual)":
    st.header(f"🎯 Analisis Residual: {label_map[var_name]}")
    residual = actual_test[var_name] - pred_test[var_name]
    col_a, col_b = st.columns(2)
    with col_a:
        fig_res = px.scatter(x=pred_test[var_name], y=residual, labels={'x': 'Prediksi', 'y': 'Error'},
                             title="Residual Scatter Plot", color_discrete_sequence=['#0B3C5D'])
        fig_res.add_hline(y=0, line_dash="dash", line_color="#F2C94C")
        st.plotly_chart(fig_res, use_container_width=True)
    with col_b:
        fig_hist = px.histogram(residual, nbins=20, title="Distribusi Error", color_discrete_sequence=['#F2C94C'])
        st.plotly_chart(fig_hist, use_container_width=True)

# ======================================================
# 7. HALAMAN 3: PROFIL PENELITI (SESUAI URUTAN REVISI)
# ======================================================
else:
    st.header("👤 Profil Peneliti & Akademik")
    st.info(f"**Identitas Peneliti:**\n* Nama Peneliti: Amanda Rahmannisa\n* NIM: 06111282227058")
    st.warning(f"**Dosen Pembimbing:**\n* Dr. Melly Ariska, S.Pd., M.Sc.")
    st.success(f"**Informasi Akademik:**\n* Program Studi: Pendidikan Fisika\n* Fakultas: Keguruan dan Ilmu Pendidikan\n* Universitas: Universitas Sriwijaya\n* Tahun: 2025")
    st.divider()
    st.subheader("🛠️ Metadata Konfigurasi Model")
    st.table(metadata_df)