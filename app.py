"""
app.py - Streamlit UI for the ML Laptop Recommendation System (Orijinal Versiyon)
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

os.environ["OMP_NUM_THREADS"] = "2"

from src.ml_pipeline import (
    run_pipeline,
    apply_hard_constraints,
    recommend,
    USAGE_PROFILES,
    PORTABILITY_LIMITS,
)

# ───── Page config ─────
st.set_page_config(
    page_title="Laptop Öneri Sistemi",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───── Custom CSS ─────
st.markdown("""
<style>
/* Global */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
h1, h2, h3, h4 { color: #ffffff !important; }
p, li, label, span { color: #f0f0f5 !important; }

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 22px 18px;
    backdrop-filter: blur(12px);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(100, 100, 255, 0.25);
}
.metric-label { font-size: 0.78rem; color: #d0d0f0 !important; margin-bottom: 2px; }
.metric-value { font-size: 1.5rem; font-weight: 700; color: #a5b4fc !important; }

/* Laptop result card */
.laptop-card {
    background: linear-gradient(145deg, rgba(40,40,80,0.85), rgba(25,25,55,0.95));
    border: 1px solid rgba(124,131,255,0.3);
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 16px;
    backdrop-filter: blur(16px);
}
.laptop-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #ffffff !important;
    margin-bottom: 4px;
}
.laptop-subtitle {
    font-size: 0.88rem;
    color: #d0d0ff !important;
    margin-bottom: 16px;
}
.score-badge {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #fff !important;
    padding: 6px 18px;
    border-radius: 24px;
    font-weight: 700;
    font-size: 1.05rem;
    margin-bottom: 14px;
}
.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 14px 0;
}
</style>
""", unsafe_allow_html=True)


# ───── Load & cache pipeline ─────
def load_pipeline():
    from src.create_data import create_dataset
    import os
    csv = "data/laptop_dataset_2025_2026.csv"
    if not os.path.exists(csv):
        create_dataset(csv)
    return run_pipeline(csv)


df_full, km_model, km_scaler = load_pipeline()


# ───── Sidebar ─────
with st.sidebar:
    st.markdown("## 🎯 Tercihleriniz")
    st.markdown("---")

    usage_options = ["Seçiniz..."] + list(USAGE_PROFILES.keys())
    usage = st.selectbox(
        "💡 Kullanım Amacı",
        usage_options,
        help="Laptopunuzu hangi amaçla kullanacaksınız?",
    )
    
    perf_options = ["Seçiniz...", "Fark Etmez", "Giriş Seviyesi", "Orta Segment", "Üst Segment"]
    perf_category = st.selectbox(
        "⚡ Performans Sınıfı",
        perf_options,
        help="K-Means ile donanım özelliklerine göre belirlenen segmentler",
    )

    portability_options = ["Seçiniz..."] + list(PORTABILITY_LIMITS.keys())
    portability = st.selectbox(
        "🎒 Ağırlık",
        portability_options,
        help="Toplam ağırlık (laptop + adaptör)",
    )


    st.markdown(
        "<small style='color:#666'>Veriler statik veri kümesinden alınmıştır. "
        "Gerçek zamanlı veri kazıma kullanılmamaktadır.</small>",
        unsafe_allow_html=True,
    )
    


# ───── Header ─────
st.markdown(
    "<h1 style='text-align:center;background:linear-gradient(90deg,#667eea,#764ba2);"
    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
    "font-size:2.6rem;margin-bottom:0'>💻 Laptop Öneri Sistemi</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:#8888bb!important;font-size:1.05rem;"
    "margin-top:0'>Makine Öğrenmesi Tabanlı Akıllı Laptop Tavsiye Motoru</p>",
    unsafe_allow_html=True,
)

# ───── Tabs ─────
tab_rec, tab_cluster, tab_data = st.tabs(
    ["🏆 Öneriler", "📊 Kümeleme Analizi", "📋 Veri Kümesi"]
)

# ──────────────── TAB 1: Recommendations ────────────────
with tab_rec:
    if "Seçiniz..." in [usage, perf_category, portability]:
        st.info("👈 Lütfen size en uygun laptop önerilerini görebilmek için sol menüden tüm tercihlerinizi belirleyin.")
    else:
        filtered = apply_hard_constraints(df_full, perf_category, portability)

        filtered = apply_hard_constraints(df_full, perf_category, portability)
        results = recommend(filtered, usage, top_n=3)

        relaxed_msg = ""
        if len(results) < 3:
            # Fallback 1: Öncelikle Taşınabilirlikten (Ağırlıktan) feragat et, Performansı koru!
            if portability != "Fark Etmez":
                relaxed = apply_hard_constraints(df_full, perf_category, "Fark Etmez")
                results = pd.concat([results, recommend(relaxed, usage, top_n=3)]).drop_duplicates(subset=["Marka", "Model"]).head(3)
                relaxed_msg = "⚠️ Seçilen katı kriterlere uyan yeterli cihaz bulunamadığı için **Taşınabilirlik (Ağırlık)** kriteri esnetildi."

            # Fallback 2: Ağırlığı esnetmek yetmediyse, Performans Sınıfını da esnet!
            if len(results) < 3 and perf_category != "Fark Etmez":
                relaxed = apply_hard_constraints(df_full, "Fark Etmez", "Fark Etmez")
                results = pd.concat([results, recommend(relaxed, usage, top_n=3)]).drop_duplicates(subset=["Marka", "Model"]).head(3)
                relaxed_msg = "⚠️ Ağırlıktan feragat etmek de yeterli olmadığı için **Taşınabilirlik ve Performans** kriterleri tamamen esnetildi."

            # Fallback 3: Hâlâ bulunamadıysa (ki çok zor), son çare tüm kuralları by-pass et.
            if len(results) < 3:
                results = pd.concat([results, recommend(df_full, usage, top_n=3)]).drop_duplicates(subset=["Marka", "Model"]).head(3)
                relaxed_msg = "⚠️ Seçilen katı kriterlere uyan cihaz bulunamadığı için **tüm kısıtlamalar** esnetildi."

        if results.empty:
            st.warning("⚠️ Ne yazık ki bu amaca uygun hiçbir bilgisayar bulunamadı.")
        else:
            if relaxed_msg:
                st.info(relaxed_msg)

            st.markdown("### 📋 Tercihleriniz")
            st.info(f"**Kullanım:** {usage} &nbsp;|&nbsp; **Performans Sınıfı:** {perf_category} &nbsp;|&nbsp; **Ağırlık:** {portability}")
            st.markdown("---")

            st.markdown(f"### 🎯 En Uygun {len(results)} Laptop")

            cols = st.columns(len(results))
            medals = ["🥇", "🥈", "🥉"]

            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i]:
                    st.markdown(f"""
                    <div class="laptop-card">
                        <div style="text-align:center;font-size:2.2rem">{medals[i]}</div>
                        <div class="laptop-title" style="text-align:center">
                            {row['Marka']} {row['Model']}
                        </div>
                        <div class="laptop-subtitle" style="text-align:center">
                            {row['Yil']} &bull; {row.get('Performans_Sinifi','')}
                        </div>
                        <div style="text-align:center">
                            <span class="score-badge">Uygunluk: %{row['Uygunluk_Skoru']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Sub-metrics in columns
                    m1, m2 = st.columns(2)
                    m1.metric("CPU Skoru", f"{row['CPU_Skoru']}")
                    m2.metric("GPU Skoru", f"{row['GPU_Skoru']}")
                    m3, m4 = st.columns(2)
                    m3.metric("RAM", f"{row['RAM_GB']} GB")
                    m4.metric("SSD", f"{row['SSD_GB']} GB")
                    m5, m6 = st.columns(2)
                    m5.metric("Pil", f"{row['Pil_Video_Saat']} saat")
                    m6.metric("Ağırlık", f"{row['Total_Weight']:.1f} kg")

                    # Detailed features expander
                    with st.expander("🛠 Detaylı Özellikler", expanded=False):
                        st.markdown(f"**Ekran:** {row['Ekran_Boyutu']}\" | {row.get('Ekran_Cozunurluk', '-')}")
                        st.markdown(f"**Grafik Birimi:** {row['Ekran_Karti']} ({row['VRAM_GB']} GB VRAM)")
                        st.markdown(f"**RAM Hızı:** {row['RAM_MHz']} MHz")
                        st.markdown(f"**SSD Hızı:** {row['SSD_Hiz_MBs']} MB/s")
                        st.markdown(f"**Bağlantılar:** {row.get('Port_Detay', '-')}")

            # Radar chart comparison
            if len(results) > 0:
                st.markdown("### 📈 Karşılaştırma Radar Grafiği")
                categories = ["CPU", "GPU", "RAM", "SSD Hız", "Soğutma", "Pil", "Ağırlık"]
                fig_radar = go.Figure()
                colors = ["#4facfe", "#00f2fe", "#f093fb"]
                for i, (_, row) in enumerate(results.iterrows()):
                    vals = [
                        row["CPU_norm"], row["GPU_norm"], row["RAM_norm"],
                        row["SSD_Hiz_norm"], row["Sogutma_norm"], row["Pil_norm"],
                        row["Portability"],
                    ]
                    vals.append(vals[0])  # close the polygon
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals,
                        theta=categories + [categories[0]],
                        fill="toself",
                        name=f"{row['Marka']} {row['Model']}",
                        line=dict(color=colors[i % len(colors)]),
                        opacity=0.7,
                    ))

                fig_radar.update_layout(
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#c8c8e0"),
                    legend=dict(font=dict(color="#c8c8e0")),
                    margin=dict(t=40, b=40),
                    height=420,
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            st.markdown("---")
            st.markdown("#### 🛍️ Satın Alma Kanalları")
            
            st.markdown(
                """
                <style>
                .store-box {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 12px;
                    border-radius: 8px;
                    color: white !important;
                    font-weight: bold;
                    text-decoration: none;
                    font-size: 1rem;
                    transition: transform 0.2s, opacity 0.2s;
                }
                .store-box img {
                    width: 20px;
                    height: 20px;
                    margin-right: 8px;
                    border-radius: 4px;
                    background-color: white;
                    padding: 2px;
                }
                .store-box:hover {
                    transform: scale(1.02);
                    opacity: 0.9;
                }
                .vatan { background-color: #0033a0; }
                .media { background-color: #df0000; }
                .itopya { background-color: #f37021; }
                .ince { background-color: #00a1ff; }
                .teknosa { background-color: #ff6600; }
                
                .store-grid {
                    display: grid;
                    grid-template-columns: repeat(5, 1fr);
                    gap: 10px;
                    margin-top: 15px;
                    margin-bottom: 20px;
                }
                </style>
                
                <div class="store-grid">
                    <a href="https://www.vatanbilgisayar.com/notebook/" target="_blank" class="store-box vatan">
                        <img src="https://www.google.com/s2/favicons?domain=vatanbilgisayar.com&sz=64"> Vatan
                    </a>
                    <a href="https://www.mediamarkt.com.tr/tr/category/_diz%C3%BCst%C3%BC-bilgisayarlar-504926.html" target="_blank" class="store-box media">
                        <img src="https://www.google.com/s2/favicons?domain=mediamarkt.com.tr&sz=64"> MediaMarkt
                    </a>
                    <a href="https://www.itopya.com/laptop-notebook_k13" target="_blank" class="store-box itopya">
                        <img src="https://www.google.com/s2/favicons?domain=itopya.com&sz=64"> İtopya
                    </a>
                    <a href="https://www.incehesap.com/notebook-fiyatlari/" target="_blank" class="store-box ince">
                        <img src="https://www.google.com/s2/favicons?domain=incehesap.com&sz=64"> İncehesap
                    </a>
                    <a href="https://www.teknosa.com/laptop-notebook-c-116004" target="_blank" class="store-box teknosa">
                        <img src="https://www.google.com/s2/favicons?domain=teknosa.com&sz=64"> Teknosa
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )

# ──────────────── TAB 2: K-Means Clustering ────────────────
with tab_cluster:
    st.markdown("### 📊 İşlemci (CPU) Çoklu Çekirdek Performansı vs Güç Tüketimi")
    st.markdown(
        "Aşağıdaki grafik, cihazların güç tüketimi (Watt) ile "
        "CPU (Multithread) performansları arasındaki ilişkiyi göstermektedir."
    )

    fig_cpu = go.Figure()

    # Referans görsele BİREBİR uyan statik veri seti (Sadece CPU'lar: AMD, Intel, Apple M)
    # Çizgili olanlar (Lines + Markers)
    traces = [
        {"name": "Ryzen AI MAX+ 395", "x": [40, 49, 63, 83], "y": [5200, 5800, 6350, 6200], "color": "#FF1493", "mode": "lines+markers"},
        {"name": "Ryzen AI 9 HX 470", "x": [45, 67, 82], "y": [4100, 4400, 4600], "color": "#FF8C00", "mode": "lines+markers"},
        {"name": "Ultra X9 388H", "x": [22, 33, 46], "y": [3350, 3950, 4550], "color": "#00BFFF", "mode": "lines+markers"},
        {"name": "Ultra 7 356H", "x": [44, 59, 60], "y": [4200, 4700, 4680], "color": "#B0C4DE", "mode": "lines+markers"},
        
        # Sadece noktalı olanlar (Apple M Serisi)
        {"name": "M4 Pro 14C", "x": [35], "y": [5800], "color": "#A9A9A9", "mode": "markers"},
        {"name": "M5 10C", "x": [26], "y": [4200], "color": "#000000", "mode": "markers"},
        {"name": "M5 Pro 15C", "x": [39], "y": [6600], "color": "#32CD32", "mode": "markers"},
        {"name": "M5 Max 18C", "x": [44], "y": [7600], "color": "#9ACD32", "mode": "markers"},
        {"name": "M5 Pro 18C", "x": [43], "y": [7400], "color": "#7FFF00", "mode": "markers"}
    ]

    for t in traces:
        fig_cpu.add_trace(go.Scatter(
            x=t["x"],
            y=t["y"],
            mode=t["mode"],
            name=t["name"],
            marker=dict(size=9, color=t["color"], line=dict(width=1, color="#fff")),
            line=dict(width=3, color=t["color"]) if "lines" in t["mode"] else None,
            hovertemplate="<b>%{data.name}</b><br>Güç Tüketimi: %{x} W<br>Çoklu Çekirdek Skoru: %{y}<extra></extra>"
        ))

    fig_cpu.update_layout(
        title="Cinebench 2026 Performance by Power Draw",
        xaxis_title="AVERAGE WATTAGE (LOWER IS BETTER)",
        yaxis_title="MULTITHREAD SCORE (HIGHER IS BETTER)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,12,41,0.6)",
        font=dict(color="#c8c8e0"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        height=650,
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.02, title="")
    )
    st.plotly_chart(fig_cpu, use_container_width=True)




# ──────────────── TAB 3: Raw Dataset ────────────────
with tab_data:
    st.markdown("### 📋 Tüm Veri Kümesi")
    display_cols = [
        "Marka", "Model", "Yil", "Performans_Sinifi", "CPU_Skoru", "GPU_Skoru",
        "RAM_GB", "SSD_GB", "Pil_Video_Saat", "Total_Weight"
    ]
    st.dataframe(
        df_full[display_cols].style.format({"Total_Weight": "{:.1f} kg"}),
        use_container_width=True,
        height=520,
    )

# ───── Footer ─────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#555!important;font-size:0.82rem'>"
    "ML Laptop Öneri Sistemi &bull; Scikit-Learn &amp; Streamlit &bull; 2025</p>",
    unsafe_allow_html=True,
)
