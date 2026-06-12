"""
app.py - Akıllı Dizüstü Bilgisayar Öneri Sistemi İçin Streamlit Arayüzü
Çalıştırma Komutu: streamlit run app.py

Bu dosya; makine öğrenmesi tabanlı dizüstü bilgisayar tavsiye sisteminin kullanıcı 
arayüzünü (UI) yönetir. Kullanıcıdan alınan tercihler doğrultusunda arka plandaki 
makine öğrenmesi boru hattını (ML Pipeline) tetikler, donanım kısıtlamalarını uygular 
ve elde edilen sonuçları modern görsel bileşenler ile kullanıcıya sunar.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. KÜTÜPHANELERİN İÇE AKTARILMASI (IMPORTS)
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st            # Arayüz bileşenlerini (butonlar, seçim kutuları, sekmeler vb.) oluşturur.
import pandas as pd               # Veri setini yüklemek, filtrelemek ve manipüle etmek için kullanılır.
import numpy as np                # Vektörel ve matematiksel işlemler gerçekleştirmek üzere dahil edilmiştir.
import plotly.express as px       # Grafik oluşturma işlemlerini kolaylaştıran yüksek seviyeli kütüphane.
import plotly.graph_objects as go # Radar ve saçılım grafikleri gibi daha özelleştirilmiş grafikleri çizmek için kullanılır.
import os                         # Dosya yolu kontrolü ve çevre değişkeni yönetimi için işletim sistemi arayüzü.

# OpenMP (Open Multi-Processing) iş parçacığı (thread) sınırını 2 olarak belirliyoruz.
# Bu ayar; K-Means gibi CPU'yu yoğun kullanan algoritmaların arka planda gereksiz yere
# tüm CPU çekirdeklerini kilitlemesini ve Streamlit arayüzünün donmasını engeller.
os.environ["OMP_NUM_THREADS"] = "2"

# Kendi yazdığımız makine öğrenmesi modülümüzden (src/ml_pipeline.py) gerekli bileşenleri alıyoruz:
# - run_pipeline: Tüm veri işleme ve ML kümeleme sürecini başlatan ana fonksiyon.
# - apply_hard_constraints: Kullanıcı tercihlerine göre katı filtreleme uygulayan fonksiyon.
# - recommend: Ağırlıklandırılmış yakınlık/benzerlik hesaplayıp öneri listesi sunan fonksiyon.
# - USAGE_PROFILES: Kullanım amaçlarına göre donanım ağırlıklarını tutan sözlük.
# - PORTABILITY_LIMITS: Taşınabilirlik sınıflarına göre maksimum ağırlık limitlerini içeren sözlük.
from src.ml_pipeline import (
    run_pipeline,
    apply_hard_constraints,
    recommend,
    USAGE_PROFILES,
    PORTABILITY_LIMITS,
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. SAYFA VE TEMA YAPILANDIRMASI (PAGE CONFIG & CUSTOM CSS)
# ─────────────────────────────────────────────────────────────────────────────

# Streamlit uygulamasının tarayıcı ayarlarını yapıyoruz.
# - page_title: Sekmede görünecek başlık metni.
# - page_icon: Sekme ikonu olarak kullanılacak emoji.
# - layout="wide": Sayfa içeriğinin tüm ekrana yayılmasını (geniş mod) sağlar.
# - initial_sidebar_state="expanded": Sol menünün (sidebar) başlangıçta açık gelmesini sağlar.
st.set_page_config(
    page_title="Laptop Öneri Sistemi",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# HTML ve CSS kullanarak Streamlit'in varsayılan arayüz stilini özelleştiriyoruz.
# "unsafe_allow_html=True" parametresi, Streamlit'in ham HTML ve CSS enjeksiyonuna izin vermesini sağlar.
st.markdown("""
<style>
/* 1. Global Sayfa Arka Planı: Koyu mavi ve mor tonlarında degrade (linear-gradient) geçiş */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}

/* 2. Sol Menü (Sidebar) Arka Planı: Sayfa ile uyumlu dikey degrade koyu tema */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}

/* 3. Başlıklar: Tüm H1, H2, H3 ve H4 başlıklarının yazı rengini beyaz yapar */
h1, h2, h3, h4 { color: #ffffff !important; }

/* 4. Genel Yazılar: Paragraf, liste, etiket ve metinlerin açık gri/mavi tonda olmasını sağlar */
p, li, label, span { color: #f0f0f5 !important; }

/* 5. Metrik Kartları Tasarımı (CPU, GPU vb. değerlerin kutucukları):
   - Arka plan hafif şeffaf beyaz (rgba),
   - Sınırlar belirginleştirilmiş (border),
   - Köşeler yuvarlatılmış (border-radius),
   - Arkadaki katmanı bulanıklaştıran cam efekti (backdrop-filter: blur),
   - Üzerine gelindiğinde tetiklenecek animasyon süresi (transition) belirlendi. */
.metric-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 22px 18px;
    backdrop-filter: blur(12px);
    transition: transform 0.2s, box-shadow 0.2s;
}

/* Fareyle metrik kartının üzerine gelindiğinde (hover):
   - translateY(-4px): Kartı hafifçe yukarı kaldırır, derinlik hissi verir.
   - box-shadow: Kartın etrafında morumsu bir parlama (gölge) efekti oluşturur. */
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(100, 100, 255, 0.25);
}

/* Metrik etiketlerinin (örn. CPU Skoru) yazı boyutu ve rengi */
.metric-label { font-size: 0.78rem; color: #d0d0f0 !important; margin-bottom: 2px; }

/* Metrik değerlerinin (örn. 5200) yazı boyutu, kalınlığı ve rengi */
.metric-value { font-size: 1.5rem; font-weight: 700; color: #a5b4fc !important; }

/* Önerilen Laptop Sonuç Kartları:
   - Koyu lacivert/mor degrade arka plan,
   - İnce morumsu çerçeve (rgba(124,131,255,0.3)),
   - Cam efekti ile premium bir görünüm sağlar. */
.laptop-card {
    background: linear-gradient(145deg, rgba(40,40,80,0.85), rgba(25,25,55,0.95));
    border: 1px solid rgba(124,131,255,0.3);
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 16px;
    backdrop-filter: blur(16px);
}

/* Laptop kartlarındaki başlık stili */
.laptop-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #ffffff !important;
    margin-bottom: 4px;
}

/* Laptop kartlarındaki alt başlık (Yıl ve Performans Sınıfı) stili */
.laptop-subtitle {
    font-size: 0.88rem;
    color: #d0d0ff !important;
    margin-bottom: 16px;
}

/* Uygunluk Skoru Rozeti (%94 gibi):
   - Mavi-mor degrade arka plan,
   - Oval köşeler (border-radius: 24px) ile modern buton görünümü. */
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

/* Yatay ayraç çizgisi tasarımı: Çok hafif şeffaf beyaz çizgi */
.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 14px 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. VERİ HATTI VE ÖNBELLEKLEME (DATA PIPELINE LOADING)
# ─────────────────────────────────────────────────────────────────────────────

def load_pipeline():
    """
    Veri kümesini kontrol eden, yoksa oluşturan ve 
    makine öğrenmesi boru hattını çalıştırarak verileri yükleyen yardımcı fonksiyon.
    """
    # Projeyi modüler kılmak amacıyla veri seti üretici fonksiyonu yerel olarak çağırıyoruz.
    from src.create_data import create_dataset
    import os
    
    csv = "data/laptop_dataset_2025_2026.csv"
    
    # Eğer belirtilen dosya yolunda CSV dosyası bulunamazsa:
    if not os.path.exists(csv):
        # Sentetik/gerçekçi laptop veri kümesini oluşturmak için ilgili fonksiyonu çalıştır.
        create_dataset(csv)
        
    # ml_pipeline.py içerisindeki run_pipeline fonksiyonunu çağırarak;
    # 1. Temizlenmiş ve ölçeklenmiş tam veri çerçevesini (df_full)
    # 2. Eğitilmiş K-Means model nesnesini (km_model)
    # 3. Veri normalizasyonunda kullanılan Scaler nesnesini (km_scaler) döndürür.
    return run_pipeline(csv)


# Fonksiyonu tetikleyip arayüzde kullanacağımız değişkenleri dolduruyoruz.
df_full, km_model, km_scaler = load_pipeline()


# ─────────────────────────────────────────────────────────────────────────────
# 4. SOL MENÜ (SIDEBAR) KONTROLLERİ VE FİLTRELER
# ─────────────────────────────────────────────────────────────────────────────

# "with st.sidebar:" bloğu, altındaki tüm bileşenlerin sol menüde yer almasını sağlar.
with st.sidebar:
    st.markdown("## 🎯 Tercihleriniz")
    st.markdown("---") # Ayırıcı çizgi

    # Kullanım Amacı Seçimi:
    # USAGE_PROFILES sözlüğünün anahtarlarını (Oyun, Ofis, Yazılım vb.) liste olarak alıp seçim kutusuna veriyoruz.
    usage_options = ["Seçiniz..."] + list(USAGE_PROFILES.keys())
    usage = st.selectbox(
        "💡 Kullanım Amacı",
        usage_options,
        help="Laptopunuzu hangi amaçla kullanacaksınız? Seçiminiz donanım önceliklerini (ağırlıklarını) belirler.",
    )
    
    # Performans Sınıfı Seçimi:
    # Cihazların donanım seviyelerini (Giriş, Orta, Üst) K-Means kümelerine göre filtrelemek için kullanılır.
    perf_options = ["Seçiniz...", "Fark Etmez", "Giriş Seviyesi", "Orta Segment", "Üst Segment"]
    perf_category = st.selectbox(
        "⚡ Performans Sınıfı",
        perf_options,
        help="K-Means kümeleme algoritması ile donanım özelliklerine göre belirlenen segmentler.",
    )

    # Ağırlık/Taşınabilirlik Seçimi:
    # Cihazın toplam ağırlık kısıtlamasını belirtir.
    portability_options = ["Seçiniz..."] + list(PORTABILITY_LIMITS.keys())
    portability = st.selectbox(
        "🎒 Ağırlık",
        portability_options,
        help="Toplam ağırlık (laptop ve adaptör ağırlığının toplamı).",
    )

    # Uygulamanın statik bir veri setiyle çalıştığını belirten yasal/teknik bilgilendirme yazısı.
    st.markdown(
        "<small style='color:#666'>Veriler statik veri kümesinden alınmıştır. "
        "Gerçek zamanlı veri kazıma kullanılmamaktadır.</small>",
        unsafe_allow_html=True,
    )
    


# ─────────────────────────────────────────────────────────────────────────────
# 5. ANA SAYFA BAŞLIKLARI VE GİRİŞ METİNLERİ
# ─────────────────────────────────────────────────────────────────────────────

# CSS degrade arka planlı ve ortalanmış ana başlık.
st.markdown(
    "<h1 style='text-align:center;background:linear-gradient(90deg,#667eea,#764ba2);"
    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
    "font-size:2.6rem;margin-bottom:0'>💻 Laptop Öneri Sistemi</h1>",
    unsafe_allow_html=True,
)
# Tamamlayıcı alt başlık.
st.markdown(
    "<p style='text-align:center;color:#8888bb!important;font-size:1.05rem;"
    "margin-top:0'>Makine Öğrenmesi Tabanlı Akıllı Laptop Tavsiye Motoru</p>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. SEKMELİ KULLANICI ARAYÜZÜ (TABS)
# ─────────────────────────────────────────────────────────────────────────────

# Uygulamayı daha düzenli hale getirmek için 3 adet sekme tanımlıyoruz.
tab_rec, tab_cluster, tab_data = st.tabs(
    ["🏆 Öneriler", "📊 Kümeleme Analizi", "📋 Veri Kümesi"]
)

# ──────────────── TAB 1: ÖNERİLER SEKME MANTIĞI ────────────────
with tab_rec:
    # Kontrol: Eğer kullanıcı henüz sol menüden tüm tercih kutularını doldurmadıysa uyarı gösteriyoruz.
    if "Seçiniz..." in [usage, perf_category, portability]:
        st.info("👈 Lütfen size en uygun laptop önerilerini görebilmek için sol menüden tüm tercihlerinizi belirleyin.")
    else:
        # AŞAMA 1: Kullanıcının belirlediği Performans ve Ağırlık kısıtlarına göre veri setini filtrele.
        filtered = apply_hard_constraints(df_full, perf_category, portability)
        
        # AŞAMA 2: Filtrelenmiş veri seti içinden kullanım amacına en uygun ilk 3 laptopu seç.
        results = recommend(filtered, usage, top_n=3)

        relaxed_msg = ""
        # EĞER kısıtlamalar çok katıysa ve 3'ten az sonuç döndüyse, ESNETME (Fallback) algoritmalarını çalıştırıyoruz:
        if len(results) < 3:
            
            # Fallback 1: Taşınabilirlik (Ağırlık) kısıtlamasını "Fark Etmez" yaparak kaldır, performansı koru.
            if portability != "Fark Etmez":
                relaxed = apply_hard_constraints(df_full, perf_category, "Fark Etmez")
                # Mevcut sonuçlarla yeni sonuçları birleştir, marka-model bazında mükerrer kayıtları sil ve ilk 3'ü al.
                results = pd.concat([results, recommend(relaxed, usage, top_n=3)]).drop_duplicates(subset=["Marka", "Model"]).head(3)
                relaxed_msg = "⚠️ Seçilen katı kriterlere uyan yeterli cihaz bulunamadığı için **Taşınabilirlik (Ağırlık)** kriteri esnetildi."

            # Fallback 2: Ağırlığı esnetmek yetmediyse, Performans Sınıfı kısıtlamasını da esneterek kaldır.
            if len(results) < 3 and perf_category != "Fark Etmez":
                relaxed = apply_hard_constraints(df_full, "Fark Etmez", "Fark Etmez")
                results = pd.concat([results, recommend(relaxed, usage, top_n=3)]).drop_duplicates(subset=["Marka", "Model"]).head(3)
                relaxed_msg = "⚠️ Ağırlıktan feragat etmek de yeterli olmadığı için **Taşınabilirlik ve Performans** kriterleri tamamen esnetildi."

            # Fallback 3: Eğer hala 3 cihaz bulunamadıysa (veri kümesi darlığı vb.), tüm kısıtlamaları devre dışı bırak ve tüm veri setini öneriye tabi tut.
            if len(results) < 3:
                results = pd.concat([results, recommend(df_full, usage, top_n=3)]).drop_duplicates(subset=["Marka", "Model"]).head(3)
                relaxed_msg = "⚠️ Seçilen katı kriterlere uyan cihaz bulunamadığı için **tüm kısıtlamalar** esnetildi."

        # Tüm adımlara rağmen sonuç bulunamadıysa uyarı bas.
        if results.empty:
            st.warning("⚠️ Ne yazık ki bu amaca uygun hiçbir bilgisayar bulunamadı.")
        else:
            # Herhangi bir filtre esnetildiyse arayüzde mavi kutuda mesaj yayınla.
            if relaxed_msg:
                st.info(relaxed_msg)

            # Kullanıcının yaptığı seçimlerin özet gösterimi.
            st.markdown("### 📋 Tercihleriniz")
            st.info(f"**Kullanım:** {usage} &nbsp;|&nbsp; **Performans Sınıfı:** {perf_category} &nbsp;|&nbsp; **Ağırlık:** {portability}")
            st.markdown("---")

            st.markdown(f"### 🎯 En Uygun {len(results)} Laptop")

            # Gelen sonuç adedine göre dinamik olarak yan yana sütunlar oluşturuyoruz.
            cols = st.columns(len(results))
            medals = ["🥇", "🥈", "🥉"] # Öneri sıralamasına göre madalyalar

            # Her bir önerilen laptopu ilgili sütuna yerleştiriyoruz.
            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i]:
                    # Özelleştirilmiş HTML kart tasarımı. Laptop marka, model, yıl ve uygunluk yüzdesini gösterir.
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

                    # Metrik Kartları:
                    # columns(2) yardımıyla yan yana ikişerli metrik yerleşimleri yapıyoruz.
                    m1, m2 = st.columns(2)
                    m1.metric("CPU Skoru", f"{row['CPU_Skoru']}")
                    m2.metric("GPU Skoru", f"{row['GPU_Skoru']}")
                    
                    m3, m4 = st.columns(2)
                    m3.metric("RAM", f"{row['RAM_GB']} GB")
                    m4.metric("SSD", f"{row['SSD_GB']} GB")
                    
                    m5, m6 = st.columns(2)
                    m5.metric("Pil", f"{row['Pil_Video_Saat']} saat")
                    m6.metric("Ağırlık", f"{row['Total_Weight']:.1f} kg")

                    # Detaylı Özellikler (st.expander):
                    # Kullanıcı tıkladığında açılan, varsayılan olarak kapalı (expanded=False) olan detay kutusu.
                    with st.expander("🛠 Detaylı Özellikler", expanded=False):
                        st.markdown(f"**Ekran:** {row['Ekran_Boyutu']}\" | {row.get('Ekran_Cozunurluk', '-')}")
                        st.markdown(f"**Grafik Birimi:** {row['Ekran_Karti']} ({row['VRAM_GB']} GB VRAM)")
                        st.markdown(f"**RAM Hızı:** {row['RAM_MHz']} MHz")
                        st.markdown(f"**SSD Hızı:** {row['SSD_Hiz_MBs']} MB/s")
                        st.markdown(f"**Bağlantılar:** {row.get('Port_Detay', '-')}")

            # ───── RADAR GRAFİĞİ İLE KARŞILAŞTIRMA ANALİZİ ─────
            if len(results) > 0:
                st.markdown("### 📈 Karşılaştırma Radar Grafiği")
                # Radar grafiğinde yer alacak normalize edilmiş boyutlar.
                categories = ["CPU", "GPU", "RAM", "SSD Hız", "Soğutma", "Pil", "Ağırlık"]
                fig_radar = go.Figure()
                colors = ["#4facfe", "#00f2fe", "#f093fb"] # Farklı laptoplar için çizgi renkleri.
                
                # Her bir cihaz için verileri okuyup grafik üzerine çizgi (Scatterpolar) olarak ekliyoruz.
                for i, (_, row) in enumerate(results.iterrows()):
                    # ml_pipeline tarafından 0 ile 1 arasına ölçeklenmiş özellikleri alıyoruz.
                    vals = [
                        row["CPU_norm"], row["GPU_norm"], row["RAM_norm"],
                        row["SSD_Hiz_norm"], row["Sogutma_norm"], row["Pil_norm"],
                        row["Portability"],
                    ]
                    # Polar (radar) grafiklerde şeklin kapanması için ilk elemanı listenin sonuna tekrar ekliyoruz.
                    vals.append(vals[0])
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals,                                        # Yarıçap (özelliklerin değerleri)
                        theta=categories + [categories[0]],            # Eksen isimleri
                        fill="toself",                                 # Alanın içini doldur
                        name=f"{row['Marka']} {row['Model']}",          # Lejanttaki adı
                        line=dict(color=colors[i % len(colors)]),      # Çizgi rengi
                        opacity=0.7,                                   # Saydamlık derecesi
                    ))

                # Radar grafiğinin görsel şablon, arkaplan, yazı tipi ve boşluk ayarları.
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)", # İç arkaplan şeffaf
                        radialaxis=dict(visible=True, range=[0, 1], showticklabels=False), # Eksen çizgileri aralığı 0-1, etiketler kapalı
                    ),
                    paper_bgcolor="rgba(0,0,0,0)", # Dış çerçeve arkaplanı şeffaf
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#c8c8e0"),    # Açık mavi/gri yazı rengi
                    legend=dict(font=dict(color="#c8c8e0")),
                    margin=dict(t=40, b=40),       # Üst ve alt kenar boşlukları
                    height=420,                    # Grafik yüksekliği (piksel)
                )
                # Grafiği Streamlit arayüzüne genişlik otomatik ayarlanacak şekilde (use_container_width) yerleştir.
                st.plotly_chart(fig_radar, use_container_width=True)

            st.markdown("---")
            st.markdown("#### 🛍️ Satın Alma Kanalları")
            
            # HTML ve CSS kullanarak tasarlanmış satın alma kanalları yönlendirme butonları gridi.
            # Faviconları otomatik çekebilmek için Google Favicon API (s2/favicons) servisi kullanılmıştır.
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

# ──────────────── TAB 2: KÜMELEME ANALİZİ SEKME MANTIĞI ────────────────
with tab_cluster:
    st.markdown("### 📊 İşlemci (CPU) Çoklu Çekirdek Performansı vs Güç Tüketimi")
    st.markdown(
        "Aşağıdaki grafik, cihazların güç tüketimi (Watt) ile "
        "CPU (Multithread) performansları arasındaki ilişkiyi göstermektedir."
    )

    fig_cpu = go.Figure()

    # Referans grafik şablonuna uygun olarak hazırlanmış statik işlemci veri seti listesi.
    # AMD, Intel ve Apple M işlemcilerin Güç Tüketimi (TDP / Watt - X ekseni) ve Cinebench Çoklu Çekirdek Skoru (Y ekseni) değerleri.
    # Lines+markers modundakiler farklı TDP limitlerinde test edilmiş serileri gösterir (Çizgi çekilir).
    # Sadece markers modundakiler (Apple M) tek bir güç seviyesinde ölçülen statik noktalardır.
    traces = [
        {"name": "Ryzen AI MAX+ 395", "x": [40, 49, 63, 83], "y": [5200, 5800, 6350, 6200], "color": "#FF1493", "mode": "lines+markers"},
        {"name": "Ryzen AI 9 HX 470", "x": [45, 67, 82], "y": [4100, 4400, 4600], "color": "#FF8C00", "mode": "lines+markers"},
        {"name": "Ultra X9 388H", "x": [22, 33, 46], "y": [3350, 3950, 4550], "color": "#00BFFF", "mode": "lines+markers"},
        {"name": "Ultra 7 356H", "x": [44, 59, 60], "y": [4200, 4700, 4680], "color": "#B0C4DE", "mode": "lines+markers"},
        
        # Apple M serisi (tekil ölçüm noktaları)
        {"name": "M4 Pro 14C", "x": [35], "y": [5800], "color": "#A9A9A9", "mode": "markers"},
        {"name": "M5 10C", "x": [26], "y": [4200], "color": "#000000", "mode": "markers"},
        {"name": "M5 Pro 15C", "x": [39], "y": [6600], "color": "#32CD32", "mode": "markers"},
        {"name": "M5 Max 18C", "x": [44], "y": [7600], "color": "#9ACD32", "mode": "markers"},
        {"name": "M5 Pro 18C", "x": [43], "y": [7400], "color": "#7FFF00", "mode": "markers"}
    ]

    # Her bir işlemci kaydını döngüyle Plotly grafiğine ekliyoruz.
    for t in traces:
        fig_cpu.add_trace(go.Scatter(
            x=t["x"],                                                   # X ekseni (Watt)
            y=t["y"],                                                   # Y ekseni (Çoklu Çekirdek Skoru)
            mode=t["mode"],                                             # Çizim modu (çizgi+nokta veya sadece nokta)
            name=t["name"],                                             # İşlemci adı
            marker=dict(size=9, color=t["color"], line=dict(width=1, color="#fff")), # Nokta boyutu ve sınır rengi
            line=dict(width=3, color=t["color"]) if "lines" in t["mode"] else None,  # Çizgi kalınlığı ve rengi (varsa)
            # Fareyle üzerine gelindiğinde görünecek bilgi kutusu (hover) tasarımı.
            hovertemplate="<b>%{data.name}</b><br>Güç Tüketimi: %{x} W<br>Çoklu Çekirdek Skoru: %{y}<extra></extra>"
        ))

    # Performans Analizi Grafiğinin yerleşim, başlık ve eksen ayarları.
    fig_cpu.update_layout(
        title="Cinebench 2026 Performance by Power Draw",
        xaxis_title="AVERAGE WATTAGE (LOWER IS BETTER)", # X Eksen Başlığı: Düşük olması daha iyidir.
        yaxis_title="MULTITHREAD SCORE (HIGHER IS BETTER)", # Y Eksen Başlığı: Yüksek olması daha iyidir.
        paper_bgcolor="rgba(0,0,0,0)",                       # Dış panel arkaplanı transparan
        plot_bgcolor="rgba(15,12,41,0.6)",                   # Çizim alanı koyu lacivert tonunda yarı şeffaf
        font=dict(color="#c8c8e0"),                          # Yazı tipi rengi
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False), # İnce x-ızgara çizgileri
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False), # İnce y-ızgara çizgileri
        height=650,                                          # Grafik yüksekliği (piksel)
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.02, title="") # Sağ taraftaki lejant yerleşimi
    )
    st.plotly_chart(fig_cpu, use_container_width=True)




# ──────────────── TAB 3: VERİ KÜMESİ SEKME MANTIĞI ────────────────
with tab_data:
    st.markdown("### 📋 Tüm Veri Kümesi")
    
    # Kullanıcıya tüm veri setini ham halde gösterirken karmaşıklığı önlemek adına sadece ana sütunları seçiyoruz.
    display_cols = [
        "Marka", "Model", "Yil", "Performans_Sinifi", "CPU_Skoru", "GPU_Skoru",
        "RAM_GB", "SSD_GB", "Pil_Video_Saat", "Total_Weight"
    ]
    
    # st.dataframe: Verileri interaktif ve sıralanabilir bir tablo olarak ekrana basar.
    # df_full[display_cols].style.format: Toplam ağırlık sütununu virgülden sonra tek hane göstermek üzere biçimlendirir (örn. "1.8 kg").
    st.dataframe(
        df_full[display_cols].style.format({"Total_Weight": "{:.1f} kg"}),
        use_container_width=True, # Tablonun ekranı kaplamasını sağlar
        height=520,                # Tablonun dikey boyut limiti
    )

# ─────────────────────────────────────────────────────────────────────────────
# 7. SAYFA ALTI BİLGİSİ (FOOTER)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#555!important;font-size:0.82rem'>"
    "ML Laptop Öneri Sistemi &bull; Scikit-Learn &amp; Streamlit &bull; 2025-2026</p>",
    unsafe_allow_html=True,
)
