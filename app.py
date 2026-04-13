import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# --- 🎨 ÖZEL ARAYÜZ TASARIMI (CSS) ---
def local_css():
    st.markdown("""
        <style>
        .main { background-color: #0e1117; color: #ffffff; }
        .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #ff4b4b; color: white; border: none; font-weight: bold; }
        .stButton>button:hover { background-color: #ff2b2b; border: none; }
        .laptop-card { background-color: #1e2130; padding: 20px; border-radius: 15px; border-left: 5px solid #ff4b4b; margin-bottom: 20px; box-shadow: 2px 2px 10px rgba(0,0,0,0.3); }
        .metric-text { font-size: 24px; font-weight: bold; color: #ff4b4b; }
        .device-title { font-size: 20px; font-weight: bold; margin-bottom: 5px; }
        </style>
    """, unsafe_allow_html=True)

# --- 🛠️ VERİ VE ANALİZ MOTORU ---
@st.cache_data
def load_data():
    # Dosya yolu hatasını önlemek için kontrol
    try:
        df = pd.read_csv('veri_seti.csv')
        df['Mobilite_Puani'] = 10 - df['Agirlik_Kg']
        return df
    except FileNotFoundError:
        return pd.DataFrame()

def main():
    local_css()
    
    st.title("🤖 Akıllı Bilgisayar Öneren Web Sistemi")
    st.markdown("**Bursa Uludağ Üniversitesi Bilgisayar Mühendisliği Bölümü**")
    st.markdown("---")

    df = load_data()
    if df.empty:
        st.error("❌ 'veri_seti.csv' dosyası bulunamadı. Lütfen dosyanın app.py ile aynı klasörde olduğundan emin olun.")
        return

    # --- Sidebar ---
    with st.sidebar:
        st.header("🔍 Tercihlerinizi Belirleyin")
        
        amac = st.selectbox("🎯 Kullanım Amacı:", 
            ["Yazılım Geliştirme", "Oyun", "Grafik Tasarım", "Ofis / İş", "Öğrenci / Günlük"])
        
        os_pref = st.radio("💻 İşletim Sistemi:", ["Windows", "macOS"])
        
        budget_range = st.slider("💰 Bütçe Aralığı (TL):", 
                                         15000, 160000, (30000, 90000), step=1000)
        
        submit = st.button("Analiz Et ve Öner 🚀")

    # --- Ana Ekran Mesajı ---
    if not submit:
        st.info("Lütfen kriterlerinizi belirledikten sonra sol taraftaki butona basarak analizi başlatın.")
        # Deprecated parametre düzeltildi: use_column_width -> use_container_width 
        st.image("https://images.unsplash.com/photo-1496181133206-80ce9b88a853?auto=format&fit=crop&q=80&w=1000", use_container_width=True)

    if submit:
        min_p, max_p = budget_range
        
        # 1. Sert Filtreleme (Hard Constraints) [cite: 25]
        filtered_df = df[(df['OS'] == os_pref) & 
                         (df['Fiyat'] >= min_p) & 
                         (df['Fiyat'] <= max_p)].copy()
        
        if filtered_df.empty:
            st.error(f"❌ Üzgünüz, {min_p} - {max_p} TL aralığında uygun bir {os_pref} cihaz bulunamadı.")
        else:
            # 2. Normalizasyon (Min-Max) [cite: 24, 35]
            scaler = MinMaxScaler()
            features = ['CPU_Skor', 'GPU_Skor', 'RAM_GB', 'Mobilite_Puani']
            scaler.fit(df[features])
            scaled_data = scaler.transform(filtered_df[features])
            
            # 3. Hedef Vektörler
            if amac == "Yazılım Geliştirme": target = [[95, 55, 32, 5]]
            elif amac == "Oyun": target = [[90, 96, 64, 2]]
            elif amac == "Grafik Tasarım": target = [[95, 92, 32, 4]]
            elif amac == "Ofis / İş": target = [[75, 40, 16, 8]]
            else: target = [[65, 35, 16, 9]]
            
            target_scaled = scaler.transform(target)
            
            # 4. KNN Analizi (Euclidean) [cite: 22, 36]
            knn = NearestNeighbors(n_neighbors=min(3, len(filtered_df)), metric='euclidean')
            knn.fit(scaled_data)
            distances, indices = knn.kneighbors(target_scaled)
            
            # 5. Sonuçların Görselleştirilmesi 
            st.subheader(f"✅ {amac} İçin Belirlenen En İyi 3 Eşleşme")
            
            results = filtered_df.iloc[indices[0]].copy()
            results['Uyum_Skoru'] = np.round(100 - (distances[0] * 12), 1)
            results['Uyum_Skoru'] = results['Uyum_Skoru'].clip(lower=10, upper=100)

            for idx, row in results.iterrows():
                st.markdown(f"""
                <div class="laptop-card">
                    <div class="device-title">🏆 {row['Model']}</div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <b>Fiyat:</b> {row['Fiyat']:,} TL <br>
                            <b>Donanım:</b> {row['RAM_GB']}GB RAM | {row['OS']} <br>
                            <b>Ağırlık:</b> {row['Agirlik_Kg']} Kg
                        </div>
                        <div class="metric-text">%{row['Uyum_Skoru']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            # Balonlar kaldırıldı.

if __name__ == "__main__":
    main()