"""
ml_pipeline.py - Akıllı Bilgisayar Öneri Sistemi İçin Makine Öğrenmesi Boru Hattı
(K-Means ve Uzman Sistem Tabanlı Ana İşlem Motoru)

Bu modül aşağıdaki temel görevleri yerine getirir:
  1. Özellik Mühendisliği (Donanım güçleri, verimlilik ve ağırlık hesaplamaları)
  2. MinMaxScaler ile Veri Normalizasyonu ve Taşınabilirlik (Portability) skorlaması
  3. K-Means Kümeleme Algoritması ile Performans Segmentasyonu (Giriş, Orta, Üst)
  4. Kullanıcı Tercihlerine Göre Katı Kısıt (Hard-Constraint) Filtrelemesi
  5. Kural Tabanlı Uzman Sistem Destekli Öneri Motoru (Recommendation Engine)
"""

import numpy as np                 # Vektörel ve matris tabanlı matematiksel işlemler için.
import pandas as pd                # Veri analizi ve DataFrame manipülasyonu için.
from sklearn.preprocessing import MinMaxScaler  # Değerleri 0 ile 1 arasına ölçeklemek için.
from sklearn.cluster import KMeans              # Performans segmentasyonu kümelemesi için.

# ─────────────────────────────────────────────────────────────────────────────
# 1. KULLANIM SENARYOLARINA GÖRE HEDEF DONANIM PROFİLLERİ (USAGE PROFILES)
# ─────────────────────────────────────────────────────────────────────────────
# Her bir kullanım profili, bir laptopun sahip olması gereken ideal donanım ağırlıklarını temsil eder.
# Vektör Düzeni: [CPU, GPU, RAM_norm, SSD_speed_norm, Cooling, Battery, Portability]
# Bu değerler, öneri aşamasında cihazların uygunluk skorlarını hesaplarken temel ağırlık katsayıları olarak rol oynar.
# Not: Bu sözlük dış modüller ve entegrasyonlar tarafından referans alınabileceği için korunmuştur.
USAGE_PROFILES = {
    "Veri Bilimi / AI":   [0.90, 0.40, 0.90, 0.80, 0.60, 0.60, 0.60], # CPU, RAM ve SSD odaklı
    "Yazılım Geliştirme":          [0.85, 0.30, 0.80, 0.80, 0.50, 0.80, 0.70], # CPU, Pil ve RAM odaklı
    "Oyun (Gaming)":               [0.85, 0.95, 0.80, 0.70, 0.90, 0.20, 0.20], # GPU, CPU ve Soğutma odaklı, Pil önemsiz
    "Video Düzenleme":             [0.85, 0.85, 0.90, 0.90, 0.80, 0.40, 0.40], # Dengeli güçlü CPU/GPU ve yüksek hızlı depolama
    "Ofis / Gündelik":      [0.40, 0.10, 0.30, 0.40, 0.30, 0.95, 0.90], # Pil ömrü ve hafiflik (taşınabilirlik) öncelikli
}

# Normalizasyon sonrasında benzerlik hesaplamasında kullanılacak olan ana özellik sütunları
FEATURE_COLS = [
    "CPU_norm", "GPU_norm", "RAM_norm",
    "SSD_Hiz_norm", "Sogutma_norm", "Pil_norm", "Portability",
]

# Taşınabilirlik (Ağırlık) sınıf kısıtlamalarının sınır değerleri (kg cinsinden)
PORTABILITY_LIMITS = {
    "Hafif (< 1.8 kg)":   (0.0, 1.8),
    "Orta (1.8 – 2.5 kg)": (1.8, 2.5),
    "Ağır (> 2.5 kg)":    (2.5, 99.0),
    "Fark Etmez":         (0.0, 99.0),
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. AZALAN VERİM (DIMINISHING RETURNS) FONKSİYONLARI
# ─────────────────────────────────────────────────────────────────────────────
# Bu fonksiyonlar, donanım kapasitesi arttıkça kullanıcının elde edeceği marjinal faydanın
# doğrusal olmayan bir şekilde azalmasını simüle eder. Örneğin, 32GB'tan sonraki her GB RAM
# kullanıcıya ilk 16GB kadar büyük bir fark hissettirmez.

def calculate_diminishing_ram(ram):
    """
    RAM kapasitesi arttıkça doğrusal artışı sınırlandırır.
    32 GB'a kadar olan kısım tam etki yapar, 32-64 GB arası %25 etki yapar,
    64 GB üzeri ise sadece %10 oranında puan kazandırır.
    """
    if ram <= 32: return float(ram)
    elif ram <= 64: return 32.0 + (ram - 32.0) * 0.25
    else: return 32.0 + 32.0 * 0.25 + (ram - 64.0) * 0.1

def calculate_diminishing_ssd(ssd):
    """
    SSD kapasitesindeki azalan verimi simüle eder.
    2000 GB (2 TB) kapasiteye kadar tam puan verilirken, 2 TB üzerindeki ek kapasite
    kullanıcıya sadece %10 oranında ek puan katkısı sağlar.
    """
    if ssd <= 2000: return float(ssd)
    else: return 2000.0 + (ssd - 2000.0) * 0.1

# ─────────────────────────────────────────────────────────────────────────────
# 3. ÖZNİTELİK MÜHENDİSLİĞİ (FEATURE ENGINEERING)
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mevcut sütunları birleştirerek sisteme anlamlı yeni öznitelikler kazandırır.
    """
    df = df.copy()
    
    # Toplam Ağırlık = Laptop Ağırlığı + Adaptör Ağırlığı (Kullanıcının sırtında taşıyacağı gerçek yük)
    df["Total_Weight"] = df["Laptop_Agirlik_kg"] + df["Adaptor_Agirlik_kg"]
    
    # Ham Güç = CPU Performansı + GPU Performansı
    df["Total_Raw_Power"] = df["CPU_Skoru"] + df["GPU_Skoru"]
    
    # Güç Verimliliği = Toplam Güç / Harcanan Enerji (TDP Watt)
    df["Verimlilik"] = df["Total_Raw_Power"] / df["TDP_Watt"]
    
    # Azalan verim yasasına göre RAM ve SSD kapasitelerinin efektif değerlerini hesapla
    df["RAM_Effective"] = df["RAM_GB"].apply(calculate_diminishing_ram)
    df["SSD_Effective"] = df["SSD_GB"].apply(calculate_diminishing_ssd)
    
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 4. ÖZNİTELİK NORMALİZASYONU (FEATURE NORMALIZATION)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tüm sayısal donanım özelliklerini 0 ile 1 arasına (MinMaxScaler ile) çeker.
    Ağırlık özelliğini tersine çevirerek "Portability" (Taşınabilirlik) puanına dönüştürür.
    """
    df = df.copy()
    scaler = MinMaxScaler()

    # Orijinal sütunların hangi yeni normalize edilmiş sütun isimlerine eşleneceği sözlüğü
    cols_to_norm = {
        "CPU_Skoru": "CPU_norm",
        "CPU_Single_Core": "CPU_Single_norm",
        "CPU_Multi_Core": "CPU_Multi_norm",
        "GPU_Skoru": "GPU_norm",
        "RAM_Effective": "RAM_norm",  # Kısıtlandırılmış/efektif RAM kapasitesi normalize ediliyor
        "SSD_Hiz_MBs": "SSD_Hiz_norm",
        "SSD_Effective": "SSD_Kapasite_norm",
        "Sogutma_Skoru": "Sogutma_norm",
        "Pil_Video_Saat": "Pil_norm",
    }

    # Belirtilen özellikleri sütun sütun normalize et
    for src, dst in cols_to_norm.items():
        df[dst] = scaler.fit_transform(df[[src]]).flatten()

    # Ağırlık normalizasyonu ve Taşınabilirlik (Portability) dönüşümü:
    # Ağırlık arttıkça taşınabilirlik azalmalıdır. Bu yüzden 1.0 değerinden çıkarıyoruz (1.0 - w_scaled).
    w = df[["Total_Weight"]].values
    w_scaled = scaler.fit_transform(w).flatten()
    df["Portability"] = 1.0 - w_scaled

    return df

# ─────────────────────────────────────────────────────────────────────────────
# 5. PERFORMANS KÜMELEME (K-MEANS CLUSTERING)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_performance(df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42):
    """
    Laptopları donanım güçlerine ve enerji verimliliklerine göre 3 farklı segmente
    (Giriş Seviyesi, Orta Segment, Üst Segment) kümeleyen K-Means algoritması.
    """
    df = df.copy()
    
    # Kümelemede girdi olarak Toplam Ham Güç ve Verimlilik değerlerini kullanıyoruz
    X = df[["Total_Raw_Power", "Verimlilik"]].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means modelini çalıştırıp küme numaralarını (Cluster) tahmin et
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df["Cluster"] = km.fit_predict(X_scaled)

    # Oluşan kümeleri toplam güçlerine göre sıralayarak isimlendirmeyi garantiliyoruz.
    # Bu sayede güç değeri en düşük olan küme her zaman "Giriş Seviyesi" olur.
    centres = km.cluster_centers_
    power_order = np.argsort(centres[:, 0]) # Kümelerin ilk özniteliğe (Güç) göre sıralanmış indeksleri
    
    label_map = {}
    ordered_names = ["Giriş Seviyesi", "Orta Segment", "Üst Segment"]
    for rank, cid in enumerate(power_order):
        label_map[cid] = ordered_names[rank]

    # Küme numaralarını (0, 1, 2) segment isimlerine ("Giriş Seviyesi", vb.) eşliyoruz
    df["Performans_Sinifi"] = df["Cluster"].map(label_map)
    
    # ── UZMAN KURAL MÜDAHALELERİ (EXPERT OVERRIDES) ──
    # Makine öğrenmesi modeli (K-Means), Apple Silicon (M4/M5) işlemcilerin aşırı yüksek verimlilik değerlerinden 
    # ötürü bu cihazları yanlış kümelere atayabilir. Bu sapmaları düzeltmek için uzman kuralları devreye alıyoruz:
    
    # Kural 1: Üst seviye Apple silikon işlemcileri doğrudan Üst Segmente çek
    apple_top_tier = ["M4 Max", "M5 Max", "M5 Pro", "M4 Pro"]
    df.loc[df["Ekran_Karti"].isin(apple_top_tier), "Performans_Sinifi"] = "Üst Segment"

    # Kural 2: NVIDIA RTX ekran kartlarına göre segmentleri kesin olarak sabitle
    df.loc[df["Ekran_Karti"].isin(["RTX 5090", "RTX 5080"]), "Performans_Sinifi"] = "Üst Segment"
    df.loc[df["Ekran_Karti"].isin(["RTX 5070", "RTX 5070 Ti"]), "Performans_Sinifi"] = "Orta Segment"
    df.loc[df["Ekran_Karti"].isin(["RTX 5060", "RTX 5050"]), "Performans_Sinifi"] = "Giriş Seviyesi"

    return df, km, scaler

# ─────────────────────────────────────────────────────────────────────────────
# 6. KATI KISIT FİLTRELEME (HARD CONSTRAINTS FILTER)
# ─────────────────────────────────────────────────────────────────────────────

def apply_hard_constraints(
    df: pd.DataFrame,
    perf_category: str,
    portability_key: str,
) -> pd.DataFrame:
    """
    Kullanıcının kesin sınırlarını (Ağırlık ve Segment tercihlerini) 
    veri kümesine katı filtre olarak uygular. Uymayan cihazları tamamen eler.
    """
    filtered = df.copy()

    # Katı Ağırlık Kısıtı Filtresi: Belirlenen ağırlık aralığının dışındaki tüm cihazları eler.
    lo, hi = PORTABILITY_LIMITS[portability_key]
    filtered = filtered[
        (filtered["Total_Weight"] >= lo) & (filtered["Total_Weight"] < hi)
    ]

    # Katı Performans Kısıtı Filtresi: "Fark Etmez" seçilmediyse uymayan segmentteki cihazları eler.
    if perf_category != "Fark Etmez":
        filtered = filtered[filtered["Performans_Sinifi"] == perf_category]

    return filtered.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# 7. UZMAN SİSTEM TABANLI ÖNERİ MOTORU (RECOMMENDATION ENGINE)
# ─────────────────────────────────────────────────────────────────────────────

def recommend(
    df_filtered: pd.DataFrame,
    usage_key: str,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Filtrelenmiş veri seti üzerinde uzman sistem kuralları ve ağırlıklı puanlama
    yöntemi kullanarak kullanıcının amacına en uygun dizüstü bilgisayarları listeler.
    """
    # Veri seti boşsa doğrudan boş veri setini döndürür
    if df_filtered.empty:
        return df_filtered

    df_res = df_filtered.copy()
    
    # ── EK PUAN (BONUS) HESAPLAMA SİSTEMİ ──
    # Temel donanım skorlarının haricinde, cihazların sahip olduğu avantajlı ekstra nitelikleri
    # nihai uygunluk skoruna artı puan olarak yansıtıyoruz.
    bonus_vram = df_res["VRAM_GB"] / 4.0                               # Her 4 GB VRAM için +1 Puan
    bonus_port = df_res["Portlar"] / 2.0                               # Her 2 port için +1 Puan
    bonus_ram = (df_res["RAM_MHz"] - 4800).clip(lower=0) / 1000.0       # 4800 MHz üzerindeki her 1000 MHz RAM hızı için +1 Puan
    bonus_ssd = (df_res["SSD_Hiz_MBs"] - 3000).clip(lower=0) / 1000.0   # 3000 MB/s üzerindeki her 1000 MB/s SSD hızı için +1 Puan
    bonus_ssd_kapasite = df_res["SSD_Kapasite_norm"] * 3.0             # Yüksek depolama kapasitesi için maksimum +3 Puan
    # 16 inç ve 18 inç gibi büyük ekranlar için ekran bonusu
    bonus_ekran = (df_res["Ekran_Boyutu"] >= 16.0).astype(float) * 1.5 + (df_res["Ekran_Boyutu"] >= 18.0).astype(float) * 1.5
    
    # Ekran çözünürlük ifadelerine göre çözünürlük bonusunu hesaplayan iç yardımcı fonksiyon
    def get_res_bonus(res_str):
        res_str = str(res_str).upper()
        if "4K" in res_str or "UHD" in res_str: return 3.0
        if "3K" in res_str or "QHD" in res_str or "WQXGA" in res_str or "RETINA" in res_str: return 2.0
        return 0.0
        
    bonus_coz = df_res["Ekran_Cozunurluk"].apply(get_res_bonus)
    
    # Tüm bonusların toplanması
    total_bonus = bonus_vram + bonus_port + bonus_ram + bonus_ssd + bonus_ssd_kapasite + bonus_ekran + bonus_coz
    
    # Çift Kayıt Engelleme Yardımcı Fonksiyonu (Deduplication):
    # Aynı laptop modelinin farklı RAM-SSD varyasyonlarının ilk 3 sırayı kapatmasını önler.
    # Her model serisinden sadece en yüksek puanlı tek bir cihazın kalmasını sağlar.
    def deduplicate_df(d: pd.DataFrame) -> pd.DataFrame:
        seen = set()
        keep = []
        for idx, row in d.iterrows():
            # Parantez öncesindeki ana model ismini alıyoruz (Örn: "ROG Zephyrus G16 (RTX 5080)" -> "ROG Zephyrus G16")
            base_model = str(row["Model"]).split("(")[0].strip()
            if base_model not in seen:
                seen.add(base_model)
                keep.append(idx)
        return d.loc[keep].reset_index(drop=True)
        
    # ── UZMAN SİSTEM MÜHENDİSLİK KURALLARI (EXPERT RULES) ──

    # Kural: Ofis ve Yazılım dışındaki ağır grafik gerektiren işlerde dahili/paylaşımlı (iGPU) kartları tamamen ele.
    if usage_key not in ["Ofis / Gündelik", "Yazılım Geliştirme"]:
        igpus = ["Radeon 890M", "Arc Graphics", "UHD Graphics"]
        df_res = df_res[~df_res["Ekran_Karti"].isin(igpus)].copy()
        if df_res.empty: return df_res

    # ──── SENARYO 1: VERİ BİLİMİ / YAPAY ZEKA (AI) ────
    if usage_key == "Veri Bilimi / AI":
        # Katı Kural: AI geliştiricileri sadece NVIDIA CUDA çekirdekleri taşıyan kartları veya Apple Unified Memory kullanan kartları seçebilir.
        allowed_gpus = ["RTX 5070", "RTX 5070 Ti", "RTX 5080", "RTX 5090", "M4 Pro", "M4 Max", "M5 Pro", "M5 Max"]
        df_res = df_res[df_res["Ekran_Karti"].isin(allowed_gpus)].copy()
        
        # Katı Kural: AI projelerinde veri indirme/aktarma hızı için WiFi 7 teknolojisi zorunludur.
        df_res = df_res[df_res["WiFi_7"] == True].copy()
        if df_res.empty: return df_res
            
        # Yapay Zeka Ağırlıklı Uygunluk Skor Formülü:
        # GPU gücü (%30), VRAM kapasitesi (%25), RAM kapasitesi (%25), SSD kapasitesi (%15) ve CPU gücü (%5)
        df_res["Uygunluk_Skoru"] = (
            df_res["GPU_norm"] * 30 +
            (df_res["VRAM_GB"] / 32.0) * 25 +  
            df_res["RAM_norm"] * 25 +          
            df_res["SSD_Kapasite_norm"] * 15 + 
            df_res["CPU_norm"] * 5
        ).round(1)
        
        # Ek donanım bonuslarını ekle
        df_res["Uygunluk_Skoru"] += total_bonus[df_res.index]

    # ──── SENARYO 2: VİDEO DÜZENLEME / RENDER (VIDEO EDITING) ────
    elif usage_key == "Video Düzenleme":
        # Video düzenleme uzmanları arasında Apple ve Windows cihazlarının seçimi çok farklı dinamiklere sahiptir.
        # Bu nedenle veri setini Apple ve Windows olarak ikiye ayırıp ayrı formüllerle puanlıyoruz.
        apple_df = df_res[df_res["Marka"] == "Apple"].copy()
        win_df = df_res[df_res["Marka"] != "Apple"].copy()
        
        if not apple_df.empty:
            # Apple puanlaması: CPU gücü (%30), GPU gücü (%30) ve SSD hızı (%40)
            apple_df["Uygunluk_Skoru"] = (apple_df["CPU_norm"] * 30 + apple_df["GPU_norm"] * 30 + apple_df["SSD_Hiz_norm"] * 40).round(1)
            apple_df["Uygunluk_Skoru"] += total_bonus[apple_df.index]
            
            # Sıralama için geçici bir sütunda sıralama puanını saklayıp uzman kurallarını buraya ekliyoruz
            # Bu sayede gerçek uygunluk skoru yapay olarak şişmez ve anlamlı ölçekte kalır.
            apple_df["Siralama_Skoru"] = apple_df["Uygunluk_Skoru"].copy()
            
            # Kural: Video düzenlemede ağır/kalın kasalarda Max çipleri, hafif kasalarda ise Pro çipleri öncelik kazanmalıdır.
            is_light = apple_df["Total_Weight"] < 2.0
            
            # Hafif Kasalar İçin (Örn: MacBook Pro 14 inç)
            apple_df.loc[is_light & (apple_df["Ekran_Karti"] == "M5 Pro"), "Siralama_Skoru"] += 1000
            apple_df.loc[is_light & (apple_df["Ekran_Karti"] == "M4 Pro"), "Siralama_Skoru"] += 900
            apple_df.loc[is_light & (apple_df["Ekran_Karti"] == "M5 Max"), "Siralama_Skoru"] += 800
            
            # Kalın Kasalar İçin (Örn: MacBook Pro 16 inç)
            apple_df.loc[~is_light & (apple_df["Ekran_Karti"] == "M5 Max"), "Siralama_Skoru"] += 1000
            apple_df.loc[~is_light & (apple_df["Ekran_Karti"] == "M4 Max"), "Siralama_Skoru"] += 900
            apple_df.loc[~is_light & (apple_df["Ekran_Karti"] == "M5 Pro"), "Siralama_Skoru"] += 800
            
            apple_df = apple_df.sort_values(by="Siralama_Skoru", ascending=False)
            apple_df = deduplicate_df(apple_df)
            
        if not win_df.empty:
            # Windows puanlaması: GPU gücü (%40), CPU gücü (%30) ve Ekran Panel kalitesi (%30)
            win_df["Uygunluk_Skoru"] = (win_df["GPU_norm"] * 40 + win_df["CPU_norm"] * 30 + win_df["Panel_Kalitesi"] / 100.0 * 30).round(1)
            win_df["Uygunluk_Skoru"] += total_bonus[win_df.index]
            win_df = win_df.sort_values(by="Uygunluk_Skoru", ascending=False)
            win_df = deduplicate_df(win_df)
            
        # Sonuçları dengeli sunmak için en iyi Apple modelleri ile en iyi Windows modellerini birleştiriyoruz.
        top_apples = apple_df.head(2) if not apple_df.empty else pd.DataFrame()
        top_wins = win_df.head(top_n - len(top_apples)) if not win_df.empty else pd.DataFrame()
        
        final_res = pd.concat([top_apples, top_wins])
        if not final_res.empty:
             # Skorlar zaten 0-100 ölçeğinde hesaplandığından çarpım yapılmadan 85-100 arasına sınırlanır (clip)
             final_res["Uygunluk_Skoru"] = np.clip(final_res["Uygunluk_Skoru"], 85, 100).round(1)
        return final_res.head(top_n)

    # ──── SENARYO 3: YAZILIM GELİŞTİRME (SOFTWARE DEVELOPMENT) ────
    elif usage_key == "Yazılım Geliştirme":
        # Yazılım geliştirme puanlaması: CPU gücü (%40), RAM frekansı (%30), SSD hızı (%25) ve GPU (%5)
        df_res["Uygunluk_Skoru"] = (
            df_res["CPU_norm"] * 40 +
            (df_res["RAM_MHz"] / 8533) * 30 +
            df_res["SSD_Hiz_norm"] * 25 +
            df_res["GPU_norm"] * 5
        ).round(1)
        # İndeks bazlı güvenli hizalama ile bonusları ekle
        df_res["Uygunluk_Skoru"] += total_bonus[df_res.index]

    # ──── SENARYO 4: OFİS VE GÜNDELİK KULLANIM (OFFICE / CASUAL) ────
    elif usage_key == "Ofis / Gündelik":
        # Katı Kural: Günlük ve ofis kullanımları için kalın ve kaba "Oyuncu Laptopları" elenir.
        df_res = df_res[df_res["Is_Gaming"] == False].copy()
        if df_res.empty: return df_res
        
        # Ofis puanlaması: Pil ömrü (%35), Hafiflik/Taşınabilirlik (%30), Ekran kalitesi (%30) ve CPU (%5)
        df_res["Uygunluk_Skoru"] = (
            df_res["Pil_norm"] * 35 +
            df_res["Portability"] * 30 +
            (df_res["Panel_Kalitesi"] / 100.0) * 30 +
            df_res["CPU_norm"] * 5
        ).round(1)
        # Günlük kullanımda ekran çözünürlüğü göz yormaması için kritik olduğundan çözünürlük bonusunu ikiye katlıyoruz.
        df_res["Uygunluk_Skoru"] += total_bonus[df_res.index] + (bonus_coz[df_res.index] * 2.0)
        
    # ──── SENARYO 5: OYUN BİLGİSAYARLARI (GAMING) ────
    elif usage_key == "Oyun (Gaming)":
        # Katı Kural: Oyun oynamak isteyen kullanıcılara sadece gerçek oyuncu serisi (Windows) bilgisayarları önerilir. 
        # Apple cihazları ve dahili grafikli ince ultrabooklar elenir.
        df_res = df_res[(df_res["Is_Gaming"] == True) & (df_res["Marka"] != "Apple")].copy()
        if df_res.empty: return df_res
        
        # Oyun performansı puanlaması: GPU gücü (%50), Soğutma verimliliği (%20), 
        # Tekli çekirdek hızı (%15 - fps için önemlidir), Çoklu çekirdek (%10) ve RAM (%5)
        df_res["Uygunluk_Skoru"] = (
            df_res["GPU_norm"] * 50 +
            df_res["Sogutma_norm"] * 20 +
            df_res["CPU_Single_norm"] * 15 +  
            df_res["CPU_Multi_norm"] * 10 +
            df_res["RAM_norm"] * 5
        ).round(1)
        df_res["Uygunluk_Skoru"] += total_bonus[df_res.index]

    # Nihai skorların 100 üzerinden anlamlı bir yüzdeye normalize edilmesi.
    if not df_res.empty:
        max_score = df_res["Uygunluk_Skoru"].max()
        if max_score > 0:
            df_res["Uygunluk_Skoru"] = (df_res["Uygunluk_Skoru"] / max_score * 99).round(1)
        else:
            df_res["Uygunluk_Skoru"] = 80.0
            
    # Sonuçları önce uygunluk skoruna, eşitlik durumunda ise Marka Güvenilirlik Skoruna göre sıralıyoruz.
    df_res = df_res.sort_values(
        by=["Uygunluk_Skoru", "Marka_Skoru"],
        ascending=[False, False],
    ).reset_index(drop=True)

    # Aynı modelin farklı donanım kombinasyonlarının listeyi doldurmasını önlemek için yinelenenleri eliyoruz.
    df_res = deduplicate_df(df_res)

    return df_res.head(top_n)

# ─────────────────────────────────────────────────────────────────────────────
# 8. MAÇ BAŞLATICI - BORU HATTI (PIPELINE EXECUTION)
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(csv_path: str = "data/laptop_dataset_2025_2026.csv"):
    """
    Sırasıyla tüm adımları (Öznitelik üretimi, Normalizasyon ve K-Means Kümelemesi)
    tetikleyerek veri setini son kullanıma hazır hale getiren ana boru hattı fonksiyonu.
    """
    df = pd.read_csv(csv_path)           # CSV dosyasını diske yükle.
    df = engineer_features(df)           # Yeni öznitelikleri üret (Toplam ağırlık, verimlilik vb.)
    df = normalize_features(df)          # MinMaxScaler ile verileri 0-1 arasına normalize et.
    df, km, scaler = cluster_performance(df) # K-Means ile donanım segmentasyon kümelerini belirle.
    return df, km, scaler                # Veri setini, modeli ve ölçekleyiciyi geri döndür.
