"""
ml_pipeline.py - Machine Learning Pipeline for Laptop Recommendation
(KNN ve K-Means Tabanlı Orijinal Versiyon - Fiyatsız)

This module implements:
  1. Feature Engineering (Total_Weight)
  2. MinMaxScaler normalization with inverted weight (portability)
  3. K-Means clustering for Performance Segmentation (Giriş, Orta, Üst)
  4. Hard-constraint filtering (Performance + portability)
  5. KNN-based recommendation engine (Euclidean distance)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# ────────────────────────────────────────────
# Usage-scenario target profiles
# Each vector: [CPU, GPU, RAM_norm, SSD_speed_norm, Cooling, Battery, Portability]
# ────────────────────────────────────────────
USAGE_PROFILES = {
    "Veri Bilimi / AI":   [0.90, 0.40, 0.90, 0.80, 0.60, 0.60, 0.60],
    "Yazılım Geliştirme":          [0.85, 0.30, 0.80, 0.80, 0.50, 0.80, 0.70],
    "Oyun (Gaming)":               [0.85, 0.95, 0.80, 0.70, 0.90, 0.20, 0.20],
    "Video Düzenleme":             [0.85, 0.85, 0.90, 0.90, 0.80, 0.40, 0.40],
    "Ofis / Gündelik":      [0.40, 0.10, 0.30, 0.40, 0.30, 0.95, 0.90],
}

FEATURE_COLS = [
    "CPU_norm", "GPU_norm", "RAM_norm",
    "SSD_Hiz_norm", "Sogutma_norm", "Pil_norm", "Portability",
]

PORTABILITY_LIMITS = {
    "Hafif (< 1.8 kg)":   (0.0, 1.8),
    "Orta (1.8 – 2.5 kg)": (1.8, 2.5),
    "Ağır (> 2.5 kg)":    (2.5, 99.0),
    "Fark Etmez":         (0.0, 99.0),
}

def calculate_diminishing_ram(ram):
    if ram <= 32: return float(ram)
    elif ram <= 64: return 32.0 + (ram - 32.0) * 0.25
    else: return 32.0 + 32.0 * 0.25 + (ram - 64.0) * 0.1

def calculate_diminishing_ssd(ssd):
    if ssd <= 2000: return float(ssd)
    else: return 2000.0 + (ssd - 2000.0) * 0.1

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Total_Weight"] = df["Laptop_Agirlik_kg"] + df["Adaptor_Agirlik_kg"]
    df["Total_Raw_Power"] = df["CPU_Skoru"] + df["GPU_Skoru"]
    df["Verimlilik"] = df["Total_Raw_Power"] / df["TDP_Watt"]
    
    # Bonus/Ceza Dengesi (Diminishing Returns)
    df["RAM_Effective"] = df["RAM_GB"].apply(calculate_diminishing_ram)
    df["SSD_Effective"] = df["SSD_GB"].apply(calculate_diminishing_ssd)
    
    return df

def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scaler = MinMaxScaler()

    cols_to_norm = {
        "CPU_Skoru": "CPU_norm",
        "CPU_Single_Core": "CPU_Single_norm",
        "CPU_Multi_Core": "CPU_Multi_norm",
        "GPU_Skoru": "GPU_norm",
        "RAM_Effective": "RAM_norm",  # Eskiden RAM_GB idi, şimdi kısıtlanmış hali
        "SSD_Hiz_MBs": "SSD_Hiz_norm",
        "SSD_Effective": "SSD_Kapasite_norm",
        "Sogutma_Skoru": "Sogutma_norm",
        "Pil_Video_Saat": "Pil_norm",
    }

    for src, dst in cols_to_norm.items():
        df[dst] = scaler.fit_transform(df[[src]]).flatten()

    w = df[["Total_Weight"]].values
    w_scaled = scaler.fit_transform(w).flatten()
    df["Portability"] = 1.0 - w_scaled

    return df

def cluster_performance(df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42):
    df = df.copy()
    # "Watt başına alınan performans (İşlem gücü)" analizi için Verimlilik kullanılıyor
    X = df[["Total_Raw_Power", "Verimlilik"]].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df["Cluster"] = km.fit_predict(X_scaled)

    centres = km.cluster_centers_
    power_order = np.argsort(centres[:, 0])
    
    label_map = {}
    ordered_names = ["Giriş Seviyesi", "Orta Segment", "Üst Segment"]
    for rank, cid in enumerate(power_order):
        label_map[cid] = ordered_names[rank]

    df["Performans_Sinifi"] = df["Cluster"].map(label_map)
    
    # Kural: Apple'ın Max ve yeni nesil Pro çipleri performansta çok yüksek olmasına rağmen 
    # Watt/Performans dengesinden dolayı kümelemede sapabilir. Bunları manuel olarak Üst Segment yapıyoruz.
    apple_top_tier = ["M4 Max", "M5 Max", "M5 Pro", "M4 Pro"]
    df.loc[df["Ekran_Karti"].isin(apple_top_tier), "Performans_Sinifi"] = "Üst Segment"

    # Kural: Yapay Zeka (AI) ve Genel Kullanım için Segment Sabitlemesi
    df.loc[df["Ekran_Karti"].isin(["RTX 5090", "RTX 5080"]), "Performans_Sinifi"] = "Üst Segment"
    df.loc[df["Ekran_Karti"].isin(["RTX 5070", "RTX 5070 Ti"]), "Performans_Sinifi"] = "Orta Segment"
    df.loc[df["Ekran_Karti"].isin(["RTX 5060", "RTX 5050"]), "Performans_Sinifi"] = "Giriş Seviyesi"

    return df, km, scaler

def apply_hard_constraints(
    df: pd.DataFrame,
    perf_category: str,
    portability_key: str,
) -> pd.DataFrame:
    filtered = df.copy()

    # Orijinal kural: KATI filtre. Ağırlığı geçen elenir.
    lo, hi = PORTABILITY_LIMITS[portability_key]
    filtered = filtered[
        (filtered["Total_Weight"] >= lo) & (filtered["Total_Weight"] < hi)
    ]

    # Orijinal kural: KATI filtre. Segment uymayan elenir.
    if perf_category != "Fark Etmez":
        filtered = filtered[filtered["Performans_Sinifi"] == perf_category]

    return filtered.reset_index(drop=True)

def recommend(
    df_filtered: pd.DataFrame,
    usage_key: str,
    top_n: int = 3,
) -> pd.DataFrame:
    if df_filtered.empty:
        return df_filtered

    df_res = df_filtered.copy()
    
    # ── EK PUAN (BONUS) HESAPLAMASI ──
    # Kullanıcının detaylı donanım kriterlerini parametre yerine skorlara ek puan olarak yansıtıyoruz
    bonus_vram = df_res["VRAM_GB"] / 4.0
    bonus_port = df_res["Portlar"] / 2.0
    bonus_ram = (df_res["RAM_MHz"] - 4800).clip(lower=0) / 1000.0
    bonus_ssd = (df_res["SSD_Hiz_MBs"] - 3000).clip(lower=0) / 1000.0
    bonus_ssd_kapasite = df_res["SSD_Kapasite_norm"] * 3.0
    bonus_ekran = (df_res["Ekran_Boyutu"] >= 16.0).astype(float) * 1.5 + (df_res["Ekran_Boyutu"] >= 18.0).astype(float) * 1.5
    
    def get_res_bonus(res_str):
        res_str = str(res_str).upper()
        if "4K" in res_str or "UHD" in res_str: return 3.0
        if "3K" in res_str or "QHD" in res_str or "WQXGA" in res_str or "RETINA" in res_str: return 2.0
        return 0.0
        
    bonus_coz = df_res["Ekran_Cozunurluk"].apply(get_res_bonus)
    total_bonus = bonus_vram + bonus_port + bonus_ram + bonus_ssd + bonus_ssd_kapasite + bonus_ekran + bonus_coz
    
    def deduplicate_df(d: pd.DataFrame) -> pd.DataFrame:
        seen = set()
        keep = []
        for idx, row in d.iterrows():
            base_model = str(row["Model"]).split("(")[0].strip()
            if base_model not in seen:
                seen.add(base_model)
                keep.append(idx)
        return d.loc[keep].reset_index(drop=True)
        
    # ── UZMAN SİSTEM (EXPERT SYSTEM) KURALLARI ──

    # Kural: "Ofis / Gündelik" ve "Yazılım Geliştirme" haricindeki ağır senaryolarda paylaşımlı (iGPU) kartları komple ele.
    if usage_key not in ["Ofis / Gündelik", "Yazılım Geliştirme"]:
        igpus = ["Radeon 890M", "Arc Graphics", "UHD Graphics"]
        df_res = df_res[~df_res["Ekran_Karti"].isin(igpus)].copy()
        if df_res.empty: return df_res

    if usage_key == "Veri Bilimi / AI":
        allowed_gpus = ["RTX 5070", "RTX 5070 Ti", "RTX 5080", "RTX 5090", "M4 Pro", "M4 Max", "M5 Pro", "M5 Max"]
        df_res = df_res[df_res["Ekran_Karti"].isin(allowed_gpus)].copy()
        df_res = df_res[df_res["WiFi_7"] == True].copy()
        if df_res.empty: return df_res
            
        # AI Spesifik Skorlama: Yüksek VRAM, GPU Gücü, RAM ve SSD Kapasitesi ön planda
        df_res["Uygunluk_Skoru"] = (
            df_res["GPU_norm"] * 30 +
            (df_res["VRAM_GB"] / 32.0) * 25 +  # VRAM miktarı AI için çok kritik
            df_res["RAM_norm"] * 25 +          # RAM kapasitesi
            df_res["SSD_Kapasite_norm"] * 15 + # SSD Kapasitesi
            df_res["CPU_norm"] * 5
        ).round(1)
        
        # Ek puanları ekle
        df_res["Uygunluk_Skoru"] += total_bonus[df_res.index]

    elif usage_key == "Video Düzenleme":
        apple_df = df_res[df_res["Marka"] == "Apple"].copy()
        win_df = df_res[df_res["Marka"] != "Apple"].copy()
        
        if not apple_df.empty:
            apple_df["Uygunluk_Skoru"] = (apple_df["CPU_norm"] * 30 + apple_df["GPU_norm"] * 30 + apple_df["SSD_Hiz_norm"] * 40).round(1)
            apple_df["Uygunluk_Skoru"] += total_bonus[apple_df.index]
            
            # Kural: Video düzenlemede ağır/orta kasalarda M5 Max > M4 Max, hafif kasalarda M5 Pro > M4 Pro kazanmalı
            is_light = apple_df["Total_Weight"] < 2.0
            
            # Hafif Kasalar İçin (Örn: 14 inç)
            apple_df.loc[is_light & (apple_df["Ekran_Karti"] == "M5 Pro"), "Uygunluk_Skoru"] += 1000
            apple_df.loc[is_light & (apple_df["Ekran_Karti"] == "M4 Pro"), "Uygunluk_Skoru"] += 900
            apple_df.loc[is_light & (apple_df["Ekran_Karti"] == "M5 Max"), "Uygunluk_Skoru"] += 800
            
            # Kalın Kasalar İçin (Örn: 16 inç)
            apple_df.loc[~is_light & (apple_df["Ekran_Karti"] == "M5 Max"), "Uygunluk_Skoru"] += 1000
            apple_df.loc[~is_light & (apple_df["Ekran_Karti"] == "M4 Max"), "Uygunluk_Skoru"] += 900
            apple_df.loc[~is_light & (apple_df["Ekran_Karti"] == "M5 Pro"), "Uygunluk_Skoru"] += 800
            
            apple_df = apple_df.sort_values(by="Uygunluk_Skoru", ascending=False)
            apple_df = deduplicate_df(apple_df)
            
        if not win_df.empty:
            win_df["Uygunluk_Skoru"] = (win_df["GPU_norm"] * 40 + win_df["CPU_norm"] * 30 + win_df["Panel_Kalitesi"] / 100.0 * 30).round(1)
            win_df["Uygunluk_Skoru"] += total_bonus[win_df.index]
            win_df = win_df.sort_values(by="Uygunluk_Skoru", ascending=False)
            win_df = deduplicate_df(win_df)
            
        top_apples = apple_df.head(2) if not apple_df.empty else pd.DataFrame()
        top_wins = win_df.head(top_n - len(top_apples)) if not win_df.empty else pd.DataFrame()
        
        final_res = pd.concat([top_apples, top_wins])
        if not final_res.empty:
             final_res["Uygunluk_Skoru"] = np.clip(final_res["Uygunluk_Skoru"] * 100, 85, 100).round(1)
        return final_res.head(top_n)

    elif usage_key == "Yazılım Geliştirme":
        df_res["Uygunluk_Skoru"] = (
            df_res["CPU_norm"] * 40 +
            (df_res["RAM_MHz"] / 8533) * 30 +
            df_res["SSD_Hiz_norm"] * 25 +
            df_res["GPU_norm"] * 5
        ).round(1)
        df_res["Uygunluk_Skoru"] += total_bonus

    elif usage_key == "Ofis / Gündelik":
        # Katı Kural: Ofis ve Gündelik kullanımda devasa "Oyuncu Laptopları" (Gaming) önerilmez.
        df_res = df_res[df_res["Is_Gaming"] == False].copy()
        if df_res.empty: return df_res
        
        df_res["Uygunluk_Skoru"] = (
            df_res["Pil_norm"] * 35 +
            df_res["Portability"] * 30 +
            (df_res["Panel_Kalitesi"] / 100.0) * 30 +
            df_res["CPU_norm"] * 5
        ).round(1)
        # Bu kategori için çözünürlük ve panel çok önemli olduğundan, çözünürlük bonusunu 2'ye katlıyoruz
        df_res["Uygunluk_Skoru"] += total_bonus[df_res.index] + (bonus_coz[df_res.index] * 2.0)
        
    elif usage_key == "Oyun (Gaming)":
        # Katı Kural: Oyun için sadece "Gerçek" Gaming laptoplar yarışabilir. Apple ve Ultrabooklar elenir.
        df_res = df_res[(df_res["Is_Gaming"] == True) & (df_res["Marka"] != "Apple")].copy()
        if df_res.empty: return df_res
        
        df_res["Uygunluk_Skoru"] = (
            df_res["GPU_norm"] * 50 +
            df_res["Sogutma_norm"] * 20 +
            df_res["CPU_Single_norm"] * 15 +  # Oyunlar tekli çekirdek sever
            df_res["CPU_Multi_norm"] * 10 +
            df_res["RAM_norm"] * 5
        ).round(1)
        df_res["Uygunluk_Skoru"] += total_bonus[df_res.index]

    # Genel Skorları 100 üzerinden gösterilebilir hale getir
    if not df_res.empty:
        max_score = df_res["Uygunluk_Skoru"].max()
        if max_score > 0:
            df_res["Uygunluk_Skoru"] = (df_res["Uygunluk_Skoru"] / max_score * 99).round(1)
        else:
            df_res["Uygunluk_Skoru"] = 80.0
            
    df_res = df_res.sort_values(
        by=["Uygunluk_Skoru", "Marka_Skoru"],
        ascending=[False, False],
    ).reset_index(drop=True)

    # ── KURAL: Aynı modelin farklı konfigürasyonlarını eleme (Deduplication) ──
    df_res = deduplicate_df(df_res)

    return df_res.head(top_n)

def run_pipeline(csv_path: str = "data/laptop_dataset_2025_2026.csv"):
    df = pd.read_csv(csv_path)
    df = engineer_features(df)
    df = normalize_features(df)
    df, km, scaler = cluster_performance(df)
    return df, km, scaler
