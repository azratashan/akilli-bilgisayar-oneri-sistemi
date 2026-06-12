"""
evaluation.py - Model Evaluation Metrics for the Laptop Recommendation System

Provides clustering quality metrics and Elbow/Silhouette analysis.

Bu modül; makine öğrenmesi modelimizin (K-Means) kümeleme kalitesini 
ölçmek, en uygun küme sayısını belirlemek (Elbow/Dirsek Yöntemi) ve 
kümeleme başarısını istatistiksel metriklerle doğrulamak için kullanılır.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. GEREKLİ VERİ BİLİMİ VE ML METRİK KÜTÜPHANELERİ
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np          # Matematiksel ve matrisel işlemler için.
import pandas as pd         # Veri çerçevesi (DataFrame) işlemleri için.
from sklearn.cluster import KMeans             # K-Means kümeleme algoritması nesnesi.
from sklearn.preprocessing import MinMaxScaler  # Özellikleri 0-1 arasına ölçeklemek için kullanılan nesne.

# Kümeleme kalitesini değerlendirmede kullanılan 3 temel akademik metrik:
# - silhouette_score: Elemanların kendi kümesine benzerliğini ve yan kümelerden uzaklığını ölçer (-1 ile +1 arası; +1'e yakınlık daha iyidir).
# - davies_bouldin_score: Kümelerin birbirine olan uzaklığı ve küme içi yayılım oranını ölçer (0'a yakınlık / düşük değer daha iyidir).
# - calinski_harabasz_score: Kümeler arası varyansın küme içi varyansa oranını ölçer (Daha yüksek değerler, daha net ayrılmış kümeleri belirtir).
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. ELBOW (DİRSEK) VE SILHOUETTE VERİSİ HESAPLAMA FONKSİYONU
# ─────────────────────────────────────────────────────────────────────────────

def compute_elbow_data(X_scaled: np.ndarray, k_range: range = range(2, 11)):
    """
    Belirli bir K (küme sayısı) aralığı için K-Means modelini eğiterek 
    atalet (inertia/WCSS) ve siluet skorlarını hesaplar. 
    Optimum küme sayısını belirlemede kullanılan görsel grafik verilerini hazırlar.
    """
    inertias = []     # Her k değeri için hesaplanan WCSS (Within-Cluster Sum of Squares) değerlerini tutar.
    silhouettes = []  # Her k değeri için hesaplanan ortalama siluet skorlarını tutar.
    
    for k in k_range:
        # K-Means modelini yapılandırıyoruz:
        # - n_clusters=k: Mevcut döngüdeki küme sayısı.
        # - random_state=42: Sonuçların her çalıştırmada aynı çıkmasını sağlar (Tekrarlanabilirlik).
        # - n_init=10: En iyi centroid (merkez) yerleşimini bulmak için algoritmanın 10 kez farklı başlangıçla çalışmasını sağlar.
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        
        # Modeli ölçeklenmiş veriye fit eder (eğitir) ve her veri noktasının küme etiketini (label) tahmin eder.
        labels = km.fit_predict(X_scaled)
        
        # km.inertia_: Kümelerin merkezlerine olan uzaklıklarının kareler toplamıdır.
        # Bu değerin küme sayısı arttıkça düşmesi beklenir. "Dirsek" (Elbow) yaptığı nokta ideal k değeridir.
        inertias.append(km.inertia_)
        
        # Kümenin genel siluet skorunu hesaplayıp listeye ekliyoruz.
        silhouettes.append(silhouette_score(X_scaled, labels))
        
    # Hesaplanan sonuçları grafik çiziminde kolayca kullanılabilmesi için bir sözlük yapısında döndürüyoruz.
    return {
        "k_values": list(k_range),
        "inertias": inertias,
        "silhouettes": silhouettes,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 3. KÜMELEME METRİKLERİ HESAPLAMA FONKSİYONU
# ─────────────────────────────────────────────────────────────────────────────

def compute_cluster_metrics(X_scaled: np.ndarray, labels: np.ndarray) -> dict:
    """
    Verilen küme etiketlerine göre silhouette, davies-bouldin 
    ve calinski-harabasz skorlarını hesaplayıp 4 ondalık basamağa yuvarlayarak döndürür.
    """
    return {
        # Silhouette Skoru: +1'e ne kadar yakınsa kümeler o kadar belirgin ve doğrudur.
        "silhouette": round(silhouette_score(X_scaled, labels), 4),
        
        # Davies-Bouldin Skoru: Değer ne kadar küçükse (0'a yakınsa) kümeler arası ayrışma o kadar başarılıdır.
        "davies_bouldin": round(davies_bouldin_score(X_scaled, labels), 4),
        
        # Calinski-Harabasz Skoru: Yüksek değerler, kümelerin kendi içinde yoğun ve birbirlerinden uzak olduğunu gösterir.
        "calinski_harabasz": round(calinski_harabasz_score(X_scaled, labels), 4),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 4. TÜM MODEL DEĞERLENDİRME AKIŞI (EVALUATION PIPELINE)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pipeline(df: pd.DataFrame, km_model: KMeans, km_scaler: MinMaxScaler):
    """
    Genel Değerlendirme Fonksiyonu: Elbow analizi verilerini toplar ve 
    mevcut eğitilmiş modelin tüm performans metriklerini hesaplar.
    
    Girdiler:
    - df: Laptop veri kümesi (Pandas DataFrame)
    - km_model: Eğitilmiş K-Means model nesnesi
    - km_scaler: Eğitilmiş MinMaxScaler normalizasyon nesnesi
    """
    # Emniyet Kontrolü: Eğer model veya scaler nesnesi boş (None) ise işlem yapmadan None döndür.
    if km_model is None or km_scaler is None:
        return None
        
    # Fiyat bilgisi olmayan boş satırları eliyoruz (Kümeleme özellikleri fiyat ve ham gücü temel alır).
    df_valid = df.dropna(subset=["Fiyat_TL"])
    
    # Metrik hesaplayabilmek için veri setinde en az 3 satır bulunmalıdır (Yetersiz veri durumunda hata alınmasını engeller).
    if len(df_valid) < 3: 
        return None
        
    # Modeli eğitirken kullandığımız özellikleri (Fiyat ve Toplam Ham Güç) seçip numpy matrisine dönüştürüyoruz.
    X = df_valid[["Fiyat_TL", "Total_Raw_Power"]].values
    
    # Verileri eğitilmiş scaler nesnesi ile normalize (0 ile 1 arasına) ediyoruz.
    # UYARI: Yeni bir fit işlemi yapmıyoruz, sadece mevcut scaler'ı uyguluyoruz (transform).
    X_scaled = km_scaler.transform(X)
    
    # Mevcut K-Means modelini kullanarak veri noktalarının hangi kümelere ait olduğunu tahmin ediyoruz.
    labels = km_model.predict(X_scaled)

    # Optimum küme analizi grafikleri için Elbow (Dirsek) verilerini hesaplıyoruz.
    elbow = compute_elbow_data(X_scaled)
    
    # Emniyet Kontrolü: Silhouette skorunun hata vermemesi için veri kümesinde en az 2 farklı küme etiketi bulunmalıdır.
    # (Tüm elemanlar tek bir kümeye atanmışsa siluet skoru matematiksel olarak hesaplanamaz.)
    if len(set(labels)) > 1:
        metrics = compute_cluster_metrics(X_scaled, labels)
    else:
        # Tek küme varsa tüm metrikleri sıfır kabul et.
        metrics = {"silhouette": 0, "davies_bouldin": 0, "calinski_harabasz": 0}
        
    # Mevcut modelin atalet (inertia/WCSS) değerini 4 basamağa yuvarlayarak metriklere ekliyoruz.
    metrics["inertia"] = round(km_model.inertia_, 4)
    
    # Mevcut modelin kaç küme (k) ile eğitildiğini de metrik yapısına kaydediyoruz.
    metrics["n_clusters"] = km_model.n_clusters

    # Çıktıları arayüzde (Streamlit) veya analiz raporlarında kullanılmak üzere topluca döndürüyoruz.
    return {"elbow_data": elbow, "current_metrics": metrics}
