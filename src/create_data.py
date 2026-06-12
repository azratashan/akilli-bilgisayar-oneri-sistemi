"""
create_data.py - Statik Dizüstü Bilgisayar Veri Seti Üreticisi (Uzman Sistem Versiyonu)

Bu modül, kural tabanlı uzman bilgisayar öneri sistemi için
sentetik ve tutarlı bir veri kümesi oluşturur ve CSV formatında dışa aktarır.
Veri seti, güncel RTX 5000 serisi ve Apple M4/M5 donanımları odaklıdır.

Modülün temel amacı: Öneri motoru ve görselleştirme katmanları için
arka planda kullanılacak gerçekçi, tutarlı ve ölçeklenebilir verileri
sentetik olarak üretmektir.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. GEREKLİ KÜTÜPHANELERİN İÇE AKTARILMASI (IMPORTS)
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd  # Üretilen verileri tablo yapısına (DataFrame) dönüştürmek ve CSV'ye yazmak için.
import random        # Rastgelelik katarak veri setini çeşitlendirmek ve daha gerçekçi kılmak için.

# Rastgele sayı üretecinin başlangıç durumunu (seed) sabitleyerek;
# kod her çalıştırıldığında birebir aynı veri kümesinin üretilmesini sağlıyoruz (Tekrarlanabilirlik / Reproducibility).
random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 2. BİLGİ TABANLI YARDIMCI SÖZLÜKLER (GRAPHICS SPECIFICATIONS)
# ─────────────────────────────────────────────────────────────────────────────

# GPU_TDP: Grafik işlemcilerin (GPU) tipik güç tüketimi sınırları (TDP - Thermal Design Power).
# Watt cinsindendir. Güç tüketimi yüksek olan kartların soğutma ve adaptör ağırlık gereksinimleri de artar.
GPU_TDP = {
    "RTX 5090": 220, "RTX 5080": 190, "RTX 5070 Ti": 160, 
    "RTX 5070": 140, "RTX 5060": 115, "RTX 5050": 85,
    "M5 Max": 60, "M5 Pro": 40, "M5": 25,
    "M4 Max": 60, "M4 Pro": 40, "M4": 25,
    "Radeon 890M": 35, "Arc Graphics": 28, "UHD Graphics": 20
}

# GPU_VRAM: Ekran kartlarının sahip olduğu ayrılmış grafik bellek (VRAM - Video RAM) kapasiteleri.
# Gigabayt (GB) cinsindendir. Yapay zeka ve oyun performansı için kritik bir parametredir.
GPU_VRAM = {
    "RTX 5090": 24, "RTX 5080": 16, "RTX 5070 Ti": 12, 
    "RTX 5070": 12, "RTX 5060": 8, "RTX 5050": 8,
    "M5 Max": 64, "M5 Pro": 36, "M5": 24,
    "M4 Max": 48, "M4 Pro": 36, "M4": 24,
    "Radeon 890M": 4, "Arc Graphics": 4, "UHD Graphics": 2
}

# GPU_SCORE: Donanımların uzman sistem tarafından değerlendirilecek bağıl ham performans skorları (0-100 arası).
GPU_SCORE = {
    "RTX 5090": 100, "RTX 5080": 92, "RTX 5070 Ti": 82, 
    "RTX 5070": 78, "RTX 5060": 65, "RTX 5050": 55,
    "M5 Max": 95, "M5 Pro": 82, "M5": 55,
    "M4 Max": 90, "M4 Pro": 78, "M4": 50,
    "Radeon 890M": 50, "Arc Graphics": 45, "UHD Graphics": 25
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. ŞABLON VERİ KÜMESİ (LAPTOP TEMPLATES)
# ─────────────────────────────────────────────────────────────────────────────
# Gerçek dünyadaki laptop modelleri referans alınarak oluşturulmuş tuple (demet) listesi.
# Yapısal düzen: 
# (Marka, Model İsmi, [Desteklediği GPU Listesi], Ekran Boyutu, Çözünürlük Metni, Laptop Ağırlığı(kg), 
#  Soğutma Skoru(0-100), Pil Süresi(Saat), Ekran Panel Puanı(0-100), Oyuncu Bilgisayarı Mı, Port Detayları, NPU Var Mı)
TEMPLATES = [
    ("ASUS", "ROG Strix SCAR 18", ["RTX 5080", "RTX 5090"], 18.0, "2560x1600 (QHD+)", 3.1, 95, 3, 90, True, "2x Thunderbolt 4, 1x HDMI 2.1, 2x USB-A 3.2, Ethernet", True),
    ("ASUS", "ROG Strix G16", ["RTX 5070 Ti", "RTX 5080"], 16.0, "2560x1600 (QHD+)", 2.5, 90, 4, 88, True, "1x Thunderbolt 4, 1x USB-C 3.2, 1x HDMI 2.1, 2x USB-A", True),
    ("ASUS", "ROG Zephyrus G16", ["RTX 5070", "RTX 5070 Ti", "RTX 5080", "RTX 5090"], 16.0, "2560x1600 (QHD+)", 1.85, 88, 6, 95, True, "1x Thunderbolt 4, 1x USB-C 3.2, 1x HDMI 2.1, 2x USB-A", True),
    ("ASUS", "TUF Gaming A15", ["RTX 5050", "RTX 5060", "RTX 5070"], 15.6, "1920x1080 (FHD)", 2.2, 75, 6, 75, True, "1x USB-C 4.0, 1x USB-C 3.2, 1x HDMI 2.1, 2x USB-A", True),
    ("Lenovo", "Legion Pro 7i", ["RTX 5070 Ti", "RTX 5080", "RTX 5090"], 16.0, "2560x1600 (WQXGA)", 2.8, 92, 4, 88, True, "1x Thunderbolt 4, 1x USB-C, 1x HDMI 2.1, 4x USB-A", True),
    ("Lenovo", "Legion Slim 5", ["RTX 5060", "RTX 5070"], 16.0, "2560x1600 (WQXGA)", 2.3, 80, 6, 82, True, "2x USB-C, 1x HDMI 2.1, 2x USB-A", True),
    ("Lenovo", "Yoga Pro 9i", ["RTX 5060", "RTX 5070", "RTX 5070 Ti"], 16.0, "3200x2000 (3.2K Mini-LED)", 2.1, 78, 7, 98, False, "1x Thunderbolt 4, 1x HDMI 2.1, 2x USB-A, SD", True),
    ("MSI", "Titan 18 HX", ["RTX 5090"], 18.0, "3840x2400 (UHD+)", 3.6, 98, 2, 92, True, "2x Thunderbolt 4, 1x HDMI 2.1, 3x USB-A, SD, Ethernet", False),
    ("MSI", "Vector 16 HX", ["RTX 5070", "RTX 5070 Ti", "RTX 5080"], 16.0, "2560x1600 (QHD+)", 2.67, 88, 3, 85, True, "1x Thunderbolt 4, 1x USB-C, 1x HDMI 2.1, 3x USB-A", True),
    ("MSI", "Stealth 16", ["RTX 5060", "RTX 5070", "RTX 5070 Ti"], 16.0, "2560x1600 (QHD+)", 1.99, 82, 5, 88, True, "1x Thunderbolt 4, 1x USB-C, 1x HDMI 2.1, 2x USB-A", True),
    ("MSI", "Cyborg 15", ["RTX 5050", "RTX 5060"], 15.6, "1920x1080 (FHD)", 1.98, 68, 4, 65, True, "1x USB-C, 1x HDMI 2.1, 2x USB-A", False),
    ("Razer", "Blade 16", ["RTX 5070", "RTX 5070 Ti", "RTX 5080", "RTX 5090"], 16.0, "2560x1600 (OLED)", 2.45, 88, 5, 95, True, "1x Thunderbolt 4, 1x USB-C, 1x HDMI 2.1, 3x USB-A", True),
    ("Razer", "Blade 14", ["RTX 5060", "RTX 5070"], 14.0, "2560x1600 (QHD+)", 1.84, 82, 6, 90, True, "2x USB-C 4.0, 1x HDMI 2.1, 2x USB-A", True),
    ("Alienware", "m18 R3", ["RTX 5080", "RTX 5090"], 18.0, "2560x1600 (QHD+)", 3.9, 94, 2, 85, True, "2x Thunderbolt 4, 1x HDMI 2.1, miniDP, 3x USB-A", False),
    ("Alienware", "x16 R3", ["RTX 5070", "RTX 5070 Ti", "RTX 5080"], 16.0, "2560x1600 (QHD+)", 2.6, 86, 4, 90, True, "2x Thunderbolt 4, 1x HDMI 2.1, 2x USB-A", True),
    ("Dell", "XPS 16", ["RTX 5060", "RTX 5070"], 16.3, "3840x2400 (4K OLED)", 2.13, 75, 8, 98, False, "3x Thunderbolt 4, SD Reader (Adapter req)", True),
    ("Dell", "XPS 14", ["RTX 5050"], 14.5, "3200x2000 (3.2K OLED)", 1.68, 70, 9, 95, False, "3x Thunderbolt 4", True),
    ("Acer", "Predator Helios 18", ["RTX 5070 Ti", "RTX 5080", "RTX 5090"], 18.0, "2560x1600 (QHD+)", 3.25, 92, 3, 85, True, "2x Thunderbolt 4, 1x HDMI 2.1, 3x USB-A", True),
    ("Acer", "Nitro V 15", ["RTX 5050", "RTX 5060"], 15.6, "1920x1080 (FHD)", 2.1, 70, 5, 70, True, "1x Thunderbolt 4, 1x HDMI 2.1, 2x USB-A", True),
    ("Gigabyte", "Aorus 16X", ["RTX 5070", "RTX 5070 Ti", "RTX 5080"], 16.0, "2560x1600 (QHD+)", 2.3, 84, 5, 82, True, "2x Type-C, 1x HDMI 2.1, 2x USB-A", True),
    ("Apple", "MacBook Pro 16", ["M4 Pro", "M4 Max", "M5 Pro", "M5 Max"], 16.2, "3456x2234 (Liquid Retina XDR)", 2.15, 88, 20, 100, False, "3x Thunderbolt 5, 1x HDMI 2.1, SDXC", True),
    ("Apple", "MacBook Pro 14", ["M4 Pro", "M4 Max", "M5 Pro", "M5 Max"], 14.2, "3024x1964 (Liquid Retina XDR)", 1.6, 80, 18, 100, False, "3x Thunderbolt 5, 1x HDMI 2.1, SDXC", True),
    ("Apple", "MacBook Air 15", ["M4", "M5"], 15.3, "2880x1864 (Liquid Retina)", 1.51, 55, 20, 95, False, "2x Thunderbolt / USB 4, MagSafe 3", True),
    ("Samsung", "Galaxy Book5 Ultra", ["RTX 5050", "RTX 5070"], 16.0, "2880x1800 (3K AMOLED)", 1.86, 75, 12, 98, False, "2x Thunderbolt 4, 1x HDMI 2.1, 1x USB-A", True),
    ("Samsung", "Galaxy Book5 Pro 360", ["Arc Graphics"], 16.0, "2880x1800 (3K AMOLED)", 1.69, 60, 18, 98, False, "2x Thunderbolt 4, 1x USB-A, 1x HDMI", True),
    ("ASUS", "Zenbook S16", ["Radeon 890M"], 16.0, "2880x1800 (3K OLED)", 1.5, 65, 15, 95, False, "2x USB4, 1x USB-A 3.2, 1x HDMI 2.1", True)
]

# ─────────────────────────────────────────────────────────────────────────────
# 4. VERİ ÜRETİM VE DENGELEME ALGORİTMASI (DATA GENERATION)
# ─────────────────────────────────────────────────────────────────────────────

def generate_laptops():
    """
    Şablon nesnelerini okuyarak, donanım seviyelerine göre RAM-SSD kombinasyonları
    oluşturan ve her kombinasyon için gerçekçi test skorları türeten ana fonksiyon.
    """
    data = [] # Üretilecek ham veri sözlüklerini tutacak olan ana liste.
    
    for t in TEMPLATES:
        # Şablon tuple'ının elemanlarını anlamlı yerel değişkenlere ayırıyoruz (Unpacking).
        marka, model_base, gpus, ekran, coz, agirlik, sogutma, pil, panel, is_gaming, port_detay, npu = t
        
        # Her şablonun desteklediği her bir grafik kartı için ayrı kombinasyonlar üretiyoruz.
        for gpu in gpus:
            
            # Cihazın taşıdığı GPU gücüne göre mantıklı RAM ve SSD seçeneklerini (tuple listesi olarak) belirliyoruz:
            if gpu in ["RTX 5090", "M5 Max", "M4 Max"]:
                # Amiral gemisi GPU'lar için yüksek kapasiteler (Örn: 64GB RAM / 2TB SSD, 128GB RAM / 4TB SSD)
                configs = [(64, 2000), (128, 4000), (64, 4000)]
            elif gpu in ["RTX 5080"]:
                configs = [(32, 1000), (32, 2000), (64, 2000), (64, 4000)]
            elif gpu in ["RTX 5070", "RTX 5070 Ti", "M5 Pro", "M4 Pro"]:
                configs = [(16, 512), (16, 1000), (32, 1000), (32, 2000), (48, 1000), (64, 2000)]
            elif gpu in ["RTX 5060", "RTX 5050"]:
                # Orta segment GPU'lar için standart kapasiteler
                configs = [(16, 512), (16, 1000), (32, 1000)]
            else:
                # Giriş seviyesi veya paylaşımlı (dahili) ekran kartları için daha düşük seçenekler
                configs = [(16, 512), (24, 1000), (32, 1000)]
                
            # Belirlenen RAM ve SSD konfigürasyonları üzerinde dönüyoruz:
            for ram, ssd in configs:
                # Yeni nesil bileşenleri barındıran (RTX 5000 serisi veya Apple M5) cihazlar için model yılını 2026,
                # diğerleri için 2025 yılı olarak atıyoruz.
                yil = 2026 if ("50" in gpu or "M5" in gpu) else 2025
                
                # Gerçekçi Watt (TDP) ve performans saçılımı sağlamak için rastgele sapma (variance) çarpanı türetiyoruz:
                if "Apple" in marka:
                    # Apple çiplerinin güç profilleri ve performans verimlilikleri son derece sabittir.
                    # Bu nedenle grafiklerde bir "Nokta" kümesi oluşturmaları için çok dar bir aralık (%2) seçilmiştir.
                    variance = random.uniform(0.98, 1.02)
                else:
                    # Intel, AMD ve NVIDIA donanımları, üreticinin kasadaki güç limitlerine göre geniş güç aralıklarında yapılandırılır.
                    # Bu nedenle grafiklerde anlamlı bir "Çizgisel" dağılım sağlamak için daha geniş bir sapma (%25) eklenir.
                    variance = random.uniform(0.75, 1.25)
                    
                # Belirlenen sapma değerine göre gerçekçi TDP (Watt) değeri hesaplanır.
                actual_tdp = int(GPU_TDP[gpu] * variance)
                
                # CPU performans puanı, ekran kartı performansına paralel olacak şekilde normalize edilerek (50-100 arasına) çekilir.
                cpu_skor = min(100, max(50, int(GPU_SCORE[gpu] * variance)))
                
                # Tekli ve Çoklu Çekirdek Puanları (Geekbench & Cinebench tarzında) simüle ediliyor.
                # Grafiklerde dikey yöndeki yapay bozulmaları önlemek adına dar rastgele katsayı aralıkları seçilmiştir.
                cpu_single = int(cpu_skor * random.uniform(28, 29))
                cpu_multi = int(cpu_skor * random.uniform(165, 170))
                
                # RAM hızını (MHz), RAM boyutu ve marka tipine göre belirliyoruz (Apple LPDDR5/Unified Memory hız avantajları gözetilmiştir).
                ram_mhz = 8000 if (ram >= 64 and "Apple" in marka) else (6400 if ram >= 64 else (7467 if (ram >= 32 and "Apple" in marka) else (6000 if ram >= 32 else 5600)))
                
                # SSD okuma/yazma hızını (MB/s) kapasiteye göre simüle ediyoruz. Büyük kapasiteli diskler daha hızlıdır.
                # Apple cihazlarında kullanılan NVMe mimarisine ek 500 MB/s hız bonusu veriyoruz.
                ssd_hiz = (7500 if ssd >= 2000 else 6000) + (500 if "Apple" in marka else 0)
                
                # Sahip olunan port sayısını, virgülle ayrılmış detay metnindeki kelime sayısını analiz ederek hesaplıyoruz.
                port_count = len(port_detay.split(",")) + 1
                
                # Güç gereksinimi yüksek oyuncu bilgisayarlarının şarj adaptörleri ağırdır (1.1kg veya 0.8kg).
                # Günlük kullanım ve ofis bilgisayarlarının adaptörleri ise hafiftir (0.2kg).
                adaptor = 1.1 if gpu in ["RTX 5090", "RTX 5080"] else (0.8 if is_gaming else 0.2)
                
                # Markaların teknik servis, malzeme kalitesi ve marka prestij katsayı puanı (10 üzerinden).
                marka_skoru = 10 if marka in ["Apple", "ASUS", "Lenovo"] else (9 if marka in ["Razer", "Samsung", "Dell"] else 8)
                
                # Oluşturulan tüm nitelikleri bir sözlük yapısında bir araya getirip listemize ekliyoruz.
                data.append({
                    "Marka": marka, 
                    "Model": f"{model_base} ({gpu})", 
                    "Yil": yil, 
                    "CPU_Skoru": cpu_skor,
                    "CPU_Single_Core": cpu_single, 
                    "CPU_Multi_Core": cpu_multi,
                    "GPU_Skoru": GPU_SCORE[gpu],
                    "RAM_GB": ram, 
                    "RAM_MHz": ram_mhz, 
                    "SSD_GB": ssd, 
                    "SSD_Hiz_MBs": ssd_hiz, 
                    "Ekran_Boyutu": ekran,
                    "Ekran_Cozunurluk": coz, 
                    "Laptop_Agirlik_kg": agirlik, 
                    "Adaptor_Agirlik_kg": adaptor, 
                    "Sogutma_Skoru": sogutma,
                    "Pil_Video_Saat": pil, 
                    "Portlar": port_count, 
                    "Port_Detay": port_detay, 
                    "Marka_Skoru": marka_skoru,
                    "Ekran_Karti": gpu, 
                    "VRAM_GB": GPU_VRAM[gpu], 
                    "NPU_Var": npu, 
                    "WiFi_7": yil == 2026,
                    "Panel_Kalitesi": panel, 
                    "Is_Gaming": is_gaming, 
                    "TDP_Watt": actual_tdp
                })
    return data

# generate_laptops() fonksiyonunu çağırarak ham listeyi oluşturuyoruz.
DATA = generate_laptops()

# ─────────────────────────────────────────────────────────────────────────────
# 5. CSV DOSYALAMA AŞAMASI (FILE GENERATION)
# ─────────────────────────────────────────────────────────────────────────────

def create_dataset(output_path: str = "laptop_dataset_2025_2026.csv") -> pd.DataFrame:
    """
    Türetilen ham liste verisini Pandas veri çerçevesine dönüştürür,
    gerçekçilik açısından satırları karıştırır ve diske yazar.
    """
    # Ham Python listesinden Pandas DataFrame'i oluşturuyoruz.
    df = pd.DataFrame(DATA)

    # Veri setini baştan sona rastgele karıştırıyoruz (frac=1 parametresi tüm veriyi karıştırır).
    # "random_state=42" değeri karıştırma şeklinin her zaman aynı kalmasını garanti eder.
    # "reset_index(drop=True)" ile eski indis numaralarını sıfırlayıp 0'dan yeniden başlatıyoruz.
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Excel'in Türkçe karakterleri (ş, ç, ğ, ı, ö, ü) düzgün okuyabilmesi için "utf-8-sig" kodlamasını kullanıyoruz.
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    # Konsola işlemin başarı durumunu ve satır sayısını yazdırıyoruz.
    print(f"Veri kumesi olusturuldu: {output_path} ({len(df)} satir)")
    return df

# Eğer bu betik doğrudan Python yorumlayıcısı ile çalıştırılırsa (modül olarak çağrılmak yerine):
if __name__ == "__main__":
    df = create_dataset()
