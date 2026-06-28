# -----------------------------------------------------------------------------
# Proje: Sahte Yorumların Tespiti (NİHAİ KARAR VEREN UZMAN MODEL)
# Sürüm: 4.1 (EDA Grafikleri, temizlemeden önceye alındı)
# -----------------------------------------------------------------------------

import pandas as pd
import re
import nltk
import ssl
import numpy as np
import textstat
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack
from textblob import TextBlob
import matplotlib.pyplot as plt # Grafik için eklendi
import seaborn as sns          # Grafik için eklendi

print("Nihai Karar Veren Uzman Model projesi başlatıldı. (Sürüm: 4.1 - EDA + Uzman Sistem)")

# --- ÖN HAZIRLIKLAR ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
from nltk.corpus import stopwords

# --- ADIM 1: Veri Setini Yükleme ---
try: 
    df = pd.read_csv('fake_reviews_dataset.csv')
    print("Veri seti yüklendi")
except FileNotFoundError:
    print("\n!!! HATA: 'fake_reviews_dataset.csv' dosyası bulunamadı! !!!")
    exit()

# --- ADIM 2: ÖZNİTELİK MÜHENDİSLİĞİ (BÖLÜM 1: Ham Veri Analizi) ---
# YORUM: Metin temizleme (küçük harfe çevirme) işleminden ÖNCE ham veri
# üzerinden hesaplanması gereken tüm metrikler buraya taşındı.
print("\n[ADIM 2] Öznitelik Mühendisliği (Bölüm 1: Ham Veri Analizi) yapılıyor...")
df['label_numeric'] = df['label'].apply(lambda x: 1 if x == 'CG' else 0)

# Orijinal (ham) metni bir değişkene al
original_text = df['text_'].astype(str)

# Ham metin üzerinden metrikleri hesapla
df['yorum_uzunlugu'] = original_text.apply(len)
df['duygu_skoru'] = original_text.apply(lambda x: TextBlob(x).sentiment.polarity)
df['buyuk_harf_orani'] = original_text.apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))
df['unlem_sayisi'] = original_text.str.count('!')
df['okunabilirlik_skoru'] = original_text.apply(textstat.flesch_reading_ease)
print(" -> Ham metin öznitelikleri (uzunluk, duygu, büyük harf) hesaplandı.")


# --- YENİ ADIM: 2. Hafta Sunumu için Görsel Kanıtlar (EDA) ---
# YORUM: Bu bölümün tamamı, metin temizleme adımından ÖNCEYE TAŞINDI.
# Böylece grafikler, küçük harfe dönüştürülmemiş ham veriyi yansıtır.
print("\n[ANALİZ] 2. Hafta sunumu için Görsel Kanıtlar (Grafikler) oluşturuluyor...")
print("Lütfen grafikleri inceleyin ve pencereleri kapatarak devam edin...")

# 1. GRAFİK: Duygu Skoru Dağılımı
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='duygu_skoru', hue='label', 
             multiple='stack', bins=30, palette={'CG': 'red', 'OR': 'blue'})
plt.title('Duygu Skorlarının Sahte (CG) ve Gerçek (OR) Yorumlara Göre Dağılımı', fontsize=16)
plt.xlabel('Duygu Skoru (Negatif <-> Pozitif)', fontsize=12)
plt.ylabel('Yorum Sayısı', fontsize=12)
plt.legend(title='Etiket', labels=['Gerçek (OR)', 'Sahte (CG)'])
plt.show()

# 2. GRAFİK: Yorum Uzunluğu Dağılımı
plt.figure(figsize=(10, 6))
df_short = df[df['yorum_uzunlugu'] < 600] # Grafiği okunaklı kılmak için filtrele
sns.histplot(data=df_short, x='yorum_uzunlugu', hue='label', 
             multiple='stack', bins=50, palette={'CG': 'red', 'OR': 'blue'})
plt.title('Yorum Uzunluklarının Dağılımı (600 Karakterden Kısa)', fontsize=16)
plt.xlabel('Yorum Uzunluğu (Karakter Sayısı)', fontsize=12)
plt.ylabel('Yorum Sayısı', fontsize=12)
plt.legend(title='Etiket', labels=['Gerçek (OR)', 'Sahte (CG)'])
plt.show()

# 3. GRAFİK: Büyük Harf Oranı (Box Plot)
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='label', y='buyuk_harf_orani', 
             palette={'CG': 'red', 'OR': 'blue'})
plt.title('Büyük Harf Oranının Dağılımı (Sahte vs Gerçek)', fontsize=16)
plt.xlabel('Yorum Etiketi', fontsize=12)
plt.ylabel('Büyük Harf Oranı ', fontsize=12)
plt.xticks(ticks=[0, 1], labels=['Gerçek (OR)', 'Sahte (CG)'])
plt.show()

print(" -> Grafikler gösterildi. Model eğitimine devam ediliyor...")
# --- (Grafik bölümü sonu) ---


# --- ADIM 2: ÖZNİTELİK MÜHENDİSLİĞİ (BÖLÜM 2: Metin Temizleme) ---
# YORUM: Grafik adımı tamamlandıktan sonra, şimdi ML modeli için
# metni temizleyip küçük harfe çeviriyoruz.
print("\n[ADIM 2 Devam] Metin temizleme (NLP için) yapılıyor...")
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower() # <-- BÜYÜK HARFLER BURADA KÜÇÜLTÜLÜYOR
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Temizlenmiş metin sütununu (modelin kullanacağı) oluştur
df['cleaned_text'] = df['text_'].apply(clean_text)
print(" -> 'cleaned_text' sütunu (NLP için) başarıyla oluşturuldu.")


# --- ADIM 3 & 4: Veriyi Ayırma ve Birleştirme ---
# YORUM: Buradan sonrası değişmedi. X_text, temizlenmiş metni kullanıyor.
X_text = df['cleaned_text'] 
X_features = df[['yorum_uzunlugu', 'duygu_skoru', 'buyuk_harf_orani', 'unlem_sayisi', 'okunabilirlik_skoru']]
y = df['label_numeric']
X_train_text, X_test_text, X_train_features, X_test_features, y_train, y_test = train_test_split(
    X_text, X_features, y, test_size=0.20, random_state=42, stratify=y)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)
X_train_features_np = X_train_features.to_numpy()
X_test_features_np = X_test_features.to_numpy()
X_train_combined = hstack([X_train_tfidf, X_train_features_np])
X_test_combined = hstack([X_test_tfidf, X_test_features_np])

# --- ADIM 5: Modeli Eğitme ---
print("\n[ADIM 5] Uzman model eğitiliyor...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=80, random_state=42, n_jobs=-1, class_weight='balanced')
rf_model.fit(X_train_combined, y_train)
print(" -> Model başarıyla eğitildi.")

# --- ADIM 6: Performans Değerlendirme (2. Hafta Raporu) ---
y_pred_rf = rf_model.predict(X_test_combined)
print("\n--- [ADIM 6] Model Performansı (2. Hafta Raporu) ---")
print(f"Genel Doğruluk (Accuracy): {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf, target_names=['Gerçek (OR)', 'Sahte (CG)']))

# --- ADIM 7: NİHAİ KARAR MOTORU İLE CANLI TAHMİN (3. Hafta Raporu) ---
print("\n" + "="*60)
print(" " * 12 + "NİHAİ KARAR MOTORU (CANLI TEST v3.3)")
print("="*60)

while True:
    user_input = input("\nLütfen bir yorum girin (çıkmak için 'çıkış'): ")
    if user_input.lower() == 'çıkış':
        break

    # 1. Girilen yorum için TÜM özellikleri hesapla
    # YORUM: Canlı test bölümü, ham metin ve temizlenmiş metni
    # zaten ayrı ayrı ele aldığı için burada bir değişiklik gerekmiyor.
    cleaned_input = clean_text(user_input) # ML için temizle
    
    # Ham metin üzerinden kuralları hesapla
    yorum_uzunlugu_input = len(user_input)
    duygu_skoru_input = TextBlob(user_input).sentiment.polarity
    buyuk_harf_orani_input = sum(1 for c in user_input if c.isupper()) / (len(user_input) + 1)
    unlem_sayisi_input = user_input.count('!')
    okunabilirlik_skoru_input = textstat.flesch_reading_ease(user_input)
    kelime_sayisi = len(user_input.split())

    # 2. ML için veriyi hazırla ve ML tahminini al
    vectorized_text = vectorizer.transform([cleaned_input]) # Temizlenmiş veriyi kullan
    input_features = np.array([[yorum_uzunlugu_input, duygu_skoru_input, buyuk_harf_orani_input, unlem_sayisi_input, okunabilirlik_skoru_input]])
    combined_input = hstack([vectorized_text, input_features])
    ml_prediction_is_cg = (rf_model.predict(combined_input)[0] == 1)

    # 3. Kural Tabanlı Risk Puanı Hesaplama (GÜNCELLENMİŞ KURALLAR v3.3)
    risk_puanı = 0
    analiz_notlari = []
    
    # Kural 0: Temel Spam
    if kelime_sayisi <= 7 and duygu_skoru_input > 0.7:
        risk_puanı += 35
        analiz_notlari.append("Risk (Kural 0): Yorum çok kısa ve genel bir övgü içeriyor.")
        
    if buyuk_harf_orani_input > 0.3 or unlem_sayisi_input >= 3:
        risk_puanı += 30
        analiz_notlari.append("Risk (Kural 0): Abartılı yazı stili (BÜYÜK HARF / !!!) kullanılmış.")

    # Kural 1: "Sinsi Pazarlama" tespiti (brand/recommend)
    if "brand" in user_input.lower():
        risk_puanı += 20
        analiz_notlari.append("Risk (Kural 1): Yorum 'marka' odaklı (pazarlama dili).")
    if "recommend" in user_input.lower():
        risk_puanı += 20
        analiz_notlari.append("Risk (Kural 1): Yorum 'tavsiye' kelimesi içeriyor (pazarlama dili).")

    # Kural 2: "Sinsi Pazarlama" tespiti (Model Kodu)
    if re.search(r'\b[A-Za-z]+[-_][0-9]{3,}\b', user_input) or re.search(r'\b[A-Z]{2,}[0-9]{3,}\b', user_input):
        risk_puanı += 35
        analiz_notlari.append("Risk (Kural 2): Yorum, 'X-1000' gibi spesifik Model Kodu içeriyor.")

    # Kural 3: "Sahte Negatif" tespiti
    if duygu_skoru_input < -0.1 and kelime_sayisi < 50:
        risk_puanı += 40
        analiz_notlari.append("Risk (Kural 3): Yorum negatif ve kısa/orta uzunlukta (karalama olabilir).")
        
    # Kural 4: "Hikaye Anlatan" pazarlama tespiti (Duygu Bağımsız)
    story_keywords = ['vacation', 'trip', 'family', 'graduation', 'daughter', 'husband', 
                      'gift', 'rome', 'italy', 'job', 'boss', 'colleagues', 'meetings']
    if any(keyword in user_input.lower() for keyword in story_keywords):
        risk_puanı += 25
        analiz_notlari.append("Risk (Kural 4): Yorum, üründen çok 'duygusal bir hikaye' anlatıyor (Duygudan bağımsız).")

    # Kural 5: "Klişe Pazarlama" tespiti
    cliche_keywords = ['game-changer', 'must-have', 'top-notch', 'life-changing']
    if any(keyword in user_input.lower() for keyword in cliche_keywords):
        risk_puanı += 25
        analiz_notlari.append("Risk (Kural 5): Yorum, 'game-changer' gibi klişe pazarlama dili kullanıyor.")

    # Kural 6: "Nötr Bot" tespiti
    if duygu_skoru_input == 0.0 and kelime_sayisi > 15:
        risk_puanı += 30
        analiz_notlari.append("Risk (Kural 6): Yorum duygusuz ve bir rapor gibi (Nötr Bot tespiti).")

    # --- Güvenilirlik (Negatif Risk Puanı) Kuralı ---
    if yorum_uzunlugu_input > 200 and -0.1 < duygu_skoru_input < 0.7:
        risk_puanı -= 40
        analiz_notlari.append("Güvenilirlik: Yorum, dengeli ve detaylı bir kullanıcı deneyimi sunuyor.")
        
    if risk_puanı < 0: risk_puanı = 0 # Puan negatif olamaz

    # 4. Nihai Risk Seviyesi Belirleme
    risk_seviyesi = "DÜŞÜK"
    if risk_puanı >= 50 or (ml_prediction_is_cg and risk_puanı > 20):
        risk_seviyesi = "YÜKSEK"
    elif risk_puanı >= 25:
        risk_seviyesi = "ORTA"
    
    # Raporu Göster
    print("\n" + ("-" * 15) + " ANALİZ RAPORU " + ("-" * 15))
    print(f"| Girdiğiniz Yorum: '{user_input}'")
    print("-" * 48)
    print(f"| ML Model Tipi Tespiti : '{'BİLGİSAYAR ÜRETİMİ' if ml_prediction_is_cg else 'İNSAN YAZIMI'}'")
    print(f"| Kural Tabanlı Risk Puanı: {risk_puanı}")
    if analiz_notlari:
        for not_ in analiz_notlari:
            print(f"|   -> {not_}")
    print(f"| Analiz Sonucu Risk Seviyesi: {risk_seviyesi}")
    print("-" * 48)

    # Nihai Karar Bölümü
    if risk_seviyesi == "YÜKSEK" or risk_seviyesi == "ORTA":
        print(" > SONUÇ: Bu yorum, analizler sonucunda SAHTE olarak sınıflandırılmıştır.")
    else:
        print(" > SONUÇ: Bu yorum, analizler sonucunda GERÇEK olarak sınıflandırılmıştır.")
    print("-" * 48)

print("\nNihai Karar Motoru kapatıldı.")