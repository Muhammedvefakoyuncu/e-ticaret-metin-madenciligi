# E-Ticaret Sahte Yorum Tespiti: Hibrit Uzman Sistem 

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Model-Random%20Forest-orange)
![NLP](https://img.shields.io/badge/NLP-NLTK%20%7C%20TextBlob-yellow)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)
![TÜBİTAK](https://img.shields.io/badge/Onay-TÜBİTAK-success)

##  Proje Vizyonu
Bu proje, e-ticaret platformlarındaki manipülatif ve sahte ürün yorumlarını tespit etmek amacıyla geliştirilmiş, **TÜBİTAK tarafından onaylanmış** hibrit bir karar motorudur. 

Sistem, geleneksel makine öğrenmesi sınıflandırmalarının ötesine geçerek; makine öğrenmesi (Random Forest) tahminlerini, dinamik bir **Kural Tabanlı Uzman Sistem (Rule-Based Expert System)** ile harmanlar. Yorumun sadece içeriğini değil, "yazılış biçimini" ve "psikolojik motivasyonunu" da analiz eden çok katmanlı bir mimariye sahiptir.

##  Hibrit Mimari ve Karar Motoru (v4.1)

Model, sınıflandırma yaparken şu 4 temel aşamadan geçer:

### 1. Stilometrik ve Duygu Analizi (Öznitelik Mühendisliği)
Ham metin temizlenmeden önce, yorumun yapısal karakteristiği çıkarılır:
* **Flesch Reading Ease (Okunabilirlik Skoru):** `textstat` kütüphanesi ile metnin karmaşıklık seviyesi ölçülür.
* **Duygu Polaritesi (Sentiment Polarity):** `TextBlob` kullanılarak metnin duygusal yönelimi hesaplanır.
* **Görsel-Biçimsel Metrikler:** Büyük harf kullanım oranı, ünlem sayısı ve toplam yorum uzunluğu.

### 2. Doğal Dil İşleme (NLP) ve Vektörizasyon
* Metinler küçük harfe çevrilir, noktalama işaretleri ve rakamlar temizlenir.
* `NLTK` kullanılarak İngilizce stop-word (etkisiz kelimeler) filtrelemesi yapılır.
* Temizlenmiş metinler **TF-IDF (Term Frequency-Inverse Document Frequency)** ile (1, 2) n-gram aralığında vektörize edilir.

### 3. Makine Öğrenmesi Katmanı
* TF-IDF matrisi ile stilometrik sayısal özellikler `scipy.sparse.hstack` kullanılarak tek bir öznitelik uzayında birleştirilir.
* Dengesiz veri setleriyle başa çıkabilmek için `class_weight='balanced'` parametresiyle optimize edilmiş **Random Forest Classifier** eğitilir.

### 4. Canlı Nihai Karar Motoru (Kural Tabanlı Heuristikler)
Makine öğrenmesi tahmini alındıktan sonra, özel olarak geliştirilmiş algoritmik kurallar devreye girerek bir "Risk Puanı" hesaplar. Tespit edilen anomali profilleri:
* **Sinsi Pazarlama (Stealth Marketing):** Spesifik model kodları (örn: X-1000) veya "brand/recommend" gibi kelime kalıpları kullanan yorumlar.
* **Hikaye Anlatanlar (Emotional Storytelling):** Üründen ziyade aile, tatil, hediye gibi duygusal kelimelerle manipülasyon yapanlar.
* **Nötr Botlar (Neutral Bots):** Duygu skoru tam olarak 0.0 olan ancak uzun metinli, rapor benzeri otomatik üretimler.
* **Sahte Negatifler (Fake Negatives):** Rakipleri karalamak amacıyla yazılan kısa ve aşırı negatif yorumlar.

## Veri Seti ve Keşifsel Veri Analizi (EDA)
Projede **Kaggle Fake Reviews Dataset** kullanılmıştır. Model eğitimi öncesinde veri setinin karakteristik özelliklerini anlamak için `Matplotlib` ve `Seaborn` ile detaylı analizler yapılmıştır:
* Gerçek ve Sahte yorumların Duygu Skoru dağılımları.
* Etiket bazlı Yorum Uzunluğu yığılmaları.
* Sınıflara göre Büyük Harf kullanım oranlarının Box-Plot analizi.

## Kurulum ve Çalıştırma

Projeyi yerel ortamınızda test etmek için:

1. Depoyu klonlayın:
   ```bash
   git clone https://github.com/Muhammedvefakoyuncu/e-ticaret-metin-madenciligi.git
2. Gerekli kütüphaneleri kurun (NLTK corpora indirmeleri kodun içinde otomatik yapılmaktadır):
   ```bash
   pip install pandas numpy scikit-learn nltk textblob textstat matplotlib seaborn
   python main.py      
