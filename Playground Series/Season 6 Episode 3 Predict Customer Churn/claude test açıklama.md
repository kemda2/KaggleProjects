# ilk açıklama

Gerçek bir senior analistin kafasındaki karar ağacına göre açıklayayım — her testin **ne zaman devreye girdiğini** ve **ne soruya cevap verdiğini** anlat:

---

## Testlerin Kullanım Mantığı

Bir analist veriyi ilk gördüğünde şu 6 soruyu sırasıyla sorar:

---

### 1️⃣ "Bu veri güvenilir mi?" — Veri Kalitesi Soruları

Bunlar **her zaman, her projede** yapılır. Modele girmeden önce zorunlu.

| Test | Soru | Ne zaman kritik |
|---|---|---|
| Business Logic Validation | "Tenure=0 ama TotalCharges>0 mümkün mü?" | Finansal/telco verisi |
| Duplicate Feature Detection | "İki feature aynı şeyi mi söylüyor?" | Her zaman |
| String Anomaly Detection | "'Male' ve 'male' farklı kategori mi sayılıyor?" | Ham/birleşik veri |
| Near-Constant Feature | "Bu feature neredeyse hiç değişmiyor mu?" | Her zaman |
| Cross-feature Consistency | "Bu kombinasyon domain'de mümkün mü?" | Domain bilgisi gerektiren veri |
| Benford's Law | "Sayılar doğal görünüyor mu?" | Finansal veri, şüpheli dataset |
| Null Pattern Clustering | "Eksiklikler sistematik mi?" | Null oranı > %5 olan dataset |
| MCAR/MAR/MNAR | "Neden eksik bu değer?" | Null olan her feature |

---

### 2️⃣ "Model gerçekten öğreniyor mu?" — Model Sağlığı Soruları

Bunlar **model eğitildikten sonra** yapılır. "AUC iyi ama model saçmalıyor olabilir mi?" sorusu.

| Test | Soru | Ne zaman kritik |
|---|---|---|
| DummyClassifier Baseline | "Model şans üstünde mi?" | İmbalanced veri, düşük AUC |
| Permutation Test (Global) | "Tüm öğrenim şans eseri mi?" | AUC > 0.80 ama şüphelenildiğinde |
| Calibration (Brier) | "Olasılıklar gerçeği yansıtıyor mu?" | Threshold kullanacaksan zorunlu |
| Nested CV | "CV skonum şişirilmiş mi?" | Hyperparameter tuning yapıldıysa |
| Learning Curve | "Daha fazla veri yardımcı olur mu?" | Augmentation kararı verilecekse |
| Seed Stability | "Sonuçlar tekrarlanabilir mi?" | Final model seçiminden önce |

---

### 3️⃣ "Hangi feature'lar gerçekten önemli?" — Feature Önceliklendirme Soruları

Bunlar **FE yapmadan önce** yapılır. "Hangi feature'a zaman harcamalıyım?" sorusu.

| Test | Soru | Ne zaman kritik |
|---|---|---|
| Leakage Detection | "Bu feature production'da olmayacak bilgi mi taşıyor?" | **Her zaman, ilk test** |
| SHAP + Importance | "Model hangi feature'a bakıyor?" | Her zaman |
| Effect Size (Cohen d / Cramér V) | "İstatistiksel değil, pratik olarak önemli mi?" | Büyük dataset (n>10k) |
| MI vs Pearson tutarsızlığı | "Non-linear ilişki var mı?" | Pearson düşük ama feature önemli görünüyorsa |
| KS / Mann-Whitney U | "Churn=Yes ve No farklı mı?" | Numeric feature analizi |
| Kruskal-Wallis | "Bu kategoriler target üzerinde fark yaratıyor mu?" | Kategorik feature analizi |
| VIF | "Bu feature başka bir feature'ın kopyası mı?" | Çok sayıda numeric feature varsa |
| Bonferroni/FDR | "Bu p-value gerçek mi, şans mı?" | 20+ feature test edildiyse |

---

### 4️⃣ "Train ve test aynı dünyada mı?" — Drift Soruları

Bunlar **özellikle Kaggle'da ve production'da** kritik. "Model test'te de çalışacak mı?"

| Test | Soru | Ne zaman kritik |
|---|---|---|
| PSI | "Feature dağılımı kayıyor mu?" | Production monitoring, Kaggle test seti |
| Chi-Square Goodness of Fit | "Kategorik dağılımlar aynı mı?" | Kategorik feature'lar varsa |
| KS Test | "Numeric dağılımlar aynı mı?" | Her numeric feature |
| Adversarial Validation | "Train ve testi ayırt edebiliyor muyuz?" | Kaggle'da neredeyse zorunlu |
| OOF vs Test Dağılımı | "Tahminlerim test'te de geçerli mi?" | Threshold seçimi yapılacaksa |
| OOD Detection | "Test'te hiç görülmemiş değer var mı?" | Her Kaggle yarışması |

---

### 5️⃣ "Model production'da nasıl davranır?" — Business Soruları

Bunlar **model teslim edilmeden önce** yapılır. Kaggle'da puanı artırmaz ama gerçek dünyada olmadan model kabul görmez.

| Test | Soru | Ne zaman kritik |
|---|---|---|
| Threshold Optimizasyonu | "0.5 doğru eşik mi?" | Maliyet asimetrik olduğunda zorunlu |
| Lift & Gain Chart | "Top %10'u hedeflersek ne kadar churner yakalarız?" | Satış ekibine model gösterilecekse |
| Decile Analizi | "Model sıralama kalitesi ne?" | Operasyonel kullanım olacaksa |
| Calibration | "Müşterinin churn ihtimali %70 diyorsam doğru mu?" | Risk skorlaması yapılacaksa |

---

### 6️⃣ "Skoru nasıl artırırım?" — Optimizasyon Soruları

Bunlar **temel pipeline çalıştıktan sonra** yapılır. "Son birkaç puan için."

| Test | Soru | Ne zaman kritik |
|---|---|---|
| Optimal Binning | "qcut mı, IV-optimal bin mı?" | IV odaklı FE yapılıyorsa |
| Encoding Karşılaştırması | "Target encoding gerçekten en iyi mi?" | Kategorik feature'lar varsa |
| Interaction Bootstrap CI | "Bu çarpım feature'ı gerçekten faydalı mı?" | FE'de interaction eklenecekse |
| Post-processing | "Rank transform skor artırıyor mu?" | Submission öncesi |
| Submission Korelasyonu | "Ensemble'a bu modeli eklemeli miyim?" | Birden fazla model varsa |
| Target Encoding Leakage Sim. | "Encoding pipeline doğru mu?" | TE yapıldıysa her zaman |

---

## Şimdi Senin Soruya Doğrudan Cevap

Mevcut pipeline'da bu kategorilerin kapsamı şöyle:

| Kategori | Kaç test var | Kaçı eksik |
|---|---|---|
| 1. Veri kalitesi | ~8/13 | 5 eksik |
| 2. Model sağlığı | ~5/8 | 3 eksik |
| 3. Feature önceliklendirme | ~10/12 | 2 eksik |
| 4. Drift tespiti | ~7/8 | 1 eksik |
| 5. Business | ~5/5 | 0 eksik ✅ |
| 6. Optimizasyon | ~7/9 | 2 eksik |


# Sonradan eklenen

Önce mevcut kodu okuyayım, sonra eksikleri ekleyip tüm testlerin açıklamalarını yazacağım.İyi. Şimdi eksik 13 testi ekleyeceğim ve ardından tüm testlerin açıklama rehberini yazacağım. İki dosya paralel üretelim.54 section, 2701 satır, syntax temiz. Şimdi rehber dökümanını yazalım:Her iki dosya hazır. İşte ne üretildi:

---

**`complete_extended_pipeline.py`** — 2701 satır, 54 section (26–81):

Eklenen 13 yeni section:

| Section | Test |
|---|---|
| 64 | **Adversarial Validation** — Tüm feature kombinasyonunun train/test shift'i |
| 65 | **Dummy Classifier Baseline** — Model şans üstünde mi? |
| 66 | **Permutation Test** — Global anlamlılık, p-value |
| 67 | **VIF** — Multicollinearity tespiti |
| 68 | **MI vs Pearson** — Non-linear ilişki haritası |
| 69 | **Spearman vs Pearson** — Monoton non-linear tespiti |
| 70 | **Duplicate Feature Detection** |
| 71 | **Near-Constant Feature** (eşik %95'e düşürüldü) |
| 72 | **String Anomaly** — Yazım tutarsızlığı |
| 73 | **Cross-Feature Consistency** — Domain kuralları |
| 74 | **Stratification Quality** — Fold bazlı target rate |
| 75 | **Seed Stability** — 5 seed × 5-fold |
| 76–78 | **SHAP Dependence + Stability + Hard Sample Force** |
| 79 | **Benford's Law** |
| 80 | **McNemar's Test** — Model karşılaştırma |
| 81 | **Master Score** (/19) |

---

**`test_rationale_guide.md`** — 699 satır:

Her testin neden var olduğu, hangi soruyu yanıtladığı, ne zaman kritik olduğu ve sonuca göre ne yapılacağı — 6 kategoride tam açıklama. Sonunda "hangi testi ne zaman atlayabilirim" tablosu ve tüm pipeline akış şeması.