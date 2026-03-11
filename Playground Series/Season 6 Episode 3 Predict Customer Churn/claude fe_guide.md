# 🚀 Senior Data Pipeline — FE Öncesi Analiz Rehberi

**Testler · Kurallar · Aksiyonlar · FE Hazırlık Kılavuzu**

Bu döküman **63 bölümlük pipeline'ın** (Section 1–63) çıktılarını **eğer X ise → Y yap** formatında özetler. FE'ye geçmeden önce her section'ı sırasıyla uygulayın.

---

## İçindekiler

1. [Temel Prensipler](#1-temel-prensipler)
2. [Section Bazlı Karar Kuralları (S1–S24)](#2-section-bazlı-karar-kuralları)
3. [Ek Analizler Section 26–34](#3-ek-analizler--section-2634-karar-kuralları)
4. [Yeni Analizler Section 35–63](#4-yeni-analizler--section-3563)
5. [Sırayla Uygulanacak FE Reçetesi](#5-sırayla-uygulanacak-fe-reçetesi)
6. [Master Hızlı Karar Referans Tablosu](#6-master-hızlı-karar-referans-tablosu)
7. [Telco Churn Dataset — Özel Notlar](#7-telco-churn-dataset--özel-notlar)

---

## 1. Temel Prensipler

> **Altın Kural:** Feature Engineering'e geçmeden önce aşağıdaki 9 kategori temizlenmeden hiçbir model güvenilir değildir.

| # | Kural |
|---|-------|
| **KURAL 1** | Leak varsa FE yapma. Önce kaldır. AUC > 0.95 solo feature → her zaman şüpheli. |
| **KURAL 2** | Business logic anomalilerini DROP etme, FLAG ekle. Model bu bilgiyi kullanabilir. |
| **KURAL 3** | Sentinel değerleri (`No internet service`, `No phone service`) her zaman `"No"` ile değiştir. |
| **KURAL 4** | Test'te OOD değer varsa → Winsorize (numeric) veya Unknown token (kategorik). |
| **KURAL 5** | Hard sample > %15 ise agresif FE gerekli. Hard sample < %8 ise temel FE yeterli. |
| **KURAL 6** | SHAP interaksiyon çifti varsa → o ikili için çarpım feature oluştur. |
| **KURAL 7** | CV std > 0.005 ise FE'ye geçme, önce CV stratejisini düzelt. |
| **KURAL 8** | Label noise > %10 ise `sample_weight` veya `label_smoothing` şart. |
| **KURAL 9** | Her FE adımından sonra baseline AUC ile karşılaştır. Düşerse geri al. |
| **KURAL 10** | Calibration bozuksa (Brier > 0.20) threshold analizi güvenilir değil. Önce kalibre et. |
| **KURAL 11** | p-value tek başına yeterli değil. Her zaman effect size (Cohen d / Cramér V) ile birlikte değerlendir. |
| **KURAL 12** | Outlier tipi belirlemeden Winsorize uygulama. Hata mı, uç değer mi, segment mi? |
| **KURAL 13** | Nested CV ile optimism bias'ı ölç. Standard CV her zaman şişirilmiş skor verir. |
| **KURAL 14** | OOF ve test dağılımları farklıysa threshold güvenilir değil. KS testi yap. |
| **KURAL 15** | Encoding stratejisini karşılaştır. Varsayılan label encoding her zaman en iyi değildir. |

---

## 2. Section Bazlı Karar Kuralları

### Section 1–2: Memory, Shape & Genel Audit

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Train > 500MB | dtypes'ı downcast et (float32, int32). `pd.read_csv` ile `dtype=dict` kullan. | 🟡 ORTA |
| Test'te train'de olmayan kolon var | Bu kolonu test'ten de kaldır. Feature alignment yap. | 🔴 KRİTİK |
| Null Pct > %30 (herhangi bir kolon) | `is_null` flag ekle, sonra impute et. Doğrudan drop etme. | 🟡 ORTA |
| Null Gap (test - train) > 5 puan | Bu kolon drift'li. Adversarial test section'ına bak. | 🟡 ORTA |
| Top Val Pct > %99.5 (bir kolonda) | CONSTANT kolon → kesinlikle drop et. | 🔴 KRİTİK |
| Top Val Pct %95–99.5 arası | QUASI constant → önce FE dene, sonra karar ver. | 🟢 DÜŞÜK |

---

### Section 3: Target Distribution

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| İkili sınıf, Ratio < 0.2 (ciddi dengesizlik) | `is_unbalance=True` (LightGBM) + `scale_pos_weight` hesapla. SMOTE son çare. | 🔴 KRİTİK |
| İkili sınıf, Ratio 0.2–0.5 | `class_weight="balanced"` veya `scale_pos_weight`. | 🟡 ORTA |
| Ratio > 0.5 | Dengeli. Özel işlem gerekmez. | 🟢 DÜŞÜK |
| Regresyon, \|skew\| > 1 | Target'a `log1p` uygula. Tahminleri `expm1` ile geri dönüştür. | 🟡 ORTA |
| Regresyon, target'ta negatif değer var | `log1p` yerine Box-Cox veya signed log. | 🟡 ORTA |

---

### Section 4: Duplicates & Constants

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Duplicate satır > 0 | `drop_duplicates(keep="first")` — ID olmayan kolonlarda. | 🟡 ORTA |
| Constant kolon tespit edildi | Model eğitiminden önce kesinlikle drop et. FE'de kullanma. | 🔴 KRİTİK |
| Quasi constant (>%95) ama IV > 0.1 | Drop etme, binary flag'e çevir: `is_{col}_dominant`. | 🟡 ORTA |

---

### Section 5: Statistical Risk & Drift (KS Test)

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| KS p >= 0.05 → "NONE" | Drift yok. Normal kullan. | 🟢 DÜŞÜK |
| KS < 0.05 ama KS istatistiği < 0.05 → "Statistical Only" | İstatistiksel ama pratik etki yok. Göz ardı edebilirsin. | 🟢 DÜŞÜK |
| "Mild Drift" (KS 0.05–0.10) | Bu feature'ı normalize et (`RobustScaler` veya `QuantileTransformer`). | 🟡 ORTA |
| "Severe Drift" (KS > 0.10) | Adversarial feature olarak değerlendir. Train'den çıkarmayı dene. | 🔴 KRİTİK |
| VIF > 10 (bir kolon için) | Multicollinearity var. Yüksek VIF'li kolonu drop et veya PCA uygula. | 🟡 ORTA |
| Skew > 2 (bir kolon için) | `log1p` veya `sqrt` transform uygula. | 🟡 ORTA |
| Outlier% > 15 | Winsorize (1st–99th percentile clip) uygula. | 🟡 ORTA |

---

### Section 6–7: Numeric & Categorical Korelasyon

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Pearson \|r\| > 0.85 (iki numeric arasında) | Birini drop et veya farkını/oranını yeni feature yap. | 🟡 ORTA |
| Cramer's V > 0.8 (iki kategorik arasında) | "Redundant" — birini drop et. | 🟡 ORTA |
| Cramer's V 0.5–0.8 | İkisini birleştiren interaction feature oluştur. | 🟢 DÜŞÜK |
| \|r\| ile target > 0.3 (herhangi bir feature) | Güçlü ayrıştırıcı. SHAP ve PDP ile detay incele. | 🟢 DÜŞÜK |
| Cramer's V ile target > 0.3 | Güçlü kategorik. Target encoding için aday. | 🟢 DÜŞÜK |

---

### Section 8: Hierarchical Dependency (Sentinel)

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Sentinel tespit edildi (`"No internet service"` gibi) | Bu değeri `"No"` ile değiştir. Hiyerarşik parent-child ilişkisini kır. | 🔴 KRİTİK |
| Sentinel olan feature'ın churn rate'i diğerlerinden > 10 puan farklı | Binary flag ekle: `has_internet_service` (0/1). | 🟡 ORTA |

---

### Section 10: Categorical Overlap

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Test'te yeni kategori değeri var (Overlap < 100%) | Target encoding veya `"Unknown"` token ekle. Label encoding kullanma. | 🔴 KRİTİK |
| Only_Test sayısı > 100 | Bu kategori istatistiksel anlamlı. Ayrı bir `"Other"` grubu oluştur. | 🟡 ORTA |
| Only_Test sayısı < 10 | `"Unknown"` ile birleştir. | 🟢 DÜŞÜK |

---

### Section 12: Information Value (IV)

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| IV > 0.5 (şüpheli) | Leak kontrolü yap (Section 26). Bu feature'ın production'da var olup olmadığını kontrol et. | 🔴 KRİTİK |
| IV 0.3–0.5 (güçlü) | Feature mükemmel. Interaction ve binning için öncelikli aday. | 🟢 DÜŞÜK |
| IV 0.1–0.3 (orta) | Kullan. Target encoding veya binning ile güçlendir. | 🟢 DÜŞÜK |
| IV 0.02–0.1 (zayıf) | Diğer feature'larla interaction yaratmaya çalış. | 🟢 DÜŞÜK |
| IV < 0.02 (işe yaramaz) | Drop et. Model gürültü katıyor. | 🟡 ORTA |

---

### Section 13 & 18: Adversarial AUC

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Adversarial AUC < 0.52 | Drift yok. Tüm feature'ları kullan. | 🟢 DÜŞÜK |
| Adversarial AUC 0.52–0.60 | Minimal drift. Sadece top adversarial feature'ları izle. | 🟢 DÜŞÜK |
| Adversarial AUC 0.60–0.70 | Top 3 adversarial feature'ı normalize et veya drop et. KS sonuçlarıyla karşılaştır. | 🟡 ORTA |
| Adversarial AUC > 0.70 | **BÜYÜK SORUN.** Sızıntı veya ciddi drift. Pipeline'ı durdur, veriyi incele. | 🔴 KRİTİK |
| Tek feature Adv Imp > 0.10 | O feature drift'li. Dağılımını plot et. Normalize veya drop. | 🟡 ORTA |

---

### Section 16: Baseline Model Skoru

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Baseline AUC > 0.90 | Çok güçlü. FE'den beklenti: +0.001–0.005. Ensemble ve hyperparameter tuning'e odaklan. | 🟢 DÜŞÜK |
| Baseline AUC 0.85–0.90 | İyi. Top feature interaksiyon FE'si +0.005–0.015 kazandırabilir. | 🟢 DÜŞÜK |
| Baseline AUC 0.80–0.85 | Orta. Agresif FE, encoding, augmentation gerekli. | 🟡 ORTA |
| Baseline AUC < 0.80 | Zayıf. Encoding stratejisini değiştir. Stacking veya target encoding dene. | 🔴 KRİTİK |

---

### Section 17: CV Stabilitesi

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Seed std < 0.001 | Çok kararlı. 5-fold yeterli. CV'ye güven. | 🟢 DÜŞÜK |
| Seed std 0.001–0.003 | Makul. Sabit seed kullan. CV güvenilir. | 🟢 DÜŞÜK |
| Seed std 0.003–0.005 | Orta varyans. `RepeatedStratifiedKFold(3×5)` kullan. | 🟡 ORTA |
| Seed std > 0.005 | **KARARLI DEĞİL.** 10-fold veya repeated CV şart. Leak veya veri problemi olabilir. | 🔴 KRİTİK |
| Outlier fold sayısı > 0 | O fold'daki satırların dağılımını incele. Veri spliti problemi olabilir. | 🟡 ORTA |

---

### Section 19: Permutation Importance

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| LGB ve Perm importance UYUŞUYOR (R_Diff ≤ 5) | Built-in importance güvenilir. Her ikisini de kullanabilirsin. | 🟢 DÜŞÜK |
| R_Diff > 5 (1–3 feature) | Bu feature'lar için permutation importance'a göre karar ver, built-in'e değil. | 🟡 ORTA |
| R_Diff > 5 (3+ feature) | Tüm kararlar için permutation importance kullan. Built-in yanıltıcı. | 🟡 ORTA |
| Perm Mean < 0 (negatif) | **Bu feature modele ZARAR veriyor. KALDIR.** | 🔴 KRİTİK |
| Perm Mean ≈ 0 (< 0.0001) | Sıfıra yakın. Kaldır, AUC artarsa kalıcı olarak sil. | 🟡 ORTA |

---

### Section 20: Segment Analizi

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Segment churn spread > 40 puan | GÜÇLÜ sinyal. Top segment'ler için binary risk flag oluştur. | 🔴 KRİTİK |
| Segment churn spread 20–40 puan | Orta sinyal. Segment × feature interaction'ları dene. | 🟡 ORTA |
| Churn rate > %50 olan segment var | Bu segment için özel bir `"high_risk"` binary flag ekle. | 🟡 ORTA |
| Churn rate < %5 olan segment var | `"low_risk"` binary flag ekle. | 🟡 ORTA |

---

### Section 21: Feature Stability

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| CV_coef > 1.0 (çok kararsız feature) | Kaldırmayı dene. CV yükselirse kalıcı sil. | 🟡 ORTA |
| CV_coef 0.5–1.0 | İzle. FE sonrası tekrar kontrol et. | 🟢 DÜŞÜK |
| CV_coef < 0.5 (kararlı) | Güvenilir feature. FE kararlarında önceliklendir. | 🟢 DÜŞÜK |

---

### Section 22: Original Data Karşılaştırma

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Original data target rate farkı < 5 puan | Güvenli concat. `pd.concat([train, original])` + source flag. | 🟢 DÜŞÜK |
| Target rate farkı 5–10 puan | Ağırlıklandırarak concat et. Original satırlara `sample_weight` ekle. | 🟡 ORTA |
| Target rate farkı > 10 puan | Dikkatli kullan. Ayrı bir `"is_original"` feature ekle, model farkı öğrensin. | 🟡 ORTA |
| KS test yüksek (numeric feature kayıyor) | Bu feature'ları original data ile uyumlu hale getir veya augment'te kullanma. | 🟡 ORTA |

---

### Section 23: Monotonicity Check

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Spearman ρ > 0.7 (pozitif monoton) | `monotone_constraints = +1` ekle (LightGBM). Model daha stabil olur. | 🟢 DÜŞÜK |
| Spearman ρ < -0.7 (negatif monoton) | `monotone_constraints = -1` ekle. | 🟢 DÜŞÜK |
| Non-monotonic ama güçlü feature (MI yüksek) | Decile binning yap. `pd.qcut` ile 10 gruba böl. | 🟡 ORTA |

---

### Section 24: Post-Cleaning Interaction Analizi

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| n_services Spearman ρ < -0.7 (çok güçlü) | `n_services` feature'ı oluştur. Çok güçlü sinyal. | 🔴 KRİTİK |
| n_services Spearman ρ -0.3 ile -0.7 arası | `n_services` ekle. Orta güç. | 🟡 ORTA |
| Interaction deviation > 15 puan | Bu kombinasyon için binary flag oluştur: `internet_yes_security_no` gibi. | 🟡 ORTA |
| Interaction deviation > 30 puan | ÇOK GÜÇLÜ. Mutlaka flag ekle. Model tek başına yakalayamıyor olabilir. | 🔴 KRİTİK |

---

## 3. Ek Analizler — Section 26–34 Karar Kuralları

### Section 26: Leakage Detection

> **Amaç:** Tek feature ile modelin hedefi neredeyse mükemmel tahmin ettiği durumları tespit et.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Solo AUC > 0.99 (bir feature için) | **KESİN LEAK.** Hiç düşünmeden drop et. Production'da bu bilgi olmaz. | 🔴 KRİTİK |
| Solo AUC 0.95–0.99 | Muhtemelen leak. Domain knowledge ile kontrol et. Test verisinde bu feature var mı? | 🔴 KRİTİK |
| Solo AUC 0.85–0.95 | Şüpheli. Temporal sıralamayı kontrol et. Churn'den sonra mı oluşmuş? | 🟡 ORTA |
| Solo AUC 0.75–0.85 | Güçlü feature. Normal. SHAP ile detay incele. | 🟢 DÜŞÜK |
| Solo AUC < 0.75 | Normal. Kullan. | 🟢 DÜŞÜK |

---

### Section 27: Label Kalitesi & Target Noise

> **Amaç:** Aynı özellik profiline sahip müşterilerin farklı churn label'ı taşıyıp taşımadığını ölç.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Conflict rate < %5 | Label kalitesi yeterli. Normal devam et. | 🟢 DÜŞÜK |
| Conflict rate %5–10 | `label_smoothing=0.05` parametresini LightGBM'e ekle. | 🟡 ORTA |
| Conflict rate > %10 | `sample_weight` hesapla: düşük güven satırlarına düşük ağırlık ver. | 🔴 KRİTİK |
| Birebir duplicate feature ama farklı target var | `drop_duplicates` YAPMA. Bu satırları 0.5 ile ağırlıklandır. | 🟡 ORTA |

---

### Section 28: Business Logic Validation

> **Amaç:** Domain'e özgü mantık ihlallerini tespit et. Telco için: `TotalCharges ≈ tenure × MonthlyCharges`.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| TotalCharges/expected oranı bozuk < %2 | Önemsiz. Devam et. | 🟢 DÜŞÜK |
| Bozuk satır oranı %2–10 | `is_charges_anomaly` binary flag ekle. Churn rate'i normal ile karşılaştır. | 🟡 ORTA |
| Bozuk satır oranı > %10 | `is_charges_anomaly` zorunlu. Bozuk satırlar farklı churn pattern'ı taşıyabilir. | 🔴 KRİTİK |
| tenure=0 ama TotalCharges > 0 | Veri hatası. TotalCharges'ı 0 ile impute et veya anomaly flag ekle. | 🔴 KRİTİK |
| Sentinel değerleri tespit edildi | Tüm sentinel'leri `"No"` ile değiştir. FE adım 1 bu olmalı. | 🔴 KRİTİK |

---

### Section 29: Cohort Drift Analizi

> **Amaç:** tenure bazlı cohort'lar arasındaki churn rate farklarını ölç. Zamana bağlı pattern'ları yakala.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Cohort churn spread < 10 puan | Cohort etkisi zayıf. Lineer tenure yeterli. | 🟢 DÜŞÜK |
| Cohort churn spread 10–20 puan | `tenure_bin` feature ekle: `pd.qcut(tenure, 5)`. | 🟡 ORTA |
| Cohort churn spread > 20 puan | GÜÇLÜ cohort etkisi. `tenure_group` kategorik feature oluştur. `tenure × MonthlyCharges` interaction ekle. | 🔴 KRİTİK |
| Spearman ρ < -0.7 (tenure↑ churn↓) | Ters monoton. `monotone_constraints=-1` ekle tenure için. | 🟡 ORTA |

---

### Section 30: Stratified Target EDA (KDE)

> **Amaç:** Her numeric feature'ın Churn=Yes vs No dağılımlarını karşılaştır. Güçlü ayrıştırıcıları belirle.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| KS istatistiği > 0.3 | ÇOK GÜÇLÜ feature. SHAP ve interaction için öncelikli aday. FE'de mutlaka kullan. | 🟢 DÜŞÜK |
| KS istatistiği 0.15–0.3 | Orta güçlü. Binning veya log transform ile güçlendir. | 🟢 DÜŞÜK |
| KS istatistiği < 0.05 | Zayıf ayrıştırıcı. Drop et veya başka feature'larla birleştir. | 🟡 ORTA |
| Ortalama farkı büyük ama KS düşük | Aykırı değer etkisi var. Winsorize sonra tekrar test et. | 🟡 ORTA |

---

### Section 31: SHAP Değerleri & Interaction

> **Amaç:** Feature'ların model üzerindeki gerçek etkisini ölç. Non-linear pattern'ları ve interaksiyonları tespit et.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| SHAP mean abs > 0.1 (bir feature) | En önemli feature. Tüm interaksiyon kombinasyonlarını dene. | 🟢 DÜŞÜK |
| SHAP scatter'da V-şekli (non-linear) | Bu feature'a binning veya log transform uygula. | 🟡 ORTA |
| SHAP interaksiyon korelasyonu > 0.3 | Bu ikili için çarpım feature oluştur: `col1 × col2`. | 🟡 ORTA |
| SHAP değeri negatif olan satır oranı > %30 | Feature bazı segmentlerde ters etki yapıyor. Segment bazlı impute veya transform dene. | 🟡 ORTA |
| Non-linear Pearson korelasyon < 0.7 | Linear olmayan pattern. `pd.qcut` ile 10–20 bin dene. | 🟡 ORTA |

---

### Section 32: Hard Sample Analizi

> **Amaç:** Modelin 5 fold'un 4+'ında yanlış tahmin ettiği "zor" satırları tespit et ve analiz et.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Hard sample oranı < %8 | Sağlıklı model. Temel FE yeterli. | 🟢 DÜŞÜK |
| Hard sample oranı %8–15 | Target encoding veya stacking bu satırlar için yardımcı olabilir. Segment analizi yap. | 🟡 ORTA |
| Hard sample oranı > %15 | Agresif FE gerekli. Hard sample'ların feature profili incelenmeli. | 🔴 KRİTİK |
| Hard sample ortalama bir feature'da > %20 farklı | O feature için hard sample'lara özel transform veya flag oluştur. | 🟡 ORTA |
| Hard sample'ların çoğu belirli bir segment'te | O segment için ayrı model veya özel FE oluştur (stacking için base layer). | 🟡 ORTA |

---

### Section 33: Test Set OOD Analizi

> **Amaç:** Test verisinde train'de hiç görülmemiş değerleri tespit et. Model bu bölgede extrapolate ediyor.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| OOD yok | Test ve train aynı dağılımda. Güvenli. | 🟢 DÜŞÜK |
| Yeni kategorik değer (az, < 10 satır) | `"Unknown"` token ile birleştir. | 🟡 ORTA |
| Yeni kategorik değer (fazla, > 100 satır) | Target encoding kullan. Label encoding bu durumda yanıltıcı. | 🔴 KRİTİK |
| Numeric değer train range dışında | Winsorize: `clip(lower=train_q01, upper=train_q99)`. | 🟡 ORTA |
| OOD satır oranı > %20 | Test'e özel normalization gerekebilir. `QuantileTransformer(output_distribution="normal")` dene. | 🔴 KRİTİK |

---

## 4. Yeni Analizler — Section 35–63

### Section 35: PSI — Population Stability Index

> **Amaç:** Feature dağılımlarının train→test kaymasını pratik bir metrikle ölç. KS testinden daha yorumlanabilir; production monitoring standardıdır.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| PSI < 0.10 | Stabil. Normal kullan. | 🟢 DÜŞÜK |
| PSI 0.10–0.25 | İzle. `RobustScaler` veya `QuantileTransformer` ile normalize et. | 🟡 ORTA |
| PSI > 0.25 | Ciddi kayma. Drop veya yeniden encode et. Model bu feature'dan güvenilir öğrenemez. | 🔴 KRİTİK |

---

### Section 36: Model Calibration (Brier Score & Calibration Curve)

> **Amaç:** Modelin ürettiği olasılıkların gerçek olasılıkları yansıtıp yansıtmadığını ölç. Kötü kalibrasyon → yanlış threshold seçimi.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Brier Score < 0.10 | İyi kalibre. Threshold analizi güvenilir. | 🟢 DÜŞÜK |
| Brier Score 0.10–0.20 | Kabul edilebilir. Isotonic regression dene. | 🟡 ORTA |
| Brier Score > 0.20 | Kötü kalibrasyon. Platt scaling zorunlu. | 🔴 KRİTİK |
| Calibration curve diyagonalden uzak | `CalibratedClassifierCV(method="isotonic")` uygula. | 🟡 ORTA |

---

### Section 37: Threshold & Cost-Sensitive Analiz

> **Amaç:** AUC optimize etmek ≠ business optimize etmek. False negative (kaçan churner) genellikle false positive'den çok daha pahalıdır.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Optimal threshold 0.5'ten > 0.1 uzakta | `predict_proba >= opt_threshold` kullan. Default 0.5 yanlış. | 🔴 KRİTİK |
| FN/FP maliyet oranı bilinmiyor | Domain uzmanıyla konuş. `fn_cost=5, fp_cost=1` makul başlangıç. | 🔴 KRİTİK |
| Precision-Recall eğrisi düşük | Recall odaklı threshold seç (daha fazla churner yakala). | 🟡 ORTA |
| İki farklı threshold yakın AUC veriyor | Business metriğe göre seç (F1 değil, cost). | 🟡 ORTA |

---

### Section 38: Lift & Gain Chart + Decile Analizi

> **Amaç:** "Top %10'u hedeflersek gerçek churner'ların kaçını yakalıyoruz?" sorusunu yanıtla.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Top decile lift > 2.5x | Mükemmel model. Satış ekibine "top %10'u hedefle" de. | 🟢 DÜŞÜK |
| Top decile lift 1.5–2.5x | İyi model. FE ile lift'i artır. | 🟢 DÜŞÜK |
| Top decile lift < 1.5x | Zayıf. FE gerekli. | 🔴 KRİTİK |
| Decile churn rate monoton değil | Model tutarsız. Threshold veya FE problemi. | 🟡 ORTA |
| Cumulative gain @%30 < %60 | Model ilk 3 decile'da yeterince odaklanmıyor. | 🟡 ORTA |

---

### Section 39: Normalite Testleri (Shapiro-Wilk / D'Agostino)

> **Amaç:** Feature'ların normal dağılıma uyup uymadığını belirle. Encoding, imputation ve istatistiksel test seçimini etkiler.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| p < 0.05 (non-normal) | T-test yerine Mann-Whitney U kullan. | 🟢 DÜŞÜK |
| Non-normal + skew > 1 | `log1p` veya `sqrt` transform uygula. | 🟡 ORTA |
| Non-normal + kurtosis > 5 | Heavy-tail dağılım. Winsorize + transform kombinasyonu. | 🟡 ORTA |
| Tüm feature'lar non-normal | Tüm parametrik varsayımları kaldır. Non-parametrik pipeline kur. | 🟡 ORTA |

---

### Section 40: Levene & Mann-Whitney U (Grup Farklılığı)

> **Amaç:** Churn=Yes ve No grupları arasındaki farkı hem istatistiksel hem pratik olarak ölç.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| MWU p < 0.05 + Cohen d > 0.8 | **Büyük pratik etki.** Bu feature FE'de en yüksek öncelik. | 🔴 KRİTİK |
| MWU p < 0.05 + Cohen d 0.5–0.8 | Orta etki. Interaction ve binning için güçlü aday. | 🟡 ORTA |
| MWU p < 0.05 + Cohen d < 0.2 | İstatistiksel anlamlı ama pratik değil. Büyük veri etkisi. Drop adayı. | 🟡 ORTA |
| Levene p < 0.05 | Gruplar farklı varyanslı. Welch t-test veya Mann-Whitney kullan. | 🟢 DÜŞÜK |

---

### Section 41: Kruskal-Wallis (Kategorik Feature Etkisi)

> **Amaç:** Kategorik feature'ların target üzerindeki istatistiksel anlamlılığını ölç.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| p < 0.05 + Eta² > 0.14 | Büyük etki. Target encoding için en güçlü aday. | 🔴 KRİTİK |
| p < 0.05 + Eta² 0.06–0.14 | Orta etki. Kullan. | 🟡 ORTA |
| p < 0.05 + Eta² < 0.01 | İstatistiksel ama pratik değil. Drop et. | 🟡 ORTA |
| p >= 0.05 | Anlamlı değil. Drop et. | 🟡 ORTA |

---

### Section 42: Çoklu Karşılaştırma Düzeltmesi (Bonferroni / BH-FDR)

> **Amaç:** Yüzlerce p-value testi yapılırken yanlış pozitif şişmesini önle.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Bonferroni sonrası anlamlı | Kesinlikle önemli feature. FE'de öncelikli kullan. | 🟢 DÜŞÜK |
| Sadece BH-FDR'da anlamlı | Dikkatli kullan. Büyük veri etkisi olabilir. Effect size kontrol et. | 🟡 ORTA |
| Düzeltme sonrası hiç anlamlı kalmadı | Feature'lar gerçekten güçsüz. Agresif FE gerekli. | 🔴 KRİTİK |

---

### Section 43: Effect Size Özeti

> **Amaç:** İstatistiksel anlamlılık ≠ pratik anlamlılık. Büyük veri setlerinde küçük farklar da anlamlı çıkar.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Cohen d > 0.8 (numeric) | Büyük etki. FE'de en önce bu feature'ı kullan. | 🔴 KRİTİK |
| Cohen d 0.2–0.5 | Küçük/orta etki. Kullan ama önceliği düşük. | 🟢 DÜŞÜK |
| Cohen d < 0.2 | Pratik etkisi önemsiz. Drop veya combine. | 🟡 ORTA |
| Cramér V > 0.5 (kategorik) | Büyük etki. Target encoding zorunlu. | 🔴 KRİTİK |
| Cramér V < 0.1 | Pratik etkisi yok. Drop adayı. | 🟡 ORTA |

---

### Section 44: Eksik Değer Mekanizması (MCAR / MAR / MNAR)

> **Amaç:** Eksikliğin nedenini belirle. Her mekanizma farklı imputation stratejisi gerektirir.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| **MCAR** (tamamen rastgele) | Mean/median impute güvenli. | 🟢 DÜŞÜK |
| **MAR** (diğer değişkenlere bağlı) | KNN veya model-based impute. `IterativeImputer` kullan. | 🟡 ORTA |
| **MNAR** (kendi değeriyle bağlantılı, gap > 10p) | `is_null` flag ZORUNLU. İmputation tek başına yetmez. | 🔴 KRİTİK |
| Null rate > %30 herhangi bir kolonda | `is_null` flag ekle, sonra impute et. DROP etme. | 🔴 KRİTİK |

---

### Section 45: Outlier Tipi Sınıflandırması

> **Amaç:** Outlier tipini belirle. Hepsine aynı Winsorize uygulamak yanlış.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| **Hata tipi** (z-score > 5, domain imkansız) | İmpute et veya drop et. | 🔴 KRİTİK |
| **Segment tipi** (Isolation Forest anormal + churn rate farklı > 15p) | `is_anomaly` flag ekle. Model bu bilgiyi kullanır. | 🟡 ORTA |
| **Gerçek uç değer** (domain mümkün ama aşırı) | Winsorize (1st–99th percentile clip). | 🟡 ORTA |
| Tüm outlier'lara aynı işlem uygulandı | Hata. Önce tipi belirle, sonra strateji seç. | 🔴 KRİTİK |

---

### Section 46: PCA — Özellik Fazlalığı

> **Amaç:** Feature'lar arasındaki redundancy miktarını ölç.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| 2–3 bileşen %80+ varyansı açıklıyor | Ciddi redundancy. VIF ve korelasyonla hangi feature'ların gereksiz olduğunu bul. | 🟡 ORTA |
| Redundant feature > toplam'ın %40'ı | Bir kısmını drop et veya ratio/difference feature yap. | 🟡 ORTA |
| Feature'lar çeşitli bilgi taşıyor | Düşük redundancy. Hepsini kullan. | 🟢 DÜŞÜK |

---

### Section 47: Null Pattern Clustering

> **Amaç:** Birden fazla feature'ın aynı anda null olduğu sistematik pattern'ları tespit et.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Null pattern'ın churn rate'i genel ortalamadan > 10p farklı | Null pattern kombinasyonunu feature olarak ekle. | 🟡 ORTA |
| Null pattern homojen | Özel feature gerekmez. Standart imputation yeterli. | 🟢 DÜŞÜK |

---

### Section 48: Rare Category Analizi

> **Amaç:** Düşük frekanslı kategorilerin ne yapılacağına karar ver.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Frekans < %1 + churn rate ortalamanın > 2 katı | DROP etme. Ayrı binary flag oluştur. Güçlü sinyal. | 🔴 KRİTİK |
| Frekans < %1 + churn rate ortalamayla benzer | `"Other"` ile birleştir. | 🟡 ORTA |
| Test'te rare kategorinin yüksek frekansı | OOD riski. Target encoding kullan. | 🟡 ORTA |

---

### Section 49: Target Encoding Leakage Simülasyonu

> **Amaç:** CV dışında target encoding yapılırsa ne kadar skor şiştiğini ölç.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Leakage gap > 0.005 | KFold target encoding ZORUNLU. Naif encoding kullanma. | 🔴 KRİTİK |
| Leakage gap 0.001–0.005 | Küçük leak. Yine de KFold TE iyi pratik. | 🟡 ORTA |
| Leakage gap < 0.001 | Minimal leak. Buna rağmen KFold TE önerilir. | 🟢 DÜŞÜK |

---

### Section 50: Optimal Binning (IV-Optimal)

> **Amaç:** `pd.qcut` (eşit frekans) her zaman IV'yi maksimize etmez. Optimal binning karşılaştır.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Optimal binning IV > qcut IV + 0.01 | `OptimalBinning` kullan. Belirgin kazanç var. | 🟡 ORTA |
| Fark < 0.01 | `pd.qcut` yeterli. | 🟢 DÜŞÜK |

---

### Section 51: Partial Dependence Plots (PDP)

> **Amaç:** SHAP marginal etkiyi gösterir, PDP ortalama etkiyi. İkisi farklı bilgi verir.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| PDP düz çizgi | Doğrusal katkı. Transform gerekmez. | 🟢 DÜŞÜK |
| PDP dalgalı / non-linear | `pd.qcut` ile binning feature oluştur. | 🟡 ORTA |
| PDP'de ani sıçrama noktası var | O değer binning threshold'u. Özel binary flag oluştur. | 🟡 ORTA |
| PDP ve SHAP farklı yön gösteriyor | Heterojen segment etkisi var. Segment bazlı analiz yap. | 🟡 ORTA |

---

### Section 52: Anomaly Score as Feature

> **Amaç:** Isolation Forest veya LOF skoru bazen çok güçlü feature'dır.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Anomaly score — target korelasyonu > 0.1 | `anomaly_score_iso` feature olarak ekle. | 🟡 ORTA |
| Anomaly score — target korelasyonu > 0.2 | Güçlü sinyal. Kesinlikle ekle. | 🔴 KRİTİK |
| Korelasyon < 0.05 | Zayıf. Ekleme. | 🟢 DÜŞÜK |

---

### Section 53: Target Rate Stabilitesi (20 Quantile)

> **Amaç:** Feature'ı 20 quantile'a böl. Ani sıçrayan quantile = önemli threshold.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Smooth monoton artış/azalış | `monotone_constraints` ekle. | 🟢 DÜŞÜK |
| Ani sıçrama noktası tespit edildi | O quantile'da binary threshold feature oluştur. | 🟡 ORTA |
| Chaotik dalgalanma | Non-informative. Drop veya transform et. | 🟡 ORTA |

---

### Section 54: Learning Curve Analizi

> **Amaç:** Model underfitting mi, overfitting mi? Augmentation yardımcı olur mu?

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Train-Val gap > 0.05 | Overfitting. `min_child_samples` ve `reg_lambda` artır. | 🔴 KRİTİK |
| Val AUC < 0.80 (tüm veri boyutlarında) | Underfitting. Daha güçlü FE veya daha derin model. | 🔴 KRİTİK |
| Val AUC hâlâ yükseliyor (en büyük sample'da) | Augmentation (original data concat) yardımcı olacak. | 🟡 ORTA |
| Val AUC platoya ulaştı | Daha fazla veri değil, daha iyi feature gerekiyor. | 🟢 DÜŞÜK |

---

### Section 55: Nested CV & Bias-Variance

> **Amaç:** Standard CV şişirilmiş skor verebilir. Nested CV ile gerçek performansı ölç.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Optimism bias > 0.01 | Model evaluation şişirilmiş. Nested CV kullan. Sonuçlara güvenme. | 🔴 KRİTİK |
| Optimism bias < 0.005 | Standard CV güvenilir. | 🟢 DÜŞÜK |
| Variance proxy (std) > 0.01 | Regularizasyon artır veya `RepeatedStratifiedKFold`. | 🟡 ORTA |
| Bias proxy (1-AUC) > 0.15 | Yüksek bias. Daha güçlü model veya FE. | 🟡 ORTA |

---

### Section 56: OOF vs Test Dağılımı

> **Amaç:** OOF ve test tahmin dağılımları farklıysa threshold yanlış seçilir.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| KS(OOF, Test) < 0.05 | Dağılımlar benzer. Threshold güvenilir. | 🟢 DÜŞÜK |
| KS(OOF, Test) 0.05–0.10 | Küçük fark. İzle. | 🟡 ORTA |
| KS(OOF, Test) > 0.10 | Dağılımlar farklı. Rank transformation veya isotonic regression uygula. | 🔴 KRİTİK |
| Test mean prob >> OOF mean prob | Covariate shift var. Test skor beklentini aşağı çek. | 🟡 ORTA |

---

### Section 57: Satır Sırası Bağımlılığı

> **Amaç:** Ardışık satırlar korelasyonluysa ve CV shuffle=False ise ciddi leak riski var.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Autocorrelation lag-1 > 0.2 | Sıra bağımlı. `TimeSeriesSplit` ile karşılaştırma yap. | 🟡 ORTA |
| Autocorrelation lag-1 < 0.05 | Bağımsız. `shuffle=True` ile `StratifiedKFold` güvenli. | 🟢 DÜŞÜK |

---

### Section 58: Feature Interaction Anlamlılığı (Bootstrap CI)

> **Amaç:** İki feature'ın çarpımından gelen AUC artışının güven aralığını ölç.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| 95% CI alt sınırı > 0 | Interaction gerçekten anlamlı. Ekle. | 🟢 DÜŞÜK |
| 95% CI üst sınırı < 0 | Interaction zararlı. Ekleme. | 🟡 ORTA |
| CI sıfırı kapsıyor | Belirsiz. Daha büyük veriyle tekrar dene. | 🟡 ORTA |
| CI çok geniş | Yüksek variance. Tekrarlanabilir değil. Ekleme. | 🟡 ORTA |

---

### Section 59: Encoding Stratejisi Karşılaştırması

> **Amaç:** Label encoding, target encoding, frequency encoding, ordinal encoding — hangisi en iyi?

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Target encoding en yüksek AUC | KFold TE kullan. | 🟢 DÜŞÜK |
| Frequency encoding en yüksek AUC | Kategori frekansı önemli bilgi taşıyor. `value_counts()` map. | 🟢 DÜŞÜK |
| Label encoding yeterli | Basit tut. | 🟢 DÜŞÜK |
| Tüm encodinglar benzer AUC | Bu kategorik feature zayıf. Drop adayı. | 🟡 ORTA |

---

### Section 60: Post-Processing Optimizasyonu

> **Amaç:** Ham tahminlere transformation uygulayarak AUC artır. Daima OOF üzerinde test et.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Rank transform AUC > original + 0.0005 | Test tahminlerine de rank transform uygula. | 🟢 DÜŞÜK |
| Power transform (p≠1) AUC > original + 0.0005 | `proba^p` uygula. Optimal p'yi OOF'tan bul. | 🟢 DÜŞÜK |
| Hiçbiri anlamlı iyileştirme yapmıyor | Ham tahminleri kullan. Basitliği koru. | 🟢 DÜŞÜK |

---

### Section 61: Submission Korelasyon Analizi (Ensemble Diversity)

> **Amaç:** Farklı modellerin tahminlerinin ne kadar çeşitli olduğunu ölç.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Model çifti korelasyonu < 0.90 | Yüksek çeşitlilik. Blend et → +0.002–0.005 bekleniyor. | 🟢 DÜŞÜK |
| Model çifti korelasyonu 0.90–0.95 | Orta çeşitlilik. Blend faydalı olabilir. | 🟢 DÜŞÜK |
| Model çifti korelasyonu > 0.98 | Aynı şeyi öğrenmişler. Bu çiftten birini ensemble'dan çıkar. | 🟡 ORTA |

---

### Section 62: Chi-Square Goodness of Fit (Kategorik Drift)

> **Amaç:** Kategorik feature dağılımlarının train ve test arasında istatistiksel olarak aynı olup olmadığını test et.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| p >= 0.05 | Dağılımlar benzer. Label encoding güvenli. | 🟢 DÜŞÜK |
| p < 0.05 (drift var) | Target encoding kullan. Label encoding drift'i büyütür. | 🔴 KRİTİK |
| Birden fazla kategorik drift'li | Tüm kategorik feature'lar için target encoding uygula. | 🔴 KRİTİK |

---

## 5. Sırayla Uygulanacak FE Reçetesi

> **Her adımdan sonra CV AUC'yi kontrol et. Düşerse geri al.**

### Faza 0: Zorunlu Temizlik (FE'den Önce)

- Sentinel değerleri `"No"` ile değiştir (`No internet service`, `No phone service`)
- Leak feature'ları drop et
- Constant kolonları drop et
- Negatif permutation importance feature'ları drop et
- `tenure=0` & `TotalCharges>0` anomalilerini düzelt
- OOD kategorik değerler için Unknown token ekle
- OOD numeric değerler için Winsorize uygula

### Faza 1: Temel Matematiksel Dönüşümler

```python
charges_residual = TotalCharges - (tenure * MonthlyCharges)
avg_monthly      = TotalCharges / (tenure + 1)
log1p_tenure     = np.log1p(tenure)
log1p_tc         = np.log1p(TotalCharges)
log1p_mc         = np.log1p(MonthlyCharges)
cost_per_month   = MonthlyCharges / (tenure + 1)   # birim başına maliyet
```

Yüksek skew feature'lara `log1p` veya `sqrt` uygula.

### Faza 2: Kategorik Dönüşümler

```python
n_services       = (OnlineSecurity=="Yes") + (OnlineBackup=="Yes") + (DeviceProtection=="Yes")
                 + (TechSupport=="Yes") + (StreamingTV=="Yes") + (StreamingMovies=="Yes")
n_security       = (OnlineSecurity=="Yes") + (DeviceProtection=="Yes") + (TechSupport=="Yes")
n_streaming      = (StreamingTV=="Yes") + (StreamingMovies=="Yes")
has_internet     = (InternetService != "No").astype(int)
is_mtm           = (Contract == "Month-to-month").astype(int)
is_echeck        = (PaymentMethod == "Electronic check").astype(int)
```

### Faza 3: Risk Flag'leri (Segment Tabanlı)

```python
# Segment churn rate > %50 olan kombinasyonlar için binary flag
is_high_risk     = (is_mtm & (InternetService=="Fiber optic") & (TechSupport=="No")).astype(int)
is_charges_anomaly = (abs(ratio - 1) > 0.5).astype(int)   # business logic
is_hard_sample   = fold_preds[:,].std(axis=1) > 0.4        # opsiyonel
# Cohort spread > 20 puan ise:
tenure_group     = pd.cut(tenure, bins=[0,12,24,36,48,100], labels=["0-12","12-24","24-36","36-48","48+"])
```

### Faza 4: SHAP Tabanlı Interaction'lar

```python
# SHAP interaksiyon çiftleri için çarpım feature'ları
tenure_x_mc      = tenure * MonthlyCharges
tc_x_contract    = TotalCharges * is_mtm

# Non-linear feature'lar için binning
tenure_bin       = pd.qcut(tenure, 10, labels=False, duplicates="drop")
mc_bin           = pd.qcut(MonthlyCharges, 10, labels=False, duplicates="drop")

# Top feature oranları
mc_per_service   = MonthlyCharges / (n_services + 1)
```

### Faza 5: Target Encoding (CV İçinde)

```python
from category_encoders import TargetEncoder
# IV > 0.1 olan tüm kategorik feature'lar
# KFold target encoding kullan — sızıntı önlemek için
# OOD kategorik değerler için de target encoding şart
```

### Faza 6: Monotone Constraints & Model Parametreleri

```python
# Section 23'ten al
monotone_constraints = [+1, -1, 0, ...]   # sırası feats listesiyle aynı olmalı

params = {
    "monotone_constraints": monotone_constraints,
    "is_unbalance": True,           # veya scale_pos_weight
    "label_smoothing": 0.05,        # sadece conflict rate > %5 ise
}
```

---

## 6. Master Hızlı Karar Referans Tablosu

Pipeline çıktısını okurken bu tabloyu yanında tut:

| Metrik / Bulgu | Eşik Değer | Aksiyon |
|---|---|---|
| Solo AUC | > 0.95 | 🔴 LEAK — DROP |
| Adversarial AUC | > 0.70 | 🔴 BÜYÜK SORUN — DUR |
| Adversarial AUC | 0.60–0.70 | 🟡 Top drift feature normalize et |
| KS Drift | Severe (> 0.10) | 🔴 Drop veya normalize |
| VIF | > 10 | 🟡 Drop veya PCA |
| IV | > 0.5 | 🔴 Leak kontrolü |
| IV | < 0.02 | 🟡 Drop et |
| Perm Importance | < 0 | 🔴 DROP |
| CV std (seed) | > 0.005 | 🔴 CV stratejisini düzelt |
| Hard Sample % | > 15% | 🔴 Agresif FE gerekli |
| Conflict Rate | > 10% | 🔴 sample_weight |
| Cohort Spread | > 20 puan | 🟡 tenure_group feature |
| KDE KS | < 0.05 | 🟡 Feature zayıf — drop |
| Segment Spread | > 40 puan | 🟡 Risk flag oluştur |
| OOD Rate | > 20% | 🔴 Target encoding şart |
| Cramer V | > 0.8 | 🟡 Redundant — birini drop |
| Pearson \|r\| | > 0.85 | 🟡 Korelasyonlu — birini drop |
| SHAP interaction | > 0.3 | 🟢 Çarpım feature oluştur |
| Baseline AUC | < 0.80 | 🔴 Encoding'i yeniden düşün |
| Imbalance Ratio | < 0.2 | 🔴 is_unbalance=True |
| **Brier Score** | **> 0.20** | **🔴 Platt scaling zorunlu** |
| **PSI** | **> 0.25** | **🔴 Feature ciddi kayma — drop/encode** |
| **PSI** | **0.10–0.25** | **🟡 RobustScaler ile normalize et** |
| **Optimal Threshold** | **0.5'ten > 0.1 uzak** | **🔴 Cost-optimal threshold kullan** |
| **Top Decile Lift** | **< 1.5x** | **🔴 Zayıf model — FE gerekli** |
| **Optimism Bias** | **> 0.01** | **🔴 Nested CV kullan** |
| **OOF-Test KS** | **> 0.10** | **🔴 Rank transform uygula** |
| **Cohen d** | **< 0.2** | **🟡 Pratik etkisi yok — drop** |
| **Cohen d** | **> 0.8** | **🟢 Büyük etki — FE öncelikli** |
| **Leakage gap (TE)** | **> 0.005** | **🔴 KFold TE zorunlu** |
| **MNAR gap** | **> 10 puan** | **🔴 is_null flag zorunlu** |
| **Outlier tipi** | **Segment (gap>15p)** | **🟡 is_anomaly flag ekle** |
| **Autocorr lag-1** | **> 0.2** | **🟡 TimeSeriesSplit değerlendir** |
| **Hard Sample %** | **> 15%** | **🔴 Agresif FE gerekli** |
| **Conflict Rate** | **> 10%** | **🔴 sample_weight zorunlu** |

| Anomaly Score (ISO/LOF) | > 0.2 korelasyon | 🟡 Feature olarak ekle |
| Bootstrap CI (interaction) | Alt sınır > 0 | 🟢 Interaction ekle |
| Encoding karşılaştırma | Target enc en iyi | 🟢 KFold TE kullan |
| Nested CV optimism | > 0.01 | 🔴 CV stratejisini düzelt |
| OOF-Test KS | > 0.10 | 🔴 Rank transform uygula |
| Autocorrelation lag-1 | > 0.2 | 🟡 TimeSeriesSplit dene |
| Post-processing | Rank/power +0.0005 | 🟢 Test'e de uygula |

---

## 7. Telco Churn Dataset — Özel Notlar

| Not | Detay |
|---|---|
| **Playground Series** | Bu veri sentetik olarak üretilmiştir. Orijinal IBM Telco verisi (blastchar) ile augment edilebilir. |
| **Augment Kuralı** | Target rate farkı < 5 puan → `pd.concat([train, original])`. Fark > 5 puan → `is_original=1` flag ile concat et. |
| **En Güçlü Feature'lar** | `Contract`, `tenure`, `MonthlyCharges`, `TotalCharges`, `InternetService`, `TechSupport` |
| **Domain Bilgisi** | `Month-to-month + Electronic check + Fiber optic + No TechSupport` kombinasyonu maksimum churn risk segmentidir. |
| **charges_residual** | `TotalCharges − (tenure × MonthlyCharges)` → müşterinin beklenenden fazla/az ödeyip ödemediğini gösterir. Çok güçlü feature. |

### Öncelikli FE Listesi — Telco İçin

```python
# 1. En öncelikli
charges_residual      = TotalCharges - (tenure * MonthlyCharges)

# 2. Servis sayaçları
n_services            = sum(df[c] == "Yes" for c in inet_cols)

# 3. Binary sözleşme flag'leri
is_month_to_month     = (Contract == "Month-to-month").astype(int)

# 4. Yüksek risk segmenti
is_high_risk_segment  = (
    (Contract == "Month-to-month") &
    (InternetService == "Fiber optic") &
    (TechSupport == "No")
).astype(int)

# 5. Maliyet/servis oranı
avg_monthly_per_svc   = MonthlyCharges / (n_services + 1)

# 6. Tenure interaksiyonu
tenure_x_monthly      = tenure * MonthlyCharges

# 7. Sözleşme normalize tenure
contract_map          = {"Month-to-month": 1, "One year": 12, "Two year": 24}
contract_tenure_ratio = tenure / df["Contract"].map(contract_map)
```

---

> **Son Not:** Her FE adımı sonrası 5-fold LightGBM AUC'sini taban çizgisiyle karşılaştır. Yükseliş yoksa geri al. Birden fazla feature aynı anda ekleme — **tek tek test et.**
