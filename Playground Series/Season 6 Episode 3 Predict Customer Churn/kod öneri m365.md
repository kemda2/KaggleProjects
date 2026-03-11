# 🚀 Senior Data Pipeline — FE Öncesi Analiz Rehberi

**Testler · Kurallar · Aksiyonlar · FE Hazırlık Kılavuzu**

Bu döküman 34 bölümlük pipeline'ın (Section 1–34) çıktılarını **eğer X ise → Y yap** formatında özetler. FE'ye geçmeden önce her section'ı sırasıyla uygulayın.

---

## İçindekiler

1. [Temel Prensipler](#1-temel-prensipler)
2. [Section Bazlı Karar Kuralları (S1–S24)](#2-section-bazlı-karar-kuralları)
3. [Ek Analizler Section 26–34](#3-ek-analizler--section-2634-karar-kuralları)
4. [Sırayla Uygulanacak FE Reçetesi](#4-sırayla-uygulanacak-fe-reçetesi)
5. [Hızlı Karar Referans Tablosu](#5-hızlı-karar-referans-tablosu)
6. [Telco Churn Dataset — Özel Notlar](#6-telco-churn-dataset--özel-notlar)

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

## 4. Sırayla Uygulanacak FE Reçetesi

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

## 5. Hızlı Karar Referans Tablosu

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

---

## 6. Telco Churn Dataset — Özel Notlar

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