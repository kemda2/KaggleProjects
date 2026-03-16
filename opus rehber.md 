

# 🧠 Train-Test Pipeline: NEDEN → YORUM → SONRAKI ADIMA ETKİSİ

Her adımın **mantığını**, **sonuç yorumlamasını** ve **domino etkisini** detaylıca açıklıyorum.

---

## 📌 1. EDA (Keşifsel Veri Analizi)

### 🔍 NEDEN YAPILIR?

```
Veriyi hiç tanımadan model kurmak, gözleri kapalı araba
kullanmak gibidir. EDA ile:

• Verinin yapısını (kaç satır, kaç kolon, hangi tipler) öğrenirsin
• Target değişkenin davranışını anlarsın
• Potansiyel sorunları (eksik veri, aykırı değer, dengesizlik) erkenden tespit edersin
• Feature'ların dağılımını görerek hangi dönüşümlerin gerektiğini anlarsın
• Train ile Test arasındaki farkları yakalar ve data drift'i tespit edersin
```

### 📊 SONUÇLAR NASIL YORUMLANIR?

```python
# ═══════════════════════════════════════════════════
# 1.1 SHAPE KONTROLÜ
# ═══════════════════════════════════════════════════
print(f"Train: {train.shape}")  # (10000, 50)
print(f"Test:  {test.shape}")   # (5000, 49)  ← target yok

# YORUM:
# • Test'te 49 kolon → target kolonu yok, bu normal
# • Train'de 10000 satır → çoğu model için yeterli
# • 50 feature → VIF ve feature selection gerekebilir
# • Satır/Kolon oranı: 10000/50 = 200 → iyi (min 10-20× olmalı)
#   Eğer oran < 10 ise → overfitting riski yüksek, regularization şart

# ═══════════════════════════════════════════════════
# 1.2 TARGET DAĞILIMI (Regression)
# ═══════════════════════════════════════════════════
print(f"Skewness: {train['target'].skew():.4f}")   # 2.34
print(f"Kurtosis: {train['target'].kurtosis():.4f}") # 8.12

# YORUM:
# Skewness = 2.34 → Sağa çarpık (pozitif skew)
#   |skew| < 0.5  → Simetrik, dönüşüm gerekmez
#   |skew| 0.5-1   → Hafif çarpık, dönüşüm düşünülebilir
#   |skew| > 1     → Ciddi çarpık, LOG DÖNÜŞÜMÜ GEREKLİ ✅
#
# Kurtosis = 8.12 → Sivri (leptokurtic), kuyruklar kalın
#   kurt ≈ 3  → Normal dağılıma yakın
#   kurt > 3  → Sivri, aykırı değer olabilir
#   kurt < 3  → Basık (platykurtic)
#
# 🔗 SONRAKİ ADIMA ETKİSİ:
#   → Adım 4'te np.log1p(target) dönüşümü uygulanacak
#   → Lineer model kullanılacaksa normallik ön koşulu sağlanmalı
#   → Tahmin sonrası np.expm1() ile geri dönüşüm yapılacak

# ═══════════════════════════════════════════════════
# 1.3 TARGET DAĞILIMI (Classification)
# ═══════════════════════════════════════════════════
print(train['target'].value_counts(normalize=True))
# 0    0.92
# 1    0.08

# YORUM:
# Sınıf 0: %92, Sınıf 1: %8 → CİDDİ DENGESİZLİK
# Imbalance ratio = 0.08/0.92 = 0.087
#
#   Ratio > 0.5   → Dengeli, özel işlem gerekmez
#   Ratio 0.2-0.5 → Hafif dengesiz, class_weight='balanced' yeterli
#   Ratio < 0.2   → Ciddi dengesiz, SMOTE veya özel strateji gerekli ✅
#
# 🔗 SONRAKİ ADIMA ETKİSİ:
#   → Adım 8'de SMOTE veya class_weight kullanılacak
#   → Accuracy yerine F1, AUC, Precision-Recall kullanılacak
#   → StratifiedKFold ZORUNLU (normal KFold kullanılamaz)
#   → Threshold optimization yapılacak (varsayılan 0.5 optimal olmayabilir)

# ═══════════════════════════════════════════════════
# 1.4 TRAIN vs TEST DAĞILIM KARŞILAŞTIRMASI
# ═══════════════════════════════════════════════════
from scipy import stats

for col in common_cols:
    if train[col].dtype in ['int64','float64']:
        stat, p = stats.ks_2samp(train[col].dropna(), test[col].dropna())
        if p < 0.05:
            print(f"⚠️  {col}: FARKLI (KS p={p:.4f})")
        else:
            print(f"✅ {col}: Benzer  (KS p={p:.4f})")

# YORUM:
# KS Test (Kolmogorov-Smirnov 2-sample):
#   H₀: İki dağılım aynı populasyondan gelir
#   p > 0.05 → Aynı dağılım ✅ → Feature güvenle kullanılabilir
#   p < 0.05 → Farklı dağılım ❌ → DATA DRIFT var
#
# Eğer birçok feature'da drift varsa:
#   → Model train'de öğrendiğini test'te uygulayamaz
#   → Adversarial validation düşünülmeli
#   → Domain adaptation gerekebilir
#   → O feature'lar modelden çıkarılabilir
#
# 🔗 SONRAKİ ADIMA ETKİSİ:
#   → Drift olan feature'lar Adım 6'da feature selection'da elenebilir
#   → Model performansı test'te düşük çıkarsa sebebi budur
#   → Adversarial validation ile train/test ayrılabilirliği kontrol edilir
```

### 🔗 BİR SONRAKİ ADIMA GENEL ETKİSİ:

```
EDA SONUCU                          →  ETKİLEDİĞİ ADIM
─────────────────────────────────────────────────────────
Eksik veri tespit edildi             →  Adım 2 (Imputation)
Aykırı değerler görüldü             →  Adım 3 (Outlier handling)
Target çarpık                       →  Adım 4 (Log dönüşümü)
Feature'lar farklı ölçeklerde       →  Adım 7 (Scaling)
Sınıf dengesizliği var              →  Adım 8 (SMOTE/class_weight)
Çok fazla feature var               →  Adım 6 (Feature selection)
Train-Test drift var                →  Feature çıkarma / dikkatli olma
```

---

## 📌 2. EKSİK VERİ ANALİZİ

### 🔍 NEDEN YAPILIR?

```
• Çoğu ML algoritması NaN değerlerle çalışamaz (LR, SVM, KNN)
  (XGBoost, LightGBM NaN'ı handle edebilir ama yine de analiz gerekli)
• Eksiklik rastgele değilse (MAR/MNAR), bilgi kaybı + bias oluşur
• Yanlış imputation modeli tamamen yanıltabilir
• Eksikliğin KENDİSİ bir bilgi taşıyabilir (feature olarak kullanılabilir)
```

### 📊 SONUÇLAR NASIL YORUMLANIR?

```python
# ═══════════════════════════════════════════════════
# 2.1 EKSİKLİK ORANI
# ═══════════════════════════════════════════════════
missing_pct = train.isnull().mean() * 100

# YORUM ve KARAR TABLOSU:
# ─────────────────────────────────────────────────
# Eksiklik %      Karar
# ─────────────────────────────────────────────────
# 0%              Sorun yok
# 0-5%            Median/Mode imputation yeterli
# 5-15%           KNN Imputer veya Iterative Imputer
# 15-30%          Feature olarak "is_missing" flag ekle + imputation
# 30-50%          Dikkatli ol, feature çıkarmayı düşün
# >50%            Genellikle DROP ET (bilgi güvenilir değil)
# ─────────────────────────────────────────────────

for col in missing_pct[missing_pct > 0].index:
    pct = missing_pct[col]
    if pct > 50:
        print(f"🗑️  {col}: %{pct:.1f} → DROP ÖNERİLİR")
    elif pct > 15:
        print(f"🏷️  {col}: %{pct:.1f} → is_missing FLAG + KNN Imputer")
    elif pct > 5:
        print(f"🔧 {col}: %{pct:.1f} → KNN / Iterative Imputer")
    else:
        print(f"✅ {col}: %{pct:.1f} → Simple Imputer (median/mode)")

# ═══════════════════════════════════════════════════
# 2.2 EKSİKLİK PATTERN'İ
# ═══════════════════════════════════════════════════
import missingno as msno
msno.heatmap(train)

# YORUM:
# Heatmap'te iki kolon arasında korelasyon yüksekse:
#   → İkisi birlikte eksik → aynı kaynaktan geliyor olabilir
#   → Birlikte impute etmek (Multivariate Imputer) daha mantıklı
#
# Eğer eksiklik tamamen rastgele (monotone pattern yok):
#   → MCAR olabilir → Simple imputation yeterli
# Eğer belirli bir sıra/blok halinde eksikse:
#   → Sistematik → MAR/MNAR olabilir → Dikkat!

# ═══════════════════════════════════════════════════
# 2.3 EKSİKLİK TARGET İLE İLİŞKİLİ Mİ? (MAR Kontrolü)
# ═══════════════════════════════════════════════════
for col in train.columns[train.isnull().any()]:
    group_missing    = train[train[col].isnull()]['target']
    group_notmissing = train[train[col].notna()]['target']
    
    if len(group_missing) > 1:
        stat, p = stats.mannwhitneyu(group_missing, group_notmissing)
        
        if p < 0.05:
            print(f"⚠️  {col}: Eksiklik TARGET ile İLİŞKİLİ (p={p:.4f})")
            # → Bu eksikliğin kendisi bilgi taşıyor!
            # → "is_missing" flag MUTLAKA eklenmeli
            # → Silmek bias yaratır
        else:
            print(f"✅ {col}: Eksiklik TARGET ile ilişkisiz (p={p:.4f})")
            # → Güvenle impute edilebilir

# ═══════════════════════════════════════════════════
# 2.4 IMPUTATION STRATEJİSİ ve ETKİSİ
# ═══════════════════════════════════════════════════

# ÖNEMLİ: Imputer SADECE train'den öğrenir!
from sklearn.impute import SimpleImputer, KNNImputer

# Neden train'den öğrenilir?
# → Test verisi "gelecek"tir. Gelecekten bilgi sızdırmak = DATA LEAKAGE
# → Örnek: Test'in medyanı train'den farklıysa, 
#           test medyanını kullanmak leakage'dır

num_imputer = KNNImputer(n_neighbors=5)
num_imputer.fit(train[num_cols])              # ← SADECE TRAIN
train[num_cols] = num_imputer.transform(train[num_cols])
test[num_cols]  = num_imputer.transform(test[num_cols])  # ← AYNI PARAMETRE

# DOĞRULAMA: Imputation sonrası dağılım bozuldu mu?
for col in num_cols:
    print(f"{col}: Before skew={original_skew:.2f}, After skew={train[col].skew():.2f}")
    # Eğer skewness çok değiştiyse → imputation yöntemi uygun değil
```

### 🔗 BİR SONRAKİ ADIMA ETKİSİ:

```
EKSİK VERİ SONUCU                    →  ETKİLEDİĞİ ADIM
─────────────────────────────────────────────────────────
Eksiklik target ile ilişkili (MAR)    →  is_missing flag → Feature Engineering
%50+ eksik kolon silindi              →  Feature sayısı azaldı → Model basitleşti
KNN Imputer kullanıldı                →  Dağılım korundu → Normallik testleri daha güvenilir
Imputation sonrası yeni değerler      →  Aykırı değer tespiti tekrar yapılmalı (Adım 3)
Train'den öğrenilen parametreler      →  Pipeline'a kaydedilmeli (Adım 20)
```

---

## 📌 3. AYKIRI DEĞER TESPİTİ

### 🔍 NEDEN YAPILIR?

```
• Aykırı değerler ortalamayı ve standart sapmayı BÜYÜK ÖLÇÜDE etkiler
• Lineer modeller aykırı değerlere ÇOK HASSASTIR (loss = squared error)
• Tree-based modeller daha dayanıklıdır ama yine de split noktaları etkilenir
• Bazı aykırı değerler GERÇEK BİLGİ taşır (fraud detection'da olduğu gibi)
• Bazıları ise VERİ HATASI'dır (yanlış giriş, ölçüm hatası)
• Aykırı değerler normallik testlerini BOZAR → yanlış test seçimine yol açar
```

### 📊 SONUÇLAR NASIL YORUMLANIR?

```python
# ═══════════════════════════════════════════════════
# 3.1 IQR YÖNTEMİ
# ═══════════════════════════════════════════════════
def detect_and_report_outliers(df, col, k=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    pct = len(outliers) / len(df) * 100
    
    print(f"\n{'='*50}")
    print(f"Feature: {col}")
    print(f"Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
    print(f"Alt sınır: {lower:.2f}, Üst sınır: {upper:.2f}")
    print(f"Aykırı sayısı: {len(outliers)} ({pct:.1f}%)")
    
    # YORUM VE KARAR
    if pct == 0:
        print("✅ Aykırı değer yok")
        decision = "none"
    elif pct < 1:
        print("⚠️  Az sayıda aykırı → Veri hatası olabilir → İNCELE")
        decision = "investigate"
    elif pct < 5:
        print("🔶 Makul aykırı → Kırpma (capping) önerilir")
        decision = "cap"
    else:
        print("🔴 Çok fazla aykırı → Dağılım doğal olarak geniş olabilir")
        print("   → Dönüşüm (log) veya robust model kullan")
        decision = "transform"
    
    return decision, lower, upper

for col in num_cols:
    decision, lo, hi = detect_and_report_outliers(train, col)

# ═══════════════════════════════════════════════════
# 3.2 AYKIRI DEĞERLERİN TARGET İLE İLİŞKİSİ
# ═══════════════════════════════════════════════════
# Bu çok kritik! Aykırı değeri silmeden önce target'a etkisini kontrol et

for col in num_cols:
    mask, lo, hi = detect_outliers_iqr(train, col)
    if mask.sum() > 0:
        target_with_outlier    = train.loc[mask, 'target'].mean()
        target_without_outlier = train.loc[~mask, 'target'].mean()
        
        diff = abs(target_with_outlier - target_without_outlier)
        print(f"\n{col}:")
        print(f"  Target (aykırı var):    {target_with_outlier:.4f}")
        print(f"  Target (aykırı yok):    {target_without_outlier:.4f}")
        print(f"  Fark:                   {diff:.4f}")
        
        if diff > train['target'].std() * 0.5:
            print(f"  ⚠️  AYKIRI DEĞERLER TARGET'I ETKİLİYOR → SİLME, BİLGİ TAŞIYOR!")
            # → Bu feature'daki aykırı değerler önemli sinyal
            # → Silmek yerine: dönüşüm uygula veya is_outlier flag ekle
        else:
            print(f"  ✅ Target'a etkisi düşük → Güvenle kırpılabilir/silinebilir")

# ═══════════════════════════════════════════════════
# 3.3 KARAR MATRİSİ
# ═══════════════════════════════════════════════════
"""
DURUM                              KARAR
──────────────────────────────────────────────────────
Veri hatası (yaş=999)              → SİL veya NaN yap
Az sayıda, target'a etkisiz        → SİL veya KIRP (cap)
Çok sayıda, dağılım geniş          → LOG DÖNÜŞÜMÜ
Target ile ilişkili                 → BIRAK + is_outlier flag
Tree-based model kullanılacak       → BIRAK (dayanıklı)
Linear model kullanılacak           → KIRP veya DÖNÜŞTÜR
"""

# ═══════════════════════════════════════════════════
# 3.4 KIRPMA (CAPPING) UYGULAMASI
# ═══════════════════════════════════════════════════
# Sadece TRAIN'den öğrenilen sınırlarla!

boundaries = {}
for col in num_cols:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    boundaries[col] = (lower, upper)
    
    # Train'e uygula
    train[col] = train[col].clip(lower, upper)
    # Test'e AYNI sınırlarla uygula
    test[col] = test[col].clip(lower, upper)

# NEDEN test'e de uygulanır?
# → Model train'deki [lower, upper] aralığını öğrendi
# → Test'te bu aralığın dışında değer gelirse model beklenmedik davranır
# → Ama sınırlar TRAIN'den gelir, TEST'ten hesaplanmaz!
```

### 🔗 BİR SONRAKİ ADIMA ETKİSİ:

```
AYKIRI DEĞER KARARI                   →  ETKİLEDİĞİ ADIM
─────────────────────────────────────────────────────────
Aykırılar silindi                     →  Örneklem küçüldü → CV'de dikkat
Kırpma (capping) yapıldı             →  Dağılım değişti → Normallik testi tekrar
Log dönüşümü gerekli                  →  Adım 4'te uygulanacak
is_outlier flag eklendi               →  Yeni feature → Feature selection'a dahil
Aykırılar bırakıldı (tree model)      →  Scaling'de robust scaler kullan
Satır silindiyse                      →  Eksik veri imputation parametreleri değişebilir
```

---

## 📌 4. DAĞILIM & DÖNÜŞÜM TESTLERİ

### 🔍 NEDEN YAPILIR?

```
• LİNEER MODELLER: Normal dağılım varsayar. Feature'lar çarpıksa 
  katsayılar (coefficients) güvenilir OLMAZ
• MESAFE BAZLI MODELLER (KNN, SVM): Dağılım homojen değilse 
  uzaklık hesapları yanıltıcı olur
• TREE-BASED MODELLER: Dağılım umursamaz ama target çarpıksa 
  RMSE metriği büyük değerlere odaklanır → log dönüşümü yardımcı olur
• NEURAL NETWORKS: Aktivasyon fonksiyonları belirli aralıklarda çalışır,
  çarpık dağılım gradient'leri bozar
```

### 📊 SONUÇLAR NASIL YORUMLANIR?

```python
# ═══════════════════════════════════════════════════
# 4.1 SKEWNESS (ÇARPIKLIK) ANALİZİ
# ═══════════════════════════════════════════════════
for col in num_cols:
    sk = train[col].skew()
    
    if abs(sk) < 0.5:
        status = "✅ Simetrik → Dönüşüm gerekmez"
        action = "none"
    elif abs(sk) < 1:
        status = "🔶 Hafif çarpık → Opsiyonel dönüşüm"
        action = "optional"
    elif abs(sk) < 2:
        status = "🔴 Çarpık → SQRT dönüşümü önerilir"
        action = "sqrt"
    else:
        status = "🔴🔴 Aşırı çarpık → LOG dönüşümü zorunlu"
        action = "log"
    
    print(f"{col}: skew={sk:.4f} → {status}")

# NEDEN SKEWNESS ÖNEMLİ?
# ─────────────────────────────────────────────
# 1. Ortalama ≠ Medyan olur → mean imputation yanıltıcı
# 2. StandardScaler'ın varsayımı bozulur → scaling yanlış olur
# 3. Pearson korelasyonu güvenilmez olur → yanlış feature selection
# 4. Lineer modelin katsayıları bias'lı olur

# ═══════════════════════════════════════════════════
# 4.2 DÖNÜŞÜM SEÇİMİ ve UYGULAMASI
# ═══════════════════════════════════════════════════

# Hangi dönüşüm ne zaman?
"""
DÖNÜŞÜM           KOŞUL                              FORMÜL
──────────────────────────────────────────────────────────────
Log (log1p)        Sağa çarpık, pozitif değerler       log(1+x)
Karekök (sqrt)     Hafif çarpık, pozitif               √x
Box-Cox            Pozitif değerler                    (x^λ - 1) / λ
Yeo-Johnson        Pozitif VE negatif değerler ✅       Genelleştirilmiş Box-Cox
Reciprocal         Sol çarpık                          1/x
"""

from sklearn.preprocessing import PowerTransformer

# Yeo-Johnson → en güvenli seçim (negatif değerleri de handle eder)
pt = PowerTransformer(method='yeo-johnson')
pt.fit(train[num_cols])  # SADECE TRAIN'den lambda parametrelerini öğren

# Lambda değerlerini kontrol et
for col, lam in zip(num_cols, pt.lambdas_):
    print(f"{col}: lambda = {lam:.4f}")
    # lambda ≈ 1   → dönüşüm çok az etki etti → zaten normal dağılıma yakındı
    # lambda ≈ 0   → log dönüşümüne yakın
    # lambda ≈ 0.5 → sqrt dönüşümüne yakın
    # lambda ≈ -1  → reciprocal dönüşümüne yakın

# Uygula
train[num_cols] = pt.transform(train[num_cols])
test[num_cols]  = pt.transform(test[num_cols])  # AYNI lambda'lar

# DOĞRULAMA: Dönüşüm işe yaradı mı?
for col in num_cols:
    new_skew = train[col].skew()
    print(f"{col}: yeni skew = {new_skew:.4f} → "
          f"{'✅ Başarılı' if abs(new_skew) < 1 else '❌ Hâlâ çarpık'}")

# ═══════════════════════════════════════════════════
# 4.3 TARGET DÖNÜŞÜMÜ (REGRESSION İÇİN ÇOK KRİTİK)
# ═══════════════════════════════════════════════════
target_skew = train['target'].skew()
print(f"Target Skewness: {target_skew:.4f}")

if abs(target_skew) > 1:
    train['target_original'] = train['target'].copy()
    train['target'] = np.log1p(train['target'])
    print(f"Log dönüşümü sonrası skew: {train['target'].skew():.4f}")
    
    # ⚠️  UNUTMA: Tahmin sonrası geri dönüşüm gerekli!
    # y_pred_original = np.expm1(y_pred_log)

# NEDEN TARGET DÖNÜŞTÜRÜLÜR?
# → RMSE büyük değerlerin hatasını KARE alarak çok büyütür
# → Log dönüşümü büyük-küçük değer farkını azaltır
# → Model hem 100 hem 100.000 olan değerleri daha dengeli öğrenir
# → Residuals daha normal dağılır → lineer model varsayımları sağlanır

# ⚠️  DİKKAT:
# → Target'ta 0 varsa → log1p kullan (log(0) = -∞)
# → Target'ta negatif varsa → log dönüşümü YAPILAMAZ
#   → Yeo-Johnson veya farklı metrik (MAE) düşün
```

### 🔗 BİR SONRAKİ ADIMA ETKİSİ:

```
DÖNÜŞÜM SONUCU                       →  ETKİLEDİĞİ ADIM
─────────────────────────────────────────────────────────
Feature'lar normalleşti               →  Pearson korelasyonu güvenilir (Adım 5)
                                       →  StandardScaler doğru çalışır (Adım 7)
                                       →  Lineer model varsayımları sağlanır (Adım 11)
Target log dönüştürüldü               →  Model log-space'te öğrenir
                                       →  Tahmin sonrası np.expm1() uygulanmalı
                                       →  RMSE metriği RMSLE'ye dönüşür
Lambda parametreleri kaydedildi        →  Test'e aynı dönüşüm uygulanır
Dönüşüm başarısız oldu                →  Non-parametrik yöntemlere yönel
                                       →  Robust scaler kullan
```

---

## 📌 5. KORELASYON & ÇOKLİ DOĞRUSALLIK

### 🔍 NEDEN YAPILIR?

```
• FEATURE-TARGET KORELASYONu: Hangi feature'lar target'ı açıklıyor?
  → Düşük korelasyonlu feature'lar bilgi taşımayabilir → atılabilir
  
• FEATURE-FEATURE KORELASYONu: İki feature aynı bilgiyi mi taşıyor?
  → Yüksek korelasyon = REDUNDANCY → gereksiz karmaşıklık
  → Lineer modellerde katsayılar INSTABLE olur (işaret bile değişebilir!)
  → VIF yüksekse güven aralıkları genişler → yorumlama imkansızlaşır
  
• TREE MODELLERİ: Multicollinearity'den etkilenmez ama
  → Feature importance paylaşılır → yanlış yorumlanır
  → Permutation importance'ta yanıltıcı sonuçlar çıkar
```

### 📊 SONUÇLAR NASIL YORUMLANIR?

```python
# ═══════════════════════════════════════════════════
# 5.1 FEATURE-TARGET KORELASYONU
# ═══════════════════════════════════════════════════
from scipy import stats

correlation_results = []
for col in num_cols:
    # Pearson (lineer ilişki)
    r_pearson, p_pearson = stats.pearsonr(train[col], train['target'])
    # Spearman (monoton ilişki - non-linear da yakalar)
    r_spearman, p_spearman = stats.spearmanr(train[col], train['target'])
    
    correlation_results.append({
        'Feature': col,
        'Pearson_r': r_pearson,
        'Pearson_p': p_pearson,
        'Spearman_r': r_spearman,
        'Spearman_p': p_spearman
    })

corr_df = pd.DataFrame(correlation_results)

# YORUM TABLOSU
for _, row in corr_df.iterrows():
    col = row['Feature']
    r_p = abs(row['Pearson_r'])
    r_s = abs(row['Spearman_r'])
    
    # Korelasyon gücü
    if r_p < 0.1:
        strength = "Yok denecek kadar zayıf"
    elif r_p < 0.3:
        strength = "Zayıf"
    elif r_p < 0.5:
        strength = "Orta"
    elif r_p < 0.7:
        strength = "Güçlü"
    else:
        strength = "Çok güçlü"
    
    # Pearson vs Spearman farkı
    diff = abs(r_s - r_p)
    if diff > 0.1:
        linearity = "⚠️  Non-lineer ilişki var (Spearman > Pearson)"
        # → Lineer model bu ilişkiyi yakalayamaz
        # → Polinom feature veya tree model gerekli
    else:
        linearity = "✅ İlişki doğrusal"
    
    # p-value yorumu
    if row['Pearson_p'] > 0.05:
        significance = "❌ İstatistiksel olarak ANLAMSIZ"
        # → Bu feature'ı çıkarmayı düşün
    else:
        significance = "✅ Anlamlı"
    
    print(f"\n{col}: |r|={r_p:.4f} ({strength})")
    print(f"  {significance}")
    print(f"  {linearity}")

# ═══════════════════════════════════════════════════
# 5.2 FEATURE-FEATURE KORELASYONU
# ═══════════════════════════════════════════════════
corr_matrix = train[num_cols].corr().abs()

# Yüksek korelasyonlu çiftleri bul
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        r = corr_matrix.iloc[i, j]
        if r > 0.85:
            col_i = corr_matrix.columns[i]
            col_j = corr_matrix.columns[j]
            
            # Hangisi target ile daha çok korelasyonlu?
            r_target_i = abs(train[col_i].corr(train['target']))
            r_target_j = abs(train[col_j].corr(train['target']))
            
            drop_candidate = col_j if r_target_i > r_target_j else col_i
            keep = col_i if r_target_i > r_target_j else col_j
            
            print(f"⚠️  {col_i} ↔ {col_j}: r={r:.4f}")
            print(f"   Target korr: {col_i}={r_target_i:.4f}, {col_j}={r_target_j:.4f}")
            print(f"   ÖNERİ: {drop_candidate} ÇIKAR, {keep} BIRAK")
            
            high_corr_pairs.append((col_i, col_j, r, drop_candidate))

# KARAR KURALLARI:
"""
r > 0.95  → Neredeyse aynı bilgi → BİRİNİ KESİNLİKLE ÇIKAR
r > 0.85  → Çok yüksek → Target ile daha az ilişkili olanı çıkar
r > 0.70  → Yüksek → Lineer modelde sorun, tree'de genellikle OK
r < 0.70  → Kabul edilebilir
"""

# ═══════════════════════════════════════════════════
# 5.3 VIF (VARIANCE INFLATION FACTOR)
# ═══════════════════════════════════════════════════
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

X_vif = sm.add_constant(train[num_cols].dropna())
vif_data = pd.DataFrame()
vif_data['Feature'] = X_vif.columns
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) 
                   for i in range(X_vif.shape[1])]
vif_data = vif_data[vif_data['Feature'] != 'const'].sort_values('VIF', ascending=False)

for _, row in vif_data.iterrows():
    vif = row['VIF']
    col = row['Feature']
    
    if vif < 5:
        print(f"✅ {col}: VIF={vif:.2f} → Multicollinearity YOK")
    elif vif < 10:
        print(f"🔶 {col}: VIF={vif:.2f} → ORTA düzey → İzle")
    else:
        print(f"🔴 {col}: VIF={vif:.2f} → CİDDİ multicollinearity → ÇIKAR veya PCA")

# VIF NEDİR?
# VIF = 1 / (1 - R²) → Feature diğer feature'larla regresyona sokulduğunda R²
# VIF = 1   → Hiç korelasyon yok
# VIF = 5   → R² = 0.80 → %80'i diğer feature'larla açıklanıyor
# VIF = 10  → R² = 0.90 → %90'ı diğer feature'larla açıklanıyor → ÇOK TEHLİKELİ
# VIF = 100 → R² = 0.99 → Neredeyse tamamen redundant

# ─── VIF YÜKSEKSE NE YAPILIR? ───
# 1. En yüksek VIF'li feature'ı çıkar → VIF'leri tekrar hesapla → iteratif
# 2. PCA ile boyut indirgeme
# 3. Ridge/Lasso regularization (katsayıları sınırlar)
# 4. İki feature'ı birleştir (interaction veya oran)

# İteratif VIF çıkarma
def iterative_vif_removal(df, cols, threshold=5):
    remaining = cols.copy()
    dropped = []
    
    while True:
        X = sm.add_constant(df[remaining])
        vifs = [variance_inflation_factor(X.values, i) 
                for i in range(1, X.shape[1])]
        max_vif = max(vifs)
        
        if max_vif < threshold:
            break
        
        max_idx = vifs.index(max_vif)
        drop_col = remaining[max_idx]
        remaining.remove(drop_col)
        dropped.append((drop_col, max_vif))
        print(f"Çıkarıldı: {drop_col} (VIF={max_vif:.2f})")
    
    print(f"\nKalan feature'lar: {remaining}")
    print(f"Çıkarılan feature'lar: {dropped}")
    return remaining, dropped
```

### 🔗 BİR SONRAKİ ADIMA ETKİSİ:

```
KORELASYON SONUCU                     →  ETKİLEDİĞİ ADIM
─────────────────────────────────────────────────────────
Target ile ilişkisiz feature          →  Feature selection'da elenecek (Adım 6)
Yüksek feature-feature korelasyon     →  Birini çıkar veya PCA (Adım 6)
VIF > 10 tespit edildi                →  Ridge/Lasso kullan (Adım 9)
Non-lineer ilişki tespit edildi       →  Polinom feature ekle / Tree model seç
Spearman > Pearson fark               →  Lineer model yetersiz → non-linear model
Feature çıkarıldı                     →  Model karmaşıklığı azaldı → overfitting riski ↓
```

---

## 📌 6. FEATURE SELECTION TESTLERİ

### 🔍 NEDEN YAPILIR?

```
• GEREKSİZ FEATURE → Model gürültü öğrenir → OVERFITTING
• ÇOK FEATURE → "Curse of Dimensionality" → performans düşer
• AZ FEATURE → "Underfitting" → model yeterince öğrenemez
• DOĞRU FEATURE SETİ → Daha hızlı eğitim, daha iyi genelleme
• YORUMLANABİLİRLİK → Az feature = kolay açıklama
```

### 📊 SONUÇLAR NASIL YORUMLANIR?

```python
# ═══════════════════════════════════════════════════
# 6.1 F-TEST (Lineer İlişki Testi)
# ═══════════════════════════════════════════════════
from sklearn.feature_selection import f_regression, f_classif, SelectKBest

# Regression için
selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(train[num_cols], train['target'])

f_results = pd.DataFrame({
    'Feature': num_cols,
    'F_Score': selector.scores_,
    'p_value': selector.pvalues_
}).sort_values('F_Score', ascending=False)

for _, row in f_results.iterrows():
    col = row['Feature']
    f = row['F_Score']
    p = row['p_value']
    
    if p > 0.05:
        print(f"❌ {col}: F={f:.2f}, p={p:.4f} → İSTATİSTİKSEL OLARAK ANLAMSIZ")
        # → Bu feature target ile LİNEER ilişki taşımıyor
        # → Çıkarmayı düşün VEYA non-lineer ilişki kontrol et (MI)
    elif p < 0.001:
        print(f"✅ {col}: F={f:.2f}, p={p:.6f} → ÇOK ANLAMLI")
    else:
        print(f"🔶 {col}: F={f:.2f}, p={p:.4f} → Anlamlı ama zayıf")

# F-TEST NE SÖYLER?
# → "Bu feature'ın ortalaması target gruplarında farklı mı?"
# → F skoru yüksek = Feature, target'ı iyi ayırıyor
# → p < 0.05 = Bu fark istatistiksel olarak anlamlı
# → DİKKAT: Sadece LİNEER ilişkiyi ölçer!

# ═══════════════════════════════════════════════════
# 6.2 MUTUAL INFORMATION (Non-lineer İlişki)
# ═══════════════════════════════════════════════════
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

mi = mutual_info_regression(train[num_cols], train['target'], random_state=42)
mi_df = pd.DataFrame({
    'Feature': num_cols, 
    'MI': mi
}).sort_values('MI', ascending=False)

for _, row in mi_df.iterrows():
    col = row['Feature']
    mi_score = row['MI']
    
    if mi_score < 0.01:
        print(f"❌ {col}: MI={mi_score:.4f} → HİÇ BİLGİ TAŞIMIYOR → ÇIKAR")
    elif mi_score < 0.05:
        print(f"🔶 {col}: MI={mi_score:.4f} → Az bilgi → İzle")
    else:
        print(f"✅ {col}: MI={mi_score:.4f} → Bilgi taşıyor → BIRAK")

# MI NE SÖYLER?
# → "Bu feature target hakkında ne kadar BİLGİ taşıyor?"
# → Lineer + non-lineer ilişkileri yakalar
# → MI = 0 → tamamen bağımsız
# → MI yüksek → güçlü ilişki (herhangi bir formda)

# ⚠️  F-TEST vs MI KARŞILAŞTIRMASI:
for col in num_cols:
    f_p = f_results[f_results['Feature']==col]['p_value'].values[0]
    mi_val = mi_df[mi_df['Feature']==col]['MI'].values[0]
    
    if f_p > 0.05 and mi_val > 0.05:
        print(f"⚠️  {col}: F-test ANLAMSIZ ama MI YÜKSEK")
        print(f"   → NON-LİNEER ilişki var! → Tree model veya polinom feature")
    elif f_p < 0.05 and mi_val < 0.01:
        print(f"🤔 {col}: F-test ANLAMLI ama MI DÜŞÜK → Zayıf lineer ilişki")

# ═══════════════════════════════════════════════════
# 6.3 LASSO İLE FEATURE SEÇİMİ
# ═══════════════════════════════════════════════════
from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso.fit(train[num_cols], train['target'])

lasso_results = pd.DataFrame({
    'Feature': num_cols,
    'Coefficient': lasso.coef_,
    'Abs_Coef': np.abs(lasso.coef_)
}).sort_values('Abs_Coef', ascending=False)

print(f"Best alpha: {lasso.alpha_:.6f}")
# alpha büyükse → çok feature elendi → belki çok agresif
# alpha küçükse → az feature elendi → belki yeterli değil

eliminated = lasso_results[lasso_results['Abs_Coef'] == 0]['Feature'].tolist()
kept = lasso_results[lasso_results['Abs_Coef'] > 0]['Feature'].tolist()

print(f"\nKorunan ({len(kept)}): {kept}")
print(f"Elenen  ({len(eliminated)}): {eliminated}")

# LASSO NE SÖYLER?
# → L1 regularization katsayıları TAM SIFIR'a çeker
# → Katsayısı 0 olan feature = Modele katkısı YOK
# → Kalan feature'lar = En önemli feature'lar
# → DİKKAT: Çoklu doğrusallık varsa rastgele birini seçer!

# ═══════════════════════════════════════════════════
# 6.4 RFE (Recursive Feature Elimination)
# ═══════════════════════════════════════════════════
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor

rfecv = RFECV(
    estimator=RandomForestRegressor(n_estimators=100, random_state=42),
    step=1, cv=5,
    scoring='neg_root_mean_squared_error',
    min_features_to_select=5
)
rfecv.fit(train[num_cols], train['target'])

print(f"Optimal feature sayısı: {rfecv.n_features_}")
print(f"Seçilen feature'lar: {[c for c, s in zip(num_cols, rfecv.support_) if s]}")

# RFECV Eğrisi
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score'])+1),
         -rfecv.cv_results_['mean_test_score'])
plt.xlabel('Feature Sayısı')
plt.ylabel('CV RMSE')
plt.title('RFECV - Optimal Feature Sayısı')
plt.axvline(x=rfecv.n_features_, color='r', linestyle='--')
plt.show()

# YORUM:
# → Eğri belirli bir noktada düzleşiyorsa → o noktadaki feature sayısı optimal
# → Daha fazla feature eklemek performansı artırmıyorsa → overfitting riski

# ═══════════════════════════════════════════════════
# 6.5 KONSENSÜS (Birden Fazla Yöntem)
# ═══════════════════════════════════════════════════
# En güvenilir yaklaşım: Birden fazla yöntemin oy çokluğu

votes = pd.DataFrame(index=num_cols)
votes['F_test'] = f_results.set_index('Feature')['p_value'] < 0.05
votes['MI'] = mi_df.set_index('Feature')['MI'] > 0.01
votes['LASSO'] = ~lasso_results.set_index('Feature')['Abs_Coef'].eq(0)
votes['RFE'] = pd.Series(rfecv.support_, index=num_cols)
votes['Total'] = votes.sum(axis=1)

print(votes.sort_values('Total', ascending=False))

# YORUM:
# Total = 4 → Tüm yöntemler hemfikir → KESİNLİKLE KORU
# Total = 3 → Çoğunluk → Koru
# Total = 2 → Kararsız → Domain bilgisi ile karar ver
# Total = 1 → Çoğunluk atıyor → ÇIKAR (ama domain bilgisi kontrol et)
# Total = 0 → Tüm yöntemler reddediyor → KESİNLİKLE ÇIKAR

selected_features = votes[votes['Total'] >= 3].index.tolist()
print(f"\nFinal seçilen feature'lar ({len(selected_features)}): {selected_features}")
```

### 🔗 BİR SONRAKİ ADIMA ETKİSİ:

```
FEATURE SELECTION SONUCU              →  ETKİLEDİĞİ ADIM
─────────────────────────────────────────────────────────
Feature sayısı azaldı                 →  Model daha hızlı eğitilir (Adım 9)
                                       →  Overfitting riski azalır
                                       →  CV daha stabil sonuç verir
Gereksiz feature çıkarıldı            →  Model complexity ↓ → generalization ↑
Non-lineer ilişki tespit edildi       →  Tree-based model seçilmeli (Adım 9)
                                       →  Veya polinom/interaction feature ekle
LASSO feature eledi                   →  Ridge yerine LASSO/ElasticNet kullan
Çok az feature kaldı (<5)             →  Underfitting riski → daha fazla feature engineer
```

---

## 📌 7. SCALING & ENCODING

### 🔍 NEDEN YAPILIR?

```python
# SCALING NEDEN?
# ─────────────────────────────────────────────
# Feature 1 (Yaş):      18 - 65      (aralık: 47)
# Feature 2 (Maaş):     3000 - 50000 (aralık: 47000)
# 
# Scaling yapılmazsa:
# → Gradient descent maaşa çok daha hassas olur (büyük gradient)
# → KNN uzaklığı maaş tarafından domine edilir
# → SVM'in margin hesabı yanlış olur
# → Regularization (L1/L2) feature'lara eşit uygulanmaz
#
# Scaling GEREKMEyen modeller:
# → Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost
# → Çünkü split noktaları ölçekten bağımsız

# HANGİ SCALER NE ZAMAN?
"""
SCALER              KOŞUL                          FORMÜL
──────────────────────────────────────────────────────────
StandardScaler      Normal dağılım                  (x-μ)/σ
MinMaxScaler        Belirli aralık gerekli [0,1]    (x-min)/(max-min)
RobustScaler        Aykırı değer var                (x-median)/IQR
MaxAbsScaler        Sparse matrix                   x/|max|
"""

# ENCODING NEDEN?
# → ML modelleri sayı ister, "İstanbul", "Ankara" anlayamaz
# → Yanlış encoding = model yanlış ilişki öğrenir
#
# HANGİ ENCODING NE ZAMAN?
"""
ENCODING             KOŞUL                          DİKKAT
──────────────────────────────────────────────────────────────
OneHotEncoding       Nominal, az kategori (<15)      Boyut patlaması
OrdinalEncoding      Sıralı kategorik               Sıra bilgisi korunur
LabelEncoding        Target encoding (tree)          Sadece tree modeller
TargetEncoding       Yüksek kardinalite              Data leakage riski!
FrequencyEncoding    Yüksek kardinalite              Basit ve etkili
BinaryEncoding       Orta kardinalite                OneHot'tan az kolon
"""

# ⚠️  EN ÖNEMLİ KURAL:
# Scaler/Encoder SADECE TRAIN'den FIT edilir!
# Test'e sadece TRANSFORM uygulanır!

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train[num_cols])         # ← mean ve std TRAIN'den öğrenildi
train[num_cols] = scaler.transform(train[num_cols])  # train dönüştürüldü
test[num_cols]  = scaler.transform(test[num_cols])   # AYNI mean ve std ile

# NEDEN Test'ten fit edilmez?
# → Test "gelecek veri"dir
# → Gelecekten bilgi sızdırmak = overly optimistic results
# → Gerçek dünyada test verisi YOKTUR, sadece yeni veri gelir
```

### 🔗 BİR SONRAKİ ADIMA ETKİSİ:

```
SCALING/ENCODING SONUCU               →  ETKİLEDİĞİ ADIM
──────────────────────────────────────────────────────────
StandardScaler uygulandı              →  Gradient descent hızlı yakınsar (Adım 9)
                                       →  Regularization eşit uygulanır
OneHotEncoding yapıldı                →  Feature sayısı arttı → VIF tekrar kontrol?
                                       →  Sparse matrix → memory dikkat
RobustScaler kullanıldı               →  Aykırı değerler scaling'i bozmaz
Yanlış encoding                       →  Model sahte ilişki öğrenir → YANLIŞ TAHMİN
Pipeline'a eklendi                    →  Leakage önlendi → güvenilir CV (Adım 9)
```

---

## 📌 8. CLASS IMBALANCE (Classification)

### 🔍 NEDEN YAPILIR?

```python
# PROBLEM:
# 1000 hasta, 950 sağlıklı, 50 hasta
# Model her zaman "sağlıklı" derse → %95 accuracy!
# Ama HİÇBİR HASTAYI BULAMADI → işe yaramaz model

# Dengesizlik neden tehlikeli?
# → Model çoğunluk sınıfını EZBERLER
# → Azınlık sınıfını tamamen GÖRMEZDEN GELİR
# → Accuracy YANILTICI olur (yüksek ama anlamsız)
# → Recall (azınlık için) ≈ 0 olur

# Hangi metrikler güvenilir?
"""
DENGESİZ VERİDE METRİK SEÇİMİ:
──────────────────────────────────────────────
❌ Accuracy        → YANILTICI, kullanma
✅ F1-Score        → Precision-Recall dengesi
✅ AUC-ROC         → Threshold-bağımsız performans
✅ Precision-Recall AUC → Dengesiz veride ROC'dan daha bilgilendirici
✅ Cohen's Kappa   → Şansa göre düzeltilmiş
✅ MCC             → Tüm confusion matrix'i kullanır
"""
```

### 📊 SONUÇLAR NASIL YORUMLANIR?

```python
# ═══════════════════════════════════════════════════
# 8.1 DENGESİZLİK SEVİYESİ
# ═══════════════════════════════════════════════════
class_dist = train['target'].value_counts(normalize=True)
ratio = class_dist.min() / class_dist.max()

if ratio > 0.5:
    print(f"✅ Dengeli (ratio={ratio:.4f}) → Özel işlem gerekmez")
    strategy = "none"
elif ratio > 0.2:
    print(f"🔶 Hafif dengesiz (ratio={ratio:.4f}) → class_weight='balanced'")
    strategy = "class_weight"
elif ratio > 0.05:
    print(f"🔴 Dengesiz (ratio={ratio:.4f}) → SMOTE + class_weight")
    strategy = "smote"
else:
    print(f"🔴🔴 Aşırı dengesiz (ratio={ratio:.4f}) → SMOTE + Threshold + Ensemble")
    strategy = "advanced"

# ═══════════════════════════════════════════════════
# 8.2 SMOTE UYGULAMASI ve YORUMU
# ═══════════════════════════════════════════════════
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, sampling_strategy=0.5)
# sampling_strategy=0.5 → azınlık/çoğunluk = 0.5 olacak şekilde üret
# sampling_strategy='auto' → tam eşitlik

X_res, y_res = smote.fit_resample(X_train, y_train)

print(f"Önce:  {dict(pd.Series(y_train).value_counts())}")
print(f"Sonra: {dict(pd.Series(y_res).value_counts())}")

# ⚠️  DİKKAT: SMOTE SADECE TRAIN'E UYGULANIR!
# → Test'e ASLA uygulanmaz (gerçek dünyayı temsil etmeli)
# → Cross-validation'da her fold'un TRAIN kısmına ayrı uygulanır
# → CV dışında SMOTE yapmak = DATA LEAKAGE

# Doğru yol: imblearn Pipeline kullanmak
from imblearn.pipeline import Pipeline as ImbPipeline

imb_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(random_state=42))
])
# Bu pipeline CV içinde her fold'a ayrı SMOTE uygular ✅
```

### 🔗 BİR SONRAKİ ADIMA ETKİSİ:

```
IMBALANCE KARARI                      →  ETKİLEDİĞİ ADIM
──────────────────────────────────────────────────────────
SMOTE uygulandı                       →  Train boyutu arttı → CV daha uzun
class_weight kullanılacak             →  Model parametrelerine eklenir (Adım 9)
Accuracy kullanılmayacak              →  Scoring=F1/AUC olacak (Adım 9)
Threshold optimize edilecek           →  Adım 14'te optimal threshold aranır
StratifiedKFold zorunlu               →  CV stratejisi belirlendi (Adım 9)
```

---

## 📌 9. CROSS-VALIDATION & MODEL KURMA

### 🔍 NEDEN YAPILIR?

```
• Modeli TEK BİR train-validation split ile değerlendirmek GÜVENİLMEZ
  → O split'e özel olabilir (şanslı/şanssız)
  
• CV ile K farklı split'te test eder → ORTALAMA performansı ölçersin
  → Standart sapma düşükse → model STABİL
  → Standart sapma yüksekse → model GÜVENSIZ
  
• Doğru CV stratejisi seçmezsen sonuçlar YANILTICI olur:
  → Classification'da normal KFold → bazı fold'larda azınlık sınıf 0 olabilir
  → Zaman serisinde shuffle=True → gelecek bilgisi sızar
  → Gruplu veride normal CV → aynı grup train+test'te olabilir
```

### 📊 SONUÇLAR NASIL YORUMLANIR?

```python
# ═══════════════════════════════════════════════════
# 9.1 CV STRATEJİSİ SEÇİMİ (ÇOK KRİTİK!)
# ═══════════════════════════════════════════════════

"""
VERİ TİPİ              CV STRATEJİSİ              NEDEN
──────────────────────────────────────────────────────────
Regression              KFold(5, shuffle=True)      Standart
Classification          StratifiedKFold(5)          Sınıf dengesini korur
Zaman serisi            TimeSeriesSplit(5)          Gelecek bilgisi sızmaz
Gruplu veri             GroupKFold(5)               Aynı grup ayrılmaz
Küçük veri (n<500)      RepeatedStratifiedKFold     Daha güvenilir tahmin
Çok küçük veri (n<100)  LeaveOneOut                 Her gözlem test olur
"""

# ═══════════════════════════════════════════════════
# 9.2 MODEL KARŞILAŞTIRMA ve YORUMLAMA
# ═══════════════════════════════════════════════════
from sklearn.model_selection import cross_validate, KFold

cv = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'LinearReg': LinearRegression(),
    'Ridge':     Ridge(alpha=1.0),
    'Lasso':     Lasso(alpha=0.1),
    'RF':        RandomForestRegressor(n_estimators=100, random_state=42),
    'GBM':       GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    scores = cross_validate(model, X_train, y_train, cv=cv,
                            scoring='neg_root_mean_squared_error',
                            return_train_score=True)
    
    train_rmse = -scores['train_score'].mean()
    val_rmse   = -scores['test_score'].mean()
    val_std    = scores['test_score'].std()
    gap        = val_rmse - train_rmse
    
    results[name] = {
        'Train_RMSE': train_rmse,
        'Val_RMSE': val_rmse,
        'Val_Std': val_std,
        'Gap': gap
    }

results_df = pd.DataFrame(results).T

# YORUM TABLOSU
for name, row in results_df.iterrows():
    print(f"\n{'='*50}")
    print(f"MODEL: {name}")
    print(f"Train RMSE: {row['Train_RMSE']:.4f}")
    print(f"Val   RMSE: {row['Val_RMSE']:.4f} ± {row['Val_Std']:.4f}")
    print(f"Gap:        {row['Gap']:.4f}")
    
    # ─── GAP YORUMU (Train vs Val farkı) ───
    gap_ratio = row['Val_RMSE'] / row['Train_RMSE'] if row['Train_RMSE'] > 0 else 0
    
    if gap_ratio < 1.1:
        print("✅ OVERFIT YOK → İyi genelleme")
    elif gap_ratio < 1.3:
        print("🔶 HAFİF OVERFIT → Regularization artırılabilir")
    else:
        print("🔴 CİDDİ OVERFIT → Model çok karmaşık!")
        print("   ÇÖZÜMLER:")
        print("   → max_depth azalt")
        print("   → n_estimators azalt")  
        print("   → min_samples_leaf artır")
        print("   → Feature sayısını azalt")
        print("   → Regularization ekle (Ridge/Lasso)")
    
    # ─── STD YORUMU (Stabilite) ───
    cv_coeff = row['Val_Std'] / row['Val_RMSE']
    if cv_coeff < 0.1:
        print("✅ Model STABİL (düşük varyans)")
    elif cv_coeff < 0.2:
        print("🔶 ORTA stabilite → Daha fazla veri yardımcı olabilir")
    else:
        print("🔴 İNSTABİL → Her fold'da farklı sonuç → GÜVENİLMEZ")
        print("   ÇÖZÜMLER:")
        print("   → Daha fazla fold kullan (10-fold)")
        print("   → RepeatedKFold kullan")
        print("   → Model basitleştir")

# ─── EN İYİ MODEL SEÇİMİ ───
best_model = results_df['Val_RMSE'].idxmin()
print(f"\n🏆 En iyi model: {best_model}")
print(f"   Val RMSE: {results_df.loc[best_model, 'Val_RMSE']:.4f}")

# ─── ÖNEMLİ KARAR: LİNEER vs TREE ───
linear_best = results_df.loc[['LinearReg','Ridge','Lasso'], 'Val_RMSE'].min()
tree_best   = results_df.loc[['RF','GBM'], 'Val_RMSE'].min()

if tree_best < linear_best * 0.95:
    print("🌳 Tree modeller belirgin şekilde daha iyi → Non-lineer ilişkiler var")
    print("   → Lineer model varsayım testlerine GEREK YOK")
    print("   → Hiperparametre optimizasyonu tree model üzerinde yapılacak")
elif linear_best < tree_best * 0.95:
    print("📏 Lineer modeller daha iyi → İlişki doğrusal")
    print("   → Model varsayım testleri YAPILMALI (Adım 11)")
    print("   → Ridge/Lasso seçimi → VIF'e göre karar")
else:
    print("≈ Performans benzer → Ensemble veya stacking düşünülebilir")
```

### 🔗 BİR SONRAKİ ADIMA ETKİSİ:

```
CV SONUCU                              →  ETKİLEDİĞİ ADIM
──────────────────────────────────────────────────────────
Overfit tespit edildi                  →  Hiperparametre ayarla (Adım 12)
                                        →  Regularization ekle
                                        →  Feature selection agresifleştir
Model instabil                         →  Daha fazla veri / daha basit model
Lineer model kazandı                   →  Varsayım testleri yapılmalı (Adım 11)
Tree model kazandı                     →  Varsayım testleri ATLANIR
                                        →  Hiperparametre optimizasyonu (Adım 12)
En iyi model belirlendi                →  Bu model üzerinde fine-tuning
```

---

## 📌 10. LEARNING & VALIDATION CURVE

### 📊 SONUÇLAR NASIL YORUMLANIR?

```python
# ═══════════════════════════════════════════════════
# 10.1 LEARNING CURVE YORUMU
# ═══════════════════════════════════════════════════

"""
SENARYO 1: HIGH BIAS (Underfitting)
─────────────────────────────────────
Train RMSE:  ████████░░ (yüksek)
Val RMSE:    █████████░ (yüksek, train'e yakın)

→ İkisi de yüksek ve birbirine yakın
→ Model yeteri kadar öğreneMİYOR
→ ÇÖZÜM: Daha karmaşık model, daha fazla feature, polinom özellikler

SENARYO 2: HIGH VARIANCE (Overfitting)  
─────────────────────────────────────
Train RMSE:  ██░░░░░░░░ (çok düşük)
Val RMSE:    ████████░░ (yüksek, train'den uzak)

→ Train çok iyi, Val kötü → EZBERLEME
→ ÇÖZÜM: Daha fazla veri, regularization, feature azaltma, pruning

SENARYO 3: İDEAL
─────────────────────────────────────
Train RMSE:  ███░░░░░░░ (düşük)
Val RMSE:    ████░░░░░░ (düşük, train'e yakın)

→ İkisi de düşük ve birbirine yakın
→ Model iyi genelleme yapıyor ✅

SENARYO 4: DAHA FAZLA VERİ GEREKLİ
─────────────────────────────────────
Train RMSE:  ██░░░░░░░░ → ███░░░░░░░ (artıyor)
Val RMSE:    ████████░░ → ██████░░░░ (azalıyor ama hâlâ uzak)

→ Veri arttıkça ikisi birbirine yaklaşıyor ama henüz yakınsamadı
→ DAHA FAZLA VERİ TOPLA
"""

# ═══════════════════════════════════════════════════
# 10.2 VALIDATION CURVE YORUMU
# ═══════════════════════════════════════════════════

"""
Örnek: max_depth parametresi

max_depth=2:  Train ████░░  Val ███░░░  → UNDERFITTING (çok basit)
max_depth=5:  Train █████░  Val █████░  → OPTİMAL ✅
max_depth=20: Train ██████  Val ███░░░  → OVERFITTING (çok karmaşık)

→ Val score'un en yüksek olduğu parametre değeri = OPTİMAL
→ Val score artmıyor ama Train artmaya devam ediyorsa → OVERFIT başlıyor
→ O noktadan sonra parametre artırmak ZARARLI
"""
```

---

## 📌 11. MODEL VARSAYIM TESTLERİ (Lineer Model)

### 🔍 NEDEN YAPILIR?

```
SADECE LİNEER MODELLER İÇİN GEÇERLİ!
Tree-based modellerde bu testler ANLAMSIZDIR.

Lineer regresyonun 5 varsayımı:
1. Doğrusallık:        Y = β₀ + β₁X₁ + ... + ε (doğrusal ilişki)
2. Bağımsızlık:        Gözlemler birbirinden bağımsız
3. Normallik:          Kalıntılar (residuals) normal dağılır
4. Homoskedastisite:   Kalıntıların varyansı sabit
5. Multicollinearity:  Feature'lar birbirine bağımlı değil

Bu varsayımlar BOZULURSA:
→ Katsayılar (β) YANLI (biased) olur
→ p-değerleri GÜVENSIZ olur
→ Güven aralıkları YANLŞ olur
→ Tahminler SİSTEMATİK OLARAK HATALI olur
```

### 📊 SONUÇLAR NASIL YORUMLANIR?

```python
# ═══════════════════════════════════════════════════
# 11.1 KALINTILAR NORMALLİK TESTİ
# ═══════════════════════════════════════════════════
import statsmodels.api as sm
from scipy import stats

X_sm = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_sm).fit()
residuals = ols_model.resid

# Shapiro-Wilk
stat, p = stats.shapiro(residuals[:5000])
print(f"Shapiro-Wilk: p={p:.4f}")

# Jarque-Bera (summary'de zaten var)
stat, p = stats.jarque_bera(residuals)
print(f"Jarque-Bera: p={p:.4f}")

"""
YORUM:
p > 0.05  → Kalıntılar NORMAL dağılıyor ✅
             → Katsayıların p-değerleri güvenilir
             → Güven aralıkları geçerli

p < 0.05  → Kalıntılar NORMAL DEĞİL ❌
             ÇÖZÜMLER:
             1. Target'a log/boxcox dönüşümü uygula
             2. Aykırı değerleri çıkar/kırp
             3. Eksik non-lineer ilişki var → polinom feature ekle
             4. Bootstrap güven aralıkları kullan
             5. Robust regression (RLM) kullan
             
⚠️  ÖNEMLİ: n > 30 ise Central Limit Theorem sayesinde
             normallik varsayımı rahatlatılabilir.
             n > 500 ise Shapiro-Wilk çok hassas olur,
             küçük sapmaları bile yakalar → Q-Q plot'a bak
"""

# ═══════════════════════════════════════════════════
# 11.2 HETEROSKEDASTİSİTE TESTİ
# ═══════════════════════════════════════════════════
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

bp_stat, bp_p, _, _ = het_breuschpagan(residuals, ols_model.model.exog)
print(f"Breusch-Pagan: p={bp_p:.4f}")

"""
YORUM:
p > 0.05  → Varyans SABİT (homoskedastik) ✅
             → OLS katsayıları verimli (efficient)
             → Standart hatalar doğru

p < 0.05  → Varyans DEĞİŞKEN (heteroskedastik) ❌
             → OLS katsayıları hâlâ unbiased AMA
             → Standart hatalar YANLIŞ → p-değerleri güvensiz
             → Güven aralıkları yanlış
             
             ÇÖZÜMLER:
             1. Robust standart hatalar kullan (HC3):
                robust = ols_model.get_robustcov_results(cov_type='HC3')
             2. WLS (Weighted Least Squares) kullan
             3. Target'a log dönüşümü uygula
             4. Heteroskedastisiteye dayanıklı model (GLS)
"""

# ═══════════════════════════════════════════════════
# 11.3 OTOKORELASYON TESTİ
# ═══════════════════════════════════════════════════
from statsmodels.stats.stattools import durbin_watson

dw = durbin_watson(residuals)
print(f"Durbin-Watson: {dw:.4f}")

"""
YORUM:
DW ≈ 2.0         → Otokorelasyon YOK ✅ (ideal: 1.5-2.5)
DW → 0           → POZİTİF otokorelasyon ❌
                     → Ardışık kalıntılar aynı yönde
DW → 4           → NEGATİF otokorelasyon ❌
                     → Ardışık kalıntılar zıt yönde

DW < 1.5 veya DW > 2.5 → SORUN VAR

NEDEN ÖNEMLİ?
→ Otokorelasyon varsa standart hatalar KÜÇÜK tahmin edilir
→ p-değerleri olduğundan KÜÇÜK çıkar → yanlış "anlamlı" sonuç
→ Zaman serisi verisinde ÇOK YAYGIN

ÇÖZÜMLER:
1. Zaman bazlı feature'lar ekle (lag, trend)
2. GLS (Generalized Least Squares) kullan
3. HAC (Heteroskedasticity & Autocorrelation Consistent) standart hatalar
4. ARIMA modellerine geç
"""

# ═══════════════════════════════════════════════════
# 11.4 DOĞRUSALLIK TESTİ (Ramsey RESET)
# ═══════════════════════════════════════════════════
from statsmodels.stats.diagnostic import linear_reset

reset = linear_reset(ols_model, power=3)
print(f"Ramsey RESET: F={reset.fvalue:.4f}, p={reset.pvalue:.4f}")

"""
YORUM:
p > 0.05  → Model spesifikasyonu DOĞRU ✅
             → Doğrusal model yeterli
             → Eksik non-lineer terim yok

p < 0.05  → Model spesifikasyonu YANLIŞ ❌
             → Non-lineer ilişki var ama model doğrusal
             ÇÖZÜMLER:
             1. Polinom feature'lar ekle (x², x³, x₁*x₂)
             2. Tree-based modele geç
             3. GAM (Generalized Additive Model) kullan
             4. Spline regresyon kullan
"""

# ═══════════════════════════════════════════════════
# 11.5 ETKİLİ GÖZLEMLER (Cook's Distance)
# ═══════════════════════════════════════════════════
influence = ols_model.get_influence()
cooks_d = influence.cooks_distance[0]

n = len(y_train)
influential = (cooks_d > 4/n).sum()
print(f"Etkili gözlem sayısı (Cook's D > 4/n): {influential}")

"""
YORUM:
Cook's Distance NEDİR?
→ Her gözlemin modelden ÇIKARILDIĞINDA tüm tahminlerin
  ne kadar değişeceğini ölçer

Cook's D < 4/n   → Normal gözlem ✅
Cook's D > 4/n   → Etkili gözlem → İNCELE
Cook's D > 1     → ÇOK ETKİLİ → Muhtemelen aykırı değer veya hata

ÇÖZÜMLER:
1. Etkili gözlemleri incele → veri hatası mı?
2. Çıkar ve modeli tekrar kur → sonuç ne kadar değişti?
3. Robust regression kullan (Huber, Bisquare)
4. Feature'ları dönüştür
"""
```

### 🔗 BİR SONRAKİ ADIMA ETKİSİ:

```
VARSAYIM TESTİ SONUCU                  →  ETKİLEDİĞİ ADIM
──────────────────────────────────────────────────────────
Normallik sağlanmıyor                  →  Log dönüşümü → Adım 4'e geri dön
                                        →  veya Bootstrap kullan
Heteroskedastisite var                 →  Robust SE kullan → Model güncelle
                                        →  veya WLS'e geç
Otokorelasyon var                      →  Zaman feature'ları ekle
                                        →  veya GLS kullan
Doğrusallık sağlanmıyor                →  Tree model'e geç → Adım 12
                                        →  veya polinom feature ekle
Etkili gözlem var                      →  Outlier analizi → Adım 3'e geri dön
                                        →  veya robust regression
TÜM VARSAYIMLAR SAĞLANIYOR            →  Lineer model güvenle kullanılabilir ✅
                                        →  Hiperparametre optimizasyonu (Adım 12)
```

---

## 📌 12. HİPERPARAMETRE OPTİMİZASYONU

### 📊 SONUÇLAR NASIL YORUMLANIR?

```python
# ═══════════════════════════════════════════════════
# 12.1 OPTİMİZASYON SONUCU YORUMLAMA
# ═══════════════════════════════════════════════════

# GridSearch / Optuna sonrası:
print(f"Best params: {best_params}")
print(f"Best CV RMSE: {best_score:.4f}")
print(f"Default CV RMSE: {default_score:.4f}")
print(f"İyileşme: {(default_score - best_score)/default_score*100:.2f}%")

"""
İYİLEŞME YORUMU:
%0-2   → Marjinal iyileşme → Varsayılan parametreler zaten iyi
          → Optimizasyona daha fazla zaman harcama
%2-10  → Makul iyileşme → Optimizasyon işe yaradı ✅
%10+   → Büyük iyileşme → Varsayılan parametreler çok uygunsuzmuş
          → Belki daha geniş aralıkta aramak lazım
"""

# ═══════════════════════════════════════════════════
# 12.2 PARAMETRELERİN YORUMU
# ═══════════════════════════════════════════════════

"""
Random Forest / GBM İÇİN:
──────────────────────────────────────────────────
max_depth düşük çıktıysa (3-5)   → Veri basit, derin ağaç gereksiz
max_depth yüksek çıktıysa (15+)  → Karmaşık ilişkiler var
                                    → Ama overfit riski artmış olabilir

n_estimators çok yüksek (500+)    → Daha fazla artırmak diminishing returns
                                    → Eğitim süresi uzar ama performans artmaz

min_samples_leaf yüksek (10+)     → Regularization etkisi → overfit önleniyor
min_samples_leaf = 1              → Overfitting riski yüksek

learning_rate düşük (0.01-0.05)   → Yavaş öğreniyor ama daha iyi genelleme
                                    → n_estimators yüksek olmalı (trade-off)
learning_rate yüksek (0.1+)       → Hızlı ama overfitting riski
"""

# ═══════════════════════════════════════════════════
# 12.3 OPTİMİZASYON SONRASI OVERFIT KONTROLÜ
# ═══════════════════════════════════════════════════

# Optimize edilmiş model ile CV tekrar
optimized_model = GradientBoostingRegressor(**best_params, random_state=42)
cv_scores = cross_validate(optimized_model, X_train, y_train, cv=5,
                           scoring='neg_root_mean_squared_error',
                           return_train_score=True)

train_rmse = -cv_scores['train_score'].mean()
val_rmse   = -cv_scores['test_score'].mean()
gap = val_rmse - train_rmse

print(f"Optimized Train RMSE: {train_rmse:.4f}")
print(f"Optimized Val RMSE:   {val_rmse:.4f}")
print(f"Gap: {gap:.4f}")

if gap / train_rmse > 0.2:
    print("⚠️  Optimizasyon sonrası OVERFIT artmış! → Regularize et")
else:
    print("✅ Optimizasyon başarılı, overfit yok")
```

---

## 📌 13. MODEL KARŞILAŞTIRMA TESTLERİ

### 🔍 NEDEN YAPILIR?

```
Model A: CV RMSE = 10.23 ± 0.45
Model B: CV RMSE = 10.31 ± 0.52

Soru: Model A GERÇEKTEN daha mı iyi, yoksa bu fark TESADÜF mü?

CV skorları rastgele değişkenlerdir. Her çalıştırmada farklı çıkar.
İstatistiksel test yapmadan "daha düşük = daha iyi" demek YANLIŞ olabilir.
```

### 📊 SONUÇLAR NASIL YORUMLANIR?

```python
# ═══════════════════════════════════════════════════
# 13.1 PAIRED T-TEST (CV SKORLARI)
# ═══════════════════════════════════════════════════

# Aynı CV fold'larında iki modelin skorları
cv = KFold(n_splits=10, shuffle=True, random_state=42)
scores_A = cross_val_score(model_A, X_train, y_train, cv=cv,
                           scoring='neg_root_mean_squared_error')
scores_B = cross_val_score(model_B, X_train, y_train, cv=cv,
                           scoring='neg_root_mean_squared_error')

# Önce farkların normalliğini kontrol et
diff = scores_A - scores_B
stat, p_norm = stats.shapiro(diff)

if p_norm > 0.05:
    # Normal → Paired t-test
    stat, p = stats.ttest_rel(scores_A, scores_B)
    test_name = "Paired t-test"
else:
    # Normal değil → Wilcoxon
    stat, p = stats.wilcoxon(scores_A, scores_B)
    test_name = "Wilcoxon signed-rank"

print(f"{test_name}: stat={stat:.4f}, p={p:.4f}")

"""
YORUM:
p > 0.05  → İki model arasında İSTATİSTİKSEL FARK YOK
             → Daha basit/hızlı modeli seç (Occam's Razor)
             → Veya ensemble/stacking düşün

p < 0.05  → Modeller arasında ANLAMLI FARK VAR
             → Daha iyi olan modeli seç
             → p < 0.01 ise çok güçlü kanıt
             → p < 0.001 ise kesin fark

⚠️  DİKKAT: CV fold sayısı az (5-10) olduğunda
             t-test'in gücü düşüktür.
             Gerçek fark olsa bile tespit edemeyebilir.
             → RepeatedKFold kullan (n_repeats=10)
"""

# ═══════════════════════════════════════════════════
# 13.2 BİRDEN FAZLA MODEL (3+): Friedman + Nemenyi
# ═══════════════════════════════════════════════════

all_scores = np.column_stack([scores_A, scores_B, scores_C, scores_D])
stat, p = stats.friedmanchisquare(*all_scores.T)
print(f"Friedman: chi2={stat:.4f}, p={p:.4f}")

"""
YORUM:
p > 0.05  → Modeller arasında FARK YOK
             → En basit modeli seç
             
p < 0.05  → EN AZ İKİ model arasında fark var
             → HANGİLERİ? → Post-hoc Nemenyi testi
"""

if p < 0.05:
    import scikit_posthocs as sp
    nemenyi = sp.posthoc_nemenyi_friedman(all_scores)
    print(nemenyi)
    # p < 0.05 olan çiftler = İSTATİSTİKSEL OLARAK FARKLI
    # p > 0.05 olan çiftler = FARK YOK
```

### 🔗 BİR SONRAKİ ADIMA ETKİSİ:

```
KARŞILAŞTIRMA SONUCU                   →  ETKİLEDİĞİ ADIM
──────────────────────────────────────────────────────────
Modeller arasında fark yok             →  En basit/hızlı modeli seç
                                        →  Yorumlanabilirlik ön planda
Bir model belirgin iyi                 →  O model final model olarak seçilir
                                        →  Hiperparametre fine-tuning yapılır
Fark marjinal                          →  Ensemble/Stacking düşün
                                        →  Veya business requirement'a göre seç
```

---

## 📌 14-15. FİNAL TAHMİN & DEĞERLENDİRME

### 📊 SONUÇLAR NASIL YORUMLANIR?

```python
# ═══════════════════════════════════════════════════
# 14.1 REGRESSION METRİKLERİ YORUMU
# ═══════════════════════════════════════════════════

"""
METRİK       NE SÖYLER                              İYİ DEĞERİ
───────────────────────────────────────────────────────────────────
RMSE         Ortalama hata büyüklüğü (büyük            Düşük
             hatalara daha hassas)                      (domain'e göre)
             
MAE          Ortalama mutlak hata (robust)              Düşük
             
MAPE         Yüzdesel hata → scale-independent          < %10 iyi
                                                        < %20 kabul

MedAE        Medyan mutlak hata → aykırılara robust     Düşük

R²           Açıklanan varyans oranı                    > 0.7 iyi
             (1 = mükemmel, 0 = ortalama kadar,         > 0.9 çok iyi
              negatif = ortalamadan kötü!)               < 0 → model çöp

Adj R²       Feature sayısı ile cezalandırılmış R²      > 0.7 iyi
             R² ile arasındaki fark büyükse              
             → gereksiz feature var                      

Max Error    En kötü tahmin                             İş gereksinimi
"""

# ═══════════════════════════════════════════════════
# REGRESSION YORUM ÖRNEĞİ
# ═══════════════════════════════════════════════════
"""
RMSE  = 15000      (Ev fiyatı tahmini, ortalama 300K)
MAE   = 12000      
MAPE  = 5.2%       → Ortalama %5.2 hata → İYİ ✅
R²    = 0.87       → Varyansın %87'si açıklanıyor → İYİ ✅
Adj R² = 0.85      → R² ile fark 0.02 → feature sayısı uygun ✅
Max Error = 120000 → En kötü tahmin 120K şaştı → İNCELE
                     (muhtemelen lüks ev veya aykırı değer)
                     
RMSE > MAE → Büyük hataların etkisi var (bazı evlerde çok şaşıyor)
→ Aykırı değerleri tekrar kontrol et
→ Log dönüşümü denenmeli

CV RMSE: 15200 ± 800
Test RMSE: 15000
→ CV ile test uyumlu ✅ → Model güvenilir
→ Eğer Test RMSE >> CV RMSE → Data drift veya overfitting sorunu
"""

# ═══════════════════════════════════════════════════
# 14.2 CLASSIFICATION METRİKLERİ YORUMU
# ═══════════════════════════════════════════════════

"""
METRİK       NE SÖYLER                              YORUMU
───────────────────────────────────────────────────────────────────
Accuracy     Doğru tahmin oranı                      Dengesizde YANILTICI!

Precision    "Pozitif dediklerimin kaçı gerçekten     Yanlış alarm maliyeti
             pozitif?"                                yüksekse ÖNEMLİ
             
Recall       "Gerçek pozitiflerin kaçını              Kaçırma maliyeti
             yakaladım?"                              yüksekse ÖNEMLİ
             
F1           Precision-Recall harmonik ortalaması     Dengeli metrik

AUC-ROC      Threshold-bağımsız ayırt edicilik        > 0.7 kabul
             0.5 = rastgele, 1.0 = mükemmel           > 0.8 iyi
                                                      > 0.9 çok iyi

PR-AUC       Dengesiz veride ROC'dan daha bilgilen.   Baseline = pozitif oran

MCC          -1 ile +1 arası, tüm confusion matrix    > 0.3 kabul
             0 = rastgele                              > 0.5 iyi

Brier Score  Olasılık kalibrasyonu                    < 0.25 iyi
             0 = mükemmel kalibrasyon                  < 0.1 çok iyi

Cohen Kappa  Şansa göre düzeltilmiş uyum              > 0.6 iyi
"""

# ═══════════════════════════════════════════════════
# CLASSIFICATION YORUM ÖRNEĞİ (Hastalık Tespiti)
# ═══════════════════════════════════════════════════
"""
Confusion Matrix:
              Predicted
              Neg    Pos
Actual Neg    920     30    → 30 sağlıklı kişiye yanlışlıkla "hasta" dedik (FP)
Actual Pos     8     42    → 8 hastayı KAÇIRDIK! (FN) ← CİDDİ!

Accuracy  = 0.962  → %96 doğru → İYİ GİBİ GÖRÜNÜYOR ama...
Precision = 0.583  → "Hasta" dediğimizin %58'i gerçekten hasta
Recall    = 0.840  → Hastaların %84'ünü yakaladık, %16'sını KAÇIRDIK!
F1        = 0.689  → Dengeli bakış
AUC       = 0.921  → Model genel olarak İYİ ayırt ediyor

YORUMLAR:
1. Recall = 0.84 → Tıbbi uygulamada YETERSİZ
   → 100 hastanın 16'sı teşhis edilmiyor → TEHLİKELİ
   → ÇÖZÜM: Threshold'u 0.5'ten 0.3'e düşür → Recall artar
   
2. Precision = 0.58 → Her "hasta" dediğimizin %42'si sağlıklı
   → Gereksiz tedavi maliyeti → ama hayat kurtarmaktan daha az önemli
   
3. KARAR: Tıbbi uygulamada RECALL > PRECISION öncelikli
   → Threshold düşürülecek
"""

# ═══════════════════════════════════════════════════
# 14.3 THRESHOLD OPTİMİZASYONU YORUMU
# ═══════════════════════════════════════════════════

"""
Threshold = 0.5 (default):
  Precision = 0.58, Recall = 0.84, F1 = 0.69

Threshold = 0.3 (düşürüldü):
  Precision = 0.42, Recall = 0.96, F1 = 0.59
  → Neredeyse tüm hastaları yakalıyor ama yanlış alarm arttı

Threshold = 0.7 (yükseltildi):
  Precision = 0.78, Recall = 0.62, F1 = 0.69
  → "Hasta" dediğimize güvenebiliriz ama çok hasta kaçırıyoruz

İŞ GEREKSİNİMİNE GÖRE SEÇ:
• Fraud detection    → Recall öncelikli (kaçırmak pahalı)
• Spam filtresi      → Precision öncelikli (normal maili spam'e atma)
• Hastalık tespiti   → Recall öncelikli (hasta kaçırma)
• Reklam targeting   → Precision öncelikli (bütçe optimizasyonu)
"""

# ═══════════════════════════════════════════════════
# 15.1 OVERFIT KONTROLÜ YORUMU
# ═══════════════════════════════════════════════════

"""
SENARYO    Train RMSE   Test RMSE   Ratio    YORUM
───────────────────────────────────────────────────────
İdeal      10.2         10.8        1.06     ✅ İyi genelleme
Hafif OF   10.2         13.5        1.32     🔶 Hafif overfit
Ciddi OF   2.1          15.3        7.29     🔴 Ciddi overfit → Model çöp!
Underfit   18.5         19.2        1.04     🔴 Hem train hem test kötü

Ratio < 1.1  → Mükemmel genelleme
Ratio 1.1-1.3 → Kabul edilebilir hafif overfit
Ratio > 1.3  → CİDDİ OVERFIT → Model karmaşıklığını azalt
"""
```

---

## 📌 16. MODEL AÇIKLANABILIRLIK (SHAP)

### 📊 SONUÇLAR NASIL YORUMLANIR?

```python
"""
SHAP Summary Plot YORUMU:

Feature: Yaş
•••••••• (kırmızı noktalar sağda, mavi solda)
→ Yaş ARTTIKÇA (kırmızı) SHAP değeri POZİTİF → Target'ı ARTIRIYOR
→ Yaş AZALDIKÇA (mavi) SHAP değeri NEGATİF → Target'ı AZALTIYOR
→ "Yaş arttıkça ev fiyatı artıyor" → Mantıklı ✅

Feature: Oda Sayısı
•••••••• (karışık, pattern yok)
→ Oda sayısı ile target arasında NET BİR İLİŞKİ YOK
→ Bu feature'ın etkisi değişken ve düzensiz
→ Belki interaction var (oda sayısı + konum birlikte etkili)

SHAP Değeri = 0 → Bu feature bu tahmine HİÇ ETKİ ETMİYOR
SHAP Değeri > 0 → Bu feature bu tahmini YUKARI çekiyor
SHAP Değeri < 0 → Bu feature bu tahmini AŞAĞI çekiyor

|SHAP| büyükse → Feature ÇOK ETKİLİ
|SHAP| küçükse → Feature AZ ETKİLİ
"""
```

---

## 📌 17-18. GÜVEN ARALIĞI & KALİBRASYON

### 📊 SONUÇLAR NASIL YORUMLANIR?

```python
"""
17. TAHMİN GÜVEN ARALIĞI YORUMU:

Tahmin: 250.000 TL   CI: [220.000, 280.000]
→ "%95 olasılıkla gerçek değer 220K-280K arasında"
→ Aralık DAR → Model EMİN → Güvenilir tahmin
→ Aralık GENİŞ → Model EMİN DEĞİL → Dikkatli ol

Bazı tahminlerde CI çok geniş çıkıyorsa:
→ O bölgede yeterli train verisi yok
→ Extrapolation yapılıyor olabilir
→ Feature kombinasyonu nadir


18. KALİBRASYON YORUMU:

Kalibrasyon Eğrisi:
• Çizgi 45° doğrusuna YAKIN → Model KALİBRE ✅
  → "%70 olasılık" dediğinde gerçekten %70'i pozitif
  
• Çizgi 45° ÜZERİNDE → Model ÇEKİNGEN (underconfident)
  → "%70 olasılık" dediğinde aslında %85'i pozitif
  
• Çizgi 45° ALTINDA → Model AŞIRI GÜVENLI (overconfident)
  → "%70 olasılık" dediğinde aslında %50'si pozitif → TEHLİKELİ

Brier Score:
< 0.1   → Mükemmel kalibrasyon
0.1-0.25 → İyi
> 0.25  → Kötü → Platt Scaling veya Isotonic Regression uygula
"""
```

---

## 📌 19. DATA LEAKAGE KONTROLÜ

### 🔍 NEDEN YAPILIR?

```python
"""
DATA LEAKAGE = En tehlikeli hata!

CV'de R² = 0.99 ama gerçek dünyada performans çöp?
→ Muhtemelen DATA LEAKAGE var!

LEAKAGE TİPLERİ:
──────────────────────────────────────────────────────
1. TARGET LEAKAGE:
   Feature'lardan biri target'tan türetilmiş
   Örnek: "Tedavi sonucu" feature'ı → "Hayatta kaldı mı?" target'ı
   → Tedavi sonucu ancak sonuç belli olduktan sonra bilinir!
   
2. TRAIN-TEST CONTAMINATION:
   Test bilgisi train'e sızmış
   Örnek: Tüm veriyle scaling yapıp sonra split etmek
   → Test'in mean/std'si train'e sızmış olur
   
3. TEMPORAL LEAKAGE:
   Gelecek bilgisi geçmişe sızmış
   Örnek: Müşterinin "sonraki ay harcaması" feature'ı → "churn" tahmini
   → Sonraki ay bilgisi tahmin anında mevcut değil!

NASIL ANLAŞILIR:
• Bir feature'ın korelasyonu > 0.95 → ŞÜPHELİ
• CV skoru gerçek dışı yüksek (R² > 0.99) → ŞÜPHELİ
• Feature importance'ta beklenmedik bir feature 1. sırada → ŞÜPHELİ
• Test skoru CV skorundan ÇOK farklı → LEAKAGE İŞARETİ
"""
```

---

## 📌 MASTER ÖZET: DOMINO ETKİSİ

```
ADIM 1: EDA
  │ "Veri çarpık, eksik, dengesiz"
  ▼
ADIM 2: Eksik Veri
  │ "MAR tespit edildi → is_missing flag eklendi"
  │ "Imputer train'den öğrenildi"
  ▼
ADIM 3: Aykırı Değer
  │ "IQR ile %3 aykırı bulundu → target ile ilişkili → BİRAKILDI"
  │ "Kalan aykırılar kırpıldı (capping)"
  ▼
ADIM 4: Dönüşüm
  │ "Target log dönüştürüldü (skew: 2.3 → 0.1)"
  │ "Yeo-Johnson ile feature'lar normalleştirildi"
  ▼
ADIM 5: Korelasyon
  │ "3 feature çifti r>0.90 → 3 feature çıkarıldı"
  │ "VIF>10: 2 feature → PCA veya çıkar"
  │ "Non-lineer ilişki tespit edildi (Spearman > Pearson)"
  ▼
ADIM 6: Feature Selection
  │ "F-test + MI + LASSO + RFE konsensüsü"
  │ "50 → 25 feature'a düşürüldü"
  ▼
ADIM 7: Scaling
  │ "RobustScaler (aykırılar nedeniyle)"
  │ "OneHotEncoding (kategorik feature'lar)"
  ▼
ADIM 8: Imbalance (Classification)
  │ "Ratio = 0.08 → SMOTE + class_weight"
  │ "StratifiedKFold zorunlu"
  ▼
ADIM 9: CV & Model
  │ "5 model karşılaştırıldı"
  │ "GBM en iyi: RMSE=10.2±0.4, overfit ratio=1.08"
  │ "Tree > Linear → varsayım testleri ATLANDI"
  ▼
ADIM 10: Learning Curve
  │ "Hafif overfit tespit edildi → regularization artırıldı"
  ▼
ADIM 12: Hiperparametre
  │ "Optuna ile 100 trial → %5 iyileşme"
  │ "max_depth=8, lr=0.05, n_est=300"
  ▼
ADIM 13: Model Karşılaştırma
  │ "Paired t-test: GBM vs RF → p=0.012 → GBM ANLAMLI DAHA İYİ"
  ▼
ADIM 14: Final Test Tahmini
  │ "Test RMSE=10.5 ≈ CV RMSE=10.2 → GENELLEMELEBİLİR ✅"
  │ "R²=0.87, MAPE=5.2%"
  ▼
ADIM 16: SHAP
  │ "En önemli 5 feature açıklandı"
  │ "İş birimine rapor hazırlandı"
  ▼
📊 FİNAL RAPOR & DEPLOYMENT
```

> 🎯 **Her adım bir öncekinin çıktısına bağlıdır. Bir adımı atlamak veya yanlış yapmak, tüm sonraki adımları bozar. Bu yüzden Senior Data Analyst sistematik ve sıralı çalışır.**