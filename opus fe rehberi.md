

# Senior DS: FE Öncesi Tam Kontrol Listesi ve FE Karar Mekanizması

Aşağıda her adımı **neden yapıyoruz**, **nasıl yapıyoruz**, **sonuca göre ne kararı alıyoruz** üçlüsüyle açıklıyorum.

---

## BÖLÜM 1: FE ÖNCESİ KONTROLLER

```
Sıra:
  0. Yarışma / Problem Anlama
  1. Veri Yükleme & İlk Bakış
  2. Hedef Değişken Analizi
  3. Veri Tipi Kontrolü & Düzeltme
  4. Eksik Veri Analizi
  5. Duplike Satır Kontrolü
  6. Kardinalite Analizi
  7. Sabit / Quasi-Sabit Sütun Kontrolü
  8. Outlier Analizi
  9. Dağılım Analizi (Skewness & Kurtosis)
 10. Korelasyon Analizi
 11. Multicollinearity (VIF)
 12. Feature-Target İlişki Analizi
 13. Train-Test Dağılım Farkı (Adversarial Validation)
 14. Data Leakage Kontrolü
 15. Class Imbalance Analizi
 16. Temporal / Sıralama Pattern Kontrolü
```

---

### ADIM 0 — Yarışma / Problem Anlama

```python
"""
KOD YAZMADAN ÖNCE OKU:
- Yarışma açıklama sayfası
- Discussion tab (özellikle ilk haftanın postları)
- Evaluation metric ne? (Bu yarışmada ROC-AUC)
- Veri nereden gelmiş? (Sentetik mi, gerçek mi?)
- Playground Series → genelde sentetik veri, 
  bazen orijinal dataset blend edilince skor artar
- Domain bilgisi: Telco churn → müşteri kaybı tahmini
"""

# Bu yarışma için bilmemiz gerekenler:
# - Metrik: ROC-AUC (probability sıralaması önemli, threshold değil)
# - Domain: Telekomünikasyon müşteri kaybı
# - Sentetik veri: Orijinal dataset (IBM Telco) ile blend denenebilir
# - Churn genelde imbalanced olur (~25-30% pozitif)
```

**Neden:** Metrik AUC ise logloss optimize etmek yerine ranking kalitesine odaklanırız. Domain bilgisi hangi feature'ların anlamlı olacağını belirler.

---

### ADIM 1 — Veri Yükleme & İlk Bakış

```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

# ── 1a. Boyut ──
print(f"Train shape: {train.shape}")
print(f"Test  shape: {test.shape}")
print(f"Train/Test satır oranı: {len(train)/len(test):.1f}x")

# ── 1b. Sütun isimleri eşleşiyor mu? ──
train_cols = set(train.columns)
test_cols  = set(test.columns)

only_in_train = train_cols - test_cols
only_in_test  = test_cols - train_cols

print(f"\nSadece train'de: {only_in_train}")  # target + varsa ekstra
print(f"Sadece test'te:  {only_in_test}")      # boş olmalı

# ── 1c. İlk/son satırlar ──
print("\n", train.head())
print("\n", train.tail())

# ── 1d. Genel bilgi ──
print("\n", train.info())
print("\n", train.describe())
print("\n", train.describe(include="object"))
```

**Karar:**
- `only_in_test` boş değilse → veri yükleme hatası var
- Satır oranı çok düşükse (örn. train 500, test 50000) → overfitting riski yüksek, regularization agresif olmalı
- `describe()` → min/max değerleri mantıklı mı, negatif olmaması gereken yerde negatif var mı?

---

### ADIM 2 — Hedef Değişken Analizi

```python
TARGET = "Churn"

# ── 2a. Tip kontrolü ──
print(f"Target dtype: {train[TARGET].dtype}")
print(f"Unique values: {train[TARGET].unique()}")
print(f"Nunique: {train[TARGET].nunique()}")

# ── 2b. Dağılım ──
print(f"\nValue counts:\n{train[TARGET].value_counts()}")
print(f"\nOranlar:\n{train[TARGET].value_counts(normalize=True)}")

# ── 2c. Eksik var mı? ──
print(f"\nTarget'ta eksik: {train[TARGET].isna().sum()}")

# ── 2d. Eğer string ise encode et ──
if train[TARGET].dtype == object:
    print("\n⚠ Target string tipinde, encode edilmeli")
    # Örn: "Yes"->1, "No"->0
    label_map = {"Yes": 1, "No": 0}
    train[TARGET] = train[TARGET].map(label_map)
    print(f"Encode sonrası:\n{train[TARGET].value_counts()}")

# ── 2e. İmbalance oranı ──
pos_ratio = train[TARGET].mean()
neg_ratio = 1 - pos_ratio
imbalance_ratio = max(pos_ratio, neg_ratio) / min(pos_ratio, neg_ratio)
print(f"\nPozitif oran: {pos_ratio:.3f}")
print(f"İmbalance ratio: {imbalance_ratio:.1f}:1")

if imbalance_ratio > 3:
    print("⚠ Ciddi imbalance — class_weight veya SMOTE düşün")
elif imbalance_ratio > 1.5:
    print("⚠ Hafif imbalance — scale_pos_weight ayarla")
else:
    print("✓ Dengeli dağılım")
```

**Karar:**
| İmbalance Oranı | Aksiyon |
|---|---|
| < 1.5:1 | Bir şey yapma |
| 1.5:1 – 3:1 | `scale_pos_weight` veya `is_unbalance=True` |
| 3:1 – 10:1 | Stratified CV + class_weight |
| > 10:1 | SMOTE / undersampling + focal loss düşün |

---

### ADIM 3 — Veri Tipi Kontrolü & Düzeltme

```python
# ── 3a. Mevcut tipler ──
print("Veri tipleri:\n")
for col in train.columns:
    print(f"  {col:25s} → {str(train[col].dtype):10s} | "
          f"örnek: {train[col].iloc[0]}")

# ── 3b. Sayısal olması gereken ama object olan sütunlar ──
for col in train.select_dtypes(include="object").columns:
    # Sayıya çevirmeyi dene
    converted = pd.to_numeric(train[col], errors="coerce")
    n_convertible = converted.notna().sum()
    n_total = len(train[col].dropna())
    
    if n_convertible / max(n_total, 1) > 0.5:
        print(f"⚠ '{col}' object ama {n_convertible}/{n_total} "
              f"değer sayıya çevrilebilir")
        # Çevrilemeyenler nedir?
        mask = converted.isna() & train[col].notna()
        if mask.any():
            print(f"  Çevrilemeyen değerler: "
                  f"{train.loc[mask, col].unique()[:10]}")

# ── 3c. Düzeltme (bu dataset için TotalCharges) ──
if "TotalCharges" in train.columns and train["TotalCharges"].dtype == object:
    print(f"\n'TotalCharges' object → numeric dönüşüm:")
    problematic = train[
        pd.to_numeric(train["TotalCharges"], errors="coerce").isna()
        & train["TotalCharges"].notna()
    ]["TotalCharges"].unique()
    print(f"  Sorunlu değerler: {problematic}")
    # Genelde boş string " " → bunlar yeni müşteriler (tenure=0)

# ── 3d. Boolean gibi davranan int sütunlar ──
for col in train.select_dtypes(include=[np.number]).columns:
    unique_vals = sorted(train[col].dropna().unique())
    if len(unique_vals) <= 3:
        print(f"  '{col}' sadece {unique_vals} değer alıyor "
              f"→ categorical gibi davranabilir")

# ── 3e. ID sütunu kontrol ──
if "id" in train.columns or "customerID" in train.columns:
    id_col = "id" if "id" in train.columns else "customerID"
    is_unique = train[id_col].nunique() == len(train)
    print(f"\n'{id_col}' unique mi: {is_unique}")
    if not is_unique:
        print("⚠ ID sütunu unique değil — duplike olabilir!")
```

**Karar:**
- Object ama sayısal → `pd.to_numeric(errors='coerce')` ile düzelt, `NaN` olanları incele
- 0/1 integer → model için sorun değil ama EDA'da categorical gibi ele al
- ID sütununu feature olarak kullanma, index yap

---

### ADIM 4 — Eksik Veri Analizi

```python
# ── 4a. Eksik veri tablosu ──
def missing_report(df, name="DataFrame"):
    """Kapsamlı eksik veri raporu."""
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    
    report = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct.round(2),
        "dtype": df.dtypes,
    })
    report = report[report["missing_count"] > 0].sort_values(
        "missing_pct", ascending=False
    )
    
    print(f"\n{'='*50}")
    print(f"Eksik Veri Raporu — {name}")
    print(f"{'='*50}")
    
    if len(report) == 0:
        print("✓ Hiç eksik veri yok")
    else:
        print(report)
        print(f"\nToplam eksik hücreli sütun: {len(report)}")
    
    return report

train_missing = missing_report(train, "Train")
test_missing  = missing_report(test, "Test")

# ── 4b. Train'de var ama test'te yok (veya tersi) ──
if len(train_missing) > 0 or len(test_missing) > 0:
    all_missing_cols = set(train_missing.index) | set(test_missing.index)
    for col in all_missing_cols:
        tr_pct = train[col].isna().mean() * 100
        te_pct = test[col].isna().mean() * 100
        print(f"  {col}: train={tr_pct:.1f}%, test={te_pct:.1f}%")
        if abs(tr_pct - te_pct) > 10:
            print(f"    ⚠ Ciddi fark — dağılım kayması olabilir")

# ── 4c. Eksiklik pattern'i — rastgele mi yoksa sistemli mi? ──
if train.isnull().any().any():
    # Eksik olan satırların target dağılımı farklı mı?
    for col in train.columns[train.isnull().any()]:
        if col == TARGET:
            continue
        has_missing = train[col].isna()
        if has_missing.sum() > 10:
            target_with = train.loc[has_missing, TARGET].mean()
            target_without = train.loc[~has_missing, TARGET].mean()
            print(f"\n  '{col}' eksik olanlarda target ort: {target_with:.3f}")
            print(f"  '{col}' dolu olanlarda target ort:   {target_without:.3f}")
            if abs(target_with - target_without) > 0.05:
                print(f"    ⚠ Eksiklik bilgi taşıyor! "
                      f"→ is_missing flag ekle")

# ── 4d. Eksiklik korelasyonu (birden fazla sütunda eksik varsa) ──
missing_cols = train.columns[train.isnull().any()].tolist()
if len(missing_cols) > 1:
    missing_corr = train[missing_cols].isnull().corr()
    print(f"\nEksiklik korelasyon matrisi:\n{missing_corr}")
    # Yüksek korelasyon → eksiklikler birlikte oluşuyor
    # → Tek bir "missing_group" flag yeterli olabilir
```

**Karar Tablosu:**

| Durum | Aksiyon |
|---|---|
| Eksik yok | Devam et |
| < %5 eksik, rastgele | Median/Mode ile doldur |
| < %5 eksik, sistemli (target ile ilişkili) | `is_missing` flag + doldur |
| %5-30 eksik | Flag + model-based imputation (iterative) |
| > %30 eksik | Sütunu silmeyi düşün veya sadece flag tut |
| Train-Test eksiklik oranı çok farklı | Dikkat: drift belirtisi |

---

### ADIM 5 — Duplike Satır Kontrolü

```python
# ── 5a. Tam duplikeler ──
feature_cols = [c for c in train.columns if c not in [TARGET, "id"]]

n_dup = train.duplicated(subset=feature_cols, keep=False).sum()
n_dup_unique = train.duplicated(subset=feature_cols, keep="first").sum()

print(f"Tam duplike satır: {n_dup} ({n_dup_unique} fazlalık)")
print(f"Oran: {n_dup_unique/len(train)*100:.2f}%")

# ── 5b. Duplikelerde target tutarlı mı? ──
if n_dup > 0:
    dup_mask = train.duplicated(subset=feature_cols, keep=False)
    dup_groups = train[dup_mask].groupby(feature_cols)[TARGET]
    
    inconsistent = 0
    for name, group in dup_groups:
        if group.nunique() > 1:
            inconsistent += 1
    
    print(f"\nTutarsız duplike grupları (aynı X, farklı y): {inconsistent}")
    if inconsistent > 0:
        print("⚠ Noise var — bu satırları silme, model noise'a dayanıklı olmalı")

# ── 5c. Near-duplicate kontrolü (sayısal sütunlar için) ──
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
if TARGET in num_cols:
    num_cols.remove(TARGET)

# Basit yöntem: yuvarlayıp bak
if len(num_cols) > 0:
    train_rounded = train[num_cols].round(1)
    near_dup = train_rounded.duplicated(keep="first").sum()
    print(f"\nNear-duplicate (1 decimal): {near_dup}")
```

**Karar:**
- Duplike yok → devam
- Duplike var, target tutarlı → feature olarak "count" eklenebilir
- Duplike var, target tutarsız → noise, silme ama farkında ol
- Near-duplicate çok fazla → feature resolution düşük, binning anlamsız olabilir

---

### ADIM 6 — Kardinalite Analizi

```python
# ── 6a. Her sütunun unique değer sayısı ──
print("Kardinalite Analizi:")
print(f"{'Sütun':30s} {'Tip':10s} {'Nunique':>8s} {'Oran':>8s}")
print("-" * 60)

for col in train.columns:
    if col in ["id", TARGET]:
        continue
    nunique = train[col].nunique()
    ratio = nunique / len(train)
    flag = ""
    if nunique == 1:
        flag = " ← SABİT, SİL"
    elif nunique == 2:
        flag = " ← binary"
    elif nunique <= 10:
        flag = " ← low cardinality"
    elif nunique <= 50:
        flag = " ← medium cardinality"
    elif ratio > 0.5:
        flag = " ← ⚠ HIGH cardinality"
    
    print(f"  {col:28s} {str(train[col].dtype):10s} "
          f"{nunique:8d} {ratio:8.4f}{flag}")

# ── 6b. Kategorik sütunlardaki değerler ──
print("\nKategorik Değerler:")
for col in train.select_dtypes(include="object").columns:
    if col in ["id"]:
        continue
    vals = train[col].value_counts()
    print(f"\n  {col} ({len(vals)} unique):")
    for v, c in vals.items():
        print(f"    {v:30s}: {c:6d} ({c/len(train)*100:5.1f}%)")

# ── 6c. Test'te train'de olmayan değerler var mı? ──
print("\nTrain-Test kategorik uyumsuzluk:")
for col in train.select_dtypes(include="object").columns:
    if col in ["id", TARGET]:
        continue
    train_vals = set(train[col].dropna().unique())
    test_vals  = set(test[col].dropna().unique())
    
    only_train = train_vals - test_vals
    only_test  = test_vals - train_vals
    
    if only_train or only_test:
        print(f"  {col}:")
        if only_train:
            print(f"    Sadece train'de: {only_train}")
        if only_test:
            print(f"    Sadece test'te:  {only_test}")
            print(f"    ⚠ OrdinalEncoder'da handle_unknown gerekli!")
    else:
        print(f"  {col}: ✓ eşleşiyor")
```

**Karar:**

| Kardinalite | Encoding Yöntemi |
|---|---|
| 2 (binary) | Map: Yes/No → 1/0 |
| 3-10 (low) | Ordinal + Frequency + Target Encoding |
| 10-50 (medium) | Target Encoding + Frequency (one-hot çok sütun yapar) |
| 50+ (high) | Target Encoding + Frequency + Hashing |
| Test'te yeni değer | `unknown_value=-1` veya global mean |

---

### ADIM 7 — Sabit / Quasi-Sabit Sütun Kontrolü

```python
# ── 7a. Sabit sütunlar (tek değer) ──
constant_cols = []
for col in train.columns:
    if train[col].nunique(dropna=False) <= 1:
        constant_cols.append(col)
        print(f"  ⚠ SABİT: '{col}' → sadece {train[col].unique()}")

# ── 7b. Quasi-sabit (bir değer %99+ dominant) ──
quasi_constant_cols = []
threshold = 0.99

for col in train.columns:
    if col in ["id", TARGET] or col in constant_cols:
        continue
    top_freq = train[col].value_counts(normalize=True).iloc[0]
    if top_freq >= threshold:
        quasi_constant_cols.append(col)
        top_val = train[col].value_counts().index[0]
        print(f"  ⚠ QUASI-SABİT: '{col}' → '{top_val}' "
              f"%{top_freq*100:.1f} dominant")

# ── 7c. Train'de sabit değil ama test'te sabit (veya tersi) ──
for col in train.columns:
    if col in ["id", TARGET]:
        continue
    if col in test.columns:
        tr_nunique = train[col].nunique()
        te_nunique = test[col].nunique()
        if tr_nunique > 1 and te_nunique <= 1:
            print(f"  ⚠ '{col}' train'de {tr_nunique} unique ama "
                  f"test'te {te_nunique}!")

print(f"\nSilinecekler: {constant_cols}")
print(f"Dikkat edilecekler: {quasi_constant_cols}")
```

**Karar:**
- Sabit sütun → sil (bilgi taşımıyor)
- Quasi-sabit → genelde sil, ama minority class target ile çok ilişkiliyse tut
- Quasi-sabit kontrolü: `target` mean'i minority'de farklı mı?

```python
# Quasi-sabit ama bilgi taşıyabilir mi?
for col in quasi_constant_cols:
    dominant_val = train[col].value_counts().index[0]
    minority_mask = train[col] != dominant_val
    
    if minority_mask.sum() > 10:
        target_minority = train.loc[minority_mask, TARGET].mean()
        target_majority = train.loc[~minority_mask, TARGET].mean()
        diff = abs(target_minority - target_majority)
        print(f"  {col}: minority target={target_minority:.3f}, "
              f"majority target={target_majority:.3f}, fark={diff:.3f}")
        if diff > 0.1:
            print(f"    → TUTULMALI, bilgi taşıyor")
```

---

### ADIM 8 — Outlier Analizi

```python
import matplotlib
# matplotlib.use('Agg')  # Kaggle'da gerekirse
import matplotlib.pyplot as plt

num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in ["id", TARGET]]

print("Outlier Analizi (IQR Yöntemi):")
print(f"{'Sütun':25s} {'Q1':>8s} {'Q3':>8s} {'IQR':>8s} "
      f"{'Alt':>8s} {'Üst':>8s} {'Out#':>6s} {'Out%':>6s}")
print("-" * 85)

outlier_report = {}

for col in num_cols:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = ((train[col] < lower) | (train[col] > upper))
    n_out = outliers.sum()
    pct_out = n_out / len(train) * 100
    
    outlier_report[col] = {
        "n_outliers": n_out, 
        "pct": pct_out,
        "lower": lower, 
        "upper": upper,
    }
    
    flag = " ⚠" if pct_out > 5 else ""
    print(f"  {col:23s} {Q1:8.1f} {Q3:8.1f} {IQR:8.1f} "
          f"{lower:8.1f} {upper:8.1f} {n_out:6d} {pct_out:5.1f}%{flag}")

# ── 8b. Outlier'ların target ile ilişkisi ──
print("\nOutlier'larda target dağılımı:")
for col in num_cols:
    info = outlier_report[col]
    if info["n_outliers"] > 10:
        outlier_mask = (
            (train[col] < info["lower"]) | (train[col] > info["upper"])
        )
        target_out = train.loc[outlier_mask, TARGET].mean()
        target_in  = train.loc[~outlier_mask, TARGET].mean()
        print(f"  {col}: outlier target={target_out:.3f}, "
              f"normal target={target_in:.3f}")

# ── 8c. Z-score yöntemi (ek kontrol) ──
print("\nZ-score > 3 olan gözlemler:")
from scipy import stats as sp_stats
for col in num_cols:
    z = np.abs(sp_stats.zscore(train[col].dropna()))
    extreme = (z > 3).sum()
    if extreme > 0:
        print(f"  {col}: {extreme} gözlem (z > 3)")

# ── 8d. Train vs Test outlier farkı ──
print("\nTrain vs Test uç değer karşılaştırması:")
for col in num_cols:
    if col in test.columns:
        tr_max, tr_min = train[col].max(), train[col].min()
        te_max, te_min = test[col].max(), test[col].min()
        
        if te_max > tr_max * 1.5 or te_min < tr_min * 1.5:
            print(f"  ⚠ {col}: train=[{tr_min:.1f}, {tr_max:.1f}], "
                  f"test=[{te_min:.1f}, {te_max:.1f}]")
```

**Karar:**

| Durum | Aksiyon |
|---|---|
| < %1 outlier, target ile ilişkisiz | Bırak |
| < %5 outlier, target ile ilişkili | Bırak (bilgi taşıyor) |
| > %5 outlier | Clipping (Q1-1.5*IQR, Q3+1.5*IQR) veya log dönüşümü |
| Test'te train range dışı değer | Clipping uygula, yoksa model extrapolation yapamaz |
| Mantıksal hata (negatif tenure vb.) | Düzelt veya sil |

---

### ADIM 9 — Dağılım Analizi

```python
from scipy.stats import skew, kurtosis, ks_2samp, shapiro

print("Dağılım Analizi:")
print(f"{'Sütun':25s} {'Mean':>10s} {'Std':>10s} {'Skew':>8s} "
      f"{'Kurt':>8s} {'Durum':>15s}")
print("-" * 80)

for col in num_cols:
    s = skew(train[col].dropna())
    k = kurtosis(train[col].dropna())
    m = train[col].mean()
    sd = train[col].std()
    
    status = ""
    if abs(s) > 2:
        status = "ÇOK ÇARPIK"
    elif abs(s) > 1:
        status = "ÇARPIK"
    elif abs(s) > 0.5:
        status = "hafif çarpık"
    else:
        status = "simetrik"
    
    if k > 7:
        status += " + heavy-tail"
    
    print(f"  {col:23s} {m:10.2f} {sd:10.2f} {s:8.2f} {k:8.2f}   {status}")

# ── 9b. Train vs Test dağılım karşılaştırması (KS test) ──
print("\nKolmogorov-Smirnov Test (Train vs Test):")
for col in num_cols:
    if col in test.columns:
        stat, pval = ks_2samp(
            train[col].dropna(), 
            test[col].dropna()
        )
        flag = "⚠ FARKLI" if pval < 0.01 else "✓ benzer"
        print(f"  {col:25s} KS={stat:.4f}  p={pval:.4f}  {flag}")

# ── 9c. Kategorikler için Chi-square test ──
from scipy.stats import chi2_contingency

print("\nKategorik Dağılım Karşılaştırması (Chi-square):")
for col in train.select_dtypes(include="object").columns:
    if col in [TARGET, "id"] or col not in test.columns:
        continue
    
    # Frekans tablosu
    tr_counts = train[col].value_counts()
    te_counts = test[col].value_counts()
    
    all_vals = sorted(set(tr_counts.index) | set(te_counts.index))
    observed = np.array([
        [tr_counts.get(v, 0) for v in all_vals],
        [te_counts.get(v, 0) for v in all_vals],
    ])
    
    if observed.shape[1] > 1:
        chi2, pval, dof, expected = chi2_contingency(observed)
        flag = "⚠ FARKLI" if pval < 0.01 else "✓ benzer"
        print(f"  {col:25s} chi2={chi2:.1f}  p={pval:.4f}  {flag}")
```

**Karar:**

| Durum | Aksiyon |
|---|---|
| \|skew\| > 2 | `np.log1p()` veya `PowerTransformer` veya binning |
| \|skew\| 1-2 | `np.sqrt()` veya `np.log1p()` |
| Heavy-tail (kurt > 7) | Clipping + log |
| KS test p < 0.01 | Train-test drift var, bu feature'a az güven |
| Chi-square p < 0.01 | Kategorik drift, dikkat |

---

### ADIM 10 — Korelasyon Analizi

```python
import seaborn as sns

# ── 10a. Feature-Feature korelasyonu ──
corr = train[num_cols + [TARGET]].corr()

# Yüksek korelasyonlu çiftler
print("Yüksek Korelasyonlu Feature Çiftleri (|r| > 0.85):")
seen = set()
for i, col1 in enumerate(num_cols):
    for j, col2 in enumerate(num_cols):
        if i >= j:
            continue
        r = corr.loc[col1, col2]
        if abs(r) > 0.85 and (col1, col2) not in seen:
            seen.add((col1, col2))
            print(f"  {col1} ↔ {col2}: r = {r:.3f}")

# ── 10b. Feature-Target korelasyonu ──
print("\nFeature-Target Korelasyonu (Pearson):")
target_corr = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False)
for feat, r in target_corr.items():
    bar = "█" * int(r * 50)
    print(f"  {feat:25s} {r:.4f} {bar}")

# ── 10c. Spearman korelasyonu (non-linear ilişkiler için) ──
spearman_corr = train[num_cols + [TARGET]].corr(method="spearman")
print("\nSpearman vs Pearson fark (non-linearity göstergesi):")
for col in num_cols:
    p = abs(corr.loc[col, TARGET])
    s = abs(spearman_corr.loc[col, TARGET])
    diff = abs(s - p)
    if diff > 0.05:
        direction = "Spearman > Pearson" if s > p else "Pearson > Spearman"
        print(f"  {col:25s} Pearson={p:.3f} Spearman={s:.3f} "
              f"→ {direction} (non-linear ilişki)")

# ── 10d. Cramér's V (kategorik-kategorik ilişki) ──
def cramers_v(x, y):
    confusion = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion)
    n = len(x)
    r, k = confusion.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1) + 1e-10))

cat_cols = train.select_dtypes(include="object").columns.tolist()
cat_cols = [c for c in cat_cols if c not in ["id", TARGET]]

if len(cat_cols) > 1:
    print("\nCramér's V (kategorik-kategorik):")
    for i, c1 in enumerate(cat_cols):
        for j, c2 in enumerate(cat_cols):
            if i >= j:
                continue
            v = cramers_v(train[c1], train[c2])
            if v > 0.5:
                print(f"  {c1} ↔ {c2}: V = {v:.3f} ⚠ yüksek")
```

**Karar:**

| Durum | Aksiyon |
|---|---|
| \|r\| > 0.95 (feature-feature) | Birini sil (daha düşük target korelasyonu olanı) |
| \|r\| 0.85-0.95 | PCA düşün veya fark/oran feature'ı oluştur |
| Spearman >> Pearson | Non-linear dönüşüm uygula (log, sqrt, polynomial) |
| Feature-Target r ≈ 0 | Sil veya interaction'larda kullan |
| Cramér's V > 0.7 | Kategorik sütunları birleştir |

---

### ADIM 11 — Multicollinearity (VIF)

```python
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# VIF hesapla (sadece numeric, eksik ve inf olmadan)
X_vif = train[num_cols].copy()
X_vif = X_vif.replace([np.inf, -np.inf], np.nan).dropna()

# Ölçekle (VIF ölçeğe duyarlı)
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_vif), 
    columns=num_cols
)

print("Variance Inflation Factor (VIF):")
print(f"{'Sütun':25s} {'VIF':>10s} {'Durum':>15s}")
print("-" * 55)

vif_results = {}
for i, col in enumerate(num_cols):
    try:
        vif = variance_inflation_factor(X_scaled.values, i)
        status = ""
        if vif > 10:
            status = "⚠ ÇOK YÜKSEK"
        elif vif > 5:
            status = "⚠ YÜKSEK"
        else:
            status = "✓ OK"
        vif_results[col] = vif
        print(f"  {col:23s} {vif:10.1f}   {status}")
    except:
        print(f"  {col:23s} {'HATA':>10s}")

# VIF > 10 olanlar için aksiyon önerisi
high_vif = [k for k, v in vif_results.items() if v > 10]
if high_vif:
    print(f"\n⚠ VIF > 10 olan sütunlar: {high_vif}")
    print("  Öneriler:")
    print("  - Birbirleriyle korelasyonlu olanlardan birini sil")
    print("  - PCA uygula")
    print("  - Ratio/diff feature oluşturup orijinalleri sil")
    print("  - Tree modeller VIF'e dayanıklıdır ama LR hassastır")
```

**Karar:**
- Tree-based modeller (LGB, XGB, Cat) → VIF önemsiz, ağaçlar otomatik halleder
- Linear modeller (LR, LDA) → VIF > 10 ciddi sorun, PCA veya sütun silme
- Ensemble'da LR varsa → VIF yüksek sütunları LR'dan çıkar ama tree'lerde tut

---

### ADIM 12 — Feature-Target İlişki Analizi

```python
from sklearn.feature_selection import mutual_info_classif

# ── 12a. Mutual Information (non-linear ilişki ölçer) ──
X_mi = train[num_cols].fillna(-999)
y_mi = train[TARGET]

mi_scores = mutual_info_classif(X_mi, y_mi, random_state=42)
mi_df = pd.DataFrame({
    "feature": num_cols, 
    "MI": mi_scores
}).sort_values("MI", ascending=False)

print("Mutual Information Scores:")
for _, row in mi_df.iterrows():
    bar = "█" * int(row["MI"] * 100)
    print(f"  {row['feature']:25s} {row['MI']:.4f} {bar}")

# ── 12b. Kategorik feature - Target ilişkisi ──
print("\nKategorik Feature - Target İlişkisi:")
for col in cat_cols:
    group_stats = train.groupby(col)[TARGET].agg(["mean", "count", "std"])
    print(f"\n  {col}:")
    for val, row in group_stats.iterrows():
        bar = "█" * int(row["mean"] * 50)
        print(f"    {str(val):30s} mean={row['mean']:.3f} "
              f"n={int(row['count']):5d} {bar}")
    
    # ANOVA testi
    groups = [group[TARGET].values for _, group in train.groupby(col)]
    if len(groups) > 1:
        f_stat, p_val = sp_stats.f_oneway(*groups)
        sig = "SİGNİFİKANT" if p_val < 0.01 else "anlamsız"
        print(f"    ANOVA: F={f_stat:.1f}, p={p_val:.6f} → {sig}")

# ── 12c. Sayısal feature'ların target grubuna göre dağılımı ──
print("\nSayısal Feature'lar Target Grubuna Göre:")
for col in num_cols:
    group0 = train.loc[train[TARGET] == 0, col]
    group1 = train.loc[train[TARGET] == 1, col]
    
    # Mann-Whitney U testi (non-parametric)
    stat, pval = sp_stats.mannwhitneyu(
        group0.dropna(), group1.dropna(), alternative="two-sided"
    )
    
    diff = abs(group1.mean() - group0.mean()) / (train[col].std() + 1e-10)
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    
    print(f"  {col:25s} churn_mean={group1.mean():8.2f} "
          f"no_churn_mean={group0.mean():8.2f} "
          f"effect_size={diff:.3f} {sig}")

# ── 12d. Point-Biserial korelasyon (binary target, continuous feature) ──
print("\nPoint-Biserial Korelasyon:")
for col in num_cols:
    r, p = sp_stats.pointbiserialr(train[TARGET], train[col].fillna(0))
    if abs(r) > 0.05:
        print(f"  {col:25s} r={r:.4f} p={p:.2e}")
```

**Karar:**
- MI yüksek → kesinlikle tut, interaction'larda kullan
- MI ≈ 0 + Pearson ≈ 0 + Mann-Whitney p > 0.05 → silme adayı
- Kategorik ANOVA significant → target encoding çok faydalı olur
- Effect size > 0.5 → güçlü ayrımcı, bu feature etrafında yeni feature'lar oluştur

---

### ADIM 13 — Train-Test Dağılım Farkı (Adversarial Validation)

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

def adversarial_validation(train_df, test_df, features=None):
    """
    Train'i test'ten ayırt edebilen bir model kur.
    AUC ≈ 0.5 → dağılımlar benzer (iyi)
    AUC >> 0.5 → drift var (kötü)
    """
    if features is None:
        features = train_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Label: 0=train, 1=test
    df = pd.concat([
        train_df[features].assign(_is_test=0),
        test_df[features].assign(_is_test=1),
    ], ignore_index=True)
    
    y_adv = df.pop("_is_test")
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []
    importances = np.zeros(len(features))
    
    for tr_idx, val_idx in skf.split(df, y_adv):
        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, 
            num_leaves=31, verbose=-1,
        )
        model.fit(
            df.iloc[tr_idx], y_adv.iloc[tr_idx],
            eval_set=[(df.iloc[val_idx], y_adv.iloc[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(0)],
        )
        pred = model.predict_proba(df.iloc[val_idx])[:, 1]
        aucs.append(roc_auc_score(y_adv.iloc[val_idx], pred))
        importances += model.feature_importances_
    
    mean_auc = np.mean(aucs)
    print(f"\nAdversarial Validation AUC: {mean_auc:.4f}")
    
    if mean_auc < 0.55:
        print("✓ Mükemmel — train ve test çok benzer")
    elif mean_auc < 0.65:
        print("✓ İyi — küçük farklar var ama sorun yok")
    elif mean_auc < 0.75:
        print("⚠ Orta — bazı feature'larda drift var")
    else:
        print("⚠ Kötü — ciddi distribution shift!")
    
    # En çok ayırt edici feature'lar
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
    print(f"\nDrift'e en çok katkı eden feature'lar:")
    for feat, imp in feat_imp.head(10).items():
        print(f"  {feat:25s}: {imp:.0f}")
    
    return mean_auc, feat_imp

adv_auc, drift_features = adversarial_validation(train, test)
```

**Karar:**
- AUC < 0.55 → sorun yok, CV skoruna güvenebilirsin
- AUC 0.65-0.75 → drift yaratan feature'ları incele, belki sil veya normalize et
- AUC > 0.75 → CV'den çok public LB'ye güven, drift feature'ları silmeyi dene
- Drift feature'ı silmek skoru düşürüyorsa → silme, ama bilinçli ol

---

### ADIM 14 — Data Leakage Kontrolü

```python
# ── 14a. Target ile aşırı yüksek korelasyon ──
print("Potansiyel Leakage Kontrolü:")
for col in num_cols:
    r = abs(train[col].corr(train[TARGET]))
    if r > 0.95:
        print(f"  ⚠⚠ '{col}' target ile r={r:.3f} → LEAKAGE ŞÜPHESİ!")
    elif r > 0.80:
        print(f"  ⚠ '{col}' target ile r={r:.3f} → incele")

# ── 14b. Bir feature ile target mükemmel ayrılıyor mu? ──
for col in num_cols:
    auc = roc_auc_score(train[TARGET], train[col].fillna(0))
    auc = max(auc, 1 - auc)  # yön farketmez
    if auc > 0.95:
        print(f"  ⚠⚠ '{col}' tek başına AUC={auc:.4f} → LEAKAGE!")
    elif auc > 0.85:
        print(f"  ⚠ '{col}' tek başına AUC={auc:.4f} → kontrol et")

# ── 14c. Feature test'te yoksa leakage olabilir ──
for col in train.columns:
    if col not in test.columns and col != TARGET:
        print(f"  ⚠ '{col}' test'te yok → ya sil ya test'te de oluştur")

# ── 14d. Zamansal leakage ──
# Eğer tarih/zaman sütunu varsa:
date_cols = [c for c in train.columns if "date" in c.lower() or "time" in c.lower()]
if date_cols:
    print(f"\n  Zaman sütunları bulundu: {date_cols}")
    print("  → Train'deki max tarih < Test'deki min tarih olmalı")
    for col in date_cols:
        train_max = pd.to_datetime(train[col]).max()
        test_min  = pd.to_datetime(test[col]).min()
        if train_max >= test_min:
            print(f"  ⚠ '{col}': train max={train_max} >= test min={test_min}")
            print(f"    → ZAMANSAL LEAKAGE riski!")
```

**Karar:**
- Leakage varsa → o feature'ı kesinlikle sil
- Şüpheli feature → domain bilgisi ile doğrula ("Bu bilgi prediction time'da var mı?")
- Zaman leakage → TimeSeriesSplit kullan, random KFold kullanma

---

### ADIM 15 — Class Imbalance Analizi

```python
from collections import Counter

# ── 15a. Temel oranlar (Adım 2'de baktık ama detaylı) ──
counts = Counter(train[TARGET])
total = sum(counts.values())

print("Class Imbalance Detaylı Analiz:")
for label, count in sorted(counts.items()):
    print(f"  Class {label}: {count:6d} ({count/total*100:.1f}%)")

ratio = counts[0] / counts[1] if 1 in counts else float("inf")
print(f"\nNegatif/Pozitif oranı: {ratio:.1f}:1")

# ── 15b. CV fold'larında dağılım tutarlı mı? ──
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\nFold bazında target dağılımı:")
for fold, (tr_idx, val_idx) in enumerate(skf.split(train, train[TARGET])):
    tr_pos = train.iloc[tr_idx][TARGET].mean()
    val_pos = train.iloc[val_idx][TARGET].mean()
    print(f"  Fold {fold}: train_pos={tr_pos:.3f} val_pos={val_pos:.3f}")

# ── 15c. Stratified CV kullanıldığını doğrula ──
print("\n✓ StratifiedKFold kullanılıyor — fold'lar dengeli")

# ── 15d. Model parametreleri önerisi ──
if ratio > 3:
    spw = counts[0] / counts[1]
    print(f"\nÖnerilen scale_pos_weight: {spw:.1f}")
    print(f"Önerilen LGB is_unbalance: True")
    print(f"Alternatif: SMOTE / undersampling")
```

---

### ADIM 16 — Temporal / Sıralama Pattern Kontrolü

```python
# ── 16a. ID sıralı mı? Target ID ile ilişkili mi? ──
if "id" in train.columns:
    # ID sayısal mı?
    if train["id"].dtype in [np.int64, np.float64]:
        # ID'ye göre sıralı mı?
        is_sorted = (train["id"].diff().dropna() > 0).all()
        print(f"ID sıralı mı: {is_sorted}")
        
        # ID ile target arasında ilişki var mı?
        r = train["id"].corr(train[TARGET])
        print(f"ID-Target korelasyonu: {r:.4f}")
        if abs(r) > 0.05:
            print("  ⚠ ID target ile ilişkili — zamansal pattern olabilir")
            
            # Rolling mean ile kontrol
            sorted_by_id = train.sort_values("id")
            rolling_target = sorted_by_id[TARGET].rolling(
                window=500, min_periods=100
            ).mean()
            
            first_500 = sorted_by_id.head(500)[TARGET].mean()
            last_500  = sorted_by_id.tail(500)[TARGET].mean()
            print(f"  İlk 500 target mean: {first_500:.3f}")
            print(f"  Son 500 target mean:  {last_500:.3f}")

# ── 16b. İndex'te bilgi var mı? ──
# Bazen Kaggle index'e göre sıralı veri verir
train_sorted = train.reset_index(drop=True)
rolling = train_sorted[TARGET].rolling(1000, min_periods=100).mean()
variance_of_rolling = rolling.std()
print(f"\nTarget rolling mean std: {variance_of_rolling:.4f}")
if variance_of_rolling > 0.02:
    print("  ⚠ Sıralama ile target arasında pattern var")
    print("  → TimeSeriesSplit veya GroupKFold düşün")
else:
    print("  ✓ Sıralama ile target arasında pattern yok")
```

---

## BÖLÜM 2: FE KARARINA NASIL GEÇİLİR

Yukarıdaki 16 adımın çıktılarını topladıktan sonra bir **karar matrisi** oluşturuyoruz:

```python
"""
═══════════════════════════════════════════════════════════════
  FE KARAR MATRİSİ
═══════════════════════════════════════════════════════════════

Adım 3 çıktısı (tipler):
  → TotalCharges object → numeric'e çevir
  → SeniorCitizen int ama aslında binary → interaction'larda binary olarak kullan

Adım 4 çıktısı (eksik):
  → TotalCharges'ta boşluk var, tenure=0 olanlarda
  → Aksiyon: 0 ile doldur + is_missing flag ekle

Adım 8 çıktısı (outlier):
  → MonthlyCharges çarpık değil ama TotalCharges çarpık
  → Aksiyon: log1p(TotalCharges)

Adım 9 çıktısı (dağılım):
  → tenure sağa çarpık (skew > 1)
  → Aksiyon: log1p + binning

Adım 10 çıktısı (korelasyon):
  → tenure ve TotalCharges yüksek korelasyon (r≈0.83)
  → Aksiyon: ikisini silme ama ratio oluştur → avg_monthly = Total/(tenure+1)
  → MonthlyCharges ve TotalCharges ilişkili
  → Aksiyon: fark feature → expected_total = Monthly * tenure, diff = Total - expected

Adım 11 çıktısı (VIF):
  → tenure, TotalCharges, avg_monthly yüksek VIF
  → Aksiyon: LR için PCA, tree modeller için sorun değil

Adım 12 çıktısı (feature-target):
  → Contract en yüksek MI → interaction'larda merkeze al
  → tenure, MonthlyCharges yüksek predictive power
  → PhoneService düşük MI → silme ama service_count'ta kalsın
  → OnlineSecurity target ile güçlü ilişki → tut

Adım 13 çıktısı (drift):
  → AUC 0.52 → drift yok, CV'ye güven

Adım 14 çıktısı (leakage):
  → Leakage yok

Adım 15 çıktısı (imbalance):
  → ~27% pozitif → hafif, StratifiedKFold yeterli

═══════════════════════════════════════════════════════════════
"""
```

---

## BÖLÜM 3: FE STRATEJİSİNİ BELİRLEME

```
Her FE kararı şu 4 soruya cevap vermelidir:

1. HANGİ TİP FE?
   ┌─────────────────────────────────────────────────────────┐
   │ EDA Bulgusu              → FE Tipi                     │
   │─────────────────────────────────────────────────────────│
   │ Çarpık dağılım           → Matematiksel dönüşüm        │
   │ Yüksek korelasyonlu çift → Ratio / Fark                │
   │ Düşük MI ama domain'de   → Interaction                 │
   │   anlamlı                                               │
   │ Kategorik + target'la    → Target encoding              │
   │   ilişkili                                              │
   │ Sayısal ama non-linear   → Binning + polynomial        │
   │   ilişki                                                │
   │ Çoklu kategori           → Aggregation (count, freq)   │
   │ Domain bilgisi           → Domain-specific features     │
   └─────────────────────────────────────────────────────────┘

2. LEAKAGE RİSKİ VAR MI?
   - Target encoding → FOLD-AWARE olmalı
   - Aggregation (mean/std) → train global üzerinden mi?
     → Hayır, fold bazında yapılmalı
   - Test verisinden bilgi mi kullanıyoruz? → HAYIR

3. TRAIN-TEST TUTARLILIĞI?
   - Feature train ve test'te aynı şekilde hesaplanabilir mi?
   - Test'te olmayan bilgi kullanıyor muyuz?

4. MODEL UYUMLULUĞU?
   - Tree modeller: monotonic transform gereksiz ama zararsız
   - Linear modeller: scaling + log dönüşümü kritik
   - Her iki tip de varsa (ensemble): her ikisine uygun FE yap
```

### Sistematik FE Kategorileri

```python
"""
KATEGORI 1: MATEMATİKSEL DÖNÜŞÜMLER
═════════════════════════════════════
Tetikleyen EDA: skewness > 1, heavy tail, outlier
"""
# Ne zaman log?
# → Skew > 1 VE değerler >= 0 VE dağılım sağa çarpık
# → tenure, TotalCharges

# Ne zaman sqrt?  
# → Skew 0.5-1.5, count verisi

# Ne zaman power transform?
# → LR için herşeye uygula (Box-Cox / Yeo-Johnson)

# Ne zaman polynomial?
# → Spearman >> Pearson olduğunda (non-linear ilişki)
# → Feature-target scatter'da eğri ilişki görünce

df["log_tenure"]         = np.log1p(df["tenure"])
df["log_total_charges"]  = np.log1p(df["TotalCharges"])
df["sqrt_monthly"]       = np.sqrt(df["MonthlyCharges"])
df["tenure_sq"]          = df["tenure"] ** 2


"""
KATEGORI 2: RATIO / FARK FEATURE'LARI
══════════════════════════════════════
Tetikleyen EDA: yüksek korelasyonlu çift (r > 0.7),
                domain'de anlamlı oran
"""
# Ne zaman ratio?
# → İki feature ilişkili + oranları anlamlı
# → Böleni 0 olmamalı → (x + 1) ekle

# Ne zaman fark?
# → Beklenen vs gerçek karşılaştırması
# → Temporal change proxy

df["avg_monthly_charge"]  = df["TotalCharges"] / (df["tenure"] + 1)
df["expected_total"]      = df["MonthlyCharges"] * df["tenure"]
df["charge_deviation"]    = df["TotalCharges"] - df["expected_total"]
df["monthly_per_service"] = df["MonthlyCharges"] / (df["service_count"] + 1)


"""
KATEGORI 3: AGGREGATION / COUNT FEATURE'LARI
═════════════════════════════════════════════
Tetikleyen EDA: çoklu binary/categorical sütunlar,
                benzer domain'deki feature'lar
"""
# Ne zaman count?
# → Birden fazla binary/categorical aynı konseptte
# → "kaç tane servis kullanıyor?"

# Ne zaman frequency?
# → Kategorik sütunun nadir değerleri farklı davranıyorsa
# → EDA'da value_counts target mean ile ilişkili ise

services = ["PhoneService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", 
            "StreamingMovies"]
df["service_count"]        = (df[services] == "Yes").sum(axis=1)
df["online_service_count"] = (df[["OnlineSecurity","OnlineBackup",
                                   "DeviceProtection","TechSupport"]] == "Yes").sum(axis=1)


"""
KATEGORI 4: INTERACTION / CROSS FEATURE'LARI
═════════════════════════════════════════════
Tetikleyen EDA: iki feature'ın kombinasyonu target ile
                tek başlarından daha güçlü ilişkili
"""
# Ne zaman multiplication?
# → İki feature'ın çarpımı anlamlıysa
# → Özellikle binary * continuous

# Ne zaman concat/group?
# → İki kategorik birlikte davranıyorsa (Cramér's V yüksek)

# Domain bilgisi ile:
# → Senior + alone → risk
# → Month-to-month + electronic check → risk
# → Fiber optic + no support → risk

df["senior_alone"]   = ((df["SeniorCitizen"]==1) & 
                         (df["Partner"]=="No") & 
                         (df["Dependents"]=="No")).astype(int)
df["mtm_electronic"] = ((df["Contract"]=="Month-to-month") & 
                         (df["PaymentMethod"]=="Electronic check")).astype(int)
df["fiber_no_support"] = ((df["InternetService"]=="Fiber optic") & 
                           (df["TechSupport"]!="Yes")).astype(int)

# Kontrol: bu interaction gerçekten faydalı mı?
for feat in ["senior_alone", "mtm_electronic", "fiber_no_support"]:
    if feat in df.columns:
        auc = roc_auc_score(df[TARGET], df[feat])
        target_1 = df.loc[df[feat]==1, TARGET].mean()
        target_0 = df.loc[df[feat]==0, TARGET].mean()
        print(f"  {feat}: AUC={max(auc,1-auc):.3f}, "
              f"flag=1 target={target_1:.3f}, "
              f"flag=0 target={target_0:.3f}")


"""
KATEGORI 5: BINNING / DISCRETIZATION
═════════════════════════════════════
Tetikleyen EDA: non-linear feature-target ilişki,
                scatter'da step-function benzeri pattern
"""
# Ne zaman eşit genişlik (cut)?
# → Domain bilgisi var: tenure 0-12 yeni, 12-48 orta, 48+ sadık

# Ne zaman eşit frekans (qcut)?
# → Domain bilgisi yok, sadece dağılım bazlı bölmek istiyorsun

# Ne zaman optimal binning?
# → WoE/IV hesaplayarak informatif binler bulmak istiyorsun

df["tenure_bin"] = pd.cut(df["tenure"], 
                           bins=[-1,6,12,24,48,72,200],
                           labels=[0,1,2,3,4,5])
df["monthly_bin"] = pd.qcut(df["MonthlyCharges"], 
                              q=10, 
                              labels=False, 
                              duplicates="drop")


"""
KATEGORI 6: TARGET ENCODING
════════════════════════════
Tetikleyen EDA: kategorik feature target ile ilişkili (ANOVA p < 0.01),
                kardinalite > 5 (one-hot çok sütun yapar)
"""
# KRITIK: Her zaman FOLD-AWARE olmalı
# → Train'in tamamı üzerinden yapılan TE → leakage
# → Çözüm: Her CV fold'unda sadece train fold üzerinden hesapla

# Bayesian smoothing neden?
# → Küçük gruplarda mean güvenilir değil
# → Global mean'e doğru shrink et

# smoothing parametresi nasıl seçilir?
# → Küçük veri (< 10K) → smoothing = 50-100
# → Büyük veri (> 100K) → smoothing = 10-30
# → Hep cross-validation ile doğrula


"""
KATEGORI 7: DOMAIN-SPECIFIC
════════════════════════════
Tetikleyen EDA: domain bilgisi + EDA bulgularının kesişimi
"""
# Telco churn domain bilgisi:
# 1. "Yeni müşteriler daha çok churn yapar" → is_new (tenure <= 3)
# 2. "Sözleşmesiz müşteriler riskli" → is_month_to_month  
# 3. "Destek almayan fiber müşteriler mutsuz" → fiber_no_support
# 4. "Çok ödeyip az servis alan kızgın" → monthly_per_service yüksek
# 5. "Kağıtsız fatura + e-çek = düşük bağlılık" → epay_paperless
```

---

## BÖLÜM 4: FE SONRASI VALİDASYON

```python
"""
FE yaptıktan sonra kontrol listesi:
"""

# ── V1. NaN / Inf üretildi mi? ──
print("FE Sonrası Sağlık Kontrolü:")
for col in train.columns:
    n_nan = train[col].isna().sum()
    n_inf = np.isinf(train[col]).sum() if train[col].dtype in [np.float64, np.float32] else 0
    if n_nan > 0 or n_inf > 0:
        print(f"  ⚠ {col}: NaN={n_nan}, Inf={n_inf}")

# ── V2. Yeni feature'lar bilgi taşıyor mu? ──
new_features = ["avg_monthly_charge", "charge_deviation", "service_count",
                "senior_alone", "mtm_electronic", "fiber_no_support",
                "log_tenure", "tenure_sq"]

print("\nYeni Feature'ların Değeri:")
for feat in new_features:
    if feat in train.columns and feat in train.select_dtypes(include=[np.number]).columns:
        mi = mutual_info_classif(
            train[[feat]].fillna(0), train[TARGET], random_state=42
        )[0]
        auc = roc_auc_score(train[TARGET], train[feat].fillna(0))
        auc = max(auc, 1-auc)
        print(f"  {feat:25s} MI={mi:.4f}  AUC={auc:.4f}")

# ── V3. Feature sayısı makul mü? ──
n_features = train.shape[1] - 1  # target hariç
n_samples  = train.shape[0]
ratio = n_samples / n_features

print(f"\nFeature sayısı: {n_features}")
print(f"Sample/Feature oranı: {ratio:.0f}")
if ratio < 10:
    print("⚠ Çok fazla feature — overfitting riski")
    print("  → Feature selection veya dimensionality reduction uygula")
elif ratio < 50:
    print("✓ Makul — regularization kullan")
else:
    print("✓ İyi oran — daha fazla feature eklenebilir")

# ── V4. Yeni feature'larla baseline model skoru arttı mı? ──
from sklearn.model_selection import cross_val_score

# Sadece orijinal feature'lar
original_num = [c for c in num_cols if c in train.columns]
X_orig = train[original_num].fillna(0)

# Tüm feature'lar
all_num = train.select_dtypes(include=[np.number]).columns.tolist()
all_num = [c for c in all_num if c != TARGET]
X_all = train[all_num].fillna(0)

model = lgb.LGBMClassifier(n_estimators=200, verbose=-1, random_state=42)

score_orig = cross_val_score(model, X_orig, train[TARGET], 
                              cv=5, scoring="roc_auc").mean()
score_all  = cross_val_score(model, X_all, train[TARGET], 
                              cv=5, scoring="roc_auc").mean()

print(f"\nOrijinal features AUC:  {score_orig:.6f}")
print(f"FE sonrası AUC:         {score_all:.6f}")
print(f"İyileşme:               {score_all - score_orig:+.6f}")

if score_all > score_orig:
    print("✓ FE faydalı")
else:
    print("⚠ FE zararlı — geri al veya feature selection uygula")
```

---

## ÖZET: TAM KONTROL AKIŞ ŞEMASI

```
START
  │
  ▼
┌─────────────────────────┐
│ 0. Problem/Yarışma Oku  │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 1. Veri Yükle & .head() │
│    .info() .describe()  │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 2. Target Analizi       │──→ Binary? Multi? Regression?
│    Dağılım, Tip, NaN    │    Imbalance oranı?
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 3. Veri Tipi Düzeltme   │──→ Object→Numeric? ID→Index?
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 4. Eksik Veri           │──→ Rastgele mi? Sistemli mi?
│    MCAR/MAR/MNAR?       │    Target ile ilişkili mi?
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 5. Duplike Kontrolü     │──→ Target tutarlı mı?
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 6. Kardinalite          │──→ Binary/Low/Med/High?
│    Test uyumu?           │    Encoding stratejisi?
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 7. Sabit/Quasi-sabit    │──→ Sil veya tut?
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 8. Outlier              │──→ Clip? Log? Sil? Bırak?
│    Target ile ilişki?    │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 9. Dağılım              │──→ Skew→log? KS test?
│    Train-Test KS         │    Chi-square?
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 10. Korelasyon           │──→ r>0.95 → sil
│     Pearson/Spearman     │    Spearman>>Pearson → nonlinear
│     Cramér's V           │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 11. VIF                  │──→ LR için kritik
│     Multicollinearity    │    Tree için önemsiz
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 12. Feature-Target       │──→ MI, ANOVA, Mann-Whitney
│     İlişki Analizi       │    Point-biserial
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 13. Adversarial Val      │──→ AUC≈0.5 iyi, >>0.5 drift
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 14. Leakage Kontrolü    │──→ r>0.95? Test'te yok?
│     Temporal leakage?    │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 15. Class Imbalance      │──→ scale_pos_weight?
│     Fold dağılımı        │    SMOTE? Stratified?
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 16. Temporal Pattern     │──→ TimeSeriesSplit gerekli mi?
└────────────┬────────────┘
             ▼
  ════════════════════════
  ║  KARAR MATRİSİ OLUŞ ║
  ║  (Tüm bulguları bir  ║
  ║   tabloya yaz)        ║
  ════════════════════════
             ▼
┌─────────────────────────┐
│ FE UYGULA:              │
│ 1. Dönüşümler           │
│ 2. Ratio/Fark           │
│ 3. Aggregation          │
│ 4. Interaction          │
│ 5. Binning              │
│ 6. Target Encoding      │
│ 7. Domain-specific      │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ FE VALİDASYON:          │
│ - NaN/Inf kontrolü      │
│ - MI/AUC ile fayda      │
│ - Baseline karşılaştır  │
│ - Feature sayısı makul? │
└────────────┬────────────┘
             ▼
        MODEL EĞİTİMİ
```

Her adımda **bulgu → karar → aksiyon** döngüsü çalışır. Hiçbir adımı atlamaman, ama her adıma da saatler harcamaman gerekir — deneyimle hangi kontrollerin hangi veri setinde kritik olduğunu hızlıca anlarsın.