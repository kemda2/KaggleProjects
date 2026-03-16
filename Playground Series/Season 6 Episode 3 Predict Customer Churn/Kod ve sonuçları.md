# ADIM 1 — Veri Yükleme & İlk Bakış
print("\n\n\nADIM 1 — Veri Yükleme & İlk Bakış\n\n\n")

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

idc = "id"
target = "Churn"
train = pd.read_csv("/kaggle/input/competitions/playground-series-s6e3/train.csv",index_col=idc)
test  = pd.read_csv("/kaggle/input/competitions/playground-series-s6e3/test.csv",index_col=idc)

for df in [train, test]:
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    
# ── 1a. Boyut ──
print(f"Train shape: {train.shape}")
print(f"Test  shape: {test.shape}")
print(f"Train/Test satır oranı: {len(train)/len(test):.1f}x")

# ── 1b. Sütun isimleri eşleşiyor mu? ──
train_cols = set(train.drop(target,axis=1).columns)
test_cols  = set(test.columns)

only_in_train = train_cols - test_cols
only_in_test  = test_cols - train_cols

if only_in_train:
    print(f"\nSadece train'de: {only_in_train}")  # target + varsa ekstra
if only_in_test:
    print(f"Sadece test'te:  {only_in_test}")      # boş olmalı

# ── 1c. İlk/son satırlar ──
print("\nHead:")
display(train.head())
print("\nTail:")
display(train.tail())

# ── 1d. Genel bilgi ──
print(train.info())
display(train.describe())
display(train.describe(include="object").T)




# ADIM 2 — Hedef Değişken Analizi
print("\n\n\nADIM 2 — Hedef Değişken Analizi\n\n\n")

# ── 2a. Tip kontrolü ──
print(f"Target dtype: {train[target].dtype}")
print(f"Unique values: {train[target].unique()}")
print(f"Nunique: {train[target].nunique()}")

# ── 2b. Dağılım ──
print(f"\nValue counts:\n{train[target].value_counts()}")
print(f"\nOranlar:\n{train[target].value_counts(normalize=True)}")

# ── 2c. Eksik var mı? ──
print(f"\nTarget'ta eksik: {train[target].isna().sum()}")

# ── 2d. Eğer string ise encode et ──
if train[target].dtype == object:
    print("\n⚠ Target string tipinde, encode edilmeli")
    # Örn: "Yes"->1, "No"->0
    label_map = {"Yes": 1, "No": 0}
    train[target] = train[target].map(label_map)
    print(f"Encode sonrası:\n{train[target].value_counts()}")

# ── 2e. İmbalance oranı ──
pos_ratio = train[target].mean()
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



# ADIM 3 — Veri Tipi Kontrolü & Düzeltme
print("\n\n\nADIM 3 — Veri Tipi Kontrolü & Düzeltme\n\n\n")

# ── 3a. Mevcut tipler ──
print("\nVeri tipleri:\n")
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
        print(f"\n⚠ '{col}' object ama {n_convertible}/{n_total} "
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
        print("\n⚠ ID sütunu unique değil — duplike olabilir!")

# ── 3f. dtype güncelleme ──
for df in [train, test]:
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)



# ADIM 4 — Eksik Veri Analizi
print("\n\n\nADIM 4 — Eksik Veri Analizi\n\n\n")

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
        if col == target:
            continue
        has_missing = train[col].isna()
        if has_missing.sum() > 10:
            target_with = train.loc[has_missing, target].mean()
            target_without = train.loc[~has_missing, target].mean()
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



# ADIM 5 — Duplike Satır Kontrolü
print("\n\n\nADIM 5 — Duplike Satır Kontrolü\n\n\n")

# ── 5a. Tam duplikeler ──
feature_cols = [c for c in train.columns if c not in [target, idc]]

n_dup = train.duplicated(subset=feature_cols, keep=False).sum()
n_dup_unique = train.duplicated(subset=feature_cols, keep="first").sum()

print(f"Tam duplike satır: {n_dup} ({n_dup_unique} fazlalık)")
print(f"Oran: {n_dup_unique/len(train)*100:.2f}%")

# ── 5b. Duplikelerde target tutarlı mı? ──
if n_dup > 0:
    dup_mask = train.duplicated(subset=feature_cols, keep=False)
    dup_groups = train[dup_mask].groupby(feature_cols)[target]
    
    inconsistent = 0
    for name, group in dup_groups:
        if group.nunique() > 1:
            inconsistent += 1
    
    print(f"\nTutarsız duplike grupları (aynı X, farklı y): {inconsistent}")
    if inconsistent > 0:
        print("⚠ Noise var — bu satırları silme, model noise'a dayanıklı olmalı")

# ── 5c. Near-duplicate kontrolü (sayısal sütunlar için) ──
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
if target in num_cols:
    num_cols.remove(target)

# ══════════════════════════════════════════════════════════════
# Near-duplicate sadece numeric mi yoksa full row mu?
# ══════════════════════════════════════════════════════════════

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
all_feature_cols = [c for c in train.columns if c not in [target]]

# 1) Sadece numeric sütunlarda near-dup
train_num_rounded = train[num_cols].round(1)
near_dup_num = train_num_rounded.duplicated(keep="first").sum()

# 2) Numeric(rounded) + TÜM kategorik sütunlarda near-dup
cat_cols = [c for c in all_feature_cols if c not in num_cols]
train_combined = pd.concat([train_num_rounded, train[cat_cols]], axis=1)
near_dup_full = train_combined.duplicated(keep="first").sum()

print(f"Near-dup sadece numeric:           {near_dup_num:,} ({near_dup_num/len(train)*100:.1f}%)")
print(f"Near-dup numeric + kategorik:      {near_dup_full:,} ({near_dup_full/len(train)*100:.1f}%)")
print(f"Kategorikler ekleyince düşen:      {near_dup_num - near_dup_full:,}")




# ADIM 6 — Kardinalite Analizi
print("\n\n\nADIM 6 — Kardinalite Analizi\n\n\n")

# ── 6a. Her sütunun unique değer sayısı ──
print("Kardinalite Analizi:")
print(f"{'Sütun':30s} {'Tip':10s} {'Nunique':>8s} {'Oran':>8s}")
print("-" * 60)

for col in train.columns:
    if col in [idc, target]:
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
    if col in [idc]:
        continue
    vals = train[col].value_counts()
    print(f"\n  {col} ({len(vals)} unique):")
    for v, c in vals.items():
        print(f"    {v:30s}: {c:6d} ({c/len(train)*100:5.1f}%)")

# ── 6c. Test'te train'de olmayan değerler var mı? ──
print("\nTrain-Test kategorik uyumsuzluk:")
for col in train.select_dtypes(include="object").columns:
    if col in [idc, target]:
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

# ══════════════════════════════════════════════════════════════
# "No internet service" değerlerinin gerçekten aynı satırlar
# olduğunu doğrula
# ══════════════════════════════════════════════════════════════

internet_dependent = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]

no_internet = train[train["InternetService"] == "No"]

print("InternetService=No olan satırlarda diğer sütunlar:")
for col in internet_dependent:
    vals = no_internet[col].value_counts()
    print(f"  {col}: {dict(vals)}")
    # Beklenen: hepsi "No internet service"

# Tersi: InternetService != No ama alt servis = "No internet service" var mı?
has_internet = train[train["InternetService"] != "No"]
for col in internet_dependent:
    anomaly = (has_internet[col] == "No internet service").sum()
    if anomaly > 0:
        print(f"  ⚠ {col}: internet var ama 'No internet service' → {anomaly}")
    else:
        print(f"  ✓ {col}: tutarlı")

# Aynısını PhoneService - MultipleLines için yap
no_phone = train[train["PhoneService"] == "No"]
print(f"\nPhoneService=No olanlarda MultipleLines:")
print(f"  {dict(no_phone['MultipleLines'].value_counts())}")



# ADIM 7 — Sabit / Quasi-Sabit Sütun Kontrolü
print("\n\n\nADIM 7 — Sabit / Quasi-Sabit Sütun Kontrolü\n\n\n")

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
    if col in [idc, target] or col in constant_cols:
        continue
    top_freq = train[col].value_counts(normalize=True).iloc[0]
    if top_freq >= threshold:
        quasi_constant_cols.append(col)
        top_val = train[col].value_counts().index[0]
        print(f"  ⚠ QUASI-SABİT: '{col}' → '{top_val}' "
              f"%{top_freq*100:.1f} dominant")

# ── 7c. Train'de sabit değil ama test'te sabit (veya tersi) ──
for col in train.columns:
    if col in [idc, target]:
        continue
    if col in test.columns:
        tr_nunique = train[col].nunique()
        te_nunique = test[col].nunique()
        if tr_nunique > 1 and te_nunique <= 1:
            print(f"  ⚠ '{col}' train'de {tr_nunique} unique ama "
                  f"test'te {te_nunique}!")

print(f"\nSilinecekler: {constant_cols}")
print(f"Dikkat edilecekler: {quasi_constant_cols}")



# ADIM 8 — Outlier Analizi
print("\n\n\nADIM 8 — Outlier Analizi\n\n\n")

import matplotlib
# matplotlib.use('Agg')  # Kaggle'da gerekirse
import matplotlib.pyplot as plt

num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in [idc, target]]

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
        target_out = train.loc[outlier_mask, target].mean()
        target_in  = train.loc[~outlier_mask, target].mean()
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




# ADIM 9 — Dağılım Analizi
print("\n\n\nADIM 9 — Dağılım Analizi\n\n\n")

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
    if col in [target, "id"] or col not in test.columns:
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



# chi2 "significant" çıkan sütunlardaki GERÇEK oran farkını gör
print("\n\nPratik Oran Farkları (train vs test):")
print(f"{'Sütun':25s} {'Değer':30s} {'Train%':>8s} {'Test%':>8s} {'Fark':>8s}")
print("-" * 85)

significant_cats = ["Partner", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                    "Contract", "PaymentMethod"]

for col in significant_cats:
    tr_pct = train[col].value_counts(normalize=True)
    te_pct = test[col].value_counts(normalize=True)
    
    for val in tr_pct.index:
        t = tr_pct.get(val, 0) * 100
        e = te_pct.get(val, 0) * 100
        d = abs(t - e)
        flag = " ⚠" if d > 1.0 else ""
        print(f"  {col:23s} {str(val):28s} {t:7.2f}% {e:7.2f}% {d:7.2f}%{flag}")
    print()



# ADIM 10 — Korelasyon Analizi
print("\n\n\nADIM 10 — Korelasyon Analizi\n\n\n")

from scipy.stats import chi2_contingency
from scipy import stats as sp_stats

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

# ── 10a. Pearson korelasyonu (feature-feature) ──
print("Pearson Korelasyonu (Feature-Feature):")
corr = train[num_cols].corr()
print(corr.round(4))

print("\nYüksek korelasyonlu çiftler (|r| > 0.7):")
for i, c1 in enumerate(num_cols):
    for j, c2 in enumerate(num_cols):
        if i >= j:
            continue
        r = corr.loc[c1, c2]
        if abs(r) > 0.7:
            print(f"  ⚠ {c1} ↔ {c2}: r = {r:.4f}")

# ── 10b. Feature-Target korelasyonu ──
print("\nFeature-Target Korelasyonu:")
for col in num_cols:
    pearson_r = train[col].corr(train[target])
    spearman_r = train[col].corr(train[target], method="spearman")
    diff = abs(abs(spearman_r) - abs(pearson_r))
    
    nonlinear = " → non-linear ilişki var" if diff > 0.03 else ""
    
    print(f"  {col:25s} Pearson={pearson_r:+.4f}  "
          f"Spearman={spearman_r:+.4f}  "
          f"fark={diff:.4f}{nonlinear}")

# ── 10c. Point-Biserial korelasyon ──
print("\nPoint-Biserial Korelasyon (binary target × continuous):")
for col in num_cols:
    r, p = sp_stats.pointbiserialr(train[target], train[col])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {col:25s} r={r:+.4f}  p={p:.2e} {sig}")

# ── 10d. Cramér's V (kategorik-kategorik) ──
def cramers_v(x, y):
    confusion = pd.crosstab(x, y)
    chi2_val, _, _, _ = chi2_contingency(confusion)
    n = len(x)
    r, k = confusion.shape
    return np.sqrt(chi2_val / (n * (min(r, k) - 1) + 1e-10))

cat_cols = [c for c in train.columns 
            if train[c].dtype == "object" and c != target]

print("\nCramér's V — Kategorik ↔ Target:")
cv_results = {}
for col in cat_cols:
    # target'ı string'e çevirmeden doğrudan kullan
    v = cramers_v(train[col], train[target])
    cv_results[col] = v

# sırala
for col, v in sorted(cv_results.items(), key=lambda x: -x[1]):
    bar = "█" * int(v * 100)
    print(f"  {col:25s} V={v:.4f} {bar}")

print("\nCramér's V — Kategorik ↔ Kategorik (V > 0.3):")
for i, c1 in enumerate(cat_cols):
    for j, c2 in enumerate(cat_cols):
        if i >= j:
            continue
        v = cramers_v(train[c1], train[c2])
        if v > 0.3:
            print(f"  {c1:25s} ↔ {c2:25s}: V={v:.4f}")




# ADIM 11 — VIF
print("\n\n\nADIM 11 — Multicollinearity (VIF)\n\n\n")

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

X_vif = train[num_cols].copy()
X_vif = X_vif.replace([np.inf, -np.inf], np.nan).dropna()

scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_vif), 
    columns=num_cols
)

print("VIF Analizi:")
print(f"{'Sütun':25s} {'VIF':>10s} {'Durum'}")
print("-" * 55)

for i, col in enumerate(num_cols):
    vif = variance_inflation_factor(X_scaled.values, i)
    if vif > 10:
        status = "⚠ ÇOK YÜKSEK"
    elif vif > 5:
        status = "⚠ YÜKSEK"
    else:
        status = "✓ OK"
    print(f"  {col:23s} {vif:10.2f}   {status}")

print("""
YORUM:
- Sadece 3 sayısal sütun var, VIF sonucu sınırlı bilgi verir
- tenure ↔ TotalCharges (r=0.77) nedeniyle VIF yüksek çıkabilir
- Tree modeller (LGB/XGB/Cat) VIF'den ETKİLENMEZ
- LR için: ratio feature (avg_monthly) kullanarak decompose et
  → Orijinalleri LR'dan çıkarmayı düşünebilirsin
""")




# ADIM 12 — Feature-Target İlişki Analizi
print("\n\n\nADIM 12 — Feature-Target İlişki Analizi\n\n\n")

from sklearn.feature_selection import mutual_info_classif
from scipy import stats as sp_stats

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = [c for c in train.columns 
            if train[c].dtype == "object" and c != target]

# ── 12a. Mutual Information (sayısal) ──
X_mi = train[num_cols].fillna(0)
y_mi = train[target]

mi_scores = mutual_info_classif(X_mi, y_mi, random_state=42)
mi_df = pd.DataFrame({
    "feature": num_cols, 
    "MI": mi_scores
}).sort_values("MI", ascending=False)

print("Mutual Information (Sayısal):")
for _, row in mi_df.iterrows():
    bar = "█" * int(row["MI"] * 200)
    print(f"  {row['feature']:25s} MI={row['MI']:.4f} {bar}")

# ── 12b. Kategorik feature - Target ilişkisi ──
print("\nKategorik Feature - Target İlişkisi:")
for col in cat_cols:
    group_stats = train.groupby(col)[target].agg(["mean", "count"])
    group_stats = group_stats.sort_values("mean", ascending=False)
    print(f"\n  {col}:")
    for val, row in group_stats.iterrows():
        bar = "█" * int(row["mean"] * 80)
        print(f"    {str(val):30s} churn={row['mean']:.3f} "
              f"n={int(row['count']):>7,} {bar}")

# ── 12c. Sayısal feature'ların target grubuna göre ──
print("\n\nSayısal Feature'lar — Churn vs No Churn:")
print(f"{'Feature':25s} {'NoChurn_mean':>12s} {'Churn_mean':>12s} "
      f"{'Fark':>10s} {'EffectSize':>10s}")
print("-" * 75)

for col in num_cols:
    g0 = train.loc[train[target] == 0, col]
    g1 = train.loc[train[target] == 1, col]
    
    diff = g1.mean() - g0.mean()
    pooled_std = np.sqrt((g0.std()**2 + g1.std()**2) / 2)
    effect = abs(diff) / (pooled_std + 1e-10)
    
    stars = "★★★" if effect > 0.8 else "★★" if effect > 0.5 else "★" if effect > 0.2 else ""
    
    print(f"  {col:23s} {g0.mean():12.2f} {g1.mean():12.2f} "
          f"{diff:+10.2f} {effect:10.3f} {stars}")

# ── 12d. Tenure dağılımı churn/no-churn gruplarında ──
print("\n\nTenure Dağılımı (Churn vs No Churn):")
for label, name in [(0, "No Churn"), (1, "Churn")]:
    data = train.loc[train[target] == label, "tenure"]
    print(f"\n  {name}:")
    print(f"    mean={data.mean():.1f}  median={data.median():.1f}  "
          f"std={data.std():.1f}")
    print(f"    P10={data.quantile(0.1):.0f}  P25={data.quantile(0.25):.0f}  "
          f"P75={data.quantile(0.75):.0f}  P90={data.quantile(0.9):.0f}")



# ADIM 13 — Adversarial Validation
print("\n\n\nADIM 13 — Adversarial Validation\n\n\n")

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Encode kategorikleri hızlıca
train_adv = train.drop(target, axis=1).copy()
test_adv = test.copy()

for col in train_adv.select_dtypes(include="object").columns:
    train_adv[col] = train_adv[col].astype("category").cat.codes
    test_adv[col] = test_adv[col].astype("category").cat.codes

combined = pd.concat([
    train_adv.assign(_is_test=0),
    test_adv.assign(_is_test=1),
], ignore_index=True)

y_adv = combined.pop("_is_test")

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
aucs = []
importances = np.zeros(len(combined.columns))

for tr_idx, val_idx in skf.split(combined, y_adv):
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05,
        num_leaves=31, verbose=-1,
    )
    model.fit(
        combined.iloc[tr_idx], y_adv.iloc[tr_idx],
        eval_set=[(combined.iloc[val_idx], y_adv.iloc[val_idx])],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(0)],
    )
    pred = model.predict_proba(combined.iloc[val_idx])[:, 1]
    aucs.append(roc_auc_score(y_adv.iloc[val_idx], pred))
    importances += model.feature_importances_

mean_auc = np.mean(aucs)
print(f"Adversarial Validation AUC: {mean_auc:.4f}")

if mean_auc < 0.55:
    print("✓ Mükemmel — train ve test dağılımları neredeyse identik")
elif mean_auc < 0.65:
    print("✓ İyi — küçük farklar var ama sorun yok")
elif mean_auc < 0.75:
    print("⚠ Orta — bazı feature'larda drift var")
else:
    print("⚠ Kötü — ciddi distribution shift!")

feat_imp = pd.Series(importances, index=combined.columns).sort_values(ascending=False)
print(f"\nDrift'e en çok katkı eden feature'lar:")
for feat, imp in feat_imp.head(10).items():
    print(f"  {feat:25s}: {imp:.0f}")


# ADIM 14 — Leakage Kontrolü
print("\n\n\nADIM 14 — Data Leakage Kontrolü\n\n\n")

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

print("Potansiyel Leakage Kontrolü:")
for col in num_cols:
    r = abs(train[col].corr(train[target]))
    if r > 0.95:
        print(f"  ⚠⚠ '{col}' target ile r={r:.3f} → LEAKAGE ŞÜPHESİ!")
    elif r > 0.80:
        print(f"  ⚠ '{col}' target ile r={r:.3f} → incele")
    else:
        print(f"  ✓ '{col}' target ile r={r:.3f} → OK")

print("\nTek feature AUC kontrolü:")
for col in num_cols:
    auc = roc_auc_score(train[target], train[col].fillna(0))
    auc = max(auc, 1 - auc)
    if auc > 0.95:
        print(f"  ⚠⚠ '{col}' AUC={auc:.4f} → LEAKAGE!")
    elif auc > 0.85:
        print(f"  ⚠ '{col}' AUC={auc:.4f} → kontrol et")
    else:
        print(f"  ✓ '{col}' AUC={auc:.4f} → OK")

# Test'te olup train'de olmayan (veya tersi) sütunlar
print("\nSütun tutarlılık:")
for col in train.columns:
    if col == target:
        continue
    if col not in test.columns:
        print(f"  ⚠ '{col}' test'te yok!")
    else:
        print(f"  ✓ '{col}' her ikisinde var")




# ADIM 15 — Class Imbalance (detay)
print("\n\n\nADIM 15 — Class Imbalance Detay\n\n\n")

from sklearn.model_selection import StratifiedKFold

pos_ratio = train[target].mean()
neg_ratio = 1 - pos_ratio
ratio = neg_ratio / pos_ratio

print(f"Pozitif (Churn=1): {train[target].sum():,} ({pos_ratio:.1%})")
print(f"Negatif (Churn=0): {(train[target]==0).sum():,} ({neg_ratio:.1%})")
print(f"Neg/Pos oranı: {ratio:.2f}:1")
print(f"Önerilen scale_pos_weight: {ratio:.2f}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\nFold bazında target dağılımı:")
for fold, (tr_idx, val_idx) in enumerate(skf.split(train, train[target])):
    tr_pos = train.iloc[tr_idx][target].mean()
    val_pos = train.iloc[val_idx][target].mean()
    print(f"  Fold {fold}: train_pos={tr_pos:.4f}  val_pos={val_pos:.4f}")


# ADIM 16 — Temporal / Sıralama Pattern
print("\n\n\nADIM 16 — Temporal / Sıralama Pattern\n\n\n")

# ID ile target ilişkisi
train_reset = train.reset_index()
r = train_reset[idc].corr(train_reset[target])
print(f"ID-Target korelasyonu: {r:.4f}")

# Rolling mean
rolling = train_reset[target].rolling(5000, min_periods=1000).mean()
print(f"Target rolling mean (window=5000):")
print(f"  std:  {rolling.std():.5f}")
print(f"  min:  {rolling.min():.4f}")
print(f"  max:  {rolling.max():.4f}")

if rolling.std() < 0.01:
    print("  ✓ Sıralama ile target arasında pattern YOK")
    print("  → StratifiedKFold uygun, TimeSeriesSplit gereksiz")
else:
    print("  ⚠ Temporal pattern olabilir")


# ------------------------
# ------------------------
# ------------------------



# TÜM EDA ÖZETİ + FE KARAR MATRİSİ

## 16 Adımın Toplu Özeti

```
┌─────┬──────────────────────────────┬──────────────────────────────────┐
│ #   │ ADIM                         │ BULGU                            │
├─────┼──────────────────────────────┼──────────────────────────────────┤
│  1  │ Veri Yükleme                 │ 594K train, 254K test, 19 feat   │
│  2  │ Target                       │ Binary, %22.5 pozitif, 3.4:1     │
│  3  │ Veri Tipi                    │ SeniorCitizen str→int gerekli    │
│  4  │ Eksik Veri                   │ ✓ Yok                            │
│  5  │ Duplike                      │ ✓ Tam dup yok, near-dup %2.1     │
│  6  │ Kardinalite                  │ 6 binary, 10 low-card, 3 numeric │
│     │                              │ "No internet service" redundancy │
│     │                              │ PhoneService↔MultipleLines V=1.0 │
│  7  │ Sabit/Quasi-sabit            │ ✓ Yok                            │
│  8  │ Outlier                      │ ✓ Yok (bounded veri)             │
│  9  │ Dağılım                      │ TotalCharges hafif çarpık (0.91) │
│     │                              │ Train-test pratik fark < %1      │
│ 10  │ Korelasyon                   │ tenure↔Total r=0.77              │
│     │                              │ PaymentMethod V_target=0.48      │
│     │                              │ Contract V_target=0.47           │
│     │                              │ gender V_target=0.007 (işe yaramaz)│
│ 11  │ VIF                          │ TotalCharges VIF=8.87 (LR için)  │
│ 12  │ Feature-Target               │ tenure d=1.19 ★★★ en güçlü      │
│     │                              │ Senior churn %50 vs %19          │
│     │                              │ E-check churn %49 vs %7-8        │
│     │                              │ MTM churn %42 vs %1-6            │
│     │                              │ Fiber churn %42 vs %10           │
│     │                              │ Churn median tenure = 10 ay      │
│ 13  │ Adversarial Validation       │ AUC=0.5105 ✓ drift yok          │
│ 14  │ Leakage                      │ ✓ Yok                            │
│ 15  │ Class Imbalance              │ 3.44:1, StratifiedKFold mükemmel │
│ 16  │ Temporal Pattern             │ ✓ Yok, StratifiedKFold uygun     │
└─────┴──────────────────────────────┴──────────────────────────────────┘
```

## FE + Encoding + Modelleme — Final Kararlar

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  VERİ KALİTESİ:          Mükemmel (eksik/outlier/leak yok)     │
│  DRIFT:                  Yok (AUC 0.51)                        │
│  CV STRATEJİSİ:          StratifiedKFold (temporal pattern yok)│
│  IMBALANCE:              scale_pos_weight=3.44                  │
│  KRİTİK FEATURE'LAR:     tenure, Contract, PaymentMethod,     │
│                          InternetService, OnlineSecurity        │
│  GEREKSIZ FEATURE'LAR:   gender (~0 bilgi), PhoneService       │
│                          (MultipleLines ile redundant)          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Tam Pipeline Kodu

```python
# ══════════════════════════════════════════════════════════════
# EDA SONRASI — FE + ENCODING + MODELleme
# ══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import warnings, os, sys
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy import stats as sp_stats

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

SEED     = 42
N_SPLITS = 5
N_SEEDS  = 3
target   = "Churn"
idc      = "id"

# ══════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════

train = pd.read_csv("/kaggle/input/competitions/playground-series-s6e3/train.csv", index_col=idc)
test  = pd.read_csv("/kaggle/input/competitions/playground-series-s6e3/test.csv",  index_col=idc)

# Target encode (ADIM 2)
train[target] = train[target].map({"Yes": 1, "No": 0})

# SeniorCitizen fix (ADIM 3)
for df in [train, test]:
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Target: {train[target].mean():.4f} pozitif")


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
# Her FE kararının hangi EDA bulgusundan çıktığı yanına yazılıdır

def feature_engineering(df):
    df = df.copy()
    
    # ── ADIM 12 → tenure en güçlü sinyal (d=1.19) ──
    # Churn P25=3, P50=10, P75=25 → bu noktalarda kır
    df["is_new_customer"]   = (df["tenure"] <= 3).astype(np.int8)     # %25 churn ilk 3 ayda
    df["is_first_year"]     = (df["tenure"] <= 12).astype(np.int8)    # ilk yıl yüksek risk
    df["is_loyal"]          = (df["tenure"] >= 48).astype(np.int8)    # 4+ yıl sadık
    df["tenure_bin"]        = pd.cut(
        df["tenure"], 
        bins=[0, 3, 12, 24, 48, 72],
        labels=[0, 1, 2, 3, 4]
    ).astype(np.int8)
    
    # ADIM 9 → tenure simetrik ama uniform-like, polynomial dene
    df["log_tenure"]        = np.log1p(df["tenure"])
    df["tenure_sq"]         = (df["tenure"] ** 2).astype(np.int32)
    
    # ── ADIM 10 → tenure↔TotalCharges r=0.77, ratio ile decompose ──
    df["avg_monthly_charge"] = (df["TotalCharges"] / (df["tenure"] + 1)).astype(np.float32)
    
    # ── ADIM 12 → MonthlyCharges churn'de +20$ fark (d=0.76) ──
    df["charge_deviation"]   = (df["MonthlyCharges"] - df["avg_monthly_charge"]).astype(np.float32)
    df["expected_total"]     = (df["MonthlyCharges"] * df["tenure"]).astype(np.float32)
    df["total_charge_diff"]  = (df["TotalCharges"] - df["expected_total"]).astype(np.float32)
    
    # ADIM 9 → TotalCharges skew=0.91, log faydalı
    df["log_total_charges"]  = np.log1p(df["TotalCharges"]).astype(np.float32)
    df["log_monthly"]        = np.log1p(df["MonthlyCharges"]).astype(np.float32)
    
    # ── ADIM 12 → Electronic check %48.9 churn (6-7x fark) ──
    df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(np.int8)
    df["is_auto_pay"]         = df["PaymentMethod"].isin([
        "Credit card (automatic)", "Bank transfer (automatic)"
    ]).astype(np.int8)
    
    # ── ADIM 12 → MTM %42.1 churn, Two year %1.0 ──
    df["is_mtm"]              = (df["Contract"] == "Month-to-month").astype(np.int8)
    df["contract_ordinal"]    = df["Contract"].map({
        "Month-to-month": 0, "One year": 1, "Two year": 2
    }).astype(np.int8)
    
    # ── ADIM 12 → Fiber %41.5 churn (4x DSL) ──
    df["is_fiber"]            = (df["InternetService"] == "Fiber optic").astype(np.int8)
    df["has_internet"]        = (df["InternetService"] != "No").astype(np.int8)
    
    # ── ADIM 6 → "No internet service" redundancy doğrulandı ──
    # → has_internet flag + orijinal sütunları da tutalım (tree modeller kullanır)
    
    # ── ADIM 12 → OnlineSecurity=No %40.6, TechSupport=No %40.2 ──
    df["has_no_protection"] = (
        (df["OnlineSecurity"] != "Yes") & 
        (df["TechSupport"] != "Yes") & 
        (df["InternetService"] != "No")
    ).astype(np.int8)
    
    # ── ADIM 6 + 12 → servis sayımları ──
    online_services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    streaming       = ["StreamingTV", "StreamingMovies"]
    all_services    = ["PhoneService"] + online_services + streaming
    
    df["service_count"]         = (df[all_services] == "Yes").sum(axis=1).astype(np.int8)
    df["online_service_count"]  = (df[online_services] == "Yes").sum(axis=1).astype(np.int8)
    df["streaming_count"]       = (df[streaming] == "Yes").sum(axis=1).astype(np.int8)
    
    # ── ADIM 12 → SeniorCitizen=1 → %50 churn ──
    df["senior_alone"] = (
        (df["SeniorCitizen"] == 1) & 
        (df["Partner"] == "No") & 
        (df["Dependents"] == "No")
    ).astype(np.int8)
    df["senior_monthly"] = (df["SeniorCitizen"] * df["MonthlyCharges"]).astype(np.float32)
    
    # ── ADIM 10 → Partner↔Dependents V=0.53 → birleştir ──
    df["family_size"] = (
        (df["Partner"] == "Yes").astype(int) + 
        (df["Dependents"] == "Yes").astype(int)
    ).astype(np.int8)
    
    # ── INTERACTION'LAR (güçlü + bağımsız feature'ları çarp) ──
    # ADIM 12 → MTM + E-check = çift risk
    df["mtm_electronic"]      = (df["is_mtm"] * df["is_electronic_check"]).astype(np.int8)
    # ADIM 12 → Fiber + no support = en riskli combo
    df["fiber_no_support"]    = (df["is_fiber"] * df["has_no_protection"]).astype(np.int8)
    # ADIM 12 → paperless + e-check = düşük bağlılık
    df["epay_paperless"]      = (
        (df["PaperlessBilling"] == "Yes").astype(int) * df["is_electronic_check"]
    ).astype(np.int8)
    
    # ── Contract × tenure (ADIM 10 → ayrı ayrı güçlü, birlikte daha da güçlü) ──
    df["contract_tenure"]     = (df["contract_ordinal"] * df["tenure"]).astype(np.int16)
    df["contract_monthly"]    = (df["contract_ordinal"] * df["MonthlyCharges"]).astype(np.float32)
    
    # ── Charge ratios (ADIM 10+12) ──
    df["monthly_per_service"] = (df["MonthlyCharges"] / (df["service_count"] + 1)).astype(np.float32)
    df["total_per_service"]   = (df["TotalCharges"] / (df["service_count"] + 1)).astype(np.float32)
    df["monthly_tenure_ratio"]= (df["MonthlyCharges"] / (df["tenure"] + 1)).astype(np.float32)
    
    # ── Yeni müşteri + fiber + MTM + e-check = ultra yüksek risk ──
    df["ultra_risk"] = (
        df["is_new_customer"] * df["is_fiber"] * df["is_mtm"] * df["is_electronic_check"]
    ).astype(np.int8)
    
    return df


print("Feature Engineering...")
train = feature_engineering(train)
test  = feature_engineering(test)
print(f"Train: {train.shape}, Test: {test.shape}")


# ══════════════════════════════════════════════════════════════
# ENCODING
# ══════════════════════════════════════════════════════════════

BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
MULTI_COLS  = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]

def encode_features(tr, te):
    tr, te = tr.copy(), te.copy()
    
    # Binary → 0/1
    binary_maps = {
        "gender":           {"Male": 1, "Female": 0},
        "Partner":          {"Yes": 1, "No": 0},
        "Dependents":       {"Yes": 1, "No": 0},
        "PhoneService":     {"Yes": 1, "No": 0},
        "PaperlessBilling": {"Yes": 1, "No": 0},
    }
    for col, m in binary_maps.items():
        tr[col] = tr[col].map(m).astype(np.int8)
        te[col] = te[col].map(m).astype(np.int8)
    
    # Ordinal encoding
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    tr[MULTI_COLS] = enc.fit_transform(tr[MULTI_COLS]).astype(np.int16)
    te[MULTI_COLS] = enc.transform(te[MULTI_COLS]).astype(np.int16)
    
    # Frequency encoding
    for col in MULTI_COLS:
        freq = tr[col].value_counts(normalize=True)
        tr[f"{col}_freq"] = tr[col].map(freq).astype(np.float32)
        te[f"{col}_freq"] = te[col].map(freq).fillna(0).astype(np.float32)
    
    return tr, te

train, test = encode_features(train, test)

y      = train[target]
X      = train.drop(columns=[target])
X_test = test.copy()

print(f"Final features: {X.shape[1]}")
print(f"Target: {y.mean():.4f}")


# ══════════════════════════════════════════════════════════════
# TARGET ENCODING (fold-aware, Bayesian smoothing)
# ══════════════════════════════════════════════════════════════
# ADIM 12 → kategorik feature'lar target ile güçlü ilişkili
# ADIM 10 → redundancy var ama TE ile her sütunun kendi
#            target ilişkisini yakalarız

def target_encode_fold(X_tr, X_val, X_te, y_tr, cols, smoothing=30.0):
    """Bayesian smoothed target encoding — per fold."""
    global_mean = y_tr.mean()
    X_tr, X_val, X_te = X_tr.copy(), X_val.copy(), X_te.copy()
    
    for col in cols:
        agg = pd.DataFrame({"col": X_tr[col], "target": y_tr})
        stats = agg.groupby("col")["target"].agg(["mean", "count"])
        smooth = (
            (stats["count"] * stats["mean"] + smoothing * global_mean) /
            (stats["count"] + smoothing)
        )
        te_col = f"{col}_te"
        X_tr[te_col] = X_tr[col].map(smooth).fillna(global_mean).astype(np.float32)
        X_val[te_col] = X_val[col].map(smooth).fillna(global_mean).astype(np.float32)
        X_te[te_col]  = X_te[col].map(smooth).fillna(global_mean).astype(np.float32)
    
    return X_tr, X_val, X_te


# ══════════════════════════════════════════════════════════════
# MODEL TRAINING — Multi-seed + K-Fold
# ══════════════════════════════════════════════════════════════

# ADIM 2  → imbalance 3.44:1 → scale_pos_weight
# ADIM 15 → StratifiedKFold mükemmel çalışıyor
# ADIM 16 → temporal pattern yok → random shuffle OK

lgb_params = dict(
    n_estimators=4000, learning_rate=0.008, num_leaves=63,
    max_depth=-1, min_child_samples=20,
    colsample_bytree=0.7, subsample=0.75, subsample_freq=1,
    reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=3.44,   # ← ADIM 15
    verbose=-1,
)

xgb_params = dict(
    n_estimators=4000, learning_rate=0.008, max_depth=6,
    min_child_weight=5, subsample=0.75, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=3.44,   # ← ADIM 15
    tree_method="hist", early_stopping_rounds=150,
    verbosity=0,
)

cat_params = dict(
    iterations=4000, learning_rate=0.008, depth=6,
    l2_leaf_reg=3.0, min_data_in_leaf=20,
    auto_class_weights="Balanced",  # ← ADIM 15
    early_stopping_rounds=150, verbose=0,
)

lr_params = dict(
    C=1.0, max_iter=2000, solver="lbfgs",
    class_weight="balanced",  # ← ADIM 15
)


def train_model_cv(X, y, X_test, model_type, params, multi_cols, n_seeds=N_SEEDS):
    """Multi-seed averaged K-fold CV."""
    
    all_oof  = np.zeros((n_seeds, len(X)))
    all_test = np.zeros((n_seeds, len(X_test)))
    
    for seed_idx in range(n_seeds):
        current_seed = SEED + seed_idx * 1000
        kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=current_seed)
        oof = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))
        scores = []
        
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr  = X.iloc[tr_idx]
            X_val = X.iloc[val_idx]
            y_tr  = y.iloc[tr_idx]
            y_val = y.iloc[val_idx]
            X_te  = X_test.copy()
            
            # Per-fold target encoding
            X_tr, X_val, X_te = target_encode_fold(X_tr, X_val, X_te, y_tr, multi_cols)
            
            p = {**params, "random_state": current_seed}
            
            _stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
            try:
                if model_type == "lgb":
                    model = lgb.LGBMClassifier(**p)
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                              callbacks=[lgb.early_stopping(150, verbose=False),
                                         lgb.log_evaluation(0)])
                elif model_type == "xgb":
                    model = xgb.XGBClassifier(**p)
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                elif model_type == "cat":
                    model = CatBoostClassifier(**p)
                    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
                elif model_type == "lr":
                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_tr)
                    X_val_s = scaler.transform(X_val)
                    X_te_s = scaler.transform(X_te)
                    model = LogisticRegression(**p)
                    model.fit(X_tr_s, y_tr)
                
                if model_type == "lr":
                    val_pred  = model.predict_proba(X_val_s)[:, 1]
                    test_fold = model.predict_proba(X_te_s)[:, 1]
                else:
                    val_pred  = model.predict_proba(X_val)[:, 1]
                    test_fold = model.predict_proba(X_te)[:, 1]
            finally:
                sys.stdout = _stdout
            
            oof[val_idx] = val_pred
            test_preds  += test_fold / N_SPLITS
            scores.append(roc_auc_score(y_val, val_pred))
        
        all_oof[seed_idx]  = oof
        all_test[seed_idx] = test_preds
        oof_auc = roc_auc_score(y, oof)
        print(f"  {model_type.upper()} seed={current_seed} "
              f"OOF AUC: {oof_auc:.6f} (folds: {[f'{s:.4f}' for s in scores]})")
    
    final_oof  = all_oof.mean(axis=0)
    final_test = all_test.mean(axis=0)
    final_auc  = roc_auc_score(y, final_oof)
    print(f"  {model_type.upper()} FINAL ({n_seeds}-seed avg) OOF AUC: {final_auc:.6f}\n")
    return final_oof, final_test


# ══════════════════════════════════════════════════════════════
# TRAIN ALL MODELS
# ══════════════════════════════════════════════════════════════

print("=" * 60)
print("TRAINING")
print("=" * 60)

oof_lgb, test_lgb = train_model_cv(X, y, X_test, "lgb", lgb_params, MULTI_COLS)
oof_xgb, test_xgb = train_model_cv(X, y, X_test, "xgb", xgb_params, MULTI_COLS)
oof_cat, test_cat = train_model_cv(X, y, X_test, "cat", cat_params, MULTI_COLS)
oof_lr,  test_lr  = train_model_cv(X, y, X_test, "lr",  lr_params,  MULTI_COLS, n_seeds=1)


# ══════════════════════════════════════════════════════════════
# ENSEMBLE
# ══════════════════════════════════════════════════════════════

print("=" * 60)
print("ENSEMBLE")
print("=" * 60)

# ── 1) Weighted Average ──
w = {"lgb": 0.35, "xgb": 0.25, "cat": 0.30, "lr": 0.10}
oof_dict  = {"lgb": oof_lgb, "xgb": oof_xgb, "cat": oof_cat, "lr": oof_lr}
test_dict = {"lgb": test_lgb, "xgb": test_xgb, "cat": test_cat, "lr": test_lr}

oof_weighted  = sum(w[k] * oof_dict[k] for k in w)
test_weighted = sum(w[k] * test_dict[k] for k in w)
print(f"Weighted Avg OOF AUC: {roc_auc_score(y, oof_weighted):.6f}")

# ── 2) Rank Average ──
def rank_average(preds_list, weights):
    ranks = [sp_stats.rankdata(p) / len(p) for p in preds_list]
    return sum(w * r for w, r in zip(weights, ranks))

weight_list = [w[k] for k in oof_dict]
oof_rank  = rank_average(list(oof_dict.values()), weight_list)
test_rank = rank_average(list(test_dict.values()), weight_list)
print(f"Rank Avg OOF AUC:     {roc_auc_score(y, oof_rank):.6f}")

# ── 3) Stacking ──
from sklearn.linear_model import LogisticRegression as LR_meta

oof_stack_features  = np.column_stack(list(oof_dict.values()))
test_stack_features = np.column_stack(list(test_dict.values()))

# Pairwise products
n_models = oof_stack_features.shape[1]
for i in range(n_models):
    for j in range(i+1, n_models):
        oof_stack_features  = np.column_stack([oof_stack_features,  oof_stack_features[:,i] * oof_stack_features[:,j]])
        test_stack_features = np.column_stack([test_stack_features, test_stack_features[:,i] * test_stack_features[:,j]])

kf_meta = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_stacked  = np.zeros(len(y))
test_stacked = np.zeros(len(test_stack_features))

for tr_idx, val_idx in kf_meta.split(oof_stack_features, y):
    sc = StandardScaler()
    X_tr_m = sc.fit_transform(oof_stack_features[tr_idx])
    X_val_m = sc.transform(oof_stack_features[val_idx])
    X_te_m  = sc.transform(test_stack_features)
    
    meta = LR_meta(C=1.0, max_iter=2000)
    meta.fit(X_tr_m, y.iloc[tr_idx])
    
    oof_stacked[val_idx] = meta.predict_proba(X_val_m)[:, 1]
    test_stacked        += meta.predict_proba(X_te_m)[:, 1] / N_SPLITS

print(f"Stacked OOF AUC:      {roc_auc_score(y, oof_stacked):.6f}")

# ── Best ensemble seç ──
candidates = {
    "weighted": (oof_weighted, test_weighted),
    "rank_avg": (oof_rank, test_rank),
    "stacked":  (oof_stacked, test_stacked),
}

best_name, best_auc = "", 0
for name, (oof_c, _) in candidates.items():
    auc = roc_auc_score(y, oof_c)
    print(f"  {name:12s} → AUC: {auc:.6f}")
    if auc > best_auc:
        best_auc = auc
        best_name = name

final_oof, final_test = candidates[best_name]
print(f"\n★ Best: {best_name} (AUC: {best_auc:.6f})")

# ══════════════════════════════════════════════════════════════
# SUBMISSION
# ══════════════════════════════════════════════════════════════

sub = pd.read_csv("/kaggle/input/competitions/playground-series-s6e3/sample_submission.csv")
sub[target] = final_test
sub.to_csv("submission.csv", index=False)
print(f"\nSubmission saved: {sub.shape}")
print(f"Pred stats: mean={final_test.mean():.4f} std={final_test.std():.4f}")
print("DONE ✓")
```

## FE Traceability Özeti

```
┌──────────────────────────┬──────────────────────────────────────┐
│ FEATURE                  │ HANGİ EDA ADIMININ ÇIKTISI           │
├──────────────────────────┼──────────────────────────────────────┤
│ is_new_customer          │ ADIM 12: churn P25=3 ay              │
│ is_first_year            │ ADIM 12: churn P50=10 ay             │
│ is_loyal                 │ ADIM 12: no-churn P75=66             │
│ tenure_bin               │ ADIM 12: churn dağılım kırılımları   │
│ log_tenure               │ ADIM 9: dağılım normalizasyonu       │
│ tenure_sq                │ ADIM 10: Spearman>Pearson farkı      │
│ avg_monthly_charge       │ ADIM 10: tenure↔Total r=0.77         │
│ charge_deviation         │ ADIM 10: decompose ilişkiyi          │
│ expected_total           │ ADIM 10: tenure × Monthly            │
│ total_charge_diff        │ ADIM 10: gerçek-beklenen fark        │
│ log_total_charges        │ ADIM 9: skew=0.91                    │
│ is_electronic_check      │ ADIM 12: %48.9 churn (6-7x fark)    │
│ is_auto_pay              │ ADIM 12: auto pay %7 vs manual %49   │
│ is_mtm                   │ ADIM 12: %42.1 churn                 │
│ contract_ordinal         │ ADIM 6: doğal sıralama var           │
│ is_fiber                 │ ADIM 12: %41.5 churn (4x DSL)       │
│ has_internet             │ ADIM 6: redundancy flag              │
│ has_no_protection        │ ADIM 12: Security+Support=No → %40+  │
│ service_count            │ ADIM 6+10: aggregation               │
│ online_service_count     │ ADIM 6: servis cluster'ı             │
│ streaming_count          │ ADIM 10: TV↔Movies V=0.81            │
│ senior_alone             │ ADIM 12: Senior %50 churn            │
│ senior_monthly           │ ADIM 12: Senior × charge             │
│ family_size              │ ADIM 10: Partner↔Dependents V=0.53   │
│ mtm_electronic           │ ADIM 12: çift risk interaction       │
│ fiber_no_support         │ ADIM 12: en riskli combo             │
│ epay_paperless           │ ADIM 12: düşük bağlılık combo        │
│ contract_tenure          │ ADIM 10+12: güçlü+bağımsız interact  │
│ contract_monthly         │ ADIM 10+12: sözleşme × ücret         │
│ monthly_per_service      │ ADIM 12: hizmet başına maliyet       │
│ ultra_risk               │ ADIM 12: tüm risk faktörleri birlikte│
│ scale_pos_weight=3.44    │ ADIM 2+15: imbalance oranı           │
│ StratifiedKFold          │ ADIM 15+16: dengeli fold + no time   │
│ target_encode (per-fold) │ ADIM 12: kategorik-target ilişki     │
│ Bayesian smoothing=30    │ ADIM 6: düşük kardinalite, smoothing │
└──────────────────────────┴──────────────────────────────────────┘
```

Her FE kararı bir EDA bulgusuna dayanıyor — hiçbiri "rastgele deneme" değil.