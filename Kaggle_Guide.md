# 🏆 KAGGLE YARIŞMASI — TAM REHBER
## Senior & Grand Master Seviyesi · Baştan Sona Her Adım

---

> **Bu rehberin amacı:** Her Kaggle yarışmasına uygulanabilecek, sıfırdan submission'a ve ötesine kadar her adımı kapsayan, karar noktalarını net açıklayan, copy-paste hazır kod örnekleri içeren evrensel bir template.

---

## İÇİNDEKİLER

### TEMEL REHBER
1. [Yarışmayı Anlama & Strateji Kurma](#1-yarışmayı-anlama--strateji-kurma)
2. [Ortam Kurulumu](#2-ortam-kurulumu)
3. [Veri Yükleme & İlk Bakış](#3-veri-yükleme--ilk-bakış)
4. [Keşifsel Veri Analizi (EDA)](#4-keşifsel-veri-analizi-eda)
5. [Veri Temizleme](#5-veri-temizleme)
6. [Feature Engineering](#6-feature-engineering)
7. [Model Seçimi & Baseline](#7-model-seçimi--baseline)
8. [Cross-Validation Stratejisi](#8-cross-validation-stratejisi)
9. [Hyperparameter Optimization](#9-hyperparameter-optimization)
10. [Ensemble & Stacking](#10-ensemble--stacking)
11. [Post-Processing & Threshold Tuning](#11-post-processing--threshold-tuning)
12. [Submission & Leaderboard Stratejisi](#12-submission--leaderboard-stratejisi)
13. [Sıkışınca Ne Yapmalı](#13-sıkışınca-ne-yapmalı)
14. [Yarışma Bittikten Sonra](#14-yarışma-bittikten-sonra)
15. [Hızlı Başvuru: Metrik Sözlüğü](#15-hızlı-başvuru-metrik-sözlüğü)

### EK BÖLÜMLER
16. [Veri Boyutuna Göre Strateji](#16-veri-boyutuna-göre-strateji)
17. [NLP — Metin Özellikleri](#17-nlp--metin-özellikleri)
18. [Görüntü & Multi-modal Yarışmalar](#18-görüntü--multi-modal-yarışmalar)
19. [Neural Networks on Tabular Data](#19-neural-networks-on-tabular-data)
20. [Zaman Serisi — Detaylı Rehber](#20-zaman-serisi--detaylı-rehber)
21. [Dengesiz Veri (Imbalanced Data)](#21-dengesiz-veri-imbalanced-data)
22. [Özel Problem Tipleri](#22-özel-problem-tipleri)
23. [Experiment Tracking](#23-experiment-tracking)
24. [GPU Kullanımı](#24-gpu-kullanımı)
25. [Kaggle API & Otomasyon](#25-kaggle-api--otomasyon)
26. [Forum & Discussion Stratejisi](#26-forum--discussion-stratejisi)
27. [Takım Çalışması](#27-takım-çalışması)
28. [Özel Loss Fonksiyonları](#28-özel-loss-fonksiyonları)
29. [Post-Competition Analiz](#29-post-competition-analiz)
30. [Hızlı Başvuru Kartları](#30-hızlı-başvuru-kartları)

---

# 1. YARIŞMAYI ANLAMA & STRATEJİ KURMA

## 1.1 Yarışma Sayfasını Sistematik Okuma

Yarışma sayfasına girildiğinde sırayla şunlara bakılır:

```
OKUMA SIRASI:
1. Overview → Problem ne?
2. Evaluation → Metrik ne?  ← EN KRİTİK
3. Data → Kaç dosya, hangi format, ne kadar veri?
4. Discussion → Başka katılımcılar ne keşfetti?
5. Notebooks → Public kernel'larda baseline nerede?
```

## 1.2 Metriği Anlama (En Kritik Adım)

Metrik yarışma boyunca tüm kararları belirler. Yanlış metrik optimize etmek = sıfır ilerleme.

```python
# ─── METRİK KARAR TABLOSU ───────────────────────────────────────────────────
"""
Classification:
  AUC-ROC     → Eşik önemsiz, sıralama önemli → predict_proba kullan
  Logloss     → Kalibrasyon önemli → CalibratedClassifierCV kullan
  F1 / F-beta → Threshold tuning kritik → precision-recall curve çiz
  Accuracy    → Nadiren kullanılır, imbalance'a dikkat
  MCC         → İkili, çok dengesiz veri için
  
Regression:
  RMSE        → Büyük hatalara ceza ağır → outlier temizle, log transform
  MAE         → Robust, medyan tahmincisi optimal
  RMSLE       → log(y+1) predict et, sonra exp-1 al
  R²          → Görece metrik
  MAPE        → Küçük y değerlerinde patlayabilir
  
Special:
  MAP@K       → Sıralama önemli → LTR modelleri
  IoU/Dice    → Segmentasyon
  BLEU/ROUGE  → NLP üretim
"""

# Metriği hemen implement et — her denemede kullan
import numpy as np
from sklearn.metrics import (
    roc_auc_score, log_loss, f1_score, mean_squared_error,
    mean_absolute_error, r2_score
)

def competition_metric(y_true, y_pred):
    """Yarışma metriğini buraya yaz — değiştirme"""
    # Örnek: RMSLE
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(np.maximum(y_pred, 0))))

def weighted_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

def map_at_k(y_true, y_pred_proba, k=5):
    """MAP@K için"""
    top_k = np.argsort(y_pred_proba, axis=1)[:, -k:][:, ::-1]
    ap_list = []
    for i, (true, pred) in enumerate(zip(y_true, top_k)):
        hits = np.cumsum([1 if p == true else 0 for j, p in enumerate(pred)])
        prec = [hits[j]/(j+1) for j in range(k) if pred[j] == true]
        ap_list.append(np.mean(prec) if prec else 0.0)
    return np.mean(ap_list)
```

## 1.3 Problem Tipi Tespiti

```python
"""
PROBLEM TİPİ KARAR AĞACI:
─────────────────────────────────────────────────────
Target binary (0/1)?
  → Binary Classification
  
Target multi-class (>2 sınıf)?
  → Multi-class Classification
  → Sınıf sayısı > 10 ise → Hierarchical veya Label Encoding
  
Target sürekli sayı?
  → Regression
  → log(y+1) çarpık mı? → Log transform + regresyon
  
Target birden fazla label?
  → Multi-label Classification
  
Her satır = bir zaman noktası?
  → Time Series Forecasting
  → Group-based CV (zamana göre split)
  
Her satır = görüntü/metin?
  → Deep Learning (CNN / Transformer)
"""

# Target analizi
import pandas as pd
import numpy as np

def analyze_target(df, target_col):
    y = df[target_col].dropna()
    
    n_unique = y.nunique()
    dtype = y.dtype
    
    print(f"Target: {target_col}")
    print(f"dtype: {dtype}, unique: {n_unique}, total: {len(y)}")
    print(f"Missing: {df[target_col].isna().sum()}")
    
    if dtype == 'object' or n_unique <= 20:
        print(f"\n→ CLASSIFICATION ({n_unique} sınıf)")
        print(y.value_counts(normalize=True).round(4))
        # Imbalance check
        vc = y.value_counts(normalize=True)
        if vc.iloc[0] > 0.9:
            print("⚠️  EXTREME IMBALANCE — özel strateji gerekli")
        elif vc.iloc[0] > 0.7:
            print("⚠️  Moderate imbalance")
    else:
        print(f"\n→ REGRESSION")
        print(y.describe())
        skew = y.skew()
        print(f"Skewness: {skew:.3f}")
        if abs(skew) > 1:
            print(f"⚠️  Çarpık dağılım — log transform dene")
```

## 1.4 İlk 30 Dakika Planı

```
DAKİKA 0-5:   Metriği anla, implement et
DAKİKA 5-10:  train.csv + test.csv → shape, dtypes, missing
DAKİKA 10-20: Target dağılımı, temel EDA
DAKİKA 20-25: Public notebook'lara bak (kendi kodunu yazmadan önce)
DAKİKA 25-30: İlk baseline kur, submission oluştur
```

---

# 2. ORTAM KURULUMU

## 2.1 Gerekli Kütüphaneler

```python
# requirements — bu cell her yarışmada çalıştırılır
import warnings
warnings.filterwarnings('ignore')

# Core
import numpy as np
import pandas as pd
import os, gc, sys, time, pickle, json
from pathlib import Path
from copy import deepcopy

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
plt.style.use('seaborn-v0_8-darkgrid')

# ML
from sklearn.model_selection import (
    StratifiedKFold, KFold, GroupKFold, StratifiedGroupKFold,
    cross_val_score, cross_val_predict, train_test_split
)
from sklearn.preprocessing import (
    LabelEncoder, OrdinalEncoder, StandardScaler, RobustScaler,
    MinMaxScaler, QuantileTransformer, PowerTransformer
)
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer

# Boosting (en çok kullanılan)
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor
)
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, HuberRegressor
)

# Feature Engineering
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif,
    RFE, RFECV, VarianceThreshold
)
import category_encoders as ce  # pip install category_encoders

# HPO
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Interpretability
import shap

# Utility
from tqdm.auto import tqdm
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency

print("✅ Tüm kütüphaneler yüklendi")
```

## 2.2 Proje Klasör Yapısı

```python
# Klasör yapısı — her yarışmada aynı
from pathlib import Path

COMP_NAME = "competition-name"  # Yarışma adını yaz
BASE_DIR  = Path(f"/kaggle/input/{COMP_NAME}")
WORK_DIR  = Path("/kaggle/working")
OUT_DIR   = WORK_DIR / "output"
MODEL_DIR = WORK_DIR / "models"
FIG_DIR   = WORK_DIR / "figures"

for d in [OUT_DIR, MODEL_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TRAIN_PATH  = BASE_DIR / "train.csv"
TEST_PATH   = BASE_DIR / "test.csv"
SAMPLE_PATH = BASE_DIR / "sample_submission.csv"

print("Dosyalar:")
for f in BASE_DIR.iterdir():
    print(f"  {f.name}: {f.stat().st_size/1024**2:.1f} MB")
```

## 2.3 Seed & Reproducibility

```python
import random, torch

SEED = 42

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except:
        pass

seed_everything(SEED)
print(f"✅ Seed {SEED} ayarlandı")
```

---

# 3. VERİ YÜKLEME & İLK BAKIŞ

## 3.1 Veri Yükleme

```python
def load_data(train_path, test_path, sample_path=None):
    """Standart veri yükleme — dtype optimizasyonu dahil"""
    
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    
    print(f"Train: {train.shape}  |  Test: {test.shape}")
    
    if sample_path:
        sample = pd.read_csv(sample_path)
        print(f"Sample submission: {sample.shape}")
        return train, test, sample
    
    return train, test

train, test, sample = load_data(TRAIN_PATH, TEST_PATH, SAMPLE_PATH)
```

## 3.2 Büyük Veri için Memory Optimizasyonu

```python
def optimize_dtypes(df):
    """RAM kullanımını %40-70 azaltır"""
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object' and col_type.name != 'category':
            c_min, c_max = df[col].min(), df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)  # float16 çoğu durumda unstable
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        
        elif col_type == 'object':
            if df[col].nunique() / len(df) < 0.5:  # düşük kardinalite
                df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory: {start_mem:.1f} MB → {end_mem:.1f} MB ({100*(start_mem-end_mem)/start_mem:.0f}% azaldı)")
    return df

train = optimize_dtypes(train)
test  = optimize_dtypes(test)
```

## 3.3 Hızlı Profil

```python
def quick_profile(df, name="Dataset"):
    """30 saniyede veri setini tanı"""
    print(f"\n{'='*60}")
    print(f"📊 {name}  —  {df.shape[0]:,} satır × {df.shape[1]} sütun")
    print(f"{'='*60}")
    
    # Tip özeti
    type_counts = df.dtypes.value_counts()
    print("\nSütun Tipleri:")
    for dtype, count in type_counts.items():
        print(f"  {str(dtype):15} : {count}")
    
    # Eksik değerler
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss) > 0:
        print(f"\nEksik Değer ({len(miss)} sütun):")
        for col, n in miss.head(15).items():
            print(f"  {col:40} {n:6,}  ({n/len(df)*100:.1f}%)")
    else:
        print("\n✅ Eksik değer yok")
    
    # Duplike
    dups = df.duplicated().sum()
    print(f"\nDuplike satır: {dups:,}  ({dups/len(df)*100:.2f}%)")
    
    # Sayısal özet
    print("\nSayısal Özet:")
    print(df.describe().round(3).to_string())
    
    return miss

miss_train = quick_profile(train, "Train")
miss_test  = quick_profile(test, "Test")
```

---

# 4. KEŞİFSEL VERİ ANALİZİ (EDA)

## 4.1 Target Analizi

```python
TARGET = "target"  # ← değiştir

# ─── CLASSIFICATION ──────────────────────────────────────────────
def eda_classification_target(df, target):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    vc = df[target].value_counts()
    
    # Bar chart
    vc.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='white')
    axes[0].set_title(f'Sınıf Dağılımı: {target}', fontsize=14)
    for i, (idx, val) in enumerate(vc.items()):
        axes[0].text(i, val, f'{val:,}\n({val/len(df)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=10)
    
    # Pie
    vc.plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
    axes[1].set_title('Oran', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'target_dist.png', dpi=120, bbox_inches='tight')
    plt.show()
    
    # Imbalance ratio
    ratio = vc.max() / vc.min()
    print(f"\nImbalance ratio: {ratio:.1f}:1")
    if ratio > 10:
        print("⚠️  EXTREME IMBALANCE → SMOTE / class_weight / oversampling gerekli")
    elif ratio > 3:
        print("⚠️  Moderate imbalance → class_weight='balanced' dene")

# ─── REGRESSION ──────────────────────────────────────────────────
def eda_regression_target(df, target):
    y = df[target].dropna()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Distribution
    axes[0].hist(y, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].set_title(f'{target} Dağılımı', fontsize=13)
    axes[0].axvline(y.mean(), color='red', linestyle='--', label=f'Mean={y.mean():.2f}')
    axes[0].axvline(y.median(), color='orange', linestyle='--', label=f'Median={y.median():.2f}')
    axes[0].legend()
    
    # Log transform
    y_log = np.log1p(y.clip(lower=0))
    axes[1].hist(y_log, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[1].set_title(f'log1p({target}) Dağılımı', fontsize=13)
    
    # QQ plot
    stats.probplot(y, dist='norm', plot=axes[2])
    axes[2].set_title('Q-Q Plot (Normallik)', fontsize=13)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nİstatistikler:")
    print(f"  Mean:   {y.mean():.4f}")
    print(f"  Median: {y.median():.4f}")
    print(f"  Std:    {y.std():.4f}")
    print(f"  Skew:   {y.skew():.4f}  {'⚠️ Çarpık → log transform' if abs(y.skew()) > 1 else '✅'}")
    print(f"  Kurt:   {y.kurtosis():.4f}")
    print(f"  Min:    {y.min():.4f}")
    print(f"  Max:    {y.max():.4f}")
    print(f"  Zeros:  {(y==0).sum():,}  ({(y==0).mean()*100:.2f}%)")
    print(f"  Negatives: {(y<0).sum():,}")
```

## 4.2 Sayısal Sütun Analizi

```python
def analyze_numeric_features(df, target=None, max_cols=20):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target and target in num_cols:
        num_cols.remove(target)
    num_cols = num_cols[:max_cols]
    
    n = len(num_cols)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5*n))
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(num_cols):
        vals = df[col].dropna()
        
        # Histogram + KDE
        axes[i, 0].hist(vals, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='white')
        vals.plot(kind='kde', ax=axes[i, 0], color='red', linewidth=2)
        axes[i, 0].set_title(f'{col} — Hist+KDE', fontsize=11)
        axes[i, 0].set_xlabel(f'skew={vals.skew():.2f}  |  kurt={vals.kurtosis():.2f}')
        
        # Box plot
        axes[i, 1].boxplot(vals, vert=True, patch_artist=True,
                           boxprops=dict(facecolor='steelblue', alpha=0.6))
        q1, q3 = vals.quantile([0.25, 0.75])
        iqr = q3 - q1
        outlier_pct = ((vals < q1-1.5*iqr) | (vals > q3+1.5*iqr)).mean() * 100
        axes[i, 1].set_title(f'{col} — Box (Outlier: {outlier_pct:.1f}%)', fontsize=11)
        
        # vs Target
        if target and target in df.columns:
            if df[target].nunique() <= 10:  # classification
                for j, cls in enumerate(df[target].unique()):
                    mask = df[target] == cls
                    axes[i, 2].hist(df.loc[mask, col].dropna(), bins=30, alpha=0.5,
                                   label=str(cls), density=True)
                axes[i, 2].legend()
                axes[i, 2].set_title(f'{col} vs {target}', fontsize=11)
            else:  # regression
                axes[i, 2].scatter(df[col], df[target], alpha=0.1, s=5)
                corr = df[[col, target]].corr().iloc[0, 1]
                axes[i, 2].set_title(f'{col} vs {target} (r={corr:.3f})', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'numeric_analysis.png', dpi=100, bbox_inches='tight')
    plt.show()

# Detaylı istatistik tablosu
def numeric_stats_table(df, target=None):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target and target in num_cols:
        num_cols.remove(target)
    
    stats_list = []
    for col in num_cols:
        v = df[col].dropna()
        q1, q3 = v.quantile([0.25, 0.75])
        iqr = q3 - q1
        out_pct = ((v < q1-1.5*iqr) | (v > q3+1.5*iqr)).mean() * 100
        
        row = {
            'col': col,
            'dtype': str(df[col].dtype),
            'missing%': f"{df[col].isna().mean()*100:.1f}%",
            'unique': df[col].nunique(),
            'mean': f"{v.mean():.4f}",
            'std': f"{v.std():.4f}",
            'min': f"{v.min():.4f}",
            'q25': f"{q1:.4f}",
            'median': f"{v.median():.4f}",
            'q75': f"{q3:.4f}",
            'max': f"{v.max():.4f}",
            'skew': f"{v.skew():.3f}",
            'outlier%': f"{out_pct:.1f}%",
        }
        
        if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
            corr, _ = stats.pearsonr(df[col].fillna(df[col].median()),
                                      df[target].fillna(df[target].median()))
            row['corr_target'] = f"{corr:.4f}"
        
        stats_list.append(row)
    
    return pd.DataFrame(stats_list).set_index('col')

stats_df = numeric_stats_table(train, TARGET)
print(stats_df.to_string())
```

## 4.3 Kategorik Sütun Analizi

```python
def analyze_categorical_features(df, target=None, max_cols=15, top_n=20):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()[:max_cols]
    
    cat_stats = []
    for col in cat_cols:
        vc = df[col].value_counts()
        rare_pct = (vc < vc.sum()*0.01).sum() / len(vc) * 100
        
        row = {
            'col': col,
            'missing%': f"{df[col].isna().mean()*100:.1f}%",
            'unique': df[col].nunique(),
            'mode': str(vc.index[0]),
            'mode%': f"{vc.iloc[0]/len(df)*100:.1f}%",
            'rare%': f"{rare_pct:.1f}%",
            'encoding': 'OneHot' if df[col].nunique() <= 5
                       else 'Label/Freq' if df[col].nunique() <= 20
                       else 'Target'
        }
        
        if target and target in df.columns:
            try:
                ct = pd.crosstab(df[col], df[target])
                chi2, p, _, _ = chi2_contingency(ct)
                n = len(df)
                cramers_v = np.sqrt(chi2 / (n * (min(ct.shape)-1)))
                row['cramers_v'] = f"{cramers_v:.4f}"
                row['chi2_p'] = f"{p:.4f}"
            except:
                pass
        
        cat_stats.append(row)
    
    stats_df = pd.DataFrame(cat_stats).set_index('col')
    print(stats_df.to_string())
    
    # Top N bar charts
    fig, axes = plt.subplots(
        (len(cat_cols)+2)//3, 3,
        figsize=(18, 5 * ((len(cat_cols)+2)//3))
    )
    axes = axes.flatten()
    
    for i, col in enumerate(cat_cols):
        vc = df[col].value_counts().head(top_n)
        axes[i].barh(vc.index.astype(str), vc.values, color='steelblue')
        axes[i].set_title(f'{col} (top {top_n})', fontsize=11)
        axes[i].invert_yaxis()
    
    for j in range(len(cat_cols), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return stats_df
```

## 4.4 Korelasyon Analizi

```python
def correlation_analysis(df, target=None, threshold=0.8):
    num_df = df.select_dtypes(include=[np.number])
    
    # Pearson correlation
    corr = num_df.corr()
    
    # Heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', vmin=-1, vmax=1,
                ax=axes[0], linewidths=0.5, annot_kws={'size': 8})
    axes[0].set_title('Pearson Correlation Matrix', fontsize=13)
    
    # Target correlation bar chart
    if target and target in corr:
        target_corr = corr[target].drop(target).sort_values(key=abs, ascending=False)
        colors = ['#2563eb' if c > 0 else '#dc2626' for c in target_corr]
        axes[1].barh(range(len(target_corr)), target_corr.values, color=colors)
        axes[1].set_yticks(range(len(target_corr)))
        axes[1].set_yticklabels(target_corr.index)
        axes[1].axvline(0, color='black', linewidth=0.5)
        axes[1].set_title(f'Feature Correlation with {target}', fontsize=13)
    
    plt.tight_layout()
    plt.show()
    
    # Yüksek korelasyonlu çiftler — multicollinearity
    high_corr = []
    for i in range(len(corr)):
        for j in range(i+1, len(corr)):
            if abs(corr.iloc[i, j]) >= threshold:
                high_corr.append({
                    'col1': corr.columns[i],
                    'col2': corr.columns[j],
                    'r': corr.iloc[i, j]
                })
    
    if high_corr:
        hc_df = pd.DataFrame(high_corr).sort_values('r', key=abs, ascending=False)
        print(f"\n⚠️  Yüksek korelasyon (|r|>={threshold}) — {len(hc_df)} çift:")
        print(hc_df.to_string(index=False))
    
    return corr

# Spearman da ekle (non-linear ilişkiler için)
def spearman_analysis(df, target):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in num_cols:
        num_cols.remove(target)
    
    results = []
    for col in num_cols:
        mask = df[[col, target]].notna().all(axis=1)
        r, p = stats.spearmanr(df.loc[mask, col], df.loc[mask, target])
        results.append({'col': col, 'spearman_r': r, 'p_value': p})
    
    return pd.DataFrame(results).sort_values('spearman_r', key=abs, ascending=False)
```

## 4.5 Train vs Test Drift Analizi

```python
def train_test_drift(train, test, feature_cols=None, alpha=0.05):
    """
    KS-test ile her sütun için train/test dağılım farkını ölçer.
    Yüksek drift → model LB'de çakılabilir.
    """
    if feature_cols is None:
        feature_cols = [c for c in train.select_dtypes(include=[np.number]).columns
                       if c in test.columns]
    
    results = []
    for col in feature_cols:
        tr = train[col].dropna().values
        te = test[col].dropna().values
        if len(tr) == 0 or len(te) == 0:
            continue
        
        ks_stat, p_val = ks_2samp(tr, te)
        
        results.append({
            'col': col,
            'train_mean': tr.mean(),
            'test_mean': te.mean(),
            'train_std': tr.std(),
            'test_std': te.std(),
            'ks_stat': ks_stat,
            'p_value': p_val,
            'drift': '⚠️ YES' if p_val < alpha else '✅ NO'
        })
    
    df_res = pd.DataFrame(results).sort_values('ks_stat', ascending=False)
    
    drift_cols = df_res[df_res['p_value'] < alpha]['col'].tolist()
    print(f"\nDrift tespit edilen sütun: {len(drift_cols)}/{len(results)}")
    print(df_res.to_string(index=False))
    
    # Overlay histograms for drifted columns
    for col in drift_cols[:6]:
        fig, ax = plt.subplots(figsize=(8, 4))
        tr = train[col].dropna()
        te = test[col].dropna()
        ax.hist(tr, bins=50, alpha=0.5, label='Train', density=True, color='steelblue')
        ax.hist(te, bins=50, alpha=0.5, label='Test', density=True, color='red')
        ks = df_res[df_res['col']==col]['ks_stat'].values[0]
        ax.set_title(f'{col} — KS={ks:.3f} ⚠️ Drift', fontsize=12)
        ax.legend()
        plt.show()
    
    return drift_cols

drift_cols = train_test_drift(train, test)
```

## 4.6 Zaman Serisi EDA (Eğer Varsa)

```python
def time_series_eda(df, date_col, target, freq='M'):
    """Tarih sütunu olan yarışmalar için"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Zaman serisi
    monthly = df.groupby(df[date_col].dt.to_period(freq))[target].mean()
    monthly.index = monthly.index.astype(str)
    axes[0].plot(range(len(monthly)), monthly.values, marker='o', markersize=3)
    axes[0].set_xticks(range(0, len(monthly), max(1, len(monthly)//12)))
    axes[0].set_xticklabels(list(monthly.index)[::max(1, len(monthly)//12)], rotation=45)
    axes[0].set_title(f'{target} Zaman Serisi ({freq})', fontsize=13)
    
    # Mevsimsellik
    if df[date_col].dt.month.nunique() > 1:
        monthly_avg = df.groupby(df[date_col].dt.month)[target].mean()
        axes[1].bar(monthly_avg.index, monthly_avg.values, color='steelblue')
        axes[1].set_title('Aylık Ortalama (Mevsimsellik)', fontsize=13)
        axes[1].set_xlabel('Ay')
    
    # Haftanın günü
    dow_avg = df.groupby(df[date_col].dt.dayofweek)[target].mean()
    dow_names = ['Pzt', 'Sal', 'Çar', 'Per', 'Cum', 'Cmt', 'Paz']
    axes[2].bar(dow_names[:len(dow_avg)], dow_avg.values, color='steelblue')
    axes[2].set_title('Haftanın Günü Ortalaması', fontsize=13)
    
    plt.tight_layout()
    plt.show()
    
    # Train/test split görselleştir (sınırı işaretle)
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(range(len(df)), df[target].values, alpha=0.3, color='steelblue')
    ax.axvline(len(df)*0.8, color='red', linestyle='--', label='Train/Val Split')
    ax.set_title(f'{target} — Veri Sıralaması', fontsize=13)
    ax.legend()
    plt.show()
```

---

# 5. VERİ TEMİZLEME

## 5.1 Eksik Değer Stratejisi

```python
"""
EKSİK DEĞER KARAR ŞEMASI:
─────────────────────────────────────────────────────
Eksik oran > %70?
  → Sütunu drop et
  
Eksik oran %20-%70?
  → Yeni binary sütun ekle: col_was_missing = 1
  → Sonra impute et
  
Eksik oran < %20?
  → Sayısal → median impute (robust) ya da KNN
  → Kategorik → mode ya da "Missing" kategorisi
  
Pattern var mı (MCAR/MAR/MNAR)?
  → Eksiklik diğer sütunlarla korelasyonu var mı?
  → Eğer varsa → MICE (IterativeImputer) kullan
"""

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def handle_missing(df_train, df_test, target=None, threshold_drop=0.7):
    df_train = df_train.copy()
    df_test  = df_test.copy()
    
    dropped_cols = []
    missing_flag_cols = []
    
    for col in df_train.columns:
        if col == target:
            continue
        
        miss_rate = df_train[col].isna().mean()
        
        # Çok yüksek eksiklik → drop
        if miss_rate > threshold_drop:
            dropped_cols.append(col)
            continue
        
        # Orta eksiklik → flag ekle
        if miss_rate > 0.2:
            flag_col = f'{col}_was_missing'
            df_train[flag_col] = df_train[col].isna().astype(int)
            df_test[flag_col]  = df_test[col].isna().astype(int)
            missing_flag_cols.append(flag_col)
        
        # İmpute
        if df_train[col].dtype in ['object', 'category']:
            mode_val = df_train[col].mode()
            fill = mode_val[0] if len(mode_val) > 0 else 'Missing'
            df_train[col] = df_train[col].fillna(fill)
            df_test[col]  = df_test[col].fillna(fill)
        else:
            median_val = df_train[col].median()
            df_train[col] = df_train[col].fillna(median_val)
            df_test[col]  = df_test[col].fillna(median_val)
    
    df_train.drop(columns=dropped_cols, inplace=True)
    df_test.drop(columns=[c for c in dropped_cols if c in df_test.columns], inplace=True)
    
    print(f"Drop edilen: {dropped_cols}")
    print(f"Missing flag eklenen: {missing_flag_cols}")
    print(f"Train sonrası: {df_train.shape}  |  Test: {df_test.shape}")
    
    return df_train, df_test

# KNN Imputer (daha iyi ama yavaş)
def knn_impute(df_train, df_test, num_cols, k=5):
    imputer = KNNImputer(n_neighbors=k)
    df_train[num_cols] = imputer.fit_transform(df_train[num_cols])
    df_test[num_cols]  = imputer.transform(df_test[num_cols])
    return df_train, df_test

# MICE (en iyi ama en yavaş)
def mice_impute(df_train, df_test, num_cols):
    imputer = IterativeImputer(random_state=SEED, max_iter=10)
    df_train[num_cols] = imputer.fit_transform(df_train[num_cols])
    df_test[num_cols]  = imputer.transform(df_test[num_cols])
    return df_train, df_test
```

## 5.2 Outlier Tedavisi

```python
def handle_outliers(df, cols, method='iqr', threshold=1.5):
    """
    method: 'iqr', 'zscore', 'winsorize', 'clip'
    
    KARAR:
    - Tree modeller (LightGBM, XGBoost) → outlier önemsiz, işlem gerekmiyor
    - Linear modeller, NN → mutlaka tedavi et
    """
    df = df.copy()
    
    for col in cols:
        if col not in df.columns:
            continue
        
        v = df[col].dropna()
        
        if method == 'iqr':
            q1, q3 = v.quantile(0.25), v.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - threshold*iqr, q3 + threshold*iqr
            df[col] = df[col].clip(lo, hi)
        
        elif method == 'zscore':
            z_lo = v.mean() - threshold * v.std()
            z_hi = v.mean() + threshold * v.std()
            df[col] = df[col].clip(z_lo, z_hi)
        
        elif method == 'winsorize':
            from scipy.stats import mstats
            df[col] = pd.Series(
                mstats.winsorize(df[col].fillna(v.median()), limits=[0.01, 0.01]),
                index=df.index
            )
        
        elif method == 'quantile_clip':
            lo, hi = v.quantile(0.01), v.quantile(0.99)
            df[col] = df[col].clip(lo, hi)
    
    return df

# Outlier tespit
def detect_outliers(df, cols, method='iqr'):
    results = []
    for col in cols:
        v = df[col].dropna()
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        iqr = q3 - q1
        out = ((v < q1-1.5*iqr) | (v > q3+1.5*iqr)).sum()
        results.append({
            'col': col,
            'outlier_n': out,
            'outlier_%': f"{out/len(v)*100:.2f}%",
            'lo_fence': q1-1.5*iqr,
            'hi_fence': q3+1.5*iqr
        })
    return pd.DataFrame(results).sort_values('outlier_n', ascending=False)
```

## 5.3 Duplike Temizleme

```python
def remove_duplicates(df, subset=None, keep='first'):
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    after = len(df)
    print(f"Duplike kaldırıldı: {before-after:,} satır ({(before-after)/before*100:.2f}%)")
    return df
```

---

# 6. FEATURE ENGINEERING

> **Altın Kural:** Feature Engineering, model performansını en çok etkileyen adımdır. Özellikle tree-based modeller için iyi feature'lar HPO'dan çok daha etkilidir.

## 6.1 Sayısal Feature Engineering

```python
def numeric_fe(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # ── Log Transform (çarpık sütunlar için)
    for col in num_cols:
        if df[col].skew() > 1 and df[col].min() >= 0:
            df[f'{col}_log'] = np.log1p(df[col])
    
    # ── Polynomial features (küçük feature setlerinde)
    from sklearn.preprocessing import PolynomialFeatures
    # poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    # poly_features = poly.fit_transform(df[selected_cols])
    
    # ── Binning (sayısalı kategoriye çevir)
    for col in num_cols[:5]:  # ilk 5 sütun için örnek
        try:
            df[f'{col}_bin'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')
        except:
            pass
    
    # ── Rank transform (outlier'a robust)
    for col in num_cols:
        df[f'{col}_rank'] = df[col].rank(pct=True)
    
    # ── Rolling statistics (time series varsa)
    # df[f'{col}_rolling_mean'] = df[col].rolling(window=7, min_periods=1).mean()
    # df[f'{col}_rolling_std']  = df[col].rolling(window=7, min_periods=1).std()
    
    return df

# Manuel interaction features
def create_interactions(df, col_pairs):
    """
    col_pairs = [('col1', 'col2'), ('col3', 'col4')]
    """
    df = df.copy()
    for c1, c2 in col_pairs:
        if c1 in df.columns and c2 in df.columns:
            df[f'{c1}_x_{c2}']    = df[c1] * df[c2]
            df[f'{c1}_div_{c2}']  = df[c1] / (df[c2] + 1e-8)
            df[f'{c1}_plus_{c2}'] = df[c1] + df[c2]
            df[f'{c1}_diff_{c2}'] = df[c1] - df[c2]
    return df

# Z-score normalization (leak-free: fit on train only)
def normalize_features(df_train, df_test, cols, method='robust'):
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'quantile':
        scaler = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
    
    df_train[cols] = scaler.fit_transform(df_train[cols])
    df_test[cols]  = scaler.transform(df_test[cols])
    
    return df_train, df_test, scaler
```

## 6.2 Kategorik Feature Engineering

```python
def categorical_fe(df_train, df_test, target_col, cat_cols):
    df_train = df_train.copy()
    df_test  = df_test.copy()
    
    # ── Label Encoding (düşük kardinalite)
    for col in cat_cols:
        if df_train[col].nunique() <= 20:
            le = LabelEncoder()
            df_train[col] = le.fit_transform(df_train[col].astype(str))
            # test'te unseen label için güvenli transform
            le_classes = set(le.classes_)
            df_test[col] = df_test[col].astype(str).apply(
                lambda x: x if x in le_classes else le.classes_[0]
            )
            df_test[col] = le.transform(df_test[col])
    
    # ── Frequency Encoding
    for col in cat_cols:
        freq_map = df_train[col].value_counts(normalize=True)
        df_train[f'{col}_freq'] = df_train[col].map(freq_map).fillna(0)
        df_test[f'{col}_freq']  = df_test[col].map(freq_map).fillna(0)
    
    # ── Count Encoding
    for col in cat_cols:
        count_map = df_train[col].value_counts()
        df_train[f'{col}_count'] = df_train[col].map(count_map).fillna(0)
        df_test[f'{col}_count']  = df_test[col].map(count_map).fillna(0)
    
    return df_train, df_test

# ── Target Encoding (leak-free — CV ile)
def target_encode_cv(df_train, df_test, cat_cols, target_col, n_splits=5):
    """
    Leak-free target encoding — fold'lar arası cross-validate edilir.
    Mean-smooth ile rare kategori stabilizasyonu dahil.
    """
    from sklearn.model_selection import KFold
    
    df_train = df_train.copy()
    df_test  = df_test.copy()
    
    global_mean = df_train[target_col].mean()
    
    for col in cat_cols:
        encoded_train = np.zeros(len(df_train))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        
        for train_idx, val_idx in kf.split(df_train):
            tr = df_train.iloc[train_idx]
            means = tr.groupby(col)[target_col].mean()
            smoothed = means  # basit versiyon
            encoded_train[val_idx] = df_train[col].iloc[val_idx].map(smoothed).fillna(global_mean)
        
        df_train[f'{col}_te'] = encoded_train
        
        # Test için tüm train ile hesapla
        full_means = df_train.groupby(col)[target_col].mean()
        df_test[f'{col}_te'] = df_test[col].map(full_means).fillna(global_mean)
    
    return df_train, df_test

# ── category_encoders kullanımı (daha güçlü)
def advanced_target_encoding(df_train, df_test, cat_cols, target_col):
    """
    BinaryEncoder, TargetEncoder, WOEEncoder, CatBoostEncoder
    """
    import category_encoders as ce
    
    # CatBoost Encoding (leak-free, ordered)
    enc = ce.CatBoostEncoder(cols=cat_cols, random_state=SEED)
    df_train[cat_cols] = enc.fit_transform(df_train[cat_cols], df_train[target_col])
    df_test[cat_cols]  = enc.transform(df_test[cat_cols])
    
    return df_train, df_test, enc

# ── Rare Category Handling
def handle_rare_categories(df_train, df_test, cat_cols, min_freq=0.01):
    df_train = df_train.copy()
    df_test  = df_test.copy()
    
    for col in cat_cols:
        freq = df_train[col].value_counts(normalize=True)
        rare_vals = freq[freq < min_freq].index
        df_train[col] = df_train[col].replace(rare_vals, '__RARE__')
        df_test[col]  = df_test[col].replace(rare_vals, '__RARE__')
        # Test'te yeni kategorileri de rare yap
        train_cats = set(df_train[col].unique())
        df_test[col] = df_test[col].apply(lambda x: x if x in train_cats else '__RARE__')
    
    return df_train, df_test
```

## 6.3 Tarih/Zaman Feature Engineering

```python
def datetime_fe(df, date_cols):
    df = df.copy()
    
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Temel özellikler
        df[f'{col}_year']       = df[col].dt.year
        df[f'{col}_month']      = df[col].dt.month
        df[f'{col}_day']        = df[col].dt.day
        df[f'{col}_dayofweek']  = df[col].dt.dayofweek
        df[f'{col}_dayofyear']  = df[col].dt.dayofyear
        df[f'{col}_quarter']    = df[col].dt.quarter
        df[f'{col}_week']       = df[col].dt.isocalendar().week.astype(int)
        df[f'{col}_hour']       = df[col].dt.hour
        df[f'{col}_is_weekend'] = (df[col].dt.dayofweek >= 5).astype(int)
        df[f'{col}_is_month_start'] = df[col].dt.is_month_start.astype(int)
        df[f'{col}_is_month_end']   = df[col].dt.is_month_end.astype(int)
        
        # Döngüsel kodlama (ML için kritik — Ocak ile Aralık arasındaki uzaklık)
        df[f'{col}_month_sin'] = np.sin(2 * np.pi * df[col].dt.month / 12)
        df[f'{col}_month_cos'] = np.cos(2 * np.pi * df[col].dt.month / 12)
        df[f'{col}_dow_sin']   = np.sin(2 * np.pi * df[col].dt.dayofweek / 7)
        df[f'{col}_dow_cos']   = np.cos(2 * np.pi * df[col].dt.dayofweek / 7)
        df[f'{col}_hour_sin']  = np.sin(2 * np.pi * df[col].dt.hour / 24)
        df[f'{col}_hour_cos']  = np.cos(2 * np.pi * df[col].dt.hour / 24)
        
        # Referans noktasından gün sayısı
        ref = df[col].min()
        df[f'{col}_days_since'] = (df[col] - ref).dt.days
        df[f'{col}_days_to_end'] = (df[col].max() - df[col]).dt.days
    
    return df
```

## 6.4 Aggregation Features (Group-by)

```python
def aggregation_fe(df_train, df_test, group_cols, agg_cols, target=None):
    """
    group_cols: gruplama sütunları  ['customer_id', 'product_id']
    agg_cols:   agregasyon yapılacak sütunlar
    
    Bu tip feature'lar genellikle en güçlü feature'lar olur.
    """
    all_data = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    
    agg_dfs = []
    
    for group_col in group_cols:
        for agg_col in agg_cols:
            grp = all_data.groupby(group_col)[agg_col].agg(
                ['mean', 'std', 'min', 'max', 'median',
                 lambda x: x.quantile(0.25),
                 lambda x: x.quantile(0.75)]
            )
            grp.columns = [
                f'{group_col}_{agg_col}_mean',
                f'{group_col}_{agg_col}_std',
                f'{group_col}_{agg_col}_min',
                f'{group_col}_{agg_col}_max',
                f'{group_col}_{agg_col}_median',
                f'{group_col}_{agg_col}_q25',
                f'{group_col}_{agg_col}_q75',
            ]
            grp[f'{group_col}_{agg_col}_range'] = (
                grp[f'{group_col}_{agg_col}_max'] - grp[f'{group_col}_{agg_col}_min']
            )
            agg_dfs.append(grp)
    
    for agg_df in agg_dfs:
        df_train = df_train.merge(agg_df, on=group_col, how='left')
        df_test  = df_test.merge(agg_df, on=group_col, how='left')
    
    return df_train, df_test
```

## 6.5 Feature Selection

```python
def feature_selection(X_train, y_train, X_test, method='importance', n_features=50):
    """
    method: 'importance', 'mutual_info', 'correlation', 'rfe', 'shap'
    """
    
    if method == 'importance':
        # LightGBM feature importance
        model = lgb.LGBMClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
        model.fit(X_train, y_train)
        
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        selected = importance.head(n_features)['feature'].tolist()
        
        # Plot
        plt.figure(figsize=(10, max(6, n_features//3)))
        sns.barplot(data=importance.head(30), y='feature', x='importance', color='steelblue')
        plt.title('Feature Importance (Top 30)')
        plt.tight_layout()
        plt.show()
    
    elif method == 'mutual_info':
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(X_train.fillna(0), y_train, random_state=SEED)
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': mi
        }).sort_values('importance', ascending=False)
        selected = importance.head(n_features)['feature'].tolist()
    
    elif method == 'correlation':
        # Hedef ile düşük korelasyon + birbiriyle yüksek korelasyon olanları kaldır
        corr = X_train.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
        selected = [c for c in X_train.columns if c not in to_drop]
    
    elif method == 'variance':
        sel = VarianceThreshold(threshold=0.01)
        sel.fit(X_train.fillna(0))
        selected = X_train.columns[sel.get_support()].tolist()
    
    elif method == 'shap':
        # SHAP-based selection (en güvenilir)
        model = lgb.LGBMClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
        model.fit(X_train, y_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train[:5000])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_importance = np.abs(shap_values).mean(axis=0)
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': shap_importance
        }).sort_values('importance', ascending=False)
        selected = importance.head(n_features)['feature'].tolist()
    
    print(f"Seçilen feature sayısı: {len(selected)} / {len(X_train.columns)}")
    return selected, importance
```

---

# 7. MODEL SEÇİMİ & BASELINE

## 7.1 Problem Tipine Göre Model Matrisi

```python
"""
PROBLEM TİPİ         MODEL HİYERARŞİSİ (iyi→daha iyi)
─────────────────────────────────────────────────────────────────
Binary Classif.      LR → RF → XGB → LightGBM → CatBoost → Ensemble
Multi-class          RF → XGB → LightGBM → CatBoost → NN
Regression           Ridge → RF → XGB → LightGBM → CatBoost → NN
Time Series          ARIMA → Prophet → LightGBM (lag features) → LSTM
Imbalanced           LightGBM(scale_pos_weight) → BalancedRF → SMOTE+LGB
High-dim Text        TF-IDF+LR → TF-IDF+LGBM → BERT → DeBERTa
Tabular+Text         LGB(text features) → CatBoost → Blend(LGBM+BERT)
Images               EfficientNet → ConvNeXt → ViT → Ensemble

GOLDEN RULE:
  Kaggle tablo veri → LightGBM genellikle kazanır
  NLP → BERT/DeBERTa
  CV → EfficientNet/ViT
  Her zaman ensemble
"""
```

## 7.2 LightGBM — Ana Model

```python
import lightgbm as lgb

# ─── CLASSIFICATION ──────────────────────────────────────────────
lgb_clf_params = {
    # Temel
    'objective':        'binary',        # 'multiclass' için num_class ekle
    'metric':           'auc',           # yarışma metriğine göre değiştir
    'boosting_type':    'gbdt',          # 'dart', 'goss', 'rf'
    'num_leaves':       127,             # 2^max_depth - 1, overfit dikkat
    'max_depth':        -1,              # -1 = sınırsız
    'learning_rate':    0.05,            # HPO'da düşür: 0.01-0.1
    'n_estimators':     2000,
    
    # Sampling
    'feature_fraction': 0.7,            # colsample_bytree
    'bagging_fraction': 0.8,            # subsample
    'bagging_freq':     5,
    
    # Regularization
    'min_child_samples':20,
    'lambda_l1':        0.1,
    'lambda_l2':        0.1,
    'min_gain_to_split':0.01,
    
    # Class imbalance
    'scale_pos_weight': 1,  # pos_count/neg_count
    'is_unbalance':     False,
    
    # Performance
    'n_jobs':           -1,
    'device':           'cpu',          # 'gpu' varsa
    'random_state':     SEED,
    'verbose':          -1,
}

# ─── REGRESSION ──────────────────────────────────────────────────
lgb_reg_params = {
    'objective':        'regression',    # 'regression_l1' (MAE), 'huber', 'tweedie'
    'metric':           'rmse',
    'num_leaves':       127,
    'learning_rate':    0.05,
    'n_estimators':     2000,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq':     5,
    'lambda_l1':        0.1,
    'lambda_l2':        0.1,
    'n_jobs':           -1,
    'random_state':     SEED,
    'verbose':          -1,
}

# ─── MULTICLASS ──────────────────────────────────────────────────
lgb_multi_params = {
    'objective':        'multiclass',
    'num_class':        5,              # ← sınıf sayısı
    'metric':           'multi_logloss',
    'num_leaves':       127,
    'learning_rate':    0.05,
    'n_estimators':     2000,
    'n_jobs':           -1,
    'random_state':     SEED,
    'verbose':          -1,
}
```

## 7.3 XGBoost

```python
import xgboost as xgb

xgb_params = {
    'objective':        'binary:logistic',  # 'reg:squarederror', 'multi:softprob'
    'eval_metric':      'auc',
    'n_estimators':     2000,
    'max_depth':        6,
    'learning_rate':    0.05,
    'subsample':        0.8,
    'colsample_bytree': 0.7,
    'gamma':            0.1,
    'reg_alpha':        0.1,
    'reg_lambda':       1.0,
    'min_child_weight': 5,
    'scale_pos_weight': 1,
    'use_label_encoder': False,
    'n_jobs':           -1,
    'random_state':     SEED,
    'tree_method':      'hist',          # 'gpu_hist' for GPU
}
```

## 7.4 CatBoost

```python
from catboost import CatBoostClassifier, CatBoostRegressor

cb_params = {
    'iterations':       2000,
    'learning_rate':    0.05,
    'depth':            6,
    'l2_leaf_reg':      3,
    'border_count':     128,
    'eval_metric':      'AUC',
    'od_type':          'Iter',
    'od_wait':          50,
    'random_seed':      SEED,
    'verbose':          False,
    # CatBoost native kategorik işleme (çok güçlü)
    # cat_features: kategorik sütun isimleri veya indeksleri
}
```

## 7.5 Cross-Validation ile Eğitim Döngüsü

```python
def train_cv(X, y, X_test, params, n_splits=5, problem_type='classification',
             cat_features=None, early_stopping_rounds=100, verbose_eval=200):
    """
    Tam CV eğitim döngüsü:
    - Her fold'da model eğit
    - OOF (Out-of-Fold) prediction oluştur (stacking için)
    - Test prediction'ları ortalama
    """
    
    if problem_type == 'classification':
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        split_iter = kf.split(X, y)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        split_iter = kf.split(X)
    
    # Storage
    if problem_type == 'classification':
        n_classes = len(np.unique(y))
        oof_preds = np.zeros((len(X), n_classes)) if n_classes > 2 else np.zeros(len(X))
        test_preds = np.zeros((len(X_test), n_classes)) if n_classes > 2 else np.zeros(len(X_test))
    else:
        oof_preds  = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))
    
    feature_importance = pd.DataFrame()
    cv_scores = []
    trained_models = []
    
    print(f"\n{'='*60}")
    print(f"Training: {n_splits}-Fold CV  |  {len(X):,} samples  |  {X.shape[1]} features")
    print(f"{'='*60}")
    
    for fold, (train_idx, val_idx) in enumerate(split_iter, 1):
        print(f"\n── Fold {fold}/{n_splits} ──")
        t0 = time.time()
        
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # ─── LightGBM ───────────────────────────────────────────
        model = lgb.LGBMClassifier(**params) if problem_type == 'classification' \
                else lgb.LGBMRegressor(**params)
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(verbose_eval)
            ],
            categorical_feature=cat_features or 'auto'
        )
        
        # Predictions
        if problem_type == 'classification':
            val_pred  = model.predict_proba(X_val)
            test_fold = model.predict_proba(X_test)
            
            if val_pred.shape[1] == 2:
                val_pred  = val_pred[:, 1]
                test_fold = test_fold[:, 1]
            
            oof_preds[val_idx]  = val_pred
            test_preds         += test_fold / n_splits
            score = roc_auc_score(y_val, val_pred if val_pred.ndim == 1 else val_pred,
                                  multi_class='ovr' if val_pred.ndim > 1 else 'raise')
        else:
            val_pred  = model.predict(X_val)
            test_fold = model.predict(X_test)
            oof_preds[val_idx] = val_pred
            test_preds += test_fold / n_splits
            score = np.sqrt(mean_squared_error(y_val, val_pred))
        
        cv_scores.append(score)
        trained_models.append(model)
        
        # Feature importance
        fi = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_,
            'fold': fold
        })
        feature_importance = pd.concat([feature_importance, fi])
        
        elapsed = time.time() - t0
        print(f"  Score: {score:.5f}  |  Best iter: {model.best_iteration_}  |  {elapsed:.1f}s")
    
    # CV Summary
    print(f"\n{'='*60}")
    print(f"CV Scores: {[f'{s:.5f}' for s in cv_scores]}")
    print(f"Mean: {np.mean(cv_scores):.5f} ± {np.std(cv_scores):.5f}")
    print(f"{'='*60}")
    
    # OOF Score
    if problem_type == 'classification':
        oof_score = roc_auc_score(y, oof_preds if oof_preds.ndim == 1 else oof_preds)
        print(f"OOF Score: {oof_score:.5f}")
    else:
        oof_score = np.sqrt(mean_squared_error(y, oof_preds))
        print(f"OOF RMSE: {oof_score:.5f}")
    
    # Feature importance plot
    fi_mean = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    fi_mean.head(30).plot(kind='barh', color='steelblue')
    plt.gca().invert_yaxis()
    plt.title('Feature Importance (Top 30 — Mean across folds)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'feature_importance.png', dpi=120)
    plt.show()
    
    return {
        'oof_preds':    oof_preds,
        'test_preds':   test_preds,
        'cv_scores':    cv_scores,
        'oof_score':    oof_score,
        'importance':   fi_mean,
        'models':       trained_models
    }

# Kullanım
# results = train_cv(X, y, X_test, lgb_clf_params, n_splits=5, problem_type='classification')
```

---

# 8. CROSS-VALIDATION STRATEJİSİ

```python
"""
CV STRATEJİSİ KARAR TABLOSU:
──────────────────────────────────────────────────────────────────────
Problem                              CV Yöntemi
──────────────────────────────────────────────────────────────────────
Standard classification              StratifiedKFold(n=5)
Standard regression                  KFold(n=5)
Imbalanced classification            StratifiedKFold(n=5)  (zaten stratified)
Time series (no shuffle!)            TimeSeriesSplit veya custom date split
Group-based (aynı kullanıcı)         GroupKFold / StratifiedGroupKFold
Multi-label                          IterativeStratification (iterstrat lib)
Az veri (<1000 sample)               StratifiedKFold(n=10) veya LOOCV
Çok veri (>1M rows)                  Hold-out (80/20) ya da 3-fold
──────────────────────────────────────────────────────────────────────
"""

# ─── TIME SERIES CV ──────────────────────────────────────────────
from sklearn.model_selection import TimeSeriesSplit

def time_series_cv(X, y, X_test, params, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_preds  = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    cv_scores  = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_test) / n_splits
        
        score = np.sqrt(mean_squared_error(y_val, val_pred))
        cv_scores.append(score)
        print(f"Fold {fold}: RMSE={score:.5f}")
    
    print(f"\nCV Mean: {np.mean(cv_scores):.5f} ± {np.std(cv_scores):.5f}")
    return oof_preds, test_preds, cv_scores

# ─── GROUP K-FOLD ────────────────────────────────────────────────
def group_cv(X, y, X_test, groups, params, n_splits=5):
    """
    groups: her sample'ın ait olduğu grup (örn: customer_id)
    Aynı müşteri hem train hem validation'da olmamalı.
    """
    gkf = StratifiedGroupKFold(n_splits=n_splits)
    oof_preds  = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    cv_scores  = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        
        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred
        test_preds += model.predict_proba(X_test)[:, 1] / n_splits
        
        score = roc_auc_score(y_val, val_pred)
        cv_scores.append(score)
    
    print(f"CV: {np.mean(cv_scores):.5f} ± {np.std(cv_scores):.5f}")
    return oof_preds, test_preds, cv_scores
```

## 8.1 CV-LB Correlation Takibi

```python
"""
EN ÖNEMLİ DERS: CV ve LB arasındaki korelasyonu takip et.

CV yükseliyor ama LB düşüyor  → Overfitting / Leak
CV düşüyor ama LB yükseliyor  → CV stratejin hatalı
CV ve LB aynı yönde hareket   → ✅ Güvenilir CV

Bunu bir tabloda tut:
"""

cv_lb_tracker = pd.DataFrame(columns=['experiment', 'cv_score', 'lb_score', 'notes'])

def log_experiment(name, cv_score, lb_score=None, notes=''):
    global cv_lb_tracker
    new_row = pd.DataFrame({
        'experiment': [name],
        'cv_score':   [cv_score],
        'lb_score':   [lb_score],
        'notes':      [notes]
    })
    cv_lb_tracker = pd.concat([cv_lb_tracker, new_row], ignore_index=True)
    print(cv_lb_tracker.to_string(index=False))

# Kullanım:
# log_experiment('LightGBM baseline', 0.8523, 0.8491, 'default params')
# log_experiment('LGB + FE v1', 0.8601, 0.8574, 'added interaction features')
```

---

# 9. HYPERPARAMETER OPTIMIZATION

## 9.1 Optuna ile HPO

```python
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

def optimize_lgbm(X, y, n_trials=100, problem_type='classification', n_splits=5):
    """
    Optuna ile LightGBM HPO.
    n_trials=100: ~30-60 dakika
    n_trials=300: ~2-3 saat (competition'da değer)
    """
    
    def objective(trial):
        params = {
            'objective':        'binary' if problem_type == 'classification' else 'regression',
            'metric':           'auc' if problem_type == 'classification' else 'rmse',
            'n_estimators':     2000,
            'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'num_leaves':       trial.suggest_int('num_leaves', 20, 500),
            'max_depth':        trial.suggest_int('max_depth', 3, 12),
            'min_child_samples':trial.suggest_int('min_child_samples', 5, 200),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq':     trial.suggest_int('bagging_freq', 1, 10),
            'lambda_l1':        trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2':        trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'min_gain_to_split':trial.suggest_float('min_gain_to_split', 0, 15),
            'n_jobs':           -1,
            'random_state':     SEED,
            'verbose':          -1,
        }
        
        if problem_type == 'classification':
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        
        scores = []
        for train_idx, val_idx in kf.split(X, y if problem_type == 'classification' else np.zeros(len(X))):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = lgb.LGBMClassifier(**params) if problem_type == 'classification' \
                    else lgb.LGBMRegressor(**params)
            
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
            
            if problem_type == 'classification':
                pred = model.predict_proba(X_val)[:, 1]
                scores.append(roc_auc_score(y_val, pred))
            else:
                pred = model.predict(X_val)
                scores.append(-np.sqrt(mean_squared_error(y_val, pred)))
        
        return np.mean(scores)
    
    sampler = TPESampler(seed=SEED, n_startup_trials=20)
    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=30)
    
    direction = 'maximize'
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    
    study.optimize(objective, n_trials=n_trials,
                   timeout=3600,  # max 1 saat
                   show_progress_bar=True)
    
    print(f"\nEn iyi trial: {study.best_trial.number}")
    print(f"En iyi score: {study.best_value:.5f}")
    print(f"En iyi params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    
    # Visualization
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.show()
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.show()
    except:
        pass
    
    return study.best_params

# Kullanım
# best_params = optimize_lgbm(X_train, y_train, n_trials=100)
# best_params['n_estimators'] = 2000  # early stopping ile belirlenir
```

## 9.2 XGBoost HPO

```python
def optimize_xgb(X, y, n_trials=100, problem_type='classification'):
    def objective(trial):
        params = {
            'objective':         'binary:logistic' if problem_type == 'classification' else 'reg:squarederror',
            'n_estimators':      2000,
            'max_depth':         trial.suggest_int('max_depth', 3, 12),
            'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'subsample':         trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'gamma':             trial.suggest_float('gamma', 0, 15),
            'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight':  trial.suggest_int('min_child_weight', 1, 20),
            'n_jobs':            -1,
            'random_state':      SEED,
            'tree_method':       'hist',
        }
        
        scores = []
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        for train_idx, val_idx in kf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params) if problem_type == 'classification' \
                    else xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      early_stopping_rounds=100,
                      verbose=False)
            
            if problem_type == 'classification':
                pred = model.predict_proba(X_val)[:, 1]
                scores.append(roc_auc_score(y_val, pred))
            else:
                pred = model.predict(X_val)
                scores.append(-np.sqrt(mean_squared_error(y_val, pred)))
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"Best: {study.best_value:.5f}")
    return study.best_params
```

---

# 10. ENSEMBLE & STACKING

## 10.1 Ensemble Stratejileri

```python
"""
ENSEMBLE HİYERARŞİSİ (basit → karmaşık):
─────────────────────────────────────────────────────
1. Single model                 → baseline
2. Model averaging              → çok kolay, genellikle +0.003-0.005
3. Weighted averaging           → biraz optimizasyon gerekir
4. Rank averaging               → ranking metriklerinde çok iyi
5. Stacking / Blending          → en güçlü, overfitting riski var
6. Multi-level stacking         → grandmaster seviye
"""

# ─── Simple Average ──────────────────────────────────────────────
def simple_average(pred_list):
    return np.mean(pred_list, axis=0)

# ─── Weighted Average ────────────────────────────────────────────
def weighted_average(pred_list, weights):
    """weights: [0.4, 0.3, 0.3] gibi, toplamı 1 olmalı"""
    weights = np.array(weights) / np.sum(weights)
    return sum(w * p for w, p in zip(weights, pred_list))

# ─── Rank Average ────────────────────────────────────────────────
def rank_average(pred_list):
    """
    Kalibrasyon farklı modellerde daha iyi çalışır.
    AUC gibi ranking metriklerinde güçlü.
    """
    ranked = [pd.Series(p).rank(pct=True).values for p in pred_list]
    return np.mean(ranked, axis=0)

# ─── Optimal Weight Optimization ─────────────────────────────────
from scipy.optimize import minimize

def optimize_weights(oof_preds_list, y_true, metric_fn, n_trials=100):
    """
    OOF prediction'larına göre optimal ağırlık bul.
    """
    n_models = len(oof_preds_list)
    
    def neg_metric(weights):
        weights = np.array(weights) / np.sum(weights)
        blended = sum(w * p for w, p in zip(weights, oof_preds_list))
        return -metric_fn(y_true, blended)
    
    best_score = -np.inf
    best_weights = np.ones(n_models) / n_models
    
    for _ in range(n_trials):
        init_weights = np.random.dirichlet(np.ones(n_models))
        bounds = [(0, 1)] * n_models
        constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 1}
        
        result = minimize(neg_metric, init_weights,
                         method='SLSQP', bounds=bounds,
                         constraints=constraints)
        
        if -result.fun > best_score:
            best_score = -result.fun
            best_weights = result.x / result.x.sum()
    
    print(f"Optimal weights: {best_weights.round(3)}")
    print(f"Blended OOF score: {best_score:.5f}")
    
    return best_weights

# Kullanım:
# weights = optimize_weights([oof_lgb, oof_xgb, oof_cb], y_train, roc_auc_score)
# test_blend = weighted_average([test_lgb, test_xgb, test_cb], weights)
```

## 10.2 Stacking

```python
def stacking_ensemble(X_train, y_train, X_test, base_models, meta_model,
                      n_splits=5, problem_type='classification'):
    """
    2-level stacking:
    Level 1: Base models → OOF predictions
    Level 2: Meta model trained on OOF predictions
    """
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED) \
         if problem_type == 'classification' \
         else KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    # Level 1: OOF predictions
    train_meta = np.zeros((len(X_train), len(base_models)))
    test_meta  = np.zeros((len(X_test),  len(base_models)))
    
    for model_idx, (model_name, model) in enumerate(base_models.items()):
        print(f"\nLevel 1 — {model_name}")
        oof  = np.zeros(len(X_train))
        test_folds = np.zeros((len(X_test), n_splits))
        
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            X_tr = X_train.iloc[tr_idx]; X_val = X_train.iloc[val_idx]
            y_tr = y_train.iloc[tr_idx]; y_val = y_train.iloc[val_idx]
            
            m = deepcopy(model)
            m.fit(X_tr, y_tr)
            
            if problem_type == 'classification':
                oof[val_idx]     = m.predict_proba(X_val)[:, 1]
                test_folds[:, fold] = m.predict_proba(X_test)[:, 1]
            else:
                oof[val_idx]     = m.predict(X_val)
                test_folds[:, fold] = m.predict(X_test)
        
        train_meta[:, model_idx] = oof
        test_meta[:, model_idx]  = test_folds.mean(axis=1)
        
        if problem_type == 'classification':
            score = roc_auc_score(y_train, oof)
        else:
            score = np.sqrt(mean_squared_error(y_train, oof))
        print(f"  OOF Score: {score:.5f}")
    
    # Level 2: Meta model
    print("\nLevel 2 — Meta Model")
    meta_model.fit(train_meta, y_train)
    
    if problem_type == 'classification':
        final_pred = meta_model.predict_proba(test_meta)[:, 1]
        final_oof  = meta_model.predict_proba(train_meta)[:, 1]
        print(f"Meta OOF Score: {roc_auc_score(y_train, final_oof):.5f}")
    else:
        final_pred = meta_model.predict(test_meta)
        final_oof  = meta_model.predict(train_meta)
        print(f"Meta OOF RMSE: {np.sqrt(mean_squared_error(y_train, final_oof)):.5f}")
    
    return final_pred, final_oof, train_meta, test_meta

# Kullanım:
from sklearn.linear_model import LogisticRegression, Ridge

base_models = {
    'lgb':  lgb.LGBMClassifier(**lgb_clf_params),
    'xgb':  xgb.XGBClassifier(**xgb_params),
    'cb':   cb.CatBoostClassifier(**cb_params),
    'rf':   RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
}
meta_model = LogisticRegression(C=0.1)

# final_pred, final_oof, train_meta, test_meta = stacking_ensemble(
#     X_train, y_train, X_test,
#     base_models, meta_model
# )
```

---

# 11. POST-PROCESSING & THRESHOLD TUNING

## 11.1 Threshold Tuning (Classification)

```python
def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """
    F1, MCC, precision-recall dengesi için threshold optimize et.
    Sadece CV'de yapılmalı — test'e doğrudan uygulanmaz.
    """
    thresholds = np.arange(0.01, 0.99, 0.01)
    scores = []
    
    for thr in thresholds:
        y_pred = (y_pred_proba >= thr).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, average='binary')
        elif metric == 'mcc':
            from sklearn.metrics import matthews_corrcoef
            score = matthews_corrcoef(y_true, y_pred)
        elif metric == 'balanced_accuracy':
            from sklearn.metrics import balanced_accuracy_score
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            score = f1_score(y_true, y_pred)
        
        scores.append(score)
    
    best_idx = np.argmax(scores)
    best_thr = thresholds[best_idx]
    best_score = scores[best_idx]
    
    plt.figure(figsize=(10, 4))
    plt.plot(thresholds, scores, color='steelblue')
    plt.axvline(best_thr, color='red', linestyle='--', label=f'Best={best_thr:.2f}')
    plt.title(f'Threshold vs {metric}  (best={best_score:.4f} @ {best_thr:.2f})')
    plt.xlabel('Threshold')
    plt.ylabel(metric)
    plt.legend()
    plt.show()
    
    print(f"Optimal threshold: {best_thr:.2f}  |  {metric}: {best_score:.4f}")
    
    return best_thr

# Precision-Recall curve
def plot_pr_curve(y_true, y_pred_proba):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='steelblue', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve  (AP={ap:.4f})')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # F1 scores per threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    print(f"Best F1 threshold: {thresholds[best_idx]:.3f}  |  F1={f1_scores[best_idx]:.4f}")
    
    return thresholds[best_idx]
```

## 11.2 Prediction Clipping & Transformation

```python
# ─── Regression için ─────────────────────────────────────────────
def postprocess_regression(preds, y_train, clip=True):
    """
    Log transform kullandıysan geri çevir.
    Sınır dışı değerleri clip et.
    """
    # Log transform geri çevir
    if LOG_TARGET:
        preds = np.expm1(preds)
    
    # Clip
    if clip:
        lo = y_train.min()
        hi = y_train.max()
        preds = np.clip(preds, lo, hi)
    
    # Negatif değerleri sıfırla (pozitif target için)
    preds = np.maximum(preds, 0)
    
    return preds

# ─── Calibration (probability calibration) ───────────────────────
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

def calibrate_predictions(y_true, y_pred_proba, method='isotonic'):
    """
    Logloss metriği için kalibrasyon kritik.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression as LR
    
    if method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(y_pred_proba, y_true)
        calibrated = calibrator.predict(y_pred_proba)
    
    elif method == 'platt':  # Platt scaling
        calibrator = LR()
        calibrator.fit(y_pred_proba.reshape(-1, 1), y_true)
        calibrated = calibrator.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]
    
    # Plot calibration
    fig, ax = plt.subplots(figsize=(8, 6))
    
    frac_pos, mean_pred = calibration_curve(y_true, y_pred_proba, n_bins=20)
    ax.plot(mean_pred, frac_pos, 'b-', label='Before calibration')
    
    frac_pos2, mean_pred2 = calibration_curve(y_true, calibrated, n_bins=20)
    ax.plot(mean_pred2, frac_pos2, 'r-', label='After calibration')
    
    ax.plot([0,1], [0,1], 'k--', label='Perfect')
    ax.legend()
    ax.set_title('Calibration Curve')
    plt.show()
    
    print(f"Before logloss: {log_loss(y_true, y_pred_proba):.5f}")
    print(f"After logloss:  {log_loss(y_true, calibrated):.5f}")
    
    return calibrated, calibrator
```

---

# 12. SUBMISSION & LEADERBOARD STRATEJİSİ

## 12.1 Submission Oluşturma

```python
def create_submission(test_ids, predictions, target_col, filename=None, problem_type='classification'):
    sub = pd.DataFrame({
        'id': test_ids,
        target_col: predictions
    })
    
    # Sağlık kontrolü
    assert sub[target_col].isna().sum() == 0, "⚠️ NaN var!"
    assert np.isinf(sub[target_col]).sum() == 0, "⚠️ Inf var!"
    
    if problem_type == 'classification':
        assert sub[target_col].between(0, 1).all(), "⚠️ 0-1 dışında değer var!"
    
    print(f"Submission shape: {sub.shape}")
    print(sub[target_col].describe())
    
    if filename is None:
        filename = f"submission_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    filepath = OUT_DIR / filename
    sub.to_csv(filepath, index=False)
    print(f"✅ Saved: {filepath}")
    
    return sub

# Kullanım:
# sub = create_submission(test['id'], results['test_preds'], 'Churn',
#                         'submission_lgb_v1.csv')
```

## 12.2 Submission Stratejisi

```python
"""
SUBMISSION STRATEJİSİ:
──────────────────────────────────────────────────────────────────
Toplam submission hakkı genellikle günde 2-5.

KURAL 1: İlk gün 2 submission kullan
  - Submission 1: Baseline (mevcut durumu anla)
  - Submission 2: İlk iyileştirme (leak var mı kontrol et)

KURAL 2: CV-LB korelasyonunu kur
  - Her submission'da hem CV hem LB'yi kaydet
  - Korelasyon var mı? Yoksa CV'ne güvenme

KURAL 3: Son 2 submission'ı sakla
  - Yarışma biterken farklı 2 tahmin arasında seç
  - Public LB ile private LB arasında fark olabilir

KURAL 4: Son 7 günde selection stratejisi
  Option A: En yüksek public LB — riski yüksek (public LB leak)
  Option B: En yüksek CV — daha güvenilir
  Option C: Birini CV'e, birini public LB'ye göre seç — denge

KURAL 5: Shake-up durumları
  - Public LB < %30 veri → büyük shake-up olabilir
  - CV güvenilirse CV'e güven, public'e değil
"""

# CV - LB tracker
print(cv_lb_tracker.sort_values('cv_score', ascending=False).to_string(index=False))
```

## 12.3 Sızıntı (Data Leak) Tespiti

```python
"""
DATA LEAK BELİRTİLERİ:
1. CV çok yüksek (0.99+) ama makul görünüyor  → şüpheli
2. Public LB CV'den çok yüksek  → feature'larda sızıntı var
3. Belirli bir feature çok dominant importance  → sızıntı olabilir

LEAK TÜRÜ         ÇÖZÜM
─────────────────────────────────────────────────────
Target leak       Feature'ları tek tek çıkar, kontrol et
Time leak         train'de gelecek veri var → date sorunu
ID leak           ID ile target ilişkili → ID'yi drop et
"""

def check_for_leaks(X_train, y_train):
    """Her feature'ın tek başına AUC'unu ölç"""
    results = []
    
    for col in X_train.columns:
        try:
            vals = X_train[col].fillna(-999)
            if vals.nunique() < 2:
                continue
            
            auc = roc_auc_score(y_train, vals)
            auc = max(auc, 1 - auc)  # directional
            
            results.append({'feature': col, 'auc': auc})
        except:
            pass
    
    df_results = pd.DataFrame(results).sort_values('auc', ascending=False)
    
    suspicious = df_results[df_results['auc'] > 0.9]
    if len(suspicious) > 0:
        print(f"\n⚠️  Yüksek solo AUC (>0.90) — potansiyel leak:")
        print(suspicious.to_string(index=False))
    
    return df_results
```

---

# 13. SIKININCA NE YAPMALIYIM

## 13.1 Platoya Ulaştığında

```python
"""
PLATO AŞMA STRATEJİLERİ:
──────────────────────────────────────────────────────────────────

A) FEATURE ENGINEERING
   ✓ Aggregation features ekle (group-by)
   ✓ Interaction features (col1 * col2, col1 / col2)
   ✓ Lag features (time series)
   ✓ Target encoding farklı window ile
   ✓ PCA / UMAP embeddings
   ✓ Cluster features (KMeans, GMM)

B) MODEL
   ✓ Başka model dene (XGBoost varsa CatBoost dene)
   ✓ Neural network (TabNet, NODE, MLP)
   ✓ NN + GBDT blend
   ✓ Daha uzun early stopping
   ✓ n_estimators artır, learning_rate düşür

C) DATA
   ✓ Train + Test birleştir (pseudo-labeling)
   ✓ Farklı CV stratejisi (fold sayısı değiştir)
   ✓ Noise injection (data augmentation)
   ✓ Daha agresif feature selection

D) ENSEMBLE
   ✓ Daha fazla model ekle
   ✓ Out-of-fold stacking
   ✓ Rank averaging dene

E) METALEARNING
   ✓ Discussion bölümüne bak — ne keşfedilmiş?
   ✓ Public notebooks'ları indir ve analiz et
   ✓ Winning solutions (önceki yarışmalar) oku
"""

# Pseudo-labeling
def pseudo_labeling(X_train, y_train, X_test, model, threshold=0.95,
                    problem_type='classification'):
    """
    Yüksek güvenli test tahminlerini train'e ekle.
    Final submission'da dikkatli kullan — CV sızıntısı riski var.
    """
    model.fit(X_train, y_train)
    
    if problem_type == 'classification':
        test_preds = model.predict_proba(X_test)[:, 1]
        
        high_conf = (test_preds >= threshold) | (test_preds <= 1-threshold)
        pseudo_X = X_test[high_conf]
        pseudo_y = (test_preds[high_conf] >= 0.5).astype(int)
    else:
        test_preds = model.predict(X_test)
        # regression için: tahminlerin ortalamasına uzaklığa göre filtrele
        mu, sigma = test_preds.mean(), test_preds.std()
        high_conf = np.abs(test_preds - mu) > 1.5 * sigma
        pseudo_X = X_test[~high_conf]  # conservative: ortaya yakın olanları al
        pseudo_y = pd.Series(test_preds[~high_conf])
    
    print(f"Pseudo-labeled: {len(pseudo_X):,} / {len(X_test):,} samples")
    
    X_augmented = pd.concat([X_train, pseudo_X]).reset_index(drop=True)
    y_augmented = pd.concat([y_train, pseudo_y]).reset_index(drop=True)
    
    return X_augmented, y_augmented
```

## 13.2 SHAP ile Model Yorumlama

```python
def shap_analysis(model, X_train, X_test=None, max_display=30):
    """Modelin ne öğrendiğini anla — feature engineering ipuçları çıkar"""
    
    explainer = shap.TreeExplainer(model)
    
    # Train'in küçük bir sample'ı
    X_sample = X_train.sample(min(5000, len(X_train)), random_state=SEED)
    shap_values = explainer.shap_values(X_sample)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # binary için pozitif class
    
    # Summary plot
    plt.figure(figsize=(10, max(6, max_display//2)))
    shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
    plt.title('SHAP Summary Plot')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'shap_summary.png', dpi=120, bbox_inches='tight')
    plt.show()
    
    # Bar plot
    shap.summary_plot(shap_values, X_sample, plot_type='bar',
                      max_display=max_display, show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.show()
    
    # Dependence plots (top 5 feature)
    top_features = np.argsort(np.abs(shap_values).mean(0))[-5:]
    for feat_idx in top_features:
        shap.dependence_plot(feat_idx, shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.show()
    
    # Global importance
    global_imp = pd.DataFrame({
        'feature': X_sample.columns,
        'shap_importance': np.abs(shap_values).mean(0)
    }).sort_values('shap_importance', ascending=False)
    
    print(global_imp.head(20).to_string(index=False))
    
    return shap_values, global_imp
```

---

# 14. YARIŞMA BİTTİKTEN SONRA

## 14.1 Sonuç Analizi

```python
"""
YARIŞMA BİTİŞİNDE YAPILACAKLAR:
──────────────────────────────────────────────────────────────────
1. SONUÇLARI İNCELE
   - Public LB vs Private LB farkı neydi?
   - Shake-up oldu mu? Neden?
   - CV-LB korelasyonun ne kadar doğruydu?

2. WINNING SOLUTION OKU
   - Discussion'da 1st, 2nd, 3rd write-up'ları oku
   - Kendi çözümünle karşılaştır
   - Kaçırdığın teknikler ne?

3. NOTEBOOK YAYINLA
   - Öğrendiklerini paylaş
   - Community'ye katkı → Kaggle Expert/Master yolu

4. ÖĞRENİLENLERİ KAYDET
   - Aşağıdaki template ile notlar al

SONRAKI YARIŞMA İÇİN NOTLAR:
"""

learning_template = {
    'competition': 'competition-name',
    'final_rank': None,
    'problem_type': None,
    'best_single_model': None,
    'best_ensemble': None,
    'most_impactful_feature': None,
    'biggest_mistake': None,
    'what_worked': [],
    'what_didnt_work': [],
    'techniques_to_try_next': [],
    'cv_lb_correlation': None,  # yüksek/orta/düşük
    'shake_up': None,  # büyük/küçük
}
```

## 14.2 Retrospektif Analiz

```python
def retrospective_analysis(cv_lb_tracker):
    """Yarışma sonrası CV-LB korelasyonunu analiz et"""
    
    if len(cv_lb_tracker) < 3:
        print("Yeterli data yok")
        return
    
    df = cv_lb_tracker.dropna(subset=['lb_score'])
    
    from scipy.stats import spearmanr
    
    corr, p = spearmanr(df['cv_score'], df['lb_score'])
    
    plt.figure(figsize=(8, 6))
    plt.scatter(df['cv_score'], df['lb_score'], color='steelblue', s=80, alpha=0.8)
    for _, row in df.iterrows():
        plt.annotate(row['experiment'],
                    (row['cv_score'], row['lb_score']),
                    fontsize=8, alpha=0.7)
    
    # Trend line
    z = np.polyfit(df['cv_score'], df['lb_score'], 1)
    xl = np.linspace(df['cv_score'].min(), df['cv_score'].max(), 50)
    plt.plot(xl, np.poly1d(z)(xl), 'r--', alpha=0.5)
    
    plt.xlabel('CV Score')
    plt.ylabel('LB Score')
    plt.title(f'CV-LB Correlation  (Spearman r={corr:.3f}, p={p:.4f})')
    plt.tight_layout()
    plt.show()
    
    print(f"CV-LB Spearman: {corr:.3f}  {'✅ Güvenilir' if corr > 0.9 else '⚠️ Güvenilmez'}")
```

---

# 15. HIZLI BAŞVURU: METRİK SÖZLÜĞÜ

```python
"""
METRİK          OPTIMIZE     KARAR
────────────────────────────────────────────────────────────────────
AUC-ROC         Maximize     predict_proba kullan, threshold'u daha sonra ayarla
Logloss         Minimize     Kalibrasyon kritik — isotonic regression
F1              Maximize     Threshold tuning şart — 0.5 genellikle optimal değil
F-beta          Maximize     beta < 1 → precision; beta > 1 → recall
Accuracy        Maximize     Imbalanced'da misleading
MCC             Maximize     Çok dengeli, imbalanced için iyi
RMSE            Minimize     log transform + outlier tedavisi
RMSLE           Minimize     log1p(y) tahmin et, expm1(pred) geri çevir
MAE             Minimize     median predictor optimal; outlier'a robust
R²              Maximize     Negatif olabilir (kötü model)
MAPE            Minimize     Küçük y için patlayabilir → epsilon ekle
MAP@K           Maximize     LTR modelleri; prediction'ları sırala
────────────────────────────────────────────────────────────────────
"""

# Her metrik için custom scorer — sklearn CV'de kullanım için
from sklearn.metrics import make_scorer

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)
weighted_f1_scorer = make_scorer(f1_score, average='weighted')
roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

# Competition metric custom objective (LightGBM için)
def lgb_rmsle_objective(y_pred, dtrain):
    """RMSLE için custom LightGBM objective"""
    y_true = dtrain.get_label()
    y_pred = np.expm1(y_pred)  # log space'den çevir
    grad = (np.log1p(y_pred) - np.log1p(y_true)) / (y_pred + 1)
    hess = (-np.log1p(y_pred) + np.log1p(y_true) + 1) / (y_pred + 1)**2
    return grad, hess

def lgb_rmsle_metric(y_pred, dtrain):
    """RMSLE için custom LightGBM eval metric"""
    y_true = dtrain.get_label()
    score = rmsle(y_true, np.expm1(y_pred))
    return 'rmsle', score, False  # False = lower is better
```

---

## APPENDIX: SIK KARŞILAŞILAN SORUNLAR

```python
"""
SORUN                              ÇÖZÜM
──────────────────────────────────────────────────────────────────
OOM (Out of Memory)                optimize_dtypes(), samp(), batch predict
Uzun eğitim süresi                 n_estimators↓, 3-fold, GPU, subsample
LB'de kötü score                   Leak var mı? CV stratejisi doğru mu?
NaN in predictions                 fillna, assert sonrası kontrol
Unseen categories test'te          'unknown' kategorisi, try-except
Negative predictions                clip(0, None) veya np.maximum(pred, 0)
Sınıf yok val'da (StratifiedKFold) min_samples kontrol et
Index karışıklığı                  reset_index(drop=True) her yerde kullan
Feature sayısı train≠test          ortak sütun listesi kullan
Overfitting CV                     early stopping, regularization, data+
──────────────────────────────────────────────────────────────────
"""
```

---

*Bu rehber yaşayan bir document'tır. Her yarışmadan sonra güncelleyin.*

**Son güncelleme:** 2025

---

---

# 🏆 KAGGLE REHBER — EK BÖLÜMLER
## Eksik Kalan Tüm Konular

---

# 16. VERİ BOYUTUNA GÖRE STRATEJİ

## 16.1 Az Veri (<1000 Satır)

```python
"""
AZ VERİ SORUNLARI:
- Model kolayca overfit eder
- CV çok gürültülü
- Feature sayısı > satır sayısı olabilir (curse of dimensionality)

STRATEJİ:
1. Basit modeller önce (Ridge, Lasso, SVM)
2. Ağır regularization
3. 10-fold veya LOOCV
4. Feature sayısını agresif düşür
5. Bayesian modeller (uncertainty estimation)
6. Augmentation (veri türetme)
"""

from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold

# LOOCV — her satır bir kez validation
def loocv_train(X, y, model):
    loo = LeaveOneOut()
    preds = np.zeros(len(X))
    for train_idx, val_idx in loo.split(X):
        m = deepcopy(model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        if hasattr(m, 'predict_proba'):
            preds[val_idx] = m.predict_proba(X.iloc[val_idx])[:, 1]
        else:
            preds[val_idx] = m.predict(X.iloc[val_idx])
    return preds

# Repeated Stratified KFold (az veri için daha stabil CV)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)

# Az veri için güçlü modeller
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# Bayesian Optimization ile az feature, ağır reg
lgb_small_data = {
    'num_leaves': 15,           # çok küçük tut
    'min_child_samples': 20,    # overfit önle
    'learning_rate': 0.01,      # küçük lr
    'n_estimators': 500,
    'lambda_l1': 5.0,
    'lambda_l2': 5.0,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'random_state': SEED,
    'verbose': -1,
}

# Data augmentation for tabular (SMOTE benzeri ama regresyon için)
def gaussian_noise_augmentation(X, y, n_augment=2, noise_std=0.01):
    """Gaussian gürültü ile yeni satırlar türet"""
    X_aug_list = [X]
    y_aug_list = [y]
    
    for _ in range(n_augment):
        noise = np.random.normal(0, noise_std, X.shape)
        X_noisy = X + noise * X.std()
        X_aug_list.append(pd.DataFrame(X_noisy, columns=X.columns))
        y_aug_list.append(y)
    
    return pd.concat(X_aug_list).reset_index(drop=True), \
           pd.concat(y_aug_list).reset_index(drop=True)
```

## 16.2 Büyük Veri (>1M Satır)

```python
"""
BÜYÜK VERİ STRATEJİSİ:
- Pandas yerine Polars (10-50x hızlı)
- Dask/Vaex (out-of-memory)
- Sampling ile EDA ve HPO
- Batch prediction
- Feature engineering'i vektörize yaz
"""

# Polars ile hızlı veri işleme
import polars as pl

def load_large_csv_polars(path):
    df = pl.read_csv(path)
    print(f"Shape: {df.shape}  |  Memory: {df.estimated_size()/1024**3:.2f} GB")
    return df

# Lazy evaluation — sadece gerektiğinde compute
def polars_lazy_pipeline(path):
    result = (
        pl.scan_csv(path)
        .filter(pl.col("value") > 0)
        .with_columns([
            (pl.col("a") * pl.col("b")).alias("a_x_b"),
            pl.col("date").str.to_date().dt.year().alias("year"),
        ])
        .group_by("category")
        .agg([
            pl.col("value").mean().alias("value_mean"),
            pl.col("value").std().alias("value_std"),
            pl.col("value").count().alias("count"),
        ])
        .collect()
    )
    return result

# Dask ile out-of-memory işlem
import dask.dataframe as dd

def process_with_dask(path):
    ddf = dd.read_csv(path, blocksize="128MB")
    
    # Lazy computation
    result = ddf.groupby('category')['value'].mean()
    
    # Execute
    computed = result.compute()
    return computed

# Büyük veri için batch prediction
def batch_predict(model, X, batch_size=50000):
    preds = []
    for i in range(0, len(X), batch_size):
        batch = X.iloc[i:i+batch_size]
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(batch)[:, 1]
        else:
            pred = model.predict(batch)
        preds.append(pred)
    return np.concatenate(preds)

# LightGBM için native large data
def lgb_large_data(X_train, y_train, X_val, y_val, params):
    """LightGBM Dataset API — daha memory efficient"""
    dtrain = lgb.Dataset(X_train, label=y_train,
                         free_raw_data=False)
    dval   = lgb.Dataset(X_val, label=y_val,
                         reference=dtrain, free_raw_data=False)
    
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(200)
        ]
    )
    return model

# Feature hashing (yüksek kardinalite için)
from sklearn.feature_extraction import FeatureHasher

def hash_encode(df, col, n_features=1024):
    """Yüksek kardinaliteli kategorik sütunları hash'le"""
    hasher = FeatureHasher(n_features=n_features, input_type='string')
    vals = df[col].astype(str).values.reshape(-1, 1)
    hashed = hasher.transform(vals).toarray()
    cols = [f'{col}_hash_{i}' for i in range(n_features)]
    return pd.DataFrame(hashed, columns=cols, index=df.index)
```

## 16.3 Geniş Veri (1000+ Feature)

```python
"""
GENİŞ VERİ STRATEJİSİ:
- Variance threshold ile sıfır-varyans kaldır
- Correlation thresholding (|r| > 0.95 → birini drop et)
- PCA / t-SNE / UMAP ile boyut azalt
- Regularized models (Lasso, ElasticNet)
- Feature importance ile agresif selection
"""

from sklearn.decomposition import PCA, TruncatedSVD, NMF
from sklearn.manifold import TSNE
import umap  # pip install umap-learn

def dimensionality_reduction_pipeline(X_train, X_test, method='pca', n_components=50):
    """Boyut azaltma pipeline"""
    
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=SEED)
        
    elif method == 'svd':  # sparse matrix için
        reducer = TruncatedSVD(n_components=n_components, random_state=SEED)
        
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=SEED,
                           n_neighbors=15, min_dist=0.1)
    
    train_reduced = reducer.fit_transform(X_train)
    test_reduced  = reducer.transform(X_test)
    
    if method == 'pca':
        explained = reducer.explained_variance_ratio_.cumsum()
        print(f"PCA {n_components} bileşen: %{explained[-1]*100:.1f} varyans açıklandı")
    
    # DataFrame'e çevir
    cols = [f'{method}_{i}' for i in range(n_components)]
    tr_df = pd.DataFrame(train_reduced, columns=cols)
    te_df = pd.DataFrame(test_reduced, columns=cols)
    
    return tr_df, te_df, reducer

# Variance threshold
from sklearn.feature_selection import VarianceThreshold

def remove_low_variance(X_train, X_test, threshold=0.01):
    sel = VarianceThreshold(threshold=threshold)
    X_train_sel = sel.fit_transform(X_train)
    X_test_sel  = sel.transform(X_test)
    
    kept = X_train.columns[sel.get_support()].tolist()
    removed = set(X_train.columns) - set(kept)
    print(f"Kaldırılan: {len(removed)}  |  Kalan: {len(kept)}")
    
    return pd.DataFrame(X_train_sel, columns=kept), \
           pd.DataFrame(X_test_sel, columns=kept), sel

# Correlation thresholding
def correlation_threshold_selection(X, threshold=0.95):
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    print(f"Yüksek korelasyon ({threshold}) — drop: {len(to_drop)}")
    return X.drop(columns=to_drop), to_drop
```

---

# 17. NLP — METİN ÖZELLİKLERİ

## 17.1 Metin Temizleme

```python
import re
import string
from collections import Counter

# pip install nltk spacy langdetect
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer    = PorterStemmer()

def clean_text(text, 
               lowercase=True,
               remove_html=True,
               remove_urls=True,
               remove_punctuation=True,
               remove_numbers=False,
               remove_stopwords=True,
               lemmatize=True,
               min_word_len=2):
    """Standart metin temizleme pipeline"""
    
    if not isinstance(text, str):
        return ''
    
    if lowercase:
        text = text.lower()
    
    if remove_html:
        text = re.sub(r'<[^>]+>', ' ', text)
    
    if remove_urls:
        text = re.sub(r'http\S+|www\S+', ' ', text)
    
    # Özel karakterler
    text = re.sub(r'[^\w\s]', ' ', text) if remove_punctuation \
           else text
    
    if remove_numbers:
        text = re.sub(r'\d+', ' ', text)
    
    # Tokenize
    tokens = text.split()
    
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    
    if min_word_len:
        tokens = [t for t in tokens if len(t) >= min_word_len]
    
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens)

# Batch temizleme
def clean_text_column(df, col, **kwargs):
    cleaned = df[col].fillna('').apply(lambda x: clean_text(x, **kwargs))
    return cleaned
```

## 17.2 TF-IDF ve Count Vectorizer

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline

def tfidf_features(train_texts, test_texts, max_features=50000, ngram_range=(1,2)):
    """TF-IDF feature extraction"""
    
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,      # (1,1)=unigram, (1,2)=bi, (1,3)=tri
        min_df=2,                      # en az 2 belgede geçmeli
        max_df=0.95,                   # max %95 belgede geçmeli (çok yaygın → anlamsız)
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{2,}',
        sublinear_tf=True,             # log(tf) — büyük veri için iyi
        norm='l2',
    )
    
    train_tfidf = tfidf.fit_transform(train_texts)
    test_tfidf  = tfidf.transform(test_texts)
    
    print(f"TF-IDF matrix: {train_tfidf.shape}")
    
    return train_tfidf, test_tfidf, tfidf

# Karakter n-gram (yazım hatalarına robust)
def char_tfidf(train_texts, test_texts, max_features=50000):
    tfidf = TfidfVectorizer(
        analyzer='char_wb',            # karakter n-gram (kelime sınırlarına saygılı)
        ngram_range=(2, 4),
        max_features=max_features,
        min_df=3,
        sublinear_tf=True,
    )
    return tfidf.fit_transform(train_texts), tfidf.transform(test_texts), tfidf

# SVD ile boyut azalt (LSA)
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

def lsa_features(train_tfidf, test_tfidf, n_components=100):
    """Latent Semantic Analysis"""
    svd  = TruncatedSVD(n_components=n_components, random_state=SEED)
    norm = Normalizer(copy=False)
    
    lsa_pipe = make_pipeline(svd, norm)
    train_lsa = lsa_pipe.fit_transform(train_tfidf)
    test_lsa  = lsa_pipe.transform(test_tfidf)
    
    print(f"LSA explained variance: {svd.explained_variance_ratio_.sum()*100:.1f}%")
    return train_lsa, test_lsa
```

## 17.3 Word Embeddings

```python
# pip install gensim
from gensim.models import Word2Vec, FastText
import gensim.downloader as api

def word2vec_features(texts, vector_size=100, window=5, method='mean'):
    """Word2Vec ile cümle embedding"""
    
    tokenized = [text.split() for text in texts]
    
    # Model eğit
    w2v = Word2Vec(
        tokenized,
        vector_size=vector_size,
        window=window,
        min_count=2,
        workers=4,
        epochs=10,
        seed=SEED
    )
    
    def text_to_vec(tokens):
        vecs = [w2v.wv[w] for w in tokens if w in w2v.wv]
        if not vecs:
            return np.zeros(vector_size)
        if method == 'mean':
            return np.mean(vecs, axis=0)
        elif method == 'max':
            return np.max(vecs, axis=0)
        elif method == 'tfidf_weighted':
            # TF-IDF ağırlıklı ortalama
            return np.mean(vecs, axis=0)  # basitleştirilmiş
    
    embeddings = np.array([text_to_vec(toks) for toks in tokenized])
    return embeddings, w2v

# Pretrained embeddings (FastText)
def fasttext_features(texts, model_name='fasttext-wiki-news-subwords-300'):
    """Pretrained FastText embeddings"""
    model = api.load(model_name)
    
    def get_embedding(text):
        tokens = text.split()
        vecs = [model[w] for w in tokens if w in model]
        return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)
    
    return np.array([get_embedding(text) for text in texts])

# Sentence transformers (modern, güçlü)
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

def sentence_transformer_features(texts, model_name='all-MiniLM-L6-v2',
                                   batch_size=256):
    """
    Güçlü sentence embeddings.
    Model seçimi:
    - all-MiniLM-L6-v2: hızlı, küçük
    - all-mpnet-base-v2: iyi denge
    - paraphrase-multilingual-mpnet-base-v2: çok dilli
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return embeddings
```

## 17.4 Transformer Modeller (BERT/DeBERTa)

```python
"""
NLP YARIŞMA STANDART PIPELINE:
─────────────────────────────────────────────────────────────────
1. Baseline: TF-IDF + LightGBM
2. İyileştirme: Sentence Transformers embeddings + LightGBM
3. Advanced: Fine-tune BERT/DeBERTa
4. Final: DeBERTa + LightGBM blend

EN İYİ MODELLER (2024):
- deberta-v3-base/large: çoğu NLP yarışmasında #1
- microsoft/deberta-v3-small: hızlı baseline
- roberta-base/large: sağlam alternatif
- google/electra-base-discriminator: efficient
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup, AdamW
)

# Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {
            'input_ids':      encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].squeeze()
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

# Model
class TransformerClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # [CLS] token veya mean pooling
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS
        # pooled = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) \
        #          / attention_mask.sum(-1).unsqueeze(-1)  # mean pooling
        
        return self.classifier(self.dropout(pooled))

# Training loop
def train_transformer(model_name, X_train, y_train, X_val, y_val,
                      num_classes=2, epochs=3, batch_size=16,
                      max_length=256, lr=2e-5):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_ds = TextDataset(X_train, y_train, tokenizer, max_length)
    val_ds   = TextDataset(X_val, y_val, tokenizer, max_length)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl   = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=2)
    
    model = TransformerClassifier(model_name, num_classes).to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dl) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss()
    best_score = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            logits = model(**{k: v for k, v in batch.items() if k != 'labels'})
            loss = criterion(logits, batch['labels'])
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        # Val
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**{k: v for k, v in batch.items() if k != 'labels'})
                probs = torch.softmax(logits, dim=-1)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(batch['labels'].cpu().numpy())
        
        all_preds  = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        if num_classes == 2:
            score = roc_auc_score(all_labels, all_preds[:, 1])
        else:
            score = roc_auc_score(all_labels, all_preds, multi_class='ovr')
        
        print(f"Epoch {epoch+1}: loss={train_loss/len(train_dl):.4f}  val_auc={score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model_state = deepcopy(model.state_dict())
    
    model.load_state_dict(best_model_state)
    return model, tokenizer

# TTA — Test Time Augmentation (NLP için)
def tta_predict(model, tokenizer, texts, max_length=256, n_tta=5):
    """
    NLP TTA: farklı truncation startları ile birden fazla tahmin al, ortala.
    """
    device = next(model.parameters()).device
    model.eval()
    
    all_preds = []
    for tta_idx in range(n_tta):
        ds = TextDataset(texts, tokenizer=tokenizer, max_length=max_length)
        dl = DataLoader(ds, batch_size=64, shuffle=False)
        preds = []
        
        with torch.no_grad():
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch)
                preds.append(torch.softmax(logits, dim=-1).cpu().numpy())
        
        all_preds.append(np.concatenate(preds))
    
    return np.mean(all_preds, axis=0)
```

## 17.5 NLP Metin Feature Engineering

```python
def nlp_feature_engineering(df, text_col):
    """Metin içindeki istatistiksel özellikler"""
    df = df.copy()
    t = df[text_col].fillna('')
    
    # Temel istatistikler
    df['text_len']          = t.str.len()
    df['word_count']        = t.str.split().str.len()
    df['unique_word_count'] = t.apply(lambda x: len(set(x.lower().split())))
    df['char_per_word']     = df['text_len'] / (df['word_count'] + 1)
    df['unique_ratio']      = df['unique_word_count'] / (df['word_count'] + 1)
    
    # Noktalama
    df['punct_count']   = t.apply(lambda x: sum(1 for c in x if c in string.punctuation))
    df['upper_count']   = t.apply(lambda x: sum(1 for c in x if c.isupper()))
    df['digit_count']   = t.apply(lambda x: sum(1 for c in x if c.isdigit()))
    df['space_count']   = t.str.count(' ')
    
    # Cümle sayısı
    df['sentence_count'] = t.str.count(r'[.!?]+')
    
    # Özel kelimeler
    df['stopword_count'] = t.apply(
        lambda x: sum(1 for w in x.lower().split() if w in STOP_WORDS)
    )
    df['stopword_ratio'] = df['stopword_count'] / (df['word_count'] + 1)
    
    # Emoji
    df['emoji_count'] = t.apply(
        lambda x: sum(1 for c in x if ord(c) > 127462)
    )
    
    # Ortalama kelime uzunluğu
    df['avg_word_len'] = t.apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )
    
    return df
```

---

# 18. GÖRÜNTÜ & MULTI-MODAL YARIŞMALAR

## 18.1 Görüntü Augmentation

```python
# pip install albumentations timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import cv2

# Train augmentation (agresif)
train_transforms = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.7, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    A.GridDistortion(p=0.2),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Val/Test (sadece normalize)
val_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# MixUp augmentation
def mixup_data(x, y, alpha=0.2):
    """MixUp: iki görüntüyü ve label'ı karıştır"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    idx = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[idx, :]
    y_a, y_b = y, y[idx]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# CutMix
def cutmix_data(x, y, alpha=1.0):
    """CutMix: rastgele dikdörtgen bölge kes ve başka görüntüden yapıştır"""
    lam = np.random.beta(alpha, alpha)
    rand_idx = torch.randperm(x.size(0)).to(x.device)
    
    _, _, H, W = x.size()
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    
    x_new = x.clone()
    x_new[:, :, y1:y2, x1:x2] = x[rand_idx, :, y1:y2, x1:x2]
    
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    
    return x_new, y, y[rand_idx], lam
```

## 18.2 Pretrained Models (timm)

```python
def get_image_model(model_name='efficientnet_b4', num_classes=10,
                    pretrained=True, drop_rate=0.3):
    """
    Popüler model seçenekleri:
    - efficientnet_b4/b5/b6      : iyi performans/hız dengesi
    - convnext_base/large        : modern CNN
    - vit_base_patch16_224       : Vision Transformer
    - swin_base_patch4_window7   : Swin Transformer
    - tf_efficientnetv2_m/l      : EfficientNetV2
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model

# Image Dataset
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels=None, transforms=None):
        self.paths      = image_paths
        self.labels     = labels
        self.transforms = transforms
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(str(self.paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            img = self.transforms(image=img)['image']
        
        item = {'image': img}
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx])
        
        return item

# TTA for images
def image_tta_predict(model, test_paths, n_tta=5, device='cuda'):
    """Test Time Augmentation"""
    tta_transforms = [
        val_transforms,
        A.Compose([A.HorizontalFlip(p=1), *val_transforms.transforms]),
        A.Compose([A.VerticalFlip(p=1), *val_transforms.transforms]),
        A.Compose([A.Rotate(limit=10, p=1), *val_transforms.transforms]),
    ]
    
    all_preds = []
    model.eval()
    
    for aug in tta_transforms[:n_tta]:
        ds = ImageDataset(test_paths, transforms=aug)
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
        preds = []
        
        with torch.no_grad():
            for batch in dl:
                imgs = batch['image'].to(device)
                logits = model(imgs)
                preds.append(torch.softmax(logits, dim=-1).cpu().numpy())
        
        all_preds.append(np.concatenate(preds))
    
    return np.mean(all_preds, axis=0)
```

## 18.3 Multi-modal Pipeline

```python
def multimodal_blend(tabular_preds, image_preds, text_preds,
                     oof_tab, oof_img, oof_txt, y_true):
    """
    Tabular + Image + Text modellerini blend et.
    OOF'lar üzerinde optimize et.
    """
    from scipy.optimize import minimize
    
    def neg_auc(weights):
        w = np.abs(weights) / np.abs(weights).sum()
        blend = w[0]*oof_tab + w[1]*oof_img + w[2]*oof_txt
        return -roc_auc_score(y_true, blend)
    
    result = minimize(neg_auc, [0.33, 0.33, 0.34],
                     method='Nelder-Mead',
                     options={'xatol': 1e-6, 'fatol': 1e-6})
    
    w = np.abs(result.x) / np.abs(result.x).sum()
    print(f"Optimal weights: Tab={w[0]:.3f}  Img={w[1]:.3f}  Txt={w[2]:.3f}")
    print(f"OOF AUC: {-result.fun:.5f}")
    
    final = w[0]*tabular_preds + w[1]*image_preds + w[2]*text_preds
    return final, w
```

---

# 19. NEURAL NETWORKS ON TABULAR DATA

## 19.1 TabNet

```python
# pip install pytorch-tabnet
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

def train_tabnet(X_train, y_train, X_val, y_val, X_test,
                 problem_type='classification', max_epochs=200):
    
    cat_idxs  = []  # kategorik sütun indexleri
    cat_dims  = []  # her kategorik sütunun cardinality'si
    
    # Kategorik indexleri bul
    for i, col in enumerate(X_train.columns):
        if str(X_train[col].dtype) in ['object', 'category', 'int8', 'int16']:
            unique = X_train[col].nunique()
            if unique < 200:
                cat_idxs.append(i)
                cat_dims.append(unique + 1)
    
    if problem_type == 'classification':
        model = TabNetClassifier(
            n_d=64, n_a=64,            # embedding boyutu
            n_steps=5,                  # attention adımları
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=3,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={'step_size': 50, 'gamma': 0.9},
            mask_type='sparsemax',
            device_name='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=0,
            seed=SEED,
        )
    else:
        model = TabNetRegressor(
            n_d=64, n_a=64, n_steps=5,
            cat_idxs=cat_idxs, cat_dims=cat_dims,
            device_name='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=0, seed=SEED,
        )
    
    model.fit(
        X_train=X_train.values, y_train=y_train.values,
        eval_set=[(X_val.values, y_val.values)],
        eval_name=['val'],
        eval_metric=['auc' if problem_type == 'classification' else 'rmse'],
        max_epochs=max_epochs,
        patience=30,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
    )
    
    if problem_type == 'classification':
        val_pred  = model.predict_proba(X_val.values)[:, 1]
        test_pred = model.predict_proba(X_test.values)[:, 1]
    else:
        val_pred  = model.predict(X_val.values).ravel()
        test_pred = model.predict(X_test.values).ravel()
    
    return model, val_pred, test_pred

# TabNet feature importance
def tabnet_feature_importance(model, X):
    explain_mat, _ = model.explain(X.values)
    importance = pd.DataFrame({
        'feature':    X.columns,
        'importance': explain_mat.mean(0)
    }).sort_values('importance', ascending=False)
    return importance
```

## 19.2 MLP (PyTorch)

```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim,
                 dropout=0.3, batch_norm=True):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
        self.y = torch.FloatTensor(y.values if y is not None and hasattr(y, 'values') else y) \
                 if y is not None else None
    
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

def train_mlp(X_train, y_train, X_val, y_val, X_test,
              hidden_dims=[512, 256, 128], dropout=0.3,
              epochs=100, lr=1e-3, batch_size=512):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Normalize
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train)
    X_v    = scaler.transform(X_val)
    X_te   = scaler.transform(X_test)
    
    train_ds = TabularDataset(X_tr, y_train)
    val_ds   = TabularDataset(X_v, y_val)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False)
    
    model = MLP(X_train.shape[1], hidden_dims, 1, dropout=dropout).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()
    
    best_score = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(device), y_b.to(device)
            pred = model(X_b).squeeze()
            loss = criterion(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        model.eval()
        val_preds = []
        with torch.no_grad():
            for X_b, _ in val_dl:
                pred = torch.sigmoid(model(X_b.to(device))).cpu().numpy()
                val_preds.append(pred.squeeze())
        
        val_preds = np.concatenate(val_preds)
        score = roc_auc_score(y_val, val_preds)
        
        if score > best_score:
            best_score = score
            best_state = deepcopy(model.state_dict())
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: AUC={score:.5f}  Best={best_score:.5f}")
    
    model.load_state_dict(best_state)
    
    # Test predictions
    test_ds = TabularDataset(X_te)
    test_dl = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False)
    
    model.eval()
    test_preds = []
    with torch.no_grad():
        for X_b in test_dl:
            pred = torch.sigmoid(model(X_b.to(device))).cpu().numpy()
            test_preds.append(pred.squeeze())
    
    return model, np.concatenate(test_preds), best_score
```

---

# 20. ZAMAN SERİSİ — DETAYLI REHBER

## 20.1 Zaman Serisi Feature Engineering

```python
def lag_features(df, target_col, group_col=None, lags=[1,2,3,7,14,28,30]):
    """
    Lag features — CRITICAL: sadece geçmişe bak, gelecekten sızdırma!
    group_col varsa (ör: store_id), her grup için ayrı lag hesapla.
    """
    df = df.copy().sort_values(['date_col'] if group_col is None
                                else [group_col, 'date_col'])
    
    for lag in lags:
        if group_col:
            df[f'{target_col}_lag_{lag}'] = df.groupby(group_col)[target_col].shift(lag)
        else:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    return df

def rolling_features(df, target_col, group_col=None,
                     windows=[7, 14, 28, 90],
                     funcs=['mean', 'std', 'min', 'max', 'median']):
    """Rolling window istatistikleri — min_periods ile NaN'ı minimize et"""
    df = df.copy()
    
    for window in windows:
        for func in funcs:
            col_name = f'{target_col}_roll_{window}_{func}'
            if group_col:
                df[col_name] = df.groupby(group_col)[target_col].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).agg(func)
                )
            else:
                df[col_name] = df[target_col].shift(1).rolling(
                    window, min_periods=1
                ).agg(func)
    
    return df

def ewm_features(df, target_col, group_col=None, spans=[7, 14, 30]):
    """Exponentially Weighted Mean — recent data'ya daha fazla ağırlık"""
    df = df.copy()
    
    for span in spans:
        col_name = f'{target_col}_ewm_{span}'
        if group_col:
            df[col_name] = df.groupby(group_col)[target_col].transform(
                lambda x: x.shift(1).ewm(span=span, adjust=False).mean()
            )
        else:
            df[col_name] = df[target_col].shift(1).ewm(span=span, adjust=False).mean()
    
    return df

def trend_features(df, target_col, group_col=None, windows=[7, 28]):
    """Trend — kısa/uzun dönem ortalama farkı"""
    df = df.copy()
    
    for w in windows:
        short = df[target_col].shift(1).rolling(w//2, min_periods=1).mean()
        long  = df[target_col].shift(1).rolling(w, min_periods=1).mean()
        df[f'{target_col}_trend_{w}'] = short - long
    
    return df

def difference_features(df, target_col, periods=[1, 7, 28]):
    """Differencing — stationarity için"""
    df = df.copy()
    for p in periods:
        df[f'{target_col}_diff_{p}'] = df[target_col].diff(p)
        df[f'{target_col}_pct_{p}']  = df[target_col].pct_change(p)
    return df
```

## 20.2 Zaman Serisi CV

```python
def purged_group_timeseries_cv(df, date_col, target_col, n_splits=5,
                                gap_days=7, purge_days=14):
    """
    Kaggle zaman serisi yarışmalarında doğru CV:
    1. Temporal order koru
    2. Gap: val başlamadan önce boşluk bırak (data leakage önle)
    3. Purge: train'in sonundan birkaç günü kaldır (look-ahead leak)
    """
    dates = pd.to_datetime(df[date_col])
    min_date = dates.min()
    max_date = dates.max()
    total_days = (max_date - min_date).days
    
    fold_size = total_days // (n_splits + 1)
    
    splits = []
    for fold in range(n_splits):
        train_end = min_date + pd.Timedelta(days=(fold+1)*fold_size - gap_days)
        train_start = min_date
        
        val_start = train_end + pd.Timedelta(days=gap_days)
        val_end   = min_date + pd.Timedelta(days=(fold+2)*fold_size)
        
        # Purge: train'in son purge_days'ini kaldır
        purge_start = train_end - pd.Timedelta(days=purge_days)
        
        train_mask = (dates >= train_start) & (dates <= purge_start)
        val_mask   = (dates >= val_start)   & (dates <= val_end)
        
        splits.append((
            np.where(train_mask)[0],
            np.where(val_mask)[0]
        ))
    
    return splits

# Walk-forward validation
def walk_forward_cv(df, date_col, target, n_folds=5,
                    initial_train_ratio=0.5):
    """Her adımda train seti büyür, val sabit pencere"""
    dates = pd.to_datetime(df[date_col]).sort_values()
    n = len(df)
    
    initial_size = int(n * initial_train_ratio)
    fold_size = (n - initial_size) // n_folds
    
    splits = []
    for fold in range(n_folds):
        train_end = initial_size + fold * fold_size
        val_end   = train_end + fold_size
        
        splits.append((
            np.arange(train_end),
            np.arange(train_end, min(val_end, n))
        ))
    
    return splits
```

## 20.3 İleri Seviye Zaman Serisi

```python
# Prophet
from prophet import Prophet

def prophet_forecast(df, date_col, target_col, future_periods=365,
                     freq='D', regressors=None):
    """Meta Prophet — mevsimsellik ve tatil etkisi"""
    
    df_prophet = df[[date_col, target_col]].rename(
        columns={date_col: 'ds', target_col: 'y'}
    )
    
    model = Prophet(
        seasonality_mode='multiplicative',  # veya 'additive'
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,       # trend flexibility
        seasonality_prior_scale=10,
    )
    
    # Tatil etkisi
    from prophet.make_holidays import make_holidays_df
    holidays = make_holidays_df(year_list=[2020, 2021, 2022, 2023], country='US')
    model = Prophet(holidays=holidays)
    
    # Ekstra regressor
    if regressors:
        for reg in regressors:
            model.add_regressor(reg)
    
    model.fit(df_prophet)
    
    future = model.make_future_dataframe(periods=future_periods, freq=freq)
    forecast = model.predict(future)
    
    return model, forecast

# Gradient Boosting ile zaman serisi (lag+rolling features + LightGBM)
def lgb_time_series(df, feature_cols, target_col, date_col,
                    test_df=None, n_splits=5):
    """
    En güçlü yaklaşım çoğu Kaggle zaman serisi için:
    LightGBM + lag/rolling features + temporal CV
    """
    
    splits = walk_forward_cv(df, date_col, target_col, n_folds=n_splits)
    
    oof_preds  = np.zeros(len(df))
    test_preds = np.zeros(len(test_df)) if test_df is not None else None
    cv_scores  = []
    
    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        X_tr = df[feature_cols].iloc[train_idx]
        y_tr = df[target_col].iloc[train_idx]
        X_v  = df[feature_cols].iloc[val_idx]
        y_v  = df[target_col].iloc[val_idx]
        
        model = lgb.LGBMRegressor(**lgb_reg_params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_v, y_v)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        
        pred = model.predict(X_v)
        oof_preds[val_idx] = pred
        
        rmse = np.sqrt(mean_squared_error(y_v, pred))
        cv_scores.append(rmse)
        print(f"Fold {fold}: RMSE={rmse:.5f}")
        
        if test_df is not None:
            test_preds += model.predict(test_df[feature_cols]) / n_splits
    
    print(f"\nCV RMSE: {np.mean(cv_scores):.5f} ± {np.std(cv_scores):.5f}")
    
    return oof_preds, test_preds, cv_scores
```

---

# 21. DENGESIZ VERİ (IMBALANCED DATA)

## 21.1 Resampling Teknikleri

```python
# pip install imbalanced-learn
from imblearn.over_sampling import (
    SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, RandomOverSampler
)
from imblearn.under_sampling import (
    RandomUnderSampler, TomekLinks, EditedNearestNeighbours, NearMiss
)
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

def compare_resampling_strategies(X_train, y_train, X_val, y_val):
    """
    Farklı resampling stratejilerini karşılaştır.
    KARAR: OOF AUC ve F1'e göre seç.
    
    GENEL KURAL:
    - Oran < 10:1  → class_weight='balanced' genellikle yeterli
    - Oran 10:1-100:1 → SMOTE veya ADASYN
    - Oran > 100:1  → EasyEnsemble veya custom thresholding
    """
    
    strategies = {
        'no_resampling': None,
        'random_over':   RandomOverSampler(random_state=SEED),
        'smote':         SMOTE(random_state=SEED, k_neighbors=5),
        'borderline':    BorderlineSMOTE(random_state=SEED),
        'adasyn':        ADASYN(random_state=SEED),
        'tomek':         TomekLinks(),
        'smote_tomek':   SMOTETomek(random_state=SEED),
        'smote_enn':     SMOTEENN(random_state=SEED),
    }
    
    results = {}
    base_model = lgb.LGBMClassifier(n_estimators=300, random_state=SEED, verbose=-1)
    
    for name, sampler in strategies.items():
        if sampler is not None:
            X_res, y_res = sampler.fit_resample(X_train, y_train)
        else:
            X_res, y_res = X_train, y_train
        
        m = deepcopy(base_model)
        m.fit(X_res, y_res)
        
        val_pred = m.predict_proba(X_val)[:, 1]
        auc  = roc_auc_score(y_val, val_pred)
        f1   = f1_score(y_val, (val_pred > 0.5).astype(int))
        
        results[name] = {'auc': auc, 'f1': f1,
                         'train_size': len(y_res),
                         'pos_rate': y_res.mean()}
        print(f"{name:20s}  AUC={auc:.4f}  F1={f1:.4f}  size={len(y_res):,}")
    
    return results
```

## 21.2 Anomali Tespiti (Extreme Imbalance)

```python
"""
Pozitif oran < %1 ise → Anomaly Detection yaklaşımı daha iyi olabilir
"""
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

def anomaly_detection_ensemble(X_train, X_test, contamination=0.01):
    """Anomali tespiti modelleri ensemble"""
    
    models = {
        'isolation_forest': IsolationForest(
            contamination=contamination, random_state=SEED, n_jobs=-1
        ),
        'elliptic': EllipticEnvelope(
            contamination=contamination, random_state=SEED
        ),
    }
    
    scores = {}
    for name, model in models.items():
        # Sadece negatif (normal) örneklerle eğit
        model.fit(X_train)
        # -1 = anomali, 1 = normal
        scores[name] = -model.decision_function(X_test)  # yüksek = anomali
    
    # Ensemble: normalize et ve ortala
    all_scores = np.column_stack(list(scores.values()))
    from sklearn.preprocessing import MinMaxScaler
    normalized = MinMaxScaler().fit_transform(all_scores)
    final_score = normalized.mean(axis=1)
    
    return final_score
```

---

# 22. ÖZEL PROBLEM TİPLERİ

## 22.1 Ordinal Regression

```python
"""
Ordinal target: 1 < 2 < 3 < 4 < 5 gibi sıralı kategoriler.
Standart classification yanlış (1-5 arası mesafe eşit değil).
"""

# Yaklaşım 1: Regression olarak çöz, sonra round
def ordinal_as_regression(X_train, y_train, X_test, n_classes=5):
    model = lgb.LGBMRegressor(**lgb_reg_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return np.clip(np.round(preds).astype(int), 1, n_classes)

# Yaklaşım 2: Binary threshold cascade
def ordinal_binary_cascade(X_train, y_train, X_test, n_classes=5):
    """
    K-1 binary classifier zinciri.
    P(y > k) için ayrı model.
    """
    binary_preds = []
    
    for k in range(1, n_classes):
        y_bin = (y_train > k).astype(int)
        model = lgb.LGBMClassifier(**lgb_clf_params)
        model.fit(X_train, y_bin)
        binary_preds.append(model.predict_proba(X_test)[:, 1])
    
    # P(y = k) = P(y > k-1) - P(y > k)
    probs = np.column_stack([
        1 - binary_preds[0],
        *[binary_preds[k-1] - binary_preds[k] for k in range(1, n_classes-1)],
        binary_preds[-1]
    ])
    
    return np.argmax(probs, axis=1) + 1  # 1-indexed

# Yaklaşım 3: mord library
# pip install mord
from mord import LogisticAT, LogisticIT

def ordinal_logistic(X_train, y_train, X_test):
    model = LogisticAT(alpha=1.0)
    model.fit(X_train, y_train)
    return model.predict(X_test)
```

## 22.2 Multi-label Classification

```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold  # pip install iterative-stratification

def multilabel_cv(X, y_multilabel, params, n_splits=5):
    """
    Multi-label için özel CV — her label'ın dağılımını koru.
    """
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    n_labels = y_multilabel.shape[1]
    oof_preds = np.zeros_like(y_multilabel, dtype=float)
    
    for fold, (train_idx, val_idx) in enumerate(mskf.split(X, y_multilabel), 1):
        X_tr, X_v = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_v = y_multilabel[train_idx], y_multilabel[val_idx]
        
        # Her label için ayrı model (OneVsRest)
        model = MultiOutputClassifier(
            lgb.LGBMClassifier(**params), n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        
        preds = np.column_stack([
            m.predict_proba(X_v)[:, 1]
            for m in model.estimators_
        ])
        
        oof_preds[val_idx] = preds
        
        # Metrik: macro F1 veya label-wise AUC
        from sklearn.metrics import label_ranking_average_precision_score
        score = label_ranking_average_precision_score(y_v, preds)
        print(f"Fold {fold}: LRAP={score:.5f}")
    
    return oof_preds
```

## 22.3 Survival Analysis

```python
# pip install lifelines scikit-survival
from lifelines import KaplanMeierFitter, CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis

def survival_analysis(df, duration_col, event_col, feature_cols):
    """
    Yarışma: C-index (concordance index) metriği
    duration_col: gözlem süresi
    event_col: event oldu mu (1=evet, 0=censored)
    """
    
    # Kaplan-Meier — genel survival curve
    kmf = KaplanMeierFitter()
    kmf.fit(df[duration_col], df[event_col])
    kmf.plot_survival_function()
    plt.title('Kaplan-Meier Survival Curve')
    plt.show()
    
    # Cox PH
    cox_df = df[feature_cols + [duration_col, event_col]]
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_df, duration_col=duration_col, event_col=event_col)
    cph.print_summary()
    
    # Random Survival Forest
    y_surv = np.array(
        [(bool(e), d) for e, d in zip(df[event_col], df[duration_col])],
        dtype=[('event', '?'), ('duration', '<f8')]
    )
    
    rsf = RandomSurvivalForest(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=SEED
    )
    rsf.fit(df[feature_cols], y_surv)
    
    return cph, rsf
```

## 22.4 Ranking / Recommendation

```python
"""
MAP@K, NDCG gibi sıralama metriklerinde LightGBM lambdarank.
"""

def lightgbm_ranking(train_df, test_df, feature_cols, relevance_col, query_col):
    """
    LightGBM LambdaRank.
    relevance_col: 0-4 arası relevance score
    query_col: query/group ID (ör: user_id, search_query)
    """
    
    # Her query'nin kaç dokümanı var
    train_groups = train_df.groupby(query_col).size().values
    
    lgb_rank_params = {
        'objective':     'lambdarank',
        'metric':        'ndcg',
        'ndcg_eval_at':  [1, 5, 10],
        'learning_rate': 0.05,
        'num_leaves':    127,
        'n_estimators':  500,
        'min_child_samples': 20,
        'n_jobs':       -1,
        'verbose':      -1,
    }
    
    dtrain = lgb.Dataset(
        train_df[feature_cols],
        label=train_df[relevance_col],
        group=train_groups
    )
    
    model = lgb.train(lgb_rank_params, dtrain, num_boost_round=500)
    
    scores = model.predict(test_df[feature_cols])
    
    return scores, model

def map_at_k_metric(y_true_grouped, y_score_grouped, k=12):
    """Grouped MAP@K hesaplama"""
    aps = []
    for y_true, y_score in zip(y_true_grouped, y_score_grouped):
        sorted_idx = np.argsort(y_score)[::-1][:k]
        hits = [1 if y_true[i] == 1 else 0 for i in sorted_idx]
        
        if sum(hits) == 0:
            aps.append(0.0)
            continue
        
        precision_at_k = [
            sum(hits[:j+1]) / (j+1)
            for j in range(k) if hits[j] == 1
        ]
        aps.append(np.mean(precision_at_k))
    
    return np.mean(aps)
```

---

# 23. EXPERIMENT TRACKING

## 23.1 MLflow ile Tracking

```python
import mlflow
import mlflow.lightgbm
import mlflow.sklearn

# MLflow setup
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment(COMP_NAME)

def train_with_mlflow(X_train, y_train, X_test, params, run_name, notes=''):
    """MLflow ile tam experiment tracking"""
    
    with mlflow.start_run(run_name=run_name):
        # Parametreleri logla
        mlflow.log_params(params)
        mlflow.log_param('n_features', X_train.shape[1])
        mlflow.log_param('n_samples',  X_train.shape[0])
        mlflow.log_param('notes',      notes)
        
        # Model eğit
        results = train_cv(X_train, y_train, X_test, params)
        
        # Metrikleri logla
        mlflow.log_metric('oof_score', results['oof_score'])
        mlflow.log_metric('cv_mean',   np.mean(results['cv_scores']))
        mlflow.log_metric('cv_std',    np.std(results['cv_scores']))
        
        for fold, score in enumerate(results['cv_scores'], 1):
            mlflow.log_metric(f'fold_{fold}_score', score)
        
        # Grafikleri logla
        fig, ax = plt.subplots()
        results['importance'].head(30).plot(kind='barh', ax=ax)
        plt.tight_layout()
        mlflow.log_figure(fig, 'feature_importance.png')
        plt.close()
        
        # Modeli kaydet
        mlflow.lightgbm.log_model(results['models'][0], 'model')
        
        # Submission'ı artifact olarak kaydet
        sub = pd.DataFrame({'id': range(len(results['test_preds'])),
                            'pred': results['test_preds']})
        sub.to_csv('/tmp/submission.csv', index=False)
        mlflow.log_artifact('/tmp/submission.csv')
        
        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")
    
    return results

# Geçmiş deneyleri karşılaştır
def compare_runs():
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(COMP_NAME)
    
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.oof_score DESC"]
    )
    
    summary = []
    for run in runs:
        summary.append({
            'name':      run.data.tags.get('mlflow.runName', ''),
            'oof_score': run.data.metrics.get('oof_score', 0),
            'cv_mean':   run.data.metrics.get('cv_mean', 0),
            'run_id':    run.info.run_id[:8]
        })
    
    return pd.DataFrame(summary)
```

## 23.2 Manuel Experiment Logger

```python
"""MLflow kuramıyorsan veya Kaggle notebook'ta çalışıyorsan."""

import json
from datetime import datetime

class ExperimentLogger:
    def __init__(self, log_file='experiments.json'):
        self.log_file = WORK_DIR / log_file
        self.experiments = self._load()
    
    def _load(self):
        if self.log_file.exists():
            with open(self.log_file) as f:
                return json.load(f)
        return []
    
    def log(self, name, cv_score, lb_score=None, params=None,
            features=None, notes='', **kwargs):
        exp = {
            'id':        len(self.experiments) + 1,
            'name':      name,
            'timestamp': datetime.now().isoformat(),
            'cv_score':  cv_score,
            'lb_score':  lb_score,
            'params':    params or {},
            'n_features': len(features) if features else None,
            'features':  features[:20] if features else None,
            'notes':     notes,
            **kwargs
        }
        self.experiments.append(exp)
        self._save()
        self._print_latest()
        return exp
    
    def _save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
    
    def _print_latest(self):
        e = self.experiments[-1]
        print(f"\n[#{e['id']}] {e['name']}")
        print(f"  CV: {e['cv_score']:.5f}  |  LB: {e['lb_score'] or 'pending'}")
        print(f"  Notes: {e['notes']}")
    
    def summary(self):
        if not self.experiments:
            print("Henüz experiment yok")
            return
        
        df = pd.DataFrame(self.experiments)[
            ['id', 'name', 'cv_score', 'lb_score', 'notes']
        ].sort_values('cv_score', ascending=False)
        
        print(df.to_string(index=False))
        
        # CV-LB korelasyonu
        lb_exists = df.dropna(subset=['lb_score'])
        if len(lb_exists) >= 3:
            from scipy.stats import spearmanr
            r, p = spearmanr(lb_exists['cv_score'], lb_exists['lb_score'])
            print(f"\nCV-LB Spearman r={r:.3f}  (p={p:.4f})")
            
            if r < 0.7:
                print("⚠️  CV-LB korelasyonu düşük — CV stratejini gözden geçir!")
        
        return df

# Kullanım
logger = ExperimentLogger()
# logger.log('LGB baseline', cv_score=0.8523, lb_score=0.8491,
#            params=lgb_clf_params, notes='default params')
# logger.log('LGB + FE v1',  cv_score=0.8601, lb_score=None,
#            notes='interaction features eklendi')
```

## 23.3 Model Versiyonlama

```python
def save_model(model, name, metadata=None):
    """Model ve metadata'yı kaydet"""
    model_path = MODEL_DIR / f"{name}.pkl"
    meta_path  = MODEL_DIR / f"{name}_meta.json"
    
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    if metadata:
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"Model kaydedildi: {model_path}")
    return model_path

def load_model(name):
    import pickle
    with open(MODEL_DIR / f"{name}.pkl", 'rb') as f:
        return pickle.load(f)

def save_oof_predictions(oof_preds, name):
    """OOF prediction'larını kaydet — sonra stacking'de kullan"""
    path = OUT_DIR / f"oof_{name}.npy"
    np.save(path, oof_preds)
    print(f"OOF saved: {path}")

def load_oof_predictions(name):
    return np.load(OUT_DIR / f"oof_{name}.npy")

# Checkpoint sistemi
def save_checkpoint(state_dict, epoch, score, name='checkpoint'):
    """PyTorch modelini checkpoint olarak kaydet"""
    path = MODEL_DIR / f"{name}_epoch{epoch}_score{score:.4f}.pt"
    torch.save({
        'epoch':      epoch,
        'score':      score,
        'state_dict': state_dict,
        'timestamp':  datetime.now().isoformat()
    }, path)
    print(f"Checkpoint: {path}")
    return path
```

---

# 24. GPU KULLANIMI

## 24.1 Kaggle'da GPU Aktifleştirme

```python
"""
Kaggle Notebook'ta GPU:
Settings → Accelerator → GPU T4 x2 veya P100

GPU tipini kontrol et:
"""
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)

# PyTorch GPU kontrolü
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")
```

## 24.2 LightGBM GPU

```python
# LightGBM GPU parametreleri
lgb_gpu_params = {
    **lgb_clf_params,
    'device':        'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id':   0,
    # GPU için num_leaves düşür (memory)
    'num_leaves':    63,
}

# Eğitim ~5-10x hızlanır büyük datasette
```

## 24.3 XGBoost GPU

```python
xgb_gpu_params = {
    **xgb_params,
    'tree_method':   'gpu_hist',
    'device':        'cuda',      # yeni XGBoost versiyonu
    'predictor':     'gpu_predictor',
}
```

## 24.4 CatBoost GPU

```python
cb_gpu_params = {
    **cb_params,
    'task_type': 'GPU',
    'devices':   '0',   # GPU index
}
```

## 24.5 Mixed Precision Training (PyTorch)

```python
from torch.cuda.amp import autocast, GradScaler

def train_with_amp(model, train_dl, optimizer, criterion, device):
    """
    Automatic Mixed Precision — 2x hızlı, yarı bellek.
    FP16 forward/backward, FP32 weights.
    """
    scaler = GradScaler()
    model.train()
    
    for X_b, y_b in train_dl:
        X_b, y_b = X_b.to(device), y_b.to(device)
        
        optimizer.zero_grad()
        
        with autocast():  # FP16 forward pass
            pred = model(X_b).squeeze()
            loss = criterion(pred, y_b)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
```

---

# 25. KAGGLE API & OTOMASYON

## 25.1 Kaggle CLI

```bash
# Kurulum
pip install kaggle

# API key: kaggle.com/settings → API → Create New Token
# ~/.kaggle/kaggle.json veya Kaggle Secrets'a ekle

# Yarışma datasını indir
kaggle competitions download -c competition-name -p /kaggle/input/

# Submission gönder
kaggle competitions submit -c competition-name \
    -f /kaggle/working/output/submission.csv \
    -m "LGB v3 + FE interaction features, OOF=0.8601"

# Leaderboard görüntüle
kaggle competitions leaderboard -c competition-name --show

# Public notebooks listesi
kaggle kernels list -c competition-name --sort-by hotness
```

## 25.2 Kaggle'da Gizli Değerleri Kullanma

```python
# Kaggle Secrets API (API key saklamak için)
from kaggle_secrets import UserSecretsClient

def get_kaggle_secret(secret_name):
    secrets = UserSecretsClient()
    return secrets.get_secret(secret_name)

# Kullanım
# api_key = get_kaggle_secret("WANDB_API_KEY")
```

## 25.3 Datasets ile Model Saklama

```python
"""
Kaggle'da /kaggle/working session bitince temizlenir.
Büyük modelleri Dataset olarak yükle, sonra reuse et.

1. Output klasöründe modeli kaydet
2. "Save Version" → Dataset'e ekle
3. Yeni notebook'ta input olarak ekle
"""

def save_for_kaggle_dataset(model, name):
    """Kaggle Dataset'e yüklenebilecek format"""
    import joblib
    path = f'/kaggle/working/{name}.joblib'
    joblib.dump(model, path)
    print(f"Saved for dataset upload: {path}")
```

---

# 26. FORUM & DISCUSSION STRATEJİSİ

## 26.1 Yarışma Boyunca Yapılacaklar

```python
"""
HAFTA 1:
□ Discussion'daki tüm konuları oku
□ Data description'ı 3 kez oku — anlamadığın şeyleri sor
□ "Is [feature] available at inference time?" sorusunu sor
□ EDA notebook'larına bak — veriyi nasıl anlıyorlar?
□ Baseline public notebook'u bul ve çalıştır

HAFTA 2-3:
□ Her gün discussion'a bak — yeni keşifler paylaşılıyor mu?
□ "External data" başlığı var mı? Hangi veri kullanılabilir?
□ Evaluation metric hakkında tartışma var mı?
□ Interesting insights paylaşıldığında like at — takip et

HAFTA 4:
□ "What's working" thread'lerini oku
□ Kimler nasıl bir CV stratejisi kullanıyor?
□ Büyük shake-up bekleniyor mu? (Public LB'nin boyutuna bak)
□ Son submissions'ı planla

KAPANIŞTAN SONRA:
□ 1st, 2nd, 3rd place write-up'larını oku
□ Silver/Bronze zone write-up'ları da oku
□ Kendi çözümünü yayınla (community'ye katkı)
"""

def analyze_public_lb(leaderboard_df):
    """
    Public LB'yi analiz et:
    - Top score ne? Makul mu?
    - Score dağılımı nasıl? (Bir yerde kümeleme = plateau)
    - Kaç kişi katılmış?
    """
    print(f"Top score:    {leaderboard_df['score'].max():.5f}")
    print(f"Median score: {leaderboard_df['score'].median():.5f}")
    print(f"Your rank:    Need to check manually")
    print(f"Participants: {len(leaderboard_df)}")
    
    plt.figure(figsize=(10, 5))
    leaderboard_df['score'].hist(bins=50, color='steelblue', edgecolor='white')
    plt.title('Public LB Score Dağılımı')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.show()
```

## 26.2 Shake-up Analizi

```python
"""
SHAKE-UP TAHMİNİ:
─────────────────────────────────────────────────────
Public LB test boyutu küçükse → büyük shake-up
Zaman serisi yarışmalar → genellikle büyük shake-up
Çok fazla feature leak → LB'ye güvenme
CV ile LB korelasyonu düşükse → shake-up olabilir

STRATEJI:
Büyük shake-up bekliyorsan → en yüksek CV modelini seç
Küçük shake-up bekliyorsan → en yüksek LB modelini seç
"""

def estimate_shakeup_risk(competition_info):
    """Shake-up riskini tahmin et"""
    public_pct = competition_info.get('public_test_pct', 30)
    has_time_series = competition_info.get('time_series', False)
    cv_lb_corr = competition_info.get('cv_lb_corr', 0.9)
    
    risk = 'LOW'
    reasons = []
    
    if public_pct < 30:
        risk = 'HIGH'
        reasons.append(f"Public test only {public_pct}%")
    
    if has_time_series:
        risk = 'HIGH'
        reasons.append("Time series — future data different")
    
    if cv_lb_corr < 0.7:
        risk = 'MEDIUM' if risk == 'LOW' else risk
        reasons.append(f"Low CV-LB correlation: {cv_lb_corr:.2f}")
    
    print(f"Shake-up risk: {risk}")
    for r in reasons:
        print(f"  • {r}")
    
    if risk == 'HIGH':
        print("\n→ Son submission: CV'ye göre seç, public LB'yi yoksay")
    elif risk == 'LOW':
        print("\n→ Son submission: Public LB'ye güvenebilirsin")
    
    return risk
```

---

# 27. TAKIM ÇALIŞMASI

## 27.1 Kaggle Takım Merge

```python
"""
TAKIM OLUŞTURMA KURALLARI:
- Yarışma bitişinden önce merge gerekir
- Merge sonrası submission limiti: max(ikisinin limiti)
- "Merge Teams" butonunu takım liderine bas

TAKIM ÇALIŞMASI STRATEJİSİ:
─────────────────────────────────────────────────────
KİM NE YAPAR:
Person A: EDA + Feature Engineering + LightGBM
Person B: XGBoost + CatBoost + Neural Networks
Person C: NLP features + Stacking + Ensemble
─────────────────────────────────────────────────────

ALTIN KURAL:
OOF prediction'larını paylaş, feature'larını paylaş,
ama aynı pipeline'ı kopyalama — diversity önemli!
"""

def merge_team_oof_preds(person_a_oof, person_b_oof, person_c_oof,
                          person_a_test, person_b_test, person_c_test,
                          y_train):
    """
    Farklı takım üyelerinin OOF'larını blend et.
    """
    # Her biri için OOF score
    for name, oof in [('A', person_a_oof), ('B', person_b_oof), ('C', person_c_oof)]:
        score = roc_auc_score(y_train, oof)
        print(f"Person {name} OOF AUC: {score:.5f}")
    
    # Korelasyonları kontrol et — düşük korelasyon = iyi diversity
    print("\nOOF Correlations:")
    for (na, oa), (nb, ob) in [
        (('A', person_a_oof), ('B', person_b_oof)),
        (('A', person_a_oof), ('C', person_c_oof)),
        (('B', person_b_oof), ('C', person_c_oof)),
    ]:
        r = np.corrcoef(oa, ob)[0, 1]
        print(f"  {na}-{nb}: r={r:.4f}  {'✅ Good diversity' if r < 0.95 else '⚠️ Similar models'}")
    
    # Optimal weights
    oof_list  = [person_a_oof,  person_b_oof,  person_c_oof]
    test_list = [person_a_test, person_b_test, person_c_test]
    
    weights = optimize_weights(oof_list, y_train, roc_auc_score)
    final_test = weighted_average(test_list, weights)
    
    return final_test, weights
```

## 27.2 Zaman Yönetimi

```python
"""
4 HAFTALIK YARIŞMA TAKVİMİ:
─────────────────────────────────────────────────────────────────
HAFTA 1 — ANLAMA & BASELINE
  Gün 1-2: Veri ve problem anlama, EDA, discussion okuma
  Gün 3-4: Baseline model kur, ilk submission yap
  Gün 5-7: Feature engineering ilk round, CV stratejisi kur

HAFTA 2 — GELİŞTİRME
  Gün 8-10: Feature engineering derinleştir (aggregation, target enc)
  Gün 11-12: XGBoost/CatBoost ekle, blend dene
  Gün 13-14: HPO (Optuna) çalıştır, hafta sonu

HAFTA 3 — OPTIMIZASYON
  Gün 15-17: Feature selection, önemli FE keşifleri
  Gün 18-19: Neural network dene (TabNet/MLP)
  Gün 20-21: Stacking, ensemble optimize et

HAFTA 4 — FİNAL
  Gün 22-24: Tüm modelleri birleştir, final blend
  Gün 25-26: Submission stratejisini planla
  Gün 27-28: Final submission seçimi
  Son saat: 2 farklı submission seç, paylaş

GÜNLÜK RUTİN:
  09:00 — Discussion'ı kontrol et
  09:30 — Dün denediklerini değerlendir
  10:00 — Yeni deney başlat
  17:00 — Sonuçları kaydet, yarın için plan yap
"""
```

---

# 28. ÖZEL LOSS FONKSİYONLARI

## 28.1 Custom Objectives

```python
# ─── Focal Loss (imbalanced classification için) ─────────────────
def focal_loss_lgb(y_pred, dtrain, gamma=2.0, alpha=0.25):
    """
    Focal loss: zor örneklere daha fazla ağırlık ver.
    gamma=2 standard, alpha=class_balance
    """
    y_true = dtrain.get_label()
    
    p = 1 / (1 + np.exp(-y_pred))
    
    grad = (alpha * (1-p)**gamma * (gamma*p*np.log(p+1e-8) + p - 1) * y_true +
            (1-alpha) * p**gamma * (gamma*(1-p)*np.log(1-p+1e-8) - p) * (1-y_true))
    
    hess = (alpha * (1-p)**gamma * (gamma**2 * p * np.log(p+1e-8) + gamma*p -
             gamma*(1-p) + p) * y_true +
            (1-alpha) * p**gamma * (gamma**2 * (1-p) * np.log(1-p+1e-8) -
             gamma*(1-p) + gamma*p + 1-p) * (1-y_true))
    
    return grad, hess

# Kullanım:
# lgb.train({'objective': focal_loss_lgb, ...}, dtrain, ...)

# ─── Tweedie Loss (sigorta, sıfır ağırlıklı) ─────────────────────
# Tweedie LightGBM'de built-in:
tweedie_params = {
    'objective':      'tweedie',
    'tweedie_variance_power': 1.5,  # 1=Poisson, 2=Gamma, 1-2=Tweedie
    'metric':         'tweedie',
}

# ─── Huber Loss (outlier'a robust regression) ────────────────────
huber_params = {
    'objective':   'huber',
    'alpha':       0.9,  # quantile threshold
}

# ─── Asymmetric Loss (false positive ve negative farklı maliyet) ──
def asymmetric_loss(y_pred, dtrain, alpha=1.5):
    """
    False negative'leri false positive'lerden daha fazla cezalandır.
    alpha > 1: FN daha pahalı (dolandırıcılık tespiti gibi)
    """
    y_true = dtrain.get_label()
    residual = y_true - y_pred
    
    grad = np.where(residual > 0, -2*alpha*residual, -2*residual)
    hess = np.where(residual > 0, 2*alpha, 2.0)
    
    return grad, hess

# ─── Custom Eval Metric ──────────────────────────────────────────
def lgb_custom_metric(y_pred, dtrain):
    """Yarışmanın tam metriğini LightGBM eval'e ekle"""
    y_true = dtrain.get_label()
    
    # Örnek: Competition metric
    score = competition_metric(y_true, y_pred)
    
    return 'competition_metric', score, False  # False = lower is better
```

---

# 29. POST-COMPETITION ANALİZ

## 29.1 Hata Analizi

```python
def error_analysis(y_true, y_pred, X, threshold=0.5, n_examples=20):
    """
    Modelin nerede hata yaptığını anla — sonraki iterasyon için ipucu.
    """
    
    # Binary için
    y_class = (y_pred >= threshold).astype(int)
    
    fp_mask = (y_class == 1) & (y_true == 0)  # False Positive
    fn_mask = (y_class == 0) & (y_true == 1)  # False Negative
    tp_mask = (y_class == 1) & (y_true == 1)  # True Positive
    tn_mask = (y_class == 0) & (y_true == 0)  # True Negative
    
    print(f"TP: {tp_mask.sum()}  |  TN: {tn_mask.sum()}")
    print(f"FP: {fp_mask.sum()}  |  FN: {fn_mask.sum()}")
    
    # FP örneklerini incele
    fp_examples = X[fp_mask].head(n_examples)
    fn_examples = X[fn_mask].head(n_examples)
    
    # FP vs FN feature ortalamaları karşılaştır
    comparison = pd.DataFrame({
        'FP_mean': X[fp_mask].mean(),
        'FN_mean': X[fn_mask].mean(),
        'TP_mean': X[tp_mask].mean(),
        'Overall_mean': X.mean()
    })
    
    # En büyük farklar
    comparison['FP_vs_TP'] = abs(comparison['FP_mean'] - comparison['TP_mean'])
    significant = comparison.nlargest(15, 'FP_vs_TP')
    
    print("\nFP vs TP en büyük farklılıklar:")
    print(significant[['FP_mean', 'FN_mean', 'TP_mean']].to_string())
    
    return fp_mask, fn_mask, comparison

# Prediction güven analizi
def confidence_analysis(y_pred_proba, y_true, n_bins=10):
    """Modelin ne kadar confident olduğunu ve kalibrasyonunu analiz et"""
    
    bins = np.linspace(0, 1, n_bins+1)
    
    for i in range(n_bins):
        mask = (y_pred_proba >= bins[i]) & (y_pred_proba < bins[i+1])
        if mask.sum() == 0:
            continue
        
        actual_rate = y_true[mask].mean()
        predicted_rate = y_pred_proba[mask].mean()
        n = mask.sum()
        
        print(f"[{bins[i]:.1f}-{bins[i+1]:.1f}]  "
              f"n={n:5,}  predicted={predicted_rate:.3f}  actual={actual_rate:.3f}  "
              f"{'✅' if abs(actual_rate-predicted_rate)<0.05 else '⚠️'}")
```

## 29.2 Grand Master Seviyesi Notlar

```python
"""
KAZANAN ÇÖZÜMLERİNDEN ORTAK TEMALAR:

1. FEATURE ENGINEERING her zaman kazandırır
   - Çoğu yarışmada kazanan çözüm, "özel" FE'ye dayanır
   - Model ne kadar güçlü olursa olsun, doğru feature daha kritik
   - Domain knowledge — verinin ne anlama geldiğini anlamak

2. CV STRATEJİSİNE GÜVEN
   - LB'yi takip et ama CV'yi takip et
   - LB'ye göre overfit eden takımlar son hafta çöker

3. ENSEMBLİNG çok modelden değil, çeşitli çözümlerden
   - Aynı modelin 3 farklı random seed'i = gereksiz
   - Farklı approaches (tree + neural + linear) = güçlü ensemble

4. PSEUDOLABELİNG son silah olarak kullan
   - Test'te yüksek confident tahminleri train'e ekle
   - Doğru yapılırsa +0.002-0.005 iyileşme

5. COMMUNITY'ye kat
   - Discussion'da iyi insight paylaş → başkalarından ipucu al
   - Silver/Gold yazarların notebooklarını detaylı incele

6. OVERFITTING TUZAĞI
   - Validation set'ini birden fazla kez kullanma
   - Test prediction'larına bakarak karar verme (leakage!)
   - Nestlenmiş CV ile HPO yap, sonra fresh validation ile test et

7. ENSEMBLİNG ZAMANI
   - İlk 2 haftada ensemble yapma — önce tek model mükemmelleştir
   - Son 1 haftada ensemble ve submission stratejisi
"""
```

---

# 30. HIZLI BAŞVURU KARTLARI

## 30.1 Problem Tipi → Hızlı Karar

```
PROBLEM                 MODEL           CV              METRİK
───────────────────────────────────────────────────────────────────
Binary clf             LGB+XGB+CB      StratifiedKFold AUC/Logloss
Multi-class clf        LGB+CB          StratifiedKFold Logloss/F1
Regression             LGB+XGB         KFold           RMSE/MAE
Ordinal                LGB(reg)+round  KFold           QWK/RMSE
Multi-label            MultiOutput     MultilabelKFold Macro F1
Time series            LGB+lag/roll    WalkForward     RMSE/MAPE
Imbalanced (<1%)       LGB+AnomalyDet  Stratified      AUC/F1
NLP                    TF-IDF+LGB      Stratified      AUC/F1
NLP advanced           DeBERTa+blend   Stratified      AUC/F1
CV                     EfficientNet    Stratified      AUC/F1
Tabular NN             TabNet/MLP      Stratified      AUC/RMSE
───────────────────────────────────────────────────────────────────
```

## 30.2 Metrik → Optimal Karar

```
METRİK          LOSS FUNC         PRED TYPE     POST-PROC
─────────────────────────────────────────────────────────────
AUC-ROC         binary:logistic   proba         threshold tuning
Logloss         binary:logistic   proba(calib)  isotonic calib
F1              binary:logistic   proba         threshold=optimal
RMSE            regression        value         clip extremes
RMSLE           regression        log(y)        expm1
MAE             regression_l1     value         clip
Accuracy        softmax           argmax        —
Macro F1        multiclass        softmax       per-class thr
MAP@K           lambdarank        score         top-K rank
QWK (Kappa)     regression        round→clip    optimize thr
─────────────────────────────────────────────────────────────
```

## 30.3 Checklist — Submission Öncesi

```
□ NaN check: sub[target].isna().sum() == 0
□ Inf check: np.isinf(sub[target]).sum() == 0
□ Range check: sub[target].between(lo, hi).all()
□ Shape check: len(sub) == len(test)
□ ID check: set(sub['id']) == set(test['id'])
□ Distribution check: sub[target].describe()
□ Dosya adı: anlamlı isim (lgb_v3_oof0.8601.csv)
□ MLflow/logger'a kaydet
□ CV skoru not al
□ Submission mesajına parametreleri ekle
```

## 30.4 Feature Engineering Hızlı Referans

```
FEATURE TİPİ      FONKSIYON                         ZAMAN
────────────────────────────────────────────────────────────
Log transform     np.log1p(col.clip(0))             1 dk
Binning           pd.qcut(col, q=10)                1 dk
Rank              col.rank(pct=True)                1 dk
Interaction       col1 * col2, col1 / col2          1 dk
Target enc (CV)   CatBoostEncoder(cols)             5 dk
Frequency enc     col.map(col.value_counts())       1 dk
Group-by mean     groupby(key).transform('mean')    5 dk
Lag features      groupby(id)[col].shift(k)         5 dk
Rolling mean      shift(1).rolling(w).mean()        5 dk
Cyclic date       sin(2π*month/12)                  1 dk
UMAP embed        UMAP(n_components=10)             10 dk
PCA               PCA(n_components=50)              2 dk
────────────────────────────────────────────────────────────
```

---

*Bu ek bölümler ana rehberle birlikte kullanılmalıdır.*
*Ana rehber: Bölüm 1-15 | Ek bölümler: 16-30*