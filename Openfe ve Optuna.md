
```py
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     OpenFE + Optuna + LightGBM Pipeline (İstatistiksel TE Destekli)        ║
║  Manuel FE yerine OpenFE otomatik feature keşfi kullanılır.                ║
║  Target Encoding (fold-safe, Cramér V filtreli) korunur.                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Mimari:
   0.  CFG         — tüm kararlar buradan
   1.  Veri yükleme + birleştirme (duplikasyon kontrolü)
   2.  Target dönüşümü
   3.  Temel temizlik + tip tespiti
   4.  Cramér V testi (TE kolon seçimi için)
   5.  Label Encoding (kategorik → sayısal, OpenFE öncesi şart)
   6.  Target Encoding (fold-safe, Bayesian smoothing)
   7.  OpenFE (otomatik feature keşfi + üretimi)
   8.  Post-filter (yüksek korelasyon + sıfır varyans temizliği)
   9.  Optuna (hyperparameter search)
  10.  Final CV
  11.  Permutation Importance
  12.  Feature Importance görseli
  13.  Sonuç özeti + submission

Kaldırılanlar (OpenFE sayesinde gereksiz):
  · Manuel FE Grup A / Grup B (OpenFE otomatik keşfeder)
  · Normallik, MWU, PBC, VIF, Spearman, Kruskal testleri
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  pip install openfe  (eğer kurulu değilse)
# ═══════════════════════════════════════════════════════════════════════════════

import os
import gc
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from openfe import OpenFE, transform

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
matplotlib.use("Agg")


# ═══════════════════════════════════════════════════════════════════════════════
# 0.  CFG  —  TÜM AYARLAR
# ═══════════════════════════════════════════════════════════════════════════════
CFG = {
    # ── Veri yolları ─────────────────────────────────────────────────────────
    "train_path": "/kaggle/input/playground-series-s5e3/train.csv",
    "test_path":  "/kaggle/input/playground-series-s5e3/test.csv",
    "orig_path":  "/kaggle/input/blastchar/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "target":     "Churn",
    "id_col":     None,            # None → otomatik tespit
    "drop_cols":  [],              # her zaman atılacak ekstra sütunlar

    # ── Ham veri tipi ipuçları ([] → otomatik) ────────────────────────────────
    "num_cols":   ["tenure", "MonthlyCharges", "TotalCharges"],
    "cat_cols":   [],              # boş → otomatik tespit

    # ── OpenFE ───────────────────────────────────────────────────────────────
    "openfe_enabled":       True,
    "openfe_n_features":    30,    # OpenFE'den alınacak max feature
    "openfe_stage":         2,     # 1=hızlı, 2=derin arama
    "openfe_n_jobs":        -1,
    "openfe_n_data_blocks": 8,
    "openfe_verbose":       True,

    # ── Target Encoding ──────────────────────────────────────────────────────
    "te_enabled":       True,
    "te_smoothing":     20,
    "te_cramers_min":   0.10,      # en az bu V'ye sahip kolonlara uygula
    "te_n_splits":      5,

    # ── Post-filter ──────────────────────────────────────────────────────────
    "high_corr_threshold": 0.95,
    "min_variance":        0.01,

    # ── Optuna ───────────────────────────────────────────────────────────────
    "optuna_enabled":    True,
    "optuna_n_trials":   60,
    "optuna_timeout":    6000,     # saniye
    "optuna_cv_splits":  3,

    # ── Final Model ──────────────────────────────────────────────────────────
    "final_cv_splits":   5,
    "seed":              42,
    "early_stopping":    150,
    "base_lgbm_params": {
        "objective":        "binary",
        "metric":           "auc",
        "boosting_type":    "gbdt",
        "learning_rate":    0.03,
        "num_leaves":       63,
        "min_child_samples": 50,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "n_jobs":           -1,
        "verbose":          -1,
        "random_state":     42,
        "is_unbalance":     True,
    },

    # ── Çıktılar ─────────────────────────────────────────────────────────────
    "output_dir":    ".",
    "plot_figures":  True,
}

TARGET = CFG["target"]


# ═══════════════════════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════════════════════
def log(msg, level="INFO"):
    prefix = {"INFO": "  ·", "STEP": "\n▸", "WARN": "  ⚠", "OK": "  ✓"}
    print(f"{prefix.get(level, '  ')} {msg}")


def cramers_v_score(x, y_arr):
    """Target Encoding kolon seçimi için Cramér V hesabı."""
    ct = pd.crosstab(x, y_arr)
    chi2, p, _, _ = chi2_contingency(ct)
    n = ct.values.sum()
    k = min(ct.shape) - 1
    v = np.sqrt(chi2 / (n * k)) if (n * k) > 0 else 0.0
    return float(v), float(p)


def auto_detect_id(df):
    """ID sütununu otomatik tespit et."""
    for c in df.columns:
        if c.lower() in ("id", "customerid", "customer_id"):
            return c
    for c in df.columns:
        if c == TARGET:
            continue
        if pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() == len(df):
            return c
    return None


def remove_high_corr(df, threshold=0.95):
    """Spearman korelasyonu threshold üstü olan feature çiftlerinden birini atar."""
    corr = df.corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    return to_drop


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  VERİ YÜKLEME + BİRLEŞTİRME
# ═══════════════════════════════════════════════════════════════════════════════
log("Veri yükleme", "STEP")

train = pd.read_csv(CFG["train_path"])
test  = pd.read_csv(CFG["test_path"])
log(f"train: {train.shape}  |  test: {test.shape}")

# Orijinal veri birleştirme (duplikasyon kontrolü ile)
if CFG["orig_path"] and os.path.exists(str(CFG["orig_path"])):
    orig = pd.read_csv(CFG["orig_path"])
    log(f"orig: {orig.shape}")
    
    # Kolon eşleştirme
    if set(train.columns) == set(orig.columns):
        orig = orig[train.columns]
    else:
        shared = [c for c in train.columns if c in orig.columns]
        orig = orig[shared]
        train_temp = train[shared].copy()
        train_temp[TARGET] = train[TARGET] if TARGET in train.columns else None
        train = train_temp
    
    # Duplikasyon kontrolü
    before_len = len(train)
    combined = pd.concat([train, orig], axis=0, ignore_index=True)
    dup_cols = [c for c in combined.columns if c != TARGET]
    combined.drop_duplicates(subset=dup_cols, keep="first", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    train = combined
    
    log(f"orig birleştirildi (duplikasyon temizlendi) → train: {train.shape}", "OK")
    log(f"  Kaldırılan duplike satır: {before_len + len(orig) - len(train)}")
else:
    log("orig yok / None — sadece train kullanılıyor.")

# ID sütunu
id_col = CFG["id_col"] or auto_detect_id(test)
test_ids = test[id_col].copy() if id_col and id_col in test.columns \
           else pd.RangeIndex(len(test))
log(f"ID sütunu: {id_col}")

# Gereksiz sütunları çıkar
drop = ([id_col] if id_col else []) + CFG["drop_cols"]
train.drop(columns=[c for c in drop if c in train.columns], inplace=True)
test.drop(columns=[c for c in drop if c in test.columns], inplace=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  TARGET DÖNÜŞÜMÜ
# ═══════════════════════════════════════════════════════════════════════════════
log("Target dönüşümü", "STEP")

y_raw = train[TARGET].copy()

y = y_raw.astype(str).str.strip().map(
    {"Yes": 1, "No": 0, "1": 1, "0": 0, "True": 1, "False": 0}
)
if y.isnull().any():
    y = pd.to_numeric(y_raw, errors="coerce")
if y.isnull().any():
    raise ValueError(f"Dönüştürülemeyen target: {y_raw[y.isnull()].unique()}")
y = y.reset_index(drop=True).astype(np.int8)

# Target sütununu train'den çıkar
if TARGET in train.columns:
    train.drop(columns=[TARGET], inplace=True)

train.reset_index(drop=True, inplace=True)
log(f"Dağılım: {dict(y.value_counts().sort_index())}  oran={y.mean():.3f}", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  TEMEL TEMİZLİK + TİP TESPİTİ
# ═══════════════════════════════════════════════════════════════════════════════
log("Temel temizlik", "STEP")

num_cols = CFG["num_cols"] if CFG["num_cols"] else \
    list(train.select_dtypes(include=["number"]).columns)
cat_cols = CFG["cat_cols"] if CFG["cat_cols"] else \
    [c for c in train.columns if c not in num_cols]
num_cols = [c for c in num_cols if c in train.columns]
cat_cols = [c for c in cat_cols if c in train.columns]

# Train istatistiklerini sakla (leakage önleme)
train_medians = {}
for c in num_cols:
    if c in train.columns:
        num_series = pd.to_numeric(train[c], errors="coerce")
        train_medians[c] = num_series.median()


def clean(df, num_c, cat_c, medians, is_train=True):
    """Veri temizleme — test için train medyanları kullanılır."""
    df = df.copy()
    for c in num_c:
        if c in df.columns:
            num_series = pd.to_numeric(df[c], errors="coerce")
            if is_train:
                med_val = num_series.median()
            else:
                med_val = medians.get(c, 0.0)
            df[c] = num_series.fillna(med_val if pd.notna(med_val) else 0.0)
    for c in cat_c:
        if c in df.columns:
            df[c] = df[c].fillna("Missing").astype(str).str.strip()
    return df


train = clean(train, num_cols, cat_cols, train_medians, is_train=True)
test  = clean(test,  num_cols, cat_cols, train_medians, is_train=False)
log(f"Nümerik: {len(num_cols)}  |  Kategorik: {len(cat_cols)}", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  CRAMÉR V TESTİ (TE Kolon Seçimi İçin)
# ═══════════════════════════════════════════════════════════════════════════════
log("Cramér V testi (TE kolon seçimi)", "STEP")

cramers_v_dict = {}
for col in cat_cols:
    if col in train.columns:
        v, p = cramers_v_score(train[col], y)
        cramers_v_dict[col] = {"v": v, "p": p}
        log(f"  {col:25s} Cramér V={v:.3f}  p={p:.3e}")

# TE için anlamlı kolonları belirle
te_candidate_cols = [
    c for c, d in cramers_v_dict.items() 
    if d["v"] >= CFG["te_cramers_min"] and d["p"] < 0.05
]
log(f"TE adayları (V >= {CFG['te_cramers_min']}): {te_candidate_cols}", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  LABEL ENCODING (OpenFE sayısal girdi ister)
# ═══════════════════════════════════════════════════════════════════════════════
log("Label Encoding", "STEP")

obj_cols = train.select_dtypes(include=["object"]).columns.tolist()
le_dict = {}
for col in obj_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))
    le_dict[col] = le

log(f"Label Encoded: {obj_cols}", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  TARGET ENCODING (fold-safe, Bayesian smoothing)
# ═══════════════════════════════════════════════════════════════════════════════
log("Target Encoding", "STEP")

te_cols = [c for c in te_candidate_cols if c in train.columns] if CFG["te_enabled"] else []
log(f"TE uygulanacak kolonlar: {te_cols}")

te_train_df = pd.DataFrame(index=train.index)
te_test_df  = pd.DataFrame(index=test.index)
global_mean = float(y.mean())
sm = CFG["te_smoothing"]

if te_cols:
    skf_te = StratifiedKFold(n_splits=CFG["te_n_splits"], shuffle=True, random_state=CFG["seed"])
    
    for col in te_cols:
        col_name = f"{col}_te"
        te_train_df[col_name] = np.nan
        
        # Fold-safe encoding (train)
        for fold_idx, (tr_idx, val_idx) in enumerate(skf_te.split(train, y)):
            tr_vals = train[col].iloc[tr_idx].values
            tr_y    = y.iloc[tr_idx].values
            
            agg = pd.DataFrame({"col": tr_vals, "y": tr_y})
            st = agg.groupby("col")["y"].agg(["sum", "count"])
            st["te"] = (st["sum"] + sm * global_mean) / (st["count"] + sm)
            
            te_train_df.iloc[val_idx, te_train_df.columns.get_loc(col_name)] = \
                train[col].iloc[val_idx].map(st["te"]).values
        
        te_train_df[col_name].fillna(global_mean, inplace=True)
        
        # Full encoding (test)
        full_agg = pd.DataFrame({"col": train[col].values, "y": y.values})
        full_st = full_agg.groupby("col")["y"].agg(["sum", "count"])
        full_st["te"] = (full_st["sum"] + sm * global_mean) / (full_st["count"] + sm)
        te_test_df[col_name] = test[col].map(full_st["te"]).fillna(global_mean)
        
        log(f"  {col:25s} → {col_name}  (V={cramers_v_dict[col]['v']:.3f})")
    
    train = pd.concat([train, te_train_df], axis=1)
    test  = pd.concat([test,  te_test_df],  axis=1)

log(f"TE sonrası: {train.shape[1]} feature  (TE kolon: {len(te_cols)})", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  OpenFE — OTOMATİK FEATURE KEŞFİ
# ═══════════════════════════════════════════════════════════════════════════════
log("OpenFE otomatik feature engineering", "STEP")

ofe_feature_names = []

if CFG["openfe_enabled"]:
    ofe = OpenFE()
    
    log("OpenFE fit başlıyor …")
    ofe_features = ofe.fit(
        data=train.copy(),
        label=y.copy(),
        task="classification",
        n_jobs=CFG["openfe_n_jobs"],
        n_data_blocks=CFG["openfe_n_data_blocks"],
        stage=CFG["openfe_stage"],
        verbose=CFG["openfe_verbose"],
    )
    
    # En iyi N feature'ı seç
    n_keep = min(CFG["openfe_n_features"], len(ofe_features))
    top_ofe = ofe_features[:n_keep]
    log(f"OpenFE {len(ofe_features)} feature buldu → en iyi {n_keep} alınıyor")
    
    if top_ofe:
        # Dönüştür (orijinal sütunlar korunur + yeni sütunlar eklenir)
        cols_before = list(train.columns)
        train, test = transform(train, test, top_ofe, n_jobs=CFG["openfe_n_jobs"])
        ofe_feature_names = [c for c in train.columns if c not in cols_before]
        
        # NaN / inf temizliği
        train[ofe_feature_names] = (train[ofe_feature_names]
                                    .replace([np.inf, -np.inf], np.nan)
                                    .fillna(0))
        test[ofe_feature_names]  = (test[ofe_feature_names]
                                    .replace([np.inf, -np.inf], np.nan)
                                    .fillna(0))
        
        log(f"OpenFE eklenen: {len(ofe_feature_names)} feature", "OK")
        for i, fn in enumerate(ofe_feature_names[:10], 1):
            log(f"  {i:>2}. {fn}")
        if len(ofe_feature_names) > 10:
            log(f"  ... ve {len(ofe_feature_names) - 10} tane daha")
    else:
        log("OpenFE anlamlı feature bulamadı — ham veriyle devam", "WARN")
else:
    log("OpenFE devre dışı (CFG)", "WARN")


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  POST-FILTER — Yüksek korelasyon + sıfır varyans temizliği
# ═══════════════════════════════════════════════════════════════════════════════
log("Post-filter (korelasyon + varyans)", "STEP")

shape_before = train.shape[1]

# 8a. Sıfır (çok düşük) varyans
vt = VarianceThreshold(threshold=CFG["min_variance"])
try:
    vt.fit(train.fillna(0))
    low_var = [c for c, keep in zip(train.columns, vt.get_support()) if not keep]
    if low_var:
        log(f"Düşük varyans → {len(low_var)} feature çıkarıldı: {low_var[:10]}", "WARN")
        train.drop(columns=low_var, inplace=True, errors="ignore")
        test.drop(columns=low_var,  inplace=True, errors="ignore")
except Exception as e:
    log(f"Varyans filtresi hatası: {e}", "WARN")

# 8b. Yüksek korelasyon
high_corr_drop = remove_high_corr(train.fillna(0), CFG["high_corr_threshold"])
if high_corr_drop:
    log(f"Yüksek korelasyon (>{CFG['high_corr_threshold']}) → "
        f"{len(high_corr_drop)} feature çıkarıldı: {high_corr_drop[:10]}", "WARN")
    train.drop(columns=high_corr_drop, inplace=True, errors="ignore")
    test.drop(columns=high_corr_drop,  inplace=True, errors="ignore")

# Takip listelerini güncelle
ofe_feature_names = [c for c in ofe_feature_names if c in train.columns]
FEATURES = train.columns.tolist()

log(f"Filter: {shape_before} → {len(FEATURES)} feature", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  OPTUNA — LightGBM Hyperparameter Search
# ═══════════════════════════════════════════════════════════════════════════════
log("Optuna hyperparameter arama", "STEP")


def lgb_cv_fast(params, X, y_arr, n_splits=3):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CFG["seed"])
    scores = []
    for tr_idx, val_idx in skf.split(X, y_arr):
        d_tr  = lgb.Dataset(X.iloc[tr_idx],  label=y_arr.iloc[tr_idx])
        d_val = lgb.Dataset(X.iloc[val_idx], label=y_arr.iloc[val_idx], reference=d_tr)
        m = lgb.train(params, d_tr, num_boost_round=800,
                      valid_sets=[d_val],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
        scores.append(roc_auc_score(y_arr.iloc[val_idx], m.predict(X.iloc[val_idx])))
    return float(np.mean(scores))


def objective(trial):
    p = {
        **CFG["base_lgbm_params"],
        "num_leaves":        trial.suggest_int("num_leaves", 20, 255),
        "max_depth":         trial.suggest_int("max_depth", 3, 12),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 150),
        "subsample":         trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),
        "bagging_freq":      trial.suggest_int("bagging_freq", 1, 7),
    }
    return lgb_cv_fast(p, train, y, n_splits=CFG["optuna_cv_splits"])


best_params = dict(CFG["base_lgbm_params"])

if CFG["optuna_enabled"]:
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=CFG["seed"]),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
    )
    study.optimize(
        objective,
        n_trials=CFG["optuna_n_trials"],
        timeout=CFG["optuna_timeout"],
        show_progress_bar=True
    )
    log(f"Optuna bitti. En iyi AUC: {study.best_value:.5f}", "OK")
    log(f"En iyi params: {study.best_params}")
    best_params = {**CFG["base_lgbm_params"], **study.best_params}
    
    if CFG["plot_figures"]:
        try:
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.tight_layout()
            plt.savefig(os.path.join(CFG["output_dir"], "optuna_history.png"), dpi=100)
            plt.close()
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.tight_layout()
            plt.savefig(os.path.join(CFG["output_dir"], "optuna_params.png"), dpi=100)
            plt.close()
        except Exception as e:
            log(f"Optuna görseli: {e}", "WARN")
else:
    log("Optuna devre dışı — base params kullanılıyor.", "WARN")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. FİNAL CROSS VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
log("Final Cross Validation", "STEP")

skf_final = StratifiedKFold(
    n_splits=CFG["final_cv_splits"], shuffle=True, random_state=CFG["seed"]
)
oof_preds   = np.zeros(len(train))
test_preds  = np.zeros(len(test))
feat_imp_df = pd.DataFrame()
fold_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf_final.split(train, y), 1):
    X_tr, X_val = train.iloc[tr_idx], train.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
    d_tr  = lgb.Dataset(X_tr, label=y_tr)
    d_val = lgb.Dataset(X_val, label=y_val, reference=d_tr)
    
    model = lgb.train(
        best_params, d_tr,
        num_boost_round=5000,
        valid_sets=[d_val],
        callbacks=[lgb.early_stopping(CFG["early_stopping"], verbose=False),
                   lgb.log_evaluation(500)]
    )
    
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(test) / CFG["final_cv_splits"]
    
    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    fold_scores.append(fold_auc)
    log(f"  Fold {fold}: AUC={fold_auc:.5f}  best_iter={model.best_iteration}")
    
    fi = pd.DataFrame({
        "feature": FEATURES,
        "gain":    model.feature_importance("gain"),
        "split":   model.feature_importance("split"),
        "fold":    fold
    })
    feat_imp_df = pd.concat([feat_imp_df, fi])
    gc.collect()

oof_auc = roc_auc_score(y, oof_preds)
log(f"Ortalama: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}", "OK")
log(f"Overall OOF AUC: {oof_auc:.5f}", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. PERMUTATION IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
log("Permutation Importance (LogReg üzerinde)", "STEP")

perm_df = pd.DataFrame()
try:
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=500, random_state=42))
    ])
    pipe.fit(train.fillna(0), y)
    perm = permutation_importance(
        pipe, train.fillna(0), y,
        n_repeats=5, random_state=42, scoring="roc_auc"
    )
    perm_df = pd.DataFrame({
        "feature":   FEATURES,
        "perm_mean": perm.importances_mean,
        "perm_std":  perm.importances_std
    }).sort_values("perm_mean", ascending=False)
    
    log("Top 10 Permutation Importance:")
    for _, row in perm_df.head(10).iterrows():
        log(f"  {row['feature']:35s} {row['perm_mean']:+.5f} ± {row['perm_std']:.5f}")
except Exception as e:
    log(f"Permutation hesaplanamadı: {e}", "WARN")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. FEATURE IMPORTANCE GÖRSELİ
# ═══════════════════════════════════════════════════════════════════════════════
if CFG["plot_figures"]:
    fi_mean = feat_imp_df.groupby("feature")["gain"].mean().sort_values(ascending=True)
    
    TE_SET  = {f"{c}_te" for c in te_cols}
    OFE_SET = set(ofe_feature_names)
    
    def get_color(feat):
        if feat in TE_SET:   return "#7F77DD"   # mor — Target Encoding
        if feat in OFE_SET:  return "#1D9E75"   # yeşil — OpenFE
        return "#888780"                         # gri — Ham feature
    
    fig, axes = plt.subplots(1, 2, figsize=(22, max(8, int(len(fi_mean) * 0.3))))
    
    axes[0].barh(fi_mean.index, fi_mean.values,
                 color=[get_color(f) for f in fi_mean.index])
    axes[0].set_title("Feature Importance — Gain (tümü)", fontsize=11)
    axes[0].tick_params(labelsize=7)
    
    top25 = fi_mean.tail(25)
    axes[1].barh(top25.index, top25.values,
                 color=[get_color(f) for f in top25.index])
    axes[1].set_title("Top 25", fontsize=11)
    
    from matplotlib.patches import Patch
    axes[1].legend(handles=[
        Patch(color="#1D9E75", label="OpenFE — otomatik keşif"),
        Patch(color="#7F77DD", label="Target Encoding"),
        Patch(color="#888780", label="Ham / Label Encoded"),
    ], loc="lower right", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CFG["output_dir"], "feature_importance.png"),
                dpi=120, bbox_inches="tight")
    plt.close()
    log("feature_importance.png kaydedildi", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 13. SONUÇ ÖZETİ + SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  SONUÇ ÖZETİ")
print("═" * 60)
print(f"  Toplam feature        : {len(FEATURES)}")
print(f"    ├─ Ham / LE         : {len(FEATURES) - len(ofe_feature_names) - len(te_cols)}")
print(f"    ├─ Target Encoding  : {len(te_cols)}")
print(f"    └─ OpenFE           : {len(ofe_feature_names)}")
print(f"  Optuna trials         : {CFG['optuna_n_trials'] if CFG['optuna_enabled'] else 'devre dışı'}")
print(f"  Fold AUC'ler          : {[f'{s:.5f}' for s in fold_scores]}")
print(f"  Ortalama AUC          : {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")
print(f"  Overall OOF AUC       : {oof_auc:.5f}")
print("═" * 60)

# Best params kaydet
with open(os.path.join(CFG["output_dir"], "best_params.json"), "w") as f:
    json.dump({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
               for k, v in best_params.items()}, f, indent=2)
log("best_params.json kaydedildi", "OK")

# Submission
submission = pd.DataFrame({
    (id_col if id_col else "id"): test_ids,
    TARGET: test_preds
})
sub_path = os.path.join(CFG["output_dir"], "submission.csv")
submission.to_csv(sub_path, index=False)
log(f"submission.csv → {submission.shape}", "OK")
print(submission[TARGET].describe().to_string())

```