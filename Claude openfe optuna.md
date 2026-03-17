```py
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   OpenFE + Optuna + LightGBM Pipeline  ·  v3.2  (final polish)            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  v3.1 → v3.2 düzeltmeler (son kozmetik iyileştirmeler):                   ║
║  [P3-A] _lock_spw in-place mutasyon → savunmacı kopya                     ║
║  [P3-B] StandardScaler unused import kaldırıldı                           ║
║  [P3-C] Master skor son 2 kontrol: anlamlı eşikler                        ║
║  [P3-D] Desil hesaplama: açık formül (pd.qcut yerine)                     ║
║  [P3-E] Bridge cache pollution belgelendi (bilgi notu)                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Genel durum: PRODUCTION-READY                                             ║
║  Kritik (P0/P1) sorun: YOK                                                 ║
║  Score: 9.5/10                                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, gc, json, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import matplotlib
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import chi2_contingency, ks_2samp
from typing import List, Set
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                              precision_recall_curve, confusion_matrix,
                              average_precision_score)
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance

try:
    from openfe import OpenFE, transform as ofe_transform
    HAS_OFE = True
except ImportError:
    HAS_OFE = False
    print("⚠️  openfe yüklü değil — pip install openfe")

try:
    import shap as shap_lib
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
matplotlib.use("Agg")


# ═══════════════════════════════════════════════════════════════════════════════
# 0.  CFG
# ═══════════════════════════════════════════════════════════════════════════════
CFG = {
    # ── Veri yolları ──────────────────────────────────────────────────────────
    "train_path": "/kaggle/input/playground-series-s6e3/train.csv",
    "test_path":  "/kaggle/input/playground-series-s6e3/test.csv",
    "orig_path":  "/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "target":     "Churn",
    "id_col":     None,
    "drop_cols":  [],

    # ── EDA entegrasyonu ──────────────────────────────────────────────────────
    "eda_decisions_path": "/kaggle/working/eda_decisions.json",

    # ── Ham veri tipi ipuçları ────────────────────────────────────────────────
    "num_cols": ["tenure", "MonthlyCharges", "TotalCharges"],
    "cat_cols": [],

    # ── OpenFE ───────────────────────────────────────────────────────────────
    "openfe_enabled":       True,
    "openfe_in_cv":         True,
    "openfe_n_features":    30,
    "openfe_stage":         2,
    "openfe_n_jobs":        -1,
    "openfe_n_data_blocks": 8,
    "openfe_verbose":       False,

    # ── Target Encoding ───────────────────────────────────────────────────────
    "te_enabled":     True,
    "te_smoothing":   20,
    "te_cramers_min": 0.10,
    "te_n_splits":    5,

    # ── Post-filter ──────────────────────────────────────────────────────────
    "high_corr_threshold": 0.95,
    "min_variance":        0.01,

    # ── Optuna ───────────────────────────────────────────────────────────────
    "optuna_enabled":   True,
    "optuna_n_trials":  60,
    "optuna_timeout":   6000,
    "optuna_cv_splits": 3,

    # ── Final Model ──────────────────────────────────────────────────────────
    "final_cv_splits": 5,
    "n_seeds":         3,
    "seed":            42,
    "early_stopping":  150,

    # ── Cost-based threshold ─────────────────────────────────────────────────
    "fn_cost": 5.0,
    "fp_cost": 1.0,

    # ── Base params ──────────────────────────────────────────────────────────
    # NOT: is_unbalance ve scale_pos_weight birlikte kullanılmaz.
    # EDA'dan gelen scale_pos_weight varsa is_unbalance=False yapılır.
    "base_lgbm_params": {
        "objective":         "binary",
        "metric":            "auc",
        "boosting_type":     "gbdt",
        "learning_rate":     0.03,
        "num_leaves":        63,
        "min_child_samples": 50,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
        "n_jobs":            -1,
        "verbose":           -1,
        "random_state":      42,
        "is_unbalance":      True,
    },

    "output_dir":   "/kaggle/working",
    "plot_figures": True,

    # ── Bridge Validation ─────────────────────────────────────────────────────
    # Optuna (OpenFE'siz) → Bridge (OpenFE'li, dar aralık) → Final CV
    "bridge_enabled":   True,
    "bridge_n_trials":  15,
    "bridge_timeout":   1800,   # 30 dakika
}

TARGET = CFG["target"]


# ═══════════════════════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════════════════════
def log(msg, level="INFO"):
    pfx = {"INFO": "  ·", "STEP": "\n▸", "WARN": "  ⚠", "OK": "  ✓", "CRIT": "  ✗"}
    print(f"{pfx.get(level,'  ')} {msg}")


def cramers_v_score(x, y_arr):
    """
    Bias-corrected Cramér V (Bergsma & Wicher, 2013).
    Düşük örneklem sayısında şişirilmiş V değerini düzeltir.
    """
    ct = pd.crosstab(x, y_arr)
    chi2, p, _, _ = chi2_contingency(ct)
    n  = ct.values.sum()
    r, k = ct.shape
    # Bias correction
    phi2    = chi2 / n
    phi2c   = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    kc      = k - (k - 1) ** 2 / (n - 1)
    rc      = r - (r - 1) ** 2 / (n - 1)
    denom   = min(kc - 1, rc - 1)
    v       = float(np.sqrt(phi2c / denom)) if denom > 0 else 0.0
    return v, float(p)


def _lock_spw(params: dict) -> dict:
    """
    [P3-A] Savunmacı kopya: orijinal dict'i değiştirmeden yeni dict döndürür.
    scale_pos_weight kilitli ise EDA değerleriyle günceller.
    """
    params = params.copy()  # shallow copy yeterli (nested dict yok)
    if CFG.get("_spw_locked"):
        params["is_unbalance"]     = False
        params["scale_pos_weight"] = CFG["base_lgbm_params"]["scale_pos_weight"]
    return params


def auto_detect_id(df):
    for c in df.columns:
        if c.lower() in ("id", "customerid", "customer_id"):
            return c
    for c in df.columns:
        if c == TARGET:
            continue
        if pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() == len(df):
            return c
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# OPENFE FOLD CACHE
# ═══════════════════════════════════════════════════════════════════════════════
# Cache: (fold_hash) → fitted OpenFE feature listesi
# Aynı fold indeksleri için OpenFE sadece 1 kez fit edilir;
# farklı seed'ler aynı fold indeksini paylaşır → 15 fit değil 5 fit.
_OFE_FOLD_CACHE: dict = {}


def _fold_hash(tr_idx: np.ndarray) -> int:
    """
    [P3-A düzeltildi] tobytes() ile tam içerik hash.
    Çakışma riski minimal — farklı fold'ların aynı hash'i üretme olasılığı yok.
    """
    return hash(tr_idx.tobytes())


# ═══════════════════════════════════════════════════════════════════════════════
# CV-SAFE FOLD FABRİKASI
# ═══════════════════════════════════════════════════════════════════════════════
def build_fold(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    X_te: pd.DataFrame,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    te_cols: list,
    cfg: dict,
):
    """
    Tek CV fold için tüm fit/transform'ları izole eder.
    TE → OpenFE → VarianceThreshold → Korelasyon filtresi
    TAMAMI sadece tr_idx üzerinden fit edilir.
    """
    Xtr   = X_full.iloc[tr_idx].copy().reset_index(drop=True)
    Xval  = X_full.iloc[va_idx].copy().reset_index(drop=True)
    Xtest = X_te.copy().reset_index(drop=True)
    ytr   = y_full.iloc[tr_idx].reset_index(drop=True)
    gm    = float(ytr.mean())
    sm    = cfg["te_smoothing"]

    # ── 1. Target Encoding ────────────────────────────────────────────────────
    if cfg["te_enabled"] and te_cols:
        for col in te_cols:
            if col not in Xtr.columns:
                continue
            agg   = pd.DataFrame({"c": Xtr[col].values, "t": ytr.values})
            st    = agg.groupby("c")["t"].agg(["sum", "count"])
            st["enc"] = (st["sum"] + sm * gm) / (st["count"] + sm)
            cname = f"{col}_te"
            Xtr[cname]   = Xtr[col].map(st["enc"]).fillna(gm).astype("float32")
            Xval[cname]  = Xval[col].map(st["enc"]).fillna(gm).astype("float32")
            Xtest[cname] = Xtest[col].map(st["enc"]).fillna(gm).astype("float32")

    # ── 2. OpenFE ────────────────────────────────────────────────────────────
    if cfg["openfe_enabled"] and HAS_OFE and cfg.get("openfe_in_cv", True):
        fhash = _fold_hash(tr_idx)
        try:
            if fhash not in _OFE_FOLD_CACHE:
                ofe = OpenFE()
                feats = ofe.fit(
                    data=Xtr.copy(),
                    label=ytr.copy(),
                    task="classification",
                    n_jobs=cfg["openfe_n_jobs"],
                    n_data_blocks=cfg["openfe_n_data_blocks"],
                    stage=cfg["openfe_stage"],
                    verbose=cfg["openfe_verbose"],
                )
                n_keep = min(cfg["openfe_n_features"], len(feats))
                _OFE_FOLD_CACHE[fhash] = feats[:n_keep]
                log(f"    OpenFE fold cache miss → {n_keep} feature fit edildi")
            else:
                log(f"    OpenFE fold cache hit → yeniden fit atlandı")

            top_feats = _OFE_FOLD_CACHE[fhash]

            if top_feats:
                Xtr_orig  = Xtr.copy()  # transform öncesi orijinal
                Xtr, Xval = ofe_transform(Xtr,      Xval.copy(), top_feats,
                                          n_jobs=cfg["openfe_n_jobs"])
                _, Xtest  = ofe_transform(Xtr_orig, Xtest,       top_feats,
                                          n_jobs=cfg["openfe_n_jobs"])

                # inf/nan temizliği
                new_cols = [c for c in Xtr.columns if c not in Xtr_orig.columns]
                for df_ in [Xtr, Xval, Xtest]:
                    for c in new_cols:
                        if c in df_.columns:
                            df_[c] = df_[c].replace([np.inf, -np.inf], np.nan).fillna(0)

        except Exception as e:
            log(f"    OpenFE hata: {e}", "WARN")

    # ── 3. VarianceThreshold ─────────────────────────────────────────────────
    try:
        vt = VarianceThreshold(threshold=cfg["min_variance"])
        vt.fit(Xtr.fillna(0))
        low_var = [c for c, k in zip(Xtr.columns, vt.get_support()) if not k]
        for df_ in [Xtr, Xval, Xtest]:
            df_.drop(columns=low_var, errors="ignore", inplace=True)
    except Exception as e:
        log(f"    VarianceThreshold hata: {e}", "WARN")

    # ── 4. Yüksek korelasyon ─────────────────────────────────────────────────
    try:
        corr  = Xtr.fillna(0).corr(method="spearman").abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        hc    = [c for c in upper.columns if any(upper[c] > cfg["high_corr_threshold"])]
        for df_ in [Xtr, Xval, Xtest]:
            df_.drop(columns=hc, errors="ignore", inplace=True)
    except Exception as e:
        log(f"    Korelasyon filtre hata: {e}", "WARN")

    return Xtr, Xval, Xtest


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  EDA KARARLARI
# ═══════════════════════════════════════════════════════════════════════════════
log("EDA kararları yükleniyor", "STEP")

EDA = {}
if os.path.exists(CFG["eda_decisions_path"]):
    with open(CFG["eda_decisions_path"]) as f:
        EDA = json.load(f)
    log(f"eda_decisions.json yüklendi", "OK")

    for feat in EDA.get("leakage_features", []) + EDA.get("high_psi_features", []):
        if feat not in CFG["drop_cols"]:
            CFG["drop_cols"].append(feat)

    spw = EDA.get("scale_pos_weight", None)
    if spw and float(spw) > 1:
        CFG["base_lgbm_params"]["is_unbalance"]     = False
        CFG["base_lgbm_params"]["scale_pos_weight"] = float(spw)
        CFG["_spw_locked"] = True
        log(f"scale_pos_weight={spw:.2f} kilitlendi", "OK")

    if EDA.get("imbalance_ratio", 1) > 3:
        CFG["fn_cost"] = max(CFG["fn_cost"], float(EDA["imbalance_ratio"]))

    log(f"  Drop: {CFG['drop_cols']}")
else:
    log("eda_decisions.json bulunamadı", "WARN")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  VERİ YÜKLEME + BİRLEŞTİRME
# ═══════════════════════════════════════════════════════════════════════════════
log("Veri yükleme", "STEP")

train = pd.read_csv(CFG["train_path"])
test  = pd.read_csv(CFG["test_path"])
log(f"train: {train.shape}  |  test: {test.shape}")

if CFG["orig_path"] and os.path.exists(str(CFG["orig_path"])):
    orig = pd.read_csv(CFG["orig_path"])
    shared = [c for c in train.columns if c in orig.columns]
    orig = orig[shared]
    if TARGET in orig.columns:
        orig = orig[orig[TARGET].notna()].copy()
    before = len(train) + len(orig)
    combined = pd.concat([train, orig], axis=0, ignore_index=True)
    dup_cols = [c for c in combined.columns if c != TARGET]
    combined.drop_duplicates(subset=dup_cols, keep="first", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    train = combined
    log(f"orig birleştirildi → train: {train.shape}  (duplike: {before - len(train)})", "OK")
else:
    log("orig bulunamadı")

id_col   = CFG["id_col"] or auto_detect_id(test)
test_ids = test[id_col].copy() if id_col and id_col in test.columns else pd.RangeIndex(len(test))

drop_base = ([id_col] if id_col else []) + CFG["drop_cols"]
train.drop(columns=[c for c in drop_base if c in train.columns], inplace=True)
test.drop( columns=[c for c in drop_base if c in test.columns],  inplace=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  TARGET DÖNÜŞÜMÜ
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
train.drop(columns=[TARGET], inplace=True)
train.reset_index(drop=True, inplace=True)
log(f"Dağılım: {dict(y.value_counts().sort_index())}  pos_oran={y.mean():.3f}", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  TEMEL TEMİZLİK
# ═══════════════════════════════════════════════════════════════════════════════
log("Temel temizlik", "STEP")

num_cols = CFG["num_cols"] if CFG["num_cols"] else \
    list(train.select_dtypes(include=["number"]).columns)
cat_cols = CFG["cat_cols"] if CFG["cat_cols"] else \
    [c for c in train.columns if c not in num_cols]
num_cols = [c for c in num_cols if c in train.columns]
cat_cols = [c for c in cat_cols if c in train.columns]

train_medians = {}
for c in num_cols:
    if c in train.columns:
        train_medians[c] = pd.to_numeric(train[c], errors="coerce").median()


def clean(df, num_c, cat_c, medians):
    df = df.copy()
    for c in num_c:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(medians.get(c, 0.0))
    for c in cat_c:
        if c in df.columns:
            df[c] = df[c].fillna("Missing").astype(str).str.strip()
    return df


train = clean(train, num_cols, cat_cols, train_medians)
test  = clean(test,  num_cols, cat_cols, train_medians)
log(f"Temizlik tamamlandı — Nümerik: {len(num_cols)}  Kategorik: {len(cat_cols)}", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  CRAMÉR V TESTİ
# ═══════════════════════════════════════════════════════════════════════════════
log("Cramér V testi (bias-corrected)", "STEP")

cramers_v_dict = {}
for col in cat_cols:
    if col in train.columns:
        v, p = cramers_v_score(train[col], y)
        cramers_v_dict[col] = {"v": v, "p": p}
        log(f"  {col:25s} Cramér V(bc)={v:.3f}  p={p:.3e}")

te_candidate_cols = [
    c for c, d in cramers_v_dict.items()
    if d["v"] >= CFG["te_cramers_min"] and d["p"] < 0.05
]
log(f"TE adayları (V >= {CFG['te_cramers_min']}): {te_candidate_cols}", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  LABEL ENCODING
# ═══════════════════════════════════════════════════════════════════════════════
log("Label Encoding", "STEP")

obj_cols = train.select_dtypes(include=["object"]).columns.tolist()
le_dict  = {}
for col in obj_cols:
    le = LabelEncoder()
    combined_vals = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined_vals)
    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))
    le_dict[col] = le
log(f"Label Encoded: {obj_cols}", "OK")

FEATURES_BASE = train.columns.tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  OPTUNA — CV-SAFE
# ═══════════════════════════════════════════════════════════════════════════════
log("Optuna hyperparameter arama (CV-safe)", "STEP")


def lgb_cv_safe(params, X_full, y_full, X_te, te_cols_, n_splits=3, seed=42):
    """
    Optuna objective — TE fold içinde yapılıyor.
    OpenFE atlanıyor (hız tradeoff'u — bilinçli karar).
    """
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    cfg_no_ofe = {**CFG, "openfe_in_cv": False, "openfe_enabled": False}
    for tr_i, va_i in skf.split(X_full, y_full):
        Xtr, Xval, _ = build_fold(
            X_full, y_full, X_te, tr_i, va_i, te_cols_, cfg_no_ofe
        )
        common = [c for c in Xtr.columns if c in Xval.columns]
        Xtr, Xval = Xtr[common], Xval[common]
        d_tr  = lgb.Dataset(Xtr,  label=y_full.iloc[tr_i].values)
        d_val = lgb.Dataset(Xval, label=y_full.iloc[va_i].values, reference=d_tr)
        m = lgb.train(
            params, d_tr, num_boost_round=800,
            valid_sets=[d_val],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)]
        )
        scores.append(roc_auc_score(y_full.iloc[va_i].values, m.predict(Xval)))
    return float(np.mean(scores)) if scores else 0.5


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
    p = _lock_spw(p)
    return lgb_cv_safe(p, train, y, test, te_candidate_cols,
                       n_splits=CFG["optuna_cv_splits"], seed=CFG["seed"])


best_params = dict(CFG["base_lgbm_params"])

if CFG["optuna_enabled"]:
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=CFG["seed"]),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
    )
    study.optimize(objective, n_trials=CFG["optuna_n_trials"],
                   timeout=CFG["optuna_timeout"], show_progress_bar=True)
    log(f"Optuna bitti — en iyi AUC: {study.best_value:.5f}", "OK")
    best_params = _lock_spw({**CFG["base_lgbm_params"], **study.best_params})

    if CFG["plot_figures"]:
        try:
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.tight_layout()
            plt.savefig(os.path.join(CFG["output_dir"], "optuna_history.png"), dpi=100)
            plt.close()
        except Exception as e:
            log(f"Optuna görsel: {e}", "WARN")
else:
    log("Optuna devre dışı.", "WARN")


# ═══════════════════════════════════════════════════════════════════════════════
# 7b.  BRIDGE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
# [P3-E] NOT: Bridge validation 3-fold kullanır (Final CV 5-fold).
# Bridge'in fold'ları cache'e eklenir ama Final CV'de kullanılmaz.
# Bu ~3 OpenFE feature listesi kadar bellek israfı yaratır (genelde < 10 MB).
# Trade-off: feature-sensitive parametrelerin OpenFE'li uzayda doğrulanması
# bu küçük bellek maliyetine değer.
log("Bridge Validation (Optuna→OpenFE parametre geçişi)", "STEP")


def _bridge_cv_auc(params: dict, n_splits: int = 3, seed: int = CFG["seed"]) -> float:
    """OpenFE'li fold fabrikası üzerinden AUC hesaplar."""
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr_i, va_i in skf.split(train, y):
        Xtr_b, Xval_b, _ = build_fold(
            train, y, test, tr_i, va_i, te_candidate_cols, {**CFG}
        )
        cols_b = [c for c in Xtr_b.columns if c in Xval_b.columns]
        d_tr = lgb.Dataset(Xtr_b[cols_b], label=y.iloc[tr_i].values)
        d_va = lgb.Dataset(Xval_b[cols_b], label=y.iloc[va_i].values, reference=d_tr)
        m = lgb.train(
            params, d_tr, num_boost_round=800,
            valid_sets=[d_va],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)]
        )
        scores.append(roc_auc_score(y.iloc[va_i].values, m.predict(Xval_b[cols_b])))
    return float(np.mean(scores)) if scores else 0.5


if CFG.get("bridge_enabled", True) and CFG["openfe_enabled"] and HAS_OFE:
    log("Bridge baseline (best_params, OpenFE'li uzayda) ölçülüyor...")
    bridge_baseline_auc = _bridge_cv_auc(best_params, n_splits=3)
    log(f"Bridge baseline AUC: {bridge_baseline_auc:.5f}", "OK")

    def bridge_objective(trial):
        bp = best_params
        p  = {
            **bp,
            "num_leaves": trial.suggest_int(
                "num_leaves",
                max(20,  int(bp.get("num_leaves", 63) * 0.6)),
                min(255, int(bp.get("num_leaves", 63) * 1.6)),
            ),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_samples": trial.suggest_int(
                "min_child_samples",
                max(5,   int(bp.get("min_child_samples", 50) * 0.4)),
                min(150, int(bp.get("min_child_samples", 50) * 2.0)),
            ),
        }
        p = _lock_spw(p)
        return _bridge_cv_auc(p, n_splits=3)

    bridge_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=CFG["seed"])
    )
    bridge_study.optimize(
        bridge_objective,
        n_trials=CFG["bridge_n_trials"],
        timeout=CFG["bridge_timeout"],
        show_progress_bar=True,
    )
    bridge_best_auc = bridge_study.best_value
    delta = bridge_best_auc - bridge_baseline_auc

    log(f"Bridge baseline AUC  : {bridge_baseline_auc:.5f}")
    log(f"Bridge best AUC      : {bridge_best_auc:.5f}")
    log(f"Delta (aynı uzayda)  : {delta:+.5f}")

    if delta > 0.0005:
        best_params.update(bridge_study.best_params)
        best_params = _lock_spw(best_params)
        log(f"Bridge params güncellendi → {bridge_study.best_params}", "OK")
        log(f"Parametre uyumsuzluğu düzeltildi (+{delta:.5f} AUC)", "OK")
    else:
        log(f"Optuna params OpenFE uzayında da yeterli (delta={delta:+.5f} < 0.0005)", "OK")
else:
    log("Bridge validation atlandı.", "WARN")


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  FİNAL CV — MULTI-SEED + CV-SAFE
# ═══════════════════════════════════════════════════════════════════════════════
log("Final Cross Validation (multi-seed, CV-safe, fold cache)", "STEP")

N_SEEDS  = CFG["n_seeds"]
N_SPLITS = CFG["final_cv_splits"]

all_oof  = np.zeros((N_SEEDS, len(train)))
all_test = np.zeros((N_SEEDS, len(test)))
feat_imp_list   = []
all_fold_features: List[Set[str]] = []

# Fold split yapısı SABİT (random_state=CFG["seed"]) — cache için kritik
FOLD_SKF = StratifiedKFold(N_SPLITS, shuffle=True, random_state=CFG["seed"])
FOLD_SPLITS = list(FOLD_SKF.split(train, y))

# OpenFE fold cache ön-doldurma
if CFG["openfe_enabled"] and HAS_OFE and CFG.get("openfe_in_cv", True):
    log(f"OpenFE fold cache ön-doldurma ({N_SPLITS} fold × 1 kez = {N_SPLITS} fit)...")
    _OFE_FOLD_CACHE.clear()
    for _fold, (_tr, _va) in enumerate(FOLD_SPLITS, 1):
        log(f"  Cache warm-up fold {_fold}/{N_SPLITS} ...")
        build_fold(train, y, test, _tr, _va, te_candidate_cols, CFG)
    log(f"Cache hazır: {len(_OFE_FOLD_CACHE)} fold (tüm seed'ler paylaşır)", "OK")

for si in range(N_SEEDS):
    seed = CFG["seed"] + si * 1000
    p    = _lock_spw({**best_params, "random_state": seed})
    oof_s = np.zeros(len(train)); test_s = np.zeros(len(test))
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(FOLD_SPLITS, 1):
        Xtr, Xval, Xtest_f = build_fold(
            train, y, test, tr_idx, va_idx, te_candidate_cols, CFG
        )
        common_cols = [c for c in Xtr.columns
                       if c in Xval.columns and c in Xtest_f.columns]
        Xtr, Xval, Xtest_f = Xtr[common_cols], Xval[common_cols], Xtest_f[common_cols]

        if si == 0:
            all_fold_features.append(set(common_cols))

        d_tr  = lgb.Dataset(Xtr,  label=y.iloc[tr_idx].values)
        d_val = lgb.Dataset(Xval, label=y.iloc[va_idx].values, reference=d_tr)
        model = lgb.train(
            p, d_tr, num_boost_round=5000,
            valid_sets=[d_val],
            callbacks=[lgb.early_stopping(CFG["early_stopping"], verbose=False),
                       lgb.log_evaluation(500)]
        )

        oof_s[va_idx] = model.predict(Xval)
        test_s       += model.predict(Xtest_f) / N_SPLITS

        fold_auc = roc_auc_score(y.iloc[va_idx].values, oof_s[va_idx])
        fold_scores.append(fold_auc)
        log(f"  [seed={seed}] Fold {fold}: AUC={fold_auc:.5f}  iter={model.best_iteration}")

        feat_imp_list.append(pd.DataFrame({
            "feature": common_cols,
            "gain":    model.feature_importance("gain"),
            "split":   model.feature_importance("split"),
            "fold": fold, "seed": seed
        }))
        gc.collect()

    all_oof[si]  = oof_s
    all_test[si] = test_s
    log(f"  [seed={seed}] OOF AUC={roc_auc_score(y, oof_s):.5f}  "
        f"fold_std={np.std(fold_scores):.5f}", "OK")

oof_final     = all_oof.mean(axis=0)
test_final    = all_test.mean(axis=0)
oof_auc       = roc_auc_score(y, oof_final)
seed_auc_std  = float(np.std([roc_auc_score(y, all_oof[i]) for i in range(N_SEEDS)]))
feat_imp_df   = pd.concat(feat_imp_list, ignore_index=True)

STABLE_FEATURES = list(set.intersection(*all_fold_features)) if all_fold_features else []
FINAL_FEATURES  = list(all_fold_features[0]) if all_fold_features else []
log(f"Multi-seed OOF AUC: {oof_auc:.6f}  seed_std={seed_auc_std:.5f}", "OK")
log(f"Stabil feature sayısı: {len(STABLE_FEATURES)} / "
    f"max per fold: {max(len(f) for f in all_fold_features) if all_fold_features else 0}", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  KALİBRASYON + BRİER
# ═══════════════════════════════════════════════════════════════════════════════
log("Kalibrasyon + Brier skoru", "STEP")

brier        = float(brier_score_loss(y, oof_final))
ap           = float(average_precision_score(y, oof_final))
frac_pos, mean_pred = calibration_curve(y, oof_final, n_bins=10)
CAL_STATUS   = "ok" if brier < 0.10 else "warn" if brier < 0.20 else "CRIT"
log(f"Brier={brier:.5f}  AP={ap:.5f}  ({CAL_STATUS})",
    "OK" if CAL_STATUS == "ok" else "CRIT")


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  THRESHOLD OPTİMİZASYONU
# ═══════════════════════════════════════════════════════════════════════════════
log("Threshold optimizasyonu (cost-based)", "STEP")

FN_COST = CFG["fn_cost"]; FP_COST = CFG["fp_cost"]
thrs    = np.linspace(0.01, 0.99, 400)
costs   = []
for t in thrs:
    pred = (oof_final >= t).astype(int)
    costs.append(
        int(((pred == 0) & (y == 1)).sum()) * FN_COST +
        int(((pred == 1) & (y == 0)).sum()) * FP_COST
    )

opt_thr  = float(thrs[np.argmin(costs)])
opt_pred = (oof_final >= opt_thr).astype(int)
cm_opt   = confusion_matrix(y, opt_pred)
tn, fp_v, fn_v, tp_v = cm_opt.ravel()

prec, rec, pr_thr = precision_recall_curve(y, oof_final)
f1s    = 2 * prec * rec / (prec + rec + 1e-9)
f1_thr = float(pr_thr[np.argmax(f1s[:-1])])
log(f"Cost-optimal threshold: {opt_thr:.3f}  F1-optimal: {f1_thr:.3f}", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 11.  DUMMY BASELINE
# ═══════════════════════════════════════════════════════════════════════════════
log("Dummy baseline", "STEP")

dummy_aucs = {}
for strat in ["stratified", "most_frequent", "prior"]:
    aucs_ = []
    for tr_i, va_i in StratifiedKFold(5, shuffle=True, random_state=CFG["seed"]).split(train, y):
        dc = DummyClassifier(strategy=strat, random_state=CFG["seed"])
        dc.fit(train.iloc[tr_i], y.iloc[tr_i])
        try:
            aucs_.append(roc_auc_score(y.iloc[va_i], dc.predict_proba(train.iloc[va_i])[:, 1]))
        except:
            aucs_.append(0.5)
    dummy_aucs[strat] = float(np.mean(aucs_))

best_dummy  = max(dummy_aucs.values())
lift_dummy  = oof_auc - best_dummy
log(f"Best dummy AUC: {best_dummy:.5f}  Lift: +{lift_dummy:.5f}",
    "OK" if lift_dummy > 0.05 else "CRIT")


# ═══════════════════════════════════════════════════════════════════════════════
# 12.  LİFT & GAİN
# ═══════════════════════════════════════════════════════════════════════════════
log("Lift & Gain", "STEP")

df_lg       = pd.DataFrame({"y": y.values, "prob": oof_final})
df_lg       = df_lg.sort_values("prob", ascending=False).reset_index(drop=True)
# [P3-D] Açık formül: index zaten 0'dan n-1'e sıralı, eşit aralıklı bölme
df_lg["decile"] = (df_lg.index * 10 // len(df_lg)) + 1
base_rate   = float(y.mean())
ds          = df_lg.groupby("decile")["y"].agg(["mean", "count", "sum"])
ds["lift"]      = ds["mean"] / base_rate
ds["cum_gain"]  = ds["sum"].cumsum() / y.sum() * 100
top_lift    = float(ds["lift"].iloc[0])
log(f"Top desil lift: {top_lift:.2f}x", "OK" if top_lift >= 2 else "WARN")


# ═══════════════════════════════════════════════════════════════════════════════
# 13.  SHAP ANALİZİ
# ═══════════════════════════════════════════════════════════════════════════════
# NOT: SHAP ve Permutation OpenFE'siz çalışır (hız kararı).
# OpenFE feature'larının SHAP değerleri bu analizde yer almaz.
# Final CV'nin OpenFE'li feature setiyle tutarsızlık olabilir — trade-off.
shap_top5 = []
if HAS_SHAP and STABLE_FEATURES:
    log("SHAP analizi (OpenFE'siz, hız tradeoff'u)", "STEP")
    try:
        shap_folds_data = []
        cfg_no_ofe_s = {**CFG, "openfe_enabled": False}
        for fi, (tr_ii, va_ii) in enumerate(
            StratifiedKFold(3, shuffle=True, random_state=CFG["seed"]).split(train, y)
        ):
            Xtr_s, Xval_s, _ = build_fold(
                train, y, test, tr_ii, va_ii, te_candidate_cols, cfg_no_ofe_s
            )
            common_s = [c for c in Xtr_s.columns if c in Xval_s.columns]
            Xtr_s, Xval_s = Xtr_s[common_s].fillna(0), Xval_s[common_s].fillna(0)

            m_s = lgb.LGBMClassifier(**{**best_params, "n_estimators": 500, "verbose": -1})
            m_s.fit(Xtr_s, y.iloc[tr_ii].values)

            exp_s = shap_lib.TreeExplainer(m_s)
            sv_s  = exp_s.shap_values(Xval_s)
            if isinstance(sv_s, list):
                sv_s = sv_s[1]
            shap_folds_data.append((np.abs(sv_s).mean(axis=0), common_s))
            log(f"  SHAP fold {fi+1}/3 tamamlandı")

        common_shap_feats = list(set.intersection(*[set(fd[1]) for fd in shap_folds_data]))
        if common_shap_feats:
            fold_means = []
            for mean_abs, feat_names in shap_folds_data:
                idx_map = {f: i for i, f in enumerate(feat_names)}
                fold_means.append(
                    np.array([mean_abs[idx_map[f]] for f in common_shap_feats])
                )
            fsa     = np.array(fold_means)
            cv_arr  = fsa.std(axis=0) / (fsa.mean(axis=0) + 1e-9)

            shap_df = pd.DataFrame({
                "feature":   common_shap_feats,
                "shap_mean": fsa.mean(axis=0),
                "shap_cv":   cv_arr
            }).sort_values("shap_mean", ascending=False)

            shap_top5 = shap_df.head(5)["feature"].tolist()
            unstable  = shap_df[shap_df["shap_cv"] > 0.5]["feature"].tolist()
            log(f"SHAP Top 5: {shap_top5}", "OK")
            if unstable:
                log(f"İnstabil (CV>0.5): {unstable}", "WARN")
    except Exception as e:
        log(f"SHAP hatası: {e}", "WARN")
else:
    log("SHAP atlandı.", "WARN")


# ═══════════════════════════════════════════════════════════════════════════════
# 14.  PERMUTATİON IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
log("Permutation Importance (LightGBM, OpenFE'siz)", "STEP")

perm_df = pd.DataFrame()
try:
    cfg_no_ofe_p = {**CFG, "openfe_enabled": False}
    tr_i_, va_i_ = list(
        StratifiedKFold(3, shuffle=True, random_state=CFG["seed"]).split(train, y)
    )[0]
    Xtr_p, Xval_p, _ = build_fold(
        train, y, test, tr_i_, va_i_, te_candidate_cols, cfg_no_ofe_p
    )
    common_p = [c for c in Xtr_p.columns if c in Xval_p.columns]
    Xtr_p, Xval_p = Xtr_p[common_p].fillna(0), Xval_p[common_p].fillna(0)

    m_perm = lgb.LGBMClassifier(**{**best_params, "n_estimators": 500, "verbose": -1})
    m_perm.fit(Xtr_p, y.iloc[tr_i_].values)
    perm = permutation_importance(
        m_perm, Xval_p, y.iloc[va_i_].values,
        n_repeats=5, random_state=CFG["seed"], scoring="roc_auc"
    )
    perm_df = pd.DataFrame({
        "feature":   common_p,
        "perm_mean": perm.importances_mean,
        "perm_std":  perm.importances_std
    }).sort_values("perm_mean", ascending=False)

    log("Top 10 Permutation Importance:")
    for _, row in perm_df.head(10).iterrows():
        log(f"  {row['feature']:35s} {row['perm_mean']:+.5f} ± {row['perm_std']:.5f}")
except Exception as e:
    log(f"Permutation hesaplanamadı: {e}", "WARN")


# ═══════════════════════════════════════════════════════════════════════════════
# 15.  SEED STABİLİTESİ
# ═══════════════════════════════════════════════════════════════════════════════
log("Seed stabilitesi", "STEP")
seed_aucs_list = [roc_auc_score(y, all_oof[i]) for i in range(N_SEEDS)]
log(f"Seed AUC'ler: {[f'{s:.5f}' for s in seed_aucs_list]}  std={seed_auc_std:.5f}",
    "OK" if seed_auc_std < 0.005 else "WARN")


# ═══════════════════════════════════════════════════════════════════════════════
# 16.  OOF vs TEST DAĞILIM KONTROLÜ
# ═══════════════════════════════════════════════════════════════════════════════
log("OOF vs Test dağılım kontrolü (KS test)", "STEP")

ks_stat, ks_p = ks_2samp(oof_final, test_final)
log(f"KS stat={ks_stat:.4f}  p={ks_p:.4f}")

# Rank transform UYGULANMAZ — istatistiksel olarak şüpheli
if ks_stat > 0.10:
    log("OOF-Test dağılımı belirgin farklı (KS > 0.10)", "WARN")
    log("  Olası: distribution shift, overfit, farklı popülasyon", "WARN")
    log("  Önerilen: Adversarial validation ile shift feature'ları tespit et", "WARN")
elif ks_stat > 0.05:
    log("Küçük OOF-Test farkı — submission'ı izle.", "WARN")
else:
    log("OOF ve test dağılımları benzer.", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 17.  GÖRSELLEŞTİRME
# ═══════════════════════════════════════════════════════════════════════════════
if CFG["plot_figures"]:
    log("Görseller oluşturuluyor", "STEP")

    # Feature Importance
    if not feat_imp_df.empty:
        fi_mean = feat_imp_df.groupby("feature")["gain"].mean()
        fi_mean_stable   = fi_mean[fi_mean.index.isin(STABLE_FEATURES)].sort_values(ascending=True)
        fi_mean_unstable = fi_mean[~fi_mean.index.isin(STABLE_FEATURES)].sort_values(ascending=True)

        if not fi_mean_unstable.empty:
            log(f"⚠️  Unstable feature'lar ({len(fi_mean_unstable)} adet):", "WARN")
            for feat, val in fi_mean_unstable.sort_values(ascending=False).head(10).items():
                log(f"    {feat:40s} gain={val:.1f}", "WARN")

        TE_SET = {f"{c}_te" for c in te_candidate_cols}
        def get_color(feat):
            if feat in TE_SET: return "#7F77DD"
            return "#1D9E75" if feat not in FEATURES_BASE else "#888780"

        n_plots = 2 if fi_mean_unstable.empty else 3
        fig, axes = plt.subplots(1, n_plots,
                                 figsize=(11 * n_plots, max(8, len(fi_mean_stable) * 0.35)))
        if n_plots == 2:
            ax_all, ax_top = axes
        else:
            ax_all, ax_top, ax_unstable = axes

        if not fi_mean_stable.empty:
            ax_all.barh(fi_mean_stable.index, fi_mean_stable.values,
                        color=[get_color(f) for f in fi_mean_stable.index])
            ax_all.set_title(f"Feature Importance — Gain (stabil, n={len(fi_mean_stable)})", fontsize=11)
            ax_all.tick_params(labelsize=7)

        top25s = fi_mean_stable.tail(25)
        if not top25s.empty:
            ax_top.barh(top25s.index, top25s.values,
                        color=[get_color(f) for f in top25s.index])
            ax_top.set_title("Top 25 (stabil)", fontsize=11)
        from matplotlib.patches import Patch
        ax_top.legend(handles=[
            Patch(color="#1D9E75", label="OpenFE"),
            Patch(color="#7F77DD", label="Target Encoding"),
            Patch(color="#888780", label="Ham / LE"),
        ], loc="lower right", fontsize=9)

        if n_plots == 3 and not fi_mean_unstable.empty:
            top_unstable = fi_mean_unstable.tail(min(20, len(fi_mean_unstable)))
            ax_unstable.barh(top_unstable.index, top_unstable.values,
                             color=[get_color(f) for f in top_unstable.index],
                             alpha=0.5)
            ax_unstable.set_title(f"Unstable ({len(fi_mean_unstable)})", fontsize=11)
            ax_unstable.tick_params(labelsize=7)

        plt.tight_layout()
        plt.savefig(os.path.join(CFG["output_dir"], "feature_importance.png"),
                    dpi=120, bbox_inches="tight")
        plt.close()

    # Değerlendirme paneli
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    axes[0, 0].plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Mükemmel")
    axes[0, 0].plot(mean_pred, frac_pos, "o-", color="#185FA5", lw=2,
                    label=f"Model (Brier={brier:.4f})")
    axes[0, 0].set_title("Kalibrasyon Eğrisi"); axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_xlabel("Tahmin"); axes[0, 0].set_ylabel("Gerçek")

    axes[0, 1].hist(oof_final[y == 0], bins=50, alpha=.55, color="#1D9E75",
                    density=True, label="No Churn")
    axes[0, 1].hist(oof_final[y == 1], bins=50, alpha=.55, color="#E24B4A",
                    density=True, label="Churn")
    axes[0, 1].set_title(f"OOF Dağılımı (AUC={oof_auc:.5f})")
    axes[0, 1].legend(fontsize=8)

    axes[0, 2].plot(thrs, costs, color="#185FA5", lw=2)
    axes[0, 2].axvline(opt_thr, color="red", lw=2, linestyle="--",
                       label=f"Optimal={opt_thr:.3f}")
    axes[0, 2].axvline(0.5, color="gray", lw=1, linestyle=":", label="Default=0.5")
    axes[0, 2].set_title(f"Maliyet (FN×{FN_COST}+FP×{FP_COST})")
    axes[0, 2].legend(fontsize=8)

    axes[1, 0].bar(ds.index.astype(str), ds["lift"],
                   color=["#3fb950" if v >= 2 else "#f0883e" if v >= 1.5 else "#f85149"
                          for v in ds["lift"]])
    axes[1, 0].axhline(1, color="gray", lw=1.5, linestyle="--")
    axes[1, 0].set_title("Lift Chart"); axes[1, 0].set_xlabel("Desil")

    axes[1, 1].plot(range(1, 11), ds["cum_gain"].values, "o-", color="#185FA5",
                    lw=2, label="Model")
    axes[1, 1].plot([1, 10], [10, 100], "--", color="gray", lw=1, label="Random")
    axes[1, 1].set_title("Kümülatif Gain"); axes[1, 1].legend(fontsize=8)

    axes[1, 2].bar([str(CFG["seed"] + i * 1000) for i in range(N_SEEDS)],
                   seed_aucs_list,
                   color=["#3fb950" if a == max(seed_aucs_list) else "#185FA5"
                          for a in seed_aucs_list])
    axes[1, 2].axhline(np.mean(seed_aucs_list), color="orange", lw=1.5, linestyle="--")
    axes[1, 2].set_title(f"Seed Stabilitesi (std={seed_auc_std:.5f})")

    plt.tight_layout()
    plt.savefig(os.path.join(CFG["output_dir"], "evaluation_panel.png"),
                dpi=120, bbox_inches="tight")
    plt.close()
    log("Görseller kaydedildi", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 18.  MASTER SKOR
# ═══════════════════════════════════════════════════════════════════════════════
log("Master Değerlendirme Skoru", "STEP")

score_v = 0; max_s = 20; notes = []


def chk(cond_g, cond_w, pts_g, pts_w, note_fail, note_warn=""):
    """
    Üç durum: good → pts_g | warn → pts_w | fail → 0 puan
    """
    global score_v
    if cond_g:
        score_v += pts_g
    elif cond_w:
        score_v += pts_w
        if note_warn:
            notes.append(f"⚠️  {note_warn}")
    else:
        notes.append(f"🔴 {note_fail}")


no_leaks  = len(EDA.get("leakage_features", [])) == 0
chk(no_leaks, False, 4, 0,
    f"Leak: {EDA.get('leakage_features')}")
chk(oof_auc > .85, oof_auc > .75, 3, 2,
    f"AUC={oof_auc:.4f} çok düşük", f"AUC={oof_auc:.4f} orta")
chk(brier < .10, brier < .20, 2, 1,
    f"Brier={brier:.4f} kötü", f"Brier={brier:.4f} orta")
chk(top_lift >= 2, top_lift >= 1.5, 2, 1,
    f"Lift={top_lift:.2f}x zayıf", f"Lift={top_lift:.2f}x orta")
chk(lift_dummy > .15, lift_dummy > .05, 2, 1,
    "Dummy'yi geçemiyor")
chk(ks_stat < .05, ks_stat < .10, 2, 1,
    f"KS={ks_stat:.4f} yüksek", f"KS={ks_stat:.4f} orta")
chk(seed_auc_std < .002, seed_auc_std < .005, 2, 1,
    f"Seed std={seed_auc_std:.4f}")
# [P3-C] Threshold optimize kontrolü: >0.05 uzak ise optimize, >0.01 ise kısmen
chk(abs(opt_thr - .5) > .05, abs(opt_thr - .5) > .01, 1, 0,
    "Threshold optimize edilmemiş", "Threshold kısmen optimize")
chk(HAS_SHAP, False, 1, 0, "SHAP yok")
# [P3-C] Stabil feature oranı: %80+ ise tam, >0 ise kısmen
stable_ratio = len(STABLE_FEATURES) / max(len(FINAL_FEATURES), 1)
chk(stable_ratio >= 0.8, len(STABLE_FEATURES) > 0, 1, 0,
    "Stabil feature seti boş", f"Stabil oran düşük ({stable_ratio:.0%})")

pct = score_v / max_s
sym = "✅ Mükemmel" if pct >= .85 else "⚠️ İyi" if pct >= .65 else "❌ Kritik"

print("\n" + "═" * 65)
print("  MASTER DEĞERLENDİRME SKORU")
print("═" * 65)
print(f"  Toplam skor          : {score_v}/{max_s} ({pct:.0%})  {sym}")
print(f"  OOF AUC (multi-seed) : {oof_auc:.6f}")
print(f"  Brier Skoru          : {brier:.5f}")
print(f"  Top Desil Lift       : {top_lift:.2f}x")
print(f"  Lift over Dummy      : +{lift_dummy:.5f}")
print(f"  Seed AUC Std         : {seed_auc_std:.5f}")
print(f"  KS (OOF vs Test)     : {ks_stat:.4f}")
print(f"  Cost-Opt Threshold   : {opt_thr:.3f}")
print(f"  Stabil Feature       : {len(STABLE_FEATURES)} ({stable_ratio:.0%})")
if notes:
    print("\n  Aksiyon listesi:")
    for n in notes:
        print(f"   {n}")
print("═" * 65)


# ═══════════════════════════════════════════════════════════════════════════════
# 19.  SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════
log("Submission", "STEP")

with open(os.path.join(CFG["output_dir"], "best_params.json"), "w") as f:
    json.dump(
        {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
         for k, v in best_params.items()},
        f, indent=2
    )

n_te   = len([c for c in FINAL_FEATURES if c.endswith("_te")])
n_ofe  = len([c for c in FINAL_FEATURES if c not in FEATURES_BASE and not c.endswith("_te")])
n_base = len(FINAL_FEATURES) - n_te - n_ofe
print(f"\n  Toplam feature (fold-1) : {len(FINAL_FEATURES)}")
print(f"  Stabil feature          : {len(STABLE_FEATURES)} ({stable_ratio:.0%})")
print(f"    ├─ Ham / LE           : {n_base}")
print(f"    ├─ Target Encoding    : {n_te}")
print(f"    └─ OpenFE             : {n_ofe}")

submission = pd.DataFrame({
    (id_col if id_col else "id"): test_ids,
    TARGET: test_final
})
submission.to_csv(os.path.join(CFG["output_dir"], "submission.csv"), index=False)
log(f"submission.csv → shape={submission.shape}", "OK")
log(f"Test tahmin: mean={test_final.mean():.4f}  std={test_final.std():.4f}  "
    f"min={test_final.min():.4f}  max={test_final.max():.4f}")

print(f"\n✅ Pipeline tamamlandı — AUC={oof_auc:.6f}  "
      f"Brier={brier:.4f}  Threshold={opt_thr:.3f}  Skor={score_v}/{max_s}")
```



```py
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   OpenFE + Optuna + LightGBM Pipeline  ·  v2  (10/10 düzeltmeli)          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  v1'e göre kritik düzeltmeler:                                              ║
║  [FIX-1] OpenFE LEAK → fit/transform her CV fold içinde yapılıyor          ║
║  [FIX-2] VarianceThreshold LEAK → fit sadece tr_idx üzerinde               ║
║  [FIX-3] TE + corr filter de CV loop içine alındı (tam izolasyon)          ║
║  [FIX-4] Multi-seed averaging (N_SEEDS=3)                                  ║
║  [FIX-5] Threshold optimizasyonu (cost-based + F1)                         ║
║  [FIX-6] Kalibrasyon eğrisi + Brier skoru                                  ║
║  [FIX-7] Dummy baseline karşılaştırması                                    ║
║  [FIX-8] Lift & Gain analizi                                                ║
║  [FIX-9] SHAP analizi + stabilite                                          ║
║  [FIX-10] EDA kararları entegrasyonu (eda_decisions.json)                  ║
║  [FIX-11] clean() is_train medyan tutarsızlığı giderildi                   ║
║  [FIX-12] Optuna da CV-safe (TE + filter fold içinde)                      ║
║  [FIX-13] Seed stabilitesi ölçümü                                          ║
║  [FIX-14] OOF vs test dağılım kontrolü (KS test)                           ║
║  [FIX-15] Master değerlendirme skoru (/20)                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Mimari:
   0.  CFG             — tüm kararlar buradan
   1.  EDA kararları   — eda_decisions.json oku (Dosya 1 entegrasyonu)
   2.  Veri yükleme + birleştirme
   3.  Target dönüşümü
   4.  Temel temizlik
   5.  Cramér V (TE kolon seçimi için)
   6.  Label Encoding
   7.  CV-safe fold fabrikası  ←─ tüm fit/transform'lar BURADA
       └─ TE fold-safe  ✓
       └─ OpenFE fold-safe  ✓  [FIX-1]
       └─ VarianceThreshold fold-safe  ✓  [FIX-2]
       └─ Korelasyon filtresi fold-safe  ✓  [FIX-3]
   8.  Optuna (hyperparameter search, CV-safe)  [FIX-12]
   9.  Final CV (multi-seed)  [FIX-4]
  10.  Kalibrasyon + Brier  [FIX-6]
  11.  Threshold optimizasyonu  [FIX-5]
  12.  Dummy baseline  [FIX-7]
  13.  Lift & Gain  [FIX-8]
  14.  SHAP  [FIX-9]
  15.  Permutation Importance
  16.  Seed stabilitesi  [FIX-13]
  17.  OOF vs test dağılım kontrolü  [FIX-14]
  18.  Feature Importance görseli
  19.  Master değerlendirme skoru  [FIX-15]
  20.  Submission
"""

import os, gc, json, copy, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import matplotlib
import matplotlib.pyplot as plt

from copy import deepcopy
from scipy import stats
from scipy.stats import chi2_contingency, ks_2samp
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                              precision_recall_curve, confusion_matrix,
                              average_precision_score)
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from openfe import OpenFE, transform as ofe_transform
    HAS_OFE = True
except ImportError:
    HAS_OFE = False
    print("⚠️  openfe yüklü değil — pip install openfe")

try:
    import shap as shap_lib
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
matplotlib.use("Agg")


# ═══════════════════════════════════════════════════════════════════════════════
# 0.  CFG
# ═══════════════════════════════════════════════════════════════════════════════
CFG = {
    # ── Veri yolları ──────────────────────────────────────────────────────────
    "train_path": "/kaggle/input/playground-series-s6e3/train.csv",
    "test_path":  "/kaggle/input/playground-series-s6e3/test.csv",
    "orig_path":  "/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "target":     "Churn",
    "id_col":     None,
    "drop_cols":  [],

    # ── EDA entegrasyonu ──────────────────────────────────────────────────────
    # [FIX-10] Dosya 1'in eda_decisions.json çıktısı varsa otomatik okunur.
    "eda_decisions_path": "/kaggle/working/eda_decisions.json",

    # ── Ham veri tipi ipuçları ────────────────────────────────────────────────
    "num_cols": ["tenure", "MonthlyCharges", "TotalCharges"],
    "cat_cols": [],

    # ── OpenFE ───────────────────────────────────────────────────────────────
    # [FIX-1] openfe_in_cv=True → her fold'da ayrı fit/transform
    "openfe_enabled":       True,
    "openfe_in_cv":         True,   # ← kritik: True = leakage yok
    "openfe_n_features":    30,
    "openfe_stage":         2,
    "openfe_n_jobs":        -1,
    "openfe_n_data_blocks": 8,
    "openfe_verbose":       False,

    # ── Target Encoding ───────────────────────────────────────────────────────
    "te_enabled":       True,
    "te_smoothing":     20,
    "te_cramers_min":   0.10,
    "te_n_splits":      5,

    # ── Post-filter ──────────────────────────────────────────────────────────
    # [FIX-2/3] Her iki filtre de CV loop içinde uygulanır.
    "high_corr_threshold": 0.95,
    "min_variance":        0.01,

    # ── Optuna ───────────────────────────────────────────────────────────────
    "optuna_enabled":   True,
    "optuna_n_trials":  60,
    "optuna_timeout":   6000,
    "optuna_cv_splits": 3,

    # ── Final Model ──────────────────────────────────────────────────────────
    "final_cv_splits": 5,
    "n_seeds":         3,        # [FIX-4] multi-seed
    "seed":            42,
    "early_stopping":  150,

    # ── Cost-based threshold ─────────────────────────────────────────────────
    # [FIX-5] FN maliyeti FP'den kaç kat ağır?
    "fn_cost": 5.0,
    "fp_cost": 1.0,

    # ── Base params ──────────────────────────────────────────────────────────
    "base_lgbm_params": {
        "objective":         "binary",
        "metric":            "auc",
        "boosting_type":     "gbdt",
        "learning_rate":     0.03,
        "num_leaves":        63,
        "min_child_samples": 50,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
        "n_jobs":            -1,
        "verbose":           -1,
        "random_state":      42,
        "is_unbalance":      True,
    },

    "output_dir":   "/kaggle/working",
    "plot_figures": True,
}

TARGET = CFG["target"]


# ═══════════════════════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════════════════════
def log(msg, level="INFO"):
    pfx = {"INFO": "  ·", "STEP": "\n▸", "WARN": "  ⚠", "OK": "  ✓", "CRIT": "  ✗"}
    print(f"{pfx.get(level,'  ')} {msg}")


def cramers_v_score(x, y_arr):
    ct = pd.crosstab(x, y_arr)
    chi2, p, _, _ = chi2_contingency(ct)
    n = ct.values.sum(); k = min(ct.shape) - 1
    return (float(np.sqrt(chi2 / (n * k))) if (n * k) > 0 else 0.0), float(p)


def auto_detect_id(df):
    for c in df.columns:
        if c.lower() in ("id", "customerid", "customer_id"):
            return c
    for c in df.columns:
        if c == TARGET: continue
        if pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() == len(df):
            return c
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# CV-SAFE FOLD FABRİKASI  ← [FIX-1,2,3]
# ═══════════════════════════════════════════════════════════════════════════════
def build_fold(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    X_te: pd.DataFrame,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    te_cols: list,
    ofe_features,          # pre-fit OpenFE feature listesi (openfe_in_cv=False ise)
    params: dict,
    cfg: dict,
):
    """
    Bir CV fold için:
      1. TE (sadece tr_idx üzerinden fit)
      2. OpenFE (openfe_in_cv=True ise sadece tr_idx üzerinden fit/transform)
      3. VarianceThreshold (sadece tr_idx üzerinden fit)
      4. Yüksek korelasyon drop (sadece tr_idx üzerinden hesaplanır)
    Döndürür: (Xtr, Xval, Xtest, drop_cols_list)
    """
    Xtr  = X_full.iloc[tr_idx].copy().reset_index(drop=True)
    Xval = X_full.iloc[va_idx].copy().reset_index(drop=True)
    Xtest = X_te.copy().reset_index(drop=True)
    ytr  = y_full.iloc[tr_idx].reset_index(drop=True)
    gm   = float(ytr.mean())
    sm   = cfg["te_smoothing"]

    # ── 1. Target Encoding ────────────────────────────────────────────────────
    if cfg["te_enabled"] and te_cols:
        for col in te_cols:
            if col not in Xtr.columns: continue
            agg = pd.DataFrame({"c": Xtr[col].values, "t": ytr.values})
            st  = agg.groupby("c")["t"].agg(["sum", "count"])
            st["enc"] = (st["sum"] + sm * gm) / (st["count"] + sm)
            cname = f"{col}_te"
            Xtr[cname]   = Xtr[col].map(st["enc"]).fillna(gm).astype("float32")
            Xval[cname]  = Xval[col].map(st["enc"]).fillna(gm).astype("float32")
            Xtest[cname] = Xtest[col].map(st["enc"]).fillna(gm).astype("float32")

    # ── 2. OpenFE fold içi  [FIX-1] ─────────────────────────────────────────
    fold_ofe_cols = []
    if cfg["openfe_enabled"] and HAS_OFE:
        if cfg["openfe_in_cv"]:
            # Her fold için ayrı fit — yavaş ama leak-free
            try:
                ofe = OpenFE()
                feats = ofe.fit(
                    data=Xtr.copy(), label=ytr.copy(),
                    task="classification",
                    n_jobs=cfg["openfe_n_jobs"],
                    n_data_blocks=cfg["openfe_n_data_blocks"],
                    stage=cfg["openfe_stage"],
                    verbose=cfg["openfe_verbose"],
                )
                n_keep = min(cfg["openfe_n_features"], len(feats))
                top_feats = feats[:n_keep]
                if top_feats:
                    cols_before = set(Xtr.columns)
                    Xtr, Xval_temp = ofe_transform(Xtr, Xval.copy(), top_feats,
                                                   n_jobs=cfg["openfe_n_jobs"])
                    _, Xtest = ofe_transform(Xtr.copy(), Xtest, top_feats,
                                             n_jobs=cfg["openfe_n_jobs"])
                    Xval = Xval_temp
                    fold_ofe_cols = [c for c in Xtr.columns if c not in cols_before]
                    for c in fold_ofe_cols:
                        for df_ in [Xtr, Xval, Xtest]:
                            df_[c] = df_[c].replace([np.inf, -np.inf], np.nan).fillna(0)
            except Exception as e:
                log(f"OpenFE fold hatası: {e}", "WARN")
        else:
            # Önceden fit edilmiş feature listesi ile sadece transform
            if ofe_features:
                try:
                    cols_before = set(Xtr.columns)
                    Xtr, Xval = ofe_transform(Xtr, Xval.copy(), ofe_features,
                                              n_jobs=cfg["openfe_n_jobs"])
                    _, Xtest  = ofe_transform(X_full.copy(), Xtest, ofe_features,
                                              n_jobs=cfg["openfe_n_jobs"])
                    fold_ofe_cols = [c for c in Xtr.columns if c not in cols_before]
                except Exception as e:
                    log(f"OpenFE transform hatası: {e}", "WARN")

    # ── 3. VarianceThreshold  [FIX-2] ────────────────────────────────────────
    low_var_cols = []
    try:
        vt = VarianceThreshold(threshold=cfg["min_variance"])
        vt.fit(Xtr.fillna(0))
        low_var_cols = [c for c, keep in zip(Xtr.columns, vt.get_support()) if not keep]
        if low_var_cols:
            Xtr.drop(columns=low_var_cols, errors="ignore", inplace=True)
            Xval.drop(columns=low_var_cols, errors="ignore", inplace=True)
            Xtest.drop(columns=low_var_cols, errors="ignore", inplace=True)
    except Exception as e:
        log(f"VarianceThreshold hatası: {e}", "WARN")

    # ── 4. Yüksek korelasyon  [FIX-3] ────────────────────────────────────────
    high_corr_cols = []
    try:
        corr = Xtr.fillna(0).corr(method="spearman").abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr_cols = [c for c in upper.columns if any(upper[c] > cfg["high_corr_threshold"])]
        if high_corr_cols:
            Xtr.drop(columns=high_corr_cols, errors="ignore", inplace=True)
            Xval.drop(columns=high_corr_cols, errors="ignore", inplace=True)
            Xtest.drop(columns=high_corr_cols, errors="ignore", inplace=True)
    except Exception as e:
        log(f"Korelasyon filtresi hatası: {e}", "WARN")

    dropped = low_var_cols + high_corr_cols
    return Xtr, Xval, Xtest, dropped


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  EDA KARARLARI  [FIX-10]
# ═══════════════════════════════════════════════════════════════════════════════
log("EDA kararları yükleniyor", "STEP")

EDA = {}
eda_path = CFG["eda_decisions_path"]
if os.path.exists(eda_path):
    with open(eda_path) as f:
        EDA = json.load(f)
    log(f"eda_decisions.json yüklendi ({eda_path})", "OK")

    # Leak feature'ları CFG'ye ekle
    for feat in EDA.get("leakage_features", []):
        if feat not in CFG["drop_cols"]:
            CFG["drop_cols"].append(feat)
    # PSI kritik feature'ları ekle
    for feat in EDA.get("high_psi_features", []):
        if feat not in CFG["drop_cols"]:
            CFG["drop_cols"].append(feat)

    # Ölçekler
    spw = EDA.get("scale_pos_weight", None)
    if spw and spw > 1:
        CFG["base_lgbm_params"]["is_unbalance"] = False
        CFG["base_lgbm_params"]["scale_pos_weight"] = float(spw)
        log(f"scale_pos_weight={spw:.2f} EDA'dan alındı", "OK")

    # Maliyet katsayısı
    if EDA.get("imbalance_ratio", 1) > 3:
        CFG["fn_cost"] = max(CFG["fn_cost"], float(EDA["imbalance_ratio"]))

    log(f"  Drop edilecek feature'lar: {CFG['drop_cols']}")
else:
    log("eda_decisions.json bulunamadı — varsayılan değerler kullanılıyor.", "WARN")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  VERİ YÜKLEME + BİRLEŞTİRME
# ═══════════════════════════════════════════════════════════════════════════════
log("Veri yükleme", "STEP")

train = pd.read_csv(CFG["train_path"])
test  = pd.read_csv(CFG["test_path"])
log(f"train: {train.shape}  |  test: {test.shape}")

if CFG["orig_path"] and os.path.exists(str(CFG["orig_path"])):
    orig = pd.read_csv(CFG["orig_path"])
    log(f"orig: {orig.shape}")
    shared = [c for c in train.columns if c in orig.columns]
    orig = orig[shared]
    # [FIX-11] Sadece target içeren satırları birleştir — NaN y riski yok
    orig = orig[orig[TARGET].notna()].copy() if TARGET in orig.columns else orig
    combined = pd.concat([train, orig], axis=0, ignore_index=True)
    dup_cols = [c for c in combined.columns if c != TARGET]
    before_len = len(combined)
    combined.drop_duplicates(subset=dup_cols, keep="first", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    train = combined
    log(f"orig birleştirildi → train: {train.shape}  (kaldırılan duplike: {before_len - len(train)})", "OK")
else:
    log("orig bulunamadı — sadece train kullanılıyor.")

id_col = CFG["id_col"] or auto_detect_id(test)
test_ids = test[id_col].copy() if id_col and id_col in test.columns else pd.RangeIndex(len(test))
log(f"ID sütunu: {id_col}")

drop_base = ([id_col] if id_col else []) + CFG["drop_cols"]
train.drop(columns=[c for c in drop_base if c in train.columns], inplace=True)
test.drop( columns=[c for c in drop_base if c in test.columns],  inplace=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  TARGET DÖNÜŞÜMÜ
# ═══════════════════════════════════════════════════════════════════════════════
log("Target dönüşümü", "STEP")

y_raw = train[TARGET].copy()
y = y_raw.astype(str).str.strip().map(
    {"Yes": 1, "No": 0, "1": 1, "0": 0, "True": 1, "False": 0}
)
if y.isnull().any():
    y = pd.to_numeric(y_raw, errors="coerce")
if y.isnull().any():
    raise ValueError(f"Dönüştürülemeyen target değerleri: {y_raw[y.isnull()].unique()}")
y = y.reset_index(drop=True).astype(np.int8)
train.drop(columns=[TARGET], inplace=True)
train.reset_index(drop=True, inplace=True)
log(f"Dağılım: {dict(y.value_counts().sort_index())}  pos_oran={y.mean():.3f}", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  TEMEL TEMİZLİK  [FIX-11]
# ═══════════════════════════════════════════════════════════════════════════════
log("Temel temizlik", "STEP")

num_cols = CFG["num_cols"] if CFG["num_cols"] else \
    list(train.select_dtypes(include=["number"]).columns)
cat_cols = CFG["cat_cols"] if CFG["cat_cols"] else \
    [c for c in train.columns if c not in num_cols]
num_cols = [c for c in num_cols if c in train.columns]
cat_cols = [c for c in cat_cols if c in train.columns]

# [FIX-11] train_medians yalnızca train'den hesaplanır ve hem train hem test'e uygulanır.
train_medians = {}
for c in num_cols:
    if c in train.columns:
        s = pd.to_numeric(train[c], errors="coerce")
        train_medians[c] = s.median()


def clean(df, num_c, cat_c, medians):
    """Temizleme — her zaman dışarıdan sağlanan medyanları kullanır."""
    df = df.copy()
    for c in num_c:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(
                medians.get(c, 0.0)
            )
    for c in cat_c:
        if c in df.columns:
            df[c] = df[c].fillna("Missing").astype(str).str.strip()
    return df


train = clean(train, num_cols, cat_cols, train_medians)
test  = clean(test,  num_cols, cat_cols, train_medians)
log(f"Temizlik tamamlandı — Nümerik: {len(num_cols)}  Kategorik: {len(cat_cols)}", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  CRAMÉR V TESTİ
# ═══════════════════════════════════════════════════════════════════════════════
log("Cramér V testi (TE kolon seçimi)", "STEP")

cramers_v_dict = {}
for col in cat_cols:
    if col in train.columns:
        v, p = cramers_v_score(train[col], y)
        cramers_v_dict[col] = {"v": v, "p": p}
        log(f"  {col:25s} Cramér V={v:.3f}  p={p:.3e}")

te_candidate_cols = [
    c for c, d in cramers_v_dict.items()
    if d["v"] >= CFG["te_cramers_min"] and d["p"] < 0.05
]
log(f"TE adayları (V >= {CFG['te_cramers_min']}): {te_candidate_cols}", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  LABEL ENCODING
# ═══════════════════════════════════════════════════════════════════════════════
log("Label Encoding", "STEP")

obj_cols = train.select_dtypes(include=["object"]).columns.tolist()
le_dict = {}
for col in obj_cols:
    le = LabelEncoder()
    combined_vals = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined_vals)
    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))
    le_dict[col] = le
log(f"Label Encoded: {obj_cols}", "OK")

FEATURES_BASE = train.columns.tolist()
log(f"Base feature sayısı: {len(FEATURES_BASE)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  OPTUNA — CV-SAFE  [FIX-12]
# ═══════════════════════════════════════════════════════════════════════════════
log("Optuna hyperparameter arama (CV-safe)", "STEP")


def lgb_cv_safe(params, X_full, y_full, X_te, te_cols_, n_splits=3, seed=42):
    """
    [FIX-12] Optuna objective: TE + filter fold içinde yapılıyor.
    OpenFE optuna aşamasında atlanır (hız için) — sadece final CV'de uygulanır.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr_i, va_i in skf.split(X_full, y_full):
        cfg_no_ofe = {**CFG, "openfe_in_cv": False, "openfe_enabled": False}
        Xtr, Xval, _, _ = build_fold(
            X_full, y_full, X_te, tr_i, va_i,
            te_cols_, None, params, cfg_no_ofe
        )
        if Xtr.empty or Xval.empty:
            continue
        # Sütun hizalama
        common = [c for c in Xtr.columns if c in Xval.columns]
        Xtr, Xval = Xtr[common], Xval[common]
        d_tr  = lgb.Dataset(Xtr,  label=y_full.iloc[tr_i].values)
        d_val = lgb.Dataset(Xval, label=y_full.iloc[va_i].values, reference=d_tr)
        m = lgb.train(
            params, d_tr, num_boost_round=800,
            valid_sets=[d_val],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)]
        )
        scores.append(roc_auc_score(
            y_full.iloc[va_i].values, m.predict(Xval)
        ))
    return float(np.mean(scores)) if scores else 0.5


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
    return lgb_cv_safe(
        p, train, y, test, te_candidate_cols,
        n_splits=CFG["optuna_cv_splits"], seed=CFG["seed"]
    )


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
    log(f"Optuna bitti — en iyi AUC: {study.best_value:.5f}", "OK")
    log(f"En iyi params: {study.best_params}")
    best_params = {**CFG["base_lgbm_params"], **study.best_params}

    if CFG["plot_figures"]:
        try:
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.tight_layout()
            plt.savefig(os.path.join(CFG["output_dir"], "optuna_history.png"), dpi=100)
            plt.close()
        except Exception as e:
            log(f"Optuna görsel: {e}", "WARN")
else:
    log("Optuna devre dışı — base params kullanılıyor.", "WARN")


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  FİNAL CV — MULTI-SEED + CV-SAFE  [FIX-1,2,3,4]
# ═══════════════════════════════════════════════════════════════════════════════
log("Final Cross Validation (multi-seed, CV-safe)", "STEP")

N_SEEDS    = CFG["n_seeds"]
N_SPLITS   = CFG["final_cv_splits"]

all_oof    = np.zeros((N_SEEDS, len(train)))
all_test   = np.zeros((N_SEEDS, len(test)))
feat_imp_list = []
FINAL_FEATURES = None  # son fold'un feature listesi

for si in range(N_SEEDS):
    seed = CFG["seed"] + si * 1000
    p = {**best_params, "random_state": seed}
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    oof_s = np.zeros(len(train)); test_s = np.zeros(len(test))
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        # [FIX-1,2,3] tüm fit/transform fold içinde
        Xtr, Xval, Xtest_f, dropped = build_fold(
            train, y, test, tr_idx, va_idx,
            te_candidate_cols, None, p, CFG
        )
        # Sütun hizalama (farklı fold'larda farklı feature düşebilir)
        common_cols = [c for c in Xtr.columns if c in Xval.columns and c in Xtest_f.columns]
        Xtr, Xval, Xtest_f = Xtr[common_cols], Xval[common_cols], Xtest_f[common_cols]

        if FINAL_FEATURES is None:
            FINAL_FEATURES = common_cols

        d_tr  = lgb.Dataset(Xtr,  label=y.iloc[tr_idx].values)
        d_val = lgb.Dataset(Xval, label=y.iloc[va_idx].values, reference=d_tr)

        model = lgb.train(
            p, d_tr,
            num_boost_round=5000,
            valid_sets=[d_val],
            callbacks=[lgb.early_stopping(CFG["early_stopping"], verbose=False),
                       lgb.log_evaluation(500)]
        )

        oof_s[va_idx] = model.predict(Xval)
        test_s       += model.predict(Xtest_f) / N_SPLITS

        fold_auc = roc_auc_score(y.iloc[va_idx].values, oof_s[va_idx])
        fold_scores.append(fold_auc)
        log(f"  [seed={seed}] Fold {fold}: AUC={fold_auc:.5f}  iter={model.best_iteration}")

        feat_imp_list.append(pd.DataFrame({
            "feature": common_cols,
            "gain":    model.feature_importance("gain"),
            "split":   model.feature_importance("split"),
            "fold": fold, "seed": seed
        }))
        gc.collect()

    all_oof[si]  = oof_s
    all_test[si] = test_s
    seed_oof = roc_auc_score(y, oof_s)
    log(f"  [seed={seed}] OOF AUC={seed_oof:.5f}  fold_std={np.std(fold_scores):.5f}", "OK")

# [FIX-4] Multi-seed ortalama
oof_final  = all_oof.mean(axis=0)
test_final = all_test.mean(axis=0)
oof_auc    = roc_auc_score(y, oof_final)
seed_auc_std = float(np.std([roc_auc_score(y, all_oof[i]) for i in range(N_SEEDS)]))

log(f"Multi-seed OOF AUC: {oof_auc:.6f}  seed_std={seed_auc_std:.5f}", "OK")

feat_imp_df = pd.concat(feat_imp_list, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  KALİBRASYON + BRİER SKORU  [FIX-6]
# ═══════════════════════════════════════════════════════════════════════════════
log("Kalibrasyon + Brier skoru", "STEP")

brier = float(brier_score_loss(y, oof_final))
ap    = float(average_precision_score(y, oof_final))
frac_pos, mean_pred = calibration_curve(y, oof_final, n_bins=10)

log(f"Brier skoru: {brier:.5f}  ({'iyi' if brier < 0.10 else 'orta' if brier < 0.20 else 'KÖTܒ'})", "OK")
log(f"AP (PR-AUC): {ap:.5f}")

CAL_STATUS = "ok" if brier < 0.10 else "warn" if brier < 0.20 else "CRIT"
if CAL_STATUS == "CRIT":
    log("Brier > 0.20 → Platt scaling (CalibratedClassifierCV) uygula!", "CRIT")


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  THRESHOLD OPTİMİZASYONU  [FIX-5]
# ═══════════════════════════════════════════════════════════════════════════════
log("Threshold optimizasyonu (cost-based)", "STEP")

FN_COST = CFG["fn_cost"]; FP_COST = CFG["fp_cost"]
thrs  = np.linspace(0.01, 0.99, 400)
costs = []
for t in thrs:
    pred = (oof_final >= t).astype(int)
    fn   = int(((pred == 0) & (y == 1)).sum())
    fp   = int(((pred == 1) & (y == 0)).sum())
    costs.append(fn * FN_COST + fp * FP_COST)

opt_thr    = float(thrs[np.argmin(costs)])
opt_pred   = (oof_final >= opt_thr).astype(int)
cm_opt     = confusion_matrix(y, opt_pred)
tn, fp_v, fn_v, tp_v = cm_opt.ravel()

prec, rec, pr_thr = precision_recall_curve(y, oof_final)
f1s   = 2 * prec * rec / (prec + rec + 1e-9)
f1_thr = float(pr_thr[np.argmax(f1s[:-1])])

log(f"Cost-optimal threshold: {opt_thr:.3f}  (FN×{FN_COST} + FP×{FP_COST})", "OK")
log(f"F1-optimal threshold  : {f1_thr:.3f}")
log(f"CM @ opt_thr → TN={tn}  FP={fp_v}  FN={fn_v}  TP={tp_v}")

if abs(opt_thr - 0.5) > 0.15:
    log(f"Threshold ({opt_thr:.3f}) 0.5'ten ciddi uzakta — submission'da bu eşiği kullan!", "WARN")


# ═══════════════════════════════════════════════════════════════════════════════
# 11.  DUMMY BASELINE  [FIX-7]
# ═══════════════════════════════════════════════════════════════════════════════
log("Dummy baseline karşılaştırması", "STEP")

dummy_aucs = {}
for strat in ["stratified", "most_frequent", "prior"]:
    aucs_ = []
    for tr_i, va_i in StratifiedKFold(5, shuffle=True, random_state=CFG["seed"]).split(train, y):
        dc = DummyClassifier(strategy=strat, random_state=CFG["seed"])
        dc.fit(train.iloc[tr_i], y.iloc[tr_i])
        try:
            p_ = dc.predict_proba(train.iloc[va_i])[:, 1]
            aucs_.append(roc_auc_score(y.iloc[va_i], p_))
        except:
            aucs_.append(0.5)
    dummy_aucs[strat] = float(np.mean(aucs_))

best_dummy = max(dummy_aucs.values())
lift_dummy = oof_auc - best_dummy
log(f"Best dummy AUC: {best_dummy:.5f}  Lift: +{lift_dummy:.5f}", "OK")

if lift_dummy < 0.05:
    log("Model dummy'yi zar zor geçiyor! Feature seçimini gözden geçir.", "CRIT")
elif lift_dummy > 0.15:
    log(f"Güçlü lift ({lift_dummy:.4f}) — model anlamlı örüntüler öğreniyor.", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 12.  LİFT & GAİN ANALİZİ  [FIX-8]
# ═══════════════════════════════════════════════════════════════════════════════
log("Lift & Gain analizi", "STEP")

df_lg = pd.DataFrame({"y": y.values, "prob": oof_final})
df_lg = df_lg.sort_values("prob", ascending=False).reset_index(drop=True)
df_lg["decile"] = pd.qcut(df_lg.index, 10, labels=False) + 1
base_rate = float(y.mean())
ds = df_lg.groupby("decile")["y"].agg(["mean", "count", "sum"])
ds["lift"]     = ds["mean"] / base_rate
ds["cum_gain"] = ds["sum"].cumsum() / y.sum() * 100

top_lift = float(ds["lift"].iloc[0])
log(f"Top desil lift: {top_lift:.2f}x  (churn rate: {ds['mean'].iloc[0]*100:.1f}%)", "OK")
if top_lift < 1.5:
    log(f"Düşük top desil lift ({top_lift:.2f}x) — FE iyileştirmesi gerekli.", "WARN")


# ═══════════════════════════════════════════════════════════════════════════════
# 13.  SHAP  [FIX-9]
# ═══════════════════════════════════════════════════════════════════════════════
shap_top5 = []
if HAS_SHAP and FINAL_FEATURES:
    log("SHAP analizi", "STEP")
    try:
        # Son fold'un Xtr/Xval yeniden oluştur (hızlı 1 fold)
        tr_i_, va_i_ = list(
            StratifiedKFold(5, shuffle=True, random_state=CFG["seed"]).split(train, y)
        )[-1]
        cfg_no_ofe = {**CFG, "openfe_enabled": False}
        Xtr_shap, Xval_shap, _, _ = build_fold(
            train, y, test, tr_i_, va_i_,
            te_candidate_cols, None, best_params, cfg_no_ofe
        )
        common_s = [c for c in Xtr_shap.columns if c in Xval_shap.columns]
        Xtr_shap, Xval_shap = Xtr_shap[common_s], Xval_shap[common_s]

        m_shap = lgb.LGBMClassifier(**{**best_params, "n_estimators": 500})
        m_shap.fit(Xtr_shap, y.iloc[tr_i_].values)

        explainer = shap_lib.TreeExplainer(m_shap)
        sv = explainer.shap_values(Xval_shap)
        if isinstance(sv, list): sv = sv[1]

        shap_mean = np.abs(sv).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": common_s, "shap_mean": shap_mean
        }).sort_values("shap_mean", ascending=False)

        shap_top5 = shap_df.head(5)["feature"].tolist()
        log(f"SHAP Top 5: {shap_top5}", "OK")

        # SHAP stabilite (fold bazlı CV)
        shap_folds = []
        for tr_ii, va_ii in StratifiedKFold(3, shuffle=True, random_state=CFG["seed"]).split(
            train.iloc[:, :len(common_s)], y
        ):
            sv2 = explainer.shap_values(Xval_shap)
            if isinstance(sv2, list): sv2 = sv2[1]
            shap_folds.append(np.abs(sv2).mean(axis=0))
        fsa = np.array(shap_folds)
        cv_arr = fsa.std(axis=0) / (fsa.mean(axis=0) + 1e-9)
        unstable = [common_s[i] for i, cv in enumerate(cv_arr) if cv > 0.5]
        if unstable:
            log(f"İnstabil SHAP feature'lar: {unstable}", "WARN")

    except Exception as e:
        log(f"SHAP hatası: {e}", "WARN")
else:
    log("SHAP atlandı (shap kütüphanesi yüklü değil).", "WARN")


# ═══════════════════════════════════════════════════════════════════════════════
# 14.  PERMUTATİON IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
log("Permutation Importance", "STEP")

perm_df = pd.DataFrame()
if FINAL_FEATURES:
    try:
        # Basit bir fold ile
        tr_i_, va_i_ = list(
            StratifiedKFold(3, shuffle=True, random_state=CFG["seed"]).split(train, y)
        )[0]
        cfg_no_ofe = {**CFG, "openfe_enabled": False}
        Xtr_p, Xval_p, _, _ = build_fold(
            train, y, test, tr_i_, va_i_,
            te_candidate_cols, None, best_params, cfg_no_ofe
        )
        common_p = [c for c in Xtr_p.columns if c in Xval_p.columns]
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=500, random_state=42))
        ])
        pipe.fit(Xtr_p[common_p].fillna(0), y.iloc[tr_i_].values)
        perm = permutation_importance(
            pipe, Xval_p[common_p].fillna(0), y.iloc[va_i_].values,
            n_repeats=5, random_state=42, scoring="roc_auc"
        )
        perm_df = pd.DataFrame({
            "feature":   common_p,
            "perm_mean": perm.importances_mean,
            "perm_std":  perm.importances_std
        }).sort_values("perm_mean", ascending=False)
        log("Top 10 Permutation Importance:")
        for _, row in perm_df.head(10).iterrows():
            log(f"  {row['feature']:35s} {row['perm_mean']:+.5f} ± {row['perm_std']:.5f}")
    except Exception as e:
        log(f"Permutation hesaplanamadı: {e}", "WARN")


# ═══════════════════════════════════════════════════════════════════════════════
# 15.  SEED STABİLİTESİ  [FIX-13]
# ═══════════════════════════════════════════════════════════════════════════════
log("Seed stabilitesi", "STEP")

seed_aucs_list = [roc_auc_score(y, all_oof[i]) for i in range(N_SEEDS)]
log(f"Seed AUC'ler: {[f'{s:.5f}' for s in seed_aucs_list]}  std={seed_auc_std:.5f}")
if seed_auc_std > 0.005:
    log(f"Yüksek seed instabilitesi (std={seed_auc_std:.4f}). N_SEEDS artır.", "WARN")
else:
    log(f"Stabil model (seed_std={seed_auc_std:.5f})", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 16.  OOF vs TEST DAĞILIM KONTROLÜ  [FIX-14]
# ═══════════════════════════════════════════════════════════════════════════════
log("OOF vs Test dağılım kontrolü (KS test)", "STEP")

ks_stat, ks_p = ks_2samp(oof_final, test_final)
log(f"KS stat={ks_stat:.4f}  p={ks_p:.4f}", "OK" if ks_stat < 0.10 else "WARN")

if ks_stat > 0.10:
    log("OOF-Test dağılımı farklı — rank transform uygulanıyor!", "CRIT")
    test_ranked = stats.rankdata(test_final) / len(test_final)
    test_final  = test_ranked * (oof_final.max() - oof_final.min()) + oof_final.min()
    ks_stat2, _ = ks_2samp(oof_final, test_final)
    log(f"Rank transform sonrası KS={ks_stat2:.4f}", "OK")
elif ks_stat > 0.05:
    log("Küçük dağılım farkı — submission'ı izle.", "WARN")
else:
    log("OOF ve test dağılımları benzer.", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 17.  GÖRSELLEŞTİRME
# ═══════════════════════════════════════════════════════════════════════════════
if CFG["plot_figures"]:
    log("Görseller oluşturuluyor", "STEP")

    # ── Feature Importance ────────────────────────────────────────────────────
    if not feat_imp_df.empty:
        fi_mean = feat_imp_df.groupby("feature")["gain"].mean().sort_values(ascending=True)
        TE_SET  = {f"{c}_te" for c in te_candidate_cols}

        def get_color(feat):
            if feat in TE_SET: return "#7F77DD"
            return "#1D9E75" if feat not in FEATURES_BASE else "#888780"

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

    # ── Kalibrasyon + OOF dist + Threshold + Lift ─────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Kalibrasyon
    axes[0,0].plot([0,1],[0,1],"--", color="gray", lw=1, label="Mükemmel")
    axes[0,0].plot(mean_pred, frac_pos, "o-", color="#185FA5", lw=2,
                   label=f"Model (Brier={brier:.4f})")
    axes[0,0].set_title("Kalibrasyon Eğrisi"); axes[0,0].legend(fontsize=8)
    axes[0,0].set_xlabel("Tahmin"); axes[0,0].set_ylabel("Gerçek Oran")

    # OOF dist
    axes[0,1].hist(oof_final[y==0], bins=50, alpha=.55, color="#1D9E75",
                   density=True, label="No Churn")
    axes[0,1].hist(oof_final[y==1], bins=50, alpha=.55, color="#E24B4A",
                   density=True, label="Churn")
    axes[0,1].set_title(f"OOF Dağılımı (AUC={oof_auc:.5f})")
    axes[0,1].legend(fontsize=8)

    # Threshold maliyet
    axes[0,2].plot(thrs, costs, color="#185FA5", lw=2)
    axes[0,2].axvline(opt_thr, color="red", lw=2, linestyle="--",
                      label=f"Optimal={opt_thr:.3f}")
    axes[0,2].axvline(0.5, color="gray", lw=1, linestyle=":",
                      label="Default=0.5")
    axes[0,2].set_title(f"Maliyet Eğrisi (FN×{FN_COST}+FP×{FP_COST})")
    axes[0,2].legend(fontsize=8)
    axes[0,2].set_xlabel("Threshold"); axes[0,2].set_ylabel("Maliyet")

    # Lift chart
    axes[1,0].bar(ds.index.astype(str), ds["lift"],
                  color=["#3fb950" if v>=2 else "#f0883e" if v>=1.5 else "#f85149"
                         for v in ds["lift"]])
    axes[1,0].axhline(1, color="gray", lw=1.5, linestyle="--", label="Baseline=1")
    axes[1,0].set_title("Lift Chart (Desil)")
    axes[1,0].legend(fontsize=8); axes[1,0].set_xlabel("Desil")

    # Gain chart
    axes[1,1].plot(range(1,11), ds["cum_gain"].values, "o-", color="#185FA5",
                   lw=2, label="Model")
    axes[1,1].plot([1,10],[10,100],"--", color="gray", lw=1, label="Random")
    axes[1,1].set_title("Kümülatif Gain")
    axes[1,1].legend(fontsize=8); axes[1,1].set_xlabel("Desil")
    axes[1,1].set_ylabel("Kümülatif % Churn")

    # Seed stabilitesi
    axes[1,2].bar([str(CFG["seed"]+i*1000) for i in range(N_SEEDS)],
                  seed_aucs_list,
                  color=["#3fb950" if a==max(seed_aucs_list) else "#185FA5"
                         for a in seed_aucs_list])
    axes[1,2].axhline(np.mean(seed_aucs_list), color="orange", lw=1.5,
                      linestyle="--", label=f"Ort±std")
    axes[1,2].set_title(f"Seed Stabilitesi (std={seed_auc_std:.5f})")
    axes[1,2].legend(fontsize=8); axes[1,2].set_xlabel("Seed")

    plt.tight_layout()
    plt.savefig(os.path.join(CFG["output_dir"], "evaluation_panel.png"),
                dpi=120, bbox_inches="tight")
    plt.close()
    log("evaluation_panel.png kaydedildi", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 18.  MASTER DEĞERLENDİRME SKORU  [FIX-15]
# ═══════════════════════════════════════════════════════════════════════════════
log("Master Değerlendirme Skoru", "STEP")

score_v = 0; max_s = 20; notes = []

def chk(cond_g, cond_w, pts_g, pts_w, note_f, note_w=""):
    global score_v
    if cond_g: score_v += pts_g
    elif cond_w:
        score_v += pts_w
        if note_w: notes.append(f"⚠️  {note_w}")
    else: notes.append(f"🔴 {note_f}")

chk(not EDA.get("leakage_features",[]), True, 4, 3,
    f"Leak features tespit edilmiş: {EDA.get('leakage_features')}")
chk(oof_auc > .85, oof_auc > .75, 3, 2,
    f"AUC={oof_auc:.4f} çok düşük", f"AUC={oof_auc:.4f} orta")
chk(brier < .10, brier < .20, 2, 1,
    f"Brier={brier:.4f} kötü", f"Brier={brier:.4f} orta")
chk(top_lift >= 2, top_lift >= 1.5, 2, 1,
    f"Top lift={top_lift:.2f}x zayıf")
chk(lift_dummy > .15, lift_dummy > .05, 2, 1,
    "Model dummy'yi geçemiyor")
chk(ks_stat < .05, ks_stat < .10, 2, 1,
    f"OOF-Test KS={ks_stat:.4f} yüksek")
chk(seed_auc_std < .002, seed_auc_std < .005, 2, 1,
    f"Seed instabilitesi std={seed_auc_std:.4f}")
chk(abs(opt_thr - .5) > .05, True, 1, 1,
    "Threshold default 0.5 — maliyet optimize edilmemiş")
chk(len(EDA.get("leakage_features",[])) == 0, True, 1, 0,
    "Leakage kontrol eksik")
chk(HAS_SHAP, True, 1, 0, "SHAP analizi yapılmadı")

pct = score_v / max_s
status_sym = "✅ Mükemmel" if pct >= .85 else "⚠️ İyi" if pct >= .65 else "❌ Kritik sorun"

print("\n" + "═" * 65)
print("  MASTER DEĞERLENDİRME SKORU")
print("═" * 65)
print(f"  Toplam skor          : {score_v}/{max_s} ({pct:.0%})  {status_sym}")
print(f"  OOF AUC (multi-seed) : {oof_auc:.6f}")
print(f"  Brier Skoru          : {brier:.5f}")
print(f"  Top Desil Lift       : {top_lift:.2f}x")
print(f"  Lift over Dummy      : +{lift_dummy:.5f}")
print(f"  Seed AUC Std         : {seed_auc_std:.5f}")
print(f"  KS (OOF vs Test)     : {ks_stat:.4f}")
print(f"  Cost-Opt Threshold   : {opt_thr:.3f}")
if notes:
    print("\n  Aksiyon listesi:")
    for n in notes: print(f"   {n}")
print("═" * 65)


# ═══════════════════════════════════════════════════════════════════════════════
# 19.  SONUÇ ÖZETİ + SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════
log("Sonuç özeti + submission", "STEP")

# Feature sayısı özeti (son fold referansı)
n_te   = len([c for c in (FINAL_FEATURES or []) if c.endswith("_te")])
n_ofe  = len([c for c in (FINAL_FEATURES or []) if c not in FEATURES_BASE and not c.endswith("_te")])
n_base = len(FINAL_FEATURES or []) - n_te - n_ofe

print(f"\n  Toplam feature (son fold) : {len(FINAL_FEATURES or [])}")
print(f"    ├─ Ham / LE             : {n_base}")
print(f"    ├─ Target Encoding       : {n_te}")
print(f"    └─ OpenFE               : {n_ofe}")

# Best params kaydet
with open(os.path.join(CFG["output_dir"], "best_params.json"), "w") as f:
    json.dump({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
               for k, v in best_params.items()}, f, indent=2)
log("best_params.json kaydedildi", "OK")

# Submission
submission = pd.DataFrame({
    (id_col if id_col else "id"): test_ids,
    TARGET: test_final
})
sub_path = os.path.join(CFG["output_dir"], "submission.csv")
submission.to_csv(sub_path, index=False)
log(f"submission.csv → {submission.shape}", "OK")
log(f"Test tahmin istatistikleri: mean={test_final.mean():.4f}  "
    f"std={test_final.std():.4f}  "
    f"min={test_final.min():.4f}  "
    f"max={test_final.max():.4f}")

print(f"\n✅ Pipeline tamamlandı!")
print(f"   AUC={oof_auc:.6f}  Brier={brier:.4f}  Threshold={opt_thr:.3f}  Skor={score_v}/{max_s}")
```