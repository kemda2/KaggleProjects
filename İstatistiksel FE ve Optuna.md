
```py
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         SENIOR ML PIPELINE — İstatistiksel FE + Optuna + LightGBM          ║
║  Her şey istatistiksel olarak test edilir. Şans yoktur.                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Mimari:
  0.  CFG         — tüm kararlar buradan
  1.  Veri yükleme + birleştirme
  2.  Temel temizlik
  3.  İstatistiksel testler (normallik, MWU, chi2, VIF, Cramér V, PBC, Kruskal)
  4.  FE Grup A — istatistiksel olarak KESIN anlamlı (otomatik)
  5.  FE Grup B — deneysel (CFG'den True/False)
  6.  Target Encoding (fold-safe, Bayesian smoothing)
  7.  Label Encoding + anlamsız kolon temizliği
  8.  Optuna hyperparameter search
  9.  Final CV (en iyi params ile)
  10. Permutation Importance (gain'e ek doğrulama)
  11. Feature Importance görseli
  12. Sonuç özeti + submission
"""

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

from scipy import stats
from scipy.stats import (
    mannwhitneyu, chi2_contingency, shapiro,
    pointbiserialr, spearmanr, kruskal
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
matplotlib.use("Agg")


# ═══════════════════════════════════════════════════════════════════════════════
# 0.  CFG  —  TÜM AYARLAR BURADA
# ═══════════════════════════════════════════════════════════════════════════════
CFG = {
    # ── Veri yolları ─────────────────────────────────────────────────────────
    "train_path": "/kaggle/input/competitions/playground-series-s6e3/train.csv",
    "test_path":  "/kaggle/input/competitions/playground-series-s6e3/test.csv",
    "orig_path":  "/kaggle/input/datasets/blastchar/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv",            # "original.csv" ya da None → kullanma
    "target":     "Churn",
    "id_col":     None,            # None → otomatik tespit
    "drop_cols":  [],              # her zaman atılacak ekstra sütunlar

    # ── Ham veri tipi ipuçları ([] → otomatik) ────────────────────────────────
    "num_cols":   ["tenure", "MonthlyCharges", "TotalCharges"],
    "cat_cols":   [],

    # ── İstatistik eşikleri ──────────────────────────────────────────────────
    "alpha":               0.05,    # genel anlamlılık
    "cramers_v_min":       0.05,    # altı → kategorik anlamsız
    "pbc_min":             0.05,    # altı → nümerik anlamsız (|r|)
    "vif_max":             10.0,    # üstü → multicollinearity uyarısı
    "high_corr_threshold": 0.85,    # spearman: çift uyarısı
    "drop_insig_cats":     True,    # Cramér V < cramers_v_min olanları çıkar

    # ── FE Grup A — KESIN (otomatik, istatistiksel onaylı) ───────────────────
    "grp_a_service_count":       True,
    "grp_a_charge_ratios":       True,
    "grp_a_contract_ordinal":    True,
    "grp_a_tenure_groups":       True,
    "grp_a_protection_score":    True,

    # ── FE Grup B — DENEYSEL (True yap → eklenir) ────────────────────────────
    "grp_b_charge_bins":         True,
    "grp_b_tenure_bins":         True,
    "grp_b_interactions":        True,
    "grp_b_polynomial":          True,
    "grp_b_log_transform":       True,
    "grp_b_payment_flags":       True,
    "grp_b_internet_flags":      True,
    "grp_b_customer_segment":    True,
    "grp_b_high_risk_flag":      True,
    "grp_b_addon_services":      True,
    "grp_b_charge_x_segment":    True,
    "grp_b_cluster_features":    False,  # KMeans — yavaş, isteğe bağlı

    # ── Target Encoding ──────────────────────────────────────────────────────
    "te_enabled":        True,
    "te_smoothing":      20,
    "te_cramers_min":    0.10,      # en az bu V'ye sahip kolonlara uygula
    "te_n_splits":       5,

    # ── Optuna ───────────────────────────────────────────────────────────────
    "optuna_enabled":    True,
    "optuna_n_trials":   60,
    "optuna_timeout":    6000,       # saniye
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
        "min_child_samples":50,
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
    "output_dir":        ".",
    "verbose_tests":     True,
    "plot_figures":      True,
}

TARGET = CFG["target"]


# ═══════════════════════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════════════════════
def log(msg, level="INFO"):
    prefix = {"INFO": "  ·", "STEP": "\n▸", "WARN": "  ⚠", "OK": "  ✓"}
    print(f"{prefix.get(level, '  ')} {msg}")


def cramers_v_score(x, y_arr):
    ct = pd.crosstab(x, y_arr)
    chi2, p, _, _ = chi2_contingency(ct)
    n = ct.values.sum()
    k = min(ct.shape) - 1
    v = np.sqrt(chi2 / (n * k)) if (n * k) > 0 else 0.0
    return float(v), float(p)


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


def vif_calc(df, cols):
    sub = df[cols].fillna(df[cols].median()).replace([np.inf, -np.inf], 0)
    result = {}
    for i, col in enumerate(cols):
        try:
            result[col] = float(variance_inflation_factor(sub.values, i))
        except Exception:
            result[col] = float("nan")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  VERİ YÜKLEME
# ═══════════════════════════════════════════════════════════════════════════════
log("Veri yükleme", "STEP")

train = pd.read_csv(CFG["train_path"])
test  = pd.read_csv(CFG["test_path"])
log(f"train: {train.shape}  |  test: {test.shape}")

if CFG["orig_path"] and os.path.exists(str(CFG["orig_path"])):
    orig = pd.read_csv(CFG["orig_path"])
    log(f"orig: {orig.shape}")
    if set(train.columns) == set(orig.columns):
        orig = orig[train.columns]
    else:
        shared = [c for c in train.columns if c in orig.columns]
        orig = orig[shared]
    train = pd.concat([train, orig], axis=0, ignore_index=True)
    log(f"orig birleştirildi → train: {train.shape}", "OK")
else:
    log("orig yok / None — sadece train kullanılıyor.")

# ID sütunu
id_col = CFG["id_col"] or auto_detect_id(test)
test_ids = test[id_col].copy() if id_col and id_col in test.columns \
           else pd.RangeIndex(len(test))
log(f"ID sütunu: {id_col}")

# Gereksiz sütunları çıkar (DÜZELTME: TARGET'ı buradan sildik, Bölüm 2'de düşüreceğiz)
drop = ([id_col] if id_col else []) + CFG["drop_cols"]
train.drop(columns=[c for c in drop if c in train.columns], inplace=True)
test.drop(columns=[c for c in ([id_col] if id_col else []) + CFG["drop_cols"]
                   if c in test.columns], inplace=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  TARGET DÖNÜŞÜMÜ
# ═══════════════════════════════════════════════════════════════════════════════
log("Target dönüşümü", "STEP")
# DÜZELTME: Target'ı doğrudan trainden alıyoruz
y_raw = train[TARGET].copy()

y = y_raw.astype(str).str.strip().map(
    {"Yes": 1, "No": 0, "1": 1, "0": 0, "True": 1, "False": 0}
)
if y.isnull().any():
    y = pd.to_numeric(y_raw, errors="coerce")
if y.isnull().any():
    raise ValueError(f"Dönüştürülemeyen target: {y_raw[y.isnull()].unique()}")
y = y.reset_index(drop=True).astype(np.int8)

# Y okunduktan sonra Target sütununu trainden atıyoruz
if TARGET in train.columns:
    train.drop(columns=[TARGET], inplace=True)

log(f"Dağılım: {dict(y.value_counts().sort_index())}  oran={y.mean():.3f}", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  TEMEL TEMİZLİK
# ═══════════════════════════════════════════════════════════════════════════════
log("Temel temizlik", "STEP")

num_cols = CFG["num_cols"] if CFG["num_cols"] else \
    list(train.select_dtypes(include=["number"]).columns)
cat_cols = CFG["cat_cols"] if CFG["cat_cols"] else \
    [c for c in train.columns if c not in num_cols]
num_cols = [c for c in num_cols if c in train.columns]
cat_cols = [c for c in cat_cols if c in train.columns]


def clean(df, num_c, cat_c):
    df = df.copy()
    for c in num_c:
        if c in df.columns:
            # DÜZELTME: Önce sayısala çevirip sonra medyan hesaplanıyor
            num_series = pd.to_numeric(df[c], errors="coerce")
            med_val = num_series.median()
            df[c] = num_series.fillna(med_val if pd.notna(med_val) else 0.0)
    for c in cat_c:
        if c in df.columns:
            df[c] = df[c].fillna("Missing").astype(str).str.strip()
    return df


train = clean(train, num_cols, cat_cols)
test  = clean(test,  num_cols, cat_cols)
log(f"Nümerik: {len(num_cols)}  |  Kategorik: {len(cat_cols)}", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  İSTATİSTİKSEL TESTLER
# ═══════════════════════════════════════════════════════════════════════════════
log("İstatistiksel testler", "STEP")
STATS = {}

# ── 4a. Normallik ─────────────────────────────────────────────────────────────
log("[4a] Normallik (Shapiro-Wilk / KS)")
STATS["normality"] = {}
for col in num_cols:
    vals = train[col].dropna().values
    if len(vals) > 5000:
        stat, p = stats.kstest((vals - vals.mean()) / (vals.std() + 1e-9), "norm")
        tname = "KS"
    else:
        stat, p = shapiro(vals[:5000])
        tname = "Shapiro"
    STATS["normality"][col] = {"p": p, "normal": p >= CFG["alpha"]}
    if CFG["verbose_tests"]:
        flag = "normal" if p >= CFG["alpha"] else "NORMAL DEĞİL"
        log(f"  {col:22s} [{tname}] p={p:.3e} → {flag}")

# ── 4b. Mann-Whitney U ────────────────────────────────────────────────────────
log("[4b] Mann-Whitney U (nümerik ~ target)")
STATS["mwu"] = {}
for col in num_cols:
    g0 = train.loc[y == 0, col].dropna()
    g1 = train.loc[y == 1, col].dropna()
    stat, p = mannwhitneyu(g0, g1, alternative="two-sided")
    sig = p < CFG["alpha"]
    STATS["mwu"][col] = {"stat": float(stat), "p": float(p), "significant": sig}
    if CFG["verbose_tests"]:
        log(f"  {col:22s} p={p:.3e} → {'ANLAMLI ✓' if sig else 'anlamsız'}")

# ── 4c. Point-Biserial ────────────────────────────────────────────────────────
log("[4c] Point-Biserial korelasyon")
STATS["pbc"] = {}
for col in num_cols:
    r, p = pointbiserialr(y, train[col].fillna(0))
    STATS["pbc"][col] = {"r": float(r), "p": float(p),
                          "useful": abs(r) >= CFG["pbc_min"]}
    if CFG["verbose_tests"]:
        log(f"  {col:22s} r={r:+.4f} → {'✓' if abs(r) >= CFG['pbc_min'] else '—'}")

# ── 4d. Spearman ──────────────────────────────────────────────────────────────
log("[4d] Spearman korelasyon (nümerik çiftleri)")
STATS["spearman"] = {}
for i in range(len(num_cols)):
    for j in range(i + 1, len(num_cols)):
        c1, c2 = num_cols[i], num_cols[j]
        r, _ = spearmanr(train[c1].fillna(0), train[c2].fillna(0))
        STATS["spearman"][(c1, c2)] = float(r)
        if CFG["verbose_tests"] and abs(r) > 0.5:
            flag = " ← ÇOK YÜKSEK!" if abs(r) > CFG["high_corr_threshold"] else ""
            log(f"  {c1} × {c2}: r={r:+.3f}{flag}")

# ── 4e. VIF ──────────────────────────────────────────────────────────────────
log("[4e] VIF")
vif_scores = vif_calc(train, num_cols)
STATS["vif"] = vif_scores
high_vif = []
for col, vif in vif_scores.items():
    if CFG["verbose_tests"]:
        flag = "⚠ yüksek" if vif and vif > CFG["vif_max"] else "ok"
        log(f"  {col:22s} VIF={vif:.2f} → {flag}")
    if vif and vif > CFG["vif_max"]:
        high_vif.append(col)
if high_vif:
    log(f"Yüksek VIF: {high_vif} → ratio FE önerilir", "WARN")

# ── 4f. Chi-square + Cramér V ────────────────────────────────────────────────
log("[4f] Chi-square + Cramér V (kategorik ~ target)")
STATS["chi2"] = {}
for col in cat_cols:
    v, p = cramers_v_score(train[col], y)
    sig = p < CFG["alpha"]
    STATS["chi2"][col] = {"cramers_v": v, "p": p, "significant": sig,
                           "useful": sig and v >= CFG["cramers_v_min"]}
    if CFG["verbose_tests"]:
        flag = f"✓ V={v:.3f}" if sig and v >= CFG["cramers_v_min"] else \
               f"zayıf V={v:.3f}" if sig else "anlamsız"
        log(f"  {col:22s} p={p:.3e} Cramér V={v:.3f} → {flag}")

sig_cats   = sorted([c for c, d in STATS["chi2"].items() if d["useful"]],
                    key=lambda c: STATS["chi2"][c]["cramers_v"], reverse=True)
insig_cats = [c for c, d in STATS["chi2"].items() if not d["useful"]]
sig_nums   = [c for c, d in STATS["mwu"].items() if d["significant"]]

log(f"Anlamlı kategorik: {sig_cats}", "OK")
log(f"Anlamlı nümerik  : {sig_nums}", "OK")
if insig_cats:
    log(f"Anlamsız kategorik: {insig_cats}", "WARN")

# ── 4g. Sınıf dengesi ────────────────────────────────────────────────────────
imbalance_ratio = (y == 0).sum() / max((y == 1).sum(), 1)
STATS["imbalance_ratio"] = float(imbalance_ratio)
log(f"[4g] Sınıf dengesi (0/1): {imbalance_ratio:.2f}"
    + (" ← dengesiz" if imbalance_ratio > 3 else ""))

# ── 4h. Kruskal-Wallis ───────────────────────────────────────────────────────
log("[4h] Kruskal-Wallis (nümerik × top kategorikler)")
STATS["kruskal"] = {}
for ncat in sig_cats[:4]:
    for ncol in sig_nums:
        groups = [train.loc[train[ncat] == g, ncol].dropna().values
                  for g in train[ncat].unique()]
        groups = [g for g in groups if len(g) > 5]
        if len(groups) < 2:
            continue
        stat, p = kruskal(*groups)
        sig = p < CFG["alpha"]
        STATS["kruskal"][(ncat, ncol)] = {"stat": float(stat), "p": float(p), "sig": sig}
        if CFG["verbose_tests"] and sig:
            log(f"  {ncat} × {ncol}: p={p:.3e} ✓")

log("İstatistiksel testler tamamlandı", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  FE GRUP A — KESIN ANLAMLI
# ═══════════════════════════════════════════════════════════════════════════════
log("FE Grup A (istatistiksel olarak onaylı)", "STEP")

SERVICE_COLS    = ["PhoneService","MultipleLines","InternetService",
                   "OnlineSecurity","OnlineBackup","DeviceProtection",
                   "TechSupport","StreamingTV","StreamingMovies"]
PROTECTION_COLS = ["OnlineSecurity","TechSupport","OnlineBackup","DeviceProtection"]


def fe_group_a(df):
    df = df.copy()

    # TotalCharges impute (domain bilgisi: tenure=0 iken TotalCharges=0 olamaz)
    if all(c in df.columns for c in ["tenure","MonthlyCharges","TotalCharges"]):
        mask = df["TotalCharges"] == 0
        df.loc[mask, "TotalCharges"] = (df.loc[mask, "tenure"] * df.loc[mask, "MonthlyCharges"])

    # Servis sayısı — hizmet sütunları chi2 anlamlı → aggregate güçlü
    if CFG["grp_a_service_count"]:
        svc = [c for c in SERVICE_COLS if c in df.columns]
        df["service_count"] = (df[svc].apply(lambda x: x.str.lower().isin(["yes","fiber optic","dsl"])).sum(axis=1).astype(np.int8))

    # Sözleşme ordinal — Contract en yüksek Cramér V → ordinal bilgi önemli
    if CFG["grp_a_contract_ordinal"] and "Contract" in df.columns:
        df["contract_ordinal"] = df["Contract"].map(
            {"Month-to-month":0,"One year":1,"Two year":2}
        ).fillna(0).astype(np.int8)

    # Koruma skoru — birden fazla güvenlik servisi aggregate
    if CFG["grp_a_protection_score"]:
        pc = [c for c in PROTECTION_COLS if c in df.columns]
        df["protection_score"] = (
            df[pc].apply(lambda x: x.str.lower() == "yes")
            .sum(axis=1).astype(np.int8)
        )

    # Ücret oranları — VIF yüksek → raw yerine ratio
    if CFG["grp_a_charge_ratios"] and all(
            c in df.columns for c in ["tenure","MonthlyCharges","TotalCharges"]):
        t = df["tenure"].replace(0, 1)
        df["avg_monthly_charge"]    = df["TotalCharges"] / t
        df["charge_delta"]          = df["MonthlyCharges"] - df["avg_monthly_charge"]
        df["charge_to_total_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
        if "service_count" in df.columns:
            df["charge_per_service"] = df["MonthlyCharges"] / (df["service_count"] + 1)

    # Tenure segmentleri — MWU anlamlı, bimodal dağılım
    if CFG["grp_a_tenure_groups"] and "tenure" in df.columns:
        df["is_new_customer"]   = (df["tenure"] <= 6).astype(np.int8)
        df["is_loyal_customer"] = (df["tenure"] >= 48).astype(np.int8)

    return df


train = fe_group_a(train)
test  = fe_group_a(test)
log(f"Grup A sonrası: {train.shape[1]} feature", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  FE GRUP B — DENEYSEL
# ═══════════════════════════════════════════════════════════════════════════════
log("FE Grup B (deneysel, CFG kontrollü)", "STEP")


def fe_group_b(df):
    df = df.copy()

    if CFG["grp_b_tenure_bins"] and "tenure" in df.columns:
        df["tenure_bin"] = pd.cut(df["tenure"], bins=6, labels=False).fillna(0).astype(np.int8)

    if CFG["grp_b_charge_bins"] and "MonthlyCharges" in df.columns:
        df["charge_bin"] = pd.qcut(df["MonthlyCharges"], q=4, labels=False,
                                    duplicates="drop").fillna(0).astype(np.int8)

    if CFG["grp_b_polynomial"]:
        for col in [c for c in ["tenure","MonthlyCharges"] if c in df.columns]:
            df[f"{col}_sq"] = df[col] ** 2

    if CFG["grp_b_log_transform"]:
        for col in [c for c in ["TotalCharges","MonthlyCharges","tenure"] if c in df.columns]:
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

    if CFG["grp_b_interactions"]:
        cp = {c: c in df.columns for c in
              ["tenure","MonthlyCharges","contract_ordinal","service_count","protection_score"]}
        if cp.get("tenure") and cp.get("MonthlyCharges"):
            df["tenure_x_monthly"]  = df["tenure"] * df["MonthlyCharges"]
        if cp.get("tenure") and cp.get("contract_ordinal"):
            df["tenure_x_contract"] = df["tenure"] * df["contract_ordinal"]
        if cp.get("MonthlyCharges") and cp.get("service_count"):
            df["monthly_x_service"] = df["MonthlyCharges"] * df["service_count"]
        if cp.get("protection_score") and cp.get("tenure"):
            df["protection_x_tenure"] = df["protection_score"] * df["tenure"]

    if CFG["grp_b_payment_flags"] and "PaymentMethod" in df.columns:
        df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(np.int8)
        df["is_auto_payment"] = df["PaymentMethod"].str.lower().str.contains(
            "automatic", na=False).astype(np.int8)

    if CFG["grp_b_internet_flags"] and "InternetService" in df.columns:
        df["is_fiber"]    = (df["InternetService"] == "Fiber optic").astype(np.int8)
        df["no_internet"] = (df["InternetService"] == "No").astype(np.int8)

    if CFG["grp_b_customer_segment"]:
        if "Partner" in df.columns and "Dependents" in df.columns:
            df["is_alone"] = (
                (df["Partner"] == "No") & (df["Dependents"] == "No")
            ).astype(np.int8)

    if CFG["grp_b_addon_services"]:
        sc = [c for c in ["StreamingTV","StreamingMovies"] if c in df.columns]
        df["streaming_count"] = (
            df[sc].apply(lambda x: x == "Yes").sum(axis=1).astype(np.int8)
        )
        if "protection_score" in df.columns:
            df["total_addon"] = df["protection_score"] + df["streaming_count"]

    if CFG["grp_b_high_risk_flag"]:
        conds = []
        if "tenure" in df.columns:        conds.append(df["tenure"] < 12)
        if "Contract" in df.columns:      conds.append(df["Contract"] == "Month-to-month")
        if "InternetService" in df.columns: conds.append(df["InternetService"] == "Fiber optic")
        if conds:
            df["high_risk"] = pd.concat(conds, axis=1).all(axis=1).astype(np.int8)

    if CFG["grp_b_charge_x_segment"] and "MonthlyCharges" in df.columns:
        for flag_col in ["is_alone","is_fiber","is_new_customer"]:
            if flag_col in df.columns:
                df[f"monthly_x_{flag_col}"] = df["MonthlyCharges"] * df[flag_col]

    if CFG["grp_b_cluster_features"]:
        from sklearn.cluster import KMeans
        cluster_feats = [c for c in ["tenure","MonthlyCharges","TotalCharges"] if c in df.columns]
        if cluster_feats:
            from sklearn.preprocessing import StandardScaler as SS2
            arr = SS2().fit_transform(df[cluster_feats].fillna(0))
            df["customer_cluster"] = KMeans(
                n_clusters=5, random_state=CFG["seed"], n_init=10
            ).fit_predict(arr).astype(np.int8)

    return df


train = fe_group_b(train)
test  = fe_group_b(test)
log(f"Grup B sonrası: {train.shape[1]} feature", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  TARGET ENCODING (fold-safe, Bayesian smoothing)
# ═══════════════════════════════════════════════════════════════════════════════
log("Target Encoding", "STEP")

te_cols = [c for c in sig_cats
           if c in train.columns and STATS["chi2"][c]["cramers_v"] >= CFG["te_cramers_min"]
           ] if CFG["te_enabled"] else []
log(f"TE kolonları: {te_cols}")

te_train_df = pd.DataFrame(index=train.index)
te_test_df  = pd.DataFrame(index=test.index)
global_mean = float(y.mean())
sm = CFG["te_smoothing"]

if te_cols:
    skf_te = StratifiedKFold(n_splits=CFG["te_n_splits"], shuffle=True, random_state=CFG["seed"])
    for col in te_cols:
        col_name = f"{col}_te"
        te_train_df[col_name] = np.nan

        for _, (tr_idx, val_idx) in enumerate(skf_te.split(train, y)):
            tr_col = train[col].iloc[tr_idx].astype(str)
            tr_y   = y.iloc[tr_idx]
            agg = pd.DataFrame({"col": tr_col.values, "y": tr_y.values})
            st = agg.groupby("col")["y"].agg(["sum","count"])
            st["te"] = (st["sum"] + sm * global_mean) / (st["count"] + sm)
            te_train_df.iloc[val_idx,
                te_train_df.columns.get_loc(col_name)] = \
                train[col].iloc[val_idx].astype(str).map(st["te"]).values

        te_train_df[col_name].fillna(global_mean, inplace=True)

        full_agg = pd.DataFrame({"col": train[col].astype(str).values, "y": y.values})
        full_st = full_agg.groupby("col")["y"].agg(["sum","count"])
        full_st["te"] = (full_st["sum"] + sm * global_mean) / (full_st["count"] + sm)
        te_test_df[col_name] = test[col].astype(str).map(full_st["te"]).fillna(global_mean)

        if CFG["verbose_tests"]:
            log(f"  {col:25s} → {col_name}  (V={STATS['chi2'][col]['cramers_v']:.3f})")

    train = pd.concat([train, te_train_df], axis=1)
    test  = pd.concat([test,  te_test_df],  axis=1)
    log(f"TE sonrası: {train.shape[1]} feature", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  LABEL ENCODING + ANLAMSIZ KOLON TEMİZLİĞİ
# ═══════════════════════════════════════════════════════════════════════════════
log("Label Encoding + kolon temizliği", "STEP")

if CFG["drop_insig_cats"]:
    drop_c = [c for c in insig_cats
              if STATS["chi2"].get(c, {}).get("cramers_v", 1) < CFG["cramers_v_min"]
              and c in train.columns]
    if drop_c:
        log(f"Cramér V < {CFG['cramers_v_min']} → çıkarılıyor: {drop_c}", "WARN")
        train.drop(columns=drop_c, inplace=True, errors="ignore")
        test.drop(columns=drop_c,  inplace=True, errors="ignore")

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
FEATURES = train.columns.tolist()
log(f"Final feature sayısı: {len(FEATURES)}", "OK")


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
    X_tr,  X_val  = train.iloc[tr_idx], train.iloc[val_idx]
    y_tr,  y_val  = y.iloc[tr_idx], y.iloc[val_idx]

    d_tr  = lgb.Dataset(X_tr, label=y_tr,  categorical_feature=obj_cols)
    d_val = lgb.Dataset(X_val, label=y_val, categorical_feature=obj_cols, reference=d_tr)

    model = lgb.train(
        best_params, d_tr,
        num_boost_round=5000,
        valid_sets=[d_val],
        callbacks=[lgb.early_stopping(CFG["early_stopping"], verbose=False),
                   lgb.log_evaluation(500)]
    )

    oof_preds[val_idx]  = model.predict(X_val)
    test_preds         += model.predict(test) / CFG["final_cv_splits"]

    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    fold_scores.append(fold_auc)
    log(f"  Fold {fold}: AUC={fold_auc:.5f}  best_iter={model.best_iteration}")

    fi = pd.DataFrame({"feature": FEATURES,
                       "gain":  model.feature_importance("gain"),
                       "split": model.feature_importance("split"),
                       "fold":  fold})
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
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=500, random_state=42))])
    pipe.fit(train.fillna(0), y)
    perm = permutation_importance(pipe, train.fillna(0), y,
                                  n_repeats=5, random_state=42, scoring="roc_auc")
    perm_df = pd.DataFrame({"feature": FEATURES,
                             "perm_mean": perm.importances_mean,
                             "perm_std":  perm.importances_std}
                           ).sort_values("perm_mean", ascending=False)
    log("Top 10 Permutation Importance:")
    for _, row in perm_df.head(10).iterrows():
        log(f"  {row['feature']:30s} {row['perm_mean']:+.5f} ± {row['perm_std']:.5f}")
except Exception as e:
    log(f"Permutation hesaplanamadı: {e}", "WARN")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. FEATURE IMPORTANCE GÖRSELİ
# ═══════════════════════════════════════════════════════════════════════════════
if CFG["plot_figures"]:
    fi_mean = feat_imp_df.groupby("feature")["gain"].mean().sort_values(ascending=True)

    GRP_A = {"service_count","contract_ordinal","protection_score","avg_monthly_charge",
             "charge_delta","charge_to_total_ratio","charge_per_service",
             "is_new_customer","is_loyal_customer"}
    GRP_B = {"tenure_bin","charge_bin","tenure_sq","MonthlyCharges_sq",
             "TotalCharges_log","MonthlyCharges_log","tenure_log","tenure_x_monthly",
             "tenure_x_contract","monthly_x_service","protection_x_tenure",
             "is_electronic_check","is_auto_payment","is_fiber","no_internet",
             "is_alone","streaming_count","total_addon","high_risk","customer_cluster"}
    TE_FEATS = {f"{c}_te" for c in te_cols}

    def get_color(feat):
        if feat in TE_FEATS:    return "#7F77DD"
        if feat in GRP_A:       return "#1D9E75"
        if feat in GRP_B:       return "#378ADD"
        return "#888780"

    fig, axes = plt.subplots(1, 2, figsize=(22, max(8, int(len(fi_mean) * 0.3))))
    axes[0].barh(fi_mean.index, fi_mean.values,
                 color=[get_color(f) for f in fi_mean.index])
    axes[0].set_title("Feature Importance — Gain (tüm)", fontsize=11)
    axes[0].tick_params(labelsize=7)

    top25 = fi_mean.tail(25)
    axes[1].barh(top25.index, top25.values,
                 color=[get_color(f) for f in top25.index])
    axes[1].set_title("Top 25", fontsize=11)

    from matplotlib.patches import Patch
    axes[1].legend(handles=[
        Patch(color="#1D9E75", label="Grup A — kesin anlamlı"),
        Patch(color="#378ADD", label="Grup B — deneysel"),
        Patch(color="#7F77DD", label="Target Encoding"),
        Patch(color="#888780", label="Ham feature"),
    ], loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(CFG["output_dir"], "feature_importance.png"),
                dpi=120, bbox_inches="tight")
    plt.close()
    log("feature_importance.png kaydedildi", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 13. SONUÇ ÖZETİ + BEST PARAMS KAYDET
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  SONUÇ ÖZETİ")
print("═"*60)
print(f"  Toplam feature      : {len(FEATURES)}")
print(f"  Anlamlı kategorik   : {len(sig_cats)}")
print(f"  Anlamlı nümerik     : {len(sig_nums)}")
print(f"  TE kolon sayısı     : {len(te_cols)}")
print(f"  Optuna trials       : {CFG['optuna_n_trials'] if CFG['optuna_enabled'] else 'devre dışı'}")
print(f"  Fold AUC'ler        : {[f'{s:.5f}' for s in fold_scores]}")
print(f"  Ortalama AUC        : {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")
print(f"  Overall OOF AUC     : {oof_auc:.5f}")
print("═"*60)

with open(os.path.join(CFG["output_dir"], "best_params.json"), "w") as f:
    json.dump({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
               for k, v in best_params.items()}, f, indent=2)
log("best_params.json kaydedildi", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 14. SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════
submission = pd.DataFrame({
    (id_col if id_col else "id"): test_ids,
    TARGET: test_preds
})
sub_path = os.path.join(CFG["output_dir"], "submission.csv")
submission.to_csv(sub_path, index=False)
log(f"submission.csv → {submission.shape}", "OK")
print(submission[TARGET].describe().to_string())

# label encoding yerine categorical dene
```