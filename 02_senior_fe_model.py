# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DOSYA 2 — SENIOR FE + MODEL + DEĞERLENDİRME PİPELİNE                    ║
# ║  Playground Series S6E3 · Telco Customer Churn                              ║
# ║  ÖNCE Dosya 1'i çalıştır → eda_decisions.json oluşsun.                    ║
# ║  Çıktı: submission.csv + /kaggle/working/model_report.html                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import subprocess, sys, os, warnings, json, base64, io
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy import stats
from scipy.stats import ks_2samp, rankdata, mcnemar
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, brier_score_loss, roc_curve,
                              precision_recall_curve, confusion_matrix, average_precision_score)
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    import shap as shap_lib
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ── Görsel tema ─────────────────────────────────────────────────────────────────
DARK  = "#0d1117"; CARD  = "#161b22"; BLUE  = "#58a6ff"
TXT   = "#c9d1d9"; GRID  = "#30363d"; RED   = "#f85149"
GREEN = "#3fb950"; ORANGE= "#f0883e"; GOLD  = "#e3b341"
PURP  = "#bc8cff"

PLOT_DIR = Path("/kaggle/working/model_plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def dark_ax(ax, title="", xl="", yl=""):
    ax.set_facecolor(CARD); ax.figure.patch.set_facecolor(DARK)
    ax.title.set_color(BLUE); ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT); ax.tick_params(colors=TXT, labelsize=8)
    for s in ax.spines.values(): s.set_color(GRID)
    if title: ax.set_title(title, color=BLUE, fontsize=10, fontweight="bold", pad=7)
    if xl: ax.set_xlabel(xl, color=TXT)
    if yl: ax.set_ylabel(yl, color=TXT)

def save_fig(fig, name):
    path = PLOT_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=DARK, edgecolor="none")
    plt.close(fig)
    return path

# ── HTML Reporter ────────────────────────────────────────────────────────────────
class ModelReporter:
    CSS = """<html><head><meta charset="utf-8"><style>
body{font-family:'Segoe UI',monospace;background:#0a0e1a;color:#c9d1d9;max-width:1400px;margin:0 auto;padding:0}
.hero{background:#0d1117;border-bottom:2px solid #30363d;padding:36px 40px 28px;text-align:center}
.hero h1{font-size:1.9rem;font-weight:700;color:#58a6ff;margin-bottom:6px}
.hero p{color:#8b949e;font-size:.9rem}
.content{padding:0 40px 48px}
.sec{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:22px;margin:18px 0}
.sec-hdr{padding-bottom:11px;border-bottom:1px solid #30363d;margin-bottom:14px}
.sec-hdr h2{color:#58a6ff;font-size:.98rem;font-weight:600;margin:0}
.rule{background:#0a1f0f;border-left:3px solid #3fb950;padding:8px 13px;margin:7px 0;border-radius:0 6px 6px 0;font-size:.81rem}
.rule-lbl{color:#3fb950;font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.4px}
.dec{display:flex;gap:10px;align-items:flex-start;padding:10px 14px;border-radius:8px;margin:8px 0;font-size:.83rem}
.dec-ok{background:#0d2818;border:1px solid #3fb950}.dec-ok .di{color:#3fb950}
.dec-warn{background:#1a1505;border:1px solid #f0883e}.dec-warn .di{color:#f0883e}
.dec-crit{background:#2d1014;border:1px solid #f85149}.dec-crit .di{color:#f85149}
.dl{font-size:.68rem;font-weight:800;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px}
.dec-ok .dl{color:#3fb950}.dec-warn .dl{color:#f0883e}.dec-crit .dl{color:#f85149}
.log{font-family:monospace;font-size:.78rem;padding:5px 11px;border-radius:4px;margin:3px 0;border-left:3px solid #30363d;background:#0d1117;color:#8b949e}
.log-ok{border-left-color:#3fb950;color:#3fb950}.log-warn{border-left-color:#f0883e;color:#f0883e}
.log-crit{border-left-color:#f85149;color:#f85149;font-weight:600}
table{border-collapse:collapse;width:100%;margin:11px 0;font-size:.79rem}
th{background:#21262d;color:#58a6ff;padding:8px 11px;border:1px solid #30363d;text-align:left;font-size:.75rem}
td{padding:6px 11px;border:1px solid #21262d;color:#c9d1d9}
tr:hover td{background:#0d1117}
.plot-wrap{margin:13px 0;text-align:center}
.plot-wrap img{max-width:100%;border-radius:8px;border:1px solid #30363d}
.plot-cap{font-size:.71rem;color:#8b949e;margin-top:5px;font-family:monospace}
.metrics{display:flex;gap:10px;flex-wrap:wrap;margin:11px 0}
.mc{background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:11px 15px;min-width:110px;text-align:center}
.mc b{display:block;font-size:1.25rem;font-family:monospace;color:#58a6ff}
.mc span{font-size:.69rem;color:#8b949e}
.mc.ok b{color:#3fb950}.mc.warn b{color:#f0883e}.mc.crit b{color:#f85149}
.score-bar-bg{background:#21262d;border-radius:6px;height:14px;overflow:hidden;margin:8px 0}
.score-bar-fill{height:100%;border-radius:6px}
</style></head><body>
<div class="hero"><h1>&#129302; Model Pipeline Raporu — Telco Churn</h1>
<p>Playground Series S6E3 &nbsp;·&nbsp; FE, Encoding, Model, Kalibrasyon, Threshold, SHAP</p></div>
<div class="content">"""

    def __init__(self):
        self._h = [self.CSS]
        self._decisions = []
        self._pc = 0

    def section(self, title):
        print(f"\n{'═'*70}\n{title}\n{'═'*70}")
        self._h.append(f'<div class="sec"><div class="sec-hdr"><h2>{title}</h2></div>')

    def end(self):  self._h.append("</div>")

    def rule(self, text):
        self._h.append(f'<div class="rule"><div class="rule-lbl">&#128204; Kural</div>{text}</div>')

    def log(self, text, cls="info"):
        pfx = {"info":"ℹ","ok":"✅","warn":"⚠️","crit":"🔴"}.get(cls,"ℹ")
        print(f"  {pfx} {text}")
        css = {"info":"","ok":"log-ok","warn":"log-warn","crit":"log-crit"}.get(cls,"")
        self._h.append(f'<div class="log {css}">{pfx} {text}</div>')

    def decision(self, text, status="ok"):
        icons={"ok":"✅","warn":"⚠️","crit":"❌"}
        css={"ok":"dec-ok","warn":"dec-warn","crit":"dec-crit"}
        lbl={"ok":"DEVAM","warn":"DİKKAT","crit":"KRİTİK"}
        print(f"  {icons[status]} KARAR: {text}")
        self._h.append(f'<div class="dec {css[status]}"><div class="di">{icons[status]}</div>'
                       f'<div><div class="dl">{lbl[status]}</div>{text}</div></div>')
        self._decisions.append({"status":status,"text":text})

    def metrics(self, items):
        self._h.append('<div class="metrics">')
        for lbl, val, cls in items:
            self._h.append(f'<div class="mc {cls}"><b>{val}</b><span>{lbl}</span></div>')
        self._h.append("</div>")

    def table(self, df, caption=""):
        self._h.append(df.to_html(index=False,border=0,classes="",na_rep="—",escape=False))
        if caption:
            self._h.append(f'<p style="font-size:.71rem;color:#8b949e;margin:3px 0 10px">{caption}</p>')

    def plot(self, fig, name, caption=""):
        self._pc += 1
        fname = f"{self._pc:02d}_{name}"
        path = save_fig(fig, fname)
        with open(path,"rb") as f: b64=base64.b64encode(f.read()).decode()
        cap = caption or name.replace("_"," ").title()
        self._h.append(f'<div class="plot-wrap"><img src="data:image/png;base64,{b64}">'
                       f'<div class="plot-cap">{cap}</div></div>')

    def save(self, path="/kaggle/working/model_report.html"):
        self._h.append("</div></body></html>")
        Path(path).write_text("\n".join(self._h), encoding="utf-8")
        print(f"\n✅ HTML raporu: {path}")

R = ModelReporter()

# ════════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1 — EDA Kararlarını Yükle
# ════════════════════════════════════════════════════════════════════════════════
R.section("📋 BÖLÜM 1 — EDA Kararlarını Yükle")

EDA_PATH = "/kaggle/working/eda_decisions.json"
if Path(EDA_PATH).exists():
    with open(EDA_PATH) as f:
        EDA = json.load(f)
    R.log("eda_decisions.json başarıyla yüklendi.", "ok")
else:
    R.log("eda_decisions.json bulunamadı! Varsayılan değerler kullanılıyor.", "warn")
    EDA = {
        "drop_features":[], "mnar_features":[], "high_psi_features":[],
        "conflict_rate":0.0, "label_smoothing":False, "sample_weight":False,
        "ood_cat_features":[], "ood_num_features":[], "sentinel_cols":[],
        "cohort_feature":False, "leakage_features":[], "suspect_features":[],
        "imbalance_ratio":3.44, "scale_pos_weight":3.44,
        "high_iv_features":[], "low_iv_features":[], "nonlinear_features":[],
        "high_vif_features":[], "temporal_pattern":False,
        "test_distribution_ok":True, "notes":[]
    }

for key, val in EDA.items():
    if val and val != 0 and val is not False and val != []:
        R.log(f"  EDA kararı → {key}: {val}", "info")

SEED     = int(EDA.get("best_seed_from_eda", 42))
N_SPLITS = 5
N_SEEDS  = 3
TARGET   = "Churn"
IDC      = "id"
FN_COST  = 5.0   # Kaçan churner maliyeti (FP'den 5x pahalı)
FP_COST  = 1.0   # Gereksiz müdahale maliyeti
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2 — Veri Yükleme & EDA Temizliği
# ════════════════════════════════════════════════════════════════════════════════
R.section("🔄 BÖLÜM 2 — Veri Yükleme & EDA Temizliği")

train = pd.read_csv("/kaggle/input/playground-series-s6e3/train.csv", index_col=IDC)
test  = pd.read_csv("/kaggle/input/playground-series-s6e3/test.csv",  index_col=IDC)

# Target encode
train[TARGET] = train[TARGET].map({"Yes":1,"No":0})

# SeniorCitizen fix
for df in [train, test]:
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

# Leakage feature drop
if EDA["leakage_features"]:
    train.drop(columns=EDA["leakage_features"], errors="ignore", inplace=True)
    test.drop( columns=EDA["leakage_features"], errors="ignore", inplace=True)
    R.log(f"Leak feature'lar drop edildi: {EDA['leakage_features']}", "crit")

# Drop features (sabit vb.)
if EDA["drop_features"]:
    drop_f = [c for c in EDA["drop_features"] if c not in EDA["leakage_features"]]
    train.drop(columns=drop_f, errors="ignore", inplace=True)
    test.drop( columns=drop_f, errors="ignore", inplace=True)
    R.log(f"Diğer drop feature'lar: {drop_f}", "warn")

# Sentinel değerlerini 'No' ile değiştir (EDA ADIM 5 kararı)
SENTINEL_REPLACE = ["No internet service","No phone service","Unknown","N/A"]
for df in [train, test]:
    for col in df.select_dtypes(["object"]).columns:
        df[col] = df[col].replace(SENTINEL_REPLACE, "No")
R.log("Sentinel değerler 'No' ile değiştirildi.", "ok")

# TotalCharges string → numeric
for df in [train, test]:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(0, inplace=True)

# OOD sayısal → Winsorize
for col in EDA.get("ood_num_features", []):
    if col in train.columns:
        lo, hi = train[col].quantile(.01), train[col].quantile(.99)
        for df in [train, test]:
            df[col] = df[col].clip(lo, hi)
        R.log(f"Winsorize: {col} → [{lo:.2f}, {hi:.2f}]", "warn")

# MNAR flagları
for col in EDA.get("mnar_features", []):
    if col in train.columns:
        train[f"is_null_{col}"] = train[col].isnull().astype(int)
        test[ f"is_null_{col}"] = test[col].isnull().astype(int)
        R.log(f"MNAR flag oluşturuldu: is_null_{col}", "ok")

R.log(f"Temizleme sonrası — Train: {train.shape}   Test: {test.shape}", "ok")
Y = train[TARGET]
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3 — Feature Engineering (EDA bulgularına referanslı)
# ════════════════════════════════════════════════════════════════════════════════
R.section("⚙️ BÖLÜM 3 — Feature Engineering")
R.rule("Her FE adımının yanında hangi EDA bulgusuna dayandığı belirtilmiştir.")

def feature_engineering(df, fit_df=None):
    df = df.copy()
    ref = fit_df if fit_df is not None else df

    # ── EDA ADIM 9: tenure en güçlü sinyal (Cohen d=1.19) ──────────────────────
    df["is_new_customer"] = (df["tenure"] <= 3).astype("int8")
    df["is_first_year"]   = (df["tenure"] <= 12).astype("int8")
    df["is_loyal"]        = (df["tenure"] >= 48).astype("int8")
    df["log_tenure"]      = np.log1p(df["tenure"]).astype("float32")
    df["tenure_sq"]       = (df["tenure"] ** 2).astype("int32")

    # ── EDA ADIM 18: Cohort drift > 10 puan → tenure_bin ──────────────────────
    if EDA.get("cohort_feature", False):
        df["tenure_group"] = pd.cut(
            df["tenure"], bins=[0,3,12,24,48,72],
            labels=[0,1,2,3,4]).astype("float32")

    # ── EDA ADIM 11: tenure↔TotalCharges r=0.77 → ratio ile decompose ─────────
    df["avg_monthly_charge"]  = (df["TotalCharges"] / (df["tenure"]+1)).astype("float32")
    df["charge_deviation"]    = (df["MonthlyCharges"] - df["avg_monthly_charge"]).astype("float32")
    df["expected_total"]      = (df["MonthlyCharges"] * df["tenure"]).astype("float32")
    df["total_charge_diff"]   = (df["TotalCharges"] - df["expected_total"]).astype("float32")
    df["charges_residual"]    = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")

    # ── EDA ADIM 8: Non-normal → log transform ─────────────────────────────────
    df["log_total_charges"]   = np.log1p(df["TotalCharges"]).astype("float32")
    df["log_monthly"]         = np.log1p(df["MonthlyCharges"]).astype("float32")

    # ── EDA ADIM 10: Electronic check %48.9 churn ──────────────────────────────
    df["is_electronic_check"] = (df["PaymentMethod"]=="Electronic check").astype("int8")
    df["is_auto_pay"]         = df["PaymentMethod"].isin(
        ["Credit card (automatic)","Bank transfer (automatic)"]).astype("int8")

    # ── EDA ADIM 10: MTM %42.1 churn, Two year %1.0 ───────────────────────────
    df["is_mtm"]           = (df["Contract"]=="Month-to-month").astype("int8")
    df["contract_ordinal"] = df["Contract"].map(
        {"Month-to-month":0,"One year":1,"Two year":2}).fillna(0).astype("int8")

    # ── EDA ADIM 10: Fiber %41.5 churn ─────────────────────────────────────────
    df["is_fiber"]          = (df["InternetService"]=="Fiber optic").astype("int8")
    df["has_internet"]      = (df["InternetService"]!="No").astype("int8")

    # ── EDA ADIM 10: OnlineSecurity=No %40.6, TechSupport=No %40.2 ─────────────
    df["has_no_protection"] = (
        (df["OnlineSecurity"]!="Yes") &
        (df["TechSupport"]!="Yes") &
        (df["InternetService"]!="No")
    ).astype("int8")

    # ── Servis sayımları ────────────────────────────────────────────────────────
    online_svcs = ["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport"]
    streaming   = ["StreamingTV","StreamingMovies"]
    all_svcs    = ["PhoneService"] + online_svcs + streaming
    df["service_count"]        = (df[all_svcs]=="Yes").sum(axis=1).astype("int8")
    df["online_service_count"] = (df[online_svcs]=="Yes").sum(axis=1).astype("int8")
    df["streaming_count"]      = (df[streaming]=="Yes").sum(axis=1).astype("int8")
    df["n_security"]           = ((df["OnlineSecurity"]=="Yes").astype(int) +
                                   (df["DeviceProtection"]=="Yes").astype(int) +
                                   (df["TechSupport"]=="Yes").astype(int))

    # ── EDA ADIM 10: SeniorCitizen=1 → %50 churn ───────────────────────────────
    df["senior_alone"] = (
        (df["SeniorCitizen"]==1) &
        (df["Partner"]=="No") &
        (df["Dependents"]=="No")
    ).astype("int8")
    df["senior_monthly"] = (df["SeniorCitizen"] * df["MonthlyCharges"]).astype("float32")
    df["family_size"] = (
        (df["Partner"]=="Yes").astype(int) +
        (df["Dependents"]=="Yes").astype(int)
    ).astype("int8")

    # ── Interaction feature'lar ─────────────────────────────────────────────────
    df["mtm_electronic"]   = (df["is_mtm"] * df["is_electronic_check"]).astype("int8")
    df["fiber_no_support"] = (df["is_fiber"] * df["has_no_protection"]).astype("int8")
    df["epay_paperless"]   = (
        (df["PaperlessBilling"]=="Yes").astype(int) * df["is_electronic_check"]
    ).astype("int8")
    df["contract_tenure"]  = (df["contract_ordinal"] * df["tenure"]).astype("int16")
    df["contract_monthly"] = (df["contract_ordinal"] * df["MonthlyCharges"]).astype("float32")

    # ── Oran feature'lar ────────────────────────────────────────────────────────
    df["monthly_per_service"]  = (df["MonthlyCharges"] / (df["service_count"]+1)).astype("float32")
    df["total_per_service"]    = (df["TotalCharges"]   / (df["service_count"]+1)).astype("float32")
    df["monthly_tenure_ratio"] = (df["MonthlyCharges"] / (df["tenure"]+1)).astype("float32")

    # ── Ultra risk segmenti ─────────────────────────────────────────────────────
    df["ultra_risk"] = (
        df["is_new_customer"] * df["is_fiber"] *
        df["is_mtm"] * df["is_electronic_check"]
    ).astype("int8")

    # ── Business logic ihlali flag (EDA ADIM 5) ─────────────────────────────────
    tc = df["TotalCharges"]; ten = df["tenure"]; mc = df["MonthlyCharges"]
    expected = ten * mc
    ratio = tc / (expected + 1e-9)
    df["is_charges_anomaly"] = ((ratio < 0.5) | (ratio > 2.0)).astype("int8")
    df["is_new_tc_zero"]     = ((ten==0) & (tc>0)).astype("int8")

    # ── EDA ADIM 18: cohort bazlı churn oranı ──────────────────────────────────
    df["high_risk_combo"] = (
        (df["is_mtm"]==1) &
        (df["is_fiber"]==1) &
        (df["has_no_protection"]==1)
    ).astype("int8")

    return df

print("Feature Engineering uygulanıyor...")
train = feature_engineering(train)
test  = feature_engineering(test)
R.log(f"FE sonrası — Train: {train.shape}   Test: {test.shape}", "ok")

# FE özet tablosu
new_feats = [c for c in train.columns if c not in
             pd.read_csv("/kaggle/input/playground-series-s6e3/train.csv",nrows=1).columns.tolist()]
R.table(pd.DataFrame({"Yeni Feature":new_feats,
                       "Dtype":[str(train[c].dtype) for c in new_feats]}))
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# BÖLÜM 4 — Encoding
# ════════════════════════════════════════════════════════════════════════════════
R.section("🔤 BÖLÜM 4 — Encoding (Binary + Ordinal + Frequency)")

BINARY_COLS = ["gender","Partner","Dependents","PhoneService","PaperlessBilling"]
MULTI_COLS  = ["MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
               "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
               "Contract","PaymentMethod"]
BINARY_MAPS = {
    "gender":          {"Male":1,"Female":0},
    "Partner":         {"Yes":1,"No":0},
    "Dependents":      {"Yes":1,"No":0},
    "PhoneService":    {"Yes":1,"No":0},
    "PaperlessBilling":{"Yes":1,"No":0},
}

def encode_features(tr, te):
    tr, te = tr.copy(), te.copy()
    for col, m in BINARY_MAPS.items():
        if col in tr.columns:
            tr[col] = tr[col].map(m).fillna(0).astype("int8")
            te[col] = te[col].map(m).fillna(0).astype("int8")

    mc_present = [c for c in MULTI_COLS if c in tr.columns]
    if mc_present:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        tr[mc_present] = enc.fit_transform(tr[mc_present]).astype("int16")
        te[mc_present] = enc.transform(te[mc_present]).astype("int16")
        # Frequency encoding
        for col in mc_present:
            freq = tr[col].value_counts(normalize=True)
            tr[f"{col}_freq"] = tr[col].map(freq).fillna(0).astype("float32")
            te[f"{col}_freq"] = te[col].map(freq).fillna(0).astype("float32")
    return tr, te

train, test = encode_features(train, test)
Y = train[TARGET]
FEATS_ALL = [c for c in train.columns if c != TARGET]
R.log(f"Encoding tamamlandı. Toplam feature: {len(FEATS_ALL)}", "ok")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# BÖLÜM 5 — Per-Fold Target Encoding (CV-safe)
# ════════════════════════════════════════════════════════════════════════════════
def target_encode_fold(X_tr, X_val, X_te, y_tr, cols, smoothing=30.0):
    gm = y_tr.mean()
    X_tr, X_val, X_te = X_tr.copy(), X_val.copy(), X_te.copy()
    for col in cols:
        if col not in X_tr.columns: continue
        agg = pd.DataFrame({"c":X_tr[col],"t":y_tr})
        st = agg.groupby("c")["t"].agg(["mean","count"])
        sm = ((st["count"]*st["mean"]+smoothing*gm)/(st["count"]+smoothing))
        tc = f"{col}_te"
        X_tr[tc]  = X_tr[col].map(sm).fillna(gm).astype("float32")
        X_val[tc] = X_val[col].map(sm).fillna(gm).astype("float32")
        X_te[tc]  = X_te[col].map(sm).fillna(gm).astype("float32")
    return X_tr, X_val, X_te

TE_COLS = [c for c in MULTI_COLS if c in FEATS_ALL]

# ════════════════════════════════════════════════════════════════════════════════
# BÖLÜM 6 — Model Eğitimi (Multi-Seed + KFold + TE)
# ════════════════════════════════════════════════════════════════════════════════
R.section("🤖 BÖLÜM 6 — Model Eğitimi")
R.rule("Multi-seed averaging + per-fold target encoding + early stopping.")

X_full = train[FEATS_ALL].copy()
X_test = test[FEATS_ALL].copy()
SPW    = float(EDA.get("scale_pos_weight", 3.44))

LGB_PARAMS = dict(
    n_estimators=4000, learning_rate=0.008, num_leaves=63,
    max_depth=-1, min_child_samples=20,
    colsample_bytree=0.7, subsample=0.75, subsample_freq=1,
    reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=SPW, verbose=-1,
)
XGB_PARAMS = dict(
    n_estimators=4000, learning_rate=0.008, max_depth=6,
    min_child_weight=5, subsample=0.75, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=SPW, tree_method="hist",
    early_stopping_rounds=150, verbosity=0,
)
if HAS_CAT:
    CAT_PARAMS = dict(
        iterations=4000, learning_rate=0.008, depth=6,
        l2_leaf_reg=3.0, min_data_in_leaf=20,
        auto_class_weights="Balanced",
        early_stopping_rounds=150, verbose=0,
    )

def train_cv(X, y, X_te, model_type, params, n_seeds=N_SEEDS):
    all_oof  = np.zeros((n_seeds, len(X)))
    all_test = np.zeros((n_seeds, len(X_te)))

    for si in range(n_seeds):
        seed = SEED + si * 1000
        kf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
        oof = np.zeros(len(X)); tp = np.zeros(len(X_te)); scores = []

        for fold, (tri, vai) in enumerate(kf.split(X, y)):
            Xtr, Xva, y_tr, y_va = X.iloc[tri], X.iloc[vai], y.iloc[tri], y.iloc[vai]
            Xte_f = X_te.copy()
            Xtr, Xva, Xte_f = target_encode_fold(Xtr, Xva, Xte_f, y_tr, TE_COLS)

            p = {**params, "random_state":seed}
            try:
                if model_type == "lgb":
                    m = lgb.LGBMClassifier(**p)
                    m.fit(Xtr, y_tr, eval_set=[(Xva, y_va)],
                          callbacks=[lgb.early_stopping(150,verbose=False),lgb.log_evaluation(-1)])
                elif model_type == "xgb":
                    m = xgb.XGBClassifier(**p)
                    m.fit(Xtr, y_tr, eval_set=[(Xva, y_va)], verbose=False)
                elif model_type == "cat" and HAS_CAT:
                    m = CatBoostClassifier(**p)
                    m.fit(Xtr, y_tr, eval_set=(Xva, y_va), verbose=0)
                else:
                    continue
                vp = m.predict_proba(Xva)[:,1]
                oof[vai] = vp; tp += m.predict_proba(Xte_f)[:,1] / N_SPLITS
                scores.append(roc_auc_score(y_va, vp))
            except Exception as e:
                print(f"  [{model_type} fold{fold+1}] hata: {e}")

        all_oof[si]  = oof
        all_test[si] = tp
        oof_auc = roc_auc_score(y, oof)
        print(f"  {model_type.upper()} seed={seed} OOF={oof_auc:.5f}  folds={[f'{s:.4f}' for s in scores]}")

    final_oof  = all_oof.mean(axis=0)
    final_test = all_test.mean(axis=0)
    print(f"  ★ {model_type.upper()} FINAL OOF={roc_auc_score(y, final_oof):.5f}\n")
    return final_oof, final_test

print("LightGBM eğitiliyor...")
oof_lgb, test_lgb = train_cv(X_full, Y, X_test, "lgb", LGB_PARAMS)

print("XGBoost eğitiliyor...")
oof_xgb, test_xgb = train_cv(X_full, Y, X_test, "xgb", XGB_PARAMS)

if HAS_CAT:
    print("CatBoost eğitiliyor...")
    oof_cat, test_cat = train_cv(X_full, Y, X_test, "cat", CAT_PARAMS, n_seeds=2)
else:
    oof_cat  = np.zeros_like(oof_lgb); test_cat = np.zeros_like(test_lgb)
    R.log("CatBoost yüklü değil, atlandı.", "warn")

# Logistic Regression (baseline)
print("Logistic Regression eğitiliyor...")
lr_oof  = np.zeros(len(X_full)); lr_test = np.zeros(len(X_test))
skf_lr  = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
for tri, vai in skf_lr.split(X_full, Y):
    Xtr, Xva, Xte_f = X_full.iloc[tri], X_full.iloc[vai], X_test.copy()
    Xtr, Xva, Xte_f = target_encode_fold(Xtr, Xva, Xte_f, Y.iloc[tri], TE_COLS)
    sc = StandardScaler()
    m_lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", solver="lbfgs")
    m_lr.fit(sc.fit_transform(Xtr), Y.iloc[tri])
    lr_oof[vai]  = m_lr.predict_proba(sc.transform(Xva))[:,1]
    lr_test     += m_lr.predict_proba(sc.transform(Xte_f))[:,1] / N_SPLITS

print(f"  ★ LR FINAL OOF={roc_auc_score(Y, lr_oof):.5f}")

OOF_PREDS  = {"lgb": oof_lgb, "xgb": oof_xgb, "cat": oof_cat, "lr": lr_oof}
TEST_PREDS = {"lgb": test_lgb,"xgb": test_xgb,"cat": test_cat,"lr": lr_test}

model_scores = {k: roc_auc_score(Y, v) for k, v in OOF_PREDS.items()}
for k, v in model_scores.items():
    R.log(f"  {k.upper()} OOF AUC: {v:.5f}", "info")

R.end()

# ════════════════════════════════════════════════════════════════════════════════
# BÖLÜM 7 — Dummy Baseline Karşılaştırması
# ════════════════════════════════════════════════════════════════════════════════
R.section("🎲 BÖLÜM 7 — Dummy Baseline Karşılaştırması")
R.rule("Lift over dummy < 0.05 → model anlamlı bir şey öğrenmemiş!")

dummy_scores = {}
for strat in ["stratified","most_frequent","prior"]:
    aucs_ = []
    for tri, vai in StratifiedKFold(5,shuffle=True,random_state=SEED).split(X_full, Y):
        dc = DummyClassifier(strategy=strat, random_state=SEED)
        dc.fit(X_full.iloc[tri], Y.iloc[tri])
        try:
            p = dc.predict_proba(X_full.iloc[vai])[:,1]
            aucs_.append(roc_auc_score(Y.iloc[vai], p))
        except: aucs_.append(.5)
    dummy_scores[strat] = float(np.mean(aucs_))

best_dummy = max(dummy_scores.values())
best_model_auc = max(model_scores.values())
lift_over_dummy = best_model_auc - best_dummy

fig, ax = plt.subplots(figsize=(10, 4), facecolor=DARK)
labels  = list(dummy_scores.keys()) + list(model_scores.keys())
values  = list(dummy_scores.values()) + list(model_scores.values())
colors_ = [GRID, GRID, GRID] + [GREEN if v==best_model_auc else BLUE for v in model_scores.values()]
bars = ax.bar(labels, values, color=colors_, edgecolor=GRID, alpha=.85)
ax.axhline(.5, color=RED, lw=1, linestyle="--", label="Random (0.5)")
ax.axhline(best_dummy, color=ORANGE, lw=1.5, linestyle=":", label=f"Best dummy={best_dummy:.4f}")
for i, (l, v) in enumerate(zip(labels, values)):
    ax.text(i, v+.003, f"{v:.4f}", ha="center", color=TXT, fontsize=8)
dark_ax(ax, "Model vs Dummy Baseline Karşılaştırması", yl="AUC")
ax.legend(facecolor=CARD, labelcolor=TXT, fontsize=8)
plt.tight_layout()
R.plot(fig, "dummy_baseline", f"Dummy baseline — Lift={lift_over_dummy:+.4f}")

R.metrics([
    ("Best Model AUC", f"{best_model_auc:.5f}", "ok"),
    ("Best Dummy AUC", f"{best_dummy:.4f}", "info"),
    ("Lift over Dummy", f"+{lift_over_dummy:.4f}",
     "ok" if lift_over_dummy>.1 else "warn" if lift_over_dummy>.05 else "crit"),
])
if lift_over_dummy < .05:
    R.decision(f"Model dummy'yi zar zor geçiyor (lift={lift_over_dummy:.4f}). Feature'ları kontrol et!", "crit")
elif lift_over_dummy > .15:
    R.decision(f"Güçlü lift ({lift_over_dummy:.4f}). Model anlamlı örüntüler öğreniyor.", "ok")
else:
    R.decision(f"Orta lift ({lift_over_dummy:.4f}). FE ile iyileştirme mümkün.", "warn")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# BÖLÜM 8 — Permutation Testi (Global Anlamlılık)
# ════════════════════════════════════════════════════════════════════════════════
R.section("🔀 BÖLÜM 8 — Permutation Testi (Model İstatistiksel Anlamlılığı)")
R.rule("p-değeri > 0.05 → model istatistiksel olarak anlamlı değil (şans eseri öğrenmiş olabilir).")

n_perm = 100
tri_p, vai_p = list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(X_full, Y))[0]
Xtr_p, Xva_p = X_full.iloc[tri_p].copy(), X_full.iloc[vai_p].copy()
Xte_p        = X_test.copy()
Xtr_p, Xva_p, Xte_p = target_encode_fold(Xtr_p, Xva_p, Xte_p, Y.iloc[tri_p], TE_COLS)

m_perm = lgb.LGBMClassifier(200, is_unbalance=True, verbose=-1, n_jobs=-1, random_state=SEED)
m_perm.fit(Xtr_p, Y.iloc[tri_p])
real_auc = roc_auc_score(Y.iloc[vai_p], m_perm.predict_proba(Xva_p)[:,1])

perm_aucs = []
for i in range(n_perm):
    yp = Y.iloc[tri_p].sample(frac=1, random_state=i).values
    mp = lgb.LGBMClassifier(50, verbose=-1, n_jobs=-1, random_state=i)
    mp.fit(Xtr_p, yp)
    perm_aucs.append(roc_auc_score(Y.iloc[vai_p], mp.predict_proba(Xva_p)[:,1]))
p_val = float((np.array(perm_aucs) >= real_auc).mean())

fig, ax = plt.subplots(figsize=(9, 4), facecolor=DARK)
ax.hist(perm_aucs, bins=30, color=BLUE, alpha=.8, edgecolor=CARD, density=True, label="Permutation AUC dağılımı")
ax.axvline(real_auc, color=RED, lw=2.5, linestyle="--", label=f"Real AUC={real_auc:.4f}")
ax.axvline(np.percentile(perm_aucs,95), color=ORANGE, lw=1.5, linestyle=":", label="95. percentile")
dark_ax(ax, "Permutation Testi", xl="AUC", yl="Yoğunluk")
ax.legend(facecolor=CARD, labelcolor=TXT, fontsize=8)
plt.tight_layout()
R.plot(fig, "permutation_test", f"Permutation testi — p={p_val:.4f}")

R.metrics([
    ("Real AUC",   f"{real_auc:.4f}", "info"),
    ("p-değeri",   f"{p_val:.4f}",    "ok" if p_val<.05 else "crit"),
    ("Anlamlı?",   "Evet" if p_val<.05 else "Hayır", "ok" if p_val<.05 else "crit"),
])
if p_val > .05:
    R.decision(f"Model istatistiksel olarak ANLAMLI DEĞİL (p={p_val:.4f}). Feature seçimini gözden geçir!", "crit")
else:
    R.decision(f"Model istatistiksel olarak anlamlı (p={p_val:.4f}).", "ok")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# BÖLÜM 9 — Model Kalibrasyonu & Brier Skoru
# ════════════════════════════════════════════════════════════════════════════════
R.section("🎯 BÖLÜM 9 — Model Kalibrasyonu & Brier Skoru")
R.rule("Brier > 0.20 → Platt scaling zorunlu. 0.10–0.20 → isotonic regression dene.")
R.rule("Kalibrasyon eğrisi diyagonalden uzaksa → olasılıklar güvenilmez, threshold yanlış optimizasyon üretir.")

MAIN_OOF = oof_lgb
brier = float(brier_score_loss(Y, MAIN_OOF))
frac_pos, mean_pred = calibration_curve(Y, MAIN_OOF, n_bins=10)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=DARK)

# Kalibrasyon eğrisi
axes[0].plot([0,1],[0,1], color=GRID, linestyle="--", lw=1.5, label="Mükemmel")
axes[0].plot(mean_pred, frac_pos, color=BLUE, lw=2, marker="o", ms=5,
              label=f"Model (Brier={brier:.4f})")
axes[0].fill_between(mean_pred, frac_pos, mean_pred, alpha=.1, color=BLUE)
dark_ax(axes[0], "Kalibrasyon Eğrisi", xl="Tahmin Ortalaması", yl="Gerçek Oran")
axes[0].legend(facecolor=CARD, labelcolor=TXT, fontsize=8)

# OOF olasılık dağılımı
axes[1].hist(MAIN_OOF[Y==0], bins=50, alpha=.55, color=GREEN, density=True, label="No Churn", edgecolor=CARD)
axes[1].hist(MAIN_OOF[Y==1], bins=50, alpha=.55, color=RED,   density=True, label="Churn",    edgecolor=CARD)
dark_ax(axes[1], "Tahmin Olasılığı Dağılımı", xl="Olasılık")
axes[1].legend(facecolor=CARD, labelcolor=TXT, fontsize=8)

# Model karşılaştırma (calibration per model)
for k, oof_p in OOF_PREDS.items():
    if not np.any(oof_p > 0): continue
    try:
        fp_k, mp_k = calibration_curve(Y, oof_p, n_bins=10)
        axes[2].plot(mp_k, fp_k, lw=1.5, marker=".", ms=4, label=f"{k.upper()} (B={brier_score_loss(Y,oof_p):.4f})")
    except: pass
axes[2].plot([0,1],[0,1], color=GRID, linestyle="--", lw=1, label="Mükemmel")
dark_ax(axes[2], "Model Başına Kalibrasyon", xl="Tahmin", yl="Gerçek")
axes[2].legend(facecolor=CARD, labelcolor=TXT, fontsize=8)
plt.tight_layout()
R.plot(fig, "kalibrasyon", "Model kalibrasyon analizi — Brier skoru ve kalibrasyon eğrisi")

cal_status = "ok" if brier<.1 else "warn" if brier<.2 else "crit"
R.metrics([
    ("Brier Skoru", f"{brier:.4f}", cal_status),
    ("AUC-ROC",     f"{model_scores['lgb']:.5f}", "ok"),
    ("AP Score",    f"{average_precision_score(Y,MAIN_OOF):.4f}", "ok"),
])
if brier > .2:
    R.decision("Brier > 0.20. Platt scaling (CalibratedClassifierCV) hemen uygula!", "crit")
elif brier > .1:
    R.decision(f"Orta kalibrasyon (Brier={brier:.4f}). Isotonic regression dene.", "warn")
else:
    R.decision(f"İyi kalibrasyon (Brier={brier:.4f}).", "ok")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# BÖLÜM 10 — Threshold Optimizasyonu (Cost-Sensitive)
# ════════════════════════════════════════════════════════════════════════════════
R.section(f"💰 BÖLÜM 10 — Threshold Optimizasyonu (FN×{FN_COST} + FP×{FP_COST})")
R.rule(f"Churn problem: kaçan churner (FN) = {FN_COST}x müdahale (FP) maliyeti. ASLA default 0.5 kullanma!")
R.rule("Optimal threshold → toplam iş maliyetini minimize eder.")

thrs = np.linspace(.01, .99, 400)
costs = []
for t in thrs:
    pred = (MAIN_OOF >= t).astype(int)
    fn   = ((pred==0) & (Y==1)).sum()
    fp   = ((pred==1) & (Y==0)).sum()
    costs.append(fn*FN_COST + fp*FP_COST)

opt_thr  = float(thrs[np.argmin(costs)])
opt_pred = (MAIN_OOF >= opt_thr).astype(int)
cm_opt   = confusion_matrix(Y, opt_pred)

prec, rec, pr_thr = precision_recall_curve(Y, MAIN_OOF)
f1_scores = 2*prec*rec/(prec+rec+1e-9)
f1_thr    = float(pr_thr[np.argmax(f1_scores[:-1])])

fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=DARK)

# Maliyet eğrisi
axes[0,0].plot(thrs, costs, color=BLUE, lw=2)
axes[0,0].axvline(opt_thr, color=RED, lw=2, linestyle="--", label=f"Optimal={opt_thr:.3f}")
axes[0,0].axvline(.5, color=GRID, lw=1, linestyle=":", label="Default=0.5")
dark_ax(axes[0,0], f"Maliyet Eğrisi (FN×{FN_COST}+FP×{FP_COST})", xl="Threshold", yl="Toplam Maliyet")
axes[0,0].legend(facecolor=CARD, labelcolor=TXT, fontsize=8)

# PR eğrisi
axes[0,1].plot(rec, prec, color=ORANGE, lw=2)
best_f1_idx = np.argmax(f1_scores[:-1])
axes[0,1].scatter([rec[best_f1_idx]], [prec[best_f1_idx]], color=GREEN, s=80, zorder=5,
                   label=f"Best F1 thr={f1_thr:.3f}")
dark_ax(axes[0,1], "Precision-Recall Eğrisi", xl="Recall", yl="Precision")
axes[0,1].legend(facecolor=CARD, labelcolor=TXT, fontsize=8)

# Karışıklık matrisi
axes[1,0].imshow(cm_opt, cmap="RdYlGn", aspect="auto", vmin=0)
for i in range(2):
    for j in range(2):
        axes[1,0].text(j, i, f"{cm_opt[i,j]:,}", ha="center", va="center",
                        color="white", fontsize=14, fontweight="bold")
dark_ax(axes[1,0], f"Karışıklık Matrisi (thr={opt_thr:.3f})")
axes[1,0].set_xticks([0,1]); axes[1,0].set_yticks([0,1])
axes[1,0].set_xticklabels(["Tahmin: Hayır","Tahmin: Evet"], color=TXT)
axes[1,0].set_yticklabels(["Gerçek: Hayır","Gerçek: Evet"], color=TXT)

# F1 vs Threshold
axes[1,1].plot(pr_thr, f1_scores[:-1], color=PURP, lw=2)
axes[1,1].axvline(f1_thr, color=GOLD, lw=2, linestyle="--",
                   label=f"Best F1={max(f1_scores):.4f} @ {f1_thr:.3f}")
axes[1,1].axvline(opt_thr, color=RED, lw=1.5, linestyle=":",
                   label=f"Cost-optimal={opt_thr:.3f}")
dark_ax(axes[1,1], "F1 Skoru vs Threshold", xl="Threshold", yl="F1")
axes[1,1].legend(facecolor=CARD, labelcolor=TXT, fontsize=8)
plt.tight_layout()
R.plot(fig, "threshold_cost", "Threshold optimizasyonu — maliyet eğrisi, PR, CM, F1")

tn, fp_v, fn_v, tp_v = cm_opt.ravel()
R.metrics([
    ("Cost-Opt Threshold", f"{opt_thr:.3f}", "ok" if abs(opt_thr-.5)<.2 else "warn"),
    ("F1-Opt Threshold",   f"{f1_thr:.3f}",  "info"),
    ("TN", f"{tn:,}", "info"), ("FP", f"{fp_v:,}", "info"),
    ("FN", f"{fn_v:,}", "warn"), ("TP", f"{tp_v:,}", "info"),
    ("Precision", f"{tp_v/(tp_v+fp_v+1e-9):.4f}", "info"),
    ("Recall",    f"{tp_v/(tp_v+fn_v+1e-9):.4f}", "info"),
])

if abs(opt_thr - .5) > .15:
    R.decision(f"Optimal eşik ({opt_thr:.3f}) 0.5'ten ciddi uzakta. Default threshold submission'a ASLA!", "crit")
elif abs(opt_thr - .5) > .05:
    R.decision(f"Optimal eşik ({opt_thr:.3f}). Submission öncesi threshold ayarla.", "warn")
else:
    R.decision(f"Optimal eşik ({opt_thr:.3f}) standart değere yakın.", "ok")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# BÖLÜM 11 — Lift & Gain Analizi
# ════════════════════════════════════════════════════════════════════════════════
R.section("📈 BÖLÜM 11 — Lift & Gain Analizi")
R.rule("Top desil lift > 2x → modelin güçlü iş değeri var. < 1.5x → FE ile iyileştir.")

df_lg = pd.DataFrame({"y":Y.values,"prob":MAIN_OOF}).sort_values("prob",ascending=False).reset_index(drop=True)
df_lg["decile"] = pd.qcut(df_lg.index, 10, labels=False) + 1
base_rate = Y.mean()
ds = df_lg.groupby("decile")["y"].agg(["mean","count","sum"])
ds["lift"]     = ds["mean"] / base_rate
ds["cum_gain"] = ds["sum"].cumsum() / Y.sum() * 100

fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=DARK)

# Lift chart
axes[0].bar(ds.index.astype(str), ds["lift"],
             color=[GREEN if v>=2 else ORANGE if v>=1.5 else RED for v in ds["lift"]],
             edgecolor=GRID, alpha=.85)
axes[0].axhline(1.0, color=GRID, lw=1.5, linestyle="--", label="Baseline=1")
axes[0].axhline(2.0, color=GREEN, lw=1,   linestyle=":",  label="2x Lift")
dark_ax(axes[0], "Lift Chart (Desil Bazlı)", xl="Desil", yl="Lift")
axes[0].legend(facecolor=CARD, labelcolor=TXT, fontsize=8)

# Gain chart
axes[1].plot(range(1,11), ds["cum_gain"].values, color=BLUE, lw=2, marker="o", ms=5, label="Model Gain")
axes[1].plot([1,10],[10,100], color=GRID, linestyle="--", lw=1, label="Random")
axes[1].fill_between(range(1,11), ds["cum_gain"].values, np.linspace(10,100,10), alpha=.1, color=BLUE)
dark_ax(axes[1], "Kümülatif Gain Grafiği", xl="Desil", yl="Kümülatif % Churn")
axes[1].legend(facecolor=CARD, labelcolor=TXT, fontsize=8)

# Churn rate per decile
axes[2].bar(ds.index.astype(str), ds["mean"]*100, color=ORANGE, edgecolor=GRID, alpha=.85)
axes[2].axhline(base_rate*100, color=RED, lw=1.5, linestyle="--", label=f"Overall={base_rate*100:.1f}%")
dark_ax(axes[2], "Desil Başına Churn Oranı %", xl="Desil", yl="Churn %")
axes[2].legend(facecolor=CARD, labelcolor=TXT, fontsize=8)
plt.tight_layout()
R.plot(fig, "lift_gain", "Lift & Gain analizi — model iş değeri")

top_lift = float(ds["lift"].iloc[0])
top_rate = float(ds["mean"].iloc[0]*100)
R.log(f"Top desil lift: {top_lift:.2f}x  (churn rate: %{top_rate:.1f})", "info")
if top_lift >= 3:
    R.decision(f"Mükemmel top desil lift ({top_lift:.2f}x). Model yüksek risk müşterileri iyi tanımlıyor.", "ok")
elif top_lift >= 2:
    R.decision(f"İyi top desil lift ({top_lift:.2f}x). İş kurallarına göre uygulayabilirsin.", "ok")
else:
    R.decision(f"Düşük top desil lift ({top_lift:.2f}x). FE veya model iyileştirme gerekli.", "warn")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# BÖLÜM 12 — Learning Curve Analizi
# ════════════════════════════════════════════════════════════════════════════════
R.section("📚 BÖLÜM 12 — Learning Curve Analizi")
R.rule("Train-Val gap > 0.05 → overfitting. Val AUC hâlâ artıyorsa → daha fazla veri/augmentation.")

sizes = np.linspace(.1, 1., 7); tr_s=[]; va_s=[]
skf_lc = StratifiedKFold(3, shuffle=True, random_state=SEED)
Xenc = X_full.copy()
for c in Xenc.select_dtypes(["object"]).columns:
    Xenc[c] = pd.factorize(Xenc[c])[0]
Xenc = Xenc.fillna(-1)

for sz in sizes:
    n = max(100, int(len(Xenc)*sz))
    idx = np.random.RandomState(SEED).choice(len(Xenc), n, replace=False)
    Xs = Xenc.iloc[idx]; ys = Y.iloc[idx]
    ft=[]; fv=[]
    for tri, vai in skf_lc.split(Xs, ys):
        m = lgb.LGBMClassifier(200, .05, is_unbalance=True, verbose=-1, n_jobs=-1, random_state=SEED)
        m.fit(Xs.iloc[tri], ys.iloc[tri])
        ft.append(roc_auc_score(ys.iloc[tri], m.predict_proba(Xs.iloc[tri])[:,1]))
        fv.append(roc_auc_score(ys.iloc[vai], m.predict_proba(Xs.iloc[vai])[:,1]))
    tr_s.append(float(np.mean(ft))); va_s.append(float(np.mean(fv)))
ns = [int(len(Xenc)*s) for s in sizes]

fig, ax = plt.subplots(figsize=(9, 4), facecolor=DARK)
ax.plot(ns, tr_s, color=GREEN, lw=2, marker="o", ms=5, label="Train AUC")
ax.plot(ns, va_s, color=BLUE,  lw=2, marker="s", ms=5, label="Val AUC")
ax.fill_between(ns, tr_s, va_s, alpha=.1, color=ORANGE)
dark_ax(ax, "Learning Curve", xl="Eğitim Örnek Sayısı", yl="AUC")
ax.legend(facecolor=CARD, labelcolor=TXT, fontsize=8)
plt.tight_layout()
R.plot(fig, "learning_curve", "Learning curve — overfitting/underfitting tespiti")

gap = tr_s[-1] - va_s[-1]
val_slope = va_s[-1] - va_s[-3]
R.log(f"Train-Val gap: {gap:.4f}   Val son slope: {val_slope:+.4f}", "info")
if gap > .05:
    R.decision(f"Overfitting gap={gap:.4f}. Regularizasyon artır veya feature'ları azalt.", "warn")
elif val_slope > .005:
    R.decision("Val AUC hâlâ artıyor. Orijinal veri ile augmentation faydalı olur.", "ok")
else:
    R.decision("Learning curve düzleşti. Feature engineering'e odaklan.", "warn")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# BÖLÜM 13 — SHAP Analizi
# ════════════════════════════════════════════════════════════════════════════════
R.section("🔬 BÖLÜM 13 — SHAP Feature Önem & Etkileşim Analizi")
R.rule("SHAP std/mean > 0.5 → feature fold'lar arası instabil. Drop adayı.")

if HAS_SHAP:
    try:
        # Son fold modelini yeniden eğit (tek fold, hızlı)
        Xtr_shap = Xenc.iloc[:int(len(Xenc)*.8)]
        Xva_shap = Xenc.iloc[int(len(Xenc)*.8):]
        y_tr_shap = Y.iloc[:int(len(Y)*.8)]
        y_va_shap = Y.iloc[int(len(Y)*.8):]

        m_shap = lgb.LGBMClassifier(500, .05, is_unbalance=True, verbose=-1, n_jobs=-1, random_state=SEED)
        m_shap.fit(Xtr_shap, y_tr_shap)

        explainer = shap_lib.TreeExplainer(m_shap)
        sv = explainer.shap_values(Xva_shap)
        if isinstance(sv, list): sv = sv[1]

        shap_mean = np.abs(sv).mean(axis=0)
        shap_df   = pd.DataFrame({"Feature": FEATS_ALL,
                                   "SHAP Mean": shap_mean}).sort_values("SHAP Mean", ascending=False)
        top5 = shap_df.head(5)["Feature"].tolist()

        R.table(shap_df.head(20).round(5))

        fig = plt.figure(figsize=(18, 10), facecolor=DARK)
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=.4, wspace=.35)

        # Bar
        ax_bar = fig.add_subplot(gs[0, :2])
        top15 = shap_df.head(15).iloc[::-1]
        ax_bar.barh(top15["Feature"], top15["SHAP Mean"], color=BLUE, edgecolor=GRID, alpha=.85)
        dark_ax(ax_bar, "SHAP Feature Önem Sıralaması (Top 15)")

        # Stability: fold bazlı SHAP CV
        ax_stab = fig.add_subplot(gs[0, 2])
        shap_folds = []
        for fi, (tri, vai) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SEED).split(Xenc, Y)):
            e2 = shap_lib.TreeExplainer(m_shap)
            sv2 = e2.shap_values(Xenc.iloc[vai])
            if isinstance(sv2, list): sv2 = sv2[1]
            shap_folds.append(np.abs(sv2).mean(axis=0))
        fsa = np.array(shap_folds)
        cv_arr = fsa.std(axis=0) / (fsa.mean(axis=0) + 1e-9)
        df_stab = pd.DataFrame({"Feature":FEATS_ALL,"CV":cv_arr}).sort_values("CV",ascending=False).head(10)
        ax_stab.barh(df_stab["Feature"], df_stab["CV"],
                      color=[RED if v>.5 else ORANGE if v>.3 else GREEN for v in df_stab["CV"]],
                      edgecolor=GRID, alpha=.85)
        ax_stab.axvline(.5, color=RED, lw=1, linestyle="--", label="Instabilite (0.5)")
        dark_ax(ax_stab, "SHAP Stabilitesi (CV = std/mean)")
        ax_stab.legend(facecolor=CARD, labelcolor=TXT, fontsize=7)

        # Top 3 scatter
        for i, col in enumerate(top5[:3]):
            ax = fig.add_subplot(gs[1, i])
            ci = FEATS_ALL.index(col) if col in FEATS_ALL else 0
            sc = ax.scatter(Xva_shap[col].values, sv[:, ci],
                             c=y_va_shap.values, cmap="RdYlGn_r",
                             alpha=.4, s=7, vmin=0, vmax=1)
            ax.axhline(0, color=GRID, lw=1, linestyle="--")
            dark_ax(ax, f"SHAP: {col}", xl=col, yl="SHAP")

        R.plot(fig, "shap_analysis", "SHAP analizi — önem, stabilite, dependence scatter")

        # Interaction heatmap (top 5)
        if len(top5) >= 3:
            shap_top = pd.DataFrame(sv[:, [FEATS_ALL.index(f) for f in top5 if f in FEATS_ALL]],
                                     columns=[f for f in top5 if f in FEATS_ALL])
            inter = shap_top.corr()
            fig2, ax2 = plt.subplots(figsize=(7, 5), facecolor=DARK)
            sns.heatmap(inter, ax=ax2, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                         linewidths=.5, linecolor=GRID)
            dark_ax(ax2, "SHAP Etkileşim (Top 5 Feature)")
            ax2.tick_params(colors=TXT, labelsize=8)
            plt.tight_layout()
            R.plot(fig2, "shap_interaction", "SHAP feature etkileşim matrisi")

            strong_inter = [(top5[i], top5[j]) for i in range(len(top5))
                             for j in range(i+1, len(top5)) if abs(inter.iloc[i,j]) > .3]
            if strong_inter:
                R.decision(f"SHAP etkileşim çiftleri: {strong_inter} → çarpım feature oluştur.", "ok")

        unstable = df_stab[df_stab["CV"] > .5]["Feature"].tolist()
        if unstable:
            R.decision(f"Instabil SHAP feature'lar: {unstable} → drop adayı.", "warn")
        R.log(f"SHAP Top 5: {top5}", "ok")

    except Exception as e:
        R.log(f"SHAP hatası: {e}", "warn")
else:
    R.log("shap kütüphanesi yüklü değil. pip install shap ile kur.", "warn")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# BÖLÜM 14 — Hard Sample Analizi
# ════════════════════════════════════════════════════════════════════════════════
R.section("🎯 BÖLÜM 14 — Hard Sample Analizi")
R.rule("5 fold'un 4+'ında yanlış tahmin edilen satırlar → hard sample. Pattern incele.")

fold_mat = np.zeros((len(X_full), N_SPLITS))
for fold, (tri, vai) in enumerate(StratifiedKFold(N_SPLITS,shuffle=True,random_state=SEED).split(Xenc,Y)):
    m_h = lgb.LGBMClassifier(300, is_unbalance=True, verbose=-1, n_jobs=-1, random_state=SEED)
    m_h.fit(Xenc.iloc[tri], Y.iloc[tri])
    fold_mat[vai, fold] = m_h.predict_proba(Xenc.iloc[vai])[:,1]

wrong = 5 - (fold_mat>.5).astype(int).T.__eq__(Y.values).T.sum(axis=1)
is_hard   = wrong >= 4
hard_rate = float(is_hard.mean() * 100)
num_cols_hard = [c for c in FEATS_ALL if Xenc[c].dtype in [np.float32, np.float64, int, np.int8, np.int16, np.int32]]

R.metrics([
    ("Hard Sample Sayısı", f"{is_hard.sum():,}", "crit" if hard_rate>15 else "warn" if hard_rate>8 else "ok"),
    ("Hard Oran",          f"{hard_rate:.2f}%",  "crit" if hard_rate>15 else "warn" if hard_rate>8 else "ok"),
])

if is_hard.sum() > 5:
    hd = train[is_hard.values]; nd = train[~is_hard.values]
    comp = []
    for col in num_cols_hard[:10]:
        if col in train.columns:
            hm = hd[col].mean(); nm = nd[col].mean()
            comp.append({"Feature":col, "Hard Ort.":round(hm,3), "Normal Ort.":round(nm,3),
                          "Fark%":round(abs(hm-nm)/(abs(nm)+1e-9)*100,1)})
    df_c = pd.DataFrame(comp).sort_values("Fark%",ascending=False)
    R.table(df_c.head(10))

    top_diff = df_c.head(6)["Feature"].tolist()
    if top_diff:
        fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK)
        x = np.arange(len(top_diff)); w=.35
        hm_v = [df_c[df_c["Feature"]==c]["Hard Ort."].values[0]   for c in top_diff]
        nm_v = [df_c[df_c["Feature"]==c]["Normal Ort."].values[0] for c in top_diff]
        ax.bar(x-w/2, hm_v, w, label="Hard", color=RED,   alpha=.8, edgecolor=GRID)
        ax.bar(x+w/2, nm_v, w, label="Normal",color=GREEN, alpha=.8, edgecolor=GRID)
        ax.set_xticks(x); ax.set_xticklabels(top_diff, rotation=30, ha="right", color=TXT)
        dark_ax(ax, "Hard vs Normal — Feature Karşılaştırması")
        ax.legend(facecolor=CARD, labelcolor=TXT, fontsize=9)
        plt.tight_layout()
        R.plot(fig, "hard_samples", "Hard sample feature profili")

if hard_rate > 15:
    R.decision(f"Yüksek hard sample (%{hard_rate:.1f}). Agresif FE + target encoding zorunlu.", "crit")
elif hard_rate > 8:
    R.decision(f"Orta hard sample (%{hard_rate:.1f}). Stacking veya meta-feature dene.", "warn")
else:
    R.decision(f"Düşük hard sample (%{hard_rate:.1f}). Model güçlü.", "ok")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# BÖLÜM 15 — Seed Stabilitesi
# ════════════════════════════════════════════════════════════════════════════════
R.section("🌱 BÖLÜM 15 — Seed Stabilitesi")
R.rule("AUC std > 0.005 → model instabil. RepeatedStratifiedKFold(5,3) kullan.")

seeds_list = [0, 42, 123, 456, 789]
seed_aucs  = []
for seed in seeds_list:
    aucs_s = []
    for tri, vai in StratifiedKFold(5,shuffle=True,random_state=seed).split(Xenc,Y):
        m_s = lgb.LGBMClassifier(200,.05,is_unbalance=True,verbose=-1,n_jobs=-1,random_state=seed)
        m_s.fit(Xenc.iloc[tri],Y.iloc[tri])
        aucs_s.append(roc_auc_score(Y.iloc[vai],m_s.predict_proba(Xenc.iloc[vai])[:,1]))
    seed_aucs.append(float(np.mean(aucs_s)))
auc_std   = float(np.std(seed_aucs))
best_seed = seeds_list[int(np.argmax(seed_aucs))]

fig, ax = plt.subplots(figsize=(9, 4), facecolor=DARK)
ax.bar([str(s) for s in seeds_list], seed_aucs,
        color=[GREEN if a==max(seed_aucs) else BLUE for a in seed_aucs],
        edgecolor=GRID, alpha=.85)
ax.axhline(float(np.mean(seed_aucs)), color=ORANGE, lw=1.5, linestyle="--",
            label=f"Ort={np.mean(seed_aucs):.4f}±{auc_std:.4f}")
dark_ax(ax, "Seed Stabilitesi — 5 Farklı Seed", xl="Seed", yl="AUC")
ax.legend(facecolor=CARD, labelcolor=TXT, fontsize=8)
plt.tight_layout()
R.plot(fig, "seed_stability", "Seed stabilitesi analizi")

R.metrics([
    ("AUC Std",     f"{auc_std:.5f}",   "ok" if auc_std<.002 else "warn" if auc_std<.005 else "crit"),
    ("Best Seed",   str(best_seed),      "info"),
    ("AUC Aralığı", f"{max(seed_aucs)-min(seed_aucs):.5f}", "info"),
])
if auc_std > .005:
    R.decision(f"Yüksek seed instabilitesi (std={auc_std:.4f}). RepeatedStratifiedKFold(5,3) kullan.", "warn")
else:
    R.decision(f"Stabil model (std={auc_std:.5f}).", "ok")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# BÖLÜM 16 — Ensemble
# ════════════════════════════════════════════════════════════════════════════════
R.section("🏆 BÖLÜM 16 — Ensemble (Weighted + Rank + Stacking)")

W = {"lgb":0.35, "xgb":0.25, "cat":0.30 if HAS_CAT else 0.0, "lr":0.10}
if not HAS_CAT: W["lgb"]+=0.30

# Weighted
oof_w  = sum(W[k]*OOF_PREDS[k]  for k in W)
test_w = sum(W[k]*TEST_PREDS[k] for k in W)

# Rank average
def rank_avg(preds, weights):
    ranks = [stats.rankdata(p)/len(p) for p in preds]
    return sum(wt*r for wt,r in zip(weights, ranks))

wl = [W[k] for k in OOF_PREDS]
oof_r  = rank_avg(list(OOF_PREDS.values()),  wl)
test_r = rank_avg(list(TEST_PREDS.values()), wl)

# Stacking (meta-LR)
oof_sf  = np.column_stack(list(OOF_PREDS.values()))
test_sf = np.column_stack(list(TEST_PREDS.values()))
for i in range(oof_sf.shape[1]):
    for j in range(i+1, oof_sf.shape[1]):
        oof_sf  = np.column_stack([oof_sf,  oof_sf[:,i]*oof_sf[:,j]])
        test_sf = np.column_stack([test_sf, test_sf[:,i]*test_sf[:,j]])

oof_stk  = np.zeros(len(Y)); test_stk = np.zeros(len(test_sf))
for tri, vai in StratifiedKFold(N_SPLITS,shuffle=True,random_state=SEED).split(oof_sf, Y):
    sc = StandardScaler()
    meta = LogisticRegression(C=1.0, max_iter=2000)
    meta.fit(sc.fit_transform(oof_sf[tri]), Y.iloc[tri])
    oof_stk[vai]  = meta.predict_proba(sc.transform(oof_sf[vai]))[:,1]
    test_stk     += meta.predict_proba(sc.transform(test_sf))[:,1] / N_SPLITS

candidates = {
    "weighted":  (oof_w,   test_w),
    "rank_avg":  (oof_r,   test_r),
    "stacked":   (oof_stk, test_stk),
}

ens_results = []
for name, (oof_c, _) in candidates.items():
    auc = roc_auc_score(Y, oof_c)
    ens_results.append({"Ensemble":name,"OOF AUC":round(auc,6)})
    R.log(f"  {name:12s} → AUC: {auc:.6f}", "info")

R.table(pd.DataFrame(ens_results))

best_name = max(candidates, key=lambda k: roc_auc_score(Y, candidates[k][0]))
final_oof, final_test = candidates[best_name]
best_auc = roc_auc_score(Y, final_oof)
R.log(f"★ Best ensemble: {best_name}  (AUC={best_auc:.6f})", "ok")

# Ensemble görselleştirme — model korelasyonu
corr_mat = pd.DataFrame({k:v for k,v in OOF_PREDS.items() if np.any(v>0)}).corr()
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=DARK)
sns.heatmap(corr_mat, ax=axes[0], cmap="RdBu_r", center=0, annot=True, fmt=".3f",
             linewidths=.5, linecolor=GRID)
dark_ax(axes[0], "Model OOF Korelasyonu (Ensemble Diversity)")
axes[0].tick_params(colors=TXT)

names  = list(ens_results)
aucs_e = [row["OOF AUC"] for row in ens_results]
axes[1].bar([r["Ensemble"] for r in ens_results], aucs_e,
             color=[GREEN if a==max(aucs_e) else BLUE for a in aucs_e],
             edgecolor=GRID, alpha=.85)
dark_ax(axes[1], "Ensemble Strateji Karşılaştırması", yl="AUC")
for i, v in enumerate(aucs_e):
    axes[1].text(i, v+.0003, f"{v:.5f}", ha="center", color=TXT, fontsize=8)
plt.tight_layout()
R.plot(fig, "ensemble", "Ensemble analizi — model korelasyonu ve strateji karşılaştırması")

# Düşük korelasyon çiftleri = iyi ensemble çeşitliliği
if not corr_mat.empty:
    for i in range(len(corr_mat)):
        for j in range(i+1, len(corr_mat)):
            r = corr_mat.iloc[i,j]
            if r < .92:
                R.decision(f"Model çeşitliliği ({corr_mat.columns[i]} ↔ {corr_mat.columns[j]}: r={r:.3f}) — ensemble faydalı.", "ok")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# BÖLÜM 17 — OOF vs Test Dağılım Kontrolü
# ════════════════════════════════════════════════════════════════════════════════
R.section("🔭 BÖLÜM 17 — OOF vs Test Tahmin Dağılım Kontrolü")
R.rule("KS > 0.10 → OOF ve test dağılımı farklı. Rank transform veya recalibration gerekli.")

ks_oof_test, ks_p = ks_2samp(final_oof, final_test)
R.log(f"KS(OOF, Test tahminleri): stat={ks_oof_test:.4f}  p={ks_p:.4f}", "info")

fig, ax = plt.subplots(figsize=(9, 4), facecolor=DARK)
ax.hist(final_oof,  bins=50, alpha=.6, color=BLUE,   density=True, label="OOF Tahminleri",  edgecolor=CARD)
ax.hist(final_test, bins=50, alpha=.6, color=ORANGE, density=True, label="Test Tahminleri", edgecolor=CARD)
dark_ax(ax, f"OOF vs Test Dağılımı (KS={ks_oof_test:.4f})", xl="Tahmin Olasılığı", yl="Yoğunluk")
ax.legend(facecolor=CARD, labelcolor=TXT, fontsize=9)
plt.tight_layout()
R.plot(fig, "oof_vs_test", "OOF vs test tahmin dağılımı karşılaştırması")

if ks_oof_test > .10:
    R.decision(f"OOF-Test dağılımı farklı (KS={ks_oof_test:.4f}). Rank transform uygula!", "crit")
    final_test = stats.rankdata(final_test) / len(final_test)
    final_test = final_test * (final_oof.max() - final_oof.min()) + final_oof.min()
    R.log("Rank transform uygulandı.", "ok")
elif ks_oof_test > .05:
    R.decision(f"Küçük dağılım farkı (KS={ks_oof_test:.4f}). Submission'ı izle.", "warn")
else:
    R.decision(f"OOF ve test dağılımları benzer (KS={ks_oof_test:.4f}).", "ok")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# BÖLÜM 18 — McNemar Testi (Model Karşılaştırması)
# ════════════════════════════════════════════════════════════════════════════════
R.section("📊 BÖLÜM 18 — McNemar Testi (LGB vs XGB)")
R.rule("p < 0.05 → iki model istatistiksel olarak farklı. Daha iyi olan tercih edilmeli.")

pf = (oof_lgb >= opt_thr).astype(int)
px = (oof_xgb >= opt_thr).astype(int)
cf = (pf == Y.values).astype(int)
cx = (px == Y.values).astype(int)
b_val = int(((cf==1)&(cx==0)).sum())
c_val = int(((cf==0)&(cx==1)).sum())

try:
    result = mcnemar([[0, b_val],[c_val, 0]], exact=False, correction=True)
    mc_p   = float(result.pvalue)
    R.log(f"McNemar b={b_val}  c={c_val}  p={mc_p:.5f}", "info")
    R.metrics([
        ("LGB OOF AUC", f"{model_scores['lgb']:.5f}", "info"),
        ("XGB OOF AUC", f"{model_scores['xgb']:.5f}", "info"),
        ("McNemar p",   f"{mc_p:.5f}", "ok" if mc_p<.05 else "warn"),
    ])
    if mc_p < .05:
        winner = "LGB" if model_scores["lgb"] > model_scores["xgb"] else "XGB"
        R.decision(f"McNemar: iki model istatistiksel olarak farklı (p={mc_p:.4f}). {winner} daha iyi.", "ok")
    else:
        R.decision(f"McNemar: iki model arasında anlamlı fark yok (p={mc_p:.4f}). Ensemble tercih et.", "warn")
except Exception as e:
    R.log(f"McNemar hatası: {e}", "warn")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# BÖLÜM 19 — Master Değerlendirme Skoru
# ════════════════════════════════════════════════════════════════════════════════
R.section("🏁 BÖLÜM 19 — Master Değerlendirme Skoru (/20)")

score = 0; max_s = 20; notes = []
auc = best_auc

def chk(cond_g, cond_w, pts_g, pts_w, note_f, note_w=""):
    global score
    if cond_g: score += pts_g
    elif cond_w:
        score += pts_w
        if note_w: notes.append(f"⚠️ {note_w}")
    else: notes.append(f"🔴 {note_f}")

chk(not EDA.get("leakage_features",[]), False, 4, 0, f"Leak features: {EDA.get('leakage_features')}")
chk(auc>.85, auc>.75, 3, 2, f"AUC={auc:.4f} çok düşük", f"AUC={auc:.4f} orta")
chk(brier<.10, brier<.20, 2, 1, f"Brier={brier:.4f} kötü kalibrasyon", f"Brier={brier:.4f} orta")
chk(adv_auc<.60, adv_auc<.70, 2, 1, f"Adversarial AUC={adv_auc:.4f} ciddi shift")
chk(top_lift>=2, top_lift>=1.5, 2, 1, f"Top lift={top_lift:.2f}x zayıf", f"Top lift={top_lift:.2f}x orta")
chk(lift_over_dummy>.15, lift_over_dummy>.05, 2, 1, "Model dummy'yi geçemiyor")
chk(hard_rate<8, hard_rate<15, 2, 1, f"Hard rate=%{hard_rate:.1f} yüksek", f"Hard rate=%{hard_rate:.1f} orta")
chk(p_val<.01, p_val<.05, 1, 0, f"Permutation p={p_val:.4f} anlamlı değil")
chk(auc_std<.002, auc_std<.005, 1, 0, f"Seed std={auc_std:.4f} instabil")
chk(EDA.get("conflict_rate",0)<5, EDA.get("conflict_rate",0)<10, 1, 0, "Label noise yüksek")

pct = score/max_s
col_ = GREEN if pct>=.8 else ORANGE if pct>=.6 else RED
status = "ok" if pct>=.8 else "warn" if pct>=.6 else "crit"

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK)
axes[0].barh(["Skor"], [score],    color=col_,   edgecolor=GRID, height=.4)
axes[0].barh(["Skor"], [max_s-score], left=[score], color=GRID, edgecolor=CARD, height=.4, alpha=.3)
axes[0].text(score/2, 0, f"{score}/{max_s}\n({pct:.0%})",
              ha="center", va="center", color="white", fontsize=18, fontweight="bold")
axes[0].set_xlim(0, max_s); axes[0].set_yticks([])
axes[0].axvline(max_s*.8,  color=GREEN,  lw=1.5, linestyle="--", label="%80 Yeşil Bölge")
axes[0].axvline(max_s*.6,  color=ORANGE, lw=1,   linestyle="--", label="%60 Sarı Bölge")
dark_ax(axes[0], "Pipeline Hazırlık Skoru")
axes[0].legend(facecolor=CARD, labelcolor=TXT, fontsize=8)

# Radar
cats_r = ["AUC","Calib.","Advers.","Lift","Hard","Perm.","Seed","Dummy"]
vals_r = [min(1,(auc-.5)/.5), max(0,1-brier/.2), max(0,(1-adv_auc)/.5),
           min(1,top_lift/3), max(0,1-hard_rate/15), max(0,1-p_val/.1),
           max(0,1-auc_std/.01), min(1,lift_over_dummy/.2)]
angles = np.linspace(0,2*np.pi,len(cats_r),endpoint=False).tolist()
vals_r += vals_r[:1]; angles += angles[:1]
ax_r = plt.subplot(1,2,2, polar=True)
ax_r.figure.patch.set_facecolor(DARK)
ax_r.set_facecolor(CARD)
ax_r.plot(angles, vals_r, color=col_, lw=2, marker="o", ms=6)
ax_r.fill(angles, vals_r, alpha=.2, color=col_)
ax_r.set_xticks(angles[:-1]); ax_r.set_xticklabels(cats_r, color=TXT, fontsize=9)
ax_r.set_title("Metrik Radar", color=BLUE, fontsize=11, fontweight="bold", pad=15)
plt.tight_layout()
R.plot(fig, "master_score", "Master değerlendirme skoru ve metrik radar")

# Özet metrik tablosu
summary_data = [
    ("Best AUC",          f"{auc:.5f}"),
    ("Brier Skoru",       f"{brier:.4f}"),
    ("Adversarial AUC",   f"{adv_auc:.4f}"),
    ("Lift over Dummy",   f"+{lift_over_dummy:.4f}"),
    ("Top Desil Lift",    f"{top_lift:.2f}x"),
    ("Hard Sample %",     f"{hard_rate:.2f}%"),
    ("Permutation p",     f"{p_val:.4f}"),
    ("Seed Std",          f"{auc_std:.5f}"),
    ("Opt. Threshold",    f"{opt_thr:.3f}"),
    ("KS OOF-Test",       f"{ks_oof_test:.4f}"),
    ("Ensemble",          best_name),
    ("Master Skor",       f"{score}/{max_s} ({pct:.0%})"),
]
R.table(pd.DataFrame(summary_data, columns=["Metrik","Değer"]))

R.metrics([
    ("Master Skor", f"{score}/{max_s}", status),
    ("Kapsam",      f"{pct:.0%}",       status),
])

if pct >= .8:
    R.decision("Pipeline mükemmel durumda. Submission hazır!", "ok")
elif pct >= .6:
    R.decision("Küçük sorunlar var. Uyarıları gider, submission yap.", "warn")
else:
    R.decision("Kritik sorunlar! Submission'dan önce mutlaka düzelt.", "crit")

if notes:
    R.log("─── Aksiyon Listesi ───", "info")
    for n in notes:
        R.log(n, "crit" if "🔴" in n else "warn")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# BÖLÜM 20 — Submission
# ════════════════════════════════════════════════════════════════════════════════
R.section("📤 BÖLÜM 20 — Submission Oluşturma")

sub = pd.read_csv("/kaggle/input/playground-series-s6e3/sample_submission.csv")
sub[TARGET] = final_test
sub.to_csv("/kaggle/working/submission.csv", index=False)

R.log(f"Submission kaydedildi: /kaggle/working/submission.csv", "ok")
R.log(f"Tahmin stats: mean={final_test.mean():.4f}  std={final_test.std():.4f}  "
      f"min={final_test.min():.4f}  max={final_test.max():.4f}", "info")

fig, ax = plt.subplots(figsize=(9, 4), facecolor=DARK)
ax.hist(final_test, bins=60, color=BLUE, alpha=.85, edgecolor=CARD)
ax.axvline(final_test.mean(), color=RED, lw=1.5, linestyle="--",
            label=f"Ort={final_test.mean():.4f}")
ax.axvline(opt_thr, color=ORANGE, lw=1.5, linestyle=":",
            label=f"Opt. Threshold={opt_thr:.3f}")
dark_ax(ax, "Test Tahmin Dağılımı", xl="Olasılık", yl="Frekans")
ax.legend(facecolor=CARD, labelcolor=TXT, fontsize=8)
plt.tight_layout()
R.plot(fig, "submission_dist", "Test tahmin dağılımı")
R.end()

# ── Çıktıları kaydet ─────────────────────────────────────────────────────────────
R.save("/kaggle/working/model_report.html")
print("\n✅ Tüm pipeline tamamlandı.")
print(f"   📈 Best OOF AUC      : {best_auc:.6f}")
print(f"   🎯 Brier Skoru       : {brier:.4f}")
print(f"   💰 Optimal Threshold : {opt_thr:.3f}")
print(f"   🏁 Master Skor       : {score}/{max_s} ({pct:.0%})")
print("   📄 /kaggle/working/model_report.html")
print("   📊 /kaggle/working/submission.csv")
print("   📊 /kaggle/working/model_plots/")
