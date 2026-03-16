# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DOSYA 1 — SENIOR EDA PİPELİNE                                             ║
# ║  Playground Series S6E3 · Telco Customer Churn                              ║
# ║  Çalıştır → eda_decisions.json + /kaggle/working/eda_report.html üretilir  ║
# ║  Bu dosyanın çıktıları Dosya 2 tarafından okunur.                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Kurulum ─────────────────────────────────────────────────────────────────────
import subprocess, sys
for pkg in ["statsmodels", "scipy"]:
    subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], capture_output=True)

import os, warnings, json, base64, io
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy import stats
from scipy.stats import (ks_2samp, chi2_contingency, shapiro, normaltest,
                          mannwhitneyu, levene, kruskal, spearmanr, pointbiserialr,
                          chi2 as chi2_dist)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import lightgbm as lgb

# ── Görsel tema ─────────────────────────────────────────────────────────────────
DARK   = "#0d1117"; CARD  = "#161b22"; BLUE  = "#58a6ff"
TXT    = "#c9d1d9"; GRID  = "#30363d"; RED   = "#f85149"
GREEN  = "#3fb950"; ORANGE= "#f0883e"; GOLD  = "#e3b341"
PURPLE = "#bc8cff"

sns.set_theme(style="darkgrid", palette="muted")
PLOT_DIR = Path("/kaggle/working/eda_plots")
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
class EDAReporter:
    CSS = """<html><head><meta charset="utf-8"><style>
body{font-family:'Segoe UI',monospace;background:#0a0e1a;color:#c9d1d9;max-width:1400px;margin:0 auto;padding:0}
.hero{background:#0d1117;border-bottom:2px solid #30363d;padding:36px 40px 28px;text-align:center}
.hero h1{font-size:1.9rem;font-weight:700;color:#58a6ff;margin-bottom:6px}
.hero p{color:#8b949e;font-size:.9rem}
.content{padding:0 40px 48px}
.sec{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:22px;margin:18px 0;scroll-margin-top:16px}
.sec-hdr{padding-bottom:11px;border-bottom:1px solid #30363d;margin-bottom:14px;display:flex;align-items:center;gap:10px}
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
.log-crit{border-left-color:#f85149;color:#f85149;font-weight:600}.log-info{border-left-color:#58a6ff}
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
.tag-red{background:#2d1014;color:#f85149;border:1px solid #f85149;border-radius:4px;padding:1px 7px;font-size:.69rem;font-weight:600}
.tag-green{background:#0d2818;color:#3fb950;border:1px solid #3fb950;border-radius:4px;padding:1px 7px;font-size:.69rem;font-weight:600}
.tag-amber{background:#1a1505;color:#f0883e;border:1px solid #f0883e;border-radius:4px;padding:1px 7px;font-size:.69rem;font-weight:600}
</style></head><body>
<div class="hero"><h1>&#128300; Senior EDA Raporu — Telco Churn</h1>
<p>Playground Series S6E3 &nbsp;·&nbsp; Tüm testler, tüm grafikler, aksiyon kararları</p></div>
<div class="content">"""

    def __init__(self):
        self._h = [self.CSS]
        self._decisions = []
        self._pc = 0

    def section(self, title):
        print(f"\n{'═'*70}\n{title}\n{'═'*70}")
        sid = title.replace(" ", "_").lower()[:30]
        self._h.append(f'<div class="sec" id="{sid}"><div class="sec-hdr"><h2>{title}</h2></div>')

    def end(self):
        self._h.append("</div>")

    def rule(self, text):
        self._h.append(f'<div class="rule"><div class="rule-lbl">&#128204; Kural</div>{text}</div>')

    def log(self, text, cls="info"):
        pfx = {"info":"ℹ","ok":"✅","warn":"⚠️","crit":"🔴"}.get(cls,"ℹ")
        print(f"  {pfx} {text}")
        css = {"info":"log-info","ok":"log-ok","warn":"log-warn","crit":"log-crit"}.get(cls,"log-info")
        self._h.append(f'<div class="log {css}">{pfx} {text}</div>')

    def decision(self, text, status="ok"):
        icons = {"ok":"✅","warn":"⚠️","crit":"❌"}
        css   = {"ok":"dec-ok","warn":"dec-warn","crit":"dec-crit"}
        lbl   = {"ok":"DEVAM","warn":"DİKKAT","crit":"KRİTİK AKSİYON"}
        print(f"  {icons[status]} KARAR: {text}")
        self._h.append(f'<div class="dec {css[status]}"><div class="di">{icons[status]}</div>'
                       f'<div><div class="dl">{lbl[status]}</div>{text}</div></div>')
        self._decisions.append({"status": status, "text": text})

    def metrics(self, items):
        self._h.append('<div class="metrics">')
        for lbl, val, cls in items:
            self._h.append(f'<div class="mc {cls}"><b>{val}</b><span>{lbl}</span></div>')
        self._h.append("</div>")

    def table(self, df, caption=""):
        self._h.append(df.to_html(index=False, border=0, classes="", na_rep="—", escape=False))
        if caption:
            self._h.append(f'<p style="font-size:.71rem;color:#8b949e;margin:3px 0 10px">{caption}</p>')

    def plot(self, fig, name, caption=""):
        self._pc += 1
        fname = f"{self._pc:02d}_{name}"
        path = save_fig(fig, fname)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        cap = caption or name.replace("_", " ").title()
        self._h.append(f'<div class="plot-wrap"><img src="data:image/png;base64,{b64}">'
                       f'<div class="plot-cap">{cap}</div></div>')

    def save(self, path="/kaggle/working/eda_report.html"):
        self._h.append("</div></body></html>")
        Path(path).write_text("\n".join(self._h), encoding="utf-8")
        print(f"\n✅ HTML raporu: {path}")

R = EDAReporter()

# ── Karar deposu (Dosya 2 tarafından okunur) ────────────────────────────────────
DECISIONS = {
    "drop_features":        [],
    "mnar_features":        [],
    "high_psi_features":    [],
    "conflict_rate":        0.0,
    "label_smoothing":      False,
    "sample_weight":        False,
    "ood_cat_features":     [],
    "ood_num_features":     [],
    "sentinel_cols":        [],
    "cohort_feature":       False,
    "leakage_features":     [],
    "suspect_features":     [],
    "imbalance_ratio":      1.0,
    "scale_pos_weight":     1.0,
    "high_iv_features":     [],
    "low_iv_features":      [],
    "nonlinear_features":   [],
    "high_vif_features":    [],
    "temporal_pattern":     False,
    "test_distribution_ok": True,
    "notes":                [],
}

def save_decisions():
    path = "/kaggle/working/eda_decisions.json"
    with open(path, "w") as f:
        json.dump(DECISIONS, f, indent=2)
    print(f"✅ Karar dosyası: {path}")

# ════════════════════════════════════════════════════════════════════════════════
# ADIM 1 — Veri Yükleme & Genel Bakış
# ════════════════════════════════════════════════════════════════════════════════
R.section("📊 ADIM 1 — Veri Yükleme & Genel Bakış")

IDC    = "id"
TARGET = "Churn"

train = pd.read_csv("/kaggle/input/playground-series-s6e3/train.csv", index_col=IDC)
test  = pd.read_csv("/kaggle/input/playground-series-s6e3/test.csv",  index_col=IDC)
orig  = pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Tip düzeltmeleri
for df in [train, test]:
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

# Target encode
train[TARGET] = train[TARGET].map({"Yes": 1, "No": 0})

Y = train[TARGET]
FEATS     = [c for c in train.columns if c != TARGET]
NUM_COLS  = ["tenure", "MonthlyCharges", "TotalCharges"]
CAT_COLS  = [c for c in FEATS if c not in NUM_COLS]

R.log(f"Train: {train.shape}   Test: {test.shape}   Orig: {orig.shape}", "info")
R.metrics([
    ("Train satır", f"{len(train):,}", ""),
    ("Test satır",  f"{len(test):,}",  ""),
    ("Feature",     str(len(FEATS)),   ""),
    ("Pozitif oran",f"{Y.mean():.3f}", "warn" if Y.mean()<0.15 else "ok"),
])

# Sütun eşleşme
only_train = set(train.drop(TARGET,axis=1).columns) - set(test.columns)
only_test  = set(test.columns) - set(train.drop(TARGET,axis=1).columns)
if only_train or only_test:
    R.log(f"Sadece train'de: {only_train}", "warn")
    R.log(f"Sadece test'te: {only_test}",  "warn")
    R.decision(f"Sütun uyumsuzluğu! Kontrol et: {only_train | only_test}", "crit")
else:
    R.log("Sütun isimleri eşleşiyor.", "ok")

# Genel dağılım grafiği
fig, axes = plt.subplots(2, 4, figsize=(18, 8), facecolor=DARK)
for i, col in enumerate(NUM_COLS + ["tenure"]):
    ax = axes[i//4, i%4] if i < 4 else axes[1, i-4]
    if col in NUM_COLS:
        axes[0, i].hist(train[col].dropna(), bins=40, color=BLUE, alpha=.8, edgecolor=CARD)
        axes[0, i].hist(test[col].dropna(),  bins=40, color=ORANGE, alpha=.5, edgecolor=CARD)
        dark_ax(axes[0, i], col)
        axes[0, i].legend(["Train","Test"], facecolor=CARD, labelcolor=TXT, fontsize=7)
target_counts = Y.value_counts()
axes[1,0].bar(["No Churn","Churn"], [target_counts.get(0,0), target_counts.get(1,0)],
               color=[GREEN, RED], edgecolor=GRID, alpha=.85)
dark_ax(axes[1,0], "Target Dağılımı")
for j in range(1, 4): axes[1, j].set_visible(False)
plt.tight_layout()
R.plot(fig, "genel_bakis", "Genel bakış — sayısal feature dağılımları + target")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# ADIM 2 — Target Analizi
# ════════════════════════════════════════════════════════════════════════════════
R.section("🎯 ADIM 2 — Target Analizi & Imbalance")
R.rule("Imbalance ratio > 3:1 → scale_pos_weight veya class_weight kullan. > 9:1 → SMOTE düşün.")

pos_r = Y.mean(); neg_r = 1 - pos_r
imbalance = neg_r / pos_r
DECISIONS["imbalance_ratio"]  = round(float(imbalance), 4)
DECISIONS["scale_pos_weight"] = round(float(imbalance), 4)

R.log(f"Pozitif (Churn=1): {Y.sum():,} ({pos_r:.3f})", "info")
R.log(f"Negatif (Churn=0): {(Y==0).sum():,} ({neg_r:.3f})", "info")
R.log(f"Neg/Pos oranı: {imbalance:.2f}:1  →  Önerilen scale_pos_weight: {imbalance:.2f}", "info")

if imbalance > 9:
    R.decision("Ciddi imbalance. SMOTE + scale_pos_weight birlikte uygula.", "crit")
elif imbalance > 3:
    R.decision(f"Orta imbalance ({imbalance:.1f}:1). scale_pos_weight={imbalance:.2f} kullan.", "warn")
    DECISIONS["notes"].append(f"scale_pos_weight={imbalance:.2f}")
else:
    R.decision("Dengeli dağılım. Standard eğitim uygundur.", "ok")

# Rolling target rate stabilitesi
fig, axes = plt.subplots(1, 3, figsize=(16, 4), facecolor=DARK)
axes[0].bar(["No Churn","Churn"], [target_counts.get(0,0), target_counts.get(1,0)],
             color=[GREEN,RED], edgecolor=GRID, alpha=.85)
dark_ax(axes[0], "Sınıf Dağılımı")
axes[1].pie([target_counts.get(0,1), target_counts.get(1,1)],
             labels=["No Churn","Churn"], colors=[GREEN,RED],
             autopct="%1.1f%%", startangle=90, textprops={"color":TXT,"fontsize":9})
axes[1].set_facecolor(DARK); axes[1].set_title("Sınıf Dengesi", color=BLUE, fontsize=10)
w = max(1, len(train)//60)
axes[2].plot(Y.rolling(w).mean().values, color=BLUE, lw=1, alpha=.9)
axes[2].axhline(pos_r, color=RED, lw=1.5, linestyle="--", label=f"Ort={pos_r:.3f}")
dark_ax(axes[2], "Target Oranı (Kayan Ort.)")
axes[2].legend(facecolor=CARD, labelcolor=TXT, fontsize=8)
plt.tight_layout()
R.plot(fig, "target_analizi", "Target dağılımı ve stabilitesi")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# ADIM 3 — Eksik Değer Analizi + MNAR Tespiti
# ════════════════════════════════════════════════════════════════════════════════
R.section("❓ ADIM 3 — Eksik Değer Analizi & MNAR Tespiti")
R.rule("MNAR: null satırlarda target oranı farklıysa (>10 puan) → is_null flag ekle.")
R.rule("MAR: başka sütunla açıklanabilir. MCAR: tamamen rastgele.")

null_tr = train[FEATS].isnull().mean() * 100
null_te = test[FEATS].isnull().mean() * 100
null_cols = null_tr[null_tr > 0].index.tolist()

if not null_cols:
    R.log("Hiç eksik değer yok.", "ok")
    R.decision("Eksik değer yok. Imputation adımını atla.", "ok")
else:
    rows = []
    mnar_feats = []
    for col in null_cols:
        mask   = train[col].isnull()
        r_null = Y[mask].mean() if mask.sum() > 10 else float("nan")
        r_obs  = Y[~mask].mean()
        gap    = abs(r_null - r_obs) * 100 if not pd.isna(r_null) else 0
        mech   = "MNAR" if gap > 10 else ("MAR/MCAR")
        if mech == "MNAR":
            mnar_feats.append(col)
        rows.append({"Feature": col,
                     "Train Null%": round(null_tr[col], 2),
                     "Test Null%":  round(null_te.get(col, 0), 2),
                     "Target(null)": f"{r_null:.3f}" if not pd.isna(r_null) else "—",
                     "Target(obs)":  f"{r_obs:.3f}",
                     "Gap pts":      f"{gap:.1f}",
                     "Mekanizma":    mech})
    DECISIONS["mnar_features"] = mnar_feats
    R.table(pd.DataFrame(rows))
    if mnar_feats:
        R.decision(f"MNAR feature'lar: {mnar_feats} → is_null flag ekle, drop etme!", "warn")
        DECISIONS["notes"].append(f"MNAR flags: {mnar_feats}")

    # Eksiklik korelasyonu
    if len(null_cols) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor=DARK)
        axes[0].barh(null_cols, null_tr[null_cols].values,
                     color=[RED if v>20 else ORANGE if v>5 else BLUE for v in null_tr[null_cols]],
                     edgecolor=GRID, alpha=.85)
        dark_ax(axes[0], "Train Null Oranı %")
        nm = train[null_cols].isnull().astype(int)
        sns.heatmap(nm.corr(), ax=axes[1], cmap="RdBu_r", center=0,
                    annot=True, fmt=".2f", linewidths=.3, linecolor=GRID)
        dark_ax(axes[1], "Null Korelasyonu")
        axes[1].tick_params(colors=TXT, labelsize=7)
        plt.tight_layout()
        R.plot(fig, "eksik_deger", "Eksik değer dağılımı ve korelasyonu")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# ADIM 4 — Duplike & Near-Duplicate
# ════════════════════════════════════════════════════════════════════════════════
R.section("📋 ADIM 4 — Duplike & Near-Duplicate Analizi")
R.rule("Aynı X, farklı Y grubu > %5 → label noise. drop_duplicates YAPMA.")

n_exact   = train[FEATS].duplicated(keep=False).sum()
n_unique  = train[FEATS].duplicated(keep="first").sum()
dup_rate  = n_unique / len(train) * 100

R.log(f"Tam duplike satır: {n_exact:,}  ({n_unique} fazlalık, %{dup_rate:.2f})", "info")

if n_exact > 0:
    tmp = train.copy()
    for c in tmp[FEATS].select_dtypes(["object"]).columns:
        tmp[c] = tmp[c].astype(str)
    inconsistent = (tmp.groupby(FEATS)[TARGET].nunique() > 1).sum()
    R.log(f"Tutarsız duplike grubu (aynı X, farklı Y): {inconsistent}", "warn" if inconsistent > 0 else "ok")
    if inconsistent > 0:
        R.decision("Label noise mevcut. sample_weight veya label_smoothing kullan.", "warn")
        DECISIONS["notes"].append("Label noise tespit edildi")

# Near-duplicate (sayısal rounded + kategorik)
train_r = train[NUM_COLS].round(1)
cat_combined = pd.concat([train_r, train[CAT_COLS]], axis=1)
n_near = cat_combined.duplicated(keep="first").sum()
R.log(f"Near-duplicate (num rounded + cat): {n_near:,} ({n_near/len(train)*100:.1f}%)", "info")

if dup_rate > 5:
    R.decision(f"Yüksek duplike oranı (%{dup_rate:.1f}). Orijinal veri ile birleştirmeyi dikkatli yap.", "warn")
else:
    R.decision(f"Duplike oranı kabul edilebilir (%{dup_rate:.2f}).", "ok")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# ADIM 5 — Veri Tipi & Kardinalite
# ════════════════════════════════════════════════════════════════════════════════
R.section("🔢 ADIM 5 — Veri Tipi, Kardinalite & Sentinel Değerler")
R.rule("Sentinel değerler ('No internet service', 'No phone service') → standart 'No' ile değiştir.")

# Kardinalite tablosu
rows = []
for col in FEATS:
    n   = train[col].nunique()
    top = train[col].value_counts(normalize=True).iloc[0] * 100
    rows.append({"Feature":col, "Tip":str(train[col].dtype),
                 "Unique":n, "Top%":f"{top:.1f}",
                 "Uyarı":("SABİT ❌" if n==1 else "near-const ⚠️" if top>99 else
                           "binary" if n==2 else "low-card" if n<=10 else "high-card" if n/len(train)>0.5 else "")})
R.table(pd.DataFrame(rows))

# Sentinel tespiti
sentinel_vals = ["No internet service","No phone service","Unknown","N/A","na","none",""]
sentinel_hits = {}
for col in CAT_COLS:
    for sv in sentinel_vals:
        cnt = (train[col] == sv).sum()
        if cnt > 0:
            sentinel_hits[f"{col}='{sv}'"] = cnt
if sentinel_hits:
    R.log("Sentinel değerler tespit edildi:", "warn")
    for k, v in sorted(sentinel_hits.items(), key=lambda x: -x[1]):
        R.log(f"  {k}: {v:,} satır", "warn")
    DECISIONS["sentinel_cols"] = [k.split("=")[0] for k in sentinel_hits]
    R.decision("Tüm sentinel değerleri 'No' ile değiştir (FE Adım 0).", "warn")
else:
    R.log("Sentinel değer yok.", "ok")

# Sabit sütunlar
constant = [c for c in FEATS if train[c].nunique() <= 1]
if constant:
    R.decision(f"Sabit sütunlar, drop et: {constant}", "crit")
    DECISIONS["drop_features"].extend(constant)
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# ADIM 6 — Outlier Analizi
# ════════════════════════════════════════════════════════════════════════════════
R.section("🔍 ADIM 6 — Outlier Analizi (IQR + Z-score)")
R.rule("IQR outlier > %5 → Winsorize (1-99. percentile). Z>4 → uç değer.")

rows = []
for col in NUM_COLS:
    v = train[col].dropna()
    Q1, Q3 = v.quantile(.25), v.quantile(.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    n_iqr = int(((v < lo) | (v > hi)).sum())
    n_z4  = int((abs(stats.zscore(v)) > 4).sum())
    r_out = Y[train[col].notna()][((v < lo) | (v > hi))].mean()
    r_in  = Y[train[col].notna()][~((v < lo) | (v > hi))].mean()
    rows.append({"Feature": col,
                 "IQR Outlier": n_iqr,
                 "IQR%": f"{n_iqr/len(train)*100:.2f}",
                 "Z>4": n_z4,
                 "Churn(outlier)": f"{r_out:.3f}",
                 "Churn(normal)": f"{r_in:.3f}",
                 "Skewness": f"{v.skew():.3f}"})
R.table(pd.DataFrame(rows))

fig, axes = plt.subplots(1, len(NUM_COLS), figsize=(14, 4), facecolor=DARK)
for i, col in enumerate(NUM_COLS):
    d = [train.loc[Y==0, col].dropna().values, train.loc[Y==1, col].dropna().values]
    bp = axes[i].boxplot(d, patch_artist=True,
                          medianprops=dict(color=GOLD, lw=2))
    bp["boxes"][0].set_facecolor(GREEN + "33")
    bp["boxes"][1].set_facecolor(RED + "33")
    axes[i].set_xticklabels(["No Churn","Churn"], color=TXT, fontsize=8)
    dark_ax(axes[i], col)
plt.tight_layout()
R.plot(fig, "outlier_boxplot", "Outlier analizi — Target grubuna göre box plot")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# ADIM 7 — Dağılım Analizi + KDE + Q-Q Plot
# ════════════════════════════════════════════════════════════════════════════════
R.section("📈 ADIM 7 — Dağılım Analizi + KDE + Q-Q Plot")
R.rule("KS istatistiği >0.3 → güçlü ayrıştırıcı. Skewness >2 → log transform dene.")

# KDE
fig, axes = plt.subplots(2, len(NUM_COLS), figsize=(15, 8), facecolor=DARK)
ks_results = {}
for i, col in enumerate(NUM_COLS):
    a = train.loc[Y==0, col].dropna()
    b = train.loc[Y==1, col].dropna()
    ks_stat, ks_p = ks_2samp(a, b)
    ks_results[col] = {"ks": ks_stat, "p": ks_p}

    axes[0,i].hist(a, bins=40, alpha=.55, color=GREEN, density=True, edgecolor=CARD, label="No Churn")
    axes[0,i].hist(b, bins=40, alpha=.55, color=RED,   density=True, edgecolor=CARD, label="Churn")
    border_col = RED if ks_stat>.3 else ORANGE if ks_stat>.15 else BLUE
    dark_ax(axes[0,i], f"{col}\n[KS={ks_stat:.3f}, p={ks_p:.4f}]")
    for s in axes[0,i].spines.values(): s.set_color(border_col)
    axes[0,i].legend(facecolor=CARD, labelcolor=TXT, fontsize=7)

    # Q-Q
    from scipy.stats import probplot
    sample = train[col].dropna().sample(min(2000,len(train[col].dropna())),random_state=42)
    res = probplot(sample, dist="norm")
    axes[1,i].scatter(res[0][0], res[0][1], alpha=.3, s=5, color=BLUE)
    x = np.array(res[0][0])
    axes[1,i].plot(x, res[1][1]+res[1][0]*x, color=RED, lw=1.5)
    dark_ax(axes[1,i], f"Q-Q: {col}")

plt.tight_layout()
R.plot(fig, "kde_qq", "KDE dağılım ve Q-Q normalite grafikleri")

strong_ks = [c for c,v in ks_results.items() if v["ks"]>.3]
if strong_ks:
    R.decision(f"Güçlü KS ayrıştırıcılar: {strong_ks} → FE'de önceliklendir.", "ok")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# ADIM 8 — Normalite Testleri (Shapiro-Wilk + D'Agostino)
# ════════════════════════════════════════════════════════════════════════════════
R.section("📐 ADIM 8 — Normalite Testleri")
R.rule("Shapiro-Wilk p<0.05 → non-normal. Non-normal feature'lar için log/rank transform uygula.")

rows = []
for col in NUM_COLS:
    v = train[col].dropna()
    samp = v.sample(min(5000, len(v)), random_state=42)
    try:    sw_p = float(shapiro(samp)[1])
    except: sw_p = 0.0
    da_p = float(normaltest(samp)[1])
    rows.append({"Feature": col,
                 "Shapiro-Wilk p": f"{sw_p:.5f}",
                 "D'Agostino p":   f"{da_p:.5f}",
                 "Skewness":       f"{v.skew():.3f}",
                 "Kurtosis":       f"{v.kurtosis():.3f}",
                 "Normal?":        "❌ Hayır" if sw_p<.05 else "✅ Evet"})
R.table(pd.DataFrame(rows))
non_normal = [r["Feature"] for r in rows if "❌" in r["Normal?"]]
if non_normal:
    DECISIONS["notes"].append(f"Non-normal features: {non_normal} → log transform")
    R.decision(f"Non-normal dağılım: {non_normal}. FE'de log1p transform uygula.", "warn")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# ADIM 9 — Mann-Whitney U + Cohen's d + Levene Testi
# ════════════════════════════════════════════════════════════════════════════════
R.section("⚖️ ADIM 9 — Mann-Whitney U, Cohen's d & Levene Testi")
R.rule("Cohen d > 0.8 → büyük etki (öncelikli FE). d < 0.2 → önemsiz (drop adayı).")
R.rule("Levene p<0.05 → varyanslar eşit değil. Model bu feature'ı farklı öğrenebilir.")

rows = []
for col in NUM_COLS:
    a = train.loc[Y==0, col].dropna().values
    b = train.loc[Y==1, col].dropna().values
    if len(a)<5 or len(b)<5: continue
    _, mwu_p  = mannwhitneyu(a, b, alternative="two-sided")
    _, lev_p  = levene(a, b)
    pool = np.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
    cd   = (np.mean(b)-np.mean(a))/(pool+1e-9)
    eff  = "Büyük" if abs(cd)>.8 else "Orta" if abs(cd)>.5 else "Küçük" if abs(cd)>.2 else "İhmal"
    rows.append({"Feature":col,"MWU p":f"{mwu_p:.5f}","Cohen d":round(cd,4),
                 "Etki":eff,"Levene p":f"{lev_p:.4f}","Anlamlı":"✅" if mwu_p<.05 else "❌"})
R.table(pd.DataFrame(rows))

fig, ax = plt.subplots(figsize=(9, 4), facecolor=DARK)
cds = [r["Cohen d"] for r in rows]
ftrs = [r["Feature"] for r in rows]
ax.barh(ftrs, cds, color=[GREEN if cd>0 else RED for cd in cds], edgecolor=GRID, alpha=.85)
ax.axvline(0, color=TXT, lw=1)
ax.axvline(.8,  color=GOLD, lw=1, linestyle="--", label="Large (0.8)")
ax.axvline(-.8, color=GOLD, lw=1, linestyle="--")
ax.axvline(.2,  color=GRID, lw=1, linestyle=":", label="Small (0.2)")
dark_ax(ax, "Cohen's d Etki Büyüklüğü")
ax.legend(facecolor=CARD, labelcolor=TXT, fontsize=8)
plt.tight_layout()
R.plot(fig, "cohen_d", "Mann-Whitney + Cohen d etki büyüklüğü")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# ADIM 10 — Chi-Square + Cramér's V + Kruskal-Wallis + Bonferroni
# ════════════════════════════════════════════════════════════════════════════════
R.section("📊 ADIM 10 — Chi-Square, Cramér's V, Kruskal-Wallis & Bonferroni FDR")
R.rule("Cramér's V > 0.3 → güçlü kategorik ilişki. FE interaction adayı.")
R.rule("Çoklu karşılaştırma: Bonferroni ile düzeltilmiş p kullan (α/n).")

# Cramér's V — kategorik ↔ target
def cramers_v(x, y):
    try:
        ct = pd.crosstab(x, y)
        chi2_v = chi2_contingency(ct)[0]
        n = ct.sum().sum(); md = min(ct.shape)-1
        return float(np.sqrt(chi2_v/(n*md))) if md>0 else 0.0
    except: return 0.0

cv_rows = []
chi_pvals = []
for col in CAT_COLS:
    try:
        ct = pd.crosstab(train[col], Y)
        chi2_v, p, dof, _ = chi2_contingency(ct)
        cv = cramers_v(train[col], Y)
        cv_rows.append({"Feature":col, "Chi² Stat":f"{chi2_v:.1f}",
                        "dof":dof, "p-değeri":f"{p:.5f}",
                        "Cramér's V":round(cv,4),
                        "Güç":("Güçlü" if cv>.3 else "Orta" if cv>.1 else "Zayıf"),
                        "Anlamlı":"✅" if p<.05 else "❌"})
        chi_pvals.append(p)
    except: pass

# Bonferroni düzeltmesi
alpha = 0.05
bonf_threshold = alpha / max(len(chi_pvals), 1)
for r in cv_rows:
    r["Bonferroni OK"] = "✅" if float(r["p-değeri"]) < bonf_threshold else "❌"

df_cv = pd.DataFrame(cv_rows).sort_values("Cramér's V", ascending=False)
R.table(df_cv)

fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor=DARK)
dfp = df_cv.sort_values("Cramér's V", ascending=True)
axes[0].barh(dfp["Feature"], dfp["Cramér's V"],
              color=[RED if v>.3 else ORANGE if v>.1 else BLUE for v in dfp["Cramér's V"]],
              edgecolor=GRID, alpha=.85)
axes[0].axvline(.3, color=RED, lw=1.5, linestyle="--", label="0.3 Güçlü")
axes[0].axvline(.1, color=ORANGE, lw=1, linestyle="--", label="0.1 Orta")
dark_ax(axes[0], "Cramér's V (Kategorik ↔ Target)")
axes[0].legend(facecolor=CARD, labelcolor=TXT, fontsize=8)

# Kategorik churn rates (top 6)
top_cats = df_cv.head(6)["Feature"].tolist()
y_pos = 0
bar_h = 0.6
feat_colors = [BLUE, ORANGE, GREEN, PURPLE, GOLD, RED]
for fi, col in enumerate(top_cats[:3]):
    ct_r = train.groupby(col)[TARGET].mean().sort_values(ascending=True)
    for vi, (val, rate) in enumerate(ct_r.items()):
        axes[1].barh(y_pos, rate*100, bar_h, color=feat_colors[fi], alpha=.75, edgecolor=GRID)
        axes[1].text(rate*100+.5, y_pos, f"{val} ({rate:.2f})", va="center", color=TXT, fontsize=7)
        y_pos += 1
    y_pos += .5
dark_ax(axes[1], "Churn Oranı — Kategorik Feature'lar (Top 3)")
axes[1].set_xlabel("Churn %", color=TXT)
plt.tight_layout()
R.plot(fig, "chi_cramers", "Chi-square, Cramér's V ve kategorik churn oranları")

strong_cv = [r["Feature"] for r in cv_rows if r["Cramér's V"] > .3]
R.log(f"Bonferroni eşiği (α={alpha}/n={len(chi_pvals)}): p < {bonf_threshold:.5f}", "info")
if strong_cv:
    R.decision(f"Güçlü Cramér's V feature'ları: {strong_cv} → target encoding öncelikli uygula.", "ok")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# ADIM 11 — Korelasyon + VIF + MI vs Pearson
# ════════════════════════════════════════════════════════════════════════════════
R.section("🔗 ADIM 11 — Korelasyon, VIF & MI vs Pearson (Non-Linear Tespit)")
R.rule("VIF > 10 → kritik çoklu bağlantı. ratio/diff feature türet veya birini drop et.")
R.rule("MI yüksek + Pearson düşük → non-linear ilişki. log/binning uygula.")

corr = train[NUM_COLS].corr()
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=DARK)

# Korelasyon matrisi
sns.heatmap(corr, ax=axes[0], cmap="RdBu_r", center=0,
            annot=True, fmt=".3f", linewidths=.3, linecolor=GRID)
dark_ax(axes[0], "Pearson Korelasyon Matrisi")
axes[0].tick_params(colors=TXT, labelsize=8)

# Target korelasyonu
tc = train[NUM_COLS].apply(lambda c: c.corr(Y.astype(float))).abs().sort_values(ascending=True)
axes[1].barh(tc.index, tc.values,
              color=[RED if v>.3 else ORANGE if v>.15 else BLUE for v in tc.values],
              edgecolor=GRID, alpha=.85)
dark_ax(axes[1], "|Korelasyon| ile Target")

# MI vs Pearson scatter
X_mi = train[NUM_COLS].fillna(0)
mi = mutual_info_classif(X_mi, Y, random_state=42)
mi_n = mi / (mi.max() + 1e-9)
pv = [abs(train[c].fillna(0).corr(Y.astype(float))) for c in NUM_COLS]
axes[2].scatter(pv, mi_n, c=[RED if (m-p)>.2 else ORANGE if (m-p)>.1 else GREEN for m,p in zip(mi_n,pv)],
                s=120, zorder=3, alpha=.9)
for i, col in enumerate(NUM_COLS):
    axes[2].annotate(col, (pv[i], mi_n[i]), xytext=(5,3), textcoords="offset points",
                     fontsize=8, color=TXT)
lim = max(max(pv), max(mi_n)) * 1.1
axes[2].plot([0,lim],[0,lim], color=GRID, linestyle="--", lw=1, label="MI=Pearson")
dark_ax(axes[2], "MI vs Pearson — Non-Linear Tespit",
        xl="Pearson |r|", yl="MI (normalize)")
axes[2].legend(facecolor=CARD, labelcolor=TXT, fontsize=8)
plt.tight_layout()
R.plot(fig, "korelasyon_vif_mi", "Korelasyon, MI vs Pearson analizi")

# VIF hesaplama
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    Xv = train[NUM_COLS].fillna(train[NUM_COLS].median())
    Xv = (Xv - Xv.mean()) / (Xv.std() + 1e-9)
    vif_rows = []
    high_vif = []
    for i, col in enumerate(NUM_COLS):
        v = float(variance_inflation_factor(Xv.values, i))
        vif_rows.append({"Feature": col, "VIF": round(v,2),
                         "Durum": "🔴 KRİTİK" if v>10 else "🟡 YÜKSEK" if v>5 else "✅ OK"})
        if v > 10: high_vif.append(col)
    R.table(pd.DataFrame(vif_rows))
    DECISIONS["high_vif_features"] = high_vif
    if high_vif:
        R.decision(f"Yüksek VIF: {high_vif}. Tree modeller için sorun yok; LR için ratio feature ekle.", "warn")
except ImportError:
    R.log("statsmodels yüklü değil, VIF atlandı.", "warn")

# Non-linear tespit
nl_feats = []
for i, col in enumerate(NUM_COLS):
    gap = mi_n[i] - abs(train[col].fillna(0).corr(Y.astype(float)))
    if gap > .15:
        nl_feats.append(col)
DECISIONS["nonlinear_features"] = nl_feats
if nl_feats:
    R.decision(f"Non-linear feature'lar: {nl_feats}. Binning + log transform dene.", "warn")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# ADIM 12 — IV / WoE Analizi
# ════════════════════════════════════════════════════════════════════════════════
R.section("💡 ADIM 12 — Information Value (IV) & Weight of Evidence (WoE)")
R.rule("IV > 0.5 = Güçlü. IV 0.1–0.3 = Orta. IV 0.02–0.1 = Zayıf. IV < 0.02 = Anlamsız.")

def calc_iv(df, col, target):
    try:
        g = df.groupby(col)[target].agg(["sum","count"])
        g.columns = ["ev","tot"]; g["nev"] = g["tot"] - g["ev"]
        te, tne = max(g["ev"].sum(),1), max(g["nev"].sum(),1)
        g["pe"] = (g["ev"]/te).clip(1e-4)
        g["pn"] = (g["nev"]/tne).clip(1e-4)
        g["woe"] = np.log(g["pn"]/g["pe"])
        g["iv"]  = (g["pn"]-g["pe"])*g["woe"]
        return float(g["iv"].sum()), g
    except: return 0.0, pd.DataFrame()

tmp = train.copy()
iv_rows = []
for col in NUM_COLS:
    try:
        tmp[col+"_bin"] = pd.qcut(tmp[col], 10, duplicates="drop").astype(str)
        iv, _ = calc_iv(tmp, col+"_bin", TARGET)
        iv_rows.append({"Feature":col,"IV":round(iv,4),"Tip":"Sayısal",
                        "Güç":("Güçlü" if iv>.5 else "Orta" if iv>.1 else "Zayıf" if iv>.02 else "Anlamsız")})
    except: pass
for col in CAT_COLS:
    iv, _ = calc_iv(train, col, TARGET)
    iv_rows.append({"Feature":col,"IV":round(iv,4),"Tip":"Kategorik",
                    "Güç":("Güçlü" if iv>.5 else "Orta" if iv>.1 else "Zayıf" if iv>.02 else "Anlamsız")})

df_iv = pd.DataFrame(iv_rows).sort_values("IV", ascending=False)
R.table(df_iv)

DECISIONS["high_iv_features"] = df_iv[df_iv["IV"]>.1]["Feature"].tolist()
DECISIONS["low_iv_features"]  = df_iv[df_iv["IV"]<.02]["Feature"].tolist()

fig, ax = plt.subplots(figsize=(10, max(5,len(iv_rows)*.4)), facecolor=DARK)
dfp = df_iv.sort_values("IV", ascending=True)
ax.barh(dfp["Feature"], dfp["IV"],
        color=[RED if v>.5 else ORANGE if v>.1 else BLUE if v>.02 else GRID for v in dfp["IV"]],
        edgecolor=GRID, alpha=.85)
for thr, col_, lbl in [(.5,RED,"0.5 Güçlü"),(.1,ORANGE,"0.1 Orta"),(.02,BLUE,"0.02 Zayıf")]:
    ax.axvline(thr, color=col_, lw=1.5, linestyle="--", label=lbl)
dark_ax(ax, "Information Value (IV) — Tüm Feature'lar")
ax.legend(facecolor=CARD, labelcolor=TXT, fontsize=8)
plt.tight_layout()
R.plot(fig, "iv_woe", "IV analizi — feature tahmin gücü sıralaması")

if DECISIONS["low_iv_features"]:
    R.decision(f"Anlamsız IV (<0.02): {DECISIONS['low_iv_features'][:5]} → FE'de düşük öncelik.", "warn")
if DECISIONS["high_iv_features"]:
    R.decision(f"Güçlü IV (>0.1): {DECISIONS['high_iv_features'][:5]} → target encoding öncelikli.", "ok")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# ADIM 13 — PSI (Population Stability Index)
# ════════════════════════════════════════════════════════════════════════════════
R.section("📉 ADIM 13 — PSI — Popülasyon Stabilitesi (Train vs Test)")
R.rule("PSI < 0.10 → stabil. 0.10–0.25 → dikkat. > 0.25 → kritik drift, feature'ı drop et.")

def psi_score(expected, actual, buckets=10):
    mn, mx = min(expected.min(),actual.min()), max(expected.max(),actual.max())
    breaks = np.linspace(mn, mx, buckets+1)
    ep = np.histogram(expected, bins=breaks)[0] / len(expected)
    ap = np.histogram(actual,   bins=breaks)[0] / len(actual)
    ep = np.where(ep==0, 1e-4, ep); ap = np.where(ap==0, 1e-4, ap)
    return float(np.sum((ap-ep)*np.log(ap/ep)))

psi_rows = []
for col in NUM_COLS:
    tr = train[col].dropna(); te = test[col].dropna()
    if len(tr)<5 or len(te)<5: continue
    p = psi_score(tr.values, te.values)
    psi_rows.append({"Feature":col,"PSI":round(p,4),
                     "Şiddet":("🔴 Kritik" if p>.25 else "🟡 Orta" if p>.1 else "✅ Stabil")})
DECISIONS["high_psi_features"] = [r["Feature"] for r in psi_rows if r["PSI"]>.25]

df_psi = pd.DataFrame(psi_rows).sort_values("PSI", ascending=False)
R.table(df_psi)

fig, ax = plt.subplots(figsize=(9, 4), facecolor=DARK)
dfp = df_psi.sort_values("PSI", ascending=True)
ax.barh(dfp["Feature"], dfp["PSI"],
        color=[RED if v>.25 else ORANGE if v>.1 else GREEN for v in dfp["PSI"]],
        edgecolor=GRID, alpha=.85)
ax.axvline(.25, color=RED,    lw=1.5, linestyle="--", label="0.25 Kritik")
ax.axvline(.10, color=ORANGE, lw=1,   linestyle="--", label="0.10 Orta")
dark_ax(ax, "PSI — Train vs Test Drift")
ax.legend(facecolor=CARD, labelcolor=TXT, fontsize=8)
plt.tight_layout()
R.plot(fig, "psi", "PSI analizi — train/test dağılım stabilitesi")

psi_max = df_psi["PSI"].max() if not df_psi.empty else 0
if DECISIONS["high_psi_features"]:
    R.decision(f"Kritik PSI: {DECISIONS['high_psi_features']} → drop et veya RobustScaler uygula.", "crit")
elif psi_max > .1:
    R.decision("Orta PSI drift var. RobustScaler ve rank transform dene.", "warn")
else:
    R.decision("PSI stabil. Train-test benzer dağılım.", "ok")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# ADIM 14 — Adversarial Validation
# ════════════════════════════════════════════════════════════════════════════════
R.section("⚔️ ADIM 14 — Adversarial Validation")
R.rule("AUC > 0.70 → ciddi covariate shift. Model test'te başarısız olabilir.")

adv_y = np.concatenate([np.zeros(len(train)), np.ones(len(test))])
Xtr = train[FEATS].copy(); Xte = test[FEATS].copy()
for c in Xtr.select_dtypes(["object","category"]).columns:
    Xtr[c] = pd.factorize(Xtr[c])[0]
    Xte[c] = pd.factorize(Xte[c])[0]
Xall = pd.concat([Xtr.fillna(-1), Xte.fillna(-1)], ignore_index=True)

adv_aucs = []; best_m = None
for tr_i, va_i in StratifiedKFold(3, shuffle=True, random_state=42).split(Xall, adv_y):
    m = lgb.LGBMClassifier(200, learning_rate=.05, max_depth=4, verbose=-1, n_jobs=-1, random_state=42)
    m.fit(Xall.iloc[tr_i], adv_y[tr_i])
    adv_aucs.append(roc_auc_score(adv_y[va_i], m.predict_proba(Xall.iloc[va_i])[:,1]))
    best_m = m
adv_auc = float(np.mean(adv_aucs))
DECISIONS["test_distribution_ok"] = adv_auc < .70

imp_df = pd.DataFrame({"Feature":FEATS,"Importance":best_m.feature_importances_}).sort_values("Importance",ascending=False).head(12)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK)
axes[0].barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1], color=ORANGE, edgecolor=GRID, alpha=.85)
dark_ax(axes[0], "Adversarial Shift — En Önemli Feature'lar")
axes[1].bar(range(1,4), adv_aucs, color=[GREEN if a<.65 else ORANGE if a<.70 else RED for a in adv_aucs],
             edgecolor=GRID, alpha=.85)
axes[1].axhline(adv_auc, color=GOLD, lw=2, linestyle="--", label=f"Ort={adv_auc:.4f}")
axes[1].axhline(.70, color=RED, lw=1.5, linestyle=":", label="0.70 Eşiği")
dark_ax(axes[1], "Adversarial AUC per Fold")
axes[1].legend(facecolor=CARD, labelcolor=TXT, fontsize=8)
plt.tight_layout()
R.plot(fig, "adversarial", f"Adversarial validation — AUC={adv_auc:.4f}")

R.log(f"Adversarial AUC: {adv_auc:.4f}", "info")
if adv_auc > .80:
    R.decision(f"Ciddi covariate shift (AUC={adv_auc:.4f}). Target encoding ve rank transform zorunlu.", "crit")
elif adv_auc > .70:
    R.decision(f"Orta shift (AUC={adv_auc:.4f}). PSI yüksek feature'ları dikkatli ele al.", "warn")
else:
    R.decision(f"Train/test dağılımı benzer (AUC={adv_auc:.4f}).", "ok")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# ADIM 15 — Leakage Tespiti (CV-based Solo AUC)
# ════════════════════════════════════════════════════════════════════════════════
R.section("🚨 ADIM 15 — Data Leakage Tespiti (5-fold Solo AUC)")
R.rule("Solo AUC > 0.95 → LEAK, hemen drop. 0.85–0.95 → domain kontrolü.")

X_enc = train[FEATS].copy()
for c in X_enc.select_dtypes(["object","category"]).columns:
    X_enc[c] = pd.factorize(X_enc[c])[0]
X_enc = X_enc.fillna(-1)

skf5 = StratifiedKFold(5, shuffle=True, random_state=42)
leak_rows = []
for col in FEATS:
    try:
        xc = X_enc[[col]]; aucs = []
        for tr_i, va_i in skf5.split(xc, Y):
            rf = RandomForestClassifier(50, max_depth=4, random_state=42, n_jobs=-1)
            rf.fit(xc.iloc[tr_i], Y.iloc[tr_i])
            aucs.append(roc_auc_score(Y.iloc[va_i], rf.predict_proba(xc.iloc[va_i])[:,1]))
        auc_m = float(np.mean(aucs))
        risk = "🔴 LEAK!" if auc_m>.95 else "🟠 Şüpheli" if auc_m>.85 else "🟡 Güçlü" if auc_m>.75 else "✅ OK"
        leak_rows.append({"Feature":col,"Solo AUC":round(auc_m,4),"Risk":risk})
    except: pass

df_leak = pd.DataFrame(leak_rows).sort_values("Solo AUC", ascending=False)
R.table(df_leak)

leaks   = [r["Feature"] for r in leak_rows if r["Solo AUC"]>.95]
suspects= [r["Feature"] for r in leak_rows if .85<r["Solo AUC"]<=.95]
DECISIONS["leakage_features"] = leaks
DECISIONS["suspect_features"] = suspects

fig, ax = plt.subplots(figsize=(10, max(4, len(FEATS)*.35)), facecolor=DARK)
dfp = df_leak.sort_values("Solo AUC", ascending=True)
ax.barh(dfp["Feature"], dfp["Solo AUC"],
        color=[RED if v>.95 else ORANGE if v>.85 else GOLD if v>.75 else GREEN for v in dfp["Solo AUC"]],
        edgecolor=GRID, alpha=.85)
ax.axvline(.95, color=RED, lw=1.5, linestyle="--", label="0.95 Leak")
ax.axvline(.85, color=ORANGE, lw=1, linestyle="--", label="0.85 Şüpheli")
dark_ax(ax, "Leakage Tespiti — Feature Başına Solo AUC")
ax.legend(facecolor=CARD, labelcolor=TXT, fontsize=8)
plt.tight_layout()
R.plot(fig, "leakage", "Solo AUC per feature — leakage tespiti")

if leaks:
    R.decision(f"SIZINTI TESPİT EDİLDİ: {leaks}. Eğitimden önce KALDIR!", "crit")
elif suspects:
    R.decision(f"Şüpheli feature'lar: {suspects}. Domain uzmanıyla kontrol et.", "warn")
else:
    R.decision("Belirgin leakage tespit edilmedi.", "ok")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# ADIM 16 — Label Kalitesi & Conflict Rate
# ════════════════════════════════════════════════════════════════════════════════
R.section("🏷️ ADIM 16 — Label Kalitesi & Conflict Rate")
R.rule("Conflict rate > %10 → sample_weight zorunlu. %5–10 → label_smoothing=0.05 dene.")

key_cols = FEATS[:min(6, len(FEATS))]
tmp = train[key_cols + [TARGET]].copy()
for c in tmp.select_dtypes(["object","category"]).columns:
    tmp[c] = tmp[c].astype(str)
try:
    grp = tmp.groupby(key_cols)[TARGET].agg(["nunique","count"])
    conflict = grp[grp["nunique"]>1]["count"].sum()
except:
    conflict = 0
conflict_rate = float(conflict / len(train) * 100)
DECISIONS["conflict_rate"] = round(conflict_rate, 4)

exact_dup = X_enc.duplicated(keep=False).sum()
R.metrics([
    ("Conflict Sayısı",   f"{int(conflict):,}",     "crit" if conflict_rate>10 else "warn" if conflict_rate>5 else "ok"),
    ("Conflict %",        f"{conflict_rate:.2f}%",   "crit" if conflict_rate>10 else "warn" if conflict_rate>5 else "ok"),
    ("Exact Dup Satır",   f"{exact_dup:,}",          "warn" if exact_dup>0 else "ok"),
])

if conflict_rate > 10:
    DECISIONS["sample_weight"] = True
    R.decision(f"Yüksek label noise (%{conflict_rate:.1f}). sample_weight = (1 / conflict_proba) uygula.", "crit")
elif conflict_rate > 5:
    DECISIONS["label_smoothing"] = True
    R.decision(f"Orta label noise (%{conflict_rate:.1f}). LightGBM'de label_smoothing=0.05 dene.", "warn")
else:
    R.decision(f"Label kalitesi iyi (%{conflict_rate:.2f}).", "ok")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# ADIM 17 — OOD Tespiti (Out-of-Distribution)
# ════════════════════════════════════════════════════════════════════════════════
R.section("🔭 ADIM 17 — OOD Tespiti (Test'te Yeni Değerler)")
R.rule("Yeni kategoriler → target encoding veya 'Unknown' token. Range dışı → Winsorize.")

ood_rows = []
for col in CAT_COLS:
    new = set(test[col].dropna().unique()) - set(train[col].dropna().unique())
    if new:
        cnt = int(test[col].isin(new).sum())
        ood_rows.append({"Kolon":col,"Tip":"Kategorik OOD","Sayı":cnt,"Değerler":str(list(new)[:3])})
        DECISIONS["ood_cat_features"].append(col)
for col in NUM_COLS:
    tr_min, tr_max = train[col].min(), train[col].max()
    n_out = int(((test[col]<tr_min)|(test[col]>tr_max)).sum())
    if n_out > 0:
        ood_rows.append({"Kolon":col,"Tip":"Sayısal Range Dışı","Sayı":n_out,
                         "Değerler":f"Train:[{tr_min:.1f},{tr_max:.1f}]"})
        DECISIONS["ood_num_features"].append(col)

if ood_rows:
    R.table(pd.DataFrame(ood_rows))
    if DECISIONS["ood_cat_features"]:
        R.decision(f"Kategorik OOD: {DECISIONS['ood_cat_features']} → target encoding veya unknown token.", "warn")
    if DECISIONS["ood_num_features"]:
        R.decision(f"Sayısal OOD: {DECISIONS['ood_num_features']} → Winsorize (1-99. percentile).", "warn")
else:
    R.log("OOD değer tespit edilmedi.", "ok")
    R.decision("Test seti train ile uyumlu dağılımda.", "ok")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# ADIM 18 — Cohort Drift Analizi
# ════════════════════════════════════════════════════════════════════════════════
R.section("⏳ ADIM 18 — Cohort Drift Analizi (Tenure Bazlı)")
R.rule("Cohort churn spread > 20 puan → tenure_group kategorik feature oluştur.")
R.rule("Cohort spread 10–20 puan → tenure_bin ekle.")

if "tenure" in train.columns:
    tmp_c = train.copy()
    tmp_c["tenure_num"] = pd.to_numeric(tmp_c["tenure"], errors="coerce")
    tmp_c["cohort"] = pd.cut(tmp_c["tenure_num"], bins=[0,12,24,36,48,100],
                              labels=["0-12","12-24","24-36","36-48","48+"], right=True)
    cs = tmp_c.groupby("cohort")[TARGET].agg(["mean","count"])
    cs.columns = ["Churn Rate","Count"]; cs["Churn Rate %"] = cs["Churn Rate"]*100
    R.table(cs.reset_index().rename(columns={"cohort":"Cohort"}).round(3))

    rates = cs["Churn Rate"].values
    cohort_spread = float((max(rates) - min(rates)) * 100)
    DECISIONS["cohort_feature"] = cohort_spread > 10

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor=DARK)
    axes[0].bar(cs.index.astype(str), cs["Churn Rate %"].values,
                color=[RED if r>30 else ORANGE if r>20 else GREEN for r in cs["Churn Rate %"]],
                edgecolor=GRID, alpha=.85)
    axes[0].axhline(Y.mean()*100, color=GOLD, lw=1.5, linestyle="--", label=f"Overall={Y.mean()*100:.1f}%")
    dark_ax(axes[0], "Tenure Cohort'a Göre Churn Oranı", xl="Cohort", yl="Churn %")
    axes[0].legend(facecolor=CARD, labelcolor=TXT, fontsize=8)

    axes[1].bar(cs.index.astype(str), cs["Count"].values, color=BLUE, edgecolor=GRID, alpha=.85)
    dark_ax(axes[1], "Cohort Örnek Sayısı", xl="Cohort", yl="N")
    plt.tight_layout()
    R.plot(fig, "cohort_drift", f"Cohort drift analizi — spread={cohort_spread:.1f} puan")

    R.log(f"Cohort churn spread: {cohort_spread:.1f} puan", "info")
    if cohort_spread > 20:
        R.decision(f"Büyük cohort farkı ({cohort_spread:.1f}p). tenure_group kategorik feature oluştur.", "warn")
    elif cohort_spread > 10:
        R.decision(f"Orta cohort farkı ({cohort_spread:.1f}p). tenure_bin (pd.cut) ekle.", "warn")
    else:
        R.decision(f"Küçük cohort farkı ({cohort_spread:.1f}p). Lineer tenure yeterli.", "ok")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# ADIM 19 — Benford Yasası
# ════════════════════════════════════════════════════════════════════════════════
R.section("🔢 ADIM 19 — Benford Yasası (Sayısal Veri Doğruluğu)")
R.rule("MAD > 0.015 → Benford'dan ciddi sapma (sentetik / manipüle veri şüphesi).")

benford_exp = np.array([np.log10(1+1/d) for d in range(1,10)])
benf_rows = []; benf_plot_col = None; benf_plot_obs = None

for col in NUM_COLS:
    v = train[col].dropna().abs(); v = v[v>0]
    if len(v)<100: continue
    fd = v.astype(str).str.replace(r"^0\.0*","",regex=True).str[0]
    fd = fd[fd.str.isdigit()&(fd!="0")]
    if len(fd)<50: continue
    obs = np.array([(fd==str(d)).mean() for d in range(1,10)])
    mad = float(np.mean(np.abs(obs-benford_exp)))
    benf_rows.append({"Feature":col,"MAD":round(mad,5),"N":len(fd),
                      "Uyum":("✅ Uyuyor" if mad<.006 else "🟡 Hafif Sapma" if mad<.015 else "🔴 Sapıyor")})
    if benf_plot_col is None: benf_plot_col=col; benf_plot_obs=obs

if benf_rows:
    R.table(pd.DataFrame(benf_rows))
    if benf_plot_col:
        fig, ax = plt.subplots(figsize=(9, 4), facecolor=DARK)
        x = np.arange(1,10)
        ax.bar(x-.2, benford_exp*100, .35, color=BLUE, alpha=.8, label="Benford Beklenen")
        ax.bar(x+.2, benf_plot_obs*100, .35, color=ORANGE, alpha=.8, label=f"Gözlenen ({benf_plot_col})")
        dark_ax(ax, "Benford Yasası Analizi", xl="İlk Hane", yl="Sıklık %")
        ax.legend(facecolor=CARD, labelcolor=TXT, fontsize=8)
        plt.tight_layout()
        R.plot(fig, "benford", "Benford yasası — ilk hane frekans analizi")
    dev = [r["Feature"] for r in benf_rows if r["MAD"]>.015]
    if dev:
        R.decision(f"Benford sapması: {dev}. Sentetik veri karışımı olabilir.", "warn")
    else:
        R.decision("Veri Benford yasasına uyuyor. Doğal dağılım onaylandı.", "ok")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# ADIM 20 — String Anomali & Near-Constant Feature Tespiti
# ════════════════════════════════════════════════════════════════════════════════
R.section("🔤 ADIM 20 — String Anomali & Near-Constant Feature")
R.rule("Büyük-küçük harf varyantı veya boşluk anomalisi → normalize et.")

# String anomaliler
str_rows = []
for col in CAT_COLS:
    vals = train[col].dropna().astype(str)
    case_var = len(set(vals.str.lower())) != len(set(vals))
    spaces   = int((vals != vals.str.strip()).sum())
    vc       = vals.value_counts(normalize=True)
    rare     = int((vc < .005).sum())
    if case_var or spaces > 0 or rare > 0:
        str_rows.append({"Feature":col,"Case Varyant":case_var,
                          "Boşluk Sorunu":spaces,"Nadir (<0.5%)":rare})
if str_rows:
    R.table(pd.DataFrame(str_rows))
    R.decision("String anomalileri düzelt: .str.strip().str.lower() uygula.", "warn")
else:
    R.log("String anomalisi yok.", "ok")

# Near-constant
nc_rows = []
for col in FEATS:
    top = train[col].value_counts(normalize=True).iloc[0]*100
    if top > 90:
        nc_rows.append({"Feature":col,"Top%":f"{top:.1f}","Top Değer":str(train[col].value_counts().index[0]),
                        "Şiddet":("❌ Near-Const" if top>99 else "⚠️ Yüksek")})
if nc_rows:
    R.table(pd.DataFrame(nc_rows))
    drop_nc = [r["Feature"] for r in nc_rows if r["Şiddet"].startswith("❌")]
    if drop_nc:
        R.decision(f"Near-constant feature'lar: {drop_nc} → drop et.", "warn")
        DECISIONS["drop_features"].extend(drop_nc)
else:
    R.log("Near-constant feature yok.", "ok")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# ADIM 21 — Temporal Pattern & Stratifikasyon Kalitesi
# ════════════════════════════════════════════════════════════════════════════════
R.section("📅 ADIM 21 — Temporal Pattern & Stratifikasyon Kalitesi")
R.rule("Rolling target std > 0.01 → temporal pattern var, TimeSeriesSplit kullan.")

train_r = train.reset_index()
id_corr = train_r.index.to_series().corr(train_r[TARGET]) if IDC in train_r.columns else 0.0
rolling_std = float(train_r[TARGET].rolling(5000, min_periods=1000).mean().std())
DECISIONS["temporal_pattern"] = rolling_std > .01

R.log(f"ID-Target korelasyonu: {id_corr:.4f}", "info")
R.log(f"Rolling target std (window=5000): {rolling_std:.5f}", "info")
if rolling_std > .01:
    R.decision("Temporal pattern var! TimeSeriesSplit kullan.", "crit")
else:
    R.decision("Temporal pattern yok. StratifiedKFold güvenle kullanılabilir.", "ok")

# Stratifikasyon kalitesi
fold_stats = []
for fold, (tr_i, va_i) in enumerate(skf5.split(X_enc, Y)):
    fold_stats.append({"Fold":fold+1, "Train N":len(tr_i), "Val N":len(va_i),
                       "Train Target%":round(Y.iloc[tr_i].mean()*100,3),
                       "Val Target%":round(Y.iloc[va_i].mean()*100,3)})
df_folds = pd.DataFrame(fold_stats)
R.table(df_folds)
rate_std = float(np.std(df_folds["Val Target%"].values))
if rate_std > 1:
    R.decision(f"Zayıf stratifikasyon (std={rate_std:.3f}%). Farklı seed dene.", "warn")
else:
    R.decision(f"İyi stratifikasyon (std={rate_std:.3f}%).", "ok")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# ADIM 22 — Kategorik Detay: Churn Rate per Category
# ════════════════════════════════════════════════════════════════════════════════
R.section("🏷️ ADIM 22 — Kategorik Feature Detay Analizi")

top_cat_feats = [r["Feature"] for r in cv_rows if r["Cramér's V"] > .1][:6]
if top_cat_feats:
    nc = min(3, len(top_cat_feats))
    nr = (len(top_cat_feats)+nc-1)//nc
    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4*nr), facecolor=DARK)
    axf = np.array(axes).flatten() if len(top_cat_feats)>1 else [axes]
    for i, col in enumerate(top_cat_feats):
        ax = axf[i]
        ct = train.groupby(col)[TARGET].mean().sort_values(ascending=True) * 100
        ax.barh(ct.index.astype(str), ct.values,
                color=[RED if v>ct.mean() else GREEN for v in ct.values],
                edgecolor=GRID, alpha=.85)
        ax.axvline(ct.mean(), color=GOLD, lw=1.5, linestyle="--")
        dark_ax(ax, col, xl="Churn %")
    for j in range(len(top_cat_feats), len(axf)): axf[j].set_visible(False)
    plt.tight_layout()
    R.plot(fig, "cat_churn_detail", "Kategorik feature başına churn oranı (Cramér V > 0.1)")
R.end()

# ════════════════════════════════════════════════════════════════════════════════
# ÖZET — EDA Karar Raporu
# ════════════════════════════════════════════════════════════════════════════════
R.section("🏁 EDA ÖZET — Kararlar & Dosya 2'ye Aktarılan Bilgiler")

print("\n" + "="*70)
print("EDA KARAR ÖZETİ")
print("="*70)
for key, val in DECISIONS.items():
    if val and val != 0 and val is not False and val != [] and val != {}:
        print(f"  {key}: {val}")

# Özet tablo
summary_items = [
    ("Hedef pozitif oran",    f"{Y.mean():.3f}",                  "warn"),
    ("Scale pos weight",      f"{DECISIONS['scale_pos_weight']:.2f}", ""),
    ("Leak feature",          str(len(DECISIONS['leakage_features'])), "crit" if DECISIONS['leakage_features'] else "ok"),
    ("MNAR feature",          str(len(DECISIONS['mnar_features'])),    "warn" if DECISIONS['mnar_features'] else "ok"),
    ("PSI kritik",            str(len(DECISIONS['high_psi_features'])), "crit" if DECISIONS['high_psi_features'] else "ok"),
    ("Yüksek IV feature",     str(len(DECISIONS['high_iv_features'])), "ok"),
    ("Label noise (%)",       f"{DECISIONS['conflict_rate']:.2f}",  "warn" if DECISIONS['conflict_rate']>5 else "ok"),
    ("Adversarial AUC",       f"{adv_auc:.4f}",                    "crit" if adv_auc>.70 else "ok"),
    ("Cohort feature gerekli",str(DECISIONS['cohort_feature']),     "warn" if DECISIONS['cohort_feature'] else "ok"),
    ("Temporal pattern",      str(DECISIONS['temporal_pattern']),   "crit" if DECISIONS['temporal_pattern'] else "ok"),
]
R.metrics(summary_items[:5])
R.metrics(summary_items[5:])

all_decisions = [d for d in R._decisions]
crit_count = sum(1 for d in all_decisions if d["status"]=="crit")
warn_count = sum(1 for d in all_decisions if d["status"]=="warn")
ok_count   = sum(1 for d in all_decisions if d["status"]=="ok")

R.log(f"Toplam karar: {len(all_decisions)} | Kritik: {crit_count} | Uyarı: {warn_count} | Tamam: {ok_count}", "info")

if crit_count > 0:
    R.decision(f"{crit_count} KRİTİK sorun var. Dosya 2'ye geçmeden önce eda_decisions.json'u incele.", "crit")
elif warn_count > 3:
    R.decision(f"{warn_count} uyarı var. Dosya 2'deki aksiyon noktalarına dikkat et.", "warn")
else:
    R.decision("EDA temiz. Dosya 2'ye geçmeye hazır.", "ok")
R.end()

# ── Çıktıları kaydet ─────────────────────────────────────────────────────────────
save_decisions()
R.save("/kaggle/working/eda_report.html")
print("\n✅ Tüm EDA tamamlandı.")
print("   📄 /kaggle/working/eda_report.html")
print("   📋 /kaggle/working/eda_decisions.json")
print("   📊 /kaggle/working/eda_plots/  (tüm grafik PNG'leri)")
