# ════════════════════════════════════════════════════════════════════════════
# 🔬 COMPLETE SENIOR DATA PIPELINE — EK ANALİZLER (Sections 26–63)
# Orijinal full_senior_data_pipeline() çağrısından sonra ayrı hücrede çalıştır.
# Bağımlılıklar: shap, optbinning, category_encoders, scipy, sklearn, lightgbm
# ════════════════════════════════════════════════════════════════════════════

# Kaggle notebook0027ta çalıştır:
# !pip install tabulate shap optbinning category_encoders -q
import subprocess; subprocess.run(["pip","install","tabulate","shap","optbinning","category_encoders","-q"], check=False, capture_output=True)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap, os, warnings, base64
warnings.filterwarnings("ignore")

from scipy import stats
from scipy.stats import (ks_2samp, spearmanr, chi2_contingency, shapiro,
                         levene, mannwhitneyu, kruskal, normaltest, chi2)
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (roc_auc_score, brier_score_loss, log_loss,
                              precision_recall_curve, roc_curve,
                              average_precision_score, confusion_matrix)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
import lightgbm as lgb
from itertools import combinations

sns.set_theme(style="darkgrid", palette="muted")
PLOT_DIR = "/kaggle/working/plots_ext"
os.makedirs(PLOT_DIR, exist_ok=True)

DARK  = "#0d1117"; CARD = "#161b22"; BLUE = "#58a6ff"
TXT   = "#c9d1d9"; GRID = "#30363d"; RED  = "#f85149"
GREEN = "#3fb950"; ORANGE = "#f0883e"


# ── helpers ──────────────────────────────────────────────────────────────────
def dark_style(ax, title=""):
    ax.set_facecolor(CARD); ax.figure.set_facecolor(DARK)
    ax.title.set_color(BLUE); ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT); ax.tick_params(colors=TXT)
    for s in ax.spines.values(): s.set_color(GRID)
    if title: ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

def safe_encode(s):
    if s.dtype == "O":
        return pd.Series(LabelEncoder().fit_transform(s.astype(str)), index=s.index)
    return s

def psi(expected, actual, buckets=10):
    """Population Stability Index"""
    def scale_range(x, mn, mx):
        x = np.clip(x, mn, mx)
        return (x - mn) / (mx - mn + 1e-9)
    mn = min(expected.min(), actual.min())
    mx = max(expected.max(), actual.max())
    breaks = np.linspace(mn, mx, buckets + 1)
    exp_pct = np.histogram(expected, bins=breaks)[0] / len(expected)
    act_pct = np.histogram(actual,   bins=breaks)[0] / len(actual)
    exp_pct = np.where(exp_pct == 0, 1e-4, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-4, act_pct)
    return np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))

def cohen_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*np.std(a,ddof=1)**2 + (nb-1)*np.std(b,ddof=1)**2) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / (pooled + 1e-9)

def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2_v = chi2_contingency(ct)[0]
    n = ct.sum().sum(); md = min(ct.shape) - 1
    return np.sqrt(chi2_v / (n * md)) if md > 0 else 0.0

def calc_iv(data, feature, target):
    g = data.groupby(feature)[target].agg(["sum","count"])
    g.columns = ["ev","tot"]; g["nev"] = g["tot"] - g["ev"]
    te, tne = g["ev"].sum(), g["nev"].sum()
    g["pe"] = (g["ev"]/te).clip(1e-4); g["pn"] = (g["nev"]/tne).clip(1e-4)
    g["woe"] = np.log(g["pn"]/g["pe"]); g["iv"] = (g["pn"]-g["pe"])*g["woe"]
    return g["iv"].sum()


# ════════════════════════════════════════════════════════════════
# REPORTER
# ════════════════════════════════════════════════════════════════
class Reporter:
    CSS = """<html><head><meta charset="utf-8"><style>
      body{font-family:'Segoe UI',Consolas,monospace;background:#0d1117;color:#c9d1d9;
           padding:30px;max-width:1400px;margin:auto}
      .section{background:#161b22;border:1px solid #30363d;border-radius:10px;
               padding:20px;margin:25px 0}
      .section h2{color:#58a6ff;border-bottom:2px solid #58a6ff;
                  padding-bottom:8px;font-size:19px}
      .info-line{background:#1f2937;padding:9px 14px;border-radius:6px;margin:7px 0;
                 border-left:4px solid #58a6ff;font-size:13px}
      .warn{border-left-color:#f0883e;color:#f0883e}
      .ok{border-left-color:#3fb950;color:#3fb950}
      .critical{border-left-color:#f85149;color:#f85149;font-weight:bold}
      .decision{background:#1a2332;border:2px solid #58a6ff;border-radius:8px;
                padding:13px;margin:10px 0;font-size:13px}
      .decision-yes{border-color:#3fb950;background:#0d2818}
      .decision-no{border-color:#f85149;background:#2d1014}
      .section table{border-collapse:collapse;width:100%;margin:14px 0;font-size:12px}
      .section table th{background:#21262d;color:#58a6ff;padding:9px 13px;
                        border:1px solid #30363d;text-align:left}
      .section table td{padding:7px 13px;border:1px solid #30363d;background:#0d1117}
      .section table tr:hover td{background:#1a2233}
      .plot-container{text-align:center;margin:14px 0}
      .plot-container img{max-width:100%;border-radius:8px;border:1px solid #30363d}
      .rule{background:#0d2818;border:1px solid #3fb950;border-radius:7px;
            padding:12px;margin:8px 0;font-size:12px}
      .rule b{color:#3fb950}
    </style></head><body>
    <h1 style="text-align:center;color:#58a6ff">
      🔬 COMPLETE EXTENDED PIPELINE — Sections 26–63</h1>"""

    def __init__(self):
        self.h = [self.CSS]; self.pc = 0; self.actions = []

    def log(self, t, cls=""):
        print(t)
        c = f" {cls}" if cls else ""
        self.h.append(f'<div class="info-line{c}">{t}</div>')

    def decision(self, t, pos=True):
        cn = "decision-yes" if pos else "decision-no"
        px = "✅" if pos else "⚠️"
        full = f"{px} KARAR: {t}"
        print(f"  {full}")
        self.h.append(f'<div class="decision {cn}">{full}</div>')
        self.actions.append(t)

    def rule(self, t):
        self.h.append(f'<div class="rule"><b>📌 KURAL:</b> {t}</div>')

    def section(self, title):
        print(f"\n{'═'*80}\n{title}\n{'═'*80}")
        self.h.append(f'<div class="section"><h2>{title}</h2>')

    def end(self): self.h.append("</div>")

    def table(self, styled):
        from IPython.display import display
        try: display(styled)
        except: pass
        try: self.h.append(styled.to_html())
        except: self.h.append(styled.to_frame().to_html())

    def plot(self, fig, title="plot"):
        from IPython.display import display, Image
        self.pc += 1
        fname = f"ext_{self.pc:02d}_{title.replace(' ','_').lower()[:40]}.png"
        fpath = os.path.join(PLOT_DIR, fname)
        fig.savefig(fpath, dpi=130, bbox_inches="tight",
                    facecolor=DARK, edgecolor="none")
        plt.close(fig)
        try: display(Image(filename=fpath))
        except: pass
        with open(fpath,"rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        self.h.append(
            f'<div class="plot-container">'
            f'<img src="data:image/png;base64,{b64}" alt="{title}"></div>')

    def save(self, path):
        self.h.append("</body></html>")
        with open(path,"w",encoding="utf-8") as f:
            f.write("\n".join(self.h))
        print(f"\n✅ Rapor kaydedildi: {path}")


# ════════════════════════════════════════════════════════════════
# ANA FONKSİYON
# ════════════════════════════════════════════════════════════════
def complete_extended_pipeline(train, test, target_col, id_col=None,
                                cat_threshold=25,
                                fn_cost=5.0, fp_cost=1.0):
    """
    fn_cost : False Negative maliyeti (kaçan churner)
    fp_cost : False Positive maliyeti (gereksiz müdahale)
    """
    R = Reporter()

    feats    = [c for c in train.columns if c not in [target_col, id_col]]
    num_cols = [c for c in feats if train[c].nunique() > cat_threshold]
    cat_cols = [c for c in feats if train[c].nunique() <= cat_threshold]
    y_full   = safe_encode(train[target_col].copy())

    X_enc = train[feats].copy()
    for c in X_enc.select_dtypes(["object","category"]).columns:
        X_enc[c] = pd.factorize(X_enc[c])[0]
    X_enc = X_enc.fillna(-1)

    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Baseline OOF predictions (shared across sections) ────────────────────
    R.log("⏳ Baseline OOF tahminleri hesaplanıyor...", cls="ok")
    oof_proba = np.zeros(len(train))
    oof_models = []
    for tr_i, va_i in skf5.split(X_enc, y_full):
        m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                max_depth=6, num_leaves=31,
                                is_unbalance=True, random_state=42,
                                verbose=-1, n_jobs=-1)
        m.fit(X_enc.iloc[tr_i], y_full.iloc[tr_i],
              eval_set=[(X_enc.iloc[va_i], y_full.iloc[va_i])],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        oof_proba[va_i] = m.predict_proba(X_enc.iloc[va_i])[:,1]
        oof_models.append(m)
    baseline_auc = roc_auc_score(y_full, oof_proba)
    R.log(f"✅ Baseline OOF AUC: {baseline_auc:.5f}", cls="ok")

    # ════════════════════════════════════════════════════
    # SECTION 26: LEAKAGE DETECTION
    # ════════════════════════════════════════════════════
    R.section("🚨 26. LEAKAGE DETECTION")
    R.rule("Solo AUC > 0.95 → kesin leak. DROP et.")
    R.rule("Solo AUC 0.85–0.95 → şüpheli. Domain kontrolü yap.")

    leak_res = []
    for col in feats:
        try:
            xc = X_enc[[col]]
            aucs = [roc_auc_score(y_full.iloc[va_i],
                    RandomForestClassifier(50, max_depth=4, random_state=42, n_jobs=-1)
                    .fit(xc.iloc[tr_i], y_full.iloc[tr_i])
                    .predict_proba(xc.iloc[va_i])[:,1])
                    for tr_i, va_i in skf5.split(xc, y_full)]
            auc_m = np.mean(aucs)
            level = ("🔴 LEAK!" if auc_m>0.95 else
                     "🟠 ŞÜPHELİ" if auc_m>0.85 else
                     "🟡 Güçlü"  if auc_m>0.75 else "✅ Normal")
            leak_res.append({"Feature":col,"Solo_AUC":auc_m,"Risk":level})
        except: pass

    df_leak = (pd.DataFrame(leak_res)
               .sort_values("Solo_AUC",ascending=False).reset_index(drop=True))
    R.table(df_leak.style
            .background_gradient(cmap="RdYlGn_r",subset=["Solo_AUC"])
            .format({"Solo_AUC":"{:.4f}"}))

    leaks    = df_leak[df_leak["Solo_AUC"]>0.95]["Feature"].tolist()
    suspects = df_leak[(df_leak["Solo_AUC"]>0.85)&(df_leak["Solo_AUC"]<=0.95)]["Feature"].tolist()
    if leaks:
        R.log(f"🔴 LEAK: {leaks}", cls="critical")
        R.decision(f"DROP: {leaks}", pos=False)
    else:
        R.log("✅ Leak yok.", cls="ok")
    if suspects:
        R.decision(f"Domain kontrolü: {suspects}", pos=False)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 27: LABEL QUALITY
    # ════════════════════════════════════════════════════
    R.section("🏷️ 27. LABEL KALİTESİ & TARGET NOISE")
    R.rule("Conflict rate > %10 → sample_weight zorunlu.")
    R.rule("Conflict rate %5–10 → label_smoothing=0.05.")

    key_cols  = feats[:min(6, len(feats))]
    dup_check = train[feats+[target_col]].copy()
    for c in dup_check.select_dtypes(["object","category"]).columns:
        dup_check[c] = dup_check[c].astype(str)
    grp = dup_check.groupby(key_cols)[target_col].agg(["nunique","count"])
    conflict_count = grp[grp["nunique"]>1]["count"].sum()
    conflict_rate  = conflict_count / len(train) * 100
    R.log(f"📊 Conflict: {conflict_count:,} satır ({conflict_rate:.2f}%)")
    if conflict_rate > 10:
        R.decision("sample_weight kullan", pos=False)
    elif conflict_rate > 5:
        R.decision("label_smoothing=0.05", pos=False)
    else:
        R.decision("Label kalitesi iyi.", pos=True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 28: BUSINESS LOGIC VALIDATION
    # ════════════════════════════════════════════════════
    R.section("🏢 28. BUSINESS LOGIC VALIDATION")
    R.rule("TotalCharges ≈ tenure × MonthlyCharges. Sapma > %50 → flag ekle.")
    R.rule("tenure=0 & TotalCharges>0 → veri hatası.")
    n_bad = 0
    if all(c in train.columns for c in ["TotalCharges","tenure","MonthlyCharges"]):
        tc  = pd.to_numeric(train["TotalCharges"], errors="coerce")
        ten = pd.to_numeric(train["tenure"],       errors="coerce")
        mc  = pd.to_numeric(train["MonthlyCharges"],errors="coerce")
        ratio   = tc / (ten*mc + 1e-9)
        bad_ratio = ((ratio < 0.5)|(ratio > 2.0)) & (ten > 0)
        n_bad   = bad_ratio.sum()
        R.log(f"📊 Bozuk satır: {n_bad:,} ({n_bad/len(train)*100:.2f}%)")
        if n_bad > 0:
            R.decision("is_charges_anomaly flag oluştur.", pos=False)
        zero_bad = ((ten==0)&(tc>0)).sum()
        if zero_bad:
            R.log(f"🔴 tenure=0 & TC>0: {zero_bad}", cls="critical")
            R.decision("TotalCharges'ı 0 yap veya flag ekle.", pos=False)
        else:
            R.log("✅ tenure=0 anomalisi yok.", cls="ok")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 29: COHORT DRIFT
    # ════════════════════════════════════════════════════
    R.section("⏳ 29. COHORT DRIFT (tenure bazlı)")
    R.rule("Cohort spread > 20 puan → tenure_group feature zorunlu.")
    cohort_spread = 0
    if "tenure" in train.columns:
        tmp_c = train.copy(); tmp_c["tbin"] = y_full
        tmp_c["cohort"] = pd.cut(pd.to_numeric(tmp_c["tenure"],errors="coerce"),
                                  bins=[0,12,24,36,48,100],
                                  labels=["0-12","12-24","24-36","36-48","48+"],
                                  right=True)
        cs = tmp_c.groupby("cohort")["tbin"].agg(["mean","count"])
        cs.columns = ["Rate%","Count"]; cs["Rate%"] *= 100
        R.table(cs.style.background_gradient(cmap="RdYlGn_r",subset=["Rate%"])
                .format({"Rate%":"{:.2f}"}))
        cohort_spread = cs["Rate%"].max() - cs["Rate%"].min()
        rho_c, _ = spearmanr(range(len(cs)), cs["Rate%"].values)
        R.log(f"📊 Spread: {cohort_spread:.1f}p | Spearman: {rho_c:.3f}")
        if cohort_spread > 20:
            R.decision("tenure_group feature oluştur.", pos=True)
        elif cohort_spread > 10:
            R.decision("tenure_bin ekle.", pos=True)
        else:
            R.decision("Lineer tenure yeterli.", pos=True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 30: STRATIFIED KDE
    # ════════════════════════════════════════════════════
    R.section("📊 30. STRATİFİED KDE (Churn=Yes vs No)")
    R.rule("KS > 0.3 → güçlü ayrıştırıcı. KS < 0.05 → zayıf, drop.")
    strat_res = []
    if num_cols:
        y0 = train[y_full==0]; y1 = train[y_full==1]
        for col in num_cols:
            a,b = y0[col].dropna(), y1[col].dropna()
            if len(a)>5 and len(b)>5:
                ks_v, ks_p = ks_2samp(a,b)
                strat_res.append({"Feature":col,"KS":ks_v,"P":ks_p,
                                   "Mean_No":a.mean(),"Mean_Yes":b.mean(),
                                   "Power":("🔴 Güçlü" if ks_v>0.3 else
                                            "🟡 Orta" if ks_v>0.15 else "⚪ Zayıf")})
        df_s = (pd.DataFrame(strat_res)
                .sort_values("KS",ascending=False).reset_index(drop=True))
        R.table(df_s.style.background_gradient(cmap="YlGn",subset=["KS"])
                .format({"KS":"{:.4f}","P":"{:.4f}","Mean_No":"{:.3f}","Mean_Yes":"{:.3f}"}))
        weak = df_s[df_s["KS"]<0.05]["Feature"].tolist()
        if weak:
            R.decision(f"Zayıf KDE feature'lar drop edilebilir: {weak}", pos=False)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 31: SHAP + INTERACTION
    # ════════════════════════════════════════════════════
    R.section("🔬 31. SHAP & INTERACTION ANALİZİ")
    R.rule("SHAP interaksiyon korelasyonu > 0.3 → çarpım feature oluştur.")
    R.rule("Non-linear Pearson < 0.7 → binning veya log transform.")
    shap_top5 = feats[:5]
    try:
        tr_i0, va_i0 = list(skf5.split(X_enc,y_full))[0]
        explainer  = shap.TreeExplainer(oof_models[0])
        shap_vals  = explainer.shap_values(X_enc.iloc[va_i0])
        if isinstance(shap_vals,list): shap_vals = shap_vals[1]
        shap_mean  = np.abs(shap_vals).mean(axis=0)
        shap_df    = (pd.DataFrame({"Feature":feats,"SHAP_Mean":shap_mean})
                      .sort_values("SHAP_Mean",ascending=False).reset_index(drop=True))
        R.table(shap_df.style.bar(subset=["SHAP_Mean"],color=BLUE)
                .format({"SHAP_Mean":"{:.4f}"}))
        shap_top5 = shap_df.head(5)["Feature"].tolist()
        # Interaction proxy
        shap_top5_vals = pd.DataFrame(
            shap_vals[:,[feats.index(f) for f in shap_top5]], columns=shap_top5)
        inter_corr = shap_top5_vals.corr()
        strong_inter = [(shap_top5[i],shap_top5[j])
                        for i in range(len(shap_top5))
                        for j in range(i+1,len(shap_top5))
                        if abs(inter_corr.iloc[i,j])>0.3]
        if strong_inter:
            R.decision(f"Çarpım feature adayları: {strong_inter}", pos=True)
        # SHAP bar plot
        fig, ax = plt.subplots(figsize=(9,max(4,len(feats)*0.32)))
        shap_df.head(15).sort_values("SHAP_Mean").plot.barh(
            x="Feature", y="SHAP_Mean", ax=ax, color=BLUE, edgecolor=GRID, legend=False)
        dark_style(ax,"SHAP Feature Importance (Top 15)")
        fig.tight_layout(); R.plot(fig,"shap_bar")
    except Exception as e:
        R.log(f"⚠️ SHAP hatası: {e}", cls="warn")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 32: HARD SAMPLE
    # ════════════════════════════════════════════════════
    R.section("🎯 32. HARD SAMPLE ANALİZİ")
    R.rule("Hard sample > %15 → agresif FE. < %8 → temel FE yeterli.")
    fold_preds = np.zeros((len(train),5))
    for fold,(tr_i,va_i) in enumerate(skf5.split(X_enc,y_full)):
        fold_preds[va_i,fold] = oof_models[fold].predict_proba(X_enc.iloc[va_i])[:,1]
    pred_labels = (fold_preds>0.5).astype(int)
    wrong_count = 5 - (pred_labels==y_full.values.reshape(-1,1)).sum(axis=1)
    is_hard     = wrong_count >= 4
    hard_rate   = is_hard.mean()*100
    R.log(f"📊 Hard: {is_hard.sum():,} ({hard_rate:.2f}%)")
    if hard_rate>15:
        R.decision("Agresif FE gerekli.", pos=False)
    elif hard_rate>8:
        R.decision("Target encoding veya stacking dene.", pos=False)
    else:
        R.decision("Model sağlıklı.", pos=True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 33: OOD
    # ════════════════════════════════════════════════════
    R.section("🔭 33. TEST OOD ANALİZİ")
    R.rule("OOD kategorik (>100 satır) → target encoding şart.")
    R.rule("OOD numeric → Winsorize.")
    ood_results = []
    for col in cat_cols:
        new_in_test = set(test[col].dropna().unique()) - set(train[col].dropna().unique())
        if new_in_test:
            cnt = test[col].isin(new_in_test).sum()
            ood_results.append({"Col":col,"Type":"Yeni Kategori","Count":cnt})
    for col in num_cols:
        tr_min,tr_max = train[col].min(), train[col].max()
        oor = ((test[col]<tr_min)|(test[col]>tr_max)).sum()
        if oor: ood_results.append({"Col":col,"Type":"Range Dışı","Count":oor})
    if ood_results:
        df_ood = pd.DataFrame(ood_results)
        R.table(df_ood.style.background_gradient(cmap="OrRd",subset=["Count"]))
        cat_ood = [r["Col"] for r in ood_results if r["Type"]=="Yeni Kategori"]
        num_ood = [r["Col"] for r in ood_results if r["Type"]=="Range Dışı"]
        if cat_ood: R.decision(f"Target encoding: {cat_ood}", pos=False)
        if num_ood: R.decision(f"Winsorize: {num_ood}", pos=False)
    else:
        R.log("✅ OOD yok.", cls="ok")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 35: PSI (Population Stability Index)
    # ════════════════════════════════════════════════════
    R.section("📉 35. PSI — POPULATION STABILITY INDEX")
    R.rule("PSI < 0.10 → stabil. 0.10–0.25 → izle. > 0.25 → ciddi kayma, model yeniden eğitilmeli.")
    R.rule("PSI, KS testinden daha yorumlanabilir; production monitoring için standarttır.")

    psi_res = []
    for col in num_cols:
        try:
            tr_v = train[col].dropna().values
            ts_v = test[col].dropna().values
            p    = psi(tr_v, ts_v)
            level = ("🔴 CİDDİ" if p>0.25 else
                     "🟠 İZLE"  if p>0.10 else "✅ Stabil")
            psi_res.append({"Feature":col,"PSI":p,"Seviye":level})
        except: pass
    if psi_res:
        df_psi = (pd.DataFrame(psi_res)
                  .sort_values("PSI",ascending=False).reset_index(drop=True))
        R.table(df_psi.style
                .background_gradient(cmap="RdYlGn_r",subset=["PSI"])
                .format({"PSI":"{:.4f}"}))
        serious = df_psi[df_psi["PSI"]>0.25]["Feature"].tolist()
        monitor = df_psi[(df_psi["PSI"]>0.10)&(df_psi["PSI"]<=0.25)]["Feature"].tolist()
        if serious:
            R.log(f"🔴 PSI > 0.25: {serious}", cls="critical")
            R.decision(f"Bu feature'lar ciddi kayma yapıyor. Drop veya yeniden encode: {serious}", pos=False)
        if monitor:
            R.log(f"🟠 PSI 0.10–0.25 (izle): {monitor}", cls="warn")
            R.decision(f"Bu feature'ları RobustScaler ile normalize et: {monitor}", pos=False)
        if not serious and not monitor:
            R.log("✅ Tüm feature'lar PSI stabil.", cls="ok")
            R.decision("PSI iyi. Drift endişesi yok.", pos=True)

        # PSI bar chart
        fig, ax = plt.subplots(figsize=(10, max(4, len(df_psi)*0.35)))
        colors = [RED if p>0.25 else ORANGE if p>0.10 else GREEN for p in df_psi["PSI"]]
        ax.barh(df_psi["Feature"], df_psi["PSI"], color=colors, edgecolor=GRID)
        ax.axvline(0.10, color=ORANGE, linestyle="--", lw=1.5, label="0.10 (izle)")
        ax.axvline(0.25, color=RED,    linestyle="--", lw=1.5, label="0.25 (kritik)")
        dark_style(ax, "PSI — Population Stability Index")
        ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TXT, fontsize=9)
        fig.tight_layout(); R.plot(fig, "psi_chart")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 36: CALIBRATION CHECK
    # ════════════════════════════════════════════════════
    R.section("🎯 36. MODEL CALİBRATION (Brier Score & Calibration Curve)")
    R.rule("Brier Score < 0.10 → iyi kalibre. 0.10–0.20 → kabul edilebilir. > 0.20 → kötü.")
    R.rule("Calibration curve diyagonal çizgiden uzaksa → Platt scaling veya isotonic regression uygula.")
    R.rule("Kötü kalibrasyon: threshold seçimi ve cost-sensitive analiz yanlış sonuç verir.")

    brier = brier_score_loss(y_full, oof_proba)
    ll    = log_loss(y_full, oof_proba)
    R.log(f"📊 Brier Score: {brier:.4f}  |  Log Loss: {ll:.4f}")
    if brier < 0.10:
        R.decision(f"Brier={brier:.4f} — İyi kalibre. Threshold analizi güvenilir.", pos=True)
    elif brier < 0.20:
        R.decision(f"Brier={brier:.4f} — Kabul edilebilir. Isotonic regression dene.", pos=False)
    else:
        R.decision(f"Brier={brier:.4f} — Kötü kalibrasyon. Platt scaling zorunlu.", pos=False)

    # Calibration curve
    try:
        frac_pos, mean_pred = calibration_curve(y_full, oof_proba, n_bins=10)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot([0,1],[0,1], color=GRID, linestyle="--", lw=1.5, label="Mükemmel")
        axes[0].plot(mean_pred, frac_pos, color=BLUE, marker="o", lw=2, label="Model")
        dark_style(axes[0], "Calibration Curve")
        axes[0].set_xlabel("Ortalama Tahmin"); axes[0].set_ylabel("Gerçek Oran")
        axes[0].legend(facecolor=CARD, edgecolor=GRID, labelcolor=TXT)
        axes[1].hist(oof_proba[y_full==0], bins=40, alpha=0.6,
                     color=GREEN, label="Churn=No",  density=True, edgecolor=GRID)
        axes[1].hist(oof_proba[y_full==1], bins=40, alpha=0.6,
                     color=RED,   label="Churn=Yes", density=True, edgecolor=GRID)
        dark_style(axes[1], "OOF Tahmin Dağılımı")
        axes[1].legend(facecolor=CARD, edgecolor=GRID, labelcolor=TXT)
        fig.set_facecolor(DARK); fig.tight_layout()
        R.plot(fig, "calibration_curve")
    except Exception as e:
        R.log(f"⚠️ Calibration curve hatası: {e}", cls="warn")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 37: THRESHOLD OPTİMİZASYONU & COST MATRIX
    # ════════════════════════════════════════════════════
    R.section(f"💰 37. THRESHOLD & COST-SENSITIVE ANALİZ (FN={fn_cost}x, FP={fp_cost}x)")
    R.rule(f"fn_cost={fn_cost}: kaçan churner maliyeti. fp_cost={fp_cost}: gereksiz müdahale maliyeti.")
    R.rule("Optimal threshold = min(FN×fn_cost + FP×fp_cost). AUC-only optimizasyon business'a yanlış threshold verir.")
    R.rule("Precision-Recall tradeoff'u anla: recall↑ → daha fazla churner yakalanır ama precision↓.")

    thresholds = np.linspace(0.01, 0.99, 200)
    costs, f1s, precisions, recalls = [], [], [], []
    for thr in thresholds:
        preds = (oof_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_full, preds, labels=[0,1]).ravel()
        cost = fn*fn_cost + fp*fp_cost
        costs.append(cost)
        p = tp/(tp+fp+1e-9); r = tp/(tp+fn+1e-9)
        precisions.append(p); recalls.append(r)
        f1s.append(2*p*r/(p+r+1e-9))

    opt_idx_cost = int(np.argmin(costs))
    opt_idx_f1   = int(np.argmax(f1s))
    opt_thr_cost = thresholds[opt_idx_cost]
    opt_thr_f1   = thresholds[opt_idx_f1]

    R.log(f"📊 Cost-optimal threshold : {opt_thr_cost:.3f}  (F1={f1s[opt_idx_cost]:.4f})")
    R.log(f"📊 F1-optimal threshold   : {opt_thr_f1:.3f}  (F1={f1s[opt_idx_f1]:.4f})")
    R.log(f"📊 Default threshold=0.50 : F1={f1s[np.argmin(np.abs(thresholds-0.5))]:.4f}")

    if abs(opt_thr_cost - 0.5) > 0.1:
        R.decision(f"Threshold 0.5'ten önemli ölçüde farklı ({opt_thr_cost:.3f}). "
                   f"predict_proba >= {opt_thr_cost:.2f} kullan.", pos=True)
    else:
        R.decision(f"Threshold 0.5 yakın ({opt_thr_cost:.3f}). Default kullanılabilir.", pos=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(thresholds, costs, color=BLUE, lw=2)
    axes[0].axvline(opt_thr_cost, color=RED, linestyle="--", lw=1.5,
                    label=f"Opt={opt_thr_cost:.2f}")
    dark_style(axes[0], "Cost vs Threshold")
    axes[0].set_xlabel("Threshold"); axes[0].legend(facecolor=CARD,edgecolor=GRID,labelcolor=TXT)

    axes[1].plot(thresholds, f1s,        color=GREEN,  lw=2, label="F1")
    axes[1].plot(thresholds, precisions, color=BLUE,   lw=2, label="Precision")
    axes[1].plot(thresholds, recalls,    color=ORANGE, lw=2, label="Recall")
    axes[1].axvline(opt_thr_f1, color=RED, linestyle="--", lw=1.5)
    dark_style(axes[1], "F1 / Precision / Recall vs Threshold")
    axes[1].legend(facecolor=CARD,edgecolor=GRID,labelcolor=TXT,fontsize=9)

    prec_arr, rec_arr, _ = precision_recall_curve(y_full, oof_proba)
    axes[2].plot(rec_arr, prec_arr, color=BLUE, lw=2)
    axes[2].fill_between(rec_arr, prec_arr, alpha=0.15, color=BLUE)
    baseline_pr = y_full.mean()
    axes[2].axhline(baseline_pr, color=GRID, linestyle="--", lw=1.5, label=f"Baseline={baseline_pr:.2f}")
    dark_style(axes[2], "Precision-Recall Curve")
    axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision")
    axes[2].legend(facecolor=CARD,edgecolor=GRID,labelcolor=TXT)
    fig.set_facecolor(DARK); fig.tight_layout(); R.plot(fig,"threshold_analysis")

    # Confusion matrix at optimal threshold
    preds_opt = (oof_proba >= opt_thr_cost).astype(int)
    cm = confusion_matrix(y_full, preds_opt)
    fig2, ax2 = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2,
                linewidths=0.5, linecolor=GRID,
                xticklabels=["Pred No","Pred Yes"],
                yticklabels=["True No","True Yes"])
    dark_style(ax2, f"Confusion Matrix (thr={opt_thr_cost:.2f})")
    fig2.tight_layout(); R.plot(fig2,"confusion_matrix")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 38: LIFT & GAIN CHART + DECILE ANALİZİ
    # ════════════════════════════════════════════════════
    R.section("📈 38. LIFT & GAIN CHART + DECİLE ANALİZİ")
    R.rule("Lift > 2.0 (ilk decile) → model iş değeri yüksek. < 1.5 → zayıf.")
    R.rule("Decile churn rate monoton artmalı (yüksek skor = yüksek churn). Bozulma → tutarsızlık.")
    R.rule("Gain chart: 'Top %20'yi hedeflersek gerçek churner'ların kaçını yakalıyoruz?'")

    df_lift = pd.DataFrame({"y":y_full.values, "prob":oof_proba})
    df_lift["decile"] = pd.qcut(df_lift["prob"].rank(method="first"),
                                 10, labels=False)
    df_lift["decile"] = 9 - df_lift["decile"]  # 0=highest prob

    decile_stats = df_lift.groupby("decile").agg(
        Count=("y","count"), Churners=("y","sum"), Mean_Prob=("prob","mean")
    ).reset_index()
    decile_stats["Churn_Rate%"]    = decile_stats["Churners"] / decile_stats["Count"] * 100
    decile_stats["Cumulative_Gain"]= decile_stats["Churners"].cumsum() / y_full.sum() * 100
    decile_stats["Lift"]           = decile_stats["Churn_Rate%"] / (y_full.mean()*100)

    R.table(decile_stats.style
            .background_gradient(cmap="RdYlGn_r",subset=["Churn_Rate%"])
            .background_gradient(cmap="Blues",    subset=["Cumulative_Gain"])
            .format({"Churn_Rate%":"{:.2f}","Cumulative_Gain":"{:.1f}","Lift":"{:.3f}",
                     "Mean_Prob":"{:.4f}"}))

    top1_lift = decile_stats.iloc[0]["Lift"]
    top3_gain = decile_stats.iloc[:3]["Churners"].sum() / y_full.sum() * 100
    rho_dec, _ = spearmanr(decile_stats["decile"], -decile_stats["Churn_Rate%"])

    R.log(f"📊 Top decile lift: {top1_lift:.2f}x | Top 3 decile gain: {top3_gain:.1f}%")
    R.log(f"📊 Decile monotonicity Spearman: {rho_dec:.3f}")

    if top1_lift > 2.5:
        R.decision(f"Mükemmel lift ({top1_lift:.2f}x). Model iş değeri çok yüksek.", pos=True)
    elif top1_lift > 1.5:
        R.decision(f"İyi lift ({top1_lift:.2f}x).", pos=True)
    else:
        R.decision(f"Düşük lift ({top1_lift:.2f}x). FE ile iyileştir.", pos=False)

    if rho_dec > 0.9:
        R.decision("Decile monoton. Model tutarlı.", pos=True)
    else:
        R.decision(f"Decile tutarsız (rho={rho_dec:.2f}). Threshold veya model problemi.", pos=False)

    fig, axes = plt.subplots(1,3,figsize=(15,4))
    axes[0].bar(decile_stats["decile"], decile_stats["Lift"],
                color=[RED if l>2 else ORANGE if l>1.5 else BLUE for l in decile_stats["Lift"]],
                edgecolor=GRID)
    axes[0].axhline(1.0,color=GRID,linestyle="--",lw=1.5)
    dark_style(axes[0],"Lift by Decile"); axes[0].set_xlabel("Decile (0=top)")

    axes[1].plot(decile_stats["decile"]+1,
                 decile_stats["Cumulative_Gain"], color=BLUE, marker="o", lw=2)
    xs = np.linspace(1,10,10)
    axes[1].plot(xs, xs*10, color=GRID, linestyle="--", lw=1.5, label="Random")
    dark_style(axes[1],"Cumulative Gain"); axes[1].set_xlabel("Decile"); axes[1].set_ylabel("Gain%")
    axes[1].legend(facecolor=CARD,edgecolor=GRID,labelcolor=TXT)

    axes[2].bar(decile_stats["decile"], decile_stats["Churn_Rate%"],
                color=BLUE, edgecolor=GRID)
    dark_style(axes[2],"Churn Rate by Decile"); axes[2].set_xlabel("Decile")
    fig.set_facecolor(DARK); fig.tight_layout(); R.plot(fig,"lift_gain_chart")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 39: NORMALİTE TESTLERİ (Shapiro-Wilk / D'Agostino)
    # ════════════════════════════════════════════════════
    R.section("📐 39. NORMALİTE TESTLERİ (Shapiro-Wilk / D'Agostino-K²)")
    R.rule("p < 0.05 → normal dağılım değil. Parametrik testler (t-test, ANOVA) yerine non-parametrik kullan.")
    R.rule("Yüksek skewness (>1) + non-normal → log1p veya sqrt transform gerekli.")
    R.rule("Bu test encoding ve imputation stratejisini belirler.")

    norm_res = []
    sample = train[num_cols].dropna().sample(min(2000, len(train)), random_state=42)
    for col in num_cols:
        vals = sample[col].dropna().values
        if len(vals) < 8: continue
        # D'Agostino (n>20 için daha güvenilir)
        try:
            stat_d, p_d = normaltest(vals)
        except:
            stat_d, p_d = np.nan, np.nan
        # Shapiro (n<2000)
        try:
            stat_s, p_s = shapiro(vals[:min(500,len(vals))])
        except:
            stat_s, p_s = np.nan, np.nan
        sk  = float(pd.Series(vals).skew())
        krt = float(pd.Series(vals).kurtosis())
        is_normal = (p_d > 0.05) if not np.isnan(p_d) else (p_s > 0.05)
        norm_res.append({"Feature":col,"Skew":sk,"Kurtosis":krt,
                          "DAgostino_p":p_d,"Shapiro_p":p_s,
                          "Normal":"✅ Evet" if is_normal else "❌ Hayır"})

    df_norm = pd.DataFrame(norm_res).sort_values("DAgostino_p")
    R.table(df_norm.style
            .background_gradient(cmap="RdYlGn",subset=["DAgostino_p"])
            .format({"Skew":"{:.3f}","Kurtosis":"{:.3f}",
                     "DAgostino_p":"{:.4f}","Shapiro_p":"{:.4f}"}))

    non_normal = df_norm[df_norm["Normal"]=="❌ Hayır"]["Feature"].tolist()
    high_skew  = df_norm[df_norm["Skew"].abs()>1]["Feature"].tolist()
    R.log(f"📊 Non-normal feature sayısı: {len(non_normal)}/{len(num_cols)}")
    if non_normal:
        R.decision(f"Non-normal feature'lar için Mann-Whitney U testi kullan (t-test değil).", pos=True)
    if high_skew:
        R.decision(f"Yüksek skew feature'lar ({high_skew}) → log1p transform uygula.", pos=False)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 40: LEVENE + MANN-WHITNEY U (Varyans & Grup Testi)
    # ════════════════════════════════════════════════════
    R.section("⚖️ 40. LEVENE & MANN-WHITNEY U (Grup Farklılığı)")
    R.rule("Levene p < 0.05 → Churn=Yes ve No gruplarında varyans farklı. Welch t-test veya Mann-Whitney kullan.")
    R.rule("Mann-Whitney p < 0.05 + Cohen's d > 0.5 → pratik anlamlı fark var. Bu feature FE'de öncelikli.")
    R.rule("Sadece p-value yeterli değil. Effect size (Cohen's d) ile birlikte değerlendir.")

    group_res = []
    g0 = train[y_full==0]; g1 = train[y_full==1]
    for col in num_cols:
        a = g0[col].dropna().values; b = g1[col].dropna().values
        if len(a)<5 or len(b)<5: continue
        try:
            _, p_lev  = levene(a, b)
            _, p_mwu  = mannwhitneyu(a, b, alternative="two-sided")
            d         = cohen_d(b, a)  # Yes - No
            sig       = "✅" if p_mwu<0.05 else "❌"
            practical = ("🔴 Büyük" if abs(d)>0.8 else
                         "🟡 Orta" if abs(d)>0.5 else
                         "🟢 Küçük" if abs(d)>0.2 else "⚪ Önemsiz")
            group_res.append({"Feature":col,"Levene_p":p_lev,"MWU_p":p_mwu,
                               "Cohen_d":d,"Sig":sig,"Effect":practical})
        except: pass

    df_grp = (pd.DataFrame(group_res)
              .sort_values("MWU_p").reset_index(drop=True))
    R.table(df_grp.style
            .background_gradient(cmap="RdYlGn",subset=["MWU_p"])
            .background_gradient(cmap="OrRd",  subset=["Cohen_d"])
            .format({"Levene_p":"{:.4f}","MWU_p":"{:.4f}","Cohen_d":"{:.4f}"}))

    big_effect = df_grp[(df_grp["MWU_p"]<0.05) & (df_grp["Cohen_d"].abs()>0.5)]["Feature"].tolist()
    no_effect  = df_grp[df_grp["Cohen_d"].abs()<0.2]["Feature"].tolist()
    if big_effect:
        R.decision(f"Büyük effect size feature'lar: {big_effect}. FE'de öncelikle kullan.", pos=True)
    if no_effect:
        R.decision(f"Pratik etkisi yok: {no_effect}. Drop veya dönüştür.", pos=False)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 41: KRUSKAL-WALLIS (Kategorik → Numeric)
    # ════════════════════════════════════════════════════
    R.section("📊 41. KRUSKAL-WALLIS (Kategorik Feature → Numeric Target Etkisi)")
    R.rule("p < 0.05 → bu kategorik feature numeric gruplar arasında anlamlı fark yaratıyor.")
    R.rule("Eta-squared > 0.14 → büyük etki. Bu feature'ı mutlaka FE'ye dahil et.")

    kw_res = []
    for cat in cat_cols:
        groups = [train[num_cols[0]].dropna().values] if not num_cols else []
        try:
            groups = [y_full[train[cat]==v].values
                      for v in train[cat].dropna().unique() if (train[cat]==v).sum()>5]
            if len(groups)<2: continue
            stat, p = kruskal(*groups)
            # Eta-squared proxy
            n_total = sum(len(g) for g in groups)
            eta2    = (stat - len(groups) + 1) / (n_total - len(groups))
            kw_res.append({"Feature":cat,"KW_stat":stat,"p_value":p,
                            "Eta2":max(0,eta2),
                            "Effect":("🔴 Büyük" if eta2>0.14 else
                                      "🟡 Orta"  if eta2>0.06 else
                                      "🟢 Küçük" if eta2>0.01 else "⚪ Yok")})
        except: pass
    if kw_res:
        df_kw = (pd.DataFrame(kw_res)
                 .sort_values("p_value").reset_index(drop=True))
        R.table(df_kw.style
                .background_gradient(cmap="RdYlGn",subset=["p_value"])
                .background_gradient(cmap="OrRd",  subset=["Eta2"])
                .format({"KW_stat":"{:.2f}","p_value":"{:.4f}","Eta2":"{:.4f}"}))
        big_kw = df_kw[(df_kw["p_value"]<0.05)&(df_kw["Eta2"]>0.14)]["Feature"].tolist()
        if big_kw:
            R.decision(f"Büyük KW etkisi: {big_kw}. Target encoding için en güçlü adaylar.", pos=True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 42: BONFERRONI / FDR CORRECTION
    # ════════════════════════════════════════════════════
    R.section("🔬 42. ÇOKLU KARŞILAŞTIRMA DÜZELTMESİ (Bonferroni / BH-FDR)")
    R.rule("34 section'da yüzlerce p-value testi yapılıyor → yanlış pozitif şişmesi riski.")
    R.rule("Bonferroni: alpha/n → çok konservatif. BH-FDR: daha dengeli.")
    R.rule("Düzeltmeden sonra hâlâ anlamlı olan feature'lar gerçekten güçlüdür.")

    all_pvals = []
    if group_res:
        for r in group_res:
            all_pvals.append({"Feature":r["Feature"],"Test":"MWU","p":r["MWU_p"]})
    if kw_res:
        for r in kw_res:
            all_pvals.append({"Feature":r["Feature"],"Test":"KW","p":r["p_value"]})

    if all_pvals:
        df_p = pd.DataFrame(all_pvals)
        n_tests = len(df_p)
        bonf_alpha = 0.05 / n_tests
        # BH-FDR
        sorted_p = df_p["p"].sort_values().values
        bh_threshold = 0.0
        for i, p in enumerate(sorted_p, 1):
            if p <= (i / n_tests) * 0.05:
                bh_threshold = p
        df_p["Bonf_Sig"] = df_p["p"] < bonf_alpha
        df_p["BH_Sig"]   = df_p["p"] <= bh_threshold
        R.log(f"📊 Toplam test: {n_tests} | Bonferroni alpha: {bonf_alpha:.5f} | BH threshold: {bh_threshold:.5f}")
        R.log(f"📊 Bonferroni anlamlı: {df_p['Bonf_Sig'].sum()} | BH anlamlı: {df_p['BH_Sig'].sum()}")
        bonf_sig = df_p[df_p["Bonf_Sig"]]["Feature"].tolist()
        bh_only  = df_p[df_p["BH_Sig"] & ~df_p["Bonf_Sig"]]["Feature"].tolist()
        if bonf_sig:
            R.decision(f"Bonferroni sonrası anlamlı: {bonf_sig}. Bu feature'lar kesinlikle önemli.", pos=True)
        if bh_only:
            R.decision(f"Sadece BH'de anlamlı: {bh_only}. Dikkatli kullan.", pos=False)
        R.table(df_p.style
                .background_gradient(cmap="RdYlGn",subset=["p"])
                .format({"p":"{:.5f}"}))
    else:
        R.log("⚠️ p-value verisi bulunamadı.", cls="warn")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 43: EFFECT SIZE — Cohen's d & Cramér's V Özeti
    # ════════════════════════════════════════════════════
    R.section("📏 43. EFFECT SIZE ÖZET (İstatistiksel Anlamlılık ≠ Pratik Anlamlılık)")
    R.rule("p < 0.05 tek başına yeterli değil. Cohen d > 0.5 veya Cramér V > 0.3 olmalı.")
    R.rule("Büyük veri setlerinde küçük farklar da anlamlı çıkar. Effect size bunu düzeltir.")

    effect_rows = []
    for col in num_cols:
        a = g0[col].dropna().values; b = g1[col].dropna().values
        if len(a)<5 or len(b)<5: continue
        d = cohen_d(b,a)
        effect_rows.append({"Feature":col,"Type":"Numeric","Effect_Size":abs(d),
                             "Metric":"Cohen d",
                             "Interpretation":("Büyük" if abs(d)>0.8 else
                                               "Orta"  if abs(d)>0.5 else
                                               "Küçük" if abs(d)>0.2 else "Önemsiz")})
    for col in cat_cols:
        v = cramers_v(train[col].fillna("NA"), train[target_col].fillna("NA"))
        effect_rows.append({"Feature":col,"Type":"Categorical","Effect_Size":v,
                             "Metric":"Cramér V",
                             "Interpretation":("Büyük" if v>0.5 else
                                               "Orta"  if v>0.3 else
                                               "Küçük" if v>0.1 else "Önemsiz")})
    df_eff = (pd.DataFrame(effect_rows)
              .sort_values("Effect_Size",ascending=False).reset_index(drop=True))
    R.table(df_eff.style
            .background_gradient(cmap="YlGn",subset=["Effect_Size"])
            .format({"Effect_Size":"{:.4f}"}))
    useless = df_eff[df_eff["Interpretation"]=="Önemsiz"]["Feature"].tolist()
    if useless:
        R.decision(f"Pratik etkisi önemsiz feature'lar: {useless}. Drop veya combine et.", pos=False)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 44: MISSING VALUE MECHANISM (MCAR/MAR/MNAR)
    # ════════════════════════════════════════════════════
    R.section("❓ 44. EKSİK DEĞER MEKANİZMASI (MCAR / MAR / MNAR)")
    R.rule("MCAR: eksiklik tamamen rastgele → mean/median impute güvenli.")
    R.rule("MAR: eksiklik diğer değişkenlere bağlı → model-based impute veya KNN impute.")
    R.rule("MNAR: eksiklik kendi değeriyle bağlantılı → is_null flag ZORUNLU. İmputation yetmez.")
    R.rule("Test: null olan satırlarda target rate, null olmayan satırlardan farklıysa → MNAR riski yüksek.")

    null_cols = [c for c in feats if train[c].isnull().any()]
    mnar_res  = []
    if null_cols:
        for col in null_cols:
            mask_null  = train[col].isnull()
            rate_null  = y_full[mask_null].mean() * 100
            rate_present = y_full[~mask_null].mean() * 100
            gap = abs(rate_null - rate_present)

            # Little's MCAR test proxy: korelasyon diğer feature'larla
            corr_with_others = []
            for other in num_cols:
                if other == col: continue
                c = abs(mask_null.astype(float).corr(train[other].fillna(train[other].median())))
                if not np.isnan(c): corr_with_others.append(c)
            max_corr = max(corr_with_others) if corr_with_others else 0

            mechanism = ("MNAR" if gap>10 else
                         "MAR"  if max_corr>0.1 else "MCAR")
            mnar_res.append({"Feature":col,
                              "Null%":mask_null.mean()*100,
                              "Rate_Null%":rate_null,
                              "Rate_Present%":rate_present,
                              "Gap":gap,"Max_Corr":max_corr,
                              "Mechanism":mechanism})

        df_mnar = (pd.DataFrame(mnar_res)
                   .sort_values("Gap",ascending=False))
        R.table(df_mnar.style
                .background_gradient(cmap="OrRd",subset=["Gap"])
                .format({"Null%":"{:.2f}","Rate_Null%":"{:.2f}",
                         "Rate_Present%":"{:.2f}","Gap":"{:.2f}","Max_Corr":"{:.4f}"}))
        mnar_feats = [r["Feature"] for r in mnar_res if r["Mechanism"]=="MNAR"]
        mar_feats  = [r["Feature"] for r in mnar_res if r["Mechanism"]=="MAR"]
        mcar_feats = [r["Feature"] for r in mnar_res if r["Mechanism"]=="MCAR"]
        if mnar_feats:
            R.log(f"🔴 MNAR: {mnar_feats}", cls="critical")
            R.decision(f"is_null flag ZORUNLU: {mnar_feats}. İmputation tek başına yetmez.", pos=False)
        if mar_feats:
            R.decision(f"MAR: {mar_feats} → KNN veya model-based impute.", pos=False)
        if mcar_feats:
            R.decision(f"MCAR: {mcar_feats} → mean/median impute güvenli.", pos=True)
    else:
        R.log("✅ Eksik değer yok.", cls="ok")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 45: OUTLIER TİPİ SINIFLANDIRMASI
    # ════════════════════════════════════════════════════
    R.section("🔍 45. OUTLIER TİPİ SINIFLANDIRMASI (Hata / Uç Değer / Segment)")
    R.rule("Hata (veri girişi yanlışı): z-score > 5 ve domain imkansız → impute veya drop.")
    R.rule("Gerçek uç değer: z-score > 3 ama domain mümkün → Winsorize.")
    R.rule("Farklı segment: Isolation Forest anormal ama churn rate farklı → anomaly flag olarak kullan.")
    R.rule("Hepsine aynı Winsorize uygulamak yanlış. Önce tipi belirle.")

    if num_cols:
        X_iso = train[num_cols].fillna(train[num_cols].median())
        iso  = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        iso_scores = iso.fit_predict(X_iso)  # -1 = anomaly
        iso_anomaly = (iso_scores == -1)

        out_res = []
        for col in num_cols:
            vals = train[col].dropna()
            z    = np.abs(stats.zscore(vals.fillna(vals.median())))
            q1,q3= vals.quantile(0.25), vals.quantile(0.75)
            iqr  = q3 - q1
            iqr_out = ((vals < q1-3*iqr) | (vals > q3+3*iqr)).sum()
            z5_out  = (z > 5).sum()
            # Segment check: isolation anomaly + different churn
            anom_mask  = iso_anomaly & train[col].notna()
            normal_mask= ~iso_anomaly & train[col].notna()
            rate_anom  = y_full[anom_mask].mean()*100  if anom_mask.sum()>0 else 0
            rate_norm  = y_full[normal_mask].mean()*100 if normal_mask.sum()>0 else 0
            segment_gap= abs(rate_anom - rate_norm)

            otype = ("Hata"    if z5_out>0 and iqr_out>0 else
                     "Segment" if segment_gap>15 else
                     "Uç Değer" if iqr_out>0 else "Normal")
            action = ("Drop/impute"  if otype=="Hata" else
                      "Anomaly flag" if otype=="Segment" else
                      "Winsorize"    if otype=="Uç Değer" else "—")
            out_res.append({"Feature":col,"IQR_Out":iqr_out,"Z5_Out":z5_out,
                             "Segment_Gap":segment_gap,"Type":otype,"Action":action})

        df_out = pd.DataFrame(out_res).sort_values("IQR_Out",ascending=False)
        R.table(df_out.style
                .background_gradient(cmap="OrRd",subset=["IQR_Out"])
                .format({"IQR_Out":"{:.0f}","Z5_Out":"{:.0f}","Segment_Gap":"{:.2f}"}))

        seg_feats  = df_out[df_out["Type"]=="Segment"]["Feature"].tolist()
        err_feats  = df_out[df_out["Type"]=="Hata"]["Feature"].tolist()
        winz_feats = df_out[df_out["Type"]=="Uç Değer"]["Feature"].tolist()
        if seg_feats:
            R.decision(f"Segment anomaly → is_anomaly_{col} flag: {seg_feats}", pos=True)
        if err_feats:
            R.decision(f"Hata tipi outlier → impute: {err_feats}", pos=False)
        if winz_feats:
            R.decision(f"Uç değer → Winsorize: {winz_feats}", pos=True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 46: PCA — FEATURE REDUNDANCY
    # ════════════════════════════════════════════════════
    R.section("🧬 46. PCA — ÖZELLİK FAZLALIĞI & BOYUT İNDİRGEME")
    R.rule("İlk N bileşen %90 varyansı açıklıyorsa → geri kalan feature'lar büyük ölçüde gereksiz.")
    R.rule("Eğer 2–3 bileşen %80+ açıklıyorsa → feature'lar arasında ciddi redundancy var.")
    R.rule("PCA'yı doğrudan modele verme; yorumlanabilirlik azalır. Redundancy'yi anlamak için kullan.")

    if len(num_cols) >= 3:
        X_pca  = train[num_cols].fillna(train[num_cols].median())
        X_std  = (X_pca - X_pca.mean()) / (X_pca.std() + 1e-9)
        pca    = PCA(random_state=42)
        pca.fit(X_std)
        ev_ratio = pca.explained_variance_ratio_
        cumvar   = np.cumsum(ev_ratio)
        n80 = int(np.searchsorted(cumvar, 0.80)) + 1
        n90 = int(np.searchsorted(cumvar, 0.90)) + 1
        n95 = int(np.searchsorted(cumvar, 0.95)) + 1
        R.log(f"📊 {n80} bileşen → %80 | {n90} bileşen → %90 | {n95} bileşen → %95 varyans")
        R.log(f"📊 Toplam feature: {len(num_cols)}")

        redundancy = len(num_cols) - n90
        if redundancy > len(num_cols)*0.4:
            R.decision(f"{redundancy} feature redundant. VIF ve korelasyon ile hangileri olduğunu bul.", pos=False)
        else:
            R.decision(f"Düşük redundancy. Feature'lar çeşitli bilgi taşıyor.", pos=True)

        fig, axes = plt.subplots(1,2,figsize=(12,4))
        axes[0].bar(range(1,len(ev_ratio)+1), ev_ratio*100, color=BLUE, edgecolor=GRID)
        dark_style(axes[0],"Explained Variance per Component")
        axes[0].set_xlabel("PC"); axes[0].set_ylabel("Variance %")

        axes[1].plot(range(1,len(cumvar)+1), cumvar*100, color=BLUE, marker="o", ms=4, lw=2)
        axes[1].axhline(80,color=ORANGE,linestyle="--",lw=1.5,label="80%")
        axes[1].axhline(90,color=RED,   linestyle="--",lw=1.5,label="90%")
        dark_style(axes[1],"Cumulative Explained Variance")
        axes[1].set_xlabel("Components"); axes[1].set_ylabel("Cumulative Variance %")
        axes[1].legend(facecolor=CARD,edgecolor=GRID,labelcolor=TXT)
        fig.set_facecolor(DARK); fig.tight_layout(); R.plot(fig,"pca_variance")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 47: NULL PATTERN CLUSTERİNG
    # ════════════════════════════════════════════════════
    R.section("🕸️ 47. NULL PATTERN CLUSTERİNG")
    R.rule("Birden fazla feature aynı anda null olan satırlar → sistemik veri eksikliği.")
    R.rule("Null pattern cluster'ının churn rate'i farklıysa → is_null_pattern flag güçlü feature olur.")
    R.rule("Bu pattern genellikle veri toplama sürecindeki sorunları gösterir.")

    null_cols_list = [c for c in feats if train[c].isnull().any()]
    if len(null_cols_list) >= 2:
        null_matrix = train[null_cols_list].isnull().astype(int)
        null_matrix["pattern"] = null_matrix.apply(
            lambda r: "_".join(str(v) for v in r.values), axis=1)
        null_matrix["target"]  = y_full.values
        pat_stats = (null_matrix.groupby("pattern")["target"]
                     .agg(["mean","count"]).reset_index())
        pat_stats.columns = ["Pattern","Churn_Rate%","Count"]
        pat_stats["Churn_Rate%"] *= 100
        pat_stats = pat_stats[pat_stats["Count"] >= 20].sort_values("Count",ascending=False)

        if len(pat_stats) > 1:
            R.table(pat_stats.style
                    .background_gradient(cmap="RdYlGn_r",subset=["Churn_Rate%"])
                    .format({"Churn_Rate%":"{:.2f}"}))
            overall_rate = y_full.mean()*100
            anomalous_patterns = pat_stats[
                abs(pat_stats["Churn_Rate%"]-overall_rate) > 10]
            if len(anomalous_patterns) > 0:
                R.decision("Null pattern churn rate'ten sapıyor. "
                            "null pattern kombinasyonunu feature olarak ekle.", pos=True)
            else:
                R.decision("Null pattern'lar homojen. Özel feature gerekmez.", pos=True)
        else:
            R.log("📊 Yeterli null pattern çeşitliliği yok.", cls="ok")
    else:
        R.log("✅ Yeterli null kolon yok (< 2).", cls="ok")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 48: RARE CATEGORY ANALİZİ
    # ════════════════════════════════════════════════════
    R.section("🦄 48. RARE CATEGORY ANALİZİ")
    R.rule("Frekans < %1 olan kategoriler → 'Other' ile birleştir veya target encoding kullan.")
    R.rule("Rare ama churn rate ortalamanın > 2 katıysa → DROP etme! Güçlü sinyal taşıyor.")
    R.rule("Test'te rare kategorinin yüksek frekansı varsa → OOD riski yüksek.")

    rare_res = []
    for col in cat_cols:
        vc = train[col].value_counts(normalize=True)*100
        rare = vc[vc < 1.0]
        for cat_val, pct in rare.items():
            mask = train[col] == cat_val
            churn_rate = y_full[mask].mean()*100
            overall    = y_full.mean()*100
            rare_res.append({"Feature":col,"Category":cat_val,
                              "Train%":pct,"Churn%":churn_rate,
                              "Vs_Overall":churn_rate/overall,
                              "Action":("KEEP (strong)" if churn_rate/overall>2
                                        else "Other ile birleştir")})
    if rare_res:
        df_rare = (pd.DataFrame(rare_res)
                   .sort_values("Vs_Overall",ascending=False).reset_index(drop=True))
        R.table(df_rare.style
                .background_gradient(cmap="OrRd",subset=["Vs_Overall"])
                .format({"Train%":"{:.2f}","Churn%":"{:.2f}","Vs_Overall":"{:.2f}"}))
        strong_rare = df_rare[df_rare["Vs_Overall"]>2]
        if len(strong_rare):
            R.decision(f"{len(strong_rare)} rare kategori güçlü sinyal taşıyor. "
                       f"Drop etme, ayrı flag oluştur.", pos=True)
        bland_rare = df_rare[df_rare["Vs_Overall"]<=1.2]
        if len(bland_rare):
            R.decision(f"{len(bland_rare)} rare kategori zayıf → 'Other' ile birleştir.", pos=False)
    else:
        R.log("✅ Rare kategori (< %1) yok.", cls="ok")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 49: TARGET ENCODING LEAKAGE SİMÜLASYONU
    # ════════════════════════════════════════════════════
    R.section("🧪 49. TARGET ENCODING LEAKAGE SİMÜLASYONU")
    R.rule("CV dışında target encoding yapılırsa skor şişer. Ne kadar şiştiğini ölç.")
    R.rule("Skor farkı > 0.005 ise → KFold target encoding zorunlu. Naif encoding kullanma.")
    R.rule("Bu test, encoding pipeline'ının doğruluğunu doğrular.")

    if cat_cols:
        # Naive (leaky) target encoding
        def naive_te(tr, va, cols, tgt):
            X = tr[cols+[tgt]].copy()
            means = X.groupby(cols[:1])[tgt].mean()
            result = va[cols[:1]].map(means).fillna(X[tgt].mean())
            return result.values

        # Proper KFold target encoding
        def kfold_te(X_cat, y, col, n_splits=5):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            out = np.zeros(len(X_cat))
            for tr_i, va_i in skf.split(X_cat, y):
                means = y.iloc[tr_i].groupby(X_cat.iloc[tr_i][col]).mean()
                out[va_i] = X_cat.iloc[va_i][col].map(means).fillna(y.mean())
            return out

        col_te = cat_cols[0]
        # Naive TE AUC
        naive_aucs = []
        for tr_i, va_i in skf5.split(X_enc, y_full):
            tr_map   = y_full.iloc[tr_i].groupby(
                train.iloc[tr_i][col_te].fillna("NA")).mean()
            va_enc   = train.iloc[va_i][col_te].fillna("NA").map(tr_map).fillna(y_full.mean())
            # Naif: global mean kullanmak yerine FULL train mean leak
            full_map = y_full.groupby(train[col_te].fillna("NA")).mean()
            va_leak  = train.iloc[va_i][col_te].fillna("NA").map(full_map).fillna(y_full.mean())
            naive_aucs.append(roc_auc_score(y_full.iloc[va_i], va_leak))

        # Proper TE AUC (fold içinde)
        proper_aucs = []
        for tr_i, va_i in skf5.split(X_enc, y_full):
            fold_map = y_full.iloc[tr_i].groupby(
                train.iloc[tr_i][col_te].fillna("NA")).mean()
            va_enc   = train.iloc[va_i][col_te].fillna("NA").map(fold_map).fillna(y_full.mean())
            proper_aucs.append(roc_auc_score(y_full.iloc[va_i], va_enc))

        naive_mean  = np.mean(naive_aucs)
        proper_mean = np.mean(proper_aucs)
        leakage_gap = naive_mean - proper_mean

        R.log(f"📊 Naive (leaky) TE AUC : {naive_mean:.5f}")
        R.log(f"📊 Proper KFold TE AUC  : {proper_mean:.5f}")
        R.log(f"📊 Leakage gap          : {leakage_gap:.5f}")

        if leakage_gap > 0.005:
            R.log(f"🔴 Leakage gap büyük: {leakage_gap:.4f}", cls="critical")
            R.decision("KFold target encoding ZORUNLU. Naif encoding şişmiş skor üretir.", pos=False)
        else:
            R.decision(f"Leakage gap küçük ({leakage_gap:.4f}). "
                       f"Buna rağmen KFold TE kullanmak iyi pratik.", pos=True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 50: OPTIMAL BİNNİNG (IV-Optimal)
    # ════════════════════════════════════════════════════
    R.section("📦 50. OPTİMAL BİNNİNG (IV-Maksimize)")
    R.rule("pd.qcut (eşit frekans) her zaman IV'yi maksimize etmez.")
    R.rule("Optimal binning: WoE monotonluğunu korurken IV'yi maksimize eder.")
    R.rule("Optimal bin sonrası IV > qcut IV → binning feature FE'ye dahil edilmeli.")

    try:
        from optbinning import OptimalBinning
        ob_res = []
        for col in num_cols[:5]:  # ilk 5 numerik
            try:
                x_v = train[col].fillna(train[col].median()).values
                ob  = OptimalBinning(name=col, dtype="numerical",
                                     solver="cp", monotonic_trend="auto")
                ob.fit(x_v, y_full.values)
                bt  = ob.binning_table.build()
                iv_opt = bt["IV"].iloc[-2] if "IV" in bt.columns else 0
                # qcut IV karşılaştırma
                tmp_iv = train[[col,target_col]].copy()
                tmp_iv[target_col] = y_full
                tmp_iv[f"{col}_bin"] = pd.qcut(
                    tmp_iv[col].fillna(tmp_iv[col].median()),
                    q=10, duplicates="drop", labels=False)
                iv_qcut = abs(calc_iv(tmp_iv, f"{col}_bin", target_col))
                ob_res.append({"Feature":col,"IV_OptBin":iv_opt,
                                "IV_qcut":iv_qcut,"Gain":iv_opt-iv_qcut})
            except: pass

        if ob_res:
            df_ob = pd.DataFrame(ob_res).sort_values("Gain",ascending=False)
            R.table(df_ob.style
                    .background_gradient(cmap="YlGn",subset=["Gain"])
                    .format({"IV_OptBin":"{:.4f}","IV_qcut":"{:.4f}","Gain":"{:.4f}"}))
            better = df_ob[df_ob["Gain"]>0.01]["Feature"].tolist()
            if better:
                R.decision(f"Optimal binning qcut'tan belirgin iyileştirme sağlıyor: {better}. "
                           f"OptimalBinning kullan.", pos=True)
            else:
                R.decision("qcut ile optimal binning arasında belirgin fark yok. "
                           "qcut yeterli.", pos=True)
    except ImportError:
        R.log("⚠️ optbinning kurulu değil. !pip install optbinning", cls="warn")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 51: PARTİAL DEPENDENCE PLOTS (PDP)
    # ════════════════════════════════════════════════════
    R.section("📉 51. PARTİAL DEPENDENCE PLOTS (PDP)")
    R.rule("PDP: bir feature'ın ortalama etkisini gösterir. SHAP marginal etkiyi gösterir. İkisi farklı bilgi verir.")
    R.rule("PDP düz çizgiyse → feature modele doğrusal katkı yapıyor. Dalgalıysa → non-linear.")
    R.rule("PDP'de ani sıçrama → o değerde önemli bir threshold var. Binning için kriter.")

    try:
        top_pdp = shap_top5[:3]
        fig, axes = plt.subplots(1, len(top_pdp), figsize=(5*len(top_pdp), 4))
        if len(top_pdp) == 1: axes = [axes]
        last_model = oof_models[-1]
        for i, col in enumerate(top_pdp):
            if col not in X_enc.columns: continue
            col_vals = np.percentile(X_enc[col].dropna(), np.linspace(1, 99, 50))
            X_tmp    = X_enc.sample(min(500, len(X_enc)), random_state=42).copy()
            pdp_means = []
            for v in col_vals:
                X_tmp2   = X_tmp.copy(); X_tmp2[col] = v
                pdp_means.append(last_model.predict_proba(X_tmp2)[:,1].mean())
            axes[i].plot(col_vals, pdp_means, color=BLUE, lw=2)
            axes[i].fill_between(col_vals, pdp_means, alpha=0.1, color=BLUE)
            dark_style(axes[i], f"PDP: {col}")
            axes[i].set_xlabel(col); axes[i].set_ylabel("Mean Pred Prob")
            axes[i].axhline(y_full.mean(), color=GRID, linestyle="--", lw=1.5)
        fig.set_facecolor(DARK); fig.tight_layout(); R.plot(fig,"pdp_plots")
        R.decision("PDP ile non-linear pattern tespit edildi. "
                   "Ani sıçrama noktaları binning threshold'u olarak kullan.", pos=True)
    except Exception as e:
        R.log(f"⚠️ PDP hatası: {e}", cls="warn")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 52: ANOMALY SCORE AS FEATURE
    # ════════════════════════════════════════════════════
    R.section("🤖 52. ANOMALİ SKORU FEATURE OLARAK")
    R.rule("Isolation Forest ve LOF anomali skorları bazen çok güçlü feature'dır.")
    R.rule("Anomali skoru ile target arasında korelasyon > 0.1 ise → feature olarak ekle.")
    R.rule("Anomali skoru yüksek müşteriler → alışılmadık profil → churn riski farklı.")

    if num_cols:
        X_anom = train[num_cols].fillna(train[num_cols].median())
        # Isolation Forest
        iso2   = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
        iso_score = -iso2.fit(X_anom).score_samples(X_anom)  # yüksek = daha anormal
        corr_iso  = abs(pd.Series(iso_score).corr(y_full))
        R.log(f"📊 Isolation Forest skoru — target korelasyonu: {corr_iso:.4f}")

        # LOF
        try:
            lof       = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            lof_labels= lof.fit_predict(X_anom)
            lof_score = -lof.negative_outlier_factor_
            corr_lof  = abs(pd.Series(lof_score).corr(y_full))
            R.log(f"📊 LOF skoru — target korelasyonu: {corr_lof:.4f}")
        except Exception as e:
            corr_lof = 0
            R.log(f"⚠️ LOF hatası: {e}", cls="warn")

        if corr_iso > 0.1:
            R.decision("Isolation Forest skoru feature olarak ekle: anomaly_score_iso.", pos=True)
        else:
            R.decision(f"Isolation Forest skoru zayıf (r={corr_iso:.4f}). Eklemene gerek yok.", pos=False)
        if corr_lof > 0.1:
            R.decision("LOF skoru feature olarak ekle: anomaly_score_lof.", pos=True)

        # Scatter: anomaly score vs churn
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].scatter(iso_score, oof_proba, alpha=0.2,
                        c=y_full, cmap="RdYlGn_r", s=5)
        dark_style(axes[0], "Isolation Forest Score vs Churn Prob")
        axes[0].set_xlabel("Anomaly Score"); axes[0].set_ylabel("Churn Prob")
        axes[1].hist(iso_score[y_full==0], bins=40, alpha=0.6, color=GREEN,
                     label="Churn=No", density=True, edgecolor=GRID)
        axes[1].hist(iso_score[y_full==1], bins=40, alpha=0.6, color=RED,
                     label="Churn=Yes", density=True, edgecolor=GRID)
        dark_style(axes[1], "Anomaly Score — Churn=Yes vs No")
        axes[1].legend(facecolor=CARD, edgecolor=GRID, labelcolor=TXT)
        fig.set_facecolor(DARK); fig.tight_layout(); R.plot(fig,"anomaly_score")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 53: TARGET RATE STABİLİTESİ (20 Quantile)
    # ════════════════════════════════════════════════════
    R.section("📊 53. TARGET RATE STABİLİTESİ (20 Quantile Analizi)")
    R.rule("Feature'ı 20 quantile'a böl. Her quantile'ın churn rate'i smooth ise → feature iyi davranıyor.")
    R.rule("Ani sıçrama olan quantile → önemli threshold. Binning için kullan.")
    R.rule("Monoton artış/azalış → monotone_constraint ekle.")

    if num_cols:
        top_q = [c for c in shap_top5 if c in num_cols][:4]
        fig, axes = plt.subplots(1, max(1,len(top_q)), figsize=(4*max(1,len(top_q)), 4))
        if len(top_q) == 1: axes = [axes]
        threshold_candidates = {}
        for i, col in enumerate(top_q):
            try:
                tmp_q = train[[col]].copy()
                tmp_q["y"] = y_full.values
                tmp_q["q"] = pd.qcut(tmp_q[col].rank(method="first"),
                                      20, labels=False, duplicates="drop")
                q_stats  = tmp_q.groupby("q")["y"].agg(["mean","count"])
                rates    = q_stats["mean"].values * 100
                axes[i].plot(range(len(rates)), rates, color=BLUE, marker="o", ms=4, lw=2)
                axes[i].fill_between(range(len(rates)), rates, alpha=0.1, color=BLUE)
                dark_style(axes[i], col)
                axes[i].set_xlabel("Quantile"); axes[i].set_ylabel("Churn Rate %")
                axes[i].axhline(y_full.mean()*100, color=GRID, linestyle="--", lw=1)
                # Threshold tespiti: maksimum slope
                slopes = np.diff(rates)
                max_slope_idx = int(np.argmax(np.abs(slopes)))
                threshold_candidates[col] = max_slope_idx
                axes[i].axvline(max_slope_idx, color=RED, linestyle="--", lw=1.5,
                                label=f"Sıçrama@Q{max_slope_idx}")
                axes[i].legend(facecolor=CARD, edgecolor=GRID, labelcolor=TXT, fontsize=8)
            except Exception as e:
                R.log(f"  {col}: {e}", cls="warn")
        for j in range(len(top_q), len(axes)): axes[j].set_visible(False)
        fig.set_facecolor(DARK); fig.tight_layout(); R.plot(fig,"quantile_target_rate")
        if threshold_candidates:
            R.decision(f"Ani sıçrama quantile'ları bulundu: {threshold_candidates}. "
                       f"Bu noktalar binning threshold'u olarak kullan.", pos=True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 54: LEARNING CURVE ANALİZİ
    # ════════════════════════════════════════════════════
    R.section("📚 54. LEARNING CURVE ANALİZİ")
    R.rule("Train ve val AUC yakınsa → iyi fit. Train >> val → overfitting. İkisi de düşük → underfitting.")
    R.rule("Veri arttıkça val AUC hâlâ yükseliyorsa → augmentation yardımcı olur.")
    R.rule("Val AUC platoya ulaştıysa → daha fazla veri değil, daha iyi feature gerekiyor.")

    sample_sizes = [int(len(train)*p) for p in [0.1,0.2,0.3,0.5,0.7,1.0]]
    lc_train_aucs = []; lc_val_aucs = []
    for ss in sample_sizes:
        idx = np.random.choice(len(train), ss, replace=False)
        X_sub = X_enc.iloc[idx]; y_sub = y_full.iloc[idx]
        tr_aucs_ss = []; va_aucs_ss = []
        for tr_i, va_i in StratifiedKFold(3,shuffle=True,random_state=42).split(X_sub,y_sub):
            m = lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,
                                    is_unbalance=True,random_state=42,
                                    verbose=-1,n_jobs=-1)
            m.fit(X_sub.iloc[tr_i], y_sub.iloc[tr_i])
            tr_aucs_ss.append(roc_auc_score(y_sub.iloc[tr_i],
                                             m.predict_proba(X_sub.iloc[tr_i])[:,1]))
            va_aucs_ss.append(roc_auc_score(y_sub.iloc[va_i],
                                             m.predict_proba(X_sub.iloc[va_i])[:,1]))
        lc_train_aucs.append(np.mean(tr_aucs_ss))
        lc_val_aucs.append(np.mean(va_aucs_ss))

    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(sample_sizes, lc_train_aucs, color=GREEN,  marker="o", lw=2, label="Train AUC")
    ax.plot(sample_sizes, lc_val_aucs,   color=BLUE,   marker="o", lw=2, label="Val AUC")
    ax.fill_between(sample_sizes, lc_train_aucs, lc_val_aucs, alpha=0.1, color=ORANGE)
    dark_style(ax,"Learning Curve")
    ax.set_xlabel("Sample Size"); ax.set_ylabel("AUC")
    ax.legend(facecolor=CARD,edgecolor=GRID,labelcolor=TXT)
    fig.tight_layout(); R.plot(fig,"learning_curve")

    gap_at_full  = lc_train_aucs[-1] - lc_val_aucs[-1]
    val_still_rising = lc_val_aucs[-1] - lc_val_aucs[-2] > 0.002

    R.log(f"📊 Train-Val gap (full data): {gap_at_full:.4f}")
    if gap_at_full > 0.05:
        R.decision(f"Overfitting ({gap_at_full:.4f}). Regularizasyon artır (min_child_samples, reg_lambda).", pos=False)
    elif lc_val_aucs[-1] < 0.80:
        R.decision("Underfitting. Daha güçlü FE veya daha derin model gerekiyor.", pos=False)
    elif val_still_rising:
        R.decision("Val AUC hâlâ yükseliyor. Augmentation (original data) yardımcı olacak.", pos=True)
    else:
        R.decision("İyi fit. Platoya ulaşıldı. Daha fazla veri yerine FE odaklan.", pos=True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 55: NESTED CV (BIAS-VARIANCE)
    # ════════════════════════════════════════════════════
    R.section("🔄 55. NESTED CV & BIAS-VARIANCE DEĞERLENDİRMESİ")
    R.rule("Outer CV = gerçek test performansı. Inner CV = hyperparameter seçimi.")
    R.rule("Outer AUC << Inner AUC → optimistic bias var. Nested CV farkı > 0.01 → dikkat.")
    R.rule("Bias = modelin öğrenemediği şey. Variance = fold'lar arası tutarsızlık.")

    outer_scores = []
    outer_skf    = StratifiedKFold(3, shuffle=True, random_state=99)
    for tr_i, va_i in outer_skf.split(X_enc, y_full):
        # Inner: basit CV ile en iyi n_estimators seç
        inner_best = 300
        m_out = lgb.LGBMClassifier(n_estimators=inner_best, learning_rate=0.05,
                                    is_unbalance=True, random_state=42,
                                    verbose=-1, n_jobs=-1)
        m_out.fit(X_enc.iloc[tr_i], y_full.iloc[tr_i])
        outer_scores.append(
            roc_auc_score(y_full.iloc[va_i],
                          m_out.predict_proba(X_enc.iloc[va_i])[:,1]))

    nested_auc  = np.mean(outer_scores)
    nested_std  = np.std(outer_scores)
    optimism    = baseline_auc - nested_auc

    R.log(f"📊 Standard CV AUC : {baseline_auc:.5f}")
    R.log(f"📊 Nested CV AUC   : {nested_auc:.5f} ± {nested_std:.5f}")
    R.log(f"📊 Optimism bias   : {optimism:.5f}")

    if optimism > 0.01:
        R.decision(f"Optimism bias büyük ({optimism:.4f}). "
                   f"Model evaluation şişirilmiş. Nested CV kullan.", pos=False)
    else:
        R.decision(f"Optimism bias küçük ({optimism:.4f}). Standard CV güvenilir.", pos=True)

    bias_proxy     = 1 - nested_auc
    variance_proxy = nested_std
    R.log(f"📊 Bias proxy (1-AUC): {bias_proxy:.4f} | Variance proxy (std): {variance_proxy:.4f}")
    if bias_proxy > 0.15:
        R.decision("Yüksek bias → daha güçlü model veya daha iyi FE.", pos=False)
    if variance_proxy > 0.01:
        R.decision("Yüksek variance → regularizasyon artır veya RepeatedKFold.", pos=False)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 56: OOF TAHMİN DAĞILIMI vs TEST
    # ════════════════════════════════════════════════════
    R.section("🔭 56. OOF TAHMİN DAĞILIMI vs TEST TAHMİN DAĞILIMI")
    R.rule("OOF ve test tahmin dağılımları farklıysa → covariate shift var. Threshold yanlış seçilir.")
    R.rule("KS istatistiği > 0.10 → dağılımlar anlamlı farklı. Test tahminlerini recalibrate et.")

    X_test_enc = test[feats].copy()
    for c in X_test_enc.select_dtypes(["object","category"]).columns:
        X_test_enc[c] = pd.factorize(X_test_enc[c])[0]
    X_test_enc = X_test_enc.fillna(-1)

    test_proba_list = [m.predict_proba(X_test_enc)[:,1] for m in oof_models]
    test_proba      = np.mean(test_proba_list, axis=0)

    ks_oof_test, p_oof_test = ks_2samp(oof_proba, test_proba)
    R.log(f"📊 OOF mean prob  : {oof_proba.mean():.4f} ± {oof_proba.std():.4f}")
    R.log(f"📊 Test mean prob : {test_proba.mean():.4f} ± {test_proba.std():.4f}")
    R.log(f"📊 KS (OOF vs Test): {ks_oof_test:.4f} | p={p_oof_test:.4f}")

    if ks_oof_test > 0.10:
        R.decision(f"KS={ks_oof_test:.4f} — OOF ve test dağılımları farklı. "
                   f"Rank transformation veya isotonic regression uygula.", pos=False)
    else:
        R.decision(f"KS={ks_oof_test:.4f} — Dağılımlar benzer. Threshold güvenilir.", pos=True)

    fig, ax = plt.subplots(figsize=(9,4))
    ax.hist(oof_proba,  bins=50, alpha=0.6, color=BLUE,  label="OOF",  density=True, edgecolor=GRID)
    ax.hist(test_proba, bins=50, alpha=0.6, color=ORANGE, label="Test", density=True, edgecolor=GRID)
    dark_style(ax,"OOF vs Test Probability Distribution")
    ax.legend(facecolor=CARD,edgecolor=GRID,labelcolor=TXT)
    fig.tight_layout(); R.plot(fig,"oof_vs_test_dist")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 57: ROW ORDER DEPENDENCY
    # ════════════════════════════════════════════════════
    R.section("📋 57. SATIR SIRASI BAĞIMLILIĞI")
    R.rule("Ardışık satırlar korelasyonlu ise ve CV shuffle=False ise → ciddi leak riski.")
    R.rule("Otokorelasyon > 0.2 olan feature varsa → zaman serisi CV (TimeSeriesSplit) değerlendir.")
    R.rule("Synthetic Kaggle verilerinde genellikle sıra bağımlılığı yok ama kontrol et.")

    autocorr_res = []
    for col in num_cols:
        try:
            vals = train[col].fillna(train[col].median()).values
            ac1  = float(pd.Series(vals).autocorr(lag=1))
            ac5  = float(pd.Series(vals).autocorr(lag=5))
            autocorr_res.append({"Feature":col,"AutoCorr_lag1":ac1,"AutoCorr_lag5":ac5})
        except: pass

    if autocorr_res:
        df_ac = (pd.DataFrame(autocorr_res)
                 .sort_values("AutoCorr_lag1",ascending=False,key=abs).reset_index(drop=True))
        R.table(df_ac.style
                .background_gradient(cmap="RdYlGn_r",subset=["AutoCorr_lag1"])
                .format({"AutoCorr_lag1":"{:.4f}","AutoCorr_lag5":"{:.4f}"}))
        high_ac = df_ac[df_ac["AutoCorr_lag1"].abs()>0.2]["Feature"].tolist()
        if high_ac:
            R.log(f"⚠️ Yüksek otokorelasyon: {high_ac}", cls="warn")
            R.decision(f"Bu feature'lar sıra bağımlı: {high_ac}. "
                       f"TimeSeriesSplit ile CV karşılaştırması yap.", pos=False)
        else:
            R.log("✅ Otokorelasyon yok. Satır sırası bağımsız.", cls="ok")
            R.decision("Shuffle=True ile StratifiedKFold güvenli.", pos=True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 58: FEATURE INTERACTION SİGNİFİKANSI (Bootstrap)
    # ════════════════════════════════════════════════════
    R.section("🔗 58. FEATURE INTERACTION SİGNİFİKANSI (Bootstrap CI)")
    R.rule("İki feature çarpımı eklendikten sonra AUC farkının güven aralığı hesaplanır.")
    R.rule("95% CI alt sınırı > 0 ise → interaction gerçekten anlamlı.")
    R.rule("CI çok geniş ise → yüksek variance. Tekrarlanabilir değil, ekleme.")

    if len(shap_top5) >= 2 and all(c in X_enc.columns for c in shap_top5[:2]):
        col_a, col_b = shap_top5[0], shap_top5[1]
        X_inter = X_enc.copy()
        X_inter[f"{col_a}_x_{col_b}"] = X_inter[col_a] * X_inter[col_b]

        # Bootstrap AUC difference
        n_bootstrap = 50
        auc_diffs   = []
        for _ in range(n_bootstrap):
            idx_b = np.random.choice(len(train), len(train), replace=True)
            X_b   = X_enc.iloc[idx_b];  X_i_b = X_inter.iloc[idx_b]
            y_b   = y_full.iloc[idx_b]
            tr_b  = idx_b[:int(len(idx_b)*0.8)]
            va_b  = idx_b[int(len(idx_b)*0.8):]
            try:
                m1 = lgb.LGBMClassifier(n_estimators=100,verbose=-1,n_jobs=-1,random_state=42)
                m1.fit(X_enc.iloc[tr_b], y_full.iloc[tr_b])
                a1 = roc_auc_score(y_full.iloc[va_b], m1.predict_proba(X_enc.iloc[va_b])[:,1])
                m2 = lgb.LGBMClassifier(n_estimators=100,verbose=-1,n_jobs=-1,random_state=42)
                m2.fit(X_inter.iloc[tr_b], y_full.iloc[tr_b])
                a2 = roc_auc_score(y_full.iloc[va_b], m2.predict_proba(X_inter.iloc[va_b])[:,1])
                auc_diffs.append(a2-a1)
            except: pass

        if auc_diffs:
            mean_diff = np.mean(auc_diffs)
            ci_lo     = np.percentile(auc_diffs, 2.5)
            ci_hi     = np.percentile(auc_diffs, 97.5)
            R.log(f"📊 {col_a} × {col_b} interaction:")
            R.log(f"   Mean AUC diff: {mean_diff:+.5f} | 95% CI: [{ci_lo:+.5f}, {ci_hi:+.5f}]")
            if ci_lo > 0:
                R.decision(f"CI alt sınırı > 0 → interaction anlamlı! Ekle: {col_a}_x_{col_b}", pos=True)
            elif ci_hi < 0:
                R.decision(f"Interaction zararlı. Ekleme.", pos=False)
            else:
                R.decision(f"CI sıfırı kapsıyor → belirsiz. Büyük veri setinde tekrar dene.", pos=False)

            fig, ax = plt.subplots(figsize=(8,4))
            ax.hist(auc_diffs, bins=20, color=BLUE, edgecolor=GRID, alpha=0.8)
            ax.axvline(0,         color=RED,    linestyle="--", lw=2, label="0")
            ax.axvline(ci_lo,     color=ORANGE, linestyle="--", lw=1.5, label="95% CI Lo")
            ax.axvline(ci_hi,     color=ORANGE, linestyle="--", lw=1.5, label="95% CI Hi")
            ax.axvline(mean_diff, color=GREEN,  linestyle="-",  lw=2,  label="Mean")
            dark_style(ax, f"Bootstrap AUC Diff: {col_a}×{col_b}")
            ax.legend(facecolor=CARD,edgecolor=GRID,labelcolor=TXT)
            fig.tight_layout(); R.plot(fig,"interaction_bootstrap")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 59: ENCODING STRATEJİSİ KARŞILAŞTIRMASI
    # ════════════════════════════════════════════════════
    R.section("🔤 59. ENCODİNG STRATEJİSİ KARŞILAŞTIRMASI")
    R.rule("Label encoding, target encoding, one-hot, binary encoding — hepsinin AUC'sini karşılaştır.")
    R.rule("En yüksek AUC'yi veren encoding stratejisini FE'de kullan.")
    R.rule("Kategorik sayısı yüksekse → target encoding. Düşükse → one-hot veya label.")

    enc_res = []
    for col in cat_cols[:3]:  # İlk 3 kategorik
        results_enc = {}
        # Label encoding (mevcut X_enc zaten bu)
        skf3 = StratifiedKFold(3,shuffle=True,random_state=42)

        for enc_name, enc_vals in [
            ("LabelEnc", pd.factorize(train[col].fillna("NA"))[0]),
            ("TargetEnc", None),  # fold içinde hesaplanacak
            ("FreqEnc",   train[col].map(train[col].value_counts()).fillna(0).values),
            ("OrdinalEnc",train[col].map(
                {v:i for i,v in enumerate(train[col].value_counts().index)}).fillna(-1).values)
        ]:
            aucs_enc = []
            for tr_i, va_i in skf3.split(X_enc, y_full):
                if enc_name == "TargetEnc":
                    fold_map = y_full.iloc[tr_i].groupby(
                        train.iloc[tr_i][col].fillna("NA")).mean()
                    vals_tr = train.iloc[tr_i][col].fillna("NA").map(fold_map).fillna(y_full.mean()).values.reshape(-1,1)
                    vals_va = train.iloc[va_i][col].fillna("NA").map(fold_map).fillna(y_full.mean()).values.reshape(-1,1)
                else:
                    vals_tr = enc_vals[tr_i].reshape(-1,1)
                    vals_va = enc_vals[va_i].reshape(-1,1)
                m_enc = lgb.LGBMClassifier(n_estimators=100,verbose=-1,n_jobs=-1,random_state=42)
                try:
                    m_enc.fit(vals_tr, y_full.iloc[tr_i])
                    aucs_enc.append(roc_auc_score(y_full.iloc[va_i],
                                                   m_enc.predict_proba(vals_va)[:,1]))
                except: aucs_enc.append(0.5)
            results_enc[enc_name] = np.mean(aucs_enc)
        best_enc = max(results_enc, key=results_enc.get)
        enc_res.append({"Feature":col, **results_enc, "Best":best_enc})

    if enc_res:
        df_enc = pd.DataFrame(enc_res)
        enc_cols = [c for c in df_enc.columns if c not in ["Feature","Best"]]
        R.table(df_enc.style
                .highlight_max(subset=enc_cols, color="#0d2818", axis=1)
                .format({c:"{:.4f}" for c in enc_cols}))
        for _, row in df_enc.iterrows():
            R.log(f"  {row['Feature']}: En iyi encoding → {row['Best']}")
        target_enc_wins = df_enc[df_enc["Best"]=="TargetEnc"]["Feature"].tolist()
        if target_enc_wins:
            R.decision(f"Target encoding en iyi: {target_enc_wins}. KFold TE kullan.", pos=True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 60: POST-PROCESSING OPTİMİZASYONU
    # ════════════════════════════════════════════════════
    R.section("🔧 60. POST-PROCESSING OPTİMİZASYONU")
    R.rule("Rank transformation: tahminleri 0-1 arası uniform dağılıma çeker. Bazen +0.001 AUC.")
    R.rule("Isotonic regression: kalibrasyon bozuksa tahminleri düzeltir.")
    R.rule("Power transform (p<1): düşük tahminleri yukarı iter. Imbalanced sınıf için faydalı.")
    R.rule("Bu testleri daima OOF üzerinde yap, test üzerinde değil.")

    # Rank transform
    oof_rank = pd.Series(oof_proba).rank(pct=True).values
    auc_rank = roc_auc_score(y_full, oof_rank)

    # Power transforms
    auc_powers = {}
    for p in [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
        transformed = np.clip(oof_proba ** p, 0, 1)
        auc_powers[p] = roc_auc_score(y_full, transformed)

    best_power = max(auc_powers, key=auc_powers.get)
    R.log(f"📊 Original AUC         : {baseline_auc:.5f}")
    R.log(f"📊 Rank transform AUC   : {auc_rank:.5f} (Δ={auc_rank-baseline_auc:+.5f})")
    R.log(f"📊 Best power ({best_power:.1f}) AUC : {auc_powers[best_power]:.5f} "
          f"(Δ={auc_powers[best_power]-baseline_auc:+.5f})")

    if auc_rank > baseline_auc + 0.0005:
        R.decision("Rank transform AUC artırıyor. Test tahminlerine de uygula.", pos=True)
    if auc_powers[best_power] > baseline_auc + 0.0005:
        R.decision(f"Power transform p={best_power:.1f} AUC artırıyor. "
                   f"Test tahminlerine uygula: proba^{best_power:.1f}", pos=True)
    else:
        R.decision("Post-processing anlamlı iyileştirme yapmıyor. "
                   "Ham tahminleri kullan.", pos=True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 61: SUBMISSION KORELASYON ANALİZİ
    # ════════════════════════════════════════════════════
    R.section("🏆 61. SUBMİSSİON KORELASYON ANALİZİ (Ensemble Diversity)")
    R.rule("İyi ensemble için model tahminlerinin korelasyonu < 0.95 olmalı.")
    R.rule("Korelasyon > 0.98 → bu iki model aynı şeyi öğrenmiş. Ensemble'a ekleme.")
    R.rule("Korelasyon < 0.90 → çeşitlilik yüksek. Blend veya stack ile +0.002–0.005 kazanabilirsin.")

    # Farklı seed/param'larla modeller
    submission_preds = {"LGB_base": test_proba}
    for seed in [0, 123]:
        sp = np.zeros(len(test))
        for tr_i, va_i in skf5.split(X_enc, y_full):
            m_s = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                      num_leaves=31, is_unbalance=True,
                                      random_state=seed, verbose=-1, n_jobs=-1)
            m_s.fit(X_enc.iloc[tr_i], y_full.iloc[tr_i])
            sp += m_s.predict_proba(X_test_enc)[:,1] / 5
        submission_preds[f"LGB_seed{seed}"] = sp

    sub_df   = pd.DataFrame(submission_preds)
    sub_corr = sub_df.corr()
    R.table(sub_corr.style
            .background_gradient(cmap="RdYlGn_r")
            .format("{:.4f}"))

    low_corr_pairs = [(sub_corr.columns[i], sub_corr.columns[j],
                       sub_corr.iloc[i,j])
                      for i in range(len(sub_corr))
                      for j in range(i+1,len(sub_corr))
                      if sub_corr.iloc[i,j] < 0.95]
    high_corr_pairs= [(sub_corr.columns[i], sub_corr.columns[j])
                      for i in range(len(sub_corr))
                      for j in range(i+1,len(sub_corr))
                      if sub_corr.iloc[i,j] > 0.98]

    if low_corr_pairs:
        R.decision(f"Çeşitli modeller var: {[(a,b) for a,b,_ in low_corr_pairs]}. "
                   f"Blend et → +0.002–0.005 bekleniyor.", pos=True)
    if high_corr_pairs:
        R.decision(f"Yüksek korelasyon: {high_corr_pairs}. "
                   f"Bu çiftlerden birini ensemble'dan çıkar.", pos=False)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 62: CHI-SQUARE GOODNESS OF FIT (Train vs Test Dağılım)
    # ════════════════════════════════════════════════════
    R.section("📊 62. CHI-SQUARE GOODNESS OF FIT (Kategorik Dağılım Testi)")
    R.rule("Train ve test kategorik dağılımları istatistiksel olarak aynı mı?")
    R.rule("Chi-square p < 0.05 → dağılımlar farklı. Bu feature drift'li.")
    R.rule("KS test numerikleri, Chi-square kategorikleri test eder. İkisi birlikte tam tablo verir.")

    chi_res = []
    for col in cat_cols:
        tr_vc = train[col].value_counts(normalize=True)
        ts_vc = test[col].value_counts(normalize=True)
        common = sorted(set(tr_vc.index) & set(ts_vc.index))
        if len(common) < 2: continue
        tr_counts = (tr_vc[common] * len(train)).values.astype(int)
        ts_counts = (ts_vc[common] * len(test)).values.astype(int)
        if tr_counts.sum() == 0 or ts_counts.sum() == 0: continue
        try:
            combined = np.array([tr_counts, ts_counts])
            chi2_val, p_val, _, _ = chi2_contingency(combined)
            chi_res.append({"Feature":col,"Chi2":chi2_val,"p_value":p_val,
                             "Drift":("🔴 Farklı" if p_val<0.05 else "✅ Benzer")})
        except: pass

    if chi_res:
        df_chi = pd.DataFrame(chi_res).sort_values("p_value")
        R.table(df_chi.style
                .background_gradient(cmap="RdYlGn",subset=["p_value"])
                .format({"Chi2":"{:.2f}","p_value":"{:.4f}"}))
        drifted_cats = df_chi[df_chi["p_value"]<0.05]["Feature"].tolist()
        if drifted_cats:
            R.log(f"🔴 Kategorik drift: {drifted_cats}", cls="critical")
            R.decision(f"Kategorik drift'li feature'lar: {drifted_cats}. "
                       f"Target encoding kullan (label encoding drift'i büyütür).", pos=False)
        else:
            R.log("✅ Tüm kategorik feature'lar train/test'te benzer dağılımda.", cls="ok")
            R.decision("Kategorik drift yok. Label encoding güvenli.", pos=True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 64: ADVERSARIAL VALİDATION
    # ════════════════════════════════════════════════════
    R.section("⚔️ 64. ADVERSARIAL VALİDATION")
    R.rule("Train ve test'i ayırt edebiliyor muyuz? AUC > 0.70 → ciddi covariate shift.")
    R.rule("AUC = 0.50 → train ve test ayrıt edilemiyor, dağılımlar benzer. İdeal durum.")
    R.rule("Hangi feature'lar ayırımı yapıyor? Bu feature'lar en çok shift yapan feature'lardır.")
    R.rule("Shift yapan feature'larda target encoding veya PSI-based drop stratejisi uygula.")

    adv_y    = np.concatenate([np.zeros(len(train)), np.ones(len(test))])
    adv_X_tr = train[feats].copy()
    adv_X_ts = test[feats].copy()
    for c in adv_X_tr.select_dtypes(["object","category"]).columns:
        adv_X_tr[c] = pd.factorize(adv_X_tr[c])[0]
        adv_X_ts[c] = pd.factorize(adv_X_ts[c])[0]
    adv_X = pd.concat([adv_X_tr.fillna(-1), adv_X_ts.fillna(-1)], ignore_index=True)
    adv_y_s = pd.Series(adv_y)

    adv_aucs = []
    adv_skf  = StratifiedKFold(5, shuffle=True, random_state=42)
    adv_model= None
    for tr_i, va_i in adv_skf.split(adv_X, adv_y_s):
        m_adv = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                    max_depth=4, verbose=-1, n_jobs=-1, random_state=42)
        m_adv.fit(adv_X.iloc[tr_i], adv_y_s.iloc[tr_i])
        adv_aucs.append(roc_auc_score(adv_y_s.iloc[va_i],
                                       m_adv.predict_proba(adv_X.iloc[va_i])[:,1]))
        adv_model = m_adv
    adv_auc = np.mean(adv_aucs)

    R.log(f"📊 Adversarial AUC: {adv_auc:.4f}")
    if adv_auc > 0.80:
        R.log(f"🔴 CİDDİ SHIFT: {adv_auc:.4f}", cls="critical")
        R.decision("Covariate shift çok yüksek. Bu model test'te güvenilir olmayabilir. "
                   "Shift yapan feature'ları drop et veya target encode et.", pos=False)
    elif adv_auc > 0.70:
        R.log(f"🟠 ORTA SHIFT: {adv_auc:.4f}", cls="warn")
        R.decision("Anlamlı covariate shift var. PSI > 0.10 feature'ları önce normalize et.", pos=False)
    elif adv_auc > 0.60:
        R.log(f"🟡 HAFIF SHIFT: {adv_auc:.4f}", cls="warn")
        R.decision("Küçük shift var. Model genellikle robust. Yine de PSI'yi izle.", pos=True)
    else:
        R.log(f"✅ SHIFT YOK: {adv_auc:.4f}", cls="ok")
        R.decision("Train ve test ayırt edilemiyor. Mükemmel dağılım benzerliği.", pos=True)

    # Hangi feature'lar ayırıyor
    if adv_model is not None:
        adv_imp = pd.DataFrame({
            "Feature": feats,
            "Importance": adv_model.feature_importances_
        }).sort_values("Importance", ascending=False).head(10)
        R.table(adv_imp.style.bar(subset=["Importance"], color=ORANGE)
                .format({"Importance": "{:.0f}"}))
        shift_feats = adv_imp.head(5)["Feature"].tolist()
        if adv_auc > 0.65:
            R.decision(f"En çok shift yapan feature'lar: {shift_feats}. "
                       f"Bu feature'lar için PSI ve KS'yi öncelikli incele.", pos=False)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 65: DUMMY CLASSIFIER BASELINE
    # ════════════════════════════════════════════════════
    R.section("🎲 65. DUMMY CLASSİFİER BASELINE (Şans Üstünde mi?)")
    R.rule("Model Stratified majority baseline'ı ne kadar geçiyor?")
    R.rule("AUC farkı < 0.05 → model yeterince öğrenmemiş. Ciddi FE veya model değişikliği gerekli.")
    R.rule("Random model AUC = 0.50. Stratified baseline genellikle 0.50–0.60 civarı.")
    R.rule("Bu test modelin 'gerçekten öğrenip öğrenmediğini' doğrular.")

    from sklearn.dummy import DummyClassifier
    dummy_aucs = {"stratified":[], "most_frequent":[], "prior":[]}
    for tr_i, va_i in skf5.split(X_enc, y_full):
        for strategy in dummy_aucs:
            dc = DummyClassifier(strategy=strategy, random_state=42)
            dc.fit(X_enc.iloc[tr_i], y_full.iloc[tr_i])
            try:
                proba = dc.predict_proba(X_enc.iloc[va_i])[:,1]
                dummy_aucs[strategy].append(roc_auc_score(y_full.iloc[va_i], proba))
            except:
                dummy_aucs[strategy].append(0.5)

    df_dummy = pd.DataFrame({
        "Strategy":   list(dummy_aucs.keys()),
        "AUC":        [np.mean(v) for v in dummy_aucs.values()],
    })
    df_dummy["vs_Model"] = baseline_auc - df_dummy["AUC"]
    R.table(df_dummy.style
            .background_gradient(cmap="YlGn", subset=["vs_Model"])
            .format({"AUC":"{:.4f}","vs_Model":"{:+.4f}"}))

    best_dummy = df_dummy["AUC"].max()
    lift_over_dummy = baseline_auc - best_dummy
    R.log(f"📊 Model AUC      : {baseline_auc:.4f}")
    R.log(f"📊 Best Dummy AUC : {best_dummy:.4f}")
    R.log(f"📊 Lift over dummy: {lift_over_dummy:+.4f}")

    if lift_over_dummy > 0.15:
        R.decision(f"Model dummy'yi {lift_over_dummy:.3f} AUC ile açık ara geçiyor. "
                   f"Gerçekten öğreniyor.", pos=True)
    elif lift_over_dummy > 0.05:
        R.decision(f"Model dummy'yi {lift_over_dummy:.3f} geçiyor. Kabul edilebilir ama FE ile artır.", pos=True)
    else:
        R.decision(f"Model dummy'yi sadece {lift_over_dummy:.3f} geçiyor. "
                   f"Ciddi sorun: imbalance, leak veya zayıf feature'lar.", pos=False)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 66: PERMUTATION TEST (Global Anlamlılık)
    # ════════════════════════════════════════════════════
    R.section("🔀 66. PERMUTATION TEST (Modelin Global Anlamlılığı)")
    R.rule("Target'ı karıştırınca AUC ne oluyor? Gerçek AUC bu dağılımın neresinde?")
    R.rule("p-value = P(permuted AUC >= gerçek AUC). p < 0.05 → model anlamlı.")
    R.rule("p > 0.05 → model şans üstünde değil. Tüm pipeline'ı sorgula.")
    R.rule("Bu test özellikle küçük veri setlerinde önemlidir.")

    n_perm = 50
    perm_aucs = []
    y_arr = y_full.values.copy()
    for _ in range(n_perm):
        y_perm = np.random.permutation(y_arr)
        p_aucs = []
        for tr_i, va_i in StratifiedKFold(3, shuffle=True,
                                           random_state=np.random.randint(1000)).split(
                X_enc, y_full):
            try:
                m_p = lgb.LGBMClassifier(n_estimators=100, verbose=-1,
                                          n_jobs=-1, random_state=42)
                m_p.fit(X_enc.iloc[tr_i], y_perm[tr_i])
                p_aucs.append(roc_auc_score(y_perm[va_i],
                                             m_p.predict_proba(X_enc.iloc[va_i])[:,1]))
            except: p_aucs.append(0.5)
        perm_aucs.append(np.mean(p_aucs))

    perm_aucs = np.array(perm_aucs)
    p_val_perm = (perm_aucs >= baseline_auc).mean()
    R.log(f"📊 Gerçek AUC         : {baseline_auc:.5f}")
    R.log(f"📊 Permuted AUC mean  : {perm_aucs.mean():.5f} ± {perm_aucs.std():.5f}")
    R.log(f"📊 Permutation p-value: {p_val_perm:.4f}")

    if p_val_perm < 0.01:
        R.decision(f"p={p_val_perm:.4f} — Model istatistiksel olarak çok anlamlı (p<0.01).", pos=True)
    elif p_val_perm < 0.05:
        R.decision(f"p={p_val_perm:.4f} — Model anlamlı (p<0.05).", pos=True)
    else:
        R.decision(f"p={p_val_perm:.4f} — Model şans üstünde değil! "
                   f"Leak kontrolü veya FE gerekli.", pos=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(perm_aucs, bins=20, color=GRID, edgecolor=CARD, alpha=0.8, label="Permuted AUC")
    ax.axvline(baseline_auc, color=RED, lw=2.5, label=f"Gerçek AUC={baseline_auc:.4f}")
    ax.axvline(perm_aucs.mean(), color=ORANGE, lw=1.5, linestyle="--",
               label=f"Perm mean={perm_aucs.mean():.4f}")
    dark_style(ax, "Permutation Test — Model Anlamlılığı")
    ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TXT)
    fig.tight_layout(); R.plot(fig, "permutation_test")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 67: VIF (Variance Inflation Factor)
    # ════════════════════════════════════════════════════
    R.section("📐 67. VIF — MULTICOLLINEARITY (Çoklu Doğrusal Bağlantı)")
    R.rule("VIF > 10 → ciddi multicollinearity. Bu feature başka bir feature'ın doğrusal kombinasyonu.")
    R.rule("VIF 5–10 → orta. İzle ama hemen drop etme.")
    R.rule("VIF < 5 → güvenli.")
    R.rule("VIF yüksek feature DROP edilmez; ya diğeriyle birleştirilir ya da ikisinden biri seçilir.")
    R.rule("LightGBM multicollinearity'ye robust ama interaction feature'lar için VIF kritik.")

    if len(num_cols) >= 2:
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            X_vif = train[num_cols].fillna(train[num_cols].median())
            X_vif = (X_vif - X_vif.mean()) / (X_vif.std() + 1e-9)
            vif_data = []
            for i, col in enumerate(num_cols):
                try:
                    v = variance_inflation_factor(X_vif.values, i)
                    vif_data.append({"Feature": col, "VIF": v,
                                     "Severity": ("🔴 Ciddi" if v>10 else
                                                   "🟡 Orta"  if v>5  else "✅ OK")})
                except: pass
            df_vif = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)
            R.table(df_vif.style
                    .background_gradient(cmap="RdYlGn_r", subset=["VIF"])
                    .format({"VIF": "{:.2f}"}))
            high_vif = df_vif[df_vif["VIF"] > 10]["Feature"].tolist()
            if high_vif:
                R.log(f"🔴 Yüksek VIF: {high_vif}", cls="critical")
                R.decision(f"VIF > 10 feature'lar: {high_vif}. "
                           f"Birbirleriyle korelasyonu kontrol et, ratio feature yap.", pos=False)
            else:
                R.log("✅ Ciddi multicollinearity yok.", cls="ok")
                R.decision("VIF uygun. Feature'lar bağımsız bilgi taşıyor.", pos=True)
        except ImportError:
            R.log("⚠️ statsmodels kurulu değil. !pip install statsmodels", cls="warn")
            # Manuel VIF proxy
            vif_proxy = []
            for col in num_cols:
                other = [c for c in num_cols if c != col]
                if not other: continue
                corrs = [abs(train[col].corr(train[o])) for o in other]
                max_c = max(corrs) if corrs else 0
                vif_proxy.append({"Feature":col, "Max_Corr_with_Others": max_c,
                                   "Risk": "🔴" if max_c>0.9 else "🟡" if max_c>0.7 else "✅"})
            df_vp = pd.DataFrame(vif_proxy).sort_values("Max_Corr_with_Others", ascending=False)
            R.table(df_vp.style
                    .background_gradient(cmap="RdYlGn_r", subset=["Max_Corr_with_Others"])
                    .format({"Max_Corr_with_Others": "{:.4f}"}))
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 68: MI vs PEARSON TUTARSIZLIĞI
    # ════════════════════════════════════════════════════
    R.section("🔗 68. MI vs PEARSON TUTARSIZLIĞI (Non-Linear İlişki Tespiti)")
    R.rule("MI yüksek + Pearson düşük → non-linear ilişki var. log/sqrt/binning transform gerekli.")
    R.rule("MI ≈ Pearson → doğrusal ilişki. Basit transform yeterli.")
    R.rule("MI düşük + Pearson düşük → feature zayıf. Drop adayı.")
    R.rule("Bu test hangi feature'lara non-linear FE uygulanacağını belirler.")

    from sklearn.feature_selection import mutual_info_classif
    if num_cols:
        X_mi = X_enc[num_cols].fillna(-1)
        mi_scores = mutual_info_classif(X_mi, y_full, random_state=42)
        mi_norm   = mi_scores / (mi_scores.max() + 1e-9)

        pearson_scores = []
        spearman_scores = []
        for col in num_cols:
            p_r  = abs(train[col].fillna(train[col].median()).corr(y_full))
            sp_r, _ = spearmanr(train[col].fillna(train[col].median()), y_full)
            pearson_scores.append(p_r)
            spearman_scores.append(abs(sp_r))

        df_mi = pd.DataFrame({
            "Feature":   num_cols,
            "MI_norm":   mi_norm,
            "Pearson_r": pearson_scores,
            "Spearman_r":spearman_scores,
        })
        df_mi["MI_Pearson_gap"]   = df_mi["MI_norm"] - df_mi["Pearson_r"]
        df_mi["Spearman_Pearson"] = df_mi["Spearman_r"] - df_mi["Pearson_r"]
        df_mi["Pattern"] = df_mi.apply(lambda r:
            "Non-linear (MI>>Pearson)" if r["MI_Pearson_gap"] > 0.2 else
            "Monoton non-linear"       if r["Spearman_Pearson"] > 0.15 else
            "Linear"                   if r["Pearson_r"] > 0.1 else
            "Weak", axis=1)
        df_mi = df_mi.sort_values("MI_norm", ascending=False).reset_index(drop=True)
        R.table(df_mi.style
                .background_gradient(cmap="YlGn",   subset=["MI_norm"])
                .background_gradient(cmap="Blues",  subset=["Pearson_r"])
                .background_gradient(cmap="OrRd",   subset=["MI_Pearson_gap"])
                .format({"MI_norm":"{:.4f}","Pearson_r":"{:.4f}",
                         "Spearman_r":"{:.4f}","MI_Pearson_gap":"{:.4f}",
                         "Spearman_Pearson":"{:.4f}"}))

        nonlinear = df_mi[df_mi["Pattern"]=="Non-linear (MI>>Pearson)"]["Feature"].tolist()
        monoton   = df_mi[df_mi["Pattern"]=="Monoton non-linear"]["Feature"].tolist()
        weak      = df_mi[df_mi["Pattern"]=="Weak"]["Feature"].tolist()

        if nonlinear:
            R.decision(f"Non-linear feature'lar: {nonlinear}. "
                       f"pd.qcut binning + PDP ile threshold bul.", pos=True)
        if monoton:
            R.decision(f"Monoton non-linear: {monoton}. log1p veya sqrt transform uygula.", pos=True)
        if weak:
            R.decision(f"Zayıf feature'lar: {weak}. Drop et veya interaction ile güçlendir.", pos=False)

        # Scatter plot: MI vs Pearson
        fig, ax = plt.subplots(figsize=(8, 5))
        for _, row in df_mi.iterrows():
            color = (RED if row["Pattern"]=="Non-linear (MI>>Pearson)" else
                     ORANGE if row["Pattern"]=="Monoton non-linear" else
                     GREEN  if row["Pattern"]=="Linear" else GRID)
            ax.scatter(row["Pearson_r"], row["MI_norm"], color=color, s=60, zorder=3)
            ax.annotate(row["Feature"], (row["Pearson_r"], row["MI_norm"]),
                        textcoords="offset points", xytext=(5,3),
                        fontsize=7, color=TXT)
        ax.plot([0,1],[0,1], color=GRID, linestyle="--", lw=1.5, label="MI=Pearson (linear)")
        dark_style(ax, "MI vs Pearson — Non-Linear Detection")
        ax.set_xlabel("Pearson |r|"); ax.set_ylabel("MI (normalized)")
        ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TXT)
        fig.tight_layout(); R.plot(fig, "mi_vs_pearson")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 69: SPEARMAN vs PEARSON TUTARSIZLIĞI
    # ════════════════════════════════════════════════════
    R.section("📈 69. SPEARMAN vs PEARSON (Monoton Non-Linear Tespiti)")
    R.rule("Spearman >> Pearson → monoton ama non-linear ilişki. log1p/sqrt transform işe yarar.")
    R.rule("İkisi de yüksek → doğrusal. Transform gerekmez.")
    R.rule("İkisi de düşük → ya non-monoton non-linear ya da zayıf feature.")
    R.rule("Bu test Section 68'i tamamlar: MI tüm non-linearity'yi yakalar, Spearman sadece monoton olanı.")

    if num_cols:
        sp_res = []
        for col in num_cols:
            vals = train[col].fillna(train[col].median())
            p_r  = abs(vals.corr(y_full))
            sp_r, sp_p = spearmanr(vals, y_full)
            sp_r = abs(sp_r)
            gap  = sp_r - p_r
            sp_res.append({"Feature":col,"Pearson":p_r,"Spearman":sp_r,
                            "Gap(Sp-Pe)":gap,
                            "Transform":("log1p/sqrt" if gap>0.10 else
                                          "Yok gerekli" if p_r>0.05 else "Drop adayı")})
        df_sp = (pd.DataFrame(sp_res)
                 .sort_values("Gap(Sp-Pe)",ascending=False).reset_index(drop=True))
        R.table(df_sp.style
                .background_gradient(cmap="OrRd", subset=["Gap(Sp-Pe)"])
                .format({"Pearson":"{:.4f}","Spearman":"{:.4f}","Gap(Sp-Pe)":"{:.4f}"}))
        transform_needed = df_sp[df_sp["Gap(Sp-Pe)"]>0.10]["Feature"].tolist()
        if transform_needed:
            R.decision(f"Monoton non-linear feature'lar: {transform_needed}. "
                       f"log1p veya sqrt transform uygula, sonra tekrar ölç.", pos=True)
        else:
            R.decision("Spearman-Pearson gap düşük. Mevcut feature'lar linear veya zaten transform edilmiş.", pos=True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 70: DUPLICATE FEATURE DETECTION
    # ════════════════════════════════════════════════════
    R.section("👯 70. DUPLICATE FEATURE DETECTION")
    R.rule("Farklı isimde ama aynı bilgiyi taşıyan feature'lar varsa model confusion yaşar.")
    R.rule("Pearson |r| > 0.99 → neredeyse duplicate. Birini kaldır.")
    R.rule("Aynı değer sayısı / toplam satır > 0.99 → birebir aynı. Kesinlikle birini kaldır.")
    R.rule("Bu korelasyon matrisinden farklı: sadece neredeyse mükemmel çiftlere bakıyoruz.")

    dup_feat_res = []
    if len(num_cols) >= 2:
        for i, c1 in enumerate(num_cols):
            for c2 in num_cols[i+1:]:
                try:
                    r = abs(train[c1].fillna(0).corr(train[c2].fillna(0)))
                    # Exact value match
                    exact = (train[c1].fillna(-999) == train[c2].fillna(-999)).mean()
                    if r > 0.95 or exact > 0.95:
                        dup_feat_res.append({
                            "Feature_A": c1, "Feature_B": c2,
                            "Pearson_r": r, "Exact_Match%": exact*100,
                            "Action": ("DROP birini" if r>0.99 or exact>0.99
                                       else "İzle / ratio yap")})
                except: pass

    for c1 in cat_cols:
        for c2 in cat_cols:
            if c1 >= c2: continue
            try:
                v = cramers_v(train[c1].fillna("NA"), train[c2].fillna("NA"))
                if v > 0.95:
                    dup_feat_res.append({"Feature_A":c1,"Feature_B":c2,
                                         "Pearson_r":v,"Exact_Match%":0,
                                         "Action":"DROP birini (Cramér V > 0.95)"})
            except: pass

    if dup_feat_res:
        df_dup_f = pd.DataFrame(dup_feat_res).sort_values("Pearson_r", ascending=False)
        R.table(df_dup_f.style
                .background_gradient(cmap="OrRd", subset=["Pearson_r"])
                .format({"Pearson_r":"{:.4f}","Exact_Match%":"{:.2f}"}))
        drop_dups = df_dup_f[df_dup_f["Action"].str.startswith("DROP")]
        if len(drop_dups):
            pairs = list(zip(drop_dups["Feature_A"], drop_dups["Feature_B"]))
            R.decision(f"{len(pairs)} duplicate çift: {pairs}. Her çiftten birini kaldır.", pos=False)
    else:
        R.log("✅ Duplicate feature yok (r > 0.95 veya exact match > 0.95 çift bulunamadı).", cls="ok")
        R.decision("Feature'lar birbirinden bağımsız bilgi taşıyor.", pos=True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 71: NEAR-CONSTANT FEATURE DETECTION
    # ════════════════════════════════════════════════════
    R.section("📉 71. NEAR-CONSTANT FEATURE DETECTION")
    R.rule("Tek değer > %95 → neredeyse sabit. Modele katkısı minimal, gürültü ekliyor.")
    R.rule("Mevcut pipeline > %99.5 için kesiyor — bu çok yüksek. %95 daha doğru eşik.")
    R.rule("Variance Threshold: std < 0.01 olan numeric feature'lar da near-constant.")
    R.rule("Near-constant feature'ı DROP et. Modele zarar vermese de anlamsız.")

    near_const = []
    for col in feats:
        vc   = train[col].value_counts(normalize=True)
        top_pct = vc.iloc[0] * 100 if len(vc) > 0 else 100
        std_val = train[col].std() if train[col].dtype in [float, int, np.float64] else None
        if top_pct > 95:
            near_const.append({"Feature":col,"Top_Value":vc.index[0],
                                "Top_Pct%":top_pct,"Std":std_val,
                                "Action":("DROP" if top_pct>99 else "İzle")})
        elif std_val is not None and std_val < 0.01:
            near_const.append({"Feature":col,"Top_Value":"(numeric near-zero std)",
                                "Top_Pct%":None,"Std":std_val,"Action":"DROP"})

    if near_const:
        df_nc = pd.DataFrame(near_const)
        R.table(df_nc.style.format({"Top_Pct%":"{:.2f}","Std":"{:.6f}"}))
        drop_nc = [r["Feature"] for r in near_const if r["Action"]=="DROP"]
        watch_nc= [r["Feature"] for r in near_const if r["Action"]=="İzle"]
        if drop_nc:
            R.decision(f"Near-constant feature DROP: {drop_nc}", pos=False)
        if watch_nc:
            R.decision(f"Borderline near-constant (izle): {watch_nc}", pos=False)
    else:
        R.log("✅ Near-constant feature yok (top value < %95).", cls="ok")
        R.decision("Tüm feature'lar yeterli variance'a sahip.", pos=True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 72: STRING ANOMALY DETECTION (Kategorik Yazım Tutarsızlığı)
    # ════════════════════════════════════════════════════
    R.section("🔤 72. STRING ANOMALY DETECTION (Kategorik Yazım Tutarsızlığı)")
    R.rule("'Male', 'male', 'MALE', ' male' → aynı değer farklı yazılmış. Ayrı kategori sayılır, yanlış.")
    R.rule("Lowercase + strip normalizasyonu sonrası kategori sayısı azalıyorsa sorun var.")
    R.rule("Bu özellikle original dataset ile Kaggle synthetic verisi birleştirildiğinde önemli.")
    R.rule("Normalizasyon sonrası rare category analizi tekrar yap.")

    string_issues = []
    for col in cat_cols:
        orig_nunique = train[col].dropna().nunique()
        normalized   = train[col].dropna().astype(str).str.strip().str.lower()
        norm_nunique = normalized.nunique()
        if norm_nunique < orig_nunique:
            # Hangi değerler birleşiyor?
            orig_vc = train[col].value_counts()
            norm_vc = normalized.value_counts()
            collapsed = orig_nunique - norm_nunique
            string_issues.append({
                "Feature":col,
                "Original_Unique":orig_nunique,
                "Normalized_Unique":norm_nunique,
                "Collapsed":collapsed,
                "Action":"str.strip().str.lower() uygula"
            })

    if string_issues:
        df_str = pd.DataFrame(string_issues)
        R.table(df_str.style
                .background_gradient(cmap="OrRd", subset=["Collapsed"]))
        R.log(f"⚠️ {len(string_issues)} feature'da yazım tutarsızlığı var.", cls="warn")
        affected = [r["Feature"] for r in string_issues]
        R.decision(f"Bu feature'lara .str.strip().str.lower() uygula: {affected}. "
                   f"Sonra rare category analizini tekrar çalıştır.", pos=False)
    else:
        R.log("✅ Yazım tutarsızlığı yok. Kategorik değerler normalize.", cls="ok")
        R.decision("String normalizasyona gerek yok.", pos=True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 73: CROSS-FEATURE CONSISTENCY (Domain Rules)
    # ════════════════════════════════════════════════════
    R.section("🏗️ 73. CROSS-FEATURE CONSISTENCY (Domain Kuralları)")
    R.rule("İmkansız feature kombinasyonları → veri kalitesi problemi veya encoding hatası.")
    R.rule("Bu satırları DROP etme; is_domain_violation flag ekle.")
    R.rule("Telco'ya özel: 'No internet service' ama 'StreamingTV=Yes' → imkansız.")
    R.rule("Violation oranı > %5 → veri kaynağını sorgula.")

    domain_violations = []

    # Telco-specific domain rules
    telco_rules = []
    if "InternetService" in train.columns:
        inet_dependent = [c for c in ["OnlineSecurity","OnlineBackup","DeviceProtection",
                                       "TechSupport","StreamingTV","StreamingMovies"]
                          if c in train.columns]
        for dep in inet_dependent:
            # No internet but has service
            mask = ((train["InternetService"]=="No") &
                    (train[dep].isin(["Yes","No"])) &
                    (~train[dep].isin(["No internet service","No"])))
            n_viol = mask.sum()
            if n_viol > 0:
                domain_violations.append({
                    "Rule": f"InternetService=No but {dep}=Yes",
                    "Count": n_viol, "Rate%": n_viol/len(train)*100
                })
                telco_rules.append(mask)

    if "PhoneService" in train.columns and "MultipleLines" in train.columns:
        mask2 = ((train["PhoneService"]=="No") &
                 (train["MultipleLines"]=="Yes"))
        n2 = mask2.sum()
        if n2 > 0:
            domain_violations.append({"Rule":"PhoneService=No but MultipleLines=Yes",
                                       "Count":n2,"Rate%":n2/len(train)*100})
            telco_rules.append(mask2)

    # Generic: negative numeric values where impossible
    for col in num_cols:
        if col.lower() in ["tenure","totalcharges","monthlycharges"]:
            neg_count = (train[col] < 0).sum()
            if neg_count > 0:
                domain_violations.append({"Rule":f"{col} < 0 (imkansız)",
                                           "Count":neg_count,"Rate%":neg_count/len(train)*100})

    if domain_violations:
        df_dv = pd.DataFrame(domain_violations)
        R.table(df_dv.style
                .background_gradient(cmap="OrRd", subset=["Rate%"])
                .format({"Rate%":"{:.2f}"}))
        total_viol = sum(r["Count"] for r in domain_violations)
        viol_rate  = total_viol / len(train) * 100
        R.log(f"📊 Toplam violation satırı: {total_viol:,} ({viol_rate:.2f}%)")
        if viol_rate > 5:
            R.decision(f"Yüksek domain violation oranı ({viol_rate:.1f}%). "
                       f"Veri kaynağını sorgula. is_domain_violation flag zorunlu.", pos=False)
        else:
            R.decision("Domain violation düşük. is_domain_violation flag ekle, model kullansın.", pos=True)
    else:
        R.log("✅ Domain violation yok. Feature kombinasyonları tutarlı.", cls="ok")
        R.decision("Cross-feature consistency iyi.", pos=True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 74: STRATIFICATION QUALITY CHECK
    # ════════════════════════════════════════════════════
    R.section("⚖️ 74. STRATİFİKASYON KALİTESİ (Fold Bazlı Target Rate)")
    R.rule("Her fold'un target rate'i birbirine benzer mi? Fark > %2 → CV güvenilmez.")
    R.rule("Imbalanced veri setinde StratifiedKFold kullanılmazsa fold'lar arası ciddi fark oluşabilir.")
    R.rule("Bu test CV stratejisinin doğruluğunu doğrular.")
    R.rule("Ayrıca: her fold'da train ve val dağılımı benzer mi? KS testi ile ölç.")

    strat_quality = []
    for fold, (tr_i, va_i) in enumerate(skf5.split(X_enc, y_full)):
        tr_rate = y_full.iloc[tr_i].mean() * 100
        va_rate = y_full.iloc[va_i].mean() * 100
        strat_quality.append({
            "Fold": fold+1,
            "Train_Rate%": tr_rate,
            "Val_Rate%":   va_rate,
            "Diff": abs(tr_rate - va_rate),
            "Train_N": len(tr_i),
            "Val_N":   len(va_i)
        })
    df_strat = pd.DataFrame(strat_quality)
    R.table(df_strat.style
            .background_gradient(cmap="RdYlGn_r", subset=["Diff"])
            .format({"Train_Rate%":"{:.2f}","Val_Rate%":"{:.2f}","Diff":"{:.3f}"}))

    max_diff = df_strat["Diff"].max()
    overall_rate = y_full.mean() * 100
    R.log(f"📊 Overall target rate: {overall_rate:.2f}%")
    R.log(f"📊 Max fold diff      : {max_diff:.3f}%")

    if max_diff > 2.0:
        R.decision(f"Fold farkı {max_diff:.2f}% — StratifiedKFold düzgün çalışmıyor. "
                   f"sklearn versiyonunu kontrol et.", pos=False)
    elif max_diff > 0.5:
        R.decision(f"Fold farkı {max_diff:.2f}% — Küçük sapma. Normal kabul edilir.", pos=True)
    else:
        R.decision(f"Mükemmel stratification. Tüm fold'lar benzer target rate ({max_diff:.2f}% max diff).", pos=True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 75: SEED STABİLİTE ANALİZİ
    # ════════════════════════════════════════════════════
    R.section("🎯 75. SEED STABİLİTE ANALİZİ (Model Tekrarlanabilirliği)")
    R.rule("AUC std across seeds > 0.005 → model instabil. Sonuçlara güvenme.")
    R.rule("Instabilite nedenleri: az veri, yüksek variance, CV folds az, non-deterministic ops.")
    R.rule("Kaggle'da: final submission için en yüksek seed'i değil, ortalamayı kullan.")
    R.rule("RepeatedStratifiedKFold (5x3) instabiliteyi azaltır.")

    seeds = [0, 42, 123, 456, 789]
    seed_aucs = []
    for seed in seeds:
        s_aucs = []
        skf_s  = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr_i, va_i in skf_s.split(X_enc, y_full):
            m_s = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                      is_unbalance=True, random_state=seed,
                                      verbose=-1, n_jobs=-1)
            m_s.fit(X_enc.iloc[tr_i], y_full.iloc[tr_i])
            s_aucs.append(roc_auc_score(y_full.iloc[va_i],
                                         m_s.predict_proba(X_enc.iloc[va_i])[:,1]))
        seed_aucs.append({"Seed":seed,"AUC":np.mean(s_aucs),"Std":np.std(s_aucs)})

    df_seed = pd.DataFrame(seed_aucs)
    R.table(df_seed.style
            .background_gradient(cmap="YlGn", subset=["AUC"])
            .format({"AUC":"{:.5f}","Std":"{:.5f}"}))

    seed_auc_std = df_seed["AUC"].std()
    seed_auc_range = df_seed["AUC"].max() - df_seed["AUC"].min()
    R.log(f"📊 AUC across seeds: {df_seed['AUC'].mean():.5f} ± {seed_auc_std:.5f}")
    R.log(f"📊 AUC range        : {seed_auc_range:.5f}")

    if seed_auc_std > 0.005:
        R.decision(f"Yüksek seed instabilitesi (std={seed_auc_std:.4f}). "
                   f"RepeatedStratifiedKFold(5,3) veya daha fazla fold kullan.", pos=False)
    elif seed_auc_std > 0.002:
        R.decision(f"Orta instabilite (std={seed_auc_std:.4f}). "
                   f"Final submission için seed ortalaması al.", pos=False)
    else:
        R.decision(f"Yüksek stabilite (std={seed_auc_std:.4f}). "
                   f"Model tekrarlanabilir. Tek seed yeterli.", pos=True)

    best_seed = int(df_seed.loc[df_seed["AUC"].idxmax(), "Seed"])
    R.log(f"📊 En yüksek AUC seed: {best_seed} — AUC={df_seed['AUC'].max():.5f}")
    R.decision(f"Önerilen seed: {best_seed} (en yüksek AUC). "
               f"Ama instabilite varsa 5 seed ortalaması kullan.", pos=True)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar([str(s) for s in seeds], df_seed["AUC"], color=BLUE, edgecolor=GRID)
    ax.errorbar([str(s) for s in seeds], df_seed["AUC"], yerr=df_seed["Std"],
                fmt="none", color=RED, capsize=4, lw=1.5)
    ax.axhline(df_seed["AUC"].mean(), color=ORANGE, linestyle="--", lw=1.5,
               label=f"Mean={df_seed['AUC'].mean():.4f}")
    dark_style(ax, "AUC Stability Across Seeds")
    ax.set_xlabel("Random Seed"); ax.set_ylabel("AUC")
    ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TXT)
    fig.tight_layout(); R.plot(fig, "seed_stability")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 76: SHAP DEPENDENCE PLOTS
    # ════════════════════════════════════════════════════
    R.section("🔍 76. SHAP DEPENDENCE PLOTS (Non-Linear Threshold Görselleştirme)")
    R.rule("SHAP dependence: bir feature'ın değeri arttıkça SHAP etkisi nasıl değişiyor?")
    R.rule("Düz çizgi → linear etki. S-eğrisi veya kırılma noktası → non-linear.")
    R.rule("Kırılma noktası = binning threshold. Orada binary feature oluştur.")
    R.rule("Renk kodlaması (interaction feature): SHAP'ın en çok hangi feature ile interact ettiğini gösterir.")

    try:
        tr_i0, va_i0 = list(skf5.split(X_enc, y_full))[0]
        explainer2  = shap.TreeExplainer(oof_models[0])
        shap_vals2  = explainer2.shap_values(X_enc.iloc[va_i0])
        if isinstance(shap_vals2, list): shap_vals2 = shap_vals2[1]

        top_dep = [c for c in shap_top5 if c in X_enc.columns][:4]
        n_plots = len(top_dep)
        if n_plots > 0:
            fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
            if n_plots == 1: axes = [axes]
            for i, col in enumerate(top_dep):
                col_idx  = list(X_enc.columns).index(col)
                shap_col = shap_vals2[:, col_idx]
                feat_col = X_enc.iloc[va_i0][col].values
                scatter  = axes[i].scatter(feat_col, shap_col,
                                            c=y_full.iloc[va_i0].values,
                                            cmap="RdYlGn_r", alpha=0.3, s=8)
                axes[i].axhline(0, color=GRID, linestyle="--", lw=1)
                dark_style(axes[i], f"SHAP Dependence: {col}")
                axes[i].set_xlabel(col); axes[i].set_ylabel("SHAP value")
            fig.set_facecolor(DARK); fig.tight_layout()
            R.plot(fig, "shap_dependence")
            R.decision("SHAP dependence plot'larda kırılma noktaları incelendi. "
                       "Non-linear threshold'ları PDP ile de doğrula.", pos=True)
    except Exception as e:
        R.log(f"⚠️ SHAP dependence hatası: {e}", cls="warn")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 77: FEATURE STABILITY ACROSS FOLDS (SHAP STD)
    # ════════════════════════════════════════════════════
    R.section("📊 77. FEATURE STABİLİTESİ ACROSS FOLDS (SHAP Std)")
    R.rule("SHAP importance her fold'da benzer mi? Yüksek std → feature instabil.")
    R.rule("SHAP std / mean > 0.5 → bu feature fold'dan fold'a çok değişiyor. Güvenilmez.")
    R.rule("Instabil feature'lar modeli overfit'e iter. Regularizasyon veya drop.")
    R.rule("Stabil + önemli → kesinlikle FE'ye dahil et.")

    try:
        fold_shap_means = []
        for fold_idx, (tr_i, va_i) in enumerate(skf5.split(X_enc, y_full)):
            exp = shap.TreeExplainer(oof_models[fold_idx])
            sv  = exp.shap_values(X_enc.iloc[va_i])
            if isinstance(sv, list): sv = sv[1]
            fold_shap_means.append(np.abs(sv).mean(axis=0))

        fold_shap_arr = np.array(fold_shap_means)
        shap_mean_arr = fold_shap_arr.mean(axis=0)
        shap_std_arr  = fold_shap_arr.std(axis=0)
        shap_cv_arr   = shap_std_arr / (shap_mean_arr + 1e-9)

        df_shap_stab = pd.DataFrame({
            "Feature":  feats,
            "SHAP_Mean":shap_mean_arr,
            "SHAP_Std": shap_std_arr,
            "CV(std/mean)": shap_cv_arr,
            "Stability": ["✅ Stabil" if cv<0.3 else
                           "🟡 Orta"  if cv<0.5 else
                           "🔴 Instabil" for cv in shap_cv_arr]
        }).sort_values("SHAP_Mean", ascending=False).reset_index(drop=True)

        R.table(df_shap_stab.style
                .background_gradient(cmap="YlGn",   subset=["SHAP_Mean"])
                .background_gradient(cmap="OrRd",   subset=["CV(std/mean)"])
                .format({"SHAP_Mean":"{:.4f}","SHAP_Std":"{:.4f}","CV(std/mean)":"{:.3f}"}))

        unstable = df_shap_stab[df_shap_stab["CV(std/mean)"]>0.5]["Feature"].tolist()
        stable_important = df_shap_stab[
            (df_shap_stab["CV(std/mean)"]<0.3) &
            (df_shap_stab["SHAP_Mean"] > df_shap_stab["SHAP_Mean"].median())
        ]["Feature"].tolist()

        if unstable:
            R.decision(f"Instabil feature'lar: {unstable}. "
                       f"Regularizasyon artır veya bu feature'ları daha dikkatli kullan.", pos=False)
        if stable_important:
            R.decision(f"Stabil ve önemli feature'lar: {stable_important}. "
                       f"FE'de öncelikli bunları kullan.", pos=True)
    except Exception as e:
        R.log(f"⚠️ SHAP stability hatası: {e}", cls="warn")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 78: HARD SAMPLE SHAP FORCE (Bireysel Hata Analizi)
    # ════════════════════════════════════════════════════
    R.section("🔎 78. HARD SAMPLE SHAP FORCE (Neden Yanlış Tahmin?)")
    R.rule("Hard sample'ların neden yanlış tahmin edildiğini SHAP ile açıkla.")
    R.rule("Sistematik bir özellik varsa → o feature için özel FE yap.")
    R.rule("Örnek: Tüm hard sample'lar tenure=0-3 ise → yeni müşteri segmenti özel FE istiyor.")
    R.rule("Bu analiz FE öncelik listesini netleştirir.")

    try:
        # Hard sample'ları belirle
        wrong_count_hs = 5 - (fold_preds > 0.5).astype(int).__eq__(
            y_full.values.reshape(-1,1)).sum(axis=1)
        is_hard_hs = wrong_count_hs >= 4

        if is_hard_hs.sum() >= 10:
            hard_idx   = np.where(is_hard_hs)[0][:min(200, is_hard_hs.sum())]
            hard_X     = X_enc.iloc[hard_idx]
            hard_y     = y_full.iloc[hard_idx]

            exp_hard   = shap.TreeExplainer(oof_models[-1])
            sv_hard    = exp_hard.shap_values(hard_X)
            if isinstance(sv_hard, list): sv_hard = sv_hard[1]

            hard_shap_mean = np.abs(sv_hard).mean(axis=0)
            df_hard_shap   = pd.DataFrame({
                "Feature":  feats,
                "SHAP_Hard":hard_shap_mean,
                "SHAP_All": np.abs(explainer2.shap_values(X_enc.iloc[va_i0])
                                   if not isinstance(explainer2.shap_values(X_enc.iloc[va_i0]), list)
                                   else explainer2.shap_values(X_enc.iloc[va_i0])[1]).mean(axis=0)
            })
            df_hard_shap["Hard_vs_All_ratio"] = (
                df_hard_shap["SHAP_Hard"] / (df_hard_shap["SHAP_All"] + 1e-9))
            df_hard_shap = df_hard_shap.sort_values("Hard_vs_All_ratio", ascending=False)

            R.table(df_hard_shap.style
                    .background_gradient(cmap="OrRd", subset=["Hard_vs_All_ratio"])
                    .format({"SHAP_Hard":"{:.4f}","SHAP_All":"{:.4f}",
                             "Hard_vs_All_ratio":"{:.3f}"}))

            dominant = df_hard_shap.head(3)["Feature"].tolist()
            R.log(f"📊 Hard sample'larda baskın feature'lar: {dominant}")
            R.decision(f"Hard sample'larda {dominant} feature'ları normal sample'lardan "
                       f"çok daha etkili. Bu feature'lar için özel FE veya interaction yap.", pos=True)
        else:
            R.log("📊 Yeterli hard sample yok (< 10). Bu section atlandı.", cls="warn")
    except Exception as e:
        R.log(f"⚠️ Hard sample SHAP hatası: {e}", cls="warn")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 79: BENFORD'S LAW (Sayısal Veri Doğrulama)
    # ════════════════════════════════════════════════════
    R.section("🔢 79. BENFORD'S LAW (Sayısal Veri Doğruluğu)")
    R.rule("Gerçek sayısal veriler Benford dağılımına uyar: ilk hane 1 → %30, 2 → %17, ...")
    R.rule("Sapma büyükse → veri manipülasyonu, sentetik oluşturma veya truncation sinyali.")
    R.rule("Kaggle synthetic verilerinde Benford'a uymama normal olabilir. Ama kontrol et.")
    R.rule("Chi-square p < 0.01 + MAD > 0.015 → ciddi sapma.")

    benford_expected = np.array([np.log10(1 + 1/d) for d in range(1, 10)])

    benford_res = []
    for col in num_cols:
        vals = train[col].dropna().abs()
        vals = vals[vals > 0]
        if len(vals) < 100: continue
        first_digits = vals.astype(str).str.replace(r'0\.0*', '', regex=True).str[0]
        first_digits = first_digits[first_digits.str.isdigit() &
                                    (first_digits != '0')]
        if len(first_digits) < 50: continue
        observed = np.array([
            (first_digits == str(d)).mean() for d in range(1, 10)])
        chi2_stat = len(first_digits) * np.sum(
            (observed - benford_expected)**2 / benford_expected)
        mad = np.mean(np.abs(observed - benford_expected))
        benford_res.append({
            "Feature": col, "Chi2": chi2_stat, "MAD": mad,
            "N": len(first_digits),
            "Conformity": ("✅ Uyuyor" if mad < 0.006 else
                           "🟡 Hafif sapma" if mad < 0.015 else "🔴 Sapıyor")
        })

    if benford_res:
        df_benf = pd.DataFrame(benford_res).sort_values("MAD", ascending=False)
        R.table(df_benf.style
                .background_gradient(cmap="RdYlGn_r", subset=["MAD"])
                .format({"Chi2":"{:.2f}","MAD":"{:.5f}"}))
        deviating = df_benf[df_benf["MAD"] > 0.015]["Feature"].tolist()
        if deviating:
            R.log(f"⚠️ Benford'dan sapan feature'lar: {deviating}", cls="warn")
            R.decision(f"Benford sapması: {deviating}. Sentetik veri yapısı veya encoding "
                       f"problemi olabilir. Domain kontrolü yap.", pos=False)
        else:
            R.log("✅ Sayısal veriler Benford dağılımına uyuyor.", cls="ok")
            R.decision("Benford testi geçildi. Sayısal veriler doğal görünüyor.", pos=True)
    else:
        R.log("📊 Benford analizi için yeterli sayısal veri yok.", cls="warn")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 80: MCNEMAR'S TEST (Model Karşılaştırma)
    # ════════════════════════════════════════════════════
    R.section("📊 80. McNEMAR'S TEST (İki Model Arasındaki Fark Anlamlı Mı?)")
    R.rule("FE öncesi ve sonrası model farkı istatistiksel olarak anlamlı mı?")
    R.rule("p < 0.05 → fark gerçek. p > 0.05 → fark şans eseri olabilir.")
    R.rule("AUC farkı 0.001 bile olsa McNemar ile anlamlı olup olmadığını kontrol et.")
    R.rule("Bu test 'bu FE adımı gerçekten fark yarattı mı?' sorusunu yanıtlar.")

    from scipy.stats import mcnemar
    # Baseline model (mevcut) vs Simplified model (daha az feature)
    # Feature önem sıralamasına göre alt yarı feature'lar çıkarılır
    n_half = max(1, len(feats)//2)
    top_feats = shap_df.head(n_half)["Feature"].tolist() if 'shap_df' in dir() else feats[:n_half]
    X_simplified = X_enc[top_feats]

    oof_simple = np.zeros(len(train))
    for tr_i, va_i in skf5.split(X_enc, y_full):
        m_sim = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                    is_unbalance=True, verbose=-1,
                                    n_jobs=-1, random_state=42)
        m_sim.fit(X_simplified.iloc[tr_i], y_full.iloc[tr_i])
        oof_simple[va_i] = m_sim.predict_proba(X_simplified.iloc[va_i])[:,1]

    auc_simple = roc_auc_score(y_full, oof_simple)

    # McNemar contingency table
    pred_full    = (oof_proba  >= opt_thr_cost).astype(int)
    pred_simple  = (oof_simple >= opt_thr_cost).astype(int)
    correct_full   = (pred_full   == y_full.values).astype(int)
    correct_simple = (pred_simple == y_full.values).astype(int)

    # b: full correct, simple wrong; c: full wrong, simple correct
    b = ((correct_full==1) & (correct_simple==0)).sum()
    c = ((correct_full==0) & (correct_simple==1)).sum()

    try:
        result_mc = mcnemar([[0, b],[c, 0]], exact=False, correction=True)
        mc_p = result_mc.pvalue
        R.log(f"📊 Full model AUC     : {baseline_auc:.5f}")
        R.log(f"📊 Simplified AUC     : {auc_simple:.5f}")
        R.log(f"📊 McNemar p-value    : {mc_p:.5f} (b={b}, c={c})")
        if mc_p < 0.05:
            R.decision(f"McNemar p={mc_p:.4f} — Modeller istatistiksel olarak farklı. "
                       f"Full model anlamlı şekilde daha iyi.", pos=True)
        else:
            R.decision(f"McNemar p={mc_p:.4f} — Fark istatistiksel değil. "
                       f"Daha az feature ile gidebilirsin (simpler model).", pos=False)
    except Exception as e:
        R.log(f"⚠️ McNemar hatası: {e}", cls="warn")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 63 (renumbered 81): NİHAİ KAPSAMLI HAZIRLIK SKORU
    # ════════════════════════════════════════════════════
    R.section("🏁 81. NİHAİ KAPSAMLI HAZIRLIK SKORU & MASTER ÖZET")

    score = 0; notes = []

    # Leakage
    if not leaks: score += 3
    else: notes.append(f"🚨 {len(leaks)} LEAK feature — KALDIR")

    # Calibration
    if brier < 0.10: score += 2
    elif brier < 0.20: score += 1; notes.append("⚠️ Kalibrasyon orta — isotonic regression dene")
    else: notes.append("🔴 Kötü kalibrasyon — Platt scaling zorunlu")

    # PSI
    serious_psi = [r["Feature"] for r in psi_res if r["PSI"]>0.25] if psi_res else []
    if not serious_psi: score += 2
    else: notes.append(f"🔴 PSI ciddi: {serious_psi}")

    # Hard samples
    if hard_rate < 8: score += 2
    elif hard_rate < 15: score += 1; notes.append(f"⚠️ Hard sample %{hard_rate:.1f}")
    else: notes.append(f"🔴 Yüksek hard sample %{hard_rate:.1f}")

    # Label noise
    if conflict_rate < 5: score += 2
    elif conflict_rate < 10: score += 1; notes.append("⚠️ Orta label noise")
    else: notes.append("🔴 Yüksek label noise")

    # Optimism bias
    if optimism < 0.005: score += 1
    else: notes.append(f"⚠️ Optimism bias {optimism:.4f}")

    # OOF vs Test dist
    if ks_oof_test < 0.05: score += 1
    else: notes.append(f"⚠️ OOF-Test KS={ks_oof_test:.4f}")

    # Adversarial validation
    if adv_auc < 0.65: score += 2
    elif adv_auc < 0.70: score += 1; notes.append(f"⚠️ Adversarial AUC={adv_auc:.4f}")
    else: notes.append(f"🔴 Ciddi covariate shift: adv_auc={adv_auc:.4f}")

    # Dummy lift
    if lift_over_dummy > 0.10: score += 1
    else: notes.append(f"🔴 Model dummy'yi yeterince geçmiyor: lift={lift_over_dummy:.4f}")

    # Permutation test
    if p_val_perm < 0.01: score += 1
    else: notes.append(f"🔴 Permutation p={p_val_perm:.4f} — model anlamlı değil")

    # Seed stability
    if seed_auc_std < 0.002: score += 1
    elif seed_auc_std < 0.005: pass
    else: notes.append(f"⚠️ Seed instabilitesi std={seed_auc_std:.4f}")

    max_score = 19
    R.log(f"\n{'═'*60}")
    R.log(f"📊 MASTER HAZIRLIK SKORU: {score}/{max_score}")  
    if score >= 16:
        R.log("🟢 MÜKEMMEL — FE'ye geç.", cls="ok")
        R.decision("Veri temiz, model güvenilir. FE reçetesini uygula.", pos=True)
    elif score >= 12:
        R.log("🟡 İYİ — Küçük düzeltmelerle geç.", cls="warn")
        R.decision("Aşağıdaki uyarıları gider, sonra FE'ye geç.", pos=False)
    elif score >= 7:
        R.log("🟠 ORTA — Önemli problemler var.", cls="warn")
        R.decision("Kritik sorunları çöz. FE'ye henüz geçme.", pos=False)
    else:
        R.log("🔴 DÜŞÜK — Temel problemler çözülmeden devam etme.", cls="critical")
        R.decision("Fundamental problemler var. Pipeline'ı düzelt.", pos=False)

    R.log(f"\n📋 KRİTİK NOTLAR ({len(notes)} adet):")
    for n in notes:
        cl = "critical" if "🚨" in n or "🔴" in n else "warn"
        R.log(f"  {n}", cls=cl)

    R.log("\n📊 ÖZET METRİKLER:")
    R.log(f"  Adversarial AUC    : {adv_auc:.4f}")
    R.log(f"  Dummy Lift         : {lift_over_dummy:+.4f}")
    R.log(f"  Permutation p-val  : {p_val_perm:.4f}")
    R.log(f"  Seed Stability std : {seed_auc_std:.5f}")
    R.log(f"  Baseline AUC       : {baseline_auc:.5f}")
    R.log(f"  Nested CV AUC      : {nested_auc:.5f} ± {nested_std:.5f}")
    R.log(f"  Brier Score        : {brier:.4f}")
    R.log(f"  Optimal Threshold  : {opt_thr_cost:.3f} (cost) / {opt_thr_f1:.3f} (F1)")
    R.log(f"  Top Decile Lift    : {top1_lift:.2f}x")
    R.log(f"  Hard Sample %      : {hard_rate:.2f}%")
    R.log(f"  PSI ciddi drift    : {serious_psi if serious_psi else 'Yok'}")
    R.log(f"  Max PSI            : {max(r['PSI'] for r in psi_res):.4f}" if psi_res else "  PSI: —")
    R.log(f"  Cohort Spread      : {cohort_spread:.1f} puan")
    R.log(f"  Conflict Rate      : {conflict_rate:.2f}%")
    R.log(f"  Optimism Bias      : {optimism:.5f}")
    R.log(f"  OOF-Test KS        : {ks_oof_test:.4f}")

    R.log("\n🔧 MASTER FE ÖNCELIK LİSTESİ:")
    priority_actions = [a for a in R.actions if any(
        kw in a for kw in ["oluştur","ekle","kullan","uygula","Ekle","Kullan"])]
    for i, a in enumerate(priority_actions[:15], 1):
        R.log(f"  {i:2d}. {a}", cls="ok")
    R.end()

    R.save("/kaggle/working/complete_extended_report.html")
    return {
        "baseline_auc":      baseline_auc,
        "nested_auc":        nested_auc,
        "brier":             brier,
        "opt_threshold":     opt_thr_cost,
        "top_lift":          top1_lift,
        "hard_rate":         hard_rate,
        "conflict_rate":     conflict_rate,
        "serious_psi":       serious_psi,
        "adversarial_auc":   adv_auc,
        "lift_over_dummy":   lift_over_dummy,
        "permutation_pval":  p_val_perm,
        "seed_auc_std":      seed_auc_std,
        "best_seed":         best_seed,
        "oof_proba":         oof_proba,
        "test_proba":        test_proba,
        "fold_preds":        fold_preds,
        "readiness_score":   score,
        "leaks":             leaks,
    }


# ═══════════════════════════════════════════════════════════════
# ÇALIŞTIR
# ═══════════════════════════════════════════════════════════════
# train  = pd.read_csv("/kaggle/input/playground-series-s6e3/train.csv")
# test   = pd.read_csv("/kaggle/input/playground-series-s6e3/test.csv")
#
# results = complete_extended_pipeline(
#     train, test,
#     target_col="Churn",
#     id_col="id",
#     fn_cost=5.0,   # kaçan churner maliyeti  (örn: 5x)
#     fp_cost=1.0    # gereksiz müdahale maliyeti
# )
