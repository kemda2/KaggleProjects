```python
# ════════════════════════════════════════════════════════════════════════════
# 🔬 EXTENDED SENIOR DATA PIPELINE — EK ANALİZLER (Sectionlar 26-34)
# Mevcut pipeline'ın sonuna, full_senior_data_pipeline() çağrısından ÖNCE
# veya ayrı bir hücre olarak çalıştırın.
# ════════════════════════════════════════════════════════════════════════════

!pip install tabulate shap -q

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os, warnings, base64
warnings.filterwarnings("ignore")

from scipy.stats import ks_2samp, spearmanr, chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
import lightgbm as lgb
from itertools import combinations

sns.set_theme(style="darkgrid", palette="muted")
PLOT_DIR = "/kaggle/working/plots_ext"
os.makedirs(PLOT_DIR, exist_ok=True)

DARK="#0d1117"; CARD="#161b22"; BLUE="#58a6ff"; TXT="#c9d1d9"; GRID="#30363d"

def dark_style(ax, title=""):
    ax.set_facecolor(CARD); ax.figure.set_facecolor(DARK)
    ax.title.set_color(BLUE); ax.xaxis.label.set_color(TXT); ax.yaxis.label.set_color(TXT)
    ax.tick_params(colors=TXT)
    for s in ax.spines.values(): s.set_color(GRID)
    if title: ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

def safe_encode(s):
    if s.dtype == "O":
        return pd.Series(LabelEncoder().fit_transform(s.astype(str)), index=s.index)
    return s

# ════════════════════════════════════════════════════════════════
# REPORTER (bağımsız — orijinal pipeline'dan ayrı kayıt)
# ════════════════════════════════════════════════════════════════
class ExtReporter:
    CSS = """<html><head><meta charset="utf-8">
    <style>
      body{font-family:'Segoe UI',Consolas,monospace;background:#0d1117;color:#c9d1d9;padding:30px;max-width:1400px;margin:auto}
      .section{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px;margin:25px 0}
      .section h2{color:#58a6ff;border-bottom:2px solid #58a6ff;padding-bottom:8px;font-size:20px}
      .info-line{background:#1f2937;padding:10px 15px;border-radius:6px;margin:8px 0;border-left:4px solid #58a6ff;font-size:14px}
      .warn{border-left-color:#f0883e;color:#f0883e}
      .ok{border-left-color:#3fb950;color:#3fb950}
      .critical{border-left-color:#f85149;color:#f85149;font-weight:bold}
      .decision{background:#1a2332;border:2px solid #58a6ff;border-radius:8px;padding:15px;margin:12px 0;font-size:13px}
      .decision-yes{border-color:#3fb950;background:#0d2818}
      .decision-no{border-color:#f85149;background:#2d1014}
      .section table{border-collapse:collapse;width:100%;margin:15px 0;font-size:13px}
      .section table th{background:#21262d;color:#58a6ff;padding:10px 14px;border:1px solid #30363d;text-align:left}
      .section table td{padding:8px 14px;border:1px solid #30363d;background:#0d1117}
      .section table tr:hover td{background:#1a2233}
      .plot-container{text-align:center;margin:15px 0}
      .plot-container img{max-width:100%;border-radius:8px;border:1px solid #30363d}
      .rule-box{background:#0d2818;border:1px solid #3fb950;border-radius:8px;padding:14px;margin:10px 0;font-size:13px;color:#c9d1d9}
      .rule-box b{color:#3fb950}
    </style></head><body>
    <h1 style="text-align:center;color:#58a6ff">🔬 EXTENDED PIPELINE REPORT — Sections 26-34</h1>"""

    def __init__(self):
        self.h = [self.CSS]; self.pc = 0; self.actions = []

    def log(self, text, cls=""):
        print(text)
        c = f" {cls}" if cls else ""
        self.h.append(f'<div class="info-line{c}">{text}</div>')

    def decision(self, text, pos=True):
        cn = "decision-yes" if pos else "decision-no"
        px = "✅" if pos else "⚠️"
        full = f"{px} KARAR: {text}"
        print(f"  {full}")
        self.h.append(f'<div class="decision {cn}">{full}</div>')
        self.actions.append(text)

    def rule(self, text):
        self.h.append(f'<div class="rule-box"><b>📌 KURAL:</b> {text}</div>')

    def section(self, title):
        print(f"\n{'═'*80}\n{title}\n{'═'*80}")
        self.h.append(f'<div class="section"><h2>{title}</h2>')

    def end(self): self.h.append("</div>")

    def table(self, styled):
        from IPython.display import display
        display(styled)
        self.h.append(styled.to_html())

    def plot(self, fig, title="plot"):
        from IPython.display import display, Image
        self.pc += 1
        fname = f"ext_{self.pc:02d}_{title.replace(' ','_').lower()}.png"
        fpath = os.path.join(PLOT_DIR, fname)
        fig.savefig(fpath, dpi=150, bbox_inches="tight", facecolor=DARK, edgecolor="none")
        plt.close(fig)
        display(Image(filename=fpath))
        with open(fpath,"rb") as f: b64 = base64.b64encode(f.read()).decode()
        self.h.append(f'<div class="plot-container"><img src="data:image/png;base64,{b64}" alt="{title}"></div>')

    def save(self, path):
        self.h.append("</body></html>")
        with open(path,"w",encoding="utf-8") as f: f.write("\n".join(self.h))
        print(f"\n✅ HTML kaydedildi: {path}")


# ════════════════════════════════════════════════════════════════
# ANA FONKSİYON
# ════════════════════════════════════════════════════════════════
def extended_pipeline(train, test, target_col, id_col=None, cat_threshold=25):
    R = ExtReporter()
    ext_actions = []

    feats = [c for c in train.columns if c not in [target_col, id_col]]
    num_cols = [c for c in feats if train[c].nunique() > cat_threshold]
    cat_cols = [c for c in feats if train[c].nunique() <= cat_threshold]
    y_full = safe_encode(train[target_col].copy())

    X_enc = train[feats].copy()
    for c in X_enc.select_dtypes(["object","category"]).columns:
        X_enc[c] = pd.factorize(X_enc[c])[0]
    X_enc = X_enc.fillna(-1)

    # ════════════════════════════════════════════════════
    # SECTION 26: LEAKAGE DETECTION
    # ════════════════════════════════════════════════════
    R.section("🚨 26. LEAKAGE DETECTION (Veri Sızıntısı)")
    R.rule("Tek bir feature ile AUC > 0.95 ise → bu feature hedef değişkeni tahmin ediyor demektir, production'da olmayacak bir bilgi taşıyor olabilir. KALDIR.")
    R.rule("Feature, target'ı mükemmel ayırıyorsa (AUC>0.99) → kesinlikle sızdırma. Hiç düşünmeden drop et.")
    R.rule("0.85-0.95 arası → şüpheli, domain bilgisiyle kontrol et.")

    leak_res = []
    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for col in feats:
        try:
            xc = X_enc[[col]]
            aucs = []
            for tr_i, va_i in skf5.split(xc, y_full):
                rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42, n_jobs=-1)
                rf.fit(xc.iloc[tr_i], y_full.iloc[tr_i])
                aucs.append(roc_auc_score(y_full.iloc[va_i], rf.predict_proba(xc.iloc[va_i])[:,1]))
            auc_m = np.mean(aucs)
            level = "🔴 LEAK!" if auc_m > 0.95 else ("🟠 ŞÜPHELİ" if auc_m > 0.85 else ("🟡 Güçlü" if auc_m > 0.75 else "✅ Normal"))
            leak_res.append({"Feature": col, "Solo_AUC": auc_m, "Risk": level})
        except:
            pass

    df_leak = pd.DataFrame(leak_res).sort_values("Solo_AUC", ascending=False).reset_index(drop=True)
    R.table(df_leak.style.background_gradient(cmap="RdYlGn_r", subset=["Solo_AUC"]).format({"Solo_AUC": "{:.4f}"}))

    leaks = df_leak[df_leak["Solo_AUC"] > 0.95]["Feature"].tolist()
    suspects = df_leak[(df_leak["Solo_AUC"] > 0.85) & (df_leak["Solo_AUC"] <= 0.95)]["Feature"].tolist()

    if leaks:
        R.log(f"🔴 LEAK TESPİT EDİLDİ: {leaks}", cls="critical")
        R.decision(f"LEAK özellikleri KALDIR: {leaks}. Model eğitiminde KULLANMA!", pos=False)
        ext_actions.append(f"🚨 LEAK DROP: {leaks}")
    else:
        R.log("✅ Belirgin leak yok.", cls="ok")

    if suspects:
        R.log(f"🟠 Şüpheli yüksek solo-AUC: {suspects}", cls="warn")
        R.decision(f"Domain kontrolü: {suspects} → production'da bu bilgiler var mı?", pos=False)
        ext_actions.append(f"⚠️ Leak şüphelisi kontrol et: {suspects}")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 27: LABEL QUALITY / TARGET NOISE
    # ════════════════════════════════════════════════════
    R.section("🏷️ 27. LABEL KALİTESİ & TARGET NOISE")
    R.rule("Aynı feature profiline sahip satırlarda farklı target → label noise. %5'ten fazlası sorunlu.")
    R.rule("Conflict rate > %10 ise → ağırlıklı kayıp fonksiyonu (sample_weight) veya noise-robust loss kullan.")
    R.rule("Conflict rate < %5 ise → normal devam et.")

    dup_check = train[feats + [target_col]].copy()
    for c in dup_check.select_dtypes(["object","category"]).columns:
        dup_check[c] = dup_check[c].astype(str)

    key_cols = feats[:min(6, len(feats))]
    grp = dup_check.groupby(key_cols)[target_col].agg(["nunique","count"])
    conflicted = grp[grp["nunique"] > 1]
    conflict_count = conflicted["count"].sum()
    conflict_rate = conflict_count / len(train) * 100

    R.log(f"📊 Conflict satır sayısı: {conflict_count:,} ({conflict_rate:.2f}%)")
    if conflict_rate > 10:
        R.log(f"🔴 Yüksek conflict oranı: %{conflict_rate:.2f}", cls="critical")
        R.decision("sample_weight veya noise-robust loss kullan. Label kalitesi düşük.", pos=False)
        ext_actions.append("⚠️ Label noise yüksek → sample_weight")
    elif conflict_rate > 5:
        R.log(f"🟠 Orta conflict: %{conflict_rate:.2f}", cls="warn")
        R.decision("LightGBM'de label_smoothing=0.05 dene.", pos=False)
        ext_actions.append("⚠️ Orta label noise → label_smoothing")
    else:
        R.log(f"✅ Düşük conflict: %{conflict_rate:.2f}", cls="ok")
        R.decision("Label kalitesi iyi. Normal devam et.", pos=True)

    # Near-duplicate profiling (tüm feature'larla)
    full_dup = train[feats].copy()
    for c in full_dup.select_dtypes(["object","category"]).columns:
        full_dup[c] = full_dup[c].astype(str)
    exact_dups = full_dup.duplicated(keep=False).sum()
    R.log(f"📊 Birebir feature duplicate (farklı target olabilir): {exact_dups:,}")
    if exact_dups > 0:
        exact_conflict = train[full_dup.duplicated(keep=False)].groupby(feats)[target_col].nunique()
        n_conflict = (exact_conflict > 1).sum()
        R.log(f"   → Bunların {n_conflict} grubunda farklı target var")
        if n_conflict > 0:
            R.decision(f"{n_conflict} grup birebir aynı feature ama farklı target. drop_duplicates YAPMA, ağırlıklandır.", pos=False)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 28: BUSINESS LOGIC VALIDATION
    # ════════════════════════════════════════════════════
    R.section("🏢 28. BUSINESS LOGIC VALIDATION")
    R.rule("TotalCharges ≈ tenure × MonthlyCharges olmalı. Sapma > %50 ise veri problemi.")
    R.rule("tenure=0 ama TotalCharges>0 → imkansız. Bu satırları işaretle veya düzelt.")
    R.rule("tenure=0 ve TotalCharges=0 → yeni müşteri, normal.")
    R.rule("MonthlyCharges çok yüksek ama hiç servisi yok → anomali, incelemeye değer.")
    R.rule("Hatalı satırları DROP etme, flag ekle (is_anomaly). Model bu bilgiyi kullanabilir.")

    bl_issues = []

    # TotalCharges consistency
    if all(c in train.columns for c in ["TotalCharges","tenure","MonthlyCharges"]):
        tc = pd.to_numeric(train["TotalCharges"], errors="coerce")
        ten = pd.to_numeric(train["tenure"], errors="coerce")
        mc = pd.to_numeric(train["MonthlyCharges"], errors="coerce")

        expected = ten * mc
        ratio = tc / (expected + 1e-9)
        bad_ratio = ((ratio < 0.5) | (ratio > 2.0)) & (ten > 0)
        n_bad = bad_ratio.sum()
        R.log(f"📊 TotalCharges/expected oranı bozuk satır: {n_bad:,} ({n_bad/len(train)*100:.2f}%)")
        bl_issues.append(("TotalCharges_ratio_bad", n_bad))

        if n_bad > 0:
            R.decision(f"{n_bad} satırda TotalCharges anomalisi. 'is_charges_anomaly' flag oluştur.", pos=False)
            ext_actions.append("🏷️ is_charges_anomaly flag ekle")

            # Anomali satırların churn rate'i
            churn_anom = y_full[bad_ratio].mean() * 100
            churn_norm = y_full[~bad_ratio].mean() * 100
            R.log(f"   Anomali satır churn rate: %{churn_anom:.2f} vs Normal: %{churn_norm:.2f}")
            if abs(churn_anom - churn_norm) > 10:
                R.decision("Anomali satırlar farklı churn pattern'ı var. FLAG güçlü feature olacak!", pos=True)

        # tenure=0 kontrol
        zero_ten_pos_tc = ((ten == 0) & (tc > 0)).sum()
        if zero_ten_pos_tc > 0:
            R.log(f"🔴 tenure=0 ama TotalCharges>0: {zero_ten_pos_tc} satır", cls="critical")
            R.decision("Bu satırlarda TotalCharges'ı 0 yap veya is_anomaly flag ekle.", pos=False)
            ext_actions.append(f"🔧 tenure=0/TC>0 düzelt: {zero_ten_pos_tc} satır")
        else:
            R.log("✅ tenure=0 & TotalCharges>0 anomalisi yok.", cls="ok")

        # charges residual dağılımı
        residual = tc - expected
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(residual.dropna(), bins=60, color=BLUE, edgecolor=GRID, alpha=0.8)
        dark_style(axes[0], "Charges Residual (TC - tenure×MC)")
        axes[1].scatter(expected.sample(min(1000, len(expected)), random_state=42),
                        tc.sample(min(1000, len(tc)), random_state=42),
                        alpha=0.3, color=BLUE, s=10)
        x_line = np.linspace(0, expected.max(), 100)
        axes[1].plot(x_line, x_line, color="#f85149", lw=2, label="Perfect")
        dark_style(axes[1], "Expected vs Actual TotalCharges")
        axes[1].legend(facecolor=CARD, edgecolor=GRID, labelcolor=TXT)
        fig.set_facecolor(DARK); fig.tight_layout()
        R.plot(fig, "charges_residual")
    else:
        R.log("⚠️ TotalCharges/tenure/MonthlyCharges kolonları bulunamadı.", cls="warn")

    # Sentinel değer kontrol (genel)
    sentinel_vals = ["No internet service", "No phone service", "Unknown", "N/A", "na", "none"]
    sent_counts = {}
    for c in cat_cols:
        for sv in sentinel_vals:
            cnt = (train[c] == sv).sum()
            if cnt > 0:
                sent_counts[f"{c}='{sv}'"] = cnt
    if sent_counts:
        R.log(f"⚠️ Sentinel değerler tespit edildi:", cls="warn")
        for k, v in sorted(sent_counts.items(), key=lambda x: -x[1]):
            R.log(f"   {k}: {v:,} satır")
        R.decision("Tüm sentinel değerleri 'No' ile değiştir (FE adım 1).", pos=True)
        ext_actions.append("🔧 Sentinel → 'No' dönüşümü")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 29: TEMPORAL / COHORT DRIFT ANALİZİ
    # ════════════════════════════════════════════════════
    R.section("⏳ 29. COHORT DRIFT ANALİZİ (tenure bazlı)")
    R.rule("Erken cohort (tenure<12) vs geç cohort (tenure>48) arasında churn rate büyük farklıysa → cohort-based feature gereklidir.")
    R.rule("Churn rate farkı > 15 puan → tenure grupları bazında ayrı özellik mühendisliği yap.")
    R.rule("KS drift erken vs geç cohort arasında yüksekse → model farklı dönemlerde genelleşemiyor demektir.")

    if "tenure" in train.columns:
        tmp_c = train.copy()
        tmp_c["tbin"] = y_full
        tmp_c["tenure_num"] = pd.to_numeric(tmp_c["tenure"], errors="coerce")
        tmp_c["cohort"] = pd.cut(tmp_c["tenure_num"],
                                  bins=[0, 12, 24, 36, 48, 100],
                                  labels=["0-12","12-24","24-36","36-48","48+"],
                                  right=True)
        cohort_stats = tmp_c.groupby("cohort")["tbin"].agg(["mean","count"])
        cohort_stats.columns = ["Churn_Rate%","Count"]
        cohort_stats["Churn_Rate%"] *= 100

        R.table(cohort_stats.style.background_gradient(cmap="RdYlGn_r", subset=["Churn_Rate%"]).format({"Churn_Rate%":"{:.2f}"}))

        rates = cohort_stats["Churn_Rate%"].values
        cohort_range = rates.max() - rates.min()
        rho_c, _ = spearmanr(range(len(rates)), rates)

        R.log(f"📊 Cohort churn spread: {cohort_range:.1f} puan | Spearman: {rho_c:.3f}")

        if cohort_range > 20:
            R.decision(f"Cohort farkı {cohort_range:.1f} puan — BÜYÜK. tenure_group kategorik feature oluştur.", pos=True)
            ext_actions.append("🎯 tenure_group kategorik feature")
        elif cohort_range > 10:
            R.decision(f"Cohort farkı {cohort_range:.1f} puan — orta. tenure_bin ekle.", pos=True)
            ext_actions.append("🎯 tenure_bin feature")
        else:
            R.decision(f"Cohort farkı düşük ({cohort_range:.1f}p). Lineer tenure yeterli.", pos=True)

        # Her cohort'ta numeric feature dağılımı kayıyor mu?
        if "MonthlyCharges" in train.columns:
            mc_cohorts = []
            for coh, grp in tmp_c.groupby("cohort"):
                mc_cohorts.append({"Cohort": str(coh), "MC_Mean": grp["MonthlyCharges"].mean(), "MC_Std": grp["MonthlyCharges"].std()})
            df_coh = pd.DataFrame(mc_cohorts)
            R.table(df_coh.style.background_gradient(cmap="Blues", subset=["MC_Mean"]).format(precision=2))

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        cohort_stats["Churn_Rate%"].plot(kind="bar", ax=axes[0], color=[
            "#f85149" if r > 30 else "#f0883e" if r > 20 else "#3fb950" for r in cohort_stats["Churn_Rate%"]], edgecolor=GRID)
        dark_style(axes[0], "Churn Rate by Tenure Cohort")
        axes[0].set_ylabel("Churn Rate (%)")

        cohort_stats["Count"].plot(kind="bar", ax=axes[1], color=BLUE, edgecolor=GRID)
        dark_style(axes[1], "Sample Count by Tenure Cohort")
        fig.set_facecolor(DARK); fig.tight_layout()
        R.plot(fig, "cohort_drift")
    else:
        R.log("⚠️ tenure kolonu bulunamadı.", cls="warn")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 30: STRATIFIED TARGET EDA (KDE)
    # ════════════════════════════════════════════════════
    R.section("📊 30. STRATİFİED TARGET EDA (Churn=Yes vs No dağılımları)")
    R.rule("Churn=Yes ve Churn=No dağılımları büyük ölçüde örtüşüyorsa → o feature zayıf ayrıştırıcıdır.")
    R.rule("KS istatistiği yüksek (>0.3) → feature güçlü ayrıştırıcıdır, FE'de önceliklendir.")
    R.rule("KS < 0.05 → feature neredeyse işe yaramıyor, düşür veya transform et.")

    if num_cols:
        y0 = train[y_full == 0]; y1 = train[y_full == 1]
        strat_res = []
        for col in num_cols:
            a, b = y0[col].dropna(), y1[col].dropna()
            if len(a) > 5 and len(b) > 5:
                ks_val, ks_p = ks_2samp(a, b)
                mean_diff = b.mean() - a.mean()
                strat_res.append({"Feature": col, "KS": ks_val, "P_val": ks_p,
                                   "Mean_No": a.mean(), "Mean_Yes": b.mean(),
                                   "Mean_Diff": mean_diff,
                                   "Power": "🔴 Güçlü" if ks_val > 0.3 else ("🟡 Orta" if ks_val > 0.15 else "⚪ Zayıf")})

        df_strat = pd.DataFrame(strat_res).sort_values("KS", ascending=False).reset_index(drop=True)
        R.table(df_strat.style.background_gradient(cmap="YlGn", subset=["KS"]).format(precision=4))

        top_feats = df_strat[df_strat["KS"] > 0.15]["Feature"].tolist()[:6]
        weak_feats = df_strat[df_strat["KS"] < 0.05]["Feature"].tolist()

        if weak_feats:
            R.log(f"⚪ Çok zayıf ayrıştırıcılar: {weak_feats}", cls="warn")
            R.decision(f"Bu feature'lar düşürülmeli veya transform edilmeli: {weak_feats}", pos=False)
            ext_actions.append(f"⚠️ Zayıf KDE feature'lar: {weak_feats}")

        # KDE plotlar
        nc = min(3, len(top_feats)); nr = (len(top_feats) + nc - 1) // nc if top_feats else 1
        if top_feats:
            fig, axes = plt.subplots(nr, nc, figsize=(15, nr * 4))
            axes_flat = np.array(axes).flatten() if len(top_feats) > 1 else [axes]
            for i, col in enumerate(top_feats):
                a_d, b_d = y0[col].dropna(), y1[col].dropna()
                axes_flat[i].hist(a_d, bins=40, alpha=0.5, color="#3fb950", label="Churn=No", density=True, edgecolor=GRID)
                axes_flat[i].hist(b_d, bins=40, alpha=0.5, color="#f85149", label="Churn=Yes", density=True, edgecolor=GRID)
                dark_style(axes_flat[i], col)
                axes_flat[i].legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TXT)
            for j in range(len(top_feats), len(axes_flat)):
                axes_flat[j].set_visible(False)
            fig.suptitle("Numeric Features — Churn=Yes vs No", color=BLUE, fontsize=16, fontweight="bold")
            fig.set_facecolor(DARK); fig.tight_layout(rect=[0,0,1,0.95])
            R.plot(fig, "stratified_kde")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 31: SHAP INTERACTION (Top-5 feature)
    # ════════════════════════════════════════════════════
    R.section("🔬 31. SHAP DEĞERLERİ & İNTERACTION ANALİZİ")
    R.rule("SHAP mean abs değeri > 0.05 → feature modelde önemli etki yapıyor.")
    R.rule("SHAP değeri scatter'ında V-şekli veya renk gradyanı → non-linear ilişki var, binning/transform yardımcı olur.")
    R.rule("İki feature arasında SHAP interaction > 0.02 → bu ikili için çarpım feature'ı oluştur.")
    R.rule("SHAP negatif olan satırlar çoksa → bu feature bazı segmentlerde ters etki yapıyor, segment bazlı FE düşün.")

    try:
        skf_shap = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        tr_i, va_i = list(skf_shap.split(X_enc, y_full))[0]
        m_shap = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                                     num_leaves=31, is_unbalance=True, random_state=42, verbose=-1, n_jobs=-1)
        m_shap.fit(X_enc.iloc[tr_i], y_full.iloc[tr_i])

        explainer = shap.TreeExplainer(m_shap)
        shap_vals = explainer.shap_values(X_enc.iloc[va_i])
        if isinstance(shap_vals, list): shap_vals = shap_vals[1]

        shap_mean = np.abs(shap_vals).mean(axis=0)
        shap_df = pd.DataFrame({"Feature": feats, "SHAP_Mean": shap_mean}).sort_values("SHAP_Mean", ascending=False).reset_index(drop=True)
        R.table(shap_df.style.bar(subset=["SHAP_Mean"], color="#636efa").format({"SHAP_Mean": "{:.4f}"}))

        top5 = shap_df.head(5)["Feature"].tolist()
        R.log(f"🏆 Top 5 SHAP feature: {top5}")

        # SHAP beeswarm / summary plot
        fig, ax = plt.subplots(figsize=(10, max(5, len(feats) * 0.35)))
        shap.summary_plot(shap_vals, X_enc.iloc[va_i], feature_names=feats,
                          plot_type="bar", show=False, color=BLUE)
        ax = plt.gca(); dark_style(ax, "SHAP Feature Importance")
        fig.set_facecolor(DARK); fig.tight_layout()
        R.plot(fig, "shap_summary")

        # Top-3 için scatter
        top3 = shap_df.head(3)["Feature"].tolist()
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
        for i, col in enumerate(top3):
            idx = feats.index(col)
            axes2[i].scatter(X_enc.iloc[va_i][col], shap_vals[:, idx],
                             alpha=0.3, c=shap_vals[:, idx], cmap="RdBu_r", s=8)
            dark_style(axes2[i], f"SHAP: {col}")
            axes2[i].axhline(0, color=GRID, lw=1, linestyle="--")
            axes2[i].set_xlabel(col); axes2[i].set_ylabel("SHAP value")
        fig2.set_facecolor(DARK); fig2.tight_layout()
        R.plot(fig2, "shap_scatter_top3")

        # Interaction matrix (top-5 arası)
        R.log("📊 SHAP Interaction proxy (korelasyon bazlı):")
        shap_top5 = pd.DataFrame(shap_vals[:, [feats.index(f) for f in top5]], columns=top5)
        inter_corr = shap_top5.corr()
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        sns.heatmap(inter_corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    ax=ax3, linewidths=0.5, linecolor=GRID)
        dark_style(ax3, "SHAP Interaction (Top-5)")
        fig3.tight_layout()
        R.plot(fig3, "shap_interaction")

        strong_inter = [(top5[i], top5[j]) for i in range(len(top5)) for j in range(i+1,len(top5)) if abs(inter_corr.iloc[i,j]) > 0.3]
        if strong_inter:
            R.log(f"⚡ Güçlü SHAP interaksiyon çiftleri: {strong_inter}", cls="ok")
            R.decision(f"Bu çiftler için çarpım feature oluştur: {strong_inter}", pos=True)
            ext_actions.append(f"🎯 SHAP interaction pairs FE: {strong_inter}")

        # Non-linear kontrol
        nonlinear = []
        for col in top5:
            idx = feats.index(col)
            corr_lin = abs(np.corrcoef(X_enc.iloc[va_i][col].fillna(-1), shap_vals[:, idx])[0, 1])
            if corr_lin < 0.7:
                nonlinear.append(col)
        if nonlinear:
            R.log(f"🔀 Non-linear SHAP pattern: {nonlinear}", cls="warn")
            R.decision(f"Bu feature'lar için binning veya log transform dene: {nonlinear}", pos=False)
            ext_actions.append(f"🔧 Non-linear → binning/log: {nonlinear}")

    except Exception as e:
        R.log(f"⚠️ SHAP hesaplanamadı: {e}", cls="warn")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 32: HARD SAMPLE ANALİZİ (CV tutarsız satırlar)
    # ════════════════════════════════════════════════════
    R.section("🎯 32. HARD SAMPLE ANALİZİ (Model için zor satırlar)")
    R.rule("5 fold'un 4+'ında yanlış tahmin edilen satırlar → 'hard samples'. Bunlar:")
    R.rule("  a) Gerçek belirsizlik (label noise) → kaldır veya ağırlığını azalt")
    R.rule("  b) Önemli bir pattern → özel FE ile yakala")
    R.rule("Hard sample oranı > %15 ise → model yetersiz, daha güçlü FE gerekli")
    R.rule("Hard sample'ların feature profili normal dağılımdan sapıyorsa → segment-specific FE yap")

    fold_preds = np.zeros((len(train), 5))
    skf_hard = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_i, va_i) in enumerate(skf_hard.split(X_enc, y_full)):
        m_h = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                                   is_unbalance=True, random_state=42, verbose=-1, n_jobs=-1)
        m_h.fit(X_enc.iloc[tr_i], y_full.iloc[tr_i],
                eval_set=[(X_enc.iloc[va_i], y_full.iloc[va_i])],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        fold_preds[va_i, fold] = m_h.predict_proba(X_enc.iloc[va_i])[:, 1]

    pred_labels = (fold_preds > 0.5).astype(int)
    correct = (pred_labels == y_full.values.reshape(-1, 1)).sum(axis=1)
    wrong_count = 5 - correct
    is_hard = wrong_count >= 4

    hard_rate = is_hard.mean() * 100
    R.log(f"📊 Hard sample sayısı: {is_hard.sum():,} ({hard_rate:.2f}%)")

    if hard_rate > 15:
        R.log(f"🔴 Hard sample oranı yüksek: %{hard_rate:.2f}", cls="critical")
        R.decision("Daha agresif FE gerekiyor. Hard sample'ların feature profilini incele.", pos=False)
        ext_actions.append(f"⚠️ Hard sample %{hard_rate:.1f} → güçlü FE gerekli")
    elif hard_rate > 8:
        R.log(f"🟠 Orta hard sample: %{hard_rate:.2f}", cls="warn")
        R.decision("Target encoding veya stacking bu satırlar için yardımcı olabilir.", pos=False)
    else:
        R.log(f"✅ Düşük hard sample: %{hard_rate:.2f}", cls="ok")
        R.decision("Model sağlıklı. Hard sample'lar manageable.", pos=True)

    # Hard sample profili
    if is_hard.sum() > 10:
        hard_df = train[is_hard].copy()
        normal_df = train[~is_hard].copy()
        R.log("\n📊 Hard vs Normal — Numeric Mean Karşılaştırma:")
        if num_cols:
            comp_rows = []
            for col in num_cols:
                hm = hard_df[col].mean(); nm = normal_df[col].mean()
                comp_rows.append({"Feature": col, "Hard_Mean": hm, "Normal_Mean": nm,
                                   "Diff%": abs(hm - nm) / (abs(nm) + 1e-9) * 100})
            df_comp = pd.DataFrame(comp_rows).sort_values("Diff%", ascending=False)
            R.table(df_comp.style.background_gradient(cmap="OrRd", subset=["Diff%"]).format(precision=3))
            big_diff = df_comp[df_comp["Diff%"] > 20]["Feature"].tolist()
            if big_diff:
                R.decision(f"Hard sample'lar bu feature'larda çok farklı: {big_diff}. Bu feature'lar için özel transform uygula.", pos=True)
                ext_actions.append(f"🎯 Hard sample odaklı FE: {big_diff}")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 33: TEST SET OOD (Out-of-Distribution) ANALİZİ
    # ════════════════════════════════════════════════════
    R.section("🔭 33. TEST SET OOD (Out-of-Distribution) ANALİZİ")
    R.rule("Test'te train'de hiç görülmemiş feature kombinasyonları varsa → model bu bölgede extrapolate ediyor.")
    R.rule("OOD satır oranı > %20 ise → test'e özel FE veya ensemble gereklidir.")
    R.rule("Test'te yeni kategorik değerler varsa → unknown token ekle veya target encoding kullan.")
    R.rule("Numeric OOD (train range dışı) → clip veya Winsorize uygula.")

    ood_results = []

    # Kategorik: yeni değerler
    for col in cat_cols:
        train_vals = set(train[col].dropna().unique())
        test_vals = set(test[col].dropna().unique())
        new_in_test = test_vals - train_vals
        if new_in_test:
            cnt = test[col].isin(new_in_test).sum()
            ood_results.append({"Col": col, "Type": "Yeni Kategori", "Count": cnt, "Values": str(list(new_in_test)[:5])})

    # Numeric: range dışı
    for col in num_cols:
        tr_min, tr_max = train[col].min(), train[col].max()
        out_of_range = ((test[col] < tr_min) | (test[col] > tr_max)).sum()
        if out_of_range > 0:
            ood_results.append({"Col": col, "Type": "Range Dışı", "Count": out_of_range,
                                 "Values": f"Train:[{tr_min:.1f},{tr_max:.1f}] Test_min={test[col].min():.1f} max={test[col].max():.1f}"})

    if ood_results:
        df_ood = pd.DataFrame(ood_results)
        R.table(df_ood.style.background_gradient(cmap="OrRd", subset=["Count"]))
        total_ood = sum(r["Count"] for r in ood_results)
        ood_rate = total_ood / len(test) * 100
        R.log(f"📊 OOD toplam: {total_ood:,} ({ood_rate:.1f}% test satırı)")

        cat_ood = [r["Col"] for r in ood_results if r["Type"] == "Yeni Kategori"]
        num_ood = [r["Col"] for r in ood_results if r["Type"] == "Range Dışı"]

        if cat_ood:
            R.decision(f"Yeni kategoriler: {cat_ood} → Target Encoding veya 'Unknown' token ekle.", pos=False)
            ext_actions.append(f"🔧 OOD kategoriler: {cat_ood} → target encoding")
        if num_ood:
            R.decision(f"Range dışı numeric: {num_ood} → Winsorize (clip at 1st-99th percentile).", pos=False)
            ext_actions.append(f"🔧 OOD numeric: {num_ood} → Winsorize")
    else:
        R.log("✅ Test'te OOD değer yok.", cls="ok")
        R.decision("Test ve train aynı değer aralığında. Güvenli.", pos=True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 34: FE ÖNCESİ NİHAİ HAZIRLIK SKORU
    # ════════════════════════════════════════════════════
    R.section("🏁 34. FE ÖNCESİ NİHAİ HAZIRLIK SKORU & ÖZET")
    R.rule("Bu bölüm tüm testlerin çıktısını birleştirir ve FE'ye geçiş kararı verir.")

    readiness_score = 0
    readiness_notes = []

    # Leakage
    if not leaks:
        readiness_score += 2
    else:
        readiness_notes.append(f"🚨 {len(leaks)} leak feature var — FE ÖNCESİ KALDIR")

    # Label noise
    if conflict_rate < 5:
        readiness_score += 2
    elif conflict_rate < 10:
        readiness_score += 1
        readiness_notes.append("⚠️ Orta label noise — label_smoothing ekle")
    else:
        readiness_notes.append("🔴 Yüksek label noise — sample_weight zorunlu")

    # Business logic
    if "TotalCharges" in train.columns:
        if n_bad < len(train) * 0.02:
            readiness_score += 1
        else:
            readiness_notes.append("⚠️ Business logic anomalileri yüksek — flag ekle")

    # OOD
    if not ood_results:
        readiness_score += 2
    elif len(ood_results) < 3:
        readiness_score += 1
        readiness_notes.append("⚠️ Küçük OOD problemi — Winsorize/unknown token")
    else:
        readiness_notes.append("🔴 Büyük OOD problemi — target encoding şart")

    # Hard samples
    if hard_rate < 8:
        readiness_score += 2
    elif hard_rate < 15:
        readiness_score += 1
    else:
        readiness_notes.append("🔴 Yüksek hard sample — agresif FE gerekli")

    R.log(f"\n📊 HAZIRLIK SKORU: {readiness_score}/9")
    if readiness_score >= 7:
        R.log("🟢 YÜKSELİŞ HAZIR", cls="ok")
        R.decision("Veri temiz, FE'ye geç. Aşağıdaki sırayla ilerle.", pos=True)
    elif readiness_score >= 5:
        R.log("🟡 ORTA HAZIRLIK", cls="warn")
        R.decision("Kritik düzeltmeleri yap, sonra FE'ye geç.", pos=False)
    else:
        R.log("🔴 DÜŞÜK HAZIRLIK", cls="critical")
        R.decision("Temel problemleri çöz. FE'ye geçme.", pos=False)

    R.log("\n📋 KRİTİK NOTLAR:")
    for n in readiness_notes:
        R.log(f"  {n}", cls="critical" if "🚨" in n else "warn")

    R.log("\n📋 TÜM EKSTENSİYON AKSİYONLARI:")
    for i, a in enumerate(ext_actions, 1):
        cl = "critical" if "🚨" in a else ("warn" if "⚠" in a else "ok")
        R.log(f"  {i}. {a}", cls=cl)

    R.end()
    R.save("/kaggle/working/extended_pipeline_report.html")
    return ext_actions, fold_preds


# ═══════════════════════════════════════════════════════════
# ÇALIŞTIR (orijinal pipeline'dan dönen train/test/target ile)
# ═══════════════════════════════════════════════════════════
# Önceki hücrede full_senior_data_pipeline() çalıştırıldıktan sonra:
#
#   ext_actions, fold_preds = extended_pipeline(
#       train, test, target_col, id_col
#   )
#
# Tek başına da çalışır:
#
#   train = pd.read_csv("/kaggle/input/playground-series-s6e3/train.csv")
#   test  = pd.read_csv("/kaggle/input/playground-series-s6e3/test.csv")
#   ext_actions, fold_preds = extended_pipeline(train, test, "Churn", "id")
```

# 🚀 Senior Data Pipeline — FE Öncesi Analiz Rehberi

**Testler · Kurallar · Aksiyonlar · FE Hazırlık Kılavuzu**

Bu döküman 34 bölümlük pipeline'ın (Section 1–34) çıktılarını **eğer X ise → Y yap** formatında özetler. FE'ye geçmeden önce her section'ı sırasıyla uygulayın.

---

## İçindekiler

1. [Temel Prensipler](#1-temel-prensipler)
2. [Section Bazlı Karar Kuralları (S1–S24)](#2-section-bazlı-karar-kuralları)
3. [Ek Analizler Section 26–34](#3-ek-analizler--section-2634-karar-kuralları)
4. [Sırayla Uygulanacak FE Reçetesi](#4-sırayla-uygulanacak-fe-reçetesi)
5. [Hızlı Karar Referans Tablosu](#5-hızlı-karar-referans-tablosu)
6. [Telco Churn Dataset — Özel Notlar](#6-telco-churn-dataset--özel-notlar)

---

## 1. Temel Prensipler

> **Altın Kural:** Feature Engineering'e geçmeden önce aşağıdaki 9 kategori temizlenmeden hiçbir model güvenilir değildir.

| # | Kural |
|---|-------|
| **KURAL 1** | Leak varsa FE yapma. Önce kaldır. AUC > 0.95 solo feature → her zaman şüpheli. |
| **KURAL 2** | Business logic anomalilerini DROP etme, FLAG ekle. Model bu bilgiyi kullanabilir. |
| **KURAL 3** | Sentinel değerleri (`No internet service`, `No phone service`) her zaman `"No"` ile değiştir. |
| **KURAL 4** | Test'te OOD değer varsa → Winsorize (numeric) veya Unknown token (kategorik). |
| **KURAL 5** | Hard sample > %15 ise agresif FE gerekli. Hard sample < %8 ise temel FE yeterli. |
| **KURAL 6** | SHAP interaksiyon çifti varsa → o ikili için çarpım feature oluştur. |
| **KURAL 7** | CV std > 0.005 ise FE'ye geçme, önce CV stratejisini düzelt. |
| **KURAL 8** | Label noise > %10 ise `sample_weight` veya `label_smoothing` şart. |
| **KURAL 9** | Her FE adımından sonra baseline AUC ile karşılaştır. Düşerse geri al. |

---

## 2. Section Bazlı Karar Kuralları

### Section 1–2: Memory, Shape & Genel Audit

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Train > 500MB | dtypes'ı downcast et (float32, int32). `pd.read_csv` ile `dtype=dict` kullan. | 🟡 ORTA |
| Test'te train'de olmayan kolon var | Bu kolonu test'ten de kaldır. Feature alignment yap. | 🔴 KRİTİK |
| Null Pct > %30 (herhangi bir kolon) | `is_null` flag ekle, sonra impute et. Doğrudan drop etme. | 🟡 ORTA |
| Null Gap (test - train) > 5 puan | Bu kolon drift'li. Adversarial test section'ına bak. | 🟡 ORTA |
| Top Val Pct > %99.5 (bir kolonda) | CONSTANT kolon → kesinlikle drop et. | 🔴 KRİTİK |
| Top Val Pct %95–99.5 arası | QUASI constant → önce FE dene, sonra karar ver. | 🟢 DÜŞÜK |

---

### Section 3: Target Distribution

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| İkili sınıf, Ratio < 0.2 (ciddi dengesizlik) | `is_unbalance=True` (LightGBM) + `scale_pos_weight` hesapla. SMOTE son çare. | 🔴 KRİTİK |
| İkili sınıf, Ratio 0.2–0.5 | `class_weight="balanced"` veya `scale_pos_weight`. | 🟡 ORTA |
| Ratio > 0.5 | Dengeli. Özel işlem gerekmez. | 🟢 DÜŞÜK |
| Regresyon, \|skew\| > 1 | Target'a `log1p` uygula. Tahminleri `expm1` ile geri dönüştür. | 🟡 ORTA |
| Regresyon, target'ta negatif değer var | `log1p` yerine Box-Cox veya signed log. | 🟡 ORTA |

---

### Section 4: Duplicates & Constants

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Duplicate satır > 0 | `drop_duplicates(keep="first")` — ID olmayan kolonlarda. | 🟡 ORTA |
| Constant kolon tespit edildi | Model eğitiminden önce kesinlikle drop et. FE'de kullanma. | 🔴 KRİTİK |
| Quasi constant (>%95) ama IV > 0.1 | Drop etme, binary flag'e çevir: `is_{col}_dominant`. | 🟡 ORTA |

---

### Section 5: Statistical Risk & Drift (KS Test)

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| KS p >= 0.05 → "NONE" | Drift yok. Normal kullan. | 🟢 DÜŞÜK |
| KS < 0.05 ama KS istatistiği < 0.05 → "Statistical Only" | İstatistiksel ama pratik etki yok. Göz ardı edebilirsin. | 🟢 DÜŞÜK |
| "Mild Drift" (KS 0.05–0.10) | Bu feature'ı normalize et (`RobustScaler` veya `QuantileTransformer`). | 🟡 ORTA |
| "Severe Drift" (KS > 0.10) | Adversarial feature olarak değerlendir. Train'den çıkarmayı dene. | 🔴 KRİTİK |
| VIF > 10 (bir kolon için) | Multicollinearity var. Yüksek VIF'li kolonu drop et veya PCA uygula. | 🟡 ORTA |
| Skew > 2 (bir kolon için) | `log1p` veya `sqrt` transform uygula. | 🟡 ORTA |
| Outlier% > 15 | Winsorize (1st–99th percentile clip) uygula. | 🟡 ORTA |

---

### Section 6–7: Numeric & Categorical Korelasyon

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Pearson \|r\| > 0.85 (iki numeric arasında) | Birini drop et veya farkını/oranını yeni feature yap. | 🟡 ORTA |
| Cramer's V > 0.8 (iki kategorik arasında) | "Redundant" — birini drop et. | 🟡 ORTA |
| Cramer's V 0.5–0.8 | İkisini birleştiren interaction feature oluştur. | 🟢 DÜŞÜK |
| \|r\| ile target > 0.3 (herhangi bir feature) | Güçlü ayrıştırıcı. SHAP ve PDP ile detay incele. | 🟢 DÜŞÜK |
| Cramer's V ile target > 0.3 | Güçlü kategorik. Target encoding için aday. | 🟢 DÜŞÜK |

---

### Section 8: Hierarchical Dependency (Sentinel)

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Sentinel tespit edildi (`"No internet service"` gibi) | Bu değeri `"No"` ile değiştir. Hiyerarşik parent-child ilişkisini kır. | 🔴 KRİTİK |
| Sentinel olan feature'ın churn rate'i diğerlerinden > 10 puan farklı | Binary flag ekle: `has_internet_service` (0/1). | 🟡 ORTA |

---

### Section 10: Categorical Overlap

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Test'te yeni kategori değeri var (Overlap < 100%) | Target encoding veya `"Unknown"` token ekle. Label encoding kullanma. | 🔴 KRİTİK |
| Only_Test sayısı > 100 | Bu kategori istatistiksel anlamlı. Ayrı bir `"Other"` grubu oluştur. | 🟡 ORTA |
| Only_Test sayısı < 10 | `"Unknown"` ile birleştir. | 🟢 DÜŞÜK |

---

### Section 12: Information Value (IV)

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| IV > 0.5 (şüpheli) | Leak kontrolü yap (Section 26). Bu feature'ın production'da var olup olmadığını kontrol et. | 🔴 KRİTİK |
| IV 0.3–0.5 (güçlü) | Feature mükemmel. Interaction ve binning için öncelikli aday. | 🟢 DÜŞÜK |
| IV 0.1–0.3 (orta) | Kullan. Target encoding veya binning ile güçlendir. | 🟢 DÜŞÜK |
| IV 0.02–0.1 (zayıf) | Diğer feature'larla interaction yaratmaya çalış. | 🟢 DÜŞÜK |
| IV < 0.02 (işe yaramaz) | Drop et. Model gürültü katıyor. | 🟡 ORTA |

---

### Section 13 & 18: Adversarial AUC

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Adversarial AUC < 0.52 | Drift yok. Tüm feature'ları kullan. | 🟢 DÜŞÜK |
| Adversarial AUC 0.52–0.60 | Minimal drift. Sadece top adversarial feature'ları izle. | 🟢 DÜŞÜK |
| Adversarial AUC 0.60–0.70 | Top 3 adversarial feature'ı normalize et veya drop et. KS sonuçlarıyla karşılaştır. | 🟡 ORTA |
| Adversarial AUC > 0.70 | **BÜYÜK SORUN.** Sızıntı veya ciddi drift. Pipeline'ı durdur, veriyi incele. | 🔴 KRİTİK |
| Tek feature Adv Imp > 0.10 | O feature drift'li. Dağılımını plot et. Normalize veya drop. | 🟡 ORTA |

---

### Section 16: Baseline Model Skoru

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Baseline AUC > 0.90 | Çok güçlü. FE'den beklenti: +0.001–0.005. Ensemble ve hyperparameter tuning'e odaklan. | 🟢 DÜŞÜK |
| Baseline AUC 0.85–0.90 | İyi. Top feature interaksiyon FE'si +0.005–0.015 kazandırabilir. | 🟢 DÜŞÜK |
| Baseline AUC 0.80–0.85 | Orta. Agresif FE, encoding, augmentation gerekli. | 🟡 ORTA |
| Baseline AUC < 0.80 | Zayıf. Encoding stratejisini değiştir. Stacking veya target encoding dene. | 🔴 KRİTİK |

---

### Section 17: CV Stabilitesi

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Seed std < 0.001 | Çok kararlı. 5-fold yeterli. CV'ye güven. | 🟢 DÜŞÜK |
| Seed std 0.001–0.003 | Makul. Sabit seed kullan. CV güvenilir. | 🟢 DÜŞÜK |
| Seed std 0.003–0.005 | Orta varyans. `RepeatedStratifiedKFold(3×5)` kullan. | 🟡 ORTA |
| Seed std > 0.005 | **KARARLI DEĞİL.** 10-fold veya repeated CV şart. Leak veya veri problemi olabilir. | 🔴 KRİTİK |
| Outlier fold sayısı > 0 | O fold'daki satırların dağılımını incele. Veri spliti problemi olabilir. | 🟡 ORTA |

---

### Section 19: Permutation Importance

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| LGB ve Perm importance UYUŞUYOR (R_Diff ≤ 5) | Built-in importance güvenilir. Her ikisini de kullanabilirsin. | 🟢 DÜŞÜK |
| R_Diff > 5 (1–3 feature) | Bu feature'lar için permutation importance'a göre karar ver, built-in'e değil. | 🟡 ORTA |
| R_Diff > 5 (3+ feature) | Tüm kararlar için permutation importance kullan. Built-in yanıltıcı. | 🟡 ORTA |
| Perm Mean < 0 (negatif) | **Bu feature modele ZARAR veriyor. KALDIR.** | 🔴 KRİTİK |
| Perm Mean ≈ 0 (< 0.0001) | Sıfıra yakın. Kaldır, AUC artarsa kalıcı olarak sil. | 🟡 ORTA |

---

### Section 20: Segment Analizi

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Segment churn spread > 40 puan | GÜÇLÜ sinyal. Top segment'ler için binary risk flag oluştur. | 🔴 KRİTİK |
| Segment churn spread 20–40 puan | Orta sinyal. Segment × feature interaction'ları dene. | 🟡 ORTA |
| Churn rate > %50 olan segment var | Bu segment için özel bir `"high_risk"` binary flag ekle. | 🟡 ORTA |
| Churn rate < %5 olan segment var | `"low_risk"` binary flag ekle. | 🟡 ORTA |

---

### Section 21: Feature Stability

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| CV_coef > 1.0 (çok kararsız feature) | Kaldırmayı dene. CV yükselirse kalıcı sil. | 🟡 ORTA |
| CV_coef 0.5–1.0 | İzle. FE sonrası tekrar kontrol et. | 🟢 DÜŞÜK |
| CV_coef < 0.5 (kararlı) | Güvenilir feature. FE kararlarında önceliklendir. | 🟢 DÜŞÜK |

---

### Section 22: Original Data Karşılaştırma

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Original data target rate farkı < 5 puan | Güvenli concat. `pd.concat([train, original])` + source flag. | 🟢 DÜŞÜK |
| Target rate farkı 5–10 puan | Ağırlıklandırarak concat et. Original satırlara `sample_weight` ekle. | 🟡 ORTA |
| Target rate farkı > 10 puan | Dikkatli kullan. Ayrı bir `"is_original"` feature ekle, model farkı öğrensin. | 🟡 ORTA |
| KS test yüksek (numeric feature kayıyor) | Bu feature'ları original data ile uyumlu hale getir veya augment'te kullanma. | 🟡 ORTA |

---

### Section 23: Monotonicity Check

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Spearman ρ > 0.7 (pozitif monoton) | `monotone_constraints = +1` ekle (LightGBM). Model daha stabil olur. | 🟢 DÜŞÜK |
| Spearman ρ < -0.7 (negatif monoton) | `monotone_constraints = -1` ekle. | 🟢 DÜŞÜK |
| Non-monotonic ama güçlü feature (MI yüksek) | Decile binning yap. `pd.qcut` ile 10 gruba böl. | 🟡 ORTA |

---

### Section 24: Post-Cleaning Interaction Analizi

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| n_services Spearman ρ < -0.7 (çok güçlü) | `n_services` feature'ı oluştur. Çok güçlü sinyal. | 🔴 KRİTİK |
| n_services Spearman ρ -0.3 ile -0.7 arası | `n_services` ekle. Orta güç. | 🟡 ORTA |
| Interaction deviation > 15 puan | Bu kombinasyon için binary flag oluştur: `internet_yes_security_no` gibi. | 🟡 ORTA |
| Interaction deviation > 30 puan | ÇOK GÜÇLÜ. Mutlaka flag ekle. Model tek başına yakalayamıyor olabilir. | 🔴 KRİTİK |

---

## 3. Ek Analizler — Section 26–34 Karar Kuralları

### Section 26: Leakage Detection

> **Amaç:** Tek feature ile modelin hedefi neredeyse mükemmel tahmin ettiği durumları tespit et.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Solo AUC > 0.99 (bir feature için) | **KESİN LEAK.** Hiç düşünmeden drop et. Production'da bu bilgi olmaz. | 🔴 KRİTİK |
| Solo AUC 0.95–0.99 | Muhtemelen leak. Domain knowledge ile kontrol et. Test verisinde bu feature var mı? | 🔴 KRİTİK |
| Solo AUC 0.85–0.95 | Şüpheli. Temporal sıralamayı kontrol et. Churn'den sonra mı oluşmuş? | 🟡 ORTA |
| Solo AUC 0.75–0.85 | Güçlü feature. Normal. SHAP ile detay incele. | 🟢 DÜŞÜK |
| Solo AUC < 0.75 | Normal. Kullan. | 🟢 DÜŞÜK |

---

### Section 27: Label Kalitesi & Target Noise

> **Amaç:** Aynı özellik profiline sahip müşterilerin farklı churn label'ı taşıyıp taşımadığını ölç.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Conflict rate < %5 | Label kalitesi yeterli. Normal devam et. | 🟢 DÜŞÜK |
| Conflict rate %5–10 | `label_smoothing=0.05` parametresini LightGBM'e ekle. | 🟡 ORTA |
| Conflict rate > %10 | `sample_weight` hesapla: düşük güven satırlarına düşük ağırlık ver. | 🔴 KRİTİK |
| Birebir duplicate feature ama farklı target var | `drop_duplicates` YAPMA. Bu satırları 0.5 ile ağırlıklandır. | 🟡 ORTA |

---

### Section 28: Business Logic Validation

> **Amaç:** Domain'e özgü mantık ihlallerini tespit et. Telco için: `TotalCharges ≈ tenure × MonthlyCharges`.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| TotalCharges/expected oranı bozuk < %2 | Önemsiz. Devam et. | 🟢 DÜŞÜK |
| Bozuk satır oranı %2–10 | `is_charges_anomaly` binary flag ekle. Churn rate'i normal ile karşılaştır. | 🟡 ORTA |
| Bozuk satır oranı > %10 | `is_charges_anomaly` zorunlu. Bozuk satırlar farklı churn pattern'ı taşıyabilir. | 🔴 KRİTİK |
| tenure=0 ama TotalCharges > 0 | Veri hatası. TotalCharges'ı 0 ile impute et veya anomaly flag ekle. | 🔴 KRİTİK |
| Sentinel değerleri tespit edildi | Tüm sentinel'leri `"No"` ile değiştir. FE adım 1 bu olmalı. | 🔴 KRİTİK |

---

### Section 29: Cohort Drift Analizi

> **Amaç:** tenure bazlı cohort'lar arasındaki churn rate farklarını ölç. Zamana bağlı pattern'ları yakala.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Cohort churn spread < 10 puan | Cohort etkisi zayıf. Lineer tenure yeterli. | 🟢 DÜŞÜK |
| Cohort churn spread 10–20 puan | `tenure_bin` feature ekle: `pd.qcut(tenure, 5)`. | 🟡 ORTA |
| Cohort churn spread > 20 puan | GÜÇLÜ cohort etkisi. `tenure_group` kategorik feature oluştur. `tenure × MonthlyCharges` interaction ekle. | 🔴 KRİTİK |
| Spearman ρ < -0.7 (tenure↑ churn↓) | Ters monoton. `monotone_constraints=-1` ekle tenure için. | 🟡 ORTA |

---

### Section 30: Stratified Target EDA (KDE)

> **Amaç:** Her numeric feature'ın Churn=Yes vs No dağılımlarını karşılaştır. Güçlü ayrıştırıcıları belirle.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| KS istatistiği > 0.3 | ÇOK GÜÇLÜ feature. SHAP ve interaction için öncelikli aday. FE'de mutlaka kullan. | 🟢 DÜŞÜK |
| KS istatistiği 0.15–0.3 | Orta güçlü. Binning veya log transform ile güçlendir. | 🟢 DÜŞÜK |
| KS istatistiği < 0.05 | Zayıf ayrıştırıcı. Drop et veya başka feature'larla birleştir. | 🟡 ORTA |
| Ortalama farkı büyük ama KS düşük | Aykırı değer etkisi var. Winsorize sonra tekrar test et. | 🟡 ORTA |

---

### Section 31: SHAP Değerleri & Interaction

> **Amaç:** Feature'ların model üzerindeki gerçek etkisini ölç. Non-linear pattern'ları ve interaksiyonları tespit et.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| SHAP mean abs > 0.1 (bir feature) | En önemli feature. Tüm interaksiyon kombinasyonlarını dene. | 🟢 DÜŞÜK |
| SHAP scatter'da V-şekli (non-linear) | Bu feature'a binning veya log transform uygula. | 🟡 ORTA |
| SHAP interaksiyon korelasyonu > 0.3 | Bu ikili için çarpım feature oluştur: `col1 × col2`. | 🟡 ORTA |
| SHAP değeri negatif olan satır oranı > %30 | Feature bazı segmentlerde ters etki yapıyor. Segment bazlı impute veya transform dene. | 🟡 ORTA |
| Non-linear Pearson korelasyon < 0.7 | Linear olmayan pattern. `pd.qcut` ile 10–20 bin dene. | 🟡 ORTA |

---

### Section 32: Hard Sample Analizi

> **Amaç:** Modelin 5 fold'un 4+'ında yanlış tahmin ettiği "zor" satırları tespit et ve analiz et.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| Hard sample oranı < %8 | Sağlıklı model. Temel FE yeterli. | 🟢 DÜŞÜK |
| Hard sample oranı %8–15 | Target encoding veya stacking bu satırlar için yardımcı olabilir. Segment analizi yap. | 🟡 ORTA |
| Hard sample oranı > %15 | Agresif FE gerekli. Hard sample'ların feature profili incelenmeli. | 🔴 KRİTİK |
| Hard sample ortalama bir feature'da > %20 farklı | O feature için hard sample'lara özel transform veya flag oluştur. | 🟡 ORTA |
| Hard sample'ların çoğu belirli bir segment'te | O segment için ayrı model veya özel FE oluştur (stacking için base layer). | 🟡 ORTA |

---

### Section 33: Test Set OOD Analizi

> **Amaç:** Test verisinde train'de hiç görülmemiş değerleri tespit et. Model bu bölgede extrapolate ediyor.

| Koşul / Bulgu | Yapılacak Aksiyon | Öncelik |
|---|---|---|
| OOD yok | Test ve train aynı dağılımda. Güvenli. | 🟢 DÜŞÜK |
| Yeni kategorik değer (az, < 10 satır) | `"Unknown"` token ile birleştir. | 🟡 ORTA |
| Yeni kategorik değer (fazla, > 100 satır) | Target encoding kullan. Label encoding bu durumda yanıltıcı. | 🔴 KRİTİK |
| Numeric değer train range dışında | Winsorize: `clip(lower=train_q01, upper=train_q99)`. | 🟡 ORTA |
| OOD satır oranı > %20 | Test'e özel normalization gerekebilir. `QuantileTransformer(output_distribution="normal")` dene. | 🔴 KRİTİK |

---

## 4. Sırayla Uygulanacak FE Reçetesi

> **Her adımdan sonra CV AUC'yi kontrol et. Düşerse geri al.**

### Faza 0: Zorunlu Temizlik (FE'den Önce)

- Sentinel değerleri `"No"` ile değiştir (`No internet service`, `No phone service`)
- Leak feature'ları drop et
- Constant kolonları drop et
- Negatif permutation importance feature'ları drop et
- `tenure=0` & `TotalCharges>0` anomalilerini düzelt
- OOD kategorik değerler için Unknown token ekle
- OOD numeric değerler için Winsorize uygula

### Faza 1: Temel Matematiksel Dönüşümler

```python
charges_residual = TotalCharges - (tenure * MonthlyCharges)
avg_monthly      = TotalCharges / (tenure + 1)
log1p_tenure     = np.log1p(tenure)
log1p_tc         = np.log1p(TotalCharges)
log1p_mc         = np.log1p(MonthlyCharges)
cost_per_month   = MonthlyCharges / (tenure + 1)   # birim başına maliyet
```

Yüksek skew feature'lara `log1p` veya `sqrt` uygula.

### Faza 2: Kategorik Dönüşümler

```python
n_services       = (OnlineSecurity=="Yes") + (OnlineBackup=="Yes") + (DeviceProtection=="Yes")
                 + (TechSupport=="Yes") + (StreamingTV=="Yes") + (StreamingMovies=="Yes")
n_security       = (OnlineSecurity=="Yes") + (DeviceProtection=="Yes") + (TechSupport=="Yes")
n_streaming      = (StreamingTV=="Yes") + (StreamingMovies=="Yes")
has_internet     = (InternetService != "No").astype(int)
is_mtm           = (Contract == "Month-to-month").astype(int)
is_echeck        = (PaymentMethod == "Electronic check").astype(int)
```

### Faza 3: Risk Flag'leri (Segment Tabanlı)

```python
# Segment churn rate > %50 olan kombinasyonlar için binary flag
is_high_risk     = (is_mtm & (InternetService=="Fiber optic") & (TechSupport=="No")).astype(int)
is_charges_anomaly = (abs(ratio - 1) > 0.5).astype(int)   # business logic
is_hard_sample   = fold_preds[:,].std(axis=1) > 0.4        # opsiyonel
# Cohort spread > 20 puan ise:
tenure_group     = pd.cut(tenure, bins=[0,12,24,36,48,100], labels=["0-12","12-24","24-36","36-48","48+"])
```

### Faza 4: SHAP Tabanlı Interaction'lar

```python
# SHAP interaksiyon çiftleri için çarpım feature'ları
tenure_x_mc      = tenure * MonthlyCharges
tc_x_contract    = TotalCharges * is_mtm

# Non-linear feature'lar için binning
tenure_bin       = pd.qcut(tenure, 10, labels=False, duplicates="drop")
mc_bin           = pd.qcut(MonthlyCharges, 10, labels=False, duplicates="drop")

# Top feature oranları
mc_per_service   = MonthlyCharges / (n_services + 1)
```

### Faza 5: Target Encoding (CV İçinde)

```python
from category_encoders import TargetEncoder
# IV > 0.1 olan tüm kategorik feature'lar
# KFold target encoding kullan — sızıntı önlemek için
# OOD kategorik değerler için de target encoding şart
```

### Faza 6: Monotone Constraints & Model Parametreleri

```python
# Section 23'ten al
monotone_constraints = [+1, -1, 0, ...]   # sırası feats listesiyle aynı olmalı

params = {
    "monotone_constraints": monotone_constraints,
    "is_unbalance": True,           # veya scale_pos_weight
    "label_smoothing": 0.05,        # sadece conflict rate > %5 ise
}
```

---

## 5. Hızlı Karar Referans Tablosu

Pipeline çıktısını okurken bu tabloyu yanında tut:

| Metrik / Bulgu | Eşik Değer | Aksiyon |
|---|---|---|
| Solo AUC | > 0.95 | 🔴 LEAK — DROP |
| Adversarial AUC | > 0.70 | 🔴 BÜYÜK SORUN — DUR |
| Adversarial AUC | 0.60–0.70 | 🟡 Top drift feature normalize et |
| KS Drift | Severe (> 0.10) | 🔴 Drop veya normalize |
| VIF | > 10 | 🟡 Drop veya PCA |
| IV | > 0.5 | 🔴 Leak kontrolü |
| IV | < 0.02 | 🟡 Drop et |
| Perm Importance | < 0 | 🔴 DROP |
| CV std (seed) | > 0.005 | 🔴 CV stratejisini düzelt |
| Hard Sample % | > 15% | 🔴 Agresif FE gerekli |
| Conflict Rate | > 10% | 🔴 sample_weight |
| Cohort Spread | > 20 puan | 🟡 tenure_group feature |
| KDE KS | < 0.05 | 🟡 Feature zayıf — drop |
| Segment Spread | > 40 puan | 🟡 Risk flag oluştur |
| OOD Rate | > 20% | 🔴 Target encoding şart |
| Cramer V | > 0.8 | 🟡 Redundant — birini drop |
| Pearson \|r\| | > 0.85 | 🟡 Korelasyonlu — birini drop |
| SHAP interaction | > 0.3 | 🟢 Çarpım feature oluştur |
| Baseline AUC | < 0.80 | 🔴 Encoding'i yeniden düşün |
| Imbalance Ratio | < 0.2 | 🔴 is_unbalance=True |

---

## 6. Telco Churn Dataset — Özel Notlar

| Not | Detay |
|---|---|
| **Playground Series** | Bu veri sentetik olarak üretilmiştir. Orijinal IBM Telco verisi (blastchar) ile augment edilebilir. |
| **Augment Kuralı** | Target rate farkı < 5 puan → `pd.concat([train, original])`. Fark > 5 puan → `is_original=1` flag ile concat et. |
| **En Güçlü Feature'lar** | `Contract`, `tenure`, `MonthlyCharges`, `TotalCharges`, `InternetService`, `TechSupport` |
| **Domain Bilgisi** | `Month-to-month + Electronic check + Fiber optic + No TechSupport` kombinasyonu maksimum churn risk segmentidir. |
| **charges_residual** | `TotalCharges − (tenure × MonthlyCharges)` → müşterinin beklenenden fazla/az ödeyip ödemediğini gösterir. Çok güçlü feature. |

### Öncelikli FE Listesi — Telco İçin

```python
# 1. En öncelikli
charges_residual      = TotalCharges - (tenure * MonthlyCharges)

# 2. Servis sayaçları
n_services            = sum(df[c] == "Yes" for c in inet_cols)

# 3. Binary sözleşme flag'leri
is_month_to_month     = (Contract == "Month-to-month").astype(int)

# 4. Yüksek risk segmenti
is_high_risk_segment  = (
    (Contract == "Month-to-month") &
    (InternetService == "Fiber optic") &
    (TechSupport == "No")
).astype(int)

# 5. Maliyet/servis oranı
avg_monthly_per_svc   = MonthlyCharges / (n_services + 1)

# 6. Tenure interaksiyonu
tenure_x_monthly      = tenure * MonthlyCharges

# 7. Sözleşme normalize tenure
contract_map          = {"Month-to-month": 1, "One year": 12, "Two year": 24}
contract_tenure_ratio = tenure / df["Contract"].map(contract_map)
```

---

> **Son Not:** Her FE adımı sonrası 5-fold LightGBM AUC'sini taban çizgisiyle karşılaştır. Yükseliş yoksa geri al. Birden fazla feature aynı anda ekleme — **tek tek test et.**