# Türkçe çıktı
```python
!pip install tabulate -q

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import skew, ks_2samp, chi2_contingency
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from IPython.display import display, Image
import base64, io, os, warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="darkgrid", palette="muted")
PLOT_DIR = "/kaggle/working/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# 🔧 REPORTER — print/display/plot → HTML + MD
# ══════════════════════════════════════════════════════════════
class Reporter:

    CSS = """
    <html><head><meta charset="utf-8">
    <style>
      body{font-family:'Segoe UI',Consolas,monospace;background:#0d1117;color:#c9d1d9;padding:30px;max-width:1400px;margin:auto}
      .section{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px;margin:25px 0}
      .section h2{color:#58a6ff;border-bottom:2px solid #58a6ff;padding-bottom:8px;font-size:20px}
      .info-line{background:#1f2937;padding:10px 15px;border-radius:6px;margin:8px 0;border-left:4px solid #58a6ff;font-size:14px}
      .warn{border-left-color:#f0883e;color:#f0883e}
      .ok{border-left-color:#3fb950;color:#3fb950}
      .critical{border-left-color:#f85149;color:#f85149;font-weight:bold}
      .section table{border-collapse:collapse;width:100%;margin:15px 0;font-size:13px}
      .section table th{background:#21262d;color:#58a6ff;padding:10px 14px;border:1px solid #30363d;text-align:left}
      .section table td{padding:8px 14px;border:1px solid #30363d;background:#0d1117}
      .section table tr:hover td{background:#1a2233}
      .plot-container{text-align:center;margin:15px 0}
      .plot-container img{max-width:100%;border-radius:8px;border:1px solid #30363d}
    </style></head><body>
    <h1 style="text-align:center;color:#58a6ff">🚀 SENIOR DATA PIPELINE REPORT</h1>
    """

    def __init__(self, plot_dir=PLOT_DIR):
        self.h = [self.CSS]
        self.m = ["# 🚀 SENIOR DATA PIPELINE REPORT\n\n"]
        self.plot_dir = plot_dir
        self.plot_count = 0

    def log(self, text, cls=""):
        print(text)
        c = f" {cls}" if cls else ""
        self.h.append(f'<div class="info-line{c}">{text}</div>')
        b = "**" if cls in ("critical", "warn") else ""
        self.m.append(f"> {b}{text}{b}\n\n")

    def section(self, title):
        print(f"\n{'═'*80}\n{title}\n{'═'*80}")
        self.h.append(f'<div class="section"><h2>{title}</h2>')
        self.m.append(f"\n---\n\n## {title}\n\n")

    def end(self):
        self.h.append("</div>")

    def table(self, styled):
        display(styled)
        self.h.append(styled.to_html())
        df = styled.data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(filter(None, (str(x) for x in c))) for c in df.columns]
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        try:
            self.m.append(f"\n{df.to_markdown(floatfmt='.4f')}\n\n")
        except Exception:
            self.m.append(f"\n```\n{df.to_string()}\n```\n\n")

    def plot(self, fig, title="plot"):
        """fig → notebook'ta göster + PNG kaydet + HTML'ye embed + MD'ye referans."""
        self.plot_count += 1
        fname = f"{self.plot_count:02d}_{title.replace(' ','_').lower()}.png"
        fpath = os.path.join(self.plot_dir, fname)

        # PNG kaydet
        fig.savefig(fpath, dpi=150, bbox_inches="tight",
                    facecolor="#0d1117", edgecolor="none")
        plt.close(fig)

        # Notebook'ta göster
        display(Image(filename=fpath))

        # HTML'ye base64 embed
        with open(fpath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        self.h.append(f'<div class="plot-container">'
                      f'<img src="data:image/png;base64,{b64}" alt="{title}">'
                      f'</div>')

        # MD'ye referans
        self.m.append(f"\n![{title}](plots/{fname})\n\n")

    def save(self, html_path, md_path):
        self.h.append("</body></html>")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.h))
        print(f"\n✅ HTML kaydedildi  : {html_path}")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.m))
        print(f"✅ Markdown kaydedildi: {md_path}")
        print(f"📁 Grafikler          : {self.plot_dir}/ ({self.plot_count} adet)")


# ══════════════════════════════════════════════════════════════
# 🎨 PLOT FONKSİYONLARI
# ══════════════════════════════════════════════════════════════
DARK = "#0d1117"
CARD = "#161b22"
BLUE = "#58a6ff"
TXT  = "#c9d1d9"
GRID = "#30363d"

def dark_style(ax, title=""):
    ax.set_facecolor(CARD)
    ax.figure.set_facecolor(DARK)
    ax.title.set_color(BLUE)
    ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT)
    ax.tick_params(colors=TXT)
    for s in ax.spines.values():
        s.set_color(GRID)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)


def plot_target_dist(train, target_col, task):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if task == "classification":
        vc = train[target_col].value_counts()
        colors = ["#3fb950", "#f85149"] if len(vc) == 2 else sns.color_palette("muted", len(vc))
        axes[0].bar(vc.index.astype(str), vc.values, color=colors, edgecolor=GRID)
        dark_style(axes[0], f"{target_col} — Count")
        axes[0].set_ylabel("Count")
        axes[1].pie(vc.values, labels=vc.index.astype(str), autopct="%1.1f%%",
                    colors=colors, textprops={"color": TXT},
                    wedgeprops={"edgecolor": GRID})
        axes[1].set_facecolor(DARK)
        axes[1].set_title(f"{target_col} — Ratio", color=BLUE, fontsize=14, fontweight="bold")
    else:
        axes[0].hist(train[target_col].dropna(), bins=50, color=BLUE, edgecolor=GRID, alpha=0.8)
        dark_style(axes[0], f"{target_col} — Distribution")
        sns.boxplot(x=train[target_col].dropna(), ax=axes[1], color=BLUE)
        dark_style(axes[1], f"{target_col} — Boxplot")
    fig.set_facecolor(DARK)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(train, num_cols, target_col):
    tc = train[target_col].copy()
    if tc.dtype == "O":
        tc = LabelEncoder().fit_transform(tc.astype(str))
    cols = num_cols + [target_col]
    data = train[num_cols].copy()
    data[target_col] = tc
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(max(8, len(cols)), max(6, len(cols)*0.8)))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax,
                linewidths=0.5, linecolor=GRID,
                cbar_kws={"shrink": 0.8})
    dark_style(ax, "Korelasyon Matrisi (Numerik + Target)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_cramers_heatmap(train, cat_cols, target_col):
    all_cols = cat_cols + [target_col]
    n = len(all_cols)
    mat = pd.DataFrame(np.zeros((n, n)), index=all_cols, columns=all_cols)
    for i in range(n):
        for j in range(i, n):
            v = cramers_v(train[all_cols[i]].fillna("_NA_"), train[all_cols[j]].fillna("_NA_"))
            mat.iloc[i, j] = v
            mat.iloc[j, i] = v
    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.9), max(8, n * 0.7)))
    sns.heatmap(mat.astype(float), mask=mask, annot=True, fmt=".2f",
                cmap="YlOrRd", vmin=0, vmax=1, ax=ax,
                linewidths=0.5, linecolor=GRID,
                cbar_kws={"shrink": 0.8})
    dark_style(ax, "Cramér's V Matrisi (Kategorik + Target)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_feature_importance(imp_df, top_n=15):
    df = imp_df.head(top_n).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    bars = ax.barh(df["Feature"], df["Importance"], color=BLUE, edgecolor=GRID)
    for bar, val in zip(bars, df["Importance"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", color=TXT, fontsize=10)
    dark_style(ax, f"Top {top_n} Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


def plot_bivariate_top(train, target_col, cat_cols, task, top_n=6):
    tmp = train.copy()
    if tmp[target_col].dtype == "O":
        le = LabelEncoder()
        tmp[target_col] = le.fit_transform(tmp[target_col].astype(str))
    cols = cat_cols[:top_n]
    ncols = 3
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten() if nrows * ncols > 1 else [axes]
    for i, col in enumerate(cols):
        grp = tmp.groupby(col)[target_col].mean() * (100 if task == "classification" else 1)
        grp = grp.sort_values(ascending=False)
        colors = sns.color_palette("RdYlGn_r", len(grp))
        axes[i].barh(grp.index.astype(str), grp.values, color=colors, edgecolor=GRID)
        dark_style(axes[i], col)
        lbl = "Churn Rate (%)" if task == "classification" else f"Mean {target_col}"
        axes[i].set_xlabel(lbl)
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Bivariate — Kategorik vs Target", color=BLUE, fontsize=16, fontweight="bold")
    fig.set_facecolor(DARK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_numeric_dist(train, test, num_cols):
    n = len(num_cols)
    if n == 0:
        return None
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3.5))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for i, col in enumerate(num_cols):
        axes[i].hist(train[col].dropna(), bins=50, alpha=0.6, color="#58a6ff", label="Train", edgecolor=GRID)
        axes[i].hist(test[col].dropna(), bins=50, alpha=0.5, color="#f0883e", label="Test", edgecolor=GRID)
        dark_style(axes[i], col)
        axes[i].legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TXT)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Numerik Dağılımlar — Train vs Test", color=BLUE, fontsize=16, fontweight="bold")
    fig.set_facecolor(DARK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_iv_chart(div):
    df = div.sort_values("IV", ascending=True)
    colors = []
    for v in df["IV"]:
        if v > 0.5:   colors.append("#f85149")
        elif v > 0.3: colors.append("#3fb950")
        elif v > 0.1: colors.append("#f0883e")
        elif v > 0.02:colors.append("#58a6ff")
        else:          colors.append("#484f58")
    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.35)))
    ax.barh(df.index, df["IV"], color=colors, edgecolor=GRID)
    ax.axvline(x=0.02, color="#484f58", linestyle="--", alpha=0.7, label="Weak (0.02)")
    ax.axvline(x=0.1, color="#f0883e", linestyle="--", alpha=0.7, label="Medium (0.1)")
    ax.axvline(x=0.3, color="#3fb950", linestyle="--", alpha=0.7, label="Strong (0.3)")
    dark_style(ax, "Information Value (IV)")
    ax.set_xlabel("IV")
    ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TXT, fontsize=9)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# 🧮 YARDIMCI
# ══════════════════════════════════════════════════════════════
def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct)[0]
    n = ct.sum().sum()
    md = min(ct.shape) - 1
    return np.sqrt(chi2 / (n * md)) if md > 0 else 0.0

def calc_iv(data, feature, target):
    g = data.groupby(feature)[target].agg(["sum","count"])
    g.columns = ["ev","tot"]; g["nev"] = g["tot"] - g["ev"]
    te, tne = g["ev"].sum(), g["nev"].sum()
    g["pe"] = (g["ev"]/te).clip(1e-4); g["pn"] = (g["nev"]/tne).clip(1e-4)
    g["woe"] = np.log(g["pn"]/g["pe"]); g["iv"] = (g["pn"]-g["pe"])*g["woe"]
    return g["iv"].sum()

def drift_label(ks, p):
    if p >= 0.05:    return "✅ YOK"
    if ks < 0.05:    return "🟡 Sadece İstatistiksel"
    if ks < 0.10:    return "🟠 Hafif Drift"
    return "🔴 Ciddi Drift"


# ══════════════════════════════════════════════════════════════
# 🚀 ANA PİPELINE
# ══════════════════════════════════════════════════════════════
def full_senior_data_pipeline(path, train_name, test_name, target_col, id_col=None):

    R = Reporter()
    actions = []

    train = pd.read_csv(f"{path}/{train_name}")
    test  = pd.read_csv(f"{path}/{test_name}")

    feats = [c for c in train.columns if c not in [target_col, id_col]]
    THR = 25
    num_cols = [c for c in feats if train[c].nunique() > THR]
    cat_cols = [c for c in feats if train[c].nunique() <= THR]
    nu = train[target_col].nunique()
    task = "classification" if (train[target_col].dtype == "O" or nu <= 20) else "regression"

    # ── 0 ──
    R.section("📋 0. PIPELINE OVERVIEW")
    R.log(f"🚀 Pipeline | ID: {id_col} | Target: {target_col}")
    R.log(f"📊 {len(feats)} Feature: {len(num_cols)} Numeric, {len(cat_cols)} Categoric")
    R.log(f"🎯 Task: {task.upper()} (target {nu} unique)", cls="ok")
    R.end()

    # ── 1 ──
    R.section("🌟 1. BELLEK VE BOYUT")
    for nm, df in zip(["Train","Test"], [train, test]):
        mem = df.memory_usage(deep=True).sum()/1e6
        R.log(f"📐 {nm}: {df.shape[0]:,} × {df.shape[1]} | 💾 {mem:.1f} MB")
    R.end()

    # ── 2 ──
    R.section("🌟 2. GENEL VERİ AUDIT")
    role = {c: ("Numeric" if c in num_cols else "Categoric") for c in feats}
    met = {
        "Meta": pd.DataFrame({"Dtype": train[feats].dtypes.astype(str), "Role": pd.Series(role)}),
        "N-Unique": pd.concat([train[feats].nunique().rename("Train"),
                                test[feats].nunique().rename("Test")], axis=1),
        "Null Pct": pd.concat([(train[feats].isnull().mean()*100).rename("Train"),
                                (test[feats].isnull().mean()*100).rename("Test")], axis=1),
        "Top Val Pct": pd.concat([
            train[feats].apply(lambda x: x.value_counts(normalize=True).iloc[0]*100
                                if not x.dropna().empty else 0).rename("Train"),
            test[feats].apply(lambda x: x.value_counts(normalize=True).iloc[0]*100
                               if not x.dropna().empty else 0).rename("Test")], axis=1),
    }
    da = pd.concat(met.values(), axis=1, keys=met.keys())
    da[("Diff","Null Gap")] = (da[("Null Pct","Test")] - da[("Null Pct","Train")]).round(2)
    da = da.sort_values(by=[("Meta","Role"),("N-Unique","Train")], ascending=[False,False])
    R.table(da.style
        .background_gradient(cmap="YlGnBu",
            subset=[c for c in da.columns if c[0] in ["N-Unique","Null Pct","Top Val Pct"]])
        .background_gradient(cmap="Reds", subset=[("Diff","Null Gap")])
        .format(precision=2))
    high_null = [c for c in feats if train[c].isnull().mean() > 0.3]
    if high_null:
        actions.append(f"🗑️ >%30 null: {high_null}")
    R.end()

    # ── 3 ── TARGET + GRAFİK
    R.section("🌟 3. TARGET DAĞILIMI")
    if task == "classification":
        vc = train[target_col].value_counts()
        vcp = train[target_col].value_counts(normalize=True)*100
        dt = pd.DataFrame({"Count": vc, "Pct (%)": vcp, "Bar": vcp})
        R.table(dt.style.bar(subset=["Bar"], color="#636efa").format({"Pct (%)":"{:.2f}","Bar":"{:.1f}"}))
        ir = vc.min()/vc.max()
        if ir < 0.2:
            R.log(f"⚠️ Şiddetli dengesizlik! Min/Max: {ir:.3f}", cls="critical")
            actions.append("⚖️ Şiddetli imbalance → SMOTE / focal loss")
        elif ir < 0.5:
            R.log(f"⚠️ Orta dengesizlik. Min/Max: {ir:.3f}", cls="warn")
            actions.append("⚖️ Hafif imbalance → class_weight='balanced'")
        else:
            R.log(f"✅ Dengeli. Min/Max: {ir:.3f}", cls="ok")
    else:
        desc = train[target_col].describe().to_frame().T
        desc["skew"] = skew(train[target_col].dropna())
        R.table(desc.style.format(precision=3))
        if abs(desc["skew"].iloc[0]) > 1:
            R.log("⚠️ Target çarpık → log1p düşün", cls="warn")
            actions.append("📐 Target skewed → np.log1p()")

    R.plot(plot_target_dist(train, target_col, task), "target_distribution")
    R.end()

    # ── 4 ──
    R.section("🌟 4. DUPLICATE & SABİT SÜTUN")
    dup = train.duplicated().sum()
    if dup:
        R.log(f"⚠️ {dup:,} duplicate ({dup/len(train)*100:.2f}%)", cls="warn")
        actions.append(f"🔄 {dup} duplicate → drop")
    else:
        R.log("✅ Duplicate yok.", cls="ok")
    const = []
    for c in feats:
        tp = train[c].value_counts(normalize=True).iloc[0]*100
        if tp >= 99.5:   const.append({"Feature": c, "Top%": tp, "Status": "🔴 CONSTANT"})
        elif tp >= 95:   const.append({"Feature": c, "Top%": tp, "Status": "🟡 QUASI"})
    if const:
        dc = pd.DataFrame(const).set_index("Feature")
        R.table(dc.style.map(lambda x: "color:#f85149" if "CONSTANT" in str(x) else "color:#f0883e",
                             subset=["Status"]).format({"Top%": "{:.2f}"}))
        drc = [r["Feature"] for r in const if r["Top%"] >= 99.5]
        if drc: actions.append(f"🗑️ Sabit: {drc}")
    else:
        R.log("✅ Sabit/quasi-sabit yok.", cls="ok")
    R.end()

    # ── 5 ── RISK & DRIFT + NUMERİK GRAFİK
    R.section("🌟 5. STATISTICAL RISK & DRIFT")
    if num_cols:
        stats = []
        for c in num_cols:
            tr, ts = train[c].dropna(), test[c].dropna()
            q1, q3 = tr.quantile(0.25), tr.quantile(0.75)
            iqr = q3-q1
            out = ((tr < q1-1.5*iqr)|(tr > q3+1.5*iqr)).mean()*100
            ks, p = ks_2samp(tr, ts)
            stats.append({"Sütun": c, "Skew": skew(tr), "Outlier (%)": out,
                          "KS-Stat": ks, "P-Value": p, "Drift": drift_label(ks, p)})
        vd = train[num_cols].dropna()
        vr = [variance_inflation_factor(vd.values, i) for i in range(len(num_cols))]
        tmi = train[num_cols+[target_col]].dropna()
        if tmi[target_col].dtype == "O":
            tmi[target_col] = LabelEncoder().fit_transform(tmi[target_col])
        mi_fn = mutual_info_classif if task == "classification" else mutual_info_regression
        mi = mi_fn(tmi[num_cols], tmi[target_col], random_state=42)
        da2 = pd.DataFrame(stats).set_index("Sütun")
        da2["VIF"] = vr; da2["MI_Score"] = mi
        R.table(da2.style
            .background_gradient(cmap="OrRd", subset=["VIF","Skew"])
            .background_gradient(cmap="YlGn", subset=["MI_Score"])
            .map(lambda x: "color:red;font-weight:bold" if "Ciddi" in str(x)
                 else ("color:orange" if "Hafif" in str(x) or "İstatistiksel" in str(x)
                        else "color:green"), subset=["Drift"])
            .format(precision=3))
        rd = [s["Sütun"] for s in stats if "Ciddi" in s["Drift"] or "Hafif" in s["Drift"]]
        if rd: actions.append(f"📉 Gerçek drift: {rd}")
        hv = [num_cols[i] for i,v in enumerate(vr) if v > 10]
        if hv: actions.append(f"📊 VIF>10: {hv}")
        lmi = [num_cols[i] for i,v in enumerate(mi) if v < 0.01]
        if lmi: actions.append(f"📉 MI≈0: {lmi}")

        fig_nd = plot_numeric_dist(train, test, num_cols)
        if fig_nd:
            R.plot(fig_nd, "numeric_distributions")
    else:
        R.log("ℹ️ Numerik sütun yok.")
    R.end()

    # ── 6 ── KORELASYON + HEATMAP
    R.section("🌟 6. NUMERİK KORELASYON")
    if len(num_cols) >= 2:
        corr = train[num_cols].corr()
        hcp = []
        for i in range(len(num_cols)):
            for j in range(i+1, len(num_cols)):
                r = corr.iloc[i, j]
                if abs(r) > 0.85:
                    hcp.append({"Feature_1": num_cols[i], "Feature_2": num_cols[j],
                                "|r|": abs(r), "Dir": "➕" if r>0 else "➖"})
        if hcp:
            dhc = pd.DataFrame(hcp).sort_values("|r|", ascending=False).reset_index(drop=True)
            R.log(f"⚠️ {len(dhc)} çift |r|>0.85:", cls="warn")
            R.table(dhc.style.background_gradient(cmap="Reds", subset=["|r|"]).format({"|r|":"{:.4f}"}))
            actions.append(f"🔗 {len(dhc)} yüksek korelasyonlu çift")
        else:
            R.log("✅ |r|>0.85 çift yok.", cls="ok")
        tc = train[target_col].copy()
        if tc.dtype == "O": tc = LabelEncoder().fit_transform(tc.astype(str))
        tcr = train[num_cols].corrwith(pd.Series(tc, index=train.index)).abs().sort_values(ascending=False)
        dtc = pd.DataFrame({"Feature": tcr.index, f"|r| with {target_col}": tcr.values}).reset_index(drop=True)
        R.table(dtc.style.bar(subset=[f"|r| with {target_col}"], color="#3fb950")
                .format({f"|r| with {target_col}": "{:.4f}"}))
        R.plot(plot_correlation_heatmap(train, num_cols, target_col), "correlation_heatmap")
    else:
        R.log("ℹ️ Yeterli numerik sütun yok.")
    R.end()

    # ── 7 ── CRAMÉR'S V + HEATMAP
    R.section("🌟 7. CRAMÉR'S V — KATEGORİK KORELASYON")
    if len(cat_cols) >= 2:
        hcv = []
        for i in range(len(cat_cols)):
            for j in range(i+1, len(cat_cols)):
                v = cramers_v(train[cat_cols[i]].fillna("_NA_"), train[cat_cols[j]].fillna("_NA_"))
                if v > 0.5:
                    hcv.append({"Feature_1": cat_cols[i], "Feature_2": cat_cols[j],
                                "V": v, "Risk": "🔴 Redundant" if v > 0.8 else "🟡 High"})
        if hcv:
            dcv = pd.DataFrame(hcv).sort_values("V", ascending=False).reset_index(drop=True)
            R.log(f"⚠️ {len(dcv)} çift V>0.5:", cls="warn")
            R.table(dcv.style.background_gradient(cmap="Reds", subset=["V"]).format({"V":"{:.4f}"}))
            red = dcv[dcv["V"] > 0.8]
            if len(red):
                actions.append(f"🔗 Redundant kategorik: {list(zip(red['Feature_1'],red['Feature_2']))}")
        else:
            R.log("✅ V>0.5 çift yok.", cls="ok")
        tcv = {}
        tgf = train[target_col].fillna("_NA_")
        for c in cat_cols:
            tcv[c] = cramers_v(train[c].fillna("_NA_"), tgf)
        dtcv = (pd.DataFrame.from_dict(tcv, orient="index", columns=[f"V with {target_col}"])
                .sort_values(f"V with {target_col}", ascending=False))
        R.table(dtcv.style.bar(subset=[f"V with {target_col}"], color="#636efa").format(precision=4))
        R.plot(plot_cramers_heatmap(train, cat_cols, target_col), "cramers_v_heatmap")
    else:
        R.log("ℹ️ Yeterli kategorik sütun yok.")
    R.end()

    # ── 8 ── HİYERARŞİK
    R.section("🌟 8. HİYERARŞİK BAĞIMLILIK")
    sentinels = ["No internet service", "No phone service"]
    hier = []
    for col in cat_cols:
        for sv in sentinels:
            mask = train[col] == sv
            sc = mask.sum()
            if sc == 0: continue
            for pcol in cat_cols:
                if pcol == col: continue
                for pval in train[pcol].unique():
                    if (train[pcol]==pval).sum() != sc: continue
                    pm = train[pcol] == pval
                    if (mask & pm).sum() == sc:
                        tgt = train[target_col].copy()
                        if tgt.dtype == "O": tgt = (tgt=="Yes").astype(int)
                        hier.append({"Child": col, "Sentinel": sv, "Parent": pcol,
                                     "Parent_Val": pval, "Count": sc,
                                     f"{target_col}_Rate%": tgt[mask].mean()*100})
    if hier:
        dh = pd.DataFrame(hier)
        R.log(f"⚠️ {len(dh)} hiyerarşik bağımlılık!", cls="critical")
        R.table(dh.style.format({f"{target_col}_Rate%":"{:.2f}"})
                .background_gradient(cmap="OrRd", subset=["Count"]))
        R.log("💡 Sentinel → NaN veya parent ile merge et.", cls="warn")
        actions.append(f"🔗 Hiyerarşik: {list(dh['Child'].unique())}")
    else:
        R.log("✅ Hiyerarşik bağımlılık yok.", cls="ok")
    R.end()

    # ── 9 ── TÜRETİLMİŞ
    R.section("🌟 9. TÜRETİLMİŞ FEATURE İLİŞKİLERİ")
    if len(num_cols) >= 2:
        derived = []
        cc = num_cols[:10]
        for i in range(len(cc)):
            for j in range(i+1, len(cc)):
                for k in range(len(cc)):
                    if k in (i,j): continue
                    a, b, c = train[cc[i]], train[cc[j]], train[cc[k]]
                    prod = a*b
                    if prod.std() > 0 and c.std() > 0:
                        rp = prod.corr(c)
                        if abs(rp) > 0.90:
                            derived.append({"İlişki": f"{cc[i]} × {cc[j]}",
                                            "Hedef": cc[k], "|r|": abs(rp), "Tip": "Product"})
                    if (b.abs()>1e-9).all():
                        rat = a/b
                        if rat.std()>0 and c.std()>0:
                            rr = rat.corr(c)
                            if abs(rr) > 0.90:
                                derived.append({"İlişki": f"{cc[i]} / {cc[j]}",
                                                "Hedef": cc[k], "|r|": abs(rr), "Tip": "Ratio"})
        if derived:
            dd = pd.DataFrame(derived).sort_values("|r|", ascending=False).reset_index(drop=True)
            R.log(f"⚠️ {len(dd)} türetilmiş ilişki:", cls="critical")
            R.table(dd.style.background_gradient(cmap="Reds", subset=["|r|"]).format({"|r|":"{:.4f}"}))
            actions.append(f"🔗 Türetilmiş: {dd['Hedef'].unique().tolist()} → Residual")
        else:
            R.log("✅ Türetilmiş ilişki yok.", cls="ok")
    else:
        R.log("ℹ️ Yeterli numerik sütun yok.")
    R.end()

    # ── 10 ── OVERLAP
    R.section("🌟 10. KATEGORİK OVERLAP")
    if cat_cols:
        rc = []
        for c in cat_cols:
            trs = set(train[c].dropna().unique())
            tss = set(test[c].dropna().unique())
            ov = (len(tss&trs)/len(tss)*100) if tss else 100
            rc.append({"Sütun": c, "Tr": len(trs), "Ts": len(tss),
                       "Overlap%": ov, "Only_Test": len(tss-trs)})
        drc = pd.DataFrame(rc).set_index("Sütun")
        R.table(drc.style.map(lambda x: "color:red" if x<100 else "color:green", subset=["Overlap%"]))
        lov = [r["Sütun"] for r in rc if r["Overlap%"] < 100]
        if lov: actions.append(f"🔀 Yeni kategoriler: {lov}")
    else:
        R.log("ℹ️ Kategorik sütun yok.")
    R.end()

    # ── 11 ── BIVARIATE + GRAFİK
    R.section("🌟 11. TARGET BIVARIATE")
    if cat_cols:
        tmp = train.copy()
        if tmp[target_col].dtype == "O":
            le = LabelEncoder()
            tmp[target_col] = le.fit_transform(tmp[target_col].astype(str))
            tl = le.classes_[1] if len(le.classes_) > 1 else "Positive"
        else:
            tl = "Target"
        bv = []
        for c in cat_cols:
            grp = tmp.groupby(c)[target_col]
            if task == "classification":
                for idx, val in (grp.mean()*100).items():
                    bv.append({"Feature": c, "Category": idx,
                               f"{tl}_Rate%": val, "Count": grp.count().get(idx, 0)})
            else:
                for idx, val in grp.mean().items():
                    bv.append({"Feature": c, "Category": idx,
                               f"Mean_{target_col}": val, "Count": grp.count().get(idx, 0)})
        dbv = pd.DataFrame(bv).set_index(["Feature","Category"])
        rc2 = [c for c in dbv.columns if "Rate" in c or "Mean" in c][0]
        R.table(dbv.style.background_gradient(cmap="RdYlGn_r", subset=[rc2]).format(precision=2))
        R.plot(plot_bivariate_top(train, target_col, cat_cols, task), "bivariate_top")
    else:
        R.log("ℹ️ Kategorik sütun yok.")
    R.end()

    # ── 12 ── IV + GRAFİK
    R.section("🌟 12. INFORMATION VALUE (IV)")
    if task == "classification" and cat_cols:
        tmp_iv = train.copy()
        if tmp_iv[target_col].dtype == "O":
            vals = tmp_iv[target_col].unique()
            pos = vals[1] if len(vals) > 1 else vals[0]
            tmp_iv[target_col] = (tmp_iv[target_col] == pos).astype(int)
        ivr = {}
        for c in cat_cols:
            ivr[c] = calc_iv(tmp_iv, c, target_col)
        div = (pd.DataFrame.from_dict(ivr, orient="index", columns=["IV"])
               .sort_values("IV", ascending=False))
        def iv_lbl(x):
            if x > 0.5:  return "🔴 Suspicious"
            if x > 0.3:  return "🟢 Strong"
            if x > 0.1:  return "🟡 Medium"
            if x > 0.02: return "🔵 Weak"
            return "⚪ Useless"
        div["Strength"] = div["IV"].apply(iv_lbl)
        R.table(div.style.bar(subset=["IV"], color="#636efa").format({"IV":"{:.4f}"}))
        R.plot(plot_iv_chart(div), "information_value")
        sus = div[div["IV"] > 0.5].index.tolist()
        if sus:
            R.log(f"⚠️ IV>0.5 leakage riski: {sus}", cls="critical")
            actions.append(f"🚨 IV suspicious: {sus}")
        useless = div[div["IV"] < 0.02].index.tolist()
        if useless: actions.append(f"📉 IV<0.02: {useless}")
    else:
        R.log("ℹ️ IV sadece binary classification için.")
    R.end()

    # ── 13 ── ADVERSARIAL & IMPORTANCE + GRAFİK
    R.section("🌟 13. ADVERSARIAL & FEATURE IMPORTANCE")
    adf = pd.concat([train[feats], test[feats]], axis=0)
    ya = [0]*len(train)+[1]*len(test)
    for c in adf.select_dtypes(["object","category"]).columns:
        adf[c] = pd.factorize(adf[c])[0]
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    auc = cross_val_score(rf, adf.fillna(-1), ya, cv=5, scoring="roc_auc").mean()
    if auc > 0.8:
        R.log(f"🕵️ Adversarial AUC: {auc:.4f} — 🔴 Ciddi!", cls="critical")
        actions.append(f"🚨 Adv AUC={auc:.4f}")
    elif auc > 0.6:
        R.log(f"🕵️ Adversarial AUC: {auc:.4f} — ⚠️ Hafif drift.", cls="warn")
    else:
        R.log(f"🕵️ Adversarial AUC: {auc:.4f} — ✅ Güvenli.", cls="ok")

    timp = train[feats+[target_col]].dropna()
    for c in timp.select_dtypes(["object","category"]).columns:
        timp[c] = LabelEncoder().fit_transform(timp[c].astype(str))
    rf_imp = (RandomForestClassifier if task == "classification" else RandomForestRegressor)(
        n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf_imp.fit(timp.drop(columns=[target_col]), timp[target_col])
    imp = (pd.DataFrame({"Feature": feats, "Importance": rf_imp.feature_importances_})
           .sort_values("Importance", ascending=False).reset_index(drop=True))
    imp["Cumulative"] = imp["Importance"].cumsum()
    imp["Rank"] = range(1, len(imp)+1)
    R.table(imp.style.bar(subset=["Importance"], color="#636efa")
            .background_gradient(cmap="YlGn", subset=["Cumulative"])
            .format({"Importance":"{:.4f}","Cumulative":"{:.4f}"}))
    R.plot(plot_feature_importance(imp), "feature_importance")

    li = set(imp[imp["Importance"] < 0.005]["Feature"])
    hd = set(c for c in cat_cols if train[c].value_counts(normalize=True).iloc[0] > 0.90)
    tu = li & hd
    if tu:
        R.log(f"🗑️ Imp<0.005 + Dom>90%: {tu}", cls="warn")
        actions.append(f"🗑️ Useless: {tu}")
    la = imp[imp["Importance"] < 0.01]["Feature"].tolist()
    if la: actions.append(f"📉 Imp<0.01: {la[:5]}{'...' if len(la)>5 else ''}")
    t80 = imp[imp["Cumulative"] <= 0.80]["Feature"].tolist()
    R.log(f"📊 Top 80% bilgi: {len(t80)} feature → {t80[:8]}{'...' if len(t80)>8 else ''}")
    R.end()

    # ── 14 ── MISSING
    R.section("🌟 14. MISSING VALUE PATERNLERİ")
    nc = [c for c in feats if train[c].isnull().any()]
    if nc:
        ns = []
        for c in nc:
            np_ = train[c].isnull().mean()*100
            tgt2 = train[target_col].copy()
            if tgt2.dtype == "O": tgt2 = LabelEncoder().fit_transform(tgt2.astype(str))
            if task == "classification":
                rn = tgt2[train[c].isnull()].mean()*100 if train[c].isnull().any() else 0
                rnp = tgt2[train[c].notnull()].mean()*100
                ns.append({"Feature": c, "Null%": np_, "Rate_NULL%": rn, "Rate_PRESENT%": rnp,
                           "Gap": abs(rn-rnp)})
            else:
                mn = train.loc[train[c].isnull(), target_col].mean()
                mnp = train.loc[train[c].notnull(), target_col].mean()
                ns.append({"Feature": c, "Null%": np_, "Mean_NULL": mn, "Mean_PRESENT": mnp,
                           "Gap": abs(mn-mnp) if pd.notna(mn) else 0})
        dn = pd.DataFrame(ns).set_index("Feature").sort_values("Gap", ascending=False)
        R.table(dn.style.background_gradient(cmap="OrRd", subset=["Gap","Null%"]).format(precision=2))
        inull = dn[dn["Gap"] > 5].index.tolist()
        if inull:
            R.log(f"💡 Bilgi taşıyan null: {inull}", cls="warn")
            actions.append(f"🏷️ is_null flag: {inull}")
    else:
        R.log("✅ Missing yok.", cls="ok")
    R.end()

    # ── 15 ── ORİJİNAL VERİ
    R.section("🌟 15. PLAYGROUND SERIES — ORİJİNAL VERİ")
    R.log("📌 Playground Series → orijinal veri augmentation yapılabilir.")
    R.log("📌 Bu veri → IBM Telco Customer Churn (Kaggle'da mevcut).")
    R.log("💡 pd.concat([train, original]) + source flag ekle.", cls="warn")
    actions.append("📦 Orijinal veri augmentation")
    R.end()

    # ── 16 ── AKSİYON
    R.section("🎯 16. AKSİYON ÖZETİ — NE YAPMALI?")
    if actions:
        for i, a in enumerate(actions, 1):
            cl = "critical" if any(x in a for x in ["🚨","🔴"]) else ("warn" if "⚠" in a else "")
            R.log(f"  {i}. {a}", cls=cl)
    else:
        R.log("✅ Kritik sorun yok.", cls="ok")
    R.log("")
    R.log("📌 Önerilen sonraki adımlar:", cls="ok")
    for s in ["1️⃣ Sentinel'leri NaN'a çevir", "2️⃣ Orijinal veri augment",
              "3️⃣ FE: residual, interaction, aggregation", "4️⃣ Target / CatBoost encoding",
              "5️⃣ Baseline: LGB + StratifiedKFold", "6️⃣ Optuna tuning",
              "7️⃣ Ensemble: LGB + XGB + CatBoost"]:
        R.log(f"  {s}")
    R.end()

    # ── KAYDET ──
    R.save(
        html_path="/kaggle/working/pipeline_report.html",
        md_path="/kaggle/working/pipeline_report.md",
    )

    return train, test, target_col, id_col


# ═══════════════════════════
# ÇALIŞTIR
# ═══════════════════════════
train, test, target_col, id_col = full_senior_data_pipeline(
    "/kaggle/input/competitions/playground-series-s6e3/",
    "train.csv", "test.csv", "Churn", "id"
)
```

# ingilizce çıktı
```python
!pip install tabulate -q

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, ks_2samp, chi2_contingency
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from IPython.display import display, Image
import base64, os, warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="darkgrid", palette="muted")
PLOT_DIR = "/kaggle/working/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# SAFE ENCODE — always returns pandas Series
# ══════════════════════════════════════════════════════════════
def safe_encode_target(series):
    if series.dtype == "O":
        return pd.Series(
            LabelEncoder().fit_transform(series.astype(str)),
            index=series.index
        )
    return series


# ══════════════════════════════════════════════════════════════
# REPORTER
# ══════════════════════════════════════════════════════════════
class Reporter:

    CSS = """
    <html><head><meta charset="utf-8">
    <style>
      body{font-family:'Segoe UI',Consolas,monospace;background:#0d1117;color:#c9d1d9;padding:30px;max-width:1400px;margin:auto}
      .section{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px;margin:25px 0}
      .section h2{color:#58a6ff;border-bottom:2px solid #58a6ff;padding-bottom:8px;font-size:20px}
      .info-line{background:#1f2937;padding:10px 15px;border-radius:6px;margin:8px 0;border-left:4px solid #58a6ff;font-size:14px}
      .warn{border-left-color:#f0883e;color:#f0883e}
      .ok{border-left-color:#3fb950;color:#3fb950}
      .critical{border-left-color:#f85149;color:#f85149;font-weight:bold}
      .section table{border-collapse:collapse;width:100%;margin:15px 0;font-size:13px}
      .section table th{background:#21262d;color:#58a6ff;padding:10px 14px;border:1px solid #30363d;text-align:left}
      .section table td{padding:8px 14px;border:1px solid #30363d;background:#0d1117}
      .section table tr:hover td{background:#1a2233}
      .plot-container{text-align:center;margin:15px 0}
      .plot-container img{max-width:100%;border-radius:8px;border:1px solid #30363d}
    </style></head><body>
    <h1 style="text-align:center;color:#58a6ff">🚀 SENIOR DATA PIPELINE REPORT</h1>
    """

    def __init__(self, plot_dir=PLOT_DIR):
        self.h = [self.CSS]
        self.m = ["# 🚀 SENIOR DATA PIPELINE REPORT\n\n"]
        self.plot_dir = plot_dir
        self.plot_count = 0

    def log(self, text, cls=""):
        print(text)
        c = f" {cls}" if cls else ""
        self.h.append(f'<div class="info-line{c}">{text}</div>')
        b = "**" if cls in ("critical", "warn") else ""
        self.m.append(f"> {b}{text}{b}\n\n")

    def section(self, title):
        print(f"\n{'═'*80}\n{title}\n{'═'*80}")
        self.h.append(f'<div class="section"><h2>{title}</h2>')
        self.m.append(f"\n---\n\n## {title}\n\n")

    def end(self):
        self.h.append("</div>")

    def table(self, styled):
        display(styled)
        self.h.append(styled.to_html())
        df = styled.data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(filter(None, (str(x) for x in c))) for c in df.columns]
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        try:
            self.m.append(f"\n{df.to_markdown(floatfmt='.4f')}\n\n")
        except Exception:
            self.m.append(f"\n```\n{df.to_string()}\n```\n\n")

    def plot(self, fig, title="plot"):
        self.plot_count += 1
        fname = f"{self.plot_count:02d}_{title.replace(' ','_').lower()}.png"
        fpath = os.path.join(self.plot_dir, fname)
        fig.savefig(fpath, dpi=150, bbox_inches="tight",
                    facecolor="#0d1117", edgecolor="none")
        plt.close(fig)
        display(Image(filename=fpath))
        with open(fpath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        self.h.append(f'<div class="plot-container">'
                      f'<img src="data:image/png;base64,{b64}" alt="{title}">'
                      f'</div>')
        self.m.append(f"\n![{title}](plots/{fname})\n\n")

    def save(self, html_path, md_path):
        self.h.append("</body></html>")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.h))
        print(f"\n✅ HTML saved  : {html_path}")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.m))
        print(f"✅ Markdown saved: {md_path}")
        print(f"📁 Plots         : {self.plot_dir}/ ({self.plot_count} files)")


# ══════════════════════════════════════════════════════════════
# PLOT FUNCTIONS
# ══════════════════════════════════════════════════════════════
DARK = "#0d1117"; CARD = "#161b22"; BLUE = "#58a6ff"; TXT = "#c9d1d9"; GRID = "#30363d"

def dark_style(ax, title=""):
    ax.set_facecolor(CARD); ax.figure.set_facecolor(DARK)
    ax.title.set_color(BLUE); ax.xaxis.label.set_color(TXT); ax.yaxis.label.set_color(TXT)
    ax.tick_params(colors=TXT)
    for s in ax.spines.values(): s.set_color(GRID)
    if title: ax.set_title(title, fontsize=14, fontweight="bold", pad=12)


def plot_target_dist(train, target_col, task):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if task == "classification":
        vc = train[target_col].value_counts()
        colors = ["#3fb950", "#f85149"] if len(vc) == 2 else sns.color_palette("muted", len(vc))
        axes[0].bar(vc.index.astype(str), vc.values, color=colors, edgecolor=GRID)
        dark_style(axes[0], f"{target_col} — Count")
        axes[0].set_ylabel("Count")
        axes[1].pie(vc.values, labels=vc.index.astype(str), autopct="%1.1f%%",
                    colors=colors, textprops={"color": TXT}, wedgeprops={"edgecolor": GRID})
        axes[1].set_facecolor(DARK)
        axes[1].set_title(f"{target_col} — Ratio", color=BLUE, fontsize=14, fontweight="bold")
    else:
        axes[0].hist(train[target_col].dropna(), bins=50, color=BLUE, edgecolor=GRID, alpha=0.8)
        dark_style(axes[0], f"{target_col} — Distribution")
        sns.boxplot(x=train[target_col].dropna(), ax=axes[1], color=BLUE)
        dark_style(axes[1], f"{target_col} — Boxplot")
    fig.set_facecolor(DARK); fig.tight_layout()
    return fig


def plot_correlation_heatmap(train, num_cols, target_col):
    data = train[num_cols].copy()
    data[target_col] = safe_encode_target(train[target_col].copy())
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n * 0.8)))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax,
                linewidths=0.5, linecolor=GRID, cbar_kws={"shrink": 0.8})
    dark_style(ax, "Correlation Matrix (Numeric + Target)")
    ax.tick_params(axis="x", rotation=45); fig.tight_layout()
    return fig


def plot_cramers_heatmap(train, cat_cols, target_col):
    all_cols = cat_cols + [target_col]
    n = len(all_cols)
    mat = pd.DataFrame(np.zeros((n, n)), index=all_cols, columns=all_cols)
    for i in range(n):
        for j in range(i, n):
            v = cramers_v(train[all_cols[i]].fillna("_NA_"), train[all_cols[j]].fillna("_NA_"))
            mat.iloc[i, j] = v; mat.iloc[j, i] = v
    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.9), max(8, n * 0.7)))
    sns.heatmap(mat.astype(float), mask=mask, annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0, vmax=1, ax=ax, linewidths=0.5, linecolor=GRID, cbar_kws={"shrink": 0.8})
    dark_style(ax, "Cramer's V Matrix (Categorical + Target)")
    ax.tick_params(axis="x", rotation=45); fig.tight_layout()
    return fig


def plot_feature_importance(imp_df, top_n=15):
    df = imp_df.head(top_n).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    bars = ax.barh(df["Feature"], df["Importance"], color=BLUE, edgecolor=GRID)
    for bar, val in zip(bars, df["Importance"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", color=TXT, fontsize=10)
    dark_style(ax, f"Top {top_n} Feature Importance")
    ax.set_xlabel("Importance"); fig.tight_layout()
    return fig


def plot_bivariate_top(train, target_col, cat_cols, task, top_n=6):
    tmp = train.copy()
    tmp[target_col] = safe_encode_target(tmp[target_col].copy())
    cols = cat_cols[:top_n]
    ncols = 3; nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten() if nrows * ncols > 1 else [axes]
    for i, col in enumerate(cols):
        grp = tmp.groupby(col)[target_col].mean() * (100 if task == "classification" else 1)
        grp = grp.sort_values(ascending=False)
        colors = sns.color_palette("RdYlGn_r", len(grp))
        axes[i].barh(grp.index.astype(str), grp.values, color=colors, edgecolor=GRID)
        dark_style(axes[i], col)
        axes[i].set_xlabel("Target Rate (%)" if task == "classification" else f"Mean {target_col}")
    for j in range(len(cols), len(axes)): axes[j].set_visible(False)
    fig.suptitle("Bivariate — Categorical vs Target", color=BLUE, fontsize=16, fontweight="bold")
    fig.set_facecolor(DARK); fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_numeric_dist(train, test, num_cols):
    n = len(num_cols)
    if n == 0: return None
    ncols = min(3, n); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3.5))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for i, col in enumerate(num_cols):
        axes[i].hist(train[col].dropna(), bins=50, alpha=0.6, color="#58a6ff", label="Train", edgecolor=GRID)
        axes[i].hist(test[col].dropna(), bins=50, alpha=0.5, color="#f0883e", label="Test", edgecolor=GRID)
        dark_style(axes[i], col)
        axes[i].legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TXT)
    for j in range(n, len(axes)): axes[j].set_visible(False)
    fig.suptitle("Numeric Distributions — Train vs Test", color=BLUE, fontsize=16, fontweight="bold")
    fig.set_facecolor(DARK); fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_iv_chart(div):
    df = div.sort_values("IV", ascending=True)
    colors = []
    for v in df["IV"]:
        if v > 0.5:    colors.append("#f85149")
        elif v > 0.3:  colors.append("#3fb950")
        elif v > 0.1:  colors.append("#f0883e")
        elif v > 0.02: colors.append("#58a6ff")
        else:           colors.append("#484f58")
    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.35)))
    ax.barh(df.index, df["IV"], color=colors, edgecolor=GRID)
    ax.axvline(x=0.02, color="#484f58", linestyle="--", alpha=0.7, label="Weak (0.02)")
    ax.axvline(x=0.1, color="#f0883e", linestyle="--", alpha=0.7, label="Medium (0.1)")
    ax.axvline(x=0.3, color="#3fb950", linestyle="--", alpha=0.7, label="Strong (0.3)")
    dark_style(ax, "Information Value (IV)")
    ax.set_xlabel("IV")
    ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TXT, fontsize=9)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct)[0]
    n = ct.sum().sum(); md = min(ct.shape) - 1
    return np.sqrt(chi2 / (n * md)) if md > 0 else 0.0


def calc_iv(data, feature, target):
    g = data.groupby(feature)[target].agg(["sum", "count"])
    g.columns = ["ev", "tot"]; g["nev"] = g["tot"] - g["ev"]
    te, tne = g["ev"].sum(), g["nev"].sum()
    g["pe"] = (g["ev"] / te).clip(1e-4); g["pn"] = (g["nev"] / tne).clip(1e-4)
    g["woe"] = np.log(g["pn"] / g["pe"]); g["iv"] = (g["pn"] - g["pe"]) * g["woe"]
    return g["iv"].sum()


def drift_label(ks, p):
    if p >= 0.05:   return "✅ NONE"
    if ks < 0.05:   return "🟡 Statistical Only"
    if ks < 0.10:   return "🟠 Mild Drift"
    return "🔴 Severe Drift"


# ══════════════════════════════════════════════════════════════
# 🚀 MAIN PIPELINE
# ══════════════════════════════════════════════════════════════
def full_senior_data_pipeline(path, train_name, test_name, target_col, id_col=None):

    R = Reporter()
    actions = []

    train = pd.read_csv(f"{path}/{train_name}")
    test  = pd.read_csv(f"{path}/{test_name}")

    feats = [c for c in train.columns if c not in [target_col, id_col]]
    THR = 25
    num_cols = [c for c in feats if train[c].nunique() > THR]
    cat_cols = [c for c in feats if train[c].nunique() <= THR]
    nu = train[target_col].nunique()
    task = "classification" if (train[target_col].dtype == "O" or nu <= 20) else "regression"

    # ── 0. OVERVIEW ──
    R.section("📋 0. PIPELINE OVERVIEW")
    R.log(f"🚀 Pipeline started | ID: {id_col} | Target: {target_col}")
    R.log(f"📊 {len(feats)} Features: {len(num_cols)} Numeric, {len(cat_cols)} Categorical")
    R.log(f"🎯 Task: {task.upper()} (target has {nu} unique values)", cls="ok")
    R.end()

    # ── 1. MEMORY & SHAPE ──
    R.section("🌟 1. MEMORY & SHAPE")
    for nm, df in zip(["Train", "Test"], [train, test]):
        mem = df.memory_usage(deep=True).sum() / 1e6
        R.log(f"📐 {nm}: {df.shape[0]:,} × {df.shape[1]} | 💾 {mem:.1f} MB")
    R.end()

    # ── 2. GENERAL AUDIT ──
    R.section("🌟 2. GENERAL DATA AUDIT")
    role = {c: ("Numeric" if c in num_cols else "Categorical") for c in feats}
    met = {
        "Meta": pd.DataFrame({"Dtype": train[feats].dtypes.astype(str), "Role": pd.Series(role)}),
        "N-Unique": pd.concat([train[feats].nunique().rename("Train"),
                                test[feats].nunique().rename("Test")], axis=1),
        "Null Pct": pd.concat([(train[feats].isnull().mean() * 100).rename("Train"),
                                (test[feats].isnull().mean() * 100).rename("Test")], axis=1),
        "Top Val Pct": pd.concat([
            train[feats].apply(lambda x: x.value_counts(normalize=True).iloc[0] * 100
                                if not x.dropna().empty else 0).rename("Train"),
            test[feats].apply(lambda x: x.value_counts(normalize=True).iloc[0] * 100
                               if not x.dropna().empty else 0).rename("Test")], axis=1),
    }
    da = pd.concat(met.values(), axis=1, keys=met.keys())
    da[("Diff", "Null Gap")] = (da[("Null Pct", "Test")] - da[("Null Pct", "Train")]).round(2)
    da = da.sort_values(by=[("Meta", "Role"), ("N-Unique", "Train")], ascending=[False, False])
    R.table(da.style
        .background_gradient(cmap="YlGnBu",
            subset=[c for c in da.columns if c[0] in ["N-Unique", "Null Pct", "Top Val Pct"]])
        .background_gradient(cmap="Reds", subset=[("Diff", "Null Gap")])
        .format(precision=2))
    high_null = [c for c in feats if train[c].isnull().mean() > 0.3]
    if high_null:
        actions.append(f"🗑️ >30% null columns: {high_null} → drop or advanced imputation")
    R.end()

    # ── 3. TARGET DISTRIBUTION ──
    R.section("🌟 3. TARGET DISTRIBUTION")
    if task == "classification":
        vc = train[target_col].value_counts()
        vcp = train[target_col].value_counts(normalize=True) * 100
        dt = pd.DataFrame({"Count": vc, "Pct (%)": vcp, "Bar": vcp})
        R.table(dt.style.bar(subset=["Bar"], color="#636efa")
                .format({"Pct (%)": "{:.2f}", "Bar": "{:.1f}"}))
        ir = vc.min() / vc.max()
        if ir < 0.2:
            R.log(f"⚠️ Severe imbalance! Min/Max ratio: {ir:.3f}", cls="critical")
            actions.append("⚖️ Severe imbalance → SMOTE / focal loss / undersampling")
        elif ir < 0.5:
            R.log(f"⚠️ Moderate imbalance. Min/Max ratio: {ir:.3f}", cls="warn")
            actions.append("⚖️ Moderate imbalance → class_weight='balanced'")
        else:
            R.log(f"✅ Balanced distribution. Min/Max ratio: {ir:.3f}", cls="ok")
    else:
        desc = train[target_col].describe().to_frame().T
        desc["skew"] = skew(train[target_col].dropna())
        R.table(desc.style.format(precision=3))
        if abs(desc["skew"].iloc[0]) > 1:
            R.log("⚠️ Target is skewed → consider log1p transform", cls="warn")
            actions.append("📐 Target skewed → np.log1p()")
    R.plot(plot_target_dist(train, target_col, task), "target_distribution")
    R.end()

    # ── 4. DUPLICATES & CONSTANT COLUMNS ──
    R.section("🌟 4. DUPLICATES & CONSTANT COLUMNS")
    dup = train.duplicated().sum()
    if dup:
        R.log(f"⚠️ {dup:,} duplicate rows ({dup / len(train) * 100:.2f}%)", cls="warn")
        actions.append(f"🔄 {dup} duplicates → drop_duplicates()")
    else:
        R.log("✅ No duplicate rows.", cls="ok")

    const = []
    for c in feats:
        tp = train[c].value_counts(normalize=True).iloc[0] * 100
        if tp >= 99.5:
            const.append({"Feature": c, "Top%": tp, "Status": "🔴 CONSTANT"})
        elif tp >= 95:
            const.append({"Feature": c, "Top%": tp, "Status": "🟡 QUASI-CONSTANT"})
    if const:
        dc = pd.DataFrame(const).set_index("Feature")
        R.table(dc.style.map(
            lambda x: "color:#f85149;font-weight:bold" if "CONSTANT" in str(x)
            else "color:#f0883e", subset=["Status"]).format({"Top%": "{:.2f}"}))
        drop_c = [r["Feature"] for r in const if r["Top%"] >= 99.5]
        if drop_c:
            actions.append(f"🗑️ Constant columns: {drop_c} → drop")
    else:
        R.log("✅ No constant or quasi-constant columns.", cls="ok")
    R.end()

    # ── 5. STATISTICAL RISK & DRIFT ──
    R.section("🌟 5. STATISTICAL RISK & DRIFT")
    if num_cols:
        stats = []
        for c in num_cols:
            tr, ts = train[c].dropna(), test[c].dropna()
            q1, q3 = tr.quantile(0.25), tr.quantile(0.75)
            iqr = q3 - q1
            out = ((tr < q1 - 1.5 * iqr) | (tr > q3 + 1.5 * iqr)).mean() * 100
            ks, p = ks_2samp(tr, ts)
            stats.append({"Column": c, "Skew": skew(tr), "Outlier (%)": out,
                          "KS-Stat": ks, "P-Value": p, "Drift": drift_label(ks, p)})

        vd = train[num_cols].dropna()
        vr = [variance_inflation_factor(vd.values, i) for i in range(len(num_cols))]

        tmi = train[num_cols + [target_col]].dropna()
        tmi[target_col] = safe_encode_target(tmi[target_col])
        mi_fn = mutual_info_classif if task == "classification" else mutual_info_regression
        mi = mi_fn(tmi[num_cols], tmi[target_col], random_state=42)

        da2 = pd.DataFrame(stats).set_index("Column")
        da2["VIF"] = vr; da2["MI_Score"] = mi

        R.table(da2.style
            .background_gradient(cmap="OrRd", subset=["VIF", "Skew"])
            .background_gradient(cmap="YlGn", subset=["MI_Score"])
            .map(lambda x: "color:red;font-weight:bold" if "Severe" in str(x)
                 else ("color:orange" if "Mild" in str(x) or "Statistical" in str(x)
                        else "color:green"), subset=["Drift"])
            .format(precision=3))

        real_drift = [s["Column"] for s in stats if "Severe" in s["Drift"] or "Mild" in s["Drift"]]
        if real_drift:
            actions.append(f"📉 Real drift detected: {real_drift}")
        high_vif = [num_cols[i] for i, v in enumerate(vr) if v > 10]
        if high_vif:
            actions.append(f"📊 VIF > 10 (multicollinearity): {high_vif} → PCA or drop")
        low_mi = [num_cols[i] for i, v in enumerate(mi) if v < 0.01]
        if low_mi:
            actions.append(f"📉 MI ≈ 0 (no information): {low_mi}")

        fig_nd = plot_numeric_dist(train, test, num_cols)
        if fig_nd:
            R.plot(fig_nd, "numeric_distributions")
    else:
        R.log("ℹ️ No numeric columns found.", cls="ok")
    R.end()

    # ── 6. NUMERIC CORRELATION ──
    R.section("🌟 6. NUMERIC CORRELATION")
    if len(num_cols) >= 2:
        corr = train[num_cols].corr()
        hcp = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                r = corr.iloc[i, j]
                if abs(r) > 0.85:
                    hcp.append({"Feature_1": num_cols[i], "Feature_2": num_cols[j],
                                "|r|": abs(r), "Direction": "➕ Positive" if r > 0 else "➖ Negative"})
        if hcp:
            dhc = pd.DataFrame(hcp).sort_values("|r|", ascending=False).reset_index(drop=True)
            R.log(f"⚠️ {len(dhc)} pairs with |r| > 0.85:", cls="warn")
            R.table(dhc.style.background_gradient(cmap="Reds", subset=["|r|"]).format({"|r|": "{:.4f}"}))
            actions.append(f"🔗 {len(dhc)} highly correlated pairs → drop one or PCA")
        else:
            R.log("✅ No pairs with |r| > 0.85.", cls="ok")

        tc = safe_encode_target(train[target_col].copy())
        tcr = train[num_cols].corrwith(tc).abs().sort_values(ascending=False)
        dtc = pd.DataFrame({"Feature": tcr.index,
                            f"|r| with {target_col}": tcr.values}).reset_index(drop=True)
        R.log(f"🎯 Feature ↔ {target_col} correlation:")
        R.table(dtc.style.bar(subset=[f"|r| with {target_col}"], color="#3fb950")
                .format({f"|r| with {target_col}": "{:.4f}"}))
        R.plot(plot_correlation_heatmap(train, num_cols, target_col), "correlation_heatmap")
    else:
        R.log("ℹ️ Not enough numeric columns for correlation analysis.")
    R.end()

    # ── 7. CRAMER'S V ──
    R.section("🌟 7. CRAMER'S V — CATEGORICAL CORRELATION")
    if len(cat_cols) >= 2:
        hcv = []
        for i in range(len(cat_cols)):
            for j in range(i + 1, len(cat_cols)):
                v = cramers_v(train[cat_cols[i]].fillna("_NA_"), train[cat_cols[j]].fillna("_NA_"))
                if v > 0.5:
                    hcv.append({"Feature_1": cat_cols[i], "Feature_2": cat_cols[j],
                                "Cramer_V": v, "Risk": "🔴 Redundant" if v > 0.8 else "🟡 High"})
        if hcv:
            dcv = pd.DataFrame(hcv).sort_values("Cramer_V", ascending=False).reset_index(drop=True)
            R.log(f"⚠️ {len(dcv)} pairs with Cramer's V > 0.5:", cls="warn")
            R.table(dcv.style.background_gradient(cmap="Reds", subset=["Cramer_V"])
                    .format({"Cramer_V": "{:.4f}"}))
            red = dcv[dcv["Cramer_V"] > 0.8]
            if len(red):
                actions.append(f"🔗 Redundant categorical pairs (V>0.8): "
                               f"{list(zip(red['Feature_1'], red['Feature_2']))}")
        else:
            R.log("✅ No pairs with Cramer's V > 0.5.", cls="ok")

        tcv = {}
        tgf = train[target_col].fillna("_NA_")
        for c in cat_cols:
            tcv[c] = cramers_v(train[c].fillna("_NA_"), tgf)
        dtcv = (pd.DataFrame.from_dict(tcv, orient="index", columns=[f"V with {target_col}"])
                .sort_values(f"V with {target_col}", ascending=False))
        R.log(f"🎯 Categorical ↔ {target_col} Cramer's V:")
        R.table(dtcv.style.bar(subset=[f"V with {target_col}"], color="#636efa").format(precision=4))
        R.plot(plot_cramers_heatmap(train, cat_cols, target_col), "cramers_v_heatmap")
    else:
        R.log("ℹ️ Not enough categorical columns.")
    R.end()

    # ── 8. HIERARCHICAL DEPENDENCY ──
    R.section("🌟 8. HIERARCHICAL DEPENDENCY DETECTION")
    sentinels = ["No internet service", "No phone service"]
    hier = []
    for col in cat_cols:
        for sv in sentinels:
            mask = train[col] == sv
            sc = mask.sum()
            if sc == 0:
                continue
            for pcol in cat_cols:
                if pcol == col:
                    continue
                for pval in train[pcol].unique():
                    if (train[pcol] == pval).sum() != sc:
                        continue
                    pm = train[pcol] == pval
                    if (mask & pm).sum() == sc:
                        tgt = safe_encode_target(train[target_col].copy())
                        hier.append({"Child": col, "Sentinel": sv, "Parent": pcol,
                                     "Parent_Value": pval, "Count": sc,
                                     f"{target_col}_Rate%": tgt[mask].mean() * 100})
    if hier:
        dh = pd.DataFrame(hier)
        n_children = dh["Child"].nunique()
        R.log(f"⚠️ {len(dh)} hierarchical dependencies found! "
              f"{n_children} columns depend on a parent:", cls="critical")
        R.table(dh.style.format({f"{target_col}_Rate%": "{:.2f}"})
                .background_gradient(cmap="OrRd", subset=["Count"]))
        R.log("💡 Action: convert sentinel values to NaN or merge with parent column.", cls="warn")
        affected = list(dh["Child"].unique())
        actions.append(f"🔗 Hierarchical dependency: {affected} → replace sentinel with NaN or encode")
    else:
        R.log("✅ No hierarchical dependencies found.", cls="ok")
    R.end()

    # ── 9. DERIVED FEATURE RELATIONSHIPS ──
    R.section("🌟 9. DERIVED FEATURE RELATIONSHIPS")
    if len(num_cols) >= 2:
        derived = []
        cc = num_cols[:10]
        for i in range(len(cc)):
            for j in range(i + 1, len(cc)):
                for k in range(len(cc)):
                    if k in (i, j):
                        continue
                    a, b, c_ = train[cc[i]], train[cc[j]], train[cc[k]]
                    prod = a * b
                    if prod.std() > 0 and c_.std() > 0:
                        rp = prod.corr(c_)
                        if abs(rp) > 0.90:
                            derived.append({"Relationship": f"{cc[i]} × {cc[j]}",
                                            "Target_Col": cc[k], "|r|": abs(rp), "Type": "Product"})
                    if (b.abs() > 1e-9).all():
                        rat = a / b
                        if rat.std() > 0 and c_.std() > 0:
                            rr = rat.corr(c_)
                            if abs(rr) > 0.90:
                                derived.append({"Relationship": f"{cc[i]} / {cc[j]}",
                                                "Target_Col": cc[k], "|r|": abs(rr), "Type": "Ratio"})
        if derived:
            dd = pd.DataFrame(derived).sort_values("|r|", ascending=False).reset_index(drop=True)
            R.log(f"⚠️ {len(dd)} derived relationships detected:", cls="critical")
            R.table(dd.style.background_gradient(cmap="Reds", subset=["|r|"]).format({"|r|": "{:.4f}"}))
            R.log("💡 Drop the derived column or use residual (actual - predicted) instead.", cls="warn")
            actions.append(f"🔗 Derived features: {dd['Target_Col'].unique().tolist()} → use residual")

            for _, row in dd.iterrows():
                sep = " × " if "×" in row["Relationship"] else " / "
                parts = row["Relationship"].split(sep)
                if len(parts) == 2 and all(p.strip() in train.columns for p in parts):
                    p1, p2 = parts[0].strip(), parts[1].strip()
                    pred = train[p1] * train[p2] if "×" in row["Relationship"] else train[p1] / train[p2]
                    residual = train[row["Target_Col"]] - pred
                    tgt_r = safe_encode_target(train[target_col].copy())
                    valid = pd.notna(residual) & pd.notna(tgt_r)
                    if valid.sum() > 100:
                        mi_r = (mutual_info_classif if task == "classification"
                                else mutual_info_regression)(
                            residual[valid].values.reshape(-1, 1),
                            tgt_r[valid].values,
                            random_state=42
                        )[0]
                        R.log(f"   📊 Residual ({row['Target_Col']} - {row['Relationship']}) MI: {mi_r:.4f}")
        else:
            R.log("✅ No derived feature relationships detected.", cls="ok")
    else:
        R.log("ℹ️ Not enough numeric columns.")
    R.end()

    # ── 10. CATEGORICAL OVERLAP ──
    R.section("🌟 10. CATEGORICAL OVERLAP ANALYSIS")
    if cat_cols:
        rc = []
        for c in cat_cols:
            trs = set(train[c].dropna().unique())
            tss = set(test[c].dropna().unique())
            ov = (len(tss & trs) / len(tss) * 100) if tss else 100
            rc.append({"Column": c, "Train_Uniq": len(trs), "Test_Uniq": len(tss),
                       "Overlap%": ov, "Only_in_Test": len(tss - trs)})
        drc = pd.DataFrame(rc).set_index("Column")
        R.table(drc.style.map(lambda x: "color:red" if x < 100 else "color:green",
                              subset=["Overlap%"]))
        lov = [r["Column"] for r in rc if r["Overlap%"] < 100]
        if lov:
            actions.append(f"🔀 New categories in test: {lov} → add unknown handling")
    else:
        R.log("ℹ️ No categorical columns.", cls="ok")
    R.end()

    # ── 11. TARGET BIVARIATE ──
    R.section("🌟 11. TARGET BIVARIATE ANALYSIS")
    if cat_cols:
        tmp = train.copy()
        if tmp[target_col].dtype == "O":
            le = LabelEncoder()
            le.fit(tmp[target_col].astype(str))
            tmp[target_col] = le.transform(tmp[target_col].astype(str))
            tl = le.classes_[1] if len(le.classes_) > 1 else "Positive"
        else:
            tl = "Target"
        bv = []
        for c in cat_cols:
            grp = tmp.groupby(c)[target_col]
            if task == "classification":
                for idx, val in (grp.mean() * 100).items():
                    bv.append({"Feature": c, "Category": idx,
                               f"{tl}_Rate%": val, "Count": grp.count().get(idx, 0)})
            else:
                for idx, val in grp.mean().items():
                    bv.append({"Feature": c, "Category": idx,
                               f"Mean_{target_col}": val, "Count": grp.count().get(idx, 0)})
        dbv = pd.DataFrame(bv).set_index(["Feature", "Category"])
        rc2 = [c for c in dbv.columns if "Rate" in c or "Mean" in c][0]
        R.table(dbv.style.background_gradient(cmap="RdYlGn_r", subset=[rc2]).format(precision=2))
        R.plot(plot_bivariate_top(train, target_col, cat_cols, task), "bivariate_top")
    else:
        R.log("ℹ️ No categorical columns for bivariate analysis.")
    R.end()

    # ── 12. INFORMATION VALUE ──
    R.section("🌟 12. INFORMATION VALUE (IV)")
    if task == "classification" and cat_cols:
        tmp_iv = train.copy()
        tmp_iv[target_col] = safe_encode_target(tmp_iv[target_col].copy())
        ivr = {}
        for c in cat_cols:
            ivr[c] = calc_iv(tmp_iv, c, target_col)
        div = (pd.DataFrame.from_dict(ivr, orient="index", columns=["IV"])
               .sort_values("IV", ascending=False))

        def iv_lbl(x):
            if x > 0.5:   return "🔴 Suspicious"
            if x > 0.3:   return "🟢 Strong"
            if x > 0.1:   return "🟡 Medium"
            if x > 0.02:  return "🔵 Weak"
            return "⚪ Useless"

        div["Strength"] = div["IV"].apply(iv_lbl)
        R.table(div.style.bar(subset=["IV"], color="#636efa").format({"IV": "{:.4f}"}))
        R.plot(plot_iv_chart(div), "information_value")

        sus = div[div["IV"] > 0.5].index.tolist()
        if sus:
            R.log(f"⚠️ IV > 0.5 — potential leakage risk: {sus}", cls="critical")
            actions.append(f"🚨 Suspicious IV: {sus} → check for leakage")
        useless = div[div["IV"] < 0.02].index.tolist()
        if useless:
            actions.append(f"📉 IV < 0.02 (no predictive power): {useless}")
    else:
        R.log("ℹ️ IV is only computed for binary classification tasks.")
    R.end()

    # ── 13. ADVERSARIAL VALIDATION & FEATURE IMPORTANCE ──
    R.section("🌟 13. ADVERSARIAL VALIDATION & FEATURE IMPORTANCE")

    adf = pd.concat([train[feats], test[feats]], axis=0)
    ya = [0] * len(train) + [1] * len(test)
    for c in adf.select_dtypes(["object", "category"]).columns:
        adf[c] = pd.factorize(adf[c])[0]

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    auc = cross_val_score(rf, adf.fillna(-1), ya, cv=5, scoring="roc_auc").mean()

    if auc > 0.8:
        R.log(f"🕵️ Adversarial AUC: {auc:.4f} — 🔴 Severe drift / leakage!", cls="critical")
        actions.append(f"🚨 Adversarial AUC = {auc:.4f} → investigate leakage")
    elif auc > 0.6:
        R.log(f"🕵️ Adversarial AUC: {auc:.4f} — ⚠️ Mild drift detected.", cls="warn")
        actions.append(f"⚠️ Adversarial AUC = {auc:.4f} → check drifted columns")
    else:
        R.log(f"🕵️ Adversarial AUC: {auc:.4f} — ✅ Train/Test distributions are safe.", cls="ok")

    timp = train[feats + [target_col]].dropna()
    for c in timp.select_dtypes(["object", "category"]).columns:
        timp[c] = pd.Series(LabelEncoder().fit_transform(timp[c].astype(str)),
                            index=timp.index)

    rf_imp = (RandomForestClassifier if task == "classification" else RandomForestRegressor)(
        n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf_imp.fit(timp.drop(columns=[target_col]), timp[target_col])

    imp = (pd.DataFrame({"Feature": feats, "Importance": rf_imp.feature_importances_})
           .sort_values("Importance", ascending=False).reset_index(drop=True))
    imp["Cumulative"] = imp["Importance"].cumsum()
    imp["Rank"] = range(1, len(imp) + 1)

    R.table(imp.style.bar(subset=["Importance"], color="#636efa")
            .background_gradient(cmap="YlGn", subset=["Cumulative"])
            .format({"Importance": "{:.4f}", "Cumulative": "{:.4f}"}))
    R.plot(plot_feature_importance(imp), "feature_importance")

    low_imp = set(imp[imp["Importance"] < 0.005]["Feature"])
    high_dom = set(c for c in cat_cols if train[c].value_counts(normalize=True).iloc[0] > 0.90)
    truly_useless = low_imp & high_dom
    if truly_useless:
        R.log(f"🗑️ Importance < 0.005 AND dominant > 90%: {truly_useless} → safe to drop", cls="warn")
        actions.append(f"🗑️ Double-filtered useless features: {truly_useless}")

    low_all = imp[imp["Importance"] < 0.01]["Feature"].tolist()
    if low_all:
        actions.append(f"📉 Importance < 0.01: {low_all[:5]}{'...' if len(low_all) > 5 else ''}")

    top80 = imp[imp["Cumulative"] <= 0.80]["Feature"].tolist()
    R.log(f"📊 Top 80% information carried by {len(top80)} features: "
          f"{top80[:8]}{'...' if len(top80) > 8 else ''}")
    R.end()

    # ── 14. MISSING VALUE PATTERNS ──
    R.section("🌟 14. MISSING VALUE PATTERNS")
    nc = [c for c in feats if train[c].isnull().any()]
    if nc:
        ns = []
        for c in nc:
            null_pct = train[c].isnull().mean() * 100
            tgt2 = safe_encode_target(train[target_col].copy())
            if task == "classification":
                rate_null = tgt2[train[c].isnull()].mean() * 100 if train[c].isnull().any() else 0
                rate_present = tgt2[train[c].notnull()].mean() * 100
                ns.append({"Feature": c, "Null%": null_pct,
                           "Rate_if_NULL%": rate_null, "Rate_if_PRESENT%": rate_present,
                           "Gap": abs(rate_null - rate_present)})
            else:
                mean_null = train.loc[train[c].isnull(), target_col].mean()
                mean_present = train.loc[train[c].notnull(), target_col].mean()
                ns.append({"Feature": c, "Null%": null_pct,
                           "Mean_if_NULL": mean_null, "Mean_if_PRESENT": mean_present,
                           "Gap": abs(mean_null - mean_present) if pd.notna(mean_null) else 0})
        dn = pd.DataFrame(ns).set_index("Feature").sort_values("Gap", ascending=False)
        R.table(dn.style.background_gradient(cmap="OrRd", subset=["Gap", "Null%"]).format(precision=2))
        informative = dn[dn["Gap"] > 5].index.tolist()
        if informative:
            R.log(f"💡 Informative nulls detected: {informative} → create is_null flags!", cls="warn")
            actions.append(f"🏷️ Create is_null binary flags: {informative}")
    else:
        R.log("✅ No missing values found.", cls="ok")
    R.end()

    # ── 15. PLAYGROUND SERIES NOTE ──
    R.section("🌟 15. PLAYGROUND SERIES — ORIGINAL DATA NOTE")
    R.log("📌 Playground Series competitions are based on existing public datasets.")
    R.log("📌 This dataset originates from: IBM Telco Customer Churn (available on Kaggle).")
    R.log("📌 Augmenting with original data typically yields +0.001–0.005 CV score improvement.")
    R.log("💡 Use pd.concat([train, original], ignore_index=True) with a 'source' flag column.", cls="warn")
    actions.append("📦 Augment with original Telco dataset (add source flag)")
    R.end()

    # ── 16. ACTION SUMMARY ──
    R.section("🎯 16. ACTION SUMMARY — WHAT TO DO NEXT")
    if actions:
        for i, a in enumerate(actions, 1):
            cl = "critical" if any(x in a for x in ["🚨", "🔴"]) else ("warn" if "⚠" in a else "")
            R.log(f"  {i}. {a}", cls=cl)
    else:
        R.log("✅ No critical issues found. Ready for modeling.", cls="ok")

    R.log("")
    R.log("📌 Recommended next steps:", cls="ok")
    for s in [
        "1️⃣  Handle sentinel values in hierarchical columns (replace with NaN)",
        "2️⃣  Augment with original dataset (add source flag)",
        "3️⃣  Feature engineering: residuals, interactions, aggregations",
        "4️⃣  Encoding strategy: target encoding / CatBoost native categorical",
        "5️⃣  Baseline model: LightGBM + StratifiedKFold (5-fold CV)",
        "6️⃣  Hyperparameter tuning with Optuna",
        "7️⃣  Final ensemble: LightGBM + XGBoost + CatBoost blending",
    ]:
        R.log(f"  {s}")
    R.end()

    # ── SAVE ──
    R.save(
        html_path="/kaggle/working/pipeline_report.html",
        md_path="/kaggle/working/pipeline_report.md",
    )

    return train, test, target_col, id_col


# ═══════════════════════════
# RUN
# ═══════════════════════════
train, test, target_col, id_col = full_senior_data_pipeline(
    "/kaggle/input/competitions/playground-series-s6e3/",
    "train.csv", "test.csv", "Churn", "id"
)
```