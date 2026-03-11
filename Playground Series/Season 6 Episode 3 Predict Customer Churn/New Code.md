```python
!pip install tabulate -q

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, ks_2samp, chi2_contingency, spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from IPython.display import display, Image
import lightgbm as lgb
import base64, os, warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="darkgrid", palette="muted")
PLOT_DIR = "/kaggle/working/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def safe_encode_target(series):
    if series.dtype == "O":
        return pd.Series(LabelEncoder().fit_transform(series.astype(str)), index=series.index)
    return series


class Reporter:
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
    </style></head><body>
    <h1 style="text-align:center;color:#58a6ff">🚀 SENIOR DATA PIPELINE REPORT</h1>"""

    def __init__(self, plot_dir=PLOT_DIR):
        self.h = [self.CSS]; self.m = ["# 🚀 SENIOR DATA PIPELINE REPORT\n\n"]
        self.plot_dir = plot_dir; self.plot_count = 0

    def log(self, text, cls=""):
        print(text)
        c = f" {cls}" if cls else ""
        self.h.append(f'<div class="info-line{c}">{text}</div>')
        b = "**" if cls in ("critical","warn") else ""
        self.m.append(f"> {b}{text}{b}\n\n")

    def decision(self, text, positive=True):
        cn = "decision-yes" if positive else "decision-no"
        px = "✅" if positive else "⚠️"
        full = f"{px} DECISION: {text}"
        print(f"  {full}")
        self.h.append(f'<div class="decision {cn}">{full}</div>')
        self.m.append(f"> **{full}**\n\n")

    def section(self, title):
        print(f"\n{'═'*80}\n{title}\n{'═'*80}")
        self.h.append(f'<div class="section"><h2>{title}</h2>')
        self.m.append(f"\n---\n\n## {title}\n\n")

    def end(self): self.h.append("</div>")

    def table(self, styled):
        display(styled); self.h.append(styled.to_html())
        df = styled.data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(filter(None,(str(x) for x in c))) for c in df.columns]
        if isinstance(df.index, pd.MultiIndex): df = df.reset_index()
        try: self.m.append(f"\n{df.to_markdown(floatfmt='.4f')}\n\n")
        except: self.m.append(f"\n```\n{df.to_string()}\n```\n\n")

    def plot(self, fig, title="plot"):
        self.plot_count += 1
        fname = f"{self.plot_count:02d}_{title.replace(' ','_').lower()}.png"
        fpath = os.path.join(self.plot_dir, fname)
        fig.savefig(fpath, dpi=150, bbox_inches="tight", facecolor="#0d1117", edgecolor="none")
        plt.close(fig)
        display(Image(filename=fpath))
        with open(fpath,"rb") as f: b64 = base64.b64encode(f.read()).decode()
        self.h.append(f'<div class="plot-container"><img src="data:image/png;base64,{b64}" alt="{title}"></div>')
        self.m.append(f"\n![{title}](plots/{fname})\n\n")

    def save(self, html_path, md_path):
        self.h.append("</body></html>")
        with open(html_path,"w",encoding="utf-8") as f: f.write("\n".join(self.h))
        print(f"\n✅ HTML saved  : {html_path}")
        with open(md_path,"w",encoding="utf-8") as f: f.write("\n".join(self.m))
        print(f"✅ Markdown saved: {md_path}")
        print(f"📁 Plots         : {self.plot_dir}/ ({self.plot_count} files)")


# ── PLOT HELPERS ──
DARK="#0d1117"; CARD="#161b22"; BLUE="#58a6ff"; TXT="#c9d1d9"; GRID="#30363d"

def dark_style(ax, title=""):
    ax.set_facecolor(CARD); ax.figure.set_facecolor(DARK)
    ax.title.set_color(BLUE); ax.xaxis.label.set_color(TXT); ax.yaxis.label.set_color(TXT)
    ax.tick_params(colors=TXT)
    for s in ax.spines.values(): s.set_color(GRID)
    if title: ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

def plot_target_dist(train, target_col, task):
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    if task=="classification":
        vc=train[target_col].value_counts()
        cl=["#3fb950","#f85149"] if len(vc)==2 else sns.color_palette("muted",len(vc))
        axes[0].bar(vc.index.astype(str),vc.values,color=cl,edgecolor=GRID); dark_style(axes[0],f"{target_col} — Count"); axes[0].set_ylabel("Count")
        axes[1].pie(vc.values,labels=vc.index.astype(str),autopct="%1.1f%%",colors=cl,textprops={"color":TXT},wedgeprops={"edgecolor":GRID})
        axes[1].set_facecolor(DARK); axes[1].set_title(f"{target_col} — Ratio",color=BLUE,fontsize=14,fontweight="bold")
    else:
        axes[0].hist(train[target_col].dropna(),bins=50,color=BLUE,edgecolor=GRID,alpha=0.8); dark_style(axes[0],f"{target_col} — Distribution")
        sns.boxplot(x=train[target_col].dropna(),ax=axes[1],color=BLUE); dark_style(axes[1],f"{target_col} — Boxplot")
    fig.set_facecolor(DARK); fig.tight_layout(); return fig

def plot_correlation_heatmap(train, num_cols, target_col):
    data=train[num_cols].copy(); data[target_col]=safe_encode_target(train[target_col].copy())
    corr=data.corr(); mask=np.triu(np.ones_like(corr,dtype=bool),k=1); n=len(corr)
    fig,ax=plt.subplots(figsize=(max(8,n),max(6,n*0.8)))
    sns.heatmap(corr,mask=mask,annot=True,fmt=".2f",cmap="RdBu_r",center=0,vmin=-1,vmax=1,ax=ax,linewidths=0.5,linecolor=GRID,cbar_kws={"shrink":0.8})
    dark_style(ax,"Correlation Matrix"); ax.tick_params(axis="x",rotation=45); fig.tight_layout(); return fig

def plot_cramers_heatmap(train, cat_cols, target_col):
    ac=cat_cols+[target_col]; n=len(ac); mat=pd.DataFrame(np.zeros((n,n)),index=ac,columns=ac)
    for i in range(n):
        for j in range(i,n):
            v=cramers_v(train[ac[i]].fillna("_NA_"),train[ac[j]].fillna("_NA_")); mat.iloc[i,j]=v; mat.iloc[j,i]=v
    mask=np.triu(np.ones_like(mat,dtype=bool),k=1)
    fig,ax=plt.subplots(figsize=(max(10,n*0.9),max(8,n*0.7)))
    sns.heatmap(mat.astype(float),mask=mask,annot=True,fmt=".2f",cmap="YlOrRd",vmin=0,vmax=1,ax=ax,linewidths=0.5,linecolor=GRID,cbar_kws={"shrink":0.8})
    dark_style(ax,"Cramer's V Matrix"); ax.tick_params(axis="x",rotation=45); fig.tight_layout(); return fig

def plot_feature_importance(imp_df, top_n=15):
    df=imp_df.head(top_n).sort_values("Importance"); fig,ax=plt.subplots(figsize=(10,max(4,top_n*0.35)))
    bars=ax.barh(df["Feature"],df["Importance"],color=BLUE,edgecolor=GRID)
    for bar,val in zip(bars,df["Importance"]): ax.text(bar.get_width()+0.002,bar.get_y()+bar.get_height()/2,f"{val:.4f}",va="center",color=TXT,fontsize=10)
    dark_style(ax,f"Top {top_n} Feature Importance"); ax.set_xlabel("Importance"); fig.tight_layout(); return fig

def plot_bivariate_top(train, target_col, cat_cols, task, top_n=6):
    tmp=train.copy(); tmp[target_col]=safe_encode_target(tmp[target_col].copy())
    cols=cat_cols[:top_n]; nc=3; nr=(len(cols)+nc-1)//nc
    fig,axes=plt.subplots(nr,nc,figsize=(15,nr*4)); axes=axes.flatten() if nr*nc>1 else [axes]
    for i,col in enumerate(cols):
        grp=tmp.groupby(col)[target_col].mean()*(100 if task=="classification" else 1); grp=grp.sort_values(ascending=False)
        axes[i].barh(grp.index.astype(str),grp.values,color=sns.color_palette("RdYlGn_r",len(grp)),edgecolor=GRID); dark_style(axes[i],col)
        axes[i].set_xlabel("Target Rate (%)" if task=="classification" else f"Mean {target_col}")
    for j in range(len(cols),len(axes)): axes[j].set_visible(False)
    fig.suptitle("Bivariate — Categorical vs Target",color=BLUE,fontsize=16,fontweight="bold"); fig.set_facecolor(DARK); fig.tight_layout(rect=[0,0,1,0.95]); return fig

def plot_numeric_dist(train, test, num_cols):
    n=len(num_cols)
    if n==0: return None
    nc=min(3,n); nr=(n+nc-1)//nc; fig,axes=plt.subplots(nr,nc,figsize=(15,nr*3.5)); axes=np.array(axes).flatten() if n>1 else [axes]
    for i,col in enumerate(num_cols):
        axes[i].hist(train[col].dropna(),bins=50,alpha=0.6,color="#58a6ff",label="Train",edgecolor=GRID)
        axes[i].hist(test[col].dropna(),bins=50,alpha=0.5,color="#f0883e",label="Test",edgecolor=GRID)
        dark_style(axes[i],col); axes[i].legend(fontsize=9,facecolor=CARD,edgecolor=GRID,labelcolor=TXT)
    for j in range(n,len(axes)): axes[j].set_visible(False)
    fig.suptitle("Numeric Distributions — Train vs Test",color=BLUE,fontsize=16,fontweight="bold"); fig.set_facecolor(DARK); fig.tight_layout(rect=[0,0,1,0.95]); return fig

def plot_iv_chart(div):
    df=div.sort_values("IV",ascending=True)
    colors=["#f85149" if v>0.5 else "#3fb950" if v>0.3 else "#f0883e" if v>0.1 else "#58a6ff" if v>0.02 else "#484f58" for v in df["IV"]]
    fig,ax=plt.subplots(figsize=(10,max(4,len(df)*0.35))); ax.barh(df.index,df["IV"],color=colors,edgecolor=GRID)
    ax.axvline(x=0.02,color="#484f58",linestyle="--",alpha=0.7,label="Weak"); ax.axvline(x=0.1,color="#f0883e",linestyle="--",alpha=0.7,label="Medium"); ax.axvline(x=0.3,color="#3fb950",linestyle="--",alpha=0.7,label="Strong")
    dark_style(ax,"Information Value (IV)"); ax.set_xlabel("IV"); ax.legend(facecolor=CARD,edgecolor=GRID,labelcolor=TXT,fontsize=9); fig.tight_layout(); return fig

def cramers_v(x,y):
    ct=pd.crosstab(x,y); chi2=chi2_contingency(ct)[0]; n=ct.sum().sum(); md=min(ct.shape)-1
    return np.sqrt(chi2/(n*md)) if md>0 else 0.0

def calc_iv(data,feature,target):
    g=data.groupby(feature)[target].agg(["sum","count"]); g.columns=["ev","tot"]; g["nev"]=g["tot"]-g["ev"]
    te,tne=g["ev"].sum(),g["nev"].sum(); g["pe"]=(g["ev"]/te).clip(1e-4); g["pn"]=(g["nev"]/tne).clip(1e-4)
    g["woe"]=np.log(g["pn"]/g["pe"]); g["iv"]=(g["pn"]-g["pe"])*g["woe"]; return g["iv"].sum()

def drift_label(ks,p):
    if p>=0.05: return "✅ NONE"
    if ks<0.05: return "🟡 Statistical Only"
    if ks<0.10: return "🟠 Mild Drift"
    return "🔴 Severe Drift"

def full_senior_data_pipeline(path, train_name, test_name, target_col, id_col=None, cat_threshold=25, original_path=None):
    R = Reporter()
    actions = []
    results = {}

    train = pd.read_csv(f"{path}/{train_name}")
    test  = pd.read_csv(f"{path}/{test_name}")

    feats = [c for c in train.columns if c not in [target_col, id_col]]
    THR=cat_threshold
    num_cols = [c for c in feats if train[c].nunique()>THR]
    cat_cols = [c for c in feats if train[c].nunique()<=THR]
    nu = train[target_col].nunique()
    task = "classification" if (train[target_col].dtype=="O" or nu<=20) else "regression"
    y_full = safe_encode_target(train[target_col].copy())

    # ════════════════════════════════════════════════════
    # SECTION 0: OVERVIEW
    # ════════════════════════════════════════════════════
    R.section("📋 0. PIPELINE OVERVIEW")
    R.log(f"🚀 Pipeline | ID: {id_col} | Target: {target_col}")
    R.log(f"📊 {len(feats)} Features: {len(num_cols)} Numeric, {len(cat_cols)} Categorical")
    R.log(f"🎯 Task: {task.upper()} (target has {nu} unique values)", cls="ok")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 1: MEMORY & SHAPE
    # ════════════════════════════════════════════════════
    R.section("🌟 1. MEMORY & SHAPE")
    for nm,df in zip(["Train","Test"],[train,test]):
        mem=df.memory_usage(deep=True).sum()/1e6; R.log(f"📐 {nm}: {df.shape[0]:,} × {df.shape[1]} | 💾 {mem:.1f} MB")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 2: GENERAL AUDIT
    # ════════════════════════════════════════════════════
    R.section("🌟 2. GENERAL DATA AUDIT")
    role={c:("Numeric" if c in num_cols else "Categorical") for c in feats}
    met={"Meta":pd.DataFrame({"Dtype":train[feats].dtypes.astype(str),"Role":pd.Series(role)}),
         "N-Unique":pd.concat([train[feats].nunique().rename("Train"),test[feats].nunique().rename("Test")],axis=1),
         "Null Pct":pd.concat([(train[feats].isnull().mean()*100).rename("Train"),(test[feats].isnull().mean()*100).rename("Test")],axis=1),
         "Top Val Pct":pd.concat([train[feats].apply(lambda x:x.value_counts(normalize=True).iloc[0]*100 if not x.dropna().empty else 0).rename("Train"),
                                   test[feats].apply(lambda x:x.value_counts(normalize=True).iloc[0]*100 if not x.dropna().empty else 0).rename("Test")],axis=1)}
    da=pd.concat(met.values(),axis=1,keys=met.keys()); da[("Diff","Null Gap")]=(da[("Null Pct","Test")]-da[("Null Pct","Train")]).round(2)
    da=da.sort_values(by=[("Meta","Role"),("N-Unique","Train")],ascending=[False,False])
    R.table(da.style.background_gradient(cmap="YlGnBu",subset=[c for c in da.columns if c[0] in ["N-Unique","Null Pct","Top Val Pct"]]).background_gradient(cmap="Reds",subset=[("Diff","Null Gap")]).format(precision=2))
    hn=[c for c in feats if train[c].isnull().mean()>0.3]
    if hn: actions.append(f"🗑️ >30% null: {hn}")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 3: TARGET DISTRIBUTION
    # ════════════════════════════════════════════════════
    R.section("🌟 3. TARGET DISTRIBUTION")
    if task=="classification":
        vc=train[target_col].value_counts(); vcp=train[target_col].value_counts(normalize=True)*100
        dt=pd.DataFrame({"Count":vc,"Pct (%)":vcp,"Bar":vcp})
        R.table(dt.style.bar(subset=["Bar"],color="#636efa").format({"Pct (%)":"{:.2f}","Bar":"{:.1f}"}))
        ir=vc.min()/vc.max()
        if ir<0.2: R.log(f"⚠️ Severe imbalance! Ratio: {ir:.3f}",cls="critical"); actions.append("⚖️ Severe imbalance → SMOTE/focal loss")
        elif ir<0.5: R.log(f"⚠️ Moderate imbalance. Ratio: {ir:.3f}",cls="warn"); actions.append("⚖️ Moderate imbalance → class_weight='balanced'")
        else: R.log(f"✅ Balanced. Ratio: {ir:.3f}",cls="ok")
    else:
        desc=train[target_col].describe().to_frame().T; desc["skew"]=skew(train[target_col].dropna())
        R.table(desc.style.format(precision=3))
        if abs(desc["skew"].iloc[0])>1: R.log("⚠️ Skewed → log1p",cls="warn"); actions.append("📐 Target skewed → log1p")
    R.plot(plot_target_dist(train,target_col,task),"target_distribution"); R.end()

    # ════════════════════════════════════════════════════
    # SECTION 4: DUPLICATES & CONSTANTS
    # ════════════════════════════════════════════════════
    R.section("🌟 4. DUPLICATES & CONSTANT COLUMNS")
    dup=train.duplicated().sum()
    if dup: R.log(f"⚠️ {dup:,} duplicates ({dup/len(train)*100:.2f}%)",cls="warn"); actions.append(f"🔄 {dup} duplicates → drop")
    else: R.log("✅ No duplicates.",cls="ok")
    const=[]
    for c in feats:
        tp=train[c].value_counts(normalize=True).iloc[0]*100
        if tp>=99.5: const.append({"Feature":c,"Top%":tp,"Status":"🔴 CONSTANT"})
        elif tp>=95: const.append({"Feature":c,"Top%":tp,"Status":"🟡 QUASI"})
    if const:
        dc=pd.DataFrame(const).set_index("Feature")
        R.table(dc.style.map(lambda x:"color:#f85149" if "CONSTANT" in str(x) else "color:#f0883e",subset=["Status"]).format({"Top%":"{:.2f}"}))
        drc=[r["Feature"] for r in const if r["Top%"]>=99.5]
        if drc: actions.append(f"🗑️ Constants: {drc}")
    else: R.log("✅ No constant columns.",cls="ok")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 5: STATISTICAL RISK & DRIFT
    # ════════════════════════════════════════════════════
    R.section("🌟 5. STATISTICAL RISK & DRIFT")
    if num_cols:
        stats=[]
        for c in num_cols:
            tr,ts=train[c].dropna(),test[c].dropna(); q1,q3=tr.quantile(0.25),tr.quantile(0.75); iqr=q3-q1
            out=((tr<q1-1.5*iqr)|(tr>q3+1.5*iqr)).mean()*100; ks,p=ks_2samp(tr,ts)
            stats.append({"Column":c,"Skew":skew(tr),"Outlier%":out,"KS":ks,"P":p,"Drift":drift_label(ks,p)})
        vd=train[num_cols].dropna(); vr=[variance_inflation_factor(vd.values,i) for i in range(len(num_cols))]
        tmi=train[num_cols+[target_col]].dropna(); tmi[target_col]=safe_encode_target(tmi[target_col])
        mi_fn=mutual_info_classif if task=="classification" else mutual_info_regression
        mi=mi_fn(tmi[num_cols],tmi[target_col],random_state=42)
        da2=pd.DataFrame(stats).set_index("Column"); da2["VIF"]=vr; da2["MI"]=mi
        R.table(da2.style.background_gradient(cmap="OrRd",subset=["VIF","Skew"]).background_gradient(cmap="YlGn",subset=["MI"])
                .map(lambda x:"color:red;font-weight:bold" if "Severe" in str(x) else("color:orange" if "Mild" in str(x) or "Statistical" in str(x) else "color:green"),subset=["Drift"]).format(precision=3))
        rd=[s["Column"] for s in stats if "Severe" in s["Drift"] or "Mild" in s["Drift"]]
        if rd: actions.append(f"📉 Real drift: {rd}")
        hv=[num_cols[i] for i,v in enumerate(vr) if v>10]
        if hv: actions.append(f"📊 VIF>10: {hv}")
        fig_nd=plot_numeric_dist(train,test,num_cols)
        if fig_nd: R.plot(fig_nd,"numeric_distributions")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 6: NUMERIC CORRELATION
    # ════════════════════════════════════════════════════
    R.section("🌟 6. NUMERIC CORRELATION")
    if len(num_cols)>=2:
        corr=train[num_cols].corr(); hcp=[]
        for i in range(len(num_cols)):
            for j in range(i+1,len(num_cols)):
                r=corr.iloc[i,j]
                if abs(r)>0.85: hcp.append({"F1":num_cols[i],"F2":num_cols[j],"|r|":abs(r)})
        if hcp: R.log(f"⚠️ {len(hcp)} pairs |r|>0.85",cls="warn"); R.table(pd.DataFrame(hcp).style.background_gradient(cmap="Reds",subset=["|r|"]).format({"|r|":"{:.4f}"})); actions.append(f"🔗 {len(hcp)} correlated pairs")
        else: R.log("✅ No |r|>0.85 pairs.",cls="ok")
        tc=safe_encode_target(train[target_col].copy()); tcr=train[num_cols].corrwith(tc).abs().sort_values(ascending=False)
        dtc=pd.DataFrame({"Feature":tcr.index,f"|r| with {target_col}":tcr.values}).reset_index(drop=True)
        R.table(dtc.style.bar(subset=[f"|r| with {target_col}"],color="#3fb950").format({f"|r| with {target_col}":"{:.4f}"}))
        R.plot(plot_correlation_heatmap(train,num_cols,target_col),"correlation_heatmap")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 7: CRAMER'S V
    # ════════════════════════════════════════════════════
    R.section("🌟 7. CRAMER'S V — CATEGORICAL CORRELATION")
    if len(cat_cols)>=2:
        hcv=[]
        for i in range(len(cat_cols)):
            for j in range(i+1,len(cat_cols)):
                v=cramers_v(train[cat_cols[i]].fillna("_NA_"),train[cat_cols[j]].fillna("_NA_"))
                if v>0.5: hcv.append({"F1":cat_cols[i],"F2":cat_cols[j],"V":v,"Risk":"🔴 Redundant" if v>0.8 else "🟡 High"})
        if hcv: R.log(f"⚠️ {len(hcv)} pairs V>0.5",cls="warn"); R.table(pd.DataFrame(hcv).sort_values("V",ascending=False).reset_index(drop=True).style.background_gradient(cmap="Reds",subset=["V"]).format({"V":"{:.4f}"}))
        else: R.log("✅ No V>0.5 pairs.",cls="ok")
        tcv={c:cramers_v(train[c].fillna("_NA_"),train[target_col].fillna("_NA_")) for c in cat_cols}
        dtcv=pd.DataFrame.from_dict(tcv,orient="index",columns=[f"V with {target_col}"]).sort_values(f"V with {target_col}",ascending=False)
        R.table(dtcv.style.bar(subset=[f"V with {target_col}"],color="#636efa").format(precision=4))
        R.plot(plot_cramers_heatmap(train,cat_cols,target_col),"cramers_v_heatmap")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 8: HIERARCHICAL DEPENDENCY
    # ════════════════════════════════════════════════════
    R.section("🌟 8. HIERARCHICAL DEPENDENCY")
    sents=["No internet service","No phone service"]; hier=[]
    for col in cat_cols:
        for sv in sents:
            mask=train[col]==sv; sc=mask.sum()
            if sc==0: continue
            for pcol in cat_cols:
                if pcol==col: continue
                for pv in train[pcol].unique():
                    if (train[pcol]==pv).sum()!=sc: continue
                    if (mask&(train[pcol]==pv)).sum()==sc:
                        tgt=safe_encode_target(train[target_col].copy())
                        hier.append({"Child":col,"Sentinel":sv,"Parent":pcol,"Parent_Val":pv,"Count":sc,f"{target_col}_Rate%":tgt[mask].mean()*100})
    if hier:
        dh=pd.DataFrame(hier); R.log(f"⚠️ {len(dh)} hierarchical dependencies!",cls="critical")
        R.table(dh.style.format({f"{target_col}_Rate%":"{:.2f}"}).background_gradient(cmap="OrRd",subset=["Count"]))
        R.log("💡 Replace sentinel → 'No' or NaN",cls="warn"); actions.append(f"🔗 Hierarchical: {list(dh['Child'].unique())}")
    else: R.log("✅ No hierarchical dependencies.",cls="ok")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 9: DERIVED FEATURES
    # ════════════════════════════════════════════════════
    R.section("🌟 9. DERIVED FEATURE RELATIONSHIPS")
    if len(num_cols)>=2:
        derived=[]; cc=num_cols[:10]
        for i in range(len(cc)):
            for j in range(i+1,len(cc)):
                for k in range(len(cc)):
                    if k in(i,j): continue
                    a,b,c_=train[cc[i]],train[cc[j]],train[cc[k]]; prod=a*b
                    if prod.std()>0 and c_.std()>0:
                        rp=prod.corr(c_)
                        if abs(rp)>0.90: derived.append({"Rel":f"{cc[i]} × {cc[j]}","Target":cc[k],"|r|":abs(rp),"Type":"Product"})
                    if(b.abs()>1e-9).all():
                        rat=a/b
                        if rat.std()>0 and c_.std()>0:
                            rr=rat.corr(c_)
                            if abs(rr)>0.90: derived.append({"Rel":f"{cc[i]} / {cc[j]}","Target":cc[k],"|r|":abs(rr),"Type":"Ratio"})
        if derived:
            dd=pd.DataFrame(derived).sort_values("|r|",ascending=False).reset_index(drop=True)
            R.log(f"⚠️ {len(dd)} derived relationships:",cls="critical")
            R.table(dd.style.background_gradient(cmap="Reds",subset=["|r|"]).format({"|r|":"{:.4f}"}))
            actions.append(f"🔗 Derived: {dd['Target'].unique().tolist()} → residual")
        else: R.log("✅ No derived relationships.",cls="ok")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 10: CATEGORICAL OVERLAP
    # ════════════════════════════════════════════════════
    R.section("🌟 10. CATEGORICAL OVERLAP")
    if cat_cols:
        rc=[]; 
        for c in cat_cols:
            trs,tss=set(train[c].dropna().unique()),set(test[c].dropna().unique())
            ov=(len(tss&trs)/len(tss)*100) if tss else 100
            rc.append({"Col":c,"Tr":len(trs),"Ts":len(tss),"Overlap%":ov,"Only_Test":len(tss-trs)})
        drc=pd.DataFrame(rc).set_index("Col")
        R.table(drc.style.map(lambda x:"color:red" if x<100 else "color:green",subset=["Overlap%"]))
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 11: TARGET BIVARIATE
    # ════════════════════════════════════════════════════
    R.section("🌟 11. TARGET BIVARIATE")
    if cat_cols:
        tmp=train.copy()
        if tmp[target_col].dtype=="O":
            le=LabelEncoder(); le.fit(tmp[target_col].astype(str)); tmp[target_col]=le.transform(tmp[target_col].astype(str))
            tl=le.classes_[1] if len(le.classes_)>1 else "Positive"
        else: tl="Target"
        bv=[]
        for c in cat_cols:
            grp=tmp.groupby(c)[target_col]
            for idx,val in (grp.mean()*100).items(): bv.append({"Feature":c,"Category":idx,f"{tl}_Rate%":val,"Count":grp.count().get(idx,0)})
        dbv=pd.DataFrame(bv).set_index(["Feature","Category"]); rc2=[c for c in dbv.columns if "Rate" in c][0]
        R.table(dbv.style.background_gradient(cmap="RdYlGn_r",subset=[rc2]).format(precision=2))
        R.plot(plot_bivariate_top(train,target_col,cat_cols,task),"bivariate_top")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 12: INFORMATION VALUE
    # ════════════════════════════════════════════════════
    R.section("🌟 12. INFORMATION VALUE (IV)")
    div = None
    if task=="classification" and cat_cols:
        tmp_iv=train.copy(); tmp_iv[target_col]=safe_encode_target(tmp_iv[target_col])
        ivr={c:calc_iv(tmp_iv,c,target_col) for c in cat_cols}
        div=pd.DataFrame.from_dict(ivr,orient="index",columns=["IV"]).sort_values("IV",ascending=False)
        div["Strength"]=div["IV"].apply(lambda x:"🔴 Suspicious" if x>0.5 else "🟢 Strong" if x>0.3 else "🟡 Medium" if x>0.1 else "🔵 Weak" if x>0.02 else "⚪ Useless")
        R.table(div.style.bar(subset=["IV"],color="#636efa").format({"IV":"{:.4f}"}))
        R.plot(plot_iv_chart(div),"information_value")
        sus=div[div["IV"]>0.5].index.tolist()
        if sus: R.log(f"⚠️ IV>0.5 risk: {sus}",cls="critical"); actions.append(f"🚨 IV suspicious: {sus}")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 13: ADVERSARIAL & IMPORTANCE (RF-based EDA)
    # ════════════════════════════════════════════════════
    R.section("🌟 13. ADVERSARIAL & FEATURE IMPORTANCE (EDA)")
    adf=pd.concat([train[feats],test[feats]],axis=0); ya=[0]*len(train)+[1]*len(test)
    for c in adf.select_dtypes(["object","category"]).columns: adf[c]=pd.factorize(adf[c])[0]
    rf=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42,n_jobs=-1)
    auc_adv=cross_val_score(rf,adf.fillna(-1),ya,cv=5,scoring="roc_auc").mean()
    if auc_adv>0.8: R.log(f"🕵️ Adversarial AUC: {auc_adv:.4f} — 🔴 Severe!",cls="critical")
    elif auc_adv>0.6: R.log(f"🕵️ Adversarial AUC: {auc_adv:.4f} — ⚠️ Mild drift.",cls="warn")
    else: R.log(f"🕵️ Adversarial AUC: {auc_adv:.4f} — ✅ Safe.",cls="ok")
    timp=train[feats+[target_col]].dropna()
    for c in timp.select_dtypes(["object","category"]).columns: timp[c]=pd.Series(LabelEncoder().fit_transform(timp[c].astype(str)),index=timp.index)
    rf_imp=(RandomForestClassifier if task=="classification" else RandomForestRegressor)(n_estimators=100,max_depth=8,random_state=42,n_jobs=-1)
    rf_imp.fit(timp.drop(columns=[target_col]),timp[target_col])
    imp=pd.DataFrame({"Feature":feats,"Importance":rf_imp.feature_importances_}).sort_values("Importance",ascending=False).reset_index(drop=True)
    imp["Cumulative"]=imp["Importance"].cumsum(); imp["Rank"]=range(1,len(imp)+1)
    R.table(imp.style.bar(subset=["Importance"],color="#636efa").background_gradient(cmap="YlGn",subset=["Cumulative"]).format({"Importance":"{:.4f}","Cumulative":"{:.4f}"}))
    R.plot(plot_feature_importance(imp),"feature_importance")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 14: MISSING PATTERNS
    # ════════════════════════════════════════════════════
    R.section("🌟 14. MISSING VALUE PATTERNS")
    nc=[c for c in feats if train[c].isnull().any()]
    if nc:
        ns=[]
        for c in nc:
            np_=train[c].isnull().mean()*100; tgt2=safe_encode_target(train[target_col].copy())
            rn=tgt2[train[c].isnull()].mean()*100; rnp=tgt2[train[c].notnull()].mean()*100
            ns.append({"Feature":c,"Null%":np_,"Rate_NULL%":rn,"Rate_PRESENT%":rnp,"Gap":abs(rn-rnp)})
        dn=pd.DataFrame(ns).set_index("Feature").sort_values("Gap",ascending=False)
        R.table(dn.style.background_gradient(cmap="OrRd",subset=["Gap","Null%"]).format(precision=2))
        inull=dn[dn["Gap"]>5].index.tolist()
        if inull: actions.append(f"🏷️ is_null flags: {inull}")
    else: R.log("✅ No missing values.",cls="ok")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 15: PLAYGROUND SERIES NOTE
    # ════════════════════════════════════════════════════
    R.section("🌟 15. PLAYGROUND SERIES — ORIGINAL DATA")
    R.log("📌 This is a Playground Series competition — original data can be used for augmentation.")
    R.log("📌 Origin: IBM Telco Customer Churn (available on Kaggle).")
    R.log("💡 pd.concat([train, original]) + source flag.",cls="warn")
    R.end()

    # ══════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════
    #       🧪 VALIDATION TESTS (SECTIONS 16-24)
    # ══════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════

    X_all = train[feats].copy()
    for c in X_all.select_dtypes(['object','category']).columns: X_all[c]=X_all[c].astype('category')

    # ════════════════════════════════════════════════════
    # SECTION 16: BASELINE MODEL
    # ════════════════════════════════════════════════════
    R.section("🧪 16. BASELINE MODEL (Zero FE)")
    R.log("Running 5-fold LightGBM with NO feature engineering...")
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    bl_scores=[]; bl_models=[]
    for fold,(tr_i,va_i) in enumerate(skf.split(X_all,y_full)):
        m=lgb.LGBMClassifier(n_estimators=1000,learning_rate=0.05,max_depth=6,num_leaves=31,subsample=0.8,colsample_bytree=0.8,is_unbalance=True,random_state=42,verbose=-1,n_jobs=-1)
        m.fit(X_all.iloc[tr_i],y_full.iloc[tr_i],eval_set=[(X_all.iloc[va_i],y_full.iloc[va_i])],callbacks=[lgb.early_stopping(50),lgb.log_evaluation(0)])
        p=m.predict_proba(X_all.iloc[va_i])[:,1]; a=roc_auc_score(y_full.iloc[va_i],p)
        bl_scores.append(a); bl_models.append(m); R.log(f"  Fold {fold}: AUC = {a:.5f}")
    bl_mean=np.mean(bl_scores); bl_std=np.std(bl_scores)
    results['baseline_mean']=bl_mean; results['baseline_std']=bl_std
    R.log(f"\n📊 BASELINE: {bl_mean:.5f} ± {bl_std:.5f}",cls="ok")

    if bl_mean>0.90: R.decision("Baseline >0.90 — very strong. Focus on ensemble & tuning. Expect +0.001-0.005 from FE.",True)
    elif bl_mean>0.85: R.decision("Baseline 0.85-0.90 — good. FE can push +0.005-0.015. Focus on top feature interactions.",True)
    elif bl_mean>0.80: R.decision("Baseline 0.80-0.85 — moderate. Aggressive FE, encoding, augmentation needed.",True)
    else: R.decision("Baseline <0.80 — weak. Rethink approach: different encoding, target encoding, stacking.",False)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 17: CV STABILITY
    # ════════════════════════════════════════════════════
    R.section("🧪 17. CV STABILITY ACROSS SEEDS")
    seeds=[42,123,456,789,2024]; seed_scores=[]
    for seed in seeds:
        sk=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed); fsc=[]
        for tr_i,va_i in sk.split(X_all,y_full):
            m=lgb.LGBMClassifier(n_estimators=500,learning_rate=0.05,max_depth=6,is_unbalance=True,random_state=seed,verbose=-1,n_jobs=-1)
            m.fit(X_all.iloc[tr_i],y_full.iloc[tr_i],eval_set=[(X_all.iloc[va_i],y_full.iloc[va_i])],callbacks=[lgb.early_stopping(50),lgb.log_evaluation(0)])
            fsc.append(roc_auc_score(y_full.iloc[va_i],m.predict_proba(X_all.iloc[va_i])[:,1]))
        seed_scores.append(np.mean(fsc)); R.log(f"  Seed {seed}: AUC = {np.mean(fsc):.5f}")
    s_std=np.std(seed_scores); s_range=max(seed_scores)-min(seed_scores)
    results['cv_std']=s_std
    R.log(f"\n📊 Seed std: {s_std:.5f} | Range: {s_range:.5f}")

    if s_std<0.001: R.decision("Very stable (std<0.001). Trust CV, 5-fold enough.",True)
    elif s_std<0.003: R.decision("Reasonably stable (std<0.003). Use fixed seed.",True)
    elif s_std<0.005: R.decision("Moderate variance. Use RepeatedStratifiedKFold (3×5).",False)
    else: R.decision("UNSTABLE (std>0.005). Use 10-fold or repeated CV. Check leakage.",False); actions.append("🚨 CV unstable → repeated CV")

    outlier_folds=sum(1 for s in bl_scores if abs(s-bl_mean)>2*bl_std)
    if outlier_folds>0: R.decision(f"{outlier_folds} outlier fold(s). Investigate data distribution in those folds.",False)
    else: R.decision("All folds consistent. No outliers.",True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 18: PER-FEATURE ADVERSARIAL
    # ════════════════════════════════════════════════════
    R.section("🧪 18. PER-FEATURE ADVERSARIAL IMPORTANCE")
    rf_adv=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42,n_jobs=-1)
    rf_adv.fit(adf.fillna(-1),ya)
    adv_imp=pd.DataFrame({"Feature":feats,"Adv_Imp":rf_adv.feature_importances_}).sort_values("Adv_Imp",ascending=False).reset_index(drop=True)
    results['adv_auc']=auc_adv
    R.log(f"📊 Adversarial AUC: {auc_adv:.4f}")
    R.table(adv_imp.head(10).style.bar(subset=["Adv_Imp"],color="#f0883e").format({"Adv_Imp":"{:.4f}"}))

    if auc_adv<0.52: R.decision("AUC<0.52 → NO drift. Train/test same distribution.",True)
    elif auc_adv<0.60: R.decision("AUC 0.52-0.60 → minimal drift. Monitor top features.",True)
    elif auc_adv<0.70: R.decision("AUC 0.60-0.70 → moderate drift. Consider dropping/normalizing top adversarial features.",False); actions.append(f"⚠️ Drift features: {adv_imp.head(3)['Feature'].tolist()}")
    else: R.decision("AUC>0.70 → SEVERE drift/leakage! STOP and investigate.",False); actions.append("🚨 Severe drift!")

    hi_adv=adv_imp[adv_imp["Adv_Imp"]>0.10]["Feature"].tolist()
    if hi_adv: R.decision(f"Individual drifted features: {hi_adv}. Check distributions.",False)
    else: R.decision("No single feature dominates adversarial model.",True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 19: PERMUTATION IMPORTANCE
    # ════════════════════════════════════════════════════
    R.section("🧪 19. PERMUTATION IMPORTANCE")
    last_m=bl_models[-1]; last_fold=list(skf.split(X_all,y_full))[-1]
    Xvp=X_all.iloc[last_fold[1]]; yvp=y_full.iloc[last_fold[1]]
    perm=permutation_importance(last_m,Xvp,yvp,n_repeats=10,random_state=42,scoring='roc_auc',n_jobs=-1)
    perm_df=pd.DataFrame({"Feature":feats,"Perm_Mean":perm.importances_mean,"Perm_Std":perm.importances_std})
    lgb_fi=pd.DataFrame({"Feature":feats,"LGB_Imp":last_m.feature_importances_})
    comp=pd.merge(lgb_fi,perm_df,on="Feature")
    comp["R_LGB"]=comp["LGB_Imp"].rank(ascending=False); comp["R_Perm"]=comp["Perm_Mean"].rank(ascending=False)
    comp["R_Diff"]=abs(comp["R_LGB"]-comp["R_Perm"])
    comp=comp.sort_values("Perm_Mean",ascending=False).reset_index(drop=True)
    R.table(comp.style.bar(subset=["Perm_Mean"],color="#3fb950").background_gradient(cmap="OrRd",subset=["R_Diff"]).format({"Perm_Mean":"{:.4f}","Perm_Std":"{:.4f}","LGB_Imp":"{:.0f}","R_LGB":"{:.0f}","R_Perm":"{:.0f}","R_Diff":"{:.0f}"}))

    brd=comp[comp["R_Diff"]>5]
    if len(brd)==0: R.decision("LGB and permutation importance AGREE. Ranking reliable.",True)
    elif len(brd)<=3: R.decision(f"Minor disagreements: {brd['Feature'].tolist()}. Trust permutation over built-in.",False)
    else: R.decision("MAJOR disagreements. Use permutation importance for all decisions.",False); actions.append("⚠️ Use permutation importance")

    neg_p=comp[comp["Perm_Mean"]<0]["Feature"].tolist()
    if neg_p: R.log(f"🔴 Negative perm importance: {neg_p}",cls="critical"); R.decision(f"Features HURT model: {neg_p}. Remove them.",False); actions.append(f"🗑️ Negative perm: {neg_p}")
    zero_p=comp[(comp["Perm_Mean"]>=0)&(comp["Perm_Mean"]<0.0001)]["Feature"].tolist()
    if zero_p: R.decision(f"Near-zero perm: {zero_p}. Try removing — if CV improves, keep removed.",False)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 20: SEGMENT ANALYSIS
    # ════════════════════════════════════════════════════
    R.section("🧪 20. HIGH-RISK SEGMENT PROFILING")
    tmp_s=train.copy(); tmp_s["tbin"]=safe_encode_target(tmp_s[target_col].copy())
    seg_cols=[c for c in ["Contract","InternetService","PaymentMethod"] if c in tmp_s.columns]
    if len(seg_cols)>=2:
        segs=[]
        for keys,grp in tmp_s.groupby(seg_cols):
            if len(grp)>=100:
                row=dict(zip(seg_cols,keys if isinstance(keys,tuple) else [keys]))
                row["Count"]=len(grp); row["Churn%"]=grp["tbin"].mean()*100; segs.append(row)
        if segs:
            df_seg=pd.DataFrame(segs).sort_values("Churn%",ascending=False).reset_index(drop=True)
            R.log("🔴 TOP 5 HIGHEST RISK:",cls="critical")
            R.table(df_seg.head(5).style.background_gradient(cmap="Reds",subset=["Churn%"]).format({"Churn%":"{:.2f}"}))
            R.log("🟢 TOP 5 SAFEST:",cls="ok")
            R.table(df_seg.tail(5).style.background_gradient(cmap="Greens",subset=["Churn%"]).format({"Churn%":"{:.2f}"}))
            spread=df_seg["Churn%"].max()-df_seg["Churn%"].min(); results['seg_spread']=spread
            if spread>40: R.decision(f"Spread={spread:.1f}%. HUGE. Create binary risk flags from top segments.",True); actions.append("🎯 Create risk flags from segments")
            elif spread>20: R.decision(f"Spread={spread:.1f}%. Moderate. Interaction features will help.",True); actions.append("🎯 Interaction features from segments")
            else: R.decision(f"Spread={spread:.1f}%. Low. Try other combinations.",False)

            fig,ax=plt.subplots(figsize=(12,max(4,len(df_seg)*0.4)))
            colors=["#f85149" if r>30 else "#f0883e" if r>15 else "#3fb950" for r in df_seg["Churn%"]]
            labels=[' | '.join(str(df_seg.iloc[i][c]) for c in seg_cols) for i in range(len(df_seg))]
            ax.barh(range(len(df_seg)),df_seg["Churn%"],color=colors,edgecolor=GRID)
            ax.set_yticks(range(len(df_seg))); ax.set_yticklabels(labels,fontsize=9)
            dark_style(ax,"Segments — Churn Rate"); ax.set_xlabel("Churn Rate (%)"); fig.tight_layout()
            R.plot(fig,"segment_analysis")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 21: FEATURE STABILITY
    # ════════════════════════════════════════════════════
    R.section("🧪 21. FEATURE STABILITY ACROSS FOLDS")
    fi_list=[]
    for fold,(tr_i,va_i) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X_all,y_full)):
        m=lgb.LGBMClassifier(n_estimators=500,verbose=-1,n_jobs=-1,is_unbalance=True,random_state=42)
        m.fit(X_all.iloc[tr_i],y_full.iloc[tr_i],eval_set=[(X_all.iloc[va_i],y_full.iloc[va_i])],callbacks=[lgb.early_stopping(50),lgb.log_evaluation(0)])
        fi_list.append(pd.DataFrame({"Feature":feats,f"F{fold}":m.feature_importances_}).set_index("Feature"))
    stab=pd.concat(fi_list,axis=1); stab["Mean"]=stab.mean(axis=1); stab["Std"]=stab.std(axis=1)
    stab["CV_coef"]=stab["Std"]/(stab["Mean"]+1e-10); stab=stab.sort_values("Mean",ascending=False)
    R.table(stab[["Mean","Std","CV_coef"]].style.background_gradient(cmap="RdYlGn_r",subset=["CV_coef"]).format(precision=2))

    v_unstable=stab[stab["CV_coef"]>1.0].index.tolist()
    m_unstable=stab[stab["CV_coef"]>0.5].index.tolist()
    if v_unstable: R.decision(f"Very unstable (CV>1.0): {v_unstable}. Consider removing.",False); actions.append(f"⚠️ Unstable features: {v_unstable}")
    elif m_unstable: R.decision(f"Moderately unstable (CV>0.5): {m_unstable}. Monitor.",False)
    else: R.decision("All features stable (CV<0.5). Reliable ranking.",True)
    results['stable_top']=stab.head(5).index.tolist()
    R.log(f"🏆 Core stable features: {results['stable_top']}")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 22: ORIGINAL DATA COMPARISON
    # ════════════════════════════════════════════════════
    R.section("🧪 22. ORIGINAL DATA COMPARISON")
    original=None
    if original_path and os.path.exists(original_path): original=pd.read_csv(original_path)
    else:
        for p in ['/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv','/kaggle/input/blastchar-telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv']:
            if os.path.exists(p): original=pd.read_csv(p); break
    if original is not None:
        R.log(f"📦 Original: {original.shape[0]:,} | Synthetic: {train.shape[0]:,} ({train.shape[0]//original.shape[0]}x)")
        for col in num_cols:
            if col in original.columns:
                ov=pd.to_numeric(original[col],errors='coerce').dropna(); sv=train[col].dropna()
                ks,p=ks_2samp(ov,sv); R.log(f"  {col}: KS={ks:.4f} | orig μ={ov.mean():.1f} → synth μ={sv.mean():.1f}")
        if target_col in original.columns:
            or_=((original[target_col]=='Yes').mean()*100); sr_=((train[target_col]=='Yes').mean()*100); rd_=abs(or_-sr_)
            R.log(f"  Target: orig={or_:.2f}% → synth={sr_:.2f}% (Δ={rd_:.2f}%)")
            if rd_>5: R.decision(f"Rate differs by {rd_:.1f}%. Reweight when augmenting.",False)
            else: R.decision(f"Rate similar (Δ={rd_:.1f}%). Safe to concat.",True)
        R.decision("Augment: pd.concat([train, original]) + source='original' flag.",True)
        actions.append("📦 Augment with original data")
    else:
        R.log("⚠️ Original data not found.",cls="warn")
        R.decision("Add 'blastchar/telco-customer-churn' as Kaggle input.",False)
        actions.append("📦 Add original Telco dataset")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 23: MONOTONICITY CHECK
    # ════════════════════════════════════════════════════
    R.section("🧪 23. MONOTONICITY CHECK")
    tmp_m=train.copy(); tmp_m["tbin"]=safe_encode_target(tmp_m[target_col].copy())
    mono_res=[]
    for col in num_cols:
        try:
            tmp_m[f"{col}_d"]=pd.qcut(tmp_m[col],10,duplicates='drop',labels=False)
            dec=tmp_m.groupby(f"{col}_d")["tbin"].mean()*100; rates=dec.values
            rho,_=spearmanr(range(len(rates)),rates)
            is_inc=all(rates[i]<=rates[i+1]+0.5 for i in range(len(rates)-1))
            is_dec=all(rates[i]>=rates[i+1]-0.5 for i in range(len(rates)-1))
            if is_inc: d,cn="📈 Monotonic ↑",1
            elif is_dec: d,cn="📉 Monotonic ↓",-1
            else: d,cn="🔀 Non-monotonic",0
            mono_res.append({"Feature":col,"Direction":d,"Spearman":rho,"Constraint":cn})
            R.log(f"  {col}: {d} (ρ={rho:.3f})")
            R.log(f"    Deciles: {' → '.join(f'{r:.1f}%' for r in rates)}")
        except: pass

    if mono_res:
        R.table(pd.DataFrame(mono_res)[["Feature","Direction","Spearman","Constraint"]].style.format({"Spearman":"{:.3f}"}))
        constraints=[r["Constraint"] for r in mono_res]
        has_mono=any(c!=0 for c in constraints)
        if has_mono:
            R.decision(f"Apply monotone_constraints = {constraints} in LightGBM.",True)
            actions.append(f"📈 monotone_constraints = {constraints}")
        else:
            R.decision("No monotonic relationships. Consider binning.",True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 24: POST-CLEANING INTERACTIONS
    # ════════════════════════════════════════════════════
    R.section("🧪 24. POST-CLEANING INTERACTION ANALYSIS")
    tmp_i=train.copy(); inet_cols=[c for c in ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'] if c in tmp_i.columns]
    for c in inet_cols: tmp_i[c]=tmp_i[c].replace('No internet service','No')
    if 'MultipleLines' in tmp_i.columns: tmp_i['MultipleLines']=tmp_i['MultipleLines'].replace('No phone service','No')
    tmp_i["tbin"]=safe_encode_target(tmp_i[target_col].copy())

    if inet_cols:
        tmp_i['n_services']=sum((tmp_i[c]=='Yes').astype(int) for c in inet_cols)
        svc=tmp_i.groupby('n_services')['tbin'].agg(['mean','count']); svc['rate']=svc['mean']*100
        R.log("\n📊 Churn by number of services:")
        for n,row in svc.iterrows(): R.log(f"  {n} services: {row['rate']:5.1f}% ({row['count']:>6,} customers) {'█'*int(row['rate']/2)}")
        rho,_=spearmanr(svc.index,svc['rate'])
        if rho<-0.7: R.decision(f"Strong negative (ρ={rho:.2f}): more services = less churn. n_services is POWERFUL.",True); actions.append("🎯 Create n_services (strong)")
        elif rho<-0.3: R.decision(f"Moderate negative (ρ={rho:.2f}). n_services will help.",True); actions.append("🎯 Create n_services")
        elif rho>0.3: R.decision(f"Positive (ρ={rho:.2f}): unusual — investigate overcharging.",False)
        else: R.decision(f"Weak (ρ={rho:.2f}). Try n_security and n_streaming separately.",True)

    # Strong interactions
    if 'InternetService' in tmp_i.columns and inet_cols:
        overall=tmp_i['tbin'].mean()*100; strong=[]
        for col in inet_cols:
            for(inet,val),grp in tmp_i.groupby(['InternetService',col]):
                if len(grp)>=50:
                    r=grp['tbin'].mean()*100; dev=abs(r-overall)
                    if dev>15: strong.append({"Interaction":f"{inet}_{val}","Churn%":r,"Deviation":dev,"Count":len(grp),"Signal":"🔴 HIGH" if r>overall else "🟢 LOW"})
        if strong:
            df_st=pd.DataFrame(strong).sort_values("Deviation",ascending=False).reset_index(drop=True)
            R.log(f"\n⚡ {len(df_st)} strong interactions (>15% deviation):",cls="ok")
            R.table(df_st.style.background_gradient(cmap="RdYlGn_r",subset=["Churn%"]).format({"Churn%":"{:.2f}","Deviation":"{:.2f}"}))
            R.decision(f"Create binary flags for top interactions.",True)
            actions.append(f"🎯 Interaction flags: {df_st.head(3)['Interaction'].tolist()}")
    R.end()

    # ══════════════════════════════════════════════════════════
    # SECTION 25: FINAL ACTION SUMMARY
    # ══════════════════════════════════════════════════════════
    R.section("🎯 25. FINAL ACTION SUMMARY")
    R.log(f"📊 Baseline AUC: {results['baseline_mean']:.5f} ± {results['baseline_std']:.5f}")
    R.log(f"📊 CV Stability: std = {results.get('cv_std',0):.5f}")
    R.log(f"📊 Adversarial AUC: {results.get('adv_auc',0):.4f}")
    R.log(f"📊 Core features: {results.get('stable_top',[])}")

    score=0
    if results['baseline_mean']>0.85: score+=2
    elif results['baseline_mean']>0.80: score+=1
    if results.get('cv_std',1)<0.003: score+=2
    elif results.get('cv_std',1)<0.005: score+=1
    if results.get('adv_auc',1)<0.55: score+=2
    elif results.get('adv_auc',1)<0.65: score+=1

    if score>=5: R.log("🟢 READINESS: HIGH",cls="ok"); R.decision("Clean data, stable CV, no drift. Proceed to FE & modeling.",True)
    elif score>=3: R.log("🟡 READINESS: MODERATE",cls="warn"); R.decision("Fix warnings below before heavy modeling.",False)
    else: R.log("🔴 READINESS: LOW",cls="critical"); R.decision("Fix fundamental problems before modeling.",False)

    R.log("\n📋 ALL ACTIONS:")
    for i,a in enumerate(actions,1):
        cl="critical" if "🚨" in a else("warn" if "⚠" in a else "ok")
        R.log(f"  {i}. {a}",cls=cl)

    R.log("\n🔧 FE RECIPE:")
    R.log("  1️⃣  Sentinel → 'No' (No internet service, No phone service)")
    R.log("  2️⃣  Drop gender, PhoneService")
    R.log("  3️⃣  Create n_services, n_security, n_streaming")
    R.log("  4️⃣  charges_residual = TotalCharges - tenure × MonthlyCharges")
    R.log("  5️⃣  avg_monthly = TotalCharges / (tenure + 1)")
    if results.get('stable_top'): R.log(f"  6️⃣  Interactions between: {results['stable_top'][:4]}")
    if any('risk' in a.lower() for a in actions): R.log("  7️⃣  Binary risk flags from segments")
    if mono_res and any(r['Constraint']!=0 for r in mono_res):
        R.log(f"  8️⃣  monotone_constraints = {[r['Constraint'] for r in mono_res]}")

    R.log(f"\n🏁 TARGETS:")
    R.log(f"    Current: {results['baseline_mean']:.5f}")
    R.log(f"    Good FE: {results['baseline_mean']+0.005:.5f} - {results['baseline_mean']+0.015:.5f}")
    R.log(f"    If < {results['baseline_mean']:.5f} after FE → REVERT")
    R.end()

    R.save("/kaggle/working/pipeline_report.html","/kaggle/working/pipeline_report.md")
    return train, test, target_col, id_col, results, actions


# ═══════════════════════════
# RUN
# ═══════════════════════════
train, test, target_col, id_col, results, actions = full_senior_data_pipeline(
    "/kaggle/input/playground-series-s6e3",
    "train.csv", "test.csv", "Churn", "id",
    original_path="/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
```

| #  | Section                    | Ne yapıyor & Neden gerekli |
|----|----------------------------|-----------------------------|
| 0  | Overview                   | Task type (classification/regression) otomatik algılanır. Feature sayısı raporlanır. |
| 1  | Memory & Shape             | Veri büyüklüğünü anla. >1GB ise dtype optimization gerekir. |
| 2  | General Audit              | Her sütunun dtype, unique count, null oranı, dominant value oranı. Train-test null gap. |
| 3  | Target Distribution        | Class balance kontrolü. Imbalance stratejisi belirlenir (SMOTE vs class_weight). |
| 4  | Duplicates & Constants     | Duplicate satırlar ve quasi-constant sütunlar. >99.5% aynı değer = sil. |
| 5  | Statistical Risk & Drift   | KS testi + VIF + MI Score. Drift yorumu KS+N birlikte değerlendirilir. |
| 6  | Numeric Correlation        | Pearson \|r\|>0.85 çiftler ve target korelasyon. Yüksek korelasyon = drop one veya PCA. |
| 7  | Cramér's V                 | Kategorik↔kategorik ilişki. V>0.8=redundant, V>0.5=high. Target ile V sıralaması. |
| 8  | Hierarchical Dependency    | "No internet service" gibi sentinel değerler. Aynı count + aynı parent = bağımlı sütun. |
| 9  | Derived Features           | A×B≈C veya A/B≈C ilişkileri. \|r\|>0.90 ise C türetilmiş → residual oluştur. |
| 10 | Categorical Overlap        | Test'te train'de olmayan kategori var mı? <100% = unknown handling gerekli. |
| 11 | Target Bivariate           | Her kategorik değer için target rate. Hangi kategori en riskli/güvenli? |
| 12 | Information Value (IV)     | Her kategorik feature'ın prediktif gücü. IV>0.5 suspicious ama adversarial ile doğrula. |
| 13 | Adversarial & Importance   | RF-based EDA: drift tespiti + importance. Cumulative importance ile top-N belirlenir. |
| 14 | Missing Patterns           | Null'larda target rate farkı var mı? Gap>5% = is_null flag oluştur. |
| 15 | Playground Series Note     | Original data augmentation hatırlatması. |
| 16 | 🧪 BASELINE MODEL         | Sıfır FE ile LightGBM 5-fold CV. Karşılaştırma noktası. Bu olmadan FE'nin işe yarayıp yaramadığını bilemezsin. |
| 17 | 🧪 CV STABILITY           | 5 farklı seed ile CV. std<0.003 = güvenilir. std>0.005 = repeated CV gerekli. Outlier fold = data segmenti farklı. |
| 18 | 🧪 PER-FEATURE ADV.       | Hangi feature train/test'i en çok ayırıyor? Genel AUC düşük olsa bile tek feature drifted olabilir. |
| 19 | 🧪 PERMUTATION IMP.       | Feature'ı shuffle et → skor ne kadar düşer? LGB built-in importance yanıltıcı olabilir (correlated features arasında dağıtır). Negative perm = feature zararlı → sil. |
| 20 | 🧪 SEGMENT ANALYSIS       | En tehlikeli/güvenli müşteri profilleri. Spread>40% = binary risk flag oluştur. FE fikirleri buradan çıkar. |
| 21 | 🧪 FEATURE STABILITY      | Importance fold'lar arası tutarlı mı? CV_coef>1.0 = very unstable → silmeyi düşün. Top stable features = core predictors. |
| 22 | 🧪 ORIGINAL DATA          | IBM Telco orijinal veri ile karşılaştır. Target rate farkı <5% = safe to concat. >5% = reweight when augmenting. |
| 23 | 🧪 MONOTONICITY           | tenure↑ → churn↓ gibi ilişkiler monoton mu? Monoton ise → LGB monotone_constraints ekle. Overfitting'i önler, generalization artar. |
| 24 | 🧪 INTERACTIONS           | Sentinel temizleme SONRASI etkileşimler. Fiber+NoSecurity gibi kombinasyonlar. n_services count: ρ<-0.7 = powerful feature. |
| 25 | FINAL SUMMARY             | Readiness score (0-6 puan). 5-6=HIGH, 3-4=MODERATE, 0-2=LOW. Tüm aksiyonlar + FE recipe + hedef skor. |











# Senior Analist — Eksik Testler ve Ek Adımlar

Hayır, direkt ilerlemez. Arada **kritik doğrulama adımları** var. Bunları atlamak yarışmada sıralama kaybettirir.

---

## Eksik Test 1: BASELINE ÖNCE, FE SONRA

### Neden gerekli?

```
Feature engineering yaptın, CV skoru 0.842 çıktı.
Soru: Bu iyi mi kötü mü?

Baseline yoksa bilemezsin:
  → Baseline 0.840 idi → FE sadece +0.002 kattı (boşuna uğraşmışsın)
  → Baseline 0.810 idi → FE +0.032 kattı (harika!)
  → Baseline 0.845 idi → FE -0.003 kaybettirdi (FE zararlı olmuş!)

HER ZAMAN önce baseline kur, sonra karşılaştır.
```

### Kod:

```python
# ══════════════════════════════════════
# BASELINE — HİÇBİR FE YAPMADAN
# ══════════════════════════════════════
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

def quick_baseline(train, target_col, id_col, cat_cols):
    """Sıfır FE ile baseline skor."""
    feats = [c for c in train.columns if c not in [target_col, id_col]]
    X = train[feats].copy()
    y = (train[target_col] == 'Yes').astype(int) if train[target_col].dtype == 'O' else train[target_col]
    
    for c in X.select_dtypes(['object', 'category']).columns:
        X[c] = X[c].astype('category')
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            is_unbalance=True, random_state=42, verbose=-1,
            n_jobs=-1
        )
        model.fit(X_tr, y_tr, 
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        
        pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, pred)
        scores.append(auc)
        print(f"  Fold {fold}: AUC = {auc:.5f}")
    
    mean_auc = np.mean(scores)
    print(f"\n📊 BASELINE AUC: {mean_auc:.5f} ± {np.std(scores):.5f}")
    return mean_auc

baseline_score = quick_baseline(train, 'Churn', 'id', cat_cols)
# Bu skoru kaydet — her FE adımından sonra karşılaştıracaksın
```

### Sonra her FE adımında:

```python
# FE sonrası
fe_score = quick_baseline(train_fe, 'Churn', 'id', cat_cols)

print(f"Baseline : {baseline_score:.5f}")
print(f"After FE : {fe_score:.5f}")
print(f"Delta    : {fe_score - baseline_score:+.5f}")  # + ise iyi, - ise geri al
```

---

## Eksik Test 2: CV-LB KORELASYON TESTİ

### Neden gerekli?

```
Kaggle'da en büyük tuzak:
  CV skoru harika → Leaderboard'da düşük skor
  
Bu genelde CV stratejisinin yanlış olmasından kaynaklanır.
Senior analist ÖNCE CV güvenilirliğini test eder.
```

### Kod:

```python
# ══════════════════════════════════════
# CV STRATEJİSİ DOĞRULAMA
# ══════════════════════════════════════

# Test 1: Farklı seed'lerle CV stabilitesi
seeds = [42, 123, 456, 789, 2024]
all_scores = []

for seed in seeds:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_scores = []
    for tr_idx, val_idx in skf.split(X, y):
        # ... model fit & predict ...
        fold_scores.append(auc)
    all_scores.append(np.mean(fold_scores))

print(f"Scores across seeds: {[f'{s:.5f}' for s in all_scores]}")
print(f"Std across seeds   : {np.std(all_scores):.5f}")

# IF std > 0.003 → CV unstable → daha fazla fold veya repeated KFold kullan
# IF std < 0.001 → CV güvenilir → devam et


# Test 2: Fold-to-fold varyansı
# Eğer bir fold diğerlerinden çok farklıysa → data leakage veya outlier segment var
print("\nFold-level analysis:")
for i, s in enumerate(fold_scores):
    flag = " ⚠️" if abs(s - np.mean(fold_scores)) > 2 * np.std(fold_scores) else ""
    print(f"  Fold {i}: {s:.5f}{flag}")
```

---

## Eksik Test 3: PER-FEATURE ADVERSARIAL IMPORTANCE

### Neden gerekli?

```
Adversarial AUC = 0.5104 → GENEL drift yok.
Ama belirli bir feature'da GİZLİ drift olabilir.
Hangi feature en çok train/test ayrımına katkı yapıyor?
```

### Kod:

```python
# ══════════════════════════════════════
# ADVERSARIAL PER-FEATURE ANALYSIS
# ══════════════════════════════════════

adf = pd.concat([train[feats], test[feats]], axis=0)
ya = [0]*len(train) + [1]*len(test)

for c in adf.select_dtypes(['object','category']).columns:
    adf[c] = pd.factorize(adf[c])[0]

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
rf.fit(adf.fillna(-1), ya)

adv_imp = pd.DataFrame({
    'Feature': feats,
    'Adv_Importance': rf.feature_importances_
}).sort_values('Adv_Importance', ascending=False)

print("Features that MOST distinguish train from test:")
print(adv_imp.head(10).to_string(index=False))

# IF bir feature adversarial importance'ı yüksekse
# → o feature'da train/test farkı var
# → modelde kullanırken dikkatli ol
# → belki o feature'ı sil veya normalize et
```

---

## Eksik Test 4: PERMUTATION IMPORTANCE

### Neden gerekli?

```
RF Feature Importance → YANILTICI olabilir:
  → High cardinality feature'lara bias'lı
  → Correlated feature'lar arasında importance dağıtır
  
Permutation Importance → daha güvenilir:
  → Feature'ı shuffle et → skor ne kadar düşer?
  → Gerçek katkıyı ölçer
```

### Kod:

```python
# ══════════════════════════════════════
# PERMUTATION IMPORTANCE
# ══════════════════════════════════════
from sklearn.inspection import permutation_importance

# Baseline model (son fold)
perm = permutation_importance(
    model, X_val, y_val,
    n_repeats=10, random_state=42, scoring='roc_auc', n_jobs=-1
)

perm_df = pd.DataFrame({
    'Feature': feats,
    'Perm_Importance_Mean': perm.importances_mean,
    'Perm_Importance_Std': perm.importances_std,
}).sort_values('Perm_Importance_Mean', ascending=False)

print("Permutation Importance (top 10):")
print(perm_df.head(10).to_string(index=False))

# RF Importance ile karşılaştır
comparison = pd.merge(
    imp[['Feature','Importance']].rename(columns={'Importance':'RF_Imp'}),
    perm_df[['Feature','Perm_Importance_Mean']].rename(columns={'Perm_Importance_Mean':'Perm_Imp'}),
    on='Feature'
)
comparison['Rank_RF'] = comparison['RF_Imp'].rank(ascending=False)
comparison['Rank_Perm'] = comparison['Perm_Imp'].rank(ascending=False)
comparison['Rank_Diff'] = abs(comparison['Rank_RF'] - comparison['Rank_Perm'])

print("\nFeatures with BIGGEST rank difference (suspicious):")
print(comparison.sort_values('Rank_Diff', ascending=False).head(5).to_string(index=False))

# IF rank farkı büyükse → RF importance yanıltıcı
# → Permutation'a güven
```

---

## Eksik Test 5: SUBGROUP / SEGMENT ANALİZİ

### Neden gerekli?

```
Genel churn rate %22.5 ama bu ORTALAMA.
Bazı segmentlerde %70 olabilir, bazılarında %1.
Model bu segmentleri yakalıyor mu?
```

### Kod:

```python
# ══════════════════════════════════════
# HIGH-RISK SEGMENT PROFILING
# ══════════════════════════════════════

tmp = train.copy()
tmp['Churn_bin'] = (tmp['Churn'] == 'Yes').astype(int)

# En tehlikeli kombinasyonlar
segments = []

for inet in ['Fiber optic', 'DSL', 'No']:
    for contract in ['Month-to-month', 'One year', 'Two year']:
        for payment in train['PaymentMethod'].unique():
            mask = ((tmp['InternetService'] == inet) & 
                    (tmp['Contract'] == contract) & 
                    (tmp['PaymentMethod'] == payment))
            count = mask.sum()
            if count > 100:  # yeterli sample
                rate = tmp.loc[mask, 'Churn_bin'].mean() * 100
                segments.append({
                    'InternetService': inet,
                    'Contract': contract,
                    'PaymentMethod': payment,
                    'Count': count,
                    'Churn_Rate%': rate
                })

df_seg = pd.DataFrame(segments).sort_values('Churn_Rate%', ascending=False)

print("TOP 5 HIGHEST RISK SEGMENTS:")
print(df_seg.head(5).to_string(index=False))

print("\nTOP 5 SAFEST SEGMENTS:")
print(df_seg.tail(5).to_string(index=False))

# Bu bilgi FE için kullanılacak:
# → is_high_risk = (Fiber + Month-to-month + Electronic check)
# → risk_score = segment bazlı churn rate
```

---

## Eksik Test 6: FEATURE STABİLİTESİ ACROSS FOLDS

### Neden gerekli?

```
Bir feature Fold 1'de Rank 1, Fold 5'te Rank 15 ise
→ o feature UNSTABLE
→ modele noise katıyor
→ silmek skoru artırabilir
```

### Kod:

```python
# ══════════════════════════════════════
# FEATURE STABILITY ACROSS CV FOLDS
# ══════════════════════════════════════

fold_importances = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    model = lgb.LGBMClassifier(n_estimators=500, verbose=-1, n_jobs=-1)
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    
    fold_imp = pd.DataFrame({
        'Feature': feats,
        f'Fold_{fold}': model.feature_importances_
    })
    fold_importances.append(fold_imp.set_index('Feature'))

# Birleştir
stability = pd.concat(fold_importances, axis=1)
stability['Mean'] = stability.mean(axis=1)
stability['Std'] = stability.std(axis=1)
stability['CV'] = stability['Std'] / (stability['Mean'] + 1e-10)  # coefficient of variation
stability = stability.sort_values('Mean', ascending=False)

print("Feature Stability (low CV = stable, high CV = unstable):")
print(stability[['Mean', 'Std', 'CV']].to_string())

unstable = stability[stability['CV'] > 0.5].index.tolist()
if unstable:
    print(f"\n⚠️ Unstable features (CV > 0.5): {unstable}")
    print("→ Consider dropping or regularizing these")
```

---

## Eksik Test 7: ORİJİNAL VERİ KARŞILAŞTIRMASI

### Neden gerekli?

```
Playground Series = synthetic data üretilmiş
Orijinal IBM Telco verisi ile karşılaştırmak:
  1. Synthetic artifact'leri bulmak
  2. Orijinal veriyi augmentation olarak kullanmak
  3. Hangi pattern'lerin gerçek, hangilerinin yapay olduğunu anlamak
```

### Kod:

```python
# ══════════════════════════════════════
# ORIGINAL vs SYNTHETIC COMPARISON
# ══════════════════════════════════════

# Kaggle'dan indir: "blastchar/telco-customer-churn"
original = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(f"Original : {original.shape}")
print(f"Synthetic: {train.shape}")
print(f"Ratio    : {train.shape[0] / original.shape[0]:.0f}x larger")

# Dağılım karşılaştırması
for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    if col in original.columns and col in train.columns:
        orig_vals = pd.to_numeric(original[col], errors='coerce').dropna()
        synt_vals = train[col].dropna()
        ks, p = ks_2samp(orig_vals, synt_vals)
        print(f"\n{col}: KS={ks:.4f}, p={p:.4f}")
        print(f"  Original  mean={orig_vals.mean():.2f}, std={orig_vals.std():.2f}")
        print(f"  Synthetic mean={synt_vals.mean():.2f}, std={synt_vals.std():.2f}")

# Target dağılım karşılaştırması
orig_churn = original['Churn'].value_counts(normalize=True) * 100
synt_churn = train['Churn'].value_counts(normalize=True) * 100
print(f"\nChurn Rate - Original: {orig_churn.get('Yes', 0):.2f}%")
print(f"Churn Rate - Synthetic: {synt_churn.get('Yes', 0):.2f}%")

# Augmentation
original['source'] = 'original'
train['source'] = 'synthetic'
augmented = pd.concat([train, original], ignore_index=True)
print(f"\nAugmented shape: {augmented.shape}")
```

---

## Eksik Test 8: MONOTONİK İLİŞKİ KONTROLÜ

### Neden gerekli?

```
İş mantığı olarak:
  tenure arttıkça → churn AZALMALI (sadık müşteri)
  MonthlyCharges arttıkça → churn ARTMALI (pahalı = memnuniyetsiz)

Model bunu doğal öğreniyor mu?
Yoksa monotonicity constraint koymak gerekir mi?
```

### Kod:

```python
# ══════════════════════════════════════
# MONOTONICITY CHECK
# ══════════════════════════════════════

tmp = train.copy()
tmp['Churn_bin'] = (tmp['Churn'] == 'Yes').astype(int)

for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    # 10 eşit parçaya böl
    tmp[f'{col}_decile'] = pd.qcut(tmp[col], 10, duplicates='drop', labels=False)
    
    dec_analysis = tmp.groupby(f'{col}_decile')['Churn_bin'].agg(['mean', 'count'])
    dec_analysis['churn_rate'] = dec_analysis['mean'] * 100
    
    # Monotonluk testi
    rates = dec_analysis['churn_rate'].values
    is_monotonic_inc = all(rates[i] <= rates[i+1] for i in range(len(rates)-1))
    is_monotonic_dec = all(rates[i] >= rates[i+1] for i in range(len(rates)-1))
    
    mono_status = ("📈 Monotonic Increasing" if is_monotonic_inc 
                   else "📉 Monotonic Decreasing" if is_monotonic_dec 
                   else "🔀 Non-monotonic")
    
    print(f"\n{col}: {mono_status}")
    print(dec_analysis[['churn_rate', 'count']].to_string())
    
    # IF non-monotonic → model zorluk çekebilir
    # → monotonicity constraint ekle:
    # LightGBM: monotone_constraints = [1, -1, 0, ...]
    # 1 = increasing, -1 = decreasing, 0 = free
```

---

## Eksik Test 9: NULL PATTERN INTERACTION

### Neden gerekli?

```
Bu veride null yok AMA "No internet service" ve "No phone service"
aslında "gizli null" gibi davranıyor.

Sentinel temizleme SONRASI → bunlar binary oluyor.
Binary sonrası interaction'ları kontrol etmek gerekli.
```

### Kod:

```python
# ══════════════════════════════════════
# POST-CLEANING INTERACTION TEST  
# ══════════════════════════════════════

# Sentinel temizleme sonrası
train_clean = train.copy()
internet_cols = ['OnlineSecurity','OnlineBackup','DeviceProtection',
                 'TechSupport','StreamingTV','StreamingMovies']
for c in internet_cols:
    train_clean[c] = train_clean[c].replace('No internet service', 'No')
train_clean['MultipleLines'] = train_clean['MultipleLines'].replace('No phone service', 'No')

# InternetService ile her bir hizmetin etkileşimi
train_clean['Churn_bin'] = (train_clean['Churn'] == 'Yes').astype(int)

for col in internet_cols:
    ct = train_clean.groupby(['InternetService', col])['Churn_bin'].agg(['mean','count'])
    ct['churn_rate'] = ct['mean'] * 100
    print(f"\n{'='*60}")
    print(f"InternetService × {col}")
    print(ct[['churn_rate','count']].to_string())
    
# Fiber optic + No Security → %55+ churn beklenir
# DSL + Yes Security → %5-8 churn beklenir
# Bu etkileşimler FE olarak eklenebilir
```

---

## Tam Workflow — Senior Analist Adımları

```
╔══════════════════════════════════════════════════════════════════╗
║                  SENIOR ANALYST WORKFLOW                         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  PHASE 1: UNDERSTANDING (Şu an buradayız)                      ║
║  ┌──────────────────────────────────────────────────────────┐   ║
║  │ ✅ Pipeline raporu (tamamlandı)                         │   ║
║  │ ✅ Rapor yorumlama (tamamlandı)                         │   ║
║  │ ⬜ Baseline model (EKSİK — Test 1)                      │   ║
║  │ ⬜ CV stabilitesi (EKSİK — Test 2)                      │   ║
║  │ ⬜ Per-feature adversarial (EKSİK — Test 3)             │   ║
║  │ ⬜ Permutation importance (EKSİK — Test 4)              │   ║
║  │ ⬜ Segment analizi (EKSİK — Test 5)                     │   ║
║  │ ⬜ Feature stability (EKSİK — Test 6)                   │   ║
║  │ ⬜ Original data karşılaştırma (EKSİK — Test 7)         │   ║
║  │ ⬜ Monotonicity check (EKSİK — Test 8)                  │   ║
║  │ ⬜ Post-cleaning interaction (EKSİK — Test 9)           │   ║
║  └──────────────────────────────────────────────────────────┘   ║
║                           ↓                                      ║
║  PHASE 2: CLEANING & FE                                         ║
║  ┌──────────────────────────────────────────────────────────┐   ║
║  │ ⬜ Sentinel temizleme                                    │   ║
║  │ ⬜ Useless feature silme                                 │   ║
║  │ ⬜ Aggregation features                                  │   ║
║  │ ⬜ Residual features                                     │   ║
║  │ ⬜ Interaction features                                  │   ║
║  │ ⬜ Her FE adımından sonra CV ile doğrulama               │   ║
║  └──────────────────────────────────────────────────────────┘   ║
║                           ↓                                      ║
║  PHASE 3: MODELING                                               ║
║  ┌──────────────────────────────────────────────────────────┐   ║
║  │ ⬜ LightGBM + Optuna                                    │   ║
║  │ ⬜ XGBoost + Optuna                                     │   ║
║  │ ⬜ CatBoost (native categorical)                        │   ║
║  │ ⬜ Ensemble (weighted average)                           │   ║
║  └──────────────────────────────────────────────────────────┘   ║
║                           ↓                                      ║
║  PHASE 4: VALIDATION                                             ║
║  ┌──────────────────────────────────────────────────────────┐   ║
║  │ ⬜ CV-LB korelasyon testi (ilk submission)              │   ║
║  │ ⬜ Prediction distribution analizi                      │   ║
║  │ ⬜ Error analizi (hangi segmentlerde hata yapıyor?)     │   ║
║  └──────────────────────────────────────────────────────────┘   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Hangi Testler KRİTİK, Hangileri OPSİYONEL?

```
╔════╦═══════════════════════════════════╦══════════╦════════════════════════╗
║  # ║ Test                              ║ Öncelik  ║ Neden                  ║
╠════╬═══════════════════════════════════╬══════════╬════════════════════════╣
║  1 ║ Baseline model                    ║ 🔴 ŞART  ║ Karşılaştırma         ║
║    ║                                   ║          ║ noktası olmadan FE     ║
║    ║                                   ║          ║ yapılmaz               ║
╠════╬═══════════════════════════════════╬══════════╬════════════════════════╣
║  2 ║ CV stabilitesi                    ║ 🔴 ŞART  ║ Unstable CV =         ║
║    ║                                   ║          ║ leaderboard sürpriz    ║
╠════╬═══════════════════════════════════╬══════════╬════════════════════════╣
║  5 ║ Segment analizi                   ║ 🔴 ŞART  ║ FE fikirleri buradan  ║
║    ║                                   ║          ║ çıkar                  ║
╠════╬═══════════════════════════════════╬══════════╬════════════════════════╣
║  7 ║ Original data karşılaştırma       ║ 🟡 YÜKSEK║ Playground Series'e   ║
║    ║                                   ║          ║ özgü, skor kazandırır  ║
╠════╬═══════════════════════════════════╬══════════╬════════════════════════╣
║  8 ║ Monotonicity check                ║ 🟡 YÜKSEK║ Constraint ekleme     ║
║    ║                                   ║          ║ kararı için gerekli    ║
╠════╬═══════════════════════════════════╬══════════╬════════════════════════╣
║  4 ║ Permutation importance            ║ 🟡 ORTA  ║ RF importance'ı       ║
║    ║                                   ║          ║ doğrular               ║
╠════╬═══════════════════════════════════╬══════════╬════════════════════════╣
║  3 ║ Per-feature adversarial           ║ 🟢 DÜŞÜK ║ AUC=0.51 olduğu için ║
║    ║                                   ║          ║ zaten sorun yok        ║
╠════╬═══════════════════════════════════╬══════════╬════════════════════════╣
║  6 ║ Feature stability                 ║ 🟢 DÜŞÜK ║ İleri seviye          ║
║    ║                                   ║          ║ optimizasyon           ║
╠════╬═══════════════════════════════════╬══════════╬════════════════════════╣
║  9 ║ Post-cleaning interaction         ║ 🟢 DÜŞÜK ║ FE aşamasında da      ║
║    ║                                   ║          ║ yapılabilir            ║
╚════╩═══════════════════════════════════╩══════════╩════════════════════════╝
```

> **Özet:** Direkt FE'ye geçme. Önce **baseline kur** (Test 1), **CV stabilitesini kontrol et** (Test 2), **segment analizi yap** (Test 5). Bu üçü olmadan yapılan FE kör uçuş gibi — nereye gittiğini göremezsin.



Kodun çok uzun olduğu için iki parçada veriyorum. **Her ikisini de aynı hücreye yapıştır.**

## PARÇA 1: Imports + Reporter + Helpers + Plots

```python
!pip install tabulate -q

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, ks_2samp, chi2_contingency, spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from IPython.display import display, Image
import lightgbm as lgb
import base64, os, warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="darkgrid", palette="muted")
PLOT_DIR = "/kaggle/working/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def safe_encode_target(series):
    if series.dtype == "O":
        return pd.Series(LabelEncoder().fit_transform(series.astype(str)), index=series.index)
    return series


class Reporter:
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
    </style></head><body>
    <h1 style="text-align:center;color:#58a6ff">🚀 SENIOR DATA PIPELINE REPORT</h1>"""

    def __init__(self, plot_dir=PLOT_DIR):
        self.h = [self.CSS]; self.m = ["# 🚀 SENIOR DATA PIPELINE REPORT\n\n"]
        self.plot_dir = plot_dir; self.plot_count = 0

    def log(self, text, cls=""):
        print(text)
        c = f" {cls}" if cls else ""
        self.h.append(f'<div class="info-line{c}">{text}</div>')
        b = "**" if cls in ("critical","warn") else ""
        self.m.append(f"> {b}{text}{b}\n\n")

    def decision(self, text, positive=True):
        cn = "decision-yes" if positive else "decision-no"
        px = "✅" if positive else "⚠️"
        full = f"{px} DECISION: {text}"
        print(f"  {full}")
        self.h.append(f'<div class="decision {cn}">{full}</div>')
        self.m.append(f"> **{full}**\n\n")

    def section(self, title):
        print(f"\n{'═'*80}\n{title}\n{'═'*80}")
        self.h.append(f'<div class="section"><h2>{title}</h2>')
        self.m.append(f"\n---\n\n## {title}\n\n")

    def end(self): self.h.append("</div>")

    def table(self, styled):
        display(styled); self.h.append(styled.to_html())
        df = styled.data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(filter(None,(str(x) for x in c))) for c in df.columns]
        if isinstance(df.index, pd.MultiIndex): df = df.reset_index()
        try: self.m.append(f"\n{df.to_markdown(floatfmt='.4f')}\n\n")
        except: self.m.append(f"\n```\n{df.to_string()}\n```\n\n")

    def plot(self, fig, title="plot"):
        self.plot_count += 1
        fname = f"{self.plot_count:02d}_{title.replace(' ','_').lower()}.png"
        fpath = os.path.join(self.plot_dir, fname)
        fig.savefig(fpath, dpi=150, bbox_inches="tight", facecolor="#0d1117", edgecolor="none")
        plt.close(fig)
        display(Image(filename=fpath))
        with open(fpath,"rb") as f: b64 = base64.b64encode(f.read()).decode()
        self.h.append(f'<div class="plot-container"><img src="data:image/png;base64,{b64}" alt="{title}"></div>')
        self.m.append(f"\n![{title}](plots/{fname})\n\n")

    def save(self, html_path, md_path):
        self.h.append("</body></html>")
        with open(html_path,"w",encoding="utf-8") as f: f.write("\n".join(self.h))
        print(f"\n✅ HTML saved  : {html_path}")
        with open(md_path,"w",encoding="utf-8") as f: f.write("\n".join(self.m))
        print(f"✅ Markdown saved: {md_path}")
        print(f"📁 Plots         : {self.plot_dir}/ ({self.plot_count} files)")


# ── PLOT HELPERS ──
DARK="#0d1117"; CARD="#161b22"; BLUE="#58a6ff"; TXT="#c9d1d9"; GRID="#30363d"

def dark_style(ax, title=""):
    ax.set_facecolor(CARD); ax.figure.set_facecolor(DARK)
    ax.title.set_color(BLUE); ax.xaxis.label.set_color(TXT); ax.yaxis.label.set_color(TXT)
    ax.tick_params(colors=TXT)
    for s in ax.spines.values(): s.set_color(GRID)
    if title: ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

def plot_target_dist(train, target_col, task):
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    if task=="classification":
        vc=train[target_col].value_counts()
        cl=["#3fb950","#f85149"] if len(vc)==2 else sns.color_palette("muted",len(vc))
        axes[0].bar(vc.index.astype(str),vc.values,color=cl,edgecolor=GRID); dark_style(axes[0],f"{target_col} — Count"); axes[0].set_ylabel("Count")
        axes[1].pie(vc.values,labels=vc.index.astype(str),autopct="%1.1f%%",colors=cl,textprops={"color":TXT},wedgeprops={"edgecolor":GRID})
        axes[1].set_facecolor(DARK); axes[1].set_title(f"{target_col} — Ratio",color=BLUE,fontsize=14,fontweight="bold")
    else:
        axes[0].hist(train[target_col].dropna(),bins=50,color=BLUE,edgecolor=GRID,alpha=0.8); dark_style(axes[0],f"{target_col} — Distribution")
        sns.boxplot(x=train[target_col].dropna(),ax=axes[1],color=BLUE); dark_style(axes[1],f"{target_col} — Boxplot")
    fig.set_facecolor(DARK); fig.tight_layout(); return fig

def plot_correlation_heatmap(train, num_cols, target_col):
    data=train[num_cols].copy(); data[target_col]=safe_encode_target(train[target_col].copy())
    corr=data.corr(); mask=np.triu(np.ones_like(corr,dtype=bool),k=1); n=len(corr)
    fig,ax=plt.subplots(figsize=(max(8,n),max(6,n*0.8)))
    sns.heatmap(corr,mask=mask,annot=True,fmt=".2f",cmap="RdBu_r",center=0,vmin=-1,vmax=1,ax=ax,linewidths=0.5,linecolor=GRID,cbar_kws={"shrink":0.8})
    dark_style(ax,"Correlation Matrix"); ax.tick_params(axis="x",rotation=45); fig.tight_layout(); return fig

def plot_cramers_heatmap(train, cat_cols, target_col):
    ac=cat_cols+[target_col]; n=len(ac); mat=pd.DataFrame(np.zeros((n,n)),index=ac,columns=ac)
    for i in range(n):
        for j in range(i,n):
            v=cramers_v(train[ac[i]].fillna("_NA_"),train[ac[j]].fillna("_NA_")); mat.iloc[i,j]=v; mat.iloc[j,i]=v
    mask=np.triu(np.ones_like(mat,dtype=bool),k=1)
    fig,ax=plt.subplots(figsize=(max(10,n*0.9),max(8,n*0.7)))
    sns.heatmap(mat.astype(float),mask=mask,annot=True,fmt=".2f",cmap="YlOrRd",vmin=0,vmax=1,ax=ax,linewidths=0.5,linecolor=GRID,cbar_kws={"shrink":0.8})
    dark_style(ax,"Cramer's V Matrix"); ax.tick_params(axis="x",rotation=45); fig.tight_layout(); return fig

def plot_feature_importance(imp_df, top_n=15):
    df=imp_df.head(top_n).sort_values("Importance"); fig,ax=plt.subplots(figsize=(10,max(4,top_n*0.35)))
    bars=ax.barh(df["Feature"],df["Importance"],color=BLUE,edgecolor=GRID)
    for bar,val in zip(bars,df["Importance"]): ax.text(bar.get_width()+0.002,bar.get_y()+bar.get_height()/2,f"{val:.4f}",va="center",color=TXT,fontsize=10)
    dark_style(ax,f"Top {top_n} Feature Importance"); ax.set_xlabel("Importance"); fig.tight_layout(); return fig

def plot_bivariate_top(train, target_col, cat_cols, task, top_n=6):
    tmp=train.copy(); tmp[target_col]=safe_encode_target(tmp[target_col].copy())
    cols=cat_cols[:top_n]; nc=3; nr=(len(cols)+nc-1)//nc
    fig,axes=plt.subplots(nr,nc,figsize=(15,nr*4)); axes=axes.flatten() if nr*nc>1 else [axes]
    for i,col in enumerate(cols):
        grp=tmp.groupby(col)[target_col].mean()*(100 if task=="classification" else 1); grp=grp.sort_values(ascending=False)
        axes[i].barh(grp.index.astype(str),grp.values,color=sns.color_palette("RdYlGn_r",len(grp)),edgecolor=GRID); dark_style(axes[i],col)
        axes[i].set_xlabel("Target Rate (%)" if task=="classification" else f"Mean {target_col}")
    for j in range(len(cols),len(axes)): axes[j].set_visible(False)
    fig.suptitle("Bivariate — Categorical vs Target",color=BLUE,fontsize=16,fontweight="bold"); fig.set_facecolor(DARK); fig.tight_layout(rect=[0,0,1,0.95]); return fig

def plot_numeric_dist(train, test, num_cols):
    n=len(num_cols)
    if n==0: return None
    nc=min(3,n); nr=(n+nc-1)//nc; fig,axes=plt.subplots(nr,nc,figsize=(15,nr*3.5)); axes=np.array(axes).flatten() if n>1 else [axes]
    for i,col in enumerate(num_cols):
        axes[i].hist(train[col].dropna(),bins=50,alpha=0.6,color="#58a6ff",label="Train",edgecolor=GRID)
        axes[i].hist(test[col].dropna(),bins=50,alpha=0.5,color="#f0883e",label="Test",edgecolor=GRID)
        dark_style(axes[i],col); axes[i].legend(fontsize=9,facecolor=CARD,edgecolor=GRID,labelcolor=TXT)
    for j in range(n,len(axes)): axes[j].set_visible(False)
    fig.suptitle("Numeric Distributions — Train vs Test",color=BLUE,fontsize=16,fontweight="bold"); fig.set_facecolor(DARK); fig.tight_layout(rect=[0,0,1,0.95]); return fig

def plot_iv_chart(div):
    df=div.sort_values("IV",ascending=True)
    colors=["#f85149" if v>0.5 else "#3fb950" if v>0.3 else "#f0883e" if v>0.1 else "#58a6ff" if v>0.02 else "#484f58" for v in df["IV"]]
    fig,ax=plt.subplots(figsize=(10,max(4,len(df)*0.35))); ax.barh(df.index,df["IV"],color=colors,edgecolor=GRID)
    ax.axvline(x=0.02,color="#484f58",linestyle="--",alpha=0.7,label="Weak"); ax.axvline(x=0.1,color="#f0883e",linestyle="--",alpha=0.7,label="Medium"); ax.axvline(x=0.3,color="#3fb950",linestyle="--",alpha=0.7,label="Strong")
    dark_style(ax,"Information Value (IV)"); ax.set_xlabel("IV"); ax.legend(facecolor=CARD,edgecolor=GRID,labelcolor=TXT,fontsize=9); fig.tight_layout(); return fig

def cramers_v(x,y):
    ct=pd.crosstab(x,y); chi2=chi2_contingency(ct)[0]; n=ct.sum().sum(); md=min(ct.shape)-1
    return np.sqrt(chi2/(n*md)) if md>0 else 0.0

def calc_iv(data,feature,target):
    g=data.groupby(feature)[target].agg(["sum","count"]); g.columns=["ev","tot"]; g["nev"]=g["tot"]-g["ev"]
    te,tne=g["ev"].sum(),g["nev"].sum(); g["pe"]=(g["ev"]/te).clip(1e-4); g["pn"]=(g["nev"]/tne).clip(1e-4)
    g["woe"]=np.log(g["pn"]/g["pe"]); g["iv"]=(g["pn"]-g["pe"])*g["woe"]; return g["iv"].sum()

def drift_label(ks,p):
    if p>=0.05: return "✅ NONE"
    if ks<0.05: return "🟡 Statistical Only"
    if ks<0.10: return "🟠 Mild Drift"
    return "🔴 Severe Drift"
```

## PARÇA 2: Ana Pipeline Fonksiyonu + Çalıştırma (Aynı hücreye devam)

```python
def full_senior_data_pipeline(path, train_name, test_name, target_col, id_col=None,
                               original_path=None):
    R = Reporter()
    actions = []
    results = {}

    train = pd.read_csv(f"{path}/{train_name}")
    test  = pd.read_csv(f"{path}/{test_name}")

    feats = [c for c in train.columns if c not in [target_col, id_col]]
    THR=25
    num_cols = [c for c in feats if train[c].nunique()>THR]
    cat_cols = [c for c in feats if train[c].nunique()<=THR]
    nu = train[target_col].nunique()
    task = "classification" if (train[target_col].dtype=="O" or nu<=20) else "regression"
    y_full = safe_encode_target(train[target_col].copy())

    # ════════════════════════════════════════════════════
    # SECTION 0: OVERVIEW
    # ════════════════════════════════════════════════════
    R.section("📋 0. PIPELINE OVERVIEW")
    R.log(f"🚀 Pipeline | ID: {id_col} | Target: {target_col}")
    R.log(f"📊 {len(feats)} Features: {len(num_cols)} Numeric, {len(cat_cols)} Categorical")
    R.log(f"🎯 Task: {task.upper()} (target has {nu} unique values)", cls="ok")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 1: MEMORY & SHAPE
    # ════════════════════════════════════════════════════
    R.section("🌟 1. MEMORY & SHAPE")
    for nm,df in zip(["Train","Test"],[train,test]):
        mem=df.memory_usage(deep=True).sum()/1e6; R.log(f"📐 {nm}: {df.shape[0]:,} × {df.shape[1]} | 💾 {mem:.1f} MB")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 2: GENERAL AUDIT
    # ════════════════════════════════════════════════════
    R.section("🌟 2. GENERAL DATA AUDIT")
    role={c:("Numeric" if c in num_cols else "Categorical") for c in feats}
    met={"Meta":pd.DataFrame({"Dtype":train[feats].dtypes.astype(str),"Role":pd.Series(role)}),
         "N-Unique":pd.concat([train[feats].nunique().rename("Train"),test[feats].nunique().rename("Test")],axis=1),
         "Null Pct":pd.concat([(train[feats].isnull().mean()*100).rename("Train"),(test[feats].isnull().mean()*100).rename("Test")],axis=1),
         "Top Val Pct":pd.concat([train[feats].apply(lambda x:x.value_counts(normalize=True).iloc[0]*100 if not x.dropna().empty else 0).rename("Train"),
                                   test[feats].apply(lambda x:x.value_counts(normalize=True).iloc[0]*100 if not x.dropna().empty else 0).rename("Test")],axis=1)}
    da=pd.concat(met.values(),axis=1,keys=met.keys()); da[("Diff","Null Gap")]=(da[("Null Pct","Test")]-da[("Null Pct","Train")]).round(2)
    da=da.sort_values(by=[("Meta","Role"),("N-Unique","Train")],ascending=[False,False])
    R.table(da.style.background_gradient(cmap="YlGnBu",subset=[c for c in da.columns if c[0] in ["N-Unique","Null Pct","Top Val Pct"]]).background_gradient(cmap="Reds",subset=[("Diff","Null Gap")]).format(precision=2))
    hn=[c for c in feats if train[c].isnull().mean()>0.3]
    if hn: actions.append(f"🗑️ >30% null: {hn}")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 3: TARGET DISTRIBUTION
    # ════════════════════════════════════════════════════
    R.section("🌟 3. TARGET DISTRIBUTION")
    if task=="classification":
        vc=train[target_col].value_counts(); vcp=train[target_col].value_counts(normalize=True)*100
        dt=pd.DataFrame({"Count":vc,"Pct (%)":vcp,"Bar":vcp})
        R.table(dt.style.bar(subset=["Bar"],color="#636efa").format({"Pct (%)":"{:.2f}","Bar":"{:.1f}"}))
        ir=vc.min()/vc.max()
        if ir<0.2: R.log(f"⚠️ Severe imbalance! Ratio: {ir:.3f}",cls="critical"); actions.append("⚖️ Severe imbalance → SMOTE/focal loss")
        elif ir<0.5: R.log(f"⚠️ Moderate imbalance. Ratio: {ir:.3f}",cls="warn"); actions.append("⚖️ Moderate imbalance → class_weight='balanced'")
        else: R.log(f"✅ Balanced. Ratio: {ir:.3f}",cls="ok")
    else:
        desc=train[target_col].describe().to_frame().T; desc["skew"]=skew(train[target_col].dropna())
        R.table(desc.style.format(precision=3))
        if abs(desc["skew"].iloc[0])>1: R.log("⚠️ Skewed → log1p",cls="warn"); actions.append("📐 Target skewed → log1p")
    R.plot(plot_target_dist(train,target_col,task),"target_distribution"); R.end()

    # ════════════════════════════════════════════════════
    # SECTION 4: DUPLICATES & CONSTANTS
    # ════════════════════════════════════════════════════
    R.section("🌟 4. DUPLICATES & CONSTANT COLUMNS")
    dup=train.duplicated().sum()
    if dup: R.log(f"⚠️ {dup:,} duplicates ({dup/len(train)*100:.2f}%)",cls="warn"); actions.append(f"🔄 {dup} duplicates → drop")
    else: R.log("✅ No duplicates.",cls="ok")
    const=[]
    for c in feats:
        tp=train[c].value_counts(normalize=True).iloc[0]*100
        if tp>=99.5: const.append({"Feature":c,"Top%":tp,"Status":"🔴 CONSTANT"})
        elif tp>=95: const.append({"Feature":c,"Top%":tp,"Status":"🟡 QUASI"})
    if const:
        dc=pd.DataFrame(const).set_index("Feature")
        R.table(dc.style.map(lambda x:"color:#f85149" if "CONSTANT" in str(x) else "color:#f0883e",subset=["Status"]).format({"Top%":"{:.2f}"}))
        drc=[r["Feature"] for r in const if r["Top%"]>=99.5]
        if drc: actions.append(f"🗑️ Constants: {drc}")
    else: R.log("✅ No constant columns.",cls="ok")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 5: STATISTICAL RISK & DRIFT
    # ════════════════════════════════════════════════════
    R.section("🌟 5. STATISTICAL RISK & DRIFT")
    if num_cols:
        stats=[]
        for c in num_cols:
            tr,ts=train[c].dropna(),test[c].dropna(); q1,q3=tr.quantile(0.25),tr.quantile(0.75); iqr=q3-q1
            out=((tr<q1-1.5*iqr)|(tr>q3+1.5*iqr)).mean()*100; ks,p=ks_2samp(tr,ts)
            stats.append({"Column":c,"Skew":skew(tr),"Outlier%":out,"KS":ks,"P":p,"Drift":drift_label(ks,p)})
        vd=train[num_cols].dropna(); vr=[variance_inflation_factor(vd.values,i) for i in range(len(num_cols))]
        tmi=train[num_cols+[target_col]].dropna(); tmi[target_col]=safe_encode_target(tmi[target_col])
        mi_fn=mutual_info_classif if task=="classification" else mutual_info_regression
        mi=mi_fn(tmi[num_cols],tmi[target_col],random_state=42)
        da2=pd.DataFrame(stats).set_index("Column"); da2["VIF"]=vr; da2["MI"]=mi
        R.table(da2.style.background_gradient(cmap="OrRd",subset=["VIF","Skew"]).background_gradient(cmap="YlGn",subset=["MI"])
                .map(lambda x:"color:red;font-weight:bold" if "Severe" in str(x) else("color:orange" if "Mild" in str(x) or "Statistical" in str(x) else "color:green"),subset=["Drift"]).format(precision=3))
        rd=[s["Column"] for s in stats if "Severe" in s["Drift"] or "Mild" in s["Drift"]]
        if rd: actions.append(f"📉 Real drift: {rd}")
        hv=[num_cols[i] for i,v in enumerate(vr) if v>10]
        if hv: actions.append(f"📊 VIF>10: {hv}")
        fig_nd=plot_numeric_dist(train,test,num_cols)
        if fig_nd: R.plot(fig_nd,"numeric_distributions")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 6: NUMERIC CORRELATION
    # ════════════════════════════════════════════════════
    R.section("🌟 6. NUMERIC CORRELATION")
    if len(num_cols)>=2:
        corr=train[num_cols].corr(); hcp=[]
        for i in range(len(num_cols)):
            for j in range(i+1,len(num_cols)):
                r=corr.iloc[i,j]
                if abs(r)>0.85: hcp.append({"F1":num_cols[i],"F2":num_cols[j],"|r|":abs(r)})
        if hcp: R.log(f"⚠️ {len(hcp)} pairs |r|>0.85",cls="warn"); R.table(pd.DataFrame(hcp).style.background_gradient(cmap="Reds",subset=["|r|"]).format({"|r|":"{:.4f}"})); actions.append(f"🔗 {len(hcp)} correlated pairs")
        else: R.log("✅ No |r|>0.85 pairs.",cls="ok")
        tc=safe_encode_target(train[target_col].copy()); tcr=train[num_cols].corrwith(tc).abs().sort_values(ascending=False)
        dtc=pd.DataFrame({"Feature":tcr.index,f"|r| with {target_col}":tcr.values}).reset_index(drop=True)
        R.table(dtc.style.bar(subset=[f"|r| with {target_col}"],color="#3fb950").format({f"|r| with {target_col}":"{:.4f}"}))
        R.plot(plot_correlation_heatmap(train,num_cols,target_col),"correlation_heatmap")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 7: CRAMER'S V
    # ════════════════════════════════════════════════════
    R.section("🌟 7. CRAMER'S V — CATEGORICAL CORRELATION")
    if len(cat_cols)>=2:
        hcv=[]
        for i in range(len(cat_cols)):
            for j in range(i+1,len(cat_cols)):
                v=cramers_v(train[cat_cols[i]].fillna("_NA_"),train[cat_cols[j]].fillna("_NA_"))
                if v>0.5: hcv.append({"F1":cat_cols[i],"F2":cat_cols[j],"V":v,"Risk":"🔴 Redundant" if v>0.8 else "🟡 High"})
        if hcv: R.log(f"⚠️ {len(hcv)} pairs V>0.5",cls="warn"); R.table(pd.DataFrame(hcv).sort_values("V",ascending=False).reset_index(drop=True).style.background_gradient(cmap="Reds",subset=["V"]).format({"V":"{:.4f}"}))
        else: R.log("✅ No V>0.5 pairs.",cls="ok")
        tcv={c:cramers_v(train[c].fillna("_NA_"),train[target_col].fillna("_NA_")) for c in cat_cols}
        dtcv=pd.DataFrame.from_dict(tcv,orient="index",columns=[f"V with {target_col}"]).sort_values(f"V with {target_col}",ascending=False)
        R.table(dtcv.style.bar(subset=[f"V with {target_col}"],color="#636efa").format(precision=4))
        R.plot(plot_cramers_heatmap(train,cat_cols,target_col),"cramers_v_heatmap")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 8: HIERARCHICAL DEPENDENCY
    # ════════════════════════════════════════════════════
    R.section("🌟 8. HIERARCHICAL DEPENDENCY")
    sents=["No internet service","No phone service"]; hier=[]
    for col in cat_cols:
        for sv in sents:
            mask=train[col]==sv; sc=mask.sum()
            if sc==0: continue
            for pcol in cat_cols:
                if pcol==col: continue
                for pv in train[pcol].unique():
                    if (train[pcol]==pv).sum()!=sc: continue
                    if (mask&(train[pcol]==pv)).sum()==sc:
                        tgt=safe_encode_target(train[target_col].copy())
                        hier.append({"Child":col,"Sentinel":sv,"Parent":pcol,"Parent_Val":pv,"Count":sc,f"{target_col}_Rate%":tgt[mask].mean()*100})
    if hier:
        dh=pd.DataFrame(hier); R.log(f"⚠️ {len(dh)} hierarchical dependencies!",cls="critical")
        R.table(dh.style.format({f"{target_col}_Rate%":"{:.2f}"}).background_gradient(cmap="OrRd",subset=["Count"]))
        R.log("💡 Replace sentinel → 'No' or NaN",cls="warn"); actions.append(f"🔗 Hierarchical: {list(dh['Child'].unique())}")
    else: R.log("✅ No hierarchical dependencies.",cls="ok")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 9: DERIVED FEATURES
    # ════════════════════════════════════════════════════
    R.section("🌟 9. DERIVED FEATURE RELATIONSHIPS")
    if len(num_cols)>=2:
        derived=[]; cc=num_cols[:10]
        for i in range(len(cc)):
            for j in range(i+1,len(cc)):
                for k in range(len(cc)):
                    if k in(i,j): continue
                    a,b,c_=train[cc[i]],train[cc[j]],train[cc[k]]; prod=a*b
                    if prod.std()>0 and c_.std()>0:
                        rp=prod.corr(c_)
                        if abs(rp)>0.90: derived.append({"Rel":f"{cc[i]} × {cc[j]}","Target":cc[k],"|r|":abs(rp),"Type":"Product"})
                    if(b.abs()>1e-9).all():
                        rat=a/b
                        if rat.std()>0 and c_.std()>0:
                            rr=rat.corr(c_)
                            if abs(rr)>0.90: derived.append({"Rel":f"{cc[i]} / {cc[j]}","Target":cc[k],"|r|":abs(rr),"Type":"Ratio"})
        if derived:
            dd=pd.DataFrame(derived).sort_values("|r|",ascending=False).reset_index(drop=True)
            R.log(f"⚠️ {len(dd)} derived relationships:",cls="critical")
            R.table(dd.style.background_gradient(cmap="Reds",subset=["|r|"]).format({"|r|":"{:.4f}"}))
            actions.append(f"🔗 Derived: {dd['Target'].unique().tolist()} → residual")
        else: R.log("✅ No derived relationships.",cls="ok")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 10: CATEGORICAL OVERLAP
    # ════════════════════════════════════════════════════
    R.section("🌟 10. CATEGORICAL OVERLAP")
    if cat_cols:
        rc=[]; 
        for c in cat_cols:
            trs,tss=set(train[c].dropna().unique()),set(test[c].dropna().unique())
            ov=(len(tss&trs)/len(tss)*100) if tss else 100
            rc.append({"Col":c,"Tr":len(trs),"Ts":len(tss),"Overlap%":ov,"Only_Test":len(tss-trs)})
        drc=pd.DataFrame(rc).set_index("Col")
        R.table(drc.style.map(lambda x:"color:red" if x<100 else "color:green",subset=["Overlap%"]))
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 11: TARGET BIVARIATE
    # ════════════════════════════════════════════════════
    R.section("🌟 11. TARGET BIVARIATE")
    if cat_cols:
        tmp=train.copy()
        if tmp[target_col].dtype=="O":
            le=LabelEncoder(); le.fit(tmp[target_col].astype(str)); tmp[target_col]=le.transform(tmp[target_col].astype(str))
            tl=le.classes_[1] if len(le.classes_)>1 else "Positive"
        else: tl="Target"
        bv=[]
        for c in cat_cols:
            grp=tmp.groupby(c)[target_col]
            for idx,val in (grp.mean()*100).items(): bv.append({"Feature":c,"Category":idx,f"{tl}_Rate%":val,"Count":grp.count().get(idx,0)})
        dbv=pd.DataFrame(bv).set_index(["Feature","Category"]); rc2=[c for c in dbv.columns if "Rate" in c][0]
        R.table(dbv.style.background_gradient(cmap="RdYlGn_r",subset=[rc2]).format(precision=2))
        R.plot(plot_bivariate_top(train,target_col,cat_cols,task),"bivariate_top")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 12: INFORMATION VALUE
    # ════════════════════════════════════════════════════
    R.section("🌟 12. INFORMATION VALUE (IV)")
    div = None
    if task=="classification" and cat_cols:
        tmp_iv=train.copy(); tmp_iv[target_col]=safe_encode_target(tmp_iv[target_col])
        ivr={c:calc_iv(tmp_iv,c,target_col) for c in cat_cols}
        div=pd.DataFrame.from_dict(ivr,orient="index",columns=["IV"]).sort_values("IV",ascending=False)
        div["Strength"]=div["IV"].apply(lambda x:"🔴 Suspicious" if x>0.5 else "🟢 Strong" if x>0.3 else "🟡 Medium" if x>0.1 else "🔵 Weak" if x>0.02 else "⚪ Useless")
        R.table(div.style.bar(subset=["IV"],color="#636efa").format({"IV":"{:.4f}"}))
        R.plot(plot_iv_chart(div),"information_value")
        sus=div[div["IV"]>0.5].index.tolist()
        if sus: R.log(f"⚠️ IV>0.5 risk: {sus}",cls="critical"); actions.append(f"🚨 IV suspicious: {sus}")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 13: ADVERSARIAL & IMPORTANCE (RF-based EDA)
    # ════════════════════════════════════════════════════
    R.section("🌟 13. ADVERSARIAL & FEATURE IMPORTANCE (EDA)")
    adf=pd.concat([train[feats],test[feats]],axis=0); ya=[0]*len(train)+[1]*len(test)
    for c in adf.select_dtypes(["object","category"]).columns: adf[c]=pd.factorize(adf[c])[0]
    rf=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42,n_jobs=-1)
    auc_adv=cross_val_score(rf,adf.fillna(-1),ya,cv=5,scoring="roc_auc").mean()
    if auc_adv>0.8: R.log(f"🕵️ Adversarial AUC: {auc_adv:.4f} — 🔴 Severe!",cls="critical")
    elif auc_adv>0.6: R.log(f"🕵️ Adversarial AUC: {auc_adv:.4f} — ⚠️ Mild drift.",cls="warn")
    else: R.log(f"🕵️ Adversarial AUC: {auc_adv:.4f} — ✅ Safe.",cls="ok")
    timp=train[feats+[target_col]].dropna()
    for c in timp.select_dtypes(["object","category"]).columns: timp[c]=pd.Series(LabelEncoder().fit_transform(timp[c].astype(str)),index=timp.index)
    rf_imp=(RandomForestClassifier if task=="classification" else RandomForestRegressor)(n_estimators=100,max_depth=8,random_state=42,n_jobs=-1)
    rf_imp.fit(timp.drop(columns=[target_col]),timp[target_col])
    imp=pd.DataFrame({"Feature":feats,"Importance":rf_imp.feature_importances_}).sort_values("Importance",ascending=False).reset_index(drop=True)
    imp["Cumulative"]=imp["Importance"].cumsum(); imp["Rank"]=range(1,len(imp)+1)
    R.table(imp.style.bar(subset=["Importance"],color="#636efa").background_gradient(cmap="YlGn",subset=["Cumulative"]).format({"Importance":"{:.4f}","Cumulative":"{:.4f}"}))
    R.plot(plot_feature_importance(imp),"feature_importance")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 14: MISSING PATTERNS
    # ════════════════════════════════════════════════════
    R.section("🌟 14. MISSING VALUE PATTERNS")
    nc=[c for c in feats if train[c].isnull().any()]
    if nc:
        ns=[]
        for c in nc:
            np_=train[c].isnull().mean()*100; tgt2=safe_encode_target(train[target_col].copy())
            rn=tgt2[train[c].isnull()].mean()*100; rnp=tgt2[train[c].notnull()].mean()*100
            ns.append({"Feature":c,"Null%":np_,"Rate_NULL%":rn,"Rate_PRESENT%":rnp,"Gap":abs(rn-rnp)})
        dn=pd.DataFrame(ns).set_index("Feature").sort_values("Gap",ascending=False)
        R.table(dn.style.background_gradient(cmap="OrRd",subset=["Gap","Null%"]).format(precision=2))
        inull=dn[dn["Gap"]>5].index.tolist()
        if inull: actions.append(f"🏷️ is_null flags: {inull}")
    else: R.log("✅ No missing values.",cls="ok")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 15: PLAYGROUND SERIES NOTE
    # ════════════════════════════════════════════════════
    R.section("🌟 15. PLAYGROUND SERIES — ORIGINAL DATA")
    R.log("📌 This is a Playground Series competition — original data can be used for augmentation.")
    R.log("📌 Origin: IBM Telco Customer Churn (available on Kaggle).")
    R.log("💡 pd.concat([train, original]) + source flag.",cls="warn")
    R.end()

    # ══════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════
    #       🧪 VALIDATION TESTS (SECTIONS 16-24)
    # ══════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════

    X_all = train[feats].copy()
    for c in X_all.select_dtypes(['object','category']).columns: X_all[c]=X_all[c].astype('category')

    # ════════════════════════════════════════════════════
    # SECTION 16: BASELINE MODEL
    # ════════════════════════════════════════════════════
    R.section("🧪 16. BASELINE MODEL (Zero FE)")
    R.log("Running 5-fold LightGBM with NO feature engineering...")
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    bl_scores=[]; bl_models=[]
    for fold,(tr_i,va_i) in enumerate(skf.split(X_all,y_full)):
        m=lgb.LGBMClassifier(n_estimators=1000,learning_rate=0.05,max_depth=6,num_leaves=31,subsample=0.8,colsample_bytree=0.8,is_unbalance=True,random_state=42,verbose=-1,n_jobs=-1)
        m.fit(X_all.iloc[tr_i],y_full.iloc[tr_i],eval_set=[(X_all.iloc[va_i],y_full.iloc[va_i])],callbacks=[lgb.early_stopping(50),lgb.log_evaluation(0)])
        p=m.predict_proba(X_all.iloc[va_i])[:,1]; a=roc_auc_score(y_full.iloc[va_i],p)
        bl_scores.append(a); bl_models.append(m); R.log(f"  Fold {fold}: AUC = {a:.5f}")
    bl_mean=np.mean(bl_scores); bl_std=np.std(bl_scores)
    results['baseline_mean']=bl_mean; results['baseline_std']=bl_std
    R.log(f"\n📊 BASELINE: {bl_mean:.5f} ± {bl_std:.5f}",cls="ok")

    if bl_mean>0.90: R.decision("Baseline >0.90 — very strong. Focus on ensemble & tuning. Expect +0.001-0.005 from FE.",True)
    elif bl_mean>0.85: R.decision("Baseline 0.85-0.90 — good. FE can push +0.005-0.015. Focus on top feature interactions.",True)
    elif bl_mean>0.80: R.decision("Baseline 0.80-0.85 — moderate. Aggressive FE, encoding, augmentation needed.",True)
    else: R.decision("Baseline <0.80 — weak. Rethink approach: different encoding, target encoding, stacking.",False)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 17: CV STABILITY
    # ════════════════════════════════════════════════════
    R.section("🧪 17. CV STABILITY ACROSS SEEDS")
    seeds=[42,123,456,789,2024]; seed_scores=[]
    for seed in seeds:
        sk=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed); fsc=[]
        for tr_i,va_i in sk.split(X_all,y_full):
            m=lgb.LGBMClassifier(n_estimators=500,learning_rate=0.05,max_depth=6,is_unbalance=True,random_state=seed,verbose=-1,n_jobs=-1)
            m.fit(X_all.iloc[tr_i],y_full.iloc[tr_i],eval_set=[(X_all.iloc[va_i],y_full.iloc[va_i])],callbacks=[lgb.early_stopping(50),lgb.log_evaluation(0)])
            fsc.append(roc_auc_score(y_full.iloc[va_i],m.predict_proba(X_all.iloc[va_i])[:,1]))
        seed_scores.append(np.mean(fsc)); R.log(f"  Seed {seed}: AUC = {np.mean(fsc):.5f}")
    s_std=np.std(seed_scores); s_range=max(seed_scores)-min(seed_scores)
    results['cv_std']=s_std
    R.log(f"\n📊 Seed std: {s_std:.5f} | Range: {s_range:.5f}")

    if s_std<0.001: R.decision("Very stable (std<0.001). Trust CV, 5-fold enough.",True)
    elif s_std<0.003: R.decision("Reasonably stable (std<0.003). Use fixed seed.",True)
    elif s_std<0.005: R.decision("Moderate variance. Use RepeatedStratifiedKFold (3×5).",False)
    else: R.decision("UNSTABLE (std>0.005). Use 10-fold or repeated CV. Check leakage.",False); actions.append("🚨 CV unstable → repeated CV")

    outlier_folds=sum(1 for s in bl_scores if abs(s-bl_mean)>2*bl_std)
    if outlier_folds>0: R.decision(f"{outlier_folds} outlier fold(s). Investigate data distribution in those folds.",False)
    else: R.decision("All folds consistent. No outliers.",True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 18: PER-FEATURE ADVERSARIAL
    # ════════════════════════════════════════════════════
    R.section("🧪 18. PER-FEATURE ADVERSARIAL IMPORTANCE")
    rf_adv=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42,n_jobs=-1)
    rf_adv.fit(adf.fillna(-1),ya)
    adv_imp=pd.DataFrame({"Feature":feats,"Adv_Imp":rf_adv.feature_importances_}).sort_values("Adv_Imp",ascending=False).reset_index(drop=True)
    results['adv_auc']=auc_adv
    R.log(f"📊 Adversarial AUC: {auc_adv:.4f}")
    R.table(adv_imp.head(10).style.bar(subset=["Adv_Imp"],color="#f0883e").format({"Adv_Imp":"{:.4f}"}))

    if auc_adv<0.52: R.decision("AUC<0.52 → NO drift. Train/test same distribution.",True)
    elif auc_adv<0.60: R.decision("AUC 0.52-0.60 → minimal drift. Monitor top features.",True)
    elif auc_adv<0.70: R.decision("AUC 0.60-0.70 → moderate drift. Consider dropping/normalizing top adversarial features.",False); actions.append(f"⚠️ Drift features: {adv_imp.head(3)['Feature'].tolist()}")
    else: R.decision("AUC>0.70 → SEVERE drift/leakage! STOP and investigate.",False); actions.append("🚨 Severe drift!")

    hi_adv=adv_imp[adv_imp["Adv_Imp"]>0.10]["Feature"].tolist()
    if hi_adv: R.decision(f"Individual drifted features: {hi_adv}. Check distributions.",False)
    else: R.decision("No single feature dominates adversarial model.",True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 19: PERMUTATION IMPORTANCE
    # ════════════════════════════════════════════════════
    R.section("🧪 19. PERMUTATION IMPORTANCE")
    last_m=bl_models[-1]; last_fold=list(skf.split(X_all,y_full))[-1]
    Xvp=X_all.iloc[last_fold[1]]; yvp=y_full.iloc[last_fold[1]]
    perm=permutation_importance(last_m,Xvp,yvp,n_repeats=10,random_state=42,scoring='roc_auc',n_jobs=-1)
    perm_df=pd.DataFrame({"Feature":feats,"Perm_Mean":perm.importances_mean,"Perm_Std":perm.importances_std})
    lgb_fi=pd.DataFrame({"Feature":feats,"LGB_Imp":last_m.feature_importances_})
    comp=pd.merge(lgb_fi,perm_df,on="Feature")
    comp["R_LGB"]=comp["LGB_Imp"].rank(ascending=False); comp["R_Perm"]=comp["Perm_Mean"].rank(ascending=False)
    comp["R_Diff"]=abs(comp["R_LGB"]-comp["R_Perm"])
    comp=comp.sort_values("Perm_Mean",ascending=False).reset_index(drop=True)
    R.table(comp.style.bar(subset=["Perm_Mean"],color="#3fb950").background_gradient(cmap="OrRd",subset=["R_Diff"]).format({"Perm_Mean":"{:.4f}","Perm_Std":"{:.4f}","LGB_Imp":"{:.0f}","R_LGB":"{:.0f}","R_Perm":"{:.0f}","R_Diff":"{:.0f}"}))

    brd=comp[comp["R_Diff"]>5]
    if len(brd)==0: R.decision("LGB and permutation importance AGREE. Ranking reliable.",True)
    elif len(brd)<=3: R.decision(f"Minor disagreements: {brd['Feature'].tolist()}. Trust permutation over built-in.",False)
    else: R.decision("MAJOR disagreements. Use permutation importance for all decisions.",False); actions.append("⚠️ Use permutation importance")

    neg_p=comp[comp["Perm_Mean"]<0]["Feature"].tolist()
    if neg_p: R.log(f"🔴 Negative perm importance: {neg_p}",cls="critical"); R.decision(f"Features HURT model: {neg_p}. Remove them.",False); actions.append(f"🗑️ Negative perm: {neg_p}")
    zero_p=comp[(comp["Perm_Mean"]>=0)&(comp["Perm_Mean"]<0.0001)]["Feature"].tolist()
    if zero_p: R.decision(f"Near-zero perm: {zero_p}. Try removing — if CV improves, keep removed.",False)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 20: SEGMENT ANALYSIS
    # ════════════════════════════════════════════════════
    R.section("🧪 20. HIGH-RISK SEGMENT PROFILING")
    tmp_s=train.copy(); tmp_s["tbin"]=safe_encode_target(tmp_s[target_col].copy())
    seg_cols=[c for c in ["Contract","InternetService","PaymentMethod"] if c in tmp_s.columns]
    if len(seg_cols)>=2:
        segs=[]
        for keys,grp in tmp_s.groupby(seg_cols):
            if len(grp)>=100:
                row=dict(zip(seg_cols,keys if isinstance(keys,tuple) else [keys]))
                row["Count"]=len(grp); row["Churn%"]=grp["tbin"].mean()*100; segs.append(row)
        if segs:
            df_seg=pd.DataFrame(segs).sort_values("Churn%",ascending=False).reset_index(drop=True)
            R.log("🔴 TOP 5 HIGHEST RISK:",cls="critical")
            R.table(df_seg.head(5).style.background_gradient(cmap="Reds",subset=["Churn%"]).format({"Churn%":"{:.2f}"}))
            R.log("🟢 TOP 5 SAFEST:",cls="ok")
            R.table(df_seg.tail(5).style.background_gradient(cmap="Greens",subset=["Churn%"]).format({"Churn%":"{:.2f}"}))
            spread=df_seg["Churn%"].max()-df_seg["Churn%"].min(); results['seg_spread']=spread
            if spread>40: R.decision(f"Spread={spread:.1f}%. HUGE. Create binary risk flags from top segments.",True); actions.append("🎯 Create risk flags from segments")
            elif spread>20: R.decision(f"Spread={spread:.1f}%. Moderate. Interaction features will help.",True); actions.append("🎯 Interaction features from segments")
            else: R.decision(f"Spread={spread:.1f}%. Low. Try other combinations.",False)

            fig,ax=plt.subplots(figsize=(12,max(4,len(df_seg)*0.4)))
            colors=["#f85149" if r>30 else "#f0883e" if r>15 else "#3fb950" for r in df_seg["Churn%"]]
            labels=[' | '.join(str(df_seg.iloc[i][c]) for c in seg_cols) for i in range(len(df_seg))]
            ax.barh(range(len(df_seg)),df_seg["Churn%"],color=colors,edgecolor=GRID)
            ax.set_yticks(range(len(df_seg))); ax.set_yticklabels(labels,fontsize=9)
            dark_style(ax,"Segments — Churn Rate"); ax.set_xlabel("Churn Rate (%)"); fig.tight_layout()
            R.plot(fig,"segment_analysis")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 21: FEATURE STABILITY
    # ════════════════════════════════════════════════════
    R.section("🧪 21. FEATURE STABILITY ACROSS FOLDS")
    fi_list=[]
    for fold,(tr_i,va_i) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X_all,y_full)):
        m=lgb.LGBMClassifier(n_estimators=500,verbose=-1,n_jobs=-1,is_unbalance=True,random_state=42)
        m.fit(X_all.iloc[tr_i],y_full.iloc[tr_i],eval_set=[(X_all.iloc[va_i],y_full.iloc[va_i])],callbacks=[lgb.early_stopping(50),lgb.log_evaluation(0)])
        fi_list.append(pd.DataFrame({"Feature":feats,f"F{fold}":m.feature_importances_}).set_index("Feature"))
    stab=pd.concat(fi_list,axis=1); stab["Mean"]=stab.mean(axis=1); stab["Std"]=stab.std(axis=1)
    stab["CV_coef"]=stab["Std"]/(stab["Mean"]+1e-10); stab=stab.sort_values("Mean",ascending=False)
    R.table(stab[["Mean","Std","CV_coef"]].style.background_gradient(cmap="RdYlGn_r",subset=["CV_coef"]).format(precision=2))

    v_unstable=stab[stab["CV_coef"]>1.0].index.tolist()
    m_unstable=stab[stab["CV_coef"]>0.5].index.tolist()
    if v_unstable: R.decision(f"Very unstable (CV>1.0): {v_unstable}. Consider removing.",False); actions.append(f"⚠️ Unstable features: {v_unstable}")
    elif m_unstable: R.decision(f"Moderately unstable (CV>0.5): {m_unstable}. Monitor.",False)
    else: R.decision("All features stable (CV<0.5). Reliable ranking.",True)
    results['stable_top']=stab.head(5).index.tolist()
    R.log(f"🏆 Core stable features: {results['stable_top']}")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 22: ORIGINAL DATA COMPARISON
    # ════════════════════════════════════════════════════
    R.section("🧪 22. ORIGINAL DATA COMPARISON")
    original=None
    if original_path and os.path.exists(original_path): original=pd.read_csv(original_path)
    else:
        for p in ['/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv','/kaggle/input/blastchar-telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv']:
            if os.path.exists(p): original=pd.read_csv(p); break
    if original is not None:
        R.log(f"📦 Original: {original.shape[0]:,} | Synthetic: {train.shape[0]:,} ({train.shape[0]//original.shape[0]}x)")
        for col in num_cols:
            if col in original.columns:
                ov=pd.to_numeric(original[col],errors='coerce').dropna(); sv=train[col].dropna()
                ks,p=ks_2samp(ov,sv); R.log(f"  {col}: KS={ks:.4f} | orig μ={ov.mean():.1f} → synth μ={sv.mean():.1f}")
        if target_col in original.columns:
            or_=((original[target_col]=='Yes').mean()*100); sr_=((train[target_col]=='Yes').mean()*100); rd_=abs(or_-sr_)
            R.log(f"  Target: orig={or_:.2f}% → synth={sr_:.2f}% (Δ={rd_:.2f}%)")
            if rd_>5: R.decision(f"Rate differs by {rd_:.1f}%. Reweight when augmenting.",False)
            else: R.decision(f"Rate similar (Δ={rd_:.1f}%). Safe to concat.",True)
        R.decision("Augment: pd.concat([train, original]) + source='original' flag.",True)
        actions.append("📦 Augment with original data")
    else:
        R.log("⚠️ Original data not found.",cls="warn")
        R.decision("Add 'blastchar/telco-customer-churn' as Kaggle input.",False)
        actions.append("📦 Add original Telco dataset")
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 23: MONOTONICITY CHECK
    # ════════════════════════════════════════════════════
    R.section("🧪 23. MONOTONICITY CHECK")
    tmp_m=train.copy(); tmp_m["tbin"]=safe_encode_target(tmp_m[target_col].copy())
    mono_res=[]
    for col in num_cols:
        try:
            tmp_m[f"{col}_d"]=pd.qcut(tmp_m[col],10,duplicates='drop',labels=False)
            dec=tmp_m.groupby(f"{col}_d")["tbin"].mean()*100; rates=dec.values
            rho,_=spearmanr(range(len(rates)),rates)
            is_inc=all(rates[i]<=rates[i+1]+0.5 for i in range(len(rates)-1))
            is_dec=all(rates[i]>=rates[i+1]-0.5 for i in range(len(rates)-1))
            if is_inc: d,cn="📈 Monotonic ↑",1
            elif is_dec: d,cn="📉 Monotonic ↓",-1
            else: d,cn="🔀 Non-monotonic",0
            mono_res.append({"Feature":col,"Direction":d,"Spearman":rho,"Constraint":cn})
            R.log(f"  {col}: {d} (ρ={rho:.3f})")
            R.log(f"    Deciles: {' → '.join(f'{r:.1f}%' for r in rates)}")
        except: pass

    if mono_res:
        R.table(pd.DataFrame(mono_res)[["Feature","Direction","Spearman","Constraint"]].style.format({"Spearman":"{:.3f}"}))
        constraints=[r["Constraint"] for r in mono_res]
        has_mono=any(c!=0 for c in constraints)
        if has_mono:
            R.decision(f"Apply monotone_constraints = {constraints} in LightGBM.",True)
            actions.append(f"📈 monotone_constraints = {constraints}")
        else:
            R.decision("No monotonic relationships. Consider binning.",True)
    R.end()

    # ════════════════════════════════════════════════════
    # SECTION 24: POST-CLEANING INTERACTIONS
    # ════════════════════════════════════════════════════
    R.section("🧪 24. POST-CLEANING INTERACTION ANALYSIS")
    tmp_i=train.copy(); inet_cols=[c for c in ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'] if c in tmp_i.columns]
    for c in inet_cols: tmp_i[c]=tmp_i[c].replace('No internet service','No')
    if 'MultipleLines' in tmp_i.columns: tmp_i['MultipleLines']=tmp_i['MultipleLines'].replace('No phone service','No')
    tmp_i["tbin"]=safe_encode_target(tmp_i[target_col].copy())

    if inet_cols:
        tmp_i['n_services']=sum((tmp_i[c]=='Yes').astype(int) for c in inet_cols)
        svc=tmp_i.groupby('n_services')['tbin'].agg(['mean','count']); svc['rate']=svc['mean']*100
        R.log("\n📊 Churn by number of services:")
        for n,row in svc.iterrows(): R.log(f"  {n} services: {row['rate']:5.1f}% ({row['count']:>6,} customers) {'█'*int(row['rate']/2)}")
        rho,_=spearmanr(svc.index,svc['rate'])
        if rho<-0.7: R.decision(f"Strong negative (ρ={rho:.2f}): more services = less churn. n_services is POWERFUL.",True); actions.append("🎯 Create n_services (strong)")
        elif rho<-0.3: R.decision(f"Moderate negative (ρ={rho:.2f}). n_services will help.",True); actions.append("🎯 Create n_services")
        elif rho>0.3: R.decision(f"Positive (ρ={rho:.2f}): unusual — investigate overcharging.",False)
        else: R.decision(f"Weak (ρ={rho:.2f}). Try n_security and n_streaming separately.",True)

    # Strong interactions
    if 'InternetService' in tmp_i.columns and inet_cols:
        overall=tmp_i['tbin'].mean()*100; strong=[]
        for col in inet_cols:
            for(inet,val),grp in tmp_i.groupby(['InternetService',col]):
                if len(grp)>=50:
                    r=grp['tbin'].mean()*100; dev=abs(r-overall)
                    if dev>15: strong.append({"Interaction":f"{inet}_{val}","Churn%":r,"Deviation":dev,"Count":len(grp),"Signal":"🔴 HIGH" if r>overall else "🟢 LOW"})
        if strong:
            df_st=pd.DataFrame(strong).sort_values("Deviation",ascending=False).reset_index(drop=True)
            R.log(f"\n⚡ {len(df_st)} strong interactions (>15% deviation):",cls="ok")
            R.table(df_st.style.background_gradient(cmap="RdYlGn_r",subset=["Churn%"]).format({"Churn%":"{:.2f}","Deviation":"{:.2f}"}))
            R.decision(f"Create binary flags for top interactions.",True)
            actions.append(f"🎯 Interaction flags: {df_st.head(3)['Interaction'].tolist()}")
    R.end()

    # ══════════════════════════════════════════════════════════
    # SECTION 25: FINAL ACTION SUMMARY
    # ══════════════════════════════════════════════════════════
    R.section("🎯 25. FINAL ACTION SUMMARY")
    R.log(f"📊 Baseline AUC: {results['baseline_mean']:.5f} ± {results['baseline_std']:.5f}")
    R.log(f"📊 CV Stability: std = {results.get('cv_std',0):.5f}")
    R.log(f"📊 Adversarial AUC: {results.get('adv_auc',0):.4f}")
    R.log(f"📊 Core features: {results.get('stable_top',[])}")

    score=0
    if results['baseline_mean']>0.85: score+=2
    elif results['baseline_mean']>0.80: score+=1
    if results.get('cv_std',1)<0.003: score+=2
    elif results.get('cv_std',1)<0.005: score+=1
    if results.get('adv_auc',1)<0.55: score+=2
    elif results.get('adv_auc',1)<0.65: score+=1

    if score>=5: R.log("🟢 READINESS: HIGH",cls="ok"); R.decision("Clean data, stable CV, no drift. Proceed to FE & modeling.",True)
    elif score>=3: R.log("🟡 READINESS: MODERATE",cls="warn"); R.decision("Fix warnings below before heavy modeling.",False)
    else: R.log("🔴 READINESS: LOW",cls="critical"); R.decision("Fix fundamental problems before modeling.",False)

    R.log("\n📋 ALL ACTIONS:")
    for i,a in enumerate(actions,1):
        cl="critical" if "🚨" in a else("warn" if "⚠" in a else "ok")
        R.log(f"  {i}. {a}",cls=cl)

    R.log("\n🔧 FE RECIPE:")
    R.log("  1️⃣  Sentinel → 'No' (No internet service, No phone service)")
    R.log("  2️⃣  Drop gender, PhoneService")
    R.log("  3️⃣  Create n_services, n_security, n_streaming")
    R.log("  4️⃣  charges_residual = TotalCharges - tenure × MonthlyCharges")
    R.log("  5️⃣  avg_monthly = TotalCharges / (tenure + 1)")
    if results.get('stable_top'): R.log(f"  6️⃣  Interactions between: {results['stable_top'][:4]}")
    if any('risk' in a.lower() for a in actions): R.log("  7️⃣  Binary risk flags from segments")
    if mono_res and any(r['Constraint']!=0 for r in mono_res):
        R.log(f"  8️⃣  monotone_constraints = {[r['Constraint'] for r in mono_res]}")

    R.log(f"\n🏁 TARGETS:")
    R.log(f"    Current: {results['baseline_mean']:.5f}")
    R.log(f"    Good FE: {results['baseline_mean']+0.005:.5f} - {results['baseline_mean']+0.015:.5f}")
    R.log(f"    If < {results['baseline_mean']:.5f} after FE → REVERT")
    R.end()

    R.save("/kaggle/working/pipeline_report.html","/kaggle/working/pipeline_report.md")
    return train, test, target_col, id_col, results, actions


# ═══════════════════════════
# RUN
# ═══════════════════════════
train, test, target_col, id_col, results, actions = full_senior_data_pipeline(
    "/kaggle/input/playground-series-s6e3",
    "train.csv", "test.csv", "Churn", "id",
    original_path="/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
```

---

## Her Section Ne Yapıyor — Detaylı Açıklama

```
╔═══════╦════════════════════════════╦════════════════════════════════════════════════╗
║  #    ║ Section                    ║ Ne yapıyor & Neden gerekli                     ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  0    ║ Overview                   ║ Task type (classification/regression)          ║
║       ║                            ║ otomatik algılanır. Feature sayısı raporlanır. ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  1    ║ Memory & Shape             ║ Veri büyüklüğünü anla. >1GB ise dtype         ║
║       ║                            ║ optimization gerekir.                          ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  2    ║ General Audit              ║ Her sütunun dtype, unique count, null oranı,   ║
║       ║                            ║ dominant value oranı. Train-test null gap.     ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  3    ║ Target Distribution        ║ Class balance kontrolü. Imbalance stratejisi   ║
║       ║                            ║ belirlenir (SMOTE vs class_weight).            ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  4    ║ Duplicates & Constants     ║ Duplicate satırlar ve quasi-constant sütunlar. ║
║       ║                            ║ >99.5% aynı değer = sil.                      ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  5    ║ Statistical Risk & Drift   ║ KS testi + VIF + MI Score. Drift yorumu        ║
║       ║                            ║ KS+N birlikte değerlendirilir.                 ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  6    ║ Numeric Correlation        ║ Pearson |r|>0.85 çiftler ve target korelasyon.║
║       ║                            ║ Yüksek korelasyon = drop one veya PCA.        ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  7    ║ Cramér's V                 ║ Kategorik↔kategorik ilişki. V>0.8=redundant,  ║
║       ║                            ║ V>0.5=high. Target ile V sıralaması.           ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  8    ║ Hierarchical Dependency    ║ "No internet service" gibi sentinel değerler.  ║
║       ║                            ║ Aynı count + aynı parent = bağımlı sütun.     ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  9    ║ Derived Features           ║ A×B≈C veya A/B≈C ilişkileri. |r|>0.90        ║
║       ║                            ║ ise C türetilmiş → residual oluştur.           ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  10   ║ Categorical Overlap        ║ Test'te train'de olmayan kategori var mı?      ║
║       ║                            ║ <100% = unknown handling gerekli.              ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  11   ║ Target Bivariate           ║ Her kategorik değer için target rate.           ║
║       ║                            ║ Hangi kategori en riskli/güvenli?              ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  12   ║ Information Value (IV)     ║ Her kategorik feature'ın prediktif gücü.       ║
║       ║                            ║ IV>0.5 suspicious AMA adversarial ile doğrula. ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  13   ║ Adversarial & Importance   ║ RF-based EDA: drift tespiti + importance.      ║
║       ║                            ║ Cumulative importance ile top-N belirlenir.    ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  14   ║ Missing Patterns           ║ Null'larda target rate farkı var mı?           ║
║       ║                            ║ Gap>5% = is_null flag oluştur.                 ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  15   ║ Playground Series Note     ║ Original data augmentation hatırlatması.       ║
║       ║                            ║                                                ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  16   ║ 🧪 BASELINE MODEL         ║ Sıfır FE ile LightGBM 5-fold CV.              ║
║       ║                            ║ Karşılaştırma noktası. Bu olmadan FE'nin       ║
║       ║                            ║ işe yarayıp yaramadığını bilemezsin.           ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  17   ║ 🧪 CV STABILITY           ║ 5 farklı seed ile CV. std<0.003 = güvenilir.  ║
║       ║                            ║ std>0.005 = repeated CV gerekli.              ║
║       ║                            ║ Outlier fold = data segmenti farklı.           ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  18   ║ 🧪 PER-FEATURE ADV.       ║ Hangi feature train/test'i en çok ayırıyor?   ║
║       ║                            ║ Genel AUC düşük olsa bile tek feature          ║
║       ║                            ║ drifted olabilir.                              ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  19   ║ 🧪 PERMUTATION IMP.       ║ Feature'ı shuffle et → skor ne kadar düşer?   ║
║       ║                            ║ LGB built-in importance yanıltıcı olabilir     ║
║       ║                            ║ (correlated features arasında dağıtır).        ║
║       ║                            ║ Negative perm = feature zararlı → SİL.        ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  20   ║ 🧪 SEGMENT ANALYSIS       ║ En tehlikeli/güvenli müşteri profilleri.       ║
║       ║                            ║ Spread>40% = binary risk flag oluştur.         ║
║       ║                            ║ FE fikirleri buradan çıkar.                    ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  21   ║ 🧪 FEATURE STABILITY      ║ Importance fold'lar arası tutarlı mı?          ║
║       ║                            ║ CV_coef>1.0 = very unstable → silmeyi düşün.  ║
║       ║                            ║ Top stable features = core predictors.         ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  22   ║ 🧪 ORIGINAL DATA          ║ IBM Telco orijinal veri ile karşılaştır.       ║
║       ║                            ║ Target rate farkı <5% = safe to concat.        ║
║       ║                            ║ >5% = reweight when augmenting.                ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  23   ║ 🧪 MONOTONICITY           ║ tenure↑ → churn↓ gibi ilişkiler monoton mu?   ║
║       ║                            ║ Monoton ise → LGB monotone_constraints ekle.  ║
║       ║                            ║ Overfitting'i önler, generalization artar.     ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  24   ║ 🧪 INTERACTIONS           ║ Sentinel temizleme SONRASI etkileşimler.       ║
║       ║                            ║ Fiber+NoSecurity gibi kombinasyonlar.          ║
║       ║                            ║ n_services count: ρ<-0.7 = powerful feature.  ║
╠═══════╬════════════════════════════╬════════════════════════════════════════════════╣
║  25   ║ FINAL SUMMARY             ║ Readiness score (0-6 puan).                    ║
║       ║                            ║ 5-6=HIGH, 3-4=MODERATE, 0-2=LOW.              ║
║       ║                            ║ Tüm aksiyonlar + FE recipe + hedef skor.      ║
╚═══════╩════════════════════════════╩════════════════════════════════════════════════╝
```

## Her Decision'ın If-Else Mantığı

```
TEST 16 — BASELINE:
  IF   AUC > 0.90  → Ensemble & tuning odaklı ilerle
  ELIF AUC > 0.85  → FE + tuning yeterli (+0.005-0.015 beklenen)
  ELIF AUC > 0.80  → Agresif FE + encoding + augmentation
  ELSE             → Yaklaşımı değiştir (stacking, target encoding)

TEST 17 — CV STABILITY:
  IF   std < 0.001  → Çok stabil, 5-fold yeterli
  ELIF std < 0.003  → Stabil, sabit seed kullan
  ELIF std < 0.005  → RepeatedStratifiedKFold(3×5) kullan
  ELSE              → 10-fold veya 5×5 repeated, leakage kontrol

TEST 18 — ADVERSARIAL:
  IF   AUC < 0.52  → Drift yok, güvenli
  ELIF AUC < 0.60  → Minimal, top feature'ları izle
  ELIF AUC < 0.70  → Moderate, drifted feature'ları sil/normalize et
  ELSE              → SEVERE, DUR ve leakage araştır

TEST 19 — PERMUTATION:
  IF   rank_diff hep < 5   → İki metod uyumlu, güvenilir
  ELIF rank_diff 1-3 tane  → Küçük fark, perm'e güven
  ELSE                      → Büyük fark, sadece perm kullan
  +IF  perm < 0            → Feature zararlı, SİL
  +IF  perm ≈ 0            → Deneysel sil, CV ile doğrula

TEST 20 — SEGMENTS:
  IF   spread > 40%  → Risk flag oluştur
  ELIF spread > 20%  → Interaction feature oluştur
  ELSE                → Başka kombinasyonlar dene

TEST 21 — STABILITY:
  IF   all CV_coef < 0.5   → Hepsi stabil, güvenilir
  ELIF some CV_coef > 0.5  → Moderate, izle
  ELIF some CV_coef > 1.0  → Çok unstable, silmeyi düşün

TEST 22 — ORIGINAL DATA:
  IF   found + rate_diff < 5%   → Direkt concat
  IF   found + rate_diff > 5%   → Reweight ile concat
  IF   not found                → Kaggle input olarak ekle

TEST 23 — MONOTONICITY:
  IF   all monotonic    → Tüm features'a constraint uygula
  ELIF mixed            → Sadece monotonic olanlara constraint
  ELSE                  → Constraint koyma, binning dene

TEST 24 — INTERACTIONS:
  IF   ρ < -0.7  → n_services çok güçlü, oluştur
  ELIF ρ < -0.3  → n_services yardımcı, oluştur
  ELIF ρ > 0.3   → Anormal, araştır
  ELSE            → n_security / n_streaming ayrı dene
  +IF  deviation > 15%  → Binary interaction flag oluştur

TEST 25 — READINESS:
  IF   score 5-6  → 🟢 HIGH — FE & modellemeye geç
  ELIF score 3-4  → 🟡 MODERATE — önce uyarıları çöz
  ELSE             → 🔴 LOW — temelden düzelt
```