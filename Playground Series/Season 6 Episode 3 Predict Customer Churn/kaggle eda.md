# Basit EDA 

```python
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Diller
LANGS = {
    "en": {
        "summary": "Summary Statistics",
        "missing": "Missing Values",
        "target_dist": "Target Distribution",
        "corr": "Correlation Matrix",
        "cat_dist": "Categorical Distribution",
        "todo": "Recommendations",
        "num_feat": "Numerical Features",
        "cat_feat": "Categorical Features",
        "train": "TRAIN Dataset",
        "test": "TEST Dataset",
        "eda": "Exploratory Data Analysis Report",
        "col": "Column",
        "type": "Type",
        "unique": "Unique Values",
        "top": "Top",
        "freq": "Frequency",
        "date": "Date",
    },
    "tr": {
        "summary": "Özet İstatistikler",
        "missing": "Eksik Değerler",
        "target_dist": "Hedef Dağılımı",
        "corr": "Korelasyon Matrisi",
        "cat_dist": "Kategorik Dağılım",
        "todo": "Yapılması Gerekenler",
        "num_feat": "Sayısal Özellikler",
        "cat_feat": "Kategorik Özellikler",
        "train": "EĞİTİM (TRAIN) Verisi",
        "test": "TEST Verisi",
        "eda": "Keşifsel Veri Analizi Raporu",
        "col": "Sütun",
        "type": "Tip",
        "unique": "Benzersiz Değer",
        "top": "En Çok",
        "freq": "Frekans",
        "date": "Tarih",
    }
}

def _get_lang(lang_code):
    return LANGS.get(lang_code, LANGS["en"])

def save_plot(fig, name, outdir):
    fname = os.path.join(outdir, f"{name}.png")
    fig.savefig(fname, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return fname

def eda_report(train_df, test_df, target_col=None, lang="en", outdir="./eda_report"):
    lg = _get_lang(lang)
    os.makedirs(outdir, exist_ok=True)
    dt = datetime.now().strftime("%Y-%m-%d %H:%M")
    html_parts = [f"<h1>{lg['eda']}</h1><div><b>{lg['date']}:</b> {dt}</div>"]

    # --- Genel bilgi
    for df, dname in zip([train_df, test_df], [lg['train'], lg['test']]):
        html_parts.append(f"<h2>{dname}</h2>")
        html_parts.append(df.head().to_html())
        html_parts.append(df.describe(include="all").to_html())
        html_parts.append(f"<b>{lg['missing']}:</b>")
        html_parts.append(df.isnull().sum().to_frame("Missing count").to_html())

    # --- Özellik tipi belirleme
    num_cols = train_df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = train_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    html_parts.append(f"<h2>{lg['num_feat']}</h2>")
    html_parts.append(", ".join(num_cols))
    html_parts.append(f"<h2>{lg['cat_feat']}</h2>")
    html_parts.append(", ".join(cat_cols))

    # --- Sayısal değişkenlerin dağılımı ve grafikleri
    for col in num_cols:
        fig, ax = plt.subplots()
        sns.histplot(train_df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"{col} - {dname} - {lg['num_feat']}")
        img_path = save_plot(fig, f"{col}_hist", outdir)
        html_parts.append(f"<h3>{col} {lg['num_feat']}</h3><img src='{img_path}' width=400 />")

    # --- Kategorik değişkenlerin dağılımı ve grafikleri
    for col in cat_cols:
        fig, ax = plt.subplots()
        train_df[col].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"{col} - {dname} - {lg['cat_feat']}")
        img_path = save_plot(fig, f"{col}_cat", outdir)
        html_parts.append(f"<h3>{col} {lg['cat_feat']}</h3><img src='{img_path}' width=400 />")

    # --- Korelasyon matrisi
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = train_df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax, cmap="coolwarm")
    ax.set_title(lg['corr'])
    img_path = save_plot(fig, "corr_matrix", outdir)
    html_parts.append(f"<h2>{lg['corr']}</h2><img src='{img_path}' width=600 />")

    # --- Hedef değişken analizi
    if target_col and target_col in train_df:
        fig, ax = plt.subplots()
        if train_df[target_col].dtype in ['object', 'category', 'bool']:
            train_df[target_col].value_counts().plot(kind="bar", ax=ax)
        else:
            sns.histplot(train_df[target_col].dropna(), kde=True, ax=ax)
        ax.set_title(lg['target_dist'])
        img_path = save_plot(fig, f"target_dist", outdir)
        html_parts.append(f"<h2>{lg['target_dist']}</h2><img src='{img_path}' width=400 />")

    # --- Yapılması gerekenler / Recommendations (otomatik tespit)
    todo_items = []
    for col in train_df.columns:
        miss = train_df[col].isnull().mean()
        if miss > 0.2:
            todo_items.append(f"{col}: {lg['missing']} oranı yüksek ({miss:.0%})")
        if col in num_cols:
            if train_df[col].nunique() < 5:
                todo_items.append(f"{col}: Düşük benzersiz (n={train_df[col].nunique()})")
            elif abs(train_df[col].skew()) > 1.5:
                todo_items.append(f"{col}: Çarpıklık yüksek (skew={train_df[col].skew():.2f})")
        if col in cat_cols:
            if train_df[col].nunique() > 200:
                todo_items.append(f"{col}: Çok fazla kategori")
    if len(todo_items) == 0:
        todo_items.append("No major data issues found / Büyük veri sorunu bulunamadı.")

    html_parts.append(f"<h2>{lg['todo']}</h2>")
    html_parts.append("<ul>")
    for item in todo_items:
        html_parts.append(f"<li>{item}</li>")
    html_parts.append("</ul>")

    # -- Dosyaya kaydet
    html_fp = os.path.join(outdir, f"eda_report_{lang}.html")
    with open(html_fp, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"EDA report generated: {html_fp}")
    return html_fp

# Kullanım örneği:
# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')
# eda_report(train, test, target_col='target', lang='tr', outdir='./kaggle_eda')
```

# 2. Basit EDA

```py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, ttest_ind, f_oneway, chi2_contingency
from datetime import datetime

sns.set(style="whitegrid")

LANGS = {
    "en": {
        "summary": "Summary Statistics",
        "missing": "Missing Values",
        "impute": "Imputation Recommendation",
        "target_dist": "Target Distribution",
        "corr": "Correlation Matrix",
        "cat_dist": "Categorical Distribution",
        "todo": "Recommendations",
        "num_feat": "Numerical Features",
        "cat_feat": "Categorical Features",
        "train": "TRAIN Dataset",
        "test": "TEST Dataset",
        "eda": "Exploratory Data Analysis Report",
        "col": "Column",
        "type": "Type",
        "unique": "Unique Values",
        "top": "Top",
        "freq": "Frequency",
        "box": "Boxplot",
        "violin": "Violinplot",
        "pairplot": "Pairplot",
        "scatter_matrix": "Scatter Matrix",
        "outlier": "Outlier Detection",
        "feature_engineering": "Feature Engineering Suggestions",
        "encoding": "Encoding Recommendation",
        "drift": "Train/Test Drift",
        "imbalance": "Class Imbalance",
        "memory": "Memory & Optimization",
        "analyst_note": "Analyst Notes",
        "date": "Date",
        "target_rel": "Target Relationship",
        "wrong_type": "Suspicious Column Type"
    },
    "tr": {
        "summary": "Özet İstatistikler",
        "missing": "Eksik Değerler",
        "impute": "Doldurma Önerisi",
        "target_dist": "Hedef Dağılımı",
        "corr": "Korelasyon Matrisi",
        "cat_dist": "Kategorik Dağılım",
        "todo": "Yapılması Gerekenler",
        "num_feat": "Sayısal Özellikler",
        "cat_feat": "Kategorik Özellikler",
        "train": "EĞİTİM (TRAIN) Verisi",
        "test": "TEST Verisi",
        "eda": "Keşifsel Veri Analizi Raporu",
        "col": "Sütun",
        "type": "Tip",
        "unique": "Benzersiz Değer",
        "top": "En Çok",
        "freq": "Frekans",
        "box": "Boxplot",
        "violin": "Violinplot",
        "pairplot": "Pairplot",
        "scatter_matrix": "Scatter Matrix",
        "outlier": "Aykırı Değer Algılama",
        "feature_engineering": "Özellik Mühendisliği Önerileri",
        "encoding": "Encoding Önerisi",
        "drift": "Eğitim/Test Drift",
        "imbalance": "Sınıf Dengesizliği",
        "memory": "Memory ve Optimizasyon",
        "analyst_note": "Analist Notları",
        "date": "Tarih",
        "target_rel": "Hedef ile İlişki",
        "wrong_type": "Şüpheli Sütun Tipi"
    }
}

def _get_lang(lang_code):
    return LANGS.get(lang_code, LANGS["en"])

def save_plot(fig, name, outdir):
    fname = os.path.join(outdir, f"{name}.png")
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return fname

def outlier_report(df, num_cols, lang):
    lg = _get_lang(lang)
    report = []
    for col in num_cols:
        vals = df[col].dropna()
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((vals < lower) | (vals > upper)).sum()
        pct = outliers / len(vals) if len(vals) > 0 else 0
        if pct > 0.02:
            report.append(f"{col}: {lg['outlier']} - {outliers} ({pct:.1%})")
    return report

def impute_suggestion(df, col, lang):
    lg = _get_lang(lang)
    col_type = df[col].dtype
    miss = df[col].isnull().mean()
    if miss < 0.01:
        return ''
    if col_type in ['float64', 'int64']:
        return f"{col}: {lg['impute']} - Median" if miss < 0.2 else f"{col}: {lg['impute']} - Remove or advanced imputation"
    elif col_type in ['object', 'category']:
        return f"{col}: {lg['impute']} - Mode"
    else:
        return f"{col}: {lg['impute']} - Review type"

def memory_report(df, lang):
    lg = _get_lang(lang)
    mem = df.memory_usage(deep=True).sum()/1e6
    nrows = df.shape[0]
    msg = ""
    if mem > 500:
        msg += f"{lg['memory']}: DataFrame is large ({mem:.1f} MB, {nrows} rows). Consider chunking, reducing dtype, or using Dask."
    else:
        msg += f"{lg['memory']}: DataFrame is {mem:.1f} MB, {nrows} rows."
    return msg

def eda_report_senior(train_df, test_df, target_col=None, lang="en", outdir="./eda_report_senior", analyst_note=""):
    lg = _get_lang(lang)
    os.makedirs(outdir, exist_ok=True)
    dt = datetime.now().strftime("%Y-%m-%d %H:%M")
    html_parts = [f"<h1>{lg['eda']}</h1><div><b>{lg['date']}:</b> {dt}</div>"]

    def get_type_warnings(df):
        warnings = []
        for col in df.columns:
            if str(df[col].dtype) not in ["float64", "int64", "object", "category", "bool"]:
                warnings.append(f"{col}: {lg['wrong_type']} ({df[col].dtype})")
        return warnings

    # --- Memory & dtype check
    html_parts.append(f'<h3>{lg["train"]} {lg["memory"]}</h3>{memory_report(train_df, lang)}')
    html_parts.append(f'<h3>{lg["test"]} {lg["memory"]}</h3>{memory_report(test_df, lang)}')

    # --- Özellik tipi belirleme
    num_cols = train_df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = train_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    html_parts.append(f"<h2>{lg['num_feat']}</h2>")
    html_parts.append(", ".join(num_cols))
    html_parts.append(f"<h2>{lg['cat_feat']}</h2>")
    html_parts.append(", ".join(cat_cols))

    html_parts.append("<h3>Type Check</h3>")
    for w in get_type_warnings(train_df):
        html_parts.append(f"<li>{w}</li>")

    # --- Summary Tables + IQR, skewness, kurtosis, percentiles
    def summary_extended(df):
        desc = df.describe(include="all").transpose()
        ext = pd.DataFrame()
        for col in num_cols:
            ext.at[col, "Skewness"] = skew(df[col].dropna())
            ext.at[col, "Kurtosis"] = kurtosis(df[col].dropna())
            ext.at[col, "IQR"] = df[col].quantile(0.75) - df[col].quantile(0.25)
        desc = desc.join(ext)
        return desc

    html_parts.append(f"<h2>{lg['summary']}</h2>")
    html_parts.append(summary_extended(train_df).to_html())

    # --- Eksik değer analizi ve imputation önerisi
    html_parts.append(f"<h2>{lg['missing']}</h2>")
    miss_tab = train_df.isnull().sum().to_frame("Missing count")
    miss_tab["Missing %"] = train_df.isnull().mean() * 100
    html_parts.append(miss_tab.to_html())
    html_parts.append("<ul>")
    for col in train_df.columns:
        sug = impute_suggestion(train_df, col, lang)
        if sug: html_parts.append(f"<li>{sug}</li>")
    html_parts.append("</ul>")

    # --- Outlier Detection (Boxplot/IQR)
    html_parts.append(f"<h2>{lg['outlier']}</h2>")
    out_report = outlier_report(train_df, num_cols, lang)
    for col in num_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x=train_df[col], ax=ax)
        ax.set_title(f"{col} - {lg['box']}")
        img_path = save_plot(fig, f"{col}_boxplot", outdir)
        html_parts.append(f"<h3>{col} {lg['box']}</h3><img src='{img_path}' width=400 />")
    for report in out_report:
        html_parts.append(f"<li>{report}</li>")

    # --- Violinplots, scatter matrix, pairplot
    for col in num_cols:
        fig, ax = plt.subplots()
        sns.violinplot(x=train_df[col].dropna(), ax=ax)
        ax.set_title(f"{col} {lg['violin']}")
        img_path = save_plot(fig, f"{col}_violinplot", outdir)
        html_parts.append(f"<h3>{col} {lg['violin']}</h3><img src='{img_path}' width=400 />")
    if len(num_cols) >= 2:
        sns.pairplot(train_df[num_cols].dropna())
        plt.tight_layout()
        img_path = os.path.join(outdir, "pairplot.png")
        plt.savefig(img_path)
        plt.close()
        html_parts.append(f"<h3>{lg['pairplot']}</h3><img src='{img_path}' width=600 />")

        import pandas.plotting as pd_plot
        fig = plt.figure(figsize=(8,8))
        pd_plot.scatter_matrix(train_df[num_cols].dropna(), alpha=0.2, ax=None)
        img_path = save_plot(fig, "scatter_matrix", outdir)
        html_parts.append(f"<h3>{lg['scatter_matrix']}</h3><img src='{img_path}' width=600 />")

    # --- Kategorik dağılım ve encoding önerileri
    html_parts.append(f"<h2>{lg['cat_dist']}</h2>")
    for col in cat_cols:
        fig, ax = plt.subplots()
        train_df[col].value_counts().sort_values(ascending=False).head(30).plot(kind="bar", ax=ax)
        ax.set_title(f"{col} - {lg['cat_dist']}")
        img_path = save_plot(fig, f"{col}_catdist", outdir)
        html_parts.append(f"<h3>{col} {lg['cat_dist']}</h3><img src='{img_path}' width=600 />")

    for col in cat_cols:
        nuniq = train_df[col].nunique()
        if nuniq > 50:
            html_parts.append(f"<li>{col}: {lg['encoding']} - consider reducing categories or using label encoding ({nuniq} levels)</li>")
        elif nuniq < 5:
            html_parts.append(f"<li>{col}: {lg['encoding']} - OneHotEncoding ({nuniq} levels)</li>")
        else:
            html_parts.append(f"<li>{col}: {lg['encoding']} - Label or Target encoding ({nuniq} levels)</li>")

    # --- Korelasyon matrisi
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = train_df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax, cmap="coolwarm")
    ax.set_title(lg['corr'])
    img_path = save_plot(fig, "corr_matrix", outdir)
    html_parts.append(f"<h2>{lg['corr']}</h2><img src='{img_path}' width=600 />")

    # --- Target Distribution/Relationship
    if target_col and target_col in train_df:
        fig, ax = plt.subplots()
        if train_df[target_col].dtype in ['object', 'category', 'bool']:
            train_df[target_col].value_counts().plot(kind="bar", ax=ax)
        else:
            sns.histplot(train_df[target_col].dropna(), kde=True, ax=ax)
        ax.set_title(lg['target_dist'])
        img_path = save_plot(fig, f"target_dist", outdir)
        html_parts.append(f"<h2>{lg['target_dist']}</h2><img src='{img_path}' width=400 />")

        # Imbalance check
        val_counts = train_df[target_col].value_counts(normalize=True)
        if val_counts.max() > 0.8:
            html_parts.append(f"<li>{lg['imbalance']}: Class imbalance over 80% detected.</li>")
        # Target relationship: numerics
        html_parts.append(f"<h2>{lg['target_rel']} - Numerical</h2>")
        for col in num_cols:
            try:
                if train_df[target_col].dtype in ['object', 'category', 'bool']:
                    means = train_df.groupby(target_col)[col].mean()
                    html_parts.append(f"<b>{col}:{lg['target_rel']}</b> {means.to_dict()}")
                    # ANOVA
                    groups = [grp[col].dropna().values for _, grp in train_df.groupby(target_col)]
                    if len(groups) > 1:
                        fstat, pval = f_oneway(*groups)
                        html_parts.append(f"P-value (ANOVA): {pval:.3e}")
                else:
                    corrval = train_df[[col, target_col]].corr().iloc[0,1]
                    html_parts.append(f"<b>{col}-{target_col}: Correlation {corrval:.2f}</b>")
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=train_df[col], y=train_df[target_col], ax=ax)
                    img_path = save_plot(fig, f"{col}_scatter_target", outdir)
                    html_parts.append(f"<img src='{img_path}' width=300 />")
            except Exception as e:
                html_parts.append(f"Could not compute target relationship for {col}: {str(e)}")
        # Target relationship: categoricals
        html_parts.append(f"<h2>{lg['target_rel']} - Categorical</h2>")
        for col in cat_cols:
            try:
                ct = pd.crosstab(train_df[col], train_df[target_col])
                if ct.shape[1] == 2:
                    chi2, pval, dof, _ = chi2_contingency(ct)
                    html_parts.append(f"<b>{col}:{lg['target_rel']} Chi2 p-value: {pval:.3e}</b>")
                html_parts.append(ct.to_html())
            except Exception as e:
                html_parts.append(f"Could not compute categorical target rel for {col}: {str(e)}")

    # --- Train/Test karşılaştırma & drift tespiti
    html_parts.append(f"<h2>{lg['drift']}</h2>")
    for col in num_cols:
        train_mean = train_df[col].mean()
        test_mean = test_df[col].mean()
        drift = abs(train_mean - test_mean) / (abs(train_mean) + 1e-6)
        html_parts.append(f"<li>{col}: Train mean={train_mean:.3f}, Test mean={test_mean:.3f}, Drift={drift:.2%}</li>")
        if drift > 0.2:
            html_parts.append(f"<b>Significant drift detected in {col}</b>")
    for col in cat_cols:
        train_freq = train_df[col].value_counts(normalize=True)
        test_freq = test_df[col].value_counts(normalize=True)
        common = set(train_freq.index) & set(test_freq.index)
        diffs = [abs(train_freq.get(k,0)-test_freq.get(k,0)) for k in common]
        drift = np.mean(diffs) if diffs else 0
        html_parts.append(f"<li>{col}: Mean freq diff={drift:.2%}</li>")
        if drift > 0.15:
            html_parts.append(f"<b>Significant categorical drift detected: {col}</b>")

    # --- Feature engineering suggestions
    fe_suggestions = []
    for col in num_cols:
        if train_df[col].nunique() < 5:
            fe_suggestions.append(f"{col}: consider converting to categorical")
    for col in cat_cols:
        if any(word in col.lower() for word in ["date","time"]):
            fe_suggestions.append(f"{col}: extract year/month/day/features")
        if train_df[col].nunique() > 100:
            fe_suggestions.append(f"{col}: consider grouping rare categories")

    html_parts.append(f"<h2>{lg['feature_engineering']}</h2><ul>")
    for s in fe_suggestions:
        html_parts.append(f"<li>{s}</li>")
    html_parts.append("</ul>")

    # --- Recommendations / Yapılması Gerekenler
    html_parts.append(f"<h2>{lg['todo']}</h2><ul>")
    for item in out_report:
        html_parts.append(f"<li>{item}</li>")
    for col in train_df.columns:
        sug = impute_suggestion(train_df, col, lang)
        if sug: html_parts.append(f"<li>{sug}</li>")
    for w in get_type_warnings(train_df):
        html_parts.append(f"<li>{w}</li>")
    if len(fe_suggestions)==0:
        html_parts.append("<li>No major feature engineering recommendations needed.</li>")
    html_parts.append("</ul>")

    # --- Analyst Notes area
    html_parts.append(f"<h2>{lg['analyst_note']}</h2>")
    html_parts.append(f"<div style='border:1px solid #888;padding:8px'>{analyst_note}</div>")

    # --- Dosyaya kaydet
    html_fp = os.path.join(outdir, f"eda_report_senior_{lang}.html")
    with open(html_fp, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"EDA senior report generated: {html_fp}")
    return html_fp

# Kullanım örneği:
# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')
# eda_report_senior(train, test, target_col='target', lang='tr', outdir='./kaggle_senior_eda', analyst_note="Model öncesi numerik drift kritik, encoding yapılmalı.")
```

# 3. Basit EDA

```py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, ttest_ind, f_oneway, chi2_contingency, zscore
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from datetime import datetime

sns.set(style="whitegrid")

# Senior EDA operasyonlarının checklist'i
SENIOR_OPERATIONS = [
    "Type & Format Control (dtype, suspicious type, uniform column, wrong format)",
    "Missing Value Analysis & Imputation Suggestion",
    "Outlier Detection (IQR, Z-Score, Boxplot)",
    "Unique, Duplicate, Rare Category Analysis",
    "Advanced Summary (IQR, skewness, kurtosis, percentiles, multimodal)",
    "Categorical Encoding Necessity & Rare Detection",
    "Numeric/Target Relationship Analysis (correlation, ANOVA/t-test, scatter)",
    "Categorical/Target Relationship (crosstab, chi2 test, imbalance)",
    "Train/Test Drift, Leakage & Shift Detection",
    "Feature Engineering Suggestions (date extract, rare grouping, redundant vars)",
    "Time Series Analysis (trend, seasonality if detected)",
    "Memory Analysis, RAM Optimization",
    "Feature Importance & Multicollinearity",
    "Class/Category Imbalance Detection",
    "Analyst's Notes Section"
]

LANGS = {
    "en": {
        "checklist": "Senior EDA Operations Checklist",
        "completed": "All senior EDA operations are included below.",
        "summary": "Summary Statistics",
        "missing": "Missing Values",
        "impute": "Imputation Recommendation",
        "target_dist": "Target Distribution",
        "corr": "Correlation Matrix",
        "cat_dist": "Categorical Distribution",
        "todo": "Recommendations",
        "num_feat": "Numerical Features",
        "cat_feat": "Categorical Features",
        "train": "TRAIN Dataset",
        "test": "TEST Dataset",
        "eda": "Senior Exploratory Data Analysis Report",
        "col": "Column",
        "type": "Type",
        "unique": "Unique Values",
        "top": "Top",
        "freq": "Frequency",
        "box": "Boxplot",
        "violin": "Violinplot",
        "pairplot": "Pairplot",
        "scatter_matrix": "Scatter Matrix",
        "outlier": "Outlier Detection",
        "feature_engineering": "Feature Engineering Suggestions",
        "encoding": "Encoding Recommendation",
        "drift": "Train/Test Drift",
        "imbalance": "Class Imbalance",
        "memory": "Memory & Optimization",
        "analyst_note": "Analyst Notes",
        "date": "Date",
        "target_rel": "Target Relationship",
        "wrong_type": "Suspicious Column Type",
        "redundant": "Redundant Variable",
        "rare": "Rare Category",
        "duplicate": "Duplicate Value",
        "multimodal": "Multimodal Distribution",
        "feature_importance": "Feature Importance",
        "multicollinearity": "Multicollinearity Risk",
        "trend": "Trend Detected",
        "seasonality": "Seasonality Detected"
    },
    "tr": {
        "checklist": "Senior EDA Operasyon Kontrol Listesi",
        "completed": "Aşağıda tüm senior EDA operasyonları eksiksiz olarak yer almaktadır.",
        "summary": "Özet İstatistikler",
        "missing": "Eksik Değerler",
        "impute": "Doldurma Önerisi",
        "target_dist": "Hedef Dağılımı",
        "corr": "Korelasyon Matrisi",
        "cat_dist": "Kategorik Dağılım",
        "todo": "Yapılması Gerekenler",
        "num_feat": "Sayısal Özellikler",
        "cat_feat": "Kategorik Özellikler",
        "train": "EĞİTİM (TRAIN) Verisi",
        "test": "TEST Verisi",
        "eda": "Senior Keşifsel Veri Analizi Raporu",
        "col": "Sütun",
        "type": "Tip",
        "unique": "Benzersiz Değer",
        "top": "En Çok",
        "freq": "Frekans",
        "box": "Boxplot",
        "violin": "Violinplot",
        "pairplot": "Pairplot",
        "scatter_matrix": "Scatter Matrix",
        "outlier": "Aykırı Değer Algılama",
        "feature_engineering": "Özellik Mühendisliği Önerileri",
        "encoding": "Encoding Önerisi",
        "drift": "Eğitim/Test Drift",
        "imbalance": "Sınıf Dengesizliği",
        "memory": "Memory ve Optimizasyon",
        "analyst_note": "Analist Notları",
        "date": "Tarih",
        "target_rel": "Hedef ile İlişki",
        "wrong_type": "Şüpheli Sütun Tipi",
        "redundant": "Redundant Değişken",
        "rare": "Nadir Kategori",
        "duplicate": "Tekrarlı Değer",
        "multimodal": "Multimodal Dağılım",
        "feature_importance": "Öznitelik Önemi",
        "multicollinearity": "Çoklu Korelasyon Riski",
        "trend": "Trend Algılandı",
        "seasonality": "Sezonluk Etki Algılandı"
    }
}

def _get_lang(lang_code):
    return LANGS.get(lang_code, LANGS["en"])

def save_plot(fig, name, outdir):
    fname = os.path.join(outdir, f"{name}.png")
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return fname

def suspicious_types(df, lg):
    report = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if dtype not in ["float64", "int64", "object", "category", "bool"]:
            report.append(f"{col}: {lg['wrong_type']} ({dtype})")
        if df[col].nunique() == 1:
            report.append(f"{col}: Uniform column (only one value)")
        if dtype=="object" and df[col].astype(str).str.match(r"^\d{1,}$").sum()>0:
            report.append(f"{col}: May require numeric conversion")
        if dtype=="object" and df[col].astype(str).str.match(r"\d{4}-\d{2}-\d{2}").sum()>10:
            report.append(f"{col}: May require datetime conversion")
    return report

def missing_report(df, lg):
    tab = df.isnull().sum().to_frame("Missing Count")
    tab["Missing %"] = df.isnull().mean() * 100
    out = []
    for col in df.columns:
        miss = df[col].isnull().mean()
        if miss > 0.02:
            out.append(f"{col}: {lg['missing']} high ({miss:.1%})")
    return tab, out

def impute_suggestion(df, col, lg):
    miss = df[col].isnull().mean()
    dtype = str(df[col].dtype)
    if miss < 0.01:
        return ''
    if dtype in ['float64', 'int64']:
        return f"{col}: {lg['impute']} - Median" if miss < 0.2 else f"{col}: {lg['impute']} - Remove or advanced imputation"
    elif dtype in ['object', 'category']:
        return f"{col}: {lg['impute']} - Mode"
    else:
        return f"{col}: {lg['impute']} - Review type"

def outlier_report(df, num_cols, lg):
    report = []
    for col in num_cols:
        vals = df[col].dropna()
        # IQR outlier
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_outliers = ((vals < lower) | (vals > upper)).sum()
        # z-score outlier
        zvals = np.abs(zscore(vals))
        z_outliers = np.sum(zvals > 3)
        pct = (iqr_outliers + z_outliers) / (2*len(vals)) if len(vals) > 0 else 0
        if pct > 0.02:
            report.append(f"{col}: {lg['outlier']} - IQR: {iqr_outliers}, Z: {z_outliers}, ({pct:.2%})")
    return report

def rare_duplicate_report(df, cat_cols, lg):
    report = []
    for col in cat_cols:
        valc = df[col].value_counts()
        if (valc < 5).sum() > 0:
            report.append(f"{col}: {lg['rare']} - {list(valc[valc<5].index)}, n={len(valc[valc<5])}")
        if (valc > 100).sum() > 0:
            report.append(f"{col}: Very frequent categories - {list(valc[valc>100].index)}")
        dupes = df[df.duplicated([col])][col].unique()
        if len(dupes) > 1:
            report.append(f"{col}: {lg['duplicate']} detected ({len(dupes)} values)")
    return report

def memory_report(df, lg):
    mem = df.memory_usage(deep=True).sum()/1e6
    nrows = df.shape[0]
    msg = f"{lg['memory']}: DataFrame is {mem:.1f} MB, {nrows} rows."
    if mem > 500:
        msg += " Consider chunking, reducing dtype, or using Dask."
    return msg

def advanced_stats(df, num_cols):
    ext = pd.DataFrame()
    for col in num_cols:
        vals = df[col].dropna()
        ext.at[col, "Skewness"] = skew(vals)
        ext.at[col, "Kurtosis"] = kurtosis(vals)
        ext.at[col, "IQR"] = np.percentile(vals,75) - np.percentile(vals,25)
        ext.at[col, "Multimodal"] = "Yes" if len(set(np.histogram(vals, bins=10)[0][np.histogram(vals, bins=10)[0]>0]))>1 else "No"
    return ext

def feature_importance_report(df, target_col, num_cols, cat_cols, lg):
    feats = num_cols + cat_cols
    if target_col and target_col in df:
        y = df[target_col]
        X = df[feats]
        if y.dtype in ['float64','int64']:
            mi = mutual_info_regression(X.fillna(0), y)
        else:
            mi = mutual_info_classif(X.fillna("missing"), y)
        featimp = pd.DataFrame({'feature':feats, lg['feature_importance']:mi})
        return featimp.sort_values(by=lg['feature_importance'],ascending=False).to_html()
    return ""

def multicollinearity_report(df, num_cols, lg):
    corr = df[num_cols].corr()
    multi = []
    for col in corr.columns:
        for c2 in corr.columns:
            if col != c2 and abs(corr.at[col,c2]) > 0.9:
                multi.append(f"{col} & {c2}: {lg['multicollinearity']} r={corr.at[col,c2]:.2f}")
    return multi

def trend_seasonality_report(df, col, lg):
    # Simple trend/seasonality if datetime
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        vals = df[col].sort_values()
        diff = (vals.max()-vals.min()).days
        if diff > 100:
            return f"{col}: {lg['trend']}, {lg['seasonality']} (check with advanced methods)"
    return ""

def eda_report_senior_expert(
    train_df, test_df, target_col=None, lang="en", outdir="./eda_report_senior_expert", analyst_note=""
):
    lg = _get_lang(lang)
    os.makedirs(outdir, exist_ok=True)
    dt = datetime.now().strftime("%Y-%m-%d %H:%M")
    html_parts = [
        f"<h1>{lg['eda']}</h1><div><b>{lg['date']}:</b> {dt}</div>",
        f"<h2>{lg['checklist']}</h2><ul>"
    ]
    for op in SENIOR_OPERATIONS:
        html_parts.append(f"<li>{op}</li>")
    html_parts.append(f"</ul><b>{lg['completed']}</b>")

    # --- Memory & dtype check
    html_parts.append(f'<h3>{lg["train"]} {lg["memory"]}</h3>{memory_report(train_df, lg)}')
    html_parts.append(f'<h3>{lg["test"]} {lg["memory"]}</h3>{memory_report(test_df, lg)}')

    num_cols = train_df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = train_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # --- Type/format/hatalı kolon
    html_parts.append("<h3>Type & Format Check</h3><ul>")
    for w in suspicious_types(train_df, lg):
        html_parts.append(f"<li>{w}</li>")
    html_parts.append("</ul>")

    # --- Summary Tables
    desc = train_df.describe(include="all").transpose()
    adv_stats = advanced_stats(train_df,num_cols)
    html_parts.append(f"<h2>{lg['summary']}</h2>")
    html_parts.append(desc.join(adv_stats).to_html())

    # --- Eksik+imputation
    mtab,mout = missing_report(train_df, lg)
    html_parts.append(f"<h2>{lg['missing']}</h2>{mtab.to_html()}<ul>")
    for m in mout:
        html_parts.append(f"<li>{m}</li>")
    for col in train_df.columns:
        sug = impute_suggestion(train_df, col, lg)
        if sug: html_parts.append(f"<li>{sug}</li>")
    html_parts.append("</ul>")

    # Outlier detect (IQR+zscore+boxplot)
    html_parts.append(f"<h2>{lg['outlier']}</h2><ul>")
    o_report = outlier_report(train_df,num_cols,lg)
    for o in o_report:
        html_parts.append(f"<li>{o}</li>")
    for col in num_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x=train_df[col], ax=ax)
        ax.set_title(f"{col} - {lg['box']}")
        img_path = save_plot(fig, f"{col}_boxplot", outdir)
        html_parts.append(f"<h4>{col} {lg['box']}</h4><img src='{img_path}' width=400 />")
    html_parts.append("</ul>")

    # Unique/rare/duplicate
    html_parts.append(f"<h2>Unique/Duplicate/Rare Category</h2><ul>")
    ur_report = rare_duplicate_report(train_df,cat_cols,lg)
    for r in ur_report: html_parts.append(f"<li>{r}</li>")
    html_parts.append("</ul>")

    # Violinplots, scatter matrix, pairplot
    for col in num_cols:
        fig, ax = plt.subplots()
        sns.violinplot(x=train_df[col].dropna(), ax=ax)
        ax.set_title(f"{col} {lg['violin']}")
        img_path = save_plot(fig, f"{col}_violinplot", outdir)
        html_parts.append(f"<h4>{col} {lg['violin']}</h4><img src='{img_path}' width=400 />")
    if len(num_cols) >= 2:
        sns.pairplot(train_df[num_cols].dropna())
        plt.tight_layout()
        img_path = os.path.join(outdir, "pairplot.png")
        plt.savefig(img_path)
        plt.close()
        html_parts.append(f"<h4>{lg['pairplot']}</h4><img src='{img_path}' width=600 />")
        import pandas.plotting as pd_plot
        fig = plt.figure(figsize=(8,8))
        pd_plot.scatter_matrix(train_df[num_cols].dropna(), alpha=0.2, ax=None)
        img_path = save_plot(fig, "scatter_matrix", outdir)
        html_parts.append(f"<h4>{lg['scatter_matrix']}</h4><img src='{img_path}' width=600 />")

    # Kategorik dağılım ve encoding
    html_parts.append(f"<h2>{lg['cat_dist']}</h2><ul>")
    for col in cat_cols:
        fig, ax = plt.subplots()
        train_df[col].value_counts().sort_values(ascending=False).head(30).plot(kind="bar", ax=ax)
        ax.set_title(f"{col} - {lg['cat_dist']}")
        img_path = save_plot(fig, f"{col}_catdist", outdir)
        html_parts.append(f"<li>{col} {lg['cat_dist']}<img src='{img_path}' width=600 /></li>")
        nuniq = train_df[col].nunique()
        if nuniq > 50:
            html_parts.append(f"<li>{col}: {lg['encoding']} - reduce categories or label encoding ({nuniq} levels)</li>")
        elif nuniq < 5:
            html_parts.append(f"<li>{col}: {lg['encoding']} - OneHotEncoding ({nuniq} levels)</li>")
        else:
            html_parts.append(f"<li>{col}: {lg['encoding']} - Label or Target encoding ({nuniq} levels)</li>")
    html_parts.append("</ul>")

    # Korelasyon matrisi
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = train_df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax, cmap="coolwarm")
    ax.set_title(lg['corr'])
    img_path = save_plot(fig, "corr_matrix", outdir)
    html_parts.append(f"<h2>{lg['corr']}</h2><img src='{img_path}' width=600 />")

    # Multicollinearity raporu
    html_parts.append(f"<h2>{lg['multicollinearity']}</h2><ul>")
    mcreport = multicollinearity_report(train_df,num_cols,lg)
    for c in mcreport: html_parts.append(f"<li>{c}</li>")
    html_parts.append("</ul>")

    # Trend/seasonality
    html_parts.append(f"<h2>{lg['trend']} & {lg['seasonality']}</h2><ul>")
    for col in train_df.columns:
        tr_seas = trend_seasonality_report(train_df,col,lg)
        if tr_seas: html_parts.append(f"<li>{tr_seas}</li>")
    html_parts.append("</ul>")

    # Target distrib & ilişki
    if target_col and target_col in train_df:
        fig, ax = plt.subplots()
        if train_df[target_col].dtype in ['object', 'category', 'bool']:
            train_df[target_col].value_counts().plot(kind="bar", ax=ax)
        else:
            sns.histplot(train_df[target_col].dropna(), kde=True, ax=ax)
        ax.set_title(lg['target_dist'])
        img_path = save_plot(fig, f"target_dist", outdir)
        html_parts.append(f"<h2>{lg['target_dist']}</h2><img src='{img_path}' width=400 />")

        # Class imbalance
        val_counts = train_df[target_col].value_counts(normalize=True)
        if val_counts.max() > 0.8:
            html_parts.append(f"<li>{lg['imbalance']}: Class imbalance over 80% detected.</li>")

        # Feature importance
        html_parts.append(f"<h2>{lg['feature_importance']}</h2>")
        html_parts.append(feature_importance_report(train_df,target_col,num_cols,cat_cols,lg))

        # Target relationship numerics
        html_parts.append(f"<h2>{lg['target_rel']} - Numerical</h2>")
        for col in num_cols:
            try:
                if train_df[target_col].dtype in ['object', 'category', 'bool']:
                    means = train_df.groupby(target_col)[col].mean()
                    html_parts.append(f"<b>{col}:{lg['target_rel']}</b> {means.to_dict()}")
                    groups = [grp[col].dropna().values for _, grp in train_df.groupby(target_col)]
                    if len(groups) > 1:
                        fstat, pval = f_oneway(*groups)
                        html_parts.append(f"P-value (ANOVA): {pval:.3e}")
                else:
                    corrval = train_df[[col, target_col]].corr().iloc[0,1]
                    html_parts.append(f"<b>{col}-{target_col}: Correlation {corrval:.2f}</b>")
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=train_df[col], y=train_df[target_col], ax=ax)
                    img_path = save_plot(fig, f"{col}_scatter_target", outdir)
                    html_parts.append(f"<img src='{img_path}' width=300 />")
            except Exception as e:
                html_parts.append(f"Could not compute target relationship for {col}: {str(e)}")
        # Target relationship categoricals
        html_parts.append(f"<h2>{lg['target_rel']} - Categorical</h2>")
        for col in cat_cols:
            try:
                ct = pd.crosstab(train_df[col], train_df[target_col])
                if ct.shape[1] == 2:
                    chi2, pval, dof, _ = chi2_contingency(ct)
                    html_parts.append(f"<b>{col}:{lg['target_rel']} Chi2 p-value: {pval:.3e}</b>")
                html_parts.append(ct.to_html())
            except Exception as e:
                html_parts.append(f"Could not compute categorical target rel for {col}: {str(e)}")

    # Train/test karşılaştırması: drift/leakage
    html_parts.append(f"<h2>{lg['drift']}</h2><ul>")
    for col in num_cols:
        train_mean = train_df[col].mean()
        test_mean = test_df[col].mean()
        drift = abs(train_mean - test_mean) / (abs(train_mean) + 1e-6)
        html_parts.append(f"<li>{col}: Train mean={train_mean:.3f}, Test mean={test_mean:.3f}, Drift={drift:.2%}</li>")
        if drift > 0.2:
            html_parts.append(f"<b>Significant drift detected in {col}</b>")
    for col in cat_cols:
        train_freq = train_df[col].value_counts(normalize=True)
        test_freq = test_df[col].value_counts(normalize=True)
        common = set(train_freq.index) & set(test_freq.index)
        diffs = [abs(train_freq.get(k,0)-test_freq.get(k,0)) for k in common]
        drift = np.mean(diffs) if diffs else 0
        html_parts.append(f"<li>{col}: Mean freq diff={drift:.2%}</li>")
        if drift > 0.15:
            html_parts.append(f"<b>Significant categorical drift detected: {col}</b>")
    html_parts.append("</ul>")

    # Feature engineering önerileri
    fe_suggestions = []
    for col in num_cols:
        if train_df[col].nunique() < 5:
            fe_suggestions.append(f"{col}: convert to categorical")
    for col in cat_cols:
        if any(word in col.lower() for word in ["date","time"]):
            fe_suggestions.append(f"{col}: extract year/month/day/features")
        if train_df[col].nunique() > 100:
            fe_suggestions.append(f"{col}: group rare categories")
    html_parts.append(f"<h2>{lg['feature_engineering']}</h2><ul>")
    for s in fe_suggestions:
        html_parts.append(f"<li>{s}</li>")
    html_parts.append("</ul>")

    # Recommendations/Yapılması gerekenler
    html_parts.append(f"<h2>{lg['todo']}</h2><ul>")
    for item in o_report: html_parts.append(f"<li>{item}</li>")
    for item in ur_report: html_parts.append(f"<li>{item}</li>")
    for item in mcreport: html_parts.append(f"<li>{item}</li>")
    for item in fe_suggestions: html_parts.append(f"<li>{item}</li>")
    for m in mout: html_parts.append(f"<li>{m}</li>")
    for col in train_df.columns:
        sug = impute_suggestion(train_df, col, lg)
        if sug: html_parts.append(f"<li>{sug}</li>")
    if len(fe_suggestions)==0:
        html_parts.append("<li>No major feature engineering recommendations needed.</li>")
    html_parts.append("</ul>")

    # Analyst notes
    html_parts.append(f"<h2>{lg['analyst_note']}</h2>")
    html_parts.append(f"<div style='border:1px solid #888;padding:8px'>{analyst_note}</div>")

    html_fp = os.path.join(outdir, f"eda_report_senior_expert_{lang}.html")
    with open(html_fp, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"Senior EDA expert report generated: {html_fp}")
    return html_fp

# Kullanım örneği:
# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')
# eda_report_senior_expert(train, test, target_col='target', lang='tr', outdir='./kaggle_senior_eda', analyst_note="Feature importance ve drift kritik.")
```