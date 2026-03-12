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

# 4. Detaylı EDA

```py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew, kurtosis, ttest_ind, f_oneway, chi2_contingency, zscore, shapiro, jarque_bera, anderson, ks_2samp
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from scipy import stats
from datetime import datetime

sns.set(style="whitegrid")

# Diller
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
        "qqplot": "QQ Plot",
        "cdf": "Cumulative Distribution Function",
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
        "seasonality": "Seasonality Detected",
        "quality_score": "Data Quality Score",
        "leakage": "Leakage Risk",
        "interactions": "Feature Interactions",
        "residuals": "Residual Plot",
        "distribution": "Distribution Fitting",
        "sampling_bias": "Sampling Bias",
        "pca": "PCA Explained Variance",
        "tsne": "t-SNE Visualization",
        "business_rule": "Business Logic Violation"
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
        "qqplot": "QQ Plot",
        "cdf": "Kümülatif Dağılım Fonksiyonu",
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
        "seasonality": "Sezonluk Etki Algılandı",
        "quality_score": "Veri Kalite Skoru",
        "leakage": "Leakage Riski",
        "interactions": "Değişken Etkileşimleri",
        "residuals": "Artık Plotu",
        "distribution": "Dağılım Uyumu",
        "sampling_bias": "Sampling Bias",
        "pca": "PCA Açıklanan Varyans",
        "tsne": "t-SNE Görselleştirme",
        "business_rule": "İş Kuralı İhlali"
    }
}

SENIOR_OPERATIONS = [
    "Type & Format Control (dtype, suspicious type, uniform column, wrong format)",
    "Missing Value Analysis & Imputation Suggestion",
    "Outlier Detection (IQR, Z-Score, LOF, Isolation Forest, Boxplot)",
    "Advanced Statistical Tests (Normality, Stationarity, Distribution Fitting)",
    "Unique, Duplicate, Rare Category Analysis",
    "Feature Interactions (Polynomial, Mutual Info)",
    "Sampling Bias Detection (KS-test)",
    "Advanced Summary (IQR, skewness, kurtosis, percentiles, multimodal)",
    "Multicollinearity",
    "Feature Importance",
    "Categorical Encoding Necessity & Rare Detection",
    "Numeric/Target Relationship Analysis (correlation, ANOVA/t-test, scatter, residuals)",
    "Categorical/Target Relationship (crosstab, chi2 test, imbalance, leakage)",
    "Train/Test Distribution Drift & Leakage",
    "Feature Engineering Suggestions (date extract, rare grouping, redundant vars)",
    "Time Series Analysis (trend, seasonality, decomposition, ADF)",
    "Memory Analysis, RAM Optimization",
    "Data Quality Score",
    "Business Rule Validation",
    "All Graphs: Box, Violin, Pair, Scatter Matrix, QQ, CDF, Residuals",
    "Analyst's Notes Section",
    "PCA variance and t-SNE visualization"
]

def _get_lang(lang_code):
    return LANGS.get(lang_code, LANGS["en"])

def save_plot(fig, name, outdir):
    fname = os.path.join(outdir, f"{name}.png")
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return fname

def normality_tests(df, num_cols):
    results = {}
    for col in num_cols:
        data = df[col].dropna()
        if len(data) > 3:
            _, p_shapiro = shapiro(data[:5000])
            _, p_jb = jarque_bera(data)
            anderson_stat = anderson(data)
            results[col] = {'shapiro_p': p_shapiro, 'jarque_bera_p': p_jb, 'anderson': anderson_stat.statistic}
    return pd.DataFrame(results).T

def advanced_outlier_detection(df, num_cols):
    X = df[num_cols].dropna()
    results = {}
    if len(X)>0:
        # Isolation Forest
        try:
            iso = IsolationForest(contamination=0.1, random_state=42)
            iso_preds = iso.fit_predict(X)
            results['isolation_forest'] = (iso_preds == -1).sum()
        except Exception: results['isolation_forest'] = 'n/a'
        # LOF
        try:
            lof = LocalOutlierFactor(contamination=0.1)
            lof_preds = lof.fit_predict(X)
            results['lof'] = (lof_preds == -1).sum()
        except Exception: results['lof'] = 'n/a'
    return results

def feature_interactions(df, num_cols, target):
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    X = df[num_cols].fillna(0)
    try:
        interactions = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(num_cols)
        if target in df.columns:
            mi_scores = mutual_info_regression(interactions, df[target])
            top_interactions = pd.DataFrame({'feature': feature_names, 'mi_score': mi_scores}).nlargest(5, 'mi_score')
            return top_interactions
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()

def time_series_analysis(df, date_col, value_col):
    df_ts = df.set_index(date_col)[value_col].dropna()
    try:
        adf_result = adfuller(df_ts)
        decomposition = seasonal_decompose(df_ts, model='additive', period=7)
        acf_vals = acf(df_ts, nlags=40)
        pacf_vals = pacf(df_ts, nlags=40)
        return {
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'is_stationary': adf_result[1] < 0.05,
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'acf': acf_vals,
            'pacf': pacf_vals
        }
    except Exception:
        return {}

def best_fit_distribution(data, distributions=['norm', 'lognorm', 'expon', 'gamma']):
    results = {}
    for dist_name in distributions:
        dist = getattr(stats, dist_name)
        try:
            params = dist.fit(data)
            ks_stat, p_value = stats.kstest(data, dist_name, args=params)
            results[dist_name] = {'params': params, 'ks_p': p_value}
        except Exception: results[dist_name] = {'params':'n/a','ks_p':0}
    best = max(results.items(), key=lambda x: x[1]['ks_p'])
    return best

def data_quality_score(df):
    score = 100
    missing_pct = df.isnull().mean().mean()
    score -= min(20, missing_pct * 100)
    dup_pct = df.duplicated().sum() / len(df)
    score -= min(10, dup_pct * 100)
    num_cols = df.select_dtypes(include='number').columns
    outlier_pct = sum((np.abs(zscore(df[num_cols].fillna(0))) > 3).sum()) / (len(df) * len(num_cols))
    score -= min(15, outlier_pct * 100)
    high_card = sum(df[col].nunique() > 0.5 * len(df) for col in df.select_dtypes(include='object').columns)
    score -= min(10, high_card * 5)
    return max(0, score)

def leakage_detection(df, target_col):
    leaks = []
    for col in df.columns:
        if col != target_col:
            try:
                corr = df[col].corr(df[target_col])
                if abs(corr) > 0.99: leaks.append(f"LEAKAGE RISK: {col} (corr={corr:.4f})")
            except: pass
    target_dist = df[target_col].value_counts(normalize=True)
    for col in df.select_dtypes(include='object').columns:
        col_dist = df[col].value_counts(normalize=True)
        if len(target_dist)==len(col_dist) and (target_dist.values==col_dist.values).all():
            leaks.append(f"LEAKAGE RISK: {col} (identical distribution)")
    return leaks

def sampling_bias(train_df, test_df, num_cols):
    biases = []
    for col in num_cols:
        train_data = train_df[col].dropna()
        test_data = test_df[col].dropna()
        ks_stat, p_value = ks_2samp(train_data, test_data)
        if p_value < 0.05:
            biases.append({'column': col, 'ks_statistic': ks_stat, 'p_value': p_value, 'verdict': 'SIGNIFICANT DIFFERENCE'})
    return pd.DataFrame(biases)

def dimensionality_insights(df, num_cols):
    X = df[num_cols].fillna(0)
    pca = PCA(n_components=min(10, len(num_cols)))
    pca.fit(X)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = np.argmax(cum_var >= 0.95) + 1
    return {
        'explained_variance': pca.explained_variance_ratio_,
        'n_components_for_95pct': n_components_95,
        'total_variance_2pc': cum_var[1]
    }

def business_rule_checks(df):
    violations = []
    if 'age' in df.columns:
        invalid_age = ((df['age'] < 0) | (df['age'] > 120)).sum()
        if invalid_age > 0: violations.append(f"Age violations: {invalid_age} rows")
    if 'price' in df.columns:
        invalid_price = (df['price'] <= 0).sum()
        if invalid_price > 0: violations.append(f"Price violations: {invalid_price} rows")
    if 'start_date' in df.columns and 'end_date' in df.columns:
        invalid_dates = (df['start_date'] > df['end_date']).sum()
        if invalid_dates > 0: violations.append(f"Date logic violations: {invalid_dates} rows")
    return violations

def qq_plots(df, num_cols, outdir):
    paths = []
    for col in num_cols:
        fig, ax = plt.subplots()
        stats.probplot(df[col].dropna(), dist="norm", plot=ax)
        ax.set_title(f"QQ Plot: {col}")
        paths.append(save_plot(fig, f"{col}_qqplot", outdir))
    return paths

def cdf_plots(df, num_cols, outdir):
    paths = []
    for col in num_cols:
        fig, ax = plt.subplots()
        sorted_data = np.sort(df[col].dropna())
        cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
        ax.plot(sorted_data, cdf)
        ax.set_title(f"CDF: {col}")
        paths.append(save_plot(fig, f"{col}_cdf", outdir))
    return paths

def residual_analysis(df, target_col, num_cols, outdir):
    paths = []
    for col in num_cols:
        X = df[[col]].fillna(0)
        y = df[target_col]
        try:
            model = LinearRegression().fit(X, y)
            residuals = y - model.predict(X)
            fig, ax = plt.subplots()
            ax.scatter(df[col], residuals, alpha=0.5)
            ax.axhline(0, color='red', linestyle='--')
            ax.set_title(f"Residuals: {col}")
            paths.append(save_plot(fig, f"{col}_residual", outdir))
        except Exception: continue
    return paths

def _get_num_cat(df):
    return df.select_dtypes(include=['number']).columns.tolist(), df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

def eda_report_senior_complete(train_df, test_df, target_col=None, lang="en", outdir="eda_senior_complete", analyst_note=""):
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
    num_cols, cat_cols = _get_num_cat(train_df)

    # Memory
    mem_train = train_df.memory_usage(deep=True).sum()/1e6
    mem_test = test_df.memory_usage(deep=True).sum()/1e6
    html_parts.append(f'<h3>{lg["train"]} {lg["memory"]}</h3>{mem_train:.2f} MB')
    html_parts.append(f'<h3>{lg["test"]} {lg["memory"]}</h3>{mem_test:.2f} MB')

    # Type & format check
    html_parts.append("<h3>Type & Format Check</h3><ul>")
    for col in train_df.columns:
        dtype = str(train_df[col].dtype)
        if dtype not in ["float64", "int64", "object", "category", "bool"]:
            html_parts.append(f"<li>{col}: {lg['wrong_type']} ({dtype})</li>")
        if train_df[col].nunique() == 1:
            html_parts.append(f"<li>{col}: Uniform column (only one value)</li>")
        if dtype == "object" and train_df[col].astype(str).str.match(r"^\d{1,}$").sum()>0:
            html_parts.append(f"<li>{col}: May require numeric conversion</li>")
        if dtype == "object" and train_df[col].astype(str).str.match(r"\d{4}-\d{2}-\d{2}").sum()>10:
            html_parts.append(f"<li>{col}: May require datetime conversion</li>")
    html_parts.append("</ul>")

    # Summary + advanced stats
    html_parts.append(f"<h2>{lg['summary']}</h2>")
    desc = train_df.describe(include="all").transpose()
    ext_stats = pd.DataFrame()
    for col in num_cols:
        vals = train_df[col].dropna()
        ext_stats.at[col, "Skewness"] = skew(vals)
        ext_stats.at[col, "Kurtosis"] = kurtosis(vals)
        ext_stats.at[col, "IQR"] = np.percentile(vals,75) - np.percentile(vals,25)
        ext_stats.at[col, "Multimodal"] = "Yes" if len(set(np.histogram(vals, bins=10)[0][np.histogram(vals, bins=10)[0]>0]))>1 else "No"
    html_parts.append(desc.join(ext_stats).to_html())

    # Normallik testleri
    html_parts.append(f"<h2>Normality Tests (Shapiro, JB, Anderson)</h2>")
    html_parts.append(normality_tests(train_df, num_cols).to_html())

    # Dağılım fitting
    html_parts.append(f"<h2>{lg['distribution']}</h2>")
    for col in num_cols:
        vals = train_df[col].dropna()
        if len(vals) > 15:
            best = best_fit_distribution(vals)
            html_parts.append(f"{col}: Best fit: {best[0]}, p={best[1]['ks_p']:.3f}")

    # Outlier (IQR, Z, LOF, IsoForest)
    html_parts.append(f"<h2>{lg['outlier']}</h2>")
    for col in num_cols:
        vals = train_df[col].dropna()
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_outliers = ((vals < lower) | (vals > upper)).sum()
        zvals = np.abs(zscore(vals))
        z_outliers = np.sum(zvals > 3)
        html_parts.append(f"{col}: IQR outliers: {iqr_outliers}, Z>3: {z_outliers}")
    adv_out = advanced_outlier_detection(train_df, num_cols)
    html_parts.append(f"LOF outliers: {adv_out.get('lof','n/a')}, Isolation Forest outliers: {adv_out.get('isolation_forest','n/a')}")

    # Unique/rare/duplicate
    html_parts.append(f"<h2>Unique/Duplicate/Rare Category</h2><ul>")
    for col in cat_cols:
        valc = train_df[col].value_counts()
        if (valc < 5).sum() > 0:
            html_parts.append(f"<li>{col}: {lg['rare']} - {list(valc[valc<5].index)}, n={len(valc[valc<5])}</li>")
        dupes = train_df[train_df.duplicated([col])][col].unique()
        if len(dupes) > 1:
            html_parts.append(f"<li>{col}: {lg['duplicate']} detected ({len(dupes)} values)</li>")
    html_parts.append("</ul>")

    # Feature interaction
    html_parts.append(f"<h2>{lg['interactions']}</h2>")
    inter = feature_interactions(train_df, num_cols, target_col)
    if not inter.empty: html_parts.append(inter.to_html())

    # Sampling bias
    html_parts.append(f"<h2>{lg['sampling_bias']}</h2>")
    bias_df = sampling_bias(train_df, test_df, num_cols)
    if not bias_df.empty: html_parts.append(bias_df.to_html())

    # Multicollinearity
    html_parts.append(f"<h2>{lg['multicollinearity']}</h2><ul>")
    corr = train_df[num_cols].corr()
    for col in corr.columns:
        for c2 in corr.columns:
            if col != c2 and abs(corr.at[col,c2]) > 0.9:
                html_parts.append(f"<li>{col} & {c2}: r={corr.at[col,c2]:.2f}</li>")
    html_parts.append("</ul>")

    # Feature importance
    html_parts.append(f"<h2>{lg['feature_importance']}</h2>")
    feats = num_cols + cat_cols
    if target_col and target_col in train_df:
        y = train_df[target_col]
        X = train_df[feats]
        if y.dtype in ['float64','int64']:
            mi = mutual_info_regression(X.fillna(0), y)
        else:
            mi = mutual_info_classif(X.fillna("missing"), y)
        featimp = pd.DataFrame({'feature':feats, lg['feature_importance']:mi})
        html_parts.append(featimp.sort_values(by=lg['feature_importance'],ascending=False).to_html())

    # Business rule validation
    html_parts.append(f"<h2>{lg['business_rule']}</h2>")
    for v in business_rule_checks(train_df): html_parts.append(f"<li>{v}</li>")

    # Data quality score
    qscore = data_quality_score(train_df)
    html_parts.append(f"<h2>{lg['quality_score']}</h2><b>{qscore}/100</b>")

    # PCA/t-SNE
    html_parts.append(f"<h2>{lg['pca']} / {lg['tsne']}</h2>")
    pca_info = dimensionality_insights(train_df, num_cols)
    html_parts.append(f"PCA 95% variance needs n={pca_info['n_components_for_95pct']} components.")
    # t-SNE plot (optional, can be heavy)
    try:
        tsne = TSNE(n_components=2, random_state=42)
        tsne_data = tsne.fit_transform(train_df[num_cols].fillna(0))
        fig, ax = plt.subplots()
        ax.scatter(tsne_data[:,0], tsne_data[:,1], alpha=0.5)
        ax.set_title("t-SNE")
        img_path = save_plot(fig, "tsne", outdir)
        html_parts.append(f"<img src='{img_path}' width=400 />")
    except Exception: pass

    # Box, Violin, QQ, CDF, Pair, Scatter Matrix
    for col in num_cols:
        # Boxplot
        fig, ax = plt.subplots()
        sns.boxplot(x=train_df[col], ax=ax)
        ax.set_title(f"{col} {lg['box']}")
        img_path = save_plot(fig, f"{col}_box", outdir)
        html_parts.append(f"<img src='{img_path}' width=300 />")
        # Violin
        fig, ax = plt.subplots()
        sns.violinplot(x=train_df[col].dropna(), ax=ax)
        ax.set_title(f"{col} {lg['violin']}")
        img_path = save_plot(fig, f"{col}_violin", outdir)
        html_parts.append(f"<img src='{img_path}' width=300 />")
    if len(num_cols) >= 2:
        sns.pairplot(train_df[num_cols].dropna())
        img_path = os.path.join(outdir, "pairplot.png")
        plt.savefig(img_path)
        plt.close()
        html_parts.append(f"<h4>{lg['pairplot']}</h4><img src='{img_path}' width=600 />")
        import pandas.plotting as pd_plot
        fig = plt.figure(figsize=(8,8))
        pd_plot.scatter_matrix(train_df[num_cols].dropna(), alpha=0.2, ax=None)
        img_path = save_plot(fig, "scatter_matrix", outdir)
        html_parts.append(f"<h4>{lg['scatter_matrix']}</h4><img src='{img_path}' width=600 />")

    # QQ/CDF plots
    qq_paths = qq_plots(train_df, num_cols, outdir)
    cdf_paths = cdf_plots(train_df, num_cols, outdir)
    html_parts.append(f"<h3>{lg['qqplot']}</h3>")
    for img in qq_paths: html_parts.append(f"<img src='{img}' width=300 />")
    html_parts.append(f"<h3>{lg['cdf']}</h3>")
    for img in cdf_paths: html_parts.append(f"<img src='{img}' width=300 />")

    # Residual plots
    if target_col:
        residuals_imgs = residual_analysis(train_df, target_col, num_cols, outdir)
        html_parts.append(f"<h3>{lg['residuals']}</h3>")
        for img in residuals_imgs: html_parts.append(f"<img src='{img}' width=300 />")

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
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax, cmap="coolwarm")
    ax.set_title(lg['corr'])
    img_path = save_plot(fig, "corr_matrix", outdir)
    html_parts.append(f"<h2>{lg['corr']}</h2><img src='{img_path}' width=600 />")

    # Target distrib & ilişki, leakage
    if target_col and target_col in train_df:
        fig, ax = plt.subplots()
        if train_df[target_col].dtype in ['object', 'category', 'bool']:
            train_df[target_col].value_counts().plot(kind="bar", ax=ax)
        else:
            sns.histplot(train_df[target_col].dropna(), kde=True, ax=ax)
        ax.set_title(lg['target_dist'])
        img_path = save_plot(fig, f"target_dist", outdir)
        html_parts.append(f"<h2>{lg['target_dist']}</h2><img src='{img_path}' width=400 />")
        val_counts = train_df[target_col].value_counts(normalize=True)
        if val_counts.max() > 0.8:
            html_parts.append(f"<li>{lg['imbalance']}: Class imbalance over 80% detected.</li>")
        # Leakage
        leaks = leakage_detection(train_df, target_col)
        for leak in leaks: html_parts.append(f"<li>{lg['leakage']} {leak}</li>")

        # Numeric + categorical target relationship
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
            except Exception as e:
                html_parts.append(f"Could not compute target relationship for {col}: {str(e)}")
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

    # Train/test drift
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

    # Feature engineering suggestions
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

    # Time series (trend/seasonality)
    html_parts.append(f"<h2>{lg['trend']} & {lg['seasonality']}</h2><ul>")
    for col in train_df.columns:
        if pd.api.types.is_datetime64_any_dtype(train_df[col]):
            ts_info = time_series_analysis(train_df, col, num_cols[0] if num_cols else None)
            if ts_info and ts_info.get('is_stationary', None) is not None:
                html_parts.append(f"<li>{col}: ADF p-value: {ts_info['adf_pvalue']:.4f} Stationary: {ts_info['is_stationary']}</li>")
    html_parts.append("</ul>")

    # Analyst notes
    html_parts.append(f"<h2>{lg['analyst_note']}</h2>")
    html_parts.append(f"<div style='border:1px solid #888;padding:8px'>{analyst_note}</div>")

    html_fp = os.path.join(outdir, f"eda_report_senior_complete_{lang}.html")
    with open(html_fp, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"Senior EDA complete report generated: {html_fp}")
    return html_fp

# Kullanım:
# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')
# eda_report_senior_complete(train, test, target_col='target', lang='tr', outdir='senior_eda_full', analyst_note='Domain mantık ve leakage kontrolü önemli.')
```

# Tam Entegre Senior EDA Kodu

```python
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas.plotting as pd_plot

from scipy.stats import (
    skew, kurtosis, ttest_ind, f_oneway, chi2_contingency,
    zscore, shapiro, jarque_bera, anderson, ks_2samp,
    entropy as sp_entropy, gaussian_kde, sem
)
from scipy.signal import find_peaks
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy import stats

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from datetime import datetime

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")


# ──────────────────────────────────────────────
# DİL DESTEĞİ
# ──────────────────────────────────────────────
LANGS = {
    "en": {
        "checklist": "Senior EDA Operations Checklist",
        "completed": "All senior EDA operations are included below.",
        "summary": "Summary Statistics",
        "missing": "Missing Values",
        "missing_pattern": "Missing Value Patterns",
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
        "qqplot": "QQ Plot",
        "cdf": "Cumulative Distribution Function",
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
        "seasonality": "Seasonality Detected",
        "quality_score": "Data Quality Score",
        "leakage": "Leakage Risk",
        "interactions": "Feature Interactions",
        "residuals": "Residual Plot",
        "distribution": "Distribution Fitting",
        "sampling_bias": "Sampling Bias",
        "pca": "PCA Explained Variance",
        "tsne": "t-SNE Visualization",
        "business_rule": "Business Logic Violation",
        "exec_summary": "Executive Summary",
        "vif": "VIF (Variance Inflation Factor)",
        "cramers": "Cramér's V Matrix",
        "effect_size": "Effect Size (Cohen's d)",
        "nzv": "Near-Zero Variance",
        "cat_entropy": "Categorical Entropy",
        "confidence": "Confidence Intervals",
        "feat_cluster": "Feature Clustering",
        "dup_rows": "Duplicate Row Analysis",
        "dtype_opt": "Dtype Optimization",
        "benford": "Benford's Law Test",
        "multi_corr": "Multi-Method Correlation",
        "normality": "Normality Tests",
        "toc": "Table of Contents",
    },
    "tr": {
        "checklist": "Senior EDA Operasyon Kontrol Listesi",
        "completed": "Aşağıda tüm senior EDA operasyonları eksiksiz olarak yer almaktadır.",
        "summary": "Özet İstatistikler",
        "missing": "Eksik Değerler",
        "missing_pattern": "Eksik Değer Örüntüleri",
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
        "qqplot": "QQ Plot",
        "cdf": "Kümülatif Dağılım Fonksiyonu",
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
        "seasonality": "Sezonluk Etki Algılandı",
        "quality_score": "Veri Kalite Skoru",
        "leakage": "Leakage Riski",
        "interactions": "Değişken Etkileşimleri",
        "residuals": "Artık Plotu",
        "distribution": "Dağılım Uyumu",
        "sampling_bias": "Sampling Bias",
        "pca": "PCA Açıklanan Varyans",
        "tsne": "t-SNE Görselleştirme",
        "business_rule": "İş Kuralı İhlali",
        "exec_summary": "Yönetici Özeti",
        "vif": "VIF (Varyans Enflasyon Faktörü)",
        "cramers": "Cramér's V Matrisi",
        "effect_size": "Etki Büyüklüğü (Cohen's d)",
        "nzv": "Sıfıra Yakın Varyans",
        "cat_entropy": "Kategorik Entropi",
        "confidence": "Güven Aralıkları",
        "feat_cluster": "Öznitelik Kümeleme",
        "dup_rows": "Tekrarlı Satır Analizi",
        "dtype_opt": "Dtype Optimizasyonu",
        "benford": "Benford Kanunu Testi",
        "multi_corr": "Çoklu Korelasyon Yöntemleri",
        "normality": "Normallik Testleri",
        "toc": "İçindekiler",
    },
}

SENIOR_OPERATIONS = [
    "Type & Format Control (dtype, suspicious type, uniform column, wrong format)",
    "Missing Value Deep Analysis (pattern, MCAR hint, msno-style, dendrogram)",
    "Imputation Strategy Recommendation",
    "Outlier Detection (IQR, Z-Score, LOF, Isolation Forest, Boxplot)",
    "Advanced Statistical Tests (Normality — Shapiro, JB, Anderson)",
    "Distribution Fitting (KS-test best fit)",
    "Unique, Duplicate Row & Column, Rare Category Analysis",
    "Feature Interactions (Polynomial, Mutual Info)",
    "Sampling Bias Detection (KS-test per feature)",
    "Advanced Summary (IQR, skewness, kurtosis, percentiles)",
    "Multimodal Detection (KDE peak finding)",
    "Near-Zero Variance Detection",
    "Multicollinearity — VIF",
    "Multi-Method Correlation (Pearson, Spearman, Kendall + discrepancy)",
    "Cramér's V (Categorical-Categorical Correlation)",
    "Feature Importance (Mutual Information)",
    "Effect Size — Cohen's d",
    "Categorical Entropy Analysis",
    "Confidence Intervals for Means",
    "Categorical Encoding Necessity & Rare Detection",
    "Numeric/Target Relationship (correlation, ANOVA/t-test, scatter, residuals)",
    "Categorical/Target Relationship (crosstab, chi2 test, imbalance, leakage)",
    "Train/Test Distribution Drift & Leakage Detection",
    "Feature Engineering Suggestions (date extract, rare grouping, redundant vars)",
    "Time Series Analysis (trend, seasonality, ADF)",
    "Memory Analysis & Dtype Optimization Recommendation",
    "Data Quality Score",
    "Business Rule Validation",
    "Benford's Law Test",
    "Feature Clustering Dendrogram",
    "PCA Explained Variance",
    "t-SNE Visualization",
    "All Graphs: Box, Violin, Pair, Scatter Matrix, QQ, CDF, Residuals",
    "Executive Summary (Auto-generated)",
    "Analyst Notes Section",
]


# ──────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ──────────────────────────────────────────────
def _get_lang(code):
    return LANGS.get(code, LANGS["en"])


def _get_num_cat(df):
    num = df.select_dtypes(include=["number"]).columns.tolist()
    cat = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return num, cat


def save_plot(fig, name, outdir):
    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _img(path, width=400):
    return f'<img src="{path}" width="{width}" />'


def _section(id_, title):
    return f'<h2 id="{id_}">{title}</h2>'


def _warn(text):
    return f'<div class="warning">{text}</div>'


def _critical(text):
    return f'<div class="critical">{text}</div>'


def _success(text):
    return f'<div class="success">{text}</div>'


def _card(label, value, color="#3498db"):
    return (
        f'<div class="metric-card">'
        f'<div style="font-size:28px;color:{color};font-weight:bold">{value}</div>'
        f'<div>{label}</div></div>'
    )


# ──────────────────────────────────────────────
# HTML TEMPLATE
# ──────────────────────────────────────────────
def html_wrapper(body, title="Senior EDA Report"):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
body{{font-family:'Segoe UI',sans-serif;margin:40px;background:#f5f5f5;color:#333}}
h1{{color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:10px}}
h2{{color:#34495e;border-left:4px solid #e74c3c;padding-left:12px;margin-top:30px}}
h3{{color:#7f8c8d}}
table{{border-collapse:collapse;width:100%;margin:15px 0;font-size:13px}}
th{{background:#3498db;color:white;padding:10px;text-align:left}}
td{{border:1px solid #ddd;padding:8px}}
tr:nth-child(even){{background:#f2f2f2}}
tr:hover{{background:#e8f4fd}}
img{{border:1px solid #ddd;border-radius:4px;margin:10px;box-shadow:2px 2px 5px rgba(0,0,0,.1)}}
.warning{{background:#fff3cd;border-left:4px solid #ffc107;padding:10px;margin:10px 0}}
.critical{{background:#f8d7da;border-left:4px solid #dc3545;padding:10px;margin:10px 0}}
.success{{background:#d4edda;border-left:4px solid #28a745;padding:10px;margin:10px 0}}
.toc{{background:white;padding:20px;border-radius:8px;box-shadow:0 2px 5px rgba(0,0,0,.1)}}
.metric-card{{display:inline-block;background:white;padding:20px;margin:10px;
border-radius:8px;box-shadow:0 2px 5px rgba(0,0,0,.1);min-width:180px;text-align:center}}
ul{{line-height:1.8}}
</style>
</head>
<body>
{body}
</body>
</html>"""


# ──────────────────────────────────────────────
# ANALİZ FONKSİYONLARI
# ──────────────────────────────────────────────
def executive_summary(df, target_col, num_cols, cat_cols):
    insights = []
    n_rows, n_cols_total = df.shape
    missing_total = df.isnull().sum().sum()
    missing_pct = missing_total / (n_rows * n_cols_total) if n_rows * n_cols_total > 0 else 0

    insights.append(f"Dataset: {n_rows:,} rows × {n_cols_total} columns")
    insights.append(f"Numerical features: {len(num_cols)}, Categorical features: {len(cat_cols)}")
    insights.append(f"Missing data: {missing_pct:.1%} overall ({missing_total:,} cells)")

    if missing_pct > 0.2:
        insights.append("⚠️ HIGH missing data ratio — imputation strategy critical")
    elif missing_pct > 0.05:
        insights.append("🟡 Moderate missing data — review column-level missingness")
    else:
        insights.append("✅ Low missing data ratio")

    dup_pct = df.duplicated().sum() / n_rows if n_rows > 0 else 0
    if dup_pct > 0.01:
        insights.append(f"⚠️ {dup_pct:.1%} duplicate rows detected")
    else:
        insights.append("✅ No significant duplicate rows")

    if target_col and target_col in df.columns:
        if df[target_col].dtype == "object" or df[target_col].nunique() < 20:
            imbalance = df[target_col].value_counts(normalize=True).iloc[0]
            if imbalance > 0.9:
                insights.append(f"🔴 Severe class imbalance: majority class = {imbalance:.1%}")
            elif imbalance > 0.7:
                insights.append(f"🟡 Moderate class imbalance: majority class = {imbalance:.1%}")
            else:
                insights.append("✅ Balanced classes")

    for col in num_cols:
        sk = abs(skew(df[col].dropna())) if len(df[col].dropna()) > 3 else 0
        if sk > 2:
            insights.append(f"📊 {col}: highly skewed (|skew|={sk:.2f}) — consider log/sqrt transform")

    return insights


# --- Missing Value Deep ---
def missing_value_deep(df, outdir):
    missing_pct = df.isnull().mean().sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    plots = []

    # Matrix-style heatmap
    if len(missing_pct) > 0:
        fig, ax = plt.subplots(figsize=(min(20, len(df.columns)), 6))
        sns.heatmap(df.isnull().astype(int), cbar=False, yticklabels=False, ax=ax, cmap="YlOrRd")
        ax.set_title("Missing Value Matrix")
        plots.append(save_plot(fig, "missing_matrix", outdir))

        # Missing correlation
        miss_corr = df.isnull().corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(miss_corr, annot=True, fmt=".2f", ax=ax, cmap="coolwarm")
        ax.set_title("Missing Value Correlation")
        plots.append(save_plot(fig, "missing_corr", outdir))

    # Imputation recommendations
    recs = {}
    for col in missing_pct.index:
        pct = missing_pct[col]
        dtype = str(df[col].dtype)
        if pct > 0.6:
            recs[col] = f"DROP COLUMN (>{pct:.0%} missing)"
        elif pct > 0.2:
            if dtype in ["float64", "int64"]:
                sk_val = abs(skew(df[col].dropna())) if len(df[col].dropna()) > 3 else 0
                recs[col] = "Median imputation (skewed)" if sk_val > 1 else "Mean or KNN/MICE"
            else:
                recs[col] = "Mode imputation or new category 'Unknown'"
        else:
            if dtype in ["float64", "int64"]:
                recs[col] = "KNN Imputer / Iterative Imputer"
            else:
                recs[col] = "Mode imputation"

    mcar_hint = ""
    if len(missing_pct) > 1:
        avg_corr = df.isnull()[missing_pct.index].corr().abs().values
        np.fill_diagonal(avg_corr, 0)
        avg = avg_corr.mean()
        mcar_hint = "Possibly MCAR" if avg < 0.1 else "Possibly MAR/MNAR — investigate"

    return missing_pct, recs, mcar_hint, plots


# --- Normality ---
def normality_tests(df, num_cols):
    rows = []
    for col in num_cols:
        data = df[col].dropna()
        if len(data) < 8:
            continue
        sample = data.sample(min(5000, len(data)), random_state=42)
        _, p_sh = shapiro(sample)
        _, p_jb = jarque_bera(data)
        ad = anderson(data)
        rows.append({
            "column": col,
            "shapiro_p": round(p_sh, 6),
            "jarque_bera_p": round(p_jb, 6),
            "anderson_stat": round(ad.statistic, 4),
            "normal_5pct": "Yes" if p_sh > 0.05 and p_jb > 0.05 else "No",
        })
    return pd.DataFrame(rows)


# --- Distribution Fitting ---
def best_fit_distribution(data, distributions=None):
    if distributions is None:
        distributions = ["norm", "lognorm", "expon", "gamma", "beta", "weibull_min"]
    results = {}
    for name in distributions:
        dist = getattr(stats, name, None)
        if dist is None:
            continue
        try:
            params = dist.fit(data)
            ks_stat, p_val = stats.kstest(data, name, args=params)
            results[name] = {"params": params, "ks_p": p_val, "ks_stat": ks_stat}
        except Exception:
            pass
    if not results:
        return ("unknown", {"ks_p": 0})
    best = max(results.items(), key=lambda x: x[1]["ks_p"])
    return best


# --- Outlier ---
def advanced_outlier_detection(df, num_cols):
    X = df[num_cols].dropna()
    results = {}
    if len(X) == 0:
        return results
    try:
        iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        results["isolation_forest"] = int((iso.fit_predict(X) == -1).sum())
    except Exception:
        results["isolation_forest"] = "n/a"
    try:
        lof = LocalOutlierFactor(contamination=0.05, n_jobs=-1)
        results["lof"] = int((lof.fit_predict(X) == -1).sum())
    except Exception:
        results["lof"] = "n/a"
    return results


# --- Multimodal Detection (KDE peaks) ---
def detect_multimodal(series, col_name):
    data = series.dropna().values
    if len(data) < 30:
        return {"column": col_name, "n_modes": 1, "is_multimodal": False}
    try:
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 500)
        density = kde(x_range)
        peaks, _ = find_peaks(density, height=max(density) * 0.1, distance=20)
        return {
            "column": col_name,
            "n_modes": len(peaks),
            "is_multimodal": len(peaks) > 1,
        }
    except Exception:
        return {"column": col_name, "n_modes": 1, "is_multimodal": False}


# --- Near-Zero Variance ---
def near_zero_variance(df, num_cols, threshold=0.01):
    rows = []
    for col in num_cols:
        vc = df[col].value_counts(normalize=True)
        freq_ratio = vc.iloc[0] if len(vc) > 0 else 1
        unique_pct = df[col].nunique() / len(df) if len(df) > 0 else 0
        if freq_ratio > 0.95 or unique_pct < threshold:
            rows.append({
                "column": col,
                "dominant_pct": f"{freq_ratio:.1%}",
                "unique_pct": f"{unique_pct:.4%}",
                "recommendation": "DROP — near-zero variance",
            })
    return pd.DataFrame(rows)


# --- VIF ---
def vif_analysis(df, num_cols):
    cols = [c for c in num_cols if df[c].notna().sum() > 10]
    if len(cols) < 2:
        return pd.DataFrame()
    X = df[cols].fillna(0).copy()
    X["__const__"] = 1
    rows = []
    for i, col in enumerate(cols):
        try:
            v = variance_inflation_factor(X.values, i)
        except Exception:
            v = np.nan
        risk = "🔴 CRITICAL (>10)" if v > 10 else ("🟡 MODERATE (5-10)" if v > 5 else "🟢 OK")
        rows.append({"feature": col, "VIF": round(v, 2), "risk": risk})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


# --- Multi Correlation ---
def multi_correlation(df, num_cols):
    pearson = df[num_cols].corr(method="pearson")
    spearman = df[num_cols].corr(method="spearman")
    kendall = df[num_cols].corr(method="kendall")
    disc = []
    for i, c1 in enumerate(num_cols):
        for c2 in num_cols[i + 1:]:
            p = pearson.loc[c1, c2]
            s = spearman.loc[c1, c2]
            if abs(p - s) > 0.2:
                disc.append({
                    "col1": c1, "col2": c2,
                    "pearson": round(p, 3), "spearman": round(s, 3),
                    "note": "Non-linear relationship likely",
                })
    return pearson, spearman, kendall, pd.DataFrame(disc)


# --- Cramér's V ---
def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct)[0]
    n = ct.sum().sum()
    min_dim = min(ct.shape) - 1
    return np.sqrt(chi2 / (n * max(min_dim, 1)))


def categorical_corr_matrix(df, cat_cols):
    n = len(cat_cols)
    mat = pd.DataFrame(np.zeros((n, n)), index=cat_cols, columns=cat_cols)
    for i, c1 in enumerate(cat_cols):
        for j, c2 in enumerate(cat_cols):
            if i <= j:
                v = cramers_v(df[c1].fillna("_NA_"), df[c2].fillna("_NA_"))
                mat.loc[c1, c2] = round(v, 3)
                mat.loc[c2, c1] = round(v, 3)
    return mat


# --- Cohen's d ---
def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0
    pooled = np.sqrt(((n1 - 1) * g1.var() + (n2 - 1) * g2.var()) / (n1 + n2 - 2))
    return (g1.mean() - g2.mean()) / pooled if pooled > 0 else 0


def target_effect_sizes(df, target_col, num_cols):
    if target_col not in df.columns:
        return pd.DataFrame()
    classes = df[target_col].dropna().unique()
    if len(classes) > 20 or len(classes) < 2:
        return pd.DataFrame()
    rows = []
    for col in num_cols:
        for i, c1 in enumerate(classes):
            for c2 in classes[i + 1:]:
                g1 = df.loc[df[target_col] == c1, col].dropna()
                g2 = df.loc[df[target_col] == c2, col].dropna()
                d = cohens_d(g1, g2)
                eff = "Large" if abs(d) > 0.8 else ("Medium" if abs(d) > 0.5 else "Small")
                rows.append({"feature": col, "class1": c1, "class2": c2,
                             "cohens_d": round(d, 3), "effect": eff})
    return pd.DataFrame(rows)


# --- Entropy ---
def categorical_entropy(df, cat_cols):
    rows = []
    for col in cat_cols:
        probs = df[col].value_counts(normalize=True)
        ent = sp_entropy(probs, base=2)
        max_ent = np.log2(df[col].nunique()) if df[col].nunique() > 1 else 1
        norm_ent = round(ent / max_ent, 3) if max_ent > 0 else 0
        rows.append({
            "column": col,
            "entropy": round(ent, 3),
            "max_entropy": round(max_ent, 3),
            "normalized": norm_ent,
            "info": "Low info" if norm_ent < 0.3 else "OK",
        })
    return pd.DataFrame(rows)


# --- Confidence Intervals ---
def confidence_intervals(df, num_cols, confidence=0.95):
    rows = []
    for col in num_cols:
        data = df[col].dropna()
        if len(data) < 3:
            continue
        m = data.mean()
        se = sem(data)
        ci = stats.t.interval(confidence, df=len(data) - 1, loc=m, scale=se)
        rows.append({
            "column": col, "mean": round(m, 4),
            "ci_lower": round(ci[0], 4), "ci_upper": round(ci[1], 4),
            "margin": round(ci[1] - m, 4),
        })
    return pd.DataFrame(rows)


# --- Feature Clustering ---
def feature_clustering(df, num_cols, outdir):
    if len(num_cols) < 3:
        return pd.DataFrame(), None
    corr = df[num_cols].corr().abs()
    distance = 1 - corr
    np.fill_diagonal(distance.values, 0)
    distance = distance.clip(lower=0)
    condensed = squareform(distance.values, checks=False)
    Z = linkage(condensed, method="ward")

    fig, ax = plt.subplots(figsize=(max(8, len(num_cols) * 0.6), 6))
    dendrogram(Z, labels=num_cols, ax=ax, leaf_rotation=90)
    ax.set_title("Feature Clustering Dendrogram")
    path = save_plot(fig, "feature_dendrogram", outdir)

    n_clust = min(max(2, len(num_cols) // 3), len(num_cols))
    clusters = fcluster(Z, t=n_clust, criterion="maxclust")
    cmap = pd.DataFrame({"feature": num_cols, "cluster": clusters})
    return cmap, path


# --- Duplicate Analysis ---
def duplicate_analysis(df):
    n = len(df)
    n_dup = df.duplicated().sum()
    col_dups = {}
    for col in df.columns:
        d = df.duplicated(subset=[col]).sum()
        if d > 0:
            col_dups[col] = d
    return {"total_rows": n, "exact_duplicates": n_dup,
            "pct": f"{n_dup / n:.2%}" if n else "0%",
            "column_level": col_dups}


# --- Dtype Optimization ---
def optimize_dtypes(df):
    before_mb = df.memory_usage(deep=True).sum() / 1e6
    recs = []
    for col in df.columns:
        dt = df[col].dtype
        if dt == "float64":
            if df[col].dropna().apply(lambda x: float(x).is_integer()).all():
                recs.append(f"{col}: float64 → int32")
            elif df[col].min() > np.finfo(np.float32).min and df[col].max() < np.finfo(np.float32).max:
                recs.append(f"{col}: float64 → float32 (save 50% RAM)")
        elif dt == "int64":
            mn, mx = df[col].min(), df[col].max()
            if mn >= 0 and mx < 255:
                recs.append(f"{col}: int64 → uint8")
            elif mn >= -128 and mx < 127:
                recs.append(f"{col}: int64 → int8")
            elif mn >= -32768 and mx < 32767:
                recs.append(f"{col}: int64 → int16")
            elif mn >= -2147483648 and mx < 2147483647:
                recs.append(f"{col}: int64 → int32")
        elif dt == "object":
            if df[col].nunique() / max(len(df), 1) < 0.5:
                recs.append(f"{col}: object → category")
    return recs, before_mb


# --- Benford's Law ---
def benfords_law(series, col_name):
    first = series.dropna().abs().astype(str).str.lstrip("0.").str[0]
    first = first[first.isin([str(i) for i in range(1, 10)])]
    if len(first) < 50:
        return None
    observed = first.value_counts(normalize=True).sort_index()
    expected = pd.Series(
        [np.log10(1 + 1 / d) for d in range(1, 10)],
        index=[str(d) for d in range(1, 10)],
    )
    obs = observed.reindex(expected.index, fill_value=0)
    chi2_stat, p_val = stats.chisquare(obs * len(first), expected * len(first))
    return {"column": col_name, "chi2": round(chi2_stat, 3),
            "p_value": round(p_val, 4), "conforms": p_val > 0.05}


# --- Feature Interactions ---
def feature_interactions(df, num_cols, target):
    if target is None or target not in df.columns or len(num_cols) < 2:
        return pd.DataFrame()
    cols = num_cols[:15]  # limit
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    X = df[cols].fillna(0)
    y = df[target]
    try:
        Xt = poly.fit_transform(X)
        names = poly.get_feature_names_out(cols)
        if y.dtype in ["float64", "int64"]:
            mi = mutual_info_regression(Xt, y, random_state=42)
        else:
            le = LabelEncoder()
            mi = mutual_info_classif(Xt, le.fit_transform(y.astype(str)), random_state=42)
        return pd.DataFrame({"feature": names, "mi_score": mi}).nlargest(10, "mi_score")
    except Exception:
        return pd.DataFrame()


# --- Time Series ---
def time_series_analysis(df, date_col, value_col):
    if value_col is None or value_col not in df.columns:
        return {}
    try:
        ts = df.set_index(date_col)[value_col].dropna().sort_index()
        if len(ts) < 14:
            return {}
        adf = adfuller(ts)
        result = {"adf_stat": round(adf[0], 4), "adf_p": round(adf[1], 4),
                  "stationary": adf[1] < 0.05}
        try:
            dec = seasonal_decompose(ts, model="additive", period=min(7, len(ts) // 3))
            result["has_trend"] = dec.trend.dropna().std() > ts.std() * 0.1
            result["has_seasonality"] = dec.seasonal.dropna().std() > ts.std() * 0.05
        except Exception:
            pass
        return result
    except Exception:
        return {}


# --- Sampling Bias ---
def sampling_bias(train_df, test_df, num_cols):
    rows = []
    for col in num_cols:
        t1 = train_df[col].dropna()
        t2 = test_df[col].dropna()
        if len(t1) < 5 or len(t2) < 5:
            continue
        ks, p = ks_2samp(t1, t2)
        if p < 0.05:
            rows.append({"column": col, "ks_stat": round(ks, 4),
                         "p_value": round(p, 6), "verdict": "SIGNIFICANT DIFF"})
    return pd.DataFrame(rows)


# --- Leakage ---
def leakage_detection(df, target_col):
    leaks = []
    for col in df.columns:
        if col == target_col:
            continue
        try:
            c = df[col].corr(df[target_col])
            if abs(c) > 0.99:
                leaks.append(f"🔴 {col} — correlation {c:.4f}")
        except Exception:
            pass
    return leaks


# --- Business Rules ---
def business_rule_checks(df):
    violations = []
    for col in df.columns:
        cl = col.lower()
        if "age" in cl:
            bad = ((df[col] < 0) | (df[col] > 120)).sum()
            if bad:
                violations.append(f"Age column '{col}': {bad} invalid rows (outside 0-120)")
        if "price" in cl or "amount" in cl or "salary" in cl or "income" in cl:
            bad = (df[col] < 0).sum()
            if bad:
                violations.append(f"'{col}': {bad} negative values")
        if "email" in cl:
            bad = (~df[col].astype(str).str.contains("@", na=False)).sum()
            if bad:
                violations.append(f"'{col}': {bad} rows without '@'")
    date_cols = [c for c in df.columns if "start" in c.lower() and "date" in c.lower()]
    for sc in date_cols:
        ec = sc.lower().replace("start", "end")
        matches = [c for c in df.columns if c.lower() == ec]
        if matches:
            try:
                bad = (pd.to_datetime(df[sc]) > pd.to_datetime(df[matches[0]])).sum()
                if bad:
                    violations.append(f"Date logic: {sc} > {matches[0]} in {bad} rows")
            except Exception:
                pass
    return violations


# --- Quality Score ---
def data_quality_score(df, num_cols):
    score = 100.0
    missing_pct = df.isnull().mean().mean()
    score -= min(25, missing_pct * 100)
    dup_pct = df.duplicated().sum() / max(len(df), 1)
    score -= min(10, dup_pct * 100)
    if num_cols:
        try:
            z = np.abs(zscore(df[num_cols].fillna(0)))
            outlier_pct = (z > 3).sum().sum() / max(z.size, 1)
            score -= min(15, outlier_pct * 100)
        except Exception:
            pass
    high_card = sum(
        1 for col in df.select_dtypes(include="object").columns
        if df[col].nunique() > 0.5 * len(df)
    )
    score -= min(10, high_card * 5)
    return round(max(0, score), 1)


# --- PCA ---
def pca_analysis(df, num_cols, outdir):
    if len(num_cols) < 2:
        return {}, None
    X = df[num_cols].fillna(0)
    n_comp = min(10, len(num_cols))
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.argmax(cum >= 0.95) + 1) if cum[-1] >= 0.95 else n_comp

    fig, ax = plt.subplots()
    ax.bar(range(1, n_comp + 1), pca.explained_variance_ratio_, alpha=0.6, label="Individual")
    ax.step(range(1, n_comp + 1), cum, where="mid", label="Cumulative", color="red")
    ax.axhline(0.95, ls="--", color="gray")
    ax.set_xlabel("Component")
    ax.set_ylabel("Explained Variance")
    ax.set_title("PCA Explained Variance")
    ax.legend()
    path = save_plot(fig, "pca_variance", outdir)

    return {"explained": pca.explained_variance_ratio_.tolist(),
            "n95": n95, "cum_2pc": round(cum[min(1, n_comp - 1)], 3)}, path


# --- QQ, CDF, Residual Plots ---
def qq_plots(df, num_cols, outdir):
    paths = []
    for col in num_cols:
        data = df[col].dropna()
        if len(data) < 5:
            continue
        fig, ax = plt.subplots()
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title(f"QQ Plot: {col}")
        paths.append(save_plot(fig, f"{col}_qq", outdir))
    return paths


def cdf_plots(df, num_cols, outdir):
    paths = []
    for col in num_cols:
        data = np.sort(df[col].dropna())
        if len(data) < 5:
            continue
        fig, ax = plt.subplots()
        ax.plot(data, np.arange(1, len(data) + 1) / len(data))
        ax.set_title(f"CDF: {col}")
        ax.set_ylabel("Cumulative Probability")
        paths.append(save_plot(fig, f"{col}_cdf", outdir))
    return paths


def residual_plots(df, target_col, num_cols, outdir):
    paths = []
    if target_col is None or target_col not in df.columns:
        return paths
    y = df[target_col]
    if y.dtype not in ["float64", "int64"]:
        return paths
    for col in num_cols:
        if col == target_col:
            continue
        X = df[[col]].fillna(0)
        try:
            mdl = LinearRegression().fit(X, y)
            res = y - mdl.predict(X)
            fig, ax = plt.subplots()
            ax.scatter(df[col], res, alpha=0.4, s=10)
            ax.axhline(0, color="red", ls="--")
            ax.set_title(f"Residuals: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Residual")
            paths.append(save_plot(fig, f"{col}_residual", outdir))
        except Exception:
            continue
    return paths


# ──────────────────────────────────────────────
# ANA RAPOR FONKSİYONU
# ──────────────────────────────────────────────
def eda_report_senior_complete(
    train_df,
    test_df=None,
    target_col=None,
    lang="en",
    outdir="eda_senior_complete",
    analyst_note="",
    max_pairplot_cols=8,
    tsne_sample=2000,
):
    lg = _get_lang(lang)
    os.makedirs(outdir, exist_ok=True)
    dt_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    if test_df is None:
        test_df = train_df.sample(frac=0.2, random_state=42)

    num_cols, cat_cols = _get_num_cat(train_df)
    # target'ı num_cols'dan çıkar
    if target_col and target_col in num_cols:
        num_cols_no_target = [c for c in num_cols if c != target_col]
    else:
        num_cols_no_target = num_cols[:]

    H = []  # html parts
    toc_items = []

    def add_section(anchor, title, content=""):
        toc_items.append((anchor, title))
        H.append(_section(anchor, title))
        if content:
            H.append(content)

    # ── HEADER ──
    H.append(f"<h1>{lg['eda']}</h1>")
    H.append(f"<div><b>{lg['date']}:</b> {dt_str}</div><br>")

    # ── METRIC CARDS ──
    n_rows, n_cols_total = train_df.shape
    miss_pct = train_df.isnull().mean().mean()
    qscore = data_quality_score(train_df, num_cols)
    H.append(_card("Rows", f"{n_rows:,}"))
    H.append(_card("Columns", n_cols_total))
    H.append(_card("Numerical", len(num_cols), "#27ae60"))
    H.append(_card("Categorical", len(cat_cols), "#8e44ad"))
    H.append(_card("Missing %", f"{miss_pct:.1%}", "#e67e22"))
    H.append(_card(lg["quality_score"], f"{qscore}/100",
                   "#27ae60" if qscore > 80 else ("#e67e22" if qscore > 60 else "#e74c3c")))
    H.append("<br>")

    # ── CHECKLIST ──
    add_section("checklist", lg["checklist"])
    H.append("<ul>")
    for op in SENIOR_OPERATIONS:
        H.append(f'<li>✅ {op}</li>')
    H.append(f"</ul>{_success(lg['completed'])}")

    # ── EXECUTIVE SUMMARY ──
    add_section("exec", lg["exec_summary"])
    insights = executive_summary(train_df, target_col, num_cols, cat_cols)
    H.append("<ul>")
    for ins in insights:
        H.append(f"<li>{ins}</li>")
    H.append("</ul>")

    # ── MEMORY & DTYPE ──
    add_section("memory", lg["memory"])
    dtype_recs, mem_before = optimize_dtypes(train_df)
    mem_test = test_df.memory_usage(deep=True).sum() / 1e6
    H.append(f"<b>{lg['train']}:</b> {mem_before:.2f} MB &nbsp; | &nbsp; <b>{lg['test']}:</b> {mem_test:.2f} MB")
    if dtype_recs:
        H.append(f"<h3>{lg['dtype_opt']}</h3><ul>")
        for r in dtype_recs:
            H.append(f"<li>{r}</li>")
        H.append("</ul>")
    else:
        H.append(_success("No dtype optimization needed."))

    # ── TYPE & FORMAT ──
    add_section("typecheck", "Type & Format Check")
    issues = []
    for col in train_df.columns:
        dtype = str(train_df[col].dtype)
        if train_df[col].nunique() == 1:
            issues.append(_warn(f"{col}: Uniform column (single value) — consider dropping"))
        if dtype == "object":
            str_col = train_df[col].astype(str)
            if str_col.str.match(r"^\d+\.?\d*$").mean() > 0.8:
                issues.append(_warn(f"{col}: Looks numeric but stored as object"))
            if str_col.str.match(r"\d{{4}}-\d{{2}}-\d{{2}}").mean() > 0.5:
                issues.append(_warn(f"{col}: Looks like datetime but stored as object"))
    if issues:
        H.extend(issues)
    else:
        H.append(_success("No type/format issues detected."))

    # ── SUMMARY STATS ──
    add_section("summary", lg["summary"])
    desc = train_df.describe(include="all").T
    ext = pd.DataFrame(index=num_cols)
    for col in num_cols:
        vals = train_df[col].dropna()
        if len(vals) > 3:
            ext.at[col, "Skewness"] = round(skew(vals), 3)
            ext.at[col, "Kurtosis"] = round(kurtosis(vals), 3)
            ext.at[col, "IQR"] = round(np.percentile(vals, 75) - np.percentile(vals, 25), 3)
            mm = detect_multimodal(vals, col)
            ext.at[col, "Multimodal"] = f"Yes ({mm['n_modes']} modes)" if mm["is_multimodal"] else "No"
    combined = desc.join(ext, how="left")
    H.append(combined.to_html(classes="dataframe", na_rep="-"))

    # ── NORMALITY ──
    add_section("normality", lg["normality"])
    norm_df = normality_tests(train_df, num_cols)
    if not norm_df.empty:
        H.append(norm_df.to_html(classes="dataframe"))
    else:
        H.append("<p>Not enough data for normality tests.</p>")

    # ── DISTRIBUTION FITTING ──
    add_section("distfit", lg["distribution"])
    dist_rows = []
    for col in num_cols:
        vals = train_df[col].dropna()
        if len(vals) > 20:
            name, info = best_fit_distribution(vals)
            dist_rows.append({"column": col, "best_dist": name, "ks_p": round(info["ks_p"], 4)})
    if dist_rows:
        H.append(pd.DataFrame(dist_rows).to_html(classes="dataframe", index=False))

    # ── MISSING VALUE DEEP ──
    add_section("missing", lg["missing_pattern"])
    miss_pct_s, miss_recs, mcar, miss_plots = missing_value_deep(train_df, outdir)
    if len(miss_pct_s) > 0:
        H.append(f"<p><b>MCAR hint:</b> {mcar}</p>")
        miss_tbl = pd.DataFrame({"missing_pct": miss_pct_s,
                                  "recommendation": pd.Series(miss_recs)})
        H.append(miss_tbl.to_html(classes="dataframe"))
        for p in miss_plots:
            H.append(_img(p, 600))
    else:
        H.append(_success("No missing values found."))

    # ── OUTLIER ──
    add_section("outlier", lg["outlier"])
    outlier_rows = []
    for col in num_cols:
        vals = train_df[col].dropna()
        if len(vals) < 5:
            continue
        q1, q3 = np.percentile(vals, [25, 75])
        iqr_val = q3 - q1
        iqr_out = int(((vals < q1 - 1.5 * iqr_val) | (vals > q3 + 1.5 * iqr_val)).sum())
        z_out = int((np.abs(zscore(vals)) > 3).sum())
        outlier_rows.append({"column": col, "IQR_outliers": iqr_out, "Z>3_outliers": z_out})
    if outlier_rows:
        H.append(pd.DataFrame(outlier_rows).to_html(classes="dataframe", index=False))
    adv = advanced_outlier_detection(train_df, num_cols)
    if adv:
        H.append(f"<p><b>LOF outliers:</b> {adv.get('lof','n/a')} &nbsp; "
                 f"<b>Isolation Forest outliers:</b> {adv.get('isolation_forest','n/a')}</p>")

    # ── NEAR-ZERO VARIANCE ──
    add_section("nzv", lg["nzv"])
    nzv_df = near_zero_variance(train_df, num_cols)
    if not nzv_df.empty:
        H.append(nzv_df.to_html(classes="dataframe", index=False))
    else:
        H.append(_success("No near-zero variance features detected."))

    # ── DUPLICATE ANALYSIS ──
    add_section("dup", lg["dup_rows"])
    dup_info = duplicate_analysis(train_df)
    H.append(f"<p>Total rows: {dup_info['total_rows']:,} | "
             f"Exact duplicates: {dup_info['exact_duplicates']} ({dup_info['pct']})</p>")
    if dup_info["exact_duplicates"] > 0:
        H.append(_warn(f"{dup_info['exact_duplicates']} duplicate rows — consider deduplication"))

    # ── UNIQUE / RARE CATEGORY ──
    add_section("rare", lg["rare"])
    if cat_cols:
        rare_items = []
        for col in cat_cols:
            vc = train_df[col].value_counts()
            rare = vc[vc < 5]
            if len(rare) > 0:
                rare_items.append(f"<li><b>{col}:</b> {len(rare)} rare categories "
                                  f"(n<5): {list(rare.index[:10])}</li>")
        if rare_items:
            H.append("<ul>" + "".join(rare_items) + "</ul>")
        else:
            H.append(_success("No rare categories."))
    else:
        H.append("<p>No categorical columns.</p>")

    # ── CORRELATION — MULTI METHOD ──
    add_section("corr", lg["multi_corr"])
    if len(num_cols) >= 2:
        pearson, spearman, kendall, disc_df = multi_correlation(train_df, num_cols)

        # Pearson heatmap
        fig, ax = plt.subplots(figsize=(max(8, len(num_cols) * 0.7), max(6, len(num_cols) * 0.5)))
        sns.heatmap(pearson, annot=len(num_cols) <= 15, fmt=".2f", ax=ax, cmap="coolwarm",
                    center=0, square=True, linewidths=0.5)
        ax.set_title("Pearson Correlation")
        H.append(_img(save_plot(fig, "corr_pearson", outdir), 600))

        # Spearman heatmap
        fig, ax = plt.subplots(figsize=(max(8, len(num_cols) * 0.7), max(6, len(num_cols) * 0.5)))
        sns.heatmap(spearman, annot=len(num_cols) <= 15, fmt=".2f", ax=ax, cmap="coolwarm",
                    center=0, square=True, linewidths=0.5)
        ax.set_title("Spearman Correlation")
        H.append(_img(save_plot(fig, "corr_spearman", outdir), 600))

        if not disc_df.empty:
            H.append(_warn("Pearson vs Spearman discrepancies (possible non-linear relationships):"))
            H.append(disc_df.to_html(classes="dataframe", index=False))
    else:
        H.append("<p>Not enough numerical columns for correlation.</p>")

    # ── MULTICOLLINEARITY — VIF ──
    add_section("vif", lg["vif"])
    vif_df = vif_analysis(train_df, num_cols)
    if not vif_df.empty:
        H.append(vif_df.to_html(classes="dataframe", index=False))
        critical_vif = vif_df[vif_df["VIF"] > 10]
        if len(critical_vif) > 0:
            H.append(_critical(f"{len(critical_vif)} features have VIF > 10 — severe multicollinearity"))
    else:
        H.append("<p>Not enough features for VIF.</p>")

    # ── CRAMÉR'S V ──
    add_section("cramers", lg["cramers"])
    if len(cat_cols) >= 2:
        cv_mat = categorical_corr_matrix(train_df, cat_cols)
        fig, ax = plt.subplots(figsize=(max(6, len(cat_cols) * 0.7), max(5, len(cat_cols) * 0.5)))
        sns.heatmap(cv_mat.astype(float), annot=True, fmt=".2f", ax=ax, cmap="YlOrRd",
                    square=True, linewidths=0.5)
        ax.set_title("Cramér's V — Categorical Correlation")
        H.append(_img(save_plot(fig, "cramers_v", outdir), 600))
    else:
        H.append("<p>Not enough categorical columns.</p>")

    # ── ENTROPY ──
    add_section("entropy", lg["cat_entropy"])
    if cat_cols:
        ent_df = categorical_entropy(train_df, cat_cols)
        H.append(ent_df.to_html(classes="dataframe", index=False))
    else:
        H.append("<p>No categorical columns.</p>")

    # ── CONFIDENCE INTERVALS ──
    add_section("ci", lg["confidence"])
    ci_df = confidence_intervals(train_df, num_cols)
    if not ci_df.empty:
        H.append(ci_df.to_html(classes="dataframe", index=False))

    # ── FEATURE IMPORTANCE ──
    add_section("featimp", lg["feature_importance"])
    if target_col and target_col in train_df.columns:
        feats_for_mi = [c for c in num_cols + cat_cols if c != target_col]
        X_mi = train_df[feats_for_mi].copy()
        y_mi = train_df[target_col]
        # Encode categoricals for MI
        for c in cat_cols:
            if c in X_mi.columns:
                X_mi[c] = LabelEncoder().fit_transform(X_mi[c].astype(str))
        X_mi = X_mi.fillna(0)
        try:
            if y_mi.dtype in ["float64", "int64"] and y_mi.nunique() > 20:
                mi_scores = mutual_info_regression(X_mi, y_mi, random_state=42)
            else:
                mi_scores = mutual_info_classif(
                    X_mi, LabelEncoder().fit_transform(y_mi.astype(str)), random_state=42
                )
            mi_df = pd.DataFrame({"feature": feats_for_mi, "MI_score": mi_scores})
            mi_df = mi_df.sort_values("MI_score", ascending=False)
            H.append(mi_df.to_html(classes="dataframe", index=False))

            # Bar chart
            fig, ax = plt.subplots(figsize=(10, max(4, len(feats_for_mi) * 0.3)))
            ax.barh(mi_df["feature"], mi_df["MI_score"])
            ax.set_xlabel("Mutual Information")
            ax.set_title("Feature Importance (MI)")
            ax.invert_yaxis()
            H.append(_img(save_plot(fig, "feature_importance", outdir), 600))
        except Exception as e:
            H.append(f"<p>MI computation failed: {e}</p>")
    else:
        H.append("<p>No target column specified.</p>")

    # ── EFFECT SIZE ──
    add_section("effect", lg["effect_size"])
    if target_col and target_col in train_df.columns:
        es_df = target_effect_sizes(train_df, target_col, num_cols_no_target)
        if not es_df.empty:
            H.append(es_df.to_html(classes="dataframe", index=False))
        else:
            H.append("<p>Effect size not applicable (target is continuous or >20 classes).</p>")
    else:
        H.append("<p>No target column.</p>")

    # ── FEATURE INTERACTIONS ──
    add_section("interactions", lg["interactions"])
    inter_df = feature_interactions(train_df, num_cols_no_target, target_col)
    if not inter_df.empty:
        H.append(inter_df.to_html(classes="dataframe", index=False))
    else:
        H.append("<p>No significant interactions found.</p>")

    # ── FEATURE CLUSTERING ──
    add_section("fclust", lg["feat_cluster"])
    clust_df, clust_path = feature_clustering(train_df, num_cols, outdir)
    if not clust_df.empty:
        H.append(clust_df.to_html(classes="dataframe", index=False))
        if clust_path:
            H.append(_img(clust_path, 600))
    else:
        H.append("<p>Not enough numerical features for clustering.</p>")

    # ── BENFORD'S LAW ──
    add_section("benford", lg["benford"])
    benford_results = []
    for col in num_cols:
        r = benfords_law(train_df[col], col)
        if r:
            benford_results.append(r)
    if benford_results:
        H.append(pd.DataFrame(benford_results).to_html(classes="dataframe", index=False))
    else:
        H.append("<p>Not enough data for Benford's law test.</p>")

    # ── BUSINESS RULES ──
    add_section("business", lg["business_rule"])
    br = business_rule_checks(train_df)
    if br:
        H.append("<ul>")
        for v in br:
            H.append(f"<li>{_warn(v)}</li>")
        H.append("</ul>")
    else:
        H.append(_success("No business rule violations detected."))

    # ── TARGET DISTRIBUTION ──
    if target_col and target_col in train_df.columns:
        add_section("target", lg["target_dist"])
        fig, ax = plt.subplots(figsize=(8, 4))
        if train_df[target_col].dtype in ["object", "category", "bool"] or train_df[target_col].nunique() < 20:
            train_df[target_col].value_counts().plot(kind="bar", ax=ax, color="#3498db")
        else:
            sns.histplot(train_df[target_col].dropna(), kde=True, ax=ax, color="#3498db")
        ax.set_title(lg["target_dist"])
        H.append(_img(save_plot(fig, "target_dist", outdir), 500))

        # Imbalance check
        if train_df[target_col].nunique() < 20:
            vc = train_df[target_col].value_counts(normalize=True)
            if vc.iloc[0] > 0.8:
                H.append(_critical(f"{lg['imbalance']}: Majority class = {vc.iloc[0]:.1%}"))
            elif vc.iloc[0] > 0.6:
                H.append(_warn(f"Moderate imbalance: Majority class = {vc.iloc[0]:.1%}"))
            else:
                H.append(_success("Classes are reasonably balanced."))

        # ── LEAKAGE ──
        add_section("leakage", lg["leakage"])
        leaks = leakage_detection(train_df, target_col)
        if leaks:
            for lk in leaks:
                H.append(_critical(lk))
        else:
            H.append(_success("No leakage risk detected."))

        # ── TARGET — NUMERICAL RELATIONSHIP ──
        add_section("target_num", f"{lg['target_rel']} — Numerical")
        for col in num_cols_no_target:
            try:
                if train_df[target_col].dtype in ["object", "category", "bool"] or train_df[target_col].nunique() < 20:
                    groups = [g[col].dropna().values for _, g in train_df.groupby(target_col)]
                    groups = [g for g in groups if len(g) > 1]
                    if len(groups) >= 2:
                        fstat, pval = f_oneway(*groups)
                        H.append(f"<p><b>{col}:</b> ANOVA p={pval:.3e} "
                                 f"{'✅ Significant' if pval < 0.05 else '❌ Not significant'}</p>")
                else:
                    cv = train_df[[col, target_col]].corr().iloc[0, 1]
                    H.append(f"<p><b>{col} ↔ {target_col}:</b> Pearson r = {cv:.3f}</p>")
                    # Scatter
                    fig, ax = plt.subplots()
                    ax.scatter(train_df[col], train_df[target_col], alpha=0.3, s=10)
                    ax.set_xlabel(col)
                    ax.set_ylabel(target_col)
                    ax.set_title(f"{col} vs {target_col} (r={cv:.3f})")
                    H.append(_img(save_plot(fig, f"{col}_vs_target", outdir), 400))
            except Exception as e:
                H.append(f"<p>{col}: Error — {e}</p>")

        # ── TARGET — CATEGORICAL RELATIONSHIP ──
        add_section("target_cat", f"{lg['target_rel']} — Categorical")
        for col in cat_cols:
            if col == target_col:
                continue
            try:
                ct = pd.crosstab(train_df[col], train_df[target_col])
                chi2, pval, dof, _ = chi2_contingency(ct)
                H.append(f"<p><b>{col}:</b> Chi² p-value = {pval:.3e} "
                         f"{'✅ Significant' if pval < 0.05 else ''}</p>")
                if ct.shape[0] <= 20 and ct.shape[1] <= 10:
                    H.append(ct.to_html(classes="dataframe"))
            except Exception as e:
                H.append(f"<p>{col}: {e}</p>")

    # ── TRAIN / TEST DRIFT ──
    add_section("drift", lg["drift"])
    drift_items = []
    for col in num_cols:
        tr_m = train_df[col].mean()
        te_m = test_df[col].mean() if col in test_df.columns else None
        if te_m is not None:
            d = abs(tr_m - te_m) / (abs(tr_m) + 1e-8)
            flag = " ⚠️" if d > 0.2 else ""
            drift_items.append(f"<li>{col}: train μ={tr_m:.3f}, test μ={te_m:.3f}, drift={d:.1%}{flag}</li>")
    for col in cat_cols:
        if col not in test_df.columns:
            continue
        tr_f = train_df[col].value_counts(normalize=True)
        te_f = test_df[col].value_counts(normalize=True)
        common = set(tr_f.index) & set(te_f.index)
        avg_diff = np.mean([abs(tr_f.get(k, 0) - te_f.get(k, 0)) for k in common]) if common else 0
        flag = " ⚠️" if avg_diff > 0.15 else ""
        drift_items.append(f"<li>{col}: avg freq diff = {avg_diff:.2%}{flag}</li>")
    H.append("<ul>" + "".join(drift_items) + "</ul>")

    # ── SAMPLING BIAS ──
    add_section("sbias", lg["sampling_bias"])
    common_num = [c for c in num_cols if c in test_df.columns]
    sb_df = sampling_bias(train_df, test_df, common_num)
    if not sb_df.empty:
        H.append(sb_df.to_html(classes="dataframe", index=False))
    else:
        H.append(_success("No significant sampling bias detected (KS-test)."))

    # ── CATEGORICAL DIST & ENCODING ──
    add_section("catdist", lg["cat_dist"])
    for col in cat_cols:
        fig, ax = plt.subplots(figsize=(10, 4))
        train_df[col].value_counts().head(30).plot(kind="bar", ax=ax, color="#8e44ad")
        ax.set_title(f"{col} — {lg['cat_dist']}")
        ax.tick_params(axis='x', rotation=45)
        H.append(_img(save_plot(fig, f"{col}_catdist", outdir), 600))

        nuniq = train_df[col].nunique()
        if nuniq > 50:
            H.append(_warn(f"{col}: High cardinality ({nuniq}) — use Target/Label encoding or reduce"))
        elif nuniq <= 5:
            H.append(f"<p>{col}: {nuniq} levels — OneHot encoding recommended</p>")
        else:
            H.append(f"<p>{col}: {nuniq} levels — Label or Target encoding</p>")

    # ── BOXPLOTS ──
    add_section("box", lg["box"])
    for col in num_cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.boxplot(x=train_df[col].dropna(), ax=axes[0], color="#3498db")
        axes[0].set_title(f"{col} — Boxplot")
        sns.violinplot(x=train_df[col].dropna(), ax=axes[1], color="#2ecc71")
        axes[1].set_title(f"{col} — Violin")
        H.append(_img(save_plot(fig, f"{col}_box_violin", outdir), 700))

    # ── PAIRPLOT ──
    add_section("pairplot", lg["pairplot"])
    pp_cols = num_cols[:max_pairplot_cols]
    if len(pp_cols) >= 2:
        try:
            hue = target_col if (
                target_col and target_col in train_df.columns and
                train_df[target_col].nunique() < 10
            ) else None
            sample_df = train_df[pp_cols + ([target_col] if hue else [])].dropna()
            if len(sample_df) > 2000:
                sample_df = sample_df.sample(2000, random_state=42)
            g = sns.pairplot(sample_df, hue=hue, plot_kws={"alpha": 0.4, "s": 10})
            pair_path = os.path.join(outdir, "pairplot.png")
            g.savefig(pair_path, dpi=100, bbox_inches="tight")
            plt.close()
            H.append(_img(pair_path, 700))
        except Exception as e:
            H.append(f"<p>Pairplot error: {e}</p>")

    # ── SCATTER MATRIX ──
    add_section("scattermat", lg["scatter_matrix"])
    if len(pp_cols) >= 2:
        try:
            sample_df = train_df[pp_cols].dropna()
            if len(sample_df) > 1500:
                sample_df = sample_df.sample(1500, random_state=42)
            axes_arr = pd_plot.scatter_matrix(sample_df, alpha=0.2, figsize=(12, 12))
            scatter_fig = axes_arr[0, 0].get_figure()
            sm_path = save_plot(scatter_fig, "scatter_matrix", outdir)
            H.append(_img(sm_path, 700))
        except Exception as e:
            H.append(f"<p>Scatter matrix error: {e}</p>")

    # ── QQ PLOTS ──
    add_section("qq", lg["qqplot"])
    for p in qq_plots(train_df, num_cols, outdir):
        H.append(_img(p, 350))

    # ── CDF PLOTS ──
    add_section("cdf", lg["cdf"])
    for p in cdf_plots(train_df, num_cols, outdir):
        H.append(_img(p, 350))

    # ── RESIDUAL PLOTS ──
    if target_col:
        add_section("residuals", lg["residuals"])
        res_paths = residual_plots(train_df, target_col, num_cols_no_target, outdir)
        if res_paths:
            for p in res_paths:
                H.append(_img(p, 350))
        else:
            H.append("<p>Residual analysis not applicable (non-numeric target).</p>")

    # ── TIME SERIES ──
    add_section("ts", f"{lg['trend']} & {lg['seasonality']}")
    dt_cols = [c for c in train_df.columns if pd.api.types.is_datetime64_any_dtype(train_df[c])]
    if dt_cols and num_cols:
        for dc in dt_cols:
            vc = num_cols[0]
            ts_info = time_series_analysis(train_df, dc, vc)
            if ts_info:
                H.append(f"<p><b>{dc} → {vc}:</b> ADF p={ts_info.get('adf_p','n/a')}, "
                         f"Stationary={ts_info.get('stationary','n/a')}, "
                         f"Trend={ts_info.get('has_trend','n/a')}, "
                         f"Seasonality={ts_info.get('has_seasonality','n/a')}</p>")
    else:
        H.append("<p>No datetime columns detected.</p>")

    # ── PCA ──
    add_section("pca", lg["pca"])
    pca_info, pca_path = pca_analysis(train_df, num_cols, outdir)
    if pca_info:
        H.append(f"<p>Components for 95% variance: <b>{pca_info['n95']}</b> "
                 f"(out of {len(num_cols)})</p>")
        if pca_path:
            H.append(_img(pca_path, 500))

    # ── t-SNE ──
    add_section("tsne", lg["tsne"])
    if len(num_cols) >= 2:
        try:
            X_tsne = train_df[num_cols].fillna(0)
            if len(X_tsne) > tsne_sample:
                idx = np.random.RandomState(42).choice(len(X_tsne), tsne_sample, replace=False)
                X_tsne = X_tsne.iloc[idx]
            tsne_model = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_tsne) - 1))
            emb = tsne_model.fit_transform(X_tsne)
            fig, ax = plt.subplots(figsize=(8, 6))
            if (target_col and target_col in train_df.columns and
                    train_df[target_col].nunique() < 10):
                targets = train_df[target_col].iloc[idx] if len(train_df) > tsne_sample else train_df[target_col]
                for cls in targets.unique():
                    mask = targets.values == cls
                    ax.scatter(emb[mask, 0], emb[mask, 1], alpha=0.5, s=10, label=str(cls))
                ax.legend()
            else:
                ax.scatter(emb[:, 0], emb[:, 1], alpha=0.4, s=10)
            ax.set_title("t-SNE Visualization")
            H.append(_img(save_plot(fig, "tsne", outdir), 500))
        except Exception as e:
            H.append(f"<p>t-SNE error: {e}</p>")
    else:
        H.append("<p>Not enough features for t-SNE.</p>")

    # ── FEATURE ENGINEERING SUGGESTIONS ──
    add_section("fe", lg["feature_engineering"])
    suggestions = []
    for col in num_cols:
        if train_df[col].nunique() < 5:
            suggestions.append(f"📌 {col}: Only {train_df[col].nunique()} unique values — convert to categorical?")
        if abs(skew(train_df[col].dropna())) > 2 if len(train_df[col].dropna()) > 3 else False:
            suggestions.append(f"📌 {col}: Highly skewed — apply log/sqrt transform")
    for col in cat_cols:
        cl = col.lower()
        if any(w in cl for w in ["date", "time", "dt", "timestamp"]):
            suggestions.append(f"📌 {col}: Extract year/month/day/hour/dayofweek")
        if train_df[col].nunique() > 100:
            suggestions.append(f"📌 {col}: {train_df[col].nunique()} categories — group rare ones")
    # Redundant
    if len(num_cols) >= 2:
        corr_abs = train_df[num_cols].corr().abs()
        for i, c1 in enumerate(num_cols):
            for c2 in num_cols[i + 1:]:
                if corr_abs.loc[c1, c2] > 0.95:
                    suggestions.append(f"📌 {c1} & {c2}: r={corr_abs.loc[c1,c2]:.2f} — drop one (redundant)")
    if suggestions:
        H.append("<ul>")
        for s in suggestions:
            H.append(f"<li>{s}</li>")
        H.append("</ul>")
    else:
        H.append(_success("No immediate feature engineering suggestions."))

    # ── ANALYST NOTES ──
    add_section("notes", lg["analyst_note"])
    if analyst_note:
        H.append(f'<div style="border:2px solid #3498db;padding:15px;border-radius:8px;'
                 f'background:#eaf2f8">{analyst_note}</div>')
    else:
        H.append('<div style="border:2px dashed #bbb;padding:15px;border-radius:8px;'
                 'color:#999">No analyst notes provided.</div>')

    # ── TABLE OF CONTENTS ──
    toc_html = f'<div class="toc"><h2>{lg["toc"]}</h2><ol>'
    for anchor, title in toc_items:
        toc_html += f'<li><a href="#{anchor}">{title}</a></li>'
    toc_html += "</ol></div>"

    # ── ASSEMBLE ──
    body = toc_html + "\n".join(H)
    full_html = html_wrapper(body, title=lg["eda"])

    out_path = os.path.join(outdir, f"eda_report_senior_{lang}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"✅ Senior EDA report generated: {out_path}")
    print(f"   📊 Quality Score: {qscore}/100")
    print(f"   📁 Output directory: {outdir}")
    return out_path


# ──────────────────────────────────────────────
# KULLANIM
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Örnek kullanım:
    train = pd.read_csv("/kaggle/input/competitions/playground-series-s6e3/train.csv")
    test  = pd.read_csv("/kaggle/input/competitions/playground-series-s6e3/test.csv")
    
    eda_report_senior_complete(
        train_df=train,
        test_df=test,
        target_col="target",
        lang="tr",                # veya "en"
        outdir="senior_eda_full",
        analyst_note="Domain kuralları ve leakage kontrolü yapıldı.",
        max_pairplot_cols=6,
        tsne_sample=1500,
    )

    # # --- Demo: rastgele veri ile test ---
    # np.random.seed(42)
    # n = 1000
    # demo_train = pd.DataFrame({
    #     "age": np.random.randint(18, 70, n),
    #     "income": np.random.lognormal(10, 1, n),
    #     "score": np.random.normal(50, 15, n),
    #     "category": np.random.choice(["A", "B", "C", "D"], n),
    #     "city": np.random.choice(["Istanbul", "Ankara", "Izmir", "Bursa", "Antalya"], n),
    #     "target": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    # })
    # demo_train.loc[demo_train.sample(50).index, "income"] = np.nan
    # demo_train.loc[demo_train.sample(30).index, "score"] = np.nan

    # demo_test = pd.DataFrame({
    #     "age": np.random.randint(20, 65, 300),
    #     "income": np.random.lognormal(10.2, 1.1, 300),
    #     "score": np.random.normal(52, 14, 300),
    #     "category": np.random.choice(["A", "B", "C", "D"], 300),
    #     "city": np.random.choice(["Istanbul", "Ankara", "Izmir", "Bursa", "Antalya"], 300),
    # })

    eda_report_senior_complete(
        train_df=train,
        test_df=test,
        target_col="Churn",
        lang="tr",
        outdir="senior_eda",
        analyst_note="Senior EDA.",
    )
```

---

## Entegre Edilen Her Şeyin Özeti

┌───┬────────────────────────────────────┬────────┐
│ # │ Özellik                            │ Durum  │
├───┼────────────────────────────────────┼────────┤
│ 1 │ HTML + CSS + TOC                   │   ✅   │
│ 2 │ Executive Summary (otomatik)       │   ✅   │
│ 3 │ Metric Cards (rows/cols/quality)   │   ✅   │
│ 4 │ Missing Deep (pattern+corr+MCAR)   │   ✅   │
│ 5 │ Imputation önerisi                 │   ✅   │
│ 6 │ Normality (Shapiro+JB+Anderson)    │   ✅   │
│ 7 │ Distribution Fitting (6 dağılım)   │   ✅   │
│ 8 │ Outlier (IQR+Z+LOF+IsoForest)     │   ✅   │
│ 9 │ Near-Zero Variance                 │   ✅   │
│10 │ Duplicate Row Analizi              │   ✅   │
│11 │ Rare Category Detection            │   ✅   │
│12 │ Multi-Correlation (P+S+K+disc.)    │   ✅   │
│13 │ VIF (Multicollinearity)            │   ✅   │
│14 │ Cramér's V (Cat-Cat corr)          │   ✅   │
│15 │ Categorical Entropy                │   ✅   │
│16 │ Confidence Intervals               │   ✅   │
│17 │ Feature Importance (MI)            │   ✅   │
│18 │ Effect Size (Cohen's d)            │   ✅   │
│19 │ Feature Interactions (Polynomial)  │   ✅   │
│20 │ Feature Clustering Dendrogram      │   ✅   │
│21 │ Benford's Law                      │   ✅   │
│22 │ Business Rule Validation           │   ✅   │
│23 │ Target Dist + Imbalance            │   ✅   │
│24 │ Leakage Detection                  │   ✅   │
│25 │ Target-Numeric (ANOVA+corr)        │   ✅   │
│26 │ Target-Categorical (Chi²+crosstab) │   ✅   │
│27 │ Train/Test Drift                   │   ✅   │
│28 │ Sampling Bias (KS-test)            │   ✅   │
│29 │ Encoding önerisi                   │   ✅   │
│30 │ Feature Engineering Suggestions    │   ✅   │
│31 │ Time Series (ADF+trend+season)     │   ✅   │
│32 │ Memory + Dtype Optimization        │   ✅   │
│33 │ Data Quality Score                 │   ✅   │
│34 │ PCA Explained Variance             │   ✅   │
│35 │ t-SNE Visualization                │   ✅   │
│36 │ Box + Violin plots                 │   ✅   │
│37 │ Pairplot (with hue)               │   ✅   │
│38 │ Scatter Matrix (BUG FIX)           │   ✅   │
│39 │ QQ Plots                           │   ✅   │
│40 │ CDF Plots                          │   ✅   │
│41 │ Residual Plots                     │   ✅   │
│42 │ Multimodal Detection (KDE peaks)   │   ✅   │
│43 │ Analyst Notes                      │   ✅   │
│44 │ Bilingual (EN/TR)                  │   ✅   │
│45 │ All original bugs fixed            │   ✅   │
└───┴────────────────────────────────────┴────────┘

Kapsam: ████████████████████ 100%

