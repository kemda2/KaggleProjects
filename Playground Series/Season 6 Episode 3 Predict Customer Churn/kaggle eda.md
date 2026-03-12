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