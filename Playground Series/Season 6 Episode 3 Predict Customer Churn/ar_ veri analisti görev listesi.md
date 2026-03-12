

# Senior Data Analistinin A'dan Z'ye Tüm İş Akışı

---

## 🔷 FAZI 1: PROBLEMİ ANLAMA (Business Understanding)

```
"Veriyi açmadan ÖNCE yapılacak en kritik adım"
```

### 1.1 — İş Problemini Tanımlama
```markdown
Sorulacak sorular:
─────────────────
• Bu projenin iş hedefi ne? (churn azaltma, satış tahmini, segmentasyon...)
• Başarı kriteri ne? (accuracy mi, precision mı, gelir artışı mı?)
• Sonuçlar nasıl kullanılacak? (dashboard, API, rapor, otomatik karar)
• Kim kullanacak? (C-level, pazarlama ekibi, otomasyon sistemi)
• Zaman kısıtı var mı?
• Daha önce bu problem çözülmeye çalışıldı mı? Sonuç ne oldu?
```

### 1.2 — Başarı Metriğini Belirleme
```python
# Örnek: E-ticaret churn problemi
business_goal = "Aylık churn oranını %15'ten %10'a düşürmek"
success_metric = "Precision ≥ 0.80 (yanlış alarm maliyeti yüksek)"
baseline = "Hiçbir şey yapmasak churn oranı %15 → naive baseline"
```

---

## 🔷 FAZ 2: VERİYİ TOPLAMA & İLK TEMAS

### 2.1 — Veri Kaynaklarını Belirleme
```python
# Olası kaynaklar
sources = {
    "SQL DB": "PostgreSQL, MySQL, MSSQL",
    "Data Warehouse": "BigQuery, Redshift, Snowflake",
    "Dosyalar": "CSV, Excel, Parquet, JSON",
    "API": "REST API, GraphQL",
    "Cloud Storage": "S3, GCS, Azure Blob"
}
```

### 2.2 — Veriyi Yükleme
```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─── CSV'den ─────────────────────────────────────
df = pd.read_csv('data.csv', 
                  parse_dates=['date_col'],
                  dtype={'id': str, 'category': 'category'},
                  low_memory=False,
                  encoding='utf-8')

# ─── SQL'den ─────────────────────────────────────
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:pass@host:5432/db')
df = pd.read_sql("""
    SELECT * 
    FROM customers c
    JOIN orders o ON c.id = o.customer_id
    WHERE o.order_date >= '2023-01-01'
""", engine)

# ─── Parquet (büyük veri için) ────────────────────
df = pd.read_parquet('data.parquet', engine='pyarrow')

# ─── Birden fazla dosya ──────────────────────────
import glob
files = glob.glob('data/*.csv')
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
```

### 2.3 — İlk Bakış (First Look)
```python
# ─── Temel Bilgiler ──────────────────────────────
print(f"Shape: {df.shape}")                    # (satır, sütun)
print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

df.head(10)          # İlk 10 satır
df.tail(5)           # Son 5 satır
df.sample(10)        # Rastgele 10 satır (daha güvenilir)
df.info()            # Dtypes + non-null sayıları
df.describe()        # Sayısal istatistikler
df.describe(include='object')  # Kategorik istatistikler
df.dtypes.value_counts()       # Tip dağılımı

# ─── Sütun Listesi ───────────────────────────────
print(df.columns.tolist())

# ─── Unique değerler ─────────────────────────────
for col in df.select_dtypes(include='object').columns:
    print(f"{col}: {df[col].nunique()} unique → {df[col].unique()[:10]}")
```

---

## 🔷 FAZ 3: VERİ KALİTE ANALİZİ (Data Quality Assessment)

### 3.1 — Eksik Veri Analizi
```python
# ─── Eksik Veri Tablosu ──────────────────────────
def missing_report(df):
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    dtypes = df.dtypes
    
    report = pd.DataFrame({
        'Missing': missing,
        'Percent': percent.round(2),
        'Dtype': dtypes
    })
    return report[report['Missing'] > 0].sort_values('Percent', ascending=False)

print(missing_report(df))

# ─── Eksik Veri Paterni (Rastgele mi, sistematik mi?) ──
import missingno as msno
import matplotlib.pyplot as plt

msno.matrix(df, figsize=(12,6))     # Eksiklik matrisi
msno.heatmap(df, figsize=(10,6))    # Eksiklikler arası korelasyon
msno.dendrogram(df, figsize=(10,6)) # Hangileri birlikte eksik
plt.show()
```

### 3.2 — Duplikasyon Kontrolü
```python
# ─── Tam duplikasyon ─────────────────────────────
print(f"Tam duplike satır: {df.duplicated().sum()}")
print(f"Oran: %{df.duplicated().mean()*100:.2f}")

# ─── Belirli sütunlara göre duplikasyon ──────────
key_cols = ['customer_id', 'order_date', 'product_id']
print(f"Key bazlı duplike: {df.duplicated(subset=key_cols).sum()}")

# ─── Duplikeleri incele ──────────────────────────
dupes = df[df.duplicated(subset=key_cols, keep=False)]
dupes.sort_values(key_cols).head(20)
```

### 3.3 — Veri Tipi Uyumsuzlukları
```python
# ─── Yanlış tipteki sütunları tespit et ──────────
# Sayı gibi görünen ama object olan sütunlar
for col in df.select_dtypes(include='object').columns:
    try:
        converted = pd.to_numeric(df[col], errors='coerce')
        valid_pct = converted.notna().mean() * 100
        if valid_pct > 80:
            print(f"⚠️ {col}: %{valid_pct:.0f}'i numeric → tip düzeltilmeli")
    except:
        pass

# ─── Tarih gibi görünen object sütunlar ─────────
for col in df.select_dtypes(include='object').columns:
    sample = df[col].dropna().iloc[:5]
    print(f"{col}: {sample.values}")
```

### 3.4 — Tutarsızlık Kontrolü
```python
# ─── Kategorik değerlerdeki tutarsızlıklar ───────
for col in df.select_dtypes(include='object').columns:
    vals = df[col].unique()
    # Whitespace sorunu
    space_issues = [v for v in vals if isinstance(v, str) and (v != v.strip())]
    # Büyük/küçük harf sorunu
    if len(vals) != len(set(str(v).lower() for v in vals if pd.notna(v))):
        print(f"⚠️ {col}: case tutarsızlığı var")

# ─── Mantıksal tutarsızlıklar ───────────────────
# Örnek: bitiş tarihi < başlangıç tarihi
if 'start_date' in df.columns and 'end_date' in df.columns:
    invalid = df[df['end_date'] < df['start_date']]
    print(f"Tarih tutarsızlığı: {len(invalid)} satır")

# Örnek: Negatif olmaması gereken değerler
if 'price' in df.columns:
    print(f"Negatif fiyat: {(df['price'] < 0).sum()}")
    
# Örnek: Yüzde değerleri 0-100 arasında mı
if 'rate' in df.columns:
    print(f"Aralık dışı oran: {((df['rate']<0)|(df['rate']>100)).sum()}")
```

---

## 🔷 FAZ 4: KEŞİFÇİ VERİ ANALİZİ (EDA)

### 4.1 — Univariate Analiz (Tek Değişken)

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# ─── Sayısal Değişkenler ─────────────────────────
numeric_cols = df.select_dtypes(include=np.number).columns

for i, col in enumerate(numeric_cols[:6]):
    ax = axes[i//3, i%3]
    
    # Histogram + KDE
    sns.histplot(df[col], kde=True, ax=ax, bins=50)
    
    # Ortalama ve medyan çizgileri
    ax.axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean: {df[col].mean():.2f}')
    ax.axvline(df[col].median(), color='green', linestyle='--', label=f'Median: {df[col].median():.2f}')
    ax.legend()
    ax.set_title(f'{col} | Skew: {df[col].skew():.2f} | Kurt: {df[col].kurtosis():.2f}')

plt.tight_layout()
plt.show()

# ─── Detaylı İstatistikler ───────────────────────
for col in numeric_cols:
    print(f"\n{'='*50}")
    print(f"📊 {col}")
    print(f"  Mean:   {df[col].mean():.4f}")
    print(f"  Median: {df[col].median():.4f}")
    print(f"  Std:    {df[col].std():.4f}")
    print(f"  Skew:   {df[col].skew():.4f}")  # |skew| > 1 → ciddi çarpıklık
    print(f"  Kurt:   {df[col].kurtosis():.4f}")
    print(f"  Min:    {df[col].min()}")
    print(f"  Max:    {df[col].max()}")
    print(f"  Q1:     {df[col].quantile(0.25):.4f}")
    print(f"  Q3:     {df[col].quantile(0.75):.4f}")
    print(f"  IQR:    {df[col].quantile(0.75) - df[col].quantile(0.25):.4f}")
```

```python
# ─── Kategorik Değişkenler ───────────────────────
cat_cols = df.select_dtypes(include=['object', 'category']).columns

for col in cat_cols:
    print(f"\n{'='*50}")
    print(f"📊 {col} | Unique: {df[col].nunique()}")
    print(df[col].value_counts().head(15))
    print(f"Top %: {df[col].value_counts(normalize=True).iloc[0]*100:.1f}%")
    
    if df[col].nunique() <= 20:
        fig, ax = plt.subplots(figsize=(10, 4))
        df[col].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(col)
        plt.xticks(rotation=45)
        plt.show()
```

### 4.2 — Bivariate Analiz (İki Değişken)

```python
# ─── Target ile İlişki (Sayısal hedef) ──────────
target = 'churn'  # veya 'revenue', 'price' vs.

# Sayısal vs Target
for col in numeric_cols:
    if col != target:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        # Scatter
        axes[0].scatter(df[col], df[target], alpha=0.3, s=5)
        axes[0].set_xlabel(col)
        axes[0].set_ylabel(target)
        
        # Korelasyon
        corr = df[col].corr(df[target])
        axes[0].set_title(f'{col} vs {target} | r={corr:.3f}')
        
        # Kategorik target ise boxplot
        if df[target].nunique() <= 5:
            sns.boxplot(x=target, y=col, data=df, ax=axes[1])
        
        plt.tight_layout()
        plt.show()

# ─── Kategorik vs Target ────────────────────────
for col in cat_cols:
    if col != target and df[col].nunique() <= 20:
        fig, ax = plt.subplots(figsize=(10, 4))
        if df[target].nunique() <= 5:
            # Kategorik target: grouped bar
            ct = pd.crosstab(df[col], df[target], normalize='index')
            ct.plot(kind='bar', stacked=True, ax=ax)
        else:
            # Sayısal target: box per category
            sns.boxplot(x=col, y=target, data=df, ax=ax)
        plt.title(f'{col} vs {target}')
        plt.xticks(rotation=45)
        plt.show()
```

### 4.3 — Korelasyon Analizi

```python
# ─── Korelasyon Matrisi ──────────────────────────
corr_matrix = df[numeric_cols].corr()

# Heatmap
plt.figure(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# ─── Yüksek Korelasyonlu Çiftler ────────────────
def get_high_corr_pairs(corr_matrix, threshold=0.8):
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                pairs.append({
                    'Feature_1': corr_matrix.columns[i],
                    'Feature_2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
    return pd.DataFrame(pairs).sort_values('Correlation', 
                                            key=abs, ascending=False)

high_corr = get_high_corr_pairs(corr_matrix, 0.8)
print("⚠️ Yüksek korelasyonlu çiftler (multicollinearity riski):")
print(high_corr)

# ─── Target ile korelasyon sıralaması ────────────
target_corr = corr_matrix[target].drop(target).sort_values(key=abs, ascending=False)
print(f"\n{target} ile en yüksek korelasyon:")
print(target_corr)
```

### 4.4 — Multivariate Analiz

```python
# ─── Pair Plot (seçilmiş sütunlar) ──────────────
selected = target_corr.head(5).index.tolist() + [target]
sns.pairplot(df[selected].sample(min(1000, len(df))), 
             hue=target if df[target].nunique()<=5 else None,
             diag_kind='kde', plot_kws={'alpha':0.3, 's':10})
plt.show()

# ─── Pivot / Crosstab Analizleri ─────────────────
pd.crosstab(df['segment'], df['churn'], margins=True, normalize='index')

# ─── Gruplu İstatistikler ────────────────────────
df.groupby('segment').agg({
    'revenue': ['mean', 'median', 'sum', 'count'],
    'tenure': ['mean', 'median'],
    'churn': 'mean'
}).round(2)
```

---

## 🔷 FAZ 5: VERİ TEMİZLEME (Data Cleaning)

### 5.1 — Eksik Veri Tedavisi

```python
# ─── Strateji Seçimi ─────────────────────────────
"""
Eksiklik < %5   → Silme veya basit imputation
%5 - %30        → Model-based imputation
> %30           → Flag oluştur + impute VEYA sütunu sil
> %60           → Büyük ihtimalle sil (iş bilgisine göre)

MCAR (tamamen rastgele) → Herhangi bir yöntem OK
MAR  (koşullu rastgele) → Model-based imputation
MNAR (rastgele değil)   → Domain knowledge gerekir
"""

# ─── Sayısal: Median Imputation (outlier'a dayanıklı) ──
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy='median')
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

# ─── Sayısal: KNN Imputation (daha sofistike) ───
from sklearn.impute import KNNImputer

knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])

# ─── Sayısal: Iterative Imputer (en sofistike) ──
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

iter_imputer = IterativeImputer(max_iter=10, random_state=42)
df[numeric_cols] = iter_imputer.fit_transform(df[numeric_cols])

# ─── Kategorik: Mode veya 'Unknown' ─────────────
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        if df[col].isnull().mean() < 0.05:
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna('Unknown', inplace=True)

# ─── Eksiklik Flag'i oluştur (bilgi kaybı önleme) ──
for col in df.columns:
    if df[col].isnull().any():
        df[f'{col}_is_missing'] = df[col].isnull().astype(int)
```

### 5.2 — Outlier Tedavisi

```python
# ─── Tespit: IQR Yöntemi ────────────────────────
def detect_outliers_iqr(df, col, multiplier=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    return outliers, lower, upper

# ─── Tespit: Z-Score Yöntemi ────────────────────
from scipy import stats

def detect_outliers_zscore(df, col, threshold=3):
    z_scores = np.abs(stats.zscore(df[col].dropna()))
    return df[z_scores > threshold]

# ─── Tüm sayısal sütunlar için outlier raporu ───
print(f"{'Sütun':<25} {'Outlier #':<12} {'Oran %':<10} {'Alt Sınır':<15} {'Üst Sınır'}")
print("="*80)
for col in numeric_cols:
    outliers, lower, upper = detect_outliers_iqr(df, col)
    pct = len(outliers) / len(df) * 100
    if pct > 0:
        print(f"{col:<25} {len(outliers):<12} {pct:<10.2f} {lower:<15.2f} {upper:.2f}")

# ─── Tedavi: Capping / Winsorization ────────────
def cap_outliers(df, col, multiplier=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    df[col] = df[col].clip(lower=lower, upper=upper)
    return df

# ─── Tedavi: Log Transform (sağa çarpık veriler) ──
for col in numeric_cols:
    if df[col].skew() > 1 and (df[col] > 0).all():
        df[f'{col}_log'] = np.log1p(df[col])

# ─── Tedavi: Percentile Capping ─────────────────
for col in numeric_cols:
    lower_cap = df[col].quantile(0.01)
    upper_cap = df[col].quantile(0.99)
    df[col] = df[col].clip(lower_cap, upper_cap)
```

### 5.3 — Veri Tipi Düzeltmeleri
```python
# ─── Tarih Dönüşümleri ──────────────────────────
date_cols = ['created_at', 'updated_at', 'birth_date']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# ─── Numerik Dönüşüm ────────────────────────────
df['price'] = pd.to_numeric(df['price'].str.replace(',', '.').str.replace('₺', ''), errors='coerce')

# ─── Kategorik Optimizasyon ──────────────────────
for col in df.select_dtypes(include='object').columns:
    if df[col].nunique() / len(df) < 0.05:  # Kardinalite < %5
        df[col] = df[col].astype('category')

# ─── Memory Optimizasyonu ────────────────────────
def optimize_dtypes(df):
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].min() >= 0:
            if df[col].max() < 255: df[col] = df[col].astype(np.uint8)
            elif df[col].max() < 65535: df[col] = df[col].astype(np.uint16)
            elif df[col].max() < 4294967295: df[col] = df[col].astype(np.uint32)
        else:
            if df[col].max() < 127: df[col] = df[col].astype(np.int8)
            elif df[col].max() < 32767: df[col] = df[col].astype(np.int16)
            elif df[col].max() < 2147483647: df[col] = df[col].astype(np.int32)
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)
    
    return df

before = df.memory_usage(deep=True).sum() / 1e6
df = optimize_dtypes(df)
after = df.memory_usage(deep=True).sum() / 1e6
print(f"Memory: {before:.1f}MB → {after:.1f}MB ({(1-after/before)*100:.0f}% azalma)")
```

### 5.4 — Text / String Temizleme
```python
# ─── Genel String Temizleme ─────────────────────
for col in df.select_dtypes(include='object').columns:
    df[col] = (df[col]
               .str.strip()                    # Baş/son boşluk
               .str.lower()                    # Küçük harfe
               .str.replace(r'\s+', ' ', regex=True)  # Çoklu boşluk
               .str.replace(r'[^\w\s]', '', regex=True))  # Özel karakter

# ─── Spesifik Düzeltmeler ───────────────────────
df['city'] = df['city'].replace({
    'istanbul': 'istanbul',
    'ist': 'istanbul',
    'İstanbul': 'istanbul',
    'ist.': 'istanbul'
})
```

---

## 🔷 FAZ 6: FEATURE ENGINEERING

### 6.1 — Tarih Bazlı Özellikler
```python
# ─── Tarihten Türetme ────────────────────────────
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek  # 0=Pazartesi
df['day_name'] = df['date'].dt.day_name()
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['quarter'] = df['date'].dt.quarter
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
df['hour'] = df['datetime'].dt.hour
df['part_of_day'] = pd.cut(df['hour'], bins=[0,6,12,18,24], 
                            labels=['night','morning','afternoon','evening'])

# ─── Fark / Yaş Hesaplama ───────────────────────
df['account_age_days'] = (pd.Timestamp.now() - df['created_at']).dt.days
df['days_since_last_order'] = (pd.Timestamp.now() - df['last_order_date']).dt.days
df['order_delivery_days'] = (df['delivery_date'] - df['order_date']).dt.days

# ─── Cyclical Encoding (sin/cos) ────────────────
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

### 6.2 — Aggregation / Grup Bazlı Özellikler
```python
# ─── Müşteri Bazlı Aggregation'lar ──────────────
customer_agg = df.groupby('customer_id').agg(
    total_orders=('order_id', 'nunique'),
    total_revenue=('revenue', 'sum'),
    avg_order_value=('revenue', 'mean'),
    median_order_value=('revenue', 'median'),
    std_order_value=('revenue', 'std'),
    min_order_value=('revenue', 'min'),
    max_order_value=('revenue', 'max'),
    total_items=('quantity', 'sum'),
    unique_products=('product_id', 'nunique'),
    unique_categories=('category', 'nunique'),
    first_order=('order_date', 'min'),
    last_order=('order_date', 'max'),
    avg_days_between_orders=('order_date', lambda x: x.sort_values().diff().dt.days.mean())
).reset_index()

# ─── Recency, Frequency, Monetary (RFM) ─────────
snapshot_date = df['order_date'].max() + pd.Timedelta(days=1)
rfm = df.groupby('customer_id').agg(
    recency=('order_date', lambda x: (snapshot_date - x.max()).days),
    frequency=('order_id', 'nunique'),
    monetary=('revenue', 'sum')
).reset_index()

# ─── Window / Rolling Features ──────────────────
df = df.sort_values(['customer_id', 'order_date'])
df['rolling_3_avg_revenue'] = df.groupby('customer_id')['revenue'].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)
df['cumulative_revenue'] = df.groupby('customer_id')['revenue'].cumsum()
df['order_rank'] = df.groupby('customer_id')['order_date'].rank()

# ─── Lag Features ────────────────────────────────
df['prev_order_revenue'] = df.groupby('customer_id')['revenue'].shift(1)
df['revenue_change'] = df['revenue'] - df['prev_order_revenue']
df['revenue_pct_change'] = df['revenue'].pct_change()
```

### 6.3 — Etkileşim / Oran Özellikleri
```python
# ─── Oranlar ─────────────────────────────────────
df['revenue_per_item'] = df['revenue'] / (df['quantity'] + 1)
df['discount_rate'] = df['discount'] / (df['original_price'] + 1)
df['return_rate'] = df['returns'] / (df['total_orders'] + 1)
df['conversion_rate'] = df['purchases'] / (df['visits'] + 1)

# ─── Etkileşimler ───────────────────────────────
df['age_x_income'] = df['age'] * df['income']
df['price_per_sqm'] = df['price'] / (df['area'] + 1)

# ─── Polynomial Features (dikkatli kullan) ──────
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
selected_feats = ['age', 'income', 'tenure']
poly_features = poly.fit_transform(df[selected_feats])
poly_names = poly.get_feature_names_out(selected_feats)
df_poly = pd.DataFrame(poly_features, columns=poly_names)
```

### 6.4 — Encoding (Kategorik → Sayısal)
```python
# ─── Label Encoding (ordinal veriler için) ───────
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Sıralı kategoriler
education_order = ['ilkokul', 'lise', 'lisans', 'yuksek_lisans', 'doktora']
df['education_encoded'] = OrdinalEncoder(
    categories=[education_order]
).fit_transform(df[['education']])

# ─── One-Hot Encoding (nominal, düşük kardinalite) ──
df = pd.get_dummies(df, columns=['gender', 'city_tier'], drop_first=True)

# ─── Target Encoding (yüksek kardinalite) ───────
# Dikkat: Data leakage riski → mutlaka CV ile
from category_encoders import TargetEncoder
te = TargetEncoder(cols=['city', 'product_category'], smoothing=10)
df['city_target_enc'] = te.fit_transform(df['city'], df[target])

# ─── Frequency Encoding ─────────────────────────
for col in ['city', 'product_id']:
    freq = df[col].value_counts(normalize=True)
    df[f'{col}_freq'] = df[col].map(freq)

# ─── Binary Encoding (orta kardinalite) ─────────
from category_encoders import BinaryEncoder
be = BinaryEncoder(cols=['zip_code'])
df = be.fit_transform(df)
```

### 6.5 — Binning / Discretization
```python
# ─── Eşit Aralıklı ──────────────────────────────
df['age_bin'] = pd.cut(df['age'], bins=5, labels=['very_young','young','mid','senior','old'])

# ─── Eşit Frekanslı (Quantile) ──────────────────
df['income_quantile'] = pd.qcut(df['income'], q=4, labels=['Q1','Q2','Q3','Q4'])

# ─── Domain Knowledge Bazlı ─────────────────────
df['age_group'] = pd.cut(df['age'], 
                          bins=[0, 18, 25, 35, 50, 65, 100],
                          labels=['child','young_adult','adult','middle_age','senior','elderly'])
```

---

## 🔷 FAZ 7: FEATURE SELECTION

```python
# ─── 1. Variance Threshold ──────────────────────
from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(threshold=0.01)  # Çok düşük varyans → bilgi yok
vt.fit(df[numeric_cols])
low_var = numeric_cols[~vt.get_support()]
print(f"Düşük varyanslı (silinecek): {low_var.tolist()}")

# ─── 2. Korelasyon Bazlı ────────────────────────
def remove_collinear(df, threshold=0.90):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return to_drop

drop_cols = remove_collinear(df[numeric_cols], 0.90)
print(f"Yüksek korelasyonlu (silinecek): {drop_cols}")

# ─── 3. Mutual Information ──────────────────────
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

mi = mutual_info_classif(df[feature_cols], df[target], random_state=42)
mi_df = pd.DataFrame({'feature': feature_cols, 'MI': mi}).sort_values('MI', ascending=False)
print(mi_df.head(20))

# ─── 4. Feature Importance (Tree-based) ─────────
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 12))
sns.barplot(data=importances.head(30), x='importance', y='feature')
plt.title('Feature Importance (Random Forest)')
plt.show()

# ─── 5. Permutation Importance ──────────────────
from sklearn.inspection import permutation_importance

perm_imp = permutation_importance(rf, X_val, y_val, n_repeats=10, random_state=42)
perm_df = pd.DataFrame({
    'feature': feature_cols,
    'importance_mean': perm_imp.importances_mean,
    'importance_std': perm_imp.importances_std
}).sort_values('importance_mean', ascending=False)

# ─── 6. SHAP ────────────────────────────────────
import shap
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values[1], X_val, max_display=20)

# ─── 7. Boruta (wrapper method) ─────────────────
from boruta import BorutaPy
boruta = BorutaPy(rf, n_estimators='auto', random_state=42)
boruta.fit(X_train.values, y_train.values)
selected = X_train.columns[boruta.support_].tolist()
print(f"Boruta seçilen: {selected}")

# ─── 8. Recursive Feature Elimination ───────────
from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator=rf, step=1, cv=5, scoring='roc_auc', n_jobs=-1)
rfecv.fit(X_train, y_train)
print(f"Optimal feature sayısı: {rfecv.n_features_}")
selected = X_train.columns[rfecv.support_].tolist()
```

---

## 🔷 FAZ 8: VERİ HAZIRLAMA (Preprocessing Pipeline)

### 8.1 — Train/Test Split

```python
from sklearn.model_selection import train_test_split

# ─── Hedef Değişken Ayırma ───────────────────────
X = df.drop(columns=[target, 'id', 'date'])  # Non-feature sütunları çıkar
y = df[target]

# ─── İmbalanced data kontrolü ───────────────────
print(f"Target Dağılımı:\n{y.value_counts(normalize=True)}")
# Eğer dengesiz → stratify kullan

# ─── Split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Validasyon seti de ayır
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Train target: {y_train.value_counts(normalize=True).to_dict()}")

# ─── Time-series ise: Temporal Split ────────────
# ASLA shuffle yapma, zaman sırası koru
df = df.sort_values('date')
train_cutoff = '2023-06-01'
val_cutoff = '2023-09-01'

train = df[df['date'] < train_cutoff]
val = df[(df['date'] >= train_cutoff) & (df['date'] < val_cutoff)]
test = df[df['date'] >= val_cutoff]
```

### 8.2 — Preprocessing Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, 
                                     RobustScaler, PowerTransformer,
                                     OneHotEncoder, OrdinalEncoder)
from sklearn.impute import SimpleImputer, KNNImputer

# ─── Sütun Grupları ──────────────────────────────
numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# ─── Numeric Pipeline ────────────────────────────
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler()),  # Outlier'a dayanıklı
    # Alternatifler:
    # ('scaler', StandardScaler()),     # Normal dağılım varsayımı
    # ('scaler', MinMaxScaler()),       # 0-1 aralığına
    # ('power', PowerTransformer()),    # Gaussian'a yakınlaştır
])

# ─── Categorical Pipeline ────────────────────────
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
])

# ─── Birleştir ───────────────────────────────────
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
], remainder='drop')

# ─── Fit & Transform ────────────────────────────
X_train_processed = preprocessor.fit_transform(X_train)   # FIT sadece train'de!
X_val_processed = preprocessor.transform(X_val)            # Transform only
X_test_processed = preprocessor.transform(X_test)          # Transform only

print(f"Processed shape: {X_train_processed.shape}")

# Feature isimlerini al
feature_names = preprocessor.get_feature_names_out()
```

### 8.3 — Dengesiz Veri Tedavisi (Imbalanced Data)

```python
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

# ─── SMOTE ───────────────────────────────────────
smote = SMOTE(random_state=42, sampling_strategy=0.8)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
print(f"SMOTE sonrası: {pd.Series(y_train_resampled).value_counts()}")

# ─── Class Weights (model içinde) ───────────────
# Örnekleme yerine modelde ağırlık ver → daha güvenli
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weight_dict = dict(zip(np.unique(y_train), class_weights))
print(f"Class weights: {weight_dict}")
# Model'de: class_weight='balanced' veya class_weight=weight_dict
```

---

## 🔷 FAZ 9: MODELLEME

### 9.1 — Baseline Model
```python
from sklearn.dummy import DummyClassifier, DummyRegressor

# ─── Classification Baseline ────────────────────
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train_processed, y_train)
print(f"Baseline Accuracy: {dummy.score(X_val_processed, y_val):.4f}")

# ─── Regression Baseline ────────────────────────
dummy_reg = DummyRegressor(strategy='mean')
dummy_reg.fit(X_train_processed, y_train)
print(f"Baseline RMSE: {np.sqrt(mean_squared_error(y_val, dummy_reg.predict(X_val_processed))):.4f}")
```

### 9.2 — Birden Fazla Model Deneme

```python
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                               f1_score, roc_auc_score, classification_report,
                               confusion_matrix, mean_squared_error, r2_score,
                               mean_absolute_error)
import time

# ─── Model Sözlüğü ──────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', 
                                             random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=200, use_label_encoder=False, 
                              eval_metric='logloss', random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=200, class_weight='balanced', 
                                 random_state=42, verbose=-1),
    'CatBoost': CatBoostClassifier(iterations=200, random_seed=42, verbose=0),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(probability=True, class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
}

# ─── Hepsini Dene ────────────────────────────────
results = []
for name, model in models.items():
    start = time.time()
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_val_processed)
    y_proba = model.predict_proba(X_val_processed)[:, 1] if hasattr(model, 'predict_proba') else None
    elapsed = time.time() - start
    
    result = {
        'Model': name,
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred, average='weighted'),
        'Recall': recall_score(y_val, y_pred, average='weighted'),
        'F1': f1_score(y_val, y_pred, average='weighted'),
        'ROC_AUC': roc_auc_score(y_val, y_proba) if y_proba is not None else None,
        'Time (s)': round(elapsed, 2)
    }
    results.append(result)
    print(f"✅ {name}: F1={result['F1']:.4f}, AUC={result['ROC_AUC']:.4f}, Time={elapsed:.1f}s")

results_df = pd.DataFrame(results).sort_values('ROC_AUC', ascending=False)
print("\n📊 Model Karşılaştırma:")
print(results_df.to_string(index=False))
```

### 9.3 — Cross Validation
```python
from sklearn.model_selection import (cross_val_score, StratifiedKFold, 
                                       KFold, TimeSeriesSplit)

# ─── Stratified K-Fold (classification) ─────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# En iyi 3 model için CV
best_models = ['LightGBM', 'XGBoost', 'Random Forest']
for name in best_models:
    model = models[name]
    scores = cross_val_score(model, X_train_processed, y_train, 
                              cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"{name}: AUC = {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Folds: {scores.round(4)}")

# ─── Time Series CV ─────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X_train_processed):
    X_tr, X_vl = X_train_processed[train_idx], X_train_processed[val_idx]
    y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]
```

---

## 🔷 FAZ 10: HİPERPARAMETRE OPTİMİZASYONU

```python
# ─── Optuna (Modern, Bayesian Optimization) ─────
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1, log=True),
    }
    
    model = XGBClassifier(**params, use_label_encoder=False, 
                           eval_metric='logloss', random_state=42, n_jobs=-1)
    
    scores = cross_val_score(model, X_train_processed, y_train, 
                              cv=cv, scoring='roc_auc', n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"\nBest AUC: {study.best_value:.4f}")
print(f"Best Params: {study.best_params}")

# ─── LightGBM için Optuna ───────────────────────
def lgbm_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
    }
    
    model = LGBMClassifier(**params, class_weight='balanced', 
                            random_state=42, verbose=-1, n_jobs=-1)
    scores = cross_val_score(model, X_train_processed, y_train, 
                              cv=cv, scoring='roc_auc', n_jobs=-1)
    return scores.mean()

# ─── GridSearchCV (Kapsamlı ama yavaş) ──────────
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1]
}

grid = GridSearchCV(XGBClassifier(random_state=42), param_grid, 
                     cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
grid.fit(X_train_processed, y_train)
print(f"Best: {grid.best_score_:.4f} → {grid.best_params_}")
```

---

## 🔷 FAZ 11: MODEL DEĞERLENDİRME (Final)

### 11.1 — Classification Metrikleri
```python
# ─── Final Model ─────────────────────────────────
best_model = LGBMClassifier(**study.best_params, class_weight='balanced',
                              random_state=42, verbose=-1)
best_model.fit(X_train_processed, y_train)

y_pred = best_model.predict(X_test_processed)
y_proba = best_model.predict_proba(X_test_processed)[:, 1]

# ─── Classification Report ──────────────────────
print(classification_report(y_test, y_pred, digits=4))

# ─── Confusion Matrix ───────────────────────────
from sklearn.metrics import ConfusionMatrixDisplay
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(best_model, X_test_processed, y_test, 
                                       ax=ax, cmap='Blues', normalize='true')
plt.title('Normalized Confusion Matrix')
plt.show()

# ─── ROC Curve ───────────────────────────────────
from sklearn.metrics import RocCurveDisplay
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_estimator(best_model, X_test_processed, y_test, ax=ax)
ax.plot([0,1], [0,1], 'k--', label='Random (AUC=0.5)')
plt.title(f'ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.4f})')
plt.legend()
plt.show()

# ─── Precision-Recall Curve (imbalanced data için daha önemli) ──
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score
fig, ax = plt.subplots(figsize=(8, 6))
PrecisionRecallDisplay.from_estimator(best_model, X_test_processed, y_test, ax=ax)
ap = average_precision_score(y_test, y_proba)
plt.title(f'PR Curve (AP = {ap:.4f})')
plt.show()

# ─── Optimal Threshold Bulma ────────────────────
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold: {optimal_threshold:.4f}")
print(f"Bu threshold'da → Precision: {precisions[optimal_idx]:.4f}, "
      f"Recall: {recalls[optimal_idx]:.4f}, F1: {f1_scores[optimal_idx]:.4f}")

# Custom threshold ile tahmin
y_pred_custom = (y_proba >= optimal_threshold).astype(int)
print(f"\nCustom threshold sonuçları:")
print(classification_report(y_test, y_pred_custom, digits=4))

# ─── Calibration (Tahmin güvenilirliği) ─────────
from sklearn.calibration import CalibrationDisplay
fig, ax = plt.subplots(figsize=(8, 6))
CalibrationDisplay.from_estimator(best_model, X_test_processed, y_test, 
                                   n_bins=10, ax=ax)
plt.title('Calibration Curve')
plt.show()
```

### 11.2 — Regression Metrikleri
```python
# Regression projesi ise:
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                               r2_score, mean_absolute_percentage_error)

y_pred = model.predict(X_test_processed)

print(f"RMSE:  {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE:   {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MAPE:  {mean_absolute_percentage_error(y_test, y_pred)*100:.2f}%")
print(f"R²:    {r2_score(y_test, y_pred):.4f}")

# Residual analizi
residuals = y_test - y_pred
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].scatter(y_pred, residuals, alpha=0.3, s=5)
axes[0].axhline(0, color='r', linestyle='--')
axes[0].set_title('Residuals vs Predicted')
axes[1].hist(residuals, bins=50, edgecolor='black')
axes[1].set_title('Residual Distribution')
stats.probplot(residuals, plot=axes[2])
axes[2].set_title('Q-Q Plot')
plt.tight_layout()
plt.show()
```

### 11.3 — Model Yorumlama (Explainability)
```python
import shap

# ─── SHAP (en kapsamlı yorumlama) ───────────────
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_processed)

# Global Feature Importance
shap.summary_plot(shap_values, X_test_processed, 
                   feature_names=feature_names, max_display=20)

# Bar plot
shap.summary_plot(shap_values, X_test_processed, 
                   feature_names=feature_names, plot_type='bar', max_display=20)

# Tek bir tahmin açıklama (Local Explanation)
idx = 0  # İlk test örneği
shap.force_plot(explainer.expected_value, shap_values[idx], 
                X_test_processed[idx], feature_names=feature_names)

# Waterfall plot
shap.waterfall_plot(shap.Explanation(values=shap_values[idx],
                                      base_values=explainer.expected_value,
                                      feature_names=feature_names))

# Dependence plot (bir feature'ın etkisi)
shap.dependence_plot('feature_name', shap_values, X_test_processed,
                      feature_names=feature_names)

# ─── LIME (alternatif) ──────────────────────────
import lime.lime_tabular

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train_processed, feature_names=feature_names,
    class_names=['No Churn', 'Churn'], mode='classification'
)
exp = lime_explainer.explain_instance(X_test_processed[0], best_model.predict_proba)
exp.show_in_notebook()

# ─── Partial Dependence Plots ───────────────────
from sklearn.inspection import PartialDependenceDisplay
fig, ax = plt.subplots(figsize=(12, 6))
PartialDependenceDisplay.from_estimator(best_model, X_test_processed, 
                                         features=[0, 1, 2, (0, 1)],
                                         feature_names=feature_names, ax=ax)
plt.tight_layout()
plt.show()
```

---

## 🔷 FAZ 12: MODEL KAYDETME & PIPELINE

```python
import joblib
import pickle

# ─── Tam Pipeline (Preprocessing + Model) ───────
from sklearn.pipeline import Pipeline

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', best_model)
])

# Train pipeline
full_pipeline.fit(X_train, y_train)

# ─── Kaydetme ────────────────────────────────────
joblib.dump(full_pipeline, 'models/churn_pipeline_v1.joblib')
joblib.dump(optimal_threshold, 'models/optimal_threshold.joblib')

# ─── Yükleme & Test ─────────────────────────────
loaded_pipeline = joblib.load('models/churn_pipeline_v1.joblib')
loaded_threshold = joblib.load('models/optimal_threshold.joblib')

# Test
y_proba_loaded = loaded_pipeline.predict_proba(X_test)[:, 1]
y_pred_loaded = (y_proba_loaded >= loaded_threshold).astype(int)
print(f"Loaded model AUC: {roc_auc_score(y_test, y_proba_loaded):.4f}")

# ─── Model Metadata ─────────────────────────────
import json
from datetime import datetime

metadata = {
    'model_name': 'churn_prediction_v1',
    'model_type': 'LGBMClassifier',
    'created_at': datetime.now().isoformat(),
    'features': feature_names.tolist(),
    'hyperparameters': study.best_params,
    'metrics': {
        'auc': roc_auc_score(y_test, y_proba_loaded),
        'f1': f1_score(y_test, y_pred_loaded),
        'precision': precision_score(y_test, y_pred_loaded),
        'recall': recall_score(y_test, y_pred_loaded)
    },
    'threshold': float(optimal_threshold),
    'training_data_shape': list(X_train.shape),
    'training_date_range': f"{df['date'].min()} - {df['date'].max()}"
}

with open('models/metadata_v1.json', 'w') as f:
    json.dump(metadata, f, indent=2, default=str)
```

---

## 🔷 FAZ 13: API & DEPLOYMENT

### 13.1 — FastAPI ile REST API

```python
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Churn Prediction API", version="1.0")

# Model yükle
pipeline = joblib.load('models/churn_pipeline_v1.joblib')
threshold = joblib.load('models/optimal_threshold.joblib')

# ─── Request Schema ──────────────────────────────
class CustomerData(BaseModel):
    age: int
    tenure_months: int
    monthly_charges: float
    total_charges: float
    contract_type: str
    payment_method: str
    # ... diğer alanlar
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "tenure_months": 24,
                "monthly_charges": 79.99,
                "total_charges": 1919.76,
                "contract_type": "month-to-month",
                "payment_method": "electronic_check"
            }
        }

# ─── Response Schema ─────────────────────────────
class PredictionResponse(BaseModel):
    customer_id: str = None
    churn_probability: float
    churn_prediction: int
    risk_level: str
    threshold_used: float

# ─── Endpoints ───────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_version": "v1"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    try:
        df = pd.DataFrame([data.dict()])
        proba = pipeline.predict_proba(df)[:, 1][0]
        prediction = int(proba >= threshold)
        
        # Risk seviyesi
        if proba < 0.3: risk = "LOW"
        elif proba < 0.6: risk = "MEDIUM"
        elif proba < 0.8: risk = "HIGH"
        else: risk = "CRITICAL"
        
        return PredictionResponse(
            churn_probability=round(float(proba), 4),
            churn_prediction=prediction,
            risk_level=risk,
            threshold_used=float(threshold)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch")
def predict_batch(data: list[CustomerData]):
    df = pd.DataFrame([d.dict() for d in data])
    probas = pipeline.predict_proba(df)[:, 1]
    predictions = (probas >= threshold).astype(int)
    
    return {
        "predictions": predictions.tolist(),
        "probabilities": probas.round(4).tolist(),
        "count": len(data)
    }

# Çalıştır: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 13.2 — Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/churn_pipeline_v1.joblib
    restart: always
    
  monitoring:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

```bash
# Build & Run
docker build -t churn-api .
docker run -p 8000:8000 churn-api

# Docker Compose
docker-compose up -d
```

### 13.3 — Cloud Deployment

```python
# ─── AWS SageMaker ───────────────────────────────
import sagemaker
from sagemaker.sklearn import SKLearnModel

model = SKLearnModel(
    model_data='s3://bucket/models/model.tar.gz',
    role='arn:aws:iam::role/SageMakerRole',
    framework_version='1.2-1',
    entry_point='inference.py'
)
predictor = model.deploy(instance_type='ml.t2.medium', initial_instance_count=1)

# ─── GCP Vertex AI ──────────────────────────────
from google.cloud import aiplatform
aiplatform.init(project='my-project', location='us-central1')
model = aiplatform.Model.upload(
    display_name='churn-model',
    artifact_uri='gs://bucket/models/',
    serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest'
)
endpoint = model.deploy(machine_type='n1-standard-4')

# ─── MLflow (Model Registry) ────────────────────
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("churn_prediction")

with mlflow.start_run(run_name="lgbm_v1"):
    mlflow.log_params(study.best_params)
    mlflow.log_metrics({
        'auc': roc_auc_score(y_test, y_proba),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    })
    mlflow.sklearn.log_model(full_pipeline, "model",
                              registered_model_name="churn_model")
    mlflow.log_artifact('models/metadata_v1.json')
```

---

## 🔷 FAZ 14: MONİTÖRLEME & BAKIM

### 14.1 — Model Performance Monitoring

```python
# ─── Data Drift Detection ───────────────────────
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=X_train, current_data=X_new_production)
drift_report.save_html("reports/drift_report.html")

# ─── Manuel PSI (Population Stability Index) ────
def calculate_psi(expected, actual, bins=10):
    """PSI < 0.1: Stabil, 0.1-0.25: Dikkat, >0.25: Ciddi drift"""
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    expected_perc = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_perc = np.histogram(actual, breakpoints)[0] / len(actual)
    
    expected_perc = np.clip(expected_perc, 0.001, None)
    actual_perc = np.clip(actual_perc, 0.001, None)
    
    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return psi

# Her feature için PSI hesapla
for col in numeric_features:
    psi = calculate_psi(X_train[col].values, X_new[col].values)
    status = "✅" if psi < 0.1 else "⚠️" if psi < 0.25 else "🚨"
    print(f"{status} {col}: PSI = {psi:.4f}")

# ─── Prediction Drift ───────────────────────────
# Tahmin dağılımının zamanla değişimi
daily_predictions = df_production.groupby('date').agg(
    avg_proba=('churn_probability', 'mean'),
    median_proba=('churn_probability', 'median'),
    positive_rate=('churn_prediction', 'mean'),
    count=('churn_prediction', 'count')
).reset_index()

# Alert: positive_rate çok değiştiyse
if abs(daily_predictions['positive_rate'].iloc[-1] - baseline_positive_rate) > 0.05:
    send_alert("Prediction distribution shifted significantly!")
```

### 14.2 — Logging & Alerting

```python
import logging
from datetime import datetime

# ─── Logging Setup ───────────────────────────────
logging.basicConfig(
    filename='logs/prediction.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

def log_prediction(input_data, prediction, proba, latency):
    logging.info(f"INPUT: {input_data} | PRED: {prediction} | "
                 f"PROBA: {proba:.4f} | LATENCY: {latency:.3f}s")

# ─── Scheduled Retraining Trigger ────────────────
def check_retrain_needed():
    conditions = {
        'psi_threshold': max_psi > 0.25,
        'performance_drop': current_auc < baseline_auc - 0.05,
        'time_elapsed': (datetime.now() - last_train_date).days > 30,
        'data_volume': new_data_count > 10000
    }
    
    if any(conditions.values()):
        triggered = [k for k, v in conditions.items() if v]
        print(f"🔄 RETRAIN NEEDED! Triggers: {triggered}")
        return True
    return False
```

### 14.3 — A/B Testing
```python
# ─── Model A/B Test ──────────────────────────────
import hashlib

def get_model_variant(customer_id, split_ratio=0.5):
    """Deterministic A/B split"""
    hash_val = int(hashlib.md5(str(customer_id).encode()).hexdigest(), 16)
    return 'model_v2' if (hash_val % 100) < (split_ratio * 100) else 'model_v1'

# API'de
@app.post("/predict_ab")
def predict_ab(data: CustomerData, customer_id: str):
    variant = get_model_variant(customer_id)
    
    if variant == 'model_v2':
        pipeline = pipeline_v2
    else:
        pipeline = pipeline_v1
    
    proba = pipeline.predict_proba(pd.DataFrame([data.dict()]))[:, 1][0]
    
    # Log for analysis
    log_ab_test(customer_id, variant, proba)
    
    return {"variant": variant, "probability": proba}
```

---

## 🔷 FAZ 15: DOKÜMANTASYON & RAPORLAMA

```markdown
## Proje Dokümantasyon Yapısı
```

```
project/
├── README.md                    # Genel proje açıklaması
├── requirements.txt             # Python bağımlılıkları
├── setup.py                     # Paket kurulumu
├── .env                         # Environment variables
├── .gitignore
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Keşifçi analiz
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Modeling.ipynb
│   └── 04_Evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── load.py             # Veri yükleme
│   │   └── clean.py            # Veri temizleme
│   ├── features/
│   │   ├── build.py            # Feature engineering
│   │   └── select.py           # Feature selection
│   ├── models/
│   │   ├── train.py            # Model eğitimi
│   │   ├── predict.py          # Tahmin
│   │   └── evaluate.py         # Değerlendirme
│   └── utils/
│       ├── config.py           # Konfigürasyonlar
│       └── helpers.py          # Yardımcı fonksiyonlar
│
├── models/
│   ├── churn_pipeline_v1.joblib
│   ├── metadata_v1.json
│   └── optimal_threshold.joblib
│
├── data/
│   ├── raw/                    # Ham veri (dokunulmaz)
│   ├── processed/              # İşlenmiş veri
│   └── external/               # Dış kaynak veri
│
├── reports/
│   ├── figures/                # Grafikler
│   ├── drift_report.html       # Drift raporu
│   └── final_report.pdf        # Final rapor
│
├── api/
│   ├── app.py                  # FastAPI uygulaması
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_model.py
│
└── configs/
    ├── model_config.yaml
    └── feature_config.yaml
```

### Business Rapor Template
```python
# ─── Otomatik Rapor Oluşturma ────────────────────
from fpdf import FPDF

class ModelReport(FPDF):
    def create_report(self):
        self.add_page()
        self.set_font('Arial', 'B', 20)
        self.cell(0, 10, 'Churn Prediction Model Report', ln=True, align='C')
        
        self.set_font('Arial', '', 12)
        self.cell(0, 10, f'Date: {datetime.now().strftime("%Y-%m-%d")}', ln=True)
        self.cell(0, 10, f'Model: LightGBM v1', ln=True)
        self.cell(0, 10, f'AUC: {auc:.4f}', ln=True)
        self.cell(0, 10, f'F1 Score: {f1:.4f}', ln=True)
        
        # İş etkisi
        self.ln(10)
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Business Impact', ln=True)
        self.set_font('Arial', '', 12)
        
        at_risk = (y_proba >= threshold).sum()
        revenue_at_risk = df_test.loc[y_proba >= threshold, 'monthly_revenue'].sum()
        self.cell(0, 10, f'At-risk customers: {at_risk}', ln=True)
        self.cell(0, 10, f'Monthly revenue at risk: ${revenue_at_risk:,.2f}', ln=True)
        
        # Grafikleri ekle
        self.image('reports/figures/confusion_matrix.png', w=150)
        self.image('reports/figures/roc_curve.png', w=150)
        self.image('reports/figures/shap_summary.png', w=150)

report = ModelReport()
report.create_report()
report.output('reports/final_report.pdf')
```

---

## 🔷 ÖZET: TÜM AKIŞ TEK TABLODA

```
┌────────────────────────────────────────────────────────────────┐
│                    SENIOR DATA ANALYST WORKFLOW                 │
├────────┬───────────────────────┬───────────────────────────────┤
│  FAZ   │      ADIM             │    ARAÇLAR                    │
├────────┼───────────────────────┼───────────────────────────────┤
│   1    │ Problem Tanımlama     │ Stakeholder görüşmeleri       │
│   2    │ Veri Toplama          │ SQL, API, Pandas              │
│   3    │ Veri Kalite Kontrolü  │ missingno, pandas profiling   │
│   4    │ EDA                   │ matplotlib, seaborn, plotly   │
│   5    │ Veri Temizleme        │ pandas, sklearn imputers      │
│   6    │ Feature Engineering   │ pandas, domain knowledge      │
│   7    │ Feature Selection     │ SHAP, Boruta, MI, RFE         │
│   8    │ Preprocessing         │ sklearn Pipeline              │
│   9    │ Modelleme             │ sklearn, xgb, lgbm, catboost  │
│  10    │ Hyperparameter Tuning │ Optuna, GridSearch            │
│  11    │ Değerlendirme         │ sklearn metrics, SHAP, LIME   │
│  12    │ Model Kaydetme        │ joblib, MLflow                │
│  13    │ Deployment            │ FastAPI, Docker, Cloud        │
│  14    │ Monitoring            │ Evidently, PSI, logging       │
│  15    │ Dokümantasyon         │ Markdown, PDF reports         │
└────────┴───────────────────────┴───────────────────────────────┘

⏱️ Tipik Zaman Dağılımı:
  Problem Anlama  : %5
  Veri Toplama    : %10
  EDA + Temizleme : %35  ← En çok zaman buraya gider
  Feature Eng.    : %20
  Modelleme       : %15
  Deploy + Monitor: %15
```

> **Altın Kurallar:**
> 1. *"Garbage in, garbage out"* — Veri kalitesi her şeyden önemli
> 2. *Data leakage'a dikkat* — fit sadece train'de, transform hepsinde
> 3. *Basit başla, karmaşıklaştır* — Logistic Regression → LightGBM
> 4. *Domain knowledge > fancy algorithms* — İş bilgisi her şeyi yener
> 5. *Reproducibility* — Random seed, pipeline, versiyon kontrolü
> 6. *Test setine sadece EN SONDA 1 kez bak* — Aksi halde overfitting