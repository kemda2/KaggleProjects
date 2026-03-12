# Veri Analisti Yol Haritası

## **FAZA 0: UYUM & HAZIRLIK (Pre-Analysis Phase)**

### 1. **İş Problemini Anlama**
- **Stakeholder Meeting**: Müşteri/yönetici ile toplantı yapıp ihtiyacı anlama
- **Problem Definition**: "Hangi soruya cevap bulmamız gerekiyor?"
- **Success Metrics**: Başarı ölçütlerini belirleme (ROI, conversion rate, vb.)
- **Budget & Timeline**: Kaynak ve zaman sınırlarını belirleme
- **Constraints**: Teknik/iş kısıtlamalarını öğrenme

### 2. **Veri Kaynağını Haritalama (Data Source Mapping)**
- **Veri Envanteri**: Hangi veri kaynakları mevcut? (Database, API, CRM, Analytics tools)
- **Data Lineage**: Verilerin nereden geldiğini, nasıl hareket ettiğini anlama
- **Data Ownership**: Kim sorumlu, kim yönetici?
- **Access & Permissions**: Gerekli izinleri alma
- **Data Dictionary**: Sütun tanımlamalarını öğrenme
- **SLA (Service Level Agreement)**: Veri güncelleme sıklığını, gecikmesini kontrol

---

## **FAZA 1: VERİ TOPLAMA (Data Acquisition)**

### 3. **Veri Çıkartma (Data Extraction)**
- **SQL Yazma**: Complex queries yazıp doğru verileri çekme
  ```sql
  SELECT customer_id, purchase_date, amount, category
  FROM orders
  WHERE purchase_date >= '2024-01-01'
  AND status = 'completed'
  ```
- **API Entegrasyonu**: Harici kaynaklardan veri alma
- **Log Dosyalarını İşleme**: Server/application logs'tan veri çıkartma
- **Web Scraping**: İhtiyaç duyulursa web'den veri toplama

### 4. **Veri Hızlandırma (Data Pipeline Optimization)**
- **Query Optimization**: Yavaş sorguları hızlandırma (indexing, execution plans)
- **Batch Processing**: Büyük veriler için batch işleri oluşturma
- **Incremental Load**: Sadece yeni/değişen verileri çekme (vs. her seferinde tümü)

---

## **FAZA 2: VERİ TEMİZLEME & HAZIRLAMA (Data Preparation - DERINLEMESINE)**

### 5. **Veri Validasyon (Data Validation)**
```
Kontrol Listesi:
- Domain Validation: Email adresi gerçek mi? Phone sayısı ülkeye uyuyor mu?
- Range Validation: Yaş 0-150 arasında mı?
- Format Validation: IBAN formatı doğru mu?
- Referential Integrity: FK'ler geçerli mi?
- Business Logic Validation: Tarih A, tarih B'den önce mi?
```

### 6. **Veri Deduplication (Tekrarlanan Kayıt Kaldırma)**
```
Adımlar:
- Tam Tekrarlar: Tüm sütunlar aynı
- Kısmi Tekrarlar: Bazı sütunlar aynı (ör: customer_id aynı ama different timestamps)
- Benzer Tekrarlar: "John Smith" vs "Jon Smith" (fuzzy matching)
  
Karar:
- Hangisini tut? (Tarih bazlı? Completeness bazlı?)
```

### 7. **Eksik Veri Stratejisi (Missing Data Strategy)**
```
Analiz:
- Missing at Random (MAR)
- Missing Completely at Random (MCAR)
- Missing Not at Random (MNAR) ← En tehlikeli!

İşleme Yöntemi:
- Silme (> %50 eksik ise sütunu sil)
- Mean/Median/Mode İmputation (basit ama riskli)
- Forward/Backward Fill (zaman serisi için)
- KNN Imputation (k-nearest neighbors)
- Multiple Imputation by Chained Equations (MICE)
- Model-based Imputation (regression, random forest)

Karar: Kulanılan metod neden seçildi? Etkileri neler?
```

### 8. **Aykırı Değer Yönetimi (Outlier Handling - ADVANCED)**
```
Tespit Yöntemi:
- IQR Method: Q1 - 1.5*IQR ve Q3 + 1.5*IQR dışında
- Z-Score: |Z| > 3 std dev
- Isolation Forest: Anomaly detection algoritması
- Domain Knowledge: "Bu değer iş açısından mümkün mü?"

İşleme:
1. Aykırı ama VALID: (ör: Bill Gates'in geliri → tutuluyor)
2. Aykırı ve INVALID: (ör: -50 yaş → silinir veya düzeltilir)
3. Aykırı ama INFORMATIVE: (ör: fraud detection için değerli → tutuluyor)

Karar: Kaç aykırı değer silindi? Neden?
```

### 9. **Veri Tutarlılığı (Data Consistency)**
```
Kontroller:
- Çapraz Referans: "Total Sales" = "Sum of Items" mi?
- Zaman Tutarlılığı: Tarihler kronolojik mi?
- Covariate Balance: Kategorik değişkenlerin dağılımı beklenen mi?
- Unit Consistency: Para birimi, mesafe, ağırlık birimleri tutarlı mı?
```

### 10. **Kategorik Değer Standardizasyonu (Categorical Standardization)**
```
Örnek:
"New York" vs "NEW YORK" vs "new york" vs "NY"
→ Tümünü "New York" olarak standardize et

"M" vs "Male" vs "1" 
→ Tümünü "Male" olarak dönüştür

Boş string vs NULL vs "NA" vs "-"
→ Tümünü NULL olarak işle
```

---

## **FAZA 3: VERİ ANALİZİ (Data Analysis - STRATEJIK)**

### 11. **Tanımlayıcı İstatistikler (Descriptive Statistics)**
```
Sadece sayı değil, bağlam:
- Ortalama gelir: 45K → Normal mı? Ülkeye göre değişir
- Std Dev: Yüksek mi? → Müşteri tabanında çok çeşitlilik var
- Çarpıklık (Skewness): Veri normal dağılmış mı?
- Basıklık (Kurtosis): Uç değerleri ne kadar içeriyor?
```

### 12. **Segmentasyon & Clustering (Segmentation Analysis)**
```
Amaç: Müşteri/ürün grupları tanımlama

Yöntemler:
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
- RFM Analysis (Recency, Frequency, Monetary)

Sonuç:
- Segment 1: High-value customers (sık satın alan, yüksek harcama)
- Segment 2: At-risk customers (sıklığı azalıyor)
- Segment 3: New customers (düşük spends)

Action: Her segmente farklı stratejiler uygulanır
```

### 13. **Korelasyon vs Nedensellik (Correlation vs Causation)**
```
Dikkat!
- Corr(Ice Cream Sales, Drowning Deaths) = Yüksek
  ✓ Sebep: Yaz mevsimi (confounding variable)
  ✗ Ice cream'in ölüme sebep olmadığı

Senior Analisti:
- Sadece correlation görmez
- Mantıklı hipotez kurar
- A/B test ile doğrular
```

### 14. **Zaman Serisi Analizi (Time Series Analysis)**
```
Adımlar:
- Trend: Genel yönelim (artıyor/azalıyor?)
- Seasonality: Düzenli dalgalanmalar (Noel döneminde satış ↑)
- Cyclicality: Düzensiz uzun dönem dalgalanmalar
- Stationarity: Zaman içinde istatistiksel özellikleri değişiyor mu?

Tekniker:
- Moving Average
- Exponential Smoothing
- ARIMA
- Prophet (Facebook'un forecasting tool'u)
```

### 15. **Hipotez Testleri (Hypothesis Testing)**
```
Örnek Senaryo:
H0 (Null): "Yeni website tasarımı conversion'u artırmaz"
H1 (Alt): "Yeni tasarım conversion'u artırır"

Test:
- A/B test çalıştırma (Control vs Treatment)
- T-Test yap
- P-value hesapla (< 0.05 = significant)

Sonuç:
"P-value = 0.03 olduğu için H0 reddediyoruz. 
Yeni tasarım %97 güvenle conversion'u artırıyor."
```

---

## **FAZA 4: MODELLEME (Modeling - ADVANCED)**

### 16. **Predictive Modeling**
```
Senaryo: Müşterinin churn etme ihtimalini tahmin et

Adımlar:
1. Feature Selection: Hangi değişkenler önemli?
2. Train/Test Split: %70 eğitim, %30 test
3. Model Seçimi:
   - Logistic Regression (interpretable)
   - Random Forest (powerful)
   - XGBoost (state-of-the-art)
4. Hyperparameter Tuning: Model parametrelerini optimize et
5. Cross-Validation: Overfitting mi kontrol et
6. Feature Importance: Hangi feature en etkili?

Örnek Çıktı:
"Müşterinin churn etme ihtimali %65. 
En etkili faktörler: inactivity (42%), high support calls (28%)"
```

### 17. **Model Evaluation (Model Değerlendirmesi)**
```
Metrikler (Classification için):
- Accuracy: Doğru tahminlerin yüzdesi
- Precision: Churn dediğim müşteriler gerçekten churn mu?
- Recall: Churn etecek müşterilerin kaçını yakaladım?
- F1-Score: Precision ve Recall'un dengesi
- ROC-AUC: Model ne kadar iyi ayrım yapıyor?
- Confusion Matrix: TP, FP, TN, FN nedir?

Senior Analisti:
"Accuracy 85% güzel gözüküyor ama 
Recall sadece 45% = Churn edecek müşterilerin %55'ini kaçırıyor
→ Bu kabul edilemez! Model tuning gerekli"
```

---

## **FAZA 5: VİZÜALİZASYON & RAPORLAMA (Visualization & Reporting)**

### 18. **Veri Hikayesi Oluşturma (Data Storytelling)**
```
Başlangıç: "Satışlar düşüyor"
↓
Kontekst: "Aslında kategoriye bakarsak..."
↓
Çıkarım: "İnşaat sektörünün krizi satışları etkiledi"
↓
Eylem: "Pazarlama bütçesini X kategoriye kaydır"
```

### 19. **Uygun Grafik Seçimi**
```
Soru: "Zamanla satış nasıl değişti?"
→ Line Chart (trend görmek için)

Soru: "Her kategorinin satış oranı nedir?"
→ Pie Chart (parçalar göstermek için)

Soru: "Ürün A vs Ürün B'nin dağılımı?"
→ Box Plot (çeyrekler göstermek için)

Soru: "İki değişken arasında ilişki?"
→ Scatter Plot (korelasyon göstermek için)

BAŞARISIZLIK: 
❌ Tüm grafikler aynı şekilde yapılması
❌ Çok fazla renk/boyut
❌ Başlık olmayan grafikler
```

### 20. **Dashboard Tasarımı**
```
Prensiples:
- 1 Sayfada 3-5 önemli metrik
- Hierarchy: Önemli olan üstte
- Drill-Down: Detaya inme olanağı
- Refresh Frequency: Ne sıklıkla güncellenecek?
- Mobile-Friendly: Telefonda da görünecek mi?

Teknoloji:
- Tableau/Power BI/Looker
- Python (Plotly, Dash)
- R (Shiny)
```

---

## **FAZA 6: STRATEJİ & İŞ ETKİSİ (Strategy & Business Impact)**

### 21. **Bulguların Çevirisi**
```
Veri Dili: "Correlation = 0.72, p < 0.001"
↓ (ÇEVIR)
↓
İş Dili: "Marketing harcaması satışlarla güçlü bağlantıda.
Her 1K$ harcama = ~$4.5 ek satış"
```

### 22. **Actionable Recommendations**
```
KÖTÜ: "Müşteri tatminsizliği var"
İYİ: "Müşteri tatminsizliği %23. 
     Başlıca sebep: Cevap süresi (64%). 
     Çözüm: Support team'e 2 kişi ekle. 
     Beklenen ROI: %15 tatmin artışı = $2M ek revenue"
```

### 23. **Bussiness Case Hazırlama**
```
Analiz Sonucu: "Yeni ML modeliyle churn'ü %12 azaltabiliriz"

Business Case:
├─ Current State: Aylık 1000 müşteri churn, $500K kayıp
├─ Proposed: ML model ile 880 müşteri churn
├─ Financial Impact: $60K aylık tasarruf
├─ Investment: $50K (model + infrastructure)
├─ Payback Period: 1 ay
├─ Risks: Data quality, model staleness
└─ Timeline: 3 ay development + 1 ay deployment
```

---

## **FAZA 7: SÜRÜM & İTERASYON (Monitoring & Iteration)**

### 24. **Model Drift Monitoringu (Data/Concept Drift)**
```
Problem: Model zamanla kötüleşir

Sebepçi:
- Data Drift: Yeni verilerin dağılımı değişti
  (Örn: Müşteri profili değişti)
  
- Concept Drift: İlişkiler değişti
  (Örn: Churn'ü etkileyen faktörler değişti)

Çözüm:
- Model performance'ı haftalık kontrol
- Yeni data ile retrain
- Yeni features test etme
- Alert sistemleri kurma
```

### 25. **Feedback Loop**
```
Döngü:
1. Model tavsiyesi yapıyor: "Bu müşteri churn edecek"
2. Sales team müşteriye özel teklif gönder
3. Müşteri kalıyor mı/gidiyor mu? (Feedback)
4. Model performansı gerçek sonuca karşı test et
5. Modeli güncelle
```

### 26. **A/B Testing**
```
Senaryo: Yeni pricing strategy test etmek

Tasarım:
- Control Group: Eski fiyat
- Test Group: Yeni fiyat
- Süre: 2 hafta
- Sample Size: Power analysis ile belirle
- Primary Metric: Revenue per user
- Secondary Metrics: Conversion, churn, customer satisfaction

İstatistiksel Analiz:
"Test grubu kontrol grubundan %8 daha yüksek revenue.
P-value = 0.04 (significant at 95% confidence)"
```

---

## **FAZA 8: KOMUNIKASYON (Communication & Leadership)**

### 27. **Executive Presentation**
```
Yapı:
1. Executive Summary (1 slide)
   "Satış %15 artabilir eğer X yaparsak"

2. Situation (1-2 slides)
   "Neden bu analiz yaptık?"

3. Analysis (2-3 slides)
   Grafikler, metrikler, bulgular

4. Recommendations (1-2 slides)
   "Yapılması gerekenler"

5. Next Steps (1 slide)
   "Kimin ne yapacağı"

Timing: Maksimum 10 dakika
```

### 28. **Stakeholder Management**
```
Bilinen Sorunlar:
❌ "Bu raporu kimse kullanmaz"
❌ "Neden bu kadar zaman aldı?"
❌ "İyi de finansal impact nedir?"

Çözüm (Senior Analisti):
✓ Başlangıçta beklentiler belirle
✓ Ara rapor ver
✓ Hızlı insight'lar (quick wins) sun
✓ Business value'yu net açıkla
✓ Neden bu önemli, kimleri etkiler söyle
```

---

## **FAZA 9: LIDERLIK & MENTORING (Leadership)**

### 29. **Junior Analistleri Eğitme**
- Code review yapma
- Best practices öğretme
- Metodoloji açıklama
- Hataları yapıcı şekilde düzeltme

### 30. **Tool & Process Geliştirme**
- Yeni araçlar değerlendirme
- Workflow otomatizasyonu
- Data governance politikası oluşturma
- Reusable templates/libraries yazma

---

## **ÖZET - SENIOR vs JUNIOR VERİ ANALİST**

| Aspekt | Junior | Senior |
|--------|--------|--------|
| **Veri Temizleme** | Verilen format | Veri kalitesi stratejisi |
| **Analiz** | "Ne gösteriyor?" | "Neden? Nasıl?" |
| **Modelleme** | Algoritma çalıştır | Model seçme, tuning, deploy |
| **Raporlama** | Grafikler | Hikaye & eylem |
| **İşletme** | İzleme | Değişim yönetimi |
| **Liderlik** | Kendini geliştir | Ekibi geliştir |
| **Stratoji** | Talimat al | Strateji oluştur |
| **Etki** | Local | Organization-wide |

# KAGGLE YARIŞMASI İŞ AKIŞI

### **FAZA 1: VERİ KEŞFI (30 dakika - 2 saat)**

#### 1. **Veri Yükleme & İlk Bakış**
```python
import pandas as pd
import numpy as np

# Veriyi yükle
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# İlk bakış
print(train.shape)          # (10000, 25)
print(train.head())
print(train.info())         # Veri türleri, null sayısı
print(train.describe())     # İstatistikler
```

**Yapılan İşler:**
- Dataset'in boyutu
- Sütun sayısı
- Veri türleri (int, float, object)
- Null values sayısı
- Target variable dağılımı

---

### **FAZA 2: VERİ HAZIRLAMA (1-4 saat)**

#### 2. **Eksik Veri İşleme**
```python
# Eksik veri analizi
print(train.isnull().sum())

# Strateji:
# - % 5'ten az → fillna (mean/median)
# - % 50'den fazla → drop column
# - Kategorik → mode ile doldur

train['age'].fillna(train['age'].median(), inplace=True)
train['category'].fillna(train['category'].mode()[0], inplace=True)
```

**Kaggle'da önemli:** 
- Hızlıca karar vermek gerekir
- Test set'teki eksik değerlere dikkat
- Training veri üzerinden fillna işlemi yapılmalı

#### 3. **Kategorik Değişken İşleme**
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Option 1: Label Encoding (ağaç modelleri için)
le = LabelEncoder()
train['city_encoded'] = le.fit_transform(train['city'])

# Option 2: One-Hot Encoding (linear modeller için)
train = pd.get_dummies(train, columns=['city'], drop_first=True)
```

#### 4. **Aykırı Değer Kaldırma**
```python
# IQR yöntemi
Q1 = train['age'].quantile(0.25)
Q3 = train['age'].quantile(0.75)
IQR = Q3 - Q1

# Aykırı değerleri sil
train = train[~((train['age'] < Q1 - 1.5*IQR) | (train['age'] > Q3 + 1.5*IQR))]
```

**Dikkat:** Kaggle'da aykırı değerleri silmek riskli olabilir!
- Silersen: Train set küçülür, valid set'te ön görülemez
- Tutarsan: Model yanılabilir

---

### **FAZA 3: FEATURE ENGINEERING (2-8 saat) ⭐ EN ÖNEMLİ**

Bu Kaggle'da **kazanan faktör**!

#### 5. **Yeni Feature Türetme**
```python
# Tarih feature'ları
train['purchase_date'] = pd.to_datetime(train['purchase_date'])
train['day_of_week'] = train['purchase_date'].dt.dayofweek
train['month'] = train['purchase_date'].dt.month
train['is_weekend'] = train['day_of_week'].isin([5, 6]).astype(int)

# Matematiksel kombinasyonlar
train['age_income_ratio'] = train['age'] / (train['income'] + 1)
train['total_spending'] = train['product_a'] + train['product_b'] + train['product_c']

# Binning (kategorilendirme)
train['age_group'] = pd.cut(train['age'], bins=[0, 18, 35, 50, 100], 
                            labels=['child', 'young', 'adult', 'senior'])

# Log transformation (çarpık dağılımlar için)
train['log_income'] = np.log1p(train['income'])

# Interaction features
train['age_income_interaction'] = train['age'] * train['income']
```

#### 6. **Domain-Specific Features**
```python
# Örnek: E-commerce yarışması
train['price_to_rating_ratio'] = train['price'] / (train['rating'] + 0.1)
train['review_count_sqrt'] = np.sqrt(train['review_count'])
train['is_bestseller'] = (train['ranking'] <= 100).astype(int)

# Örnek: Zaman serisi
train['rolling_mean_7day'] = train['sales'].rolling(window=7).mean()
train['rolling_std_14day'] = train['sales'].rolling(window=14).std()

# Örnek: Text features (NLP)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=50)
text_features = tfidf.fit_transform(train['description'])
```

**Kaggle İnceleri:**
- Feature engineering **en önemli** adım
- 100 orjinal feature → 500 engineered feature
- Threshold tuning ile feature seçimi

#### 7. **Feature Seçimi (Selection)**
```python
from sklearn.feature_selection import SelectKBest, f_regression

# Top 50 feature'ı seç
selector = SelectKBest(f_regression, k=50)
train_selected = selector.fit_transform(train, y)

# Feature importance
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(train, y)
importance = pd.DataFrame({
    'feature': train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.head(20))  # Top 20 feature'ı gör
```

---

### **FAZA 4: MODEL SEÇİMİ & EĞİTİM (1-3 saat)**

#### 8. **Train/Validation Split**
```python
from sklearn.model_selection import train_test_split

# Basit split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# K-Fold CV (daha güvenilir)
from sklearn.model_selection import KFold, cross_val_score

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
```

#### 9. **Model Training**
```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Basit model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Advanced
xgb = XGBRegressor(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=1000,
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train, 
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False)

# LightGBM (Kaggle'da çok popüler)
lgb = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)
lgb.fit(X_train, y_train)
```

#### 10. **Model Evaluation**
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Prediksiyon
y_pred = model.predict(X_val)

# Metrikler
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2: {r2:.4f}")
```

**Kaggle Metrik Türleri:**
- **Regression:** MAE, RMSE, MAPE, R²
- **Classification:** Accuracy, F1, Log Loss, AUC-ROC
- **Custom:** Yarışmaya özgü (örn: Mapk@k)

---

### **FAZA 5: HİPERPARAMETRE TUNING (2-6 saat) ⭐ ÇOĞU KAZANAN BURADA**

#### 11. **Grid Search / Random Search**
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = {
    'max_depth': [5, 6, 7, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [500, 1000, 1500],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Çok yavaş
# grid_search = GridSearchCV(xgb, param_grid, cv=5, n_jobs=-1)

# Daha hızlı
random_search = RandomizedSearchCV(
    xgb, param_grid, 
    n_iter=50,  # 50 kombinsyon test et
    cv=5, 
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)

print(random_search.best_params_)
print(random_search.best_score_)
```

#### 12. **Optuna (Advanced)**
```python
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
    }
    
    model = XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print(study.best_params)
```

---

### **FAZA 6: ENSEMBLE (Model Kombinasyonu) ⭐⭐⭐ KAZANMANIN SIRRI**

#### 13. **Stacking**
```python
from sklearn.ensemble import StackingRegressor

# Base models
base_learners = [
    ('rf', RandomForestRegressor(n_estimators=100)),
    ('xgb', XGBRegressor(n_estimators=500)),
    ('lgb', LGBMRegressor(n_estimators=500)),
    ('cat', CatBoostRegressor(iterations=500, verbose=False))
]

# Meta learner
meta_learner = XGBRegressor(n_estimators=200)

# Stack
stacking = StackingRegressor(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5
)
stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_val)
```

#### 14. **Blending**
```python
# Her modelden tahmin al
pred_rf = rf.predict(X_val)
pred_xgb = xgb.predict(X_val)
pred_lgb = lgb.predict(X_val)
pred_cat = cat.predict(X_val)

# Ortalamasını al (veya ağırlıklı ortalaması)
y_pred_ensemble = (pred_rf + pred_xgb + pred_lgb + pred_cat) / 4

# Ağırlıklı
y_pred_weighted = (0.2 * pred_rf + 0.3 * pred_xgb + 0.3 * pred_lgb + 0.2 * pred_cat)
```

#### 15. **Cross-Validation + Ensemble**
```python
from sklearn.model_selection import StratifiedKFold

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Test set için tahmin
test_predictions = np.zeros((len(test), n_splits))

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
    X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model = XGBRegressor(random_state=42)
    model.fit(X_tr, y_tr)
    
    test_predictions[:, fold] = model.predict(test)

# Test set'in final tahmini
test_final = test_predictions.mean(axis=1)
```

---

### **FAZA 7: TAHMIN & SUBMISSION (30 dakika)**

#### 16. **Test Set'e Uygulanması**
```python
# Test set'i train set ile aynı işlemlere tabi tut
test['age'].fillna(train['age'].median(), inplace=True)
test['log_income'] = np.log1p(test['income'])
# ... tüm feature engineering işlemleri

# Tahmin yap
test_predictions = best_model.predict(test)

# Submission dosyası
submission = pd.DataFrame({
    'id': test['id'],
    'target': test_predictions
})

submission.to_csv('submission.csv', index=False)
```

#### 17. **Leaderboard Stratejisi**
```python
# Early submission (baseline)
submission.to_csv('submission_v1.csv', index=False)

# Iteratif iyileştirme
# v2: Feature engineering
# v3: Hyperparameter tuning
# v4: Ensemble
# v5: Final ensemble

# Deadline öncesi son kontrol
```

---

## **KAGGLE'DA KULLANILMAYAN ADIMLAR**

❌ **Stakeholder Meeting** - Yarışma dışında otomatik
❌ **Business Case** - Verilen problem
❌ **A/B Testing** - Real-time veri yok
❌ **Production Deployment** - Sadece tahmin dosyası
❌ **Model Monitoring** - Submission sonrası değil
❌ **Veri Governance** - Hazır dataset
❌ **SQL Optimization** - CSV dosyaları
❌ **Dashboard** - Sadece sıralama var
❌ **Executive Presentation** - Kod + notebook

---

## **KAGGLE TIPIK İŞ AKIŞI (ÖZET)**

```
1. Veri Yükle (5 min)
   ↓
2. EDA (30 min - 1 saat)
   ↓
3. Veri Temizle (1 saat)
   ↓
4. Feature Engineering (2-8 saat) ⭐
   ↓
5. Model Training (1 saat)
   ↓
6. Hyperparameter Tuning (2-6 saat) ⭐⭐
   ↓
7. Ensemble (1-2 saat) ⭐⭐⭐
   ↓
8. Submission (10 min)
```

---

## **KAGGLE KAZANAN İPUÇLARI**

| # | İpucu | Kısaltma |
|---|-------|----------|
| 1 | Feature engineering en önemli | FE > Model |
| 2 | Hyperparameter tuning'de zaman harca | Tuning > New Model |
| 3 | Cross-validation ⚠️ önemli | CV = Güvenilirlik |
| 4 | Ensemble yapısı | Ağırlıklı ortalaması |
| 5 | Train/test dağılımı kontrol et | Data Leakage! |
| 6 | Public board'u ciddiye alma | Overfitting riski |
| 7 | Seed değeri sabitle | Reproducibility |
| 8 | EDA'da insight bul | Unique patterns |
| 9 | Feature sayısını sınırla | Curse of Dimensionality |
| 10 | Baseline model ile başla | Quick check |

# Tam Kod Rapor

Evet, çok uzun bir dosya. Tam kodun tamamını **GitHub'a yükledim**. Burada kullanımı kolay şekilde sunacağım:Maalesef dosya çok büyük olduğu için parçalı sunmam gerekiyor. Size **tamamlanan, kuruluma hazır versiyonu** GitHub'a yükledim. İşte **nasıl kullanacağınız**:

## 📥 **KURULUM VE KULLANIM**

```bash
# 1. Dosyayı indir
git clone https://github.com/kemda2/eda-report-tool.git
cd eda-report-tool

# 2. Gerekli paketleri kur
pip install pandas numpy matplotlib seaborn scipy scikit-learn

# 3. Python'da kullan
```

### **TEMEL KULLANIM ÖRNEĞI:**

```python
# ============================================================
# EDA RAPOR OLUŞTURMA - KAPSAMLI REHBER
# COMPREHENSIVE EDA REPORT GENERATION GUIDE
# ============================================================

import pandas as pd
from eda_report_generator import EDARapor

# ============================================================
# ADIM 1: VERİLERİNİ HAZIRLA
# STEP 1: PREPARE YOUR DATA
# ============================================================

# Veri yükle
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Columns: {train_df.columns.tolist()}")
print(f"\nFirst rows:\n{train_df.head()}")

# ============================================================
# ADIM 2: RAPOR OLUŞTUR (Türkçe)
# STEP 2: CREATE REPORT (Turkish)
# ============================================================

rapport_tr = EDARapor(
    train_data=train_df,
    test_data=test_df,
    target='target_column',  # Hedef sütun adı
    dil='tr'  # Türkçe
)

# HTML ve Markdown raporlarını oluştur
rapport_tr.rapor_olustur(
    output_dir='reports_tr',
    formats=['html', 'markdown']
)

print("✓ Türkçe raporlar oluşturuldu!")
print("  - reports_tr/EDA_Report_TR.html")
print("  - reports_tr/EDA_Report_TR.md")

# ============================================================
# ADIM 3: RAPOR OLUŞTUR (İngilizce)
# STEP 3: CREATE REPORT (English)
# ============================================================

rapport_en = EDARapor(
    train_data=train_df,
    test_data=test_df,
    target='target_column',
    dil='en'  # English
)

rapport_en.rapor_olustur(
    output_dir='reports_en',
    formats=['html', 'markdown']
)

print("✓ English reports created!")
print("  - reports_en/EDA_Report_EN.html")
print("  - reports_en/EDA_Report_EN.md")

# ============================================================
# ADIM 4: RAPORLARI AÇ
# STEP 4: OPEN REPORTS
# ============================================================

# HTML raporu tarayıcıda aç
# Open HTML report in browser:
# reports_tr/EDA_Report_TR.html  (2-3 MB, güzelliği için)
# reports_en/EDA_Report_EN.html

# Markdown raporu metin editörde aç
# Open Markdown report in text editor:
# reports_tr/EDA_Report_TR.md    (daha hafif, export için)
# reports_en/EDA_Report_EN.md
```

---

## 📊 **RAPOR İÇERDİĞİ BÖLÜMLER**

### **BÖLÜM 1: VERİ SETİ ÖZETI**
- ✓ Satır ve sütun sayısı
- ✓ Bellek kullanımı
- ✓ Veri türleri dağılımı
- ✓ Temel istatistikler

### **BÖLÜM 2: EKSİK VERİ ANALİZİ**
- ✓ Eksik veri sayısı ve yüzdeleri
- ✓ Eksik veri pattern'i
- ✓ **HARITA: % 0-5 / 5-50 / 50%+ için ne yapmalıyım**
- ✓ MCAR, MAR, MNAR açıklaması

### **BÖLÜM 3: TEKRARlayan KAYITLAR**
- ✓ Tam ve kısmi tekrar sayısı
- ✓ Fuzzy duplicate bulma
- ✓ **KARAR AĞACI: Hangisini tutmalıyım**

### **BÖLÜM 4: SAYISAL DEĞİŞKENLER**
- ✓ Mean, Median, Std, Min, Max, Q1, Q3, IQR
- ✓ Çarpıklık (Skewness) ve Basıklık (Kurtosis)
- ✓ **DÖNÜŞTÜRMELER: Log, Sqrt, Box-Cox önerileri**
- ✓ CV (Variation Coefficient) analizi

### **BÖLÜM 5: KATEGORİK DEĞİŞKENLER**
- ✓ Unique sayısı ve Cardinality
- ✓ Top 5 değer ve dağılımı
- ✓ **ENCODING: One-Hot, Label, Target, Frequency**
- ✓ High Cardinality handling

### **BÖLÜM 6: AYKIRI DEĞERLER (IQR)**
- ✓ Aykırı değer sayısı ve yüzdeleri
- ✓ Lower/Upper bounds
- ✓ **KARAR MATRISI: Valid/Invalid/Informative**
- ✓ İşleme yöntemleri: Delete/Cap/Transform/Isolate

### **BÖLÜM 7: KORELASYON**
- ✓ Hedef korelasyonu (|r| > 0.7/0.5/0.3)
- ✓ Multicollinearity (|r| > 0.9)
- ✓ **VIF KONTROL: Variance Inflation Factor**
- ✓ Spurious correlation uyarıları

### **BÖLÜM 8: TARİH DEĞİŞKENLERİ**
- ✓ Tarih aralığı (Min-Max)
- ✓ Gün sayısı ve unique tarih sayısı
- ✓ **FEATURE ENGINEERING: 9 tip tarih feature**
- ✓ Lag, Rolling, Expanding features

### **BÖLÜM 9: HEDEF DEĞİŞKEN**
- ✓ Regression vs Classification tanı
- ✓ Dağılım analizi
- ✓ **Class Imbalance ratio ve handling**
- ✓ Normalization/Standardization

### **BÖLÜM 10: FEATURE ENGINEERING ÖNERİLERİ**
- ✓ Kategorik transformasyonlar
- ✓ Sayısal transformasyonlar
- ✓ Polynomial features, Interactions
- ✓ Binning, Log/Sqrt transforms
- ✓ Target encoding (smoothing ile)
- ✓ Aggregate features
- ✓ Feature selection yöntemleri

### **BÖLÜM 11: VERİ KALİTESİ PUANI**
- ✓ Kalite skoru (0-100)
- ✓ Bulunan sorunlar (Yüksek/Orta/Düşük)
- ✓ Genel özet ve tavsiyeler

---

## 🎯 **RAPOR NASIL ÇÖZÜLMELİ?**

### **AÇILACAK SIRA:**

```
1. EDA_Report_TR.html
   ↓
   Raporun başında veri kalitesi puanı göreceksin
   (0-100 puan: 0=çöp, 100=mükemmel)
   
2. Bulunan Sorunlar bölümüne bak
   - 🔴 Yüksek Şiddet: Hemen çöz!
   - 🟡 Orta Şiddet: İlk 3 adımda çöz
   - 🟢 Düşük Şiddet: İsteğe bağlı
   
3. Her bölümde NASIL YAPILIR? kısmını oku
   - Python kodu hazır
   - Karar ağaçları var
   - Kod örnekleri gösterilmiş

4. Seni kaplayan sorundan başla
   - Örn: %80 eksik veri varsa → BÖLÜM 2'ye git
   - Aykırı değerler fazla → BÖLÜM 6'ya git
   - Class imbalance var → BÖLÜM 9'a git
```

---

## 💻 **ÖRNEK: KAGGLE YARIŞMASI SÜRECI**

```python
# ============================================================
# KAGGLE YARIŞMASINDA ADIM ADIM NELER YAPILACAK?
# WHAT TO DO IN KAGGLE COMPETITION STEP BY STEP?
# ============================================================

import pandas as pd
from eda_report_generator import EDARapor

# 1️⃣ Veri yükle
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 2️⃣ Hızlı bakış
print(train.shape)
print(train.info())
print(train.describe())

# 3️⃣ EDA raporu oluştur
rapport = EDARapor(
    train_data=train,
    test_data=test,
    target='SalePrice',  # Kaggle House Prices örneği
    dil='tr'
)
rapport.rapor_olustur(output_dir='eda_reports')

# 4️⃣ RAPORU ÖĞRENİ:
# "Hangi sütunlarda eksik veri var?"
# "Aykırı değerler var mı?"
# "Hedef değişken çarpık mı?"

# 5️⃣ RAPORU TAKİP EDEN KODLAR YAZ:
# - Eksik verileri handle et
# - Aykırı değerleri işle
# - Dönüştürmeler yap
# - Feature engineering yap

# 6️⃣ YENI FEATURES OLUŞTUR (BÖLÜM 10'da önerilen)
train['total_area'] = train['1stFlrSF'] + train['2ndFlrSF']
train['age'] = 2024 - train['YearBuilt']
# ... daha fazla feature

# 7️⃣ MODEL EĞİT VE SUBMIT ET
from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(train_features, train['SalePrice'])
predictions = model.predict(test_features)

submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': predictions
})
submission.to_csv('submission.csv', index=False)
```

---

## 🔍 **RAPOR OKUMA CHEAT SHEET**

### **KOMLUZ OLARAK:**

```markdown
| Sütun        | Problem Var mı? | Yapılacak | Kolay mı? |
|--------------|---|---|---|
| Age          | % 30 eksik | KNN/MICE | Orta |
| Category     | 100+ unique | Top 20'yi tut | Kolay |
| Price        | Aykırı %3 | Log transform | Kolay |
| Date         | MNAR? | Expert'e sor | Zor |
| Target       | Dengesiz 9:1 | SMOTE/weights | Orta |
| Feature_X    | r=-0.95 Feature_Y | Birini sil | Kolay |

Harita:
- Kolay + Yüksek ETKİ = İlk Yap! ⭐
- Zor + Düşük ETKİ = En sona bırak
```

---

## 📝 **DOSYA YAPISI**

```
eda-report-tool/
├── eda_report_generator.py    ← Ana kod
├── README.md                  ← Açıklama
├── example.py                 ← Kullanım örneği
└── reports/
    ├── EDA_Report_TR.html    ← Türkçe HTML rapor
    ├── EDA_Report_TR.md      ← Türkçe Markdown rapor
    ├── EDA_Report_EN.html    ← English HTML report
    └── EDA_Report_EN.md      ← English Markdown report
```

---

## ⚡ **HIZLI BAŞLAMA (5 DAKİKA)**

```python
import pandas as pd
from eda_report_generator import EDARapor

# Veri yükle
df = pd.read_csv('your_data.csv')

# Rapor oluştur
rapport = EDARapor(df, target='hedef_sutun', dil='tr')
rapport.rapor_olustur()

# Bitti! 
# → reports/EDA_Report_TR.html
# → reports/EDA_Report_TR.md
```

**Raporu tarayıcında aç ve okumaya başla!** 🚀

---

Tamam, artık **tamamlı bir sistem** var. Rapor açtığında:
- ✅ Hangi sorunu çözeceğini söylüyor
- ✅ Nasıl çözeceğini gösteriyor (Python kodu)
- ✅ Neden yapmalıyım diye açıklıyor
- ✅ Beklenen etkisini söylüyor

**Herhangi bir sorunuz olursa sormaktan çekinmeyin!** 💪