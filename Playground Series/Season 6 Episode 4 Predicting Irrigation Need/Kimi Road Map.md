**Kaggle Playground Series S6E4** yarışması bir **sınıflandırma** problemi ve metriğimiz **Balanced Accuracy** (her sınıfın doğruluğunun ağırlıklı ortalaması). Veri seti oldukça dengesiz (High sınıfı sadece %3.3), bu yüzden bu sınıfı yakalamak kritik.

Hemen başlayalım. İlk olarak veriyi analiz edip, sonra modelleme stratejisini kuracağız.

## 🎯 1. STRATEJİK ANALİZ (Verdiğin Özetlerden Çıkarımlar)

Senin verdiğin özetlere göre çok güçlü pattern'ler var:

**🔴 Kritik Bulgular:**
1. **Soil_Moisture (Toprak Nemi)**: High ihtiyaçta ortalama **17.7**, Low ihtiyaçta **43.3** → En güçlü belirleyici! (Negatif korelasyon)
2. **Temperature_C**: High ihtiyaçta **34.6°C**, Low'da **25.3°C** → Sıcaklık artınca su ihtiyacı artıyor
3. **Rainfall_mm**: High ihtiyaçta **989mm**, Low'da **1500mm** → Yağmur azaldıkça sulama şart
4. **Wind_Speed_kmh**: High'da **14.6 km/s**, Low'da **9.2 km/s** → Rüzgar buharlaşmayı artırıyor
5. **Crop_Growth_Stage**: 
   - **Harvest** döneminde %84.8 Low ihtiyaç (mantıklı, ürün toplanıyor)
   - **Flowering** ve **Vegetative** dönemlerinde High ihtiyaç %6.4'lere çıkıyor (bitki suyu çok tüketiyor)
6. **Mulching_Used**: 
   - **Yes**: %73 Low ihtiyaç (malçlama su kaybını önlüyor)
   - **No**: %44.5 Low ihtiyaç

**⚠️ Sınıf Dengesizliği Tehlikesi:**
```
Low:    %58.7 (369,917 örnek)
Medium: %37.9 (239,074 örnek)  
High:   %3.3  (21,009 örnek)  ← Çok az!
```
Balanced Accuracy kullanıldığı için High sınıfını %100 doğru tahmin etsen bile, Low'u %50 şaşırırsan skorun düşer. Ama High'ı kaçırırsan büyük kayıp yaşarsın.

---

## 🛠️ 2. UYGULAMA PLANI (A'dan Z'ye)

Şimdi adım adım kod yazacağız. İlk olarak gerekli kütüphaneleri ve veri yükleme işlemlerini yapalım:

```python
# Gerekli kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Gelişmiş modeller
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Veriyi yükle
train = pd.read_csv('/kaggle/input/playground-series-s6e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s6e4/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s6e4/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Target distribution:\n{train['Irrigation_Need'].value_counts(normalize=True)}")
```

### 2.1 Özellik Mühendisliği (Feature Engineering)

Bu veri setinde fiziksel ilişkiler var. Örneğin:
- **Evapotranspiration Potansiyeli**: Sıcaklık + Rüzgar - Nem (su kaybı potansiyeli)
- **Su Stres İndeksi**: Yağış / (Sıcaklık + 1) 
- **Toprak Verimliliği**: pH × Organik Karbon
- **Su Kaybı Riski**: Sıcaklık × Rüzgar / Nem

```python
def feature_engineering(df):
    df = df.copy()
    
    # 1. Fiziksel İlişkiler (Su Kaybı Potansiyeli)
    # Sıcaklık ve rüzgar arttıkça, nem azaldıkça su kaybı artar
    df['Evapotranspiration_Risk'] = (df['Temperature_C'] * df['Wind_Speed_kmh']) / (df['Humidity'] + 1)
    
    # 2. Su Dengesi (Yağış vs İhtiyaç)
    df['Water_Balance'] = df['Rainfall_mm'] - (df['Temperature_C'] * 50)  # Sıcaklığa göre su ihtiyacı
    
    # 3. Toprak Verimliliği İndeksi
    df['Soil_Fertility_Index'] = df['Soil_pH'] * df['Organic_Carbon'] * (1 - abs(df['Soil_pH'] - 6.5) / 6.5)
    
    # 4. Nem Stresi (Toprak nemi düşük ama sıcaklık yüksekse)
    df['Moisture_Stress'] = np.where(
        (df['Soil_Moisture'] < 25) & (df['Temperature_C'] > 30), 1, 0
    )
    
    # 5. Önceki Sulama Etkinliği
    df['Irrigation_Efficiency'] = df['Previous_Irrigation_mm'] / (df['Field_Area_hectare'] + 0.1)
    
    # 6. İklim Zorluğu
    df['Climate_Harshness'] = (df['Temperature_C'] / df['Humidity']) * df['Wind_Speed_kmh']
    
    # 7. pH Dengesi (6.5 idealdir, uzaklaştıkça verim düşer)
    df['pH_Ideal_Distance'] = abs(df['Soil_pH'] - 6.5)
    
    # 8. Kategorik Etkileşimler (Önemli!)
    # Eğer malçlama varsa ve toprak kumluysa su tutma kapasitesi düşüktür
    df['Mulch_Soil_Interaction'] = df['Mulching_Used'] + "_" + df['Soil_Type']
    
    # 9. Büyüme dönemi riski (Flowering/Vegetative kritik)
    df['Critical_Growth_Stage'] = df['Crop_Growth_Stage'].isin(['Flowering', 'Vegetative']).astype(int)
    
    # 10. Elektriksel İletkenlik ve Nem ilişkisi (Tuzluluk indikatörü)
    df['Salinity_Stress'] = df['Electrical_Conductivity'] / (df['Soil_Moisture'] + 1)
    
    return df

# Uygula
train_fe = feature_engineering(train)
test_fe = feature_engineering(test)

print("Yeni özellikler eklendi:")
print([col for col in train_fe.columns if col not in train.columns])
```

### 2.2 Kategorik Değişken İşleme

CatBoost kullanacağımız için kategorik değişkenleri string olarak bırakabiliriz (CatBoost otomatik encoding yapar), ancak diğer modeller için encoding gerekir:

```python
# Kategorik kolonlar
cat_features = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season', 
                'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region',
                'Mulch_Soil_Interaction']  # Yeni eklediğimiz etkileşim

# Target encoding için fonksiyon (Leakage'ı önlemek için CV içinde yapmalıyız)
def target_encoding(train_df, test_df, col, target, smoothing=10):
    """
    Mean target encoding with smoothing to prevent overfitting
    """
    global_mean = train_df[target].map({'Low': 0, 'Medium': 1, 'High': 2}).mean()
    
    # Her kategori için ağırlıklı ortalama
    category_stats = train_df.groupby(col)[target].agg(['count', 'mean'])
    # String mapping'i numeric'e çevir
    mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    category_stats['mean_num'] = category_stats['mean'].map(mapping)
    
    # Smoothing
    smoothed_values = (category_stats['count'] * category_stats['mean_num'] + 
                       smoothing * global_mean) / (category_stats['count'] + smoothing)
    
    # Mapping dictionary
    encode_dict = smoothed_values.to_dict()
    
    train_df[f'{col}_encoded'] = train_df[col].map(encode_dict)
    test_df[f'{col}_encoded'] = test_df[col].map(encode_dict)
    # Test setinde görünmeyen kategoriler için global mean
    test_df[f'{col}_encoded'].fillna(global_mean, inplace=True)
    
    return train_df, test_df

# Label encoding için (Tree modelleri için)
le_dict = {}
for col in cat_features:
    if col in train_fe.columns:
        le = LabelEncoder()
        combined = pd.concat([train_fe[col], test_fe[col]], axis=0).astype(str)
        le.fit(combined)
        train_fe[col + '_le'] = le.transform(train_fe[col].astype(str))
        test_fe[col + '_le'] = le.transform(test_fe[col].astype(str))
        le_dict[col] = le
```

### 2.3 Modelleme Stratejisi

**En İyi Yaklaşım: Ensemble (CatBoost + LightGBM + XGBoost)**

Balanced Accuracy için class_weight='balanced' kullanacağız:

```python
# Hedef değişkeni encode et
target_map = {'Low': 0, 'Medium': 1, 'High': 2}
reverse_map = {v: k for k, v in target_map.items()}

y = train_fe['Irrigation_Need'].map(target_map)
X = train_fe.drop(['id', 'Irrigation_Need'], axis=1)
X_test = test_fe.drop(['id'], axis=1)

# Kategorik kolonları belirle (CatBoost için)
cat_cols_catboost = [col for col in cat_features if col in X.columns]
cat_cols_lgb = [col + '_le' for col in cat_features if col + '_le' in X.columns]

# Numerik kolonlar (diğer modeller için)
num_cols = [col for col in X.columns if col not in cat_features and col not in [c + '_le' for c in cat_features]]
num_cols = [col for col in num_cols if col not in ['id']]

# Stratified K-Fold (Dengesiz veri seti için kritik!)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Modeller
models = {
    'catboost': CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        cat_features=cat_cols_catboost,
        class_weights=[1, 2, 15],  # High'a daha fazla ağırlık (az örnek olduğu için)
        random_seed=42,
        verbose=100,
        early_stopping_rounds=100
    ),
    'lgbm': LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=31,
        class_weight='balanced',  # Otomatik dengeli ağırlık
        random_state=42,
        verbose=-1
    ),
    'xgb': XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=8,
        scale_pos_weight=10,  # Dengeleme için
        random_state=42,
        eval_metric='mlogloss'
    )
}
```

### 2.4 Cross-Validation ve Eğitim

```python
oof_preds = np.zeros((len(X), 3))  # 3 sınıf için
test_preds = np.zeros((len(X_test), 3))
feature_importance = pd.DataFrame()

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*50}")
    print(f"Fold {fold + 1}")
    print(f"{'='*50}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # CatBoost için kategorik kolonları kullan
    model_cb = models['catboost'].copy()
    model_cb.fit(X_train, y_train, 
                 eval_set=(X_val, y_val),
                 verbose=False)
    
    # LightGBM için encoded kolonları kullan
    X_train_lgb = X_train[cat_cols_lgb + num_cols]
    X_val_lgb = X_val[cat_cols_lgb + num_cols]
    X_test_lgb = X_test[cat_cols_lgb + num_cols]
    
    model_lgb = models['lgbm']
    model_lgb.fit(X_train_lgb, y_train,
                  eval_set=[(X_val_lgb, y_val)],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    
    # Tahminler
    pred_cb = model_cb.predict_proba(X_val)
    pred_lgb = model_lgb.predict_proba(X_val_lgb)
    
    # Ensemble (Ağırlıklı ortalama)
    pred_ensemble = 0.6 * pred_cb + 0.4 * pred_lgb
    
    oof_preds[val_idx] = pred_ensemble
    
    # Test tahminleri
    test_cb = model_cb.predict_proba(X_test)
    test_lgb = model_lgb.predict_proba(X_test_lgb)
    test_preds += (0.6 * test_cb + 0.4 * test_lgb) / 5
    
    # Skor hesapla
    val_pred_class = np.argmax(pred_ensemble, axis=1)
    score = balanced_accuracy_score(y_val, val_pred_class)
    print(f"Fold Balanced Accuracy: {score:.4f}")
    
    # Feature importance (CatBoost)
    fold_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model_cb.feature_importances_,
        'fold': fold
    })
    feature_importance = pd.concat([feature_importance, fold_importance])

# Genel skor
final_pred_class = np.argmax(oof_preds, axis=1)
final_score = balanced_accuracy_score(y, final_pred_class)
print(f"\n{'='*50}")
print(f"OOF Balanced Accuracy: {final_score:.4f}")
print(f"{'='*50}")
```

### 2.5 Feature Importance Analizi

```python
# Ortalama importance
mean_importance = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=mean_importance.values[:20], y=mean_importance.index[:20])
plt.title('Top 20 Önemli Özellik')
plt.tight_layout()
plt.show()

print("En önemli 10 özellik:")
print(mean_importance.head(10))
```

### 2.6 Hiperparametre Optimizasyonu (Optuna ile)

Daha iyi sonuçlar için Optuna kullanarak tuning yapabilirsin:

```python
import optuna

def objective(trial):
    params = {
        'iterations': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'random_seed': 42,
        'verbose': False,
        'class_weights': [1, 2, trial.suggest_int('high_weight', 10, 30)]
    }
    
    model = CatBoostClassifier(**params)
    
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        preds = model.predict(X_val)
        scores.append(balanced_accuracy_score(y_val, preds))
    
    return np.mean(scores)

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=20)
# print(study.best_params)
```

---

## 📊 3. DETAYLI EDA (Exploratory Data Analysis)

Şimdi veriyi daha derinlemesine analiz edelim. Bu adımları notebook'unda uygula:

### 3.1 Hedef Değişken Analizi
```python
# Sınıf dengesizliğini görselleştir
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
train['Irrigation_Need'].value_counts().plot(kind='bar', color=['green', 'orange', 'red'])
plt.title('Sınıf Dağılımı (Sayı)')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
train['Irrigation_Need'].value_counts(normalize=True).plot(kind='bar', color=['green', 'orange', 'red'])
plt.title('Sınıf Dağılımı (Oran)')
plt.xticks(rotation=0)
plt.show()
```

### 3.2 Numerik Özelliklerin Dağılımı (Hedefe Göre)
```python
# Boxplotlar ile numerik özelliklerin hedefe göre dağılımı
numeric_cols = ['Soil_Moisture', 'Temperature_C', 'Rainfall_mm', 'Wind_Speed_kmh']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for idx, col in enumerate(numeric_cols):
    ax = axes[idx//2, idx%2]
    sns.boxplot(data=train, x='Irrigation_Need', y=col, ax=ax, 
                order=['Low', 'Medium', 'High'])
    ax.set_title(f'{col} vs Irrigation_Need')
    
plt.tight_layout()
plt.show()
```

**Yorum:** Soil_Moisture'da Low ile High arasında net ayrım var. Temperature_C'de de benzer şekilde.

### 3.3 Korelasyon Matrisi
```python
# Numerik kolonların korelasyonu
numeric_df = train.select_dtypes(include=[np.number])
numeric_df['Irrigation_Need_num'] = train['Irrigation_Need'].map(target_map)

plt.figure(figsize=(14, 12))
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Korelasyon Matrisi')
plt.show()
```

### 3.4 Kategorik Özellikler vs Hedef
```python
# Crop_Growth_Stage için oran tablosu
ct = pd.crosstab(train['Crop_Growth_Stage'], train['Irrigation_Need'], normalize='index')
ct.plot(kind='bar', stacked=True, figsize=(10, 6), color=['green', 'orange', 'red'])
plt.title('Crop Growth Stage vs Irrigation Need (Oransal)')
plt.legend(loc='upper right')
plt.xticks(rotation=45)
plt.show()
```

---

## 🚀 4. GELİŞMİŞ TEKNİKLER (1. Olmanı Sağlayacaklar)

### 4.1 Pseudo-Labeling (Semi-Supervised Learning)
Test setinde yüksek güvenle tahmin edilen örnekleri eğitime ekle:

```python
# Yüksek confidence'lı test tahminlerini bul
max_confidence = np.max(test_preds, axis=1)
confident_mask = max_confidence > 0.85  # %85 üzeri güven

# Pseudo labels
pseudo_labels = np.argmax(test_preds[confident_mask], axis=1)
pseudo_df = X_test[confident_mask].copy()
pseudo_df['Irrigation_Need'] = [reverse_map[x] for x in pseudo_labels]

# Eğitim setine ekle (opsiyonel, dikkatli kullan)
# X_extended = pd.concat([X, pseudo_df.drop('Irrigation_Need', axis=1)], axis=0)
# y_extended = pd.concat([y, pd.Series(pseudo_labels)], axis=0)
```

### 4.2 Threshold Optimization
Balanced Accuracy'i maksimize eden threshold'ları bul:

```python
from sklearn.metrics import balanced_accuracy_score

def find_best_thresholds(y_true, y_proba):
    """
    Her sınıf için en iyi threshold'u bul
    """
    best_score = 0
    best_thresholds = [0.5, 0.5, 0.5]  # Başlangıç
    
    # Grid search
    for t0 in np.arange(0.3, 0.7, 0.05):
        for t1 in np.arange(0.2, 0.6, 0.05):
            # 3 sınıf için thresholdlar
            thresholds = [t0, t1, 1.0]
            preds = np.zeros_like(y_proba)
            for i, thresh in enumerate(thresholds[:-1]):
                preds[:, i] = (y_proba[:, i] > thresh).astype(int)
            
            # En yüksek olanı seç
            final_preds = np.argmax(preds, axis=1)
            score = balanced_accuracy_score(y_true, final_preds)
            if score > best_score:
                best_score = score
                best_thresholds = thresholds
                
    return best_thresholds, best_score

# OOF tahminlerinde optimize et
# best_thresh, score = find_best_thresholds(y, oof_preds)
```

### 4.3 Blending (Farklı Modellerin Birleştirilmesi)
```python
# Farklı modellerin tahminlerini birleştir
final_test_preds = 0.5 * test_preds_cb + 0.3 * test_preds_lgb + 0.2 * test_preds_xgb
final_classes = np.argmax(final_test_preds, axis=1)
```

---

## 📝 5. SUBMISSION HAZIRLAMA

```python
# Son tahminleri oluştur
submission = pd.DataFrame({
    'id': test['id'],
    'Irrigation_Need': [reverse_map[x] for x in np.argmax(test_preds, axis=1)]
})

# Dağılım kontrolü
print("Submission dağılımı:")
print(submission['Irrigation_Need'].value_counts(normalize=True))

# Kaydet
submission.to_csv('submission.csv', index=False)
print("Submission hazır!")
```

---

## ⚡ 6. KAZANMA STRATEJİSİ (Özet)

**Hemen Yapılacaklar:**
1. **Feature Engineering** kısmındaki 10 yeni özelliği ekle (özellikle `Evapotranspiration_Risk` ve `Critical_Growth_Stage`)
2. **Class weights**: `[1, 2, 15]` veya `[1, 1.5, 10]` dene (High sınıfı için agresif ağırlık)
3. **CatBoost** kullan (kategorik verilerle en iyi performansı verir)
4. **Stratified 5-Fold** CV kullan (her fold'da High sınıfı eşit dağılsın)
5. **Ensemble**: CatBoost (0.6) + LightGBM (0.4)

**İkinci Aşama İyileştirmeler:**
1. Target Encoding dene (kategorik değişkenler için)
2. Pseudo-labeling ile test verisinden faydalan
3. Hyperparameter tuning (Optuna)
4. Outlier detection (Soil_Moisture > 60 ve Temperature_C > 40 gibi aykırı değerleri kontrol et)

**Dikkat Edilmesi Gerekenler:**
- **Leakage yok**: `id` kolonunu düşürdüğünden emin ol
- **Overfitting**: Early stopping kullan, iteration sayısını çok yüksek tutma
- **High sınıfı**: Confusion matrix'te High'ın recall'ına odaklan (precision'dan önce)

Başarı şansın çok yüksek! Özellikle `Soil_Moisture` ve `Temperature_C` arasındaki etkileşimi iyi modellemelisin. High sınıfı genellikle "sıcak, kuru, rüzgarlı, çiçeklenme döneminde" olanlarda çıkıyor.

Başka hangi konuda detaylı bilgi istersin? Örneğin:
- Hiperparametre optimizasyonu detayları?
- Hata analizi (hangi örnekleri yanlış tahmin ediyoruz)?
- Orijinal veri setini nasıl entegre ederiz? (Eğer varsa)