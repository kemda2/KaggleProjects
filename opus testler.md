

# 🎯 Train-Test Senaryosunda Kullanılacak Testler & Analizler

Aşağıda **baştan sona** bir supervised learning pipeline'ında hangi testin, hangi aşamada, neden kullanıldığını bulacaksınız.

---

## 📌 GENEL AKIŞ

```
TRAIN DATA                                    TEST DATA
    │                                             │
    ▼                                             │
 1. EDA & Veri Anlama                             │
    │                                             │
 2. Eksik Veri Analizi ──────── (aynı strateji) ──┤
    │                                             │
 3. Aykırı Değer Tespiti                          │
    │                                             │
 4. Feature Engineering ─────── (aynı dönüşüm) ──┤
    │                                             │
 5. Dağılım & Dönüşüm Testleri                   │
    │                                             │
 6. Korelasyon & Multicollinearity                │
    │                                             │
 7. Feature Selection Testleri                    │
    │                                             │
 8. Model Kurma & Cross-Validation                │
    │                                             │
 9. Model Varsayım Testleri                       │
    │                                             │
10. Model Karşılaştırma Testleri                  │
    │                                             │
    └───────── Final Model ──────────────────────▶ │
                                                   ▼
                                          11. Tahmin & Değerlendirme
                                                   │
                                          12. Hata Analizi
```

> ⚠️ **KRİTİK KURAL:** Tüm dönüşümler (scaling, encoding, imputation) **sadece TRAIN'den öğrenilir**, **TEST'e uygulanır**. Data leakage'dan kaçınılır.

---

## 📌 1. EDA — SADECE TRAIN ÜZERİNDE

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

# ── Temel Bakış ──
print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")
train.info()
train.describe(include='all')

# ── Target Dağılımı (SADECE TRAIN'DE VAR) ──
# Regression ise:
train['target'].hist(bins=50)
print(f"Skewness: {train['target'].skew():.4f}")
print(f"Kurtosis: {train['target'].kurtosis():.4f}")

# Classification ise:
print(train['target'].value_counts(normalize=True))
train['target'].value_counts().plot(kind='bar')
# ─── Dengesizlik oranı ───
imbalance_ratio = train['target'].value_counts().min() / train['target'].value_counts().max()
print(f"Imbalance Ratio: {imbalance_ratio:.4f}")

# ── Train vs Test Dağılım Karşılaştırması ──
# (Data drift / covariate shift kontrolü)
common_cols = [c for c in train.columns if c in test.columns and c != 'target']
for col in common_cols:
    if train[col].dtype in ['int64','float64']:
        stat, p = stats.ks_2samp(train[col].dropna(), test[col].dropna())
        if p < 0.05:
            print(f"⚠️  {col}: Train-Test dağılımı FARKLI (KS p={p:.4f})")
```

---

## 📌 2. EKSİK VERİ ANALİZİ

```python
import missingno as msno

# ── Eksiklik Oranları ──
def missing_report(df, name=""):
    missing = df.isnull().sum()
    percent = df.isnull().mean() * 100
    report = pd.DataFrame({'Missing': missing, '%': percent})
    report = report[report['Missing'] > 0].sort_values('%', ascending=False)
    print(f"\n{'='*40} {name} {'='*40}")
    print(report)
    return report

missing_train = missing_report(train, "TRAIN")
missing_test  = missing_report(test, "TEST")

# ── Eksiklik Paterni (TRAIN) ──
msno.matrix(train)
msno.heatmap(train)  # eksik değerler arası korelasyon

# ── Eksiklik Target ile İlişkili mi? (MAR kontrolü) ──
for col in train.columns[train.isnull().any()]:
    if train['target'].dtype in ['int64','float64']:
        group_missing    = train[train[col].isnull()]['target']
        group_notmissing = train[train[col].notna()]['target']
        if len(group_missing) > 1:
            stat, p = stats.mannwhitneyu(group_missing, group_notmissing, 
                                          alternative='two-sided')
            if p < 0.05:
                print(f"⚠️  {col} eksikliği target ile İLİŞKİLİ (p={p:.4f})")

# ── Imputation (TRAIN'den öğren, TEST'e uygula) ──
from sklearn.impute import SimpleImputer, KNNImputer

# Sayısal
num_imputer = SimpleImputer(strategy='median')  # veya KNNImputer
num_imputer.fit(train[num_cols])                  # SADECE TRAIN
train[num_cols] = num_imputer.transform(train[num_cols])
test[num_cols]  = num_imputer.transform(test[num_cols])  # AYNI DÖNÜŞÜM

# Kategorik
cat_imputer = SimpleImputer(strategy='most_frequent')
cat_imputer.fit(train[cat_cols])
train[cat_cols] = cat_imputer.transform(train[cat_cols])
test[cat_cols]  = cat_imputer.transform(test[cat_cols])
```

---

## 📌 3. AYKIRI DEĞER TESPİTİ — SADECE TRAIN

> Test verisine dokunma! Gerçek dünyayı temsil ediyor.

```python
# ── IQR ──
def detect_outliers_iqr(df, col, k=1.5):
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - k*IQR, Q3 + k*IQR
    mask = (df[col] < lower) | (df[col] > upper)
    return mask, lower, upper

for col in num_cols:
    mask, lo, hi = detect_outliers_iqr(train, col)
    pct = mask.mean() * 100
    if pct > 0:
        print(f"{col}: {pct:.1f}% aykırı [{lo:.2f}, {hi:.2f}]")

# ── Isolation Forest (çok değişkenli) ──
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.05, random_state=42)
train['outlier'] = iso.fit_predict(train[num_cols])
print(f"Aykırı: {(train['outlier']==-1).sum()}")
# Kararlar: kırp / çıkar / dönüştür / bırak

# ── Aykırı Değer Kararı ve Etkisi ──
# Aykırı değerler çıkarılmadan/çıkarıldıktan sonra target ilişkisi kontrol et
```

---

## 📌 4. DAĞILIM & DÖNÜŞÜM TESTLERİ — TRAIN

> **Neden?** → Bazı modeller (Linear Reg, LDA) normal dağılım ister. Tree-based modeller umursamaz.

```python
from scipy.stats import shapiro, skew, boxcox, yeojohnson

# ── Hangi Feature'lar Skewed? ──
skewed_features = []
for col in num_cols:
    sk = train[col].skew()
    if abs(sk) > 1:
        skewed_features.append((col, sk))
        print(f"{col}: skewness = {sk:.4f} → Dönüşüm gerekebilir")

# ── Dönüşüm Uygula (TRAIN'den parametreleri öğren) ──
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson')  # negatif değerler de OK
pt.fit(train[num_cols])                        # SADECE TRAIN
train[num_cols] = pt.transform(train[num_cols])
test[num_cols]  = pt.transform(test[num_cols])  # AYNI PARAMETRE

# ── Target Dönüşümü (Regression) ──
# Target çarpıksa → log dönüşümü
if abs(train['target'].skew()) > 1:
    print(f"Target skew: {train['target'].skew():.4f} → Log dönüşümü önerilir")
    train['target_log'] = np.log1p(train['target'])
    # Tahmin sonrası: np.expm1(predictions)
```

---

## 📌 5. KORELASYON & ÇOKLİ DOĞRUSALLIK — TRAIN

```python
# ── Feature-Target Korelasyonu ──
if train['target'].dtype in ['int64', 'float64']:
    corr_with_target = train[num_cols + ['target']].corr()['target'].drop('target')
    corr_with_target = corr_with_target.abs().sort_values(ascending=False)
    print(corr_with_target)

    # Anlamlılık testleri
    for col in num_cols:
        r, p = stats.pearsonr(train[col].dropna(), 
                               train.loc[train[col].notna(), 'target'])
        if p < 0.05:
            print(f"✅ {col}: r={r:.4f}, p={p:.4f} → Anlamlı")
        else:
            print(f"❌ {col}: r={r:.4f}, p={p:.4f} → Anlamsız")

# ── Feature-Feature Korelasyonu (Multicollinearity) ──
corr_matrix = train[num_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [(col, row, corr_matrix.loc[row, col]) 
             for col in upper.columns 
             for row in upper.index 
             if upper.loc[row, col] > 0.85]
print("Yüksek korelasyonlu çiftler:")
for c1, c2, r in high_corr:
    print(f"  {c1} ↔ {c2}: r={r:.4f}")

# ── VIF (Variance Inflation Factor) ──
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

X_vif = sm.add_constant(train[num_cols].dropna())
vif = pd.DataFrame()
vif['Feature'] = X_vif.columns
vif['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
vif = vif.sort_values('VIF', ascending=False)
print(vif)
# VIF > 5 → sorun var, VIF > 10 → ciddi → Feature çıkar veya PCA uygula
```

---

## 📌 6. FEATURE SELECTION TESTLERİ — TRAIN

### 6.1 İstatistiksel Testler ile Feature Selection

| Target Tipi | Feature Tipi | Test |
|-------------|-------------|------|
| Sürekli | Sürekli | **Pearson / Spearman Korelasyon** |
| Sürekli | Kategorik | **ANOVA F-test / Kruskal-Wallis** |
| Kategorik | Sürekli | **ANOVA F-test / Mann-Whitney** |
| Kategorik | Kategorik | **Chi-Square / Fisher / Cramér's V** |

```python
from sklearn.feature_selection import (
    f_classif, f_regression, chi2, 
    mutual_info_classif, mutual_info_regression,
    SelectKBest, SelectFromModel, RFE
)

# ══════════════════════════════════════════
# REGRESSION İÇİN
# ══════════════════════════════════════════

# ── F-Test (Lineer ilişki) ──
selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(train[num_cols], train['target'])
f_scores = pd.DataFrame({
    'Feature': num_cols,
    'F-Score': selector.scores_,
    'p-value': selector.pvalues_
}).sort_values('F-Score', ascending=False)
print(f_scores)

# ── Mutual Information (Non-lineer ilişki) ──
mi_scores = mutual_info_regression(train[num_cols], train['target'], random_state=42)
mi_df = pd.DataFrame({
    'Feature': num_cols, 
    'MI': mi_scores
}).sort_values('MI', ascending=False)
print(mi_df)

# ══════════════════════════════════════════
# CLASSIFICATION İÇİN
# ══════════════════════════════════════════

# ── ANOVA F-test (sayısal feature → kategorik target) ──
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(train[num_cols], train['target'])
print(pd.DataFrame({
    'Feature': num_cols,
    'F': selector.scores_,
    'p': selector.pvalues_
}).sort_values('F', ascending=False))

# ── Chi-Square (kategorik feature → kategorik target) ──
# Sadece pozitif, kategorik/encoded değerler için
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(train[num_cols])
chi2_scores, chi2_pvalues = chi2(X_scaled, train['target'])

# ── Mutual Information (Classification) ──
mi = mutual_info_classif(train[num_cols], train['target'], random_state=42)

# ── Kategorik Feature vs Kategorik Target: Chi-Square Bağımsızlık ──
for col in cat_cols:
    contingency = pd.crosstab(train[col], train['target'])
    chi2_stat, p, dof, expected = stats.chi2_contingency(contingency)
    cramers = np.sqrt(chi2_stat / (len(train) * (min(contingency.shape) - 1)))
    print(f"{col}: Chi2={chi2_stat:.2f}, p={p:.4f}, Cramér's V={cramers:.4f}")
```

### 6.2 Model Bazlı Feature Selection

```python
# ── RFE (Recursive Feature Elimination) ──
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Regression
estimator = RandomForestRegressor(n_estimators=100, random_state=42)
rfe = RFE(estimator, n_features_to_select=10, step=1)
rfe.fit(train[num_cols], train['target'])
selected = [col for col, rank in zip(num_cols, rfe.ranking_) if rank == 1]
print(f"RFE Selected: {selected}")

# ── Feature Importance ──
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train[num_cols], train['target'])
importance = pd.DataFrame({
    'Feature': num_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print(importance)
importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10,8))

# ── Permutation Importance (Daha güvenilir) ──
from sklearn.inspection import permutation_importance
perm = permutation_importance(model, train[num_cols], train['target'],
                               n_repeats=10, random_state=42)
perm_df = pd.DataFrame({
    'Feature': num_cols,
    'Importance_Mean': perm.importances_mean,
    'Importance_Std':  perm.importances_std
}).sort_values('Importance_Mean', ascending=False)

# ── LASSO (L1) ile Feature Selection ──
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5, random_state=42).fit(train[num_cols], train['target'])
lasso_coef = pd.DataFrame({
    'Feature': num_cols,
    'Coef': np.abs(lasso.coef_)
}).sort_values('Coef', ascending=False)
eliminated = lasso_coef[lasso_coef['Coef'] == 0]['Feature'].tolist()
print(f"LASSO eliminated: {eliminated}")

# ── Boruta (Wrapper Method) ──
from boruta import BorutaPy
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
boruta = BorutaPy(rf, n_estimators='auto', random_state=42, max_iter=100)
boruta.fit(train[num_cols].values, train['target'].values)
boruta_selected = [col for col, s in zip(num_cols, boruta.support_) if s]
print(f"Boruta Selected: {boruta_selected}")
```

---

## 📌 7. SCALING & ENCODING — TRAIN'DEN ÖĞREN

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ══ ÖNEMLİ: Pipeline Kullanarak Data Leakage Önleme ══

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

# Pipeline'ı train'e fit et
preprocessor.fit(train[num_cols + cat_cols])

# Her ikisine de transform uygula
X_train = preprocessor.transform(train[num_cols + cat_cols])
X_test  = preprocessor.transform(test[num_cols + cat_cols])
y_train = train['target']
```

---

## 📌 8. CLASS IMBALANCE (SADECE CLASSİFİCATION) — TRAIN

```python
# ── Dengesizlik kontrolü ──
print(train['target'].value_counts(normalize=True))

# Eğer dengesiz ise:
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# SADECE TRAIN'E UYGULA (cross-validation içinde)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print(f"Before: {dict(pd.Series(y_train).value_counts())}")
print(f"After:  {dict(pd.Series(y_resampled).value_counts())}")

# VEYA class_weight kullan (daha temiz)
# model = RandomForestClassifier(class_weight='balanced')
```

---

## 📌 9. CROSS-VALIDATION TESTLERİ — TRAIN

> **En kritik adım:** Modelin gerçek performansını TRAIN içinde ölç.

```python
from sklearn.model_selection import (
    cross_val_score, cross_validate, 
    StratifiedKFold, KFold, RepeatedKFold,
    RepeatedStratifiedKFold, TimeSeriesSplit, 
    GroupKFold, LeaveOneOut,
    learning_curve, validation_curve
)

# ══════════════════════════════════════════
# CV STRATEJİSİ SEÇİMİ
# ══════════════════════════════════════════
# Regression         → KFold(n_splits=5, shuffle=True)
# Classification     → StratifiedKFold(n_splits=5, shuffle=True) 
# Zaman serisi       → TimeSeriesSplit(n_splits=5)
# Gruplu veri        → GroupKFold(n_splits=5)
# Küçük veri         → RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
# Çok küçük veri     → LeaveOneOut()

# ── Regression CV ──
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR

models = {
    'LinearReg': LinearRegression(),
    'Ridge':     Ridge(alpha=1.0),
    'Lasso':     Lasso(alpha=0.1),
    'RF':        RandomForestRegressor(n_estimators=100, random_state=42),
    'GBM':       GradientBoostingRegressor(n_estimators=100, random_state=42)
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}
for name, model in models.items():
    scores = cross_validate(model, X_train, y_train, cv=cv,
                            scoring=['neg_mean_squared_error', 
                                     'neg_mean_absolute_error', 'r2'],
                            return_train_score=True)
    results[name] = {
        'Train_RMSE': np.sqrt(-scores['train_neg_mean_squared_error'].mean()),
        'Val_RMSE':   np.sqrt(-scores['test_neg_mean_squared_error'].mean()),
        'Val_RMSE_std': np.sqrt(-scores['test_neg_mean_squared_error']).std(),
        'Val_MAE':    -scores['test_neg_mean_absolute_error'].mean(),
        'Val_R2':      scores['test_r2'].mean(),
        'Overfit':    (np.sqrt(-scores['train_neg_mean_squared_error'].mean()) - 
                       np.sqrt(-scores['test_neg_mean_squared_error'].mean()))
    }
    
results_df = pd.DataFrame(results).T.sort_values('Val_RMSE')
print(results_df)

# ── Classification CV ──
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_ovr']

for name, model in models.items():
    scores = cross_validate(model, X_train, y_train, cv=cv,
                            scoring=scoring, return_train_score=True)
    print(f"\n{name}:")
    print(f"  Accuracy:  {scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}")
    print(f"  F1:        {scores['test_f1_macro'].mean():.4f}")
    print(f"  AUC:       {scores['test_roc_auc_ovr'].mean():.4f}")
```

---

## 📌 10. LEARNING CURVE & VALIDATION CURVE

```python
# ── Learning Curve (Overfitting / Underfitting Tespiti) ──
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)

train_rmse = np.sqrt(-train_scores.mean(axis=1))
val_rmse   = np.sqrt(-val_scores.mean(axis=1))

plt.figure(figsize=(10,6))
plt.plot(train_sizes, train_rmse, 'o-', label='Train RMSE')
plt.plot(train_sizes, val_rmse, 'o-', label='Validation RMSE')
plt.xlabel('Training Set Size')
plt.ylabel('RMSE')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.show()

# YORUM:
# - Train ve Val birbirine yakın & yüksek → Underfitting (High Bias)
# - Train düşük & Val yüksek → Overfitting (High Variance)
# - İkisi de düşük & birbirine yakın → İdeal

# ── Validation Curve (Hiperparametre Etkisi) ──
from sklearn.model_selection import validation_curve

param_range = [10, 50, 100, 200, 500]
train_scores, val_scores = validation_curve(
    RandomForestRegressor(random_state=42),
    X_train, y_train,
    param_name='n_estimators',
    param_range=param_range,
    cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)

plt.plot(param_range, np.sqrt(-train_scores.mean(axis=1)), 'o-', label='Train')
plt.plot(param_range, np.sqrt(-val_scores.mean(axis=1)), 'o-', label='Validation')
plt.xlabel('n_estimators')
plt.ylabel('RMSE')
plt.title('Validation Curve')
plt.legend()
plt.show()
```

---

## 📌 11. MODEL VARSAYIM TESTLERİ (Lineer Modeller İçin)

> Tree-based modeller için bu testler **gerekmez**. Linear/Logistic Regression için **zorunludur**.

```python
import statsmodels.api as sm
from statsmodels.stats.diagnostic import (
    het_breuschpagan, het_white, linear_reset, 
    acorr_breusch_godfrey
)
from statsmodels.stats.stattools import durbin_watson

# ── Lineer Regresyon Fit ──
X_sm = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_sm).fit()
residuals = ols_model.resid
fitted = ols_model.fittedvalues

# ╔══════════════════════════════════════════════╗
# ║  TEST 1: Kalıntıların Normalliği             ║
# ╚══════════════════════════════════════════════╝
stat, p = stats.shapiro(residuals[:5000])  # shapiro max 5000
print(f"Shapiro-Wilk (Residuals): p={p:.4f}")

stat, p = stats.jarque_bera(residuals)
print(f"Jarque-Bera (Residuals): p={p:.4f}")

sm.qqplot(residuals, line='s')
plt.title("Residuals Q-Q Plot")
plt.show()

# ╔══════════════════════════════════════════════╗
# ║  TEST 2: Heteroskedastisite                  ║
# ╚══════════════════════════════════════════════╝
bp_stat, bp_p, _, _ = het_breuschpagan(residuals, ols_model.model.exog)
print(f"Breusch-Pagan: p={bp_p:.4f} → {'Homojen ✅' if bp_p>0.05 else 'Heterojen ❌'}")

white_stat, white_p, _, _ = het_white(residuals, ols_model.model.exog)
print(f"White Test: p={white_p:.4f}")

# Heteroskedastisite varsa → WLS veya Robust SE kullan
# robust_model = ols_model.get_robustcov_results(cov_type='HC3')

# ╔══════════════════════════════════════════════╗
# ║  TEST 3: Otokorelasyon                        ║
# ╚══════════════════════════════════════════════╝
dw = durbin_watson(residuals)
print(f"Durbin-Watson: {dw:.4f} → {'OK ✅' if 1.5<dw<2.5 else 'Otokorelasyon ❌'}")

bg = acorr_breusch_godfrey(ols_model, nlags=5)
print(f"Breusch-Godfrey: p={bg[1]:.4f}")

# ╔══════════════════════════════════════════════╗
# ║  TEST 4: Doğrusallık                          ║
# ╚══════════════════════════════════════════════╝
reset = linear_reset(ols_model, power=3)
print(f"Ramsey RESET: p={reset.pvalue:.4f} → {'Doğrusal ✅' if reset.pvalue>0.05 else 'Non-linear ❌'}")

# Residuals vs Fitted plot
plt.scatter(fitted, residuals, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.show()

# ╔══════════════════════════════════════════════╗
# ║  TEST 5: Etkili Gözlemler (Influential Obs)  ║
# ╚══════════════════════════════════════════════╝
influence = ols_model.get_influence()
cooks_d = influence.cooks_distance[0]
leverage = influence.hat_matrix_diag

n_influential = (cooks_d > 4/len(y_train)).sum()
print(f"Cook's Distance > 4/n: {n_influential} gözlem")
```

---

## 📌 12. HİPERPARAMETRE OPTİMİZASYONU — TRAIN

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import optuna

# ── GridSearch ──
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid, cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best RMSE: {-grid.best_score_:.4f}")

# ── Optuna (Daha akıllı) ──
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    }
    model = GradientBoostingRegressor(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5,
                             scoring='neg_root_mean_squared_error')
    return -scores.mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print(f"Best params: {study.best_params}")
print(f"Best RMSE: {study.best_value:.4f}")
```

---

## 📌 13. MODEL KARŞILAŞTIRMA İSTATİSTİKSEL TESTLERİ

> **Önemli:** "Model A, Model B'den gerçekten daha mı iyi, yoksa tesadüf mü?"

```python
# ══════════════════════════════════════════
# 13.1 Paired t-test (CV skorları üzerinde)
# ══════════════════════════════════════════
from scipy.stats import ttest_rel, wilcoxon

cv = KFold(n_splits=10, shuffle=True, random_state=42)

scores_A = cross_val_score(model_A, X_train, y_train, cv=cv,
                           scoring='neg_root_mean_squared_error')
scores_B = cross_val_score(model_B, X_train, y_train, cv=cv, 
                           scoring='neg_root_mean_squared_error')

# Normallik kontrolü
stat, p_norm = stats.shapiro(scores_A - scores_B)
if p_norm > 0.05:
    # Parametrik
    stat, p = ttest_rel(scores_A, scores_B)
    print(f"Paired t-test: t={stat:.4f}, p={p:.4f}")
else:
    # Non-parametrik
    stat, p = wilcoxon(scores_A, scores_B)
    print(f"Wilcoxon: stat={stat:.4f}, p={p:.4f}")

# ══════════════════════════════════════════
# 13.2 McNemar Test (Classification: hata karşılaştırma)
# ══════════════════════════════════════════
from statsmodels.stats.contingency_tables import mcnemar

pred_A = model_A.predict(X_val)
pred_B = model_B.predict(X_val)

# Doğru/Yanlış matrisi
both_correct   = ((pred_A == y_val) & (pred_B == y_val)).sum()
A_correct_only = ((pred_A == y_val) & (pred_B != y_val)).sum()
B_correct_only = ((pred_A != y_val) & (pred_B == y_val)).sum()
both_wrong     = ((pred_A != y_val) & (pred_B != y_val)).sum()

table = [[both_correct, A_correct_only],
         [B_correct_only, both_wrong]]
result = mcnemar(table, exact=False, correction=True)
print(f"McNemar p={result.pvalue:.4f}")

# ══════════════════════════════════════════
# 13.3 Corrected Paired t-test (Nadeau & Bengio)
# ══════════════════════════════════════════
# Normal paired t-test CV'de optimistik olabilir
# Düzeltme: test_size/train_size oranını hesaba kat
def corrected_paired_ttest(scores1, scores2, n_train, n_test):
    diff = scores1 - scores2
    n = len(diff)
    mean_diff = diff.mean()
    var_diff = diff.var()
    correction = (1/n + n_test/n_train)
    t_stat = mean_diff / np.sqrt(correction * var_diff)
    p_value = 2 * stats.t.sf(abs(t_stat), df=n-1)
    return t_stat, p_value

# ══════════════════════════════════════════
# 13.4 Friedman + Nemenyi (3+ model karşılaştırma)
# ══════════════════════════════════════════
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

# Her modelin CV skorları
all_scores = np.column_stack([scores_A, scores_B, scores_C])
stat, p = friedmanchisquare(*all_scores.T)
print(f"Friedman: chi2={stat:.4f}, p={p:.4f}")

if p < 0.05:
    # Post-hoc Nemenyi
    nemenyi = sp.posthoc_nemenyi_friedman(all_scores)
    print(nemenyi)

# ══════════════════════════════════════════
# 13.5 Cochran's Q (Classification: çoklu model)
# ══════════════════════════════════════════
from mlxtend.evaluate import cochrans_q, mcnemar_table
stat, p = cochrans_q(y_val, pred_A, pred_B, pred_C)
```

---

## 📌 14. FİNAL MODEL → TEST TAHMİNİ & DEĞERLENDİRME

### 14.1 REGRESSION Metrikleri

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error,
    explained_variance_score, max_error
)

# Final modeli tüm TRAIN ile eğit
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

# ── Metrikler ──
metrics = {
    'RMSE':  np.sqrt(mean_squared_error(y_test, y_pred)),
    'MAE':   mean_absolute_error(y_test, y_pred),
    'MAPE':  mean_absolute_percentage_error(y_test, y_pred) * 100,
    'MedAE': median_absolute_error(y_test, y_pred),
    'R²':    r2_score(y_test, y_pred),
    'Adj_R²': 1 - (1-r2_score(y_test, y_pred)) * (len(y_test)-1) / (len(y_test)-X_test.shape[1]-1),
    'Explained_Var': explained_variance_score(y_test, y_pred),
    'Max_Error': max_error(y_test, y_pred)
}

for name, val in metrics.items():
    print(f"{name:15s}: {val:.4f}")

# ── Residual Analizi (Test) ──
residuals = y_test - y_pred

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Actual vs Predicted
axes[0,0].scatter(y_test, y_pred, alpha=0.3)
axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0,0].set_xlabel('Actual')
axes[0,0].set_ylabel('Predicted')
axes[0,0].set_title('Actual vs Predicted')

# 2. Residuals Distribution
axes[0,1].hist(residuals, bins=50, edgecolor='black')
axes[0,1].set_title('Residuals Distribution')

# 3. Residuals vs Predicted
axes[1,0].scatter(y_pred, residuals, alpha=0.3)
axes[1,0].axhline(y=0, color='r', linestyle='--')
axes[1,0].set_title('Residuals vs Predicted')

# 4. Q-Q Plot
sm.qqplot(residuals, line='s', ax=axes[1,1])
axes[1,1].set_title('Residuals Q-Q')

plt.tight_layout()
plt.show()
```

### 14.2 CLASSIFICATION Metrikleri

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    log_loss, cohen_kappa_score, matthews_corrcoef,
    average_precision_score, brier_score_loss
)

y_pred = final_model.predict(X_test)
y_prob = final_model.predict_proba(X_test)

# ── Tüm Metrikler ──
print(classification_report(y_test, y_pred, digits=4))

metrics_cls = {
    'Accuracy':     accuracy_score(y_test, y_pred),
    'Precision':    precision_score(y_test, y_pred, average='weighted'),
    'Recall':       recall_score(y_test, y_pred, average='weighted'),
    'F1':           f1_score(y_test, y_pred, average='weighted'),
    'Cohen_Kappa':  cohen_kappa_score(y_test, y_pred),
    'MCC':          matthews_corrcoef(y_test, y_pred),
    'Log_Loss':     log_loss(y_test, y_prob),
}

# Binary classification ek metrikler
if len(np.unique(y_test)) == 2:
    metrics_cls['AUC']   = roc_auc_score(y_test, y_prob[:, 1])
    metrics_cls['Brier'] = brier_score_loss(y_test, y_prob[:, 1])
    metrics_cls['AP']    = average_precision_score(y_test, y_prob[:, 1])

for name, val in metrics_cls.items():
    print(f"{name:15s}: {val:.4f}")

# ── Confusion Matrix ──
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ── ROC Curve ──
if len(np.unique(y_test)) == 2:
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_prob[:,1]):.4f}')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

# ── Precision-Recall Curve (Dengesiz veri için ROC'dan daha bilgilendirici) ──
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob[:, 1])
    plt.plot(recall_vals, precision_vals)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

# ── Optimal Threshold Bulma ──
    from sklearn.metrics import f1_score
    best_threshold = 0.5
    best_f1 = 0
    for t in np.arange(0.1, 0.9, 0.01):
        y_pred_t = (y_prob[:, 1] >= t).astype(int)
        f1 = f1_score(y_test, y_pred_t)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    print(f"Optimal Threshold: {best_threshold:.2f}, F1: {best_f1:.4f}")
```

---

## 📌 15. OVERFITTING / GENERALIZATION TESTLERİ

```python
# ══════════════════════════════════════════
# 15.1 Train vs Test Performans Karşılaştırması
# ══════════════════════════════════════════

train_pred = final_model.predict(X_train)
test_pred  = final_model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse  = np.sqrt(mean_squared_error(y_test, test_pred))

overfit_ratio = test_rmse / train_rmse
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE:  {test_rmse:.4f}")
print(f"Ratio:      {overfit_ratio:.4f}")
print(f"Overfit:    {'EVET ⚠️' if overfit_ratio > 1.2 else 'HAYIR ✅'}")

# ══════════════════════════════════════════
# 15.2 CV Variance Kontrolü
# ══════════════════════════════════════════
cv_scores = cross_val_score(final_model, X_train, y_train, cv=10,
                            scoring='neg_root_mean_squared_error')
print(f"CV RMSE: {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"CV Coefficient of Variation: {cv_scores.std()/(-cv_scores.mean()):.4f}")
# CoV > 0.2 ise instabilite var

# ══════════════════════════════════════════
# 15.3 Bias-Variance Decomposition
# ══════════════════════════════════════════
from mlxtend.evaluate import bias_variance_decomp

avg_loss, avg_bias, avg_var = bias_variance_decomp(
    final_model, X_train, y_train.values, X_test, y_test.values,
    loss='mse', num_rounds=200, random_seed=42
)
print(f"Average Loss:     {avg_loss:.4f}")
print(f"Average Bias²:    {avg_bias:.4f}")
print(f"Average Variance: {avg_var:.4f}")
```

---

## 📌 16. MODEL AÇIKLANABILIRLIK (XAI)

```python
import shap

# ── SHAP Values ──
explainer = shap.TreeExplainer(final_model)  # Tree-based için
shap_values = explainer.shap_values(X_test)

# Global Feature Importance
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Tek bir tahmin açıklama
shap.force_plot(explainer.expected_value, shap_values[0], X_test[0],
                feature_names=feature_names)

# Beeswarm Plot
shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                  plot_type='dot')

# ── Partial Dependence Plot ──
from sklearn.inspection import PartialDependenceDisplay
PartialDependenceDisplay.from_estimator(final_model, X_train, 
                                         features=[0, 1, (0, 1)],
                                         feature_names=feature_names)
plt.show()
```

---

## 📌 17. TAHMİN GÜVEN ARALIĞI

```python
# ── Bootstrap Prediction Interval ──
def bootstrap_prediction_interval(model, X_train, y_train, X_new, 
                                   n_bootstrap=1000, alpha=0.05):
    predictions = []
    n = len(X_train)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        model_b = model.__class__(**model.get_params())
        model_b.fit(X_train[idx], y_train.values[idx])
        pred = model_b.predict(X_new)
        predictions.append(pred)
    predictions = np.array(predictions)
    lower = np.percentile(predictions, 100*alpha/2, axis=0)
    upper = np.percentile(predictions, 100*(1-alpha/2), axis=0)
    mean  = predictions.mean(axis=0)
    return mean, lower, upper

mean_pred, lower, upper = bootstrap_prediction_interval(
    final_model, X_train, y_train, X_test[:5]
)
for i in range(5):
    print(f"Pred: {mean_pred[i]:.2f}  CI: [{lower[i]:.2f}, {upper[i]:.2f}]")

# ── Conformal Prediction (Daha modern yaklaşım) ──
from mapie.regression import MapieRegressor
mapie = MapieRegressor(final_model, cv=5)
mapie.fit(X_train, y_train)
y_pred, y_pis = mapie.predict(X_test, alpha=0.05)
# y_pis[:, 0, 0] → lower bound
# y_pis[:, 1, 0] → upper bound
```

---

## 📌 18. CALIBRATION (Olasılık Kalibrasyonu — Classification)

```python
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# ── Kalibrasyon Eğrisi ──
prob_true, prob_pred = calibration_curve(y_test, y_prob[:, 1], n_bins=10)
plt.plot(prob_pred, prob_true, 'o-', label='Model')
plt.plot([0,1], [0,1], 'r--', label='Perfect')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend()
plt.show()

# ── Brier Score ──
from sklearn.metrics import brier_score_loss
brier = brier_score_loss(y_test, y_prob[:, 1])
print(f"Brier Score: {brier:.4f}")  # 0'a yakın → iyi

# ── Kalibrasyon Düzeltmesi (gerekirse) ──
cal_model = CalibratedClassifierCV(final_model, method='isotonic', cv=5)
cal_model.fit(X_train, y_train)
y_prob_cal = cal_model.predict_proba(X_test)
```

---

## 📌 19. DATA LEAKAGE KONTROLÜ

```python
# ╔══════════════════════════════════════════════╗
# ║        DATA LEAKAGE KONTROL LİSTESİ          ║
# ╚══════════════════════════════════════════════╝

# ✅ 1. Feature'lar arasında target'tan türetilmiş var mı?
for col in train.columns:
    if col != 'target':
        r, p = stats.pearsonr(train[col].dropna(), 
                               train.loc[train[col].notna(), 'target'])
        if abs(r) > 0.95:
            print(f"⚠️  LEAKAGE ŞÜPHESİ: {col} → r={r:.4f}")

# ✅ 2. Scaling/Encoding sadece train'den mi öğrenildi?
#    → Pipeline kullan (yukarıda gösterildi)

# ✅ 3. Feature selection CV içinde mi yapıldı?
#    → Doğrusu: Pipeline + CV birlikte

# ✅ 4. Zaman bazlı veri varsa gelecek bilgisi sızmış mı?
#    → TimeSeriesSplit kullan

# ✅ 5. Duplicate/near-duplicate satırlar train-test'e dağılmış mı?
common = pd.merge(train, test, how='inner', on=list(test.columns))
print(f"Train-Test overlap: {len(common)} satır")

# ── Doğru Pipeline (Leakage-free) ──
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selector', SelectKBest(f_regression, k=20)),
    ('model', GradientBoostingRegressor(random_state=42))
])

# CV skorları leakage-free
cv_scores = cross_val_score(pipeline, train.drop('target', axis=1), 
                            train['target'], cv=5,
                            scoring='neg_root_mean_squared_error')
```

---

## 📌 20. FULL PIPELINE — HEPSİ BİR ARADA

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

# ── Kolon Tanımları ──
num_cols = train.select_dtypes(include=np.number).columns.drop('target').tolist()
cat_cols = train.select_dtypes(include='object').columns.tolist()

# ── Preprocessing Pipeline ──
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# ── Full Pipeline ──
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    ))
])

# ── Cross-Validation (Leakage-Free) ──
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(full_pipeline, 
                            train.drop('target', axis=1), train['target'],
                            cv=cv, scoring='neg_root_mean_squared_error')
print(f"CV RMSE: {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Final Fit & Predict ──
full_pipeline.fit(train.drop('target', axis=1), train['target'])
test_predictions = full_pipeline.predict(test)

# ── Submission ──
submission = pd.DataFrame({
    'id': test['id'],
    'target': test_predictions
})
submission.to_csv('submission.csv', index=False)
```

---

## 📌 ÖZET: HANGİ AŞAMADA HANGİ TEST

| # | Aşama | Testler | Uygulandığı Veri |
|---|-------|---------|------------------|
| 1 | EDA | KS-2samp (train vs test drift) | Train + Test |
| 2 | Eksik Veri | MAR kontrolü (t-test, chi-sq) | Train |
| 3 | Aykırı Değer | IQR, Isolation Forest, Mahalanobis | Train |
| 4 | Normallik | Shapiro, Skewness, Q-Q Plot | Train (feature & target) |
| 5 | Dönüşüm | Box-Cox, Yeo-Johnson | Train → Test'e uygula |
| 6 | Korelasyon | Pearson, Spearman, VIF | Train |
| 7 | Feature Selection | F-test, MI, Chi-sq, LASSO, RFE, Boruta | Train (CV içinde) |
| 8 | Model Varsayımları | Breusch-Pagan, DW, RESET, Cook's D | Train (lineer model) |
| 9 | CV | KFold, StratifiedKFold | Train |
| 10 | Overfit Kontrolü | Learning Curve, Train/Test gap, Bias-Var | Train |
| 11 | Model Karşılaştırma | Paired t-test, Wilcoxon, Friedman, McNemar | Train (CV skorları) |
| 12 | Final Değerlendirme | RMSE, MAE, R², F1, AUC, Confusion Matrix | Test |
| 13 | Hata Analizi | Residual plots, QQ, SHAP | Test |
| 14 | Güven Aralığı | Bootstrap PI, Conformal Prediction | Test |
| 15 | Kalibrasyon | Calibration curve, Brier score | Test (classification) |
| 16 | Leakage Kontrolü | Pipeline bütünlüğü, korelasyon kontrolü | Train + Test |

> 🎯 **Altın Kural:** Train'den öğren → Test'e uygula → Asla tersi olmasın!