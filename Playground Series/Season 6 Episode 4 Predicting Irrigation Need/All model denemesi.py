import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

from scipy.stats import ks_2samp

import missingno as msno

import warnings
warnings.filterwarnings('ignore')

import shap

import catboost as cat
import xgboost as xgb
import lightgbm as lgb

# ========== KONFİGÜRASYON ==========
path          = "/kaggle/input/competitions/playground-series-s6e4/"
original_path = "/kaggle/input/datasets/miadul/irrigation-water-requirement-prediction-dataset/irrigation_prediction.csv"
target        = "Irrigation_Need"
id_col        = "id"
drop          =[]

y_map = {"Low":0, "Medium":1, "High":2}
n_classes = 3

tr_cols = {
    "Soil_Type": "Toprak_Tipi",
    "Soil_pH": "Toprak_pH",
    "Soil_Moisture": "Toprak_Nem_orani",
    "Organic_Carbon": "Organik_Karbon",
    "Electrical_Conductivity": "Elektrik_Iletkenligi",
    "Temperature_C": "Sicaklik_C",
    "Humidity": "Nem",
    "Rainfall_mm": "Yagis_mm",
    "Sunlight_Hours": "Guneslenme_Saati",
    "Wind_Speed_kmh": "Ruzgar_Hizi_kmh",
    "Crop_Type": "Mahsul_Tipi",
    "Crop_Growth_Stage": "Mahsul_Gelisim_Asamasi",
    "Season": "Mevsim",
    "Irrigation_Type": "Sulama_Tipi",
    "Water_Source": "Su_Kaynagi",
    "Field_Area_hectare": "Tarla_Alani_hektar",
    "Mulching_Used": "Malc_Kullanimi",
    "Previous_Irrigation_mm": "Onceki_Sulama_mm",
    "Region": "Bolge"
}

# ========== FEATURE ENGINEERING ==========
def feat_eng(train, test, target_col=None):
    # DÜZELTİLDİ: loc kullanırken hata almamak için indexler sıfırlanıyor
    train = train.reset_index(drop=True).copy()
    test = test.reset_index(drop=True).copy()
    
    # ---- Sayısal Etkileşimler ----
    train['buharlasma_gucu'] = (train['Sicaklik_C'] * train['Ruzgar_Hizi_kmh'] * train['Guneslenme_Saati']) / (train['Nem'] + 1)
    test['buharlasma_gucu']  = (test['Sicaklik_C'] * test['Ruzgar_Hizi_kmh'] * test['Guneslenme_Saati']) / (test['Nem'] + 1)
    
    train['toplam_su_girdisi'] = train['Yagis_mm'] + train['Onceki_Sulama_mm']
    test['toplam_su_girdisi']  = test['Yagis_mm'] + test['Onceki_Sulama_mm']
    
    train['su_acigi'] = train['toplam_su_girdisi'] - train['buharlasma_gucu']
    test['su_acigi']  = test['toplam_su_girdisi'] - test['buharlasma_gucu']
    
    train['ph_sapmasi'] = abs(train['Toprak_pH'] - 7)
    test['ph_sapmasi']  = abs(test['Toprak_pH'] - 7)
    
    train['toprak_kalite'] = (7 - train['ph_sapmasi']) * train['Organik_Karbon'] / (train['Elektrik_Iletkenligi'] + 0.1)
    test['toprak_kalite']  = (7 - test['ph_sapmasi']) * test['Organik_Karbon'] / (test['Elektrik_Iletkenligi'] + 0.1)
    
    train['malc_faktoru'] = train['Malc_Kullanimi'].map({'Yes': 0.6, 'No': 1.0})
    test['malc_faktoru']  = test['Malc_Kullanimi'].map({'Yes': 0.6, 'No': 1.0})
    
    train['buharlasma_malc_etkili'] = train['buharlasma_gucu'] * train['malc_faktoru']
    test['buharlasma_malc_etkili']  = test['buharlasma_gucu'] * test['malc_faktoru']
    
    train['sicaklik_nem_stres'] = train['Sicaklik_C'] * (100 - train['Nem'])
    test['sicaklik_nem_stres']  = test['Sicaklik_C'] * (100 - test['Nem'])
    
    # ---- Kategorik Target Encoding (CV'li) ----
    if target_col and target_col in train.columns:
        cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for col in cat_cols:
            train[col + '_te'] = np.nan
            for tr_idx, val_idx in skf.split(train, train[target_col]):
                means = train.iloc[tr_idx].groupby(col)[target_col].mean()
                train.loc[val_idx, col + '_te'] = train.loc[val_idx, col].map(means)
            
            global_means = train.groupby(col)[target_col].mean()
            test[col + '_te'] = test[col].map(global_means).fillna(train[target_col].mean())
            train[col + '_te'] = train[col + '_te'].fillna(train[target_col].mean())
            
    return train, test

# ========== VERİ YÜKLEME ==========
def load_data(path, target, id_column, drop):
    def extract_ids(train, test, id_column):
        test_id = test[id_column]
        train = train.drop([id_column], axis=1)
        test = test.drop([id_column], axis=1)
        return train, test, test_id

    def drop_cols(train, test, drop):
        if drop:
            train = train.drop(drop, axis=1)
            test = test.drop(drop, axis=1)
        return train, test

    def y_encoding(train, target):
        train[target] = train[target].map(y_map)
        return train

    def rename_cols(train, test):
        train = train.rename(columns=tr_cols)
        test = test.rename(columns=tr_cols)
        return train, test

    def cat_num_seperation(df, target):
        cat_cols = df.drop(target, axis=1).select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        num_cols =[c for c in df.columns if c not in cat_cols + [target]]
        return cat_cols, num_cols

    def fill_na(train, test, cat_cols, num_cols):
        num_medians = train[num_cols].median()
        cat_modes = {}
        for col in cat_cols:
            mode_val = train[col].mode()
            if not mode_val.empty:
                cat_modes[col] = mode_val[0]
            else:
                cat_modes[col] = "Missing"
        for col in num_cols:
            train[col] = train[col].fillna(num_medians[col])
        for col in cat_cols:
            train[col] = train[col].fillna(cat_modes[col])
        for col in num_cols:
            test[col] = test[col].fillna(num_medians[col])
        for col in cat_cols:
            test[col] = test[col].fillna(cat_modes[col])
        return train, test

    train, test = pd.read_csv(path+"train.csv"), pd.read_csv(path+"test.csv")
    train, test, test_id = extract_ids(train, test, id_column)
    train, test = drop_cols(train, test, drop)
    train = y_encoding(train, target)
    train, test = rename_cols(train, test)
    
    # FE uygula
    train, test = feat_eng(train, test, target)
    
    cat_cols, num_cols = cat_num_seperation(train, target)
    train, test = fill_na(train, test, cat_cols, num_cols)

    y = train[target]
    X = train.drop(target, axis=1)
    X_test = test.copy()

    return X, y, X_test, test_id, cat_cols, num_cols

# ========== EDA FONKSİYONU (İsteğe bağlı, kısaltıldı) ==========
def detailed_eda_for_fe(X, y, X_test, target_name='Irrigation_Need', 
                        cat_cols=None, num_cols=None,
                        save_fig=False, fig_dir='eda_figures', max_pairplot=5):
    """
    X, y, X_test formatında EDA yapar.
    
    Parametreler:
    ------------
    X : pd.DataFrame - Eğitim özellikleri (hedef değişken dahil değil)
    y : pd.Series - Hedef değişken
    X_test : pd.DataFrame - Test özellikleri
    target_name : str - Hedef değişkenin adı (grafiklerde kullanılacak)
    cat_cols : list - Kategorik sütun listesi (opsiyonel)
    num_cols : list - Sayısal sütun listesi (opsiyonel)
    save_fig : bool - Grafikleri kaydet (default=False)
    fig_dir : str - Kayıt klasörü
    max_pairplot : int - Pairplot için max sayısal özellik sayısı
    """
    
    # Train verisini birleştir
    train = X.copy()
    train[target_name] = y
    
    # Test verisi
    test = X_test.copy()
    
    if save_fig:
        import os
        os.makedirs(fig_dir, exist_ok=True)
    
    # Otomatik tip belirleme
    if cat_cols is None:
        cat_cols = train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        if target in cat_cols:
            cat_cols.remove(target)
    if num_cols is None:
        num_cols = train.select_dtypes(include=np.number).columns.tolist()
        if target in num_cols:
            num_cols.remove(target)
    
    print("="*80)
    print("FEATURE ENGINEERING İÇİN EDA BAŞLIYOR")
    print(f"Train shape: {train.shape} | Test shape: {test.shape}")
    print(f"Kategorik ({len(cat_cols)}): {cat_cols}")
    print(f"Sayısal ({len(num_cols)}): {num_cols}")
    print("="*80)
    
    # 1. Hedef Değişken Dağılımı
    if target and target in train.columns:
        print("\n1. HEDEF DEĞİŞKEN DAĞILIMI")
        fig, ax = plt.subplots(figsize=(8,5))
        train[target].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f'Hedef: {target}')
        ax.set_xlabel(target)
        ax.set_ylabel('Frekans')
        for i, v in enumerate(train[target].value_counts().values):
            ax.text(i, v+100, str(v), ha='center')
        plt.tight_layout()
        if save_fig:
            plt.savefig(f'{fig_dir}/target_dist.png')
        plt.show()
        print(train[target].value_counts())
    
    # 2. Sayısal Özellikler: Train vs Test Dağılımı + KS Testi
    if num_cols:
        print("\n2. SAYISAL ÖZELLİKLER - TRAIN vs TEST DAĞILIMI")
        for col in num_cols:
            fig, axes = plt.subplots(1, 2, figsize=(12,4))
            # Histogram + KDE
            sns.histplot(train[col], kde=True, color='blue', alpha=0.5, ax=axes[0], label='Train')
            sns.histplot(test[col], kde=True, color='red', alpha=0.5, ax=axes[0], label='Test')
            axes[0].set_title(f'{col}')
            axes[0].legend()
            # Boxplot
            data = pd.DataFrame({'value': pd.concat([train[col], test[col]]),
                                 'set': ['Train']*len(train) + ['Test']*len(test)})
            sns.boxplot(x='set', y='value', data=data, ax=axes[1])
            # KS testi
            ks_stat, p = ks_2samp(train[col].dropna(), test[col].dropna())
            fig.suptitle(f'KS test p={p:.4f}')
            plt.tight_layout()
            if save_fig:
                plt.savefig(f'{fig_dir}/num_{col}.png')
            plt.show()
            if p < 0.05:
                print(f"  ⚠️ {col}: Train-test farklı (p={p:.4f})")
            else:
                print(f"  ✓ {col}: Benzer dağılım (p={p:.4f})")
    
    # 3. Kategorik Özellikler: Frekans ve Yeni Kategori Kontrolü
    if cat_cols:
        print("\n3. KATEGORİK ÖZELLİKLER - TRAIN vs TEST")
        for col in cat_cols:
            fig, axes = plt.subplots(1, 2, figsize=(14,5))
            train[col].value_counts().head(10).plot(kind='bar', ax=axes[0], color='blue')
            axes[0].set_title(f'Train - {col}')
            test[col].value_counts().head(10).plot(kind='bar', ax=axes[1], color='red')
            axes[1].set_title(f'Test - {col}')
            plt.tight_layout()
            if save_fig:
                plt.savefig(f'{fig_dir}/cat_{col}.png')
            plt.show()
            new_cats = set(test[col].unique()) - set(train[col].unique())
            if new_cats:
                print(f"  ⚠️ {col}: Test'te yeni kategoriler -> {new_cats}")
            else:
                print(f"  ✓ {col}: Tüm kategoriler train'de var")
    
    # 4. Korelasyon Matrisi (Sayısal)
    if len(num_cols) > 1:
        print("\n4. KORELASYON MATRİSİ")
        plt.figure(figsize=(12,10))
        corr = train[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', square=True)
        plt.title('Sayısal Özellikler Korelasyonu')
        if save_fig:
            plt.savefig(f'{fig_dir}/corr.png')
        plt.show()
        # Yüksek korelasyon uyarısı
        high = [(c1, c2, corr.loc[c1,c2]) for i,c1 in enumerate(corr.columns) 
                for c2 in corr.columns[i+1:] if abs(corr.loc[c1,c2]) > 0.8]
        if high:
            print("⚠️ Yüksek korelasyon (>0.8):")
            for c1,c2,val in high:
                print(f"   {c1} - {c2}: {val:.2f}")
        
        # Hedefi sayısala çevir (Low=0, Medium=1, High=2)
        corr_with_target = train[num_cols + [target]].corr()[target].drop(target).sort_values()
        corr_with_target.plot(kind='barh', figsize=(10,6))
        plt.title('Sayısal Özelliklerin Hedef ile Korelasyonu')
        plt.show()
    
    # 5. Hedefe Göre Sayısal Özellikler (Boxplot)
    if target and target in train.columns and num_cols:
        print("\n5. HEDEFE GÖRE SAYISAL ÖZELLİKLER")
        for col in num_cols:
            fig, ax = plt.subplots(figsize=(12, 5))
            
            # Her sınıf için ayrı renk ve çizgi stili
            colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Yeşil, Turuncu, Kırmızı
            labels = ['Low (0)', 'Medium (1)', 'High (2)']
            
            for class_val, color, label in zip(sorted(train[target].unique()), colors, labels):
                subset = train[train[target] == class_val][col]
                sns.kdeplot(subset, fill=True, alpha=0.3, color=color, label=label, ax=ax, linewidth=2)
            
            # X ekseni etiketlerini sıklaştır (maksimum 15 tick)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(15))
            ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
            
            # Grid ve stil
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_title(f'{col} - Sınıf Bazlı Yoğunluk Dağılımı', fontsize=14, fontweight='bold')
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel('Yoğunluk', fontsize=12)
            ax.legend(title='Sulama İhtiyacı', fontsize=10)
            
            # İstatistiksel özet (isteğe bağlı)
            stats_text = f"Low: μ={train[train[target]==0][col].mean():.2f}\nMedium: μ={train[train[target]==1][col].mean():.2f}\nHigh: μ={train[train[target]==2][col].mean():.2f}"
            ax.text(0.98, 0.85, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show()

    # 6. Kategorik Özelliklerin Hedefe Göre Dağılımı (Stacked Bar Plot)
    if target and target in train.columns and cat_cols:
        print("\n6. KATEGORİK ÖZELLİKLER - HEDEFE GÖRE DAĞILIM")
        for col in cat_cols:
            # Çapraz tablo (oran)
            crosstab = pd.crosstab(train[col], train[target], normalize='index')
            
            # Grafik çiz
            ax = crosstab.plot(kind='bar', stacked=True, figsize=(8, 6), 
                               color=['#2ecc71', '#f1c40f', '#e74c3c'])  # Yeşil-Sarı-Kırmızı (Low-Medium-High)
            plt.title(f'{col} - Hedef Sınıf Oranları', fontsize=14)
            plt.ylabel('Oran', fontsize=12)
            plt.xlabel(col, fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Irrigation_Need', labels=['Low (0)', 'Medium (1)', 'High (2)'])
            
            # YÜZDE ETİKETLERİNİ EKLE (Sadece %3'ten büyük olanları göster)
            for i, (index, row) in enumerate(crosstab.iterrows()):
                cumsum = 0
                for j, (col_name, value) in enumerate(row.items()):
                    if value > 0.03:  # %3'ten küçükleri yazdırma, grafik kirlenmesin
                        ax.text(i, cumsum + value/2, f'{value:.1%}', 
                                ha='center', va='center', fontsize=9, fontweight='bold', color='black')
                    cumsum += value
            
            plt.tight_layout()
            plt.show()
        
    # 7. Aykırı Değer Analizi (IQR yöntemi)
    if num_cols:
        print("\n7. AYKIRI DEĞER ANALİZİ")
        for col in num_cols:
            plt.figure(figsize=(8,3))
            sns.boxplot(x=train[col])
            plt.title(f'{col}')
            if save_fig:
                plt.savefig(f'{fig_dir}/outlier_{col}.png')
            plt.show()
            Q1, Q3 = train[col].quantile(0.25), train[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((train[col] < Q1 - 1.5*IQR) | (train[col] > Q3 + 1.5*IQR)).sum()
            pct = outliers/len(train)*100
            if pct > 5:
                print(f"  ⚠️ {col}: %{pct:.1f} aykırı değer ({outliers})")
            else:
                print(f"  ✓ {col}: %{pct:.1f} aykırı değer")
    
    # 8. Eksik Değer Görselleştirmesi
    print("\n8. EKSİK DEĞER ANALİZİ")
    if train.isnull().sum().sum() == 0 and test.isnull().sum().sum() == 0:
        print("  ✓ Hiç eksik değer yok.")
    else:
        fig, axes = plt.subplots(1,2, figsize=(14,6))
        msno.matrix(train, ax=axes[0])
        axes[0].set_title('Train')
        msno.matrix(test, ax=axes[1])
        axes[1].set_title('Test')
        plt.tight_layout()
        if save_fig:
            plt.savefig(f'{fig_dir}/missing.png')
        plt.show()
        train_miss = train.isnull().sum()
        train_miss = train_miss[train_miss>0].sort_values(ascending=False)
        if len(train_miss):
            print("Train eksik oranları:\n", train_miss/len(train))
        test_miss = test.isnull().sum()
        test_miss = test_miss[test_miss>0].sort_values(ascending=False)
        if len(test_miss):
            print("Test eksik oranları:\n", test_miss/len(test))
    
    # 9. Pairplot (sınırlı sayıda)
    if len(num_cols) <= max_pairplot and target and target in train.columns:
        print(f"\n9. PAIRPLOT (ilk {max_pairplot} sayısal)")
        sns.pairplot(train[num_cols[:max_pairplot] + [target]], hue=target, diag_kind='kde')
        if save_fig:
            plt.savefig(f'{fig_dir}/pairplot.png')
        plt.show()
    else:
        print(f"\n8. Pairplot atlandı (sayısal sayısı {len(num_cols)} > {max_pairplot})")
    
    print("\n" + "="*80)
    print("EDA TAMAMLANDI. Yukarıdaki uyarıları FE kararlarınızda kullanın.")
    print("="*80)

# ========== MODEL HAZIRLIK ==========
def prepare_data(X, X_test, cat_cols):
    X = X.copy()
    X_test = X_test.copy()
    for col in cat_cols:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")
    return X, X_test

# ========== SHAP YARDIMCI FONKSİYONU (DÜZELTİLDİ) ==========
def get_shap_importance(explainer, sample_X):
    shap_values = explainer.shap_values(sample_X)
    
    # Multiclass için liste dönüyorsa (eski sürümler / LGBM)
    if isinstance(shap_values, list):
        shap_imp = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        # Multiclass için 3 boyutlu dönüyorsa (yeni XGBoost) -> (samples, features, classes)
        if shap_values.ndim == 3:
            shap_imp = np.abs(shap_values).mean(axis=0).mean(axis=1)
        else:
            shap_imp = np.abs(shap_values).mean(axis=0)
            
    shap_imp_df = pd.DataFrame({'feature': sample_X.columns, 'shap_importance': shap_imp})
    return shap_imp_df.sort_values('shap_importance', ascending=False).reset_index(drop=True)

# ========== ANA İŞLEM ==========
X, y, X_test, test_id, cat_cols, num_cols = load_data(path, target, id_col, drop)

detailed_eda_for_fe(X=X, y=y, X_test=X_test, target_name=target, cat_cols=cat_cols, num_cols=num_cols, save_fig=True)

prepX, prepX_test = prepare_data(X, X_test, cat_cols)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Sınıf etiketi ters haritası (0→Low, 1→Medium, 2→High)
inv_y_map = {v: k for k, v in y_map.items()}

# -----------------------------------------------------
# 1. CatBoost 
# -----------------------------------------------------
print("\nCatBoost Stratified K-Fold başlıyor...")
cat_oof = np.zeros((len(y), n_classes))
cat_test_preds = []
cat_models =[]
cat_fold_bal_acc, cat_fold_acc, cat_fold_macro_f1 = [], [],[]

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = cat.CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        cat_features=cat_cols, random_state=42,
        eval_metric="MultiClass",
        auto_class_weights='Balanced', verbose=0,
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
    
    cat_oof[val_idx] = model.predict_proba(X_val)
    cat_test_preds.append(model.predict_proba(X_test))
    cat_models.append(model)
    
    y_val_pred = model.predict(X_val)
    y_val_pred = y_val_pred.astype(int).reshape(-1)
    cat_fold_bal_acc.append(balanced_accuracy_score(y_val, y_val_pred))
    cat_fold_acc.append(accuracy_score(y_val, y_val_pred))
    cat_fold_macro_f1.append(f1_score(y_val, y_val_pred, average='macro'))
    
    print(f"CatBoost Fold {fold+1}/5 -> Balanced Accuracy: {cat_fold_bal_acc[-1]:.4f}")

print("\n📊 CatBoost Ortalama Balanced Accuracy:", np.mean(cat_fold_bal_acc))
cat_test_proba = np.mean(cat_test_preds, axis=0)

print("\nCatBoost SHAP hesaplanıyor...")
explainer_cat = shap.TreeExplainer(cat_models[0])
sample_X_cat = X.sample(min(500, len(X)), random_state=42)
shap_imp_df_cat = get_shap_importance(explainer_cat, sample_X_cat)
print("🔹 CatBoost SHAP Önem Sıralaması:\n", shap_imp_df_cat)

# 1. CatBoost OOF ve Submission
cat_oof_df = pd.DataFrame(cat_oof, columns=[f"prob_{inv_y_map[i]}" for i in range(n_classes)])
cat_oof_df['pred_class'] = np.argmax(cat_oof, axis=1)
cat_oof_df['pred_label'] = cat_oof_df['pred_class'].map(inv_y_map)
cat_oof_df.to_csv('catboost_oof.csv', index=False)

cat_sub = pd.DataFrame({
    'id': test_id,
    'Irrigation_Need': [inv_y_map[p] for p in np.argmax(cat_test_proba, axis=1)]
})
cat_sub.to_csv('catboost_submission.csv', index=False)
print("✅ CatBoost OOF ve submission kaydedildi.")
# -----------------------------------------------------
# 2. LightGBM
# -----------------------------------------------------
print("\nLightGBM Stratified K-Fold başlıyor...")
lgb_oof = np.zeros((len(y), n_classes))
lgb_test_preds = []
lgb_models =[]
lgb_fold_bal_acc, lgb_fold_acc, lgb_fold_macro_f1 = [], [],[]

for fold, (train_idx, val_idx) in enumerate(skf.split(prepX, y)):
    X_tr, X_val = prepX.iloc[train_idx], prepX.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=31,
        random_state=42, verbose=-1, class_weight='balanced'
    )
    # Lgbm kategorik tespiti otomatiktir pandas category tipleri icin
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    
    lgb_oof[val_idx] = model.predict_proba(X_val)
    lgb_test_preds.append(model.predict_proba(prepX_test))
    lgb_models.append(model)
    
    y_val_pred = model.predict(X_val).flatten().astype(int)
    lgb_fold_bal_acc.append(balanced_accuracy_score(y_val, y_val_pred))
    
    print(f"LightGBM Fold {fold+1}/5 -> Balanced Accuracy: {lgb_fold_bal_acc[-1]:.4f}")

print("\n📊 LightGBM Ortalama Balanced Accuracy:", np.mean(lgb_fold_bal_acc))
lgb_test_proba = np.mean(lgb_test_preds, axis=0)

print("\nLightGBM SHAP hesaplanıyor...")
explainer_lgb = shap.TreeExplainer(lgb_models[0])
sample_X_lgb = prepX.sample(min(500, len(prepX)), random_state=42)
shap_imp_df_lgb = get_shap_importance(explainer_lgb, sample_X_lgb)
print("🔹 LightGBM SHAP Önem Sıralaması:\n", shap_imp_df_lgb)

# 2. LightGBM OOF ve Submission
lgb_oof_df = pd.DataFrame(lgb_oof, columns=[f"prob_{inv_y_map[i]}" for i in range(n_classes)])
lgb_oof_df['pred_class'] = np.argmax(lgb_oof, axis=1)
lgb_oof_df['pred_label'] = lgb_oof_df['pred_class'].map(inv_y_map)
lgb_oof_df.to_csv('lightgbm_oof.csv', index=False)

lgb_sub = pd.DataFrame({
    'id': test_id,
    'Irrigation_Need': [inv_y_map[p] for p in np.argmax(lgb_test_proba, axis=1)]
})
lgb_sub.to_csv('lightgbm_submission.csv', index=False)
print("✅ LightGBM OOF ve submission kaydedildi.")
# -----------------------------------------------------
# 3. XGBoost
# -----------------------------------------------------
print("\nXGBoost Stratified K-Fold başlıyor...")
xgb_oof = np.zeros((len(y), n_classes))
xgb_test_preds = []
xgb_models =[]
xgb_fold_bal_acc, xgb_fold_acc, xgb_fold_macro_f1 = [], [],[]

for fold, (train_idx, val_idx) in enumerate(skf.split(xgbX, y)):
    X_tr, X_val = xgbX.iloc[train_idx], xgbX.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_tr)
    
    model = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        objective='multi:softprob', num_class=n_classes,
        random_state=42, early_stopping_rounds=50,
        eval_metric="mlogloss",
        enable_categorical=True # DÜZELTİLDİ: Kategorik veriler için yerleşik destek eklendi
    )
    model.fit(X_tr, y_tr, sample_weight=sample_weights, eval_set=[(X_val, y_val)], verbose=False)
    
    xgb_oof[val_idx] = model.predict_proba(X_val)
    xgb_test_preds.append(model.predict_proba(xgbX_test))
    xgb_models.append(model)
    
    y_val_pred = model.predict(X_val).flatten().astype(int)
    xgb_fold_bal_acc.append(balanced_accuracy_score(y_val, y_val_pred))
    
    print(f"XGBoost Fold {fold+1}/5 -> Balanced Accuracy: {xgb_fold_bal_acc[-1]:.4f}")

print("\n📊 XGBoost Ortalama Balanced Accuracy:", np.mean(xgb_fold_bal_acc))
xgb_test_proba = np.mean(xgb_test_preds, axis=0)

print("\nXGBoost SHAP hesaplanıyor...")
explainer_xgb = shap.TreeExplainer(xgb_models[0])
sample_X_xgb = xgbX.sample(min(500, len(xgbX)), random_state=42)
shap_imp_df_xgb = get_shap_importance(explainer_xgb, sample_X_xgb)
print("🔹 XGBoost SHAP Önem Sıralaması:\n", shap_imp_df_xgb)

# 3. XGBoost OOF ve Submission
xgb_oof_df = pd.DataFrame(xgb_oof, columns=[f"prob_{inv_y_map[i]}" for i in range(n_classes)])
xgb_oof_df['pred_class'] = np.argmax(xgb_oof, axis=1)
xgb_oof_df['pred_label'] = xgb_oof_df['pred_class'].map(inv_y_map)
xgb_oof_df.to_csv('xgboost_oof.csv', index=False)

xgb_sub = pd.DataFrame({
    'id': test_id,
    'Irrigation_Need': [inv_y_map[p] for p in np.argmax(xgb_test_proba, axis=1)]
})
xgb_sub.to_csv('xgboost_submission.csv', index=False)
print("✅ XGBoost OOF ve submission kaydedildi.")
# -----------------------------------------------------
# 4. Ensemble (Basit Ortalama)
# -----------------------------------------------------
ensemble_proba = (cat_test_proba + lgb_test_proba + xgb_test_proba) / 3
ensemble_preds = np.argmax(ensemble_proba, axis=1)

submission = pd.DataFrame({
    'id': test_id, 
    'Irrigation_Need': [['Low','Medium','High'][p] for p in ensemble_preds]
})
submission.to_csv('ensemble_submission.csv', index=False)

# Ensemble submission zaten kaydediliyor (mevcut kodda var)
print("\n📁 Kaydedilen dosyalar:")
print("  - catboost_oof.csv")
print("  - catboost_submission.csv")
print("  - lightgbm_oof.csv")
print("  - lightgbm_submission.csv")
print("  - xgboost_oof.csv")
print("  - xgboost_submission.csv")
print("  - ensemble_submission.csv")

print("\n✅ Tüm modeller başarıyla eğitildi ve 'ensemble_submission.csv' kaydedildi!")