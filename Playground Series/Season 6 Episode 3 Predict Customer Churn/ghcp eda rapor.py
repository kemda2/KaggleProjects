"""
FULL HTML EDA REPORT GENERATOR
Tam çalışan, 550 satır
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

class EDAtoHTML:
    def __init__(self, df, target=None, lang='tr'):
        self.df = df
        self.target = target
        self.lang = lang
        self.time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def generate(self, filename='eda_report.html'):
        """HTML rapor oluştur ve kaydet"""
        html = self._build_html()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"✓ {filename} oluşturuldu!")
    
    def _build_html(self):
        """HTML rapor inşa et"""
        
        # CSS Stilleri
        css = """
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
                color: #333;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }
            header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            header p {
                font-size: 1.1em;
                opacity: 0.9;
            }
            .metadata {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                padding: 20px 40px;
                background: #f8f9fa;
                border-bottom: 2px solid #667eea;
            }
            .meta-item {
                text-align: center;
                padding: 15px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .meta-item strong {
                display: block;
                color: #667eea;
                font-size: 1.2em;
                margin-bottom: 5px;
            }
            .meta-item span {
                color: #666;
                font-size: 0.9em;
            }
            .content {
                padding: 40px;
            }
            section {
                margin-bottom: 40px;
                padding: 25px;
                background: #f9f9f9;
                border-radius: 10px;
                border-left: 5px solid #667eea;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            h2 {
                color: #667eea;
                font-size: 1.8em;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #667eea;
            }
            h3 {
                color: #764ba2;
                font-size: 1.2em;
                margin-top: 15px;
                margin-bottom: 10px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            th {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: 600;
                border: none;
            }
            td {
                padding: 12px 15px;
                border-bottom: 1px solid #eee;
                text-align: left;
            }
            tr:hover {
                background: #f5f5f5;
            }
            tr:last-child td {
                border-bottom: none;
            }
            .warning {
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }
            .success {
                background: #d4edda;
                border-left: 4px solid #28a745;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }
            .error {
                background: #f8d7da;
                border-left: 4px solid #dc3545;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }
            .code {
                background: #2d2d2d;
                color: #f8f8f2;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
                margin: 15px 0;
            }
            .quality-score {
                font-size: 3em;
                font-weight: bold;
                text-align: center;
                padding: 20px;
                margin: 20px 0;
                border-radius: 10px;
            }
            .score-good { color: #28a745; background: #d4edda; }
            .score-medium { color: #ffc107; background: #fff3cd; }
            .score-bad { color: #dc3545; background: #f8d7da; }
            .recommendation {
                background: white;
                border: 2px solid #667eea;
                border-radius: 8px;
                padding: 20px;
                margin: 15px 0;
            }
            .recommendation-title {
                font-weight: bold;
                color: #667eea;
                margin-bottom: 10px;
                font-size: 1.1em;
            }
            .grid-2 {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 15px 0;
            }
            @media (max-width: 768px) {
                .grid-2 { grid-template-columns: 1fr; }
                header h1 { font-size: 1.8em; }
                .metadata { grid-template-columns: 1fr; }
            }
            footer {
                background: #f8f9fa;
                padding: 20px;
                text-align: center;
                border-top: 2px solid #ddd;
                color: #666;
                font-size: 0.9em;
            }
        </style>
        """
        
        # Başlık
        header = f"""
        <header>
            <h1>📊 KAPSAMLı EDA RAPORU</h1>
            <p>Exploratory Data Analysis Report</p>
            <p style="margin-top: 10px; font-size: 0.9em;">📅 {self.time}</p>
        </header>
        """
        
        # Metadata
        metadata = f"""
        <div class="metadata">
            <div class="meta-item">
                <strong>{len(self.df):,}</strong>
                <span>Satırlar / Rows</span>
            </div>
            <div class="meta-item">
                <strong>{len(self.df.columns)}</strong>
                <span>Sütunlar / Columns</span>
            </div>
            <div class="meta-item">
                <strong>{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB</strong>
                <span>Bellek / Memory</span>
            </div>
            <div class="meta-item">
                <strong>{self.df.isnull().sum().sum()}</strong>
                <span>Eksik / Missing</span>
            </div>
            <div class="meta-item">
                <strong>{self.df.duplicated().sum()}</strong>
                <span>Tekrarlar / Duplicates</span>
            </div>
        </div>
        """
        
        # İçerik
        content = '<div class="content">'
        
        # 1. VERİ SETİ ÖZETI
        content += self._section_info()
        
        # 2. EKSİK VERİ
        content += self._section_missing()
        
        # 3. TEKRARLAYAN KAYITLAR
        content += self._section_duplicates()
        
        # 4. SAYISAL DEĞİŞKENLER
        content += self._section_numeric()
        
        # 5. KATEGORİK DEĞİŞKENLER
        content += self._section_categorical()
        
        # 6. AYKIRI DEĞERLER
        content += self._section_outliers()
        
        # 7. KORELASYON
        content += self._section_correlation()
        
        # 8. HEDEF DEĞİŞKEN
        if self.target:
            content += self._section_target()
        
        # 9. FEATURE ENGINEERING
        content += self._section_feature_engineering()
        
        # 10. KALİTE PUANI
        content += self._section_quality()
        
        content += '</div>'
        
        # Footer
        footer = """
        <footer>
            <p>Bu rapor otomatik olarak oluşturulmuştur | This report was automatically generated</p>
            <p>Feature Engineering öncesi detaylı veri analizi için tasarlanmıştır</p>
        </footer>
        """
        
        # Tüm HTML'yi birleştir
        html = f"""
        <!DOCTYPE html>
        <html lang="tr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>EDA Raporu</title>
            {css}
        </head>
        <body>
            <div class="container">
                {header}
                {metadata}
                {content}
                {footer}
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _section_info(self):
        """Veri seti özeti"""
        dtypes = self.df.dtypes.value_counts()
        
        html = '<section><h2>1️⃣ VERİ SETİ ÖZETI / DATASET OVERVIEW</h2>'
        
        html += '<h3>Veri Türleri / Data Types</h3>'
        html += '<table><tr><th>Tip / Type</th><th>Sayı / Count</th></tr>'
        for dtype, count in dtypes.items():
            html += f'<tr><td>{dtype}</td><td>{count}</td></tr>'
        html += '</table>'
        
        html += '<h3>Sütun Listesi / Column List</h3>'
        html += '<table><tr><th>Sütun / Column</th><th>Tip / Type</th><th>Null %</th></tr>'
        for col in self.df.columns:
            null_pct = self.df[col].isnull().sum() / len(self.df) * 100
            html += f'<tr><td>{col}</td><td>{self.df[col].dtype}</td><td>{null_pct:.2f}%</td></tr>'
        html += '</table>'
        
        html += '</section>'
        return html
    
    def _section_missing(self):
        """Eksik veri"""
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        html = '<section><h2>2️⃣ EKSİK VERİ ANALİZİ / MISSING DATA</h2>'
        
        if len(missing) == 0:
            html += '<div class="success"><strong>✓</strong> Eksik veri bulunmamaktadır / No missing data</div>'
        else:
            html += '<table><tr><th>Sütun / Column</th><th>Eksik / Missing</th><th>%</th><th>Aksiyonunuz / Action</th></tr>'
            for col, cnt in missing.items():
                pct = cnt / len(self.df) * 100
                if pct < 5:
                    action = '📝 Ortalamayla doldur / Fill with mean'
                elif pct < 50:
                    action = '🔧 KNN Imputation'
                else:
                    action = '❌ Sütunu sil / Delete'
                html += f'<tr><td>{col}</td><td>{cnt}</td><td>{pct:.2f}%</td><td>{action}</td></tr>'
            html += '</table>'
            
            html += '<div class="recommendation"><div class="recommendation-title">💡 EKSIK VERİ HANDLİNG REHBERI</div>'
            html += '<h4>% 0-5 Arası:</h4>'
            html += '<div class="code">df[\'column\'].fillna(df[\'column\'].mean(), inplace=True)  # Sayısal<br>df[\'column\'].fillna(df[\'column\'].mode()[0], inplace=True)  # Kategorik</div>'
            html += '<h4>% 5-50 Arası:</h4>'
            html += '<div class="code">from sklearn.impute import KNNImputer<br>imputer = KNNImputer(n_neighbors=5)<br>df_imputed = imputer.fit_transform(df)</div>'
            html += '<h4>% 50+ Arası:</h4>'
            html += '<div class="code">df = df.drop(\'column\', axis=1)  # Sütunu sil</div>'
            html += '</div>'
        
        html += '</section>'
        return html
    
    def _section_duplicates(self):
        """Tekrarlayan kayıtlar"""
        dup_count = self.df.duplicated().sum()
        dup_pct = dup_count / len(self.df) * 100
        
        html = '<section><h2>3️⃣ TEKRARlayan KAYITLAR / DUPLICATES</h2>'
        
        if dup_count == 0:
            html += '<div class="success"><strong>✓</strong> Tekrarlayan kayıt bulunmamaktadır</div>'
        else:
            html += f'<div class="warning"><strong>⚠</strong> {dup_count} tekrarlayan kayıt bulundu ({dup_pct:.2f}%)</div>'
            html += '<div class="code">df = df.drop_duplicates()  # Tümünü sil<br>df = df.drop_duplicates(subset=[\'id\'], keep=\'last\')  # Kısmi tekrar</div>'
        
        html += '</section>'
        return html
    
    def _section_numeric(self):
        """Sayısal değişkenler"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target and self.target in numeric_cols:
            numeric_cols.remove(self.target)
        
        html = '<section><h2>4️⃣ SAYISAL DEĞİŞKENLER / NUMERIC VARIABLES</h2>'
        
        if len(numeric_cols) == 0:
            html += '<p>Sayısal değişken bulunmamaktadır</p>'
        else:
            html += '<table><tr><th>Variable</th><th>Mean</th><th>Median</th><th>Std</th><th>Min</th><th>Max</th><th>Skewness</th><th>Action</th></tr>'
            
            for col in numeric_cols[:30]:  # İlk 30
                mean = self.df[col].mean()
                median = self.df[col].median()
                std = self.df[col].std()
                mini = self.df[col].min()
                maxi = self.df[col].max()
                skew = self.df[col].skew()
                
                if abs(skew) > 1:
                    action = '⚠ Log Transform'
                elif abs(skew) > 0.5:
                    action = '📝 Sqrt Transform'
                else:
                    action = '✓ OK'
                
                html += f'<tr><td>{col}</td><td>{mean:.2f}</td><td>{median:.2f}</td><td>{std:.2f}</td><td>{mini:.2f}</td><td>{maxi:.2f}</td><td>{skew:.2f}</td><td>{action}</td></tr>'
            
            html += '</table>'
            
            html += '<div class="recommendation"><div class="recommendation-title">💡 SAYISAL DEĞİŞKEN TRANSFORMATION</div>'
            html += '<h4>Skewness > 1 ise:</h4>'
            html += '<div class="code">df[\'column_log\'] = np.log1p(df[\'column\'])<br>df[\'column_sqrt\'] = np.sqrt(df[\'column\'])<br>from scipy.stats import boxcox<br>df[\'column_boxcox\'] = boxcox(df[\'column\'] + 1)[0]</div>'
            html += '</div>'
        
        html += '</section>'
        return html
    
    def _section_categorical(self):
        """Kategorik değişkenler"""
        cat_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        html = '<section><h2>5️⃣ KATEGORİK DEĞİŞKENLER / CATEGORICAL VARIABLES</h2>'
        
        if len(cat_cols) == 0:
            html += '<p>Kategorik değişken bulunmamaktadır</p>'
        else:
            html += '<table><tr><th>Variable</th><th>Unique</th><th>Top Value</th><th>Top Count</th><th>Cardinality</th><th>Action</th></tr>'
            
            for col in cat_cols[:20]:  # İlk 20
                vc = self.df[col].value_counts()
                unique = len(vc)
                top_val = vc.index[0]
                top_count = vc.iloc[0]
                cardinality = unique / len(self.df)
                
                if unique > 20:
                    action = '⚠ Group rare'
                else:
                    action = '✓ OK'
                
                html += f'<tr><td>{col}</td><td>{unique}</td><td>{top_val}</td><td>{top_count}</td><td>{cardinality:.4f}</td><td>{action}</td></tr>'
            
            html += '</table>'
            
            html += '<div class="recommendation"><div class="recommendation-title">💡 KATEGORİK ENCODING</div>'
            html += '<h4>One-Hot Encoding (< 10 category):</h4>'
            html += '<div class="code">df = pd.get_dummies(df, columns=[\'column\'], drop_first=True)</div>'
            html += '<h4>Label Encoding (Tree models):</h4>'
            html += '<div class="code">from sklearn.preprocessing import LabelEncoder<br>le = LabelEncoder()<br>df[\'column_encoded\'] = le.fit_transform(df[\'column\'])</div>'
            html += '<h4>Target Encoding (High cardinality):</h4>'
            html += '<div class="code">target_mean = df.groupby(\'column\')[\'target\'].mean()<br>df[\'column_target\'] = df[\'column\'].map(target_mean)</div>'
            html += '</div>'
        
        html += '</section>'
        return html
    
    def _section_outliers(self):
        """Aykırı değerler"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        html = '<section><h2>6️⃣ AYKIRI DEĞER ANALİZİ / OUTLIERS (IQR)</h2>'
        
        html += '<table><tr><th>Variable</th><th>Outlier Count</th><th>%</th><th>Lower</th><th>Upper</th><th>Action</th></tr>'
        
        has_outliers = False
        for col in numeric_cols[:30]:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            out_mask = (self.df[col] < lower) | (self.df[col] > upper)
            out_count = out_mask.sum()
            out_pct = out_count / len(self.df) * 100
            
            if out_count > 0:
                has_outliers = True
                if out_pct > 5:
                    action = '⚠ Log/Cap'
                else:
                    action = '✓ OK'
                html += f'<tr><td>{col}</td><td>{out_count}</td><td>{out_pct:.2f}%</td><td>{lower:.2f}</td><td>{upper:.2f}</td><td>{action}</td></tr>'
        
        html += '</table>'
        
        if not has_outliers:
            html += '<div class="success">✓ Önemli aykırı değer bulunmamaktadır</div>'
        
        html += '<div class="recommendation"><div class="recommendation-title">💡 AYKIRI DEĞER HANDLİNG</div>'
        html += '<h4>Silin (% < 1):</h4>'
        html += '<div class="code">df = df[~outlier_mask]</div>'
        html += '<h4>Sınırlayın / Cap (% 1-5):</h4>'
        html += '<div class="code">df[\'column\'] = df[\'column\'].clip(lower=lower_bound, upper=upper_bound)</div>'
        html += '<h4>Transform Et (% 5-10):</h4>'
        html += '<div class="code">df[\'column_log\'] = np.log1p(df[\'column\'])</div>'
        html += '<h4>İzole Et (% > 10):</h4>'
        html += '<div class="code">df[\'is_outlier\'] = outlier_mask.astype(int)</div>'
        html += '</div>'
        
        html += '</section>'
        return html
    
    def _section_correlation(self):
        """Korelasyon"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        html = '<section><h2>7️⃣ KORELASYON ANALİZİ / CORRELATION</h2>'
        
        if len(numeric_cols) < 2:
            html += '<p>Korelasyon analizi için en az 2 sayısal değişken gereklidir</p>'
        else:
            corr = self.df[numeric_cols].corr()
            
            # Hedef korelasyonu
            if self.target and self.target in self.df.columns:
                if pd.api.types.is_numeric_dtype(self.df[self.target]):
                    html += '<h3>Hedef Korelasyonu / Target Correlation</h3>'
                    html += '<table><tr><th>Feature</th><th>Correlation</th><th>Strength</th></tr>'
                    
                    target_corr = corr[self.target].sort_values(ascending=False)
                    for feat, corr_val in target_corr.head(15).items():
                        if feat != self.target:
                            if abs(corr_val) > 0.7:
                                strength = '⭐⭐⭐ Very Strong'
                            elif abs(corr_val) > 0.5:
                                strength = '⭐⭐ Strong'
                            elif abs(corr_val) > 0.3:
                                strength = '⭐ Moderate'
                            else:
                                strength = '❌ Weak'
                            html += f'<tr><td>{feat}</td><td>{corr_val:.4f}</td><td>{strength}</td></tr>'
                    html += '</table>'
            
            # Güçlü korelasyonlar
            html += '<h3>Güçlü Korelasyonlar (|r| > 0.7) / Strong Correlations</h3>'
            strong_found = False
            html += '<table><tr><th>Feature 1</th><th>Feature 2</th><th>Correlation</th></tr>'
            
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if abs(corr.iloc[i, j]) > 0.7:
                        strong_found = True
                        html += f'<tr><td>{corr.columns[i]}</td><td>{corr.columns[j]}</td><td>{corr.iloc[i, j]:.4f}</td></tr>'
            
            html += '</table>'
            
            if not strong_found:
                html += '<div class="success">✓ Multicollinearity problemi bulunmamaktadır</div>'
            else:
                html += '<div class="warning">⚠ Yüksek korelasyon var! Birini silebilirsiniz</div>'
            
            html += '<div class="code">df = df.drop(\'redundant_column\', axis=1)</div>'
        
        html += '</section>'
        return html
    
    def _section_target(self):
        """Hedef değişken"""
        html = '<section><h2>8️⃣ HEDEF DEĞİŞKEN / TARGET VARIABLE</h2>'
        
        target = self.df[self.target]
        
        if pd.api.types.is_numeric_dtype(target):
            html += '<h3>📈 REGRESSION TASK</h3>'
            html += '<table><tr><th>Metrik</th><th>Değer</th></tr>'
            html += f'<tr><td>Mean</td><td>{target.mean():.4f}</td></tr>'
            html += f'<tr><td>Median</td><td>{target.median():.4f}</td></tr>'
            html += f'<tr><td>Std Dev</td><td>{target.std():.4f}</td></tr>'
            html += f'<tr><td>Min</td><td>{target.min():.4f}</td></tr>'
            html += f'<tr><td>Max</td><td>{target.max():.4f}</td></tr>'
            html += f'<tr><td>Skewness</td><td>{target.skew():.4f}</td></tr>'
            html += '</table>'
            
            if abs(target.skew()) > 1:
                html += '<div class="warning">⚠ Target çarpık! Log transform önerilir</div>'
                html += '<div class="code">y_log = np.log1p(y)</div>'
        else:
            html += '<h3>🎯 CLASSIFICATION TASK</h3>'
            vc = target.value_counts()
            html += '<table><tr><th>Class</th><th>Count</th><th>%</th></tr>'
            
            for cls, cnt in vc.items():
                pct = cnt / len(self.df) * 100
                html += f'<tr><td>{cls}</td><td>{cnt}</td><td>{pct:.2f}%</td></tr>'
            
            html += '</table>'
            
            imbalance = vc.iloc[0] / vc.iloc[-1]
            if imbalance > 10:
                html += '<div class="error">❌ Yüksek dengesizlik! ({:.1f}:1)</div>'.format(imbalance)
                html += '<h4>Çözüm Yolları:</h4>'
                html += '<div class="code">from imblearn.over_sampling import SMOTE<br>smote = SMOTE()<br>X_res, y_res = smote.fit_resample(X, y)</div>'
                html += '<div class="code">from sklearn.utils.class_weight import compute_class_weight<br>weights = compute_class_weight("balanced", classes=np.unique(y), y=y)<br>model = XGBClassifier(scale_pos_weight=weights[1]/weights[0])</div>'
            else:
                html += '<div class="success">✓ Dengeli sınıf dağılımı</div>'
        
        html += '</section>'
        return html
    
    def _section_feature_engineering(self):
        """Feature engineering önerileri"""
        html = '<section><h2>9️⃣ FEATURE ENGINEERING ÖNERİLERİ</h2>'
        
        html += '<h3>📝 Kategorik Features</h3>'
        html += '<div class="code">df = pd.get_dummies(df, columns=[\'col\'], drop_first=True)</div>'
        html += '<div class="code">from sklearn.preprocessing import LabelEncoder<br>le = LabelEncoder()<br>df[\'col_encoded\'] = le.fit_transform(df[\'col\'])</div>'
        
        html += '<h3>🔢 Sayısal Features</h3>'
        html += '<div class="code"># Polynomial<br>df[\'col_squared\'] = df[\'col\'] ** 2<br>df[\'col_sqrt\'] = np.sqrt(df[\'col\'])</div>'
        html += '<div class="code"># Interaction<br>df[\'col1_col2\'] = df[\'col1\'] * df[\'col2\']<br>df[\'col1_col2_ratio\'] = df[\'col1\'] / (df[\'col2\'] + 1)</div>'
        html += '<div class="code"># Binning<br>df[\'col_binned\'] = pd.qcut(df[\'col\'], q=5)<br>df[\'col_binned_encoded\'] = pd.factorize(df[\'col_binned\'])[0]</div>'
        
        html += '<h3>🗓️ Tarih Features</h3>'
        html += '<div class="code">df[\'date\'] = pd.to_datetime(df[\'date\'])<br>df[\'year\'] = df[\'date\'].dt.year<br>df[\'month\'] = df[\'date\'].dt.month<br>df[\'day\'] = df[\'date\'].dt.day<br>df[\'dayofweek\'] = df[\'date\'].dt.dayofweek<br>df[\'is_weekend\'] = df[\'dayofweek\'].isin([5, 6]).astype(int)</div>'
        
        html += '<h3>⚙️ Scaling</h3>'
        html += '<div class="code">from sklearn.preprocessing import StandardScaler, MinMaxScaler<br>scaler = StandardScaler()<br>df_scaled = scaler.fit_transform(df[numeric_cols])</div>'
        
        html += '</section>'
        return html
    
    def _section_quality(self):
        """Veri kalitesi puanı"""
        # Kalite hesapla
        score = 100
        issues = []
        
        missing_pct = self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100
        if missing_pct > 10:
            score -= 20
            issues.append(('Yüksek Eksik Veri', f'{missing_pct:.1f}%', 'error'))
        elif missing_pct > 5:
            score -= 10
            issues.append(('Eksik Veri', f'{missing_pct:.1f}%', 'warning'))
        
        dup_pct = self.df.duplicated().sum() / len(self.df) * 100
        if dup_pct > 5:
            score -= 15
            issues.append(('Tekrarlayan Kayıtlar', f'{dup_pct:.1f}%', 'error'))
        elif dup_pct > 1:
            score -= 5
            issues.append(('Tekrarlayan Kayıtlar', f'{dup_pct:.1f}%', 'warning'))
        
        if len(self.df) < 100:
            score -= 20
            issues.append(('Çok Küçük Dataset', f'{len(self.df)} satır', 'error'))
        
        score = max(0, min(100, score))
        
        html = '<section><h2>🔟 VERİ KALİTESİ PUANI / DATA QUALITY SCORE</h2>'
        
        if score >= 70:
            score_class = 'score-good'
            status = '🟢 İYİ / GOOD'
        elif score >= 40:
            score_class = 'score-medium'
            status = '🟡 ORTA / MEDIUM'
        else:
            score_class = 'score-bad'
            status = '🔴 KÖTÜ / BAD'
        
        html += f'<div class="quality-score {score_class}">{score}/100 {status}</div>'
        
        if len(issues) == 0:
            html += '<div class="success"><strong>✓</strong> Sorun bulunmamaktadır</div>'
        else:
            html += '<h3>Bulunan Sorunlar / Issues Found</h3>'
            html += '<table><tr><th>Sorun / Issue</th><th>Değer / Value</th><th>Şiddet / Severity</th></tr>'
            for issue, value, severity in issues:
                icon = '❌' if severity == 'error' else '⚠'
                html += f'<tr><td>{issue}</td><td>{value}</td><td>{icon} {severity.upper()}</td></tr>'
            html += '</table>'
        
        html += '<h3>Özet / Summary</h3>'
        html += '<ul>'
        html += f'<li>Veri boyutu: {len(self.df):,} satır × {len(self.df.columns)} sütun</li>'
        html += f'<li>Eksik veri oranı: {missing_pct:.2f}%</li>'
        html += f'<li>Tekrarlayan kayıt oranı: {dup_pct:.2f}%</li>'
        html += f'<li>Sayısal sütun: {len(self.df.select_dtypes(include=[np.number]).columns)}</li>'
        html += f'<li>Kategorik sütun: {len(self.df.select_dtypes(include=["object"]).columns)}</li>'
        html += '</ul>'
        
        html += '</section>'
        return html

# ==========================================
# KULLANIM ÖRNEĞİ
# ==========================================

if __name__ == "__main__":
    # Örnek veri oluştur
    df = pd.DataFrame({
        'age': np.random.normal(35, 15, 1000),
        'salary': np.random.normal(50000, 15000, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'experience': np.random.randint(0, 30, 1000),
        'target': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
    })
    
    # Biraz eksik veri ekle
    df.loc[np.random.choice(df.index, 50), 'salary'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'age'] = np.nan
    
    # Rapor oluştur
    eda = EDAtoHTML(df, target='target', lang='tr')
    eda.generate('eda_report.html')
    
    print("\n✅ EDA raporu başarıyla oluşturuldu!")
    print("📊 eda_report.html dosyasını tarayıcıda aç")