"""
TRAIN vs TEST EDA COMPARISON REPORT
Kaggle için Train/Test Karşılaştırması
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

class EDATrainTestReport:
    def __init__(self, train_df, test_df, target=None, lang='tr'):
        self.train = train_df
        self.test = test_df
        self.target = target
        self.lang = lang
        self.time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def generate(self, filename='eda_train_test.html'):
        """Train vs Test raporu oluştur"""
        html = self._build_html()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"✓ {filename} oluşturuldu!")
    
    def _build_html(self):
        """HTML rapor inşa et"""
        
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
                max-width: 1600px;
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
            header h1 { font-size: 2.5em; margin-bottom: 10px; }
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
            .meta-item strong { display: block; color: #667eea; font-size: 1.2em; margin-bottom: 5px; }
            .meta-item span { color: #666; font-size: 0.9em; }
            .content { padding: 40px; }
            section { margin-bottom: 40px; padding: 25px; background: #f9f9f9; border-radius: 10px; border-left: 5px solid #667eea; }
            h2 { color: #667eea; font-size: 1.8em; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #667eea; }
            h3 { color: #764ba2; font-size: 1.2em; margin-top: 15px; margin-bottom: 10px; }
            
            /* Train vs Test Comparison */
            .comparison {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 20px 0;
            }
            .comparison-box {
                padding: 20px;
                border-radius: 8px;
                border: 2px solid #667eea;
            }
            .train-box { background: #e3f2fd; }
            .test-box { background: #f3e5f5; }
            .comparison-box h4 {
                color: #667eea;
                margin-bottom: 10px;
                font-size: 1.1em;
            }
            .comparison-box p {
                margin: 5px 0;
                font-size: 0.95em;
            }
            
            /* Data Leakage Detection */
            .leakage-warning {
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }
            .leakage-error {
                background: #f8d7da;
                border-left: 4px solid #dc3545;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }
            .leakage-success {
                background: #d4edda;
                border-left: 4px solid #28a745;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
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
            }
            td {
                padding: 12px 15px;
                border-bottom: 1px solid #eee;
            }
            tr:hover { background: #f5f5f5; }
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
            .warning { background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0; border-radius: 5px; }
            .success { background: #d4edda; border-left: 4px solid #28a745; padding: 15px; margin: 15px 0; border-radius: 5px; }
            .error { background: #f8d7da; border-left: 4px solid #dc3545; padding: 15px; margin: 15px 0; border-radius: 5px; }
            
            @media (max-width: 768px) {
                .comparison { grid-template-columns: 1fr; }
                header h1 { font-size: 1.8em; }
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
        
        header = f"""
        <header>
            <h1>📊 TRAIN vs TEST EDA RAPORU</h1>
            <p>Kaggle Yarışması Veri Analizi</p>
            <p style="margin-top: 10px; font-size: 0.9em;">📅 {self.time}</p>
        </header>
        """
        
        metadata = self._create_metadata()
        
        content = '<div class="content">'
        content += self._section_overview()
        content += self._section_missing_comparison()
        content += self._section_duplicates_comparison()
        content += self._section_data_leakage()
        content += self._section_numeric_comparison()
        content += self._section_categorical_comparison()
        content += self._section_outliers_comparison()
        content += self._section_target_analysis()
        content += self._section_warnings()
        content += '</div>'
        
        footer = """
        <footer>
            <p>Bu rapor Train/Test veri setlerinin karşılaştırması için oluşturulmuştur</p>
            <p>Data Leakage ve Distribution Shift'i görmek için önemlidir!</p>
        </footer>
        """
        
        html = f"""
        <!DOCTYPE html>
        <html lang="tr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Train vs Test EDA</title>
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
    
    def _create_metadata(self):
        """Metadata bölümü"""
        return f"""
        <div class="metadata">
            <div class="meta-item">
                <strong style="color: #1e88e5;">{len(self.train):,}</strong>
                <span>Train Satırları</span>
            </div>
            <div class="meta-item">
                <strong style="color: #d81b60;">{len(self.test):,}</strong>
                <span>Test Satırları</span>
            </div>
            <div class="meta-item">
                <strong style="color: #388e3c;">{len(self.train.columns)}</strong>
                <span>Sütunlar</span>
            </div>
            <div class="meta-item">
                <strong>{round(len(self.train) / (len(self.train) + len(self.test)) * 100, 1)}%</strong>
                <span>Train Oranı</span>
            </div>
            <div class="meta-item">
                <strong>{round(len(self.test) / (len(self.train) + len(self.test)) * 100, 1)}%</strong>
                <span>Test Oranı</span>
            </div>
        </div>
        """
    
    def _section_overview(self):
        """Genel bakış"""
        html = '<section><h2>1️⃣ GENEL KARŞILAŞTIRMA / OVERVIEW</h2>'
        
        html += '<div class="comparison">'
        html += '<div class="comparison-box train-box"><h4>🔵 TRAIN SETİ</h4>'
        html += f'<p><strong>Satırlar:</strong> {len(self.train):,}</p>'
        html += f'<p><strong>Sütunlar:</strong> {len(self.train.columns)}</p>'
        html += f'<p><strong>Bellek:</strong> {self.train.memory_usage(deep=True).sum() / 1024**2:.2f} MB</p>'
        html += f'<p><strong>Eksik:</strong> {self.train.isnull().sum().sum()}</p>'
        html += f'<p><strong>Tekrarlar:</strong> {self.train.duplicated().sum()}</p>'
        html += '</div>'
        
        html += '<div class="comparison-box test-box"><h4>🟣 TEST SETİ</h4>'
        html += f'<p><strong>Satırlar:</strong> {len(self.test):,}</p>'
        html += f'<p><strong>Sütunlar:</strong> {len(self.test.columns)}</p>'
        html += f'<p><strong>Bellek:</strong> {self.test.memory_usage(deep=True).sum() / 1024**2:.2f} MB</p>'
        html += f'<p><strong>Eksik:</strong> {self.test.isnull().sum().sum()}</p>'
        html += f'<p><strong>Tekrarlar:</strong> {self.test.duplicated().sum()}</p>'
        html += '</div>'
        html += '</div>'
        
        html += '</section>'
        return html
    
    def _section_missing_comparison(self):
        """Eksik veri karşılaştırması"""
        html = '<section><h2>2️⃣ EKSİK VERİ KARŞILAŞTIRMASI / MISSING DATA</h2>'
        
        train_missing = self.train.isnull().sum()
        test_missing = self.test.isnull().sum()
        
        # Sütun sütun karşılaştır
        all_cols = set(self.train.columns) | set(self.test.columns)
        
        html += '<table><tr><th>Sütun</th><th>Train Eksik</th><th>Test Eksik</th><th>Fark</th><th>Uyarı</th></tr>'
        
        for col in sorted(all_cols):
            train_null = train_missing.get(col, 0)
            test_null = test_missing.get(col, 0)
            train_pct = train_null / len(self.train) * 100 if col in self.train.columns else 0
            test_pct = test_null / len(self.test) * 100 if col in self.test.columns else 0
            
            diff = abs(train_pct - test_pct)
            
            if diff > 10:
                warning = '⚠️ FARK VAR!'
            elif col not in self.train.columns:
                warning = '❌ TRAIN\'DA YOK'
            elif col not in self.test.columns:
                warning = '❌ TEST\'TE YOK'
            else:
                warning = '✓'
            
            html += f'<tr><td>{col}</td><td>{train_pct:.2f}%</td><td>{test_pct:.2f}%</td><td>{diff:.2f}%</td><td>{warning}</td></tr>'
        
        html += '</table>'
        html += '</section>'
        return html
    
    def _section_duplicates_comparison(self):
        """Tekrarlayan kayıtlar"""
        html = '<section><h2>3️⃣ TEKRARlayan KAYITLAR / DUPLICATES</h2>'
        
        train_dups = self.train.duplicated().sum()
        test_dups = self.test.duplicated().sum()
        train_dup_pct = train_dups / len(self.train) * 100
        test_dup_pct = test_dups / len(self.test) * 100
        
        html += '<div class="comparison">'
        html += f'<div class="comparison-box train-box"><h4>🔵 TRAIN</h4><p>{train_dups} tekrar ({train_dup_pct:.2f}%)</p></div>'
        html += f'<div class="comparison-box test-box"><h4>🟣 TEST</h4><p>{test_dups} tekrar ({test_dup_pct:.2f}%)</p></div>'
        html += '</div>'
        
        if train_dups > 0 or test_dups > 0:
            html += '<div class="warning">⚠️ Tekrarlayan kayıtlar bulundu!</div>'
            html += '<div class="code">train = train.drop_duplicates()\ntest = test.drop_duplicates()</div>'
        
        html += '</section>'
        return html
    
    def _section_data_leakage(self):
        """Data Leakage Tespiti - ÇOK ÖNEMLİ!"""
        html = '<section><h2>⚡ DATA LEAKAGE TESPİTİ (ÇOK ÖNEMLİ!) / DATA LEAKAGE</h2>'
        
        leakage_found = False
        
        # 1. Train ve Test'te aynı satırlar var mı?
        html += '<h3>1. Aynı Satırlar Test Edildi mi?</h3>'
        
        # Tam kopya kontrol
        train_concat = self.train.astype(str).apply(lambda x: '|'.join(x), axis=1)
        test_concat = self.test.astype(str).apply(lambda x: '|'.join(x), axis=1)
        exact_duplicates = train_concat.isin(test_concat).sum()
        
        if exact_duplicates > 0:
            html += f'<div class="leakage-error">❌ SORUN! {exact_duplicates} satır hem Train hem Test\'te var!</div>'
            leakage_found = True
        else:
            html += '<div class="leakage-success">✓ Tam kopya satır yok</div>'
        
        # 2. ID'lerde örtüşme var mı?
        html += '<h3>2. ID Değerleri Örtüşüyor mu?</h3>'
        
        id_cols = [col for col in self.train.columns if 'id' in col.lower()]
        for id_col in id_cols:
            if id_col in self.train.columns and id_col in self.test.columns:
                train_ids = set(self.train[id_col])
                test_ids = set(self.test[id_col])
                overlap = len(train_ids & test_ids)
                
                if overlap > 0:
                    html += f'<div class="leakage-warning">⚠️ {id_col}: {overlap} ortak ID var!</div>'
                    leakage_found = True
                else:
                    html += f'<div class="leakage-success">✓ {id_col}: ID\'lerde örtüşme yok</div>'
        
        # 3. Hedef değişken
        if self.target and self.target in self.test.columns:
            html += '<h3>3. Hedef Değişken</h3>'
            html += '<div class="leakage-warning">⚠️ TEST SET\'TE HEDEF DEĞER VAR! Normalde olmamalı.</div>'
            leakage_found = True
        
        # 4. Kategorik değişkenlerde far
        html += '<h3>4. Kategorik Değişkenlerde Dağılım Farkı</h3>'
        cat_cols = self.train.select_dtypes(include=['object']).columns.tolist()
        
        distribution_diff = False
        for col in cat_cols[:5]:  # İlk 5'i kontrol et
            if col in self.test.columns:
                train_vals = set(self.train[col].dropna().unique())
                test_vals = set(self.test[col].dropna().unique())
                
                new_in_test = test_vals - train_vals
                if len(new_in_test) > 0:
                    html += f'<div class="leakage-warning">⚠️ {col}: Test\'te {len(new_in_test)} yeni kategori var!</div>'
                    distribution_diff = True
        
        if not distribution_diff:
            html += '<div class="leakage-success">✓ Kategorik değişkenlerde yeni değer yok</div>'
        
        # 5. Sayısal değişkenlerde istatistik farkı
        html += '<h3>5. Sayısal Değişkenlerde İstatistik Farkı</h3>'
        numeric_cols = self.train.select_dtypes(include=[np.number]).columns.tolist()
        
        html += '<table><tr><th>Değişken</th><th>Train Mean</th><th>Test Mean</th><th>Fark %</th></tr>'
        large_diff = False
        
        for col in numeric_cols[:10]:
            if col in self.test.columns:
                train_mean = self.train[col].mean()
                test_mean = self.test[col].mean()
                
                if pd.notna(train_mean) and pd.notna(test_mean) and train_mean != 0:
                    diff_pct = abs((test_mean - train_mean) / train_mean * 100)
                    
                    if diff_pct > 20:
                        warning = '⚠️ FARK VAR!'
                        large_diff = True
                    else:
                        warning = '✓'
                    
                    html += f'<tr><td>{col}</td><td>{train_mean:.2f}</td><td>{test_mean:.2f}</td><td>{diff_pct:.1f}%</td></tr>'
        
        html += '</table>'
        
        # Sonuç
        html += '<h3>📋 Sonuç</h3>'
        if leakage_found or large_diff:
            html += '<div class="leakage-error">❌ DATA LEAKAGE UYARISI!</div>'
            html += '<p>Train ve Test setleri arasında anormal bir ilişki/fark var. Lütfen kontrol edin!</p>'
        else:
            html += '<div class="leakage-success">✅ LEAKAGE BULUNMADI - Güvenli!</div>'
        
        html += '</section>'
        return html
    
    def _section_numeric_comparison(self):
        """Sayısal değişkenler"""
        html = '<section><h2>4️⃣ SAYISAL DEĞİŞKENLER KARŞILAŞTIRMASI / NUMERIC</h2>'
        
        numeric_cols = self.train.select_dtypes(include=[np.number]).columns.tolist()
        
        html += '<table><tr><th>Variable</th><th>Train Mean</th><th>Test Mean</th><th>Train Std</th><th>Test Std</th><th>Fark</th></tr>'
        
        for col in numeric_cols[:20]:
            if col in self.test.columns:
                train_mean = self.train[col].mean()
                test_mean = self.test[col].mean()
                train_std = self.train[col].std()
                test_std = self.test[col].std()
                
                diff = abs(train_mean - test_mean)
                
                if diff > train_std:
                    diff_status = '⚠️ FARK'
                else:
                    diff_status = '✓'
                
                html += f'<tr><td>{col}</td><td>{train_mean:.2f}</td><td>{test_mean:.2f}</td><td>{train_std:.2f}</td><td>{test_std:.2f}</td><td>{diff_status}</td></tr>'
        
        html += '</table>'
        html += '</section>'
        return html
    
    def _section_categorical_comparison(self):
        """Kategorik değişkenler"""
        html = '<section><h2>5️⃣ KATEGORİK DEĞİŞKENLER KARŞILAŞTIRMASI / CATEGORICAL</h2>'
        
        cat_cols = self.train.select_dtypes(include=['object']).columns.tolist()
        
        html += '<table><tr><th>Variable</th><th>Train Unique</th><th>Test Unique</th><th>Train Top</th><th>Test Top</th><th>Not in Test</th></tr>'
        
        for col in cat_cols[:15]:
            if col in self.test.columns:
                train_unique = self.train[col].nunique()
                test_unique = self.test[col].nunique()
                
                train_top = self.train[col].value_counts().index[0] if len(self.train[col].value_counts()) > 0 else 'N/A'
                test_top = self.test[col].value_counts().index[0] if len(self.test[col].value_counts()) > 0 else 'N/A'
                
                train_vals = set(self.train[col].dropna().unique())
                test_vals = set(self.test[col].dropna().unique())
                not_in_test = len(train_vals - test_vals)
                
                html += f'<tr><td>{col}</td><td>{train_unique}</td><td>{test_unique}</td><td>{train_top}</td><td>{test_top}</td><td>{not_in_test}</td></tr>'
        
        html += '</table>'
        html += '</section>'
        return html
    
    def _section_outliers_comparison(self):
        """Aykırı değerler"""
        html = '<section><h2>6️⃣ AYKIRI DEĞERLER KARŞILAŞTIRMASI / OUTLIERS</h2>'
        
        numeric_cols = self.train.select_dtypes(include=[np.number]).columns.tolist()
        
        html += '<table><tr><th>Variable</th><th>Train Outlier %</th><th>Test Outlier %</th><th>Fark</th></tr>'
        
        for col in numeric_cols[:15]:
            if col in self.test.columns:
                # Train
                Q1_train = self.train[col].quantile(0.25)
                Q3_train = self.train[col].quantile(0.75)
                IQR_train = Q3_train - Q1_train
                out_train = ((self.train[col] < Q1_train - 1.5*IQR_train) | (self.train[col] > Q3_train + 1.5*IQR_train)).sum()
                out_train_pct = out_train / len(self.train) * 100
                
                # Test
                Q1_test = self.test[col].quantile(0.25)
                Q3_test = self.test[col].quantile(0.75)
                IQR_test = Q3_test - Q1_test
                out_test = ((self.test[col] < Q1_test - 1.5*IQR_test) | (self.test[col] > Q3_test + 1.5*IQR_test)).sum()
                out_test_pct = out_test / len(self.test) * 100
                
                diff = abs(out_train_pct - out_test_pct)
                
                if diff > 5:
                    status = '⚠️'
                else:
                    status = '✓'
                
                html += f'<tr><td>{col}</td><td>{out_train_pct:.2f}%</td><td>{out_test_pct:.2f}%</td><td>{diff:.2f}% {status}</td></tr>'
        
        html += '</table>'
        html += '</section>'
        return html
    
    def _section_target_analysis(self):
        """Hedef değişken (sadece train'de varsa)"""
        html = '<section><h2>7️⃣ HEDEF DEĞİŞKEN ANALİZİ / TARGET</h2>'
        
        if self.target and self.target in self.train.columns:
            target = self.train[self.target]
            
            if pd.api.types.is_numeric_dtype(target):
                html += '<h3>📈 REGRESSION</h3>'
                html += f'<p><strong>Mean:</strong> {target.mean():.4f}</p>'
                html += f'<p><strong>Median:</strong> {target.median():.4f}</p>'
                html += f'<p><strong>Std:</strong> {target.std():.4f}</p>'
                html += f'<p><strong>Skewness:</strong> {target.skew():.4f}</p>'
                
                if abs(target.skew()) > 1:
                    html += '<div class="warning">⚠️ Hedef çarpık! Log transform önerilir</div>'
            else:
                html += '<h3>🎯 CLASSIFICATION</h3>'
                vc = target.value_counts()
                html += '<table><tr><th>Class</th><th>Count</th><th>%</th></tr>'
                for cls, cnt in vc.items():
                    pct = cnt / len(self.train) * 100
                    html += f'<tr><td>{cls}</td><td>{cnt}</td><td>{pct:.2f}%</td></tr>'
                html += '</table>'
                
                if vc.iloc[0] / vc.iloc[-1] > 10:
                    html += '<div class="warning">⚠️ Yüksek dengesizlik! SMOTE kullan</div>'
        else:
            html += '<p>Test setinde hedef değişken bulunmamaktadır (normal)</p>'
        
        html += '</section>'
        return html
    
    def _section_warnings(self):
        """Genel uyarılar ve öneriler"""
        html = '<section><h2>8️⃣ ÖNERİLER VE UYARILAR / RECOMMENDATIONS</h2>'
        
        html += '<h3>✓ Yapılması Gereken İşlemler</h3>'
        html += '<div class="code">'
        html += '# 1. Train ve Test\'e aynı işlemleri uygula\n'
        html += 'train_processed = preprocess(train)\n'
        html += 'test_processed = preprocess(test)  # Aynı fonksiyon!\n\n'
        html += '# 2. Scaling\'de leakage\'ı önle\n'
        html += 'scaler.fit(train)  # SADECE train\'e fit et\n'
        html += 'train_scaled = scaler.transform(train)\n'
        html += 'test_scaled = scaler.transform(test)  # Test\'e fit etme!\n\n'
        html += '# 3. Kategorik encoding\'de\n'
        html += 'encoder.fit(train[\'col\'])  # Train\'den öğren\n'
        html += 'test_encoded = encoder.transform(test[\'col\'])  # Test\'e uyguła\n'
        html += '</div>'
        
        html += '<h3>⚠️ Data Leakage Uyarıları</h3>'
        html += '<ul>'
        html += '<li>Train statistics\'i test\'te kullanma</li>'
        html += '<li>Target leakage: Feature\'da target bilgisi olmasın</li>'
        html += '<li>Temporal leakage: Zaman sırası önemli mi kontrol et</li>'
        html += '<li>Aggregation leakage: Grouped statistics'
        html += '</ul>'
        
        html += '</section>'
        return html

# ==========================================
# KULLANIM ÖRNEĞİ
# ==========================================

if __name__ == "__main__":
    # Örnek veri (Kaggle House Prices benzeri)
    np.random.seed(42)
    
    train = pd.DataFrame({
        'Id': range(1000, 2000),
        'Age': np.random.normal(40, 15, 1000),
        'Salary': np.random.normal(50000, 15000, 1000),
        'Category': np.random.choice(['A', 'B', 'C'], 1000),
        'SalePrice': np.random.normal(200000, 50000, 1000)
    })
    
    test = pd.DataFrame({
        'Id': range(3000, 3460),
        'Age': np.random.normal(42, 15, 460),  # Biraz farklı
        'Salary': np.random.normal(48000, 16000, 460),
        'Category': np.random.choice(['A', 'B', 'C'], 460)
    })
    
    # Biraz data leakage benzeri durum ekle
    test.loc[0, 'Age'] = -50  # Imkansız değer
    train.loc[np.random.choice(train.index, 50), 'Age'] = np.nan
    
    # Rapor oluştur
    eda = EDATrainTestReport(train, test, target='SalePrice', lang='tr')
    eda.generate('eda_train_test.html')
    
    print("\n✅ Train vs Test raporu başarıyla oluşturuldu!")
    print("📊 eda_train_test.html dosyasını tarayıcıda aç")