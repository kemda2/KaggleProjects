Harika bir ekleme yapmışsın—bu “Extended Senior Data Pipeline (26–34)” bloğu pratikte Kaggle Telco Churn tarzı veri setlerinde çok gerçek işe yarar kontrolleri kapsıyor. Yine de **bir senior analistin mutlaka eklemesi gereken** bazı kritik noktalar ve **küçük ama anlamlı düzeltmeler** var. Aşağıda önce **eksik/yanlış** gördüğüm kısımları maddeler halinde belirtiyorum; ardından **düzeltilmiş ve iyileştirilmiş TAM KODU** ve **“kural kural nasıl ilerleyeceğiz” rehberini** paylaşıyorum.

***

## 🔍 Eksikler ve Düzeltmeler (Senior Review)

### 1) HTML/CSS Raporu **escape** edilmiş (🛠️ Önemli bug)

*   `ExtReporter.CSS` içinde HTML **`&lt;` ve `&gt;`** olarak kaçışlı yazılmış. Bu şekilde **çıktı dosyası ham metin gibi** görünür.  
    **Düzeltme:** Gerçek `<html> ... </html>` etiketleri kullanılmalı.

### 2) SHAP Summary Plot **figure** yanlış üretimi (🛠️ Önemli bug)

*   `shap.summary_plot` kendi figürünü üretir. `fig, ax = plt.subplots(...); shap.summary_plot(..., show=False)` şeklinde **önceden figür oluşturup** sonra `plt.gca()` almak **boş figür kaydetmenize** yol açabilir.  
    **Düzeltme:** `shap.summary_plot(..., show=False)` sonrası `fig = plt.gcf()` ile **şekli yakala** ve kaydet.

### 3) OOD (Out-of-Distribution) **oranını** yanlış hesaplıyorsun

*   Şu an `total_ood = sum(r["Count"] ...)` ile **kolon bazlı sayıları topluyorsun**; bu, **tek satırın birden çok OOD sebebiyle** birden fazla sayılmasına yol açar.  
    **Düzeltme:** **Satır bazında OOD maskesi** üret ve **en az bir OOD** olan satırların oranını hesapla.

### 4) Numerik/Sayısal kolon tespiti **güvenilir değil**

*   `num_cols = [c for c in feats if train[c].nunique() > cat_threshold]` gibi eşik mantığıyla tespit, **yüksek kardinaliteli kategorikleri** yanlışlıkla numeric muamelesi yapar.  
    **Düzeltme:** `pd.api.types.is_numeric_dtype` veya **yüksek başarıyla sayıya çevrilebilen** kolonları numeric say.

### 5) Kategorik ayrıştırma gücü (KS yerine Ki-kare / Cramér’s V) **eksik**

*   Section 30’da sadece numerikler için KS testi var. Kategorikler için **ki-kare** testi ve **Cramér’s V** ile ayrıştırıcılık ölçümü eklenmeli.

### 6) Korelasyon/Multi-kolineerlik **kontrolü eksik**

*   FE öncesi **yüksek korelasyon** (ör. |ρ|>0.95) uyarısı verilmesi, hem **model stabilitesi** hem de **aşırı-öğrenme** riskine karşı önemli.  
    **Ek:** Numeric korelasyon matrisi + eşik üstü çiftler listesi.

### 7) Permütasyon Önemi (stability) **import edilmiş ama kullanılmamış**

*   `permutation_importance` import edilmiş; **top-features** için en azından **tek fold** permütasyon önemi ölçümü ve **stabilite kontrolü** (farklı fold’larda yakın mı?) çok faydalı.

### 8) SHAP için örneklem ve bellek koruması

*   Büyük veri setlerinde SHAP maliyetli. **`sample_for_shap`** parametresi ekleyip **alt örneklem** almak gerekiyor.

### 9) İş kuralı kontrolleri dataset-agnostic uyarıları

*   Şu an **Telco** datasetine çok iyi hizmet ediyor (tenure, TotalCharges...), ancak **kolonlar yoksa** raporda **“uygulanamadı”** bilgisi var—iyi. Ayrıca **sentinel** değerlerin **gerçek dönüşümü** öneri olarak kalıyor.  
    **Not:** Bu iyi; yine de rapora **“uygularsanız, şunları değiştireceğiz”** gibi açık adım listesi ekledim.

### 10) Reprodüksiyon ve hız seçenekleri

*   Kaggle’da denenebilir; ama eğitim sayısı ve SHAP ağır.  
    **Ek:** `FAST_MODE` ve `row_sample`, `leakage_max_rows` gibi parametrelerle **kontrol** ve **hızlandırma** sağladım.

***

## ✅ Düzeltilmiş / Geliştirilmiş TAM KOD (26–34)

Aşağıdaki kod:

*   HTML raporunu **doğru** üretir,
*   SHAP summary/sactter **doğru figür** kaydeder,
*   OOD oranını **satır bazında** hesaplar,
*   Numerik/kategorik ayrımı **sağlamlaştırır**,
*   Kategorik ayrıştırma gücü (Chi-square & Cramér’s V) ekler,
*   Yüksek korelasyon **uyarısı** ekler,
*   Permütasyon önemi **stabilite** notu ekler,
*   Hız için parametreler ekler.

> **Kullanım:** Orijinal pipeline’dan `train`, `test`, `target_col`, `id_col` hazırlandıktan sonra aşağıdaki hücreyi **ayrı hücre** olarak çalıştır.  
> Dilersen tek başına da örnek CSV ile çalışıyor (en altta örnek var).

```python
# ════════════════════════════════════════════════════════════════════════════
# 🔬 EXTENDED SENIOR DATA PIPELINE v2 — EK ANALİZLER (Sectionlar 26-34)
# Kaggle için optimize edildi — HTML raporu, SHAP, OOD, Label Noise, vs.
# Gereksinimler: lightgbm, shap (Kaggle runtime'da mevcut; yoksa pip ile kurulur)
# ════════════════════════════════════════════════════════════════════════════

# (İsteğe bağlı) !pip install shap -q

import os, warnings, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ks_2samp, spearmanr, chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
import lightgbm as lgb

from itertools import combinations
from pandas.api.types import is_numeric_dtype

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid", palette="muted")

# ── Görsel tema/dizinler
PLOT_DIR = "/kaggle/working/plots_ext"
os.makedirs(PLOT_DIR, exist_ok=True)
DARK="#0d1117"; CARD="#161b22"; BLUE="#58a6ff"; TXT="#c9d1d9"; GRID="#30363d"

# ── Reprodüksiyon
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"

def dark_style(ax, title=""):
    ax.set_facecolor(CARD); ax.figure.set_facecolor(DARK)
    ax.title.set_color(BLUE); ax.xaxis.label.set_color(TXT); ax.yaxis.label.set_color(TXT)
    ax.tick_params(colors=TXT)
    for s in ax.spines.values(): s.set_color(GRID)
    if title: ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

def safe_encode(s):
    if s.dtype == "O":
        return pd.Series(LabelEncoder().fit_transform(s.astype(str)), index=s.index)
    return s

def to_numeric_safely(series, min_numeric_ratio=0.9):
    """Objeyi sayıya çevirmeyi dener; yeterli başarı yoksa None döner."""
    if is_numeric_dtype(series):
        return series
    # Obje ise dönüştürmeyi dene
    s = pd.to_numeric(series, errors="coerce")
    ratio = s.notna().mean()
    return s if ratio >= min_numeric_ratio else None

def cramers_v(conf_matrix):
    """Cramér’s V (kategorik ayrıştırma gücü)"""
    chi2 = chi2_contingency(conf_matrix)[0]
    n = conf_matrix.sum().sum()
    r, k = conf_matrix.shape
    return np.sqrt( (chi2 / n) / (min(k - 1, r - 1) + 1e-9) )

# ════════════════════════════════════════════════════════
# REPORTER
# ════════════════════════════════════════════════════════
class ExtReporter:
    CSS = """<html><head><meta charset="utf-8">
    <style>
      body{font-family:'Segoe UI',Consolas,monospace;background:#0d1117;color:#c9d1d9;padding:30px;max-width:1400px;margin:auto}
      .section{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px;margin:25px 0}
      .section h2{color:#58a6ff;border-bottom:2px solid #58a6ff;padding-bottom:8px;font-size:20px}
      .info-line{background:#1f2937;padding:10px 15px;border-radius:6px;margin:8px 0;border-left:4px solid #58a6ff;font-size:14px}
      .warn{border-left-color:#f0883e;color:#f0883e}
      .ok{border-left-color:#3fb950;color:#3fb950}
      .critical{border-left-color:#f85149;color:#f85149;font-weight:bold}
      .decision{background:#1a2332;border:2px solid #58a6ff;border-radius:8px;padding:15px;margin:12px 0;font-size:13px}
      .decision-yes{border-color:#3fb950;background:#0d2818}
      .decision-no{border-color:#f85149;background:#2d1014}
      .section table{border-collapse:collapse;width:100%;margin:15px 0;font-size:13px}
      .section table th{background:#21262d;color:#58a6ff;padding:10px 14px;border:1px solid #30363d;text-align:left}
      .section table td{padding:8px 14px;border:1px solid #30363d;background:#0d1117}
      .section table tr:hover td{background:#1a2233}
      .plot-container{text-align:center;margin:15px 0}
      .plot-container img{max-width:100%;border-radius:8px;border:1px solid #30363d}
      .rule-box{background:#0d2818;border:1px solid #3fb950;border-radius:8px;padding:14px;margin:10px 0;font-size:13px;color:#c9d1d9}
      .rule-box b{color:#3fb950}
      .code{background:#0b1220;border:1px dashed #30363d;border-radius:8px;padding:12px;margin:10px 0;font-size:12px;color:#d1d7e0}
    </style></head><body>
    <h1 style="text-align:center;color:#58a6ff">🔬 EXTENDED PIPELINE REPORT — Sections 26-34</h1>"""

    def __init__(self):
        self.h = [self.CSS]
        self.pc = 0
        self.actions = []

    def log(self, text, cls=""):
        print(text)
        c = f" {cls}" if cls else ""
        self.h.append(f'<div class="info-line{c}">{text}</div>')

    def decision(self, text, pos=True):
        cn = "decision-yes" if pos else "decision-no"
        px = "✅" if pos else "⚠️"
        full = f"{px} KARAR: {text}"
        print(f"  {full}")
        self.h.append(f'<div class="decision {cn}">{full}</div>')
        self.actions.append(text)

    def rule(self, text):
        self.h.append(f'<div class="rule-box"><b>📌 KURAL:</b> {text}</div>')

    def code(self, text):
        self.h.append(f'<div class="code">{text}</div>')

    def section(self, title):
        print(f"\n{'═'*80}\n{title}\n{'═'*80}")
        self.h.append(f'<div class="section"><h2>{title}</h2>')

    def end(self): self.h.append("</div>")

    def table(self, styled):
        from IPython.display import display
        display(styled)
        self.h.append(styled.to_html())

    def plot(self, fig, title="plot"):
        from IPython.display import display, Image
        self.pc += 1
        fname = f"ext_{self.pc:02d}_{title.replace(' ','_').lower()}.png"
        fpath = os.path.join(PLOT_DIR, fname)
        fig.savefig(fpath, dpi=150, bbox_inches="tight", facecolor=DARK, edgecolor="none")
        plt.close(fig)
        display(Image(filename=fpath))
        with open(fpath,"rb") as f: b64 = base64.b64encode(f.read()).decode()
        self.h.append(f'<div class="plot-container">data:image/png;base64,{b64}</div>')

    def save(self, path):
        self.h.append("</body></html>")
        with open(path,"w",encoding="utf-8") as f: f.write("\n".join(self.h))
        print(f"\n✅ HTML kaydedildi: {path}")

# ════════════════════════════════════════════════════════
# ANA FONKSİYON
# ════════════════════════════════════════════════════════
def extended_pipeline(
    train,
    test,
    target_col,
    id_col=None,
    cat_threshold=25,
    leakage_max_rows=20000,  # Leakage testinde satır alt-örneklem sınırı
    sample_for_shap=10000,   # SHAP için maksimum satır
    fast_mode=False          # Hızlı denemeler için modelleri küçült
):
    R = ExtReporter()
    ext_actions = []

    # ── Kolon listeleri
    feats = [c for c in train.columns if c not in [target_col, id_col]]
    # Numerikleri tespit: gerçek numeric veya yüksek oranla sayıya dönen
    num_cols = []
    for c in feats:
        if is_numeric_dtype(train[c]):
            num_cols.append(c)
        else:
            s = to_numeric_safely(train[c], min_numeric_ratio=0.9)
            if s is not None:
                num_cols.append(c)
    # Kategorikler: numerik listede olmayanlar
    cat_cols = [c for c in feats if c not in num_cols]

    # Hedefi encode et
    y_full = safe_encode(train[target_col].copy())

    # Modellemeler için encode (basit factorize + NA=-1)
    X_enc = train[feats].copy()
    for c in X_enc.select_dtypes(["object","category"]).columns:
        X_enc[c] = pd.factorize(X_enc[c].astype(str))[0]
    X_enc = X_enc.fillna(-1)

    # Hız moduna göre model parametreleri
    rf_n = 30 if fast_mode else 50
    lgb_n = 250 if fast_mode else 500

    # ════════════════════════════════════════════════════
    # 26. LEAKAGE DETECTION
    # ════════════════════════════════════════════════════
    R.section("🚨 26. LEAKAGE DETECTION (Veri Sızıntısı)")
    R.rule("Tek bir feature ile AUC > 0.95 ise → yüksek sızıntı riski, production'da olmayabilir. Kaldır.")
    R.rule("AUC>0.99 → çok büyük ihtimalle sızıntı. Kesin drop.")
    R.rule("0.85-0.95 → şüpheli. Domain kontrolü.")
    R.rule("ID-benzeri, zaman-sonrası (post-event) ipuçları ve label türevleri özellikle risklidir.")

    # Alt-örneklem (hız ve robustluk için)
    if len(X_enc) > leakage_max_rows:
        leak_idx = np.random.choice(len(X_enc), size=leakage_max_rows, replace=False)
        X_leak = X_enc.iloc[leak_idx].reset_index(drop=True)
        y_leak = y_full.iloc[leak_idx].reset_index(drop=True)
    else:
        X_leak = X_enc
        y_leak = y_full

    leak_res = []
    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for col in feats:
        try:
            xc = X_leak[[col]]
            aucs = []
            for tr_i, va_i in skf5.split(xc, y_leak):
                rf = RandomForestClassifier(n_estimators=rf_n, max_depth=4, random_state=42, n_jobs=-1)
                rf.fit(xc.iloc[tr_i], y_leak.iloc[tr_i])
                prob = rf.predict_proba(xc.iloc[va_i])[:,1]
                aucs.append(roc_auc_score(y_leak.iloc[va_i], prob))
            auc_m = float(np.mean(aucs))
            level = "🔴 LEAK!" if auc_m > 0.95 else ("🟠 ŞÜPHELİ" if auc_m > 0.85 else ("🟡 Güçlü" if auc_m > 0.75 else "✅ Normal"))
            leak_res.append({"Feature": col, "Solo_AUC": auc_m, "Risk": level})
        except Exception as e:
            pass

    df_leak = pd.DataFrame(leak_res).sort_values("Solo_AUC", ascending=False).reset_index(drop=True)
    if not df_leak.empty:
        R.table(df_leak.style.background_gradient(cmap="RdYlGn_r", subset=["Solo_AUC"]).format({"Solo_AUC": "{:.4f}"}))

    leaks = df_leak[df_leak["Solo_AUC"] > 0.95]["Feature"].tolist() if not df_leak.empty else []
    suspects = df_leak[(df_leak["Solo_AUC"] > 0.85) & (df_leak["Solo_AUC"] <= 0.95)]["Feature"].tolist() if not df_leak.empty else []

    if leaks:
        R.log(f"🔴 LEAK TESPİT: {leaks}", cls="critical")
        R.decision(f"LEAK özellikleri KALDIR: {leaks}. Model eğitiminde KULLANMA!", pos=False)
        ext_actions.append(f"🚨 LEAK DROP: {leaks}")
    else:
        R.log("✅ Belirgin leak yok.", cls="ok")

    if suspects:
        R.log(f"🟠 Şüpheli yüksek solo-AUC: {suspects}", cls="warn")
        R.decision(f"Domain kontrolü: {suspects} → production'da bu bilgiler var mı?", pos=False)
        ext_actions.append(f"⚠️ Leak şüphelisi kontrol et: {suspects}")
    R.end()

    # ════════════════════════════════════════════════════
    # 27. LABEL KALİTESİ & TARGET NOISE
    # ════════════════════════════════════════════════════
    R.section("🏷️ 27. LABEL KALİTESİ & TARGET NOISE")
    R.rule("Aynı feature profiline sahip satırlarda farklı target → label noise. %5'ten fazlası sorunlu.")
    R.rule("Conflict rate > %10 → sample_weight/robust loss.")
    R.rule("5–10% → label_smoothing veya threshold ayarı.")

    dup_check = train[feats + [target_col]].copy()
    for c in dup_check.select_dtypes(["object","category"]).columns:
        dup_check[c] = dup_check[c].astype(str)

    key_cols = feats[:min(6, len(feats))]
    grp = dup_check.groupby(key_cols)[target_col].agg(["nunique","count"])
    conflicted = grp[grp["nunique"] > 1]
    conflict_count = int(conflicted["count"].sum()) if not conflicted.empty else 0
    conflict_rate = (conflict_count / len(train) * 100) if len(train) else 0

    R.log(f"📊 Conflict satır: {conflict_count:,} ({conflict_rate:.2f}%)")
    if conflict_rate > 10:
        R.log(f"🔴 Yüksek conflict oranı: %{conflict_rate:.2f}", cls="critical")
        R.decision("sample_weight veya noise-robust loss kullan. Label kalitesi düşük.", pos=False)
        ext_actions.append("⚠️ Label noise yüksek → sample_weight")
    elif conflict_rate > 5:
        R.log(f"🟠 Orta conflict: %{conflict_rate:.2f}", cls="warn")
        R.decision("LightGBM'de label_smoothing=0.05 dene.", pos=False)
        ext_actions.append("⚠️ Orta label noise → label_smoothing")
    else:
        R.log(f"✅ Düşük conflict: %{conflict_rate:.2f}", cls="ok")
        R.decision("Label kalitesi iyi. Normal devam et.", pos=True)

    # Exact duplicate profili (tüm feature'larla)
    full_dup = train[feats].copy()
    for c in full_dup.select_dtypes(["object","category"]).columns:
        full_dup[c] = full_dup[c].astype(str)
    exact_dups = int(full_dup.duplicated(keep=False).sum())
    R.log(f"📊 Birebir feature duplicate (farklı target olabilir): {exact_dups:,}")
    if exact_dups > 0:
        exact_conflict = train[full_dup.duplicated(keep=False)].groupby(feats)[target_col].nunique()
        n_conflict = int((exact_conflict > 1).sum())
        R.log(f"   → Bunların {n_conflict} grubunda farklı target var")
        if n_conflict > 0:
            R.decision(f"{n_conflict} grup birebir aynı feature ama farklı target. drop_duplicates YAPMA, ağırlıklandır.", pos=False)
    R.end()

    # ════════════════════════════════════════════════════
    # 28. BUSINESS LOGIC VALIDATION (Telco örnekleri)
    # ════════════════════════════════════════════════════
    R.section("🏢 28. BUSINESS LOGIC VALIDATION")
    R.rule("TotalCharges ≈ tenure × MonthlyCharges olmalı. Sapma > %50 → veri problemi.")
    R.rule("tenure=0 ama TotalCharges>0 → imkansız. İşaretle/düzelt.")
    R.rule("Sentinel değerleri normalize et (örn. 'No internet service' → 'No').")

    n_bad = 0
    if all(c in train.columns for c in ["TotalCharges","tenure","MonthlyCharges"]):
        tc = pd.to_numeric(train["TotalCharges"], errors="coerce")
        ten = pd.to_numeric(train["tenure"], errors="coerce")
        mc = pd.to_numeric(train["MonthlyCharges"], errors="coerce")

        expected = ten * mc
        ratio = tc / (expected + 1e-9)
        bad_ratio = ((ratio < 0.5) | (ratio > 2.0)) & (ten > 0)
        n_bad = int(bad_ratio.sum())
        R.log(f"📊 TotalCharges/expected oranı bozuk satır: {n_bad:,} ({n_bad/len(train)*100:.2f}%)")

        if n_bad > 0:
            R.decision(f"{n_bad} satırda TotalCharges anomalisi. 'is_charges_anomaly' flag oluştur.", pos=False)
            ext_actions.append("🏷️ is_charges_anomaly flag ekle")

            ybin = y_full
            churn_anom = ybin[bad_ratio].mean() * 100
            churn_norm = ybin[~bad_ratio].mean() * 100
            R.log(f"   Anomali satır churn: %{churn_anom:.2f} vs Normal: %{churn_norm:.2f}")
            if abs(churn_anom - churn_norm) > 10:
                R.decision("Anomali satırlarında churn paterni farklı. FLAG güçlü feature olabilir.", pos=True)

        zero_ten_pos_tc = int(((ten == 0) & (tc > 0)).sum())
        if zero_ten_pos_tc > 0:
            R.log(f"🔴 tenure=0 ama TotalCharges>0: {zero_ten_pos_tc} satır", cls="critical")
            R.decision("Bu satırlarda TotalCharges'ı 0 yap veya is_anomaly flag ekle.", pos=False)
            ext_actions.append(f"🔧 tenure=0/TC>0 düzelt: {zero_ten_pos_tc} satır")
        else:
            R.log("✅ tenure=0 & TotalCharges>0 anomalisi yok.", cls="ok")

        # Residual plot
        residual = tc - expected
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(residual.dropna(), bins=60, color=BLUE, edgecolor=GRID, alpha=0.8)
        dark_style(axes[0], "Charges Residual (TC - tenure×MC)")
        # Scatter
        samp = np.random.choice(len(expected), size=min(2000, len(expected)), replace=False)
        axes[1].scatter(expected.iloc[samp], tc.iloc[samp], alpha=0.3, color=BLUE, s=10)
        mx = float(np.nanmax(expected.iloc[samp]))
        x_line = np.linspace(0, mx, 100)
        axes[1].plot(x_line, x_line, color="#f85149", lw=2, label="Perfect")
        dark_style(axes[1], "Expected vs Actual TotalCharges")
        axes[1].legend(facecolor=CARD, edgecolor=GRID, labelcolor=TXT)
        fig.set_facecolor(DARK); fig.tight_layout()
        R.plot(fig, "charges_residual")
    else:
        R.log("⚠️ TotalCharges/tenure/MonthlyCharges kolonları bulunamadı.", cls="warn")

    sentinel_vals = ["No internet service", "No phone service", "Unknown", "N/A", "na", "none", "None"]
    sent_counts = {}
    for c in cat_cols:
        ser = train[c].astype(str)
        for sv in sentinel_vals:
            cnt = int((ser == sv).sum())
            if cnt > 0:
                sent_counts[f"{c}='{sv}'"] = cnt
    if sent_counts:
        R.log(f"⚠️ Sentinel değerler tespit edildi:", cls="warn")
        for k, v in sorted(sent_counts.items(), key=lambda x: -x[1]):
            R.log(f"   {k}: {v:,} satır")
        R.decision("Sentinel değerleri normalize et (örn. hepsini 'No'). FE adım 1.", pos=True)
        ext_actions.append("🔧 Sentinel → 'No' dönüşümü")
    R.end()

    # ════════════════════════════════════════════════════
    # 29. COHORT DRIFT (tenure bazlı)
    # ════════════════════════════════════════════════════
    R.section("⏳ 29. COHORT DRIFT ANALİZİ (tenure bazlı)")
    R.rule("Erken (tenure<12) ve geç (tenure>48) cohort arasında churn farkı büyükse → cohort-based FE.")
    R.rule("Fark > 15 puan → tenure gruplara özel feature/fitting düşün.")

    if "tenure" in train.columns:
        tmp_c = train.copy()
        tmp_c["tbin"] = y_full.values
        tmp_c["tenure_num"] = pd.to_numeric(tmp_c["tenure"], errors="coerce")
        tmp_c["cohort"] = pd.cut(tmp_c["tenure_num"],
                                 bins=[0, 12, 24, 36, 48, 1000],
                                 labels=["0-12","12-24","24-36","36-48","48+"],
                                 right=True)
        cohort_stats = tmp_c.groupby("cohort")["tbin"].agg(["mean","count"])
        cohort_stats.columns = ["Churn_Rate%","Count"]
        cohort_stats["Churn_Rate%"] *= 100

        R.table(cohort_stats.style.background_gradient(cmap="RdYlGn_r", subset=["Churn_Rate%"]).format({"Churn_Rate%":"{:.2f}"}))

        rates = cohort_stats["Churn_Rate%"].values
        cohort_range = float(np.nanmax(rates) - np.nanmin(rates))
        rho_c, _ = spearmanr(np.arange(len(rates)), rates)

        R.log(f"📊 Cohort churn spread: {cohort_range:.1f} puan | Spearman: {rho_c:.3f}")
        if cohort_range > 20:
            R.decision(f"Cohort farkı {cohort_range:.1f} puan — BÜYÜK. tenure_group kategorik feature oluştur.", pos=True)
            ext_actions.append("🎯 tenure_group kategorik feature")
        elif cohort_range > 10:
            R.decision(f"Cohort farkı {cohort_range:.1f} puan — orta. tenure_bin ekle.", pos=True)
            ext_actions.append("🎯 tenure_bin feature")
        else:
            R.decision(f"Cohort farkı düşük ({cohort_range:.1f}p). Lineer tenure yeterli.", pos=True)

        # Örnek ek: MonthlyCharges cohort'ta değişiyor mu?
        if "MonthlyCharges" in train.columns:
            mc_cohorts = []
            for coh, grp in tmp_c.groupby("cohort"):
                mc_cohorts.append({"Cohort": str(coh), "MC_Mean": pd.to_numeric(grp["MonthlyCharges"], errors="coerce").mean(),
                                   "MC_Std": pd.to_numeric(grp["MonthlyCharges"], errors="coerce").std()})
            df_coh = pd.DataFrame(mc_cohorts)
            R.table(df_coh.style.background_gradient(cmap="Blues", subset=["MC_Mean"]).format(precision=2))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        cohort_stats["Churn_Rate%"].plot(kind="bar", ax=axes[0], color=[
            "#f85149" if r > 30 else "#f0883e" if r > 20 else "#3fb950" for r in cohort_stats["Churn_Rate%"]], edgecolor=GRID)
        dark_style(axes[0], "Churn Rate by Tenure Cohort"); axes[0].set_ylabel("Churn Rate (%)")
        cohort_stats["Count"].plot(kind="bar", ax=axes[1], color=BLUE, edgecolor=GRID)
        dark_style(axes[1], "Sample Count by Tenure Cohort")
        fig.set_facecolor(DARK); fig.tight_layout()
        R.plot(fig, "cohort_drift")
    else:
        R.log("⚠️ tenure kolonu yok.", cls="warn")
    R.end()

    # ════════════════════════════════════════════════════
    # 30. STRATIFIED TARGET EDA (KDE + Kategorik Ki-kare)
    # ════════════════════════════════════════════════════
    R.section("📊 30. STRATİFİED TARGET EDA (Churn=Yes vs No)")
    R.rule("KS > 0.3 → güçlü sayısal ayrıştırıcı. <0.05 → zayıf.")
    R.rule("Kategorikler için ki-kare p-değeri küçük ve Cramér’s V yüksek ise güçlü ayrıştırıcı.")

    # Numerikler için KS
    if num_cols:
        y0 = train[y_full == 0]
        y1 = train[y_full == 1]
        strat_res = []
        for col in num_cols:
            a_raw = y0[col]
            b_raw = y1[col]
            a = pd.to_numeric(a_raw, errors="coerce").dropna()
            b = pd.to_numeric(b_raw, errors="coerce").dropna()
            if len(a) > 20 and len(b) > 20:
                ks_val, ks_p = ks_2samp(a, b)
                mean_diff = b.mean() - a.mean()
                strat_res.append({"Feature": col, "KS": ks_val, "P_val": ks_p,
                                  "Mean_No": a.mean(), "Mean_Yes": b.mean(),
                                  "Mean_Diff": mean_diff,
                                  "Power": "🔴 Güçlü" if ks_val > 0.3 else ("🟡 Orta" if ks_val > 0.15 else "⚪ Zayıf")})
        df_strat = pd.DataFrame(strat_res).sort_values("KS", ascending=False).reset_index(drop=True)
        if not df_strat.empty:
            R.table(df_strat.style.background_gradient(cmap="YlGn", subset=["KS"]).format(precision=4))

            top_feats = df_strat[df_strat["KS"] > 0.15]["Feature"].tolist()[:6]
            weak_feats = df_strat[df_strat["KS"] < 0.05]["Feature"].tolist()
            if weak_feats:
                R.log(f"⚪ Çok zayıf numerik ayrıştırıcılar: {weak_feats}", cls="warn")
                R.decision(f"Bu numerik feature'lar düşürülmeli veya transform edilmeli: {weak_feats}", pos=False)
                ext_actions.append(f"⚠️ Zayıf numerik KDE: {weak_feats}")

            # KDE/Hist (üst KS'ler)
            if top_feats:
                nc = min(3, len(top_feats)); nr = (len(top_feats) + nc - 1) // nc
                fig, axes = plt.subplots(nr, nc, figsize=(15, nr * 4))
                axes_flat = np.array(axes).flatten() if len(top_feats) > 1 else [axes]
                for i, col in enumerate(top_feats):
                    a_d = pd.to_numeric(y0[col], errors="coerce").dropna()
                    b_d = pd.to_numeric(y1[col], errors="coerce").dropna()
                    axes_flat[i].hist(a_d, bins=40, alpha=0.5, color="#3fb950", label="Churn=No", density=True, edgecolor=GRID)
                    axes_flat[i].hist(b_d, bins=40, alpha=0.5, color="#f85149", label="Churn=Yes", density=True, edgecolor=GRID)
                    dark_style(axes_flat[i], col)
                    axes_flat[i].legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TXT)
                for j in range(len(top_feats), len(axes_flat)):
                    axes_flat[j].set_visible(False)
                fig.suptitle("Numeric Features — Churn=Yes vs No", color=BLUE, fontsize=16, fontweight="bold")
                fig.set_facecolor(DARK); fig.tight_layout(rect=[0,0,1,0.95])
                R.plot(fig, "stratified_kde")
    else:
        R.log("⚠️ Numerik kolon bulunamadı.", cls="warn")

    # Kategorikler için Ki-kare + Cramér’s V
    cat_power = []
    if cat_cols:
        for col in cat_cols:
            tab = pd.crosstab(train[col].astype(str), y_full)
            if tab.shape[0] >= 2 and tab.shape[1] == 2:  # en az 2 kategori ve 2 sınıf
                try:
                    chi2, p, dof, exp = chi2_contingency(tab)
                    v = cramers_v(tab)
                    cat_power.append({"Feature": col, "Chi2_p": p, "CramersV": v,
                                      "Power": "🔴 Güçlü" if v > 0.3 else ("🟡 Orta" if v > 0.15 else "⚪ Zayıf")})
                except:
                    pass
        if cat_power:
            df_catpow = pd.DataFrame(cat_power).sort_values(["CramersV"], ascending=False)
            R.table(df_catpow.style.background_gradient(cmap="YlOrRd", subset=["CramersV"]).format({"Chi2_p":"{:.2e}","CramersV":"{:.3f}"}))
    R.end()

    # ════════════════════════════════════════════════════
    # 30-b. KORELASYON / MULTI-KOLİNEERLİK UYARISI
    # ════════════════════════════════════════════════════
    R.section("🧩 30-b. KORELASYON ANALİZİ (Numeric)")
    R.rule("|ρ| > 0.95 → çok yüksek korelasyon; biri düşürülebilir (model stabilitesi).")
    high_corr_pairs = []
    if len(num_cols) >= 2:
        # Numerik kolonları güvenle sayıya çevir
        Nm = pd.DataFrame()
        for c in num_cols:
            s = to_numeric_safely(train[c], min_numeric_ratio=0.9)
            if s is not None:
                Nm[c] = s
        if Nm.shape[1] >= 2:
            cm = Nm.corr(numeric_only=True).abs()
            for i, j in combinations(cm.columns, 2):
                if cm.loc[i, j] > 0.95:
                    high_corr_pairs.append((i, j, float(cm.loc[i, j])))
            if high_corr_pairs:
                high_corr_pairs = sorted(high_corr_pairs, key=lambda x: -x[2])
                R.log(f"⚠️ Yüksek korelasyonlu çiftler (ilk 10): {high_corr_pairs[:10]}", cls="warn")
                R.decision("Bu çiftlerden biri FE'de düşürülebilir veya ayrı modellerde denenebilir.", pos=False)
                ext_actions.append("🔧 Yüksek korelasyonlu çiftlerin sadeleştirilmesi")
    else:
        R.log("ℹ️ Yeterli numerik kolon yok veya dönüştürülemedi.", cls="warn")
    R.end()

    # ════════════════════════════════════════════════════
    # 31. SHAP DEĞERLERİ & INTERACTION ANALİZİ (+ Permütasyon Önemi)
    # ════════════════════════════════════════════════════
    R.section("🔬 31. SHAP & PERMUTASYON ÖNEMİ")
    R.rule("SHAP mean abs > 0.05 → modelde anlamlı etki.")
    R.rule("Permütasyon önemi ile SHAP uyumluysa önem stabil.")
    R.rule("SHAP scatter V-şekli/renk gradyanı → non-linear; binning/log/WOE düşünülebilir.")

    try:
        # Alt örneklem (SHAP maliyeti kontrol)
        idx_all = np.arange(len(X_enc))
        if len(idx_all) > sample_for_shap:
            shap_idx = np.random.choice(idx_all, size=sample_for_shap, replace=False)
        else:
            shap_idx = idx_all
        Xs = X_enc.iloc[shap_idx].reset_index(drop=True)
        ys = y_full.iloc[shap_idx].reset_index(drop=True)

        skf_shap = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        tr_i, va_i = list(skf_shap.split(Xs, ys))[0]
        m_shap = lgb.LGBMClassifier(
            n_estimators=300 if not fast_mode else 150,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            is_unbalance=True,
            random_state=42,
            n_jobs=-1
        )
        m_shap.fit(Xs.iloc[tr_i], ys.iloc[tr_i])

        import shap
        explainer = shap.TreeExplainer(m_shap)
        shap_vals = explainer.shap_values(Xs.iloc[va_i])
        if isinstance(shap_vals, list): shap_vals = shap_vals[1]  # binary için class1

        shap_mean = np.abs(shap_vals).mean(axis=0)
        shap_df = pd.DataFrame({"Feature": feats, "SHAP_Mean": shap_mean}).sort_values("SHAP_Mean", ascending=False).reset_index(drop=True)
        R.table(shap_df.style.bar(subset=["SHAP_Mean"], color="#636efa").format({"SHAP_Mean": "{:.4f}"}))

        top5 = shap_df.head(5)["Feature"].tolist()
        R.log(f"🏆 Top 5 SHAP feature: {top5}")

        # SHAP summary (bar)
        shap.summary_plot(shap_vals, Xs.iloc[va_i], feature_names=feats, plot_type="bar", show=False, color=BLUE)
        fig = plt.gcf(); ax = plt.gca(); dark_style(ax, "SHAP Feature Importance")
        fig.set_facecolor(DARK); fig.tight_layout()
        R.plot(fig, "shap_summary")

        # Top-3 scatter
        top3 = shap_df.head(3)["Feature"].tolist()
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
        for i, col in enumerate(top3):
            idx = feats.index(col)
            axes2[i].scatter(Xs.iloc[va_i][col], shap_vals[:, idx],
                             alpha=0.3, c=shap_vals[:, idx], cmap="RdBu_r", s=8)
            dark_style(axes2[i], f"SHAP: {col}")
            axes2[i].axhline(0, color=GRID, lw=1, linestyle="--")
            axes2[i].set_xlabel(col); axes2[i].set_ylabel("SHAP value")
        fig2.set_facecolor(DARK); fig2.tight_layout()
        R.plot(fig2, "shap_scatter_top3")

        # Interaction proxy (SHAP korelasyonu top-5)
        if len(top5) >= 2:
            shap_top5 = pd.DataFrame(shap_vals[:, [feats.index(f) for f in top5]], columns=top5)
            inter_corr = shap_top5.corr()
            fig3, ax3 = plt.subplots(figsize=(7, 5))
            sns.heatmap(inter_corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                        ax=ax3, linewidths=0.5, linecolor=GRID, cbar=True)
            dark_style(ax3, "SHAP Interaction (Top-5)")
            fig3.tight_layout()
            R.plot(fig3, "shap_interaction")

            strong_inter = [(top5[i], top5[j]) for i in range(len(top5)) for j in range(i+1,len(top5)) if abs(inter_corr.iloc[i,j]) > 0.3]
            if strong_inter:
                R.log(f"⚡ Güçlü SHAP interaksiyon çiftleri: {strong_inter}", cls="ok")
                R.decision(f"Bu çiftler için çarpım feature oluştur: {strong_inter}", pos=True)
                ext_actions.append(f"🎯 SHAP interaction pairs FE: {strong_inter}")

        # Non-linear kontrol
        nonlinear = []
        for col in top5:
            idxc = feats.index(col)
            corr_lin = abs(np.corrcoef(Xs.iloc[va_i][col].fillna(-1), shap_vals[:, idxc])[0, 1])
            if corr_lin < 0.7:
                nonlinear.append(col)
        if nonlinear:
            R.log(f"🔀 Non-linear SHAP pattern: {nonlinear}", cls="warn")
            R.decision(f"Bu feature'lar için binning/log transform dene: {nonlinear}", pos=False)
            ext_actions.append(f"🔧 Non-linear → binning/log: {nonlinear}")

        # Permütasyon önemi (tek fold) — stabilite
        perm = permutation_importance(m_shap, Xs.iloc[va_i], ys.iloc[va_i], n_repeats=5, random_state=42, n_jobs=-1)
        perm_imp = pd.DataFrame({"Feature": feats, "Perm_Importance": perm.importances_mean}).sort_values("Perm_Importance", ascending=False)
        R.table(perm_imp.head(15).style.bar(subset=["Perm_Importance"], color="#ff7f0e").format({"Perm_Importance":"{:.4f}"}))
        R.log("ℹ️ SHAP ile permütasyon öneminin ilk 10 içindeki kesişimi stabilite göstergesidir.")
        inter = set(perm_imp.head(10)["Feature"]).intersection(set(shap_df.head(10)["Feature"]))
        R.log(f"   SHAP ∩ Perm_Importance (Top10): {sorted(list(inter))}")

    except Exception as e:
        R.log(f"⚠️ SHAP/Permütasyon hesaplanamadı: {e}", cls="warn")
    R.end()

    # ════════════════════════════════════════════════════
    # 32. HARD SAMPLE ANALİZİ (CV tutarsız satırlar)
    # ════════════════════════════════════════════════════
    R.section("🎯 32. HARD SAMPLE ANALİZİ (Model için zor satırlar)")
    R.rule("5 fold'un ≥4'ünde yanlış tahmin edilenler → 'hard samples'.")
    R.rule("Hard sample oranı > %15 → agresif FE/ensembling/encoding gerekebilir.")

    fold_preds = np.zeros((len(train), 5))
    skf_hard = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_i, va_i) in enumerate(skf_hard.split(X_enc, y_full)):
        m_h = lgb.LGBMClassifier(
            n_estimators=lgb_n,
            learning_rate=0.05,
            max_depth=6,
            is_unbalance=True,
            random_state=42,
            n_jobs=-1
        )
        m_h.fit(
            X_enc.iloc[tr_i], y_full.iloc[tr_i],
            eval_set=[(X_enc.iloc[va_i], y_full.iloc[va_i])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        fold_preds[va_i, fold] = m_h.predict_proba(X_enc.iloc[va_i])[:, 1]

    pred_labels = (fold_preds > 0.5).astype(int)
    correct = (pred_labels == y_full.values.reshape(-1, 1)).sum(axis=1)
    wrong_count = 5 - correct
    is_hard = wrong_count >= 4

    hard_rate = is_hard.mean() * 100
    R.log(f"📊 Hard sample sayısı: {int(is_hard.sum()):,} ({hard_rate:.2f}%)")
    if hard_rate > 15:
        R.log(f"🔴 Hard sample oranı yüksek: %{hard_rate:.2f}", cls="critical")
        R.decision("Daha agresif FE/encoding gerekli. Hard sample profillerine odaklan.", pos=False)
        ext_actions.append(f"⚠️ Hard sample %{hard_rate:.1f} → güçlü FE gerekli")
    elif hard_rate > 8:
        R.log(f"🟠 Orta hard sample: %{hard_rate:.2f}", cls="warn")
        R.decision("Target encoding / stacking bu satırlara yardımcı olabilir.", pos=False)
    else:
        R.log(f"✅ Düşük hard sample: %{hard_rate:.2f}", cls="ok")
        R.decision("Model sağlıklı. Hard sample'lar manageable.", pos=True)

    if is_hard.sum() > 10 and num_cols:
        hard_df = train[is_hard].copy()
        normal_df = train[~is_hard].copy()
        R.log("\n📊 Hard vs Normal — Numeric Mean Karşılaştırma:")
        comp_rows = []
        for col in num_cols:
            hm = pd.to_numeric(hard_df[col], errors="coerce").mean()
            nm = pd.to_numeric(normal_df[col], errors="coerce").mean()
            comp_rows.append({"Feature": col, "Hard_Mean": hm, "Normal_Mean": nm,
                               "Diff%": abs(hm - nm) / (abs(nm) + 1e-9) * 100})
        df_comp = pd.DataFrame(comp_rows).sort_values("Diff%", ascending=False)
        R.table(df_comp.style.background_gradient(cmap="OrRd", subset=["Diff%"]).format(precision=3))
        big_diff = df_comp[df_comp["Diff%"] > 20]["Feature"].tolist()
        if big_diff:
            R.decision(f"Hard sample'lar şu feature'larda farklı: {big_diff}. Özel transform uygula.", pos=True)
            ext_actions.append(f"🎯 Hard sample odaklı FE: {big_diff}")
    R.end()

    # ════════════════════════════════════════════════════
    # 33. TEST SET OOD ANALİZİ (satır-bazlı oran dahil)
    # ════════════════════════════════════════════════════
    R.section("🔭 33. TEST SET OOD (Out-of-Distribution) ANALİZİ")
    R.rule("Test'te yeni kategoriler → 'Unknown' token veya Target Encoding.")
    R.rule("Numeric OOD (train range dışı) → clip/Winsorize.")
    R.rule("OOD oranını satır bazında hesapla (en az bir OOD taşıyan satırlar).")

    ood_results = []
    # Kategorik: yeni değerler + satır maskesi
    test_ood_row_mask = pd.Series(False, index=test.index)
    for col in cat_cols:
        tr_vals = set(train[col].astype(str).dropna().unique())
        te_series = test[col].astype(str)
        new_mask = ~te_series.isin(tr_vals) & te_series.notna()
        cnt = int(new_mask.sum())
        if cnt > 0:
            new_vals = list(set(te_series[new_mask].unique()))
            ood_results.append({"Col": col, "Type": "Yeni Kategori", "Count": cnt, "Values": str(new_vals[:5])})
            test_ood_row_mask |= new_mask

    # Numeric: range dışı + satır maskesi
    for col in num_cols:
        tr_num = to_numeric_safely(train[col])
        te_num = to_numeric_safely(test[col])
        if tr_num is None or te_num is None:
            continue
        tr_min, tr_max = float(tr_num.min()), float(tr_num.max())
        out_of_range_mask = (te_num < tr_min) | (te_num > tr_max)
        out_of_range = int(out_of_range_mask.sum())
        if out_of_range > 0:
            ood_results.append({"Col": col, "Type": "Range Dışı", "Count": out_of_range,
                                 "Values": f"Train:[{tr_min:.2f},{tr_max:.2f}] Test_min={float(te_num.min()):.2f} max={float(te_num.max()):.2f}"})
            test_ood_row_mask |= out_of_range_mask

    if ood_results:
        df_ood = pd.DataFrame(ood_results)
        R.table(df_ood.style.background_gradient(cmap="OrRd", subset=["Count"]))
        ood_row_rate = test_ood_row_mask.mean() * 100
        R.log(f"📊 En az 1 OOD içeren test satırı oranı: %{ood_row_rate:.1f}")
        cat_ood = [r["Col"] for r in ood_results if r["Type"] == "Yeni Kategori"]
        num_ood = [r["Col"] for r in ood_results if r["Type"] == "Range Dışı"]

        if cat_ood:
            R.decision(f"Yeni kategoriler: {sorted(set(cat_ood))} → Target Encoding veya 'Unknown' token.", pos=False)
            ext_actions.append(f"🔧 OOD kategoriler: {sorted(set(cat_ood))} → target encoding")
        if num_ood:
            R.decision(f"Range dışı numeric: {sorted(set(num_ood))} → Winsorize (clip 1–99p).", pos=False)
            ext_actions.append(f"🔧 OOD numeric: {sorted(set(num_ood))} → Winsorize")
    else:
        R.log("✅ Test'te OOD değer yok.", cls="ok")
        R.decision("Test ve train aynı değer aralığında. Güvenli.", pos=True)
    R.end()

    # ════════════════════════════════════════════════════
    # 34. FE ÖNCESİ NİHAİ HAZIRLIK SKORU & ÖZET
    # ════════════════════════════════════════════════════
    R.section("🏁 34. FE ÖNCESİ NİHAİ HAZIRLIK SKORU & ÖZET")
    R.rule("Bu bölüm tüm test çıktılarından bir skor türetir ve FE'ye geçiş kararı verir.")

    readiness_score = 0
    readiness_notes = []

    # Leakage
    if not leaks:
        readiness_score += 2
    else:
        readiness_notes.append(f"🚨 {len(leaks)} leak feature var — FE ÖNCESİ KALDIR")

    # Label noise
    if conflict_rate < 5:
        readiness_score += 2
    elif conflict_rate < 10:
        readiness_score += 1
        readiness_notes.append("⚠️ Orta label noise — label_smoothing ekle")
    else:
        readiness_notes.append("🔴 Yüksek label noise — sample_weight zorunlu")

    # Business logic
    if "TotalCharges" in train.columns:
        if n_bad < len(train) * 0.02:
            readiness_score += 1
        else:
            readiness_notes.append("⚠️ Business logic anomalileri yüksek — flag ekle")

    # OOD
    if not ood_results:
        readiness_score += 2
    elif len(ood_results) < 3:
        readiness_score += 1
        readiness_notes.append("⚠️ Küçük OOD problemi — Winsorize/unknown token")
    else:
        readiness_notes.append("🔴 Büyük OOD problemi — target encoding şart")

    # Hard samples
    if hard_rate < 8:
        readiness_score += 2
    elif hard_rate < 15:
        readiness_score += 1
    else:
        readiness_notes.append("🔴 Yüksek hard sample — agresif FE gerekli")

    R.log(f"\n📊 HAZIRLIK SKORU: {readiness_score}/9")
    if readiness_score >= 7:
        R.log("🟢 YÜKSELİŞ HAZIR", cls="ok")
        R.decision("Veri temiz, FE'ye geç. Aşağıdaki sırayla ilerle.", pos=True)
    elif readiness_score >= 5:
        R.log("🟡 ORTA HAZIRLIK", cls="warn")
        R.decision("Kritik düzeltmeleri yap, sonra FE'ye geç.", pos=False)
    else:
        R.log("🔴 DÜŞÜK HAZIRLIK", cls="critical")
        R.decision("Temel problemleri çöz. FE'ye geçme.", pos=False)

    R.log("\n📋 KRİTİK NOTLAR:")
    for n in readiness_notes:
        R.log(f"  {n}", cls="critical" if "🚨" in n else "warn")

    R.log("\n📋 TÜM EKSTENSİYON AKSİYONLARI:")
    for i, a in enumerate(ext_actions, 1):
        cl = "critical" if "🚨" in a else ("warn" if "⚠" in a else "ok")
        R.log(f"  {i}. {a}", cls=cl)

    R.end()
    R.save("/kaggle/working/extended_pipeline_report.html")
    return ext_actions, fold_preds

# ═══════════════════════════════════════════════════════════
# ÇALIŞTIRMA TALİMATI
# ═══════════════════════════════════════════════════════════
# 1) Orijinal pipeline'dan train/test/target_col/id_col hazır olmalı.
# 2) Bu hücreyi çalıştır.
# 3) Çağır:
#       ext_actions, fold_preds = extended_pipeline(
#           train, test, target_col, id_col,
#           fast_mode=False     # hızlı denemelerde True
#       )
#
# Tek başına örnek:
#   train = pd.read_csv("/kaggle/input/playground-series-s6e3/train.csv")
#   test  = pd.read_csv("/kaggle/input/playground-series-s6e3/test.csv")
#   ext_actions, fold_preds = extended_pipeline(train, test, "Churn", "id", fast_mode=True)
```

***

## 🧭 “Kural Kural Nasıl İlerliyoruz?” — Uygulama Rehberi

1.  **Leakage (26):**
    *   **Kural:** Tek feature’la AUC>0.95 ise **drop**. 0.85–0.95 arası **domain check**.
    *   **Aksiyon:** `leaks` listesini **FE öncesi** direkt düş. Şüpheliler için veri sözlüğü/iş ekibi onayı al.

2.  **Label Noise (27):**
    *   **Kural:** Aynı profil + farklı target → noise. %5–10 arası ise `label_smoothing`, >10 ise `sample_weight`/robust loss.
    *   **Aksiyon:** `grp` ve `conflict_rate` raporuna bak; orana göre **kayıp fonksiyonu** stratejini belirle.

3.  **Business Rules (28):**
    *   **Kural:** `TotalCharges ≈ tenure×MonthlyCharges`, sentinel normalizasyonu.
    *   **Aksiyon:** `is_charges_anomaly` flag’i oluştur, `tenure=0 & TC>0` düzelt/flag. `sentinel` değerleri **FE’nin ilk adımında** normalize et.

4.  **Cohort Drift (29):**
    *   **Kural:** Tenure cohort farkı >10–20 puan ise bin/segment feature’ları.
    *   **Aksiyon:** `tenure_group` gibi kategorik/ordinal feature ekle; farklı cohortlara **farklı etki** bekle.

5.  **Stratified EDA (30):**
    *   **Kural:** Numerikler için **KS**; Kategorikler için **Ki-kare & Cramér’s V**.
    *   **Aksiyon:** Zayıf numerikleri transform/düş. Güçlü kategorikler için **target encoding** ve **interactions** önceliklendir.

6.  **Korelasyon (30-b):**
    *   **Kural:** |ρ|>0.95 çiftlerden birini düşür; multikolinerliği azalt.
    *   **Aksiyon:** Yüksek korelasyonlu çift listesine bak, **gereksiz** olanı FE’de çıkar.

7.  **SHAP + Permütasyon (31):**
    *   **Kural:** Yüksek SHAP ortalamalı feature’lar önemli; permütasyon önemli ile **uyum** stabilite göstergesidir.
    *   **Aksiyon:** SHAP **top-5** için non-linear desen varsa **binning/log**. Güçlü interaksiyon çiftleri için **çarpım**/etkileşim feature’ları üret.

8.  **Hard Samples (32):**
    *   **Kural:** 5 fold’un ≥4’ünde yanlış olanlar “hard”. Oran >%15 ise agresif FE.
    *   **Aksiyon:** Hard sample profiline bak; çok farklı duran feature’larda **segment bazlı** veya **özel transform** dene.

9.  **Test OOD (33):**
    *   **Kural:** Yeni kategori → **Unknown/Target Encoding**; range dışı sayısal → **Winsorize/clip**.
    *   **Aksiyon:** OOD **satır bazlı oran** ve kolon listesine göre FE’de **normalize** et.

10. **Hazırlık Skoru (34):**

*   **Kural:** 7/9 ve üzeri → FE’ye geç; 5–6/9 → küçük düzeltmelerden sonra FE; altı → önce problemleri çöz.
*   **Aksiyon:** Rapor sonundaki **kritik notlar** ve **ext\_actions** listesine göre sırayla ilerle.

***

## Küçük ama Yararlı İpuçları (Kaggle için)

*   **fast\_mode=True** ile ilk denemeleri hızla çalıştır, sinyalleri gör; sonrasında tam modda koş.
*   SHAP ve leakage için **alt örneklem** zaten kodda var; gerekiyorsa `leakage_max_rows` ve `sample_for_shap`’ı düşür.
*   FE sonrası **modelleme hücrende**:
    *   `is_charges_anomaly`, `tenure_group`, sentinel normalizasyonu, OOD clip / Unknown token, SHAP interaksiyonları ve **yüksek korelasyon sadeleştirmesi** adımlarını sırayla uygula.
    *   **Target encoding** için CV-safe yaklaşım uygula (fold içinde fit, o fold’un valid setine uygula).

***

İstersen bu bloğu senin mevcut notebook’una **entegre** edip, FE adımların için **otomatik dönüşümlü** bir sürüm de hazırlayabilirim (ör. `apply_recommended_fe(train, test, actions)` gibi). Nasıl ilerleyelim—**önce raporu üretip** beraber aksiyon listesini mi netleştirelim, yoksa **direkt dönüşüm yapan** bir yardımcı fonksiyon da ekleyeyim mi?


# ekle

