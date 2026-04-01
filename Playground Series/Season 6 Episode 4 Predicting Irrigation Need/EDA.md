# . Öneri KAGGLE GRANDMASTER EDA

```py

```

# 3. Öneri KAGGLE GRANDMASTER EDA

```py
# =============================================================================
# KAGGLE GRANDMASTER EDA v4.0 — 42 BÖLÜM, KAPSAMLI YAZILI RAPOR
# Sadece EDA, model yok. Tüm bilgiler metin olarak çıktı verir.
# =============================================================================

import numpy as np
import pandas as pd
import warnings
import os
from itertools import combinations
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
warnings.filterwarnings('ignore')


# =============================================================================
# 1. KONFİGÜRASYON
# =============================================================================

class EDAConfig:
    BASE_PATH           = "/kaggle/input/playground-series-s6e4/"
    TRAIN_FILE          = "train.csv"
    TEST_FILE           = "test.csv"
    SAMPLE_SUB_FILE     = "sample_submission.csv"
    ORIGINAL_FILE       = None          # Orijinal dataset varsa path

    TARGET_COL          = None          # None = otomatik
    ID_COL              = None

    CAT_THRESHOLD       = 10
    HIGH_CARD_THRESHOLD = 100
    DRIFT_THRESHOLD     = 0.05
    MISSING_THRESHOLD   = 0.50
    RARE_THRESHOLD      = 0.01
    QUASI_CONST_THRESH  = 0.97

    TOP_N               = 20
    TOP_N_CAT           = 10
    BIN_COUNT           = 10
    REG_NUNIQUE_THRESH  = 30
    INTERACT_TOP_K      = 8
    MI_SAMPLE           = 50_000
    CLUSTER_THRESH      = 0.7          # Korelasyon clustering eşiği


# =============================================================================
# 2. OTOMATİK ALGILAMA
# =============================================================================

class AutoDetector:
    @staticmethod
    def detect_target(df, cfg):
        if cfg.TARGET_COL and cfg.TARGET_COL in df.columns:
            return cfg.TARGET_COL
        for c in ['target','Target','TARGET','label','Label','class','Class',
                   'y','Y','outcome','Outcome','survived','Survived',
                   'SalePrice','price','Price','irrigation_need','Irrigation_Need']:
            if c in df.columns:
                return c
        cols = [c for c in df.columns if c.lower() not in ['id']]
        return cols[-1] if cols else None

    @staticmethod
    def detect_id(df, cfg):
        if cfg.ID_COL and cfg.ID_COL in df.columns:
            return cfg.ID_COL
        for c in ['id','ID','Id','index','sample_id','row_id']:
            if c in df.columns:
                return c
        return None

    @staticmethod
    def detect_problem(df, target, cfg):
        d  = df[target].dtype
        nu = df[target].nunique()
        if d in ('float64','float32') and nu > cfg.REG_NUNIQUE_THRESH:
            return 'regression'
        if d in ('int64','int32')     and nu > cfg.REG_NUNIQUE_THRESH:
            return 'regression'
        return 'binary' if nu == 2 else ('multiclass' if nu > 2 else 'regression')

    @staticmethod
    def detect_features(df, feat_cols, thr):
        cats, nums = [], []
        for c in feat_cols:
            if df[c].dtype == 'object' or df[c].dtype.name == 'category' or df[c].nunique() <= thr:
                cats.append(c)
            else:
                nums.append(c)
        return cats, nums


# =============================================================================
# 3. RAPOR SINIFI
# =============================================================================

class Report:
    def __init__(self):
        self.L = []

    def add(self, t=""):
        self.L.append(t)

    def sec(self, title, ch="="):
        self.L += [f"\n{ch*80}", f"{title:^80}", f"{ch*80}\n"]

    def sub(self, title):
        self.L += [f"\n{'─'*80}", f"  {title}", f"{'─'*80}\n"]

    def dump(self):
        print("\n".join(self.L))

    def save(self, fn="eda_report.txt"):
        with open(fn, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.L))
        print(f"\n💾 Saved: {fn}")


# =============================================================================
# 4. ANA EDA SINIFI
# =============================================================================

class GrandmasterEDA:

    def __init__(self, cfg):
        self.cfg     = cfg
        self.R       = Report()
        self.train   = self.test = self.sub = self.orig = None
        self.target  = self.id_col = self.ptype = None
        self.feats   = self.cats = self.nums = []
        self.quality = {}                              # Kalite skorları

    # ── helpers ────────────────────────────────────────────────────── #

    def _enc_target(self):
        t = self.train[self.target]
        return LabelEncoder().fit_transform(t) if t.dtype == 'object' else t.values.copy()

    def _corr(self, a, b):
        try:
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 10:
                return 0.0
            r = np.corrcoef(a[m], b[m])[0, 1]
            return 0.0 if np.isnan(r) else r
        except Exception:
            return 0.0

    def _cramers_v(self, x, y):
        try:
            ct = pd.crosstab(x, y)
            chi2 = stats.chi2_contingency(ct)[0]
            n = len(x)
            k = min(ct.shape) - 1
            return np.sqrt(chi2 / (n * k)) if k > 0 and n > 0 else 0.0
        except Exception:
            return 0.0

    def _entropy(self, series):
        p = series.value_counts(normalize=True, dropna=False).values
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    # ── load ───────────────────────────────────────────────────────── #

    def load(self):
        bp = self.cfg.BASE_PATH
        self.train = pd.read_csv(bp + self.cfg.TRAIN_FILE)
        self.test  = pd.read_csv(bp + self.cfg.TEST_FILE)

        sp = bp + self.cfg.SAMPLE_SUB_FILE
        if os.path.exists(sp):
            self.sub = pd.read_csv(sp)

        if self.cfg.ORIGINAL_FILE and os.path.exists(self.cfg.ORIGINAL_FILE):
            self.orig = pd.read_csv(self.cfg.ORIGINAL_FILE)

        self.target = AutoDetector.detect_target(self.train, self.cfg)
        self.id_col = AutoDetector.detect_id(self.train, self.cfg)
        self.ptype  = AutoDetector.detect_problem(self.train, self.target, self.cfg)
        self.feats  = [c for c in self.train.columns if c not in [self.target, self.id_col]]
        self.cats, self.nums = AutoDetector.detect_features(
            self.train, self.feats, self.cfg.CAT_THRESHOLD)

    # ── run ────────────────────────────────────────────────────────── #

    def run(self):
        R = self.R
        R.sec("KAGGLE GRANDMASTER EDA  v4.0  (42 SECTIONS)")
        R.add(f"  Timestamp : {pd.Timestamp.now()}")
        R.add(f"  Target    : {self.target} ({self.ptype.upper()})")

        self._s01(); self._s02(); self._s03(); self._s04(); self._s05()
        self._s06(); self._s07(); self._s08(); self._s09(); self._s10()
        self._s11(); self._s12(); self._s13(); self._s14(); self._s15()
        self._s16(); self._s17(); self._s18(); self._s19(); self._s20()
        self._s21(); self._s22(); self._s23(); self._s24(); self._s25()
        self._s26(); self._s27(); self._s28(); self._s29(); self._s30()
        self._s31(); self._s32(); self._s33(); self._s34(); self._s35()
        self._s36(); self._s37(); self._s38(); self._s39(); self._s40()
        self._s41(); self._s42()

        R.dump()
        return R

    # ================================================================ #
    # S01 – TEMEL BİLGİLER
    # ================================================================ #
    def _s01(self):
        R = self.R; R.sec("1. TEMEL VERİ BİLGİLERİ")
        R.add(f"  Train : {self.train.shape[0]:>10,} × {self.train.shape[1]:>4}")
        R.add(f"  Test  : {self.test.shape[0]:>10,} × {self.test.shape[1]:>4}")
        R.add(f"  Ratio : {self.test.shape[0]/self.train.shape[0]:.2f}x")
        R.add(f"  Target: {self.target}  dtype={self.train[self.target].dtype}")
        R.add(f"  ID    : {self.id_col}")
        R.add(f"  Cat   : {len(self.cats):>3}  {self.cats[:6]}")
        R.add(f"  Num   : {len(self.nums):>3}  {self.nums[:6]}")
        R.add(f"\n  Dtypes:")
        for dt, cnt in self.train[self.feats].dtypes.value_counts().items():
            R.add(f"    {str(dt):<15}: {cnt}")

    # ================================================================ #
    # S02 – BELLEK
    # ================================================================ #
    def _s02(self):
        R = self.R; R.sec("2. BELLEK KULLANIMI")
        tr = self.train.memory_usage(deep=True).sum()/1024**2
        te = self.test.memory_usage(deep=True).sum()/1024**2
        R.add(f"  Train={tr:.1f} MB  Test={te:.1f} MB  Total={tr+te:.1f} MB")
        dc = sum(1 for c in self.feats
                 if self.train[c].dtype=='float64' or
                    (self.train[c].dtype=='int64' and
                     self.train[c].max() < 32767 and self.train[c].min() > -32768))
        if dc:
            R.add(f"  ⚡ {dc} kolon downcast edilebilir")

    # ================================================================ #
    # S03 – SAMPLE SUBMISSION
    # ================================================================ #
    def _s03(self):
        R = self.R; R.sec("3. SAMPLE SUBMISSION ANALİZİ")
        if self.sub is None:
            R.add("  ⚠️ Bulunamadı"); return
        R.add(f"  Shape  : {self.sub.shape}")
        R.add(f"  Columns: {list(self.sub.columns)}")
        for c in self.sub.columns:
            R.add(f"    {c}: dtype={self.sub[c].dtype}  nunique={self.sub[c].nunique()}  "
                   f"sample={self.sub[c].iloc[0]}")
        non_id = [c for c in self.sub.columns if c.lower() != 'id']
        if len(non_id) == 1:
            col = non_id[0]
            if self.sub[col].dtype in ('float64','float32'):
                R.add(f"\n  📝 Probability/regression output bekleniyor → {col}")
            else:
                R.add(f"\n  📝 Class label bekleniyor → {col}")
        elif len(non_id) > 1:
            R.add(f"\n  📝 Multi-column: muhtemelen per-class probability")
            R.add(f"     → {non_id}")

    # ================================================================ #
    # S04 – ORİJİNAL DATASET
    # ================================================================ #
    def _s04(self):
        R = self.R; R.sec("4. ORİJİNAL DATASET (PS-Specific)")
        if self.orig is not None:
            R.add(f"  ✅ Yüklendi: {self.orig.shape}")
            R.add(f"  Ortak kolon: {len(set(self.train.columns)&set(self.orig.columns))}")
            R.add(f"  → Train'e eklenince skor genelde artar")
        else:
            R.add(f"  ℹ️ Yüklenmedi. PS yarışmasıysa orijinal veriyi bul ve ekle.")

    # ================================================================ #
    # S05 – HEDEF DEĞİŞKEN
    # ================================================================ #
    def _s05(self):
        R = self.R; R.sec("5. HEDEF DEĞİŞKEN ANALİZİ")
        t = self.train[self.target]

        if self.ptype == 'regression':
            d = t.describe()
            for k in d.index:
                R.add(f"  {k:<10}: {d[k]:>15.4f}")
            sk, ku = t.skew(), t.kurtosis()
            R.add(f"  {'skew':<10}: {sk:>15.4f}")
            R.add(f"  {'kurtosis':<10}: {ku:>15.4f}")
            R.add(f"  zeros={int((t==0).sum()):,}  neg={int((t<0).sum()):,}")
            if abs(sk) > 1:
                R.add(f"  ⚠️ Skewed → log1p / Box-Cox target transform düşün")
            return

        vc = t.value_counts().sort_index()
        vr = t.value_counts(normalize=True).sort_index()
        R.add(f"  {'Class':<20} {'Count':>10} {'Ratio':>10}")
        R.add(f"  {'-'*45}")
        for v in vc.index:
            bar = "█" * int(vr[v]*40)
            R.add(f"  {str(v):<20} {vc[v]:>10,} {vr[v]:>9.2%}  {bar}")
        imb = vr.max()/vr.min()
        R.add(f"\n  Imbalance: {imb:.1f}:1")
        if imb > 10:   R.add("  🚨 HIGHLY IMBALANCED")
        elif imb > 3:  R.add("  ⚠️ MODERATELY IMBALANCED")
        else:          R.add("  ✅ BALANCED")
        self.quality['imbalance'] = imb

    # ================================================================ #
    # S06 – EKSİK VERİ
    # ================================================================ #
    def _s06(self):
        R = self.R; R.sec("6. EKSİK VERİ ANALİZİ")
        tr_m = self.train.isnull().sum()
        te_m = self.test.isnull().sum()
        cols = sorted(set(tr_m[tr_m>0].index) |
                      set(c for c in te_m[te_m>0].index if c != self.target))
        if not cols:
            R.add("  ✅ Eksik veri yok"); self.quality['missing'] = 0; return

        self.quality['missing'] = len(cols)
        R.add(f"  {'Col':<25} {'Tr#':>8} {'Tr%':>7} {'Te#':>8} {'Te%':>7} {'Act':>10}")
        R.add(f"  {'-'*70}")
        for c in cols:
            tn = tr_m.get(c,0); tp = tn/len(self.train)*100
            ten = te_m.get(c,0) if c in self.test.columns else 0
            tep = ten/len(self.test)*100 if c in self.test.columns else 0
            act = "DROP" if tp > 50 else ("DRIFT!" if abs(tp-tep) > 10 else "IMPUTE")
            R.add(f"  {c:<25} {tn:>8,} {tp:>6.1f}% {ten:>8,} {tep:>6.1f}% {act:>10}")

        # Missingness korelasyonu
        miss_cols = [c for c in cols if c in self.train.columns and tr_m[c]>0]
        if len(miss_cols) >= 2:
            mc = self.train[miss_cols].isnull().corr()
            pairs = [(miss_cols[i], miss_cols[j], abs(mc.iloc[i,j]))
                     for i in range(len(miss_cols))
                     for j in range(i+1, len(miss_cols))
                     if abs(mc.iloc[i,j]) > 0.5]
            if pairs:
                R.add(f"\n  ⚠️ Birlikte eksik (corr>0.5):")
                for c1,c2,c in sorted(pairs, key=lambda x:x[2], reverse=True)[:5]:
                    R.add(f"    {c1} ↔ {c2}: {c:.3f}")

    # ================================================================ #
    # S07 – DUPLICATE
    # ================================================================ #
    def _s07(self):
        R = self.R; R.sec("7. DUPLICATE ANALİZİ")
        da = self.train.duplicated().sum()
        df = self.train[self.feats].duplicated().sum()
        R.add(f"  Tam satır dup   : {da:>8,} ({da/len(self.train):.2%})")
        R.add(f"  Feature-only dup: {df:>8,} ({df/len(self.train):.2%})")
        self.quality['duplicates'] = df

        if df > 0:
            try:
                mask = self.train[self.feats].duplicated(keep=False)
                conflict = self.train[mask].groupby(
                    self.feats, dropna=False)[self.target].nunique()
                nc = int((conflict > 1).sum())
                R.add(f"  Çelişkili (aynı X farklı y): {nc:,}")
                if nc > 0:
                    R.add(f"  ⚠️ Noisy labels → label smoothing veya cleaning")
                self.quality['label_conflicts'] = nc
            except Exception:
                R.add("  ℹ️ Çelişki kontrolü yapılamadı")

    # ================================================================ #
    # S08 – CONSTANT / QUASI-CONSTANT
    # ================================================================ #
    def _s08(self):
        R = self.R; R.sec("8. CONSTANT / QUASI-CONSTANT")
        const, quasi = [], []
        for c in self.feats:
            nu = self.train[c].nunique(dropna=False)
            if nu <= 1:
                const.append(c)
            else:
                tf = self.train[c].value_counts(normalize=True, dropna=False).iloc[0]
                if tf > self.cfg.QUASI_CONST_THRESH:
                    quasi.append((c, tf))

        self.quality['constant'] = len(const)
        self.quality['quasi_const'] = len(quasi)
        if const:
            R.add(f"  🚨 CONSTANT ({len(const)}) → DROP: {const}")
        else:
            R.add("  ✅ Constant yok")
        if quasi:
            R.add(f"  ⚠️ QUASI-CONST ({len(quasi)}):")
            for c,f in quasi:
                R.add(f"    {c}: {f:.1%}")

    # ================================================================ #
    # S09 – RARE CATEGORIES
    # ================================================================ #
    def _s09(self):
        R = self.R; R.sec("9. RARE CATEGORY TESPİTİ")
        if not self.cats:
            R.add("  ℹ️ Cat yok"); return
        thr = self.cfg.RARE_THRESHOLD
        R.add(f"  Eşik: <{thr:.1%}\n")

        any_rare = False
        for c in self.cats:
            vc = self.train[c].value_counts(normalize=True)
            rare = vc[vc < thr]
            if len(rare) > 0:
                any_rare = True
                R.add(f"  {c:<25} #rare={len(rare):>4}  rare_total={rare.sum():.1%}  "
                       f"samples={list(rare.index[:3])}")
        if not any_rare:
            R.add("  ✅ Rare yok")

        # Unseen
        R.add(f"\n  UNSEEN (test'te var train'de yok):")
        unseen = False
        for c in self.cats:
            if c not in self.test.columns: continue
            u = set(self.test[c].dropna().unique()) - set(self.train[c].dropna().unique())
            if u:
                unseen = True
                R.add(f"    {c}: {len(u)} unseen {list(u)[:5]}")
        if not unseen:
            R.add("    ✅ Yok")

    # ================================================================ #
    # S10 – KATEGORİK ANALİZ
    # ================================================================ #
    def _s10(self):
        R = self.R; R.sec("10. KATEGORİK ANALİZ")
        if not self.cats:
            R.add("  ℹ️ Cat yok"); return

        R.add(f"  {'Feature':<25} {'Uniq':>6} {'Top':>15} {'Top%':>7} {'Card':>6}")
        R.add(f"  {'-'*64}")
        for c in self.cats:
            nu = self.train[c].nunique()
            tv = self.train[c].value_counts().index[0]
            tp = self.train[c].value_counts(normalize=True).iloc[0]
            cd = "HIGH" if nu > self.cfg.HIGH_CARD_THRESHOLD else "ok"
            R.add(f"  {c:<25} {nu:>6} {str(tv)[:15]:>15} {tp:>6.1%} {cd:>6}")

        for c in self.cats[:self.cfg.TOP_N_CAT]:
            R.sub(f"📋 {c}")
            vc = self.train[c].value_counts().head(8)
            for v, cnt in vc.items():
                r = cnt/len(self.train)
                R.add(f"    {str(v):<20} {cnt:>8,} ({r:>5.1%}) {'█'*int(r*40)}")
            if self.ptype != 'regression':
                ct = pd.crosstab(self.train[c], self.train[self.target], normalize='index')
                R.add(f"    Target:")
                for v in ct.index[:5]:
                    row = ct.loc[v]
                    R.add(f"      {str(v):<18} → {row.idxmax()} ({row.max():.1%})")

    # ================================================================ #
    # S11 – NUMERİK ANALİZ
    # ================================================================ #
    def _s11(self):
        R = self.R; R.sec("11. NUMERİK ANALİZ")
        if not self.nums:
            R.add("  ℹ️ Num yok"); return
        R.add(f"  {'Feat':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Skew':>7} {'Out%':>6}")
        R.add(f"  {'-'*78}")
        for c in self.nums:
            s = self.train[c]
            Q1,Q3 = s.quantile(.25), s.quantile(.75)
            IQR = Q3-Q1
            op = ((s<Q1-1.5*IQR)|(s>Q3+1.5*IQR)).mean()*100
            R.add(f"  {c:<20} {s.mean():>10.2f} {s.std():>10.2f} "
                   f"{s.min():>10.2f} {s.max():>10.2f} {s.skew():>7.2f} {op:>5.1f}%")

        for c in self.nums[:self.cfg.TOP_N]:
            R.sub(f"🔢 {c}")
            d = self.train[c].describe()
            R.add(f"    Count={d['count']:,.0f}  Mean={d['mean']:.4f}  Std={d['std']:.4f}")
            R.add(f"    Min={d['min']:.4f} Q1={d['25%']:.4f} Med={d['50%']:.4f} "
                   f"Q3={d['75%']:.4f} Max={d['max']:.4f}")
            sk = self.train[c].skew()
            R.add(f"    Skew={sk:.3f}  Kurt={self.train[c].kurtosis():.3f}")
            R.add(f"    Zero={(self.train[c]==0).sum():,}  Neg={(self.train[c]<0).sum():,}")
            if abs(sk) > 2:
                R.add(f"    🚨 Highly skewed → transform")
            try:
                bins = pd.cut(self.train[c], bins=self.cfg.BIN_COUNT)
                bc = bins.value_counts().sort_index(); mx=bc.max()
                R.add(f"    Dist:")
                for iv,cnt in bc.items():
                    R.add(f"      {str(iv):<30} {cnt:>7,} {'█'*int(cnt/mx*25 if mx else 0)}")
            except Exception:
                pass

    # ================================================================ #
    # S12 – SINIF BAZLI İSTATİSTİK
    # ================================================================ #
    def _s12(self):
        R = self.R; R.sec("12. SINIF BAZLI İSTATİSTİKLER")
        if self.ptype == 'regression' or not self.nums:
            R.add("  ℹ️ Atlanıyor"); return
        classes = sorted(self.train[self.target].unique())
        if len(classes) > 10:
            classes = classes[:10]; R.add(f"  ℹ️ İlk 10 sınıf")
        for c in self.nums[:10]:
            R.sub(f"📊 {c}")
            hdr = f"    {'Class':<15} {'Mean':>10} {'Std':>10} {'Med':>10}"
            R.add(hdr); R.add(f"    {'-'*50}")
            for cl in classes:
                ss = self.train.loc[self.train[self.target]==cl, c]
                R.add(f"    {str(cl):<15} {ss.mean():>10.3f} {ss.std():>10.3f} {ss.median():>10.3f}")

    # ================================================================ #
    # S13 – BINNED TARGET RATE
    # ================================================================ #
    def _s13(self):
        R = self.R; R.sec("13. BINNED TARGET RATE")
        if self.ptype == 'regression' or not self.nums:
            R.add("  ℹ️ Atlanıyor"); return
        te = self._enc_target()
        for c in self.nums[:10]:
            R.sub(f"📈 {c}")
            try:
                bins = pd.qcut(self.train[c], q=self.cfg.BIN_COUNT, duplicates='drop')
                tmp = pd.DataFrame({'b': bins, 't': te})
                g = tmp.groupby('b', observed=True)['t'].agg(['mean','count'])
                R.add(f"    {'Bin':<30} {'N':>8} {'TgtMean':>10}")
                R.add(f"    {'-'*52}")
                for i,row in g.iterrows():
                    R.add(f"    {str(i):<30} {int(row['count']):>8} {row['mean']:>10.4f}")
                means = g['mean'].values
                d = np.diff(means)
                if len(d) > 0 and (np.all(d>=0) or np.all(d<=0)):
                    R.add(f"    ✅ MONOTONİK")
                elif len(d) > 0:
                    R.add(f"    ℹ️ Non-monotonic ({np.sum(np.diff(np.sign(d))!=0)} yön değ.)")
            except Exception as e:
                R.add(f"    ⚠️ {e}")

    # ================================================================ #
    # S14 – İSTATİSTİKSEL TESTLER
    # ================================================================ #
    def _s14(self):
        R = self.R; R.sec("14. İSTATİSTİKSEL TESTLER")
        te = self._enc_target()
        classes = sorted(self.train[self.target].unique())

        if self.ptype == 'regression':
            R.add(f"  {'Feat':<25} {'Pearson':>9} {'p':>10} {'Spearman':>9} {'p':>10}")
            R.add(f"  {'-'*68}")
            for c in self.nums[:self.cfg.TOP_N]:
                try:
                    v = self.train[c].fillna(self.train[c].median())
                    pr,pp = stats.pearsonr(v, self.train[self.target])
                    sr,sp = stats.spearmanr(v, self.train[self.target])
                    R.add(f"  {c:<25} {pr:>9.4f} {pp:>10.2e} {sr:>9.4f} {sp:>10.2e}")
                except Exception: pass
            return

        # Kruskal-Wallis
        R.add(f"  NUMERIK → Kruskal-Wallis:\n")
        R.add(f"  {'Feat':<25} {'H':>10} {'p':>12} {'Sig':>8}")
        R.add(f"  {'-'*60}")
        kw = []
        for c in self.nums:
            try:
                grps = [self.train.loc[self.train[self.target]==cl, c].dropna()
                        for cl in classes]
                grps = [g for g in grps if len(g)>0]
                if len(grps)>=2:
                    h,p = stats.kruskal(*grps)
                    sig = "***" if p<.001 else ("**" if p<.01 else ("*" if p<.05 else "ns"))
                    kw.append((c,h,p,sig))
            except Exception: pass
        kw.sort(key=lambda x:x[1], reverse=True)
        for c,h,p,sig in kw[:self.cfg.TOP_N]:
            R.add(f"  {c:<25} {h:>10.2f} {p:>12.2e} {sig:>8}")

        # Chi-square
        if self.cats:
            R.add(f"\n  KATEGORİK → Chi-Square:\n")
            R.add(f"  {'Feat':<25} {'Chi2':>12} {'p':>12} {'Cramér':>8} {'Sig':>6}")
            R.add(f"  {'-'*68}")
            for c in self.cats:
                try:
                    ct = pd.crosstab(self.train[c], self.train[self.target])
                    chi2,p,_,_ = stats.chi2_contingency(ct)
                    cv = self._cramers_v(self.train[c], self.train[self.target])
                    sig = "***" if p<.001 else ("**" if p<.01 else ("*" if p<.05 else "ns"))
                    R.add(f"  {c:<25} {chi2:>12.1f} {p:>12.2e} {cv:>8.4f} {sig:>6}")
                except Exception: pass

    # ================================================================ #
    # S15 – MUTUAL INFORMATION
    # ================================================================ #
    def _s15(self):
        R = self.R; R.sec("15. MUTUAL INFORMATION")
        te = self._enc_target()
        n = len(self.train); ms = self.cfg.MI_SAMPLE
        idx = np.random.choice(n, min(n,ms), replace=False) if n > ms else np.arange(n)

        X = self.train[self.feats].iloc[idx].copy()
        for c in self.cats:
            if X[c].dtype == 'object':
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))
        X = X.fillna(-999)
        y = te[idx]

        try:
            mi = (mutual_info_regression(X, y, random_state=42, n_neighbors=5)
                  if self.ptype == 'regression'
                  else mutual_info_classif(X, y, random_state=42, n_neighbors=5))
            mi_s = pd.Series(mi, index=self.feats).sort_values(ascending=False)
            R.add(f"  {'Rank':<5} {'Feat':<25} {'MI':>10}")
            R.add(f"  {'-'*44}")
            for i,(f,s) in enumerate(mi_s.head(self.cfg.TOP_N).items(),1):
                R.add(f"  {i:<5} {f:<25} {s:>10.4f}")
            R.add(f"\n  📉 Bottom 5:")
            for f,s in mi_s.tail(5).items():
                R.add(f"    {f:<25} {s:.4f}")
        except Exception as e:
            R.add(f"  ⚠️ {e}")

    # ================================================================ #
    # S16 – KORELASYON
    # ================================================================ #
    def _s16(self):
        R = self.R; R.sec("16. KORELASYON (Target ile)")
        if len(self.nums) < 2:
            R.add("  ⚠️ Yeterli num yok"); return
        te = self._enc_target()
        cdf = self.train[self.nums].copy(); cdf['__T__'] = te
        cm = cdf.corr(); tc = cm['__T__'].drop('__T__')

        R.add(f"  {'Feat':<25} {'Pearson':>10} {'|r|':>8} {'Strength':>12}")
        R.add(f"  {'-'*60}")
        for f in tc.abs().sort_values(ascending=False).head(self.cfg.TOP_N).index:
            ac = abs(tc[f])
            st = ("V.STRONG" if ac>.7 else "STRONG" if ac>.5
                  else "MODERATE" if ac>.3 else "WEAK" if ac>.1 else "NONE")
            R.add(f"  {f:<25} {tc[f]:>10.4f} {ac:>8.4f} {st:>12}")

    # ================================================================ #
    # S17 – MULTICOLLINEARITY
    # ================================================================ #
    def _s17(self):
        R = self.R; R.sec("17. MULTICOLLINEARITY")
        if len(self.nums) < 2:
            R.add("  ⚠️ Yeterli num yok"); return
        cm = self.train[self.nums].corr()
        pairs = [(self.nums[i], self.nums[j], abs(cm.iloc[i,j]))
                 for i in range(len(self.nums))
                 for j in range(i+1, len(self.nums))
                 if abs(cm.iloc[i,j]) > 0.8]
        if pairs:
            R.add(f"  🚨 |r|>0.8 çiftler:")
            for c1,c2,c in sorted(pairs, key=lambda x:x[2], reverse=True)[:15]:
                R.add(f"    {c1:<25} ↔ {c2:<25} {c:.4f}")
        else:
            R.add("  ✅ Yüksek multicollinearity yok")

    # ================================================================ #
    # S18 – FEATURE IMPORTANCE (Heuristic)
    # ================================================================ #
    def _s18(self):
        R = self.R; R.sec("18. HEURİSTİK IMPORTANCE (Combined)")
        te = self._enc_target(); scores = []
        for c in self.nums:
            v = self.train[c].fillna(self.train[c].median()).values
            scores.append((c, abs(self._corr(v, te)), 'num', 'pearson'))
        for c in self.cats:
            cv = self._cramers_v(self.train[c], self.train[self.target])
            scores.append((c, min(cv,1.0), 'cat', 'cramers_v'))
        scores.sort(key=lambda x:x[1], reverse=True)
        R.add(f"  {'#':<4} {'Feat':<25} {'Score':>8} {'Type':>6} {'Method':>10}")
        R.add(f"  {'-'*58}")
        for i,(f,s,t,m) in enumerate(scores[:self.cfg.TOP_N],1):
            R.add(f"  {i:<4} {f:<25} {s:>8.4f} {t:>6} {m:>10}")

    # ================================================================ #
    # S19 – FEATURE INTERACTION
    # ================================================================ #
    def _s19(self):
        R = self.R; R.sec("19. FEATURE INTERACTION SİNYALLERİ")
        if len(self.nums) < 2:
            R.add("  ⚠️ Num < 2"); return
        te = self._enc_target()
        ic = {c: abs(self._corr(self.train[c].fillna(0).values, te)) for c in self.nums}
        top = sorted(ic, key=ic.get, reverse=True)[:self.cfg.INTERACT_TOP_K]

        R.add(f"  Top-{len(top)} interaction (×) taraması:\n")
        R.add(f"  {'Pair':<45} {'Ind':>8} {'Inter':>8} {'Gain':>8}")
        R.add(f"  {'-'*74}")
        inters = []
        for c1,c2 in combinations(top,2):
            v1 = self.train[c1].fillna(0).values
            v2 = self.train[c2].fillna(0).values
            ic_ = abs(self._corr(v1*v2, te))
            im = max(ic[c1], ic[c2])
            inters.append((f"{c1} × {c2}", im, ic_, ic_-im))
        inters.sort(key=lambda x:x[3], reverse=True)
        for nm,im,ic_,g in inters[:10]:
            flag = " ✨" if g > 0.02 else ""
            R.add(f"  {nm:<45} {im:>8.4f} {ic_:>8.4f} {g:>+7.4f}{flag}")

        # Oran
        R.add(f"\n  Oran (÷) taraması:")
        ratios = []
        for c1,c2 in combinations(top,2):
            v1 = self.train[c1].fillna(0).values.astype(float)
            v2 = self.train[c2].fillna(0).values.astype(float)
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.where(v2!=0, v1/v2, 0)
            r = np.nan_to_num(r, 0, 0, 0)
            rc = abs(self._corr(r, te))
            im = max(ic[c1], ic[c2])
            ratios.append((f"{c1}/{c2}", rc, rc-im))
        ratios.sort(key=lambda x:x[2], reverse=True)
        for nm,rc,g in ratios[:5]:
            R.add(f"    {nm:<40} corr={rc:.4f} gain={g:+.4f}")

    # ================================================================ #
    # S20 – ROW-WISE STATS
    # ================================================================ #
    def _s20(self):
        R = self.R; R.sec("20. ROW-WISE META FEATURES")
        if len(self.nums) < 3:
            R.add("  ℹ️ Num < 3"); return
        te = self._enc_target()
        nd = self.train[self.nums].fillna(0)
        meta = pd.DataFrame({
            'row_mean': nd.mean(1), 'row_std': nd.std(1),
            'row_min': nd.min(1), 'row_max': nd.max(1),
            'row_sum': nd.sum(1), 'row_range': nd.max(1)-nd.min(1),
            'row_zeros': (nd==0).sum(1),
            'row_nulls': self.train[self.nums].isnull().sum(1),
        })
        R.add(f"  {'Meta':<20} {'Corr':>10} {'|Corr|':>8} {'Use':>6}")
        R.add(f"  {'-'*48}")
        for c in meta.columns:
            r = self._corr(meta[c].values, te)
            u = "✅" if abs(r)>0.1 else "❌"
            R.add(f"  {c:<20} {r:>10.4f} {abs(r):>8.4f} {u:>6}")

    # ================================================================ #
    # S21 – OUTLIER IMPACT
    # ================================================================ #
    def _s21(self):
        R = self.R; R.sec("21. OUTLIER ETKİSİ (Target Üzerine)")
        if not self.nums:
            R.add("  ℹ️ Num yok"); return
        te = self._enc_target()
        R.add(f"  {'Feat':<25} {'Out%':>7} {'Tgt(norm)':>11} {'Tgt(out)':>11} {'Diff':>8}")
        R.add(f"  {'-'*66}")
        for c in self.nums[:self.cfg.TOP_N]:
            s = self.train[c]
            Q1,Q3 = s.quantile(.25), s.quantile(.75)
            IQR = Q3-Q1
            m = (s<Q1-1.5*IQR)|(s>Q3+1.5*IQR)
            if m.sum()<10 or (~m).sum()<10: continue
            tn = te[~m].mean(); to = te[m].mean(); d = to-tn
            flag = " ⚠️" if abs(d)>0.1*abs(tn+1e-9) else ""
            R.add(f"  {c:<25} {m.mean():>6.1%} {tn:>11.4f} {to:>11.4f} {d:>+7.4f}{flag}")

    # ================================================================ #
    # S22 – INDEX / ORDER TREND
    # ================================================================ #
    def _s22(self):
        R = self.R; R.sec("22. INDEX / SIRALAMA TRENDİ")
        te = self._enc_target().astype(float)
        idx = np.arange(len(self.train)).astype(float)
        tc = self._corr(idx, te)
        R.add(f"  Target ↔ Index corr: {tc:.4f}")
        if abs(tc) > 0.05:
            R.add(f"  ⚠️ Sıralama trendi → temporal leak?  TimeSeriesSplit düşün")

        n = len(self.train)
        q1m = te[:n//4].mean(); q4m = te[3*n//4:].mean()
        R.add(f"  İlk çeyrek mean={q1m:.4f}  Son çeyrek mean={q4m:.4f}")

        ft = []
        for c in self.nums:
            v = self.train[c].fillna(0).values
            r = abs(self._corr(idx, v))
            if r > 0.05: ft.append((c,r))
        if ft:
            R.add(f"\n  ⚠️ Feature'larda sıralama trendi:")
            for c,r in sorted(ft, key=lambda x:x[1], reverse=True)[:10]:
                R.add(f"    {c:<25} |corr|={r:.4f}")

    # ================================================================ #
    # S23 – DRIFT
    # ================================================================ #
    def _s23(self):
        R = self.R; R.sec("23. TRAIN-TEST DRIFT")
        # Kategorik
        cd = []
        for c in self.cats:
            if c not in self.test.columns: continue
            tr = self.train[c].value_counts(normalize=True)
            te = self.test[c].value_counts(normalize=True)
            md = max(abs(tr.get(v,0)-te.get(v,0)) for v in set(tr.index)|set(te.index))
            if md > self.cfg.DRIFT_THRESHOLD:
                cd.append((c,md))
        if cd:
            R.add(f"  🚨 Cat drift ({len(cd)}):")
            for c,d in sorted(cd, key=lambda x:x[1], reverse=True):
                R.add(f"    {c:<25} {d:.1%}")
        else:
            R.add("  ✅ Cat drift yok")

        # Numerik KS
        nd = []
        for c in self.nums:
            if c not in self.test.columns: continue
            try:
                ks,p = stats.ks_2samp(self.train[c].dropna().values,
                                       self.test[c].dropna().values)
                if ks > 0.05 or p < 0.01:
                    nd.append((c,ks,p))
            except Exception: pass
        if nd:
            R.add(f"\n  🚨 Num drift ({len(nd)}):")
            R.add(f"  {'Feat':<25} {'KS':>8} {'p':>12}")
            R.add(f"  {'-'*48}")
            for c,ks,p in sorted(nd, key=lambda x:x[1], reverse=True)[:15]:
                R.add(f"  {c:<25} {ks:>8.4f} {p:>12.2e}")
        else:
            R.add(f"\n  ✅ Num drift yok")

    # ================================================================ #
    # S24 – OVERLAP
    # ================================================================ #
    def _s24(self):
        R = self.R; R.sec("24. TRAIN-TEST OVERLAP")
        cf = [c for c in self.feats if c in self.test.columns]
        if not cf: R.add("  ⚠️ Ortak feat yok"); return
        try:
            mg = pd.merge(self.train[cf], self.test[cf], on=cf, how='inner')
            oc = len(mg)
            R.add(f"  Eşleşme: {oc:,}  (train {oc/len(self.train)*100:.2f}%  "
                   f"test {oc/len(self.test)*100:.2f}%)")
            if oc > 0: R.add("  ⚠️ Overlap → pseudo-labeling fırsatı")
            else:       R.add("  ✅ Overlap yok")
        except Exception as e:
            R.add(f"  ⚠️ Memory? {e}")

    # ================================================================ #
    # S25 – ADVERSARIAL PROXY
    # ================================================================ #
    def _s25(self):
        R = self.R; R.sec("25. ADVERSARIAL VALIDATION PROXY")
        cf = [c for c in self.nums if c in self.test.columns]
        if not cf: R.add("  ⚠️ Num ortak yok"); return
        kss = []
        for c in cf:
            try:
                ks,p = stats.ks_2samp(self.train[c].dropna().values,
                                       self.test[c].dropna().values)
                kss.append((c,ks,p))
            except Exception: pass
        if not kss: return
        avg = np.mean([x[1] for x in kss])
        R.add(f"  Ort. KS: {avg:.4f}")
        if avg < 0.02:   R.add("  ✅ Çok benzer (AUC≈0.50)")
        elif avg < 0.05: R.add("  ✅ Benzer (AUC≈0.50-0.55)")
        elif avg < 0.10: R.add("  ⚠️ Hafif drift (AUC≈0.55-0.65)")
        else:            R.add("  🚨 Ciddi drift (AUC>0.65)")
        R.add(f"\n  Top KS features:")
        for c,ks,p in sorted(kss, key=lambda x:x[1], reverse=True)[:10]:
            R.add(f"    {c:<25} KS={ks:.4f} p={p:.2e}")

    # ================================================================ #
    # S26 – LEAKAGE
    # ================================================================ #
    def _s26(self):
        R = self.R; R.sec("26. DATA LEAKAGE TESPİTİ")
        susp = []
        for c in self.nums:
            if self.train[c].nunique() == len(self.train):
                susp.append((c, "UNIQUE/ROW → ID?"))
        te = self._enc_target()
        for c in self.nums:
            r = abs(self._corr(self.train[c].fillna(0).values, te))
            if r > 0.95: susp.append((c, f"Target corr={r:.4f} 🚨"))
        for c in self.feats:
            if c not in self.test.columns:
                susp.append((c, "Train'de var TEST'te YOK"))
        if susp:
            R.add(f"  🚨 POTANSİYEL ({len(susp)}):")
            for it,rs in susp: R.add(f"    {it}: {rs}")
        else:
            R.add("  ✅ Yok")

    # ================================================================ #
    # S27 – DATE/TIME DETECTION   ← YENİ
    # ================================================================ #
    def _s27(self):
        R = self.R; R.sec("27. DATE / TIME FEATURE TESPİTİ")
        date_cols = []
        for c in self.feats:
            if self.train[c].dtype == 'object':
                sample = self.train[c].dropna().head(20)
                try:
                    parsed = pd.to_datetime(sample, infer_datetime_format=True)
                    if parsed.notna().mean() > 0.8:
                        date_cols.append(c)
                except Exception:
                    pass
            # Kolon ismine göre ipucu
            low = c.lower()
            if any(k in low for k in ['date','time','timestamp','dt','year','month',
                                       'day','hour','minute','second']):
                if c not in date_cols:
                    date_cols.append(c)

        if date_cols:
            R.add(f"  🕐 Potansiyel date/time kolonları ({len(date_cols)}):")
            for c in date_cols:
                R.add(f"    {c}  dtype={self.train[c].dtype}  "
                       f"sample={self.train[c].dropna().iloc[0] if len(self.train[c].dropna())>0 else 'N/A'}")
            R.add(f"\n  💡 FE fikirleri:")
            R.add(f"    • year, month, day, dayofweek, hour çıkar")
            R.add(f"    • is_weekend, is_holiday flag")
            R.add(f"    • Epoch seconds, cyclical encoding (sin/cos)")
            R.add(f"    • İki tarih arası fark (gün/saat)")
        else:
            R.add("  ✅ Date/time kolonu tespit edilmedi")

    # ================================================================ #
    # S28 – DISTRIBUTION SHAPE   ← YENİ
    # ================================================================ #
    def _s28(self):
        R = self.R; R.sec("28. DAĞILIM ŞEKLİ TESPİTİ")
        if not self.nums:
            R.add("  ℹ️ Num yok"); return

        R.add(f"  {'Feat':<25} {'Shape':>14} {'Skew':>7} {'Kurt':>7} {'Bimodal?':>10} {'Normal?':>10}")
        R.add(f"  {'-'*78}")

        for c in self.nums[:self.cfg.TOP_N]:
            s = self.train[c].dropna()
            if len(s) < 30: continue

            sk = s.skew()
            ku = s.kurtosis()

            # Bimodal detection: Hartigan's dip test proxy
            # Basit yöntem: histogram'da 2 tepe var mı?
            try:
                counts, edges = np.histogram(s, bins=30)
                # Smooth et
                from numpy import convolve
                kernel = np.ones(3)/3
                smooth = convolve(counts, kernel, mode='same')
                # Yerel maksimumları bul
                peaks = 0
                for i in range(1, len(smooth)-1):
                    if smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1]:
                        if smooth[i] > 0.05 * smooth.max():  # %5 eşik
                            peaks += 1
                bimodal = "YES" if peaks >= 2 else "no"
            except Exception:
                bimodal = "?"

            # Normallik testi (D'Agostino)
            try:
                if len(s) > 8:
                    _, p_norm = stats.normaltest(s.sample(min(5000, len(s))))
                    normal = "YES" if p_norm > 0.05 else "no"
                else:
                    normal = "?"
            except Exception:
                normal = "?"

            # Shape label
            if abs(sk) < 0.5 and abs(ku) < 1:
                shape = "SYMMETRIC"
            elif sk > 2:
                shape = "RIGHT-SKEW"
            elif sk < -2:
                shape = "LEFT-SKEW"
            elif ku > 5:
                shape = "HEAVY-TAIL"
            elif bimodal == "YES":
                shape = "BIMODAL"
            elif abs(sk) < 1:
                shape = "NEAR-NORMAL"
            else:
                shape = "SKEWED"

            R.add(f"  {c:<25} {shape:>14} {sk:>7.2f} {ku:>7.2f} {bimodal:>10} {normal:>10}")

        R.add(f"\n  💡 Aksiyon:")
        R.add(f"    BIMODAL       → 2 küme olabilir, clustering veya binning dene")
        R.add(f"    RIGHT-SKEW    → log1p transform")
        R.add(f"    LEFT-SKEW     → square/exp transform")
        R.add(f"    HEAVY-TAIL    → clip veya winsorize")

    # ================================================================ #
    # S29 – CAT×CAT CRAMÉR'S V   ← YENİ
    # ================================================================ #
    def _s29(self):
        R = self.R; R.sec("29. KATEGORİK × KATEGORİK KORELASYON (Cramér's V)")
        if len(self.cats) < 2:
            R.add("  ℹ️ Cat < 2"); return

        R.add(f"  {'Pair':<45} {'Cramér V':>10}")
        R.add(f"  {'-'*58}")
        pairs = []
        for c1,c2 in combinations(self.cats, 2):
            cv = self._cramers_v(self.train[c1], self.train[c2])
            pairs.append((c1,c2,cv))
        pairs.sort(key=lambda x:x[2], reverse=True)

        for c1,c2,cv in pairs[:20]:
            flag = " ⚠️ REDUNDANT" if cv > 0.8 else (" 🔗" if cv > 0.5 else "")
            R.add(f"  {c1+' ↔ '+c2:<45} {cv:>10.4f}{flag}")

        high = [p for p in pairs if p[2] > 0.8]
        if high:
            R.add(f"\n  🚨 {len(high)} çift yüksek korelasyonlu → birini drop et")

    # ================================================================ #
    # S30 – FEATURE ENTROPY / INFO CONTENT   ← YENİ
    # ================================================================ #
    def _s30(self):
        R = self.R; R.sec("30. FEATURE ENTROPY / BİLGİ İÇERİĞİ")

        R.add(f"  {'Feat':<25} {'Unique':>8} {'UniqueRatio':>12} {'Entropy':>10} {'Info':>8}")
        R.add(f"  {'-'*68}")

        for c in self.feats:
            nu = self.train[c].nunique()
            ur = nu / len(self.train)
            ent = self._entropy(self.train[c])
            # Bilgi seviyesi
            if nu <= 1:
                info = "ZERO"
            elif ur > 0.95:
                info = "ID-like"
            elif ent < 0.5:
                info = "LOW"
            elif ent > 4:
                info = "HIGH"
            else:
                info = "MED"
            R.add(f"  {c:<25} {nu:>8,} {ur:>11.4f} {ent:>10.3f} {info:>8}")

    # ================================================================ #
    # S31 – ORDINAL vs NOMINAL DETECTION   ← YENİ
    # ================================================================ #
    def _s31(self):
        R = self.R; R.sec("31. ORDINAL vs NOMINAL TESPİTİ")
        if not self.cats:
            R.add("  ℹ️ Cat yok"); return

        R.add(f"  {'Feat':<25} {'Type':>10} {'Reason':>35}")
        R.add(f"  {'-'*74}")

        ordinal_keywords = ['low','medium','high','small','large','poor','fair',
                            'good','excellent','never','rarely','sometimes',
                            'often','always','none','few','many','very',
                            'strongly','agree','disagree','1st','2nd','3rd']

        for c in self.cats:
            vals = [str(v).lower() for v in self.train[c].dropna().unique()]
            # Sayısal string kontrolü
            try:
                numeric_vals = [float(v) for v in vals]
                if len(numeric_vals) == len(vals):
                    R.add(f"  {c:<25} {'ORDINAL':>10} {'Sayısal string değerler':>35}")
                    continue
            except Exception:
                pass

            # Keyword eşleşmesi
            matches = [v for v in vals if any(k in v for k in ordinal_keywords)]
            if len(matches) >= 2:
                R.add(f"  {c:<25} {'ORDINAL?':>10} {'Keyword: '+', '.join(matches[:3]):>35}")
            else:
                R.add(f"  {c:<25} {'NOMINAL':>10} {'':>35}")

        R.add(f"\n  💡 Ordinal → OrdinalEncoder (sıra koruyarak)")
        R.add(f"     Nominal → OneHotEncoder veya TargetEncoder")

    # ================================================================ #
    # S32 – NON-LINEAR DETECTION (Spearman vs Pearson)   ← YENİ
    # ================================================================ #
    def _s32(self):
        R = self.R; R.sec("32. NON-LINEAR İLİŞKİ TESPİTİ (Spearman-Pearson Gap)")
        if not self.nums:
            R.add("  ℹ️ Num yok"); return

        te = self._enc_target()
        R.add(f"  {'Feat':<25} {'Pearson':>9} {'Spearman':>9} {'Gap':>8} {'Non-Lin?':>10}")
        R.add(f"  {'-'*66}")

        for c in self.nums[:self.cfg.TOP_N]:
            v = self.train[c].fillna(self.train[c].median()).values
            pr = self._corr(v, te)
            try:
                sr, _ = stats.spearmanr(v, te)
            except Exception:
                sr = 0.0
            gap = abs(sr) - abs(pr)
            nl = "YES ⚠️" if gap > 0.1 else ("maybe" if gap > 0.05 else "no")
            R.add(f"  {c:<25} {pr:>9.4f} {sr:>9.4f} {gap:>+7.4f} {nl:>10}")

        R.add(f"\n  💡 Gap > 0.1 → feature'da non-linear ilişki var")
        R.add(f"     → Polynomial features, binning, veya tree-based model kullan")

    # ================================================================ #
    # S33 – LABEL NOISE ESTIMATION   ← YENİ
    # ================================================================ #
    def _s33(self):
        R = self.R; R.sec("33. LABEL NOISE TAHMİNİ")

        if self.ptype == 'regression':
            R.add("  ℹ️ Regression → duplicate-based noise check")
            df = self.train[self.feats].duplicated(keep=False).sum()
            R.add(f"  Feature-duplicate satır: {df:,}")
            if df > 0:
                mask = self.train[self.feats].duplicated(keep=False)
                grp = self.train[mask].groupby(self.feats, dropna=False)[self.target]
                try:
                    noise_std = grp.std().mean()
                    R.add(f"  Aynı X'ler arası target std: {noise_std:.4f}")
                    overall_std = self.train[self.target].std()
                    ratio = noise_std / overall_std if overall_std > 0 else 0
                    R.add(f"  Noise/Signal oranı: {ratio:.2%}")
                    if ratio > 0.5:
                        R.add(f"  🚨 Yüksek label noise!")
                except Exception:
                    pass
            return

        # Classification
        conflicts = self.quality.get('label_conflicts', 0)
        dups = self.quality.get('duplicates', 0)

        if dups == 0:
            R.add("  ✅ Duplicate yok → noise tahmini yapılamıyor (iyi olabilir)")
            return

        noise_rate = conflicts / max(dups, 1)
        R.add(f"  Feature duplicates : {dups:,}")
        R.add(f"  Çelişkili gruplar  : {conflicts:,}")
        R.add(f"  Tahmini noise rate : {noise_rate:.2%}")

        if noise_rate > 0.20:
            R.add(f"  🚨 Yüksek noise → label smoothing, confident learning")
        elif noise_rate > 0.05:
            R.add(f"  ⚠️ Orta noise → dikkatli ol, cleaning dene")
        else:
            R.add(f"  ✅ Düşük noise")

    # ================================================================ #
    # S34 – SYNTHETIC DATA CHECK (PS-specific)   ← YENİ
    # ================================================================ #
    def _s34(self):
        R = self.R; R.sec("34. SENTETİK VERİ KONTROLÜ (PS-Specific)")

        indicators = 0
        reasons = []

        # 1. Çok yuvarlak sayılar
        round_cols = 0
        for c in self.nums[:20]:
            s = self.train[c].dropna()
            if len(s) == 0: continue
            pct_integer = (s == s.astype(int)).mean()
            n_decimals = s.apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
            if pct_integer > 0.95 or n_decimals.mean() < 2:
                round_cols += 1
        if round_cols > len(self.nums) * 0.5:
            indicators += 1
            reasons.append(f"Çoğu numerik kolon çok yuvarlak ({round_cols}/{len(self.nums)})")

        # 2. Mükemmel dağılım (çok uniform)
        uniform_cols = 0
        for c in self.nums[:20]:
            try:
                _, p = stats.kstest(
                    stats.zscore(self.train[c].dropna().sample(min(1000, len(self.train[c].dropna())))),
                    'norm')
                if p > 0.5:
                    uniform_cols += 1
            except Exception:
                pass
        if uniform_cols > len(self.nums) * 0.3:
            indicators += 1
            reasons.append(f"Birçok kolon normal dağılıma çok uyuyor ({uniform_cols})")

        # 3. Train size belirli eşiklerde (PS genelde 100K-200K)
        n = len(self.train)
        if n in range(90000, 210000):
            indicators += 1
            reasons.append(f"Train boyutu PS tipik aralığında ({n:,})")

        # 4. Path'te playground-series var mı?
        if 'playground-series' in self.cfg.BASE_PATH.lower():
            indicators += 2
            reasons.append("Path'te 'playground-series' var")

        R.add(f"  Sentetik veri göstergeleri: {indicators}/5")
        for r in reasons:
            R.add(f"    • {r}")

        if indicators >= 2:
            R.add(f"\n  ⚠️ Büyük olasılıkla SENTETIK (Playground Series)")
            R.add(f"  💡 Aksiyon:")
            R.add(f"    1. Orijinal dataset bul ve train'e ekle")
            R.add(f"    2. Orijinal veri ile blend/stack")
            R.add(f"    3. Sentetik veri boundary'lerinde dikkatli ol")
        else:
            R.add(f"\n  ℹ️ Gerçek veri gibi görünüyor")

    # ================================================================ #
    # S35 – PERCENTILE-LEVEL DRIFT   ← YENİ
    # ================================================================ #
    def _s35(self):
        R = self.R; R.sec("35. PERCENTILE-LEVEL DRIFT")
        if not self.nums:
            R.add("  ℹ️ Num yok"); return

        pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]

        for c in self.nums[:10]:
            if c not in self.test.columns: continue
            R.sub(f"📊 {c}")
            R.add(f"    {'Pct':>6} {'Train':>12} {'Test':>12} {'Diff%':>10}")
            R.add(f"    {'-'*44}")
            any_drift = False
            for p in pcts:
                tv = self.train[c].quantile(p/100)
                tev = self.test[c].quantile(p/100)
                diff = abs(tv-tev) / (abs(tv)+1e-9) * 100
                flag = " ⚠️" if diff > 10 else ""
                if diff > 10: any_drift = True
                R.add(f"    {p:>5}% {tv:>12.4f} {tev:>12.4f} {diff:>9.1f}%{flag}")
            if not any_drift:
                R.add(f"    ✅ Tüm percentile'lar yakın")

    # ================================================================ #
    # S36 – FEATURE CLUSTERING   ← YENİ
    # ================================================================ #
    def _s36(self):
        R = self.R; R.sec("36. FEATURE CLUSTERING (Korelasyon Grupları)")
        if len(self.nums) < 3:
            R.add("  ℹ️ Num < 3"); return

        try:
            cm = self.train[self.nums].corr().abs()
            cm = cm.fillna(0)
            # Distance = 1 - correlation
            dist = 1 - cm.values
            np.fill_diagonal(dist, 0)
            dist = np.clip(dist, 0, 2)  # negatif olmasın
            # Hierarchical clustering
            condensed = squareform(dist, checks=False)
            Z = linkage(condensed, method='complete')
            clusters = fcluster(Z, t=1-self.cfg.CLUSTER_THRESH, criterion='distance')

            cluster_map = {}
            for feat, cl in zip(self.nums, clusters):
                cluster_map.setdefault(cl, []).append(feat)

            multi_clusters = {k:v for k,v in cluster_map.items() if len(v)>=2}

            if multi_clusters:
                R.add(f"  Eşik: corr > {self.cfg.CLUSTER_THRESH}")
                R.add(f"  Toplam {len(cluster_map)} küme, {len(multi_clusters)} çoklu:\n")
                for cl_id, feats in sorted(multi_clusters.items(), key=lambda x:len(x[1]), reverse=True):
                    R.add(f"    Küme {cl_id} ({len(feats)} feature):")
                    for f in feats:
                        R.add(f"      • {f}")
                R.add(f"\n  💡 Her kümeden 1 feature seç veya PCA uygula")
            else:
                R.add("  ✅ Belirgin küme yok (feature'lar bağımsız)")
        except Exception as e:
            R.add(f"  ⚠️ Clustering hatası: {e}")

    # ================================================================ #
    # S37 – TARGET TRANSFORM ANALİZİ   ← YENİ
    # ================================================================ #
    def _s37(self):
        R = self.R; R.sec("37. TARGET TRANSFORM ANALİZİ")
        if self.ptype != 'regression':
            R.add("  ℹ️ Classification → atlanıyor"); return

        t = self.train[self.target].dropna()
        R.add(f"  Original target:")
        R.add(f"    Skew={t.skew():.3f}  Kurt={t.kurtosis():.3f}")
        R.add(f"    Min={t.min():.4f}  Max={t.max():.4f}")

        transforms = {}

        # log1p (sadece pozitif)
        if (t > 0).all():
            lt = np.log1p(t)
            transforms['log1p'] = (lt.skew(), lt.kurtosis())
        elif (t >= 0).all():
            lt = np.log1p(t)
            transforms['log1p'] = (lt.skew(), lt.kurtosis())

        # sqrt (sadece non-negative)
        if (t >= 0).all():
            st = np.sqrt(t)
            transforms['sqrt'] = (st.skew(), st.kurtosis())

        # Box-Cox (sadece pozitif)
        if (t > 0).all():
            try:
                bc, lam = stats.boxcox(t.values)
                transforms[f'boxcox(λ={lam:.2f})'] = (pd.Series(bc).skew(),
                                                        pd.Series(bc).kurtosis())
            except Exception:
                pass

        # Square
        sq = t**2
        transforms['square'] = (sq.skew(), sq.kurtosis())

        # Rank
        rk = t.rank()
        transforms['rank'] = (rk.skew(), rk.kurtosis())

        R.add(f"\n  Transform karşılaştırması:")
        R.add(f"  {'Transform':<25} {'Skew':>10} {'Kurt':>10} {'Skew↓?':>8}")
        R.add(f"  {'-'*58}")
        orig_sk = abs(t.skew())
        for name, (sk, ku) in transforms.items():
            better = "✅" if abs(sk) < orig_sk else "❌"
            R.add(f"  {name:<25} {sk:>10.3f} {ku:>10.3f} {better:>8}")

        best = min(transforms.items(), key=lambda x: abs(x[1][0]))
        R.add(f"\n  🏆 En iyi transform: {best[0]} (skew={best[1][0]:.3f})")

    # ================================================================ #
    # S38 – DATA QUALITY SCORE   ← YENİ
    # ================================================================ #
    def _s38(self):
        R = self.R; R.sec("38. VERİ KALİTESİ SKORU")

        score = 100.0
        deductions = []

        # Missing
        miss_ratio = sum(self.train[c].isnull().sum() for c in self.feats) / (len(self.train) * len(self.feats))
        if miss_ratio > 0.1:
            d = 15; score -= d; deductions.append((f"Yüksek eksik veri ({miss_ratio:.1%})", d))
        elif miss_ratio > 0.01:
            d = 5; score -= d; deductions.append((f"Eksik veri ({miss_ratio:.1%})", d))

        # Duplicates
        dup_r = self.quality.get('duplicates', 0) / len(self.train)
        if dup_r > 0.1:
            d = 10; score -= d; deductions.append((f"Yüksek duplicate ({dup_r:.1%})", d))
        elif dup_r > 0.01:
            d = 3; score -= d; deductions.append((f"Duplicate ({dup_r:.1%})", d))

        # Label noise
        noise = self.quality.get('label_conflicts', 0)
        if noise > 100:
            d = 10; score -= d; deductions.append((f"Label noise ({noise:,} çelişki)", d))

        # Constant
        nc = self.quality.get('constant', 0)
        if nc > 0:
            d = 5; score -= d; deductions.append((f"Constant feature ({nc})", d))

        # Imbalance
        imb = self.quality.get('imbalance', 1)
        if imb > 10:
            d = 10; score -= d; deductions.append((f"Ciddi imbalance ({imb:.1f}:1)", d))
        elif imb > 3:
            d = 5; score -= d; deductions.append((f"Imbalance ({imb:.1f}:1)", d))

        # High cardinality
        hc = sum(1 for c in self.cats if self.train[c].nunique() > self.cfg.HIGH_CARD_THRESHOLD)
        if hc > 0:
            d = 5; score -= d; deductions.append((f"High cardinality ({hc} cat)", d))

        # Skewed features
        sk_count = sum(1 for c in self.nums if abs(self.train[c].skew()) > 2)
        if sk_count > len(self.nums) * 0.3:
            d = 5; score -= d; deductions.append((f"Çok skewed feature ({sk_count})", d))

        score = max(score, 0)

        R.add(f"  ┌──────────────────────────────────┐")
        R.add(f"  │  VERİ KALİTESİ: {score:>5.1f} / 100      │")
        R.add(f"  └──────────────────────────────────┘")
        R.add()

        if score >= 90:
            R.add(f"  ✅ MÜKEMMEL – Veri temiz, doğrudan modellemeye başla")
        elif score >= 75:
            R.add(f"  👍 İYİ – Küçük preprocessing yeterli")
        elif score >= 50:
            R.add(f"  ⚠️ ORTA – Ciddi preprocessing gerekli")
        else:
            R.add(f"  🚨 DÜŞÜK – Kapsamlı veri temizleme şart")

        if deductions:
            R.add(f"\n  Puan kırımları:")
            for reason, d in deductions:
                R.add(f"    -{d:>3} : {reason}")

    # ================================================================ #
    # S39 – FE FİKİRLERİ
    # ================================================================ #
    def _s39(self):
        R = self.R; R.sec("39. FEATURE ENGINEERING FİKİRLERİ")
        ideas = []
        if len(self.nums)>=2:
            nc = len(self.nums)
            ideas.append(f"🔢 Pairwise: ×, ÷, −, + ({nc}C2={nc*(nc-1)//2} çift)")
        ps = [c for c in self.nums if self.train[c].skew()>1 and (self.train[c]>0).all()]
        if ps: ideas.append(f"📐 Log: {ps[:5]}")
        if self.cats:
            ideas.append(f"🏷️ Target encoding ({len(self.cats)} cat)")
            ideas.append(f"🔢 Frequency encoding")
        if len(self.cats)>=2:
            ideas.append(f"🔗 Cat concat: {len(self.cats)}C2")
        if self.cats and self.nums:
            ideas.append(f"📊 GroupBy agg: cat × num → mean/std/min/max")
        if len(self.nums)>=3:
            ideas.append(f"📊 Row-wise: mean, std, min, max, sum, range")
        ideas.append(f"📦 Quantile binning")
        miss = [c for c in self.feats if self.train[c].isnull().sum()>0]
        if miss: ideas.append(f"❌ Missing indicator: {miss[:5]}")
        ideas.append(f"🎯 Polynomial (degree=2) top features")
        ideas.append(f"📉 PCA/UMAP bileşenleri")
        ideas.append(f"🔢 Rank transform")
        ideas.append(f"🔁 Cyclical encoding (sin/cos) varsa periyodik feature")

        R.add(f"  💡 ÖNERİLER:")
        for i,idea in enumerate(ideas,1):
            R.add(f"    {i:>2}. {idea}")

    # ================================================================ #
    # S40 – CV STRATEJİSİ
    # ================================================================ #
    def _s40(self):
        R = self.R; R.sec("40. CV STRATEJİSİ & METRİK")
        n = len(self.train)
        if self.ptype in ('binary','multiclass'):
            vc = self.train[self.target].value_counts()
            mc = vc.min()
            if mc < 50:
                R.add(f"  → RepeatedStratifiedKFold(5, repeats=3)  [min_class={mc}]")
            elif n < 10000:
                R.add(f"  → StratifiedKFold(10, shuffle=True)")
            else:
                R.add(f"  → StratifiedKFold(5, shuffle=True)")
        else:
            R.add(f"  → KFold(5, shuffle=True)")

        R.add(f"\n  🎯 METRİK:")
        if self.ptype == 'binary':
            R.add(f"    Primary  : AUC-ROC, LogLoss")
            R.add(f"    Secondary: F1, PR-AUC, MCC")
        elif self.ptype == 'multiclass':
            R.add(f"    Primary  : MacroF1, LogLoss")
            R.add(f"    Secondary: Accuracy, WeightedF1, Kappa")
        else:
            R.add(f"    Primary  : RMSE, MAE")
            R.add(f"    Secondary: R², RMSLE, MAPE")

        R.add(f"\n  ⚠️ YARIŞMA METRİĞİNİ KONTROL ET!")
        R.add(f"  🔀 Adversarial validation → AUC≈0.5 ise OK, >0.7 ise drift")

    # ================================================================ #
    # S41 – MODELLEME ÖNERİLERİ
    # ================================================================ #
    def _s41(self):
        R = self.R; R.sec("41. MODELLEME ÖNERİLERİ")
        recs = []
        if self.ptype != 'regression':
            imb = self.quality.get('imbalance', 1)
            if imb > 10:  recs.append("⚖️ class_weight / focal loss / SMOTE+Tomek")
            elif imb > 3: recs.append("⚖️ scale_pos_weight / is_unbalance")
        hc = [c for c in self.cats if self.train[c].nunique() > self.cfg.HIGH_CARD_THRESHOLD]
        if hc: recs.append(f"🏷️ High card → CatBoost native / target enc: {hc[:3]}")
        sk = [c for c in self.nums if abs(self.train[c].skew()) > 2]
        if sk: recs.append(f"📐 Skewed → transform: {sk[:3]}")

        recs.append("")
        recs.append("🤖 MODEL STACK:")
        if self.ptype == 'binary':
            recs.append("  L0: LightGBM, XGBoost, CatBoost, ExtraTrees")
            recs.append("  L1: LogisticRegression (stacker)")
        elif self.ptype == 'multiclass':
            recs.append("  L0: LightGBM, XGBoost, CatBoost")
            recs.append("  L1: Ridge (stacker)")
        else:
            recs.append("  L0: LightGBM, XGBoost, CatBoost, Ridge")
            recs.append("  L1: Linear blend")

        recs += ["",
            "🏆 GRANDMASTER TACTICS:",
            "  • Seed averaging (3-5 seed)",
            "  • Pseudo-labeling (conf>0.95)",
            "  • Target encoding with 5-fold regularization",
            "  • Post-processing: threshold tune, rounding, clipping",
            "  • Null importance feature selection",
            "  • Optuna hyperparameter tuning (100+ trials)",
            "  • Blend ≥3 farklı model ailesi",
            "  • Test-time augmentation (eğer uygunsa)",
            "  • Orijinal veri ekleme (PS yarışmaları)",
            "  • Out-of-fold prediction consistency check",
        ]
        for rec in recs:
            R.add(f"  {rec}")

    # ================================================================ #
    # S42 – FİNAL CHECKLIST
    # ================================================================ #
    def _s42(self):
        R = self.R; R.sec("42. PRE-MODELING CHECKLIST")

        checks = [
            ("Eksik veri", "✅" if self.quality.get('missing',0)==0
                           else f"❌ {self.quality['missing']} kolon"),
            ("Constant feature",    "✅" if self.quality.get('constant',0)==0
                                    else f"❌ {self.quality['constant']}"),
            ("Quasi-constant",      "✅" if self.quality.get('quasi_const',0)==0
                                    else f"⚠️ {self.quality['quasi_const']}"),
            ("Duplicate",           "✅" if self.quality.get('duplicates',0)==0
                                    else f"⚠️ {self.quality['duplicates']:,}"),
            ("Label noise",         "✅" if self.quality.get('label_conflicts',0)==0
                                    else f"⚠️ {self.quality.get('label_conflicts',0):,}"),
            ("Leakage kontrolü",    "→ Bölüm 26"),
            ("Encoding planı",      "✅ yok" if not self.cats
                                    else f"⚠️ {len(self.cats)} cat"),
            ("Drift kontrolü",      "→ Bölüm 23+25+35"),
            ("CV stratejisi",       "→ Bölüm 40"),
            ("Imbalance",           "✅" if self.quality.get('imbalance',1)<3
                                    else f"⚠️ {self.quality.get('imbalance',1):.1f}:1"),
            ("Skewed transform",    f"{sum(1 for c in self.nums if abs(self.train[c].skew())>2)} skewed"),
            ("Submission format",   "→ Bölüm 3"),
            ("Orijinal veri",       "→ Bölüm 4+34"),
            ("Date/time features",  "→ Bölüm 27"),
            ("Ordinal encoding",    "→ Bölüm 31"),
            ("Feature interactions", "→ Bölüm 19"),
            ("Row-wise meta",       "→ Bölüm 20"),
            ("Feature clustering",  "→ Bölüm 36"),
            ("Non-linear check",    "→ Bölüm 32"),
            ("Target transform",    "→ Bölüm 37" if self.ptype=='regression' else "N/A"),
        ]

        R.add(f"  {'#':<4} {'Kontrol':<30} {'Durum':>40}")
        R.add(f"  {'-'*78}")
        for i,(ch,st) in enumerate(checks,1):
            R.add(f"  {i:<4} {ch:<30} {st:>40}")

        R.add(f"\n{'='*80}")
        R.add(f"{'🏆 GRANDMASTER EDA TAMAMLANDI — 42 BÖLÜM':^80}")
        R.add(f"{'='*80}")


# =============================================================================
# 5. ÇALIŞTIRMA
# =============================================================================

def run_eda():
    cfg = EDAConfig()
    eda = GrandmasterEDA(cfg)
    eda.load()
    report = eda.run()
    report.save("grandmaster_eda_report.txt")
    return eda

if __name__ == "__main__":
    eda_obj = run_eda()
```

# 2. Öneri KAGGLE GRANDMASTER EDA

```py
# =============================================================================
# KAGGLE GRANDMASTER EDA v3.0 - KAPSAMLI YAZILI RAPOR (29 BÖLÜM)
# Sadece EDA, model yok. Tüm bilgiler metin olarak çıktı verir.
# =============================================================================

import numpy as np
import pandas as pd
import warnings
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy import stats
warnings.filterwarnings('ignore')

# =============================================================================
# 1. KONFİGÜRASYON
# =============================================================================

class EDAConfig:
    BASE_PATH = "/kaggle/input/playground-series-s6e4/"
    TRAIN_FILE = "train.csv"
    TEST_FILE = "test.csv"
    SAMPLE_SUB_FILE = "sample_submission.csv"
    ORIGINAL_FILE = None  # Orijinal dataset varsa path

    TARGET_COL = None
    ID_COL = None

    CAT_THRESHOLD = 10
    HIGH_CARDINALITY_THRESHOLD = 100
    DRIFT_THRESHOLD = 0.05
    MISSING_THRESHOLD = 0.5
    RARE_THRESHOLD = 0.01          # %1'den az olan kategoriler
    QUASI_CONSTANT_THRESHOLD = 0.97

    TOP_N_FEATURES = 20
    TOP_N_CATEGORIES = 10
    BIN_COUNT = 10
    REGRESSION_NUNIQUE_THRESHOLD = 30
    INTERACTION_TOP_K = 8          # Interaction için top-K feature
    SAMPLE_SIZE_FOR_MI = 50000     # MI hesabı için max sample


# =============================================================================
# 2. OTOMATİK ALGILAMA
# =============================================================================

class AutoDetector:
    @staticmethod
    def detect_target(train_df, config):
        if config.TARGET_COL and config.TARGET_COL in train_df.columns:
            return config.TARGET_COL
        candidates = [
            'target', 'Target', 'TARGET', 'label', 'Label',
            'class', 'Class', 'y', 'Y', 'outcome', 'Outcome',
            'survived', 'Survived', 'SalePrice', 'price', 'Price',
            'irrigation_need', 'Irrigation_Need',
        ]
        for col in candidates:
            if col in train_df.columns:
                return col
        cols = [c for c in train_df.columns if c.lower() not in ['id']]
        return cols[-1] if cols else None

    @staticmethod
    def detect_id(df, config):
        if config.ID_COL and config.ID_COL in df.columns:
            return config.ID_COL
        for col in ['id', 'ID', 'Id', 'index', 'sample_id', 'row_id']:
            if col in df.columns:
                return col
        return None

    @staticmethod
    def detect_problem_type(train_df, target_col, config):
        dtype = train_df[target_col].dtype
        n_unique = train_df[target_col].nunique()
        if dtype in ['float64', 'float32'] and n_unique > config.REGRESSION_NUNIQUE_THRESHOLD:
            return 'regression'
        if dtype in ['int64', 'int32'] and n_unique > config.REGRESSION_NUNIQUE_THRESHOLD:
            return 'regression'
        if n_unique == 2:
            return 'binary'
        elif n_unique > 2:
            return 'multiclass'
        return 'regression'

    @staticmethod
    def detect_features(train_df, feature_cols, threshold):
        cats, nums = [], []
        for col in feature_cols:
            unique_count = train_df[col].nunique()
            dtype = train_df[col].dtype
            if dtype == 'object' or dtype.name == 'category' or unique_count <= threshold:
                cats.append(col)
            else:
                nums.append(col)
        return cats, nums


# =============================================================================
# 3. RAPOR SINIFI
# =============================================================================

class EDAReport:
    def __init__(self):
        self.lines = []

    def add(self, text=""):
        self.lines.append(text)

    def section(self, title, char="="):
        self.lines.append(f"\n{char * 80}")
        self.lines.append(f"{title:^80}")
        self.lines.append(f"{char * 80}\n")

    def subsection(self, title):
        self.lines.append(f"\n{'─' * 80}")
        self.lines.append(f"  {title}")
        self.lines.append(f"{'─' * 80}\n")

    def print_report(self):
        print("\n".join(self.lines))

    def save(self, filename="eda_report.txt"):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.lines))
        print(f"\n💾 Rapor kaydedildi: {filename}")


# =============================================================================
# 4. ANA EDA SINIFI
# =============================================================================

class GrandmasterEDA:
    def __init__(self, config):
        self.cfg = config
        self.rpt = EDAReport()
        self.train = self.test = self.sample_sub = self.original = None
        self.target_col = self.id_col = self.problem_type = None
        self.cat_cols = self.num_cols = self.feature_cols = []

    # ── Yardımcılar ──────────────────────────────────────────────────── #
    def _encode_target(self):
        t = self.train[self.target_col]
        if t.dtype == 'object':
            return LabelEncoder().fit_transform(t)
        return t.values.copy()

    def _safe_corr(self, a, b):
        try:
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 10:
                return 0.0
            r = np.corrcoef(a[mask], b[mask])[0, 1]
            return 0.0 if np.isnan(r) else r
        except Exception:
            return 0.0

    # ── Veri Yükleme ─────────────────────────────────────────────────── #
    def load_data(self):
        self.train = pd.read_csv(self.cfg.BASE_PATH + self.cfg.TRAIN_FILE)
        self.test = pd.read_csv(self.cfg.BASE_PATH + self.cfg.TEST_FILE)

        sub_path = self.cfg.BASE_PATH + self.cfg.SAMPLE_SUB_FILE
        if os.path.exists(sub_path):
            self.sample_sub = pd.read_csv(sub_path)

        if self.cfg.ORIGINAL_FILE and os.path.exists(self.cfg.ORIGINAL_FILE):
            self.original = pd.read_csv(self.cfg.ORIGINAL_FILE)

        self.target_col = AutoDetector.detect_target(self.train, self.cfg)
        self.id_col = AutoDetector.detect_id(self.train, self.cfg)
        self.problem_type = AutoDetector.detect_problem_type(
            self.train, self.target_col, self.cfg
        )
        self.feature_cols = [
            c for c in self.train.columns if c not in [self.target_col, self.id_col]
        ]
        self.cat_cols, self.num_cols = AutoDetector.detect_features(
            self.train, self.feature_cols, self.cfg.CAT_THRESHOLD
        )

    # ── Ana Çalıştırma ───────────────────────────────────────────────── #
    def run(self):
        r = self.rpt
        r.section("KAGGLE GRANDMASTER EDA REPORT  v3.0")
        r.add(f"  Timestamp    : {pd.Timestamp.now()}")
        r.add(f"  Target       : {self.target_col}")
        r.add(f"  Problem Type : {self.problem_type.upper()}")

        self._s01_basic()
        self._s02_memory()
        self._s03_sample_submission()
        self._s04_original_dataset()
        self._s05_target()
        self._s06_missing()
        self._s07_duplicates()
        self._s08_constant()
        self._s09_rare_categories()
        self._s10_categorical()
        self._s11_numerical()
        self._s12_per_class_stats()
        self._s13_binned_target_rate()
        self._s14_statistical_tests()
        self._s15_mutual_information()
        self._s16_correlation()
        self._s17_multicollinearity()
        self._s18_feature_importance()
        self._s19_feature_interaction()
        self._s20_row_wise_stats()
        self._s21_outlier_impact()
        self._s22_index_trend()
        self._s23_drift()
        self._s24_overlap()
        self._s25_adversarial_proxy()
        self._s26_leakage()
        self._s27_fe_ideas()
        self._s28_cv_strategy()
        self._s29_recommendations()
        self._s30_checklist()

        r.print_report()
        return r

    # ================================================================ #
    # S01 – TEMEL BİLGİLER
    # ================================================================ #
    def _s01_basic(self):
        r = self.rpt
        r.section("1. TEMEL VERİ BİLGİLERİ")
        r.add(f"  SHAPE:")
        r.add(f"    Train : {self.train.shape[0]:>10,} rows × {self.train.shape[1]:>4} cols")
        r.add(f"    Test  : {self.test.shape[0]:>10,} rows × {self.test.shape[1]:>4} cols")
        r.add(f"    Ratio : test/train = {self.test.shape[0]/self.train.shape[0]:.2f}x")
        r.add()
        r.add(f"  TARGET : {self.target_col}  (dtype={self.train[self.target_col].dtype})")
        r.add(f"  ID     : {self.id_col}")
        r.add()
        r.add(f"  FEATURES ({len(self.feature_cols)}):")
        r.add(f"    Categorical : {len(self.cat_cols):>3}  {self.cat_cols[:6]}")
        r.add(f"    Numeric     : {len(self.num_cols):>3}  {self.num_cols[:6]}")
        r.add()
        r.add(f"  DTYPE BREAKDOWN:")
        for dt, cnt in self.train[self.feature_cols].dtypes.value_counts().items():
            r.add(f"    {str(dt):<15}: {cnt}")

    # ================================================================ #
    # S02 – BELLEK
    # ================================================================ #
    def _s02_memory(self):
        r = self.rpt
        r.section("2. BELLEK KULLANIMI")
        tr_mb = self.train.memory_usage(deep=True).sum() / 1024**2
        te_mb = self.test.memory_usage(deep=True).sum() / 1024**2
        r.add(f"  Train : {tr_mb:.2f} MB")
        r.add(f"  Test  : {te_mb:.2f} MB")
        r.add(f"  Total : {tr_mb+te_mb:.2f} MB")

        downcast = []
        for col in self.feature_cols:
            d = self.train[col].dtype
            if d == 'float64':
                downcast.append((col, 'float64→float32'))
            elif d == 'int64':
                mn, mx = self.train[col].min(), self.train[col].max()
                if mn >= -32768 and mx <= 32767:
                    downcast.append((col, 'int64→int16'))
                elif mn >= -2147483648 and mx <= 2147483647:
                    downcast.append((col, 'int64→int32'))
        if downcast:
            r.add(f"\n  ⚡ Downcast önerileri ({len(downcast)}):")
            for c, rec in downcast[:10]:
                r.add(f"    {c}: {rec}")
            if len(downcast) > 10:
                r.add(f"    ... +{len(downcast)-10} daha")

    # ================================================================ #
    # S03 – SAMPLE SUBMISSION
    # ================================================================ #
    def _s03_sample_submission(self):
        r = self.rpt
        r.section("3. SAMPLE SUBMISSION ANALİZİ")
        if self.sample_sub is None:
            r.add("  ⚠️ sample_submission.csv bulunamadı")
            return
        r.add(f"  Shape   : {self.sample_sub.shape}")
        r.add(f"  Columns : {list(self.sample_sub.columns)}")
        r.add(f"  Dtypes  :")
        for col in self.sample_sub.columns:
            r.add(f"    {col}: {self.sample_sub[col].dtype}  "
                   f"(nunique={self.sample_sub[col].nunique()}, "
                   f"sample={self.sample_sub[col].iloc[0]})")

        # Submission formatı tahmini
        non_id = [c for c in self.sample_sub.columns if c.lower() != 'id']
        if len(non_id) == 1:
            r.add(f"\n  📝 Tek kolon tahmin: {non_id[0]}")
            if self.sample_sub[non_id[0]].dtype == 'float64':
                r.add(f"     → Probability veya regression output bekleniyor")
            else:
                r.add(f"     → Class label bekleniyor")
        elif len(non_id) > 1:
            r.add(f"\n  📝 Çoklu kolon: muhtemelen her sınıf için probability")
            r.add(f"     Kolonlar: {non_id}")

    # ================================================================ #
    # S04 – ORİJİNAL DATASET (Playground Series)
    # ================================================================ #
    def _s04_original_dataset(self):
        r = self.rpt
        r.section("4. ORİJİNAL DATASET KONTROLÜ")
        if self.original is not None:
            r.add(f"  ✅ Orijinal dataset yüklendi: {self.original.shape}")
            r.add(f"     Train'e eklenerek kullanılabilir (PS yarışmaları)")
            common = set(self.train.columns) & set(self.original.columns)
            r.add(f"     Ortak kolonlar: {len(common)}")
        else:
            r.add(f"  ℹ️ Orijinal dataset belirtilmedi veya bulunamadı")
            r.add(f"     Playground Series ise orijinal veriyi bulmayı dene:")
            r.add(f"     → Yarışma sayfasındaki 'Overview' sekmesine bak")
            r.add(f"     → Orijinal veri train'e eklenince skor genelde artar")

    # ================================================================ #
    # S05 – HEDEF DEĞİŞKEN
    # ================================================================ #
    def _s05_target(self):
        r = self.rpt
        r.section("5. HEDEF DEĞİŞKEN ANALİZİ")
        t = self.train[self.target_col]

        if self.problem_type == 'regression':
            desc = t.describe()
            for idx in desc.index:
                r.add(f"  {idx:<10}: {desc[idx]:>15.4f}")
            sk, ku = t.skew(), t.kurtosis()
            r.add(f"  {'skew':<10}: {sk:>15.4f}")
            r.add(f"  {'kurtosis':<10}: {ku:>15.4f}")
            r.add(f"  zeros    : {(t==0).sum():,} ({(t==0).mean():.2%})")
            r.add(f"  negative : {(t<0).sum():,} ({(t<0).mean():.2%})")
            if abs(sk) > 1:
                r.add(f"\n  ⚠️ Target skewed → log1p veya Box-Cox dönüşümü düşün")
            if (t <= 0).any() and abs(sk) > 1:
                r.add(f"     Negatif/sıfır değerler var → log1p(target - min + 1)")
            return

        vc = t.value_counts().sort_index()
        vr = t.value_counts(normalize=True).sort_index()
        r.add(f"  {'Class':<20} {'Count':>10} {'Ratio':>10}")
        r.add(f"  {'-'*45}")
        for val in vc.index:
            bar = "█" * int(vr[val] * 40)
            r.add(f"  {str(val):<20} {vc[val]:>10,} {vr[val]:>9.2%}  {bar}")

        imb = vr.max() / vr.min()
        r.add(f"\n  Imbalance ratio: {imb:.1f}:1")
        if imb > 10:
            r.add(f"  🚨 HIGHLY IMBALANCED → SMOTE/ADASYN + class weights")
        elif imb > 3:
            r.add(f"  ⚠️ MODERATELY IMBALANCED → stratified CV + class weights")
        else:
            r.add(f"  ✅ BALANCED")

    # ================================================================ #
    # S06 – EKSİK VERİ
    # ================================================================ #
    def _s06_missing(self):
        r = self.rpt
        r.section("6. EKSİK VERİ ANALİZİ")
        tr_m = self.train.isnull().sum()
        te_m = self.test.isnull().sum()
        cols = tr_m[tr_m > 0].index.tolist()
        te_only = [c for c in te_m[te_m > 0].index
                    if c not in cols and c != self.target_col]
        all_m = list(set(cols + te_only))

        if not all_m:
            r.add("  ✅ Eksik veri yok")
            return

        r.add(f"  {'Column':<25} {'Tr#':>8} {'Tr%':>7} {'Te#':>8} {'Te%':>7} {'Aksiyon':>12}")
        r.add(f"  {'-'*72}")
        for col in sorted(all_m):
            tn = tr_m.get(col, 0)
            tp = tn / len(self.train) * 100
            ten = te_m.get(col, 0) if col in self.test.columns else 0
            tep = ten / len(self.test) * 100 if col in self.test.columns else 0
            act = "DROP" if tp > 50 else ("DRIFT!" if abs(tp-tep) > 10 else "IMPUTE")
            r.add(f"  {col:<25} {tn:>8,} {tp:>6.1f}% {ten:>8,} {tep:>6.1f}% {act:>12}")

        any_miss = self.train[cols].isnull().any(axis=1).sum()
        r.add(f"\n  En az 1 eksik olan satır: {any_miss:,} ({any_miss/len(self.train):.1%})")

        # Eksik korelasyon (eksikler birlikte mi oluşuyor?)
        if len(cols) >= 2:
            miss_corr = self.train[cols].isnull().corr()
            high = []
            for i in range(len(cols)):
                for j in range(i+1, len(cols)):
                    c = abs(miss_corr.iloc[i, j])
                    if c > 0.5:
                        high.append((cols[i], cols[j], c))
            if high:
                r.add(f"\n  ⚠️ Birlikte eksik olan çiftler (missingness corr > 0.5):")
                for c1, c2, c in sorted(high, key=lambda x: x[2], reverse=True)[:5]:
                    r.add(f"    {c1} ↔ {c2}: {c:.3f}")

    # ================================================================ #
    # S07 – DUPLICATE
    # ================================================================ #
    def _s07_duplicates(self):
        r = self.rpt
        r.section("7. DUPLICATE ANALİZİ")
        dup_all = self.train.duplicated().sum()
        dup_feat = self.train[self.feature_cols].duplicated().sum()
        r.add(f"  Tam satır duplicate  : {dup_all:>8,} ({dup_all/len(self.train):.2%})")
        r.add(f"  Feature-only dup     : {dup_feat:>8,} ({dup_feat/len(self.train):.2%})")

        test_feat = [c for c in self.feature_cols if c in self.test.columns]
        te_dup = self.test[test_feat].duplicated().sum()
        r.add(f"  Test feature-only dup: {te_dup:>8,} ({te_dup/len(self.test):.2%})")

        if dup_feat > 0:
            mask = self.train[self.feature_cols].duplicated(keep=False)
            dup_rows = self.train[mask]
            try:
                conflict = dup_rows.groupby(self.feature_cols, dropna=False)[
                    self.target_col].nunique()
                n_conflict = (conflict > 1).sum()
                r.add(f"\n  ⚠️ Çelişkili target (aynı X, farklı y): {n_conflict:,} grup")
                if n_conflict > 0:
                    r.add(f"     → Noisy labels! Cleaning veya label smoothing düşün")
            except Exception:
                r.add(f"\n  ℹ️ Çelişki kontrolü yapılamadı (çok fazla unique kombinasyon)")
        else:
            r.add(f"\n  ✅ Duplicate sorunu yok")

    # ================================================================ #
    # S08 – CONSTANT / QUASI-CONSTANT
    # ================================================================ #
    def _s08_constant(self):
        r = self.rpt
        r.section("8. CONSTANT / QUASI-CONSTANT ÖZELLİKLER")
        const, quasi = [], []
        for col in self.feature_cols:
            nu = self.train[col].nunique(dropna=False)
            if nu <= 1:
                const.append(col)
            else:
                top_f = self.train[col].value_counts(normalize=True, dropna=False).iloc[0]
                if top_f > self.cfg.QUASI_CONSTANT_THRESHOLD:
                    quasi.append((col, top_f))

        if const:
            r.add(f"  🚨 CONSTANT ({len(const)}) → DROP:")
            for c in const:
                r.add(f"    {c}")
        else:
            r.add(f"  ✅ Constant yok")

        if quasi:
            r.add(f"\n  ⚠️ QUASI-CONSTANT ({len(quasi)}) (>{self.cfg.QUASI_CONSTANT_THRESHOLD:.0%}):")
            for c, f in quasi:
                r.add(f"    {c}: {f:.1%} aynı değer")
        else:
            r.add(f"  ✅ Quasi-constant yok")

    # ================================================================ #
    # S09 – RARE CATEGORIES
    # ================================================================ #
    def _s09_rare_categories(self):
        r = self.rpt
        r.section("9. RARE CATEGORY TESPİTİ")
        if not self.cat_cols:
            r.add("  ℹ️ Kategorik feature yok")
            return

        threshold = self.cfg.RARE_THRESHOLD
        r.add(f"  Eşik: <%{threshold*100:.1f} olan kategoriler\n")
        r.add(f"  {'Feature':<25} {'#Rare':>8} {'%Rare':>8} {'Rare Values (sample)':>30}")
        r.add(f"  {'-'*75}")

        any_rare = False
        for col in self.cat_cols:
            vc = self.train[col].value_counts(normalize=True)
            rare_vals = vc[vc < threshold]
            if len(rare_vals) > 0:
                any_rare = True
                rare_pct = rare_vals.sum()
                samples = list(rare_vals.index[:3])
                r.add(f"  {col:<25} {len(rare_vals):>8} {rare_pct:>7.1%} {str(samples):>30}")

        if not any_rare:
            r.add(f"  ✅ Rare category yok")
        else:
            r.add(f"\n  💡 Rare kategoriler için:")
            r.add(f"     • 'Other' grubuna birleştir")
            r.add(f"     • Frequency encoding kullan")
            r.add(f"     • min_samples_leaf yüksek tut (tree modellerde)")

        # Train'de yok test'te var
        r.add(f"\n  UNSEEN CATEGORIES (test'te var, train'de yok):")
        unseen_found = False
        for col in self.cat_cols:
            if col not in self.test.columns:
                continue
            tr_vals = set(self.train[col].dropna().unique())
            te_vals = set(self.test[col].dropna().unique())
            unseen = te_vals - tr_vals
            if unseen:
                unseen_found = True
                r.add(f"    {col}: {len(unseen)} unseen → {list(unseen)[:5]}")
        if not unseen_found:
            r.add(f"    ✅ Tüm test kategorileri train'de mevcut")

    # ================================================================ #
    # S10 – KATEGORİK ANALİZ
    # ================================================================ #
    def _s10_categorical(self):
        r = self.rpt
        r.section("10. KATEGORİK ÖZELLİK ANALİZİ")
        if not self.cat_cols:
            r.add("  ℹ️ Kategorik feature yok")
            return

        r.add(f"  {'Feature':<25} {'Unique':>8} {'Top':>15} {'Top%':>8} {'Card':>8}")
        r.add(f"  {'-'*68}")
        for col in self.cat_cols:
            nu = self.train[col].nunique()
            tv = self.train[col].value_counts().index[0]
            tp = self.train[col].value_counts(normalize=True).iloc[0]
            cd = "HIGH" if nu > self.cfg.HIGH_CARDINALITY_THRESHOLD else "OK"
            r.add(f"  {col:<25} {nu:>8} {str(tv)[:15]:>15} {tp:>7.1%} {cd:>8}")

        for col in self.cat_cols[:self.cfg.TOP_N_CATEGORIES]:
            r.subsection(f"📋 {col}")
            vc = self.train[col].value_counts().head(8)
            for val, cnt in vc.items():
                ratio = cnt / len(self.train)
                bar = "█" * int(ratio * 40)
                r.add(f"    {str(val):<20} {cnt:>8,} ({ratio:>5.1%}) {bar}")

            if self.problem_type != 'regression':
                r.add(f"\n    Target ilişkisi:")
                ct = pd.crosstab(self.train[col], self.train[self.target_col],
                                  normalize='index')
                for val in ct.index[:5]:
                    row = ct.loc[val]
                    dom = row.idxmax()
                    r.add(f"      {str(val):<18} → {dom} ({row.max():.1%})")

    # ================================================================ #
    # S11 – NUMERİK ANALİZ
    # ================================================================ #
    def _s11_numerical(self):
        r = self.rpt
        r.section("11. NUMERİK ÖZELLİK ANALİZİ")
        if not self.num_cols:
            r.add("  ℹ️ Numerik feature yok")
            return

        r.add(f"  {'Feature':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Skew':>7} {'Out%':>7}")
        r.add(f"  {'-'*80}")
        for col in self.num_cols:
            s = self.train[col]
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR = Q3 - Q1
            op = ((s < Q1-1.5*IQR) | (s > Q3+1.5*IQR)).mean()*100
            r.add(f"  {col:<20} {s.mean():>10.2f} {s.std():>10.2f} "
                   f"{s.min():>10.2f} {s.max():>10.2f} {s.skew():>7.2f} {op:>6.1f}%")

        for col in self.num_cols[:self.cfg.TOP_N_FEATURES]:
            r.subsection(f"🔢 {col}")
            d = self.train[col].describe()
            r.add(f"    Count={d['count']:,.0f}  Mean={d['mean']:.4f}  Std={d['std']:.4f}")
            r.add(f"    Min={d['min']:.4f}  Q1={d['25%']:.4f}  Med={d['50%']:.4f}  "
                   f"Q3={d['75%']:.4f}  Max={d['max']:.4f}")
            sk = self.train[col].skew()
            ku = self.train[col].kurtosis()
            r.add(f"    Skew={sk:.3f}  Kurt={ku:.3f}")
            z_cnt = (self.train[col] == 0).sum()
            n_cnt = (self.train[col] < 0).sum()
            r.add(f"    Zero={z_cnt:,} ({z_cnt/len(self.train):.1%})  "
                   f"Neg={n_cnt:,} ({n_cnt/len(self.train):.1%})")

            if abs(sk) > 2:
                r.add(f"    🚨 Highly skewed → log/sqrt/Box-Cox")

            # Text histogram
            try:
                bins = pd.cut(self.train[col], bins=self.cfg.BIN_COUNT)
                bc = bins.value_counts().sort_index()
                mx = bc.max()
                r.add(f"    Distribution:")
                for iv, cnt in bc.items():
                    bl = int(cnt / mx * 25) if mx > 0 else 0
                    r.add(f"      {str(iv):<30} {cnt:>7,} {'█'*bl}")
            except Exception:
                pass

    # ================================================================ #
    # S12 – SINIF BAZLI İSTATİSTİK
    # ================================================================ #
    def _s12_per_class_stats(self):
        r = self.rpt
        r.section("12. SINIF BAZLI NUMERİK İSTATİSTİKLER")
        if self.problem_type == 'regression' or not self.num_cols:
            r.add("  ℹ️ Regression veya numerik yok – atlanıyor")
            return

        classes = sorted(self.train[self.target_col].unique())
        if len(classes) > 10:
            r.add(f"  ℹ️ {len(classes)} sınıf, ilk 10 gösteriliyor")
            classes = classes[:10]

        for col in self.num_cols[:10]:
            r.subsection(f"📊 {col} per class")
            hdr = f"    {'Class':<15} {'Mean':>10} {'Std':>10} {'Median':>10} {'Min':>10} {'Max':>10}"
            r.add(hdr)
            r.add(f"    {'-'*70}")
            for cls in classes:
                ss = self.train.loc[self.train[self.target_col] == cls, col]
                r.add(f"    {str(cls):<15} {ss.mean():>10.3f} {ss.std():>10.3f} "
                       f"{ss.median():>10.3f} {ss.min():>10.3f} {ss.max():>10.3f}")

    # ================================================================ #
    # S13 – BINNED TARGET RATE
    # ================================================================ #
    def _s13_binned_target_rate(self):
        r = self.rpt
        r.section("13. BINNED TARGET RATE (Numerik → Target)")
        if self.problem_type == 'regression' or not self.num_cols:
            r.add("  ℹ️ Regression veya numerik yok – atlanıyor")
            return

        target_enc = self._encode_target()
        for col in self.num_cols[:10]:
            r.subsection(f"📈 {col}")
            try:
                bins = pd.qcut(self.train[col], q=self.cfg.BIN_COUNT, duplicates='drop')
                tmp = pd.DataFrame({'bin': bins, 'target': target_enc})
                grp = tmp.groupby('bin', observed=True)['target'].agg(['mean', 'count'])
                r.add(f"    {'Bin':<30} {'Count':>8} {'TargetMean':>12}")
                r.add(f"    {'-'*55}")
                for idx, row in grp.iterrows():
                    r.add(f"    {str(idx):<30} {int(row['count']):>8} {row['mean']:>12.4f}")
                means = grp['mean'].values
                diffs = np.diff(means)
                if len(diffs) > 0 and (np.all(diffs >= 0) or np.all(diffs <= 0)):
                    r.add(f"    ✅ MONOTONİK → güçlü sinyal")
                elif len(diffs) > 0:
                    sc = np.sum(np.diff(np.sign(diffs)) != 0)
                    r.add(f"    ℹ️ Non-monotonic ({sc} yön değişimi)")
            except Exception as e:
                r.add(f"    ⚠️ Hata: {e}")

    # ================================================================ #
    # S14 – İSTATİSTİKSEL TESTLER
    # ================================================================ #
    def _s14_statistical_tests(self):
        r = self.rpt
        r.section("14. İSTATİSTİKSEL TESTLER")

        if self.problem_type == 'regression':
            r.add("  Pearson + Spearman korelasyon (target ile):\n")
            r.add(f"  {'Feature':<25} {'Pearson':>10} {'p-val':>10} {'Spearman':>10} {'p-val':>10}")
            r.add(f"  {'-'*70}")
            for col in self.num_cols[:self.cfg.TOP_N_FEATURES]:
                try:
                    pr, pp = stats.pearsonr(
                        self.train[col].fillna(self.train[col].median()),
                        self.train[self.target_col])
                    sr, sp = stats.spearmanr(
                        self.train[col].fillna(self.train[col].median()),
                        self.train[self.target_col])
                    r.add(f"  {col:<25} {pr:>10.4f} {pp:>10.2e} {sr:>10.4f} {sp:>10.2e}")
                except Exception:
                    pass
            return

        # Classification: Kruskal-Wallis (numerik) + Chi2 (kategorik)
        target_enc = self._encode_target()
        classes = sorted(self.train[self.target_col].unique())

        r.add("  NUMERIK → Kruskal-Wallis H-test (non-parametric ANOVA):\n")
        r.add(f"  {'Feature':<25} {'H-stat':>10} {'p-value':>12} {'Significant':>12}")
        r.add(f"  {'-'*64}")
        kw_results = []
        for col in self.num_cols:
            try:
                groups = [self.train.loc[self.train[self.target_col]==c, col].dropna()
                          for c in classes]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) >= 2:
                    h, p = stats.kruskal(*groups)
                    sig = "YES ***" if p < 0.001 else ("YES **" if p < 0.01 else
                          ("YES *" if p < 0.05 else "NO"))
                    kw_results.append((col, h, p, sig))
            except Exception:
                pass
        kw_results.sort(key=lambda x: x[1], reverse=True)
        for col, h, p, sig in kw_results[:self.cfg.TOP_N_FEATURES]:
            r.add(f"  {col:<25} {h:>10.2f} {p:>12.2e} {sig:>12}")

        if self.cat_cols:
            r.add(f"\n  KATEGORİK → Chi-Square test:\n")
            r.add(f"  {'Feature':<25} {'Chi2':>12} {'p-value':>12} {'Cramér V':>10} {'Sig':>8}")
            r.add(f"  {'-'*72}")
            for col in self.cat_cols:
                try:
                    ct = pd.crosstab(self.train[col], self.train[self.target_col])
                    chi2, p, dof, _ = stats.chi2_contingency(ct)
                    n = len(self.train)
                    k = min(ct.shape) - 1
                    cv = np.sqrt(chi2 / (n * k)) if k > 0 else 0
                    sig = "***" if p < 0.001 else ("**" if p < 0.01 else
                          ("*" if p < 0.05 else "ns"))
                    r.add(f"  {col:<25} {chi2:>12.1f} {p:>12.2e} {cv:>10.4f} {sig:>8}")
                except Exception:
                    pass

    # ================================================================ #
    # S15 – MUTUAL INFORMATION
    # ================================================================ #
    def _s15_mutual_information(self):
        r = self.rpt
        r.section("15. MUTUAL INFORMATION (Non-linear Importance)")

        target_enc = self._encode_target()

        # Sampling (MI yavaş olabilir)
        n = len(self.train)
        max_s = self.cfg.SAMPLE_SIZE_FOR_MI
        if n > max_s:
            idx = np.random.choice(n, max_s, replace=False)
            r.add(f"  ℹ️ {n:,} → {max_s:,} sample alındı (hız için)")
        else:
            idx = np.arange(n)

        # Encode categoricals
        X = self.train[self.feature_cols].iloc[idx].copy()
        for col in self.cat_cols:
            if X[col].dtype == 'object':
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X = X.fillna(-999)
        y = target_enc[idx]

        try:
            if self.problem_type == 'regression':
                mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
            else:
                mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)

            mi_df = pd.Series(mi, index=self.feature_cols).sort_values(ascending=False)

            r.add(f"\n  {'Rank':<6} {'Feature':<25} {'MI Score':>10}")
            r.add(f"  {'-'*45}")
            for i, (feat, score) in enumerate(mi_df.head(self.cfg.TOP_N_FEATURES).items(), 1):
                r.add(f"  {i:<6} {feat:<25} {score:>10.4f}")

            r.add(f"\n  📉 EN DÜŞÜK 5 (drop adayı):")
            for feat, score in mi_df.tail(5).items():
                r.add(f"    {feat:<25} {score:.4f}")

        except Exception as e:
            r.add(f"  ⚠️ MI hesaplanamadı: {e}")

    # ================================================================ #
    # S16 – KORELASYON
    # ================================================================ #
    def _s16_correlation(self):
        r = self.rpt
        r.section("16. KORELASYON ANALİZİ")
        if len(self.num_cols) < 2:
            r.add("  ⚠️ Yeterli numerik feature yok")
            return

        target_num = self._encode_target()
        cdf = self.train[self.num_cols].copy()
        cdf['__T__'] = target_num
        cm = cdf.corr()
        tc = cm['__T__'].drop('__T__')

        r.add(f"  🎯 TARGET KORELASYONLARI:")
        r.add(f"  {'Feature':<25} {'Pearson':>10} {'|Pearson|':>10} {'Strength':>12}")
        r.add(f"  {'-'*62}")
        for feat in tc.abs().sort_values(ascending=False).head(self.cfg.TOP_N_FEATURES).index:
            ac = abs(tc[feat])
            rc = tc[feat]
            st = ("V.STRONG" if ac > 0.7 else "STRONG" if ac > 0.5
                  else "MODERATE" if ac > 0.3 else "WEAK" if ac > 0.1 else "NONE")
            r.add(f"  {feat:<25} {rc:>10.4f} {ac:>10.4f} {st:>12}")

    # ================================================================ #
    # S17 – MULTICOLLINEARITY
    # ================================================================ #
    def _s17_multicollinearity(self):
        r = self.rpt
        r.section("17. MULTICOLLINEARITY")
        if len(self.num_cols) < 2:
            r.add("  ⚠️ Yeterli numerik feature yok")
            return

        cm = self.train[self.num_cols].corr()
        pairs = []
        for i in range(len(self.num_cols)):
            for j in range(i+1, len(self.num_cols)):
                c1, c2 = self.num_cols[i], self.num_cols[j]
                c = abs(cm.loc[c1, c2])
                if c > 0.8:
                    pairs.append((c1, c2, c))

        if pairs:
            r.add(f"  🚨 YÜKSEK KORELASYONLU ÇİFTLER (|r| > 0.8):")
            r.add(f"  {'Feature 1':<25} {'Feature 2':<25} {'|Corr|':>8}")
            r.add(f"  {'-'*62}")
            for c1, c2, c in sorted(pairs, key=lambda x: x[2], reverse=True)[:15]:
                r.add(f"  {c1:<25} {c2:<25} {c:>8.4f}")
            r.add(f"\n  💡 Birini drop et veya PCA uygula")
        else:
            r.add(f"  ✅ Yüksek multicollinearity yok")

    # ================================================================ #
    # S18 – FEATURE IMPORTANCE (Heuristic)
    # ================================================================ #
    def _s18_feature_importance(self):
        r = self.rpt
        r.section("18. HEURİSTİK FEATURE IMPORTANCE (Combined)")

        target_num = self._encode_target()
        scores = []

        for col in self.num_cols:
            try:
                vals = self.train[col].fillna(self.train[col].median()).values
                c = abs(self._safe_corr(vals, target_num))
                scores.append((col, c, 'numeric', 'pearson'))
            except Exception:
                scores.append((col, 0.0, 'numeric', 'pearson'))

        for col in self.cat_cols:
            try:
                ct = pd.crosstab(self.train[col], self.train[self.target_col])
                chi2 = stats.chi2_contingency(ct)[0]
                n = len(self.train)
                k = min(ct.shape) - 1
                cv = np.sqrt(chi2 / (n * k)) if k > 0 else 0
                scores.append((col, min(cv, 1.0), 'categorical', 'cramers_v'))
            except Exception:
                scores.append((col, 0.0, 'categorical', 'cramers_v'))

        scores.sort(key=lambda x: x[1], reverse=True)
        r.add(f"  {'Rank':<5} {'Feature':<25} {'Score':>8} {'Type':>12} {'Method':>12}")
        r.add(f"  {'-'*66}")
        for i, (f, s, t, m) in enumerate(scores[:self.cfg.TOP_N_FEATURES], 1):
            r.add(f"  {i:<5} {f:<25} {s:>8.4f} {t:>12} {m:>12}")

    # ================================================================ #
    # S19 – FEATURE INTERACTION
    # ================================================================ #
    def _s19_feature_interaction(self):
        r = self.rpt
        r.section("19. FEATURE INTERACTION SİNYALLERİ")
        if len(self.num_cols) < 2:
            r.add("  ⚠️ Yeterli numerik feature yok")
            return

        target_num = self._encode_target()
        top_k = self.cfg.INTERACTION_TOP_K
        # Önce individual korelasyonlar
        ind_corr = {}
        for col in self.num_cols:
            ind_corr[col] = abs(self._safe_corr(
                self.train[col].fillna(self.train[col].median()).values, target_num))

        top_feats = sorted(ind_corr, key=ind_corr.get, reverse=True)[:top_k]

        r.add(f"  Top-{top_k} feature arasında interaction (çarpım) taraması:\n")
        r.add(f"  {'Pair':<45} {'Ind_Max':>10} {'Inter_Corr':>12} {'Gain':>8}")
        r.add(f"  {'-'*80}")

        interactions = []
        for i in range(len(top_feats)):
            for j in range(i+1, len(top_feats)):
                c1, c2 = top_feats[i], top_feats[j]
                v1 = self.train[c1].fillna(0).values
                v2 = self.train[c2].fillna(0).values
                inter = v1 * v2
                inter_corr = abs(self._safe_corr(inter, target_num))
                ind_max = max(ind_corr[c1], ind_corr[c2])
                gain = inter_corr - ind_max
                interactions.append((f"{c1} × {c2}", ind_max, inter_corr, gain))

        interactions.sort(key=lambda x: x[3], reverse=True)
        for name, im, ic, g in interactions[:10]:
            flag = " ✨" if g > 0.02 else ""
            r.add(f"  {name:<45} {im:>10.4f} {ic:>12.4f} {g:>+7.4f}{flag}")

        pos_gains = [x for x in interactions if x[3] > 0.02]
        if pos_gains:
            r.add(f"\n  💡 {len(pos_gains)} interaction pozitif gain gösteriyor → FE'de kullan")
        else:
            r.add(f"\n  ℹ️ Belirgin interaction gain yok (çarpım bazlı)")

        # Oran bazlı
        r.add(f"\n  Oran (division) taraması:")
        ratio_results = []
        for i in range(len(top_feats)):
            for j in range(i+1, len(top_feats)):
                c1, c2 = top_feats[i], top_feats[j]
                v1 = self.train[c1].fillna(0).values.astype(float)
                v2 = self.train[c2].fillna(0).values.astype(float)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.where(v2 != 0, v1 / v2, 0)
                ratio = np.nan_to_num(ratio, nan=0, posinf=0, neginf=0)
                rc = abs(self._safe_corr(ratio, target_num))
                ind_max = max(ind_corr[c1], ind_corr[c2])
                gain = rc - ind_max
                ratio_results.append((f"{c1}/{c2}", rc, gain))

        ratio_results.sort(key=lambda x: x[2], reverse=True)
        for name, rc, g in ratio_results[:5]:
            flag = " ✨" if g > 0.02 else ""
            r.add(f"    {name:<40} corr={rc:.4f}  gain={g:+.4f}{flag}")

    # ================================================================ #
    # S20 – ROW-WISE STATISTICS
    # ================================================================ #
    def _s20_row_wise_stats(self):
        r = self.rpt
        r.section("20. ROW-WISE İSTATİSTİKLER (Meta-Features)")
        if len(self.num_cols) < 3:
            r.add("  ℹ️ Yeterli numerik feature yok (<3)")
            return

        target_num = self._encode_target()
        num_data = self.train[self.num_cols].fillna(0)

        meta = pd.DataFrame({
            'row_mean': num_data.mean(axis=1),
            'row_std': num_data.std(axis=1),
            'row_min': num_data.min(axis=1),
            'row_max': num_data.max(axis=1),
            'row_sum': num_data.sum(axis=1),
            'row_range': num_data.max(axis=1) - num_data.min(axis=1),
            'row_zeros': (num_data == 0).sum(axis=1),
            'row_nulls': self.train[self.num_cols].isnull().sum(axis=1),
        })

        r.add(f"  {'Meta-Feature':<20} {'Corr w/ Target':>15} {'Abs':>8} {'Useful':>8}")
        r.add(f"  {'-'*55}")
        for col in meta.columns:
            c = self._safe_corr(meta[col].values, target_num)
            ac = abs(c)
            useful = "✅ YES" if ac > 0.1 else "❌ NO"
            r.add(f"  {col:<20} {c:>15.4f} {ac:>8.4f} {useful:>8}")

        r.add(f"\n  💡 |corr| > 0.1 olanları feature olarak ekle!")

    # ================================================================ #
    # S21 – OUTLIER IMPACT ON TARGET
    # ================================================================ #
    def _s21_outlier_impact(self):
        r = self.rpt
        r.section("21. OUTLIER ETKİSİ (Target Üzerine)")
        if not self.num_cols:
            r.add("  ℹ️ Numerik feature yok")
            return

        target_enc = self._encode_target()

        r.add(f"  {'Feature':<25} {'Out%':>7} {'Target(norm)':>13} {'Target(out)':>13} {'Diff':>8}")
        r.add(f"  {'-'*70}")

        for col in self.num_cols[:self.cfg.TOP_N_FEATURES]:
            s = self.train[col]
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR = Q3 - Q1
            mask = (s < Q1-1.5*IQR) | (s > Q3+1.5*IQR)
            out_pct = mask.mean()

            if mask.sum() < 10 or (~mask).sum() < 10:
                continue

            t_norm = target_enc[~mask].mean()
            t_out = target_enc[mask].mean()
            diff = t_out - t_norm

            flag = " ⚠️" if abs(diff) > 0.1 * abs(t_norm + 1e-9) else ""
            r.add(f"  {col:<25} {out_pct:>6.1%} {t_norm:>13.4f} {t_out:>13.4f} {diff:>+7.4f}{flag}")

    # ================================================================ #
    # S22 – INDEX / ORDER TREND
    # ================================================================ #
    def _s22_index_trend(self):
        r = self.rpt
        r.section("22. INDEX / SIRALAMA TREND ANALİZİ")

        target_enc = self._encode_target().astype(float)
        idx = np.arange(len(self.train))

        # Target vs index korelasyonu
        tc = self._safe_corr(idx, target_enc)
        r.add(f"  Target ↔ Row Index korelasyonu: {tc:.4f}")
        if abs(tc) > 0.05:
            r.add(f"  ⚠️ Target'ta sıralama trendi var → temporal leak olabilir!")
            r.add(f"     → TimeSeriesSplit veya ordering-aware CV düşün")
        else:
            r.add(f"  ✅ Belirgin sıralama trendi yok")

        # Feature vs index
        r.add(f"\n  Feature ↔ Row Index korelasyonları:")
        feat_trends = []
        for col in self.num_cols:
            vals = self.train[col].fillna(0).values
            c = abs(self._safe_corr(idx, vals))
            if c > 0.05:
                feat_trends.append((col, c))

        if feat_trends:
            feat_trends.sort(key=lambda x: x[1], reverse=True)
            for col, c in feat_trends[:10]:
                r.add(f"    {col:<25} |corr|={c:.4f} ⚠️")
        else:
            r.add(f"    ✅ Feature'larda sıralama trendi yok")

        # Rolling target mean (ilk vs son çeyrek)
        n = len(self.train)
        q1_mean = target_enc[:n//4].mean()
        q4_mean = target_enc[3*n//4:].mean()
        r.add(f"\n  Target ortalaması: İlk çeyrek={q1_mean:.4f}  Son çeyrek={q4_mean:.4f}")
        if abs(q1_mean - q4_mean) > 0.05 * (abs(q1_mean) + 1e-9):
            r.add(f"  ⚠️ Temporal shift var!")

    # ================================================================ #
    # S23 – TRAIN-TEST DRIFT
    # ================================================================ #
    def _s23_drift(self):
        r = self.rpt
        r.section("23. TRAIN-TEST DRIFT ANALİZİ")

        # Kategorik
        cat_drift = []
        for col in self.cat_cols:
            if col not in self.test.columns:
                continue
            tr = self.train[col].value_counts(normalize=True)
            te = self.test[col].value_counts(normalize=True)
            vals = set(tr.index) | set(te.index)
            md = max(abs(tr.get(v, 0) - te.get(v, 0)) for v in vals)
            if md > self.cfg.DRIFT_THRESHOLD:
                cat_drift.append((col, md))

        if cat_drift:
            r.add(f"  🚨 KATEGORİK DRIFT ({len(cat_drift)}):")
            for col, d in sorted(cat_drift, key=lambda x: x[1], reverse=True):
                r.add(f"    {col:<25} max_diff={d:.1%}")
        else:
            r.add(f"  ✅ Kategorik drift yok")

        # Numerik (KS test)
        num_drift = []
        for col in self.num_cols:
            if col not in self.test.columns:
                continue
            try:
                ks, p = stats.ks_2samp(
                    self.train[col].dropna().values, self.test[col].dropna().values)
                md = abs(self.train[col].mean()-self.test[col].mean()) / (abs(self.train[col].mean())+1e-9)
                if ks > 0.05 or p < 0.01:
                    num_drift.append((col, ks, p, md))
            except Exception:
                pass

        if num_drift:
            r.add(f"\n  🚨 NUMERİK DRIFT ({len(num_drift)}):")
            r.add(f"  {'Feature':<25} {'KS':>8} {'p-val':>12} {'MeanDiff%':>10}")
            r.add(f"  {'-'*60}")
            for col, ks, p, md in sorted(num_drift, key=lambda x: x[1], reverse=True)[:15]:
                r.add(f"  {col:<25} {ks:>8.4f} {p:>12.2e} {md:>9.2%}")
        else:
            r.add(f"\n  ✅ Numerik drift yok")

    # ================================================================ #
    # S24 – TRAIN-TEST OVERLAP
    # ================================================================ #
    def _s24_overlap(self):
        r = self.rpt
        r.section("24. TRAIN-TEST OVERLAP")
        cf = [c for c in self.feature_cols if c in self.test.columns]
        if not cf:
            r.add("  ⚠️ Ortak feature yok")
            return
        try:
            merged = pd.merge(self.train[cf], self.test[cf], on=cf, how='inner')
            oc = len(merged)
            r.add(f"  Feature-bazlı eşleşme: {oc:,}")
            r.add(f"    Train'in  %{oc/len(self.train)*100:.2f}'i test'te")
            r.add(f"    Test'in   %{oc/len(self.test)*100:.2f}'si train'de")
            if oc > 0:
                r.add(f"  ⚠️ Overlap → pseudo-labeling fırsatı veya leak riski")
            else:
                r.add(f"  ✅ Overlap yok")
        except Exception as e:
            r.add(f"  ⚠️ Hesaplanamadı (memory?): {e}")

    # ================================================================ #
    # S25 – ADVERSARIAL VALIDATION PROXY
    # ================================================================ #
    def _s25_adversarial_proxy(self):
        r = self.rpt
        r.section("25. ADVERSARIAL VALIDATION PROXY (Model-Free)")
        r.add("  Train vs Test ayırt edilebilirlik skoru (distribution-based):\n")

        cf = [c for c in self.num_cols if c in self.test.columns]
        if not cf:
            r.add("  ⚠️ Ortak numerik feature yok")
            return

        ks_scores = []
        for col in cf:
            try:
                ks, p = stats.ks_2samp(
                    self.train[col].dropna().values,
                    self.test[col].dropna().values)
                ks_scores.append((col, ks, p))
            except Exception:
                pass

        if not ks_scores:
            r.add("  ⚠️ KS testi hesaplanamadı")
            return

        avg_ks = np.mean([x[1] for x in ks_scores])
        max_ks = max(ks_scores, key=lambda x: x[1])

        r.add(f"  Ortalama KS statistic: {avg_ks:.4f}")
        r.add(f"  Maksimum KS: {max_ks[0]} ({max_ks[1]:.4f})")
        r.add()

        if avg_ks < 0.02:
            r.add(f"  ✅ Dağılımlar ÇOK BENZERdir (adv. AUC ≈ 0.50)")
        elif avg_ks < 0.05:
            r.add(f"  ✅ Dağılımlar benzer (adv. AUC ≈ 0.50-0.55)")
        elif avg_ks < 0.10:
            r.add(f"  ⚠️ Hafif drift (adv. AUC ≈ 0.55-0.65)")
        else:
            r.add(f"  🚨 CİDDİ DRIFT (adv. AUC > 0.65 beklenir)")
            r.add(f"     → Gerçek adversarial validation modeli çalıştır")

        r.add(f"\n  Feature bazlı KS sıralaması:")
        r.add(f"  {'Feature':<25} {'KS':>8} {'p-val':>12}")
        r.add(f"  {'-'*48}")
        for col, ks, p in sorted(ks_scores, key=lambda x: x[1], reverse=True)[:10]:
            r.add(f"  {col:<25} {ks:>8.4f} {p:>12.2e}")

    # ================================================================ #
    # S26 – DATA LEAKAGE
    # ================================================================ #
    def _s26_leakage(self):
        r = self.rpt
        r.section("26. DATA LEAKAGE TESPİTİ")
        suspects = []

        for col in self.num_cols:
            if self.train[col].nunique() == len(self.train):
                suspects.append((col, "UNIQUE PER ROW → olası ID"))

        target_num = self._encode_target()
        for col in self.num_cols:
            try:
                c = abs(self._safe_corr(self.train[col].fillna(0).values, target_num))
                if c > 0.95:
                    suspects.append((col, f"TARGET ile mükemmel corr ({c:.4f}) 🚨"))
            except Exception:
                pass

        if len(self.num_cols) > 1:
            for i in range(len(self.num_cols)):
                for j in range(i+1, len(self.num_cols)):
                    c1, c2 = self.num_cols[i], self.num_cols[j]
                    try:
                        c = abs(self.train[c1].corr(self.train[c2]))
                        if c > 0.98:
                            suspects.append((f"{c1}↔{c2}", f"Near-perfect corr ({c:.4f})"))
                    except Exception:
                        pass

        for col in self.feature_cols:
            if col not in self.test.columns:
                suspects.append((col, "Train'de var TEST'te YOK → leak?"))

        if suspects:
            r.add(f"  🚨 POTANSİYEL LEAKAGE ({len(suspects)}):")
            for item, reason in suspects:
                r.add(f"    {item}: {reason}")
        else:
            r.add(f"  ✅ Belirgin leakage yok")

    # ================================================================ #
    # S27 – FE FİKİRLERİ
    # ================================================================ #
    def _s27_fe_ideas(self):
        r = self.rpt
        r.section("27. FEATURE ENGINEERING FİKİRLERİ")
        ideas = []

        if len(self.num_cols) >= 2:
            nc = len(self.num_cols)
            ideas.append(f"🔢 Pairwise interactions: ×, ÷, −, + ({nc}C2 = {nc*(nc-1)//2} çift)")

        pos_sk = [c for c in self.num_cols
                   if self.train[c].skew() > 1 and (self.train[c] > 0).all()]
        if pos_sk:
            ideas.append(f"📐 Log transform: {', '.join(pos_sk[:5])}")

        neg_sk = [c for c in self.num_cols if self.train[c].skew() < -1]
        if neg_sk:
            ideas.append(f"📐 Square/exp transform (neg-skew): {', '.join(neg_sk[:5])}")

        if self.cat_cols:
            ideas.append(f"🏷️ Target encoding ({len(self.cat_cols)} cat feature)")
            ideas.append(f"🔢 Frequency/count encoding")

        if len(self.cat_cols) >= 2:
            ideas.append(f"🔗 Kategorik concat: {len(self.cat_cols)}C2 çift")

        if self.cat_cols and self.num_cols:
            ideas.append(f"📊 GroupBy agg: cat×num → mean/std/min/max/median")

        ideas.append(f"📦 Quantile binning (numerik → kategorik)")

        miss = [c for c in self.feature_cols if self.train[c].isnull().sum() > 0]
        if miss:
            ideas.append(f"❌ Missing indicator flags: {', '.join(miss[:5])}")

        if len(self.num_cols) >= 3:
            ideas.append(f"📊 Row-wise: mean, std, min, max, sum, range, zeros")

        ideas.append(f"🎯 Polynomial features (degree=2) top features için")
        ideas.append(f"📉 PCA/UMAP bileşenleri ek feature olarak")
        ideas.append(f"🔢 Rank transformation (sıralama bazlı)")

        r.add(f"  💡 ÖNERİLER:")
        for i, idea in enumerate(ideas, 1):
            r.add(f"    {i:>2}. {idea}")

    # ================================================================ #
    # S28 – CV STRATEJİSİ
    # ================================================================ #
    def _s28_cv_strategy(self):
        r = self.rpt
        r.section("28. CROSS-VALIDATION STRATEJİSİ")
        n = len(self.train)

        if self.problem_type in ['binary', 'multiclass']:
            vc = self.train[self.target_col].value_counts()
            mc = vc.min()
            if mc < 50:
                r.add(f"  ⚠️ Min class={mc} → RepeatedStratifiedKFold(5, repeats=3)")
            elif n < 10000:
                r.add(f"  → StratifiedKFold(n_splits=10, shuffle=True)")
            else:
                r.add(f"  → StratifiedKFold(n_splits=5, shuffle=True)")
        else:
            r.add(f"  → KFold(n_splits=5, shuffle=True)")
            if n < 5000:
                r.add(f"  ℹ️ Küçük dataset → RepeatedKFold düşün")

        r.add(f"\n  🎯 METRİK:")
        if self.problem_type == 'binary':
            r.add(f"    Primary  : AUC-ROC, Log Loss")
            r.add(f"    Secondary: F1, Precision, Recall, PR-AUC")
        elif self.problem_type == 'multiclass':
            r.add(f"    Primary  : Macro F1, Log Loss")
            r.add(f"    Secondary: Accuracy, Weighted F1, Cohen Kappa")
        else:
            r.add(f"    Primary  : RMSE, MAE")
            r.add(f"    Secondary: R², RMSLE (pozitif target ise), MAPE")

        r.add(f"\n  ⚠️ Yarışma metriğini kontrol et! Farklıysa ona göre optimize et.")
        r.add(f"\n  🔀 ADVERSARIAL VALIDATION:")
        r.add(f"    AUC ≈ 0.5 → iyi")
        r.add(f"    AUC > 0.7 → drift var, weight veya adversarial fold")

    # ================================================================ #
    # S29 – MODELLEME ÖNERİLERİ
    # ================================================================ #
    def _s29_recommendations(self):
        r = self.rpt
        r.section("29. MODELLEME ÖNERİLERİ")

        recs = []

        if self.problem_type != 'regression':
            vc = self.train[self.target_col].value_counts(normalize=True)
            imb = vc.max() / vc.min()
            if imb > 10:
                recs.append("⚖️ class_weight='balanced' / focal loss / SMOTE+Tomek")
            elif imb > 3:
                recs.append("⚖️ scale_pos_weight / is_unbalance")

        hc = [c for c in self.cat_cols if self.train[c].nunique() > self.cfg.HIGH_CARDINALITY_THRESHOLD]
        if hc:
            recs.append(f"🏷️ High cardinality → CatBoost native veya target encoding: {hc[:3]}")

        sk = [c for c in self.num_cols if abs(self.train[c].skew()) > 2]
        if sk:
            recs.append(f"📐 Skewed → transform: {sk[:3]}")

        recs.append("")
        recs.append("🤖 MODEL STACK ÖNERİSİ:")
        if self.problem_type == 'binary':
            recs.append("   L0: LightGBM, XGBoost, CatBoost, ExtraTrees")
            recs.append("   L1: Logistic Regression (stacker)")
        elif self.problem_type == 'multiclass':
            recs.append("   L0: LightGBM, XGBoost, CatBoost")
            recs.append("   L1: Ridge / LR (stacker)")
        else:
            recs.append("   L0: LightGBM, XGBoost, CatBoost, Ridge")
            recs.append("   L1: Linear blend (stacker)")

        recs.append("")
        recs.append("🏆 GRANDMASTER TACTICS:")
        recs.append("   • Seed averaging (3-5 seed)")
        recs.append("   • Pseudo-labeling (confidence > 0.95)")
        recs.append("   • Target encoding with regularization (5-fold)")
        recs.append("   • Post-processing: threshold tuning, rounding")
        recs.append("   • Null importance feature selection")
        recs.append("   • Bayesian hyperparameter tuning (Optuna)")
        recs.append("   • Blend en az 3 farklı model ailesi")

        for rec in recs:
            r.add(f"  {rec}")

    # ================================================================ #
    # S30 – FİNAL CHECKLIST
    # ================================================================ #
    def _s30_checklist(self):
        r = self.rpt
        r.section("30. PRE-MODELING CHECKLIST")

        checks = []

        # Missing
        miss = sum(1 for c in self.feature_cols if self.train[c].isnull().sum() > 0)
        checks.append(("Eksik veri işlendi mi?",
                        "✅ Yok" if miss == 0 else f"❌ {miss} kolon eksik"))

        # Constant
        const = sum(1 for c in self.feature_cols if self.train[c].nunique() <= 1)
        checks.append(("Constant feature drop edildi mi?",
                        "✅ Yok" if const == 0 else f"❌ {const} constant var"))

        # Duplicates
        dups = self.train[self.feature_cols].duplicated().sum()
        checks.append(("Duplicate'lar incelendi mi?",
                        "✅ Yok" if dups == 0 else f"⚠️ {dups:,} duplicate"))

        # Leakage
        checks.append(("Leakage kontrolü yapıldı mı?", "✅ Bölüm 26'ya bak"))

        # Encoding
        checks.append(("Kategorik encoding planlandı mı?",
                        f"{'✅ Cat yok' if not self.cat_cols else f'⚠️ {len(self.cat_cols)} cat feature'}"))

        # Drift
        checks.append(("Train-test drift kontrol edildi mi?", "✅ Bölüm 23+25'e bak"))

        # CV
        checks.append(("CV stratejisi belirlendi mi?", "✅ Bölüm 28'e bak"))

        # Target
        if self.problem_type != 'regression':
            vc = self.train[self.target_col].value_counts(normalize=True)
            imb = vc.max() / vc.min()
            checks.append(("Imbalance ele alındı mı?",
                            f"{'✅ Balanced' if imb < 3 else f'⚠️ {imb:.1f}:1 ratio'}"))

        # Skewness
        sk = sum(1 for c in self.num_cols if abs(self.train[c].skew()) > 2)
        checks.append(("Skewed features transform edildi mi?",
                        "✅ Yok" if sk == 0 else f"⚠️ {sk} skewed feature"))

        # Sample submission
        checks.append(("Submission formatı doğru mu?",
                        "✅ Bölüm 3'e bak" if self.sample_sub is not None else "⚠️ sample_sub yok"))

        # Original data
        checks.append(("Orijinal dataset denendi mi?",
                        "⚠️ Playground Series ise orijinal veriyi ekle"))

        r.add(f"  {'#':<4} {'Kontrol':<45} {'Durum':>30}")
        r.add(f"  {'-'*82}")
        for i, (check, status) in enumerate(checks, 1):
            r.add(f"  {i:<4} {check:<45} {status:>30}")

        r.add(f"\n{'='*80}")
        r.add(f"{'🏆 EDA RAPORU TAMAMLANDI - 30 BÖLÜM':^80}")
        r.add(f"{'='*80}")


# =============================================================================
# 5. ÇALIŞTIRMA
# =============================================================================

def run_eda():
    config = EDAConfig()
    eda = GrandmasterEDA(config)
    eda.load_data()
    report = eda.run()
    report.save("grandmaster_eda_report.txt")
    return eda

if __name__ == "__main__":
    eda_obj = run_eda()
```

# 1. Öneri KAGGLE GRANDMASTER EDA


```py
# =============================================================================
# KAGGLE GRANDMASTER EDA - KAPSAMLI YAZILI RAPOR
# Sadece EDA, model yok. Tüm bilgiler metin olarak çıktı verir.
# =============================================================================

import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
from scipy import stats
warnings.filterwarnings('ignore')

# =============================================================================
# 1. KONFİGÜRASYON
# =============================================================================

class EDAConfig:
    """EDA Konfigürasyonu - Sadece burayı değiştir"""
    BASE_PATH = "/kaggle/input/competitions/playground-series-s6e4/"
    TRAIN_FILE = "train.csv"
    TEST_FILE = "test.csv"

    TARGET_COL = None      # None = otomatik algıla
    ID_COL = None          # None = otomatik algıla

    CAT_THRESHOLD = 10
    HIGH_CARDINALITY_THRESHOLD = 100
    DRIFT_THRESHOLD = 0.05
    MISSING_THRESHOLD = 0.5

    TOP_N_FEATURES = 20
    TOP_N_CATEGORIES = 10
    BIN_COUNT = 10

    # Regression tespiti: target nunique > bu değer ise regression
    REGRESSION_NUNIQUE_THRESHOLD = 30


# =============================================================================
# 2. OTOMATİK ALGILAMA
# =============================================================================

class AutoDetector:
    @staticmethod
    def detect_target(train_df, config):
        if config.TARGET_COL and config.TARGET_COL in train_df.columns:
            return config.TARGET_COL
        candidates = [
            'target', 'Target', 'TARGET', 'label', 'Label',
            'class', 'Class', 'y', 'Y',
            'irrigation_need', 'Irrigation_Need', 'target_col',
            'Survived', 'survived', 'SalePrice', 'price'
        ]
        for col in candidates:
            if col in train_df.columns:
                return col
        cols = [c for c in train_df.columns if c.lower() not in ['id']]
        return cols[-1] if cols else None

    @staticmethod
    def detect_id(df, config):
        if config.ID_COL and config.ID_COL in df.columns:
            return config.ID_COL
        for col in ['id', 'ID', 'Id', 'index', 'sample_id', 'row_id']:
            if col in df.columns:
                return col
        return None

    @staticmethod
    def detect_problem_type(train_df, target_col, config):
        dtype = train_df[target_col].dtype
        n_unique = train_df[target_col].nunique()

        # Float ve çok fazla unique → regression
        if dtype in ['float64', 'float32']:
            if n_unique > config.REGRESSION_NUNIQUE_THRESHOLD:
                return 'regression'

        # Integer ama çok fazla unique → regression
        if dtype in ['int64', 'int32'] and n_unique > config.REGRESSION_NUNIQUE_THRESHOLD:
            return 'regression'

        if n_unique == 2:
            return 'binary'
        elif n_unique > 2:
            return 'multiclass'
        return 'regression'

    @staticmethod
    def detect_features(train_df, feature_cols, threshold):
        cats, nums = [], []
        for col in feature_cols:
            unique_count = train_df[col].nunique()
            dtype = train_df[col].dtype
            if dtype == 'object' or dtype.name == 'category' or unique_count <= threshold:
                cats.append(col)
            else:
                nums.append(col)
        return cats, nums


# =============================================================================
# 3. EDA RAPOR SINIFI (isim düzeltildi)
# =============================================================================

class EDAReport:
    def __init__(self, config):
        self.config = config
        self.lines = []

    def add(self, text):
        self.lines.append(text)

    def add_section(self, title, char="="):
        self.lines.append(f"\n{char * 80}")
        self.lines.append(f"{title:^80}")
        self.lines.append(f"{char * 80}\n")

    def add_subsection(self, title):
        self.lines.append(f"\n{'─' * 80}")
        self.lines.append(f"{title}")
        self.lines.append(f"{'─' * 80}\n")

    def print_report(self):
        print("\n".join(self.lines))

    def save(self, filename="eda_report.txt"):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.lines))
        print(f"\n💾 Rapor kaydedildi: {filename}")


# =============================================================================
# 4. ANA EDA SINIFI
# =============================================================================

class GrandmasterEDA:
    def __init__(self, config):
        self.config = config
        self.report = EDAReport(config)
        self.train = None
        self.test = None
        self.target_col = None
        self.id_col = None
        self.problem_type = None
        self.cat_cols = []
        self.num_cols = []
        self.feature_cols = []

    # --------------------------------------------------------------------- #
    # VERİ YÜKLEME
    # --------------------------------------------------------------------- #
    def load_data(self):
        train_path = self.config.BASE_PATH + self.config.TRAIN_FILE
        test_path = self.config.BASE_PATH + self.config.TEST_FILE

        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)

        self.target_col = AutoDetector.detect_target(self.train, self.config)
        self.id_col = AutoDetector.detect_id(self.train, self.config)
        self.problem_type = AutoDetector.detect_problem_type(
            self.train, self.target_col, self.config
        )

        self.feature_cols = [
            c for c in self.train.columns if c not in [self.target_col, self.id_col]
        ]
        self.cat_cols, self.num_cols = AutoDetector.detect_features(
            self.train, self.feature_cols, self.config.CAT_THRESHOLD
        )

    # --------------------------------------------------------------------- #
    # _safe_encode: Hedefi sayısala çevirme yardımcısı
    # --------------------------------------------------------------------- #
    def _encode_target(self):
        if self.train[self.target_col].dtype == 'object':
            le = LabelEncoder()
            return le.fit_transform(self.train[self.target_col])
        return self.train[self.target_col].values

    # --------------------------------------------------------------------- #
    # TAM ANALİZ
    # --------------------------------------------------------------------- #
    def run_full_analysis(self):
        self.report.add_section("KAGGLE GRANDMASTER EDA REPORT", "=")
        self.report.add(f"  Timestamp   : {pd.Timestamp.now()}")
        self.report.add(f"  Target      : {self.target_col}")
        self.report.add(f"  Problem Type: {self.problem_type.upper()}")

        self._section_basic_info()          # 1
        self._section_memory()              # 2
        self._section_target()              # 3
        self._section_missing()             # 4
        self._section_duplicates()          # 5
        self._section_constant_features()   # 6
        self._section_categorical()         # 7
        self._section_numerical()           # 8
        self._section_per_class_stats()     # 9
        self._section_binned_target_rate()  # 10
        self._section_drift()               # 11
        self._section_train_test_overlap()  # 12
        self._section_correlation()         # 13
        self._section_feature_importance()  # 14
        self._section_leakage()             # 15
        self._section_fe_ideas()            # 16
        self._section_cv_strategy()         # 17
        self._section_recommendations()     # 18

        self.report.print_report()
        return self.report

    # ================================================================== #
    # SECTION 1 – TEMEL BİLGİLER
    # ================================================================== #
    def _section_basic_info(self):
        self.report.add_section("1. TEMEL VERİ BİLGİLERİ")

        self.report.add(f"  SHAPE:")
        self.report.add(f"    Train : {self.train.shape[0]:>10,} rows  × {self.train.shape[1]:>4} cols")
        self.report.add(f"    Test  : {self.test.shape[0]:>10,} rows  × {self.test.shape[1]:>4} cols")
        self.report.add(f"    Ratio : {self.test.shape[0] / self.train.shape[0]:.2f}x")
        self.report.add(f"")
        self.report.add(f"  TARGET : {self.target_col} ({self.problem_type})")
        self.report.add(f"  ID     : {self.id_col}")
        self.report.add(f"")
        self.report.add(f"  FEATURES ({len(self.feature_cols)}):")
        self.report.add(f"    Categorical : {len(self.cat_cols):>3}  → {', '.join(self.cat_cols[:8])}"
                        f"{'...' if len(self.cat_cols) > 8 else ''}")
        self.report.add(f"    Numeric     : {len(self.num_cols):>3}  → {', '.join(self.num_cols[:8])}"
                        f"{'...' if len(self.num_cols) > 8 else ''}")

        # Dtype detayı
        self.report.add(f"\n  DTYPE BREAKDOWN:")
        for dtype, count in self.train[self.feature_cols].dtypes.value_counts().items():
            self.report.add(f"    {str(dtype):<15} : {count}")

    # ================================================================== #
    # SECTION 2 – BELLEK KULLANIMI
    # ================================================================== #
    def _section_memory(self):
        self.report.add_section("2. BELLEK KULLANIMI")

        train_mem = self.train.memory_usage(deep=True).sum() / 1024 ** 2
        test_mem = self.test.memory_usage(deep=True).sum() / 1024 ** 2

        self.report.add(f"  Train : {train_mem:>8.2f} MB")
        self.report.add(f"  Test  : {test_mem:>8.2f} MB")
        self.report.add(f"  Total : {train_mem + test_mem:>8.2f} MB")

        # Optimizasyon önerileri
        opt_lines = []
        for col in self.feature_cols:
            dtype = self.train[col].dtype
            if dtype == 'float64':
                mn, mx = self.train[col].min(), self.train[col].max()
                if mn >= np.finfo(np.float32).min and mx <= np.finfo(np.float32).max:
                    opt_lines.append(col)
            elif dtype == 'int64':
                mn, mx = self.train[col].min(), self.train[col].max()
                if mn >= np.iinfo(np.int16).min and mx <= np.iinfo(np.int16).max:
                    opt_lines.append(col)
        if opt_lines:
            self.report.add(f"\n  ⚡ {len(opt_lines)} kolon daha küçük dtype'a cast edilebilir:")
            for c in opt_lines[:10]:
                self.report.add(f"      {c} ({self.train[c].dtype})")
            if len(opt_lines) > 10:
                self.report.add(f"      ... ve {len(opt_lines) - 10} daha")
        else:
            self.report.add(f"\n  ✅ Dtype optimizasyonu gerekmiyor")

    # ================================================================== #
    # SECTION 3 – HEDEF DEĞİŞKEN
    # ================================================================== #
    def _section_target(self):
        self.report.add_section("3. HEDEF DEĞİŞKEN ANALİZİ")

        if self.problem_type == 'regression':
            desc = self.train[self.target_col].describe()
            self.report.add(f"  {'Metric':<12} {'Value':>15}")
            self.report.add(f"  {'-' * 30}")
            for idx in desc.index:
                self.report.add(f"  {idx:<12} {desc[idx]:>15.4f}")
            skew = self.train[self.target_col].skew()
            kurt = self.train[self.target_col].kurtosis()
            self.report.add(f"  {'skew':<12} {skew:>15.4f}")
            self.report.add(f"  {'kurtosis':<12} {kurt:>15.4f}")
            if abs(skew) > 1:
                self.report.add(f"\n  ⚠️ Target SKEWED (skew={skew:.2f}). Log-transform veya Box-Cox düşün.")
            zero_pct = (self.train[self.target_col] == 0).mean()
            neg_pct = (self.train[self.target_col] < 0).mean()
            self.report.add(f"\n  Zero values : {zero_pct:.2%}")
            self.report.add(f"  Negative    : {neg_pct:.2%}")
            return

        # Classification
        vc = self.train[self.target_col].value_counts().sort_index()
        vr = self.train[self.target_col].value_counts(normalize=True).sort_index()

        self.report.add(f"  {'Class':<20} {'Count':>10} {'Ratio':>10}")
        self.report.add(f"  {'-' * 45}")
        for val in vc.index:
            self.report.add(f"  {str(val):<20} {vc[val]:>10,} {vr[val]:>9.2%}")

        max_r = vr.max()
        min_r = vr.min()
        imb = max_r / min_r

        self.report.add(f"\n  IMBALANCE:")
        self.report.add(f"    Dominant : {max_r:.2%}")
        self.report.add(f"    Minority : {min_r:.2%}")
        self.report.add(f"    Ratio    : {imb:.1f}:1")

        if imb > 10:
            self.report.add(f"    🚨 HIGHLY IMBALANCED → stratified split + class weights + SMOTE/ADASYN")
        elif imb > 3:
            self.report.add(f"    ⚠️ MODERATELY IMBALANCED → stratified CV + class weights")
        else:
            self.report.add(f"    ✅ BALANCED")

    # ================================================================== #
    # SECTION 4 – EKSİK VERİ
    # ================================================================== #
    def _section_missing(self):
        self.report.add_section("4. EKSİK VERİ ANALİZİ")

        tr_miss = self.train.isnull().sum()
        te_miss = self.test.isnull().sum()
        missing_cols = tr_miss[tr_miss > 0].index.tolist()
        te_only = [c for c in te_miss[te_miss > 0].index if c not in missing_cols and c != self.target_col]

        if not missing_cols and not te_only:
            self.report.add("  ✅ Hiçbir kolonda eksik veri yok")
            return

        all_missing = list(set(missing_cols + te_only))
        self.report.add(f"  Eksik veri olan kolon sayısı: {len(all_missing)}")
        self.report.add(f"")
        self.report.add(f"  {'Column':<25} {'Train':>8} {'Train%':>8} {'Test':>8} {'Test%':>8} {'Durum':>15}")
        self.report.add(f"  {'-' * 78}")

        for col in sorted(all_missing):
            tr_n = tr_miss.get(col, 0)
            tr_p = tr_n / len(self.train) * 100
            te_n = te_miss.get(col, 0)
            te_p = te_n / len(self.test) * 100 if col in self.test.columns else 0

            if tr_p > self.config.MISSING_THRESHOLD * 100:
                status = "DROP"
            elif abs(tr_p - te_p) > 10:
                status = "DRIFT!"
            else:
                status = "IMPUTE"
            self.report.add(f"  {col:<25} {tr_n:>8,} {tr_p:>7.2f}% {te_n:>8,} {te_p:>7.2f}% {status:>15}")

        # Eksik veri pattern
        missing_pattern = self.train[missing_cols].isnull()
        rows_any = missing_pattern.any(axis=1).sum()
        rows_all = missing_pattern.all(axis=1).sum()
        self.report.add(f"\n  En az 1 eksik olan satır : {rows_any:,} ({rows_any / len(self.train):.1%})")
        self.report.add(f"  Tüm eksik kolonlar dolu  : {rows_all:,}")

    # ================================================================== #
    # SECTION 5 – DUPLICATE ANALİZİ
    # ================================================================== #
    def _section_duplicates(self):
        self.report.add_section("5. DUPLICATE ANALİZİ")

        # Tam satır duplikasyonu (target dahil)
        dup_all = self.train.duplicated().sum()
        self.report.add(f"  Train - tam satır duplicate  : {dup_all:>8,} ({dup_all / len(self.train):.2%})")

        # Feature-only duplikasyonu
        feat_dup = self.train[self.feature_cols].duplicated().sum()
        self.report.add(f"  Train - feature-only dup     : {feat_dup:>8,} ({feat_dup / len(self.train):.2%})")

        test_feat = [c for c in self.feature_cols if c in self.test.columns]
        te_dup = self.test[test_feat].duplicated().sum()
        self.report.add(f"  Test  - feature-only dup     : {te_dup:>8,} ({te_dup / len(self.test):.2%})")

        if feat_dup > 0:
            # Aynı feature'lara farklı target verilmiş mi?
            dup_mask = self.train[self.feature_cols].duplicated(keep=False)
            dup_rows = self.train[dup_mask]
            conflicting = dup_rows.groupby(self.feature_cols, dropna=False)[self.target_col].nunique()
            conflict_count = (conflicting > 1).sum()
            self.report.add(f"\n  ⚠️ Aynı feature, farklı target (çelişkili): {conflict_count:,} grup")
            if conflict_count > 0:
                self.report.add(f"     → Noisy labels olabilir, dikkat!")
        else:
            self.report.add(f"\n  ✅ Duplicate sorunu yok")

    # ================================================================== #
    # SECTION 6 – CONSTANT / QUASI-CONSTANT FEATURES
    # ================================================================== #
    def _section_constant_features(self):
        self.report.add_section("6. CONSTANT / QUASI-CONSTANT ÖZELLİKLER")

        constant = []
        quasi = []

        for col in self.feature_cols:
            nuniq = self.train[col].nunique(dropna=False)
            if nuniq <= 1:
                constant.append(col)
            else:
                top_freq = self.train[col].value_counts(normalize=True, dropna=False).iloc[0]
                if top_freq > 0.97:
                    quasi.append((col, top_freq))

        if constant:
            self.report.add(f"  🚨 CONSTANT ({len(constant)}):")
            for c in constant:
                self.report.add(f"     {c}  → DROP (sıfır varyans)")
        else:
            self.report.add(f"  ✅ Constant feature yok")

        if quasi:
            self.report.add(f"\n  ⚠️ QUASI-CONSTANT ({len(quasi)})  (>97% aynı değer):")
            for c, freq in quasi:
                self.report.add(f"     {c}: dominant value {freq:.1%}")
        else:
            self.report.add(f"  ✅ Quasi-constant feature yok")

    # ================================================================== #
    # SECTION 7 – KATEGORİK ANALİZ
    # ================================================================== #
    def _section_categorical(self):
        self.report.add_section("7. KATEGORİK ÖZELLİK ANALİZİ")

        if not self.cat_cols:
            self.report.add("  ℹ️ Kategorik feature yok")
            return

        # Özet tablo
        self.report.add(f"  {'Feature':<25} {'Unique':>8} {'Top Value':<20} {'Top%':>8} {'Cardinality':>12}")
        self.report.add(f"  {'-' * 78}")

        for col in self.cat_cols:
            nuniq = self.train[col].nunique()
            top_val = self.train[col].value_counts().index[0]
            top_pct = self.train[col].value_counts(normalize=True).iloc[0]
            card = "HIGH" if nuniq > self.config.HIGH_CARDINALITY_THRESHOLD else "OK"
            self.report.add(f"  {col:<25} {nuniq:>8} {str(top_val):<20} {top_pct:>7.1%} {card:>12}")

        # Detaylı inceleme
        for col in self.cat_cols[:self.config.TOP_N_CATEGORIES]:
            self.report.add_subsection(f"  📋 {col}")

            vc = self.train[col].value_counts().head(8)
            for val, cnt in vc.items():
                r = cnt / len(self.train)
                bar = "█" * int(r * 40)
                self.report.add(f"    {str(val):<20} {cnt:>8,} ({r:>6.1%}) {bar}")

            # Train'de var test'te yok (ve tersi)
            train_vals = set(self.train[col].dropna().unique())
            test_vals = set(self.test[col].dropna().unique()) if col in self.test.columns else set()
            only_train = train_vals - test_vals
            only_test = test_vals - train_vals
            if only_train:
                self.report.add(f"\n    ⚠️ Sadece train'de ({len(only_train)}): {list(only_train)[:5]}")
            if only_test:
                self.report.add(f"    ⚠️ Sadece test'te  ({len(only_test)}): {list(only_test)[:5]}")

            # Target ilişkisi
            if self.problem_type != 'regression' and self.target_col:
                self.report.add(f"\n    Target ilişkisi:")
                ct = pd.crosstab(self.train[col], self.train[self.target_col], normalize='index')
                for val in ct.index[:5]:
                    row = ct.loc[val]
                    dominant = row.idxmax()
                    self.report.add(f"      {str(val):<20} → {dominant} ({row.max():.1%})")

    # ================================================================== #
    # SECTION 8 – NUMERİK ANALİZ
    # ================================================================== #
    def _section_numerical(self):
        self.report.add_section("8. NUMERİK ÖZELLİK ANALİZİ")

        if not self.num_cols:
            self.report.add("  ℹ️ Numerik feature yok")
            return

        # Özet tablo
        self.report.add(f"  {'Feature':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Skew':>8} {'Out%':>8}")
        self.report.add(f"  {'-' * 82}")

        for col in self.num_cols:
            s = self.train[col]
            mean_ = s.mean()
            std_ = s.std()
            min_ = s.min()
            max_ = s.max()
            skew_ = s.skew()
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR = Q3 - Q1
            out_pct = ((s < Q1 - 1.5 * IQR) | (s > Q3 + 1.5 * IQR)).mean() * 100
            self.report.add(
                f"  {col:<20} {mean_:>10.2f} {std_:>10.2f} {min_:>10.2f} {max_:>10.2f} {skew_:>8.2f} {out_pct:>7.1f}%"
            )

        # Detaylı inceleme
        for col in self.num_cols[:self.config.TOP_N_FEATURES]:
            self.report.add_subsection(f"  🔢 {col}")

            desc = self.train[col].describe()
            self.report.add(f"    Count: {desc['count']:,.0f}  Mean: {desc['mean']:.4f}  Std: {desc['std']:.4f}")
            self.report.add(f"    Min: {desc['min']:.4f}  25%: {desc['25%']:.4f}  "
                            f"50%: {desc['50%']:.4f}  75%: {desc['75%']:.4f}  Max: {desc['max']:.4f}")

            skew = self.train[col].skew()
            kurt = self.train[col].kurtosis()
            self.report.add(f"    Skew: {skew:.3f}  Kurtosis: {kurt:.3f}")

            # Zero / negative
            zero_cnt = (self.train[col] == 0).sum()
            neg_cnt = (self.train[col] < 0).sum()
            self.report.add(f"    Zero: {zero_cnt:,} ({zero_cnt / len(self.train):.1%})  "
                            f"Negative: {neg_cnt:,} ({neg_cnt / len(self.train):.1%})")

            # Outlier
            Q1 = desc['25%']
            Q3 = desc['75%']
            IQR = Q3 - Q1
            outliers = ((self.train[col] < Q1 - 1.5 * IQR) | (self.train[col] > Q3 + 1.5 * IQR)).sum()
            self.report.add(f"    Outliers (IQR): {outliers:,} ({outliers / len(self.train):.1%})")

            if abs(skew) > 2:
                self.report.add(f"    🚨 Highly skewed → log/sqrt/Box-Cox dönüşümü önerilir")
            if outliers / len(self.train) > 0.05:
                self.report.add(f"    ⚠️ %5+ outlier → clip veya winsorize düşün")

            # Histogram benzeri metin çıktısı
            try:
                bins = pd.cut(self.train[col], bins=self.config.BIN_COUNT)
                bin_counts = bins.value_counts().sort_index()
                max_cnt = bin_counts.max()
                self.report.add(f"    Distribution:")
                for interval, cnt in bin_counts.items():
                    bar_len = int(cnt / max_cnt * 30) if max_cnt > 0 else 0
                    bar = "█" * bar_len
                    self.report.add(f"      {str(interval):<30} {cnt:>7,} {bar}")
            except Exception:
                pass

    # ================================================================== #
    # SECTION 9 – SINIF BAZLI İSTATİSTİKLER
    # ================================================================== #
    def _section_per_class_stats(self):
        self.report.add_section("9. SINIF BAZLI NUMERİK İSTATİSTİKLER")

        if self.problem_type == 'regression' or not self.num_cols:
            self.report.add("  ℹ️ Regression problemi veya numerik feature yok – atlanıyor")
            return

        classes = sorted(self.train[self.target_col].unique())
        if len(classes) > 10:
            self.report.add(f"  ℹ️ {len(classes)} sınıf var, sadece ilk 10 gösteriliyor")
            classes = classes[:10]

        for col in self.num_cols[:10]:
            self.report.add_subsection(f"  📊 {col} per class")

            header = f"    {'Class':<15}"
            for stat in ['Mean', 'Std', 'Median']:
                header += f" {stat:>10}"
            self.report.add(header)
            self.report.add(f"    {'-' * 50}")

            for cls in classes:
                subset = self.train.loc[self.train[self.target_col] == cls, col]
                self.report.add(
                    f"    {str(cls):<15} {subset.mean():>10.3f} {subset.std():>10.3f} {subset.median():>10.3f}"
                )

    # ================================================================== #
    # SECTION 10 – BINNED TARGET RATE
    # ================================================================== #
    def _section_binned_target_rate(self):
        self.report.add_section("10. BINNED TARGET RATE (Numerik → Target)")

        if self.problem_type == 'regression' or not self.num_cols:
            self.report.add("  ℹ️ Regression problemi veya numerik feature yok – atlanıyor")
            return

        target_encoded = self._encode_target()
        n_bins = self.config.BIN_COUNT

        for col in self.num_cols[:10]:
            self.report.add_subsection(f"  📈 {col}")

            try:
                bins = pd.qcut(self.train[col], q=n_bins, duplicates='drop')
                df_tmp = pd.DataFrame({'bin': bins, 'target': target_encoded})
                grp = df_tmp.groupby('bin', observed=True)['target'].agg(['mean', 'count'])

                self.report.add(f"    {'Bin':<30} {'Count':>8} {'Target Mean':>12}")
                self.report.add(f"    {'-' * 55}")
                for idx, row in grp.iterrows():
                    self.report.add(f"    {str(idx):<30} {int(row['count']):>8} {row['mean']:>12.4f}")

                # Monotonluk kontrolü
                means = grp['mean'].values
                diffs = np.diff(means)
                if np.all(diffs >= 0) or np.all(diffs <= 0):
                    self.report.add(f"    ✅ MONOTONİK ilişki → güçlü sinyal")
                else:
                    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
                    self.report.add(f"    ℹ️ Non-monotonic ({sign_changes} yön değişimi)")
            except Exception as e:
                self.report.add(f"    ⚠️ Binleme hatası: {e}")

    # ================================================================== #
    # SECTION 11 – TRAIN-TEST DRIFT
    # ================================================================== #
    def _section_drift(self):
        self.report.add_section("11. TRAIN-TEST DRIFT ANALİZİ")

        # --- Kategorik drift ---
        cat_drift = []
        for col in self.cat_cols:
            if col not in self.test.columns:
                continue
            tr_dist = self.train[col].value_counts(normalize=True)
            te_dist = self.test[col].value_counts(normalize=True)
            all_vals = set(tr_dist.index) | set(te_dist.index)
            max_diff = max(abs(tr_dist.get(v, 0) - te_dist.get(v, 0)) for v in all_vals)
            if max_diff > self.config.DRIFT_THRESHOLD:
                cat_drift.append((col, max_diff))

        if cat_drift:
            self.report.add(f"  🚨 KATEGORİK DRIFT ({len(cat_drift)}):")
            self.report.add(f"  {'Column':<25} {'Max Diff':>10}")
            self.report.add(f"  {'-' * 40}")
            for col, d in sorted(cat_drift, key=lambda x: x[1], reverse=True):
                self.report.add(f"  {col:<25} {d:>9.1%}")
        else:
            self.report.add(f"  ✅ Kategorik drift yok")

        # --- Numerik drift (KS test) ---
        self.report.add("")
        num_drift = []
        for col in self.num_cols:
            if col not in self.test.columns:
                continue
            try:
                ks_stat, ks_p = stats.ks_2samp(
                    self.train[col].dropna().values,
                    self.test[col].dropna().values
                )
                mean_diff = abs(self.train[col].mean() - self.test[col].mean()) / (abs(self.train[col].mean()) + 1e-9)
                if ks_p < 0.01 or mean_diff > self.config.DRIFT_THRESHOLD:
                    num_drift.append((col, ks_stat, ks_p, mean_diff))
            except Exception:
                pass

        if num_drift:
            self.report.add(f"  🚨 NUMERİK DRIFT ({len(num_drift)}):")
            self.report.add(f"  {'Column':<25} {'KS Stat':>10} {'KS p-val':>12} {'Mean Diff%':>12}")
            self.report.add(f"  {'-' * 65}")
            for col, ks, p, md in sorted(num_drift, key=lambda x: x[1], reverse=True)[:15]:
                self.report.add(f"  {col:<25} {ks:>10.4f} {p:>12.2e} {md:>11.2%}")
        else:
            self.report.add(f"  ✅ Numerik drift yok")

    # ================================================================== #
    # SECTION 12 – TRAIN-TEST OVERLAP
    # ================================================================== #
    def _section_train_test_overlap(self):
        self.report.add_section("12. TRAIN-TEST OVERLAP")

        common_feats = [c for c in self.feature_cols if c in self.test.columns]
        if not common_feats:
            self.report.add("  ⚠️ Ortak feature yok, overlap kontrol edilemiyor")
            return

        # Feature bazlı overlap
        try:
            merged = pd.merge(
                self.train[common_feats], self.test[common_feats],
                on=common_feats, how='inner'
            )
            overlap_count = len(merged)
            self.report.add(f"  Feature-bazlı tam eşleşme: {overlap_count:,}")
            self.report.add(f"    Train satırlarının %{overlap_count / len(self.train) * 100:.2f}'i test'te mevcut")
            self.report.add(f"    Test  satırlarının %{overlap_count / len(self.test) * 100:.2f}'si train'de mevcut")

            if overlap_count > 0:
                self.report.add(f"    ⚠️ Overlap var! Pseudo-labeling veya leak olabilir.")
            else:
                self.report.add(f"    ✅ Overlap yok")
        except Exception as e:
            self.report.add(f"  ⚠️ Overlap hesaplanamadı: {e}")

    # ================================================================== #
    # SECTION 13 – KORELASYON
    # ================================================================== #
    def _section_correlation(self):
        self.report.add_section("13. KORELASYON ANALİZİ")

        if len(self.num_cols) < 2:
            self.report.add("  ⚠️ Yeterli numerik feature yok")
            return

        target_num = self._encode_target()
        corr_df = self.train[self.num_cols].copy()
        corr_df['__TARGET__'] = target_num
        corr_matrix = corr_df.corr()

        target_corr = corr_matrix['__TARGET__'].drop('__TARGET__').abs().sort_values(ascending=False)

        self.report.add(f"  🎯 TARGET KORELASYONLARI (Top {self.config.TOP_N_FEATURES}):")
        self.report.add(f"  {'Feature':<25} {'|Corr|':>10} {'Direction':>10} {'Strength':>12}")
        self.report.add(f"  {'-' * 62}")

        raw_corr = corr_matrix['__TARGET__'].drop('__TARGET__')
        for feat in target_corr.head(self.config.TOP_N_FEATURES).index:
            ac = target_corr[feat]
            rc = raw_corr[feat]
            direction = "+" if rc > 0 else "-"
            if ac > 0.7:
                strength = "V.STRONG"
            elif ac > 0.5:
                strength = "STRONG"
            elif ac > 0.3:
                strength = "MODERATE"
            elif ac > 0.1:
                strength = "WEAK"
            else:
                strength = "NONE"
            self.report.add(f"  {feat:<25} {ac:>10.4f} {direction:>10} {strength:>12}")

        # Multicollinearity
        self.report.add(f"\n  🔗 MULTICOLLINEARITY (|r| > 0.8):")
        high_pairs = []
        for i in range(len(self.num_cols)):
            for j in range(i + 1, len(self.num_cols)):
                c1, c2 = self.num_cols[i], self.num_cols[j]
                r = abs(corr_matrix.loc[c1, c2])
                if r > 0.8:
                    high_pairs.append((c1, c2, r))

        if high_pairs:
            for c1, c2, r in sorted(high_pairs, key=lambda x: x[2], reverse=True)[:10]:
                self.report.add(f"    {c1} ↔ {c2} : {r:.4f}")
            self.report.add(f"    → Birini drop etmeyi veya PCA uygulamayı düşün")
        else:
            self.report.add(f"    ✅ Yüksek multicollinearity yok")

    # ================================================================== #
    # SECTION 14 – FEATURE IMPORTANCE (Heuristic)
    # ================================================================== #
    def _section_feature_importance(self):
        self.report.add_section("14. HEURİSTİK FEATURE IMPORTANCE")

        target_num = self._encode_target()
        scores = []

        # Numerik: korelasyon
        for col in self.num_cols:
            try:
                vals = self.train[col].fillna(self.train[col].median()).values
                corr = abs(np.corrcoef(vals, target_num)[0, 1])
                if np.isnan(corr):
                    corr = 0.0
                scores.append((col, corr, 'numeric', 'corr'))
            except Exception:
                scores.append((col, 0.0, 'numeric', 'corr'))

        # Kategorik: Cramér's V (chi-square based)
        for col in self.cat_cols:
            try:
                ct = pd.crosstab(self.train[col], self.train[self.target_col])
                chi2 = stats.chi2_contingency(ct)[0]
                n = len(self.train)
                k = min(ct.shape) - 1
                if k > 0 and n > 0:
                    cramers_v = np.sqrt(chi2 / (n * k))
                else:
                    cramers_v = 0.0
                scores.append((col, min(cramers_v, 1.0), 'categorical', 'cramers_v'))
            except Exception:
                scores.append((col, 0.0, 'categorical', 'cramers_v'))

        scores.sort(key=lambda x: x[1], reverse=True)

        self.report.add(f"  {'Rank':<6} {'Feature':<25} {'Score':>10} {'Type':>12} {'Method':>12}")
        self.report.add(f"  {'-' * 70}")

        for i, (feat, score, ftype, method) in enumerate(scores[:self.config.TOP_N_FEATURES], 1):
            self.report.add(f"  {i:<6} {feat:<25} {score:>10.4f} {ftype:>12} {method:>12}")

        # En önemsizler
        bottom = scores[-5:] if len(scores) >= 5 else scores
        self.report.add(f"\n  📉 EN DÜŞÜK 5 ÖZELLİK (drop adayı):")
        for feat, score, ftype, method in bottom:
            self.report.add(f"    {feat:<25} {score:>10.4f}")

    # ================================================================== #
    # SECTION 15 – DATA LEAKAGE
    # ================================================================== #
    def _section_leakage(self):
        self.report.add_section("15. DATA LEAKAGE TESPİTİ")

        suspects = []

        # ID-like
        for col in self.num_cols:
            if self.train[col].nunique() == len(self.train):
                suspects.append((col, "UNIQUE PER ROW → olası ID"))

        # Çok yüksek feature-feature korelasyon
        if len(self.num_cols) > 1:
            for i in range(len(self.num_cols)):
                for j in range(i + 1, len(self.num_cols)):
                    c1, c2 = self.num_cols[i], self.num_cols[j]
                    try:
                        r = abs(self.train[c1].corr(self.train[c2]))
                        if r > 0.98:
                            suspects.append((f"{c1} ↔ {c2}", f"Near-perfect corr ({r:.4f})"))
                    except Exception:
                        pass

        # Target ile mükemmel korelasyon
        target_num = self._encode_target()
        for col in self.num_cols:
            try:
                r = abs(np.corrcoef(self.train[col].fillna(0).values, target_num)[0, 1])
                if r > 0.95:
                    suspects.append((col, f"Near-perfect TARGET corr ({r:.4f}) 🚨"))
            except Exception:
                pass

        # Test'te olmayan feature
        for col in self.feature_cols:
            if col not in self.test.columns:
                suspects.append((col, "Train'de var ama test'te YOK → olası leakage"))

        if suspects:
            self.report.add(f"  🚨 POTANSİYEL LEAKAGE ({len(suspects)}):")
            for item, reason in suspects:
                self.report.add(f"    {item}: {reason}")
        else:
            self.report.add(f"  ✅ Belirgin leakage tespit edilmedi")

    # ================================================================== #
    # SECTION 16 – FEATURE ENGINEERING FİKİRLERİ
    # ================================================================== #
    def _section_fe_ideas(self):
        self.report.add_section("16. FEATURE ENGINEERING FİKİRLERİ")

        ideas = []

        # Numerik çiftler arası interaction
        if len(self.num_cols) >= 2:
            ideas.append(f"🔢 Numerik interaction: çarpım, oran, fark ({len(self.num_cols)} feature → "
                         f"{len(self.num_cols) * (len(self.num_cols) - 1) // 2} çift)")

        # Log transform adayları
        pos_skewed = [c for c in self.num_cols
                      if self.train[c].skew() > 1 and (self.train[c] > 0).all()]
        if pos_skewed:
            ideas.append(f"📐 Log transform adayları (skew>1, tümü pozitif): {', '.join(pos_skewed[:5])}")

        # Polynomial features
        if len(self.num_cols) <= 10:
            ideas.append(f"📈 Polynomial features (degree=2) denenebilir ({len(self.num_cols)} feature)")

        # Kategorik encoding
        if self.cat_cols:
            ideas.append(f"🏷️ Target encoding, frequency encoding, leave-one-out encoding "
                         f"({len(self.cat_cols)} kategorik feature)")

        # Kategorik kombinasyonlar
        if len(self.cat_cols) >= 2:
            n_comb = min(len(self.cat_cols) * (len(self.cat_cols) - 1) // 2, 20)
            ideas.append(f"🔗 Kategorik birleşim (concat) features ({n_comb} çift)")

        # Aggregation features
        if self.cat_cols and self.num_cols:
            ideas.append(f"📊 GroupBy aggregations: her kategorik için numeriklerin mean/std/min/max")

        # Binning
        ideas.append(f"📦 Numerik binning (equal-width veya quantile-based)")

        # Missing indicator
        missing_cols = [c for c in self.feature_cols if self.train[c].isnull().sum() > 0]
        if missing_cols:
            ideas.append(f"❌ Missing indicator kolonları: {', '.join(missing_cols[:5])}")

        # Count encoding
        if self.cat_cols:
            ideas.append(f"🔢 Count/Frequency encoding tüm kategorikler için")

        self.report.add(f"  💡 ÖNERİLEN FEATURE ENGINEERING:")
        for i, idea in enumerate(ideas, 1):
            self.report.add(f"    {i}. {idea}")

    # ================================================================== #
    # SECTION 17 – CV STRATEJİSİ
    # ================================================================== #
    def _section_cv_strategy(self):
        self.report.add_section("17. CROSS-VALIDATION STRATEJİSİ")

        n = len(self.train)

        self.report.add(f"  Dataset boyutu: {n:,}")
        self.report.add(f"  Problem tipi  : {self.problem_type}")
        self.report.add(f"")

        if self.problem_type in ['binary', 'multiclass']:
            vc = self.train[self.target_col].value_counts()
            min_class = vc.min()

            if min_class < 50:
                self.report.add(f"  ⚠️ En küçük sınıf {min_class} sample. RepeatedStratifiedKFold önerilir.")
                self.report.add(f"  → RepeatedStratifiedKFold(n_splits=5, n_repeats=3)")
            elif n < 10000:
                self.report.add(f"  → StratifiedKFold(n_splits=5 veya 10, shuffle=True)")
            else:
                self.report.add(f"  → StratifiedKFold(n_splits=5, shuffle=True)")

            if n > 100000:
                self.report.add(f"  ℹ️ Büyük dataset. 5-fold yeterli, 10-fold uzun sürer.")
        else:
            if n < 10000:
                self.report.add(f"  → KFold(n_splits=5 veya 10, shuffle=True)")
                self.report.add(f"  ℹ️ Küçük dataset. RepeatedKFold düşün.")
            else:
                self.report.add(f"  → KFold(n_splits=5, shuffle=True)")

        self.report.add(f"\n  🎯 METRIC ÖNERİLERİ:")
        if self.problem_type == 'binary':
            self.report.add(f"    Primary   : AUC-ROC, Log Loss")
            self.report.add(f"    Secondary : F1, Precision, Recall")
        elif self.problem_type == 'multiclass':
            self.report.add(f"    Primary   : Macro F1, Log Loss")
            self.report.add(f"    Secondary : Accuracy, Weighted F1")
        else:
            self.report.add(f"    Primary   : RMSE, MAE")
            self.report.add(f"    Secondary : R², MAPE")

        self.report.add(f"\n  🔀 ADVERSARIAL VALIDATION:")
        self.report.add(f"    Train vs Test ayırt edilebilirlik testi yapın.")
        self.report.add(f"    AUC ≈ 0.5 → iyi (benzer dağılım)")
        self.report.add(f"    AUC > 0.8 → ciddi drift var, dikkat!")

    # ================================================================== #
    # SECTION 18 – MODELLEME ÖNERİLERİ
    # ================================================================== #
    def _section_recommendations(self):
        self.report.add_section("18. MODELLEME ÖNERİLERİ")

        recs = []

        # Imbalance
        if self.problem_type != 'regression':
            vc = self.train[self.target_col].value_counts(normalize=True)
            imb = vc.max() / vc.min()
            if imb > 10:
                recs.append("⚖️ Ciddi imbalance → class_weight='balanced', focal loss, SMOTE/ADASYN")
            elif imb > 3:
                recs.append("⚖️ Orta imbalance → scale_pos_weight (XGB), is_unbalance (LGB)")

        # High cardinality
        high_card = [c for c in self.cat_cols if self.train[c].nunique() > self.config.HIGH_CARDINALITY_THRESHOLD]
        if high_card:
            recs.append(f"🏷️ Yüksek cardinality → target encoding: {', '.join(high_card[:3])}")

        # Missing
        miss = [c for c in self.feature_cols if self.train[c].isnull().sum() > 0]
        if miss:
            recs.append(f"❌ Eksik veri → median/mode impute + missing indicator: {', '.join(miss[:3])}")

        # Skewed
        skewed = [c for c in self.num_cols if abs(self.train[c].skew()) > 2]
        if skewed:
            recs.append(f"📐 Skewed features → log/sqrt dönüşüm: {', '.join(skewed[:3])}")

        # Constant
        const = [c for c in self.feature_cols if self.train[c].nunique() <= 1]
        if const:
            recs.append(f"🗑️ Constant features → DROP: {', '.join(const)}")

        # Model seçimi
        recs.append("")
        recs.append("🤖 MODEL ÖNERİLERİ:")
        if self.problem_type == 'binary':
            recs.append("   1. LightGBM (hızlı, kategorik desteği)")
            recs.append("   2. XGBoost (scale_pos_weight ile)")
            recs.append("   3. CatBoost (kategorik-native)")
            recs.append("   4. Ensemble: Weighted Average veya Stacking")
        elif self.problem_type == 'multiclass':
            recs.append("   1. LightGBM (multi_logloss)")
            recs.append("   2. CatBoost (MultiClass)")
            recs.append("   3. XGBoost (mlogloss)")
            recs.append("   4. Neural Network (TabNet, MLP)")
        else:
            recs.append("   1. LightGBM (RMSE/MAE)")
            recs.append("   2. XGBoost")
            recs.append("   3. CatBoost")
            recs.append("   4. Ridge/Lasso blend")

        recs.append("")
        recs.append("🏆 GRANDMASTER İPUÇLARI:")
        recs.append("   • Post-processing: threshold tuning (binary), probability calibration")
        recs.append("   • Seed averaging: 3-5 farklı seed ile train, sonuç average")
        recs.append("   • Feature selection: Boruta, recursive elimination, null importance")
        recs.append("   • Blending: en az 3 farklı model family")
        recs.append("   • Pseudo-labeling: yüksek confidence test tahminlerini train'e ekle")

        self.report.add(f"  💡 ÖNERİLER:")
        for i, rec in enumerate(recs, 1):
            self.report.add(f"    {rec}")

        self.report.add(f"\n{'=' * 80}")
        self.report.add(f"{'🏆 EDA RAPORU TAMAMLANDI':^80}")
        self.report.add(f"{'=' * 80}")


# =============================================================================
# 5. ÇALIŞTIRMA
# =============================================================================

def run_eda():
    config = EDAConfig()
    eda = GrandmasterEDA(config)
    eda.load_data()
    report = eda.run_full_analysis()
    report.save("grandmaster_eda_report.txt")
    return eda


if __name__ == "__main__":
    eda_obj = run_eda()
```