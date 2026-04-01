# Raporlama

```py
# =============================================================================
# UNIVERSAL KAGGLE EDA PROMPT GENERATOR
# Amaç: Her yarışmada LLM'e yapıştırılacak yeterli, kompakt ama güçlü metin çıktısı üretmek
# Model yok. Grafik yok. Sadece metin.
# =============================================================================

import os
import numpy as np
import pandas as pd
import warnings
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy import stats

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONFIG
# =============================================================================

class Config:
    BASE_PATH = "/kaggle/input/competitions/playground-series-s6e4/"
    TRAIN_FILE = "train.csv"
    TEST_FILE = "test.csv"

    TARGET = "Irrigation_Need"   # zorunlu
    ID = "id"                    # yoksa None

    CAT_THRESHOLD = 12
    TOP_N = 20
    TOP_CAT = 12
    BINS = 10
    MI_SAMPLE = 50000
    INTER_K = 8
    HIGH_CARD_THRESHOLD = 100
    RARE_THRESHOLD = 0.01


# =============================================================================
# 2. REPORT
# =============================================================================

class Report:
    def __init__(self):
        self.lines = []

    def add(self, text=""):
        self.lines.append(text)

    def section(self, title):
        self.lines += [
            f"\n{'=' * 88}",
            f"{title}",
            f"{'=' * 88}\n"
        ]

    def subsection(self, title):
        self.lines += [
            f"\n{'─' * 88}",
            f"{title}",
            f"{'─' * 88}\n"
        ]

    def show(self):
        print("\n".join(self.lines))

    def save(self, filename="llm_eda_output.txt"):
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))
        print(f"\nSaved: {filename}")


# =============================================================================
# 3. EDA
# =============================================================================

class LLMReadyEDA:
    def __init__(self, cfg):
        self.cfg = cfg
        self.r = Report()
        self.train = None
        self.test = None
        self.problem_type = None
        self.feature_cols = []
        self.cat_cols = []
        self.num_cols = []

    # -------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------
    def _encode_target(self):
        y = self.train[self.cfg.TARGET]
        if y.dtype == "object" or str(y.dtype) == "category":
            le = LabelEncoder()
            return le.fit_transform(y.astype(str))
        return y.values

    def _corr(self, a, b):
        try:
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 10:
                return 0.0
            c = np.corrcoef(a[m], b[m])[0, 1]
            return 0.0 if np.isnan(c) else float(c)
        except:
            return 0.0

    def _detect_problem_type(self):
        y = self.train[self.cfg.TARGET]
        n_unique = y.nunique(dropna=False)

        if y.dtype == "object" or str(y.dtype) == "category":
            return "binary" if n_unique == 2 else "multiclass"

        if pd.api.types.is_float_dtype(y):
            return "regression" if n_unique > 20 else ("binary" if n_unique == 2 else "multiclass")

        if pd.api.types.is_integer_dtype(y):
            return "binary" if n_unique == 2 else ("multiclass" if n_unique <= 20 else "regression")

        return "multiclass"

    def _detect_features(self):
        feats = [c for c in self.train.columns if c not in [self.cfg.TARGET, self.cfg.ID]]
        cats, nums = [], []
        for c in feats:
            if (
                self.train[c].dtype == "object"
                or str(self.train[c].dtype) == "category"
                or self.train[c].nunique(dropna=False) <= self.cfg.CAT_THRESHOLD
            ):
                cats.append(c)
            else:
                nums.append(c)
        return feats, cats, nums

    # -------------------------------------------------------------------------
    # load
    # -------------------------------------------------------------------------
    def load(self):
        train_path = os.path.join(self.cfg.BASE_PATH, self.cfg.TRAIN_FILE)
        test_path = os.path.join(self.cfg.BASE_PATH, self.cfg.TEST_FILE)

        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)

        assert self.cfg.TARGET in self.train.columns, f"Target not found: {self.cfg.TARGET}"
        if self.cfg.ID is not None:
            assert self.cfg.ID in self.train.columns, f"ID not found: {self.cfg.ID}"

        self.problem_type = self._detect_problem_type()
        self.feature_cols, self.cat_cols, self.num_cols = self._detect_features()

    # -------------------------------------------------------------------------
    # run
    # -------------------------------------------------------------------------
    def run(self):
        self._header()
        self._overview()
        self._target()
        self._missing()
        self._duplicates_constant()
        self._categorical()
        self._numerical()
        self._importance()
        self._correlation()
        self._interactions()
        self._drift()
        self._leakage()
        self._action_table()
        self._final_summary()
        return self.r

    # -------------------------------------------------------------------------
    # sections
    # -------------------------------------------------------------------------
    def _header(self):
        self.r.section("LLM-READY UNIVERSAL KAGGLE EDA OUTPUT")
        self.r.add(f"Timestamp: {pd.Timestamp.now()}")
        self.r.add(f"Target: {self.cfg.TARGET}")
        self.r.add(f"ID: {self.cfg.ID}")
        self.r.add(f"Problem Type: {self.problem_type}")

    def _overview(self):
        self.r.section("1) DATA OVERVIEW")
        tr_mem = self.train.memory_usage(deep=True).sum() / 1024**2
        te_mem = self.test.memory_usage(deep=True).sum() / 1024**2

        self.r.add(f"Train Shape: {self.train.shape}")
        self.r.add(f"Test Shape: {self.test.shape}")
        self.r.add(f"Train Memory MB: {tr_mem:.2f}")
        self.r.add(f"Test Memory MB: {te_mem:.2f}")
        self.r.add(f"Feature Count: {len(self.feature_cols)}")
        self.r.add(f"Categorical Count: {len(self.cat_cols)}")
        self.r.add(f"Numerical Count: {len(self.num_cols)}")

        self.r.add("\nDtype Breakdown:")
        for dt, cnt in self.train[self.feature_cols].dtypes.value_counts().items():
            self.r.add(f"  {str(dt):<15}: {cnt}")

        self.r.add(f"\nFeature Columns: {self.feature_cols}")

    def _target(self):
        self.r.section("2) TARGET ANALYSIS")
        y = self.train[self.cfg.TARGET]

        if self.problem_type == "regression":
            desc = y.describe()
            for k in desc.index:
                self.r.add(f"{k}: {desc[k]:.6f}")
            self.r.add(f"skew: {y.skew():.6f}")
            self.r.add(f"kurtosis: {y.kurtosis():.6f}")
            self.r.add(f"zero_ratio: {(y == 0).mean():.6f}")
            self.r.add(f"negative_ratio: {(y < 0).mean():.6f}")
        else:
            vc = y.value_counts(dropna=False)
            vr = y.value_counts(normalize=True, dropna=False)
            self.r.add(f"{'Class':<25} {'Count':>12} {'Ratio':>12}")
            self.r.add("-" * 55)
            for cls in vc.index:
                self.r.add(f"{str(cls):<25} {vc[cls]:>12} {vr[cls]:>11.4%}")
            imbalance = vr.max() / max(vr.min(), 1e-12)
            self.r.add(f"\nImbalance Ratio: {imbalance:.4f}")

    def _missing(self):
        self.r.section("3) MISSING VALUE ANALYSIS")
        tr_m = self.train.isnull().sum()
        te_m = self.test.isnull().sum()

        cols = [c for c in self.feature_cols if tr_m[c] > 0 or (c in self.test.columns and te_m[c] > 0)]

        if not cols:
            self.r.add("No missing values.")
            return

        self.r.add(f"{'Column':<30} {'Train%':>10} {'Test%':>10} {'Type':>10}")
        self.r.add("-" * 65)
        y_enc = self._encode_target()

        for c in cols:
            tr_pct = tr_m[c] / len(self.train)
            te_pct = te_m[c] / len(self.test) if c in self.test.columns else 0.0

            miss_mask = self.train[c].isnull().astype(float).values
            target_corr = abs(self._corr(miss_mask, y_enc.astype(float)))

            if target_corr > 0.1:
                mech = "MNAR?"
            else:
                mech = "MCAR/MAR"

            self.r.add(f"{c:<30} {tr_pct:>9.2%} {te_pct:>9.2%} {mech:>10}")

    def _duplicates_constant(self):
        self.r.section("4) DUPLICATES / CONSTANT / QUASI-CONSTANT")
        full_dup = self.train.duplicated().sum()
        feat_dup = self.train[self.feature_cols].duplicated().sum()

        self.r.add(f"Full-row duplicates: {full_dup}")
        self.r.add(f"Feature-only duplicates: {feat_dup}")

        const_cols = []
        quasi_cols = []
        for c in self.feature_cols:
            nun = self.train[c].nunique(dropna=False)
            if nun <= 1:
                const_cols.append(c)
            else:
                top_freq = self.train[c].value_counts(normalize=True, dropna=False).iloc[0]
                if top_freq > 0.97:
                    quasi_cols.append((c, top_freq))

        self.r.add(f"Constant Columns: {const_cols if const_cols else 'None'}")
        self.r.add("Quasi-Constant Columns:")
        if quasi_cols:
            for c, f in quasi_cols:
                self.r.add(f"  {c}: top_freq={f:.4%}")
        else:
            self.r.add("  None")

    def _categorical(self):
        self.r.section("5) CATEGORICAL FEATURE SUMMARY")
        if not self.cat_cols:
            self.r.add("No categorical columns.")
            return

        self.r.add(f"{'Feature':<28} {'Unique':>8} {'TopVal':>15} {'Top%':>10} {'Rare#':>8} {'Unseen#':>8}")
        self.r.add("-" * 85)

        for c in self.cat_cols:
            vc = self.train[c].value_counts(dropna=False)
            vr = self.train[c].value_counts(normalize=True, dropna=False)
            top_val = vc.index[0]
            top_pct = vr.iloc[0]
            rare_n = int((vr < self.cfg.RARE_THRESHOLD).sum())

            unseen = 0
            if c in self.test.columns:
                unseen = len(set(self.test[c].dropna().unique()) - set(self.train[c].dropna().unique()))

            self.r.add(
                f"{c:<28} {self.train[c].nunique(dropna=False):>8} "
                f"{str(top_val)[:15]:>15} {top_pct:>9.2%} {rare_n:>8} {unseen:>8}"
            )

        if self.problem_type != "regression":
            self.r.subsection("Categorical -> Target Pattern")
            for c in self.cat_cols[:self.cfg.TOP_CAT]:
                try:
                    ct = pd.crosstab(self.train[c], self.train[self.cfg.TARGET], normalize="index")
                    patterns = []
                    for val in ct.index[:5]:
                        row = ct.loc[val]
                        patterns.append(f"{val}->{row.idxmax()}({row.max():.1%})")
                    self.r.add(f"{c}: {' | '.join(patterns)}")
                except:
                    pass

    def _numerical(self):
        self.r.section("6) NUMERICAL FEATURE SUMMARY")
        if not self.num_cols:
            self.r.add("No numerical columns.")
            return

        self.r.add(
            f"{'Feature':<24} {'Mean':>10} {'Std':>10} {'Skew':>10} "
            f"{'Out%':>8} {'Zero%':>8} {'Min':>10} {'Max':>10}"
        )
        self.r.add("-" * 100)

        for c in self.num_cols:
            s = self.train[c]
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            out_ratio = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean()
            zero_ratio = (s == 0).mean()

            self.r.add(
                f"{c:<24} {s.mean():>10.4f} {s.std():>10.4f} {s.skew():>10.4f} "
                f"{out_ratio:>7.2%} {zero_ratio:>7.2%} {s.min():>10.4f} {s.max():>10.4f}"
            )

    def _importance(self):
        self.r.section("7) FEATURE IMPORTANCE SIGNALS")
        y = self._encode_target()

        # MI
        n = len(self.train)
        idx = np.random.choice(n, min(n, self.cfg.MI_SAMPLE), replace=False) if n > self.cfg.MI_SAMPLE else np.arange(n)
        X = self.train[self.feature_cols].iloc[idx].copy()

        for c in self.cat_cols:
            if X[c].dtype == "object":
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))

        X = X.fillna(-999)
        y_sub = y[idx]

        try:
            if self.problem_type == "regression":
                mi = mutual_info_regression(X, y_sub, random_state=42, n_neighbors=5)
            else:
                mi = mutual_info_classif(X, y_sub, random_state=42, n_neighbors=5)

            mi_s = pd.Series(mi, index=self.feature_cols).sort_values(ascending=False)

            self.r.add("Top Mutual Information:")
            for i, (f, s) in enumerate(mi_s.head(self.cfg.TOP_N).items(), 1):
                ftype = "CAT" if f in self.cat_cols else "NUM"
                self.r.add(f"  {i:>2}. {f:<25} {s:.6f} [{ftype}]")

            low_signal = mi_s[mi_s < 0.005]
            self.r.add(f"\nLow MI Features (<0.005): {list(low_signal.index)}")
        except Exception as e:
            self.r.add(f"MI failed: {e}")

    def _correlation(self):
        self.r.section("8) CORRELATION / NON-LINEAR SIGNAL")
        if not self.num_cols:
            self.r.add("No numerical columns.")
            return

        y = self._encode_target()

        rows = []
        for c in self.num_cols:
            v = self.train[c].fillna(self.train[c].median()).values
            pear = self._corr(v, y)
            try:
                spear, _ = stats.spearmanr(v, y)
                if np.isnan(spear):
                    spear = 0.0
            except:
                spear = 0.0
            gap = abs(spear) - abs(pear)
            rows.append((c, pear, spear, gap))

        rows = sorted(rows, key=lambda x: abs(x[2]), reverse=True)

        self.r.add(f"{'Feature':<24} {'Pearson':>10} {'Spearman':>10} {'Gap':>10}")
        self.r.add("-" * 60)
        for c, p, s, g in rows[:self.cfg.TOP_N]:
            self.r.add(f"{c:<24} {p:>10.4f} {s:>10.4f} {g:>10.4f}")

        nonlinear = [c for c, p, s, g in rows if g > 0.10]
        self.r.add(f"\nPotential Non-Linear Features (gap>0.10): {nonlinear}")

    def _interactions(self):
        self.r.section("9) INTERACTION SIGNALS")
        if len(self.num_cols) < 2:
            self.r.add("Not enough numerical columns.")
            return

        y = self._encode_target()
        indiv = {}
        for c in self.num_cols:
            indiv[c] = abs(self._corr(self.train[c].fillna(0).values, y))

        top_feats = sorted(indiv, key=indiv.get, reverse=True)[:self.cfg.INTER_K]
        results = []

        for c1, c2 in combinations(top_feats, 2):
            v1 = self.train[c1].fillna(0).values
            v2 = self.train[c2].fillna(0).values

            mult = abs(self._corr(v1 * v2, y))
            best_ind = max(indiv[c1], indiv[c2])
            mult_gain = mult - best_ind

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(v2 != 0, v1 / v2, 0)
            ratio = np.nan_to_num(ratio, nan=0, posinf=0, neginf=0)
            ratio_corr = abs(self._corr(ratio, y))
            ratio_gain = ratio_corr - best_ind

            results.append((f"{c1} * {c2}", mult, mult_gain))
            results.append((f"{c1} / {c2}", ratio_corr, ratio_gain))

        results = sorted(results, key=lambda x: x[2], reverse=True)
        self.r.add(f"{'Interaction':<40} {'Corr':>10} {'Gain':>10}")
        self.r.add("-" * 62)
        for name, corr, gain in results[:20]:
            self.r.add(f"{name:<40} {corr:>10.4f} {gain:>10.4f}")

    def _drift(self):
        self.r.section("10) TRAIN-TEST DRIFT")
        drift_found = False

        # categorical drift
        if self.cat_cols:
            self.r.add("Categorical Drift (max abs freq diff):")
            for c in self.cat_cols:
                if c not in self.test.columns:
                    continue
                tr = self.train[c].value_counts(normalize=True)
                te = self.test[c].value_counts(normalize=True)
                all_vals = set(tr.index) | set(te.index)
                md = max(abs(tr.get(v, 0) - te.get(v, 0)) for v in all_vals)
                if md > 0.05:
                    drift_found = True
                    self.r.add(f"  {c:<25} {md:.4f}")

        # numerical drift
        if self.num_cols:
            self.r.add("\nNumerical Drift (KS + PSI):")
            self.r.add(f"{'Feature':<24} {'KS':>10} {'p-val':>12} {'PSI':>10}")
            self.r.add("-" * 62)
            for c in self.num_cols:
                if c not in self.test.columns:
                    continue
                try:
                    trv = self.train[c].dropna().values
                    tev = self.test[c].dropna().values
                    ks, p = stats.ks_2samp(trv, tev)

                    bins = np.quantile(trv, np.linspace(0, 1, 11))
                    bins = np.unique(bins)
                    if len(bins) >= 3:
                        trh = np.histogram(trv, bins=bins)[0]
                        teh = np.histogram(tev, bins=bins)[0]
                        trp = (trh + 1) / (trh.sum() + len(trh))
                        tep = (teh + 1) / (teh.sum() + len(teh))
                        psi = float(np.sum((tep - trp) * np.log(tep / trp)))
                    else:
                        psi = 0.0

                    if ks > 0.05 or psi > 0.10:
                        drift_found = True
                        self.r.add(f"{c:<24} {ks:>10.4f} {p:>12.2e} {psi:>10.4f}")
                except:
                    pass

        if not drift_found:
            self.r.add("No significant drift detected.")

    def _leakage(self):
        self.r.section("11) LEAKAGE / OVERLAP CHECK")
        y = self._encode_target()
        suspects = []

        for c in self.num_cols:
            if self.train[c].nunique(dropna=False) == len(self.train):
                suspects.append((c, "unique_per_row"))

            corr = abs(self._corr(self.train[c].fillna(0).values, y))
            if corr > 0.95:
                suspects.append((c, f"target_corr={corr:.4f}"))

        for c in self.feature_cols:
            if c not in self.test.columns:
                suspects.append((c, "missing_in_test"))

        if self.cfg.ID and self.cfg.ID in self.train.columns:
            s = self.train[self.cfg.ID]
            if pd.api.types.is_numeric_dtype(s):
                corr = abs(self._corr(s.values.astype(float), y.astype(float)))
                if corr > 0.05:
                    suspects.append((self.cfg.ID, f"id_target_corr={corr:.4f}"))

        if suspects:
            self.r.add("Leakage Suspects:")
            for c, reason in suspects:
                self.r.add(f"  {c}: {reason}")
        else:
            self.r.add("No obvious leakage suspect.")

        # overlap
        common_feats = [c for c in self.feature_cols if c in self.test.columns]
        try:
            overlap = pd.merge(self.train[common_feats], self.test[common_feats], on=common_feats, how="inner")
            self.r.add(f"\nFeature-level train-test overlap rows: {len(overlap)}")
        except Exception as e:
            self.r.add(f"\nOverlap check failed: {e}")

    def _action_table(self):
        self.r.section("12) PER-FEATURE ACTION HINTS")
        y = self._encode_target()

        # quick MI for priority
        mi_map = {}
        try:
            X = self.train[self.feature_cols].copy()
            for c in self.cat_cols:
                if X[c].dtype == "object":
                    X[c] = LabelEncoder().fit_transform(X[c].astype(str))
            X = X.fillna(-999)

            if len(X) > self.cfg.MI_SAMPLE:
                idx = np.random.choice(len(X), self.cfg.MI_SAMPLE, replace=False)
                X2 = X.iloc[idx]
                y2 = y[idx]
            else:
                X2 = X
                y2 = y

            if self.problem_type == "regression":
                mi = mutual_info_regression(X2, y2, random_state=42, n_neighbors=5)
            else:
                mi = mutual_info_classif(X2, y2, random_state=42, n_neighbors=5)
            mi_map = dict(zip(self.feature_cols, mi))
        except:
            mi_map = {c: np.nan for c in self.feature_cols}

        self.r.add(f"{'Feature':<28} {'Type':<6} {'Main Action':<30} {'Priority':<10}")
        self.r.add("-" * 80)

        for c in self.feature_cols:
            if c in self.cat_cols:
                ftype = "CAT"
                nun = self.train[c].nunique(dropna=False)
                if nun <= 1:
                    action = "DROP"
                    pr = "HIGH"
                elif nun > self.cfg.HIGH_CARD_THRESHOLD:
                    action = "TARGET/FREQ ENCODE"
                    pr = "HIGH"
                elif nun <= 5:
                    action = "ONEHOT"
                    pr = "MED"
                else:
                    action = "TARGET ENCODE"
                    pr = "MED"
            else:
                ftype = "NUM"
                s = self.train[c]
                if s.nunique(dropna=False) <= 1:
                    action = "DROP"
                    pr = "HIGH"
                else:
                    action_parts = []
                    if abs(s.skew()) > 2:
                        action_parts.append("TRANSFORM")
                    q1, q3 = s.quantile(0.25), s.quantile(0.75)
                    iqr = q3 - q1
                    out_ratio = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean()
                    if out_ratio > 0.05:
                        action_parts.append("CLIP")
                    if s.isnull().mean() > 0:
                        action_parts.append("IMPUTE")
                    if mi_map.get(c, 0) < 0.005:
                        action_parts.append("TRY_DROP")
                    action = " + ".join(action_parts) if action_parts else "KEEP"
                    pr = "HIGH" if mi_map.get(c, 0) > 0.1 else ("LOW" if mi_map.get(c, 0) < 0.005 else "MED")

            self.r.add(f"{c:<28} {ftype:<6} {action:<30} {pr:<10}")

    def _final_summary(self):
        self.r.section("13) FINAL SUMMARY FOR LLM")
        self.r.add("Copy the whole output to the LLM.")
        self.r.add("")
        self.r.add(f"Problem Type: {self.problem_type}")
        self.r.add(f"Feature Count: {len(self.feature_cols)}")
        self.r.add(f"Categorical Features: {self.cat_cols}")
        self.r.add(f"Numerical Features: {self.num_cols}")

        if self.problem_type != "regression":
            y = self.train[self.cfg.TARGET]
            vr = y.value_counts(normalize=True, dropna=False)
            self.r.add(f"Class Ratios: {vr.to_dict()}")

        self.r.add("")
        self.r.add("Ask the LLM for:")
        self.r.add("1. feature engineering ideas")
        self.r.add("2. preprocessing strategy")
        self.r.add("3. CV strategy")
        self.r.add("4. model recommendations")
        self.r.add("5. ensemble / post-processing ideas")


# =============================================================================
# 4. RUN
# =============================================================================

def run_eda():
    cfg = Config()
    eda = LLMReadyEDA(cfg)
    eda.load()
    report = eda.run()
    report.show()
    report.save("llm_eda_output.txt")
    return eda


if __name__ == "__main__":
    eda_obj = run_eda()
```

# FeatureTools

```py
import pandas as pd
import numpy as np
import featuretools as ft
from woodwork.logical_types import Categorical, Double, Integer

# =========================================================
# 1. CONFIG
# =========================================================
BASE_PATH = "/kaggle/input/competitions/playground-series-s6e4/"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

TARGET = "Irrigation_Need"
ID_COL = "id"

# EDA'dan gelen kolonlar
cat_cols = [
    'Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
    'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region'
]

num_cols = [
    'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
    'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours',
    'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm'
]

# =========================================================
# 2. LOAD
# =========================================================
train = pd.read_csv(BASE_PATH + TRAIN_FILE)
test = pd.read_csv(BASE_PATH + TEST_FILE)

train_features = train.drop(columns=[TARGET]).copy()
test_features = test.copy()

# train+test birlikte feature üret, sonra geri ayır
full = pd.concat([train_features, test_features], axis=0, ignore_index=True)

# güvenlik
if ID_COL not in full.columns:
    full[ID_COL] = np.arange(len(full))

# =========================================================
# 3. WOODWORK LOGICAL TYPES
# =========================================================
logical_types = {ID_COL: Integer}
for c in cat_cols:
    logical_types[c] = Categorical
for c in num_cols:
    logical_types[c] = Double

# =========================================================
# 4. ENTITYSET
# =========================================================
es = ft.EntitySet(id="irrigation_es")

es = es.add_dataframe(
    dataframe_name="data",
    dataframe=full,
    index=ID_COL,
    logical_types=logical_types
)

# =========================================================
# 5. FEATURETOOLS DFS
# =========================================================
# Bu veri tek tablo olduğu için aggregation yok, transform ağırlıklı.
# Çok şişmemesi için primitive set kontrollü tutuldu.

feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name="data",
    trans_primitives=[
        "add_numeric",
        "subtract_numeric",
        "multiply_numeric",
        "divide_numeric",
        "absolute",
        "percentile",
        "negate",
        "equal",
        "not_equal",
        "greater_than",
        "less_than",
        "modulo_numeric",
    ],
    agg_primitives=[],
    max_depth=1,
    features_only=False,
    verbose=True
)

# =========================================================
# 6. BASIC CLEANUP
# =========================================================
# Target yok, sadece feature matrisi
# Çok fazla NaN/inf oluşan kolonları temizle

feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)

# %95'ten fazla missing olanları at
high_missing = feature_matrix.columns[feature_matrix.isnull().mean() > 0.95]
feature_matrix = feature_matrix.drop(columns=high_missing)

# tek değerli kolonları at
nunique = feature_matrix.nunique(dropna=False)
constant_cols = nunique[nunique <= 1].index
feature_matrix = feature_matrix.drop(columns=constant_cols)

# object kolonları string'e çek
for c in feature_matrix.select_dtypes(include="object").columns:
    feature_matrix[c] = feature_matrix[c].astype(str)

print("Generated feature matrix shape:", feature_matrix.shape)
print("Dropped high-missing cols:", len(high_missing))
print("Dropped constant cols:", len(constant_cols))

# =========================================================
# 7. SPLIT BACK TO TRAIN / TEST
# =========================================================
train_ft = feature_matrix.iloc[:len(train)].copy()
test_ft = feature_matrix.iloc[len(train):].copy()

# target ekle
train_ft[TARGET] = train[TARGET].values

print("Train FT shape:", train_ft.shape)
print("Test FT shape :", test_ft.shape)

# =========================================================
# 8. OPTIONAL: SAVE
# =========================================================
train_ft.to_csv("train_featuretools.csv", index=False)
test_ft.to_csv("test_featuretools.csv", index=False)

print("Saved: train_featuretools.csv, test_featuretools.csv")
```

# FeatureTools Sonrası Temizlik

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# =========================================================
# LOAD
# =========================================================
TARGET = "Irrigation_Need"

train_ft = pd.read_csv("train_featuretools.csv")
test_ft = pd.read_csv("test_featuretools.csv")

print("Before cleaning:")
print(train_ft.shape, test_ft.shape)

# =========================================================
# 1. TARGET AYIR
# =========================================================
y = train_ft[TARGET].copy()
X_train = train_ft.drop(columns=[TARGET]).copy()
X_test = test_ft.copy()

# =========================================================
# 2. INF / AŞIRI BÜYÜK DEĞER TEMİZLİĞİ
# =========================================================
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# çok büyük abs değerleri NaN yap
for c in X_train.columns:
    if pd.api.types.is_numeric_dtype(X_train[c]):
        X_train.loc[X_train[c].abs() > 1e15, c] = np.nan
        X_test.loc[X_test[c].abs() > 1e15, c] = np.nan

# =========================================================
# 3. TRAIN-TEST UYUMLU KOLONLAR
# =========================================================
common_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[common_cols]
X_test = X_test[common_cols]

# =========================================================
# 4. YÜKSEK MISSING OLANLARI SİL
# =========================================================
high_missing_cols = X_train.columns[X_train.isnull().mean() > 0.90].tolist()
X_train = X_train.drop(columns=high_missing_cols)
X_test = X_test.drop(columns=high_missing_cols)

print("Dropped high-missing:", len(high_missing_cols))

# =========================================================
# 5. TEK DEĞERLİ KOLONLARI SİL
# =========================================================
constant_cols = [c for c in X_train.columns if X_train[c].nunique(dropna=False) <= 1]
X_train = X_train.drop(columns=constant_cols)
X_test = X_test.drop(columns=constant_cols)

print("Dropped constant:", len(constant_cols))

# =========================================================
# 6. OBJECT KOLONLARI ENCODE ET
# =========================================================
obj_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

for c in obj_cols:
    le = LabelEncoder()
    full_vals = pd.concat([X_train[c].astype(str), X_test[c].astype(str)], axis=0)
    le.fit(full_vals)
    X_train[c] = le.transform(X_train[c].astype(str))
    X_test[c] = le.transform(X_test[c].astype(str))

print("Encoded object cols:", len(obj_cols))

# =========================================================
# 7. LOW-SIGNAL FEATURE ELEME (MI bazlı)
# =========================================================
# target encode
if y.dtype == "object":
    y_enc = LabelEncoder().fit_transform(y.astype(str))
    problem_type = "classification"
elif y.nunique() <= 20:
    y_enc = y.values
    problem_type = "classification"
else:
    y_enc = y.values
    problem_type = "regression"

# sample al
sample_size = min(50000, len(X_train))
sample_idx = np.random.choice(len(X_train), sample_size, replace=False)

X_sample = X_train.iloc[sample_idx].copy()
y_sample = y_enc[sample_idx]

# eksikleri doldur
for c in X_sample.columns:
    if pd.api.types.is_numeric_dtype(X_sample[c]):
        med = X_train[c].median()
        X_sample[c] = X_sample[c].fillna(med)
        X_train[c] = X_train[c].fillna(med)
        X_test[c] = X_test[c].fillna(med)
    else:
        X_sample[c] = X_sample[c].fillna("MISSING")
        X_train[c] = X_train[c].fillna("MISSING")
        X_test[c] = X_test[c].fillna("MISSING")

# MI
if problem_type == "classification":
    mi = mutual_info_classif(X_sample, y_sample, random_state=42)
else:
    mi = mutual_info_regression(X_sample, y_sample, random_state=42)

mi_series = pd.Series(mi, index=X_sample.columns).sort_values(ascending=False)

# çok düşük MI feature'ları sil
low_mi_cols = mi_series[mi_series < 0.0005].index.tolist()

X_train = X_train.drop(columns=low_mi_cols)
X_test = X_test.drop(columns=low_mi_cols)

print("Dropped low-MI:", len(low_mi_cols))

# =========================================================
# 8. ÇOK YÜKSEK KORELASYONLU KOLONLARI SİL
# =========================================================
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

corr_matrix = X_train[num_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.995)]

X_train = X_train.drop(columns=to_drop_corr)
X_test = X_test.drop(columns=to_drop_corr)

print("Dropped highly-correlated:", len(to_drop_corr))

# =========================================================
# 9. FINAL
# =========================================================
train_clean = X_train.copy()
train_clean[TARGET] = y.values
test_clean = X_test.copy()

print("\nAfter cleaning:")
print(train_clean.shape, test_clean.shape)

train_clean.to_csv("train_featuretools_clean.csv", index=False)
test_clean.to_csv("test_featuretools_clean.csv", index=False)

print("\nSaved:")
print("train_featuretools_clean.csv")
print("test_featuretools_clean.csv")

print("\nTop 30 MI features:")
print(mi_series.head(30))
```

# 4 Varyantllı Benchmark

```python
# =============================================================================
# 4 VARIANT BENCHMARK PIPELINE
# original vs manual_fe vs featuretools_clean vs featuretools_pruned
# Çok-sınıflı tabular benchmark. Sadece CV karşılaştırması.
# =============================================================================

import os
import re
import gc
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import make_scorer, f1_score, accuracy_score, log_loss

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# =============================================================================
# 1. CONFIG
# =============================================================================

BASE_PATH = "/kaggle/input/competitions/playground-series-s6e4/"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

FT_TRAIN_FILE = "train_featuretools_clean.csv"
FT_TEST_FILE  = "test_featuretools_clean.csv"

TARGET = "Irrigation_Need"
ID_COL = "id"

N_SPLITS = 5
RANDOM_STATE = 42
USE_CATBOOST = True   # False yaparsan sadece LightGBM benchmark yapar


# =============================================================================
# 2. LOAD
# =============================================================================

train = pd.read_csv(os.path.join(BASE_PATH, TRAIN_FILE))
test = pd.read_csv(os.path.join(BASE_PATH, TEST_FILE))

ft_train = pd.read_csv(FT_TRAIN_FILE)
ft_test = pd.read_csv(FT_TEST_FILE)

# target encode
le_target = LabelEncoder()
y = le_target.fit_transform(train[TARGET].astype(str))

# =============================================================================
# 3. MANUAL FEATURE ENGINEERING
# =============================================================================

def add_manual_features(df):
    df = df.copy()
    eps = 1e-6

    # numeric interactions
    df["moisture_temp_ratio"] = df["Soil_Moisture"] / (df["Temperature_C"] + eps)
    df["temp_moisture_ratio"] = df["Temperature_C"] / (df["Soil_Moisture"] + eps)
    df["rain_temp_ratio"] = df["Rainfall_mm"] / (df["Temperature_C"] + eps)
    df["humidity_temp_ratio"] = df["Humidity"] / (df["Temperature_C"] + eps)
    df["prev_irrig_temp_ratio"] = df["Previous_Irrigation_mm"] / (df["Temperature_C"] + eps)
    df["moisture_rain_ratio"] = df["Soil_Moisture"] / (df["Rainfall_mm"] + eps)
    df["rain_moisture_ratio"] = df["Rainfall_mm"] / (df["Soil_Moisture"] + eps)

    df["moisture_temp_diff"] = df["Soil_Moisture"] - df["Temperature_C"]
    df["moisture_humidity_diff"] = df["Soil_Moisture"] - df["Humidity"]
    df["rain_prev_irrig_diff"] = df["Rainfall_mm"] - df["Previous_Irrigation_mm"]
    df["moisture_prev_irrig_sum"] = df["Soil_Moisture"] + df["Previous_Irrigation_mm"]

    df["temp_wind_product"] = df["Temperature_C"] * df["Wind_Speed_kmh"]
    df["rain_humidity_product"] = df["Rainfall_mm"] * df["Humidity"]
    df["moisture_ph_product"] = df["Soil_Moisture"] * df["Soil_pH"]
    df["field_rain_product"] = df["Field_Area_hectare"] * df["Rainfall_mm"]
    df["field_prev_irrig_product"] = df["Field_Area_hectare"] * df["Previous_Irrigation_mm"]

    # row-wise light stats on numerics
    num_cols = [
        'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
        'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours',
        'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm'
    ]
    df["row_num_mean"] = df[num_cols].mean(axis=1)
    df["row_num_std"] = df[num_cols].std(axis=1)
    df["row_num_min"] = df[num_cols].min(axis=1)
    df["row_num_max"] = df[num_cols].max(axis=1)
    df["row_num_range"] = df["row_num_max"] - df["row_num_min"]

    # categorical crosses
    df["crop_stage"] = df["Crop_Type"].astype(str) + "_" + df["Crop_Growth_Stage"].astype(str)
    df["season_region"] = df["Season"].astype(str) + "_" + df["Region"].astype(str)
    df["stage_mulching"] = df["Crop_Growth_Stage"].astype(str) + "_" + df["Mulching_Used"].astype(str)
    df["soil_season"] = df["Soil_Type"].astype(str) + "_" + df["Season"].astype(str)
    df["water_irrigation"] = df["Water_Source"].astype(str) + "_" + df["Irrigation_Type"].astype(str)

    # binary flag interactions
    mulching_yes = (df["Mulching_Used"].astype(str) == "Yes").astype(int)
    df["mulching_moisture"] = mulching_yes * df["Soil_Moisture"]
    df["mulching_temp"] = mulching_yes * df["Temperature_C"]
    df["mulching_prev_irrig"] = mulching_yes * df["Previous_Irrigation_mm"]

    return df


# =============================================================================
# 4. PRUNE FEATURETOOLS
# =============================================================================

def prune_featuretools(df, target_col=None):
    df = df.copy()

    # target ayır
    y_local = None
    if target_col is not None and target_col in df.columns:
        y_local = df[target_col].copy()
        df = df.drop(columns=[target_col])

    # primitive bazlı kolon temizliği
    # Bu veri için aşırı redundant primitive'ler
    drop_patterns = [
        r"^ABSOLUTE\(",
        r"^PERCENTILE\(",
        r"^-\(",
    ]

    cols_to_drop = []
    for c in df.columns:
        for pat in drop_patterns:
            if re.search(pat, c):
                cols_to_drop.append(c)
                break

    cols_to_drop = list(set(cols_to_drop))
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # çok yüksek isimsel redundancy: a+b ve b+a gibi şeyler zaten FT temizliğinde
    # burada sadece shape görmek için bırakıyoruz

    if y_local is not None:
        df[target_col] = y_local

    return df, cols_to_drop


# =============================================================================
# 5. DATASET VARIANTS
# =============================================================================

# A) ORIGINAL
train_original = train.copy()
test_original = test.copy()

# B) MANUAL FE
train_manual = add_manual_features(train.copy())
test_manual = add_manual_features(test.copy())

# C) FEATURETOOLS CLEAN
train_ft_clean = ft_train.copy()
test_ft_clean = ft_test.copy()

# D) FEATURETOOLS PRUNED
train_ft_pruned, dropped_pruned_cols = prune_featuretools(ft_train.copy(), target_col=TARGET)
test_ft_pruned, _ = prune_featuretools(ft_test.copy(), target_col=None)

# ensure same columns
common_ft_pruned = [c for c in train_ft_pruned.columns if c != TARGET and c in test_ft_pruned.columns]
train_ft_pruned = train_ft_pruned[common_ft_pruned + [TARGET]]
test_ft_pruned = test_ft_pruned[common_ft_pruned].copy()

print("VARIANT SHAPES")
print("original      :", train_original.shape, test_original.shape)
print("manual_fe     :", train_manual.shape, test_manual.shape)
print("ft_clean      :", train_ft_clean.shape, test_ft_clean.shape)
print("ft_pruned     :", train_ft_pruned.shape, test_ft_pruned.shape)
print("ft_pruned dropped cols:", len(dropped_pruned_cols))


# =============================================================================
# 6. PREPROCESSING HELPERS
# =============================================================================

def split_xy(train_df, test_df, target_col, id_col=None):
    drop_cols = [target_col]
    if id_col is not None and id_col in train_df.columns:
        drop_cols.append(id_col)

    X_train = train_df.drop(columns=drop_cols, errors="ignore").copy()
    X_test = test_df.drop(columns=[id_col] if (id_col is not None and id_col in test_df.columns) else [], errors="ignore").copy()

    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    y_local = train_df[target_col].copy()
    return X_train, X_test, y_local


def build_lgb_pipeline(X):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    model = LGBMClassifier(
        objective="multiclass",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])
    return pipe


def encode_for_catboost(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    for c in cat_cols:
        all_vals = pd.concat([X_train[c].astype(str), X_test[c].astype(str)], axis=0)
        le = LabelEncoder()
        le.fit(all_vals)
        X_train[c] = le.transform(X_train[c].astype(str))
        X_test[c] = le.transform(X_test[c].astype(str))

    return X_train, X_test, cat_cols


# =============================================================================
# 7. SCORING
# =============================================================================

scoring = {
    "f1_macro": make_scorer(f1_score, average="macro"),
    "accuracy": make_scorer(accuracy_score)
}

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


# =============================================================================
# 8. BENCHMARK FUNCTION
# =============================================================================

def benchmark_variant(name, train_df, test_df, target_col, id_col=None, use_catboost=True):
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {name}")
    print(f"{'='*80}")

    X_train, X_test, y_raw = split_xy(train_df, test_df, target_col, id_col=id_col)
    y_enc = le_target.transform(y_raw.astype(str))

    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("Object cols  :", len(X_train.select_dtypes(include=['object', 'category']).columns))
    print("Numeric cols :", len(X_train.select_dtypes(exclude=['object', 'category']).columns))

    results = []

    # -------------------------------------------------------------------------
    # LightGBM + preprocessing pipeline
    # -------------------------------------------------------------------------
    lgb_pipe = build_lgb_pipeline(X_train)

    for score_name, scorer in scoring.items():
        scores = cross_val_score(
            lgb_pipe,
            X_train,
            y_enc,
            cv=cv,
            scoring=scorer,
            n_jobs=1
        )
        results.append({
            "variant": name,
            "model": "LightGBM",
            "metric": score_name,
            "mean": scores.mean(),
            "std": scores.std()
        })
        print(f"LGBM {score_name:<10}: {scores.mean():.6f} ± {scores.std():.6f}")

    # -------------------------------------------------------------------------
    # CatBoost
    # -------------------------------------------------------------------------
    if use_catboost:
        X_cb_train, X_cb_test, cat_cols = encode_for_catboost(X_train, X_test)
        cat_idx = [X_cb_train.columns.get_loc(c) for c in cat_cols]

        cb_model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1",
            iterations=500,
            learning_rate=0.05,
            depth=8,
            random_seed=RANDOM_STATE,
            auto_class_weights="Balanced",
            verbose=0
        )

        for score_name, scorer in scoring.items():
            scores = cross_val_score(
                cb_model,
                X_cb_train,
                y_enc,
                cv=cv,
                scoring=scorer,
                n_jobs=1,
                fit_params={"cat_features": cat_idx}
            )
            results.append({
                "variant": name,
                "model": "CatBoost",
                "metric": score_name,
                "mean": scores.mean(),
                "std": scores.std()
            })
            print(f"CAT  {score_name:<10}: {scores.mean():.6f} ± {scores.std():.6f}")

    gc.collect()
    return pd.DataFrame(results)


# =============================================================================
# 9. RUN ALL VARIANTS
# =============================================================================

all_results = []

all_results.append(
    benchmark_variant(
        name="original",
        train_df=train_original,
        test_df=test_original,
        target_col=TARGET,
        id_col=ID_COL,
        use_catboost=USE_CATBOOST
    )
)

all_results.append(
    benchmark_variant(
        name="manual_fe",
        train_df=train_manual,
        test_df=test_manual,
        target_col=TARGET,
        id_col=ID_COL,
        use_catboost=USE_CATBOOST
    )
)

all_results.append(
    benchmark_variant(
        name="ft_clean",
        train_df=train_ft_clean,
        test_df=test_ft_clean,
        target_col=TARGET,
        id_col=ID_COL,
        use_catboost=USE_CATBOOST
    )
)

all_results.append(
    benchmark_variant(
        name="ft_pruned",
        train_df=train_ft_pruned,
        test_df=test_ft_pruned,
        target_col=TARGET,
        id_col=ID_COL,
        use_catboost=USE_CATBOOST
    )
)

results_df = pd.concat(all_results, ignore_index=True)

# =============================================================================
# 10. SUMMARY TABLE
# =============================================================================

print(f"\n{'#'*90}")
print("FINAL BENCHMARK SUMMARY")
print(f"{'#'*90}")

summary = results_df.pivot_table(
    index=["variant", "model"],
    columns="metric",
    values="mean"
).reset_index()

print(summary.sort_values(["f1_macro", "accuracy"], ascending=False))

results_df.to_csv("benchmark_results.csv", index=False)
summary.to_csv("benchmark_summary.csv", index=False)

print("\nSaved:")
print("benchmark_results.csv")
print("benchmark_summary.csv")

# En iyi varyant
best_row = summary.sort_values("f1_macro", ascending=False).iloc[0]
print("\nBEST SETUP:")
print(best_row.to_dict())

# JSON log
with open("benchmark_best.json", "w") as f:
    json.dump(best_row.to_dict(), f, indent=2)

print("benchmark_best.json saved.")
```