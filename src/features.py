import re
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, TargetEncoder

_PHONE_RELEASE_YEARS = {
    '6': 2014, '6s': 2015, '7': 2016, '8': 2017, 'x': 2017, 'xr': 2018,
    'xs': 2018, '11': 2019, 'se': 2020, '12': 2020, '13': 2021,
    '14': 2022, '15': 2023, '16': 2024, '17': 2025,
}


class TitleFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            titles = X.iloc[:, 0].astype(str).str.lower()
        else:
            titles = pd.Series(X).astype(str).str.lower()

        X_out = pd.DataFrame(index=range(len(titles)))
        X_out['is_pro'] = titles.str.contains('pro|برو', na=False).astype(int)
        X_out['is_max'] = titles.str.contains('max|ماكس', na=False).astype(int)
        X_out['is_ultra'] = titles.str.contains('ultra|الترا', na=False).astype(int)
        X_out['is_plus'] = titles.str.contains('plus|بلس', na=False).astype(int)
        X_out['has_warranty'] = titles.str.contains('كفالة|ضمان|warranty', na=False).astype(int)
        X_out['is_sealed'] = titles.str.contains('كرتون|sealed|مسكر', na=False).astype(int)
        X_out['is_new'] = titles.str.contains('جديد|new', na=False).astype(int)
        X_out['is_5g'] = titles.str.contains(r'\b5g\b|5\s*g\b', na=False).astype(int)
        X_out['is_esim'] = titles.str.contains(r'esim|e\s*sim|الكترونيه', na=False).astype(int)

        # RAM as numeric
        ram_match = titles.str.extract(r'(\d+)\s*gb\s*ram|ram\s*(\d+)\s*gb', expand=False, flags=re.IGNORECASE)
        if ram_match is not None and hasattr(ram_match, 'shape') and ram_match.shape[1] >= 2:
            ram_vals = pd.to_numeric(ram_match.iloc[:, 0], errors='coerce').fillna(
                pd.to_numeric(ram_match.iloc[:, 1], errors='coerce')
            )
            X_out['ram_gb'] = ram_vals.fillna(0).clip(0, 24).astype(float)
        else:
            X_out['ram_gb'] = 0.0

        # Battery health
        bat_match = titles.str.extract(r'(بطاريه|بطارية|battery)\s*(\d{2,3})', expand=False)
        if bat_match is not None and hasattr(bat_match, 'shape') and bat_match.shape[1] >= 2:
            pct = pd.to_numeric(bat_match.iloc[:, 1], errors='coerce')
            X_out['battery_pct'] = pct.fillna(0).clip(0, 100).astype(float)
        else:
            X_out['battery_pct'] = 0.0

        return X_out.values


class Preprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.title_feats = TitleFeatureExtractor().fit(X, y)
        cat_cols = ['brand', 'series', 'condition', 'model_name', 'color_actual']
        self.target_enc = TargetEncoder(random_state=42, smooth='auto')
        self.target_enc.fit(X[cat_cols], y)
        num_cols = ['storage_gb', 'phone_age_months']
        self.num_cols = [c for c in num_cols if c in X.columns]
        self.scaler = StandardScaler()
        self.scaler.fit(X[self.num_cols])
        return self

    def transform(self, X):
        title_feats = self.title_feats.transform(X)
        cat_cols = ['brand', 'series', 'condition', 'model_name', 'color_actual']
        cat_feats = self.target_enc.transform(X[cat_cols])
        num_feats = self.scaler.transform(X[self.num_cols])
        return np.hstack([title_feats, cat_feats, num_feats])
