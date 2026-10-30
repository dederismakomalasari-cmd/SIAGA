"""Wrapper XGBoost — fallback ke rule-based bila model belum siap."""
import os
import numpy as np

FEATURE_ORDER = [
    "num_objects",
    "obj_in_danger",
    "max_speed_px",
    "tta",
    "min_dist_to_center",
    "max_conf_in_danger",
    "max_overlap_ratio",
    "workers_in_danger",
]

LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}


class RiskAssessor:
    def __init__(self, model_path, fallback_mode="rule"):
        self.fallback_mode = fallback_mode
        self.model = None
        self._expected_n = None

        if model_path and os.path.exists(model_path):
            try:
                import xgboost as xgb
                self.model = xgb.XGBClassifier()
                self.model.load_model(model_path)
                self._expected_n = getattr(self.model, "n_features_in_", None)
                print(f"[RiskAssessor] XGBoost loaded ({self._expected_n} fitur): {model_path}")
            except Exception as e:
                print(f"[RiskAssessor] Gagal load XGBoost ({e}). Pakai rule-based.")
                self.model = None
        else:
            print("[RiskAssessor] xgb_risk.json belum ada. Pakai rule-based.")

    def _rule_based(self, f):
        if f["tta"] <= 0:
            return 0, [1.0, 0.0, 0.0]
        if f["obj_in_danger"] == 0:
            return 0, [0.95, 0.05, 0.0]
        
        # PRIORITAS 1: Pekerja di zona langsung HIGH
        if f.get("workers_in_danger", 0) > 0:
            return 2, [0.0, 0.05, 0.95]
            
        # PRIORITAS 2: Objek lain di zona, TTA mepet
        if f["tta"] <= 8 and f["obj_in_danger"] >= 1:
            return 2, [0.0, 0.10, 0.90]
        if f["tta"] <= 15 or f["obj_in_danger"] >= 2:
            return 1, [0.05, 0.85, 0.10]
        return 1, [0.15, 0.75, 0.10]

    def assess(self, features):
        for k in FEATURE_ORDER:
            features.setdefault(k, 0.0)

        if self.model is not None:
            try:
                # Jika model dilatih dengan 6 fitur lama (tanpa workers_in_danger), potong arraynya
                if self._expected_n is not None and self._expected_n != len(FEATURE_ORDER):
                    keys = FEATURE_ORDER[: self._expected_n]
                else:
                    keys = FEATURE_ORDER

                X = np.array([[features[k] for k in keys]], dtype=np.float32)
                proba = self.model.predict_proba(X)[0].tolist()
                level = int(np.argmax(proba))
                return level, proba
            except Exception as e:
                print(f"[RiskAssessor] Predict error ({e}). Fallback ke rule-based.")

        return self._rule_based(features)