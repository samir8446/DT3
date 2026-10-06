"""Stand-in for xgboost / lightgbm in environments without them: same constructor arguments and sklearn API,
backed by scikit-learn's gradient boosting. Tests the app's integration, not the libraries."""
import sys
import types

from sklearn.ensemble import GradientBoostingRegressor


class _Shim(GradientBoostingRegressor):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0, reg_lambda=1.0,
                 num_leaves=31, min_child_samples=20, random_state=None, n_jobs=None, verbosity=0, verbose=0):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                         subsample=subsample, random_state=random_state)
        self.reg_lambda, self.num_leaves, self.min_child_samples = reg_lambda, num_leaves, min_child_samples
        self.n_jobs, self.verbosity, self.verbose = n_jobs, verbosity, verbose

    def get_params(self, deep=True):
        p = super().get_params(deep)
        return {k: v for k, v in p.items() if k in ("n_estimators", "learning_rate", "max_depth", "subsample",
                                                     "random_state")}


def install() -> None:
    for name, cls in (("xgboost", "XGBRegressor"), ("lightgbm", "LGBMRegressor")):
        if name not in sys.modules:
            m = types.ModuleType(name)
            setattr(m, cls, _Shim)
            m.__spec__ = types.SimpleNamespace(name=name)  # importlib.util.find_spec needs a spec
            sys.modules[name] = m
