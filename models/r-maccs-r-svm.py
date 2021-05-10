import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Binarizer
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

from metstab_shap.grid import SVR_poly, SVR_rbf, SVR_sigmoid
from metstab_shap.config import utils_section, csv_section
from metstab_shap.config import parse_data_config, parse_representation_config, parse_task_config
from metstab_shap.data import load_data

# load data (and change to classification if needed)
data_cfg = parse_data_config('configs/data/rat.cfg')
repr_cfg = parse_representation_config('configs/repr/maccs.cfg')
task_cfg = parse_task_config('configs/task/regression.cfg')
x, y, _, test_x, test_y, smiles, test_smiles = load_data(data_cfg, **repr_cfg[utils_section])

training_features = x
training_target = y
testing_features = test_x

# Average CV score on the training set was: -0.15218493411020995
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_regression, percentile=69),
    StackingEstimator(estimator=LinearSVR(C=0.1, dual=True, epsilon=0.01, loss="epsilon_insensitive", tol=1e-05)),
    Binarizer(threshold=0.65),
    StackingEstimator(estimator=SVR_poly(C=0.001, cache_size=100, coef0=-0.1, degree=9, epsilon=1.0, gamma="scale", kernel="poly", max_iter=2000, tol=0.01)),
    StackingEstimator(estimator=SVR_rbf(C=20.0, cache_size=100, epsilon=1.0, gamma="auto", kernel="rbf", max_iter=2000, tol=0.0001)),
    StackingEstimator(estimator=SVR_sigmoid(C=20.0, cache_size=100, coef0=0.001, epsilon=0.0001, gamma=0.001, kernel="sigmoid", max_iter=2000, tol=0.01)),
    StackingEstimator(estimator=SVR_poly(C=0.1, cache_size=100, coef0=-0.1, degree=9, epsilon=1.0, gamma=0.1, kernel="poly", max_iter=2000, tol=0.001)),
    SVR_rbf(C=5.0, cache_size=100, epsilon=0.1, gamma="scale", kernel="rbf", max_iter=2000, tol=0.001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 666)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

print('Success.')
