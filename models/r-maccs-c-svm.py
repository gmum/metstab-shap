import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

from metstab_shap.grid import SVC_poly, SVC_rbf, SVC_sigmoid
from metstab_shap.config import utils_section, csv_section
from metstab_shap.config import parse_data_config, parse_representation_config, parse_task_config
from metstab_shap.data import load_data

# load data (and change to classification if needed)
data_cfg = parse_data_config('configs/data/rat.cfg')
repr_cfg = parse_representation_config('configs/repr/maccs.cfg')
task_cfg = parse_task_config('configs/task/classification.cfg')
x, y, _, test_x, test_y, smiles, test_smiles = load_data(data_cfg, **repr_cfg[utils_section])
# change y in case of classification
if 'classification' == task_cfg[utils_section]['task']:
    log_scale = True if 'log' == data_cfg[csv_section]['scale'].lower().strip() else False
    y = task_cfg[utils_section]['cutoffs'](y, log_scale)
    test_y = task_cfg[utils_section]['cutoffs'](test_y, log_scale)

training_features = x
training_target = y
testing_features = test_x

# Average CV score on the training set was: 0.8462165441829029
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=SVC_poly(C=0.01, cache_size=100, coef0=0.0001, degree=1, gamma=1e-05, kernel="poly", max_iter=2000, probability=True, tol=0.0001)),
    StackingEstimator(estimator=SVC_sigmoid(C=1.0, cache_size=100, coef0=-0.001, gamma="auto", kernel="sigmoid", max_iter=2000, probability=True, tol=1e-05)),
    StackingEstimator(estimator=SVC_sigmoid(C=1.0, cache_size=100, coef0=-1e-06, gamma="scale", kernel="sigmoid", max_iter=2000, probability=True, tol=1e-05)),
    RobustScaler(),
    StackingEstimator(estimator=SVC_rbf(C=25.0, cache_size=100, gamma=0.1, kernel="rbf", max_iter=2000, probability=True, tol=0.1)),
    MaxAbsScaler(),
    SVC_rbf(C=0.5, cache_size=100, gamma=0.1, kernel="rbf", max_iter=2000, probability=True, tol=0.001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 666)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

print('Success.')
