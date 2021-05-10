import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

from metstab_shap.grid import SVR_rbf, SVR_sigmoid
from metstab_shap.config import utils_section, csv_section
from metstab_shap.config import parse_data_config, parse_representation_config, parse_task_config
from metstab_shap.data import load_data

# load data (and change to classification if needed)
data_cfg = parse_data_config('configs/data/human.cfg')
repr_cfg = parse_representation_config('configs/repr/maccs.cfg')
task_cfg = parse_task_config('configs/task/regression.cfg')
x, y, _, test_x, test_y, smiles, test_smiles = load_data(data_cfg, **repr_cfg[utils_section])

training_features = x
training_target = y
testing_features = test_x

# Average CV score on the training set was: -0.1593348788534101
exported_pipeline = make_pipeline(
    VarianceThreshold(threshold=0.001),
    StackingEstimator(estimator=SVR_sigmoid(C=5.0, cache_size=100, coef0=0.1, epsilon=1.0, gamma=0.001, kernel="sigmoid", max_iter=2000, tol=0.1)),
    RobustScaler(),
    RobustScaler(),
    StackingEstimator(estimator=SVR_rbf(C=5.0, cache_size=100, epsilon=0.001, gamma=0.1, kernel="rbf", max_iter=2000, tol=0.0001)),
    SVR_rbf(C=1.0, cache_size=100, epsilon=0.1, gamma=0.1, kernel="rbf", max_iter=2000, tol=0.1)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 666)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

print('Success.')
