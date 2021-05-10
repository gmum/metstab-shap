import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

from metstab_shap.grid import SVR_poly, SVR_rbf
from metstab_shap.config import utils_section, csv_section
from metstab_shap.config import parse_data_config, parse_representation_config, parse_task_config
from metstab_shap.data import load_data

# load data (and change to classification if needed)
data_cfg = parse_data_config('configs/data/rat.cfg')
repr_cfg = parse_representation_config('configs/repr/krfp.cfg')
task_cfg = parse_task_config('configs/task/regression.cfg')
x, y, _, test_x, test_y, smiles, test_smiles = load_data(data_cfg, **repr_cfg[utils_section])

training_features = x
training_target = y
testing_features = test_x

# Average CV score on the training set was: -0.14668196297842537
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=SVR_poly(C=25.0, cache_size=100, coef0=0.1, degree=6, epsilon=1.0, gamma="scale", kernel="poly", max_iter=2000, tol=0.001)),
    VarianceThreshold(threshold=0.005),
    StackingEstimator(estimator=SVR_rbf(C=0.01, cache_size=100, epsilon=0.1, gamma="scale", kernel="rbf", max_iter=2000, tol=0.1)),
    SVR_rbf(C=5.0, cache_size=100, epsilon=0.01, gamma=0.01, kernel="rbf", max_iter=2000, tol=0.001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 777)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

print('Success.')
