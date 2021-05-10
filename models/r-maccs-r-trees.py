import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Binarizer, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator, ZeroCount
from tpot.export_utils import set_param_recursive

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

# Average CV score on the training set was: -0.15289999993179348
exported_pipeline = make_pipeline(
    ZeroCount(),
    MinMaxScaler(),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=5, max_features=0.25, min_samples_leaf=3, min_samples_split=14, splitter="best")),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_depth=4, max_features=0.7500000000000001, max_samples=None, min_samples_leaf=1, min_samples_split=10, n_estimators=1000)),
    Binarizer(threshold=0.9),
    ExtraTreesRegressor(bootstrap=False, max_depth=None, max_features=0.1, max_samples=0.7, min_samples_leaf=1, min_samples_split=4, n_estimators=500)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 666)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

print('Success.')
