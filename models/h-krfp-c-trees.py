import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.preprocessing import FunctionTransformer
from copy import copy

from metstab_shap.config import utils_section, csv_section
from metstab_shap.config import parse_data_config, parse_representation_config, parse_task_config
from metstab_shap.data import load_data

# load data (and change to classification if needed)
data_cfg = parse_data_config('configs/data/human.cfg')
repr_cfg = parse_representation_config('configs/repr/krfp.cfg')
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

# Average CV score on the training set was: 0.8708849509343505
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        FunctionTransformer(copy)
    ),
    ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_depth=20, max_features=0.9000000000000001, max_samples=0.5, min_samples_leaf=1, min_samples_split=2, n_estimators=500)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 666)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

print('Success.')
