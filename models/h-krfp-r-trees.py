import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from metstab_shap.config import utils_section, csv_section
from metstab_shap.config import parse_data_config, parse_representation_config, parse_task_config
from metstab_shap.data import load_data

# load data (and change to classification if needed)
data_cfg = parse_data_config('configs/data/human.cfg')
repr_cfg = parse_representation_config('configs/repr/krfp.cfg')
task_cfg = parse_task_config('configs/task/regression.cfg')
x, y, _, test_x, test_y, smiles, test_smiles = load_data(data_cfg, **repr_cfg[utils_section])

training_features = x
training_target = y
testing_features = test_x

# Average CV score on the training set was: -0.15820678861398602
exported_pipeline = RandomForestRegressor(bootstrap=True, max_depth=None, max_features=1.0, max_samples=None, min_samples_leaf=2, min_samples_split=7, n_estimators=1000)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 666)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

print('Success.')
