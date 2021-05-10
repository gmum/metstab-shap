import os
import types
import pickle
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics import get_scorer as sklearn_get_scorer
from sklearn.metrics import precision_score, recall_score, confusion_matrix, make_scorer
from metstab_shap.config import csv_section
from metstab_shap.config import parse_data_config, parse_representation_config, parse_task_config, parse_model_config
from metstab_shap.data import cutoffs_metstabon


def force_classification(model, cutoffs, **kwargs):
    """
    Augments a regressor model to perform classification.
    :param model: sklearn-like regressor
    :param cuttoffs: cuttoffs for changing regression to classification
    :param kwargs: params for cuttoffs function
    :return: model
    """

    model.old_predict = model.predict
    model.cutoffs = cutoffs
    model.cutoffs_kwargs = kwargs

    def new_predict(self, X):
        y = self.old_predict(X)
        y = self.cutoffs(y, self.cutoffs_kwargs)
        return y

    model.predict = types.MethodType(new_predict, model)
    return model


def get_scorer(scoring):
    # extension of sklearn.metrics.get_scorer
    # to use sklearn's precision and recall with average=None, and confusion_matrix
    # we cheat a little so let's not use these additional scorers in grid search or sth
    if 'precision_none' == scoring:
        return make_scorer(precision_score, greater_is_better=True, needs_proba=False, needs_threshold=False, average=None)
    elif 'recall_none' == scoring:
        return make_scorer(recall_score, greater_is_better=True, needs_proba=False, needs_threshold=False, average=None)
    elif 'confusion_matrix' == scoring:
        return make_scorer(confusion_matrix, greater_is_better=True, needs_proba=False, needs_threshold=False)
    else:
        return sklearn_get_scorer(scoring)


def debugger_decorator(func):
    def wrapper(*args, **kwargs):
        print(f'\nCalling {func} with params:')
        for a in args:
            print(a)
        for k in kwargs:
            print(f'{k}: {kwargs[k]}')
        returned_values = func(*args, **kwargs)
        print(f'\nReturned values are: {returned_values}\n')
        return returned_values

    return wrapper


def get_configs_and_model(folder_path):
    """Go through folder with results and retrieve configs and the pickled model."""
    configs = [os.path.join(folder_path, cfg) for cfg in os.listdir(folder_path) if 'cfg' in cfg]
    data_cfg = parse_data_config([dc for dc in configs if 'rat' in dc or 'human' in dc][0])
    repr_cfg = parse_representation_config([rc for rc in configs if 'maccs' in rc or 'morgan' in rc or 'krfp' in rc][0])
    task_cfg = parse_task_config([tc for tc in configs if 'regression' in tc or 'classification' in tc][0])
    model_cfg = parse_model_config([mc for mc in configs if 'nb.cfg' in mc or 'svm.cfg' in mc or 'trees.cfg' in mc][0])
    model_pickle = [os.path.join(folder_path, pkl) for pkl in os.listdir(folder_path) if 'model.pickle' in pkl][0]

    return data_cfg, repr_cfg, task_cfg, model_cfg, model_pickle


def find_and_load(directory, pattern, protocol='numpy'):
    """Scan the directory to find a filename matching the pattern and load it using numpy or pickle protocol."""
    fname = [os.path.join(directory, f) for f in os.listdir(directory) if pattern in f][0]
    if protocol == 'numpy':
        arr = np.load(fname, allow_pickle=False)
    elif protocol == 'pickle':
        with open(fname, 'rb') as f:
            arr = pickle.load(f)
    else:
        raise NotImplementedError(f"Protocol must be `numpy` or `pickle`. Is {protocol}.")
    return arr
