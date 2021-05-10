from collections import namedtuple
from copy import deepcopy
import numpy as np
from sklearn.svm import SVC, SVR
from tpot.config.classifier import classifier_config_dict
from tpot.config.regressor import regressor_config_dict


def get_grid(task, model):
    """A general function to obtain a grid for specific task and model family."""
    task = task.lower().strip()
    model = model.lower().strip()
    if 'classification' == task:
        if 'svm' == model:
            return get_svm_classification_grid()
        elif model in ['tree', 'trees']:
            return get_tree_classification_grid()
        elif model in ['nb', 'naive-bayes', 'naive_bayes']:
            return get_nb_classification_grid()
        else:
            raise ValueError(f'For classification `model` parameter must be `svm`, `tree` or `nb`, is {model}.')
    elif 'regression' == task:
        if 'svm' == model:
            return get_svm_regression_grid()
        elif model in ['tree', 'trees']:
            return get_tree_regression_grid()
        else:
            raise ValueError(f'For regression `model` parameter must be `svm` or `tree`, is {model}.')
    else:
        raise ValueError(f'`task` parameter must be `classification` or `regression`, is {task}.')


# dummy classes - a workaround for TPOT
class SVC_rbf(SVC):
    pass


class SVC_poly(SVC):
    pass


class SVC_sigmoid(SVC):
    pass


class SVR_rbf(SVR):
    pass


class SVR_poly(SVR):
    pass


class SVR_sigmoid(SVR):
    pass


def remove_classifiers(config_dict, remove_nb=False, remove_trees=False, remove_svms=False, remove_other=True):
    """Gets TPOT configuration, returns it without models from selected families."""
    nbs = []
    svms = []
    trees = []
    other_classifiers = []
    preprocessing = []

    for key in config_dict:
        if 'naive_bayes' in key:
            nbs.append(key)
        elif any(i in key for i in ['tree', 'ExtraTreesClassifier', 'RandomForestClassifier']):
            trees.append(key)
        elif 'svm' in key:
            svms.append(key)
        elif any(i in key for i in
                 ['preprocessing', 'feature_selection', 'builtins',
                  'decomposition', 'kernel_approximation', 'FeatureAgglomeration']):
            preprocessing.append(key)
        else:
            other_classifiers.append(key)

    to_remove = []
    if remove_nb:
        to_remove.extend(nbs)

    if remove_trees:
        to_remove.extend(trees)

    if remove_svms:
        to_remove.extend(svms)

    if remove_other:
        to_remove.extend(other_classifiers)

    for key in to_remove:
        del config_dict[key]

    return config_dict


def remove_regressors(config_dict, remove_trees=False, remove_svms=False, remove_other=True):
    """Gets TPOT configuration, returns it without models from selected families."""
    svms = []
    trees = []
    other_classifiers = []
    preprocessing = []

    for key in config_dict:
        if any(i in key for i in ['tree', 'ExtraTreesRegressor', 'RandomForestRegressor']):
            trees.append(key)
        elif 'svm' in key:
            svms.append(key)
        elif any(i in key for i in
                 ['preprocessing', 'feature_selection', 'builtins',
                  'decomposition', 'kernel_approximation', 'FeatureAgglomeration']):
            preprocessing.append(key)
        else:
            other_classifiers.append(key)

    to_remove = []
    if remove_trees:
        to_remove.extend(trees)

    if remove_svms:
        to_remove.extend(svms)

    if remove_other:
        to_remove.extend(other_classifiers)

    for key in to_remove:
        del config_dict[key]

    return config_dict


def get_svm_config():
    """Hyperparams for SVMs."""
    SVM_config = namedtuple('SVM_config', ['c', 'gamma', 'coef0', 'degree', 'tol', 'epsilon', 'max_iter', 'probability'])
    cfg = SVM_config(c=[0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0],
                     gamma=['auto', 'scale'] + [10 ** i for i in range(-6, 0)],
                     coef0=[-10 ** i for i in range(-6, 0)] + [0.0] + [10 ** i for i in range(-1, -7, -1)],
                     degree=list(range(1, 10)),
                     tol=[1e-05, 0.0001, 0.001, 0.01, 0.1],
                     epsilon=[0.0001, 0.001, 0.01, 0.1, 1.0],
                     max_iter=[2000,], probability=[True,])
    return cfg


def get_svm_classification_grid():
    """Hyperparameter space for SVM classifiers."""
    cfg = get_svm_config()

    cache_size = 100
    svc_rbf = {'C': cfg.c, 'gamma': cfg.gamma, 'tol': cfg.tol,
               'kernel': ['rbf'],
               'max_iter': cfg.max_iter, 'probability': cfg.probability, 'cache_size': [cache_size,]}

    svc_poly = {'C': cfg.c, 'gamma': cfg.gamma, 'tol': cfg.tol,
                'kernel': ['poly'], 'degree': cfg.degree, 'coef0': cfg.coef0,
                'max_iter': cfg.max_iter, 'probability': cfg.probability, 'cache_size': [cache_size,]}

    svc_sigmoid = {'C': cfg.c, 'gamma': cfg.gamma, 'tol': cfg.tol,
                   'kernel': ['sigmoid'], 'coef0': cfg.coef0,
                   'max_iter': cfg.max_iter, 'probability': cfg.probability, 'cache_size': [cache_size,]}

    svm_grid = deepcopy(classifier_config_dict)
    svm_grid = remove_classifiers(svm_grid, remove_svms=False, remove_nb=True, remove_trees=True, remove_other=True)

    svm_grid['metstab_shap.grid.SVC_rbf'] = svc_rbf
    svm_grid['metstab_shap.grid.SVC_poly'] = svc_poly
    svm_grid['metstab_shap.grid.SVC_sigmoid'] = svc_sigmoid

    return svm_grid


def get_svm_regression_grid():
    """Hyperparameter space for SVM regressors."""
    cfg = get_svm_config()

    cache_size = 100
    svr_rbf = {'C': cfg.c, 'gamma': cfg.gamma, 'tol': cfg.tol, 'epsilon': cfg.epsilon,
               'kernel': ['rbf'], 'max_iter': cfg.max_iter, 'cache_size': [cache_size,]}

    svr_poly = {'C': cfg.c, 'gamma': cfg.gamma, 'tol': cfg.tol, 'epsilon': cfg.epsilon,
                'kernel': ['poly'], 'degree': cfg.degree, 'coef0': cfg.coef0, 'max_iter': cfg.max_iter, 'cache_size': [cache_size,]}

    svr_sigmoid = {'C': cfg.c, 'gamma': cfg.gamma, 'tol': cfg.tol, 'epsilon': cfg.epsilon,
                   'coef0': cfg.coef0, 'kernel': ['sigmoid'], 'max_iter': cfg.max_iter, 'cache_size': [cache_size,]}

    svm_grid = deepcopy(regressor_config_dict)
    svm_grid = remove_regressors(svm_grid, remove_svms=False, remove_trees=True, remove_other=True)

    svm_grid['metstab_shap.grid.SVR_rbf'] = svr_rbf
    svm_grid['metstab_shap.grid.SVR_poly'] = svr_poly
    svm_grid['metstab_shap.grid.SVR_sigmoid'] = svr_sigmoid

    return svm_grid


def get_tree_config():
    """Hyperparams for tree models."""
    Tree_config = namedtuple('Tree_config',
                             ['n_estimators', 'max_depth', 'max_samples', 'splitter', 'max_features', 'bootstrap'])
    cfg = Tree_config(n_estimators=[10, 50, 100, 500, 1000],
                      max_depth=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, None],
                      max_samples=[None, 0.5, 0.7, 0.9],
                      splitter=['best', 'random'],
                      max_features=np.arange(0.05, 1.01, 0.05),
                      bootstrap=[True, False])

    return cfg


def get_tree_classification_grid():
    """Hyperparameter space for tree classifiers."""
    cfg = get_tree_config()

    etc = 'sklearn.ensemble.ExtraTreesClassifier'
    dtc = 'sklearn.tree.DecisionTreeClassifier'
    rfc = 'sklearn.ensemble.RandomForestClassifier'

    tree_grid = deepcopy(classifier_config_dict)
    tree_grid = remove_classifiers(tree_grid, remove_trees=False, remove_nb=True, remove_svms=True, remove_other=True)

    tree_grid[etc]['n_estimators'] = cfg.n_estimators
    tree_grid[etc]['max_depth'] = cfg.max_depth
    tree_grid[etc]['max_samples'] = cfg.max_samples

    tree_grid[dtc]['splitter'] = cfg.splitter
    tree_grid[dtc]['max_depth'] = cfg.max_depth
    tree_grid[dtc]['max_features'] = cfg.max_features

    tree_grid[rfc]['n_estimators'] = cfg.n_estimators
    tree_grid[rfc]['max_depth'] = cfg.max_depth
    tree_grid[rfc]['bootstrap'] = cfg.bootstrap
    tree_grid[rfc]['max_samples'] = cfg.max_samples

    return tree_grid


def get_tree_regression_grid():
    """Hyperparameter space for tree regressors."""
    cfg = get_tree_config()

    etr = 'sklearn.ensemble.ExtraTreesRegressor'
    dtr = 'sklearn.tree.DecisionTreeRegressor'
    rfr = 'sklearn.ensemble.RandomForestRegressor'

    tree_grid = deepcopy(regressor_config_dict)
    tree_grid = remove_regressors(tree_grid, remove_trees=False, remove_svms=True, remove_other=True)

    tree_grid[etr]['n_estimators'] = cfg.n_estimators
    tree_grid[etr]['max_depth'] = cfg.max_depth
    tree_grid[etr]['max_samples'] = cfg.max_samples

    tree_grid[dtr]['splitter'] = cfg.splitter
    tree_grid[dtr]['max_depth'] = cfg.max_depth
    tree_grid[dtr]['max_features'] = cfg.max_features

    tree_grid[rfr]['n_estimators'] = cfg.n_estimators
    tree_grid[rfr]['max_depth'] = cfg.max_depth
    tree_grid[rfr]['bootstrap'] = cfg.bootstrap
    tree_grid[rfr]['max_samples'] = cfg.max_samples

    return tree_grid


def get_nb_config():
    """Hyperparams for naive Bayes classifiers."""
    NB_config = namedtuple('NB_config', ['alpha', 'fit_prior', 'norm', 'var_smoothing'])
    cfg = NB_config(alpha=[1e-3, 1e-2, 1e-1, 1., 10., 100.], fit_prior=[True, False], norm=[True, False],
                    var_smoothing=[1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4])

    return cfg


def get_nb_classification_grid():
    """Hyperparameter space naive Bayes classifiers."""
    cfg = get_nb_config()

    bayes_grid = deepcopy(classifier_config_dict)
    bayes_grid = remove_classifiers(bayes_grid, remove_nb=True, remove_trees=True, remove_svms=True, remove_other=True)

    bayes_grid['sklearn.naive_bayes.BernoulliNB'] = {'alpha': cfg.alpha, 'fit_prior': cfg.fit_prior}
    bayes_grid['sklearn.naive_bayes.ComplementNB'] = {'alpha': cfg.alpha, 'fit_prior': cfg.fit_prior, 'norm': cfg.norm}
    bayes_grid['sklearn.naive_bayes.GaussianNB'] = {'var_smoothing': cfg.var_smoothing}
    bayes_grid['sklearn.naive_bayes.MultinomialNB'] = {'alpha': cfg.alpha, 'fit_prior': cfg.fit_prior}

    return bayes_grid
