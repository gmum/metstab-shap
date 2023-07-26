import os
import sys
import logging
import time
import warnings

import shap
import pickle
import numpy as np

from metstab_shap.data import load_data
from metstab_shap.config import parse_shap_config, utils_section
from metstab_shap.utils import get_configs_and_model, find_and_load
from metstab_shap.savingutils import save_configs, save_as_json, pickle_and_log_artifact, save_npy_and_log_artifact, LoggerWrapper

# use: python metstab_shap/calculate_shap_values.py {ml results directory} {shap results directory} {shap config}

n_args = 1 + 3

if __name__=='__main__':
    if len(sys.argv) != n_args:
        print(f"Usage: python {sys.argv[0]} input_directory saving_directory shap.cfg")
        quit(1)

    data_dir = sys.argv[1]
    saving_dir = sys.argv[2]

    # set global saving dir for this experiment and create it
    try:
        os.makedirs(saving_dir)
    except FileExistsError:
        pass

    # setup logger (everything that goes through logger or stderr will be saved in a file and sent to stdout)
    logger_wrapper = LoggerWrapper(saving_dir)
    sys.stderr.write = logger_wrapper.log_errors
    logger_wrapper.logger.info(f'Running {sys.argv[1:]}')

    # load shap configuration
    shap_cfg = parse_shap_config(sys.argv[3])
    k = shap_cfg[utils_section]["k"]
    link = shap_cfg[utils_section]["link"]
    save_configs([sys.argv[3], ], saving_dir)

    nexp = None  # uncomment if you don't want to use Neptune

    # load other configs
    data_cfg, repr_cfg, task_cfg, model_cfg, model_pickle = get_configs_and_model(data_dir)

    # load the data
    if "krfp" not in repr_cfg[utils_section]['fingerprint']:
        # MACCS or Morgan, we can calculate it for the sake of simplicity
        x, y, _, test_x, test_y, smiles, test_smiles = load_data(data_cfg, **repr_cfg[utils_section])
    else:
        # load KRFP from file in order to save 2-3 hours
        x = find_and_load(data_dir, '-x.pickle', protocol='pickle')
        y = find_and_load(data_dir, '-y.pickle', protocol='pickle')
        smiles = find_and_load(data_dir, '-smiles.pickle', protocol='pickle')
        test_x = find_and_load(data_dir, '-test_x.pickle', protocol='pickle')
        test_y = find_and_load(data_dir, '-test_y.pickle', protocol='pickle')
        test_smiles = find_and_load(data_dir, '-test_smiles.pickle', protocol='pickle')

    # we care no more about the data split so we merge it back
    X_full = np.concatenate((x, test_x), axis=0)
    y_full = np.concatenate((y, test_y), axis=0)
    smiles_full = np.concatenate((smiles, test_smiles), axis=0)

    # some smiles are present in the data multiple times
    # removing abundant smiles will speed up the calculation of SHAP values
    unique_indices = [np.where(smiles_full == sm)[0][0] for sm in np.unique(smiles_full)]
    X_full = X_full[unique_indices, :]
    y_full = y_full[unique_indices]
    smiles_full = smiles_full[unique_indices]

    objects = [X_full, y_full, smiles_full]
    fnames = ['X_full', 'true_ys', 'smiles']
    for obj, fname in zip(objects , fnames):
        save_npy_and_log_artifact(obj, saving_dir, fname, allow_pickle=False, nexp=nexp)

    # load model
    with open(model_pickle, 'rb') as f:
        model = pickle.load(f)

    # calculating SHAP values
    background_data = shap.kmeans(X_full, k)
    if 'classification' == task_cfg[utils_section]['task']:
        e = shap.KernelExplainer(model.predict_proba, background_data, link=link)
    else:
        e = shap.KernelExplainer(model.predict, background_data, link=link)  # regression
    sv = e.shap_values(X_full)

    # saving results
    pickle_and_log_artifact(background_data, saving_dir, 'background_data', nexp)
    save_npy_and_log_artifact(sv, saving_dir, 'SHAP_values', allow_pickle=False, nexp=nexp)
    save_npy_and_log_artifact(e.expected_value, saving_dir, 'expected_values', allow_pickle=False, nexp=nexp)

    if 'classification' == task_cfg[utils_section]['task']:
        save_npy_and_log_artifact(model.classes_, saving_dir, 'classes_order', allow_pickle=False, nexp=nexp)
        preds = model.predict_proba(X_full)
    else:
        preds = model.predict(X_full)
    save_npy_and_log_artifact(preds, saving_dir, 'predictions', allow_pickle=False, nexp=nexp)

