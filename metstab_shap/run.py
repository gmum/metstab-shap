import os
import sys
import time
import signal
import logging
import neptune
import numpy as np
from sklearn.metrics import confusion_matrix

import metstab_shap.grid as grid
from metstab_shap.config import utils_section, csv_section, metrics_section, force_classification_metrics_section
from metstab_shap.config import parse_model_config, parse_data_config, parse_representation_config, parse_task_config, parse_tpot_config
from metstab_shap.data import load_data
from metstab_shap.utils import force_classification, get_scorer
from metstab_shap.savingutils import save_configs, save_as_json, save_predictions, LoggerWrapper, pickle_and_log_artifact

# use: python metstab_shap/run.py configs/model/svm.cfg configs/data/rat.cfg configs/repr/maccs.cfg configs/task/classification.cfg configs/tpot-mini.cfg saving_dir

n_args = 1 + 6
neptune.init('lamiane/sandbox')
version_tag = "A"

if __name__=='__main__':
    if len(sys.argv) != n_args:
        print(f"Usage: {sys.argv[0]} model.cfg data.cfg representation.cfg task.cfg tpot.cfg main_saving_directory")
        quit(1)

    # set global saving subdir for this experiment and create it
    saving_dir = sys.argv[6]
    try:
        os.makedirs(saving_dir)
    except FileExistsError:
        pass

    # setup logger (everything that goes through logger or stderr will be saved in a file and sent to stdout)
    logger_wrapper = LoggerWrapper(saving_dir)
    sys.stderr.write = logger_wrapper.log_errors
    logger_wrapper.logger.info(f'Running {sys.argv[1:-1]}')

    # Load configs
    model_cfg = parse_model_config(sys.argv[1])
    data_cfg = parse_data_config(sys.argv[2])
    repr_cfg = parse_representation_config(sys.argv[3])
    task_cfg = parse_task_config(sys.argv[4])
    tpot_cfg = parse_tpot_config(sys.argv[5])
    save_configs(sys.argv[1:-1], saving_dir)

    # # Nicely handling interruptions from neptune UI
    def neptune_aborter():
        # closes TPOT from UI, the best found pipeline will be saved
        logging.getLogger('').info("neptune_aborter: sending Ctrl + C.")
        os.kill(os.getpid(), signal.SIGINT)

    # nexp = None  # uncomment if you don't want to use Neptune
    # leave the code below uncommented if you want to use Neptune
    nexp = neptune.create_experiment(name=saving_dir,
                                     params={'dataset': os.path.splitext(os.path.basename(sys.argv[2]))[0],
                                             'model': model_cfg[utils_section]['model'],
                                             'fingerprint': repr_cfg[utils_section]['fingerprint'],
                                             'morgan_nbits': repr_cfg[utils_section]['morgan_nbits'],
                                             'task': task_cfg[utils_section]['task'],
                                             'minimal_number_of_models': tpot_cfg[utils_section]['minimal_number_of_models'],
                                             'model_cfg': model_cfg, 'data_cfg':data_cfg,
                                             'repr_cfg': repr_cfg, 'task_cfg': task_cfg,
                                             'tpot_cfg': tpot_cfg,
                                            },
                                     tags=['metstabpred',] + sys.argv[1:-1] + [version_tag,],
                                     upload_source_files=os.path.join(os.path.dirname(os.path.realpath(__file__)), '*.py'),
                                    abort_callback=neptune_aborter)

    # load data (and change to classification if needed)
    x, y, cv_split, test_x, test_y, smiles, test_smiles = load_data(data_cfg, **repr_cfg[utils_section])

    # # # saving dataset just in case
    objects = [x, y, smiles, test_x, test_y, test_smiles]
    fnames = ['x', 'y', 'smiles', 'test_x', 'test_y', 'test_smiles']
    for obj, fname in zip(objects, fnames):
        pickle_and_log_artifact(obj, saving_dir, fname, nexp)

    # change y in case of classification
    if 'classification' == task_cfg[utils_section]['task']:
        log_scale = True if 'log' == data_cfg[csv_section]['scale'].lower().strip() else False
        y = task_cfg[utils_section]['cutoffs'](y, log_scale)
        test_y = task_cfg[utils_section]['cutoffs'](test_y, log_scale)

    # grid space and tpot configuration
    grid_space = grid.get_grid(task_cfg[utils_section]['task'], **model_cfg[utils_section])

    n_jobs = tpot_cfg[utils_section]['n_jobs']
    max_time_mins = tpot_cfg[utils_section]['max_time_mins']
    minimal_number_of_models = tpot_cfg[utils_section]['minimal_number_of_models']

    tpot_model_kwargs = { # constants
        'generations': None,
        'random_state': 666,
        'warm_start': False,
        'use_dask': True,
        'memory': 'auto',
        'verbosity': 3,
        'cv': cv_split,
        # general setup
        'n_jobs': n_jobs,
        'max_time_mins': max_time_mins,
        'max_eval_time_mins': n_jobs * max_time_mins // minimal_number_of_models,
        # per model setup
        'scoring': task_cfg[utils_section]['metric'],
        'periodic_checkpoint_folder': os.path.join(saving_dir, "./tpot_checkpoints")
         }

    # if `cv` not in... - we don't send the cv split.
    _ = [nexp.set_property(k, v) for k,v in tpot_model_kwargs.items() if 'cv' not in k]

    # run experiment
    nexp.log_text('starting time', time.strftime('%Y-%m-%d-%H-%M'))
    TPOTModel = task_cfg[utils_section]['tpot_model']
    model = TPOTModel(config_dict=grid_space, **tpot_model_kwargs)
    _ = model.fit(x, y)
    timestamp = time.strftime('%Y-%m-%d-%H-%M')
    nexp.log_text('end time', timestamp)

    # SAVING RESULTS
    model.export(os.path.join(saving_dir, f'{timestamp}-best_model_script.py'))
    nexp.log_artifact(os.path.join(saving_dir, f'{timestamp}-best_model_script.py'))
    save_as_json(model.evaluated_individuals_, saving_dir, 'evaluated_individuals.json', nexp=nexp)

    # save the model itself
    pickle_and_log_artifact(model.fitted_pipeline_, saving_dir, 'model', nexp)

    # save predictions
    save_predictions(x, y, cv_split, test_x, test_y, smiles, test_smiles, model, saving_dir)

    # SAVING SCORES
    all_scores = {}

    # # main score
    all_scores['grid_mean_cv_score'] = model._optimized_pipeline_score
    all_scores['grid_test_score'] = model.score(test_x, test_y)

    # # additional scores
    for score_name in task_cfg[metrics_section].values():
        scorer = get_scorer(score_name)
        try:
            all_scores[f'test_{score_name}'] = scorer(model, test_x, test_y)
        except RuntimeError:
            all_scores[f'test_{score_name}'] = 'RuntimeError'

    # # for regression models we perform dummy classification
    if force_classification_metrics_section in task_cfg:
        # change data and model to work with classification
        log_scale = True if 'log' == data_cfg[csv_section]['scale'].lower().strip() else False
        y = task_cfg[utils_section]['cutoffs'](y, log_scale)
        test_y = task_cfg[utils_section]['cutoffs'](test_y, log_scale)
        model = force_classification(model, task_cfg[utils_section]['cutoffs'], log_scale=log_scale)
        save_predictions(x, y, cv_split, test_x, test_y, smiles, test_smiles, model, os.path.join(saving_dir, 'forced_classification'))

        predictions = model.predict(test_x)
        for score_name in task_cfg[force_classification_metrics_section].values():
            scorer = get_scorer(score_name)
            try:
                all_scores[f'forced_classification_test_{score_name}'] = scorer(model, test_x, test_y)
            except RuntimeError:
                all_scores[f'forced_classification_test_{score_name}'] = 'RuntimeError'

    logger_wrapper.logger.info(all_scores)
    save_as_json(all_scores, saving_dir, 'best_model_scores.json', nexp=nexp)
    _ = [nexp.log_metric(k,v) for k,v in all_scores.items() if isinstance(v, float)]
