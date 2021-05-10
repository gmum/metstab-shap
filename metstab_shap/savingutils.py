import os
import sys
import time
import json
import pickle
import shutil
import logging
import numpy as np


def save_predictions(x, y, cv_split, test_x, test_y, smiles, test_smiles, model, saving_dir):
    timestamp = time.strftime('%Y-%m-%d-%H-%M')

    def _save_to_file(smiles, true_label, predicted_label, predicted_probabilities, filename):
        if predicted_probabilities is None:
            predicted_probabilities = [None, ] * len(smiles)
        try:
            os.makedirs(saving_dir)
        except FileExistsError:
            pass
        with open(os.path.join(saving_dir, f"{timestamp}-{filename}"), 'w') as fid:
            fid.write('smiles\ttrue\tpredicted\tclass_probabilities\n')
            for sm, true, pred, proba in zip(smiles, true_label, predicted_label, predicted_probabilities):
                fid.write(f"{sm}\t{true}\t{pred}\t{proba}\n")

    # training data
    for idx, (_, indices) in enumerate(cv_split):
        this_x = x[indices]
        this_y = y[indices]
        this_smiles = smiles[indices]
        predicted = model.predict(this_x)
        try:
            proba = model.predict_proba(this_x)
        except (AttributeError, RuntimeError):
            proba = None
        _save_to_file(this_smiles, this_y, predicted, proba, f'train-{idx}.predictions')

    # test data
    predicted = model.predict(test_x)
    try:
        proba = model.predict_proba(test_x)
    except (AttributeError, RuntimeError):
        proba = None
    _save_to_file(test_smiles, test_y, predicted, proba, f'test.predictions')


def save_as_json(obj, saving_dir, filename, nexp=None):
    # saves with json but uses a timestamp
    # nexp = optional neptune experiment object
    timestamp = time.strftime('%Y-%m-%d-%H-%M')
    if isinstance(obj, dict):
        for key in obj.keys():
            if isinstance(obj[key], np.ndarray):
                obj[key] = obj[key].tolist()

    with open(os.path.join(saving_dir, f'{timestamp}-{filename}'), 'w') as f:
        json.dump(obj, f, indent=2)

    if nexp:
        nexp.log_artifact(os.path.join(saving_dir, f'{timestamp}-{filename}'))


def save_configs(cfgs_list, directory):
    # stores config files in the experiment dir
    timestamp = time.strftime('%Y-%m-%d-%H-%M')
    for config_file in cfgs_list:
        filename = f"{timestamp}-{os.path.basename(config_file)}"
        shutil.copyfile(config_file, os.path.join(directory, filename))


def pickle_and_log_artifact(obj, saving_dir, filename, nexp=None):
    # saves with pickle but uses a timestamp
    # nexp = optional neptune experiment object
    timestamp = time.strftime('%Y-%m-%d-%H-%M')
    with open(os.path.join(saving_dir, f'{timestamp}-{filename}.pickle'), 'wb') as f:
        pickle.dump(obj, f, protocol=4)
    if nexp is not None:
        nexp.log_artifact(os.path.join(saving_dir, f'{timestamp}-{filename}.pickle'))


def save_npy_and_log_artifact(obj, saving_dir, filename, allow_pickle, nexp=None):
    # saves with numpy but uses a timestamp
    # nexp = optional neptune experiment object
    timestamp = time.strftime('%Y-%m-%d-%H-%M')
    np.save(os.path.join(saving_dir, f'{timestamp}-{filename}.npy'), obj, allow_pickle=allow_pickle)
    if nexp is not None:
        nexp.log_artifact(os.path.join(saving_dir, f'{timestamp}-{filename}.npy'))


class LoggerWrapper:
    def __init__(self, path='.'):
        """
        Wrapper for logging. Allows to replace sys.stderr.write so that
        error messages are redirected to sys.stdout and also saved in a file.
        use: logger = LoggerWrapper(); sys.stderr.write = logger.log_errors
        :param: path: directory to create log file
        """
        # count spaces so that the output is nicely indented
        self.trailing_spaces = 0

        # create the log file
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        self.filename = os.path.join(path, f'{timestamp}.log')
        try:
            os.mknod(self.filename)
        except FileExistsError:
            pass

        # configure logging
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-6s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=self.filename,
                            filemode='w')
        formatter = logging.Formatter('%(name)-6s: %(levelname)-8s %(message)s')

        # make a handler to redirect stuff to std.out
        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)  # bacause matplotlib throws lots of debug messages
        self.console = logging.StreamHandler(sys.stdout)
        self.console.setLevel(logging.INFO)
        self.console.setFormatter(formatter)
        self.logger.addHandler(self.console)

    def log_errors(self, msg):
        msg = msg.strip('\n')  # don't add extra enters

        if msg == ' ' * len(msg):  # if you only get spaces: don't print them, but do remember
            self.trailing_spaces += len(msg)
        elif len(msg) > 1:
            self.logger.error(' ' * self.trailing_spaces + msg)
            self.trailing_spaces = 0
