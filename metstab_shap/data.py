import os
import csv
import shutil
import hashlib
import tempfile
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem import MolFromSmiles
from padelpy import padeldescriptor  # required to calculate KlekotaRothFingerPrint

from metstab_shap.config import csv_section, utils_section

DATA = 'DATA'
test = 'test'

def load_data(data_config, fingerprint, morgan_nbits=None):
    datasets = []
    indices = []
    this_start = 0
    for path in sorted(data_config[DATA].values()):
        x, y, smiles = preprocess_dataset(path=path, data_config=data_config,
                                          fingerprint=fingerprint, morgan_nbits=morgan_nbits)
        datasets.append((x, y, smiles))
        indices.append((this_start, this_start+len(y)))
        this_start += len(y)

    x = np.vstack([el[0] for el in datasets])
    y = np.hstack([el[1] for el in datasets])
    smiles = np.hstack([el[2] for el in datasets])

    cv_split = get_cv_split(indices)

    # test set
    test_x, test_y, test_smiles = preprocess_dataset(path=data_config[utils_section][test],
                                                     data_config=data_config,
                                                     fingerprint=fingerprint,
                                                     morgan_nbits=morgan_nbits)

    return x, y, cv_split, test_x, test_y, smiles, test_smiles


def load_data_from_df(dataset_paths, smiles_index, y_index, skip_line=False, delimiter=',', scale=None, average=None):
    """
    Load multiple files from csvs, concatenate and return smiles and ys
    :param dataset_paths: list: paths to csv files with data
    :param smiles_index: int: index of the column with smiles
    :param y_index: int: index of the column with the label
    :param skip_line: boolean: True if the first line of the file contains column names, False otherwise
    :param delimiter: delimeter used in csv
    :param scale: should y be scaled? (useful with skewed distributions of y)
    :param average: if the same SMILES appears multiple times how should its values be averaged?
    :return: (smiles, labels) - np.arrays
    """

    # column names present in files?
    header = 0 if skip_line else None

    # load all files
    dfs = []
    for data_path in dataset_paths:
        dfs.append(pd.read_csv(data_path, delimiter=delimiter, header=header))

    # merge
    data_df = pd.concat(dfs)

    # scaling ys
    if scale is not None:
        if 'sqrt' == scale.lower().strip():
            data_df.iloc[:, y_index] = np.sqrt(data_df.iloc[:, y_index])
        elif 'log' == scale.lower().strip():
            data_df.iloc[:, y_index] = np.log(1 + data_df.iloc[:, y_index])
        else:
            raise NotImplementedError(f"Scale {scale} is not implemented.")

    # averaging when one smiles has multiple values
    if average is not None:
        smiles_col = data_df.iloc[:, smiles_index].name
        y_col = data_df.iloc[:, y_index].name

        data_df = data_df.loc[:, [smiles_col, y_col]]  # since now: smiles is 0, y_col is 1, dropping other columns
        smiles_index = 0
        y_index = 1
        if 'median' == average.lower().strip():
            data_df[y_col] = data_df[y_col].groupby(data_df[smiles_col]).transform('median')
        else:
            raise NotImplementedError(f"Averaging {average} is not implemented.")

    # breaking into x and y
    data_df = data_df.values
    data_x = data_df[:, smiles_index]
    data_y = data_df[:, y_index]

    if data_y.dtype == np.float64:
        data_y = data_y.astype(np.float32)

    return data_x, data_y


def preprocess_dataset(path, data_config, fingerprint, morgan_nbits=None):
    """Calculate representation for each smiles in the dataset."""
    if fingerprint == 'morgan':
        assert morgan_nbits is not None, 'Parameter `morgan_nbits` must be set when using Morgan fingerprint.'

    smiles, labels = load_data_from_df([path,], **data_config[csv_section])
    x = []
    y = []
    calculated_smiles = []

    # we go smiles by smiles because some compounds make rdkit throw errors
    for this_smiles, this_label in zip(smiles, labels):
        try:
            mol = Chem.MolFromSmiles(this_smiles)
            if fingerprint == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 6, nBits=morgan_nbits)
                fp = [int(i) for i in fp.ToBitString()]
            elif fingerprint == 'maccs':
                fp = MACCSkeys.GenMACCSKeys(mol)
                fp = np.array(fp)[1:]  # index 0 is unset
            elif fingerprint == 'krfp':
                fp = krfp(this_smiles)
            else:
                pass  # unknown fingerprint
            x.append(fp)
            y.append(this_label)
            calculated_smiles.append(this_smiles)
        except Exception as e:
            print('exp', e)
    return np.array(x), np.array(y), calculated_smiles


def krfp(smi):
    """Calculate Klekota-Roth fingerprint using padelpy."""
    # Warning: as this function uses padel it requires descriptors.xml to be
    # in the running directory and have KlekotaRothFingerprinter set to true

    # we don't want to copy and remove the descriptors.xml file for each smiles
    # separately, so we check if it exists and if it has the proper content
    cwd = os.getcwd()
    descriptors_filename = 'descriptors.xml'
    descriptors_hash = 'f6145f57ff346599b907b044316c4e71'

    try:
        with open(os.path.join(cwd, descriptors_filename), 'r') as desc_file:
            desc_file_content = desc_file.read()
        m = hashlib.md5()
        m.update(desc_file_content.encode('utf-8'))
        if m.hexdigest() == descriptors_hash:
            pass  # descriptors.xml exists and has the right content
        else:
            # the file exists but it has a wrong content
            raise RuntimeError("The descriptors.xml was found in the running directory but its content doesn't match the prototype content. Aborting.")
    except FileNotFoundError:
        # the file doesn't exist, we have to create it
        src_directory = os.path.dirname(os.path.realpath(__file__))
        shutil.copyfile(os.path.join(src_directory, descriptors_filename),
                        os.path.join(cwd, descriptors_filename))

    # # #
    # # # descriptors.xml exists and looks good, we can continue with calculating the representation
    # on prometheus we use SCRATCH, everywhere else the default location is fine
    with tempfile.TemporaryDirectory(dir=os.getenv('SCRATCH', None)) as tmpdirname:
        smi_file = os.path.join(tmpdirname, "molecules.smi")
        with open(smi_file, 'w') as sf:
            sf.write(smi)
        out = os.path.join(tmpdirname, "out.csv")
        padeldescriptor(mol_dir=smi_file, d_file=out, fingerprints=True, retainorder=True)
        fp = pd.read_csv(out).values[:,1:].reshape((-1)).astype(int)
    return fp


def get_cv_split(indices):
    iterator = []
    for val_indices in indices:
        train_indices = []
        for idxs in [list(range(*i)) for i in indices if i != val_indices]:
            train_indices.extend(idxs)
        val_indices = list(range(*val_indices))

        assert len(train_indices) + len(val_indices) == len(set(train_indices + val_indices))

        iterator.append((np.array(train_indices), np.array(val_indices)))
    return iterator


def log_stability(values):
    if isinstance(values, (list, tuple)):
        return [np.log(1+v) for v in values]
    else:
        # for int, float, np.array it'll work, for else - IDK
        return np.log(1+values)


def unlog_stability(values):
    if isinstance(values, (list, tuple)):
        return [np.exp(v)-1 for v in values]
    else:
        return np.exp(values) - 1


def cutoffs_metstabon(values, log_scale):
    """Changes regression to classification according to cutoffs from
    MetStabOn - Online Platform for Metabolic Stability Predictions (Podlewska & Kafel)
    values - np.array of metabolic stabilities
    log_scale - boolean indicating if the stability values are in log-scale (True) or not (False)
    """

    # y <= 0.6 - low
    # 0.6 < y <= 2.32 - medium
    # 2.32 < y - high

    low = 0
    medium = 1
    high = 2

    bottom_threshold = 0.6
    top_threshold = 2.32

    if log_scale:
        bottom_threshold = log_stability(bottom_threshold)
        top_threshold = log_stability(top_threshold)

    if isinstance(values, np.ndarray):
        classification = np.ones(values.shape, dtype=int)
        classification[values<=bottom_threshold] = low
        classification[values>top_threshold] = high
    elif isinstance(values, float):
        if values <= bottom_threshold:
            return low
        else:
            return medium if values <= top_threshold else high
    else:
        raise NotImplementedError(f"Supported types for `values` are numpy.ndarray and float, is {type(values)}.")

    return classification
