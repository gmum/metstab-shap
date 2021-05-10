from configparser import ConfigParser
from distutils.util import strtobool

from tpot import TPOTRegressor, TPOTClassifier

utils_section = "UTILS"
csv_section = "CSV"
metrics_section = "METRICS"
force_classification_metrics_section = "FORCE_CLASSIFICATION_METRICS"


def str_to_bool(val):
    # distutils.util.strtobool returns zeros and ones instead of bool...
    v = strtobool(val)
    if v == 0:
        return False
    elif v == 1:
        return True
    else:
        raise ValueError


def read_config(fpath):
    config = ConfigParser()
    with open(fpath, 'r') as f:
        config.read_file(f)
    return config._sections


def parse_model_config(config_path):
    config = read_config(config_path)
    return config


def parse_shap_config(config_path):
    config = read_config(config_path)
    config[utils_section]["k"] = int(config[utils_section]["k"])
    return config


def parse_data_config(config_path):
    config = read_config(config_path)

    config[utils_section]["cv"] = str_to_bool(config[utils_section]["cv"])
    config[csv_section]["smiles_index"] = int(config[csv_section]["smiles_index"])
    config[csv_section]["y_index"] = int(config[csv_section]["y_index"])
    config[csv_section]["skip_line"] = str_to_bool(config[csv_section]["skip_line"])
    config[utils_section]["calculate_parity"] = str_to_bool(config[utils_section]["calculate_parity"])
    config[utils_section]["calculate_rocauc"] = str_to_bool(config[utils_section]["calculate_rocauc"])

    if config[csv_section]["delimiter"] == '\\t' or config[csv_section]["delimiter"] == 'tab':
        config[csv_section]["delimiter"] = '\t'

    return config


def parse_representation_config(config_path):
    config = read_config(config_path)

    if config[utils_section]['morgan_nbits'] == "None":
        config[utils_section]['morgan_nbits'] = None
    else:
        config[utils_section]['morgan_nbits'] = int(config[utils_section]['morgan_nbits'])

    return config


def parse_task_config(config_path):
    config = read_config(config_path)

    if config[utils_section]['tpot_model']=='TPOTClassifier':
        config[utils_section]['tpot_model'] = TPOTClassifier
    elif config[utils_section]['tpot_model']=='TPOTRegressor':
        config[utils_section]['tpot_model'] = TPOTRegressor
    else:
        raise ValueError(f"TPOT models are TPOTClassifier and TPOTRegressor but {config[utils_section]['tpot_model']} was given.")

    if 'cutoffs' in config[utils_section]:
        if config[utils_section]['cutoffs'] == 'metstabon':
            import metstab_shap.data as data
            config[utils_section]['cutoffs'] = data.cutoffs_metstabon
        else:
            raise NotImplementedError("Only metstabon cutoffs are implemented.")

    return config


def parse_tpot_config(config_path):
    config = read_config(config_path)
    config[utils_section]['n_jobs'] = int(config[utils_section]['n_jobs'])
    config[utils_section]['max_time_mins'] = int(config[utils_section]['max_time_mins'])
    config[utils_section]['minimal_number_of_models'] = int(config[utils_section]['minimal_number_of_models'])

    return config
