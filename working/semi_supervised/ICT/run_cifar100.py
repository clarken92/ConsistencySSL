from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from my_utils.tensorflow_utils.training import set_GPUs
from global_settings import PYTHON_EXE, RESULTS_DIR, PROCESSED_DATA_DIR

dataset = "cifar100"
data_dir = join(PROCESSED_DATA_DIR, "ComputerVision", "CIFAR100_ZCA")
train_file = join(data_dir, "train.npz")
test_file = join(data_dir, "test.npz")

output_dir = abspath(join(RESULTS_DIR, "semi_supervised", "{}".format(dataset), "ICT"))


# Default settings
# =============================== #
DEFAULT_CONFIG_10000 = {
    "num_labeled": 10000,
    "input_norm": "applied",
    "flip_horizontally": True,
    "translating_pixels": 4,

    "batch_size": 100,
    "batch_size_labeled": 25,

    "epochs": 1000,
    "steps": 280000,
    "rampup_len_step": 10000,
    "rampdown_len_step": 80000,

    "lr_max": 0.1,
    "lr_momentum": 0.9,
    "weight_decay": 0.0001,
    "gauss_noise": True,

    "ema_momentum_init": 0.99,
    "ema_momentum_final": 0.99,

    "alpha": 1.0,
    "cross_ent_l": 1.0,
    # "cent_u_coeff_max": 0.0,
    "cent_u_coeff_max": 0.1,
    "cons_coeff_max": 10.0,

    "cons_mode": "mse",

    # It is important that this one is False when #labeled is small
    # Which means we also apply consistency loss on labeled data
    "cons_4_unlabeled_only": False,
}


DEFAULT_CONFIG_5000 = DEFAULT_CONFIG_10000
DEFAULT_CONFIG_5000['num_labeled'] = 5000
# =============================== #


# Run settings
# =============================== #
run_config = {
    "output_dir": output_dir,
    "dataset": dataset,
    "train_file": train_file,
    "test_file": test_file,


    # 9310gaurav
    # ------------------------ #
    "model_name": "9310gaurav",

    # 10000
    # ------------------------ #
    "run": "0_ICT_L10000_Nesterov_alpha1.0",
    # "run": "0a_ICT_L10000_Nesterov_alpha1.0",
    # "run": "0b_ICT_L10000_Nesterov_alpha1.0",
    "num_labeled": 10000,
    "alpha": 1.0,
    # ------------------------ #

    # 5000
    # ------------------------ #
    # "run": "100_ICT_L5000_Nesterov_alpha1.0",
    # "run": "100a_ICT_L5000_Nesterov_alpha1.0",
    # "run": "100b_ICT_L5000_Nesterov_alpha1.0",
    # "num_labeled": 5000,
    # "alpha": 1.0,
    # ------------------------ #

    # "force_rm_dir": True,
}
# =============================== #

if run_config['num_labeled'] == 5000:
    config = DEFAULT_CONFIG_5000
elif run_config['num_labeled'] == 10000:
    config = DEFAULT_CONFIG_10000
else:
    raise ValueError("num_labeled={}".format(run_config['num_labeled']))

config.update(run_config)
arg_str = get_arg_string(config)
set_GPUs([0])

print("Running arguments: [{}]".format(arg_str))
run_command = "{} ./train.py {}".format(PYTHON_EXE, arg_str).strip()
subprocess.call(run_command, shell=True)