from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from my_utils.tensorflow_utils.training import set_GPUs
from global_settings import PYTHON_EXE, RESULTS_DIR, PROCESSED_DATA_DIR

dataset = "cifar10"
data_dir = join(PROCESSED_DATA_DIR, "ComputerVision", "CIFAR10_ZCA")
train_file = join(data_dir, "train.npz")
test_file = join(data_dir, "test.npz")

output_dir = abspath(join(RESULTS_DIR, "semi_supervised", "{}".format(dataset), "PiModel"))


# Default settings
# =============================== #
DEFAULT_CONFIG_1000 = {
    "num_labeled": 1000,
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
    "weight_norm": False,
    "gauss_noise": True,

    "cross_ent_l": 1.0,
    "cent_u_coeff_max": 0.0,
    "cons_coeff_max": 10.0,

    "cons_mode": "mse",
    "cons_4_unlabeled_only": False,
}


DEFAULT_CONFIG_2000 = DEFAULT_CONFIG_1000
DEFAULT_CONFIG_2000['num_labeled'] = 2000


DEFAULT_CONFIG_4000 = DEFAULT_CONFIG_1000
DEFAULT_CONFIG_4000['num_labeled'] = 4000


DEFAULT_CONFIG_500 = DEFAULT_CONFIG_1000
DEFAULT_CONFIG_500['num_labeled'] = 500
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

    # 1000
    # ------------------------ #
    # "run": "0_Pi_L1000_Nesterov",
    # "num_labeled": 1000,
    # ------------------------ #

    # 2000
    # ------------------------ #
    # "run": "100_Pi_L2000_Neterov",
    # "num_labeled": 2000,
    # ------------------------ #

    # 4000
    # ------------------------ #
    # "run": "200_Pi_L4000_Nesterov",
    # "num_labeled": 4000,
    # ------------------------ #

    # 500
    # ------------------------ #
    # "run": "300_Pi_L500_Nesterov",
    # "num_labeled": 500,
    # ------------------------ #

    "force_rm_dir": True,
}
# =============================== #

if run_config['num_labeled'] == 500:
    config = DEFAULT_CONFIG_500
elif run_config['num_labeled'] == 1000:
    config = DEFAULT_CONFIG_1000
elif run_config['num_labeled'] == 2000:
    config = DEFAULT_CONFIG_2000
elif run_config['num_labeled'] == 4000:
    config = DEFAULT_CONFIG_4000
else:
    raise ValueError("num_labeled={}".format(run_config['num_labeled']))

config.update(run_config)
arg_str = get_arg_string(config)
set_GPUs([0])

print("Running arguments: [{}]".format(arg_str))
run_command = "{} ./train.py {}".format(PYTHON_EXE, arg_str).strip()
subprocess.call(run_command, shell=True)