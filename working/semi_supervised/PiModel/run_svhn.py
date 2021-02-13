from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from my_utils.tensorflow_utils.training import set_GPUs
from global_settings import PYTHON_EXE, RESULTS_DIR, PROCESSED_DATA_DIR

dataset = "svhn"
data_dir = join(PROCESSED_DATA_DIR, "ComputerVision", dataset.upper(), "bytes")
train_file = join(data_dir, "train.npz")
test_file = join(data_dir, "test.npz")

output_dir = abspath(join(RESULTS_DIR, "semi_supervised", "{}".format(dataset), "PiModel"))


# Default settings
# =============================== #
DEFAULT_CONFIG_500 = {
    "num_labeled": 500,
    "input_norm": "standard",
    "flip_horizontally": False,
    "translating_pixels": 2,

    "batch_size": 100,
    "batch_size_labeled": 25,

    "epochs": 1000,
    "steps": 280000,
    "rampup_len_step": 40000,
    "rampdown_len_step": 0,

    "lr_max": 0.03,
    "lr_momentum": 0.9,
    "weight_decay": 0.0001,
    "weight_norm": False,    # WeightNorm behave erratically so we do not use it
    "gauss_noise": True,

    "cross_ent_l": 1.0,
    "cent_u_coeff_max": 0.0,
    "cons_coeff_max": 10.0,

    "cons_mode": "mse",
    "cons_4_unlabeled_only": False,
}


DEFAULT_CONFIG_250 = DEFAULT_CONFIG_500
DEFAULT_CONFIG_250['num_labeled'] = 250


DEFAULT_CONFIG_1000 = DEFAULT_CONFIG_500
DEFAULT_CONFIG_1000['num_labeled'] = 1000
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

    # 250
    # ------------------------ #
    # ------------------------ #

    # 500
    # ------------------------ #
    # "run": "100_Pi_L500_Nesterov",
    # "num_labeled": 500,
    # "lr_max": 0.03,
    # "lr_momentum": 0.9,
    # "weight_decay": 0.0001,
    # "cons_coeff_max": 10.0,

    # "run": "101_Pi_L500_Nesterov_lr0.1",
    # "num_labeled": 500,
    # "lr_max": 0.1,
    # "lr_momentum": 0.9,
    # "weight_decay": 0.0001,
    # "cons_coeff_max": 10.0,

    # "run": "102_Pi_L500_Nesterov_Decay3e-5",
    # "num_labeled": 500,
    # "lr_max": 0.03,
    # "lr_momentum": 0.9,
    # "weight_decay": 0.00003,
    # "cons_coeff_max": 10.0,

    # "run": "103_Pi_L500_Nesterov_NoDecay",
    # "num_labeled": 500,
    # "lr_max": 0.03,
    # "lr_momentum": 0.9,
    # "weight_decay": 0.0,
    # "cons_coeff_max": 10.0,

    # "run": "104_Pi_L500_Nesterov_lr0.01_Decay3e-5",
    # "num_labeled": 500,
    # "lr_max": 0.01,
    # "lr_momentum": 0.9,
    # "weight_decay": 0.00003,
    # "cons_coeff_max": 10.0,

    # "run": "105_Pi_L500_Nesterov_lr0.005_Decay3e-5",
    # "num_labeled": 500,
    # "lr_max": 0.005,
    # "lr_momentum": 0.9,
    # "weight_decay": 0.00003,
    # "cons_coeff_max": 10.0,

    # "run": "106_Pi_L500_Nesterov_lr0.005_Decay3e-5_Rampdown80k",
    # "num_labeled": 500,
    # "lr_max": 0.005,
    # "rampdown_len_step": 80000,
    # "lr_momentum": 0.9,
    # "weight_decay": 0.00003,
    # "cons_coeff_max": 10.0,

    # Fixed: "107_Pi_L500_Nesterov_like106_labeled50",
    # "run": "107_Pi_L500_Nesterov_like106_batchlabeled50",
    # "num_labeled": 500,
    # "lr_max": 0.005,
    # "rampdown_len_step": 80000,
    # "lr_momentum": 0.9,
    # "weight_decay": 0.00003,
    # "cons_coeff_max": 10.0,
    # "batch_size_labeled": 50,


    # "run": "108_Pi_L500_Nesterov",
    # "run": "108_Pi_L500_Nesterov",
    # "run": "108a_Pi_L500_Nesterov",
    # "run": "108b_Pi_L500_Nesterov",
    # "num_labeled": 500,
    # "lr_max": 0.01,
    # "weight_decay": 0.0001,
    # "cons_coeff_max": 10.0,
    # "rampdown_len_step": 80000,
    # ------------------------ #

    # 1000
    # ------------------------ #
    # "run": "200_Pi_L1000_Nesterov",
    # "run": "200a_Pi_L1000_Nesterov",
    # "run": "200b_Pi_L1000_Nesterov",
    # "num_labeled": 1000,
    # "lr_max": 0.01,
    # "weight_decay": 0.0001,
    # "cons_coeff_max": 10.0,
    # "rampdown_len_step": 80000,
    # ------------------------ #

    "force_rm_dir": True,
}
# =============================== #

if run_config['num_labeled'] == 250:
    config = DEFAULT_CONFIG_250
elif run_config['num_labeled'] == 500:
    config = DEFAULT_CONFIG_500
elif run_config['num_labeled'] == 1000:
    config = DEFAULT_CONFIG_1000
else:
    raise ValueError("num_labeled={}".format(run_config['num_labeled']))

config.update(run_config)
arg_str = get_arg_string(config)
set_GPUs([0])

print("Running arguments: [{}]".format(arg_str))
run_command = "{} ./train.py {}".format(PYTHON_EXE, arg_str).strip()
subprocess.call(run_command, shell=True)