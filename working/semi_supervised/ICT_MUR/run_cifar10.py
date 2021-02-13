from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from my_utils.tensorflow_utils.training import set_GPUs
from global_settings import PYTHON_EXE, RESULTS_DIR, PROCESSED_DATA_DIR

dataset = "cifar10"
data_dir = join(PROCESSED_DATA_DIR, "ComputerVision", "CIFAR10_ZCA")
train_file = join(data_dir, "train.npz")
test_file = join(data_dir, "test.npz")

output_dir = abspath(join(RESULTS_DIR, "semi_supervised", "{}".format(dataset), "ICT_MUR"))


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
    "weight_norm": False,         # Weight Norm gives erratic behaviors so we do not use it

    "ema_momentum_init": 0.99,
    "ema_momentum_final": 0.99,

    "alpha": 1.0,
    "cross_ent_l": 1.0,
    "cent_u_coeff_max": 0.0,      # May also try 0.1
    "cons_coeff_max": 10.0,

    "cons_mode": "mse",
    # It is important that this one is False when #labeled is small
    # Which means we also apply consistency loss on labeled data
    "cons_4_unlabeled_only": False,

    "mur_coeff_max": 4.0,
    "mur_noise_radius": 10.0,
    "mur_opt_steps": 1,
    "mur_opt_lr": 0.01,
    "mur_4_unlabeled_only": False,
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
    # "run": "0_ICT_MUR_L1000_Nesterov_alpha1.0_CEnt0.0",
    # "run": "0b_ICT_MUR_L1000_Nesterov_alpha1.0_CEnt0.0",
    # "run": "0c_ICT_MUR_L1000_Nesterov_alpha1.0_CEnt0.0",
    # "num_labeled": 1000,
    # "alpha": 1.0,
    # "cent_u_coeff_max": 0.0,

    # "run": "1_ICT_MUR_L1000_Nesterov_alpha1.0_CEnt0.1",
    # "num_labeled": 1000,
    # "alpha": 1.0,
    # "cent_u_coeff_max": 0.1,

    # "run": "2_ICT_MUR_L1000_Nesterov_like0_Rad6.0",
    # "num_labeled": 1000,
    # "alpha": 1.0,
    # "cent_u_coeff_max": 0.0,
    # "mur_noise_radius": 6.0,

    # "run": "3_ICT_MUR_L1000_Nesterov_like0_Rad15.0",
    # "num_labeled": 1000,
    # "alpha": 1.0,
    # "cent_u_coeff_max": 0.0,
    # "mur_noise_radius": 15.0,
    #
    # "run": "4_ICT_MUR_L1000_Nesterov_like0_Rad4.0",
    # "num_labeled": 1000,
    # "alpha": 1.0,
    # "cent_u_coeff_max": 0.0,
    # "mur_noise_radius": 4.0,

    # "run": "5_ICT_MUR_L1000_Nesterov_like0_Rad20.0",
    # "run": "5a_ICT_MUR_L1000_Nesterov_like0_Rad20.0",
    # "run": "5b_ICT_MUR_L1000_Nesterov_like0_Rad20.0",
    # "num_labeled": 1000,
    # "alpha": 1.0,
    # "cent_u_coeff_max": 0.0,
    # "mur_noise_radius": 20.0,

    # "run": "6_ICT_MUR_L1000_Nesterov_like0_AttrCoeff10.0",
    # "num_labeled": 1000,
    # "alpha": 1.0,
    # "cent_u_coeff_max": 0.0,
    # "mur_noise_radius": 10.0,
    # "mur_coeff_max": 10.0,
    # ------------------------ #

    # 2000
    # ------------------------ #
    # "run": "101_ICT_MUR_L2000_Nesterov_like0_Rad20.0",
    # "run": "101a_ICT_MUR_L2000_Nesterov_like0_Rad20.0",
    # "run": "101b_ICT_MUR_L2000_Nesterov_like0_Rad20.0",
    # "num_labeled": 2000,
    # "alpha": 1.0,
    # "cent_u_coeff_max": 0.0,
    # "mur_noise_radius": 20.0,
    # ------------------------ #

    # 4000
    # ------------------------ #
    # "run": "201_ICT_MUR_L4000_Nesterov_like0_Rad20.0",
    # "run": "201a_ICT_MUR_L4000_Nesterov_like0_Rad20.0",
    # "run": "201b_ICT_MUR_L4000_Nesterov_like0_Rad20.0",
    # "num_labeled": 4000,
    # "alpha": 1.0,
    # "cent_u_coeff_max": 0.0,
    # "mur_noise_radius": 20.0,
    # ------------------------ #

    # 500
    # ------------------------ #
    # "run": "301_ICT_MUR_L500_Nesterov_like0_Rad20.0",
    # "run": "301a_ICT_MUR_L500_Nesterov_like0_Rad20.0",
    # "run": "301b_ICT_MUR_L500_Nesterov_like0_Rad20.0",
    # "num_labeled": 500,
    # "alpha": 1.0,
    # "cent_u_coeff_max": 0.0,
    # "mur_noise_radius": 20.0,
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