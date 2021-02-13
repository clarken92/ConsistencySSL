from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from my_utils.tensorflow_utils.training import set_GPUs
from global_settings import PYTHON_EXE, RESULTS_DIR, PROCESSED_DATA_DIR

dataset = "cifar10"
data_dir = join(PROCESSED_DATA_DIR, "ComputerVision", "CIFAR10_ZCA")
train_file = join(data_dir, "train.npz")
test_file = join(data_dir, "test.npz")

output_dir = abspath(join(RESULTS_DIR, "semi_supervised", "{}".format(dataset), "MeanTeacher_VD"))


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

    "ema_momentum_init": 0.99,
    "ema_momentum_final": 0.99,

    "cross_ent_l": 1.0,
    # "cent_u_coeff_max": 0.0,
    "cent_u_coeff_max": 0.1,
    "cons_coeff_max": 10.0,
    "weight_kld_coeff_max": 0.05,

    "cons_mode": "mse",

    # It is important that this one is False when #labeled is small
    # Which means we also apply consistency loss on labeled data
    "cons_4_unlabeled_only": False,

    # We should not mask weights
    "mask_weights": False,

    # Setting this one to either True or False does not change the results
    # However, when "cons_against_mean" = False, we should set it to True
    "ema_4_log_sigma2": True,
    "cons_against_mean": True,
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
    # "run": "0_MTVD_L1000_Nesterov",
    # "num_labeled": 1000,
    # "lr_max": 0.05,
    # "lr_momentum": 0.9,
    # "weight_norm": False,
    # "weight_decay": 0.0001,

    # Similar to 0
    # "run": "0a_MTVD_L1000_Nesterov_Cons15",
    # "num_labeled": 1000,
    # "lr_max": 0.05,
    # "lr_momentum": 0.9,
    # "weight_norm": False,
    # "weight_decay": 0.0001,
    # "cons_coeff_max": 15.0,

    # Similar to 0
    # "run": "0b_MTVD_L1000_Nesterov_Cons5",
    # "num_labeled": 1000,
    # "lr_max": 0.05,
    # "lr_momentum": 0.9,
    # "weight_norm": False,
    # "weight_decay": 0.0001,
    # "cons_coeff_max": 5.0,

    # "run": "0c_MTVD_L1000_Nesterov_lr0.1",
    # "num_labeled": 1000,
    # "lr_max": 0.1,
    # "lr_momentum": 0.9,
    # "weight_norm": False,
    # "weight_decay": 0.0001,

    # "run": "1_MTVD_L1000_Nesterov",
    # "num_labeled": 1000,
    # "lr_max": 0.05,
    # "lr_momentum": 0.9,
    # "weight_norm": False,
    # "weight_decay": 0.0001,
    # "rampup_len_step": 20000,
    # "ema_momentum_final": 0.99,

    # 78.01% (bad)
    # "run": "2_MTVD_L1000_Nesterov",
    # "num_labeled": 1000,
    # "lr_max": 0.01,
    # "lr_momentum": 0.9,
    # "weight_norm": False,
    # "weight_decay": 0.0001,
    # "rampup_len_step": 10000,
    # "ema_momentum_final": 0.99,
    # "steps": 280000,
    # "rampdown_len_step": 80000,

    # 83.73% (best)
    # "run": "3_MTVD_L1000_Nesterov_lr0.1",
    # "run": "3a_MTVD_L1000_Nesterov_lr0.1",
    # "run": "3b_MTVD_L1000_Nesterov_lr0.1",
    # "run": "3c_MTVD_L1000_Nesterov_lr0.1",
    # "run": "3d_MTVD_L1000_Nesterov_lr0.1",
    # "num_labeled": 1000,
    # "lr_max": 0.1,
    # "lr_momentum": 0.9,
    # "weight_decay": 0.0001,
    # "weight_norm": False,
    # "ema_momentum_final": 0.99,
    # "rampup_len_step": 10000,
    # "rampdown_len_step": 80000,
    # "steps": 280000,

    # Change a lot of things
    # 79.83% (bad)
    # "run": "4_MTVD_L1000_Nesterov_lr0.1",
    # "num_labeled": 1000,
    # "lr_max": 0.1,
    # "lr_momentum": 0.9,
    # "weight_decay": 0.0001,
    # "weight_norm": False,
    # "ema_momentum_init": 0.97,
    # "ema_momentum_final": 0.97,   # Change compared to 3
    # "rampup_len_step": 5000,      # Change compared to 3
    # "rampdown_len_step": 80000,
    # "steps": 280000,
    # "batch_size_labeled": 50,     # Change compared to 3

    # 83.53% (second best, lower than 3) but very erratic
    # "run": "5_MTVD_L1000_Nesterov_like3_decay0.0005",
    # "run": "5a_MTVD_L1000_Nesterov_like3_decay0.0005",
    # "run": "5b_MTVD_L1000_Nesterov_like3_decay0.0005",
    # "run": "5c_MTVD_L1000_Nesterov_like3_decay0.0005",
    # "run": "5d_MTVD_L1000_Nesterov_like3_decay0.0005",
    # "num_labeled": 1000,
    # "lr_max": 0.1,
    # "lr_momentum": 0.9,
    # "weight_decay": 0.0005,       # Change compared to 3
    # "weight_norm": False,
    # "ema_momentum_final": 0.99,
    # "rampup_len_step": 10000,
    # "rampdown_len_step": 80000,
    # "steps": 280000,

    # # Like 3, with weight decay reduced to 3e-5
    # Not as good as 3 (80.48%)
    # "run": "6_MTVD_L1000_Nesterov_lr0.1_3e-5",
    # "run": "6a_MTVD_L1000_Nesterov_lr0.1_3e-5",
    # "run": "6b_MTVD_L1000_Nesterov_lr0.1_3e-5",
    # "run": "6c_MTVD_L1000_Nesterov_lr0.1_3e-5",
    # "run": "6d_MTVD_L1000_Nesterov_lr0.1_3e-5",
    # "num_labeled": 1000,
    # "lr_max": 0.1,
    # "lr_momentum": 0.9,
    # "weight_decay": 0.00003,      # Change compared to 3
    # "weight_norm": False,
    # "ema_momentum_final": 0.99,
    # "rampup_len_step": 10000,
    # "rampdown_len_step": 80000,
    # "steps": 280000,

    # Least erratic but the results is NOT as good as with weight decay
    # Worst (76.45%)
    # "run": "7_MTVD_L1000_Nesterov_like3_noDecay",
    # "run": "7a_MTVD_L1000_Nesterov_like3_noDecay",
    # "run": "7b_MTVD_L1000_Nesterov_like3_noDecay",
    # "run": "7c_MTVD_L1000_Nesterov_like3_noDecay",
    # "run": "7d_MTVD_L1000_Nesterov_like3_noDecay",
    # "run": "7e_MTVD_L1000_Nesterov_like3_noDecay",
    # "num_labeled": 1000,
    # "lr_max": 0.1,
    # "lr_momentum": 0.9,
    # "weight_decay": 0.0,          # Change compared to 3
    # "weight_norm": False,
    # "ema_momentum_final": 0.99,
    # "rampup_len_step": 10000,
    # "rampdown_len_step": 80000,
    # "steps": 280000,

    # ************************************* #
    # Updated default setting to match run 3
    # ************************************* #

    # 82.22%
    # "run": "8_MTVD_L1000_Nesterov_lr0.1_lrMomen0.97",
    # "num_labeled": 1000,
    # "lr_max": 0.1,
    # "lr_momentum": 0.97,
    # "weight_decay": 0.0001,
    # "weight_norm": False,
    # "ema_momentum_final": 0.99,
    # "rampup_len_step": 10000,
    # "rampdown_len_step": 80000,
    # "steps": 280000,

    # Still not very good (82.38%)
    # "run": "9_MTVD_L1000_Nesterov_lr0.1_lrMomen0.8",
    # "num_labeled": 1000,
    # "lr_max": 0.1,
    # "lr_momentum": 0.8,
    # "weight_norm": False,
    # "weight_decay": 0.0001,
    # "rampup_len_step": 10000,
    # "ema_momentum_final": 0.99,
    # "steps": 280000,
    # "rampdown_len_step": 80000,

    # Bad (80.47%)
    # "run": "10_MTVD_L1000_Nesterov_lr0.1_EMA0.95",
    # "num_labeled": 1000,
    # "lr_max": 0.1,
    # "lr_momentum": 0.9,
    # "weight_norm": False,
    # "weight_decay": 0.0001,
    # "rampup_len_step": 10000,
    # "ema_momentum_final": 0.95,
    # "steps": 280000,
    # "rampdown_len_step": 80000,

    # 82.72%
    # "run": "11_MTVD_L1000_Like3_NoGauss",
    # "num_labeled": 1000,
    # "lr_max": 0.1,
    # "lr_momentum": 0.9,
    # "weight_norm": False,
    # "weight_decay": 0.0001,
    # "rampup_len_step": 10000,
    # "ema_momentum_final": 0.99,
    # "steps": 280000,
    # "rampdown_len_step": 80000,
    # "gauss_noise": False,

    # "run": "12_MTVD_L1000_Nesterov_like3_CEnt0.0",
    # "run": "12a_MTVD_L1000_Nesterov_like3_CEnt0.0",
    # "run": "12b_MTVD_L1000_Nesterov_like3_CEnt0.0",
    # "num_labeled": 1000,
    # "lr_max": 0.1,
    # "lr_momentum": 0.9,
    # "weight_decay": 0.0001,
    # "weight_norm": False,
    # "ema_momentum_final": 0.99,
    # "rampup_len_step": 10000,
    # "rampdown_len_step": 80000,
    # "steps": 280000,
    # "cent_u_coeff_max": 0.0,

    # "run": "13_MTVD_L1000_Nesterov_like12_CEnt0.0_KLD0.01",
    # "run": "13a_MTVD_L1000_Nesterov_like12_CEnt0.0_KLD0.01",
    # "run": "13b_MTVD_L1000_Nesterov_like12_CEnt0.0_KLD0.01",
    # "num_labeled": 1000,
    # "weight_kld_coeff_max": 0.01,
    # "cent_u_coeff_max": 0.0,

    # "run": "14_MTVD_L1000_Nesterov_like12_CEnt0.0_KLD0.1",
    # "run": "14a_MTVD_L1000_Nesterov_like12_CEnt0.0_KLD0.1",
    # "run": "14b_MTVD_L1000_Nesterov_like12_CEnt0.0_KLD0.1",
    # "num_labeled": 1000,
    # "weight_kld_coeff_max": 0.1,
    # "cent_u_coeff_max": 0.0,

    # "run": "15_MTVD_L1000_Nesterov_like12_CEnt0.0_KLD0.005",
    # "run": "15a_MTVD_L1000_Nesterov_like12_CEnt0.0_KLD0.005",
    # "run": "15b_MTVD_L1000_Nesterov_like12_CEnt0.0_KLD0.005",
    # "num_labeled": 1000,
    # "weight_kld_coeff_max": 0.01,
    # "cent_u_coeff_max": 0.0,

    # "run": "16_MTVD_L1000_Nesterov_like12_CEnt0.0_KLD0.5",
    # "run": "16a_MTVD_L1000_Nesterov_like12_CEnt0.0_KLD0.5",
    # "run": "16b_MTVD_L1000_Nesterov_like12_CEnt0.0_KLD0.5",
    # "num_labeled": 1000,
    # "weight_kld_coeff_max": 0.5,
    # "cent_u_coeff_max": 0.0,
    # ------------------------ #

    # 2000
    # ------------------------ #
    # "run": "100_MTVD_L2000_Neterov",
    # "run": "100a_MTVD_L2000_Neterov",
    # "run": "100b_MTVD_L2000_Neterov",
    # "num_labeled": 2000,
    # ------------------------ #

    # 4000
    # ------------------------ #
    # "run": "200_MTVD_L4000_Nesterov",
    # "run": "200a_MTVD_L4000_Nesterov",
    # "run": "200b_MTVD_L4000_Nesterov",
    # "num_labeled": 4000,
    # ------------------------ #

    # 500
    # ------------------------ #
    # "run": "300_MTVD_L500_Nesterov",
    # "run": "300a_MTVD_L500_Nesterov",
    # "run": "300b_MTVD_L500_Nesterov",
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