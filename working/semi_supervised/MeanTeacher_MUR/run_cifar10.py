from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from my_utils.tensorflow_utils.training import set_GPUs
from global_settings import PYTHON_EXE, RESULTS_DIR, PROCESSED_DATA_DIR

dataset = "cifar10"
data_dir = join(PROCESSED_DATA_DIR, "ComputerVision", "CIFAR10_ZCA")
train_file = join(data_dir, "train.npz")
test_file = join(data_dir, "test.npz")

output_dir = abspath(join(RESULTS_DIR, "semi_supervised", "{}".format(dataset), "MeanTeacher_MUR"))


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
    # Weight Norm gives erratic behaviors so we do not use it
    "weight_norm": False,

    "ema_momentum_init": 0.99,
    "ema_momentum_final": 0.99,

    "cross_ent_l": 1.0,
    "cent_u_coeff_max": 0.0,
    "cons_coeff_max": 10.0,
    "mur_coeff_max": 4.0,

    "cons_mode": "mse",
    "cons_4_unlabeled_only": False,

    "mur_4_unlabeled_only": False,
    # r = {4, 6, 8, 10, 20, 40}
    "mur_noise_radius": 10.0,

    # lr={0.1, 1.0, 10.0}, s={2, 5, 8}
    "mur_opt_steps": 1,
    "mur_iter_mode": "grad_asc_w_lagrangian_relax",
    "mur_opt_lr": 1.0,
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

    # Iterative approximations of x*
    # ------------------------ #
    # "run": "0_MT_MUR_L1000_Nesterov_Step2_Lr1.0_LambdaGrad",
    # "run": "0a_MT_MUR_L1000_Nesterov_Step2_Lr1.0_LambdaGrad",
    # "run": "0b_MT_MUR_L1000_Nesterov_Step2_Lr1.0_LambdaGrad",
    # "num_labeled": 1000,
    # "mur_iter_mode": "grad_asc_w_lagrangian_relax",
    # "mur_opt_steps": 2,
    # "mur_opt_lr": 1.0,

    # "run": "1_MT_MUR_L1000_Nesterov_Step5_Lr1.0_LambdaGrad",
    # "run": "1a_MT_MUR_L1000_Nesterov_Step5_Lr1.0_LambdaGrad",
    # "run": "1b_MT_MUR_L1000_Nesterov_Step5_Lr1.0_LambdaGrad",
    # "num_labeled": 1000,
    # "mur_iter_mode": "grad_asc_w_lagrangian_relax",
    # "mur_opt_steps": 5,
    # "mur_opt_lr": 1.0,

    # "run": "2_MT_MUR_L1000_Nesterov_Step2_Lr1.0_PGA",
    # "run": "2a_MT_MUR_L1000_Nesterov_Step2_Lr1.0_PGA",
    # "run": "2b_MT_MUR_L1000_Nesterov_Step2_Lr1.0_PGA",
    # "num_labeled": 1000,
    # "mur_iter_mode": "proj_grad_asc",
    # "mur_opt_steps": 2,
    # "mur_opt_lr": 1.0,

    # "run": "3_MT_MUR_L1000_Nesterov_Step5_Lr1.0_PGA",
    # "run": "3a_MT_MUR_L1000_Nesterov_Step5_Lr1.0_PGA",
    # "run": "3b_MT_MUR_L1000_Nesterov_Step5_Lr1.0_PGA",
    # "num_labeled": 1000,
    # "mur_iter_mode": "proj_grad_asc",
    # "mur_opt_steps": 5,
    # "mur_opt_lr": 1.0,

    # "run": "4_MT_MUR_L1000_Nesterov_Step2_Lr0.1_LambdaGrad",
    # "run": "4a_MT_MUR_L1000_Nesterov_Step2_Lr0.1_LambdaGrad",
    # "run": "4b_MT_MUR_L1000_Nesterov_Step2_Lr0.1_LambdaGrad",
    # "num_labeled": 1000,
    # "mur_iter_mode": "grad_asc_w_lagrangian_relax",
    # "mur_opt_steps": 2,
    # "mur_opt_lr": 0.1,

    # "run": "5_MT_MUR_L1000_Nesterov_Step5_Lr0.1_LambdaGrad",
    # "run": "5a_MT_MUR_L1000_Nesterov_Step5_Lr0.1_LambdaGrad",
    # "run": "5b_MT_MUR_L1000_Nesterov_Step5_Lr0.1_LambdaGrad",
    # "num_labeled": 1000,
    # "mur_iter_mode": "grad_asc_w_lagrangian_relax",
    # "mur_opt_steps": 5,
    # "mur_opt_lr": 0.1,

    # "run": "6_MT_MUR_L1000_Nesterov_Step2_Lr10.0_LambdaGrad",
    # "run": "6a_MT_MUR_L1000_Nesterov_Step2_Lr10.0_LambdaGrad",
    # "run": "6b_MT_MUR_L1000_Nesterov_Step2_Lr10.0_LambdaGrad",
    # "num_labeled": 1000,
    # "mur_iter_mode": "grad_asc_w_lagrangian_relax",
    # "mur_opt_steps": 2,
    # "mur_opt_lr": 10.0,

    # "run": "7_MT_MUR_L1000_Nesterov_Step2_Lr0.1_PGA",
    # "run": "7a_MT_MUR_L1000_Nesterov_Step2_Lr0.1_PGA",
    # "run": "7b_MT_MUR_L1000_Nesterov_Step2_Lr0.1_PGA",
    # "num_labeled": 1000,
    # "mur_iter_mode": "proj_grad_asc",
    # "mur_opt_steps": 2,
    # "mur_opt_lr": 0.1,

    # "run": "8_MT_MUR_L1000_Nesterov_Step5_Lr0.1_PGA",
    # "run": "8a_MT_MUR_L1000_Nesterov_Step5_Lr0.1_PGA",
    # "run": "8b_MT_MUR_L1000_Nesterov_Step5_Lr0.1_PGA",
    # "num_labeled": 1000,
    # "mur_iter_mode": "proj_grad_asc",
    # "mur_opt_steps": 5,
    # "mur_opt_lr": 0.1,

    # "run": "9_MT_MUR_L1000_Nesterov_Step2_Lr10.0_PGA",
    # "run": "9a_MT_MUR_L1000_Nesterov_Step2_Lr10.0_PGA",
    # "run": "9b_MT_MUR_L1000_Nesterov_Step2_Lr10.0_PGA",
    # "num_labeled": 1000,
    # "mur_iter_mode": "proj_grad_asc",
    # "mur_opt_steps": 2,
    # "mur_opt_lr": 10.0,

    # "run": "10_MT_MUR_L1000_Nesterov_Step5_Lr10.0_LambdaGrad",
    # "run": "10a_MT_MUR_L1000_Nesterov_Step5_Lr10.0_LambdaGrad",
    # "run": "10b_MT_MUR_L1000_Nesterov_Step5_Lr10.0_LambdaGrad",
    # "num_labeled": 1000,
    # "mur_iter_mode": "grad_asc_w_lagrangian_relax",
    # "mur_opt_steps": 5,
    # "mur_opt_lr": 10.0,

    # "run": "11_MT_MUR_L1000_Nesterov_Step5_Lr10.0_PGA",
    # "run": "11a_MT_MUR_L1000_Nesterov_Step5_Lr10.0_PGA",
    # "run": "11b_MT_MUR_L1000_Nesterov_Step5_Lr10.0_PGA",
    # "num_labeled": 1000,
    # "mur_iter_mode": "proj_grad_asc",
    # "mur_opt_steps": 5,
    # "mur_opt_lr": 10.0,

    # "run": "12_MT_MUR_L1000_Nesterov_Step8_Lr1.0_LambdaGrad",
    # "run": "12a_MT_MUR_L1000_Nesterov_Step8_Lr1.0_LambdaGrad",
    # "run": "12b_MT_MUR_L1000_Nesterov_Step8_Lr1.0_LambdaGrad",
    # "num_labeled": 1000,
    # "mur_iter_mode": "grad_asc_w_lagrangian_relax",
    # "mur_opt_steps": 8,
    # "mur_opt_lr": 1.0,

    # "run": "13_MT_MUR_L1000_Nesterov_Step8_Lr1.0_PGA",
    # "run": "13a_MT_MUR_L1000_Nesterov_Step8_Lr1.0_PGA",
    # "run": "13b_MT_MUR_L1000_Nesterov_Step8_Lr1.0_PGA",
    # "run": "13c_MT_MUR_L1000_Nesterov_Step8_Lr1.0_PGA",
    # "num_labeled": 1000,
    # "mur_iter_mode": "proj_grad_asc",
    # "mur_opt_steps": 8,
    # "mur_opt_lr": 1.0,

    # "run": "14_MT_MUR_L1000_Nesterov_Step8_Lr0.1_LambdaGrad",
    # "run": "14a_MT_MUR_L1000_Nesterov_Step8_Lr0.1_LambdaGrad",
    # "run": "14b_MT_MUR_L1000_Nesterov_Step8_Lr0.1_LambdaGrad",
    # "num_labeled": 1000,
    # "mur_iter_mode": "grad_asc_w_lagrangian_relax",
    # "mur_opt_steps": 8,
    # "mur_opt_lr": 0.1,

    # "run": "15_MT_MUR_L1000_Nesterov_Step8_Lr10.0_LambdaGrad",
    # "run": "15a_MT_MUR_L1000_Nesterov_Step8_Lr10.0_LambdaGrad",
    # "run": "15b_MT_MUR_L1000_Nesterov_Step8_Lr10.0_LambdaGrad",
    # "num_labeled": 1000,
    # "mur_iter_mode": "grad_asc_w_lagrangian_relax",
    # "mur_opt_steps": 8,
    # "mur_opt_lr": 10.0,

    # "run": "16_MT_MUR_L1000_Nesterov_Step8_Lr0.1_PGA",
    # "run": "16a_MT_MUR_L1000_Nesterov_Step8_Lr0.1_PGA",
    # "run": "16b_MT_MUR_L1000_Nesterov_Step8_Lr0.1_PGA",
    # "num_labeled": 1000,
    # "mur_iter_mode": "proj_grad_asc",
    # "mur_opt_steps": 8,
    # "mur_opt_lr": 0.1,

    # "run": "17_MT_MUR_L1000_Nesterov_Step8_Lr10.0_PGA",
    # "run": "17a_MT_MUR_L1000_Nesterov_Step8_Lr10.0_PGA",
    # "run": "17b_MT_MUR_L1000_Nesterov_Step8_Lr10.0_PGA",
    # "num_labeled": 1000,
    # "mur_iter_mode": "proj_grad_asc",
    # "mur_opt_steps": 8,
    # "mur_opt_lr": 10.0,
    # ------------------------ #

    # Radius
    # ------------------------ #
    # "run": "18_MT_MUR_L1000_Nesterov_Step1_Rad4",
    # "run": "18a_MT_MUR_L1000_Nesterov_Step1_Rad4",
    # "run": "18b_MT_MUR_L1000_Nesterov_Step1_Rad4",
    # "num_labeled": 1000,
    # "mur_noise_radius": 4.0,
    # "mur_opt_steps": 1,


    # "run": "18d_MT_MUR_L1000_Nesterov_Step1_Rad4_NoInpNorm",
    # "input_norm": "none",
    # "num_labeled": 1000,
    # "mur_noise_radius": 4.0,
    # "mur_opt_steps": 1,
    # "train_file": join(PROCESSED_DATA_DIR, "ComputerVision", "CIFAR10", "bytes", "train.npz"),
    # "test_file": join(PROCESSED_DATA_DIR, "ComputerVision", "CIFAR10", "bytes", "test.npz"),


    # "run": "19_MT_MUR_L1000_Nesterov_Step1_Rad7",
    # "run": "19a_MT_MUR_L1000_Nesterov_Step1_Rad7",
    # "run": "19b_MT_MUR_L1000_Nesterov_Step1_Rad7",
    # "num_labeled": 1000,
    # "mur_noise_radius": 7.0,
    # "mur_opt_steps": 1,

    # "run": "19d_MT_MUR_L1000_Nesterov_Step1_Rad7_NoInpNorm",
    # "input_norm": "none",
    # "num_labeled": 1000,
    # "mur_noise_radius": 7.0,
    # "mur_opt_steps": 1,
    # "train_file": join(PROCESSED_DATA_DIR, "ComputerVision", "CIFAR10", "bytes", "train.npz"),
    # "test_file": join(PROCESSED_DATA_DIR, "ComputerVision", "CIFAR10", "bytes", "test.npz"),

    # "run": "20_MT_MUR_L1000_Nesterov_Step1_Rad10",
    # "run": "20a_MT_MUR_L1000_Nesterov_Step1_Rad10",
    # "run": "20b_MT_MUR_L1000_Nesterov_Step1_Rad10",
    # "run": "20c_MT_MUR_L1000_Nesterov_Step1_Rad10",
    # "num_labeled": 1000,
    # "mur_noise_radius": 10.0,
    # "mur_opt_steps": 1,

    # "run": "20d_MT_MUR_L1000_Nesterov_Step1_Rad10_NoInpNorm",
    # "input_norm": "none",
    # "num_labeled": 1000,
    # "mur_noise_radius": 10.0,
    # "mur_opt_steps": 1,
    # "train_file": join(PROCESSED_DATA_DIR, "ComputerVision", "CIFAR10", "bytes", "train.npz"),
    # "test_file": join(PROCESSED_DATA_DIR, "ComputerVision", "CIFAR10", "bytes", "test.npz"),

    # "run": "21_MT_MUR_L1000_Nesterov_Step1_Rad20",
    # "run": "21a_MT_MUR_L1000_Nesterov_Step1_Rad20",
    # "run": "21b_MT_MUR_L1000_Nesterov_Step1_Rad20",
    # "num_labeled": 1000,
    # "mur_noise_radius": 20.0,
    # "mur_opt_steps": 1,

    # "run": "21d_MT_MUR_L1000_Nesterov_Step1_Rad20_NoInpNorm",
    # "input_norm": "none",
    # "num_labeled": 1000,
    # "mur_noise_radius": 20.0,
    # "mur_opt_steps": 1,
    # "train_file": join(PROCESSED_DATA_DIR, "ComputerVision", "CIFAR10", "bytes", "train.npz"),
    # "test_file": join(PROCESSED_DATA_DIR, "ComputerVision", "CIFAR10", "bytes", "test.npz"),

    # "run": "22_MT_MUR_L1000_Nesterov_Step1_Rad40",
    # "run": "22a_MT_MUR_L1000_Nesterov_Step1_Rad40",
    # "run": "22b_MT_MUR_L1000_Nesterov_Step1_Rad40",
    # "num_labeled": 1000,
    # "mur_noise_radius": 40.0,
    # "mur_opt_steps": 1,

    # "run": "22d_MT_MUR_L1000_Nesterov_Step1_Rad40_NoInpNorm",
    # "input_norm": "none",
    # "num_labeled": 1000,
    # "mur_noise_radius": 40.0,
    # "mur_opt_steps": 1,
    # "train_file": join(PROCESSED_DATA_DIR, "ComputerVision", "CIFAR10", "bytes", "train.npz"),
    # "test_file": join(PROCESSED_DATA_DIR, "ComputerVision", "CIFAR10", "bytes", "test.npz"),

    # "run": "23_MT_MUR_L1000_Nesterov_NoCons",
    # "run": "23a_MT_MUR_L1000_Nesterov_NoCons",
    # "run": "23b_MT_MUR_L1000_Nesterov_NoCons",
    # "num_labeled": 1000,
    # "cross_ent_l": 1.0,
    # "cent_u_coeff_max": 0.0,
    # "cons_coeff_max": 0.0,
    # "mur_coeff_max": 10.0,
    #
    # "cons_mode": "mse",
    # "cons_4_unlabeled_only": False,
    #
    # "mur_4_unlabeled_only": False,
    # "mur_noise_radius": 10.0,
    #
    # "mur_opt_steps": 1,
    # "mur_iter_mode": "grad_asc_w_lagrangian_relax",
    # "mur_opt_lr": 1.0,

    # "run": "24_MT_MUR_L1000_Nesterov_Step1_Rad60",
    # "run": "24a_MT_MUR_L1000_Nesterov_Step1_Rad60",
    # "run": "24b_MT_MUR_L1000_Nesterov_Step1_Rad60",
    # "num_labeled": 1000,
    # "mur_noise_radius": 60.0,
    # "mur_opt_steps": 1,


    # "run": "25_MT_MUR_L1000_Nesterov_Step1_Rad80",
    # "run": "25a_MT_MUR_L1000_Nesterov_Step1_Rad80",
    # "run": "25b_MT_MUR_L1000_Nesterov_Step1_Rad80",
    # "num_labeled": 1000,
    # "mur_noise_radius": 80.0,
    # "mur_opt_steps": 1,
    # ------------------------ #

    # 2000
    # ------------------------ #
    # "run": "100_MT_MUR_L1000_Nesterov",
    # "num_labeled": 2000,
    # ------------------------ #

    # 4000
    # ------------------------ #
    # "run": "200_MT_MUR_L4000_Nesterov",
    # "num_labeled": 4000,
    # ------------------------ #

    # 500
    # ------------------------ #
    # "run": "300_MT_MUR_L500_Nesterov",
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