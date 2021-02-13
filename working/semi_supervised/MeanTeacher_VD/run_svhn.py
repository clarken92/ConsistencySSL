from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from my_utils.tensorflow_utils.training import set_GPUs
from global_settings import PYTHON_EXE, RESULTS_DIR, PROCESSED_DATA_DIR

dataset = "svhn"
data_dir = join(PROCESSED_DATA_DIR, "ComputerVision", dataset.upper(), "bytes")
train_file = join(data_dir, "train.npz")
test_file = join(data_dir, "test.npz")

output_dir = abspath(join(RESULTS_DIR, "semi_supervised", "{}".format(dataset), "MeanTeacher_VD"))


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
    "rampup_len_step": 10000,
    "rampdown_len_step": 80000,

    "lr_max": 0.1,
    "lr_momentum": 0.9,
    "weight_decay": 0.0001,
    # WeightNorm behave erratically so we do not use it
    "weight_norm": False,
    "gauss_noise": True,

    "ema_momentum_init": 0.99,
    "ema_momentum_final": 0.99,

    "cross_ent_l": 1.0,
    "cent_u_coeff_max": 0.0,
    "cons_coeff_max": 10.0,
    "weight_kld_coeff_max": 0.05,

    "cons_mode": "mse",
    "cons_4_unlabeled_only": False,

    "mask_weights": False,
    "cons_against_mean": True,

    # Setting this one to either True or False does not change the results
    # However, when "cons_against_mean" = False, we should set it to True
    "ema_4_log_sigma2": True,

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
    # "run": "0_MTVD_L250_Nesterov_default",
    # "num_labeled": 250,

    # "run": "1_MTVD_L250_Nesterov_default_Cons6.0",
    # "num_labeled": 250,
    # "cons_coeff_max": 6.0,

    # "run": "2_MTVD_L250_Nesterov_default_Cons3.0",
    # "num_labeled": 250,
    # "cons_coeff_max": 3.0,

    # "run": "3_MTVD_L250_Nesterov_default_Cons6.0_KLD0.02",
    # "num_labeled": 250,
    # "cons_coeff_max": 6.0,
    # "weight_kld_coeff_max": 0.02,

    # "run": "4_MTVD_L250_Nesterov_like106",
    # "run": "4a_MTVD_L250_Nesterov_like106",
    # "run": "4b_MTVD_L250_Nesterov_like106",
    # "num_labeled": 250,
    # "batch_size_labeled": 25,
    # "lr_max": 0.03,
    # "rampup_len_step": 40000,
    # "rampdown_len_step": 0,
    # "weight_decay": 0.0002,
    # "cons_coeff_max": 12.0,
    # "ema_momentum_init": 0.99,
    # "ema_momentum_final": 0.999,

    # "run": "5_MTVD_L250_Nesterov_like106_labeled15",
    # "num_labeled": 250,
    # "batch_size_labeled": 15,
    # "lr_max": 0.03,
    # "rampup_len_step": 40000,
    # "rampdown_len_step": 0,
    # "weight_decay": 0.0002,
    # "cons_coeff_max": 12.0,
    # "ema_momentum_init": 0.99,
    # "ema_momentum_final": 0.999,

    # "run": "6_MTVD_L250_Nesterov_like106_labeled5",
    # "num_labeled": 250,
    # "batch_size_labeled": 5,
    # "lr_max": 0.03,
    # "rampup_len_step": 40000,
    # "rampdown_len_step": 0,
    # "weight_decay": 0.0002,
    # "cons_coeff_max": 12.0,
    # "ema_momentum_init": 0.99,
    # "ema_momentum_final": 0.999,

    # Focus on this, achieves the best result among runs from 0 to 11
    # "run": "7_MTVD_L250_Nesterov_like106_labeled5_kld0.01",
    # "num_labeled": 250,
    # "batch_size_labeled": 5,
    # "lr_max": 0.03,
    # "rampup_len_step": 40000,
    # "rampdown_len_step": 0,
    # "weight_decay": 0.0002,
    # "cons_coeff_max": 12.0,
    # "ema_momentum_init": 0.99,
    # "ema_momentum_final": 0.999,
    # "weight_kld_coeff_max": 0.01,

    # "run": "8_MTVD_L250_Nesterov_like106_labeled5_kld0.1",
    # "num_labeled": 250,
    # "batch_size_labeled": 5,
    # "lr_max": 0.03,
    # "rampup_len_step": 40000,
    # "rampdown_len_step": 0,
    # "weight_decay": 0.0002,
    # "cons_coeff_max": 12.0,
    # "ema_momentum_init": 0.99,
    # "ema_momentum_final": 0.999,
    # "weight_kld_coeff_max": 0.1,

    # "run": "9_MTVD_L250_Nesterov_like106_labeled5_WeightDecay0.00005",
    # "num_labeled": 250,
    # "batch_size_labeled": 5,
    # "lr_max": 0.03,
    # "rampup_len_step": 40000,
    # "rampdown_len_step": 0,
    # "weight_decay": 0.00005,
    # "cons_coeff_max": 12.0,
    # "ema_momentum_init": 0.99,
    # "ema_momentum_final": 0.999,
    # "weight_kld_coeff_max": 0.05,

    # "run": "10_MTVD_L250_Nesterov_like106_labeled5_WeightDecay0.0005",
    # "num_labeled": 250,
    # "batch_size_labeled": 5,
    # "lr_max": 0.03,
    # "rampup_len_step": 40000,
    # "rampdown_len_step": 0,
    # "weight_decay": 0.0005,
    # "cons_coeff_max": 12.0,
    # "ema_momentum_init": 0.99,
    # "ema_momentum_final": 0.999,
    # "weight_kld_coeff_max": 0.05,

    # "run": "11_MTVD_L250_Nesterov_like106_labeled1",
    # "num_labeled": 250,
    # "batch_size_labeled": 1,
    # "lr_max": 0.03,
    # "rampup_len_step": 40000,
    # "rampdown_len_step": 0,
    # "weight_decay": 0.0002,
    # "cons_coeff_max": 12.0,
    # "ema_momentum_init": 0.99,
    # "ema_momentum_final": 0.999,


    "run": "12_MTVD_L250_Nesterov_LikeMT_kld0.01",
    # "run": "12a_MTVD_L250_Nesterov_LikeMT_kld0.01",
    # "run": "12b_MTVD_L250_Nesterov_LikeMT_kld0.01",
    "num_labeled": 250,
    "batch_size_labeled": 25,
    "lr_max": 0.03,
    "rampup_len_step": 40000,
    "rampdown_len_step": 0,
    "weight_decay": 0.0002,
    "cons_coeff_max": 12.0,
    "ema_momentum_init": 0.99,
    "ema_momentum_final": 0.999,
    "weight_kld_coeff_max": 0.01,
    # ------------------------ #

    # 500
    # ------------------------ #
    # "run": "100_MTVD_L500_Nesterov_default",
    # "run": "100a_MTVD_L500_Nesterov_default",
    # "run": "100b_MTVD_L500_Nesterov_default",
    # "num_labeled": 500,

    # "run": "101_MTVD_L500_Nesterov",
    # "num_labeled": 500,
    # "batch_size_labeled": 50,
    # "lr_max": 0.03,
    # "rampup_len_step": 40000,
    # "weight_decay": 0.0001,
    # "ema_momentum_init": 0.95,
    # "ema_momentum_final": 0.99,

    # "run": "102_MTVD_L500_Nesterov",
    # "num_labeled": 500,
    # "batch_size_labeled": 25,
    # "lr_max": 0.03,
    # "rampup_len_step": 40000,
    # "weight_decay": 0.0001,
    # "cons_coeff_max": 15.0,
    # "ema_momentum_init": 0.99,
    # "ema_momentum_final": 0.99,

    # "run": "103_MTVD_L500_Nesterov",
    # "num_labeled": 500,
    # "batch_size_labeled": 25,
    # "lr_max": 0.03,
    # "rampup_len_step": 40000,
    # "weight_decay": 0.0001,
    # "cons_coeff_max": 12.0,      # Reduce consistency
    # "ema_momentum_init": 0.99,
    # "ema_momentum_final": 0.99,

    # "run": "104_MTVD_L500_Nesterov",
    # "num_labeled": 500,
    # "batch_size_labeled": 25,
    # "lr_max": 0.03,
    # "rampup_len_step": 40000,
    # "weight_decay": 0.0002,     # Increase weight decay
    # "cons_coeff_max": 15.0,
    # "ema_momentum_init": 0.99,
    # "ema_momentum_final": 0.99,

    # "run": "105_MTVD_L500_Nesterov",
    # "num_labeled": 500,
    # "batch_size_labeled": 25,
    # "lr_max": 0.03,
    # "rampup_len_step": 40000,
    # "weight_decay": 0.0001,
    # "cons_coeff_max": 15.0,
    # "ema_momentum_init": 0.99,
    # "ema_momentum_final": 0.999,   # Change final momentum

    # OK, look good
    # "run": "106_MTVD_L500_Nesterov",
    # "run": "106a_MTVD_L500_Nesterov",
    # "run": "106b_MTVD_L500_Nesterov",
    # "run": "106c_MTVD_L500_Nesterov",
    # "run": "106d_MTVD_L500_Nesterov",
    # "num_labeled": 500,
    # "batch_size_labeled": 25,
    # "lr_max": 0.03,
    # "rampup_len_step": 40000,
    # "rampdown_len_step": 0,
    # "weight_decay": 0.0002,
    # "cons_coeff_max": 12.0,
    # "ema_momentum_init": 0.99,
    # "ema_momentum_final": 0.999,

    # OK, 96.42% but not as good as 106
    # "run": "107_MTVD_L500_Nesterov",
    # "num_labeled": 500,
    # "batch_size_labeled": 25,
    # "lr_max": 0.03,
    # "steps": 280000,
    # "rampup_len_step": 40000,
    # "rampdown_len_step": 80000,
    # "weight_decay": 0.0001,
    # "cons_coeff_max": 10.0,      # Reduce consistency
    # "ema_momentum_init": 0.99,
    # "ema_momentum_final": 0.99,

    # Bad, about 95.98%
    # "run": "108_MTVD_L500_Nesterov_like102_WDecay0.00003",
    # "num_labeled": 500,
    # "batch_size_labeled": 25,
    # "lr_max": 0.03,
    # "rampup_len_step": 40000,
    # "weight_decay": 0.00003,
    # "cons_coeff_max": 15.0,
    # "ema_momentum_init": 0.99,
    # "ema_momentum_final": 0.99,

    # Max 96.64%, last 96.49%, not as good as 106
    # Curve do not well behave
    # "run": "109_MTVD_L500_Nesterov_almost_like106",
    # "num_labeled": 500,
    # "batch_size_labeled": 25,
    # "lr_max": 0.03,
    # "rampup_len_step": 40000,
    # "rampdown_len_step": 80000,
    # "weight_decay": 0.0002,
    # "cons_coeff_max": 15.0,
    # "ema_momentum_init": 0.99,
    # "ema_momentum_final": 0.999,

    # "run": "110_MTVD_L500_Nesterov_like106_BatchLabeled50",
    # "run": "110a_MTVD_L500_Nesterov_like106_BatchLabeled50",
    # "num_labeled": 500,
    # "batch_size_labeled": 50,
    # "lr_max": 0.03,
    # "rampup_len_step": 40000,
    # "rampdown_len_step": 0,
    # "weight_decay": 0.0002,
    # "cons_coeff_max": 12.0,
    # "ema_momentum_init": 0.99,
    # "ema_momentum_final": 0.999,
    # ------------------------ #

    # 1000
    # ------------------------ #
    # "run": "200_MTVD_L1000_Nesterov_default",
    # "run": "200a_MTVD_L1000_Nesterov_default",
    # "run": "200b_MTVD_L1000_Nesterov_default",
    # "num_labeled": 1000,

    # "run": "201_MTVD_L1000_Nesterov_Like106",
    # "run": "201a_MTVD_L1000_Nesterov",
    # "run": "201b_MTVD_L1000_Nesterov",
    # "num_labeled": 1000,
    # "batch_size_labeled": 25,
    # "lr_max": 0.03,
    # "rampup_len_step": 40000,
    # "rampdown_len_step": 0,
    # "weight_decay": 0.0002,
    # "cons_coeff_max": 12.0,
    # "ema_momentum_init": 0.99,
    # "ema_momentum_final": 0.999,
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