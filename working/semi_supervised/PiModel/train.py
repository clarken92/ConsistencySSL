import argparse
import os
from time import time
import pprint

import numpy as np
import tensorflow as tf

from components.semi_supervised.general import InputPerturber
from components.semi_supervised.cnn13 import MainClassifier_9310gaurav
from models.semi_supervised.pi import PiModel

from my_utils.python_utils.general import make_dir_if_not_exist
from my_utils.python_utils.arg_parsing import save_args, str2bool
from my_utils.python_utils.datasets import SimpleDataset4SSL, SimpleDataset
from my_utils.python_utils.data.normalization import Standardization, ZCA
from my_utils.python_utils.training import iterate_data, \
    ContinuousIndexSampler, ContinuousIndexSamplerGroup, BestResultsTracker
from my_utils.python_utils.annealing import SigmoidRampup, SigmoidRampdown
from my_utils.python_utils.image import uint8_to_binary_float

from my_utils.tensorflow_utils.general import HyperParamUpdater
from my_utils.tensorflow_utils.training.helper import SimpleTrainHelper, SimpleParamPrinter
from my_utils.tensorflow_utils.training.summary import ScalarSummarizer, custom_tf_scalar_summaries


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--train_file', type=str, required=True)
parser.add_argument('--test_file', type=str, required=True)

parser.add_argument('--model_name', type=str, default="9310gaurav", choices=["9310gaurav"])

parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--steps', type=int, default=280000)
parser.add_argument('--rampup_len_step', type=int, default=10000)
parser.add_argument('--rampdown_len_step', type=int, default=80000)

parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--batch_size_labeled', type=int, default=25)

parser.add_argument('--seed', type=int, default=1003)
parser.add_argument('--num_labeled', type=int, default=250)

# For data augmentation
# Input norm should be 'standard'
parser.add_argument('--input_norm', type=str, default="standard",
                    choices=["applied", "none", "zca", "standard"])
parser.add_argument('--flip_horizontally', type=str2bool, default=False)
# Translating pixels should be 4
parser.add_argument('--translating_pixels', type=int, default=2,
                    help="Translate image pixels by the value in "
                         "[-translating_pixels, translating_pixels]")

# For Adam optimizer
parser.add_argument('--lr_max', type=float, default=0.03)
parser.add_argument('--lr_momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=-1.0)
parser.add_argument('--weight_norm', type=str2bool, default=False)
parser.add_argument('--gauss_noise', type=str2bool, default=True)

parser.add_argument('--cons_mode', type=str, default="mse",
                    choices=["mse", "kld", "rev_kld", "2rand"])
parser.add_argument('--cons_4_unlabeled_only', type=str2bool, default=False)

parser.add_argument('--cross_ent_l', type=float, default=1.0)
parser.add_argument('--cent_u_coeff_max', type=float, default=0.0)
parser.add_argument('--cons_coeff_max', type=float, default=10.0)

parser.add_argument('--eval_freq', type=int, default=2000)
parser.add_argument('--log_freq', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=100)
parser.add_argument('--num_save', type=int, default=1)
parser.add_argument('--num_save_best', type=int, default=3)

parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--run', type=str, required=True)
parser.add_argument('--force_rm_dir', type=str2bool, default="False")


def main(args):
    # Create output directory
    # ===================================== #
    args.output_dir = os.path.join(args.output_dir, args.model_name, args.run)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        if args.force_rm_dir:
            import shutil
            shutil.rmtree(args.output_dir, ignore_errors=True)
            print("Removed '{}'".format(args.output_dir))
        else:
            raise ValueError("Output directory '{}' existed. 'force_rm_dir' "
                             "must be set to True!".format(args.output_dir))
        os.mkdir(args.output_dir)

    save_args(os.path.join(args.output_dir, 'config.json'), args)
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(args.__dict__)
    # ===================================== #

    # Specify data
    # ===================================== #
    if args.dataset == "mnist":
        x_shape = [28, 28, 1]
    elif args.dataset == "mnist_3" or args.dataset == "mnistm":
        x_shape = [28, 28, 3]
    elif args.dataset == "svhn" or args.dataset == "cifar10" or args.dataset == "cifar100":
        x_shape = [32, 32, 3]
    else:
        raise ValueError("Do not support dataset '{}'!".format(args.dataset))

    if args.dataset == "cifar100":
        num_classes = 100
    else:
        num_classes = 10

    print("x_shape: {}".format(x_shape))
    print("num_classes: {}".format(num_classes))
    # ===================================== #

    # Load data
    # ===================================== #
    print("Loading {}!".format(args.dataset))
    train_loader = SimpleDataset4SSL()
    train_loader.load_npz_data(args.train_file)
    train_loader.create_ssl_data(args.num_labeled, num_classes=num_classes,
                                 shuffle=True, seed=args.seed)
    if args.input_norm != "applied":
        train_loader.x = uint8_to_binary_float(train_loader.x)
    else:
        print("Input normalization has been applied on train data!")

    test_loader = SimpleDataset()
    test_loader.load_npz_data(args.test_file)
    if args.input_norm != "applied":
        test_loader.x = uint8_to_binary_float(test_loader.x)
    else:
        print("Input normalization has been applied on test data!")

    print("train_l/train_u/test: {}/{}/{}".format(
        train_loader.num_labeled_data,
        train_loader.num_unlabeled_data,
        test_loader.num_data))

    # import matplotlib.pyplot as plt
    # print("train_l.y[:10]: {}".format(train_l.y[:10]))
    # print("train_u.y[:10]: {}".format(train_u.y[:10]))
    # print("test.y[:10]: {}".format(test.y[:10]))
    # fig, axes = plt.subplots(3, 5)
    # for i in range(5):
    #     axes[0][i].imshow(train_l.x[i])
    #     axes[1][i].imshow(train_u.x[i])
    #     axes[2][i].imshow(test.x[i])
    # plt.show()

    if args.dataset == "mnist":
        train_loader.x = np.expand_dims(train_loader.x, axis=-1)
        test_loader.x = np.expand_dims(test_loader.x, axis=-1)
    elif args.dataset == "mnist_3":
        train_loader.x = np.stack([train_loader.x, train_loader.x, train_loader.x], axis=-1)
        test_loader.x = np.stack([test_loader.x, test_loader.x, test_loader.x], axis=-1)

    # Data Preprocessing + Augmentation
    # ------------------------------------- #
    if args.input_norm == 'none' or args.input_norm == 'applied':
        print("Do not apply any normalization!")
    elif args.input_norm == 'zca':
        print("Apply ZCA whitening on data!")
        normalizer = ZCA()
        normalizer.fit(train_loader.x, eps=1e-5)
        train_loader.x = normalizer.transform(train_loader.x)
        test_loader.x = normalizer.transform(test_loader.x)
    elif args.input_norm == 'standard':
        print("Apply Standardization on data!")
        normalizer = Standardization()
        normalizer.fit(train_loader.x)
        train_loader.x = normalizer.transform(train_loader.x)
        test_loader.x = normalizer.transform(test_loader.x)
    else:
        raise ValueError("Do not support 'input_norm'={}".format(args.input_norm))
    # ------------------------------------- #
    # ===================================== #

    # Hyperparameters
    # ===================================== #
    hyper_updater = HyperParamUpdater(['lr', 'cent_u_coeff', 'cons_coeff'],
                                      [args.lr_max, args.cent_u_coeff_max,
                                       args.cons_coeff_max],
                                      scope='moving_hyperparams')
    # ===================================== #

    # Build model
    # ===================================== #
    # IMPORTANT: Remember to test with No Gaussian Noise
    print("args.gauss_noise: {}".format(args.gauss_noise))

    if args.model_name == "9310gaurav":
        if args.weight_norm:
            raise ValueError("Since weight norm causes erratic "
                             "performance of SVDMeanTeacher so we do not use weight norm!")
        else:
            main_classifier = MainClassifier_9310gaurav(
                num_classes=num_classes, use_gauss_noise=args.gauss_noise)
    else:
        raise ValueError("Do not support model_name='{}'!".format(args.model_name))

    # Input Perturber
    # ------------------------------------- #
    # Input perturber only performs 'translating_pixels' (Both CIFAR-10 and SVHN) here
    input_perturber = InputPerturber(
        normalizer=None,  # We do not use normalizer here!
        flip_horizontally=args.flip_horizontally,
        flip_vertically=False,  # We do not flip images vertically!
        translating_pixels=args.translating_pixels,
        noise_std=0.0)  # We do not add noise here!
    # ------------------------------------- #

    # Main model
    # ------------------------------------- #
    model = PiModel(x_shape=x_shape, y_shape=num_classes,
                    main_classifier=main_classifier, input_perturber=input_perturber,
                    cons_mode=args.cons_mode,
                    cons_4_unlabeled_only=args.cons_4_unlabeled_only,
                    weight_decay=args.weight_decay)

    loss_coeff_dict = {
        'cross_ent_l': args.cross_ent_l,
        'cond_ent_u': hyper_updater.variables['cent_u_coeff'],
        'cons': hyper_updater.variables['cons_coeff'],
    }

    model.build(loss_coeff_dict)
    SimpleParamPrinter.print_all_params_list(trainable_only=False)
    # ------------------------------------- #
    # ===================================== #

    # Build optimizer
    # ===================================== #
    losses = model.get_loss()
    train_params = model.get_train_params()
    opt_AE = tf.train.MomentumOptimizer(learning_rate=hyper_updater.variables['lr'],
                                        momentum=args.lr_momentum, use_nesterov=True)

    # Contain both batch norm update and teacher param update
    update_ops = model.get_all_update_ops()
    print("update_ops: {}".format(update_ops))
    with tf.control_dependencies(update_ops):
        train_op_AE = opt_AE.minimize(loss=losses['loss'], var_list=train_params['loss'])
    # ===================================== #

    # Create directories
    # ===================================== #
    asset_dir = make_dir_if_not_exist(os.path.join(args.output_dir, "asset"))
    img_dir = make_dir_if_not_exist(os.path.join(asset_dir, "img"))
    log_dir = make_dir_if_not_exist(os.path.join(args.output_dir, "log"))
    train_log_file = os.path.join(log_dir, "train.log")

    summary_dir = make_dir_if_not_exist(os.path.join(args.output_dir, "summary_tf"))
    model_dir = make_dir_if_not_exist(os.path.join(args.output_dir, "model_tf"))
    # ===================================== #

    # Create session
    # ===================================== #
    config_proto = tf.ConfigProto(allow_soft_placement=True)
    config_proto.gpu_options.allow_growth = True
    config_proto.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config_proto)

    train_helper = SimpleTrainHelper(log_dir=summary_dir, save_dir=model_dir,
                                     max_to_keep=args.num_save, max_to_keep_best=args.num_save_best)
    train_helper.initialize(sess, init_variables=True, create_summary_writer=True)
    # ===================================== #

    # Start training
    # ===================================== #
    # Summarizer
    # ------------------------------------- #
    fetch_keys_AE_l = ['acc_y_l', 'cross_ent_l']
    fetch_keys_AE_u = ['acc_y_u', 'cond_ent_u', 'cons', 'l2_reg']
    fetch_keys_AE = fetch_keys_AE_l + fetch_keys_AE_u
    train_summarizer = ScalarSummarizer([(key, 'mean') for key in fetch_keys_AE])

    fetch_keys_test = ['acc_y']
    eval_summarizer = ScalarSummarizer([(key, 'mean') for key in fetch_keys_test])
    # ------------------------------------- #

    # Data sampler
    # ------------------------------------- #
    # The number of labeled data varies during training
    if args.batch_size_labeled <= 0:
        sampler = ContinuousIndexSampler(
            train_loader.num_data, args.batch_size, shuffle=True)
        sampling_separately = False
        print("batch_size_l, batch_size_u vary but their sum={}!".format(args.batch_size))

    elif 0 < args.batch_size_labeled < args.batch_size:
        batch_size_l = args.batch_size_labeled
        batch_size_u = args.batch_size - args.batch_size_labeled
        print("batch_size_l/batch_size_u: {}/{}".format(batch_size_l, batch_size_u))

        # IMPORTANT: Here we must use 'train_loader.labeled_ids' and 'train_loader.unlabeled_ids',
        # NOT 'train_loader.num_labeled_data' and 'train_loader.num_unlabeled_data'
        sampler_l = ContinuousIndexSampler(
            train_loader.labeled_ids, batch_size_l, shuffle=True)
        sampler_u = ContinuousIndexSampler(
            train_loader.unlabeled_ids, batch_size_u, shuffle=True)
        sampler = ContinuousIndexSamplerGroup(sampler_l, sampler_u)
        sampling_separately = True

    else:
        raise ValueError("'args.batch_size_labeled' must be in ({}, {})!".format(0, args.batch_size))
    # ------------------------------------- #

    # Annealer
    # ------------------------------------- #
    sigmoid_rampup_annealer = SigmoidRampup(args.rampup_len_step)
    sigmoid_rampdown_annealer = SigmoidRampdown(args.rampdown_len_step, args.steps)
    # ------------------------------------- #

    # Results Tracker
    # ------------------------------------- #
    tracker = BestResultsTracker([('acc_y', 'greater')], num_best=args.num_save_best)
    # ------------------------------------- #

    import math
    batches_per_epoch = int(math.ceil(train_loader.num_data / args.batch_size))
    global_step = 0
    log_time_start = time()

    for epoch in range(args.epochs):
        if global_step >= args.steps:
            break

        for batch in range(batches_per_epoch):
            if global_step >= args.steps:
                break

            global_step += 1

            # Update hyper parameters
            # ---------------------------------- #
            sigmoid_rampup = sigmoid_rampup_annealer.get_value(global_step)
            sigmoid_rampdown = sigmoid_rampdown_annealer.get_value(global_step)

            lr = sigmoid_rampup * sigmoid_rampdown * args.lr_max
            cent_u_coeff = sigmoid_rampup * args.cent_u_coeff_max
            cons_coeff = sigmoid_rampup * args.cons_coeff_max

            hyper_updater.update(sess, feed_dict={'lr': lr,
                                                  'cent_u_coeff': cent_u_coeff,
                                                  'cons_coeff': cons_coeff,})
            hyper_vals = hyper_updater.get_value(sess)
            hyper_vals['sigmoid_rampup'] = sigmoid_rampup
            hyper_vals['sigmoid_rampdown'] = sigmoid_rampdown
            # ---------------------------------- #

            # Train model
            # ---------------------------------- #
            if sampling_separately:
                # print("Sample separately!")
                batch_ids_l, batch_ids_u = sampler.sample_group_of_ids()

                xl, yl, label_flag_l = train_loader.fetch_batch(batch_ids_l)
                xu, yu, label_flag_u = train_loader.fetch_batch(batch_ids_u)
                assert np.all(label_flag_l), "'label_flag_l: {}'".format(label_flag_l)
                assert not np.any(label_flag_u), "'label_flag_u: {}'".format(label_flag_u)

                x = np.concatenate([xl, xu], axis=0)
                y = np.concatenate([yl, yu], axis=0)
                label_flag = np.concatenate([label_flag_l, label_flag_u], axis=0)
            else:
                # print("Sample jointly!")
                batch_ids = sampler.sample_ids()
                x, y, label_flag = train_loader.fetch_batch(batch_ids)

            _, AEm = sess.run([train_op_AE, model.get_output(fetch_keys_AE, as_dict=True)],
                              feed_dict={model.is_train: True,
                                         model.x_ph: x, model.y_ph: y,
                                         model.label_flag_ph: label_flag})

            batch_results = AEm
            train_summarizer.accumulate(batch_results, args.batch_size)
            # ---------------------------------- #

            if global_step % args.save_freq == 0:
                train_helper.save(sess, global_step)

            if global_step % args.log_freq == 0:
                log_time_end = time()
                log_time_gap = (log_time_end - log_time_start)
                log_time_start = log_time_end

                summaries, results = train_summarizer.get_summaries_and_reset(summary_prefix='train')
                train_helper.add_summaries(summaries, global_step)
                train_helper.add_summaries(custom_tf_scalar_summaries(hyper_vals, prefix="moving_hyper"), global_step)

                log_str = "\n[PiModel ({})/{}, {}], " \
                          "Epoch {}/{}, Batch {}/{} Step {} ({:.2f}s) (train)".format(
                              args.dataset, args.model_name, args.run, epoch, args.epochs,
                              batch, batches_per_epoch, global_step-1, log_time_gap) + \
                          "\n" + ", ".join(["{}: {:.4f}".format(key, results[key])
                                            for key in fetch_keys_AE_l]) + \
                          "\n" + ", ".join(["{}: {:.4f}".format(key, results[key])
                                            for key in fetch_keys_AE_u]) + \
                          "\n" + ", ".join(["{}: {:.4f}".format(key, hyper_vals[key])
                                           for key in hyper_vals])

                print(log_str)
                with open(train_log_file, "a") as f:
                    f.write(log_str)
                    f.write("\n")
                f.close()

            if global_step % args.eval_freq == 0:
                for batch_ids in iterate_data(test_loader.num_data, args.batch_size,
                                              shuffle=False, include_remaining=True):
                    x, y = test_loader.fetch_batch(batch_ids)

                    batch_results = sess.run(model.get_output(fetch_keys_test, as_dict=True),
                                             feed_dict={model.is_train: False,
                                                        model.x_ph: x, model.y_ph: y})

                    eval_summarizer.accumulate(batch_results, len(batch_ids))

                summaries, results = eval_summarizer.get_summaries_and_reset(summary_prefix='test')
                train_helper.add_summaries(summaries, global_step)

                log_str = "Epoch {}/{}, Batch {}/{} (test), acc_y: {:.4f}".format(
                              epoch, args.epochs, batch, batches_per_epoch, results['acc_y'])

                print(log_str)
                with open(train_log_file, "a") as f:
                    f.write(log_str)
                    f.write("\n")
                f.close()

                is_better = tracker.check_and_update(results, global_step)
                if is_better['acc_y']:
                    train_helper.save_best(sess, global_step=global_step)

    # Last save
    train_helper.save(sess, global_step)
    # ===================================== #


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
