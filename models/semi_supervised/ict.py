from six import iteritems
import pprint
import tensorflow as tf

from my_utils.tensorflow_utils.shaping import mixed_shape
from my_utils.tensorflow_utils.general import FlexibleTemplate
from my_utils.tensorflow_utils.activations import log_w_clip

from .mean_teacher import MeanTeacher


# Interpolation Consistency Training (OK)
class ICT(MeanTeacher):
    def __init__(self, x_shape, y_shape,
                 input_perturber, main_classifier,
                 ema_momentum=0.99, cons_mode='mse',
                 cons_4_unlabeled_only=False,
                 same_perturbed_inputs=False,
                 update_ops_4_stu_only=True,
                 weight_decay=-1.0, alpha=1.0):

        MeanTeacher.__init__(self, x_shape, y_shape,
                             input_perturber=input_perturber, main_classifier=main_classifier,
                             ema_momentum=ema_momentum, cons_mode=cons_mode,
                             cons_4_unlabeled_only=cons_4_unlabeled_only,
                             same_perturbed_inputs=same_perturbed_inputs,
                             update_ops_4_stu_only=update_ops_4_stu_only,
                             weight_decay=weight_decay)

        self._alpha = alpha
        self.beta = tf.distributions.Beta(tf.constant(alpha), tf.constant(alpha))

        print("In class [{}]:".format(self.__class__.__name__))
        print("_alpha: {}".format(self._alpha))

    # Consistency loss of ICT
    def _consistency_loss(self):
        from tensorflow.contrib.graph_editor import graph_replace

        x = self.x_ph
        
        shape_x = mixed_shape(x)
        batch_size = shape_x[0]

        rand_ids = tf.random_shuffle(tf.range(batch_size, dtype=tf.int32))
        x_rand = tf.gather(x, rand_ids)

        # (batch_size)
        lam = self.beta.sample(batch_size)
        lam_x = tf.reshape(lam, [batch_size] + [1] * len(self.x_shape))
        lam_y = tf.reshape(lam, [batch_size, 1])

        x_mixed = lam_x * x + (1 - lam_x) * x_rand

        y_prob_stu = self.get_output('y_dist_stu')['prob']
        y_prob_stu_on_x_mixed = graph_replace(y_prob_stu, {x: x_mixed})

        y_prob_tea = self.get_output('y_dist_tea')['prob']
        y_prob_tea_rand = graph_replace(y_prob_tea, {x: x_rand})
        y_prob_tea_mixed = lam_y * y_prob_tea + (1 - lam_y) * y_prob_tea_rand

        if self.cons_mode == 'mse':
            # IMPORTANT: Here, we take the sum over classes.
            # Implementations from other papers they use 'mean' instead of 'sum'.
            # This means our 'cons_coeff' should be about 10 (for CIFAR-10 and SVHN),
            # not 100 like in the original papers
            print("cons_mode=mse!")
            consistency = tf.reduce_sum(tf.square(
                y_prob_stu_on_x_mixed - tf.stop_gradient(y_prob_tea_mixed)), axis=1)
        elif self.cons_mode == 'kld':
            print("cons_mode=kld!")
            from my_utils.tensorflow_utils.distributions import KLD_2Cats_v2
            consistency = KLD_2Cats_v2(y_prob_stu_on_x_mixed, tf.stop_gradient(y_prob_tea_mixed))
        elif self.cons_mode == 'rev_kld':
            print("cons_mode=rev_kld!")
            from my_utils.tensorflow_utils.distributions import KLD_2Cats_v2
            consistency = KLD_2Cats_v2(tf.stop_gradient(y_prob_tea_mixed), y_prob_stu_on_x_mixed)
        else:
            raise ValueError("Do not support 'cons_mode'={}!".format(self.cons_mode))

        if self.cons_4_unlabeled_only:
            label_flag_inv = self.get_output('label_flag_inv')
            num_unlabeled = self.get_output('num_unlabeled')
            consistency = tf.reduce_sum(consistency * label_flag_inv, axis=0) * 1.0 / (num_unlabeled + 1e-8)
        else:
            consistency = tf.reduce_mean(consistency, axis=0)

        results = {
            'cons': consistency,
        }

        return results


# ICT+MUR
class ICT_MUR(ICT):
    def __init__(self, x_shape, y_shape,
                 input_perturber, main_classifier,
                 ema_momentum=0.99, cons_mode='mse',
                 cons_4_unlabeled_only=False,
                 same_perturbed_inputs=False,
                 update_ops_4_stu_only=True,
                 weight_decay=-1.0, alpha=1.0,

                 mur_mode="mse_wrt_point",
                 mur_4_unlabeled_only=False,
                 mur_noise_radius=10.0,

                 mur_opt_steps=1, mur_opt_lr=0.01,
                 # Only being used when mur_opt_steps > 0,
                 mur_iter_mode="proj_grad_asc"):

        ICT.__init__(self, x_shape=x_shape, y_shape=y_shape,
                     input_perturber=input_perturber, main_classifier=main_classifier,
                     ema_momentum=ema_momentum, cons_mode=cons_mode,
                     cons_4_unlabeled_only=cons_4_unlabeled_only,
                     same_perturbed_inputs=same_perturbed_inputs,
                     update_ops_4_stu_only=update_ops_4_stu_only,
                     weight_decay=weight_decay, alpha=alpha)

        possible_mur_modes = ['mse_wrt_point', 'mse_wrt_neigh']
        assert mur_mode in possible_mur_modes, "'mur_mode' must be in {}. " \
            "Found {}!".format(possible_mur_modes, mur_mode)
        assert mur_mode == "mse_wrt_point", \
            "You must set 'mur_mode'='mse_wrt_point' to obtain good performance!"

        assert mur_opt_steps >= 1, "'mur_opt_steps' must be >= 1. Found {}!".format(mur_opt_steps)

        possible_mur_iter_modes = ["grad_asc_w_lagrangian_relax", "proj_grad_asc"]
        assert mur_iter_mode in possible_mur_iter_modes, "'mur_iter_mode' must be in {}. " \
            "Found {}!".format(possible_mur_iter_modes, mur_iter_mode)

        self.mur_mode = mur_mode
        self.mur_4_unlabeled_only = mur_4_unlabeled_only
        self.mur_noise_radius = mur_noise_radius
        self.mur_opt_steps = mur_opt_steps

        self.mur_iter_mode = mur_iter_mode
        self.mur_opt_lr = mur_opt_lr

        print("In class [{}]:".format(self.__class__.__name__))
        print("mur_mode: {}".format(self.mur_mode))
        print("mur_4_unlabeled_only: {}".format(self.mur_4_unlabeled_only))
        print("mur_noise_radius: {}".format(self.mur_noise_radius))

        print("mur_opt_steps: {}".format(self.mur_opt_steps))
        print("mur_opt_lr: {}".format(self.mur_opt_lr))
        print("mur_iter_mode: {}".format(self.mur_iter_mode))

    def _mur_loss(self):
        from tensorflow.contrib.graph_editor import graph_replace

        # IMPORTANT: We use 'x_pert_stu' to ensure no perturbation on the input
        # (batch, x_dim)
        x0 = self.get_output('x_pert_stu')
        # (batch, num_classes)
        y0_prob = self.get_output('y_dist_stu')['prob']
        # (batch, )
        cond_ent0 = tf.reduce_sum(-y0_prob * log_w_clip(y0_prob), axis=1)

        normalized_axes = list(range(1, x0.shape.ndims))
        g0 = tf.gradients(cond_ent0, [x0])[0]
        g0_norm = tf.stop_gradient(tf.sqrt(tf.reduce_sum(
            g0 ** 2, axis=normalized_axes, keepdims=True)) + 1e-15)

        rad = self.mur_noise_radius

        # Direct approximation
        if self.mur_opt_steps == 1:
            print("Direct approximation of x*!")
            eps = rad * g0 / (g0_norm + 1e-8)
            x_final = tf.stop_gradient(x0 + eps)

        else:
            # 'grad_asc_w_lagrangian_relax'
            # f'(x) - norm(g_0)/r * (x-x_0) * (2 - r / norm(x-x_0))

            # 'proj_grad_asc'
            # z_{t+1} = x_{t} + lr * f'(x)
            # x_{t+1} = z_{t+1} if norm(z_{t+1} - x_0) <= r else x_0 + r * (z_{t+1} - x_0) / norm(z_{t+1} - x_0)

            lr = self.mur_opt_lr

            if self.mur_iter_mode == "grad_asc_w_lagrangian_relax":
                print("Iterative approximation of x* using vanilla gradient ascent!")
                x_t = x0
                cond_ent_t = cond_ent0

                for _ in range(self.mur_opt_steps):
                    grad_x_t = tf.gradients(cond_ent_t, [x_t])[0]

                    xt_m_x0_norm = tf.stop_gradient(tf.sqrt(tf.reduce_sum(
                        (x_t - x0) ** 2, axis=normalized_axes, keepdims=True)) + 1e-15)

                    # Update 'x_t' and 'cond_ent_t'
                    x_t = tf.stop_gradient(x_t + lr * (grad_x_t - g0_norm / self.mur_noise_radius * (x_t - x0) *
                                                       (2 - self.mur_noise_radius / (xt_m_x0_norm + 1e-15))))
                    cond_ent_t = graph_replace(cond_ent0, replacement_ts={x0: x_t})

                x_final = x_t

            elif self.mur_iter_mode == "proj_grad_asc":
                print("Iterative approximation of x* using project gradient ascent!")
                x_t = x0
                cond_ent_t = cond_ent0

                for _ in range(self.mur_opt_steps):
                    grad_x_t = tf.gradients(cond_ent_t, [x_t])[0]

                    z_t = x_t + lr * grad_x_t

                    # (batch, 1, 1, 1)
                    zt_m_x0_norm = tf.stop_gradient(tf.sqrt(tf.reduce_sum(
                        (z_t - x0) ** 2, axis=normalized_axes, keepdims=True)) + 1e-15)

                    cond = tf.cast(tf.less_equal(zt_m_x0_norm, rad), dtype=tf.float32)
                    x_t = cond * z_t + (1.0 - cond) * (x0 + (z_t - x0) / zt_m_x0_norm)
                    x_t = tf.stop_gradient(x_t)

                    cond_ent_t = graph_replace(cond_ent0, replacement_ts={x0: x_t})

                x_final = x_t

            else:
                raise ValueError(self.mur_iter_mode)

        y_prob_final = graph_replace(y0_prob, replacement_ts={x0: x_final})

        if self.mur_mode == "mse_wrt_point":
            mur = tf.reduce_sum(tf.square(tf.stop_gradient(y0_prob) - y_prob_final), axis=1)
        elif self.mur_mode == "mse_wrt_neigh":
            mur = tf.reduce_sum(tf.square(y0_prob - tf.stop_gradient(y_prob_final)), axis=1)
        else:
            raise ValueError("Do not support mur_mode={}!".format(self.mur_mode))

        if self.mur_4_unlabeled_only:
            label_flag_inv = self.get_output('label_flag_inv')
            num_unlabeled = self.get_output('num_unlabeled')
            mur = tf.reduce_sum(mur * label_flag_inv, axis=0) * 1.0 / (num_unlabeled + 1e-8)
        else:
            mur = tf.reduce_mean(mur, axis=0)

        return {
            'grad_norm_avg': tf.reduce_mean(g0_norm),
            'mur': mur
        }

    def build(self, loss_coeff_dict):
        if self._built:
            raise AssertionError("The model has already been built!")

        # Compute tensors
        # --------------------------------------- #
        tensors = self._forward()
        for key, val in iteritems(tensors):
            self.set_output(key, val)
        # --------------------------------------- #

        # Compute losses
        # --------------------------------------- #
        lc = loss_coeff_dict
        coeff_fn = self.one_if_not_exist

        loss = 0

        # Class loss
        results_class = self._class_loss()
        loss += coeff_fn(lc, 'cross_ent_l') * results_class['cross_ent_l']
        loss += coeff_fn(lc, 'cond_ent_u') * results_class['cond_ent_u']
        for key, val in iteritems(results_class):
            self.set_output(key, val)

        # Consistency loss
        results_cons = self._consistency_loss()
        loss += coeff_fn(lc, 'cons') * results_cons['cons']
        for key, val in iteritems(results_cons):
            self.set_output(key, val)

        # MUR loss
        results_mur = self._mur_loss()
        loss += coeff_fn(lc, 'mur') * results_mur['mur']
        for key, val in iteritems(results_mur):
            self.set_output(key, val)

        # Weight decay loss
        if self.weight_decay > 0:
            l2_reg = self._l2_reg_loss()
        else:
            l2_reg = tf.constant(0.0, dtype=tf.float32)
        loss += self.weight_decay * l2_reg
        self.set_output('l2_reg', l2_reg)

        # Final loss
        self.set_output('loss', loss)
        # --------------------------------------- #

        print("All loss coefficients:")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.loss_coeff_dict)
        self._built = True


class ICT_VD(ICT):
    def __init__(self, x_shape, y_shape,
                 input_perturber, main_classifier,
                 ema_momentum=0.99, cons_mode='mse',
                 cons_4_unlabeled_only=False,
                 same_perturbed_inputs=False,
                 update_ops_4_stu_only=True,
                 weight_decay=-1.0, alpha=1.0,

                 mask_weights=True,
                 ema_4_log_sigma2=False,
                 cons_against_mean=True):

        ICT.__init__(self, x_shape=x_shape, y_shape=y_shape,
                     input_perturber=input_perturber,
                     main_classifier=main_classifier,
                     ema_momentum=ema_momentum, cons_mode=cons_mode,
                     cons_4_unlabeled_only=cons_4_unlabeled_only,
                     same_perturbed_inputs=same_perturbed_inputs,
                     update_ops_4_stu_only=update_ops_4_stu_only,
                     weight_decay=weight_decay, alpha=alpha)

        # If True: Mask the weights of the deterministic model
        self.mask_weights = mask_weights

        # If True: Compute moving average for log_sigma2
        self.ema_4_log_sigma2 = ema_4_log_sigma2

        # If True: Computing consistency loss between
        # the stochastic model and the deterministic model
        self.cons_against_mean = cons_against_mean

        print("In class [{}]:".format(self.__class__.__name__))
        print("mask_weights: {}".format(self.mask_weights))
        print("ema_4_log_sigma2: {}".format(self.ema_4_log_sigma2))
        print("cons_against_mean: {}".format(self.cons_against_mean))

    def _forward(self):
        from my_utils.tensorflow_utils.bayesian import svd
        from my_utils.tensorflow_utils.activations import exp_w_clip, log_w_clip

        y = tf.one_hot(self.y_ph, depth=self.num_classes,
                       on_value=tf.constant(1, dtype=tf.int32),
                       off_value=tf.constant(0, dtype=tf.int32),
                       dtype=tf.int32)
        y = tf.stop_gradient(tf.cast(y, dtype=tf.float32))

        x_pert_stu = self.input_perturber_fn(self.x_ph)
        if self.same_perturbed_inputs:
            x_pert_tea = x_pert_stu
        else:
            x_pert_tea = self.input_perturber_fn(self.x_ph)

        # Student network
        # ----------------------------------- #
        with tf.name_scope(self._student_ns):
            y_dist_stu_sto = self.student_classifier_fn(
                x_pert_stu, weight_mode=svd.NOISY_WEIGHT_MODE)
        # ----------------------------------- #

        # We can only create teacher model here since we have called student model
        # Thus, student's variables are initialized
        # This code is borrowed from CuriousAI
        if self.teacher_classifier_fn is None:
            student_param_dict_sigma2 = {
                param.op.name: exp_w_clip(param) for param in
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  scope=self.main_classifier_name)
                if "log_sigma2" in param.op.name
            }

            student_param_dict_others = {
                param.op.name: param for param in
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  scope=self.main_classifier_name)
                if "log_sigma2" not in param.op.name
            }

            student_param_dict = dict()
            if self.ema_4_log_sigma2:
                print("Take the exponential over log_sigma2!")
                student_param_dict.update(student_param_dict_sigma2)
            student_param_dict.update(student_param_dict_others)

            teacher_ema = tf.train.ExponentialMovingAverage(self.ema_momentum)
            teacher_param_updates = teacher_ema.apply(student_param_dict.values())
            self.set_output('teacher_ema', teacher_ema)
            self.set_output('teacher_param_updates', teacher_param_updates)

            def teacher_param_getter(getter, name, *args, **kwargs):
                # assert name in student_param_dict, "Unknown variable {}.".format(name)
                if name in student_param_dict:
                    if "log_sigma2" in name:
                        print("Take the log over average of sigma2!")
                        return log_w_clip(teacher_ema.average(student_param_dict[name]))
                    else:
                        return teacher_ema.average(student_param_dict[name])
                else:
                    return getter(name, *args, **kwargs)

            self.teacher_classifier_fn = FlexibleTemplate(
                self.main_classifier_name, self._main_classifier,
                custom_getter=teacher_param_getter, is_train=self.is_train)

        # Teacher network
        # ----------------------------------- #
        with tf.name_scope(self._teacher_ns):
            # Teacher (deterministic) uses STD_WEIGHT_MODE
            y_dist_tea_det = self.teacher_classifier_fn(
                x_pert_tea, weight_mode=svd.MASKED_WEIGHT_MODE
                if self.mask_weights else svd.STD_WEIGHT_MODE)

            if self.cons_against_mean:
                print("'cons_against_mean'=True so 'y_dist_tea_sto'=None")
                y_dist_tea_sto = None
            else:
                print("'cons_against_mean'=False so 'y_dist_tea_sto'!=None")
                y_dist_tea_sto = self.teacher_classifier_fn(
                    x_pert_tea, weight_mode=svd.NOISY_WEIGHT_MODE)
        # ----------------------------------- #

        label_flag = tf.cast(self.label_flag_ph, dtype=tf.float32)
        num_labeled = tf.reduce_sum(label_flag, axis=0)

        label_flag_inv = tf.cast(tf.logical_not(self.label_flag_ph), dtype=tf.float32)
        num_unlabeled = tf.reduce_sum(label_flag_inv, axis=0)

        return {'x_pert_stu': x_pert_stu, 'x_pert_tea': x_pert_tea,
                'y': y, 'y_dist_stu_sto': y_dist_stu_sto,
                'y_dist_tea_sto': y_dist_tea_sto, 'y_dist_tea_det': y_dist_tea_det,
                'label_flag': label_flag, 'label_flag_inv': label_flag_inv,
                'num_labeled': num_labeled, 'num_unlabeled': num_unlabeled}

    def _class_loss(self):
        # This function considers both labeled and unlabeled data in a single batch
        y_idx = self.y_ph
        y = self.get_output('y')
        y_dist_stu_sto = self.get_output('y_dist_stu_sto')
        # y_dist_stu_det = self.get_output('y_dist_stu_det')
        y_dist_tea_det = self.get_output('y_dist_tea_det')

        label_flag = self.get_output('label_flag')
        label_flag_inv = self.get_output('label_flag_inv')
        num_labeled = self.get_output('num_labeled')
        num_unlabeled = self.get_output('num_unlabeled')

        y_logit_stu_sto, y_prob_stu_sto = y_dist_stu_sto['logit'], y_dist_stu_sto['prob']

        # Cross entropy loss for labeled data
        cross_ent_l = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=y_logit_stu_sto, dim=-1)
        cross_ent_l = tf.reduce_sum(cross_ent_l * label_flag, axis=0) * 1.0 / (num_labeled + 1e-8)

        # Conditional entropy loss for unlabeld data
        cond_ent_u = tf.reduce_sum(-y_prob_stu_sto * log_w_clip(y_prob_stu_sto), axis=1)
        cond_ent_u = tf.reduce_sum(cond_ent_u * label_flag_inv, axis=0) * 1.0 / (num_unlabeled + 1e-8)

        y_prob_stu_sto = y_dist_stu_sto['prob']
        y_pred_stu = tf.argmax(y_prob_stu_sto, axis=1, output_type=tf.int32)
        y_matched_stu = tf.cast(tf.equal(y_pred_stu, y_idx), dtype=tf.float32)
        acc_y_l_stu = tf.reduce_sum(y_matched_stu * label_flag, axis=0) * 1.0 / (num_labeled + 1e-8)
        acc_y_u_stu = tf.reduce_sum(y_matched_stu * label_flag_inv, axis=0) * 1.0 / (num_unlabeled + 1e-8)
        acc_y_stu = tf.reduce_mean(y_matched_stu, axis=0)

        y_prob_tea_det = y_dist_tea_det['prob']
        y_pred_tea = tf.argmax(y_prob_tea_det, axis=1, output_type=tf.int32)
        y_matched_tea = tf.cast(tf.equal(y_pred_tea, y_idx), dtype=tf.float32)
        acc_y_l_tea = tf.reduce_sum(y_matched_tea * label_flag, axis=0) * 1.0 / (num_labeled + 1e-8)
        acc_y_u_tea = tf.reduce_sum(y_matched_tea * label_flag_inv, axis=0) * 1.0 / (num_unlabeled + 1e-8)
        acc_y_tea = tf.reduce_mean(y_matched_tea, axis=0)

        y_pred = y_pred_tea
        acc_y_l = acc_y_l_tea
        acc_y_u = acc_y_u_tea
        acc_y = acc_y_tea

        results = {
            'cross_ent_l': cross_ent_l,
            'cond_ent_u': cond_ent_u,

            'y_pred_stu': y_pred_stu,
            'acc_y_l_stu': acc_y_l_stu,
            'acc_y_u_stu': acc_y_u_stu,
            'acc_y_stu': acc_y_stu,

            'y_pred_tea': y_pred_tea,
            'acc_y_l_tea': acc_y_l_tea,
            'acc_y_u_tea': acc_y_u_tea,
            'acc_y_tea': acc_y_tea,

            'y_pred': y_pred,
            'acc_y_l': acc_y_l,
            'acc_y_u': acc_y_u,
            'acc_y': acc_y,
        }

        return results

    def _consistency_loss(self):
        from tensorflow.contrib.graph_editor import graph_replace

        x = self.x_ph

        shape_x = mixed_shape(x)
        batch_size = shape_x[0]

        rand_ids = tf.random_shuffle(tf.range(batch_size, dtype=tf.int32))
        x_rand = tf.gather(x, rand_ids)

        # (batch_size)
        lam = self.beta.sample(batch_size)
        lam_x = tf.reshape(lam, [batch_size] + [1] * len(self.x_shape))
        lam_y = tf.reshape(lam, [batch_size, 1])

        x_mixed = lam_x * x + (1 - lam_x) * x_rand

        y_prob_stu = self.get_output('y_dist_stu_sto')['prob']
        y_prob_stu_on_x_mixed = graph_replace(y_prob_stu, {x: x_mixed})

        if self.cons_against_mean:
            print("Use teacher (deterministic) for consistency!")
            y_prob_tea = self.get_output('y_dist_tea_det')['prob']
        else:
            print("Use teacher (stochastic) for consistency!")
            assert 'y_dist_tea_sto' in self.output_dict, "'output_dict' must contain 'y_dist_tea_sto'!"
            y_prob_tea = self.get_output('y_dist_tea_sto')['prob']
        y_prob_tea_rand = graph_replace(y_prob_tea, {x: x_rand})
        y_prob_tea_mixed = lam_y * y_prob_tea + (1 - lam_y) * y_prob_tea_rand

        if self.cons_mode == 'mse':
            # IMPORTANT: Here, we take the sum over classes.
            # Implementations from other papers they use 'mean' instead of 'sum'.
            # This means our 'cons_coeff' should be about 10 (for CIFAR-10 and SVHN),
            # not 100 like in the original papers
            print("cons_mode=mse!")
            consistency = tf.reduce_sum(tf.square(
                y_prob_stu_on_x_mixed - tf.stop_gradient(y_prob_tea_mixed)), axis=1)
        elif self.cons_mode == 'kld':
            print("cons_mode=kld!")
            from my_utils.tensorflow_utils.distributions import KLD_2Cats_v2
            consistency = KLD_2Cats_v2(y_prob_stu_on_x_mixed, tf.stop_gradient(y_prob_tea_mixed))
        elif self.cons_mode == 'rev_kld':
            print("cons_mode=rev_kld!")
            from my_utils.tensorflow_utils.distributions import KLD_2Cats_v2
            consistency = KLD_2Cats_v2(tf.stop_gradient(y_prob_tea_mixed), y_prob_stu_on_x_mixed)
        else:
            raise ValueError("Do not support 'cons_mode'={}!".format(self.cons_mode))

        if self.cons_4_unlabeled_only:
            label_flag_inv = self.get_output('label_flag_inv')
            num_unlabeled = self.get_output('num_unlabeled')
            consistency = tf.reduce_sum(consistency * label_flag_inv, axis=0) * 1.0 / (num_unlabeled + 1e-8)
        else:
            consistency = tf.reduce_mean(consistency, axis=0)

        results = {
            'cons': consistency,
        }

        return results

    def _sparsity_loss(self):
        from my_utils.tensorflow_utils.bayesian import svd
        log_alphas = svd.collect_log_alphas_v2(
            scopes="{}/{}".format(self._student_ns, self.main_classifier_name))

        print("log_alphas: {}".format(log_alphas))
        assert len(log_alphas) > 0, "len(log_alphas) must be larger than 0!"

        weight_kld = tf.reduce_sum(tf.stack([svd.KL_qp_approx(la) for la in log_alphas]))
        num_weights = tf.reduce_sum(tf.stack([tf.reduce_prod(la.shape) for la in log_alphas]))
        weight_kld = weight_kld / (tf.cast(num_weights, dtype=tf.float32) + 1e-8)

        sparseness = svd.sparsity(log_alphas)

        return {
            'weight_kld': weight_kld,
            'sparseness': sparseness,
            'num_weights': num_weights,
        }

    def build(self, loss_coeff_dict):
        if self._built:
            raise AssertionError("The model has already been built!")

        # Compute tensors
        # --------------------------------------- #
        tensors = self._forward()
        for key, val in iteritems(tensors):
            self.set_output(key, val)
        # --------------------------------------- #

        # Compute losses
        # --------------------------------------- #
        lc = loss_coeff_dict
        coeff_fn = self.one_if_not_exist

        loss = 0
        loss_mt = 0

        # Class loss
        results_class = self._class_loss()
        loss += coeff_fn(lc, 'cross_ent_l') * results_class['cross_ent_l']
        loss += coeff_fn(lc, 'cond_ent_u') * results_class['cond_ent_u']

        loss_mt += coeff_fn(lc, 'cross_ent_l') * results_class['cross_ent_l']
        loss_mt += coeff_fn(lc, 'cond_ent_u') * results_class['cond_ent_u']
        for key, val in iteritems(results_class):
            self.set_output(key, val)

        # Consistency loss
        results_cons = self._consistency_loss()
        loss += coeff_fn(lc, 'cons') * results_cons['cons']
        loss_mt += coeff_fn(lc, 'cons') * results_cons['cons']
        for key, val in iteritems(results_cons):
            self.set_output(key, val)

        # Weight KLD
        results_sparsity = self._sparsity_loss()
        loss += coeff_fn(lc, 'weight_kld') * results_sparsity['weight_kld']
        for key, val in iteritems(results_sparsity):
            self.set_output(key, val)

        # Weight decay loss
        if self.weight_decay > 0:
            l2_reg = self._l2_reg_loss()
        else:
            l2_reg = tf.constant(0.0, dtype=tf.float32)
        loss += self.weight_decay * l2_reg
        self.set_output('l2_reg', l2_reg)

        # Final loss
        self.set_output('loss', loss)
        self.set_output('loss_mt', loss_mt)
        # --------------------------------------- #

        print("All loss coefficients:")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.loss_coeff_dict)
        self._built = True


class ICT_VD_MUR(ICT_VD):
    def __init__(self, x_shape, y_shape,
                 input_perturber, main_classifier,
                 ema_momentum=0.99, cons_mode='mse',
                 cons_4_unlabeled_only=False,
                 same_perturbed_inputs=False,
                 update_ops_4_stu_only=True,
                 weight_decay=-1.0, alpha=1.0,

                 mask_weights=True,
                 ema_4_log_sigma2=False,
                 cons_against_mean=True,

                 mur_mode="mse_wrt_point",
                 mur_4_unlabeled_only=False,
                 mur_noise_radius=10.0,

                 # Only being used when mur_opt_steps > 0,
                 mur_opt_steps=1, mur_opt_lr=0.01,
                 mur_iter_mode="proj_grad_asc"):

        ICT_VD.__init__(self, x_shape=x_shape, y_shape=y_shape,
                        input_perturber=input_perturber,
                        main_classifier=main_classifier,
                        ema_momentum=ema_momentum, cons_mode=cons_mode,
                        cons_4_unlabeled_only=cons_4_unlabeled_only,
                        same_perturbed_inputs=same_perturbed_inputs,
                        update_ops_4_stu_only=update_ops_4_stu_only,
                        weight_decay=weight_decay, alpha=alpha,

                        mask_weights=mask_weights,
                        ema_4_log_sigma2=ema_4_log_sigma2,
                        cons_against_mean=cons_against_mean)

        possible_mur_modes = ['mse_wrt_point', 'mse_wrt_neigh']
        assert mur_mode in possible_mur_modes, "'mur_mode' must be in {}. " \
                                               "Found {}!".format(possible_mur_modes, mur_mode)
        assert mur_mode == "mse_wrt_point", \
            "You must set 'mur_mode'='mse_wrt_point' to obtain good performance!"

        assert mur_opt_steps >= 1, "'mur_opt_steps' must be >= 1. Found {}!".format(mur_opt_steps)

        possible_mur_iter_modes = ["grad_asc_w_lagrangian_relax", "proj_grad_asc"]
        assert mur_iter_mode in possible_mur_iter_modes, "'mur_iter_mode' must be in {}. " \
                                                         "Found {}!".format(possible_mur_iter_modes, mur_iter_mode)

        self.mur_mode = mur_mode
        self.mur_4_unlabeled_only = mur_4_unlabeled_only
        self.mur_noise_radius = mur_noise_radius
        self.mur_opt_steps = mur_opt_steps

        self.mur_iter_mode = mur_iter_mode
        self.mur_opt_lr = mur_opt_lr

        print("In class [{}]:".format(self.__class__.__name__))
        print("mur_mode: {}".format(self.mur_mode))
        print("mur_4_unlabeled_only: {}".format(self.mur_4_unlabeled_only))
        print("mur_noise_radius: {}".format(self.mur_noise_radius))

        print("mur_opt_steps: {}".format(self.mur_opt_steps))
        print("mur_opt_lr: {}".format(self.mur_opt_lr))
        print("mur_iter_mode: {}".format(self.mur_iter_mode))
        
    def _mur_loss(self):
        from tensorflow.contrib.graph_editor import graph_replace

        # IMPORTANT: We use 'x_pert_stu' to ensure no perturbation on the input
        # (batch, x_dim)
        x0 = self.get_output('x_pert_stu')

        # IMPORTANT: The output here is 'y_dist_stu_sto' not 'y_dist_stu'
        # (batch, num_classes)
        y0_prob = self.get_output('y_dist_stu_sto')['prob']
        # (batch, )
        cond_ent0 = tf.reduce_sum(-y0_prob * log_w_clip(y0_prob), axis=1)

        normalized_axes = list(range(1, x0.shape.ndims))
        g0 = tf.gradients(cond_ent0, [x0])[0]
        g0_norm = tf.stop_gradient(tf.sqrt(tf.reduce_sum(
            g0 ** 2, axis=normalized_axes, keepdims=True)) + 1e-15)

        rad = self.mur_noise_radius

        # Direct approximation
        if self.mur_opt_steps == 1:
            print("Direct approximation of x*!")
            eps = rad * g0 / (g0_norm + 1e-8)
            x_final = tf.stop_gradient(x0 + eps)

        else:
            lr = self.mur_opt_lr

            if self.mur_iter_mode == "grad_asc_w_lagrangian_relax":
                print("Iterative approximation of x* using vanilla gradient ascent!")
                x_t = x0
                cond_ent_t = cond_ent0

                for _ in range(self.mur_opt_steps):
                    grad_x_t = tf.gradients(cond_ent_t, [x_t])[0]

                    xt_m_x0_norm = tf.stop_gradient(tf.sqrt(tf.reduce_sum(
                        (x_t - x0) ** 2, axis=normalized_axes, keepdims=True)) + 1e-15)

                    # Update 'x_t' and 'cond_ent_t'
                    x_t = tf.stop_gradient(x_t + lr * (grad_x_t - g0_norm / self.mur_noise_radius * (x_t - x0) *
                                                       (2 - self.mur_noise_radius / (xt_m_x0_norm + 1e-15))))
                    cond_ent_t = graph_replace(cond_ent0, replacement_ts={x0: x_t})

                x_final = x_t

            elif self.mur_iter_mode == "proj_grad_asc":
                print("Iterative approximation of x* using project gradient ascent!")
                x_t = x0
                cond_ent_t = cond_ent0

                for _ in range(self.mur_opt_steps):
                    grad_x_t = tf.gradients(cond_ent_t, [x_t])[0]

                    z_t = x_t + lr * grad_x_t

                    # (batch, 1, 1, 1)
                    zt_m_x0_norm = tf.stop_gradient(tf.sqrt(tf.reduce_sum(
                        (z_t - x0) ** 2, axis=normalized_axes, keepdims=True)) + 1e-15)

                    cond = tf.cast(tf.less_equal(zt_m_x0_norm, rad), dtype=tf.float32)
                    x_t = cond * z_t + (1.0 - cond) * (x0 + (z_t - x0) / zt_m_x0_norm)
                    x_t = tf.stop_gradient(x_t)

                    cond_ent_t = graph_replace(cond_ent0, replacement_ts={x0: x_t})

                x_final = x_t

            else:
                raise ValueError(self.mur_iter_mode)

        y_prob_final = graph_replace(y0_prob, replacement_ts={x0: x_final})

        if self.mur_mode == "mse_wrt_point":
            mur = tf.reduce_sum(tf.square(tf.stop_gradient(y0_prob) - y_prob_final), axis=1)
        elif self.mur_mode == "mse_wrt_neigh":
            mur = tf.reduce_sum(tf.square(y0_prob - tf.stop_gradient(y_prob_final)), axis=1)
        else:
            raise ValueError("Do not support mur_mode={}!".format(self.mur_mode))

        if self.mur_4_unlabeled_only:
            label_flag_inv = self.get_output('label_flag_inv')
            num_unlabeled = self.get_output('num_unlabeled')
            mur = tf.reduce_sum(mur * label_flag_inv, axis=0) * 1.0 / (num_unlabeled + 1e-8)
        else:
            mur = tf.reduce_mean(mur, axis=0)

        return {
            'grad_norm_avg': tf.reduce_mean(g0_norm),
            'mur': mur
        }

    def build(self, loss_coeff_dict):
        if self._built:
            raise AssertionError("The model has already been built!")

        # Compute tensors
        # --------------------------------------- #
        tensors = self._forward()
        for key, val in iteritems(tensors):
            self.set_output(key, val)
        # --------------------------------------- #

        # Compute losses
        # --------------------------------------- #
        lc = loss_coeff_dict
        coeff_fn = self.one_if_not_exist

        loss = 0

        # Class loss
        results_class = self._class_loss()
        loss += coeff_fn(lc, 'cross_ent_l') * results_class['cross_ent_l']
        loss += coeff_fn(lc, 'cond_ent_u') * results_class['cond_ent_u']
        for key, val in iteritems(results_class):
            self.set_output(key, val)

        # Consistency loss
        results_cons = self._consistency_loss()
        loss += coeff_fn(lc, 'cons') * results_cons['cons']
        for key, val in iteritems(results_cons):
            self.set_output(key, val)

        # Weight KLD
        results_sparsity = self._sparsity_loss()
        loss += coeff_fn(lc, 'weight_kld') * results_sparsity['weight_kld']
        for key, val in iteritems(results_sparsity):
            self.set_output(key, val)

        # MUR loss
        results_mur = self._mur_loss()
        loss += coeff_fn(lc, 'mur') * results_mur['mur']
        for key, val in iteritems(results_mur):
            self.set_output(key, val)

        # Weight decay loss
        if self.weight_decay > 0:
            l2_reg = self._l2_reg_loss()
        else:
            l2_reg = tf.constant(0.0, dtype=tf.float32)
        loss += self.weight_decay * l2_reg
        self.set_output('l2_reg', l2_reg)

        # Final loss
        self.set_output('loss', loss)
        # --------------------------------------- #

        print("All loss coefficients:")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.loss_coeff_dict)
        self._built = True
