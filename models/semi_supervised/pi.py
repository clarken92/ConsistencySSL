from six import iteritems
import tensorflow as tf

import pprint
pp = pprint.PrettyPrinter(indent=4)

from my_utils.python_utils.general import to_list
from my_utils.tensorflow_utils.models import LiteBaseModel
from my_utils.tensorflow_utils.activations import log_w_clip


# PiModel is unstable compared to MeanTeacher and ICT
# Pi (OK)
class PiModel(LiteBaseModel):
    def __init__(self, x_shape, y_shape,
                 input_perturber, main_classifier,
                 cons_mode='mse',
                 cons_4_unlabeled_only=False,
                 same_perturbed_inputs=False,
                 weight_decay=-1.0):

        LiteBaseModel.__init__(self)

        self.x_shape = to_list(x_shape)
        self.y_shape = to_list(y_shape)

        assert len(self.y_shape) == 1, "'y_shape' must be a scalar or an array of length 1!"
        self.num_classes = self.y_shape[0]

        self.x_ph = tf.placeholder(tf.float32, [None] + self.x_shape, name="x")
        self.y_ph = tf.placeholder(tf.int32, [None], name="y")
        self.label_flag_ph = tf.placeholder(tf.bool, [None], name="label_flag")

        # Main classifier
        self.main_classifier_name = "main_classifier"
        self.main_classifier_fn = tf.make_template(
            self.main_classifier_name, main_classifier, is_train=self.is_train)

        # Input perturber
        self.input_perturber_name = "input_perturber"
        self.input_perturber_fn = tf.make_template(
            self.input_perturber_name, input_perturber, is_train=self.is_train)

        possible_cons_modes = ['mse', 'kld', 'rev_kld', '2rand']
        assert cons_mode in possible_cons_modes, \
            "'cons_mode' must be in {}. Found {}!".format(possible_cons_modes, cons_mode)
        self.cons_mode = cons_mode
        self.cons_4_unlabeled_only = cons_4_unlabeled_only
        self.same_perturbed_inputs = same_perturbed_inputs
        self.weight_decay = weight_decay

        print("In class [{}]:".format(self.__class__.__name__))
        print("cons_mode: {}".format(self.cons_mode))
        print("cons_4_unlabeled_only: {}".format(self.cons_4_unlabeled_only))
        print("same_perturbed_inputs: {}".format(self.same_perturbed_inputs))
        print("weight_decay: {}".format(self.weight_decay))

    def _forward(self):
        y = tf.one_hot(self.y_ph, depth=self.num_classes,
                       on_value=tf.constant(1, dtype=tf.int32),
                       off_value=tf.constant(0, dtype=tf.int32),
                       dtype=tf.int32)
        y = tf.stop_gradient(tf.cast(y, dtype=tf.float32))

        x_pert = self.input_perturber_fn(self.x_ph)
        if self.same_perturbed_inputs:
            xa_pert = x_pert
        else:
            xa_pert = self.input_perturber_fn(self.x_ph)

        y_dist = self.main_classifier_fn(x_pert)
        ya_dist = self.main_classifier_fn(xa_pert)

        label_flag = tf.cast(self.label_flag_ph, dtype=tf.float32)
        num_labeled = tf.reduce_sum(label_flag, axis=0)

        label_flag_inv = tf.cast(tf.logical_not(self.label_flag_ph), dtype=tf.float32)
        num_unlabeled = tf.reduce_sum(label_flag_inv, axis=0)

        return {'x_pert': x_pert, 'xa_pert': xa_pert,
                'y': y, 'y_dist': y_dist, 'ya_dist': ya_dist,
                'label_flag': label_flag, 'label_flag_inv': label_flag_inv,
                'num_labeled': num_labeled, 'num_unlabeled': num_unlabeled}

    def _class_loss(self):
        # This function considers both labeled and unlabeled data in a single batch
        y_idx = self.y_ph
        y = self.get_output('y')
        y_dist = self.get_output('y_dist')
        label_flag = self.get_output('label_flag')
        label_flag_inv = self.get_output('label_flag_inv')
        num_labeled = self.get_output('num_labeled')
        num_unlabeled = self.get_output('num_unlabeled')

        y_logit, y_prob = y_dist['logit'], y_dist['prob']

        # Cross entropy loss for labeled data
        cross_ent_l = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=y_logit, dim=-1)
        cross_ent_l = tf.reduce_sum(cross_ent_l * label_flag, axis=0) * 1.0 / (num_labeled + 1e-8)

        # Conditional entropy loss for unlabeld data
        cond_ent_u = tf.reduce_sum(-y_prob * log_w_clip(y_prob), axis=1)
        cond_ent_u = tf.reduce_sum(cond_ent_u * label_flag_inv, axis=0) * 1.0 / (num_unlabeled + 1e-8)

        y_pred = tf.argmax(y_prob, axis=1, output_type=tf.int32)
        y_matched = tf.cast(tf.equal(y_pred, y_idx), dtype=tf.float32)
        acc_y_l = tf.reduce_sum(y_matched * label_flag, axis=0) * 1.0 / (num_labeled + 1e-8)
        acc_y_u = tf.reduce_sum(y_matched * label_flag_inv, axis=0) * 1.0 / (num_unlabeled + 1e-8)
        acc_y = tf.reduce_mean(y_matched, axis=0)

        results = {
            'cross_ent_l': cross_ent_l,
            'cond_ent_u': cond_ent_u,
            'y_pred': y_pred,
            'acc_y_l': acc_y_l,
            'acc_y_u': acc_y_u,
            'acc_y': acc_y,
        }

        return results

    def _consistency_loss(self):
        y_prob = self.get_output('y_dist')['prob']
        ya_prob = self.get_output('ya_dist')['prob']

        if self.cons_mode == 'mse':
            # IMPORTANT: Here, we take the sum over classes.
            # Implementations from other papers they use 'mean' instead of 'sum'.
            # This suggests that our 'cons_coeff' must be about 10, not 100 like other papers
            print("cons_mode=mse!")
            consistency = tf.reduce_sum(tf.square(y_prob - tf.stop_gradient(ya_prob)), axis=1)
        elif self.cons_mode == 'kld':
            print("cons_mode=kld!")
            from my_utils.tensorflow_utils.distributions import KLD_2Cats_v2
            consistency = KLD_2Cats_v2(y_prob, tf.stop_gradient(ya_prob))
        elif self.cons_mode == 'rev_kld':
            print("cons_mode=rev_kld!")
            from my_utils.tensorflow_utils.distributions import KLD_2Cats_v2
            consistency = KLD_2Cats_v2(tf.stop_gradient(ya_prob), y_prob)
        elif self.cons_mode == '2rand':
            print("cons_mode=2rand!")
            from my_utils.tensorflow_utils.activations import log_w_clip
            # IMPORTANT: We try to stop gradient here!
            # consistency = -log_w_clip(tf.reduce_sum(y_prob * ya_prob, axis=1))
            consistency = -log_w_clip(tf.reduce_sum(y_prob * tf.stop_gradient(ya_prob), axis=1))
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

    def _l2_reg_loss(self):
        from my_utils.tensorflow_utils.objectives import l2_loss
        from my_utils.tensorflow_utils.general import get_params_from_scope

        # Also use for SVDPiModel
        params = get_params_from_scope(
            self.main_classifier_name, trainable=True,
            excluded_subpattern="log_sigma2", verbose=True)

        loss = l2_loss(params)
        return loss

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

        # Weight decay loss
        if self.weight_decay > 0:
            l2_reg = self._l2_reg_loss()
        else:
            l2_reg = tf.constant(0.0, dtype=tf.float32)
        loss += self.weight_decay * l2_reg
        self.set_output('l2_reg', l2_reg)

        # Final loss
        self.set_output('loss', loss)

        print("All loss coefficients:")
        pp.pprint(self.loss_coeff_dict)
        # --------------------------------------- #

        self._built = True

    def get_loss(self):
        return {
            'loss': self.output_dict['loss'],
        }

    def get_train_params(self):
        main_classifier_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.main_classifier_name)

        assert len(main_classifier_params) > 0

        train_params = {
            'loss': main_classifier_params
        }

        for key, val in iteritems(train_params):
            assert len(val) > 0, "Loss '{}' has params {}".format(key, val)
            print("{}: {}".format(key, val))

        return train_params


# Pi+MUR (OK)
class PiModel_MUR(PiModel):
    def __init__(self, x_shape, y_shape,
                 input_perturber, main_classifier,
                 cons_mode='mse',
                 cons_4_unlabeled_only=False,
                 same_perturbed_inputs=False,
                 weight_decay=-1.0,

                 mur_mode="mse_wrt_point",
                 mur_4_unlabeled_only=False,
                 mur_noise_radius=10.0,
                 mur_opt_steps=1,

                 # Only being used when mur_opt_steps > 0,
                 mur_iter_mode="proj_grad_asc",
                 mur_opt_lr=0.01):

        PiModel.__init__(self, x_shape, y_shape,
                         input_perturber, main_classifier,
                         cons_mode=cons_mode,
                         cons_4_unlabeled_only=cons_4_unlabeled_only,
                         same_perturbed_inputs=same_perturbed_inputs,
                         weight_decay=weight_decay)

        possible_mur_modes = ['mse_wrt_point', 'mse_wrt_neigh']
        assert mur_mode in possible_mur_modes, "'mur_mode' must be in {}. " \
            "Found {}!".format(possible_mur_modes, mur_mode)
        assert mur_mode == "mse_wrt_point", \
            "You should set 'mur_mode'='mse_wrt_point' to obtain good performance!"

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

        # IMPORTANT: We use 'x_pert' to ensure no perturbation on the input
        # (batch, x_dim)
        x0 = self.get_output('x_pert')

        # (batch, num_classes)
        y0_prob = self.get_output('y_dist')['prob']
        # (batch,)
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

        # Iterative approximation
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
                    x_t = tf.stop_gradient(x_t + lr * (grad_x_t - g0_norm / rad * (x_t - x0) *
                        (2 - rad / (xt_m_x0_norm + 1e-15))))
                    cond_ent_t = graph_replace(cond_ent0, replacement_ts={x0: x_t})

                x_final = x_t

            elif self.mur_iter_mode == "proj_grad_asc":
                print("Iterative approximation of x* using projected gradient ascent!")
                x_t = x0
                cond_ent_t = cond_ent0

                for _ in range(self.mur_opt_steps):
                    grad_x_t = tf.gradients(cond_ent_t, [x_t])[0]

                    z_t = x_t + lr * grad_x_t

                    # (batch, 1, 1, 1)
                    zt_m_x0_norm = tf.stop_gradient(tf.sqrt(tf.reduce_sum(
                        (z_t - x0) ** 2, axis=normalized_axes, keepdims=True)) + 1e-15)

                    cond = tf.cast(tf.less_equal(zt_m_x0_norm, rad), dtype=tf.float32)
                    x_t = cond * z_t + (1.0 - cond) * (x0 + rad * (z_t - x0) / zt_m_x0_norm)
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

        print("All loss coefficients:")
        pp.pprint(self.loss_coeff_dict)
        # --------------------------------------- #

        self._built = True


# Pi+VD (OK)
class PiModel_VD(PiModel):
    def __init__(self, x_shape, y_shape,
                 input_perturber, main_classifier,

                 cons_mode='mse',
                 cons_4_unlabeled_only=False,
                 same_perturbed_inputs=False,
                 weight_decay=-1.0,

                 mask_weights=True,
                 cons_against_mean=True):

        PiModel.__init__(self, x_shape=x_shape, y_shape=y_shape,
                         input_perturber=input_perturber,
                         main_classifier=main_classifier,
                         cons_mode=cons_mode,
                         cons_4_unlabeled_only=cons_4_unlabeled_only,
                         same_perturbed_inputs=same_perturbed_inputs,
                         weight_decay=weight_decay)

        # If True: Mask the weights of the deterministic model
        self.mask_weights = mask_weights

        # If True: Computing consistency loss between
        # the stochastic model and the deterministic model
        self.cons_against_mean = cons_against_mean

        print("In class [{}]:".format(self.__class__.__name__))
        print("mask_weights: {}".format(self.mask_weights))
        print("cons_against_mean: {}".format(self.cons_against_mean))

    def _forward(self):
        from my_utils.tensorflow_utils.bayesian import svd

        y = tf.one_hot(self.y_ph, depth=self.num_classes,
                       on_value=tf.constant(1, dtype=tf.int32),
                       off_value=tf.constant(0, dtype=tf.int32),
                       dtype=tf.int32)
        y = tf.stop_gradient(tf.cast(y, dtype=tf.float32))

        x_pert = self.input_perturber_fn(self.x_ph)
        if self.same_perturbed_inputs:
            xa_pert = x_pert
        else:
            xa_pert = self.input_perturber_fn(self.x_ph)

        y_dist_sto = self.main_classifier_fn(x_pert, weight_mode=svd.NOISY_WEIGHT_MODE)
        ya_dist_sto = self.main_classifier_fn(xa_pert, weight_mode=svd.NOISY_WEIGHT_MODE)
        ya_dist_det = self.main_classifier_fn(
            xa_pert, weight_mode=svd.MASKED_WEIGHT_MODE if self.mask_weights else svd.STD_WEIGHT_MODE)

        label_flag = tf.cast(self.label_flag_ph, dtype=tf.float32)
        num_labeled = tf.reduce_sum(label_flag, axis=0)

        label_flag_inv = tf.cast(tf.logical_not(self.label_flag_ph), dtype=tf.float32)
        num_unlabeled = tf.reduce_sum(label_flag_inv, axis=0)

        return {'x_pert': x_pert, 'xa_pert': xa_pert,
                'y': y, 'y_dist_sto': y_dist_sto,
                'ya_dist_sto': ya_dist_sto, 'ya_dist_det': ya_dist_det,
                'label_flag': label_flag, 'label_flag_inv': label_flag_inv,
                'num_labeled': num_labeled, 'num_unlabeled': num_unlabeled}

    def _class_loss(self):
        # This function considers both labeled and unlabeled data in a single batch
        y_idx = self.y_ph
        y = self.get_output('y')
        y_dist_sto = self.get_output('ya_dist_sto')
        y_dist_det = self.get_output('ya_dist_det')
        label_flag = self.get_output('label_flag')
        label_flag_inv = self.get_output('label_flag_inv')
        num_labeled = self.get_output('num_labeled')
        num_unlabeled = self.get_output('num_unlabeled')

        y_logit_sto, y_prob_sto = y_dist_sto['logit'], y_dist_sto['prob']

        # Cross entropy loss for labeled data
        cross_ent_l = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=y_logit_sto, dim=-1)
        cross_ent_l = tf.reduce_sum(cross_ent_l * label_flag, axis=0) * 1.0 / (num_labeled + 1e-8)

        # Conditional entropy loss for unlabeld data
        cond_ent_u = tf.reduce_sum(-y_prob_sto * log_w_clip(y_prob_sto), axis=1)
        cond_ent_u = tf.reduce_sum(cond_ent_u * label_flag_inv, axis=0) * 1.0 / (num_unlabeled + 1e-8)

        y_pred_sto = tf.argmax(y_prob_sto, axis=1, output_type=tf.int32)
        y_matched_sto = tf.cast(tf.equal(y_pred_sto, y_idx), dtype=tf.float32)
        acc_y_l_sto = tf.reduce_sum(y_matched_sto * label_flag, axis=0) * 1.0 / (num_labeled + 1e-8)
        acc_y_u_sto = tf.reduce_sum(y_matched_sto * label_flag_inv, axis=0) * 1.0 / (num_unlabeled + 1e-8)
        acc_y_sto = tf.reduce_mean(y_matched_sto, axis=0)

        y_prob_det = y_dist_det['prob']
        y_pred_det = tf.argmax(y_prob_det, axis=1, output_type=tf.int32)
        y_matched_det = tf.cast(tf.equal(y_pred_det, y_idx), dtype=tf.float32)
        acc_y_l_det = tf.reduce_sum(y_matched_det * label_flag, axis=0) * 1.0 / (num_labeled + 1e-8)
        acc_y_u_det = tf.reduce_sum(y_matched_det * label_flag_inv, axis=0) * 1.0 / (num_unlabeled + 1e-8)
        acc_y_det = tf.reduce_mean(y_matched_det, axis=0)

        y_pred = y_pred_det
        acc_y_l = acc_y_l_det
        acc_y_u = acc_y_u_det
        acc_y = acc_y_det

        results = {
            'cross_ent_l': cross_ent_l,
            'cond_ent_u': cond_ent_u,

            'y_pred_sto': y_pred_sto,
            'acc_y_l_sto': acc_y_l_sto,
            'acc_y_u_sto': acc_y_u_sto,
            'acc_y_sto': acc_y_sto,

            'y_pred_det': y_pred_det,
            'acc_y_l_det': acc_y_l_det,
            'acc_y_u_det': acc_y_u_det,
            'acc_y_det': acc_y_det,

            'y_pred': y_pred,
            'acc_y_l': acc_y_l,
            'acc_y_u': acc_y_u,
            'acc_y': acc_y,
        }

        return results

    def _consistency_loss(self):
        y_prob = self.get_output('y_dist_sto')['prob']
        if self.cons_against_mean:
            ya_prob = self.get_output('ya_dist_det')['prob']
        else:
            ya_prob = self.get_output('ya_dist_sto')['prob']

        if self.cons_mode == 'mse':
            # IMPORTANT: Here, we take the sum over classes.
            # Implementations from other papers they use 'mean' instead of 'sum'.
            # This suggests that our 'cons_coeff' must be about 10, not 100 like other papers
            print("cons_mode=mse!")
            consistency = tf.reduce_sum(tf.square(y_prob - tf.stop_gradient(ya_prob)), axis=1)
        elif self.cons_mode == 'kld':
            print("cons_mode=kld!")
            from my_utils.tensorflow_utils.distributions import KLD_2Cats_v2
            consistency = KLD_2Cats_v2(y_prob, tf.stop_gradient(ya_prob))
        elif self.cons_mode == 'rev_kld':
            print("cons_mode=rev_kld!")
            from my_utils.tensorflow_utils.distributions import KLD_2Cats_v2
            consistency = KLD_2Cats_v2(tf.stop_gradient(ya_prob), y_prob)
        elif self.cons_mode == '2rand':
            print("cons_mode=2rand!")
            from my_utils.tensorflow_utils.activations import log_w_clip
            # IMPORTANT: We try to stop gradient here!
            # consistency = -log_w_clip(tf.reduce_sum(y_prob * ya_prob, axis=1))
            consistency = -log_w_clip(tf.reduce_sum(y_prob * tf.stop_gradient(ya_prob), axis=1))
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
            scopes=self.main_classifier_name)

        print("log_alphas: {}".format(log_alphas))
        assert len(log_alphas) > 0, "len(log_alphas) must be larger than 0!"

        weight_kld = tf.reduce_sum(tf.stack([svd.KL_qp_approx(la) for la in log_alphas]))
        num_weights = tf.reduce_sum(tf.stack([tf.reduce_prod(la.shape) for la in log_alphas]))
        weight_kld = weight_kld / (tf.cast(num_weights, dtype=tf.float32) + 1e-8)

        sparsity = svd.sparsity(log_alphas)

        return {
            'weight_kld': weight_kld,
            'sparsity': sparsity,
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

        # Weight decay loss
        if self.weight_decay > 0:
            l2_reg = self._l2_reg_loss()
        else:
            l2_reg = tf.constant(0.0, dtype=tf.float32)
        loss += self.weight_decay * l2_reg
        self.set_output('l2_reg', l2_reg)

        # Final loss
        self.set_output('loss', loss)

        print("All loss coefficients:")
        pp.pprint(self.loss_coeff_dict)
        # --------------------------------------- #

        self._built = True


# Pi+VD+MUR (OK)
class PiModel_VD_MUR(PiModel_VD):
    def __init__(self, x_shape, y_shape,
                 input_perturber, main_classifier,
                 cons_mode='mse',
                 cons_4_unlabeled_only=False,
                 same_perturbed_inputs=False,
                 weight_decay=-1.0,

                 mask_weights=True,
                 cons_against_mean=True,

                 mur_mode="mse_wrt_point",
                 mur_4_unlabeled_only=False,
                 mur_noise_radius=10.0,
                 mur_opt_steps=1,

                 # Only being used when mur_opt_steps > 0,
                 mur_iter_mode="proj_grad_asc",
                 mur_opt_lr=0.01):

        PiModel_VD.__init__(self, x_shape, y_shape,
                            input_perturber, main_classifier,
                            cons_mode=cons_mode,
                            cons_4_unlabeled_only=cons_4_unlabeled_only,
                            same_perturbed_inputs=same_perturbed_inputs,
                            weight_decay=weight_decay,
                            mask_weights=mask_weights,
                            cons_against_mean=cons_against_mean)

        possible_mur_modes = ['mse_wrt_point', 'mse_wrt_neigh']
        assert mur_mode in possible_mur_modes, "'mur_mode' must be in {}. " \
            "Found {}!".format(possible_mur_modes, mur_mode)
        assert mur_mode == "mse_wrt_point", \
            "You should set 'mur_mode'='mse_wrt_point' to obtain good performance!"

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

        # IMPORTANT: We use 'x_pert' to ensure no perturbation on the input
        # (batch, x_dim)
        x0 = self.get_output('x_pert')

        # IMPORTANT: The output here is 'y_dist_sto' not 'y_dist'
        # (batch, num_classes)
        y0_prob = self.get_output('y_dist_sto')['prob']
        # (batch,)
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

        # Iterative approximation
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
                    x_t = tf.stop_gradient(x_t + lr * (grad_x_t - g0_norm / rad * (x_t - x0) *
                        (2 - rad / (xt_m_x0_norm + 1e-15))))
                    cond_ent_t = graph_replace(cond_ent0, replacement_ts={x0: x_t})

                x_final = x_t

            elif self.mur_iter_mode == "proj_grad_asc":
                print("Iterative approximation of x* using projected gradient ascent!")
                x_t = x0
                cond_ent_t = cond_ent0

                for _ in range(self.mur_opt_steps):
                    grad_x_t = tf.gradients(cond_ent_t, [x_t])[0]

                    z_t = x_t + lr * grad_x_t

                    # (batch, 1, 1, 1)
                    zt_m_x0_norm = tf.stop_gradient(tf.sqrt(tf.reduce_sum(
                        (z_t - x0) ** 2, axis=normalized_axes, keepdims=True)) + 1e-15)

                    cond = tf.cast(tf.less_equal(zt_m_x0_norm, rad), dtype=tf.float32)
                    x_t = cond * z_t + (1.0 - cond) * (x0 + rad * (z_t - x0) / zt_m_x0_norm)
                    x_t = tf.stop_gradient(x_t)

                    cond_ent_t = graph_replace(cond_ent0, replacement_ts={x0: x_t})

                x_final = x_t

            else:
                raise ValueError(self.mur_iter_mode)

        y_prob_final = graph_replace(y0_prob, replacement_ts={x0: x_final})

        if self.mur_mode == "mse_wrt_neigh":
            mur = tf.reduce_sum(tf.square(y0_prob - tf.stop_gradient(y_prob_final)), axis=1)
        elif self.mur_mode == "mse_wrt_point":
            mur = tf.reduce_sum(tf.square(tf.stop_gradient(y0_prob) - y_prob_final), axis=1)
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

        print("All loss coefficients:")
        pp.pprint(self.loss_coeff_dict)
        # --------------------------------------- #

        self._built = True


# Maximum Uncertainty Training (derived from Pi+MUR)
class MUT(PiModel_MUR):
    def _mur_loss(self):
        from tensorflow.contrib.graph_editor import graph_replace

        # IMPORTANT: We use 'x_pert' to ensure no perturbation on the input
        # (batch, x_dim)
        x0 = self.get_output('x_pert')

        # (batch, num_classes)
        y0_prob = self.get_output('y_dist')['prob']
        # (batch,)
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

        # Iterative approximation
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
                    x_t = tf.stop_gradient(x_t + lr * (grad_x_t - g0_norm / rad * (x_t - x0) *
                        (2 - rad / (xt_m_x0_norm + 1e-15))))
                    cond_ent_t = graph_replace(cond_ent0, replacement_ts={x0: x_t})

                x_final = x_t

            elif self.mur_iter_mode == "proj_grad_asc":
                print("Iterative approximation of x* using projected gradient ascent!")
                x_t = x0
                cond_ent_t = cond_ent0

                for _ in range(self.mur_opt_steps):
                    grad_x_t = tf.gradients(cond_ent_t, [x_t])[0]

                    z_t = x_t + lr * grad_x_t

                    # (batch, 1, 1, 1)
                    zt_m_x0_norm = tf.stop_gradient(tf.sqrt(tf.reduce_sum(
                        (z_t - x0) ** 2, axis=normalized_axes, keepdims=True)) + 1e-15)

                    cond = tf.cast(tf.less_equal(zt_m_x0_norm, rad), dtype=tf.float32)
                    x_t = cond * z_t + (1.0 - cond) * (x0 + rad * (z_t - x0) / zt_m_x0_norm)
                    x_t = tf.stop_gradient(x_t)

                    cond_ent_t = graph_replace(cond_ent0, replacement_ts={x0: x_t})

                x_final = x_t

            else:
                raise ValueError(self.mur_iter_mode)

        y_prob_final = graph_replace(y0_prob, replacement_ts={x0: x_final})
        ya_prob = self.get_output('ya_dist')['prob']

        if self.mur_mode == "mse_wrt_point":
            mur = tf.reduce_sum(tf.square(tf.stop_gradient(ya_prob) - y_prob_final), axis=1)
        elif self.mur_mode == "mse_wrt_neigh":
            mur = tf.reduce_sum(tf.square(ya_prob - tf.stop_gradient(y_prob_final)), axis=1)
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

    # def _consistency_loss(self):
    #     raise NotImplementedError

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

        # IMPORTANT: We do not add 'cons' to loss but still record its value
        # Consistency loss
        results_cons = self._consistency_loss()
        # loss += coeff_fn(lc, 'cons') * results_cons['cons']
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

        print("All loss coefficients:")
        pp.pprint(self.loss_coeff_dict)
        # --------------------------------------- #

        self._built = True
