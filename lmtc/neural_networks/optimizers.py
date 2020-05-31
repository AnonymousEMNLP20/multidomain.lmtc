from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import zip
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import Optimizer
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import clip_ops


class AdvancedAdam(Optimizer):
    """Adam optimizer.
    Adam optimizer, with learning rate multipliers built on Keras implementation
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)

    AUTHOR: Erik Brorson
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False,
                 multipliers=None, debug_verbose=False, **kwargs):
        super(AdvancedAdam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.multipliers = multipliers
        self.debug_verbose = debug_verbose

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):

            # Learning rate multipliers
            if self.multipliers:
                multiplier = [mult for mult in self.multipliers if mult in p.name]
            else:
                multiplier = None
            if multiplier:
                new_lr_t = lr_t * self.multipliers[multiplier[0]]
                if self.debug_verbose:
                    print('Setting {} to learning rate {}'.format(multiplier[0], new_lr_t))
                    print(K.get_value(new_lr_t))
            else:
                new_lr_t = lr_t
                if self.debug_verbose:
                    print('No change in learning rate {}'.format(p.name))
                    print(K.get_value(new_lr_t))
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - new_lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - new_lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad,
                  'multipliers': self.multipliers}
        base_config = super(AdvancedAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AccumulatedAdam(Optimizer):
    """AccumulatedAdam optimizer.

    Default parameters follow those provided in the original paper.

    Arguments:
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this algorithm
          from the paper "On the Convergence of Adam and Beyond".
    """

    def __init__(self,
                 lr=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=None,
                 decay=0.,
                 amsgrad=False,
                 accumulation_steps=2,
                 **kwargs):
        super(AccumulatedAdam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.loss = K.variable(0, dtype='float32', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.accum_iters = K.variable(accumulation_steps, K.dtype(self.iterations))
        self.accum_iters_float = K.cast(self.accum_iters, K.floatx())

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = []

        lr = self.lr
        completed_updates = K.cast(math_ops.floordiv(self.iterations, self.accum_iters), K.floatx())

        if self.initial_decay > 0:
            lr = lr * (  # pylint: disable=g-no-augmented-assignment
                    1. /
                    (1. +
                     self.decay * completed_updates))

        with ops.control_dependencies([state_ops.assign_add(self.iterations, 1)]):
            t = math_ops.cast(completed_updates + 1, K.floatx())
        lr_t = lr * (
                K.sqrt(1. - math_ops.pow(self.beta_2, t)) /
                (1. - math_ops.pow(self.beta_1, t)))

        # self.iterations incremented after processing a batch
        # batch:              1 2 3 4 5 6 7 8 9
        # self.iterations:    0 1 2 3 4 5 6 7 8
        # update_switch = 1:        x       x    (if accum_iters=4)
        update_switch = K.equal((self.iterations + 1) % self.accum_iters, 0)
        update_switch = K.cast(update_switch, K.floatx())

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat, tg in zip(params, grads, ms, vs, vhats, gs):
            sum_grad = tg + g
            avg_grad = sum_grad / self.accum_iters_float

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * avg_grad
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * math_ops.square(avg_grad)
            if self.amsgrad:
                vhat_t = math_ops.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(state_ops.assign(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(state_ops.assign(m, (1 - update_switch) * m + update_switch * m_t))
            self.updates.append(state_ops.assign(v, (1 - update_switch) * v + update_switch * v_t))
            self.updates.append(state_ops.assign(tg, (1 - update_switch) * sum_grad))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(state_ops.assign(p, (1 - update_switch) * p + update_switch * new_p))
        return self.updates

    # def get_updates(self, loss, params):
    #     self.updates = [] #state_ops.assign_add(self.iterations, 1)]
    #
    #     lr = self.lr
    #     completed_updates = K.cast(math_ops.floordiv(self.iterations, self.accum_iters), K.floatx())
    #
    #     if self.initial_decay > 0:
    #         lr = lr * (  # pylint: disable=g-no-augmented-assignment
    #                 1. /
    #                 (1. +
    #                  self.decay * completed_updates))
    #
    #     with ops.control_dependencies([state_ops.assign_add(self.iterations, 1)]):
    #         t = math_ops.cast(completed_updates + 1, K.floatx())
    #     lr_t = lr * (
    #             K.sqrt(1. - math_ops.pow(self.beta_2, t)) /
    #             (1. - math_ops.pow(self.beta_1, t)))
    #
    #     # self.iterations incremented after processing a batch
    #     # batch:              1 2 3 4 5 6 7 8 9
    #     # self.iterations:    0 1 2 3 4 5 6 7 8
    #     # update_switch = 1:        x       x    (if accum_iters=4)
    #     update_switch = K.equal((self.iterations + 1) % self.accum_iters, 0)
    #     update_switch = K.cast(update_switch, K.floatx())
    #
    #     ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    #     vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    #
    #     if self.amsgrad:
    #         vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    #     else:
    #         vhats = [K.zeros(1) for _ in params]
    #     self.weights = [self.iterations] + ms + vs + vhats
    #     print(update_switch)
    #     if math_ops(update_switch):
    #         grads = self.get_gradients(self.loss / self.accum_iters_float, params)
    #         for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
    #             m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
    #             v_t = (self.beta_2 * v) + (1. - self.beta_2) * math_ops.square(g)
    #             if self.amsgrad:
    #                 vhat_t = math_ops.maximum(vhat, v_t)
    #                 p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
    #                 self.updates.append(state_ops.assign(vhat, vhat_t))
    #             else:
    #                 p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)
    #
    #             self.updates.append(state_ops.assign(m,  m_t))
    #             self.updates.append(state_ops.assign(v, v_t))
    #             new_p = p_t
    #
    #             # Apply constraints.
    #             if getattr(p, 'constraint', None) is not None:
    #                 new_p = p.constraint(new_p)
    #
    #             self.updates.append(state_ops.assign(p, new_p))
    #         self.updates.append(state_ops.assign(self.loss, 0))
    #     else:
    #         self.updates.append(state_ops.assign_add(self.loss, loss))
    #         for p, m, v, vhat in zip(params, ms, vs, vhats):
    #             self.updates.append(state_ops.assign(m, m))
    #             self.updates.append(state_ops.assign(m, m))
    #             self.updates.append(state_ops.assign(v, v))
    #             self.updates.append(state_ops.assign(vhat, vhat))
    #             self.updates.append(state_ops.assign(p, p))
    #             self.updates.append(state_ops.assign(p, p))
    #
    #     return self.updates

    def get_gradients(self, loss, params):
        """Returns gradients of `loss` with respect to `params`.

        Arguments:
            loss: Loss tensor.
            params: List of variables.

        Returns:
            List of gradient tensors.

        Raises:
            ValueError: In case any gradient cannot be computed (e.g. if gradient
              function not implemented).
        """
        grads = K.gradients(loss, params)
        # if None in grads:
        #     raise ValueError('An operation has `None` for gradient. '
        #                      'Please make sure that all of your ops have a '
        #                      'gradient defined (i.e. are differentiable). '
        #                      'Common ops without gradient: '
        #                      'K.argmax, K.round, K.eval.')
        if hasattr(self, 'clipnorm'):
            grads = [clip_ops.clip_by_norm(g, self.clipnorm) for g in grads]
        if hasattr(self, 'clipvalue'):
            grads = [
                clip_ops.clip_by_value(g, -self.clipvalue, self.clipvalue)
                for g in grads
            ]
        return grads

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
            'accum_iters': float(K.get_value(self.accum_iters))
        }
        base_config = super(AccumulatedAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
