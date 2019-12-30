import random
from collections import deque
import numpy as np
import cProfile
from keras import backend as K


class ReplayBuffer:
    """Constructs a buffer object that stores the past moves
    and samples a set of subsamples"""

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a, r, d, s2):
        """Add an experience to the buffer"""
        # S represents current state, a is action,
        # r is reward, d is whether it is the end,
        # and s2 is next state
        experience = (s, a, r, d, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample(self, batch_size):
        """Samples a total of elements equal to batch_size from buffer
        if buffer contains enough elements. Otherwise return all elements"""

        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        # Maps each experience in batch in batches of states, actions, rewards
        # and new states
        s_batch, a_batch, r_batch, d_batch, s2_batch = list(map(np.array,
                                                            list(zip(*batch))))

        return s_batch, a_batch, r_batch, d_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


def huber_loss(y_true, y_pred, clip_value):
    # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
    # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
    # for details.
    assert clip_value > 0.

    x = y_true - y_pred
    if np.isinf(clip_value):
        # Spacial case for infinity since Tensorflow does have problems
        # if we compare `K.abs(x) < np.inf`.
        return .5 * K.square(x)

    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        if hasattr(tf, 'select'):
            return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
        else:
            return tf.where(condition, squared_loss, linear_loss)  # condition, true, false
    elif K.backend() == 'theano':
        from theano import tensor as T
        return T.switch(condition, squared_loss, linear_loss)
    else:
        raise RuntimeError('Unknown backend "{}".'.format(K.backend()))
