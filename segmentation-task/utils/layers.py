import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import tf_utils

tf.config.run_functions_eagerly(True)


@tf.function
def arelu(x, alpha=0.90, beta=2.0):
    """AReLU activation function

    Args:
        x ([type]): [description]
        alpha (float, optional): [description]. Defaults to 0.90.
        beta (float, optional): [description]. Defaults to 2.0.

    Returns:
        [type]: [description]
    """
    alpha_tensor = tf.constant([alpha],
                               dtype=tf.float32,
                               name='alpha_factor')
    beta_tensor = tf.constant([beta],
                              dtype=tf.float32,
                              name='beta_factor')
    alpha_tensor = tf.clip_by_value(alpha_tensor,
                                    clip_value_min=0.01,
                                    clip_value_max=0.99)
    beta_tensor = 1 + K.sigmoid(beta_tensor)
    res = K.relu(x) * beta_tensor - \
        K.relu(-x) * alpha_tensor

    return res


@tf.function
def cbam_block(input_feature, name, ratio=8):
    """Contains the implementation of
    Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.

    Args:
        input_feature ([type]): [description]
        name ([type]): [description]
        ratio (int, optional): [description]. Defaults to 8.

    Returns:
        [type]: [description]
    """
    attention_feature = channel_attention(input_feature, 'ch_at', ratio)
    attention_feature = spatial_attention(attention_feature, 'sp_at')

    return attention_feature


@tf.function
def channel_attention(input_feature, name, ratio=8):
    """Channel attention for CBAM

    Args:
        input_feature ([type]): [description]
        name ([type]): [description]
        ratio (int, optional): [description]. Defaults to 8.

    Returns:
        [type]: [description]
    """
    kernel_initializer = tf.keras.initializers.variance_scaling()
    bias_initializer = tf.constant_initializer(value=0.0)

    channel = input_feature.get_shape()[-1]
    avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)

    assert avg_pool.get_shape()[1:] == (1, 1, channel)
    avg_pool = tf.keras.layers.Dense(
        units=channel//ratio,
        activation=tf.nn.relu,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,)(avg_pool)
    assert avg_pool.get_shape()[1:] == (1, 1, channel//ratio)
    avg_pool = tf.keras.layers.Dense(
        units=channel,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,)(avg_pool)
    assert avg_pool.get_shape()[1:] == (1, 1, channel)

    max_pool = tf.reduce_max(input_feature, axis=[1, 2], keepdims=True)
    assert max_pool.get_shape()[1:] == (1, 1, channel)
    max_pool = tf.keras.layers.Dense(
        units=channel//ratio,
        activation=tf.nn.relu,)(max_pool)
    assert max_pool.get_shape()[1:] == (1, 1, channel//ratio)
    max_pool = tf.keras.layers.Dense(
        units=channel,)(max_pool)
    assert max_pool.get_shape()[1:] == (1, 1, channel)

    scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')

    return input_feature * scale


@tf.function
def spatial_attention(input_feature, name):
    """Spatial attention for CBAM

    Args:
        input_feature ([type]): [description]
        name ([type]): [description]

    Returns:
        [type]: [description]
    """
    kernel_size = 7
    kernel_initializer = tf.keras.initializers.variance_scaling()
    avg_pool = tf.reduce_mean(input_feature, axis=[3], keepdims=True)
    assert avg_pool.get_shape()[-1] == 1
    max_pool = tf.reduce_max(input_feature, axis=[3], keepdims=True)
    assert max_pool.get_shape()[-1] == 1
    concat = tf.concat([avg_pool, max_pool], 3)
    assert concat.get_shape()[-1] == 2

    concat = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=[kernel_size, kernel_size],
        strides=[1, 1],
        padding="same",
        activation=None,
        kernel_initializer=kernel_initializer,
        use_bias=False)(concat)
    assert concat.get_shape()[-1] == 1
    concat = tf.sigmoid(concat, 'sigmoid')

    return input_feature * concat


class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022).
    """
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    @tf.function
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


class AReLU(tf.keras.layers.Layer):
    def __init__(self, alpha=0.90, beta=2.0, **kwargs):
        super(AReLU, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def get_config(self):
        config = {
            'alpha': self.alpha,
            'beta': self.beta,
        }
        base_config = super(AReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.alpha_tensor = tf.Variable([self.alpha],
                                        dtype=tf.float32,
                                        name='alpha_factor')
        self.beta_tensor = tf.Variable([self.beta],
                                       dtype=tf.float32,
                                       name='beta_factor')
        return super().build(input_shape)

    @tf.function
    def call(self, inputs):
        self.alpha_tensor = tf.clip_by_value(self.alpha_tensor,
                                             clip_value_min=0.01,
                                             clip_value_max=0.99)
        self.beta_tensor = 1 + tf.nn.sigmoid(self.beta_tensor)
        res = tf.nn.relu(inputs) * self.beta_tensor - \
            tf.nn.relu(-inputs) * self.alpha_tensor

        return res

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape
