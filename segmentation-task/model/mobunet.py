import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, DenseNet169
import utils.layers as custom_layers

tf.config.run_functions_eagerly(True)


class MobUnet(tf.keras.Model):
    def __init__(self,
                 backbone='mobilenetv2', dropout=0.0,
                 cbam=False, activation='relu', norm='batchnorm'):
        super(MobUnet, self).__init__()
        self.OUTPUT_CHANNELS = 2
        self.DROP = dropout
        self.CBAM = cbam
        self.ACT = activation
        self.NORM = norm

        input_tensor = tf.keras.layers.Input(shape=(224, 224, 3))
        if backbone == 'mobilenetv2':
            self.base_model = MobileNetV2(input_tensor=input_tensor,
                                          input_shape=(224, 224, 3),
                                          include_top=False)

            # Use the activations of these layers
            self.layer_names = [
                'block_1_expand_relu',   # 112x112
                'block_3_expand_relu',   # 56x56
                'block_6_expand_relu',   # 28x28
                'block_13_expand_relu',  # 14x14
                'block_16_project',      # 7x7
            ]
        elif backbone == 'densenet169':
            self.base_model = DenseNet169(input_tensor=input_tensor,
                                          include_top=False)

            self.layer_names = [
                'conv1/relu',
                'conv2_block6_1_relu',
                'conv3_block12_1_relu',
                'conv4_block32_1_relu',
                'conv5_block32_1_relu',
            ]
        else:
            raise ValueError

    def upsample(self, filters, size, norm_type='batchnorm', dropout=0.0, activation='relu'):
        """Upsamples an input.
        Conv2DTranspose => Batchnorm => Dropout => Relu

        Args:
            filters: number of filters
            size: filter size
            norm_type: Normalization type; either 'batchnorm' or 'instancenorm'
            dropout: If > 0.0, adds the dropout layer with given value
            activation: Activation type; either 'relu' or 'arelu'

        Raises:
            ValueError: wrong norm_type
            ValueError: wrong activation

        Returns:
            result: upsampled input (Sequential Model)
        """

        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(custom_layers.InstanceNormalization())
        else:
            raise ValueError(norm_type)

        if dropout > 0.0:
            result.add(tf.keras.layers.Dropout(dropout))

        if activation == 'relu':
            result.add(tf.keras.layers.ReLU())
        elif activation == 'arelu':
            result.add(tf.keras.layers.Activation(custom_layers.arelu))
        else:
            raise ValueError(activation)

        return result

    def call(self):
        """Builds the model

        Returns:
            model: Sequential Model
        """
        layers = \
            [self.base_model.get_layer(
                name).output for name in self.layer_names]

        # Create the feature extraction model
        self.down_stack = tf.keras.Model(
            inputs=self.base_model.input, outputs=layers)
        self.down_stack.trainable = True

        self.up_stack = [
            self.upsample(512, 3, self.NORM, self.DROP, self.ACT),  # 4x4 -> 8x8
            self.upsample(256, 3, self.NORM, self.DROP, self.ACT),  # 8x8 -> 16x16
            self.upsample(128, 3, self.NORM, self.DROP, self.ACT),  # 16x16 -> 32x32
            self.upsample(64, 3, self.NORM, self.DROP, self.ACT),   # 32x32 -> 64x64
        ]

        model = self.unet_model(self.OUTPUT_CHANNELS, cbam=self.CBAM)

        return model

    def unet_model(self, output_channels, cbam=True):
        """Build unet model

        Args:
            output_channels: number of classes
            cbam: whether to use CBAM module

        Returns:
            model: tf.keras.Model
        """
        inputs = tf.keras.layers.Input(shape=[224, 224, 3])
        x = inputs

        # Downsampling through the model
        skips = self.down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        i = 0
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])
            if cbam:
                x = custom_layers.cbam_block(x, 'C'+'_cbam_block'+str(i),
                                             ratio=8)
                i += 1

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            output_channels, 3, strides=2,
            padding='same')  # 64x64 -> 128x128

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)
