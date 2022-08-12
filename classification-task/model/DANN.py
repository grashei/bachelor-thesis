import tensorflow as tf
import numpy as np

BATCH_SIZE = 32

loss_metric = tf.keras.metrics.Mean(name="loss")
acc_metric = tf.keras.metrics.CategoricalAccuracy()

#Gradient Reversal Layer
@tf.custom_gradient
def gradient_reverse(x, lamda=1.0):
    y = tf.identity(x)
    
    def grad(dy):
        return lamda * -dy, None
    
    return y, grad


class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    def call(self, x, lamda=1.0):
        return gradient_reverse(x, lamda)

# Callback to raise lambda for domain adversarial training
class DomainClassifierCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs):
        super(DomainClassifierCallback, self).__init__()
        self.total_epochs = total_epochs
    
    def on_epoch_begin(self, epoch, logs=None):
        p = epoch / self.total_epochs
        lamda = 2 / (1 + np.exp(-100 * p, dtype=np.float32)) - 1
        lamda = lamda.astype('float32')
        self.model.lamda = lamda
        
# The DANN u-net model
class DANN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.output_channels = 2
        self.lamda = 1.0
        
        self.preprocessing_layer = tf.keras.applications.mobilenet_v2.preprocess_input
        
        self.base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            pooling='max'
        )
        
        # Use the activations of these layers
        self.layer_names = [
            'block_1_expand_relu',   # 112x112
            'block_3_expand_relu',   # 56x56
            'block_6_expand_relu',   # 28x28
            'block_13_expand_relu',  # 14x14
            'block_16_project',      # 7x7
        ]
        
        self.skip_layers = [self.base_model.get_layer(name).output for name in self.layer_names]
        
        self.down_stack = tf.keras.Model(inputs=self.base_model.input, outputs=self.skip_layers)
        self.down_stack.trainable = True
        
        self.up_stack = [
            self.upsample(512, 3, 'batchnorm', 0.0, 'relu'),  # 4x4 -> 8x8
            self.upsample(256, 3, 'batchnorm', 0.0, 'relu'),  # 8x8 -> 16x16
            self.upsample(128, 3, 'batchnorm', 0.0, 'relu'),  # 16x16 -> 32x32
            self.upsample(64, 3, 'batchnorm', 0.0, 'relu'),   # 32x32 -> 64x64
        ]
        
        self.output_layer = tf.keras.layers.Conv2DTranspose(
            self.output_channels, 3, strides=2,
            padding='same')
        
        # Domain classifier using the encoder outputs
        
        self.complementary_layers = tf.keras.Sequential([
            self.base_model.get_layer('block_16_project_BN'),
            self.base_model.get_layer('Conv_1'),
            self.base_model.get_layer('Conv_1_bn'),
            self.base_model.get_layer('out_relu'),
            self.base_model.get_layer('global_max_pooling2d')
        ])
        
        self.domain_predictor_layer1 = GradientReversalLayer()
        self.domain_predictor_layer2 = tf.keras.layers.Dense(100, activation='relu')
        self.domain_predictor_layer3 = tf.keras.layers.Dense(2, activation='softmax')
        
        
    def upsample(self, filters, size, norm_type='batchnorm', dropout=0.0, activation='relu'):
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
    
    def model(self):
        inputs = tf.keras.layers.Input(shape=[224, 224, 3])
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs, source_train=False))
    
        
    def call(self, inputs, source_train=True, lamda=1.0):
        # U-net encoder
        x = self.preprocessing_layer(inputs)
        skips = self.down_stack(x)
        x_dc = skips[-1]
        
        # U-net decoder
        x = tf.slice(skips[-1], [0, 0, 0, 0], [BATCH_SIZE, -1, -1, -1])
        skips = [tf.slice(skip, [0, 0, 0, 0], [BATCH_SIZE, -1, -1, -1]) for skip in skips]
        skips = reversed(skips[:-1])
        
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])
            
        x_unet = self.output_layer(x)
        
        # Domain classifier
        x_dc = self.complementary_layers(x_dc)
        x_dc = self.domain_predictor_layer1(x_dc)
        x_dc = self.domain_predictor_layer2(x_dc)
        x_dc = self.domain_predictor_layer3(x_dc)
        
        if source_train== True:
            return x_unet
        
        return x_unet, x_dc
    
    def train_step(self, data):
        s_images, s_labels, t_images = data
        
        d_labels = np.vstack([np.tile([1., 0.], [BATCH_SIZE, 1]),
                      np.tile([0., 1.], [BATCH_SIZE, 1])])
        d_labels = d_labels.astype('float32')
        
        images = tf.concat([s_images, t_images], 0)
        
        with tf.GradientTape() as tape:
            s_pred, d_pred = self(images, training=True, source_train=False, lamda=self.lamda)
            scce = tf.keras.losses.SparseCategoricalCrossentropy()
            cce = tf.keras.losses.CategoricalCrossentropy()
            loss_s = scce(y_true=s_labels, y_pred=s_pred)
            loss_d = cce(y_true=d_labels, y_pred=d_pred)
        
        gradients = tape.gradient([loss_s, loss_d], self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        loss_metric.update_state(loss_s)
        acc_metric.update_state(s_labels, s_pred)
        return {"loss": loss_metric.result(), "accuracy": acc_metric.result()}
    
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}