import tensorflow as tf
from tensorflow.keras.callbacks import (ReduceLROnPlateau,
                                        ModelCheckpoint,
                                        EarlyStopping)

def build_model(model, dropout, img_size, num_classes=7, finetuning=False, unet=False):
    if model == 'mobilenetv2':
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet',
            pooling='max')
    elif model == 'mobilenetv3_l':
        preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
        base_model = tf.keras.applications.MobileNetV3Large(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet',
            pooling='max')
    elif model == 'inceptionv3':
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        base_model = tf.keras.applications.inception_v3.InceptionV3(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet',
            pooling='max')
    elif model == 'efficientnetb0':
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        base_model = tf.keras.applications.efficientnet.EfficientNetB0(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet',
            pooling='max')
    elif model == 'efficientnetb5':
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        base_model = tf.keras.applications.efficientnet.EfficientNetB5(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet',
            pooling='max')
    elif model == 'densenet201':
        preprocess_input = tf.keras.applications.densenet.preprocess_input
        base_model = tf.keras.applications.densenet.DenseNet201(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet',
            pooling='max')
    elif model == 'nasnet':
        preprocess_input = tf.keras.applications.nasnet.preprocess_input
        base_model = tf.keras.applications.nasnet.NASNetLarge(
                input_shape=(img_size, img_size, 3),
                include_top=False,
                weights='imagenet',
                pooling='max')
    elif model == 'vgg19':
        preprocess_input = tf.keras.applications.vgg19.preprocess_input
        base_model = tf.keras.applications.vgg19.VGG19(
                input_shape=(img_size, img_size, 3),
                include_top=False,
                weights='imagenet',
                pooling='max')
    elif model == 'resnet':
        preprocess_input = tf.keras.applications.resnet.preprocess_input
        base_model = tf.keras.applications.resnet.ResNet101(
                input_shape=(img_size, img_size, 3),
                include_top=False,
                weights='imagenet',
                pooling='max')
    else:
        raise ValueError

    base_model.trainable = False
    
    dropout_layer = tf.keras.layers.Dropout(dropout)
    prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
    
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = preprocess_input(inputs)
    x = base_model(x)
    x = dropout_layer(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def build_unet_model(dropout, unet_model_path, img_size=224, num_classes=7):
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet',
        pooling='max')
    
    unet_model = tf.keras.models.load_model(unet_model_path)
    unet_model = unet_model.get_layer(name='model')
    
    complementary_layers = tf.keras.Sequential([
        base_model.get_layer('block_16_project_BN'),
        base_model.get_layer('Conv_1'),
        base_model.get_layer('Conv_1_bn'),
        base_model.get_layer('out_relu'),
        base_model.get_layer('global_max_pooling2d')
    ])
    
    complementary_layers.trainable = False
    unet_model.trainable = False
    
    dropout_layer = tf.keras.layers.Dropout(dropout)
    prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
    
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = preprocess_input(inputs)
    x = unet_model(x)[-1]
    x = complementary_layers(x)
    x = dropout_layer(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    return model
    

def build_concatenated_model(dropout, unet_model_path, img_size=224, num_classes=7):
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    base_model = tf.keras.applications.MobileNetV2(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet',
            pooling='max')
    
    base_model.trainable = True
    
    unet_model = tf.keras.models.load_model(unet_model_path)
    unet_model = unet_model.get_layer(name='model')
    unet_model.trainable = True
    
    # Necessary layers for classification from base model
    complementary_layers = tf.keras.Sequential([
        base_model.get_layer('block_16_project_BN'),
        base_model.get_layer('Conv_1'),
        base_model.get_layer('Conv_1_bn'),
        base_model.get_layer('out_relu'),
        base_model.get_layer('global_max_pooling2d')
    ])
    
    imagenet_model = tf.keras.applications.MobileNetV2(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet',
            pooling='max')
    
    imagenet_model.trainable = True
    
    concat_layer = tf.keras.layers.Concatenate()
    dropout_layer = tf.keras.layers.Dropout(dropout)
    prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
    
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = preprocess_input(inputs)
    x1 = imagenet_model(x)
    x2 = unet_model(x)[-1]
    x2 = complementary_layers(x2)
    x = concat_layer([x1, x2])
    x = dropout_layer(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def unfreeze_model(model, num_layers):
    print('Total layers: ', len(model.layers))
    print('Unfreeze first ', num_layers)
    print('Freeze last ', len(model.layers)-num_layers-1)
    print('First unfrozen layer: ', model.layers[-num_layers].name)
    
    for layer in model.layers[-num_layers:]:
        layer.trainable = True
    for layer in model.layers[:len(model.layers)-num_layers]:
        layer.trainable = False