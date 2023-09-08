from tensorflow.keras import layers, Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
import configs as cf 

def construct_model():
    base_model = InceptionV3(
    input_shape = (cf.IMAGE_SIZE, cf.IMAGE_SIZE, 3),
    include_top = False,
    weights = 'imagenet'
)
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    outputs = layers.Dense(29, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model
