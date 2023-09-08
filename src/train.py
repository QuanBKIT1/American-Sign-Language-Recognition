from pipeline import get_meta_data, get_data
from utils import split_meta_data
from model import construct_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import configs as cf


def train_model():
    meta_data = get_meta_data(cf.TRAINING_DIR)
    train_meta_data, val_meta_data = split_meta_data(meta_data,test_size=cf.VALIDATION_SIZE+cf.TEST_SIZE)
    val_meta_data, test_meta_data = split_meta_data(meta_data=val_meta_data,test_size=cf.TEST_SIZE/(cf.VALIDATION_SIZE+cf.TEST_SIZE))

    train_gen = get_data(train_meta_data,batch_size=32)
    val_gen = get_data(val_meta_data,batch_size=32)
    model = construct_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    earlystopper = EarlyStopping(monitor="val_accuracy", patience=20, verbose=1)
    checkpoint = ModelCheckpoint(filepath=cf.CHECKPOINT_PATH,
                                                    save_weights_only=True,
                                                    monitor='val_accuracy',
                                                    mode = 'max',
                                                    verbose=1)
    train_history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=cf.EPOCHS,
    callbacks=[earlystopper,checkpoint]
)
    return train_history, model

if __name__ == '__main__':
    train_history, model = train_model()
    # Saving the model
    model.save(cf.MODEL_PATH)