TRAINING_DIR = './data/asl_alphabet_train/asl_alphabet_train/'
TEST_DIR = './data/asl_alphabet_test/asl_alphabet_test/'

CLASSES = open("./classes.txt").readline().split()

IMAGE_SIZE = 200
CROP_SIZE = 400
BATCH_SIZE = 64
N_CLASSES = 29

TRAIN_SIZE = 0.8
VALIDATION_SIZE = 0.1
TEST_SIZE = 0.1

EPOCHS = 20

MODEL_PATH = './model/cnn_asl_model.h5'
CHECKPOINT_PATH = './checkpoint/cp.ckpt'