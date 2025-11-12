# config.py
DEBUG = False

# training
EPOCHS = 10
MOMENTUM = 0.9
LEARNING_RATE = 0.001
BATCH_SIZE = 1            # reduce drastically
THREADS = 0               # no multiprocessing
USE_CUDA = False          # since you're on CPU

# image settings
IMG_SIZE = (160, 160)     # smaller than 224x224
TRAIN_SPLIT = 0.8

# file paths
IMAGES_PATH = "dataset"
TRAINING_PATH = "train_file.csv"
VALIDATION_PATH = "test_file.csv"
TEST_PATH = "test_file.csv"

RESULTS_PATH = "results"
