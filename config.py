class Config:
    DATASET_PATH = 'penguins.csv'
    CATEGORICAL_COLUMNS = ['OriginLocation']
    TARGET_COLUMN = 'Species'
    TEST_SIZE = 0.4
    N_ITERS = 1000
    LEARNING_RATE = 0.01