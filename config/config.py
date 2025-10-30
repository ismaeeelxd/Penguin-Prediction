from models.adaline import Adaline
from models.perceptron import Preceptron
from enum import Enum
from dataclasses import dataclass

class ModelType(Enum):
    PERCEPTRON = 0
    ADALINE = 1

class Species(Enum):
    ADELIE = 0
    CHINSTRAP = 1
    GENTOO = 2

@dataclass
class Config:
    DATASET_PATH = r'data\penguins.csv'
    CATEGORICAL_FEATURES = ['OriginLocation']
    TARGET_COLUMN = 'Species'
    FEATURES_COLUMNS = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'OriginLocation', 'BodyMass']
    SELECTED_FEATURES = [Species.ADELIE, Species.CHINSTRAP]
    FEATURE_COLUMNS = ['BodyMass', 'CulmenLength']
    TEST_SIZE = 0.4
    N_ITERS = 1000
    LEARNING_RATE = 0.01
    TARGET_CLASSES = ['Adelie', 'Chinstrap', 'Gentoo']
    MODELS = [Preceptron, Adaline]
    SELECTED_MODEL = ModelType.PERCEPTRON
    SELECTED_CLASSES = [Species.ADELIE, Species.CHINSTRAP]
    BIAS = 0.0
    MSE_THRESHOLD = 1e-3 if SELECTED_MODEL == ModelType.ADALINE else None
    
