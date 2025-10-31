from models.adaline import Adaline
from models.perceptron import Preceptron
from enum import Enum

class ModelType(Enum):
    PERCEPTRON = 0
    ADALINE = 1

class PenguinClass(Enum):
    ADELIE = 0
    CHINSTRAP = 1
    GENTOO = 2

class Config:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            inst = cls._instance
            
            inst.TEST_SIZE = 0.4
            inst.DATASET_PATH = r'data\penguins.csv'
            inst.CATEGORICAL_FEATURES = ['OriginLocation']
            inst.TARGET_COLUMN = 'Species'
            inst.FEATURES_COLUMNS = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'OriginLocation', 'BodyMass']
            inst.SELECTED_FEATURES = ['CulmenDepth', 'BodyMass']
            inst.N_ITERS = 5000
            inst.LEARNING_RATE = 0.001
            inst.BIAS = 0.0
            inst.MSE_THRESHOLD = 1e-4
            
            inst.TARGET_CLASSES = {
                PenguinClass.ADELIE: 'Adelie',
                PenguinClass.CHINSTRAP: 'Chinstrap',
                PenguinClass.GENTOO: 'Gentoo'
            }

            inst.MODELS = {
                ModelType.PERCEPTRON: Preceptron,
                ModelType.ADALINE: Adaline
            }
            
            inst.SELECTED_MODEL = ModelType.PERCEPTRON
            inst.SELECTED_CLASSES = [PenguinClass.ADELIE, PenguinClass.GENTOO]

        return cls._instance
