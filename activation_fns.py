from enum import Enum
import numpy as np

class ActivationEnum(str, Enum):
    UNIT = "unit"
    SIGN = "sign"

def register(name: ActivationEnum):
    def wrapper(func):
        ACTIVATIONS[name] = func
        return func
    return wrapper

ACTIVATIONS = {}


@register(ActivationEnum.UNIT)
def unit_step_func(x):
    return np.where(x > 0 , 1, 0)

@register(ActivationEnum.SIGN)
def signum(x):
    return np.where(x >= 0, 1, -1)
