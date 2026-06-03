import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pickle
import tensorflow as tf
import sionna.phy as sn
from sionna.phy.channel.tr38901 import TDL
from src.qQ_Method.qQ_Model import qQ_MODEL
from config import SEED, CARRIER_FREQ
sn.config.seed = SEED


def prepare_model(model_class, weights_path=None, build_args=(1, 0.0), **kwargs):
    r'''
        Load weights of a model class, if exsits
    '''
    model = model_class(**kwargs)
    model(*build_args)  # call the model once to build so weights could be loaded
    if weights_path:
        with open(weights_path, 'rb') as f:
            weights = pickle.load(f)
            model.set_weights(weights)
    return model


if __name__ == "__main__":

    qQ_model = prepare_model(qQ_MODEL, 'weights-qQ_Method')
    qQ_model._channel_model = TDL(model = "A", delay_spread = 10e-9, carrier_frequency = CARRIER_FREQ, min_speed = 0.0, max_speed = 0.0)
    qQ_model.visulaize_progress = True
    qQ_model.training = True
    qQ_model(1,40)

    qQ_model = prepare_model(qQ_MODEL, 'weights-qQ_Method')
    qQ_model._channel_model = TDL(model = "A", delay_spread = 600e-9, carrier_frequency = CARRIER_FREQ, min_speed = 0.0, max_speed = 0.0)
    qQ_model.visulaize_progress = True
    qQ_model.training = True
    qQ_model(1,40)    