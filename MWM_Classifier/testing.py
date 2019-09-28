# code for predicting the spectrum of a single star in normalized space.
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import torch


def predict_class(scaled_spectra, model):
    '''
    Predict the class of a single star.
    '''

    input = torch.FloatTensor(scaled_spectra[:,None,:])
    output = model(input)
    return output
