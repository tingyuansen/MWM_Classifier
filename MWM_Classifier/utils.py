# a few low-level functions that are used throughout
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
from scipy import interpolate
import torch
import os
from . import training, testing, loadspec

# loadspec is written by David Nidever


def read_in_neural_network():

    '''
    Read in the trained networks and related results
    '''

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'neural_nets/training_loss_IR.npz')
    tmp = np.load(path)
    training_loss_IR = tmp["training_loss"]
    validation_loss_IR = tmp["validation_loss"]

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'neural_nets/training_loss_optical.npz')
    tmp = np.load(path)
    training_loss_optical = tmp["training_loss"]
    validation_loss_optical = tmp["validation_loss"]

    model_IR = training.classifier(dim_in=17037, num_neurons=30, num_channels=8,\
                               mask_size=5, stride_size=3, num_group=4)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'neural_nets/NN_normalized_spectra_IR.pt')
    state_dict = torch.load(path)
    model_IR.load_state_dict(state_dict)

    model_optical = training.classifier(dim_in=5331, num_neurons=30, num_channels=8,\
                               mask_size=5, stride_size=3, num_group=6)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'neural_nets/NN_normalized_spectra_optical.pt')
    state_dict = torch.load(path)
    model_optical.load_state_dict(state_dict)

    return model_IR, model_optical, training_loss_IR, validation_loss_IR, training_loss_optical, validation_loss_optical



def read_in_validation_spectra():

    '''
    Read in validation spectra (not used in training)
    We also only saved a small subset due to github size limit
    '''

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'spectra/IR_validation_spectra.npz')
    tmp = np.load(path)
    x_valid_IR = tmp["validation_spectra"]
    y_valid_IR = tmp["validation_labels"]

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'spectra/Optical_validation_spectra.npz')
    tmp = np.load(path)
    x_valid_optical = tmp["validation_spectra"]
    y_valid_optical = tmp["validation_labels"]

    return x_valid_IR, y_valid_IR, x_valid_optical, y_valid_optical


def read_in_IR_example():

    '''
    Read in an IR hot star example
    '''

    print("Reading in: apVisit-r8-4914-55807-100.fits")
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'spectra/apVisit-r8-4914-55807-100.fits')
    spec = loadspec.rdspec(path)
    wavelength = spec.wave
    flux = spec.flux
    flux_err = spec.err
    mask = spec.mask
    return wavelength, flux, flux_err, mask


def read_in_optical_example():

    '''
    Read in an optical FGKM example
    '''

    print("Reading in: spec-3586-55181-0220.fits")
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'spectra/spec-3586-55181-0220.fits')
    spec = loadspec.rdspec(path)
    wavelength = spec.wave
    flux = spec.flux
    flux_err = spec.err
    mask = spec.mask
    return wavelength, flux, flux_err, mask


def standardize_IR_spectrum(wave,spec):
    '''
    Interpolate onto the same wavelength grid
    '''
    # a fixed wavelength grid
    red_array = 16475 + np.arange((16947-16475)*10+1)/10.
    green_array = 15860 + np.arange((16432-15860)*10+1)/10.
    blue_array = 15145 + np.arange((15808-15145)*10+1)/10.

    # interpolate
    f_spec_red = interpolate.interp1d(wave[0,:],spec[0,:])
    f_spec_green = interpolate.interp1d(wave[1,:],spec[1,:])
    f_spec_blue = interpolate.interp1d(wave[2,:],spec[2,:])

    wavelength = np.concatenate([blue_array, green_array, red_array])
    spectrum = np.concatenate([f_spec_blue(blue_array),\
                               f_spec_green(green_array),f_spec_red(red_array)])

    return wavelength, spectrum


def standardize_optical_spectrum(wave,spec):
    '''
    Interpolate onto the same wavelength grid
    '''
    # a fixed wavelength grid
    wavelength = 3840 + np.arange((9170-3840)+1)

    # interpolate
    f_spec = interpolate.interp1d(wave,spec)
    spectrum = f_spec(wavelength)

    return wavelength, spectrum
