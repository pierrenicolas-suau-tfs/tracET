# =============================================================================================
# Ad-hoc U-Net based models for MT segmentation in cryo-ET
# =============================================================================================

__author__ = 'Antonio Martinez Sanchez (anmartinezs@um.es)'

from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Dense, Flatten
from tensorflow.keras.models import Model


def deep_finder(dim_in, Ncl):
    input = Input(shape=(dim_in, dim_in, dim_in, 1))

    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(input)
    high = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)

    x = MaxPooling3D((2, 2, 2), strides=None)(high)

    x = Conv3D(48, (3, 3, 3), padding='same', activation='relu')(x)
    mid = Conv3D(48, (3, 3, 3), padding='same', activation='relu')(x)

    x = MaxPooling3D((2, 2, 2), strides=None)(mid)

    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)

    x = UpSampling3D(size=(2, 2, 2), data_format='channels_last')(x)
    x = Conv3D(64, (2, 2, 2), padding='same', activation='relu')(x)

    x = concatenate([x, mid])
    x = Conv3D(48, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(48, (3, 3, 3), padding='same', activation='relu')(x)

    x = UpSampling3D(size=(2, 2, 2), data_format='channels_last')(x)
    x = Conv3D(48, (2, 2, 2), padding='same', activation='relu')(x)

    x = concatenate([x, high])
    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)

    output = Conv3D(Ncl, (1, 1, 1), padding='same', activation='softmax')(x)

    model = Model(input, output)
    return model


def deep_finder_regression(dim_in):
    input = Input(shape=(dim_in, dim_in, dim_in, 1))

    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(input)
    high = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)

    x = MaxPooling3D((2, 2, 2), strides=None)(high)

    x = Conv3D(48, (3, 3, 3), padding='same', activation='relu')(x)
    mid = Conv3D(48, (3, 3, 3), padding='same', activation='relu')(x)

    x = MaxPooling3D((2, 2, 2), strides=None)(mid)

    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)

    x = UpSampling3D(size=(2, 2, 2), data_format='channels_last')(x)
    x = Conv3D(64, (2, 2, 2), padding='same', activation='relu')(x)

    x = concatenate([x, mid])
    x = Conv3D(48, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(48, (3, 3, 3), padding='same', activation='relu')(x)

    x = UpSampling3D(size=(2, 2, 2), data_format='channels_last')(x)
    x = Conv3D(48, (2, 2, 2), padding='same', activation='relu')(x)

    x = concatenate([x, high])
    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)

    output = Conv3D(1, (1, 1, 1), padding='same', activation='linear')(x)

    model = Model(input, output)
    return model


def light_3D_unet_segmentation(dim_in, Ncl):
    """
    Generates U-Net model with lighter (less number of filters) convolutional layers than DeepFinder
    :param dim_in: input cubic dimension
    :param Ncl: Number of output layers
    :return: the model generated
    """

    input = Input(shape=(dim_in, dim_in, dim_in, 1))

    x = Conv3D(16, (3, 3, 3), padding='same', activation='relu')(input)
    high = Conv3D(16, (3, 3, 3), padding='same', activation='relu')(x)

    x = MaxPooling3D((2, 2, 2), strides=None)(high)

    x = Conv3D(24, (3, 3, 3), padding='same', activation='relu')(x)
    mid = Conv3D(24, (3, 3, 3), padding='same', activation='relu')(x)

    x = MaxPooling3D((2, 2, 2), strides=None)(mid)

    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)

    x = UpSampling3D(size=(2, 2, 2), data_format='channels_last')(x)
    x = Conv3D(32, (2, 2, 2), padding='same', activation='relu')(x)

    x = concatenate([x, mid])
    x = Conv3D(24, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(24, (3, 3, 3), padding='same', activation='relu')(x)

    x = UpSampling3D(size=(2, 2, 2), data_format='channels_last')(x)
    x = Conv3D(24, (2, 2, 2), padding='same', activation='relu')(x)

    x = concatenate([x, high])
    x = Conv3D(16, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(16, (3, 3, 3), padding='same', activation='relu')(x)

    output = Conv3D(Ncl, (1, 1, 1), padding='same', activation='softmax')(x)

    model = Model(input, output)
    return model


def light_3D_unet_regression(dim_in):
    """
    Generates U-Net model with lighter (less number of filters) convolutional layers than DeepFinder for regression
    :param dim_in: input cubic dimension
    :return: the model generated
    """

    input = Input(shape=(dim_in, dim_in, dim_in, 1))

    x = Conv3D(16, (3, 3, 3), padding='same', activation='relu')(input)
    high = Conv3D(16, (3, 3, 3), padding='same', activation='relu')(x)

    x = MaxPooling3D((2, 2, 2), strides=None)(high)

    x = Conv3D(24, (3, 3, 3), padding='same', activation='relu')(x)
    mid = Conv3D(24, (3, 3, 3), padding='same', activation='relu')(x)

    x = MaxPooling3D((2, 2, 2), strides=None)(mid)

    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)

    x = UpSampling3D(size=(2, 2, 2), data_format='channels_last')(x)
    x = Conv3D(32, (2, 2, 2), padding='same', activation='relu')(x)

    x = concatenate([x, mid])
    x = Conv3D(24, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(24, (3, 3, 3), padding='same', activation='relu')(x)

    x = UpSampling3D(size=(2, 2, 2), data_format='channels_last')(x)
    x = Conv3D(24, (2, 2, 2), padding='same', activation='relu')(x)

    x = concatenate([x, high])
    x = Conv3D(16, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(16, (3, 3, 3), padding='same', activation='relu')(x)

    output = Conv3D(1, (1, 1, 1), padding='same', activation='linear')(x)

    model = Model(input, output)
    return model

