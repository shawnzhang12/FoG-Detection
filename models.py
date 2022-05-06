import tensorflow.keras as keras
import tensorflow.keras.metrics as metrics
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LSTM, Dropout, TimeDistributed
from tensorflow.keras.utils import to_categorical 
import tensorflow_addons as tfa

def CNN2D_model():
    input_shape = Input(shape=(64, 9, 1))
    tower_1 = Conv2D(filters=16, kernel_size=(5,9), padding='same', activation='relu')(input_shape)
    tower_1 = MaxPooling2D((3, 1), strides=(3, 1), padding='same')(tower_1)

    tower_2 = Conv2D(filters=16, kernel_size=(5,9), padding='same', activation='relu')(tower_1)
    tower_2 = MaxPooling2D((3, 1), strides=(3, 1), padding='same')(tower_2)

    tower_3 =  Conv2D(filters=16, kernel_size=(5,9), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((3, 1), strides=(3, 1), padding='same')(tower_3)

    merged = Flatten()(tower_3)

    out = Dense(100, activation='relu')(merged)
    out = Dense(10, activation='relu')(out)
    out = Dense(1, activation='sigmoid')(out)

    model = Model(input_shape, out)
    return model

def CNN1D_model():
    input_shape = Input(shape=(128, 9, 1))
    tower_1 = Conv1D(filters=16, kernel_size=(5), padding='same', activation='relu')(input_shape)
    tower_1 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_1)

    tower_2 = Conv1D(filters=16, kernel_size=(5), padding='same', activation='relu')(tower_1)
    tower_2 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_2)

    tower_3 =  Conv1D(filters=16, kernel_size=(5), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_3)

    merged = Flatten()(tower_3)

    out = Dense(100, activation='relu')(merged)
    out = Dropout(0.1)(out)
    out = Dense(10, activation='relu')(out)
    out = Dense(1, activation='sigmoid')(out)

    model = Model(input_shape, out)
    return model

def LSTM_model():
    input_shape = Input(shape=(128, 9))
    lstm1 = LSTM(50, dropout=0.05, return_sequences=True)(input_shape)
    lstm2 = LSTM(50, dropout=0.05)(lstm1)
    out = Dense(10, activation='relu')(lstm2)
    out = Dense(1, activation='sigmoid')(out)
    model = Model(input_shape, out)
    return model