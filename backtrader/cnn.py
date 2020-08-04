from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import RandomUniform, RandomNormal
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from matplotlib import pyplot as plt
import os

class Cnn:
    
    def __init__(self):
        self.dim = 15
        self.training_data = np.zeros((0,self.dim,self.dim))
        self.training_val = np.zeros((0,1))

    def _create_model(self):
        params = {'batch_size': 60, 'conv2d_layers': {'conv2d_do_1': 0.0, 'conv2d_filters_1': 30,
                                               'conv2d_kernel_size_1': 2, 'conv2d_mp_1': 2, 'conv2d_strides_1': 1,
                                               'kernel_regularizer_1':0.0, 'conv2d_do_2': 0.01, 'conv2d_filters_2': 10,
                                               'conv2d_kernel_size_2': 2, 'conv2d_mp_2': 2, 'conv2d_strides_2': 2,
                                               'kernel_regularizer_2':0.0, 'layers': 'two'},
           'dense_layers': {'dense_do_1': 0.07, 'dense_nodes_1': 100, 'kernel_regularizer_1':0.0, 'layers': 'one'},
           'epochs': 3000, 'lr': 0.001, 'optimizer': 'adam', 'input_dim_1': 15, 'input_dim_2': 15, 'input_dim_3': 3}

        model = Sequential()

        print("Training with params {}".format(params))
        # (batch_size, timesteps, data_dim)
        # x_train, y_train = get_data_cnn(df, df.head(1).iloc[0]["timestamp"])[0:2]
        conv2d_layer1 = Conv2D(params["conv2d_layers"]["conv2d_filters_1"],
                            params["conv2d_layers"]["conv2d_kernel_size_1"],
                            strides=params["conv2d_layers"]["conv2d_strides_1"],
                            kernel_regularizer=regularizers.l2(params["conv2d_layers"]["kernel_regularizer_1"]),
                            padding='valid', activation="relu", use_bias=True,
                            kernel_initializer='glorot_uniform',
                            input_shape=(params['input_dim_1'],
                                            params['input_dim_2'], params['input_dim_3']))
        model.add(conv2d_layer1)
        if params["conv2d_layers"]['conv2d_mp_1'] >= 0:
            model.add(MaxPool2D(pool_size=2))
        model.add(Dropout(params['conv2d_layers']['conv2d_do_1']))
        if params["conv2d_layers"]['layers'] == 'two':
            conv2d_layer2 = Conv2D(params["conv2d_layers"]["conv2d_filters_2"],
                                params["conv2d_layers"]["conv2d_kernel_size_2"],
                                strides=params["conv2d_layers"]["conv2d_strides_2"],
                                kernel_regularizer=regularizers.l2(params["conv2d_layers"]["kernel_regularizer_2"]),
                                padding='valid', activation="relu", use_bias=True,
                                kernel_initializer='glorot_uniform')
            model.add(conv2d_layer2)
            if params["conv2d_layers"]['conv2d_mp_2'] >= 0:
                model.add(MaxPool2D(pool_size=2))
            model.add(Dropout(params['conv2d_layers']['conv2d_do_2']))

        model.add(Flatten())

        model.add(Dense(params['dense_layers']["dense_nodes_1"], activation='relu'))
        model.add(Dropout(params['dense_layers']['dense_do_1']))

        if params['dense_layers']["layers"] == 'two':
            model.add(Dense(params['dense_layers']["dense_nodes_2"], activation='relu',
                            kernel_regularizer=params['dense_layers']["kernel_regularizer_1"]))
            model.add(Dropout(params['dense_layers']['dense_do_2']))

        model.add(Dense(3, activation='softmax'))
        if params["optimizer"] == 'rmsprop':
            optimizer = optimizers.RMSprop(lr=params["lr"])
        elif params["optimizer"] == 'sgd':
            optimizer = optimizers.SGD(lr=params["lr"], decay=1e-6, momentum=0.9, nesterov=True)
        elif params["optimizer"] == 'adam':
            optimizer = optimizers.Adam(learning_rate=params["lr"], beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        # from keras.utils.vis_utils import plot_model use this too for diagram with plot
        model.summary(print_fn=lambda x: print(x + '\n'))
        self.model = model

    def _get_sample_weights(self):
        values = self.training_val.ravel().astype(int)  # compute_class_weight needs int labels
        class_weights = compute_class_weight('balanced', np.unique(values), values)

        print("real class weights are {}".format(class_weights), np.unique(values))
        print("value_counts", np.unique(values, return_counts=True))
        self.sample_weights = values.copy().astype(float)
        for i in np.unique(values):
            self.sample_weights[self.sample_weights == i] = class_weights[i]

    def _preprocess_data(self):
        import collections
        print(self.training_val.shape)
        print(collections.Counter(self.training_val.ravel()))

        self._get_sample_weights()

        self.one_hot_enc = OneHotEncoder(sparse=False, categories='auto')
        self.one_hot_enc.fit(self.training_val)
        print(self.training_val.shape)
        self.training_val = self.one_hot_enc.transform(self.training_val)
        print(self.training_val.shape)

        self.training_data = np.stack((self.training_data,) * 3, axis=-1)
        print(self.training_data.shape)
        if np.isnan(self.training_data).any():
            print('none')

        self._split_data()

    def _split_data(self, ratio=0.8):
        self.training_data, self.validation_data, self.training_val, self.validation_val, self.sample_weights, _ = train_test_split(self.training_data, self.training_val, self.sample_weights,
                                                                                                            train_size = ratio,
                                                                                                            test_size = 1-ratio,
                                                                                                            shuffle=True,
                                                                                                            stratify=self.training_val)

    def _plot_history(self, history):
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.plot(history.history['f1_metric'])
        plt.plot(history.history['val_f1_metric'])
        plt.title('Model Metrics')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train_loss', 'val_loss', 'train_acc', 'val_acc', 'f1', 'val_f1'], loc='upper left')
        plt.savefig(os.path.join("/home/gene/git/autoTrading/backtrader/history", 'plt'))
    
    def add_train_data(self, data, val):
        scaler = MinMaxScaler()
        d = scaler.fit_transform(np.array(data).reshape(self.dim, self.dim)).reshape(1, self.dim, self.dim)
        self.training_data = np.append(self.training_data, d, axis=0)
        self.training_val = np.append(self.training_val, np.array(val).reshape(1,1), axis=0)

    def is_trained(self):
        try:
            self.model
        except:
            return False
        else:
            return True

    def start_traning(self):
        self._create_model()
        self._preprocess_data()
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, min_delta=0.0001)
        rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.02, patience=10, verbose=1, mode='min',
                                min_delta=0.001, cooldown=1, min_lr=0.0001)
        mcp = ModelCheckpoint('/home/gene/git/autoTrading/backtrader/model/best_model', monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='min', period=1)

        history = self.model.fit(self.training_data, self.training_val, epochs=3000, verbose=0,
                                batch_size=64, shuffle=False,
                                validation_data=(self.validation_data, self.validation_val),
                                callbacks=[es, mcp, rlp],
                                sample_weight=self.sample_weights)

    def predict(self, data):
        scaler = MinMaxScaler()
        d = scaler.fit_transform(np.array(data).reshape(self.dim, self.dim)).reshape(1, self.dim, self.dim)
        d = np.stack((d,) * 3, axis=-1)
        predict_raw = self.model.predict(d)
        result = self.one_hot_enc.inverse_transform(predict_raw)
        # print(predict_raw, result[0][0])
        return result[0][0]
