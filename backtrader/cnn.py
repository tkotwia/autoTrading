import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import RandomUniform, RandomNormal
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from matplotlib import pyplot as plt
import os

def f1_weighted(y_true, y_pred):
    y_true_class = tf.math.argmax(y_true, axis=1, output_type=tf.dtypes.int32)
    y_pred_class = tf.math.argmax(y_pred, axis=1, output_type=tf.dtypes.int32)
    conf_mat = tf.math.confusion_matrix(y_true_class, y_pred_class)  # can use conf_mat[0, :], tf.slice()
    # precision = TP/TP+FP, recall = TP/TP+FN
    rows, cols = conf_mat.get_shape()
    size = y_true_class.get_shape()[0]
    precision = tf.constant([0, 0, 0])  # change this to use rows/cols as size
    recall = tf.constant([0, 0, 0])
    class_counts = tf.constant([0, 0, 0])

    def get_precision(i, conf_mat):
        print("prec check", conf_mat, conf_mat[i, i], tf.reduce_sum(conf_mat[:, i]))
        precision[i].assign(conf_mat[i, i] / tf.reduce_sum(conf_mat[:, i]))
        recall[i].assign(conf_mat[i, i] / tf.reduce_sum(conf_mat[i, :]))
        tf.add(i, 1)
        return i, conf_mat, precision, recall

    def tf_count(i):
        elements_equal_to_value = tf.equal(y_true_class, i)
        as_ints = tf.cast(elements_equal_to_value, tf.int32)
        count = tf.reduce_sum(as_ints)
        class_counts[i].assign(count)
        tf.add(i, 1)
        return count

    def condition(i, conf_mat):
        return tf.less(i, 3)

    i = tf.constant(3)
    i, conf_mat = tf.while_loop(condition, get_precision, [i, conf_mat])

    i = tf.constant(3)
    c = lambda i: tf.less(i, 3)
    b = tf_count(i)
    tf.while_loop(c, b, [i])

    weights = tf.math.divide(class_counts, size)
    numerators = tf.math.multiply(tf.math.multiply(precision, recall), tf.constant(2))
    denominators = tf.math.add(precision, recall)
    f1s = tf.math.divide(numerators, denominators)
    weighted_f1 = tf.reduce_sum(tf.math.multiply(f1s, weights))
    return weighted_f1


def f1_metric(y_true, y_pred):
    """
    this calculates precision & recall
    """

    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # mistake: y_pred of 0.3 is also considered 1
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    # y_true_class = tf.math.argmax(y_true, axis=1, output_type=tf.dtypes.int32)
    # y_pred_class = tf.math.argmax(y_pred, axis=1, output_type=tf.dtypes.int32)
    # conf_mat = tf.math.confusion_matrix(y_true_class, y_pred_class)
    # tf.Print(conf_mat, [conf_mat], "confusion_matrix")

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


get_custom_objects().update({"f1_metric": f1_metric, "f1_weighted": f1_weighted})

class Cnn:
    
    def __init__(self, model_path):
        self.dim = 15
        self.model_path = model_path

    def _create_model(self):
        params = {'batch_size': 80, 'conv2d_layers': {'conv2d_do_1': 0.2, 'conv2d_filters_1': 32,
                                               'conv2d_kernel_size_1': 3, 'conv2d_mp_1': 0, 'conv2d_strides_1': 1,
                                               'kernel_regularizer_1':0.0, 'conv2d_do_2': 0.3, 'conv2d_filters_2': 64,
                                               'conv2d_kernel_size_2': 3, 'conv2d_mp_2': 2, 'conv2d_strides_2': 1,
                                               'kernel_regularizer_2':0.0, 'layers': 'two'},
           'dense_layers': {'dense_do_1': 0.2, 'dense_nodes_1': 128, 'kernel_regularizer_1':0.0, 'layers': 'one'},
           'epochs': 3000, 'lr': 0.001, 'optimizer': 'adam', 'input_dim_1': 15, 'input_dim_2': 15, 'input_dim_3': 3}

        model = Sequential()

        print("Training with params {}".format(params))
        # (batch_size, timesteps, data_dim)
        # x_train, y_train = get_data_cnn(df, df.head(1).iloc[0]["timestamp"])[0:2]
        conv2d_layer1 = Conv2D(params["conv2d_layers"]["conv2d_filters_1"],
                            params["conv2d_layers"]["conv2d_kernel_size_1"],
                            strides=params["conv2d_layers"]["conv2d_strides_1"],
                            kernel_regularizer=regularizers.l2(params["conv2d_layers"]["kernel_regularizer_1"]),
                            padding='same', activation="relu", use_bias=True,
                            kernel_initializer='glorot_uniform',
                            input_shape=(params['input_dim_1'],
                                            params['input_dim_2'], params['input_dim_3']))
        model.add(conv2d_layer1)

        if params["conv2d_layers"]['conv2d_mp_1'] > 1:
            model.add(MaxPool2D(pool_size=0))
        model.add(Dropout(params['conv2d_layers']['conv2d_do_1']))
        if params["conv2d_layers"]['layers'] == 'two':
            conv2d_layer2 = Conv2D(params["conv2d_layers"]["conv2d_filters_2"],
                                params["conv2d_layers"]["conv2d_kernel_size_2"],
                                strides=params["conv2d_layers"]["conv2d_strides_2"],
                                kernel_regularizer=regularizers.l2(params["conv2d_layers"]["kernel_regularizer_2"]),
                                padding='same', activation="relu", use_bias=True,
                                kernel_initializer='glorot_uniform')
            model.add(conv2d_layer2)
            if params["conv2d_layers"]['conv2d_mp_2'] > 1:
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
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', f1_metric])
        # from keras.utils.vis_utils import plot_model use this too for diagram with plot
        # model.summary(print_fn=lambda x: print(x + '\n'))
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
        print("preprocess_data begin")
        print(self.training_data.shape)
        print(self.training_val.shape)
        print(collections.Counter(self.training_val.ravel()))

        self._get_sample_weights()
        # self._feature_selection()

        # self.training_data = self._filter_features(self.training_data)

        self.one_hot_enc = OneHotEncoder(sparse=False, categories='auto')
        self.training_val = self.one_hot_enc.fit_transform(self.training_val)

        self.training_data = self._reshape_as_image(self.training_data)

        self.training_data = np.stack((self.training_data,) * 3, axis=-1)


        self._split_data()

        print(self.training_data.shape)
        if np.isnan(self.training_data).any():
            print('none')

        print("preprocess_data end")


    def _split_data(self, ratio=0.8):
        self.training_data, self.validation_data, self.training_val, self.validation_val, self.sample_weights, _ = train_test_split(self.training_data, self.training_val, self.sample_weights,
                                                                                                            train_size = ratio,
                                                                                                            test_size = 1-ratio,
                                                                                                            shuffle=True,
                                                                                                            stratify=self.training_val)
    
    def _feature_selection(self):
        num_features = 225  # should be a perfect square
        topk = 270
        select_k_best = SelectKBest(f_classif, k=topk)
        select_k_best.fit(self.training_data, self.training_val)
        features_A = select_k_best.get_support(indices=True)

        select_k_best = SelectKBest(mutual_info_classif, k=topk)
        select_k_best.fit(self.training_data, self.training_val)
        features_B = select_k_best.get_support(indices=True)

        self.features = features_A[np.in1d(features_A, features_B)]
        if len(self.features) < num_features:
            raise Exception(
                'number of common features found {} < {} required features. Increase "topK"'.format(len(self.features),
                                                                                                    num_features))
        print(len(self.features))
        self.features = sorted(self.features[0:num_features])    

    def _filter_features(self, input):
        output = input
        columns = output.shape[1]
        for i in reversed(range(columns)):
            if i not in self.features:
                output = np.delete(output, i, axis=1)
        return output

    def _reshape_as_image(self, data):
        x_temp = np.zeros((len(data), self.dim, self.dim))
        for i in range(data.shape[0]):
            x_temp[i] = np.reshape(data[i], (self.dim, self.dim))
        return x_temp

    def add_train_data(self, data, val):
        scaler = MinMaxScaler(feature_range=(0, 1))
        d = scaler.fit_transform(np.array(data).reshape(self.dim, -1)).reshape(1, -1)
        if hasattr(self, 'training_data'):
            self.training_data = np.append(self.training_data, d, axis=0)
        else:
            self.training_data = d

        if hasattr(self, 'training_val'):
            self.training_val = np.append(self.training_val, np.array(val).reshape(1,1), axis=0)
        else:
            self.training_val = np.array(val).reshape(1,1)

    def is_trained(self):
        return hasattr(self, 'model')

    def _plot_history(self, history):
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.plot(history.history['f1_metric'])
        plt.plot(history.history['val_f1_metric'])

        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train_loss', 'val_loss', 'f1', 'val_f1'], loc='upper left')
        plt.show()

    def start_traning(self):
        print("start_trainging")
        if os.path.exists(self.model_path):
            print('load model...')
            self.one_hot_enc = OneHotEncoder(sparse=False, categories='auto')
            self.one_hot_enc.fit(self.training_val)
            self.model = load_model(self.model_path)
            return

        self._create_model()

        self._preprocess_data()
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, min_delta=0.0001)
        rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.02, patience=20, verbose=1, mode='min',
                                min_delta=0.001, cooldown=1, min_lr=0.0001)
        mcp = ModelCheckpoint('/home/gene/git/autoTrading/backtrader/model/best_model', monitor='val_f1_metric', verbose=1,
                            save_best_only=True, save_weights_only=False, mode='max', period=1)

        history = self.model.fit(self.training_data, self.training_val, epochs=3000, verbose=1,
                                batch_size=64, shuffle=False,
                                validation_data=(self.validation_data, self.validation_val),
                                callbacks=[mcp, rlp, es],
                                sample_weight=self.sample_weights)

        # print('save model...')
        # self.model.save(self.model_path) 
        self._plot_history(history)

    def predict(self, data):
        scaler = MinMaxScaler()
        d = scaler.fit_transform(np.array(data).reshape(self.dim, -1)).reshape(1, -1)
        # d = self._filter_features(d)
        d = self._reshape_as_image(d)
        d = np.stack((d,) * 3, axis=-1)
        predict_raw = self.model.predict(d)
        result = self.one_hot_enc.inverse_transform(predict_raw)
        # print(predict_raw, result[0][0])
        return result[0][0]
