import optuna

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import scipy as sp
from scipy.stats import pearsonr

INPUT_DIM = 10
MID_UNIT = 15
OUTPUT_DIM = 4
EPOCH = 50000

def nn_train(x_filename,y_filename):
  x_train = np.loadtxt(x_filename,delimiter=',')
  y_train = np.loadtxt(y_filename,delimiter=',')

  #r = np.zeros(29)
  #p = np.zeros(29)

  def objective(trial):
    #セッションのクリア
    K.clear_session()

    #最適化するパラメータの設定
    #中間層のユニット数
    mid_units = trial.suggest_int("mid_units", 5, 100)

    #活性化関数
    activation = trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh"])

    #optimizer
    optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam", "rmsprop"])

    #define NN framework
    model = Sequential()
    model.add(Dense(mid_units, input_dim = INPUT_DIM, activation=activation))
    model.add(Dense(OUTPUT_DIM, activation=activation))

    model.compile(optimizer=optimizer,
          loss="mean_squared_error",
          metrics=["accuracy"])

    history = model.fit(x_train, y_train, verbose=0, epochs=200, batch_size=128, validation_split=0.1)
    return 1 - history.history["val_acc"][-1]

  study = optuna.create_study()
  study.optimize(objective, n_trials=100)

  mid_units=study.best_params["mid_units"]
  activation=study.best_params["activation"]
  optimizer=study.best_params["optimizer"]

  best_params = ["mid_units:",str(mid_units),",activation:" ,activation,",optimizer:",optimizer]
  path_w  = "para_memo.txt"
  with open(path_w, mode='w') as f:
      f.writelines(best_params)

  #define NN framework
  model = Sequential()
  model.add(Dense(mid_units, input_dim = INPUT_DIM, activation=activation))
  model.add(Dense(OUTPUT_DIM, activation=activation))

  model.compile(optimizer=optimizer,
        loss="mean_squared_error",
        metrics=["accuracy"])

  train=model.fit(x=x_train, y=y_train, nb_epoch=EPOCH)
  model.save("model.h5")

  lossname = "loss_log.csv"
  np.savetxt(lossname,train.history['loss'])

if __name__ == '__main__':
  #set input, output data to train
  factor_train = "test_input.csv"
  face_train = "test_output.csv"

  nn_train(factor_train,face_train)