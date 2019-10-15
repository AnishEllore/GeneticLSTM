from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM


class SLSTM:
    def __init__(self, batch_size,epochs, look_back, lstm_units, is_eval=False, model_name=""):
        self.batch_size = batch_size
        self.look_back = look_back
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.model = load_model("saved_models/" + model_name) if is_eval else self._model()

    def _model(self):
        model = Sequential()
        model.add(LSTM(self.lstm_units, input_shape=(1, self.look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def train(self, trainX, trainY):
        self.model.fit(trainX, trainY, epochs=self.epochs, batch_size=self.batch_size, verbose=2)
