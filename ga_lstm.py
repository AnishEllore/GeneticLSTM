import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as split

from keras.layers import LSTM, Input, Dense
from keras.models import Model

from deap import base, creator, tools, algorithms

from scipy.stats import bernoulli
from bitstring import BitArray

from pyspark.sql import SparkSession
import numpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
np.random.seed(1120)


def prepare_dataset(data, window_size):
    X, Y = np.empty((0, window_size)), np.empty((0))
    for i in range(len(data) - window_size - 1):
        X = np.vstack([X, data[i:(i + window_size), 0]])
        Y = np.append(Y, data[i + window_size, 0])
    X = np.reshape(X, (len(X), window_size, 1))
    Y = np.reshape(Y, (len(Y), 1))
    return X, Y


class GeneticAlgorithm:
    def __init__(self, data, population=4, generations=4):
        self.train_data = data
        self.population = population
        self.generations = generations
        self.generations_rmse = []
        self.population = self.genetic_algorithm()

    def train_evaluate(self, ga_individual_solution):
        # Decode GA solution to integer for window_size and num_units
        window_size_bits = BitArray(ga_individual_solution[0:6])
        num_units_bits = BitArray(ga_individual_solution[6:])
        window_size = window_size_bits.uint
        num_units = num_units_bits.uint
        print('\nWindow Size: ', window_size, ', Num of Units: ', num_units)

        # Return fitness score of 100 if window_size or num_unit is zero
        if window_size*num_units == 0:
            return 100,

            # Segment the train_data based on new window_size; split into train and validation (80/20)
        X, Y = prepare_dataset(self.train_data, window_size)
        X_train, X_val, y_train, y_val = split(X, Y, test_size=0.20, random_state=1120)

        # Train LSTM model and predict on validation set
        inputs = Input(shape=(window_size, 1))
        x = LSTM(num_units, input_shape=(window_size, 1))(inputs)
        predictions = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=5, batch_size=10, shuffle=True)
        y_pred = model.predict(X_val)

        # Calculate the RMSE score as fitness score for GA
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        self.generations_rmse.append(rmse)
        print('Validation RMSE: ', rmse, '\n')
        return rmse,

    def genetic_algorithm(self):
        population_size = self.population
        num_generations = self.generations
        gene_length = 10

        # As we are trying to minimize the RMSE score, that's why using -1.0.
        # In case, when you want to maximize accuracy for instance, use 1.0
        creator.create('FitnessMax', base.Fitness, weights=(-1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register('binary', bernoulli.rvs, 0.5)
        toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n=gene_length)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)

        toolbox.register('mate', tools.cxOrdered)
        toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.6)
        toolbox.register('select', tools.selRoulette)
        toolbox.register('evaluate', self.train_evaluate)

        population = toolbox.population(n=population_size)
        r = algorithms.eaSimple(population, toolbox, cxpb=0.4, mutpb=0.1, ngen=num_generations, verbose=False)
        return population


def load_data(file_name='data'):
    df = spark.read.format("csv").option("header", "true").load('data/'+file_name+'*'+'.csv')
    df.show()
    df = df.select(df.vmed.cast('float').alias('vmed'), df.ocupacion.cast('float').alias('occupancy'),
                   df.intensidad.cast('float').alias('value'))
    df.dropna()
    df_plot = df.select('value').toPandas()
    plt.plot(df_plot)
    plt.show()
    data = numpy.array(df.select('value').collect())
    return data


def main():
    data = load_data()
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    train_size = int(len(data) * 0.8)
    train, test = data[0:train_size, :], data[train_size:len(data), :]
    print(len(train), len(test))
    gaModel = GeneticAlgorithm(train, 100, 10)
    population = gaModel.population

    best_individuals = tools.selBest(population, k=1)
    best_window_size = None
    best_num_units = None

    for bi in best_individuals:
        window_size_bits = BitArray(bi[0:6])
        num_units_bits = BitArray(bi[6:])
        best_window_size = window_size_bits.uint
        best_num_units = num_units_bits.uint
        print('\nWindow Size: ', best_window_size, ', Num of Units: ', best_num_units)

    if best_window_size * best_num_units == 0:
        return
    X_train, y_train = prepare_dataset(train, best_window_size)
    X_test, y_test = prepare_dataset(test, best_window_size)

    inputs = Input(shape=(best_window_size, 1))
    x = LSTM(best_num_units, input_shape=(best_window_size, 1))(inputs)
    predictions = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=5, batch_size=10, shuffle=True)
    y_pred = model.predict(X_test)

    testPredict = scaler.inverse_transform(y_pred)
    testY = scaler.inverse_transform(y_test)

    rmse = np.sqrt(mean_squared_error(testY, testPredict))
    print('Test RMSE: ', rmse)
    generations_rmse = numpy.array(gaModel.generations_rmse)
    plt.plot(generations_rmse)
    plt.show()



if __name__ == "__main__":
    spark = SparkSession.builder \
        .master("local") \
        .appName("GAonLSTM") \
        .config("spark.executor.memory", "2gb") \
        .getOrCreate()
    main()
