import math

import numpy as np
from lib.Data import Algorithms, Normalize
from lib.Model.NeuralNetwork.FeedForward import FeedForward
from lib.Model.SVM.SVM import SVM
from lib.Optimizer.ADAM import ADAM
from lib.Optimizer.GradientDescent import GradientDescent
from lib.Helper.Parser.ArgParse import ArgParse
import time
from sklearn.decomposition import PCA
import sys

opt = {
    'adam': ADAM,
    'desc': GradientDescent
}


def compute_estimation(model, x_test):
    return model(x_test)


def compute_nn_accuracy(expected, model, x_test):
    estimation = compute_estimation(model, x_test)
    processed_est = np.round(estimation)
    acc = np.where(processed_est == expected, 1, 0).mean()
    return acc


def compute_svm_accuracy(expected, model, x_test):
    estimation = compute_estimation(model, x_test)
    processed_est = np.where(estimation < 0, -1, 1)
    acc = np.where(processed_est == expected, 1, 0).mean()
    return acc


def debug_function(model, j, dj, i, show):
    if show:
        print("Iteration: ", i, ". Cost: ", j)


def optimizer(model_cost, debug, args):
    optimizer_type = args.optimizer_type

    return opt[optimizer_type](
        cost=model_cost,
        learning_rate=args.learning_rate,
        regularization=args.regularization,
        reg_type=args.reg_type,
        debug_step=args.debug_step,
        debug_function=None,
        max_iter=args.max_iterations,
        alpha=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        model_type=args.model_type
    )


def train_svm_model(x_train, y_train, x_test, y_test, arg):
    # Initialize SVM model
    # print('Train SVM model based in')
    svm_model = SVM(in_x=x_train, in_y=y_train)

    # Set SVM Cost model to ADAM optimizer
    svm_cost = SVM.Cost(m_Model=svm_model, batch_size=arg.batch_size)

    cost = optimizer(svm_cost, debug_function, arg)

    accuracy = compute_svm_accuracy(y_test, svm_model, x_test)
    return accuracy, cost


def train_neural_network_model(x_train, y_train, x_test, y_test, arg):
    # Initialize neural network model
    # print('Train neural network based in')
    neural_network_model = FeedForward()
    neural_network_model.LoadParameters(arg.nn_descriptor)

    # Change y_train labels from -1 to 0
    y_train = np.where(y_train == -1, 0, y_train)

    # Define cost function
    nn_cost = FeedForward.Cost(x_train, y_train, neural_network_model, batch_size=arg.batch_size)
    nn_cost.SetPropagationTypeToBinaryCrossEntropy()

    cost = optimizer(nn_cost, debug_function, arg)

    # Change y_test labels from -1 to 0
    y_test = np.where(y_test == -1, 0, y_test)
    accuracy = compute_nn_accuracy(y_test, neural_network_model, x_test)
    return accuracy, cost


def train_random_forest_model(x_train, y_train, x_test, y_test, arg):
    # First of all, split x_train in equal parts
    # Dataset is going to be split in two parts. And the split each part by a subset of x_train columns
    print('Random Forest based in')
    train_samples, train_columns = x_train.shape

    train = []
    labels = []

    x_train_1 = x_train[0:round(train_samples / 2), :]
    y_train_1 = y_train[0:round(train_samples / 2), :]

    train.append(x_train_1)
    labels.append(y_train_1)

    x_train_2 = x_train[round(train_samples / 2):train_samples, :]
    y_train_2 = y_train[round(train_samples / 2):train_samples, :]

    train.append(x_train_2)
    labels.append(y_train_2)

    enum_cols = np.linspace(0, train_columns - 1, train_columns)

    rf_models = []
    models_cols = []

    # ----- Use the first subgroup to train SVM models -----
    np.random.shuffle(enum_cols)
    cols_1_svm = np.array(enum_cols[0:math.floor(train_columns / 3)], dtype=np.uint16)
    cols_2_svm = np.array(enum_cols[math.floor(train_columns / 3):math.floor(train_columns / 3) * 2], dtype=np.uint16)
    cols_3_svm = np.array(enum_cols[math.floor(train_columns / 3) * 2:], dtype=np.uint16)

    models_cols.append(cols_1_svm)
    models_cols.append(cols_2_svm)
    models_cols.append(cols_3_svm)

    train_1_svm = train[0][:, cols_1_svm]
    train_2_svm = train[0][:, cols_2_svm]
    train_3_svm = train[0][:, cols_3_svm]

    svm_trains = [train_1_svm, train_2_svm, train_3_svm]

    for svm_train in svm_trains:
        svm = SVM(in_x=svm_train, in_y=labels[0])
        svm_cost = SVM.Cost(m_Model=svm, batch_size=arg.batch_size)
        rf_models.append(svm_cost)

    # ----- Second subgroup is going to be used for NN models -----
    np.random.shuffle(enum_cols)
    cols_1_nn = np.array(enum_cols[0:math.floor(train_columns / 2)], dtype=np.uint16)
    cols_2_nn = np.array(enum_cols[math.floor(train_columns / 2):], dtype=np.uint16)

    models_cols.append(cols_1_nn)
    models_cols.append(cols_2_nn)

    tra_1_nn = train[1][:, cols_1_nn]
    tra_2_nn = train[1][:, cols_2_nn]

    nn_trains = [tra_1_nn, tra_2_nn]
    nn_labels = np.where(labels[1] == -1, 0, labels[1])

    for nn_train in nn_trains:
        nn = FeedForward()
        nn.LoadParameters(args.nn_descriptor)
        nn_cost = FeedForward.Cost(in_X=nn_train, in_Y=nn_labels, batch_size=arg.batch_size, model=nn)
        nn_cost.SetPropagationTypeToBinaryCrossEntropy()
        rf_models.append(nn_cost)

    # Start training process for each model
    cost = None
    i = 0
    for rf_model in rf_models:
        cost = optimizer(model_cost=rf_model, debug=debug_function, args=arg)
        print("Model ", i, " trained successfully")
        i += 1

    estimations = []

    # Compute output for SVM
    for i in range(3):
        x_subset = x_test[:, models_cols[i].tolist()]
        svm_est = rf_models[i].m_Model(x_subset)
        estimations.append(np.where(svm_est < -1, -1, np.where(svm_est > 1, 1, 0)))

    # Compute output for NN
    for i in range(3, 5):
        x_subset = x_test[:, models_cols[i]]
        nn_est = rf_models[i].m_Model(x_subset)
        estimations.append(np.where(np.round(nn_est) == 0, -1, 1))

    estimations = np.array(estimations)
    # After that, compute random forest output by voting
    rf_estimation = []
    bins = [-1, 0, 1, 2]
    # Iterate over all test samples
    for i in range(estimations.shape[1]):
        # Get estimation vector
        est = estimations[:, i, 0]

        # Compute the histogram in the estimation vector and retrieve the index of the label with more votes
        max_votes = np.histogram(est, bins=bins)[0].argmax()

        # Place in the estimation the label with more votes
        rf_estimation.append(bins[max_votes])

    y_test = y_test.flatten()
    rf_estimation = np.array(rf_estimation)
    res = np.where(y_test == rf_estimation, 1, 0)
    bagging_accuracy = res.sum() / res.shape[0]

    return bagging_accuracy, cost


def concatResults(args, accuracy, cost, total_time):
    to_algo = args.model_type \
              + ',' + args.optimizer_type \
              + ',' + args.nn_descriptor \
              + ',' + args.reg_type \
              + ',' + str(args.batch_size) \
              + ',' + str(args.regularization) \
              + ',' + str(args.max_iterations) \
              + ',' + str(args.learning_rate) \
              + ',' + accuracy \
              + ',' + str(cost) \
              + ',' + total_time \
              + '\n'

    return to_algo


def write_data(data):
    file_object = open('./experiments_con_pca_sin_softmax.csv', 'a')
    file_object.write(data)
    file_object.close()


def get_total_time(start, end):
    return end - start


if __name__ == "__main__":

    # Steps for training the model:
    #   1. Choose train and test data, guarantee that both groups are balanced
    #   2. Instantiate each of the models to train
    #   3. Set hiperparameters and optimization strategy (Gradient descent - ADAM)
    #   4. Train the models
    #   5. Compute confusion matrix with test dataset
    #   6. Compare results

    parser = ArgParse()
    parser.add_argument('-es', '--experiments-steps', type=float, default=1)
    parser.add_argument('input_data_file', type=str)
    # parser.add_argument('nn_descriptor', type=str)  # Options: None or file direction
    parser.add_argument('-tr', '--train-size', type=float, default=0.7)
    parser.add_argument('-ts', '--test-size', type=float, default=0.3)
    parser.add_argument(
        '-m', '--model-type', type=str, choices=['svm', 'nn', 'random_forest'],
        default='svm'
    )
    parser.add_argument(
        '-o', '--optimizer-type', type=str, choices=['adam', 'desc'],
        default='adam'
    )
    args = parser.parse_args()
    input_data_file = args.input_data_file
    # model_type = args.model_type
    train_size = args.train_size
    test_size = args.test_size

    print('Train Size: ', train_size)
    print('Test Size: ', test_size)

    print('Load Data Start')
    input_data = np.loadtxt(input_data_file, delimiter=',')

    models = {
        'svm': train_svm_model,
        'nn': train_neural_network_model,
        'random_forest': train_random_forest_model
    }
    print('Load Data End')

    print('Start Split Data')
    X_tra, Y_tra, X_tst, Y_tst, *_ = Algorithms.SplitData(input_data, 1, train_size, test_size)

    # PCA ===============================================================
    pca = PCA(.9999999)
    pca.fit(X_tra)
    X_tra = pca.transform(X_tra)
    X_tst = pca.transform(X_tst)
    print('X_tra shape', X_tra.shape)
    print('X_tst shape', X_tst.shape)
    # PCA ===============================================================

    X_tra, x_off, x_div = Normalize.Center(X_tra)
    X_tst = X_tst - x_off
    print('End Split Data')

    meta_model_type = ['random_forest']  # Opcion 2
    meta_optimizer_type = ['adam', 'desc']
    meta_nn_descriptor = ['./dataset-tire/nn_architecture/nn_01_48_48.nn']
    meta_reg_type = ['ridge', '0', 'lasso']
    meta_batch_size = [-1, 16, 32]
    meta_regularization = [0, 0.01, 100]
    meta_max_iterations = [100, 500, 1000]
    meta_learning_rate = [1e-8, 1e-6, 1e-4, 1e-2]

    resultLine = 'model_type' \
                 + ',' + 'optimizer_type' \
                 + ',' + 'nn_descriptor' \
                 + ',' + 'reg_type' \
                 + ',' + 'batch_size' \
                 + ',' + 'regularization' \
                 + ',' + 'max_iterations' \
                 + ',' + 'learning_rate' \
                 + ',' + 'accuracy' \
                 + ',' + 'cost' \
                 + ',' + 'total_time' \
                 + '\n'

    # Init experiments
    experiments_amount = 0
    experiments_steps = args.experiments_steps

    print('============ Start Experiments')
    for exp_model in meta_model_type:
        args.model_type = exp_model
        for optimizer_type in meta_optimizer_type:
            args.optimizer_type = optimizer_type
            for nn_descriptor in meta_nn_descriptor:
                args.nn_descriptor = nn_descriptor
                for reg_type in meta_reg_type:
                    args.reg_type = reg_type
                    for batch_size in meta_batch_size:
                        args.batch_size = batch_size
                        for regularization in meta_regularization:
                            args.regularization = regularization
                            for max_iterations in meta_max_iterations:
                                args.max_iterations = max_iterations
                                for learning_rate in meta_learning_rate:
                                    args.learning_rate = learning_rate

                                    try:
                                        model = models[args.model_type]
                                        start_time = time.time()
                                        accuracy, final_cost = model(X_tra, Y_tra, X_tst, Y_tst, args)
                                        end_time = time.time()
                                        total_time = f'{get_total_time(start_time, end_time):.5f}'
                                        newAccuracy = str(accuracy * 100)
                                        resultLine += concatResults(args, newAccuracy, final_cost, total_time)

                                    except RuntimeError:
                                        print("Error!")
                                        resultLine += concatResults(args, '0', '-1', '0')

                                    finally:
                                        experiments_amount += 1
                                        print('Experiment ', experiments_amount, '=> Accuracy: NA Cost: NA')
                                        write_data(resultLine)
                                        resultLine = ''
