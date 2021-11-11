import math

import numpy as np
from lib.Data import Algorithms, Normalize
from lib.Model.NeuralNetwork.FeedForward import FeedForward
from lib.Model.SVM.SVM import SVM
from lib.Optimizer.ADAM import ADAM
from lib.Optimizer.GradientDescent import GradientDescent
from lib.Helper.Parser.ArgParse import ArgParse

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


def optimizer(model_cost, debug, arg):
    optimizer_type = args.optimizer_type

    return opt[optimizer_type](
        cost=model_cost,
        learning_rate=arg.learning_rate,
        regularization=arg.regularization,
        reg_type=args.reg_type,
        debug_step=arg.debug_step,
        debug_function=debug,
        max_iter=arg.max_iterations,
        alpha=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
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
    print('Train neural network based in')
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

    for i in range(len(train)):
        tra = train[i]
        label = labels[i]

        # First, randomize columns indexes
        np.random.shuffle(enum_cols)

        # Then, store those column in the array
        cols_1 = np.array(enum_cols[0:math.floor(train_columns / 2)], dtype=np.uint16)
        cols_2 = np.array(enum_cols[math.floor(train_columns / 2):train_columns], dtype=np.uint16)

        models_cols.append(cols_1)
        models_cols.append(cols_2)

        # Divide the columns in two groups
        tra_1 = tra[:, cols_1]
        tra_2 = tra[:, cols_2]

        # For column subgroup 1, create a SVM
        svm = SVM(in_x=tra_1, in_y=label)

        # For column subgroup 2, create a NN
        nn = FeedForward()
        nn_label = np.where(label == -1, 0, label)
        nn.LoadParameters(arg.nn_descriptor)

        svm_cost = SVM.Cost(m_Model=svm, batch_size=arg.batch_size)
        nn_cost = FeedForward.Cost(in_X=tra_2, in_Y=nn_label, batch_size=arg.batch_size, model=nn)
        nn_cost.SetPropagationTypeToBinaryCrossEntropy()

        # Append the models in an array
        rf_models.append(svm_cost)
        rf_models.append(nn_cost)

    # Start training process for each model
    cost = None
    for model in rf_models:
        cost = optimizer(model_cost=model, debug=debug_function, arg=arg)

    estimations = []

    # Compute output of each model
    for i in range(len(rf_models)):
        x_subset = x_test[:, models_cols[i]]
        model_est = rf_models[i].m_Model(x_subset)

        # If i is even, process SVM
        if i % 2 == 0:
            estimations.append(np.where(model_est < -1, -1, np.where(model_est > 1, 1, 0)))
        # Otherwise, process NN
        else:
            estimations.append(np.where(np.round(model_est) == 0, -1, 1))

    estimations = np.array(estimations)
    # After that, compute random forest output by voting
    rf_estimation = []
    bins = [-1, 0, 1]
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


def some_func(args, accuracy, cost):
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
              + '\n'

    return to_algo


def write_data(data):
    file_object = open('./experiments.csv', 'a')
    file_object.write(data)
    file_object.close()


if __name__ == "__main__":

    # Steps for training the model:
    #   1. Choose train and test data, guarantee that both groups are balanced
    #   2. Instantiate each of the models to train
    #   3. Set hiperparameters and optimization strategy (Gradient descent - ADAM)
    #   4. Train the models
    #   5. Compute confusion matrix with test dataset
    #   6. Compare results

    parser = ArgParse()
    parser.add_argument('-es', '--experiments-steps', type=float, default=10)
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
    X_tra, x_off, x_div = Normalize.Center(X_tra)
    X_tst = X_tst - x_off
    print('End Split Data')

    meta_model_type = ['svm']  ## Opcion 1
    # meta_model_type = ['nn'] ## Opcion 2
    # meta_model_type = ['random_forest'] ## Opcion 3
    meta_optimizer_type = ['adam', 'desc']
    meta_nn_descriptor = ['']  ## Opcion 1
    # meta_nn_descriptor = ['dataset-tire/nn_architecture/nn_01.nn', 'dataset-tire/nn_architecture/nn_02.nn'] ## Opcion 2
    # meta_nn_descriptor = ['dataset-tire/nn_architecture/random_forest_nn_01.nn'] ## Opcion 3  
    meta_reg_type = ['0', 'lasso', 'ridge']
    meta_batch_size = [-1, 16, 64]
    meta_regularization = [0, 0.01, 100]
    meta_max_iterations = [100, 500, 1000]
    meta_learning_rate = [1e-2, 1e-4, 1e-6, 1e-8]

    cadena = 'model_type' \
             + ',' + 'optimizer_type' \
             + ',' + 'nn_descriptor' \
             + ',' + 'reg_type' \
             + ',' + 'batch_size' \
             + ',' + 'regularization' \
             + ',' + 'max_iterations' \
             + ',' + 'learning_rate' \
             + ',' + 'accuracy' \
             + ',' + 'cost' \
             + '\n'

    experiments_amount = 0
    experiments_steps = args.experiments_steps

    for model in meta_model_type:
        args.model_type = model
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
                                args.debug_step = max_iterations - 1
                                for learning_rate in meta_learning_rate:
                                    args.learning_rate = learning_rate

                                    try:
                                        accuracy, final_cost = models[args.model_type](X_tra, Y_tra, X_tst, Y_tst, args)
                                        newAccuracy = str(accuracy * 100)
                                        cadena += some_func(args, newAccuracy, final_cost)

                                    except RuntimeError:
                                        print("Error!")
                                        cadena += some_func(args, '0', '-1')

                                    finally:
                                        experiments_amount += 1

                                        if experiments_amount % experiments_steps == 0:
                                            write_data(cadena)
                                            cadena = ''
