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

    opt[optimizer_type](
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
    print('Train SVM model based in')
    svm_model = SVM(in_x=x_train, in_y=y_train)

    # Set SVM Cost model to ADAM optimizer
    svm_cost = SVM.Cost(svm_model, batch_size=arg.batch_size)

    optimizer(svm_cost, debug_function, arg)

    accuracy = compute_svm_accuracy(y_test, svm_model, x_test)
    print("SVM accuracy: ", str(accuracy * 100), "%")


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

    optimizer(nn_cost, debug_function, arg)

    # Change y_test labels from -1 to 0
    y_test = np.where(y_test == -1, 0, y_test)
    accuracy = compute_nn_accuracy(y_test, neural_network_model, x_test)
    print("Neural Network accuracy: ", str(accuracy * 100), "%")


def train_random_forest_model(x_train, y_train, x_test, y_test, arg):
    # First of all, split x_train in equal parts
    # Dataset is going to be split in two parts. And the split each part by a subset of x_train columns
    print('Random Forest based in')
    train_samples, train_columns = x_train.shape

    train = []
    labels = []

    x_train_1 = x_train[0:round(train_samples/2), :]
    y_train_1 = y_train[0:round(train_samples/2), :]

    train.append(x_train_1)
    labels.append(y_train_1)

    x_train_2 = x_train[round(train_samples/2):train_samples, :]
    y_train_2 = y_train[round(train_samples/2):train_samples, :]

    train.append(x_train_2)
    labels.append(y_train_2)

    # For first approach, each model is going to have 1/4 of x_train columns
    rng = np.random.default_rng()

    rf_models = []

    for i in range(len(train)):
        tra = train[i]
        label = labels[i]

        # First, randomize x_train columns
        rng.shuffle(tra, axis=1)

        # Divide the columns in two groups
        tra_1 = tra[:, 0:math.floor(train_columns / 2)]
        tra_2 = tra[:, math.floor(train_columns / 2):train_columns]

        # For column subgroup 1, create a SVM
        svm = SVM(in_x=tra_1, in_y=label)

        # For column subgroup 2, create a NN
        nn = FeedForward()
        nn_label = np.where(label == -1, 0, label)
        nn.LoadParameters(arg.nn_descriptor)

        svm_cost = SVM.Cost(m_model=svm, batch_size=arg.batch_size)
        nn_cost = FeedForward.Cost(in_X=tra_2, in_Y=nn_label, batch_size=arg.batch_size, model=nn)
        nn_cost.SetPropagationTypeToBinaryCrossEntropy()

        # Append the models in an array
        rf_models.append(svm_cost)
        rf_models.append(nn_cost)

    # Start training process for each model
    for model in rf_models:
        optimizer(model_cost=model, debug=debug_function, arg=arg)

    # TODO: compute model accuracy

if __name__ == "__main__":

    # Steps for training the model:
    #   1. Choose train and test data, guarantee that both groups are balanced
    #   2. Instantiate each of the models to train
    #   3. Set hiperparameters and optimization strategy (Gradient descent - ADAM)
    #   4. Train the models
    #   5. Compute confusion matrix with test dataset
    #   6. Compare results

    parser = ArgParse()
    parser.add_argument('input_data_file', type=str)
    parser.add_argument('nn_descriptor', type=str)  # Options: None or file direction
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
    model_type = args.model_type
    train_size = args.train_size
    test_size = args.test_size

    print('Train Size: ', train_size)
    print('Test Size: ', test_size)

    input_data = np.loadtxt(input_data_file, delimiter=',')

    models = {
        'svm': train_svm_model,
        'nn': train_neural_network_model,
        'random_forest': train_random_forest_model
    }

    X_tra, Y_tra, X_tst, Y_tst, *_ = Algorithms.SplitData(input_data, 1, train_size, test_size)
    X_tra, x_off, x_div = Normalize.Center(X_tra)
    X_tst = X_tst - x_off

    models[model_type](X_tra, Y_tra, X_tst, Y_tst, args)
