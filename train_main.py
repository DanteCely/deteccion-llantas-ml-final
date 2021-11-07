import numpy as np
from lib.Data import Algorithms, Normalize
from lib.Model.NeuralNetwork.FeedForward import FeedForward
from lib.Model.SVM.SVM import SVM
from lib.Optimizer.ADAM import ADAM
from lib.Optimizer.GradientDescent import GradientDescent
from lib.Helper.Parser.ArgParse import ArgParse


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

    if optimizer_type == 'adam':
        print('Adam optimizer')
        ADAM(
            cost=model_cost,
            learning_rate=arg.learning_rate,
            regularization=arg.regularization,
            debug_step=arg.debug_step,
            debug_function=debug,
            max_iter=arg.max_iterations
        )

    elif optimizer_type == 'desc':
        print('Gradient Descent optimizer')
        GradientDescent(
            cost=model_cost,
            alpha=args.learning_rate,
            beta1=args.beta1,
            beta2=args.beta2,
            max_iter=args.max_iterations,
            epsilon=args.epsilon,
            regularization=args.regularization,
            reg_type=args.reg_type,
            debug_step=args.debug_step,
            debug_function=debug
        )


def train_svm_model(x_train, y_train, x_test, y_test, arg):
    # Initialize SVM model
    svm_model = SVM(in_x=x_train, in_y=y_train)

    # Set SVM Cost model to ADAM optimizer
    svm_cost = SVM.Cost(svm_model, batch_size=arg.batch_size)

    optimizer(svm_cost, debug_function, arg)

    accuracy = compute_svm_accuracy(y_test, svm_model, x_test)
    print("SVM accuracy: ", str(accuracy * 100), "%")


def train_neural_network_model(x_train, y_train, x_test, y_test, arg):
    # Initialize neural network model
    neural_network_model = FeedForward()
    neural_network_model.LoadParameters(arg.nn_descriptor)

    # Define cost function
    nn_cost = FeedForward.Cost(x_train, y_train, neural_network_model, batch_size=arg.batch_size)
    nn_cost.SetPropagationTypeToBinaryCrossEntropy()

    optimizer(nn_cost, debug_function, arg)

    accuracy = compute_nn_accuracy(y_test, neural_network_model, x_test)
    print("Neural Network accuracy: ", str(accuracy * 100), "%")


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
            '-m', '--model-type', type=str, choices=['svm', 'nn'],
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

    X_tra, Y_tra, X_tst, Y_tst, *_ = Algorithms.SplitData(input_data, 1, train_size, test_size)
    X_tra, x_off, x_div = Normalize.Center(X_tra)
    X_tst = X_tst - x_off

    if model_type == 'svm':
        print('Train SVM model based in')
        train_svm_model(X_tra, Y_tra, X_tst, Y_tst, args)
    elif model_type == 'nn':
        print('Train neural network based in')
        train_neural_network_model(X_tra, Y_tra, X_tst, Y_tst, args)
