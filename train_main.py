import numpy as np
from lib.Data import Algorithms, Normalize
from lib.Model.NeuralNetwork.FeedForward import FeedForward
from lib.Model.SVM.SVM import SVM
from lib.Optimizer.ADAM import ADAM
from lib.Helper.Parser.ArgParse import ArgParse


def compute_estimation(model, x_test):
    return model(x_test)


def compute_accuracy(expected, model, x_test):
    estimation = compute_estimation(model, x_test)
    processed_est = np.where(estimation < 0, -1, 1)
    acc = np.where(processed_est == expected, 1, 0).mean()
    return acc


def debug_function(model, j, dj, i, show):
    if show:
        print("Iteration: ", i, ". Cost: ", j)


def train_svm_model(x_train, y_train, x_test, y_test, arg):
    # Initialize SVM model
    svm_model = SVM(in_x=x_train, in_y=y_train)

    # Set SVM Cost model to ADAM optimizer
    svm_cost = SVM.Cost(svm_model, batch_size=arg.batch_size)

    # run Adam optimizer
    ADAM(
        cost=svm_cost,
        learning_rate=arg.learning_rate,
        regularization=arg.regularization,
        debug_step=arg.debug_step,
        debug_function=debug_function,
        max_iter=arg.max_iterations
    )

    accuracy = compute_accuracy(y_test, svm_model, x_test)
    print("SVM accuracy: ", str(accuracy * 100), "%")


def train_neural_network_model(x_train, y_train, x_test, y_test, arg):
    # Initialize neural network model
    neural_network_model = FeedForward()
    neural_network_model.LoadParameters('./dataset-tire/nn_architecture/nn_01.nn')

    # Define cost function
    nn_cost = FeedForward.Cost(x_train, y_train, neural_network_model, batch_size=arg.batch_size)
    nn_cost.SetPropagationTypeToBinaryCrossEntropy()

    # run Adam optimizer
    ADAM(
        cost=nn_cost,
        learning_rate=arg.learning_rate,
        regularization=arg.regularization,
        debug_step=arg.debug_step,
        debug_function=debug_function,
        max_iter=arg.max_iterations
    )

    accuracy = compute_accuracy(y_test, neural_network_model, x_test)
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
    args = parser.parse_args()
    input_data = np.loadtxt('./dataset-tire/input_data.csv', delimiter=',')

    X_tra, Y_tra, X_tst, Y_tst, *_ = Algorithms.SplitData(input_data, 1, train_size=0.7, test_size=0.3)
    X_tra, x_off, x_div = Normalize.Center(X_tra)
    X_tst = X_tst - x_off

    # Train SVM model based in ADAM optimizer
    # train_svm_model(X_tra, Y_tra, X_tst, Y_tst, args)

    # Train neural network based in ADAM optimizer
    train_neural_network_model(X_tra, Y_tra, X_tst, Y_tst, args)
