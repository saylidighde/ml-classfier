"""Implementations of the linear SVM with the huberized hinge loss using fast gradient method, and the backtracking method

The fast gradient method uses a concept of "momentum" to speed up the descent. The momentum term increases for dimensions whose gradients 
point in the same directions and reduces updates for dimensions whose gradients change directions. As a result, we gain faster convergence 
and reduced oscillation.

The backtracking algorithm is a subroutine for fast gradient descent. The function takes as input the initial step-size value for the backtracking rule
and the target accuracy epsilon. It determines the most optimal learning rate for the gradient descent.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.stats
import sklearn.preprocessing
import sklearn.svm
from sklearn.model_selection import train_test_split
import scipy.linalg
import sklearn.linear_model
import os
from sklearn.svm import SVC
from sklearn import metrics

#Load and standardize the dataset
train_set = pd.read_table('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/vowel.train', sep=',', header=0)
test_set = pd.read_table('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/vowel.test', sep=',', header=0)
x_train = np.asarray(train_set)[:, 2:]
y_train = np.asarray(train_set)[:, 1]
x_test = np.asarray(test_set)[:, 2:]
y_test = np.asarray(test_set)[:, 1]

# Keep track of the number of samples and dimension of each sample
n_train = len(y_train)
n_test = len(y_test)
d = np.size(x_train, 1)

def computegrad(beta, lambduh, x=x_train, y=y_train, h=0.5):
    """Computes the gradient.

        Parameters
        ----------
        X: np.ndarray
            Feature/predictor matrix of shape (n x d)

        y: np.array | np.ndarray
            Outcome/response array of shape (n,) or (n, 1)

        beta: np.array | np.ndarray
            Coefficient array of shape (d,) or (d, 1)

        lambduh: float
            Regularization coefficient

        h: float, default=0.5
            Smoothing factor

        Returns
        -------
        The gradient w.r.t. beta
    """
    
    yt = y*x.dot(beta)
    ell_prime = -(1+h-yt)/(2*h)*y*(np.abs(1-yt) <= h) - y*(yt < (1-h))
    return np.mean(ell_prime[:, np.newaxis]*x, axis=0) + 2*lambduh*beta


def objective(beta, lambduh, x=x_train, y=y_train, h=0.5):
    """Computes the objective function. Must be overridden.

        Parameters
        ----------
        X: np.ndarray
            Feature/predictor matrix of shape (n x d)

        y: np.array | np.ndarray
            Outcome/response array of shape (n,) or (n, 1)

        beta: np.array | np.ndarray
            Coefficient array of shape (d,) or (d, 1)

        lambduh: float
            Regularization coefficient

        Returns
        -------
        The objective function w.r.t. beta
    """
    
    yt = y*x.dot(beta)
    ell = (1+h-yt)**2/(4*h)*(np.abs(1-yt) <= h) + (1-yt)*(yt < (1-h))
    return np.mean(ell) + lambduh*np.dot(beta, beta)


def backtracking(beta, lambduh, eta=1, alpha=0.5, betaparam=0.8, maxiter=1000, x=x_train, y=y_train):
    """Perform backtracking line search

        Parameters
        ----------
        X: np.ndarray
            Feature/predictor matrix of shape (n x d)

        y: np.array | np.ndarray
            Outcome/response array of shape (n,) or (n, 1)

        beta: np.array | np.ndarray
            Coefficient array of shape (d,) or (d, 1)

        lam: float
            Regularization coefficient lambda

        eta: float
            Starting (maximum) step size

        alpha: float, alpha=0.5
            Constant used to define sufficient decrease condition

        betaparam: float, default=0.8
            Fraction by which we decrease t if the previous t doesn't work

        max_iter: int, default=100
            Maximum number of iterations to run the algorithm

        Returns
        -------
        eta: Step size to use

        Raises
        ------
        ValueError:
            if lmbda is negative
            if eta is non-positive
    """
    
    if lambduh < 0:
        raise ValueError("lambduh (regularization coefficient) must be strictly non-negative")

    if eta <= 0:
        raise ValueError("eta (initial learning rate) must be strictly positive")

    grad_beta = computegrad(beta, lambduh, x=x, y=y)
    norm_grad_beta = np.linalg.norm(grad_beta)
    found_eta = 0
    iter = 0

    while found_eta == 0 and iter < maxiter:
        if objective(beta - eta * grad_beta, lambduh, x=x, y=y) < \
        objective(beta, lambduh, x=x, y=y) \
        - alpha * eta * norm_grad_beta ** 2:
            found_eta = 1
        elif iter == maxiter - 1:
            print('Warning: Max number of iterations of' \
            + 'backtracking line search reached')
            break
        else:
            eta *= betaparam
            iter += 1
    return eta


def mylinearsvm(beta_init, theta_init, lambduh, eta_init, maxiter, x=x_train, y=y_train, eps=1e-6):
    """Run fast gradient descent and trains linear SVM

        Parameters
        ----------

        X: ndarray
            ndarray with shape (n, d)

        Y: array | ndarray
            array with shape (n,) or (n, 1)

        beta_init: array | ndarray
            Initialization parameters for regression coefficients,
            with shape (d, ) or (d, 1)

        eps: float
            Target accuracy to be achieved, ignored if early_stopping is False

        eta_init: float
            Initial learning rate for the algorithm

        max_iter: int
            The maximum number of iterations to run gradient descent

        lambduh: float
            Regularization parameter

        use_backtracking: boolean, default=True
            If True, then backtracking line search is used at each iteration to find the
            best value of eta (learning rate)

        Returns
        -------
        betas
            A list of (at maximum) `max_iter` betas, each of which is a np.array of size (d,)

        thetas
            A list of (at maximum) `max_iter` thetas, each of which is a np.array of size (d,)
    """
    beta = beta_init
    theta = theta_init
    grad_theta = computegrad(theta, lambduh, x=x, y=y)
    beta_vals = beta
    theta_vals = theta
    iter = 0

    while iter < maxiter and np.linalg.norm(grad_theta) > eps:
        eta = backtracking(theta, lambduh, eta=eta_init, x=x, y=y)
        beta_new = theta - eta*grad_theta
        theta = beta_new + iter/(iter+3)*(beta_new-beta)
        # Store all of the places we step to
        beta_vals = np.vstack((beta_vals, beta_new))
        theta_vals = np.vstack((theta_vals, theta))
        grad_theta = computegrad(theta, lambduh, x=x, y=y)
        beta = beta_new
        iter += 1
    return beta_vals, theta_vals


def misclassification_error(beta_dict, scaler_dict, nclasses, x=x_test, y=y_test):
    """
        Calculates misclassification error of linear SVM
    """

    predictions = np.zeros((len(x), int(nclasses * (nclasses - 1) / 2)))
    k = 0
    for i in range(1, nclasses + 1):
        for j in range(i + 1, nclasses + 1):
            # Standardize
            xnow = scaler_dict[(i,j)].transform(x)
            # Predict
            betas = beta_dict[(i, j)]
            xbeta = xnow.dot(betas)
            predictions[:, k] = (xbeta > 0) * i + (xbeta < 0) * j
            k += 1

    y_pred = np.ravel(scipy.stats.mode(predictions, axis=1)[0])
    assert len(y_pred) == len(y)
    return np.mean(y_pred != y), predictions


def create_train_set(x_train, y_train, i, j):
    i_idxs_train = np.where(y_train == i)[0]
    xi_train = x_train[i_idxs_train, :]
    yi_train = np.ones_like(i_idxs_train)
    j_idxs_train = np.where(y_train == j)[0]
    xj_train = x_train[j_idxs_train, :]
    yj_train = np.ones_like(j_idxs_train) * -1
    x_train_c = np.vstack((xi_train, xj_train))
    y_train_c = np.concatenate((yi_train, yj_train))
    return x_train_c, y_train_c


def train(lambduh, x_train, y_train, x_test, y_test):
    betas_dict = {}
    scaler_dict = {}
    nclasses = int(np.max(y_train) - np.min(y_train) + 1)

    for i in range(1, nclasses + 1):
        for j in range(i + 1, nclasses + 1):
            x_train_c, y_train_c = create_train_set(x_train, y_train, i, j)
            # Standardize the data.
            scaler = sklearn.preprocessing.StandardScaler()
            x_train_c = scaler.fit_transform(x_train_c)
            scaler_dict[(i,j)] = scaler
            # Initialize
            beta_init = np.zeros(d)
            theta_init = np.zeros(d)
            eta_init = 1 / (
            scipy.linalg.eigh(1 / len(y_train_c) * \
            x_train_c.T.dot(x_train_c), eigvals=(d - 1, d - 1),
            eigvals_only=True)[0] + lambduh)
            maxiter = 100
            # Fit the model
            betas_fastgrad, _ = mylinearsvm(beta_init, theta_init,
            lambduh, eta_init, maxiter, x=x_train_c, y=y_train_c)
            betas_dict[(i, j)] = betas_fastgrad[-1, :]

    test_error = misclassification_error(betas_dict, scaler_dict, nclasses, x_test, y_test)
    return test_error


lambduh = 1
test_error = train(lambduh, x_train, y_train, x_test, y_test)
print('Test misclassification error when lambda=', lambduh, ':', test_error)


def cross_validate(X, y, lambda_max=2 ** 3, max_iter=1000, num_lambda=10, num_folds=5, seed=0):
    """
        Run cross-validation to find the optimal lambda
    """

    # Randomly divide the data into num_folds parts
    np.random.seed(seed)
    n = len(y)
    order = list(range(n))
    np.random.shuffle(order)
    idxs = [order[int(i * n / num_folds):int((i + 1) * n / num_folds)]
    for i in range(num_folds)]
    lambduh = lambda_max
    beta = np.zeros(np.size(X, 1))
    best_error = np.inf
    print('lambda \t Misclassification error')
    # Try many possible lambdas and see which gives the lowest average mse on the test set
    for j in range(num_lambda):
        avg_error = 0
        for part_num in range(num_folds):
            # Create the training and test sets
            x_test = X[idxs[part_num], :]
            y_test = y[idxs[part_num]]
            train_idxs = idxs[0:part_num] + idxs[part_num + 1:]
            train_idxs = [item for sublist in train_idxs for item in sublist]
            x_train = X[train_idxs, :]
            y_train = y[train_idxs]
            # Compute the optimal betas. Use warm-start.
            error = train(lambduh, x_train, y_train, x_test, y_test)
            
            avg_error = avg_error + error[0]
        avg_error /= num_folds
        print('%0.2e' % lambduh, '\t', np.round(avg_error, 5))
        # Update the best error
        if avg_error < best_error:
            best_error = avg_error
            best_lambda = lambduh
        lambduh /= 2
    return best_lambda


# best_lambda = cross_validate(x_train, y_train, num_lambda=25)
# print('The best lambda found was lambda = %0.2e' % best_lambda)
# test_error = train(best_lambda, x_train, y_train, x_test, y_test)
# print('Test misclassification error when lambda=', best_lambda, ':', test_error)


"""
    Sklearn's SVC with linear kernel
"""


data_dir = os.path.abspath(os.path.dirname(dirname))

x_train = np.load(os.path.join(data_dir, 'train_features.npy'))
y_train = np.load(os.path.join(data_dir, 'train_labels.npy'))
x_test = np.load(os.path.join(data_dir, 'val_features.npy'))
y_test = np.load(os.path.join(data_dir, 'val_labels.npy'))

print(len(x_train))
print(len(y_train))

n_clf = SVC(kernel='linear', degree=3) # since we need probabilities
n_clf.fit(x_train, y_train)

predicted = n_clf.predict(x_test)
print('predictions', len(predicted), predicted)
print('ytest', len(y_test))
print('Accuracy: ',metrics.accuracy_score(y_test, predicted))




