# used for manipulating directory paths
import os
import scipy.linalg as linalg
from scipy.optimize import minimize
# Scientific and vector computation for python
import grader as grader
import numpy as np
from matplotlib import pyplot
import matplotlib as mpl

# Optimization module in scipy
from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io \
    import loadmat

# library written for this exercise providing additional functions for assignment submission, and others
import utils

#  The following command loads the dataset.
data = loadmat("C:\\Users\\lenovo\\Desktop\\ML assignments\\Anomaly Detection\\Data\\ex8data1")
X, Xval, yval = data['X'], data['Xval'], data['yval'][:, 0]


#  Visualize the example dataset
#  Plotting X, 2 columns VS each other
#  1st column of the data latency/ 2nd column throughput
# pyplot.plot(X[:, 0], X[:, 1], 'bx', mew=2, mec='k', ms=6)
# pyplot.axis([0, 30, 0, 30])
# pyplot.xlabel('Latency (ms)')
# pyplot.ylabel('Throughput (mb/s)')


def estimateGaussian(X):
    """
    Estimates the parameters (mean, variance) of a
    Gaussian distribution using the data in X.
    Args:
        X     : array(# of training examples m, # of features n)
    Returns:
        mu    : array(# of features n, 1)
        sigma2: array(# of features n, 1)
    """
    # Get useful variables.
    m, n = X.shape

    # Init mu and sigma2.
    mu = np.zeros((n, 1))
    sigma2 = np.zeros((n, 1))

    mu = np.mean(X.T, axis=1)
    mu = mu.reshape(mu.shape[0], -1)
    sigma2 = np.var(X.T, axis=1)
    sigma2 = sigma2.reshape(sigma2.shape[0], -1)

    return mu, sigma2


# Create a function to compute the probability.
def multivariateGaussian(X, mu, Sigma2):
    """
    Computes the probability density function of the examples X
    under the multivariate gaussian distribution with parameters
    mu and sigma2. If Sigma2 is a matrix, it is treated as the
    covariance matrix. If Sigma2 is a vector, it is treated as the
    sigma^2 values of the variances in each dimension (a diagonal
    covariance matrix).
    Args:
        X     : array(# of training examples m, # of features n)
        mu    : array(# of features n, 1)
        Sigma2: array(# of features n, # of features n)
    Returns:
        p     : array(# of training examples m,)
    """
    k = len(mu)

    if (Sigma2.shape[0] == 1) or (sigma2.shape[1] == 1):
        Sigma2 = linalg.diagsvd(Sigma2.flatten(),
                                len(Sigma2.flatten()),
                                len(Sigma2.flatten()))
        X = X - mu.T
        p = np.dot(np.power(2 * np.pi, - k / 2.0),
                   np.power(np.linalg.det(Sigma2), -0.5)) * \
            np.exp(-0.5 * np.sum(np.dot(X, np.linalg.pinv(Sigma2)) * X, axis=1))

    return p


# Create a function to visualize the dataset and its estimated distribution.
def visualizeFit(X, mu, sigma2):
    """
    Visualizes the dataset and its estimated distribution.
    This visualization shows the probability density function
    of the Gaussian distribution. Each example has a location
    (x1, x2) that depends on its feature values.
    Args:
        X     : array(# of training examples m, # of features n)
        mu    : array(# of features n, 1)
        sigma2: array(# of features n, 1)
    """
    X1, X2 = np.meshgrid(np.arange(0, 30, 0.5), np.arange(0, 30, 0.5))
    Z = multivariateGaussian(np.column_stack((X1.reshape(X1.size),
                                              X2.reshape(X2.size))),
                             mu, sigma2)
    Z = Z.reshape(X1.shape)

    pyplot.plot(X[:, 0], X[:, 1], 'bx', markersize=3)

    # Do not plot if there are infinities.
    if np.sum(np.isinf(Z)) == 0:
        pyplot.contour(X1, X2, Z, np.power(10, (np.arange(-20, 0.1, 3)).T))


print('Visualizing Gaussian fit.')

# Estimate mu and sigma2.
mu, sigma2 = estimateGaussian(X)

# Return the density of the multivariate normal at each data point (row) of X.
p = multivariateGaussian(X, mu, sigma2)


# Visualize the fit.
# visualizeFit(X, mu, sigma2)
# pyplot.xlabel('Latency (ms)')
# pyplot.ylabel('Throughput (mb/s)')
# pyplot.title('Figure 2: The Gaussian distribution contours \
# of the distribution fit to the dataset.')
# pyplot.show()


def selectThreshold(yval, pval):
    """
    Finds the best threshold to use for selecting outliers
    based on the results from a validation set (pval) and
    the ground truth (yval).
    Args:
        yval       : array(# of cv examples,)
        pval       : array(# of cv examples,)
    Returns:
        bestEpsilon: float
        bestF1     : float
    """
    # Init values.
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0

    stepsize = (max(pval) - min(pval)) / 1000
    for epsilon in np.arange(min(pval), max(pval), stepsize):
        # Use predictions to get a binary vector of
        # 0's and 1's of the outlier predictions.
        predictions = pval < epsilon
        tp = sum(((yval == 1) & (predictions == 1)))
        fp = sum((yval == 0) & (predictions == 1))
        fn = sum((yval == 1) & (predictions == 0))
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        F1 = 2 * prec * rec / (prec + rec)
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

    return bestEpsilon, bestF1


pval = multivariateGaussian(Xval, mu, sigma2)
epsilon, F1 = selectThreshold(yval, pval)

print('Best epsilon found using cross-validation: {}'.format(epsilon))
print('Best F1 on Cross Validation Set:  {}'.format(F1))
print('(A value epsilon of about 8.99e-05 is expected.)')

# Find the outliers in the training set and plot them.
outliers = p < epsilon

# Draw a red circle around those outliers.
# pyplot.plot(X[outliers, 0], X[outliers, 1], 'ro', markersize=10, fillstyle='none')
# visualizeFit(X, mu, sigma2)
# pyplot.xlabel('Latency (ms)')
# pyplot.ylabel('Throughput (mb/s)')
# pyplot.title('Figure 3: The classified anomalies.')
# pyplot.show()
#
data2 = loadmat('C:\\Users\\lenovo\\Desktop\\ML assignments\\Anomaly Detection\\Data\\ex8data2.mat')
X = data2["X"]
Xval = data2["Xval"]
yval = data2["yval"].flatten()

# Apply the same steps to the larger dataset.
mu, sigma2 = estimateGaussian(X)

# Training set.
p = multivariateGaussian(X, mu, sigma2)

# Cross-validation set.
pval = multivariateGaussian(Xval, mu, sigma2)

# Find the best threshold.
epsilon, F1 = selectThreshold(yval, pval)

print('Best epsilon found using cross-validation: {}'.format(epsilon))
print('Best F1 on Cross Validation Set: {}'.format(F1))
print('# Outliers found: {}'.format(sum(p < epsilon)))

print('Loading movie ratings dataset.')

# Load data.
data3 = loadmat('C:\\Users\\lenovo\\Desktop\\ML assignments\\Anomaly Detection\\Data\\ex8_movies.mat')

# Y is a 1682x943 matrix, containing ratings (1-5)
# of 1682 movies on 943 users.
Y = data3["Y"]
# R is a 1682x943 matrix, where R(i,j) = 1 if and only if
# user j gave a rating to movie i.
R = data3["R"]

# From the matrix, statistics like average rating can be computed.
print('Average rating for movie 1 (Toy Story): {0:.2f}/5'. \
      format(np.mean(Y[0, R[0, :] == 1])))

# Visualize the ratings matrix by plotting it with imshow.
pyplot.imshow(Y, aspect='auto')
pyplot.ylabel('Movies')
pyplot.xlabel('Users')
pyplot.show()
#
# Load pre-trained weights (X, Theta, num_users, num_movies, num_features).
data4 = loadmat('C:\\Users\\lenovo\\Desktop\\ML assignments\\Anomaly Detection\\Data\\ex8_movieParams.mat')
X = data4["X"]
Theta = data4["Theta"]

# Reduce the data set size so that this runs faster.
num_users = 4
num_movies = 5
num_features = 3
X = X[:num_movies, :num_features]
Theta = Theta[:num_users, :num_features]
Y = Y[:num_movies, :num_users]
R = R[:num_movies, :num_users]


# Create a function to compute the cost J and grad.
def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda_coef):
    """
    Returns the cost and gradient for
    the collaborative filtering problem.
    Args:
        params      : array(num_movies x num_features + num_users x num_features,)
        Y           : array(num_movies, num_users)
        R           : array(num_movies, num_users)
        num_users   : int
        num_movies  : int
        num_features: int
        lambda_coef : float
    Returns:
        J           : float
        grad        : array(num_movies x num_features + num_users x num_features,)
    """
    # Unfold params back into the parameters X and Theta.
    X = np.reshape(params[:num_movies * num_features], (num_movies, num_features))
    Theta = np.reshape(params[num_movies * num_features:], (num_users, num_features))

    # Init values.
    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    # Compute squared error.
    error = np.square(np.dot(X, Theta.T) - Y)

    # Compute regularization term.
    reg_term = (lambda_coef / 2) * (np.sum(np.square(Theta)) + np.sum(np.square(X)))

    # Compute cost function but sum only if R(i,j)=1; vectorized solution.
    J = (1 / 2) * np.sum(error * R) + reg_term

    # Compute the gradients.
    X_grad = np.dot((np.dot(X, Theta.T) - Y) * R, Theta) + lambda_coef * X
    Theta_grad = np.dot(((np.dot(X, Theta.T) - Y) * R).T, X) + lambda_coef * Theta

    grad = np.concatenate((X_grad.reshape(X_grad.size),
                           Theta_grad.reshape(Theta_grad.size)))

    return J, grad


# Create a list of my X and Theta.
lst_params = [X, Theta]

# Unroll parameters and then merge/concatenate.
unrolled_params = [lst_params[i].ravel() for i, _ in enumerate(lst_params)]
params = np.concatenate(unrolled_params)

# Evaluate cost function.
J, _ = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 0)

print('Cost at loaded parameters: {:0.2f}'.format(J))
print('(this value should be about 22.22)')


def computeNumericalGradient(J, theta):
    """
    Computes the numerical gradient of the function J
    around theta using "finite differences" and gives
    a numerical estimate of the gradient.
    Notes: The following code implements numerical
           gradient checking, and returns the numerical
           gradient. It sets numgrad(i) to (a numerical
           approximation of) the partial derivative of J
           with respect to the i-th input argument,
           evaluated at theta. (i.e., numgrad(i) should
           be the (approximately) the partial derivative
           of J with respect to theta(i).)
    Args:
        J      : function
        theta  : array(num_movies x num_features + num_users x num_features,)
    Returns:
        numgrad: array(num_movies x num_features + num_users x num_features,)
    """
    # Initialize parameters.
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4

    for p in range(theta.size):
        # Set the perturbation vector.
        perturb.reshape(perturb.size)[p] = e
        loss1, _ = J(theta - perturb)
        loss2, _ = J(theta + perturb)
        # Compute the Numerical Gradient.
        numgrad.reshape(numgrad.size)[p] = (loss2 - loss1) / (2 * e)
        perturb.reshape(perturb.size)[p] = 0

    return numgrad


# Create a function to check the cost function and gradients.
def checkCostFunction(lambda_coef):
    """
    Creates a collaborative filering problem
    to check the cost function and gradients.
    It will output the analytical gradients
    and the numerical gradients computed using
    computeNumericalGradient. These two gradient
    computations should result in very similar values.
    Args:
        lambda_coef : float
    """
    # Create small problem.
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

    # Zap out most entries.
    Y = np.dot(X_t, Theta_t.T)
    Y[np.random.rand(Y.shape[0], Y.shape[1]) > 0.5] = 0
    R = np.zeros(Y.shape)
    R[Y != 0] = 1

    # Run Gradient Checking.
    X = np.random.randn(X_t.shape[0], X_t.shape[1])
    Theta = np.random.randn(Theta_t.shape[0], Theta_t.shape[1])
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]

    # Create short hand for cost function.
    def costFunc(p):
        return cofiCostFunc(p, Y, R, num_users, num_movies,
                            num_features, lambda_coef)

    params = np.concatenate((X.reshape(X.size), Theta.reshape(Theta.size)))
    numgrad = computeNumericalGradient(costFunc, params)
    J, grad = cofiCostFunc(params, Y, R, num_users, num_movies,
                           num_features, lambda_coef)

    # Visually examine the two gradient computations.
    for numerical, analytical in zip(numgrad, grad):
        print('Numerical Gradient: {0:10f}, Analytical Gradient {1:10f}'. \
              format(numerical, analytical))
    print('\nThe above two columns should be very similar.\n')

    # Evaluate the norm of the difference between two solutions.
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)

    print('If the backpropagation implementation is correct, then \n' \
          'the relative difference will be small (less than 1e-9). \n' \
          '\nRelative Difference: {:.10E}'.format(diff))


print('Checking Gradients (without regularization)...\n')
# Check gradients by running checkCostFunction.
checkCostFunction(0)

J, _ = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 1.5)

print('Cost at loaded parameters (lambda_coef = 1.5): {:0.2f}' \
      '\n(this value should be 31.34)\n'.format(J))

print('Checking Gradients (with regularization)... \n')

# Check gradients by running checkCostFunction.
checkCostFunction(1.5)


# Create a function to load movies.
def loadMovieList():
    """
    Reads the fixed movie list in movie_idx.txt
    and returns a cell array of the words in movieList.
    Returns:
        movieList: list
    """
    # Read the fixed movieulary list.
    with open('C:\\Users\\lenovo\\Desktop\\ML assignments\\Anomaly Detection\\Data\\movie_ids.txt',
              encoding="ISO-8859-1") as f:
        movieList = []
        for line in f:
            movieName = line.split()[1:]
            movieList.append(" ".join(movieName))

    return movieList


movieList = loadMovieList()

# Initialize ratings.
my_ratings = np.zeros((1682, 1))

# Check the file movie_idx.txt for id of each movie in the dataset.
# For example, Toy Story (1995) has ID 0, so to rate it "4", set:
my_ratings[0] = 4

# Or suppose did not enjoy The Mask (1994), so set:
my_ratings[71] = 1

# Select a few movies and rate them:
my_ratings[8] = 3
my_ratings[12] = 3
my_ratings[32] = 2
my_ratings[44] = 5
my_ratings[60] = 5
my_ratings[63] = 4
my_ratings[67] = 3
my_ratings[85] = 5
my_ratings[117] = 1
my_ratings[153] = 4
my_ratings[155] = 5
my_ratings[164] = 5
my_ratings[174] = 4
my_ratings[178] = 5
my_ratings[193] = 4
my_ratings[354] = 2
my_ratings[442] = 4
my_ratings[478] = 5
my_ratings[514] = 5
my_ratings[606] = 5
my_ratings[633] = 5
my_ratings[639] = 5
my_ratings[649] = 5
my_ratings[954] = 5
my_ratings[1422] = 3

print('User ratings:\n')
for i, rating in enumerate(my_ratings):
    if rating > 0:
        print('Rated {} for {}'.format(rating[0], movieList[i]))

print('Training collaborative filtering...')

# Load data.
Y = data3["Y"]  # array(1682, 943)
R = data3["R"]  # array(1682, 943)

# Add my ratings to the data matrix.
Y = np.column_stack((my_ratings, Y))  # array(1682, 944)
R = np.column_stack(((my_ratings != 0), R))  # array(1682, 944)


# Create a function to normalize ratings.
def normalizeRatings(Y, R):
    """
    Preprocesses data by subtracting mean rating for every
    movie (every row). Normalizes Y so that each movie has
    a rating of 0 on average, and returns the mean rating in Ymean.
    Args:
        Y    : array(num_movies, num_users)
        R    : array(num_movies, num_users)
    Returns:
        Ynorm: array(num_movies, num_users)
        Ymean: array(num_movies, 1)
    """
    m, n = Y.shape
    Ymean = np.zeros((m, 1))
    Ynorm = np.zeros(Y.shape)
    for i in range(m):
        idx = R[i, :] == 1
        # Compute the mean only of the rated movies.
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]

    return Ynorm, Ymean


# Normalize ratings.
[Ynorm, Ymean] = normalizeRatings(Y, R)

# Get useful values.
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10

# Set initial parameters (Theta, X).
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)

initial_parameters = np.concatenate((X.reshape(X.size),
                                     Theta.reshape(Theta.size)))

# Set options.
options = {'maxiter': 100, 'disp': True}

# Set regularization.
lambda_coef = 10


# Create short hand for cost function.
def costFunc(initial_parameters):
    return cofiCostFunc(initial_parameters, Y, R, num_users,
                        num_movies, num_features, lambda_coef)


# Optimize.
results = minimize(costFunc, x0=initial_parameters,
                   options=options, method='CG', jac=True)
theta = results.x

# Unfold results back into the parameters X and Theta.
X = np.reshape(theta[:num_movies * num_features], (num_movies, num_features))
Theta = np.reshape(theta[num_movies * num_features:], (num_users, num_features))

print('\nRecommender system learning completed!')
