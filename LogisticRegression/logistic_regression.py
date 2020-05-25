import numpy as np 
import matplotlib.pyplot as plt 

class LogisticRegression:
    """ 
    This is a class implementation of multidimensional logistic regression.
      
    Attributes: 
        X (list): The independant variables list.
        Y (list): The dependant variables list(training data). 
        w (matrix): The matrix, used for the logistic regression model.
    """
    def __init__(self, filename = "", learning_rate = 0.1, steps = 1000, X = [], Y = []):
        """ 
        The constructor for LogisticRegression class. 
  
        Parameters: 
           filename (string): The data file name to read from (optional).
           X (list): The independant variables list (optional).
           Y (list): The dependant variables list (optional).
        """
        self.X = X
        self.Y = Y
        
        if filename != "" :
            self.read(filename, learning_rate, steps)
        elif X and Y :
            self.fit(X, Y, learning_rate, steps)

    def read(self, file_name, learning_rate = 0.1, steps = 1000):
        """ 
        The function to read data from a file given. 
  
        Parameters: 
            file_name (string): The data file name. 
          
        Returns: 
            Nothing. 
        """
        X = []
        Y = []

        for line in open(file_name):
            values = line.split(',')
            X.append(1, values[:-1])
            Y.append(float(values[-1]))

        self.fit(X, Y, learning_rate, steps)

    def fit(self, X, Y, learning_rate = 0.1, steps = 1000):
        """ 
        The function to fit the logistic regression model to the data. 
  
        Parameters: 
            X (list): The independant variables list.
            Y (list): The dependant variables list.
          
        Returns: 
            Nothing. 
        """
        X = np.array(X)
        Y = np.array(Y)

        w = np.random.randn(X.shape[1])
        w = gradient_descent(weights, Y, X, learning_rate, steps)

        self.w = w
        self.X = X
        self.Y = Y

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def cross_entropy(Targets, Y):
        E = 0
        for i in range(N):
            if T[i] == 1:
                E -= np.log(Y[i])
            else:
                E -= np.log(1 - Y[i])
        return E

    def get_errors_after_gradient_descent(learning_rate = 0.1, steps = 1000):
        weights = self.w
        Y_pred = sigmoid(self.X.dot(weights))
        Errors = []

        for i in range(steps):
            weights += learning_rate * np.dot((self.Y - Y_pred).T, self.X)
            Y_pred = sigmoid(self.X.dot(weights))
            Errors.append(cross_entropy(self.Y, Y_pred))

        return Errors

    def gradient_descent(weights, Targets, Xbias, learning_rate = 0.1, steps = 1000):
        Y = sigmoid(Xbias.dot(weights))

        for i in range(steps):
            weights += learning_rate * np.dot((Targets - Y).T, Xbias)
            Y = sigmoid(Xbias.dot(weights))

        return weights

    def solve_donut_problem():
        ones = np.ones((N, 1))

        # add a column of r = sqrt(x^2 + y^2)
        # r = np.sqrt( (X * X).sum(axis=1) ).reshape(-1, 1)
        r = np.zeros((N, 1))
        for i in range(N):
            r[i] = np.sqrt(X[i,:].dot(X[i,:]))
    
        Xb = np.concatenate((ones, r, X[:,1:]), axis=1)
        self.X = Xb

    def solve_xor_problem(first_column, second_column):
        N = self.X.shape[0]
        ones = np.ones((N, 1))

        xy = (X[:,first_column] * X[:,second_column]).reshape(N, 1)
        Xb = np.concatenate((ones, xy, self.X[:,1:]), axis=1)

        self.X = Xb