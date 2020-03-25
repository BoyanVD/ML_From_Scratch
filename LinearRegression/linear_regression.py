import numpy as np 
import matplotlib.pyplot as plt 

class LinearRegression:
    """ 
    This is a class implementation of multidimensional linear regression.
      
    Attributes: 
        X (list): The independant variables list.
        Y (list): The dependant variables list(training data). 
        w (matrix): The matrix, used for the linear regression equation.
        r_squared(float): The R-squared value.
    """

    def __init__(self, filename = "", X = [], Y = []):
        """ 
        The constructor for LinearRegression class. 
  
        Parameters: 
           filename (string): The data file name to read from (optional).
           X (list): The independant variables list (optional).
           Y (list): The dependant variables list (optional).
        """
        self.X = X
        self.Y = Y
        
        if filename == "" :
            self.read(filename)
        elif X and Y :
            self.fit(X, Y)

    def read(self, file_name):
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

        self.fit(X, Y)

    def fit(self, X, Y):
        """ 
        The function to fit the model to the data. 
  
        Parameters: 
            X (list): The independant variables list.
            Y (list): The dependant variables list.
          
        Returns: 
            Nothing. 
        """
        X = np.array(X)
        Y = np.array(Y)

        w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))

        self.w = w
        self.X = X
        self.Y = Y
        self.r_squared = self.r_squared()


    def predict(self, X):
        """ 
        The prediction function. 
  
        Parameters: 
            X (list): Independant variables list.
          
        Returns: 
            list: The prediction list. 
        """
        y_pred = np.dot(X, self.w)
        return y_pred

    def r_squared(self):
        """ 
        R-squared calculation function. 
          
        Returns: 
            float: R-squared value.
        """
        y_pred = self.predict(self.X)
        d1 = self.Y - y_pred
        d2 = self.Y - self.Y.mean()
        r2 = 1 - d1.dot(d1) / d2.dot(d2)

        return r2

    def make_poly(self, deg):
        """ 
        Makes the linear model, polynomial.
  
        Parameters: 
            deg (int): The degree of the polynomial regression.
          
        Returns: 
            Nothing.
        """
        n = len(self.X)
        data = [np.ones(n)]
        for d in range(deg):
            data.append(self.X**(d + 1))
        self.X = np.vstack(data).T

    def l1_regularization(self, lambda1, learning_rate = 0.001, steps = 500):
        """
        Applies L1 Regularization to the model, modifiyng the matrix, used in
        predictions calculations. For calculating the L1 Regularization matrix 
        the method uses gradient descent technique.
        
        Parameters:
            lambda1(float): The lamda constant value, used to calculate the L2 regularization matrix.
            learning_rate(float): The parameter used for gradient descent learning rate.
            steps(int): The number of steps for the gradient descent alghorithm.
        
        Returns:
            Nothing.
        """
        Dimensionality = self.X.shape[1]
        w_reg = np.random.randn(Dimensionality) / np.sqrt(Dimensionality) # randomly initialize w

        for step in range(steps):
            y_pred = X.dot(w_reg)
            delta = y_pred - self.Y
            w_reg = w_reg - learning_rate*(X.T.dot(delta) + lambda1*np.sign(w_reg))

        self.w = w_reg
        
    def l2_regularization(self, lambda2):
        """
        Applies L2 Regularization to the model, modifiyng the matrix, used in
        predictions calculations.
        
        Parameters:
            lambda2(float): The lamda constant value, used to calculate the L2 regularization matrix.
        
        Returns:
            Nothing.
        """
        w_reg = np.linalg.solve(lambda2*np.eye(self.X.shape[1]) + self.X.T.dot(self.X), self.X.T.dot(self.Y))
        self.w = w_reg
        
    def gradient_descent(self, steps, learning_rate = 0.001):
        """
        Applies alternative to the closed calculation of the "w" matrix, used by the model to make
        predictions. This method is really useful for bypassing the dummy varieble trap or any other
        problem, that may occur when calculating the matrix using the traditional way (for example getting not invertable matrix).
        In result, this method updates the model's prediction matrix.
        
        Parameters:
            steps(int) : the number of steps that you want the gradient descent to do in its way to the goal.
            learning_rate : It's optional, the default value is 0.001 . This value is used for coefficient, that represents
            the 'size' of every step, that the algorithm makes.
            
        Returns:
            Nothing.
        """
        Dim = self.X.shape[1]
        N = self.X.shape[0]
        w_grad = np.random.rand(Dim) / np.sqrt(Dim)
        for t in range(steps):
            y_pred = self.X.dot(w_grad)
            delta = y_pred - self.Y
            w_grad = w_grad - learning_rate * self.X.T.dot(delta)
        self.w = w_grad

    def gradient_descent_with_mean_squared_calculation(self, steps, learning_rate = 0.001):
        """
        Performs gradient descent, but apart from updating the matrix used for calculating predictions, it returns
        the mean squared errors matrix. The idea of this method is to provide functionality for testing different
        learning rate and number of gradient descent steps values.
        
        Parameters:
            steps(int) : the number of steps that you want the gradient descent to do in its way to the goal.
            learning_rate : It's optional, the default value is 0.001 . This value is used for coefficient, that represents
            the 'size' of every step, that the algorithm makes.
            
        Returns:
            costs(matrix) : the matrix containing the mean squared errors.
        """
        costs = []
        Dim = self.X.shape[1]
        N = self.X.shape[0]
        w_grad = np.random.rand(Dim) / np.sqrt(Dim)
        for t in range(steps):
            y_pred = self.X.dot(w_grad)
            delta = y_pred - self.Y
            w_grad = w_grad - learning_rate * self.X.T.dot(delta)
            mean_squared_error = delta.dot(delta) / N
            costs.append(mean_squared_error)
        return costs

    def show_model_info(self):
        print("The matrix, used for the predictions calculations is : w = ", self.w)
        print("The R-squared value is : ", self.r_squared)
        
    def show_input_data(self):
        plt.scatter(self.X, self.Y)
        plt.show()
        
    def show_prediction(self, y_pred):
        plt.scatter(self.X, self.Y)
        plt.plot(self.X, y_pred)
        plt.show()