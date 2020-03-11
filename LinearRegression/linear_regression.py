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
        
    def l2_regularization(self, lambda2):
        """
        Applies L2 Regularization to the model, modifiyng the matrix, used in
        predictions calculations.
        
        Parameters:
            lambda2(float): The lamda constant value, used to calculate the L2 regularization matrix.
        
        Returns:
            Nothing.
        """
        w_reg = np.linalg.solve(lambda2*np.eye(self.X.shape[0]) + self.X.T.dot(self.X), self.X.T.dot(self.Y))
        self.w = w_reg

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