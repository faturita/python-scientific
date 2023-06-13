'''

From Assembly.AI


'''

import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000) -> None:
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Verify that y has labels -1,1, and not 0,1
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        # Optimization.  Gradient descent minimize J
        # J is the separation between the support vectors and each class
        #
        # J = (1/n * sum(max(0,1 - y(wx-b))}] + ƛ ||w||^2))
        # Hinge loss + regularizer
        # -> Min hinge loss and at the same time maximize the margin

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1

                # dj/dw and dJ/db leads to the following update formula
                if condition:
                    self.w -= self.lr * (2* self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])

                    )
                    self.b -= self.lr * y_[idx]

    def predict(self,X):
        approx = np.dot(X, self.w) - self.b
        # The linear model is applied to the input, and it is transformed
        # to the space where the separation hyperplane lives.
        # The sign is later used to see on which side of the plane this new
        # data lives
        return np.sign(approx)