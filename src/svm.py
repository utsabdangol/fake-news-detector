class LinearSVM:
    def __init__(self, lam=1e-4, epochs=10):
        self.lam, self.epochs = lam, epochs

    def fit(self, X, y):                 # y must be in {-1, +1}
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        t = 0
        for _ in range(self.epochs):
            for i in np.random.permutation(n):
                t += 1
                eta = 1 / (self.lam * t)
                margin = y[i] * (X[i] @ self.w + self.b)
                self.w *= (1 - eta * self.lam)
                if margin < 1:
                    self.w += eta * y[i] * X[i]
                    self.b += eta * y[i]
        return self

    def decision(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.where(self.decision(X) >= 0, 1, -1)