import sys
dir_str = ".."
if (dir_str not in sys.path):
    sys.path.append(dir_str)

import numpy as np
from scipy.stats import chi2, multivariate_normal
from utils import polynomial_features

class BayesianRegression():
    """
    Baysian regression model.
    Parameters
    -------------
    mu0: ndarray of shape(n_features, )
      mean value of the prior Normal distribution of the parameters
    omega0: ndarray of shape(n_features, )
      The precision matrix of the prior Normal distribution of the parameters
    nu0: float
      The degree of freedom of the prior scaled inverse chi-squared distribution
    sigma_sq0: float
      The scale parameter of the prior scaled inverse chi squared distribution
    poly_degree: int
      The polynomial degree that the features should be transformed to. Allowed for polynomial regression
    cred_int: float
      The credible interval (ETI in this implementation). 95->95% credible interval of the posterior of the parameters

    Reference:
      https://github.com/mattiasvillani/BayesLearnCourse/raw/master/Slides/BayesLearnL5.pdf
    """

    def __init__(self, n_draws, mu0, omega0, nu0, sigma_sq0, poly_degree=0, cred_int=95):
        self.w = None
        self.n_draws = n_draws
        self.poly_degree = poly_degree
        self.cred_int = cred_int

        self.mu0 = mu0
        self.omega0 = omega0
        self.nu0 = nu0
        self.sigma_sq0 = sigma_sq0

    def _draw_scaled_inv_chi_sq(self, n, df, scale):
        X = chi2.rvs(size=n, df=df)
        sigma_sq = df * scale / X
        return sigma_sq

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        #if polynominal transformation
        if self.poly_degree:
            X = polynomial_features(X, degree=self.poly_degree)
        n_samples, n_features = X.shape
        X_X = np.dot(X.T, X)

        #Least square approximate of beta
        beta_hat = np.dot(np.dot(np.linalg.pinv(X_X), X.T), y)

        #The posterior parameters can be determined analytically since assuming conjugate priors for the likelyhoods.

        #Normal posterior if Normal prior / likelihood
        mu_n = np.dot(np.linalg.pinv(X_X + self.omega0), np.dot(X_X, beta_hat) + np.dot(self.omega0, self.mu0))
        omega_n = X_X + self.omega0

        #scaled inverse chi-squaredd posterior if scaled inverse chi-squared prior / likelihood
        nu_n = self.nu0 + n_samples
        sigma_sq_n = (1.0/nu_n) * (self.nu0 * self.sigma_sq0 + (np.dot(y.T, y) + np.dot(np.dot(self.mu0.T, self.omega0), self.mu0) - np.dot(mu_n.T, np.dot(omega_n, mu_n))))

        #Simulate parameter values for n_draws
        beta_draws = np.empty((self.n_draws, n_features))
        for i in range(self.n_draws):
            sigma_sq = self._draw_scaled_inv_chi_sq(n=1, df=nu_n, scale=sigma_sq_n)
            beta = multivariate_normal.rvs(size=1, mean=mu_n[:, 0], cov=sigma_sq*np.linalg.pinv(omega_n))
            #save parameter draws
            beta_draws[i, :] = beta

        #Select the mean of the simulated variables as the ones used to make predictions
        self.w = np.mean(beta_draws, axis=0)

        #lower and upper boundary of the credible interval
        l_eti = 50 - self.cred_int/2
        u_eti = 50 + self.cred_int/2
        self.eti = np.array([[np.percentile(beta_draws[:, i], q=l_eti), np.percentile(beta_draws[:, i], q=u_eti)] for i in range(n_features)])

    def predict(self, X_test, eti=False):
        X_test = np.array(X_test)
        #if polynominal features
        if self.poly_degree:
            X_test = polynomial_features(X_test, degree=self.poly_degree)
        y_pred = np.dot(X_test, self.w)
        #if the lower and upper boundaries for the 95% equal tail interval should be returned
        if eti:
            lower_w = self.eti[:, 0]
            upper_w = self.eti[:, 1]
            y_lower_pred = np.dot(X_test, lower_w)
            y_upper_pred = np.dot(X_test, upper_w)
            return y_pred, y_lower_pred, y_upper_pred
        return y_pred
