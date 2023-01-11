from time import time
import numpy as np
from sklearn.preprocessing._data import _handle_zeros_in_scale
from sklearn.utils._param_validation import Interval
from sklearn.gaussian_process import GaussianProcessRegressor
from numbers import Integral
import gpytorch, torch


class MyGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MyGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1]))

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class GaussianProcessRegressorWithTorch(GaussianProcessRegressor):
    """Gaussian process regression (GPR).
    The implementation is based the GPTorch ExactGP class, adapted to run under the
    Scikit-Learn regression class.
    :class:`GaussianProcessRegressorWithTorch`:
       * allows prediction without prior fitting (based on the GP prior)
       * provides an additional method `sample_y(X)`, which evaluates samples
         drawn from the GPR (prior or posterior) at given inputs

    Parameters
    ----------
    n_iter : int, default=1
        The number of iteration during the Adam optimization.
    normalize_y : bool, default=False
        Whether or not to normalize the target values `y` by removing the mean
        and scaling to unit-variance. This is recommended for cases where
        zero-mean, unit-variance priors are used. Note that, in this
        implementation, the normalisation is reversed before the GP predictions
        are reported.
    verbose : bool, default=True
        Whether or not to print optimization messages

    Attributes
    ----------
    X_train_ : array-like of shape (n_samples, n_features) or list of object
        Feature vectors or other representations of training data (also
        required for prediction).
    y_train_ : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values in training data (also required for prediction).
    kernel_ : kernel instance
        The kernel used for prediction. The structure of the kernel is the
        same as the one passed as parameter but with optimized hyperparameters.
    L_ : array-like of shape (n_samples, n_samples)
        Lower-triangular Cholesky decomposition of the kernel in ``X_train_``.
    alpha_ : array-like of shape (n_samples,)
        Dual coefficients of training data points in kernel space.
    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of ``self.kernel_.theta``.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        .. versionadded:: 1.0
    See Also
    --------
    GaussianProcessClassifier : Gaussian process classification (GPC)
        based on Laplace approximation.
    References
    ----------
    .. [RW2006] `Carl E. Rasmussen and Christopher K.I. Williams,
       "Gaussian Processes for Machine Learning",
       MIT Press 2006 <https://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_
    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> gpr = GaussianProcessRegressorWithTorch(n_iter=200).fit(X, y)
    >>> gpr.score(X, y)
    0.882...
    >>> gpr.predict(X[:2,:], return_std=True)
    (array([761.1..., 437.8...]), array([1.41..., 1.43...]))
    """

    _parameter_constraints: dict = {"n_iter": [Interval(Integral, 1, None, closed="left")],
                                    "normalize_y": ["boolean"],
    }

    def __init__(self,
                 n_iter=1,
                 normalize_y=False,
                 verbose=True,
                 ):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.n_iter = n_iter
        self.normalize_y = normalize_y
        self.verbose = verbose
        self.cached = False


    def fit(self, train_x, train_y):
        if self.normalize_y:
            self._y_train_mean = np.mean(train_y, axis=0)
            self._y_train_std = _handle_zeros_in_scale(np.std(train_y, axis=0), copy=False)
            train_y = (train_y - self._y_train_mean) / self._y_train_std
        else:
            self._y_train_mean = 0
            self._y_train_std = 1


        train_x = torch.tensor(train_x, dtype=torch.float)
        train_y = torch.tensor(train_y, dtype=torch.float)
        self.model = MyGP(train_x, train_y, self.likelihood)

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(),
                                      lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood,
                                                       self.model)

        for i in range(self.n_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            if self.verbose:
                print(f'Iter {i + 1:d}/{self.n_iter:d} - Loss: {loss.item():.3f}   '
                      f'noise: {self.model.likelihood.noise.item():.3f}')
            optimizer.step()

        return self


    def predict(self, test_x, return_std=False, return_cov=False):
        # Set into eval mode
        self.model.eval()
        self.likelihood.eval()
        if not self.cached:
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(200):
                results = self.model(torch.tensor(test_x, dtype=torch.float))
            self.cached = True
        else:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                results = self.model(torch.tensor(test_x, dtype=torch.float))

        pred = results.mean.numpy()
        pred = self._y_train_std * pred + self._y_train_mean

        if return_std:
            y_var = results.variance.numpy()
            y_var = np.outer(y_var, self._y_train_std ** 2).reshape(*y_var.shape, -1)
            return pred, np.sqrt(y_var)
        elif return_cov:
            y_cov = results.covariance_matrix.numpy()
            y_cov = np.outer(y_cov, self._y_train_std ** 2).reshape(*y_cov.shape, -1)
            return pred, y_cov
        else:
            return pred


    def log_marginal_likelihood(self,
                                theta=None,
                                eval_gradient=False,
                                clone_kernel=True
                                ):
        """Return log-marginal likelihood of theta for training data.
        Parameters
        ----------
        theta : array-like of shape (n_kernel_params,) default=None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.
        eval_gradient : bool, default=False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.
        clone_kernel : bool, default=True
            If True, the kernel attribute is copied. If False, the kernel
            attribute is modified, but may result in a performance improvement.
        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.
        log_likelihood_gradient : ndarray of shape (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
        print("Warning: Not implementet yet")
        return None
        
        
