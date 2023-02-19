#!/usr/bin/env python
"""A minimalistic implemention of CMA-ES without using `numpy`.

The Covariance Matrix Adaptation Evolution Strategy, CMA-ES, serves for
numerical nonlinear function minimization.

The **main functionality** is implemented in

1. class `CMAES`, and

2. function `find_minimizing_values` which is a small single-line-usage wrapper around
   `CMAES`.

This code has two **purposes**:

1. for READING and UNDERSTANDING the basic flow and the details of the
   CMA-ES *algorithm*. The source code is meant to be read. For a quick
   glance, study the first few code lines of `fmin` and the code of
   method `CMAES.tell`, where all the real work is done in about 20 lines
   of code (search "def tell" in the source). Otherwise, reading from
   the top is a feasible option, where the codes of `fmin`,
   `CMAES.__init__`, `CMAES.ask`, `CMAES.tell` are of particular
   interest.

2. apply CMA-ES when the python module `numpy` is not available.
   When `numpy` is available, `cma.fmin` or `cma.CMAEvolutionStrategy` are
   preferred to run "serious" simulations. The latter code has many more
   lines, but usually executes faster, offers a richer user interface,
   better termination options, boundary and noise handling, injection,
   automated restarts...

Dependencies: `math.exp`, `math.log` and `random.normalvariate` (modules
`matplotlib.pylab` and `sys` are optional).

Testing: call ``python purecma.py`` at the OS shell. Tested with
Python 2.6, 2.7, 3.3, 3.5, 3.6.

URL: http://github.com/CMA-ES/pycma

Last change: September, 2017, version 3.0.0

:Author: Nikolaus Hansen, 2010-2011, 2017

This code is released into the public domain (that is, you may
use and modify it however you like).

"""

___author__ = "Nikolaus Hansen"
__license__ = "public domain"
__version__ = "3.0.0"

from math import log, exp
from random import normalvariate as random_normalvariate
import numbers

def find_minimizing_values(objective_function, inital_solution, sigma, args=(), max_function_evaluations="1e3 * N**2", fitness_target=None):
    """
        non-linear non-convex minimization procedure, a functional
        interface to CMA-ES.

        Parameters
        ==========
            `objective_function`: `callable`
                a function that takes as input a `list` of floats (like
                [3.0, 2.2, 1.1]) and returns a single `float` (a scalar).
                The objective is to find ``x`` with ``objective_function(x)``
                to be as small as possible.
            `inital_solution`: `list` or sequence
                list of numbers (like `[3.2, 2, 1]`), initial solution vector,
                its length defines the search space dimension.
            `sigma`: `float`
                initial step-size, standard deviation in any coordinate
            `args`: `tuple` or sequence
                additional (optional) arguments passed to `objective_function`
            `fitness_target`: `float`
                target function value
            `max_function_evaluations`: `int` or `str`
                maximal number of function evaluations, a string
                is evaluated with ``N`` as search space dimension

        Return
        ======
        The `tuple` (``xmin``:`list`, ``es``:`CMAES`), where ``xmin`` is the
        best seen (evaluated) solution and ``es`` is the correspoding `CMAES`
        instance. Consult ``help(es.result)`` of property `result` for further
        results.

        Example
        =======
        The following example minimizes the function `ff.elli`:

        >>> try: import cma.purecma as purecma
        ... except ImportError: import purecma
        >>> def felli(x):
        ...     return sum(10**(6 * i / (len(x)-1)) * xi**2
        ...                for i, xi in enumerate(x))
        >>> res = purecma.find_minimizing_values(felli, 3 * [0.5], 0.3)  # doctest:+SKIP
        evals: ax-ratio max(std)   f-value
            7:     1.0  3.4e-01  240.2716966
        14:     1.0  3.9e-01  2341.50170536
        700:   247.9  2.4e-01  0.629102574062
        1400:  1185.9  5.3e-07  4.83466373808e-13
        1421:  1131.2  2.9e-07  5.50167024417e-14
        termination by {'tolfun': 1e-12}
        best f-value = 2.72976881789e-14
        solution = [5.284564665206811e-08, 2.4608091035303e-09, -1.3582873173543187e-10]
        >>> print(res[0])  # doctest:+SKIP
        [5.284564665206811e-08, 2.4608091035303e-09, -1.3582873173543187e-10]
        >>> res[1].result[1])  # doctest:+SKIP
        2.72976881789e-14

        Details
        =======
        After importing `purecma`, this call:

        >>> es = purecma.find_minimizing_values(pcma.ff.elli, 10 * [0.5], 0.3)[1]  # doctest:+SKIP

        and these lines:

        >>> es = purecma.CMAES(10 * [0.5], 0.3)
        >>> es.optimize(purecma.ff.elli)  # doctest:+SKIP

        :See: `CMAES`, `OOOptimizer`.
    """
    decender = CMAES(inital_solution, sigma, max_function_evaluations=max_function_evaluations, fitness_target=fitness_target)
    while not decender.stop():
        candidate_solutions = decender.ask()  # get a list of sampled candidate solutions
        fit = [ objective_function(x, *args) for x in candidate_solutions ]  # evaluate candidates
        decender.tell(candidate_solutions, fit)  # update distribution parameters

    return [ decender.best.x if decender.best.f < objective_function(decender.xmean) else decender.xmean, decender ]


class CmaesParameters:
    """static "internal" parameter setting for `CMAES`"""

    default_popsize = lambda number_of_dimensions: 4 + int(3 * log(number_of_dimensions))

    def __init__(self, number_of_dimensions, popsize=None):
        """set static, fixed "strategy" parameters once and for all."""
        # 
        # convert popsize
        # 
        if isinstance(popsize, (int, float)):
            popsize = int(popsize)
        elif callable(popsize):
            popsize = popsize(number_of_dimensions)
        elif type(popsize) == type(None):
            popsize = CmaesParameters.default_popsize(number_of_dimensions)
        else:
            raise Exception(f'''
                popsize should either be an integer or a function that accepts number_of_dimensions as an argument
                ex: popsize = lambda number_of_dimensions: math.log(number_of_dimensions) + 10
                Instead I got popsize of: {popsize}
            ''')
        
        self.dimension = number_of_dimensions

        # Strategy parameter setting: Selection
            
        self.lam = popsize
        self.mu = int(self.lam / 2)  # number of parents/points/solutions for recombination
        _weights = [log(self.lam / 2 + 0.5) - log(i + 1) if i < self.mu else 0 for i in range(self.lam)]
        w_sum = sum(_weights[: self.mu])
        self.weights = [w / w_sum for w in _weights]  # sum is one now
        self.mueff = sum(self.weights[: self.mu]) ** 2 / sum(w**2 for w in self.weights[: self.mu])  # variance-effectiveness of sum w_i x_i

        # Strategy parameter setting: Adaptation
        self.cc = (4 + self.mueff / number_of_dimensions) / (number_of_dimensions + 4 + 2 * self.mueff / number_of_dimensions)  # time constant for cumulation for C
        self.cs = (self.mueff + 2) / (number_of_dimensions + self.mueff + 5)  # time constant for cumulation for sigma control
        self.c1 = 2 / ((number_of_dimensions + 1.3) ** 2 + self.mueff)  # learning rate for rank-one update of C
        self.cmu = min([1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((number_of_dimensions + 2) ** 2 + self.mueff)])  # and for rank-mu update
        self.damps = 2 * self.mueff / self.lam + 0.3 + self.cs  # damping for sigma, usually close to 1

        # gap to postpone eigendecomposition to achieve O(number_of_dimensions**2) per eval
        # 0.5 is chosen such that eig takes 2 times the time of tell in >=20-D
        self.lazy_gap_evals = 0.5 * number_of_dimensions * self.lam * (self.c1 + self.cmu) ** -1 / N**2


class CMAES:  # could also inherit from object
    """
        class for non-linear non-convex numerical minimization with CMA-ES.

        the methods `__init__`, `ask`, `tell`, `stop`, and property `result`.

        Examples
        --------

        The Jupyter notebook or IPython are the favorite environments to
        execute these examples, both in ``%pylab`` mode. All examples
        minimize the function `elli`, output is not shown.

        First we need to import the module we want to use. We import `purecma`
        from `cma` as (aliased to) ``pcma``::

            from cma import purecma as pcma

        The shortest example uses the inherited method
        `OOOptimizer.optimize`::

            es = pcma.CMAES(8 * [0.1], 0.5).optimize(pcma.ff.elli)

        See method `CMAES.__init__` for a documentation of the input
        parameters to `CMAES`. We might have a look at the result::

            print(es.result[0])  # best solution and
            print(es.result[1])  # its function value

        Virtually the same example can be written with an explicit loop
        instead of using `optimize`, see also `find_minimizing_values`. This gives insight
        into the `CMAES` class interface and entire control over the
        iteration loop::

            pcma.find_minimizing_values??  # print source, works in jupyter/ipython only
            es = pcma.CMAES(9 * [0.5], 0.3)  # calls CMAES.__init__()

            # this loop resembles the method optimize
            while not es.stop():  # iterate
                X = es.ask()      # get candidate solutions
                f = [pcma.ff.elli(x) for x in X]  # evaluate solutions
                es.tell(X, f)     # do all the real work

            # final output
            print('termination by', es.stop())
            print('best f-value =', es.result[1])
            print('best solution =', es.result[0])

            print('potentially better solution xmean =', es.result[5])
            print("let's check f(xmean) = ", pcma.ff.elli(es.result[5]))

        A very similar example which may also save the logged data within
        the loop is the implementation of function `find_minimizing_values`.

        Details
        -------
        Most of the work is done in the method `tell`. The property
        `result` contains more useful output.

        :See: `find_minimizing_values`, `OOOptimizer.optimize`

    """

    def __init__(
        self,
        inital_solution,
        sigma,
        popsize=CmaesParameters.default_popsize,
        fitness_target=None,
        max_function_evaluations=(
            lambda popsize, number_of_dimensions: 100 * popsize + 150 * (number_of_dimensions + 3)**2 * popsize**0.5
        ),
        randn=random_normalvariate,
    ):  # mandatory  # 100 iterations plus...
        """
            Instantiate `CMAES` object instance using `inital_solution` and `sigma`.

            Parameters
            ----------
                `inital_solution`: `list`
                    of numbers (like ``[3, 2, 1.2]``), initial
                    solution vector
                `sigma`: `float`
                    initial step-size (standard deviation in each coordinate)
                `popsize`: `int` or `str`
                    population size, number of candidate samples per iteration
                `max_function_evaluations`: `int` or function
                    maximal number of function evaluations, a string is
                    evaluated with ``number_of_dimensions`` as search space dimension
                `fitness_target`: `float`
                    target function value
                `randn`: `callable`
                    normal random number generator, by default
                    `random.normalvariate`

            Details: this method initializes the dynamic state variables and
            creates a `CmaesParameters` instance for static parameters.
        """
        # process some input parameters and set static parameters
        number_of_dimensions = len(inital_solution)  # number of objective variables/problem dimension
        self.params = CmaesParameters(number_of_dimensions, popsize)
        
        # 
        # convert max_function_evaluations
        # 
        if isinstance(max_function_evaluations, (int, float)):
            max_function_evaluations = int(max_function_evaluations)
        elif callable(max_function_evaluations):
            max_function_evaluations = max_function_evaluations(number_of_dimensions, self.params.lam)
        else:
            raise Exception(f'''
                max_function_evaluations should either be an integer
                or a function that accepts number_of_dimensions and popsize as arguments
                ex: max_function_evaluations = lambda number_of_dimensions, popsize: math.log(number_of_dimensions) + 10
                Instead I got a max_function_evaluations of: {max_function_evaluations}
            ''')
            
        self.max_function_evaluations = max_function_evaluations
        self.fitness_target = fitness_target  # stop if fitness <= fitness_target
        self.randn = randn

        # initializing dynamic state variables
        self.xmean = inital_solution[:]  # initial point, distribution mean, a copy
        self.sigma = sigma
        self.pc = number_of_dimensions * [0]  # evolution path for C
        self.ps = number_of_dimensions * [0]  # and for sigma
        self.C = DecomposingPositiveMatrix(number_of_dimensions)  # covariance matrix
        self.count_eval = 0  # countiter should be equal to counteval / lam
        self.fit_vals = []  # for bookkeeping output and termination
        self.best = BestSolution()

    def ask(self):
        """sample lambda candidate solutions

        distributed according to::

            m + sigma * Normal(0,C) = m + sigma * B * D * Normal(0,I)
                                    = m + B * D * sigma * Normal(0,I)

        and return a `list` of the sampled "vectors".
        """
        self.C.update_eigensystem(self.count_eval, self.params.lazy_gap_evals)
        candidate_solutions = []
        for _k in range(self.params.lam):  # repeat lam times
            z = [self.sigma * eigenval**0.5 * self.randn(0, 1) for eigenval in self.C.eigenvalues]
            y = dot(self.C.eigenbasis, z)
            candidate_solutions.append(plus(self.xmean, y))
        return candidate_solutions

    def tell(self, arx, fitvals):
        """update the evolution paths and the distribution parameters m,
        sigma, and C within CMA-ES.

        Parameters
        ----------
            `arx`: `list` of "row vectors"
                a list of candidate solution vectors, presumably from
                calling `ask`. ``arx[k][i]`` is the i-th element of
                solution vector k.
            `fitvals`: `list`
                the corresponding objective function values, to be
                minimised
        """
        # bookkeeping and convenience short cuts
        self.count_eval += len(fitvals)  # evaluations used within tell
        N = len(self.xmean)
        par = self.params
        xold = self.xmean  # not a copy, xmean is assigned anew later

        # Sort by fitness
        arx = [arx[k] for k in argsort(fitvals)]  # sorted arx
        self.fit_vals = sorted(fitvals)  # used for termination and display only
        self.best.update(arx[0], self.fit_vals[0], self.count_eval)

        # recombination, compute new weighted mean value
        self.xmean = dot(arx[0 : par.mu], par.weights[: par.mu], transpose=True)
        #          = [sum(self.weights[k] * arx[k][i] for k in range(self.mu))
        #                                             for i in range(N)]

        # Cumulation: update evolution paths
        y = minus(self.xmean, xold)
        z = dot(self.C.invsqrt, y)  # == C**(-1/2) * (xnew - xold)
        csn = (par.cs * (2 - par.cs) * par.mueff) ** 0.5 / self.sigma
        for i in range(N):  # update evolution path ps
            self.ps[i] = (1 - par.cs) * self.ps[i] + csn * z[i]
        ccn = (par.cc * (2 - par.cc) * par.mueff) ** 0.5 / self.sigma
        # turn off rank-one accumulation when sigma increases quickly
        hsig = sum(x**2 for x in self.ps) / N / (1 - (1 - par.cs) ** (2 * self.count_eval / par.lam)) < 2 + 4.0 / (N + 1)  # ||ps||^2 / N is 1 in expectation  # account for initial value of ps  # should be smaller than 2 + ...
        for i in range(N):  # update evolution path pc
            self.pc[i] = (1 - par.cc) * self.pc[i] + ccn * hsig * y[i]

        # Adapt covariance matrix C
        # minor adjustment for the variance loss from hsig
        c1a = par.c1 * (1 - (1 - hsig**2) * par.cc * (2 - par.cc))
        self.C.multiply_with(1 - c1a - par.cmu * sum(par.weights))  # C *= 1 - c1 - cmu * sum(w)
        self.C.addouter(self.pc, par.c1)  # C += c1 * pc * pc^T, so-called rank-one update
        for k, wk in enumerate(par.weights):  # so-called rank-mu update
            if wk < 0:  # guaranty positive definiteness
                wk *= N * (self.sigma / self.C.mahalanobis_norm(minus(arx[k], xold))) ** 2
            self.C.addouter(minus(arx[k], xold), wk * par.cmu / self.sigma**2)  # C += wk * cmu * dx * dx^T

        # Adapt step-size sigma
        cn, sum_square_ps = par.cs / par.damps, sum(x**2 for x in self.ps)
        self.sigma *= exp(min(1, cn * (sum_square_ps / N - 1) / 2))

    def stop(self):
        """return satisfied termination conditions in a dictionary,

        generally speaking like ``{'termination_reason':value, ...}``,
        for example ``{'tolfun':1e-12}``, or the empty `dict` ``{}``.
        """
        response = {}
        if self.count_eval <= 0:
            return response
        if self.count_eval >= self.max_function_evaluations:
            response["max_function_evaluations"] = self.max_function_evaluations
        if self.fitness_target is not None and len(self.fit_vals) > 0 and self.fit_vals[0] <= self.fitness_target:
            response["fitness_target"] = self.fitness_target
        if self.C.condition_number > 1e14:
            response["condition"] = self.C.condition_number
        if len(self.fit_vals) > 1 and self.fit_vals[-1] - self.fit_vals[0] < 1e-12:
            response["tolfun"] = 1e-12
        if self.sigma * max(self.C.eigenvalues) ** 0.5 < 1e-11:
            # remark: max(D) >= max(diag(C))**0.5
            response["tolx"] = 1e-11
        return response

    @property
    def result(self):
        """the `tuple` ``(xbest, f(xbest), evaluations_xbest, evaluations,
        iterations, xmean, stds)``
        """
        return (self.best.x, self.best.f, self.best.evals, self.count_eval, int(self.count_eval / self.params.lam), self.xmean, [self.sigma * C_ii**0.5 for C_ii in self.C.diag])

# _____________________________________________________________________
# _________________ Fitness (Objective) Functions _____________________


class ff(object):  # instead of a submodule
    """versatile collection of test functions in static methods"""

    @staticmethod  # syntax available since 2.4
    def elli(x):
        """ellipsoid test objective function"""
        n = len(x)
        aratio = 1e3
        return sum(x[i] ** 2 * aratio ** (2.0 * i / (n - 1)) for i in range(n))

    @staticmethod
    def sphere(x):
        """sphere, ``sum(x**2)``, test objective function"""
        return sum(x[i] ** 2 for i in range(len(x)))

    @staticmethod
    def tablet(x):
        """discus test objective function"""
        return sum(xi**2 for xi in x) + (1e6 - 1) * x[0] ** 2

    @staticmethod
    def rosenbrock(x):
        """Rosenbrock test objective function"""
        n = len(x)
        if n < 2:
            raise ValueError("dimension must be greater one")
        return sum(100 * (x[i] ** 2 - x[i + 1]) ** 2 + (x[i] - 1) ** 2 for i in range(n - 1))


# _____________________________________________________________________
# _______________________ Helper Class&Functions ______________________
#
class BestSolution(object):
    """container to keep track of the best solution seen"""

    def __init__(self, x=None, f=None, evals=None):
        """take `x`, `f`, and `evals` to initialize the best solution"""
        self.x, self.f, self.evals = x, f, evals

    def update(self, x, f, evals=None):
        """update the best solution if ``f < self.f``"""
        if self.f is None or f < self.f:
            self.x = x
            self.f = f
            self.evals = evals
        return self

    @property
    def all(self):
        """``(x, f, evals)`` of the best seen solution"""
        return self.x, self.f, self.evals


class SquareMatrix(list):  # inheritance from numpy.ndarray is not recommended
    """rudimental square matrix class"""

    def __init__(self, dimension):
        """initialize with identity matrix"""
        for i in range(dimension):
            self.append(dimension * [0])
            self[i][i] = 1

    def multiply_with(self, factor):
        """multiply matrix in place with `factor`"""
        for row in self:
            for j in range(len(row)):
                row[j] *= factor
        return self

    def addouter(self, b, factor=1):
        """Add in place `factor` times outer product of vector `b`,

        without any dimensional consistency checks.
        """
        for i, row in enumerate(self):
            for j in range(len(row)):
                row[j] += factor * b[i] * b[j]
        return self

    @property
    def diag(self):
        """diagonal of the matrix as a copy (save to change)"""
        return [self[i][i] for i in range(len(self)) if i < len(self[i])]


class DecomposingPositiveMatrix(SquareMatrix):
    """Symmetric matrix maintaining its own eigendecomposition.

    If ``isinstance(C, DecomposingPositiveMatrix)``,
    the eigendecomposion (the return value of `eig`) is stored in
    the attributes `eigenbasis` and `eigenvalues` such that the i-th
    eigenvector is::

        [row[i] for row in C.eigenbasis]  # or equivalently
        [C.eigenbasis[j][i] for j in range(len(C.eigenbasis))]

    with eigenvalue ``C.eigenvalues[i]`` and hence::

        C = C.eigenbasis x diag(C.eigenvalues) x C.eigenbasis^T

    """

    def __init__(self, dimension):
        SquareMatrix.__init__(self, dimension)
        self.eigenbasis = eye(dimension)
        self.eigenvalues = dimension * [1]
        self.condition_number = 1
        self.invsqrt = eye(dimension)
        self.updated_eval = 0

    def update_eigensystem(self, current_eval, lazy_gap_evals):
        """Execute eigendecomposition of `self` if
        ``current_eval > lazy_gap_evals + last_updated_eval``.

        Assumes (for sake of simplicity) that `self` is positive
        definite and hence raises a `RuntimeError` otherwise.
        """
        if current_eval <= self.updated_eval + lazy_gap_evals:
            return self
        self._enforce_symmetry()  # probably not necessary with eig
        self.eigenvalues, self.eigenbasis = eig(self)  # O(N**3)
        if min(self.eigenvalues) <= 0:
            raise RuntimeError("The smallest eigenvalue is <= 0 after %d evaluations!" "\neigenvectors:\n%s \neigenvalues:\n%s" % (current_eval, str(self.eigenbasis), str(self.eigenvalues)))
        self.condition_number = max(self.eigenvalues) / min(self.eigenvalues)
        # now compute invsqrt(C) = C**(-1/2) = B D**(-1/2) B'
        # this is O(n^3) and takes about 25% of the time of eig
        for i in range(len(self)):
            for j in range(i + 1):
                self.invsqrt[i][j] = self.invsqrt[j][i] = sum(self.eigenbasis[i][k] * self.eigenbasis[j][k] / self.eigenvalues[k] ** 0.5 for k in range(len(self)))
        self.updated_eval = current_eval
        return self

    def mahalanobis_norm(self, dx):
        """return ``(dx^T * C^-1 * dx)**0.5``"""
        return sum(xi**2 for xi in dot(self.invsqrt, dx)) ** 0.5

    def _enforce_symmetry(self):
        for i in range(len(self)):
            for j in range(i):
                self[i][j] = self[j][i] = (self[i][j] + self[j][i]) / 2
        return self


def eye(dimension):
    """return identity matrix as `list` of "vectors" (lists themselves)"""
    m = [dimension * [0] for i in range(dimension)]
    # m = N * [N * [0]] fails because it gives N times the same reference
    for i in range(dimension):
        m[i][i] = 1
    return m


def dot(A, b, transpose=False):
    """usual dot product of "matrix" A with "vector" b.

    ``A[i]`` is the i-th row of A. With ``transpose=True``, A transposed
    is used.
    """
    if not transpose:
        return [sum(A[i][j] * b[j] for j in range(len(b))) for i in range(len(A))]
    else:
        return [sum(A[j][i] * b[j] for j in range(len(b))) for i in range(len(A[0]))]


def plus(a, b):
    """add vectors, return a + b"""
    return [a[i] + b[i] for i in range(len(a))]


def minus(a, b):
    """subtract vectors, return a - b"""
    return [a[i] - b[i] for i in range(len(a))]


def argsort(a):
    """return index list to get `a` in order, ie
    ``a[argsort(a)[i]] == sorted(a)[i]``
    """
    return sorted(range(len(a)), key=a.__getitem__)  # a.__getitem__(i) is a[i]


# ____________________________________________________________
# ____________________________________________________________
#
# C and B are arrays rather than matrices, because they are
# addressed via B[i][j], matrices can only be addressed via B[i,j]

# tred2(N, B, diagD, offdiag);
# tql2(N, diagD, offdiag, B);

# Symmetric Householder reduction to tridiagonal form, translated from
#   JAMA package.


def eig(C):
    """eigendecomposition of a symmetric matrix.

    Return the eigenvalues and an orthonormal basis
    of the corresponding eigenvectors, ``(EVals, Basis)``, where

    - ``Basis[i]``: `list`, is the i-th row of ``Basis``
    - the i-th column of ``Basis``, ie ``[Basis[j][i] for j in range(len(Basis))]``
      is the i-th eigenvector with eigenvalue ``EVals[i]``

    Details: much slower than `numpy.linalg.eigh`.
    """
    # class eig(object):
    #     def __call__(self, C):

    # Householder transformation of a symmetric matrix V into tridiagonal
    #   form.
    # -> n             : dimension
    # -> V             : symmetric nxn-matrix
    # <- V             : orthogonal transformation matrix:
    #                    tridiag matrix == V * V_in * V^t
    # <- d             : diagonal
    # <- e[0..n-1]     : off diagonal (elements 1..n-1)

    # Symmetric tridiagonal QL algorithm, iterative
    # Computes the eigensystem from a tridiagonal matrix in roughtly 3N^3
    #    operations
    # -> n     : Dimension.
    # -> d     : Diagonale of tridiagonal matrix.
    # -> e[1..n-1] : off-diagonal, output from Householder
    # -> V     : matrix output von Householder
    # <- d     : eigenvalues
    # <- e     : garbage?
    # <- V     : basis of eigenvectors, according to d

    #  tred2(N, B, diagD, offdiag); B=C on input
    #  tql2(N, diagD, offdiag, B);

    # import numpy as np
    # return np.linalg.eigh(C)  # return sorted EVs
    try:
        num_opt = False  # True doesn't work (yet)
        if num_opt:
            import numpy as np
    except ImportError:
        num_opt = False

    #  private void tred2 (int n, double V[][], double d[], double e[]) {
    def tred2(n, V, d, e):
        #  This is derived from the Algol procedures tred2 by
        #  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
        #  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
        #  Fortran subroutine in EISPACK.

        # num_opt = False  # factor 1.5 in 30-D

        d[:] = V[n - 1][:]  # d is output argument
        if num_opt:
            # V = np.asarray(V, dtype=float)
            e = np.asarray(e, dtype=float)

        # Householder reduction to tridiagonal form.

        for i in range(n - 1, 0, -1):
            # Scale to avoid under/overflow.
            h = 0.0
            if not num_opt:
                scale = 0.0
                for k in range(i):
                    scale = scale + abs(d[k])
            else:
                scale = sum(np.abs(d[0:i]))

            if scale == 0.0:
                e[i] = d[i - 1]
                for j in range(i):
                    d[j] = V[i - 1][j]
                    V[i][j] = 0.0
                    V[j][i] = 0.0
            else:

                # Generate Householder vector.
                if not num_opt:
                    for k in range(i):
                        d[k] /= scale
                        h += d[k] * d[k]
                else:
                    d[:i] /= scale
                    h = np.dot(d[:i], d[:i])

                f = d[i - 1]
                g = h**0.5

                if f > 0:
                    g = -g

                e[i] = scale * g
                h -= f * g
                d[i - 1] = f - g
                if not num_opt:
                    for j in range(i):
                        e[j] = 0.0
                else:
                    e[:i] = 0.0

                # Apply similarity transformation to remaining columns.
                for j in range(i):
                    f = d[j]
                    V[j][i] = f
                    g = e[j] + V[j][j] * f
                    if not num_opt:
                        for k in range(j + 1, i):
                            g += V[k][j] * d[k]
                            e[k] += V[k][j] * f
                        e[j] = g
                    else:
                        e[j + 1 : i] += V.T[j][j + 1 : i] * f
                        e[j] = g + np.dot(V.T[j][j + 1 : i], d[j + 1 : i])

                f = 0.0
                if not num_opt:
                    for j in range(i):
                        e[j] /= h
                        f += e[j] * d[j]
                else:
                    e[:i] /= h
                    f += np.dot(e[:i], d[:i])

                hh = f / (h + h)
                if not num_opt:
                    for j in range(i):
                        e[j] -= hh * d[j]
                else:
                    e[:i] -= hh * d[:i]

                for j in range(i):
                    f = d[j]
                    g = e[j]
                    if not num_opt:
                        for k in range(j, i):
                            V[k][j] -= f * e[k] + g * d[k]
                    else:
                        V.T[j][j:i] -= f * e[j:i] + g * d[j:i]

                    d[j] = V[i - 1][j]
                    V[i][j] = 0.0

            d[i] = h
        # end for i--

        # Accumulate transformations.

        for i in range(n - 1):
            V[n - 1][i] = V[i][i]
            V[i][i] = 1.0
            h = d[i + 1]
            if h != 0.0:
                if not num_opt:
                    for k in range(i + 1):
                        d[k] = V[k][i + 1] / h
                else:
                    d[: i + 1] = V.T[i + 1][: i + 1] / h

                for j in range(i + 1):
                    if not num_opt:
                        g = 0.0
                        for k in range(i + 1):
                            g += V[k][i + 1] * V[k][j]
                        for k in range(i + 1):
                            V[k][j] -= g * d[k]
                    else:
                        g = np.dot(V.T[i + 1][0 : i + 1], V.T[j][0 : i + 1])
                        V.T[j][: i + 1] -= g * d[: i + 1]

            if not num_opt:
                for k in range(i + 1):
                    V[k][i + 1] = 0.0
            else:
                V.T[i + 1][: i + 1] = 0.0

        if not num_opt:
            for j in range(n):
                d[j] = V[n - 1][j]
                V[n - 1][j] = 0.0
        else:
            d[:n] = V[n - 1][:n]
            V[n - 1][:n] = 0.0

        V[n - 1][n - 1] = 1.0
        e[0] = 0.0

    # Symmetric tridiagonal QL algorithm, taken from JAMA package.
    # private void tql2 (int n, double d[], double e[], double V[][]) {
    # needs roughly 3N^3 operations
    def tql2(n, d, e, V):
        #  This is derived from the Algol procedures tql2, by
        #  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
        #  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
        #  Fortran subroutine in EISPACK.

        # num_opt = False  # True doesn't work

        if not num_opt:
            for i in range(1, n):  # (int i = 1; i < n; i++):
                e[i - 1] = e[i]
        else:
            e[0 : n - 1] = e[1:n]
        e[n - 1] = 0.0

        f = 0.0
        tst1 = 0.0
        eps = 2.0**-52.0
        for l in range(n):  # (int l = 0; l < n; l++) {

            # Find small subdiagonal element

            tst1 = max(tst1, abs(d[l]) + abs(e[l]))
            m = l
            while m < n:
                if abs(e[m]) <= eps * tst1:
                    break
                m += 1

            # If m == l, d[l] is an eigenvalue,
            # otherwise, iterate.

            if m > l:
                iiter = 0
                while 1:  # do {
                    iiter += 1  # (Could check iteration count here.)

                    # Compute implicit shift

                    g = d[l]
                    p = (d[l + 1] - g) / (2.0 * e[l])
                    r = (p**2 + 1) ** 0.5  # hypot(p, 1.0)
                    if p < 0:
                        r = -r

                    d[l] = e[l] / (p + r)
                    d[l + 1] = e[l] * (p + r)
                    dl1 = d[l + 1]
                    h = g - d[l]
                    if not num_opt:
                        for i in range(l + 2, n):
                            d[i] -= h
                    else:
                        d[l + 2 : n] -= h

                    f = f + h

                    # Implicit QL transformation.

                    p = d[m]
                    c = 1.0
                    c2 = c
                    c3 = c
                    el1 = e[l + 1]
                    s = 0.0
                    s2 = 0.0

                    # hh = V.T[0].copy()  # only with num_opt
                    for i in range(m - 1, l - 1, -1):
                        # (int i = m-1; i >= l; i--) {
                        c3 = c2
                        c2 = c
                        s2 = s
                        g = c * e[i]
                        h = c * p
                        r = (p**2 + e[i] ** 2) ** 0.5  # hypot(p,e[i])
                        e[i + 1] = s * r
                        s = e[i] / r
                        c = p / r
                        p = c * d[i] - s * g
                        d[i + 1] = h + s * (c * g + s * d[i])

                        # Accumulate transformation.

                        if not num_opt:  # overall factor 3 in 30-D
                            for k in range(n):  # (int k = 0; k < n; k++){
                                h = V[k][i + 1]
                                V[k][i + 1] = s * V[k][i] + c * h
                                V[k][i] = c * V[k][i] - s * h
                        else:  # about 20% faster in 10-D
                            hh = V.T[i + 1].copy()
                            # hh[:] = V.T[i+1][:]
                            V.T[i + 1] = s * V.T[i] + c * hh
                            V.T[i] = c * V.T[i] - s * hh
                            # V.T[i] *= c
                            # V.T[i] -= s * hh

                    p = -s * s2 * c3 * el1 * e[l] / dl1
                    e[l] = s * p
                    d[l] = c * p

                    # Check for convergence.
                    if abs(e[l]) <= eps * tst1:
                        break
                # } while (Math.abs(e[l]) > eps*tst1);

            d[l] += f
            e[l] = 0.0

        # Sort eigenvalues and corresponding vectors.
        if 11 < 3:
            for i in range(n - 1):  # (int i = 0; i < n-1; i++) {
                k = i
                p = d[i]
                for j in range(i + 1, n):  # (int j = i+1; j < n; j++) {
                    if d[j] < p:  # NH find smallest k>i
                        k = j
                        p = d[j]

                if k != i:
                    d[k] = d[i]  # swap k and i
                    d[i] = p
                    for j in range(n):  # (int j = 0; j < n; j++) {
                        p = V[j][i]
                        V[j][i] = V[j][k]
                        V[j][k] = p

    # tql2
    N = len(C[0])
    V = [C[i][:] for i in range(N)]
    d = N * [0]
    e = N * [0]
    tred2(N, V, d, e)
    tql2(N, d, e, V)
    return d, V  # sorting of V-columns in place is non-trivial


def test():
    """test of the `purecma` module, called ``if __name__ == "__main__"``.

    Currently only based on `doctest`:

    >>> try: import cma.purecma as pcma
    ... except ImportError: import purecma as pcma
    >>> import random
    >>> random.seed(8)
    >>> xmin, es = pcma.find_minimizing_values(pcma.ff.rosenbrock, 4 * [0.5], 0.5)
    >>> print(es.counteval)
    1712
    >>> print(es.best.evals)
    1704
    >>> assert es.best.f < 1e-12
    >>> random.seed(5)
    >>> es = pcma.CMAES(4 * [0.5], 0.5)
    >>> es.params = pcma.CmaesParameters(es.params.dimension, es.params.lam,)
    >>> while not es.stop():
    ...     X = es.ask()
    ...     es.tell(X, [pcma.ff.rosenbrock(x) for x in X])
    >>> print("%s, %s" % (pcma.ff.rosenbrock(es.result[0]) < 1e-13,
    ...                   es.result[2] < 1600))
    True, True

    Large population size:

    >>> random.seed(4)
    >>> es = pcma.CMAES(3 * [1], 1)
    >>> es.params = pcma.CmaesParameters(es.params.dimension, 300)
    >>> try:
    ...    es = es.optimize(pcma.ff.elli)
    ... except AttributeError:  # OOOptimizer.optimize is not available
    ...     while not es.stop():
    ...         X = es.ask()
    ...         es.tell(X, [pcma.ff.elli(x) for x in X])
    >>> assert es.result[1] < 1e13
    >>> print(es.result[2])
    9300

    """
    import doctest

    print("launching doctest...")
    print(doctest.testmod(report=True, verbose=0))  # module test


# _____________________________________________________________________
# _____________________________________________________________________
#
if __name__ == "__main__":

    test()

    # find_minimizing_values(ff.rosenbrock, 10 * [0.5], 0.5)
