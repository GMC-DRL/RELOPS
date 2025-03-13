from .basic_problem import Basic_Problem
import numpy as np
import time 

class WCCI2020_Basic_Problem(Basic_Problem):
    def __init__(self, dim, shift, rotate, bias):
        self.T1 = 0
        self.dim = dim
        self.shift = shift
        self.rotate = rotate
        self.bias = bias
        self.lb = -50
        self.ub = 50
        self.FES = 0
        self.opt = self.shift
        # self.optimum = self.eval(self.get_optimal())
        self.optimum = self.func(self.get_optimal().reshape(1, -1))[0]

    def get_optimal(self):
        return self.opt

    def func(self, x):
        raise NotImplementedError
    
    def decode(self, x):
        return x * (self.ub - self.lb) + self.lb

    def sr_func(self, x, shift, rotate):
        y = x - shift
        return np.matmul(rotate, y.transpose()).transpose()
    
    def eval(self, x):
        """
        A specific version of func() with adaptation to evaluate both individual and population in MTO.
        """
        start=time.perf_counter()
        x = self.decode(x)  # the solution in MTO is constrained in a unified space [0,1]
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 1:  # x is a single individual
            x = x[:self.dim]
            y=self.func(x.reshape(1, -1))[0]
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y
        elif x.ndim == 2:  # x is a whole population
            x = x[:, :self.dim]
            y=self.func(x)
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y
        else:
            x = x[:,:,:self.dim]
            y=self.func(x.reshape(-1, x.shape[-1]))
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y

class Sphere(WCCI2020_Basic_Problem):
    def __init__(self, dim, shift, rotate, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -100
        self.ub = 100

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        return np.sum(z ** 2, -1)

class Ackley(WCCI2020_Basic_Problem):
    def __init__(self, dim, shift, rotate, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -50
        self.ub = 50

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        sum1 = -0.2 * np.sqrt(np.sum(z ** 2, -1) / self.dim)
        sum2 = np.sum(np.cos(2 * np.pi * z), -1) / self.dim
        return np.round(np.e + 20 - 20 * np.exp(sum1) - np.exp(sum2), 15) + self.bias
    

class Griewank(WCCI2020_Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -100
        self.ub = 100

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        s = np.sum(z ** 2, -1)
        p = np.ones(x.shape[0])
        for i in range(self.dim):
            p *= np.cos(z[:, i] / np.sqrt(1 + i))
        return 1 + s / 4000 - p + self.bias

class Rastrigin(WCCI2020_Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -50
        self.ub = 50

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        return np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z) + 10, -1) + self.bias
    
class Rosenbrock(WCCI2020_Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -50
        self.ub = 50

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        z += 1
        z_ = z[:, 1:]
        z = z[:, :-1]
        tmp1 = z ** 2 - z_
        return np.sum(100 * tmp1 * tmp1 + (z - 1) ** 2, -1) + self.bias

class Weierstrass(WCCI2020_Basic_Problem):
    def __init__(self, dim, shift, rotate, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -0.5
        self.ub = 0.5

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        a, b, k_max = 0.5, 3.0, 20
        sum1, sum2 = 0, 0
        for k in range(k_max + 1):
            sum1 += np.sum(np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * (z + 0.5)), -1)
            sum2 += np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * 0.5)
        return sum1 - self.dim * sum2 + self.bias
    
class Schwefel(WCCI2020_Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -500
        self.ub = 500

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        a = 4.209687462275036e+002
        b = 4.189828872724338e+002
        z += a
        g = z * np.sin(np.sqrt(np.abs(z)))
        return b * self.dim - np.sum(g,-1)