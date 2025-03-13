import numpy as np
import time

class Basic_Problem:
    """
    Abstract super class for problems and applications.
    """

    def reset(self):
        self.T1=0

    def eval(self, x):
        """
        A general version of func() with adaptation to evaluate both individual and population.
        """
        start=time.perf_counter()

        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 1:  # x is a single individual
            y=self.func(x.reshape(1, -1))[0]
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y
        elif x.ndim == 2:  # x is a whole population
            y=self.func(x)
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y
        else:
            y=self.func(x.reshape(-1, x.shape[-1]))
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y

    def func(self, x):
        raise NotImplementedError
