"""
This is a basic class for learnable backbone optimizer.
Your own backbone optimizer should inherit from this class and have the following methods:
    1. __init__(self, config) : to initialize the backbone optimizer.
    2. init_population(self, problem) : to initialize the population, calculate costs using problem.eval()
       and record some information such as pbest and gbest if needed. It's expected to return a state for
       agent to make decisions.
    3. update(self, action, problem) : to update the population or one individual in population as you wish
       using the action given by agent, calculate new costs using problem.eval() and update some records
       if needed. It's expected to return a tuple of [next_state, reward, is_done] for agent to learn.
"""
from typing import Any, Tuple

class Learnable_Optimizer:
    """
    Abstract super class for learnable backbone optimizers.
    """
    def __init__(self, config):
        self.__config = config

    def init_population(self,
                        tasks:Any) -> Any:
        raise NotImplementedError

    def update(self,
               action: Any,
               tasks:Any) -> Tuple[Any]:
        raise NotImplementedError
