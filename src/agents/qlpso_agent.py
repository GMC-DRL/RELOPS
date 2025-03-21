import numpy as np
from .TabularQ_Agent import TabularQ_Agent
from .utils import save_class
from VectorEnv.great_para_env import ParallelEnv
import torch
from typing import Any, Callable, List, Optional, Tuple, Union, Literal
from scipy.special import softmax

class QLPSO_Agent(TabularQ_Agent):
    def __init__(self, config):
        self.config = config
        # define hyperparameters that agent needs
        self.config.n_state = 4
        self.config.n_act = 4
        self.config.lr_model = 1
        self.config.alpha_decay = True
        self.config.gamma = 0.8

        self.config.epsilon = None

        # self.__q_table = np.zeros((config.n_states, config.n_actions))

        self.__alpha_max = self.config.lr_model
        # self.lr_model = self.config.lr_model
        self.__alpha_decay = self.config.alpha_decay
        self.__max_learning_step =  self.config.max_learning_step
        # self.__global_ls = 0  # a counter of accumulated learned steps
        self.device = self.config.device
        # self.__cur_checkpoint = 0
        super().__init__(self.config)

    def __get_action(self, state):  # Make action decision according to the given state
        # Get the corresponding rows from the Q-table and compute the softmax
        q_values = self.q_table[state]  # shape: (bs, n_actions)

        # Compute the action probabilities for each state
        prob = softmax(q_values)  # shape: (bs, n_actions)

        # Choose an action based on the probabilities
        action = torch.multinomial(prob, 1)  # shape: (bs, 1)

        # Return the action
        return action.squeeze().numpy()  # Return the action and remove unnecessary dimensions

    def train_episode(self, 
                      envs, 
                      para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc']='dummy',
                      asynchronous: Literal[None, 'idle', 'restart', 'continue']=None,
                      num_cpus: Optional[Union[int, None]]=1,
                      num_gpus: int=0,
                      required_info={'normalizer': 'normalizer',
                                     'gbest':'gbest'
                                     }):
        if self.device != 'cpu':
            num_gpus = max(num_gpus, 1)
        env = ParallelEnv(envs, para_mode, asynchronous, num_cpus, num_gpus)
        
        # params for training
        gamma = self.gamma
        
        state = env.reset()
        state = torch.tensor(state,dtype=torch.int64)
        
        _R = torch.zeros(len(env))
        _loss = []
        # sample trajectory
        while not env.all_done():
            action = self.__get_action(state)
            # state transient
            next_state, reward, is_end, info = env.step(action)
            _R += reward
            # update Q-table
            # TD_error = [reward[i] + gamma * self.q_table[next_state[i]].max() - self.q_table[state[i]][action[i]]\
            #     for i in range(len(state)) ]
            reward = torch.FloatTensor(reward).to(self.device)
            TD_error = reward + gamma * torch.max(self.q_table[next_state], dim = 1)[0] - self.q_table[state, action]

            _loss.append(TD_error.mean().item())
            self.q_table[state, action] += self.lr_model * TD_error

            
            self.learning_time += 1

            if self.learning_time >= (self.config.save_interval * self.cur_checkpoint):
                save_class(self.config.agent_save_dir, 'checkpoint'+str(self.cur_checkpoint), self)
                self.cur_checkpoint += 1

            if self.learning_time >= self.config.max_learning_step:
                _Rs = _R.detach().numpy().tolist()
                return_info = {'return': _Rs, 'loss': np.mean(_loss), 'learn_steps': self.learning_time, }
                for key in required_info.keys():
                    return_info[key] = env.get_env_attr(required_info[key])
                env.close()
                return self.learning_time >= self.config.max_learning_step, return_info
        
            if self.__alpha_decay:
                self.lr_model = self.__alpha_max - (self.__alpha_max - 0.1) * self.learning_time / self.__max_learning_step
            
            # store info
            state = torch.tensor(next_state, dtype = torch.int64)
            
        is_train_ended = self.learning_time >= self.config.max_learning_step
        _Rs = _R.detach().numpy().tolist()
        return_info = {'return': _Rs, 'loss': np.mean(_loss), 'learn_steps': self.learning_time, }
        for key in required_info.keys():
            return_info[key] = env.get_env_attr(required_info[key])
        env.close()

        return is_train_ended, return_info
        
    # def rollout_episode(self, env):
    #     state = env.reset()
    #     done = False
    #     R = 0  # total reward
    #     while not done:
    #         action = self.__get_action(state)
    #         next_state, reward, done = env.step(action)
    #         R += reward
    #         state = next_state
    #     return {'cost': env.optimizer.cost, 'fes': env.optimizer.fes, 'return': R}
