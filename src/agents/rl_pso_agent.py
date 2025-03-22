import math
import torch
from torch import nn
from torch.distributions import Normal
from .REINFORCE_Agent import REINFORCE_Agent
from agent.networks import MLP
from .utils import *
from agent.utils import *
from VectorEnv.great_para_env import ParallelEnv
from typing import Any, Callable, List, Optional, Tuple, Union, Literal

class PolicyNetwork(nn.Module):
    def __init__(self, config):
        super(PolicyNetwork, self).__init__()

        net_config = [{'in': config.feature_dim, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
                      {'in': 32, 'out': 8, 'drop_out': 0, 'activation': 'ReLU'},
                      {'in': 8, 'out': config.action_dim, 'drop_out': 0, 'activation': 'None'}]

        self.__mu_net = MLP(net_config)
        self.__sigma_net = MLP(net_config)

        self.__max_sigma = config.max_sigma
        self.__min_sigma = config.min_sigma

    def forward(self, x_in, require_entropy=False, require_musigma=False):
        mu = self.__mu_net(x_in)
        mu = (torch.tanh(mu) + 1.) / 2.
        sigma = self.__sigma_net(x_in)
        sigma = (torch.tanh(sigma) + 1.) / 2.
        sigma = torch.clamp(sigma, min=self.__min_sigma, max=self.__max_sigma)

        policy = Normal(mu, sigma)
        action = policy.sample()

        filter = torch.abs(action - 0.5) >= 0.5
        action = torch.where(filter, (action + 3 * sigma.detach() - mu.detach()) * (1. / 6 * sigma.detach()), action)
        log_prob = policy.log_prob(action)

        if require_entropy:
            entropy = policy.entropy()

            out = (action, log_prob, entropy)
        else:
            if require_musigma:
                out = (action, log_prob, mu, sigma)
            else:
                out = (action, log_prob)

        return out


class RL_PSO_Agent(REINFORCE_Agent):
    def __init__(self, config):
        
        # add specified config
        self.config = config
        self.config.feature_dim = 2*config.dim
        self.config.action_dim = 1
        self.config.action_shape = (1,)
        self.config.max_sigma = 0.7
        self.config.min_sigma = 0.01
        # origin RLPSO doesnt have gamma : set a default value
        self.config.gamma = self.config.min_sigma
        self.config.lr_model = 1e-5

        model = PolicyNetwork(config)

        # optimizer
        self.config.optimizer = 'Adam'
        # origin RLPSO doesn't have clip
        self.config.max_grad_norm = math.inf

        super().__init__(self.config,{'model':model},[self.config.lr_model])

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
        
        # input action_dim should be : bs, ps
        # action in (0,1) the ratio to learn from pbest & gbest
        state = env.reset()
        try:
            state = torch.FloatTensor(state).to(self.device)
        except:
            pass
        
        _R = torch.zeros(len(env))
        _loss = []
        # sample trajectory
        while not env.all_done():
            action, log_prob = self.model(state)
            action = action.reshape(len(env))
            action = action.cpu().numpy()
            log_prob = log_prob.reshape(len(env))
            
            next_state, reward, is_done,_ = env.step(action)
            reward = torch.FloatTensor(reward).to(self.device)
            _R += reward
            state = torch.FloatTensor(next_state).to(self.device)
            policy_gradient = -log_prob*reward
            loss = policy_gradient.mean()

            self.optimizer.zero_grad()
            loss.mean().backward()
            _loss.append(loss.item())
            self.optimizer.step()
            self.learning_time += 1
            if self.learning_time >= (self.config.save_interval * self.cur_checkpoint):
                save_class(self.config.agent_save_dir,'checkpoint'+str(self.cur_checkpoint),self)
                self.cur_checkpoint+=1
  
        is_train_ended = self.learning_time >= self.config.max_learning_step
        return_info = {'return': _R.numpy(), 'loss' : np.mean(_loss),'learn_steps': self.learning_time, }
        for key in required_info.keys():
            return_info[key] = env.get_env_attr(required_info[key])
        env.close()
        
        return is_train_ended, return_info


    
    # def rollout_episode(self, env):
    #     is_done = False
    #     state = env.reset()
    #     R=0
    #     while not is_done:
    #         state = torch.FloatTensor(state)
    #         action, _ = self.__nets(state)
    #         state, reward, is_done = env.step(action.cpu().numpy())
    #         R+=reward
    #     return {'cost': env.optimizer.cost, 'fes': env.optimizer.fes,'return':R}

    def rollout_batch_episode(self, 
                              envs, 
                              seeds=None,
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
        if seeds is not None:
            env.seed(seeds)
        state = env.reset()
        try:
            state = torch.FloatTensor(state).to(self.device)
        except:
            pass
        
        R = torch.zeros(len(env))
        # sample trajectory
        while not env.all_done():
            with torch.no_grad():
                action, _  = self.model(state)
            action = action.reshape(len(env))
            action = action.cpu().numpy()
            # state transient
            state, rewards, is_end, info = env.step(action)
            # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))
            R += torch.FloatTensor(rewards).squeeze()
            # store info
            try:
                state = torch.FloatTensor(state).to(self.device)
            except:
                pass
        results = {'return': R.numpy()}
        for key in required_info.keys():
            results[key] = env.get_env_attr(required_info[key])
        return results