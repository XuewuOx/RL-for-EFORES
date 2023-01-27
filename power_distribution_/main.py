import sys
import os

project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_path)

from simulation_environment.environment import rl_env
from PPO.PPO_algorithm import PPO_pipline


def training():
    '''
    Let the PPO interact with the environment, train the PPO model.
    :return:
    '''

    env = rl_env()

    ent_coef = 0
    file_name = 'common'
    rl = PPO_pipline(
        file_name=file_name,
        env=env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=None,
        n_epochs=10,
        ent_coef=ent_coef,
        seed=400,
        device='cuda',
    )
    rl.learn(total_timesteps=10000000)

    '''
       Initilization arguments:
        :param file_name: The filename where the parameter log file is stored.
        :param learning_rate: learning rate
        :param n_steps: The number of timesteps each time the ppo interacts with the environment and samples
        :param batch_size: batch size for training the model
        :param n_epochs: number of epochs per training
        :param gamma: discount factor
        :param gae_lambda: One of the parameters to calculate GAE in PPO
        :param clip_range: Parameters for PPO clipping surrogate object update ratio
        :param clip_range_vf: Parameters for PPO clipping state value object update ratio
        :param normalize_advantage: whether to normalize estimation advantage value
        :param ent_coef: param for encouraging exploration
        :param vf_coef: param for state value object loss proportion
        :param max_grad_norm: gradient clipping strength
        :param seed: Number of random seeds
        :param device: Use 'cpu' or 'gpu' to train the model
    '''
if __name__ == '__main__':

    training()
