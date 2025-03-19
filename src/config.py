import argparse
import time


def get_config(args=None):
    parser = argparse.ArgumentParser()
    # Common config
    parser.add_argument('--problem', default='bbob', choices=['bbob', 'bbob-noisy', 'bbob-torch', 'bbob-noisy-torch', 'protein', 'protein-torch','augmented-WCCI2020'],
                        help='specify the problem suite')
    parser.add_argument('--dim', type=int, default=10, help='dimension of search space')
    parser.add_argument('--upperbound', type=float, default=5, help='upperbound of search space')
    parser.add_argument('--difficulty', default='easy', choices=['easy', 'difficult'], help='difficulty level')
    parser.add_argument('--device', default='cpu', help='device to use')
    parser.add_argument('--train', default=None, action='store_true', help='switch to train mode')
    parser.add_argument('--test', default=None, action='store_true', help='switch to inference mode')
    parser.add_argument('--rollout', default=None, action='store_true', help='switch to rollout mode')
    parser.add_argument('--run_experiment', default=None, action='store_true', help='switch to run_experiment mode')
    parser.add_argument('--mgd_test', default=None, action='store_true', help='switch to mgd_test mode')
    parser.add_argument('--mte_test', default=None, action='store_true', help='switch to mte_test mode')

    # Training parameters
    parser.add_argument('--max_learning_step', type=int, default=1500000, help='the maximum learning step for training')
    parser.add_argument('--train_batch_size', type=int, default=1, help='batch size of train set')
    parser.add_argument('--train_agent', default=None, help='agent for training')
    parser.add_argument('--train_optimizer', default=None, help='optimizer for training')
    parser.add_argument('--agent_save_dir', type=str, default='agent_model/train/',
                        help='save your own trained agent model')
    parser.add_argument('--log_dir', type=str, default='output/',
                        help='logging testing output')
    parser.add_argument('--draw_interval', type=int, default=3, help='interval epochs in drawing figures')
    parser.add_argument('--agent_for_plot_training', type=str, nargs='+', default=['RL_HPSDE_Agent'],
                        help='learnable optimizer to compare')
    parser.add_argument('--n_checkpoint', type=int, default=20, help='number of training checkpoints')
    parser.add_argument('--resume_dir', type=str, help='directory to load previous checkpoint model')

    # Testing parameters
    parser.add_argument('--agent', default=None, help='None: traditional optimizer, else Learnable optimizer')
    parser.add_argument('--agent_load_dir', type=str,
                        help='load your own agent model')
    parser.add_argument('--optimizer', default=None, help='your own learnable or traditional optimizer')
    parser.add_argument('--agent_for_cp', type=str, nargs='+', default=[],
                        help='learnable optimizer to compare')
    parser.add_argument('--l_optimizer_for_cp', type=str, nargs='+', default=[],
                        help='learnable optimizer to compare')  # same length with "agent_for_cp"
    parser.add_argument('--t_optimizer_for_cp', type=str, nargs='+', default=[],
                        help='traditional optimizer to compare')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size of test set')

    # Rollout parameters
    parser.add_argument('--agent_for_rollout', type=str, nargs='+', help='learnable agent for rollout')
    parser.add_argument('--optimizer_for_rollout', type=str, nargs='+', help='learnabel optimizer for rollout')
    parser.add_argument('--plot_smooth', type=float, default=0.8,
                        help='a float between 0 and 1 to control the smoothness of figure curves')

    # parameters common to mgd_test(zero-shot) & mte_test(transfer_learning)
    parser.add_argument('--problem_from', choices=['bbob', 'bbob-noisy', 'bbob-torch', 'bbob-noisy-torch', 'protein', 'protein-torch'],
                        help='source problem set in zero-shot and transfer learning')
    parser.add_argument('--problem_to', choices=['bbob', 'bbob-noisy', 'bbob-torch', 'bbob-noisy-torch', 'protein', 'protein-torch'],
                        help='target problem set in zero-shot and transfer learning')
    parser.add_argument('--difficulty_from', default='easy', choices=['easy', 'difficult'],
                        help='difficulty of source problem set in zero-shot and transfer learning')
    parser.add_argument('--difficulty_to', default='easy', choices=['easy', 'difficult'],
                        help='difficulty of target problem set in zero-shot and transfer learning')

    # mgd_test(zero-shot) parameters
    parser.add_argument('--model_from', type=str, help='the model trained on source problem set')
    parser.add_argument('--model_to', type=str, help='the model trained on target problem set')

    # mte_test(transfer_learning) parameters
    parser.add_argument('--pre_train_rollout', type=str, help='path of pre-train models rollout result .pkl file')
    parser.add_argument('--scratch_rollout', type=str, help='path of scratch models rollout result .pkl file')

    config = parser.parse_args(args)
    config.maxFEs = 2000 * config.dim
    # for bo, maxFEs is relatively smaller due to time limit
    config.bo_maxFEs = 10 * config.dim
    config.n_logpoint = 50

    if config.run_experiment and len(config.agent_for_cp) >= 1:
        assert config.agent_load_dir is not None, "Option --agent_load_dir must be given since you specified option --agent_for_cp."

    if config.mgd_test or config.mte_test:
        config.problem = config.problem_to
        config.difficulty = config.difficulty_to

    if config.problem in ['protein', 'protein-torch']:
        config.dim = 12
        config.maxFEs = 1000
        config.bo_maxFEs = 10
        config.n_logpoint = 5

    config.run_time = f'{time.strftime("%Y%m%dT%H%M%S")}_{config.problem}_{config.difficulty}_{config.dim}D'
    config.test_log_dir = config.log_dir + '/test/' + config.run_time + '/'
    config.rollout_log_dir = config.log_dir + '/rollout/' + config.run_time + '/'
    config.mgd_test_log_dir = config.log_dir + '/mgd_test/' + config.run_time + '/'
    config.mte_test_log_dir = config.log_dir + '/mte_test/' + config.run_time + '/'

    if config.train or config.run_experiment:
        config.agent_save_dir = config.agent_save_dir + config.train_agent + '/' + config.run_time + '/'

    config.save_interval = config.max_learning_step // config.n_checkpoint
    config.log_interval = config.maxFEs // config.n_logpoint

    if 'DEAP_CMAES' not in config.t_optimizer_for_cp:
        config.t_optimizer_for_cp.append('DEAP_CMAES')
    if 'Random_search' not in config.t_optimizer_for_cp:
        config.t_optimizer_for_cp.append('Random_search')

    return config
