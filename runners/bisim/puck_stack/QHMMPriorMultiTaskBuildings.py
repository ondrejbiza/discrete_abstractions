import numpy as np
from runners.bisim.puck_stack.QHMMPriorMultiTask import QHMMPriorMultiTask
from envs.continuous_multi_task import ContinuousMultiTask, CheckGoalFactory
from agents.quotient_mdp_nbisim import QuotientMDPNBisim
from agents.solver import Solver
from envs.continuous_multi_task_buildings import ContinuousMultiTask, CheckGoalFactory


class QHMMPriorMultiTaskBuildings(QHMMPriorMultiTask):

    def __init__(self, load_path, grid_size, num_pucks, logger, saver, num_blocks, num_components, hiddens,
                 encoder_learning_rate, beta0, beta1, beta2, weight_decay, encoder_optimizer, num_steps,
                 active_task_indices, evaluation_task_indices, dones_index,
                 disable_batch_norm=False, disable_softplus=False, no_sample=False, only_one_q_value=True,
                 gt_q_values=False, disable_resize=False, oversample=False, validation_fraction=0.2,
                 validation_freq=1000, summaries=None, load_model_path=None, batch_size=100, beta3=0.0,
                 post_train_hmm=False, post_train_prior=False, post_train_t_and_prior=False,
                 zero_sd_after_training=False, cluster_predict_qs=False, cluster_predict_qs_weight=0.1,
                 hard_abstract_state=False, save_gifs=False, prune_abstraction=False,
                 prune_threshold=0.01, prune_abstraction_new_means=False, old_bn_settings=False,
                 sample_abstract_state=False, softmax_policy=False, softmax_policy_temp=1.0,
                 include_goal_states=False, discount=0.9, soft_picture_goals=False,
                 shift_q_values=False, goal_rewards_threshold=None, q_values_indices=None,
                 fast_eval=False, show_qs=False, fix_prior_training=False, model_learning_rate=None,
                 random_shape=False):

        super(QHMMPriorMultiTaskBuildings, self).__init__(
            load_path, grid_size, num_pucks, logger, saver, num_blocks, num_components, hiddens,
            encoder_learning_rate, beta0, beta1, beta2, weight_decay, encoder_optimizer, num_steps,
            active_task_indices, evaluation_task_indices, dones_index,
            disable_batch_norm=disable_batch_norm, disable_softplus=disable_softplus, no_sample=no_sample, only_one_q_value=only_one_q_value,
            gt_q_values=gt_q_values, disable_resize=disable_resize, oversample=oversample, validation_fraction=validation_fraction,
            validation_freq=validation_freq, summaries=summaries, load_model_path=load_model_path, batch_size=batch_size, beta3=beta3,
            post_train_hmm=post_train_hmm, post_train_prior=post_train_prior, post_train_t_and_prior=post_train_t_and_prior,
            zero_sd_after_training=zero_sd_after_training, cluster_predict_qs=cluster_predict_qs, cluster_predict_qs_weight=cluster_predict_qs_weight,
            hard_abstract_state=hard_abstract_state, save_gifs=save_gifs, prune_abstraction=prune_abstraction,
            prune_threshold=prune_threshold, prune_abstraction_new_means=prune_abstraction_new_means, old_bn_settings=old_bn_settings,
            sample_abstract_state=sample_abstract_state, softmax_policy=softmax_policy, softmax_policy_temp=softmax_policy_temp,
            include_goal_states=include_goal_states, discount=discount, soft_picture_goals=soft_picture_goals,
            shift_q_values=shift_q_values, goal_rewards_threshold=goal_rewards_threshold, q_values_indices=q_values_indices,
            fast_eval=fast_eval, show_qs=show_qs, fix_prior_training=fix_prior_training, model_learning_rate=model_learning_rate,
            random_shape=random_shape
        )

    def run_abstract_agent_(self, q_values, goal_idx, append_name=None):

        def classify(state):

            depth = state[0][:, :, 0]
            if not self.disable_resize:
                depth = self.model.session.run(self.resized_t, feed_dict={
                    self.images_pl: depth[np.newaxis]
                })[0]
            depth = (depth - self.means) / self.stds

            _, log_probs = self.model.encode(
                np.array([depth]), hand_states=np.array([[state[1]]]), zero_sd=self.zero_sd_after_training
            )
            log_probs = log_probs[0]

            if self.hard_abstract_state:
                idx = np.argmax(log_probs)
                output = np.zeros_like(log_probs)
                output[idx] = 1.0
            elif self.sample_abstract_state:
                idx = np.random.choice(list(range(len(log_probs))), p=np.exp(log_probs))
                output = np.zeros_like(log_probs)
                output[idx] = 1.0
            else:
                output = np.exp(log_probs)

            return output

        fcs = [
            CheckGoalFactory.check_goal_puck_stacking(2),
            CheckGoalFactory.check_goal_puck_stacking(3),
            CheckGoalFactory.check_b1_side(CheckGoalFactory.SIDE_TOP, 2),
            CheckGoalFactory.check_b1_side(CheckGoalFactory.SIDE_BOTTOM, 2),
            CheckGoalFactory.check_b1_side(CheckGoalFactory.SIDE_LEFT, 2),
            CheckGoalFactory.check_b1_side(CheckGoalFactory.SIDE_RIGHT, 2)
        ]
        fcs = [fcs[goal_idx]]

        self.game_env = ContinuousMultiTask(
            self.grid_size, self.num_pucks, 1, fcs, self.grid_size, height_noise_std=0.05, render_always=True,
            only_one_goal=True, random_shape=False
        )

        rewards_name = "rewards"
        gifs_name = "gifs"

        if append_name is not None:
            rewards_name += "_" + append_name
            gifs_name += "_" + append_name

        self.abstract_agent = QuotientMDPNBisim(
            classify, self.game_env, q_values, minatar=False, softmax_policy=self.softmax_policy,
            softmax_policy_temp=self.softmax_policy_temp
        )

        if self.save_gifs:
            gifs_path = self.saver.get_new_dir(gifs_name)
        else:
            gifs_path = None

        solver = Solver(
            self.game_env, self.abstract_agent, int(1e+7), int(1e+7), 0, max_episodes=1000, train=False,
            gif_save_path=gifs_path, rewards_file=self.saver.get_save_file(rewards_name, "dat"),
            verbose=False
        )
        solver.run()

        self.logger.info("{:s}: {:.2f}".format(rewards_name, np.mean(solver.episode_rewards)))
