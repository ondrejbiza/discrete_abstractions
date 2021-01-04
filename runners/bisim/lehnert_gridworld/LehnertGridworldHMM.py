import constants
from model.LehnertGridworldModelHMM import LehnertGridworldModelHMM
from runners.bisim.lehnert_gridworld.LehnertGridworldGMM import LehnertGridworldGMMRunner


class LehnertGridworldHMMRunner(LehnertGridworldGMMRunner):

    def __init__(self, runner_config, model_config):

        super(LehnertGridworldHMMRunner, self).__init__(runner_config, model_config)

    def prepare_model_(self):

        self.model = LehnertGridworldModelHMM(self.model_config)
        self.model.build()
        self.model.start_session()

        if self.load_model_path is not None:
            self.model.load(self.load_model_path)

        self.to_run = {
            constants.TRAIN_STEP: self.model.train_step,
            constants.TOTAL_LOSS: self.model.loss_t,
            constants.Q_LOSS: self.model.q_loss_t,
            constants.ENTROPY_LOSS: self.model.encoder_entropy_t,
            constants.PRIOR_LOG_LIKELIHOOD: self.model.prior_log_likelihood_t
        }
