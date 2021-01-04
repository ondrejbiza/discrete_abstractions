F_BAR_NETWORK_NAMESPACE = "f_bar"
PROP_NETWORK_TEMPLATE = "prop{:d}"

GAMMA_NETWORK_NAMESPACE = "gamma"

OBJECT_PUCK = "disk"
OBJECT_BLOCK = "block"
OBJECTS = [OBJECT_PUCK, OBJECT_BLOCK]

OPENRAVE_BLOCK_HEIGHT = 0.499
OPENRAVE_DISK_HEIGHT = 0.245

OPT_ADAM = "adam"
OPT_SGD = "sgd"
OPT_MOMENTUM = "momentum"

REWARD_CONFIDENCE = "confidence"
REWARD_NUM_SAMPLES = "samples"

EASY = "easy"
MEDIUM = "medium"
HARD = "hard"

DECONV_BEFORE = "deconv_before"
DECONV_AFTER = "deconv_after"
UPSAMPLE_BEFORE = "upsample_before"
UPSAMPLE_AFTER = "upsample_after"

GOAL_STATE = "goal"

DEFAULT_BLOCK_SIZE = 28

TRAIN_STEP = "train_step"
PRIOR_TRAIN_STEP = "prior_train_step"
TOTAL_LOSS = "total_loss"
TRANSITION_LOSS = "transition_loss"
Q_LOSS = "q_loss"
VAL_LOSS = "val_loss"
ADV_LOSS = "adv_loss"
REWARD_LOSS = "reward_loss"
REWARD_SOFTMAX_GRADS_NORM = "reward_softmax_grads_norm"
REWARD_LOGITS_GRADS_NORM = "reward_logits_grads_norm"
REWARD_MATRIX_GRAD_NORM = "reward_matrix_grad_norm"
TRANSITION_SOFTMAX_GRADS_NORM = "transition_softmax_grads_norm"
TRANSITION_LOGITS_GRADS_NORM = "transition_logits_grads_norm"
TRANSITION_MATRIX_NORMALIZED_GRAD_NORM = "transition_matrix_normalized_grad_norm"
TRANSITION_MATRIX_RAW_GRAD_NORM = "transition_matrix_raw_grad_norm"
PERPLEXITY = "perplexity"
STATE_PERPLEXITY = "state_perplexity"
ACTION_PERPLEXITY = "action_perplexity"
SUMMARY = "summary"
LOGIT_NORMS = "logit_norms"
KL_LOSS = "kl_loss"
ENTROPY_LOSS = "entropy_loss"
MODEL_Q_LOSS = "model_q_loss"
MODEL_TRANSITION_LOSS = "model_transition_loss"
MODEL_REWARD_LOSS = "model_reward_loss"
PRIOR_LOSS = "prior_loss"
COMMIT_LOSS = "commit_loss"
DONES_LOSS = "dones_loss"

STEP_ONE_REWARD_LOSS = "step_one_reward_loss"
STEP_TWO_REWARD_LOSS = "step_two_reward_loss"

STATES = "states"
HAND_STATES = "hand_states"
ACTIONS = "actions"
REWARDS = "rewards"
NEXT_STATES = "next_states"
NEXT_HAND_STATES = "next_hand_states"
DONES = "dones"
NEW_DONES = "new_dones"
STATE_LABELS = "state_labels"
NEXT_STATE_LABELS = "next_state_labels"
ACTION_LABELS = "action_labels"
Q_VALUES = "q_values"
NEXT_Q_VALUES = "next_q_values"
VALUES = "values"
NEXT_VALUES = "next_values"
EMBEDDINGS = "embeddings"
STATE_PROBABILITIES = "state_probabilities"
NEXT_STATE_PROBABILITIES = "next_state_probabilities"
MASKS = "masks"
STATE_ACTION_LABELS = "state_action_labels"

LOG_MARGINAL = "log_marginal"
LOG_PZ1C = "log_pz1c"
LOG_PZ2C = "log_pz2c"
PRIOR_LOG_LIKELIHOOD = "prior_log_likelihood"
DIRICHLET_PRIOR_LOG_LIKELIHOOD = "dirichlet_prior_log_likelihood"
STATE_ENCODER_ENTROPY = "state_encoder_entropy"
ACTION_ENCODER_ENTROPY = "action_encoder_entropy"

GAME_BREAKOUT = "breakout"
