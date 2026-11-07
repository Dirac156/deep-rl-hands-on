import torch
from torch.utils.tensorboard import SummaryWriter


"""Shared hyper-parameters and helpers for the CartPole REINFORCE notebook."""

REWARD_STEPS = 10         # unused: training uses steps_count=1 and full-episode returns
GAMMA = 0.99              # discount factor
LEARNING_RATE = 0.01      # Adam step size
EPISODES_TO_TRAIN = 4     # complete episodes collected per gradient step

# Best available accelerator: CUDA, then Apple MPS, then CPU
device =torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# TensorBoard logger; view with: tensorboard --logdir runs
writer = SummaryWriter()

def compte_qvals(rewards: list[float], gamma: float) -> list[float]:
    """Discounted return G_t for every step of one finished episode.

    Args:
        rewards: per-step rewards [r_0, ..., r_T].
        gamma: discount factor in [0, 1].

    Returns:
        [G_0, ..., G_T] where G_t = r_t + gamma * G_{t+1}.
    """
    sum_r = 0.0
    discounted_sums = []
    
    for r in reversed(rewards):
        sum_r = r + gamma * sum_r
        discounted_sums.append(sum_r)
    
    # Reverse back so step 0 is first: [G_0, G_1, ..., G_T]
    discounted_sums.reverse()
    return discounted_sums


