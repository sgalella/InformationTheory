import information_theory.config as config
from information_theory.metrics import entropy, mutual_information_from_table

from collections import Counter

import numpy as np
from tqdm import tqdm


def entropy_bias(probabilities, N, num_simulations):
    """
    Computes the entropy biases sampling from probabilities.

    Args:
        probabilities (list): Probabilities for each model.
        N (int): Number of observations.
        num_simulations (int): Number of simulations per each observation.

    Returns:
        list: Entropy for each observation.
    """
    entropy_low_sampled = np.zeros((num_simulations, len(N)))
    entropy_high_sampled = np.zeros((num_simulations, len(N)))

    p_low_true, p_high_true = probabilities

    for j in tqdm(range(num_simulations)):
        for i, n in enumerate(N):
            samples_low = np.random.choice(config.STATES, size=(n,), p=p_low_true)
            samples_high = np.random.choice(config.STATES, size=(n,), p=p_high_true)

            hist_low, _ = np.histogram(samples_low, bins=config.NUM_STATES)
            hist_high, _ = np.histogram(samples_high, bins=config.NUM_STATES)

            p_low_sampled = hist_low / n
            p_high_sampled = hist_high / n

            entropy_low_sampled[j][i] = entropy(p_low_sampled)
            entropy_high_sampled[j][i] = entropy(p_high_sampled)

    return entropy_low_sampled, entropy_high_sampled


def mutual_information_bias(probabilities, N, num_simulations):
    """
    Computes the mutual information bias sampling from probabilities.
    
    Args:
        probabilities (list): Probabilities for each model.
        N (int): Number of observations.
        num_simulations (int): Number of simulations per each observation.

    Returns:
        list: Mutual information for each observation.
    """
    mutual_information_low_sampled = np.zeros((num_simulations, len(N)))
    mutual_information_high_sampled = np.zeros((num_simulations, len(N)))

    p_low_true, p_high_true = probabilities

    for j in tqdm(range(num_simulations)):
        for i, n in enumerate(N):

            samples_low = np.random.choice(config.STATES, size=(n,), p=p_low_true)
            samples_high = np.random.choice(config.STATES, size=(n,), p=p_high_true)

            count_low = Counter(samples_low)
            count_high = Counter(samples_high)

            p_dist_low = np.zeros((4, ))
            p_dist_high = np.zeros((4, ))

            for state in config.STATES:
                p_dist_low[state] = count_low[state] / n
                p_dist_high[state] = count_high[state] / n

            mutual_information_low_sampled[j][i] = mutual_information_from_table(np.reshape(p_dist_low, (2, 2)))
            mutual_information_high_sampled[j][i] = mutual_information_from_table(np.reshape(p_dist_high, (2, 2)))

    return mutual_information_low_sampled, mutual_information_high_sampled
