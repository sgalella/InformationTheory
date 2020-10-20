import information_theory.config as config

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_binnings(data):
    N = data.shape[0]

    states_vector = range(1, config.NUM_STATES + 1)

    # Uniform bin width
    width_hist, width_bin = np.histogram(data, bins=config.NUM_STATES)

    # Uniform bin count
    sorted_data = sorted(data)
    total_bins = [sorted_data[idx * N // config.NUM_STATES] if idx != config.NUM_STATES else sorted_data[idx * N // config.NUM_STATES - 1]
                  for idx in range(0, config.NUM_STATES + 1)]

    count_hist, count_bin = np.histogram(data, bins=total_bins)

    fig = plt.figure(figsize=(10, 6))
    outer = gridspec.GridSpec(2, 4, wspace=0.6, hspace=0.4)

    grid0 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0:2], wspace=0.1, hspace=0.6)

    # Uniform widths Bins
    ax00 = plt.Subplot(fig, grid0[0])
    diff_width = np.diff(width_bin)
    for idx in states_vector:
        widths = diff_width[idx - 1]
        starts = width_bin[idx - 1]
        ax00.barh(0, width=widths, left=starts, color=config.COLORS[idx - 1])
    ax00.set_xlim([min(data) - 0.5, max(data) + 0.5])
    ax00.set_ylim([-0.4, 0.4])  # Default height = 0.8
    ax00.set_title('Uniform Width Bins')
    ax00.axis('off')

    # Uniform Counts Bins
    ax01 = plt.Subplot(fig, grid0[1])
    diff_count = np.diff(count_bin)
    for idx in states_vector:
        widths = diff_count[idx - 1]
        starts = count_bin[idx - 1]
        ax01.barh(0, width=widths, left=starts, color=config.COLORS[idx - 1])
    ax01.set_xlim([min(data) - 0.5, max(data) + 0.5])
    ax01.set_ylim([-0.4, 0.4])
    ax01.set_title('Uniform Count Bins')
    ax01.axis('off')

    # Individual observations
    ax02 = plt.Subplot(fig, grid0[2])
    ax02.eventplot(data, colors='k')
    ax02.set_xlim([min(data) - 0.5, max(data) + 0.5])
    ax02.set_ylim([0.5, 1.5])
    ax02.set_title('Individual Observations')
    ax02.axis('off')

    grid2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[4:6], wspace=0.1, hspace=0.5)

    ax2 = plt.Subplot(fig, grid2[0])
    width, bins = np.histogram(data, bins=30)
    centered_bins = 0.5 * (bins[1:] + bins[:-1])
    ax2.plot(centered_bins, width, 'k')
    ax2.set_ylim([0, 2 * max(width)])
    ax2.set_xlabel('Experimental value')
    ax2.set_ylabel('Counts')

    grid1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[2:4], wspace=0.1, hspace=0.5)

    # Uniform bin width plot
    ax1 = plt.Subplot(fig, grid1[0])
    width_hist, width_bin = np.histogram(data, bins=config.NUM_STATES)
    ax1.bar(states_vector, width_hist / N, color=config.COLORS)
    ax1.set_xlabel('State')
    ax1.set_ylabel('Probability')
    ax1.set_title('Uniform Bin Width Discretized Data')
    ax1.set_xticks(states_vector)

    # Uniform bin count plot
    grid3 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[6:], wspace=0.1, hspace=0.5)

    ax3 = plt.Subplot(fig, grid3[0])
    sorted_data = sorted(data)
    total_bins = [sorted_data[idx * N // config.NUM_STATES] if idx != config.NUM_STATES else sorted_data[idx * N // config.NUM_STATES - 1]
                  for idx in range(0, config.NUM_STATES + 1)]

    count_hist, count_bin = np.histogram(data, bins=total_bins)
    ax3.bar(states_vector, count_hist / N, color=config.COLORS)
    ax3.set_xticks(states_vector)
    ax3.set_xlabel('State')
    ax3.set_ylabel('Probability')
    ax3.set_title('Uniform Bin Count Discretized Data')

    xlim = ax2.get_xlim()

    ax00.set_xlim(xlim)
    ax01.set_xlim(xlim)
    ax02.set_xlim(xlim)

    fig.add_subplot(ax00)
    fig.add_subplot(ax01)
    fig.add_subplot(ax02)
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)
    fig.add_subplot(ax3)

    plt.show()


def plot_entropy(probabilities, models_entropy):

    states_vector = range(1, probabilities[0].shape[0] + 1)
    config.NUM_MODELS = 3

    fig = plt.figure(figsize=(20, 4))

    outer = gridspec.GridSpec(1, 4, wspace=0.2, hspace=0.3)

    for i in range(config.NUM_MODELS):
        ax0 = plt.Subplot(fig, outer[i])
        ax0.bar(states_vector, probabilities[i], color=config.COLORS[i])
        ax0.set_ylim([0, 1])
        ax0.set_xlabel('State')
        ax0.set_ylabel('Probability')
        ax0.set_title(f'Model {i+1}:\nProbability Distribution')
        fig.add_subplot(ax0)

    ax1 = plt.Subplot(fig, outer[3])

    ax1.bar(range(1, config.NUM_MODELS + 1), models_entropy, color=config.COLORS)
    ax1.set_xticks(range(1, config.NUM_MODELS + 1))
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Entropy (bits)')
    ax1.set_title('Example Entropy Values')

    fig.add_subplot(ax1)

    plt.show()


def plot_joint_entropy(probabilities, models_joint_entropy):

    xx, yy = np.meshgrid(range(1, 3), range(1, 3))
    x, y = np.ravel(xx), np.ravel(yy)
    x = x - config.WIDTH / 2
    y = y - config.DEPTH / 2
    z = np.zeros_like(x)

    fig = plt.figure(figsize=(20, 4))

    outer = gridspec.GridSpec(1, 5, wspace=0.2, hspace=0.3)

    grid0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0:3], wspace=0.3, hspace=0.6)

    ax0 = plt.Subplot(fig, outer[0:3])
    ax0.axis('off')
    ax0.set_title('Example Joint Entropy Probability Distributions')
    fig.add_subplot(ax0)

    ax00 = fig.add_subplot(grid0[0], projection='3d')
    ax01 = fig.add_subplot(grid0[1], projection='3d')
    ax02 = fig.add_subplot(grid0[2], projection='3d')

    ax00.bar3d(x, y, z, config.WIDTH, config.DEPTH, probabilities[0].flatten(), color=config.COLORS[0], edgecolor='k')
    ax00.set_xticks([1, 2])
    ax00.set_yticks([1, 2])
    ax00.set_xlabel('X State')
    ax00.set_ylabel('Y State')
    ax00.set_zlabel('Probability')
    ax00.set_title('Model 1')

    ax01.bar3d(x, y, z, config.WIDTH, config.DEPTH, probabilities[1].flatten(), color=config.COLORS[1], edgecolor='k')
    ax01.set_xticks([1, 2])
    ax01.set_yticks([1, 2])
    ax01.set_xlabel('X State')
    ax01.set_ylabel('Y State')
    ax01.set_zlabel('Probability')
    ax01.set_title('Model 2')

    ax02.bar3d(x, y, z, config.WIDTH, config.DEPTH, probabilities[2].flatten(), color=config.COLORS[2], edgecolor='k')
    ax02.set_xticks([1, 2])
    ax02.set_yticks([1, 2])
    ax02.set_xlabel('X State')
    ax02.set_ylabel('Y State')
    ax02.set_zlabel('Probability')
    ax02.set_title('Model 3')

    grid1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[3:], wspace=0.1, hspace=0.6)

    ax1 = plt.Subplot(fig, grid1[0], aspect=1.2)
    ax1.bar(range(1, config.NUM_MODELS + 1), models_joint_entropy, color=config.COLORS)
    ax1.set_xticks(range(1, config.NUM_MODELS + 1))
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Joint Entropy (bits)')
    ax1.set_title('Example Joint\nEntropy Values')

    fig.add_subplot(ax1)

    plt.show()


def plot_conditional_entropy(probabilities, models_conditional_entropy):

    xx, yy = np.meshgrid(range(1, 3), range(1, 3))
    x, y = np.ravel(xx), np.ravel(yy)
    x = x - config.WIDTH / 2
    y = y - config.DEPTH / 2
    z = np.zeros_like(x)

    fig = plt.figure(figsize=(20, 4))

    outer = gridspec.GridSpec(1, 5, wspace=0.2, hspace=0.3)

    grid0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0:3], wspace=0.3, hspace=0.6)

    ax0 = plt.Subplot(fig, outer[0:3])
    ax0.axis('off')
    ax0.set_title('Example Conditional Entropy Probability Distributions')
    fig.add_subplot(ax0)

    ax00 = fig.add_subplot(grid0[0], projection='3d')
    ax01 = fig.add_subplot(grid0[1], projection='3d')
    ax02 = fig.add_subplot(grid0[2], projection='3d')

    ax00.bar3d(x, y, z, config.WIDTH, config.DEPTH, probabilities[0].flatten(), color=config.COLORS[0], edgecolor='k')
    ax00.set_xticks([1, 2])
    ax00.set_yticks([1, 2])
    ax00.set_xlabel('X State')
    ax00.set_ylabel('Y State')
    ax00.set_zlabel('Probability')
    ax00.set_title('Model 1')

    ax01.bar3d(x, y, z, config.WIDTH, config.DEPTH, probabilities[1].flatten(), color=config.COLORS[1], edgecolor='k')
    ax01.set_xticks([1, 2])
    ax01.set_yticks([1, 2])
    ax01.set_xlabel('X State')
    ax01.set_ylabel('Y State')
    ax01.set_zlabel('Probability')
    ax01.set_title('Model 2')

    ax02.bar3d(x, y, z, config.WIDTH, config.DEPTH, probabilities[2].flatten(), color=config.COLORS[2], edgecolor='k')
    ax02.set_xticks([1, 2])
    ax02.set_yticks([1, 2])
    ax02.set_xlabel('X State')
    ax02.set_ylabel('Y State')
    ax02.set_zlabel('Probability')
    ax02.set_title('Model 3')

    grid1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[3:], wspace=0.1, hspace=0.6)

    ax1 = plt.Subplot(fig, grid1[0], aspect=2.4)

    ax1.bar(range(1, config.NUM_MODELS + 1), models_conditional_entropy, color=config.COLORS)
    ax1.set_xticks(range(1, config.NUM_MODELS + 1))
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Conditional Entropy (bits)')
    ax1.set_title('Example Conditional\nEntropy Values')

    fig.add_subplot(ax1)

    plt.show()


def plot_mutual_information(probabilities, models_mutual_information):

    xx, yy = np.meshgrid(range(1, 3), range(1, 3))
    x, y = np.ravel(xx), np.ravel(yy)
    x = x - config.WIDTH / 2
    y = y - config.DEPTH / 2
    z = np.zeros_like(x)

    fig = plt.figure(figsize=(20, 4))

    outer = gridspec.GridSpec(1, 5, wspace=0.2, hspace=0.3)

    grid0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0:3], wspace=0.3, hspace=0.6)

    ax0 = plt.Subplot(fig, outer[0:3])
    ax0.axis('off')
    ax0.set_title('Example Mutual Information Probability Distributions')
    fig.add_subplot(ax0)

    ax00 = fig.add_subplot(grid0[0], projection='3d')
    ax01 = fig.add_subplot(grid0[1], projection='3d')
    ax02 = fig.add_subplot(grid0[2], projection='3d')

    ax00.bar3d(x, y, z, config.WIDTH, config.DEPTH, probabilities[0].flatten(), color=config.COLORS[0], edgecolor='k')
    ax00.set_xticks([1, 2])
    ax00.set_yticks([1, 2])
    ax00.set_xlabel('X State')
    ax00.set_ylabel('Y State')
    ax00.set_zlabel('Probability')
    ax00.set_title('Model 1')

    ax01.bar3d(x, y, z, config.WIDTH, config.DEPTH, probabilities[1].flatten(), color=config.COLORS[1], edgecolor='k')
    ax01.set_xticks([1, 2])
    ax01.set_yticks([1, 2])
    ax01.set_xlabel('X State')
    ax01.set_ylabel('Y State')
    ax01.set_zlabel('Probability')
    ax01.set_title('Model 2')

    ax02.bar3d(x, y, z, config.WIDTH, config.DEPTH, probabilities[2].flatten(), color=config.COLORS[2], edgecolor='k')
    ax02.set_xticks([1, 2])
    ax02.set_yticks([1, 2])
    ax02.set_xlabel('X State')
    ax02.set_ylabel('Y State')
    ax02.set_zlabel('Probability')
    ax02.set_title('Model 3')

    grid1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[3:], wspace=0.1, hspace=0.6)

    ax1 = plt.Subplot(fig, grid1[0], aspect=2.4)

    ax1.bar(range(1, config.NUM_MODELS + 1), models_mutual_information, color=config.COLORS)
    ax1.set_xticks(range(1, config.NUM_MODELS + 1))
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Mutual Information (bits)')
    ax1.set_title('Example Mutual\nInformation Values')

    fig.add_subplot(ax1)

    plt.show()


def plot_linear_and_nonlinear(data, models_correlations, models_mutual_information):

    X_linear, Y_linear = data[0]
    X_nonlinear1, Y_nonlinear1 = data[1]
    X_nonlinear2, Y_nonlinear2 = data[2]

    fig = plt.figure(figsize=(20, 4))

    outer = gridspec.GridSpec(1, 4, wspace=0.3, hspace=0.3)

    ax0 = plt.Subplot(fig, outer[0])
    ax0.plot(X_linear, Y_linear, '.', color=config.COLORS[0])
    ax0.set_xticks([0, 0.5, 1])
    ax0.set_xlabel('X Variable')
    ax0.set_ylabel('Y Variable')
    ax0.set_title('Model 1\nLinear Interaction')

    ax1 = plt.Subplot(fig, outer[1])
    ax1.plot(X_nonlinear1, Y_nonlinear1, '.', color=config.COLORS[1])
    ax1.set_xticks([0, 0.5, 1])
    ax1.set_xlabel('X Variable')
    ax1.set_ylabel('Y Variable')
    ax1.set_title('Model 2\nNonlinear Interaction')

    ax2 = plt.Subplot(fig, outer[2])
    ax2.plot(X_nonlinear2, Y_nonlinear2, '.', color=config.COLORS[2])
    ax2.set_xticks([0, 0.5, 1])
    ax2.set_xlabel('X Variable')
    ax2.set_ylabel('Y Variable')
    ax2.set_title('Model 3\nNonlinear Interaction')

    grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[3], wspace=0.3, hspace=0.2)

    ax3 = plt.Subplot(fig, outer[3])
    ax3.axis('off')
    fig.add_subplot(ax3)

    ax30 = fig.add_subplot(grid[0])
    ax31 = fig.add_subplot(grid[1])

    ax30.bar(range(1, config.NUM_MODELS + 1), models_correlations, color=config.COLORS)
    ax30.set_ylim([min(models_correlations), 1])
    ax30.set_xticks(range(1, config.NUM_MODELS + 1))
    ax30.set_ylabel('Correlation')
    ax30.set_title('Correlation and\nMutual Information')

    ax31.bar(range(1, config.NUM_MODELS + 1), models_mutual_information, color=config.COLORS)
    ax31.set_xticks(range(1, config.NUM_MODELS + 1))
    ax31.set_xlabel('Models')
    ax31.set_ylabel('MI (bits)')

    fig.add_subplot(ax0)
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)

    plt.show()


def plot_transfer_entropy(N, spikes, models_transfer_entropy):

    spikes_X1, spikes_Y1 = spikes[0]
    spikes_X2, spikes_Y2 = spikes[1]
    spikes_X3, spikes_Y3 = spikes[2]
    spikes_X4, spikes_Y4 = spikes[3]

    fig = plt.figure(figsize=(15, 4))

    outer = gridspec.GridSpec(4, 5, wspace=0.5, hspace=0.5)

    # First
    grid0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 0:3], wspace=0.3, hspace=0)

    ax00 = plt.Subplot(fig, grid0[0])
    ax00.eventplot(spikes_Y1, color=config.COLORS[0])
    ax00.set_xlim([0, N])
    ax00.set_ylim([0.5, 1.5])
    ax00.set_xticks([])
    ax00.set_yticks([])
    ax00.set_ylabel('Y')
    ax00.set_title('Model 1 Spikes')

    ax01 = plt.Subplot(fig, grid0[1])
    ax01.eventplot(spikes_X1, color=config.COLORS[0])
    ax01.set_xlim([0, N])
    ax01.set_ylim([0.5, 1.5])
    ax01.set_xticks([])
    ax01.set_yticks([])
    ax01.set_ylabel('X')

    fig.add_subplot(ax00)
    fig.add_subplot(ax01)

    # Second
    grid1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1, :3], wspace=0.3, hspace=0)

    ax10 = plt.Subplot(fig, grid1[0])
    ax10.eventplot(spikes_Y2, color=config.COLORS[1])
    ax10.set_xlim([0, N])
    ax10.set_ylim([0.5, 1.5])
    ax10.set_xticks([])
    ax10.set_yticks([])
    ax10.set_ylabel('Y')
    ax10.set_title('Model 2 Spikes')

    ax11 = plt.Subplot(fig, grid1[1])
    ax11.eventplot(spikes_X2, color=config.COLORS[1])
    ax11.set_ylim([0.5, 1.5])
    ax11.set_xlim([0, N])
    ax11.set_xticks([])
    ax11.set_yticks([])
    ax11.set_ylabel('X')

    fig.add_subplot(ax10)
    fig.add_subplot(ax11)

    # Third
    grid2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[2, :3], wspace=0.3, hspace=0)

    ax20 = plt.Subplot(fig, grid2[0])
    ax20.eventplot(spikes_Y3, color=config.COLORS[2])
    ax20.set_ylim([0.5, 1.5])
    ax20.set_xlim([0, N])
    ax20.set_xticks([])
    ax20.set_yticks([])
    ax20.set_ylabel('Y')
    ax20.set_title('Model 3 Spikes')

    ax21 = plt.Subplot(fig, grid2[1])
    ax21.eventplot(spikes_X3, color=config.COLORS[2])
    ax21.set_ylim([0.5, 1.5])
    ax21.set_xlim([0, N])
    ax21.set_xticks([])
    ax21.set_yticks([])
    ax21.set_ylabel('X')

    fig.add_subplot(ax20)
    fig.add_subplot(ax21)

    # Fourth
    grid3 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[3, :3], wspace=0.3, hspace=0)

    ax30 = plt.Subplot(fig, grid3[0])
    ax30.eventplot(spikes_Y4, color=config.COLORS[3])
    ax30.set_xlim([0, N])
    ax30.set_ylim([0.5, 1.5])
    ax30.set_xticks([])
    ax30.set_yticks([])
    ax30.set_ylabel('Y')
    ax30.set_title('Model 4 Spikes')

    ax31 = plt.Subplot(fig, grid3[1])
    ax31.eventplot(spikes_X4, color=config.COLORS[3])
    ax31.set_xlim([0, N])
    ax31.set_ylim([0.5, 1.5])
    ax31.set_yticks([])
    ax31.set_ylabel('X')
    ax31.set_xlabel('Time(bins)')

    # Transfer entropy
    ax4 = plt.Subplot(fig, outer[:, 3:])
    ax4.bar(range(1, config.NUM_SPIKE_MODELS + 1), models_transfer_entropy, color=config.COLORS)
    ax4.set_xlabel('Model')
    ax4.set_title('Entropy Transfer\n' + r'Entropy(X $\rightarrow$ Y) Values')

    fig.add_subplot(ax30)
    fig.add_subplot(ax31)
    fig.add_subplot(ax4)

    plt.show()


def plot_biases(N, entropy_true, entropy_sampled, mutual_information_true, mutual_information_sampled):

    entropy_low_true, entropy_high_true = entropy_true
    entropy_low_sampled, entropy_high_sampled = entropy_sampled
    mutual_information_low_true, mutual_information_high_true = mutual_information_true
    mutual_information_low_sampled, mutual_information_high_sampled = mutual_information_sampled

    f, ax = plt.subplots(1, 2, figsize=(14, 4))

    ax[0].plot(N, entropy_low_sampled.mean(axis=0), color=config.COLORS[0])
    ax[0].plot(N, entropy_high_sampled.mean(axis=0), color=config.COLORS[1])
    ax[0].plot(N, len(N) * [entropy_low_true], color=config.COLORS[0], linestyle='--')
    ax[0].plot(N, len(N) * [entropy_high_true], color=config.COLORS[1], linestyle='--')
    ax[0].fill_between(N, entropy_low_sampled.min(axis=0), entropy_low_sampled.max(axis=0), color=config.COLORS[0], alpha=0.2)
    ax[0].fill_between(N, np.quantile(entropy_low_sampled, 0.25, axis=0),
                       np.quantile(entropy_low_sampled, 0.75, axis=0), color=config.COLORS[0], alpha=0.2)
    ax[0].fill_between(N, entropy_high_sampled.min(axis=0), entropy_high_sampled.max(axis=0), color=config.COLORS[1], alpha=0.2)
    ax[0].fill_between(N, np.quantile(entropy_high_sampled, 0.25, axis=0),
                       np.quantile(entropy_high_sampled, 0.75, axis=0), color=config.COLORS[1], alpha=0.2)

    ax[0].set_xscale('log')
    ax[0].set_xlim([10, 1e3])
    ax[0].set_ylim([-0.2, 2.2])
    ax[0].set_xlabel('Number of Observations')
    ax[0].set_ylabel('Entropy (bits)')
    ax[0].set_title('Entropy Bias')
    ax[0].legend(['Model 1 (Low Ent)', 'Model 2 (High Ent)', 'Model 1 (True Value)', 'Model 2 (True Value)'])

    ax[1].plot(N, mutual_information_low_sampled.mean(axis=0), color=config.COLORS[2])
    ax[1].plot(N, mutual_information_high_sampled.mean(axis=0), color=config.COLORS[3])
    ax[1].plot(N, len(N) * [mutual_information_low_true], color=config.COLORS[2], linestyle='--')
    ax[1].plot(N, len(N) * [mutual_information_high_true], color=config.COLORS[3], linestyle='--')
    ax[1].fill_between(N, mutual_information_low_sampled.min(axis=0), mutual_information_low_sampled.max(axis=0), color=config.COLORS[2], alpha=0.2)
    ax[1].fill_between(N, np.quantile(mutual_information_low_sampled, 0.25, axis=0),
                       np.quantile(mutual_information_low_sampled, 0.75, axis=0), color=config.COLORS[2], alpha=0.2)
    ax[1].fill_between(N, mutual_information_high_sampled.min(axis=0), mutual_information_high_sampled.max(axis=0), color=config.COLORS[3], alpha=0.2)
    ax[1].fill_between(N, np.quantile(mutual_information_high_sampled, 0.25, axis=0),
                       np.quantile(mutual_information_high_sampled, 0.75, axis=0), color=config.COLORS[3], alpha=0.2)

    ax[1].set_xscale('log')
    ax[1].set_xlim([10, 1e3])
    ax[1].set_ylim([-0.1, 1.10])
    ax[1].set_xlabel('Number of Observations')
    ax[1].set_ylabel('Mutual Information (bits)')
    ax[1].set_title('Mutual Information Bias')
    ax[1].legend(['Model 1 (Low MI)', 'Model 2 (High MI)', 'Model 1 (True Value)', 'Model 2 (True Value)'])

    plt.show()
