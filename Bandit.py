"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

Bandit_Reward = [1, 2, 3, 4]
NumberOfTrials = 20000
CSV_FILE = "bandit_rewards.csv"


class Bandit(ABC):
    """Abstract base class for a multi-armed bandit algorithm."""
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        """Show one ad and return the observed reward."""
        pass

    @abstractmethod
    def update(self):
        """Update the algorithm's belief about an ad after a reward."""
        pass

    @abstractmethod
    def experiment(self):
        """Run the full experiment over all trials."""
        pass

    @abstractmethod
    def report(self):
        """Store results in a CSV and log reward and regret."""
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization():
    """Plots for the learning process and the comparison of the two algorithms.

    :param eg: Epsilon-Greedy object that has already run ``experiment()``.
    :type eg: EpsilonGreedy
    :param ts: Thompson Sampling object that has already run ``experiment()``.
    :type ts: ThompsonSampling
    """

    def __init__(self, eg, ts):
        self.eg = eg
        self.ts = ts

    def plot1(self):
        """Plot the running average reward of each algorithm on linear and log scales."""
        # Visualize the performance of each bandit: linear and log
        for bandit in (self.eg, self.ts):
            rewards = np.array(bandit.rewards)
            trials = np.arange(1, len(rewards) + 1)
            running_avg = np.cumsum(rewards) / trials

            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            for ax, scale in zip(axes, ["linear", "log"]):
                ax.plot(trials, running_avg, label="Running average reward")
                ax.axhline(max(bandit.p), color="gray", linestyle="--", label="Best ad mean")
                ax.set_xscale(scale)
                ax.set_title(f"{bandit.name}: learning process ({scale} scale)")
                ax.set_xlabel("Trial")
                ax.set_ylabel("Average reward")
                ax.legend()
            fig.tight_layout()

            filename = f"plot1_{bandit.name.replace(' ', '_').replace('-', '_')}.png"
            fig.savefig(filename)
            logger.info(f"Saved {filename}")
            plt.show()

    def plot2(self):
        """Plot cumulative reward and cumulative regret of both algorithms."""
        # Compare E-greedy and thompson sampling cummulative rewards
        # Compare E-greedy and thompson sampling cummulative regrets
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for algo in (self.eg, self.ts):
            axes[0].plot(np.cumsum(algo.rewards), label=algo.name)
            axes[1].plot(np.cumsum(algo.regrets), label=algo.name)

        axes[0].set_title("Cumulative reward")
        axes[1].set_title("Cumulative regret")
        for ax in axes:
            ax.set_xlabel("Trial")
            ax.legend()
        fig.tight_layout()

        fig.savefig("plot2_comparison.png")
        logger.info("Saved plot2_comparison.png")
        plt.show()

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """Epsilon-Greedy with epsilon decaying as 1/t.

    At trial t, explore (random ad) with probability 1/t, otherwise exploit
    the ad with the highest estimated mean reward.
    """

    def __init__(self, p, n_trials=NumberOfTrials, seed=42):
        """
        :param p: true mean reward of each ad, e.g. [1, 2, 3, 4].
        :type p: list
        :param n_trials: number of trials.
        :type n_trials: int
        :param seed: random seed for reproducibility.
        :type seed: int
        """
        self.name = "Epsilon-Greedy"
        self.p = list(p)
        self.n_trials = n_trials
        self.rng = np.random.default_rng(seed)
        self.k = len(self.p)
        self.estimates = np.zeros(self.k)          # estimated mean of each ad
        self.counts = np.zeros(self.k, dtype=int)  # times each ad was shown
        self.chosen, self.rewards, self.regrets = [], [], []

    def __repr__(self):
        """
        :return: description of the algorithm and its current estimates.
        :rtype: str
        """
        return f"EpsilonGreedy(p={self.p}, estimates={np.round(self.estimates, 3).tolist()})"

    def pull(self, arm):
        """Show ad ``arm`` once and return a noisy reward from N(p[arm], 1).

        :param arm: index of the ad.
        :type arm: int
        :return: observed reward.
        :rtype: float
        """
        return self.rng.normal(self.p[arm], 1.0)

    def update(self, arm, reward):
        """Update the running-mean estimate of ad ``arm``.

        :param arm: index of the ad that was shown.
        :type arm: int
        :param reward: observed reward.
        :type reward: float
        """
        self.counts[arm] += 1
        n = self.counts[arm]
        self.estimates[arm] = ((n - 1) * self.estimates[arm] + reward) / n

    def experiment(self):
        """Run ``n_trials`` trials with epsilon = 1/t.

        :return: reward obtained at each trial.
        :rtype: list
        """
        best = max(self.p)
        for t in range(1, self.n_trials + 1):
            epsilon = 1 / t
            if self.rng.random() < epsilon:
                arm = int(self.rng.integers(self.k))   # explore
            else:
                arm = int(np.argmax(self.estimates))   # exploit

            reward = self.pull(arm)
            self.update(arm, reward)

            self.chosen.append(arm)
            self.rewards.append(reward)
            self.regrets.append(best - self.p[arm])

        logger.debug(f"Finished: {self}")
        return self.rewards

    def report(self):
        """Store the trials in the CSV file and log reward and regret."""
        # store data in csv
        write_header = not os.path.exists(CSV_FILE)
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Bandit", "Reward", "Algorithm"])
            for arm, reward in zip(self.chosen, self.rewards):
                writer.writerow([arm + 1, reward, self.name])

        # print average reward / regret (with the logger)
        n = len(self.rewards)
        logger.info(f"[{self.name}] Cumulative reward: {sum(self.rewards):,.2f}")
        logger.info(f"[{self.name}] Average reward: {sum(self.rewards) / n:.4f}")
        logger.info(f"[{self.name}] Cumulative regret: {sum(self.regrets):,.2f}")
        logger.info(f"[{self.name}] Average regret: {sum(self.regrets) / n:.4f}")
        counts = {f"Ad {i + 1}": int(c) for i, c in enumerate(self.counts)}
        logger.info(f"[{self.name}] Times each ad was shown: {counts}")

#--------------------------------------#

class ThompsonSampling(Bandit):
    """Thompson Sampling for Gaussian rewards with known precision tau.

    Reward of ad k ~ N(mu_k, 1/tau), prior mu_k ~ N(0, 1/lambda_0).
    After n rewards with sum S the posterior is N(m_k, 1/lambda_k) with
    lambda_k = lambda_0 + n*tau and m_k = tau*S / lambda_k.
    """

    def __init__(self, p, n_trials=NumberOfTrials, tau=1.0, prior_precision=1.0, seed=42):
        """
        :param p: true mean reward of each ad, e.g. [1, 2, 3, 4].
        :type p: list
        :param n_trials: number of trials.
        :type n_trials: int
        :param tau: known precision (1/variance) of the rewards.
        :type tau: float
        :param prior_precision: precision lambda_0 of the prior on each mean.
        :type prior_precision: float
        :param seed: random seed for reproducibility.
        :type seed: int
        """
        self.name = "Thompson Sampling"
        self.p = list(p)
        self.n_trials = n_trials
        self.tau = tau
        self.rng = np.random.default_rng(seed)
        self.k = len(self.p)
        self.lambdas = np.full(self.k, prior_precision, dtype=float)  # posterior precision
        self.means = np.zeros(self.k)                                 # posterior mean
        self.sum_x = np.zeros(self.k)                                 # sum of rewards
        self.counts = np.zeros(self.k, dtype=int)
        self.chosen, self.rewards, self.regrets = [], [], []

    def __repr__(self):
        """
        :return: description of the algorithm and its posterior means.
        :rtype: str
        """
        return (f"ThompsonSampling(p={self.p}, tau={self.tau}, "
                f"posterior_means={np.round(self.means, 3).tolist()})")

    def pull(self, arm):
        """Show ad ``arm`` once and return a noisy reward from N(p[arm], 1/tau).

        :param arm: index of the ad.
        :type arm: int
        :return: observed reward.
        :rtype: float
        """
        return self.rng.normal(self.p[arm], 1 / np.sqrt(self.tau))

    def update(self, arm, reward):
        """Update the posterior of ad ``arm`` with a new reward.

        :param arm: index of the ad that was shown.
        :type arm: int
        :param reward: observed reward.
        :type reward: float
        """
        self.counts[arm] += 1
        self.sum_x[arm] += reward
        self.lambdas[arm] += self.tau
        self.means[arm] = self.tau * self.sum_x[arm] / self.lambdas[arm]

    def experiment(self):
        """Run ``n_trials`` trials: sample a mean for each ad from its posterior
        and show the ad with the largest sample.

        :return: reward obtained at each trial.
        :rtype: list
        """
        best = max(self.p)
        for _ in range(self.n_trials):
            samples = self.rng.normal(self.means, 1 / np.sqrt(self.lambdas))
            arm = int(np.argmax(samples))

            reward = self.pull(arm)
            self.update(arm, reward)

            self.chosen.append(arm)
            self.rewards.append(reward)
            self.regrets.append(best - self.p[arm])

        logger.debug(f"Finished: {self}")
        return self.rewards

    def report(self):
        """Store the trials in the CSV file and log reward and regret."""
        # store data in csv
        write_header = not os.path.exists(CSV_FILE)
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Bandit", "Reward", "Algorithm"])
            for arm, reward in zip(self.chosen, self.rewards):
                writer.writerow([arm + 1, reward, self.name])

        # print average reward / regret (with the logger)
        n = len(self.rewards)
        logger.info(f"[{self.name}] Cumulative reward: {sum(self.rewards):,.2f}")
        logger.info(f"[{self.name}] Average reward: {sum(self.rewards) / n:.4f}")
        logger.info(f"[{self.name}] Cumulative regret: {sum(self.regrets):,.2f}")
        logger.info(f"[{self.name}] Average regret: {sum(self.regrets) / n:.4f}")
        counts = {f"Ad {i + 1}": int(c) for i, c in enumerate(self.counts)}
        logger.info(f"[{self.name}] Times each ad was shown: {counts}")




def comparison():
    """Run both algorithms, report them, and compare them visually."""
    # think of a way to compare the performances of the two algorithms VISUALLY and
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)  # start a fresh CSV on every run

    eg = EpsilonGreedy(Bandit_Reward)
    ts = ThompsonSampling(Bandit_Reward)

    for algo in (eg, ts):
        logger.info(f"Running {algo.name} for {algo.n_trials} trials")
        algo.experiment()
        algo.report()

    viz = Visualization(eg, ts)
    viz.plot1()
    viz.plot2()

    if sum(ts.regrets) < sum(eg.regrets):
        logger.info("Thompson Sampling has the lower cumulative regret")
    else:
        logger.info("Epsilon-Greedy has the lower cumulative regret")

if __name__=='__main__':

    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

    comparison()