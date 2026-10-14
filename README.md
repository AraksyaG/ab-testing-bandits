# A/B Testing with Epsilon-Greedy and Thompson Sampling

Homework 2 – finding the best of 4 ads using two bandit algorithms.

## How to run

```bash
pip install -r requirements.txt
python Bandit.py
```

The program runs both experiments and then opens three plot windows one after another
(close each one to continue). The plots are also saved as PNG files, and every trial of
both algorithms is stored in `bandit_rewards.csv` with the columns `Bandit, Reward, Algorithm`.

## Experiment design

There are 4 ads with true mean rewards of 1, 2, 3 and 4, so ad 4 is the best one. Each
time an ad is shown, the reward is drawn from a normal distribution around its mean with
variance 1. This means the algorithms never see the true means directly. They only see
noisy rewards and have to figure out which ad is best from those. Each algorithm runs
for 20,000 trials, and I used a fixed seed (42) so the results are the same every time.
Regret for a trial is the difference between the best mean and the mean of the ad that
was chosen. It is 0 when ad 4 is shown, and 1, 2 or 3 otherwise.

For Epsilon-Greedy, epsilon decays as 1/t, as required in the homework. At each trial
it shows a random ad with probability epsilon and otherwise shows the ad with the
highest average reward so far. This means it explores all the time at the start but
almost never after the first few hundred trials.

For Thompson Sampling, I assumed the rewards are normal with known precision tau = 1,
which is the real noise level of the data, and put a normal prior N(0, 1) on each ad's
mean. After every reward the posterior of that ad is updated. Then, at each trial,
the algorithm draws one sample from every ad's posterior and shows the ad with the highest
sample. Ads it is still unsure about have wide posteriors and occasionally win, so it
explores without needing an epsilon.

## Results

| Algorithm | Cumulative reward | Cumulative regret | Times each ad shown (1 / 2 / 3 / 4) |
|---|---|---|---|
| Epsilon-Greedy | 73,907 | 6,181 | 2 / 3 / 6,169 / 13,826 |
| Thompson Sampling | 79,967 | 7 | 1 / 1 / 2 / 19,996 |

Epsilon-Greedy performed much worse because it got stuck on ad 3. Looking at the CSV,
ad 4 was not shown even once until trial 6,175. All estimates start at 0, so an ad that
has never been shown looks worse than any ad that has, and the greedy step never picks
it. The only way to try ad 4 was through exploration, and with epsilon = 1/t that
happens only about 10 times in 20,000 trials. When ad 4 was finally shown (reward 5.8),
its estimate went above ad 3's and the algorithm switched for good. From then on the
average reward climbs towards 4, but it only reaches 3.70 because the first 6,000
trials pull it down.

Thompson Sampling found ad 4 within the first ten or so trials and stayed on it for the
rest of the experiment. It chose a worse ad only four times in total. The small dip
visible on the log scale plot is not caused by wrong choices, just by the noise in the
rewards when the average is taken over only a few trials. Interestingly, its estimates
for ads 1–3 are quite inaccurate, because it tried each of them only once or twice. But
that is fine, because the goal is to find and show the best ad, not to measure every ad perfectly.

The comparison plot shows the difference clearly. Epsilon-Greedy's cumulative regret
rises in a straight line until about trial 6,000 and then becomes flat, while Thompson
Sampling's regret stays close to zero the whole time. Overall, Thompson Sampling is the
better choice for this experiment, because it explores based on how uncertain it is
rather than on a fixed schedule.

All classes and methods are documented with reST-style docstrings (the format pyment
generates), and all output goes through `loguru` instead of `print()`.

## Bonus: how I would improve the implementation

The biggest problem I noticed is that epsilon = 1/t explores too little. The sum of 1/t
from 1 to 20,000 is only about 10.5, which means that over the whole experiment the
algorithm explores roughly 10 times. That explains why it stayed on ad 3 for so long.
A minimum value such as `max(0.01, 1/t)`, or a slower decay like `5/t`, would keep it
checking the other ads once in a while. 

Another simple fix is to show every ad once before the main loop (or start all
estimates at a high value), so that no ad is left untried just because of how the
estimates were initialized.

I would also not trust a single run. The Epsilon-Greedy result depends a lot on luck in
the first few trials, and with a different seed it might find ad 4 immediately and look
almost as good as Thompson Sampling. Running both algorithms around 100 times with
different seeds and comparing the average regret would give a much fairer picture.

For Thompson Sampling, assuming known precision works here only because I generated the
data myself, so I know the variance is 1. With real ads this would not be known, and the
results are usually click or no click, so a Beta-Bernoulli version of Thompson Sampling
would be more realistic for an actual A/B test.

Finally, the `report()` method is identical in both classes. I left it like that because I wanted 
to keep the template’s structure, but the assignment allows adding to Bandit, so 
it could be implemented there once.