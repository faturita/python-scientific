'''

https://stats.stackexchange.com/questions/207671/normal-approximation-to-the-binomial-distribution-why-np5

'''


import matplotlib.pyplot as plt
import numpy as np
np.random.seed(20190915)


def make_hists(axs, n):
    proportions = np.linspace(0.01, 0.19, len(axs))
    for i, prop in enumerate(proportions):
        # Draw n samples 10,000 times
        x = np.random.rand(n, 10_000) < prop
        means = x.mean(axis=0)
        axs[i].hist(means, bins=np.linspace(0, 0.5, n//2))
        axs[i].set_xlim([0, 0.5])
        axs[i].set_yticklabels([])
        ylim_mean = np.mean(axs[i].get_ylim())
        axs[i].text(-0.08, ylim_mean * 3/2, f'$p={prop:.2f}$', va='center')
        axs[i].text(-0.08, ylim_mean * 2/3, f'$np={n * prop:.1f}$', va='center')
    axs[0].set_title(f'$n={n}$')

def main():
    f, axs = plt.subplots(10, 2, sharex=True, figsize=(12, 8))
    make_hists(axs[:, 0], 50)
    make_hists(axs[:, 1], 250)
    f.suptitle(
        'Histograms of 10,000 sample proportions, varying $p$ and $n$',
        fontsize=14
    )
    plt.show()

main()