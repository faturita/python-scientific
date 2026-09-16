'''
Generates a bar chart for the given labels and values.

These values are the errors obtained historically in the ImageNet benchmark.
'''

import matplotlib.pyplot as plt
import numpy as np

labels = ["'11", "'12", "'13", "'14", "Human", "'15", "'16"]
values = [26.0, 16.4, 11.7, 7.3, 5.0, 3.6, 3.1]
colors = ['#8db86a', '#4f86c0', '#4f86c0', '#4f86c0', '#c0654f', '#4f86c0', '#4f86c0']

fig, ax = plt.subplots(figsize=(1944/300, 894/300))

bars = ax.bar(labels, values, color=colors, edgecolor='none', width=0.6)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(val), ha='center', va='bottom', fontsize=13)

ax.set_ylim([0, 30])
ax.set_yticks([0, 5, 10, 15, 20, 25, 30])

ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

ax.yaxis.grid(True, color='lightgray', linewidth=0.8)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig('imagenet.png', dpi=150)
plt.show()
