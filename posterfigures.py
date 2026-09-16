import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# Example data list (replace with your values)
data = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,5,5,10,10]

#data = [2,2,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5]

#data = [19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,20,20,20,20,20]
# Set the style and context for a poster
sns.set_theme(style="whitegrid")
sns.set_context("talk", font_scale=1.8)

# Create the plot
plt.figure(figsize=(12, 8))
bins = np.arange(-2, 24 + 1, 1)
sns.histplot(data, bins=bins, kde=False, color="skyblue", edgecolor="black")

# Customize axis limits
plt.xlim(-2, 23)
plt.ylim(0, 35)

# Fixed-width bars


# Customize x-ticks to include sample index and seconds
xticks = np.arange(0, 24, 2)
xtick_labels = [f"{x}\n{x*4}ms" for x in xticks]
plt.xticks(ticks=xticks, labels=xtick_labels, fontsize=18)

# Customize y-ticks
plt.yticks(fontsize=18)

# Axis labels
plt.xlabel("Sample Index (and Time)", fontsize=22, weight='bold')
plt.ylabel("Frequency", fontsize=22, weight='bold')

# Title (optional)
#plt.title("Latency between Triggering and Actual Stimulation", fontsize=24, weight='bold')

# Display
plt.tight_layout()
plt.show()