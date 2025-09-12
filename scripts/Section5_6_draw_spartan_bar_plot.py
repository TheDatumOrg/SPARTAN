import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 15})

# Draw bar plot, comparing SPARTAN family with SAX and SFA

# Fill in classification result on 128 UCR Datasets
alphas = ['SPARTAN', 'SPARTAN-R', 'SPARTAN-S', 'SFA', 'SAX']
acc_list = [0.63, 0.63, 0.62, 0.60, 0.51]
hatches = [ '||', '||', '||', '//','--']
colors = ['#5F7592', '#5F7592', '#5F7592', '#8CC8C5', '#EAC84E']

# Reversing the order to match the vertical plotting order in the original image
alphas.reverse()
acc_list.reverse()
hatches.reverse()
colors.reverse()

y = np.arange(len(alphas))  # the label locations

fig, ax = plt.subplots(figsize=(12, 3))

bar_width = 0.8
# Plotting the horizontal bars with hatches

ax.barh(y, acc_list, height=bar_width, color=colors, hatch=hatches)

# Adding text for labels, title, and custom y-axis tick labels, etc.
ax.set_xlabel('Accuracy',fontsize=25)
ax.set_yticks(y)
ax.set_yticklabels(alphas,fontsize=25)
plt.tick_params(axis='x', labelsize=22)
xmin, xmax = 0.5, 0.64
ax.set_xlim([xmin, xmax])

ax.grid(color='gray', linestyle='dashed', axis='x')
# fig.tight_layout()
output_dir = 'result/Section5_6'
os.makedirs(output_dir, exist_ok=True)

plt.savefig(f"{output_dir}/acc-sampling-spartan-s.jpg", dpi=1000, bbox_inches='tight')
plt.show()
