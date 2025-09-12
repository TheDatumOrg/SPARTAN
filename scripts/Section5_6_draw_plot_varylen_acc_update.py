import os
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 22}

matplotlib.rc('font', **font)

# Fill in the test results
num_series = [128, 640, 1280, 6400, 12800, 64000, 128000]
accuracy_sax = [0.9, 0.78, 0.59, 0.59, 0.22, 0.22, 0.34]
accuracy_sfa = [0.99, 0.30, 0.32, 0.35, 0.35, 0.34, 0.34]
accuracy_spartan = [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
accuracy_spartan_random = [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
accuracy_spartan_sampling = [0.98, 0.982, 0.97, 0.976, 0.97, 0.97, 0.976]

plt.figure(figsize=(17, 5))
# linewidth=3.5
# markersize=10.5
# markeredgewidth=3

linewidth=7
markersize=14
markeredgewidth=6


# '#EAC84E', '#8CC8C5', '#5F7592'
plt.plot(num_series, accuracy_sax, 'x-.', label='SAX', color='rosybrown', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
plt.plot(num_series, accuracy_sfa, 'x', linestyle='dotted', label='SFA', color='orange', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
plt.plot(num_series, accuracy_spartan, '^-', label='SPARTAN', color='#5F7592', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
plt.plot(num_series, accuracy_spartan_random, '^--', label='SPARTAN-R', color='#2F98CC', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
plt.plot(num_series, accuracy_spartan_sampling, '^--', label='SPARTAN-S', color='#0066cc', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)

plt.xscale('log')
plt.xlabel('Time-series length', fontsize=30)
plt.ylabel('Accuracy', fontsize=30)
# plt.title('Accuracy vs Number of Time series')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.19), ncol=5)
ymin=0.0
ymax=1.05
plt.ylim([ymin, ymax])

os.makedirs('./result/Section5_6', exist_ok=True)

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.xticks(num_series, ['1.E+03', '5.E+03', '1.E+04', '5.E+04', '1.E+05', '5.E+05', '1.E+06'])
plt.savefig("./result/Section5_6/varyinglen_acc.jpg", dpi=800, bbox_inches='tight')

plt.show()