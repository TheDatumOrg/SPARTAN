import os
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 22}

matplotlib.rc('font', **font)

# Fill in the test results
num_series = [128, 640, 1280, 6400, 12800, 64000, 128000]
accuracy_sax = [0.00008,0.00021,0.00054,0.00391,0.00746,0.0361,0.072]
accuracy_sfa = [0.000975,0.00645,0.0148,0.108,0.240,1.37,2.88]
accuracy_spartan = [0.00976,0.128,0.367,1.46,2.80,14.2,32.2]
accuracy_spartan_random = [0.0049,0.021,0.04,0.25,0.52,2.68,5.35]
accuracy_spartan_sampling = [0.00146,0.00396,0.00756,0.0475,0.0977,0.493,0.9]

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

plt.xlim([128, 128000*3])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time-series length', fontsize=30)
plt.ylabel('Runtime (s)', fontsize=30)
# plt.title('Accuracy vs Number of Time series')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.19), ncol=5)
ymin=5e-5
ymax=50
plt.ylim([ymin, ymax])

os.makedirs('./result/Section5_6', exist_ok=True)

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.xticks(num_series, ['1.E+03', '5.E+03', '1.E+04', '5.E+04', '1.E+05', '5.E+05', '1.E+06'])
plt.savefig("./result/Section5_6/varyinglen_time.jpg", dpi=800, bbox_inches='tight')

plt.show()