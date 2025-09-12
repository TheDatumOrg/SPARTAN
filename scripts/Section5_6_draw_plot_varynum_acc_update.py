import os
import matplotlib
import matplotlib.pyplot as plt


font = {'size': 22}

matplotlib.rc('font', **font)

# Fill in the test results
num_series = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6]
accuracy_sax = [0.93, 0.936667, 0.935556, 0.935556, 0.935556, 0.935556,0.935556]
accuracy_sfa = [0.985556,0.991111,0.992222,0.992222,0.991111,0.991111,0.991111]
accuracy_spartan = [0.986667, 0.9933, 0.9889, 0.9933, 0.9944, 0.9967,0.9967]
accuracy_spartan_random = [0.986667, 0.9933, 0.9889, 0.9933, 0.9944, 0.9967,0.9967]
accuracy_spartan_sampling = [0.9753, 0.9898, 0.99, 0.9951, 0.9924, 0.9917, 0.994]

plt.figure(figsize=(17, 5))
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
plt.xlabel('Number of Time series', fontsize=30)
plt.ylabel('Accuracy', fontsize=30)
# plt.title('Accuracy vs Number of Time series')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.19), ncol=5)
ymin=0.92
ymax=1.0
plt.ylim([ymin, ymax])

os.makedirs('./result/Section5_6', exist_ok=True)

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks([1e3, 1e4, 1e5, 1e6], [r'$10^3$',  r'$10^4$', r'$10^5$', r'$10^6$'])
plt.savefig("./result/Section5_6/varyingnum_acc.jpg", dpi=800, bbox_inches='tight')

plt.show()