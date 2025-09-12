import os
import matplotlib
import matplotlib.pyplot as plt


font = {'size': 22}

matplotlib.rc('font', **font)

# Fill in the test results
num_series = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6]
accuracy_sax = [0.00009,0.00022,0.00037,0.00239,0.00475,0.02706,0.04668]
accuracy_sfa = [0.0015,0.00334,0.00661,0.0394,0.0788,0.391,0.802]
accuracy_spartan = [0.00998,0.0443,0.0781,0.529,1.26,7.67,15.6]
accuracy_spartan_random = [0.00514,0.021,0.04,0.28,0.59,3.26,6.91]
accuracy_spartan_sampling = [0.00146,0.00304,0.00472,0.0218,0.0384,0.166,0.325]

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
plt.yscale('log')
plt.xlim([1e3, 3e6])
plt.xlabel('Number of Time series', fontsize=30)
plt.ylabel('Runtime (s)', fontsize=30)
# plt.title('Accuracy vs Number of Time series')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.19), ncol=5)
ymin=1e-4
ymax=50.0
plt.ylim([ymin, ymax])

os.makedirs('./result/Section5_6', exist_ok=True)

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks([1e3, 1e4, 1e5, 1e6], [r'$10^3$',  r'$10^4$', r'$10^5$', r'$10^6$'])
plt.savefig("./result/Section5_6/varyingnum_time.jpg", dpi=800, bbox_inches='tight')

plt.show()