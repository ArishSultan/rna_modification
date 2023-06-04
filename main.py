import matplotlib.pyplot as plt
import matplotlib as mpl

# Disable the use of matplotlib's font settings in the PGF output
mpl.rcParams['pgf.rcfonts'] = False

# Generate your plot using matplotlib
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])

# Save the plot as a PGF file
plt.savefig('plot.pgf', format='pgf')