import os
import matplotlib.pyplot as plt
import numpy as np
from mymodule import rmsd  # Importing the rmsd function from file

# Function for data approximation and visualization
def approximate_and_plot(file_name):
    data = np.loadtxt(file_name, delimiter='\t', dtype=float)
    x = np.arange(0, len(data), 1)

    fig, ax = plt.subplots()
    ax.plot(x, data, 'o', label='Data Points')

    colors = ['C1', 'C2', 'C3']  # Colors for polynomials of different degrees
    labels = ['Approx 1', 'Approx 2', 'Approx 3']  # Signatures for polynomials of different degrees

    # Approximation by polynomials of 1, 2, and 3 degrees
    for degree in range(1, 4):
        p = np.polyfit(x, data, degree)
        yp = np.polyval(p, x)
        ax.plot(x, yp, color=colors[degree-1], label=f'{labels[degree-1]} (RMSD: {rmsd(p, x, data):.2f})')

    ax.legend(loc='best')
    ax.set_title(f'Data approximation for \'{file_name}\'')
    ax.set_ylabel('Y', rotation = 0)
    ax.set_xlabel('X')
    ax.grid()

    # Check if the 'plots' directory exists and create it if it doesn't
    plots_dir = 'plots'
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

    # Modify the file name to include the 'plots' directory
    png_file_name = os.path.join(plots_dir, os.path.basename(file_name).replace('.txt', '.png'))
    
    plt.savefig(png_file_name)
    plt.close()  # Close the plot without displaying

# Going through 16 files
for i in range(1, 17):
    file_name = f'data_file_work2/var_{i}.txt'
    approximate_and_plot(file_name)
