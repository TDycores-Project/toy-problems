#!/usr/bin/env python3

import numpy as np
import pandas as pd
import argparse
from pylab import *
from matplotlib import use
import matplotlib.pyplot as plt



def plot():
    # Define argparse for the input variables
    parser = argparse.ArgumentParser(description='Get input arguments')
    parser.add_argument('--conv_test_result',
                        dest='conv_test_result',
                        type=str,
                        required=True,
                        help='Path to the CSV file')
    args = parser.parse_args()
    conv_test_result = args.conv_test_result

    # Load the data
    data = pd.read_csv(conv_test_result)

    res = 'mesh_res'
    fig, ax = plt.subplots()

    data = data.sort_values('u_error')
    #u_error = data['u_error'].values[0]
    h = 1/data[res]
    H1 =  0.4* h # H = C h^p
    H2 = 1.2 * h**2
    E_u = data['u_error']
    E_p = data['p_error']
    log_h = np.log10(h)
    log_H1 = np.log10(H1)
    log_H2 = np.log10(H2)
    ax.loglog(h, E_p, 'o', color='blue')
    ax.loglog(h, E_u, 'o', color='black')
    m, b = np.polyfit(log_h, log_H1, 1)
    n, c = np.polyfit(log_h, log_H2, 1)
    ax.loglog(h, 10**b * h**m, '--', color='blue', label='Pressure O(h^1)')
    ax.loglog(h, 10**c * h**n, '--', color='black', label='Velocity O(h^2)')

    ax.legend(loc='best')
    ax.set_xlabel('h')
    ax.set_ylabel('Relative Error')
    ax.set_title('Convergence by h Refinement')
    xlim(.06, .3)
    fig.tight_layout()
    plt.savefig('conv_plt_h.png', bbox_inches='tight')


if __name__ == "__main__":
    plot()
