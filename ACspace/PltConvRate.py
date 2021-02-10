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
    parser.add_argument('--ConvResult',
                        dest='ConvResult',
                        type=str,
                        required=True,
                        help='Path to the CSV file')
    args = parser.parse_args()
    ConvResult = args.ConvResult

    # Load the data
    data = pd.read_csv(ConvResult)
    fig, ax = plt.subplots()

    data = data.sort_values('u_error')
    E_u = data['u_error']
    E_p = data['p_error']
    h = 1/data['mesh_res']
    H1 =  E_p[0]* (h/h[0]) # H = C h^1
    H2 =  E_u[0]* (h/h[0])**2  # H = C h^2

    ax.loglog(h, E_p, 'o', color='blue')
    ax.loglog(h, E_u, 'o', color='black')
    ax.loglog(h, H1, '--', color='blue', label='Pressure O(h^1)')
    ax.loglog(h, H2, '--', color='black', label='Velocity O(h^2)')

    ax.legend(loc='best')
    ax.set_xlabel('h')
    ax.set_ylabel('Relative Error')
    ax.set_title('Convergence by h Refinement')
    #xlim(.06, .3)
    fig.tight_layout()
    plt.savefig('ConvRate.png', bbox_inches='tight')


if __name__ == "__main__":
    plot()
