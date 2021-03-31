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
    parser.add_argument('--InfSup',
                        dest='InfSup',
                        type=str,
                        required=True,
                        help='Path to the InfSup CSV file')
    args = parser.parse_args()
    InfSup = args.InfSup

    # Load the data
    data = pd.read_csv(InfSup)
    fig, ax = plt.subplots()

    data = data.sort_values('run')
    alpha = data['coercivity']
    beta = data['infsup']
    h = data['h']
    ax.loglog(h, alpha, '--', color='blue', label='coercivity constant')
    ax.loglog(h, beta, '--', color='black', label='infsup constant')

    ax.legend(loc='upper left')
    ax.set_xlabel('h')
    ax.set_ylabel('coercivity and infsup')
    #ax.set_title('Uniform mesh')
    #xlim(.06, .3)
    fig.tight_layout()
    plt.savefig('infsup.png',bbox_inches='tight')


if __name__ == "__main__":
    plot()
