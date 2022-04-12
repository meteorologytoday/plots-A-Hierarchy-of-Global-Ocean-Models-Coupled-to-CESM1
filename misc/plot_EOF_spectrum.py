import matplotlib as mplt
mplt.use('Agg')

import matplotlib.pyplot as plt 
import netCDF4, sys, argparse
import numpy as np
from scipy import signal
from pprint import pprint

def mavg(y, span):
 
    N = len(y)
    yy = np.zeros((N,))
    for i in range(N):
        if i >= span - 1:
            rng = slice(i-span+1,i+1)
            yy[i] = np.mean(y[rng])
        else:
            yy[i] = np.nan

    return yy


def SpectralVariance(y):
    c = np.fft.rfft(y, norm="ortho") # Ortho means normalized by sqrt N
    return abs(c)**2.0
 
parser = argparse.ArgumentParser()
parser.add_argument('--input-dir')
parser.add_argument('--output-dir')
parser.add_argument('--casenames')
parser.add_argument('--legends')
parser.add_argument('--data-file')
parser.add_argument('--varname')
parser.add_argument('--normalize', default="yes")
parser.add_argument('--logy_max', type=float, default=3)
parser.add_argument('--logy_min', type=float, default=-3)
parser.add_argument('--ylabel', default="")
parser.add_argument('--colors')
parser.add_argument('--linestyles')
parser.add_argument('--t-offset', type=float, default=0.0)
parser.add_argument('--mavg', type=int, default=1)

args = parser.parse_args()

pprint(args)

casenames = args.casenames.split(",")
legends   = args.legends.split(",")
colors    = args.colors.split(",")

