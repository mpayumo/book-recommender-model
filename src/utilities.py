import pandas as pd 
import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
%matplotlib inline
plt.style.use('fast')

import seaborn as sns
import folium
from folium.plugins import MarkerCluster

# Read CSV
def csv(path):
    df = pd.read_csv(path, encoding='latin-1')
    return df

# Frequency Distribution
def plot_distribution(col1, col2):
