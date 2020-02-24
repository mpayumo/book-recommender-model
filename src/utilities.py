import pandas as pd 
import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster


def palette():
    message = '''
    =========================================================================
    Possible values are: \n
    palette = ........ \n
    Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, 
    BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, 
    GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, 
    OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, 
    Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, 
    PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, 
    Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, 
    RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, 
    Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, 
    Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, 
    YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, 
    binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, 
    cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, 
    cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, 
    gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, 
    gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, 
    gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, 
    gray, gray_r, hot, hot_r, hsv, hsv_r, icefire, icefire_r, 
    inferno, inferno_r, jet, jet_r, magma, magma_r, mako, mako_r, 
    nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, 
    plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, rocket, rocket_r, 
    seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, 
    tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, 
    viridis, viridis_r, vlag, vlag_r, winter, winter_r
    =========================================================================
    '''
    print(message)
    
# Read CSV
def csv(path):
    df = pd.read_csv(path, encoding='iso8859-1', header='infer', delimiter=';', error_bad_lines=False)
    return df

# Frequency Table
def distribution_table(col1):
    dist = col1.value_counts().to_frame()
    dist.reset_index(inplace=True)
    return dist

# Frequency Distribution Plot
def plot_distribution(data_source, x_value, y_value, xlabel, ylabel, title, sns_palette1='CMRmap', bar_palette2='terrain'):
    '''
    Rename columns appropriately
    prior to plotting distribution.
    '''
    sns.set(style='whitegrid',
        palette=sns_palette1,
        font_scale=2.5, 
        color_codes=True)

    fig, ax = plt.subplots(figsize=(25,16))
    ax = sns.barplot(x=x_value, y=y_value,
                    data=data_source,
                    palette=bar_palette2)
    ax.set_xticklabels(ax.get_xticklabels(),
                    rotation=90,
                    fontsize=30)
    ax.set_xlabel(xlabel,
                fontsize=40)
    ax.set_ylabel(ylabel,
                fontsize=40)
    ax.set_title(title,
                fontsize=50,
                labelpad=40)
    # plt.text(3, 750000, 'Clear = 805,381',
    #         fontsize=35)
    # plt.text(8, 305000, 'Fair = 333,852',
    #         fontsize=35)
    # plt.text(38.75, 350000, 'Overcast = 381,396',
    #         fontsize=35)
    plt.tight_layout()
    plt.savefig('img/rename_this_image.jpg')
    