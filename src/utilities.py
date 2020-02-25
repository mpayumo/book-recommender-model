import pandas as pd 
import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from mpl_toolkits.mplot3d import Axes3D

    
# Read CSV
def csv(path):
    df = pd.read_csv(path, encoding='iso8859-1', header='infer', delimiter=';', error_bad_lines=False)
    return df

# Frequency Table
def distribution_table(col1):
    dist = col1.value_counts().to_frame()
    dist.reset_index(inplace=True)
    return distYear

def plot_donut(label, count, title):
    fig, ax = plt.subplots(figsize=(12, 7), subplot_kw=dict(aspect="equal"))

    wedges, texts = ax.pie(count, wedgeprops=dict(width=0.5), startangle=-40)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
            bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(label[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, **kw)

    ax.set_title(title)
    plt.savefig('donut.jpg');

def plot_exploding_donut():
    # The slices will be ordered and plotted counter-clockwise.
    labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
    sizes = [15, 30, 45, 10]
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    explode = (0, 0, 0, 0)  # explode a slice if required
    explode = (0,0,0,0,0,0,0.3,0.5,0,0.8)

    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True)
            
    #draw a circle at the center of pie to make it look like a donut
    centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1.25)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)


    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')
    plt.show();  


def plot_horizontal_distribution(x_val, y_val, df, title, color, xlim, xlabel):
    '''
    Rename columns appropriately
    prior to plotting distribution.
    '''
    sns.set(style='whitegrid',
        palette='CMRmap',
        font_scale=2.5, 
        color_codes=True)
    sns.set_color_codes('pastel')
    sns.barplot(x = x_val, y = y_val, data=df, label=title, color=color)
    # ax.legend(n_col=2, loc='lower_right', frameon=True)
    ax.set(xlim=xlim, ylabel="", xlabel=xlabel)
    sns.despine(left=False, right=False, top=False, bottom=False)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('img/new_horizontal_bar_graph.jpg');


# Frequency Distribution Plot
def plot_horizontal_distribution(x_val, y_val, df, title, color, xlim, xlabel, ylabel):
    '''
    Rename columns appropriately
    prior to plotting distribution.
    '''
    sns.set(style='whitegrid',
        palette='CMRmap',
        font_scale=2.5, 
        color_codes=True)
    sns.set_color_codes('pastel')
    sns.barplot(x = x_val, y = y_val, data=df, label=title, color=color)
    ax.set_title(title,
                fontsize=40,
                pad=15.0)
    ax.set_ylabel(ylabel,
                 fontsize=30)
    ax.set_xlabel(xlabel,
                 fontsize=30)
    sns.despine(top=True, left=False, bottom=True, right=True)
    plt.xticks(rotation=45,
              fontsize=35)
    plt.yticks(fontsize=35)
    for p in ax.patches:
        width=p.get_width()
        plt.text(5 + p.get_width(), p.get_y() + 0.55 * p.get_height(), '{:.0f}'.format(int(width)), va='center')
    plt.tight_layout()
    plt.savefig('img/new_horizontal_bar_graph.jpg');

def plot_line(x_val, y_val, title, xlabel, ylabel, xlim, ylim, color):
    ax.plot(x_val,
            y_val,
            '-.',
            linestyle='solid',
            linewidth=3,
            color=color
    )

    ax.xaxis.set_tick_params(pad=20)
    ax.tick_params(axis='x',
                pad=2,
                grid_linewidth=1)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(rotation=45,
            fontsize=25)
    plt.yticks(fontsize=25)
    plt.title(title,
            fontsize=30)
    plt.xlabel(xlabel,
            fontsize=30,
            labelpad=20)
    plt.ylabel(ylabel,
            fontsize=30,
            labelpad=20)

    plt.tight_layout()
    plt.savefig('img/timeseries.jpg');


def plot_scatter3D(x_val, y_val, z_val, x_label, y_label, z_label):
    fig = plt.figure(1, figsize=(12,7))
    plt.clf()
    ax = Axes3D(fig, rect=[0,0,0.95,1], elev=48, azim=134)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.scatter(x_val, y_val, z_val, c=labels.astype(np.float))
    plt.savefig('img/3Dscatter.jpg')

def heat_it():
    pass

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
    