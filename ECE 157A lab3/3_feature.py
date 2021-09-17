#import necessary library
import numpy as np
import os
import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report as cr
from mpl_toolkits.mplot3d import Axes3D


#output directory
outdir='./OneSVM'
if not os.path.exists(outdir):
    os.mkdir(outdir)

#load training and validation data
df=pd.read_csv('nba_players_stats_19_20_per_game.csv')
df=df.fillna(0)
selected_features=['PTS','TRB','AST']
xs=df[selected_features]

#visualize feature distrubution vis histogram and CDF
for f in selected_features:
    #Plot PDF
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.hist(df[f],50)
    plt.xlabel(f)
    plt.ylabel('Number of Occurences')
    plt.title('Distribution of ' + f)
    plt.show()
    exit()
    
    #Plot CDf
    plt.subplot(1,2,2)
    counts, bin_edges = np.histogram(df[f],bins=100)
    counts = counts/np.sum(counts)
    cdf=np.cumsum(counts)
    plt.plot(bin_edges[1:], cdf)
    plt.xlabel(f)
    plt.ylabel('percent of players')
    plt.title ('CDF of ' + f)
    
    plt.show()
    exit()
    plt.savefig(outdir + '/distribution of ' + f + '.png')
    plt.close()