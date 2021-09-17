import numpy as np
import os
import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix as cm
from sklearn.svm import OneClassSVM as OCS
from sklearn.metrics import classification_report as cr
from mpl_toolkits.mplot3d import Axes3D



#get top three outlier and see location on feature distribution
#xx[:3] column 3
three_outliers = xx[:3]
plot_num =1
plt.figure(figsize=(32, 16))
for name, _ in three_outliers:
    for f in selected_features:
        f_val = df.loc[df['Player']==name, [f]].to_numpy()[0,0]
        plt.subplot(3, 3, plot_num)
        plot_num +=1
        counts, bin_edges = np.histogram(df[f], bins=100)
        counts = counts/np.sum(counts)
        cdf = np.cumsum(counts)
        plt.plot(bin_edges[1:], cdf)
        plt.vlines(f_val, 0, 1, colors=['r'])
        plt.xlabel(f)
        plt.ylabel('Percent of Players')
        plt.title(name.split('\\')[0] + ' on CDF of ' + f)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
#top three outliers
#plot 3D graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

f1=[]
f2=[]
f3=[]
for name, _ in three_outliers:
    f1.append(df.loc[df['Player']==name, [selected_features[0]]].to_numpy()[0,0])
    f2.append(df.loc[df['Player']==name, [selected_features[1]]].to_numpy()[0,0])
    f3.append(df.loc[df['Player']==name, [selected_features[2]]].to_numpy()[0,0])
ax.scatter(f1,f2,f3,marker=(5, 2),color='r', depthshade=False)

for name, _ in three_outliers:
    df.drop(df.loc[df['Player']==name].index, inplace=True)

f1 = df.loc[df['prediction']==-1, [selected_features[0]]].to_numpy()[:,0]
f2 = df.loc[df['prediction']==-1, [selected_features[1]]].to_numpy()[:,0]
f3 = df.loc[df['prediction']==-1, [selected_features[2]]].to_numpy()[:,0]
ax.scatter(f1,f2,f3, marker='+',depthshade=False)

f1 = df.loc[df['prediction']==1, [selected_features[0]]].to_numpy()[:,0]
f2 = df.loc[df['prediction']==1, [selected_features[1]]].to_numpy()[:,0]
f3 = df.loc[df['prediction']==1, [selected_features[2]]].to_numpy()[:,0]
ax.scatter(f1,f2,f3, marker='^', depthshade=False)

ax.set_xlabel(selected_features[0])
ax.set_ylabel(selected_features[1])
ax.set_zlabel(selected_features[2])