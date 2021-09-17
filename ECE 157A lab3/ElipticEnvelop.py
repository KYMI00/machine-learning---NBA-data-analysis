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
from sklearn.covariance import EllipticEnvelope as EE
from mpl_toolkits.mplot3d import Axes3D


#output directory
outdir='./EllipticEnvelope'
if not os.path.exists(outdir):
    os.mkdir(outdir)
#load training and validation data
df=pd.read_csv('nba_players_stats_19_20_per_game.csv')
df=df.fillna(0)
selected_features=['DRB','STL','BLK']
xs=df[selected_features]

outlier_portion = 0.05
#Train Outlier Detector
clf = EE(random_state=0)
clf = clf.fit(xs)
pred = clf.predict(xs)
df = df.assign(prediction = pred)

#get Sample decision boundary and sort results
decision = clf.decision_function(xs)
df = df.assign(decision = decision)

xx=[]
for a,b in zip(df.iloc[np.where(pred==-1)[0],1], decision[np.where(pred==-1)[0]]):
    xx.append([a,round(b,2)])
xx = sorted(xx,key=lambda x: x[1])

for el in xx:
    print(el)
print(pred.shape)
print('{} outliers and {} inliders'.format(sum(pred==-1), sum(pred==1)))

#Save outlier score to a CSV
column_values = df[['Player', 'decision']].to_numpy()
sorted_column_values = np.array(sorted(column_values,key=lambda x: x[1]))

pd.DataFrame(sorted_column_values, columns= [['Player', 'Scores']]).to_csv(outdir + '/One_Class_SVM_Scores.csv', index=False)

#isolation forest
from sklearn.ensemble import IsolationForest as IF

clf = IF(random_state=0)
clf = clf.fit(xs)
pred = clf.predict(xs)
df = df.assign(prediction = pred)
