import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
import itertools
import math
from random import randint
import matplotlib.pyplot as plt

def Scaling(datapoints):
    total = sum(datapoints)
    average = total/len(datapoints)
    variance = sum([(i-average)**2 for i in datapoints])/len(datapoints)
    stdDeviation = math.sqrt(variance)
    scaledData = [(i-average)/stdDeviation for i in datapoints]
    return [scaledData, average, stdDeviation]

def reverseScale(datapoint,mu,sigma):
    result = (sigma*datapoint)+mu
    return result

os.chdir('C:\Users\\nauga\Google Drive\BuildingPrognostics\ForecastingTrainingData')#\nauga
df = pd.ExcelFile('FinalData.xlsx').parse('Sheet1')
OAT = df['Outside Air Temperature.Outside Air Temperature.Trend - Present Value ()'].values.tolist()
ORH = df['Outside Air Humidity.Outside Air Humidity.Trend - Present Value ()'].values.tolist()
DHI = df['DHI'].values.tolist()
OAT = Scaling(OAT)
scaledOAT = OAT[0]
ORH = Scaling(ORH)
scaledORH = ORH[0]
DHI = Scaling(DHI)
scaledDHI = DHI[0]
mat = [[i,j,k] for i,j,k in itertools.izip(scaledOAT,scaledORH,scaledDHI)]
matrix = np.matrix(mat)

# Using sklearn
km = KMeans(n_clusters=5)
km.fit(matrix)
# Get cluster assignment labels
labels = km.labels_
scaledClusterCenters = km.cluster_centers_
clusterCenters = []
for i in scaledClusterCenters:
    t0 = reverseScale(i[0],OAT[1],OAT[2])
    t1 = reverseScale(i[1],ORH[1],ORH[2])
    t2 = reverseScale(i[2],DHI[1],DHI[2])
    clusterCenters.append([t0,t1,t2])

cwsBTU = df['BTUs per Hour.Trend - Present Value ()'].values.tolist()
stmBTU = df['Hourly Totalization.Hourly Totals Trend ()'].values.tolist()
DT = df['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()'].values.tolist()
Output =[(a+b)/c for a,b,c in itertools.izip(cwsBTU,stmBTU,DT)]
results = pd.DataFrame([labels,Output]).T
results.columns = ["ClassLabel","Output"]
result = pd.concat([df,results], axis=1)

Cluster0 = result.loc[result['ClassLabel'] == 0]
Cluster0.reset_index(drop=True,inplace=True)
len0 = Cluster0.shape[0]
minIdx0 = randint(0, len0)
#minIdx0 = Cluster0['Output'].idxmin()
minrow0 = Cluster0.iloc[minIdx0]
minEnergy0 = minrow0['BTUs per Hour.Trend - Present Value ()'] + minrow0['Hourly Totalization.Hourly Totals Trend ()']
minTemp0 = minrow0['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()']
##########
mask = (Cluster0['BTUs per Hour.Trend - Present Value ()'] + Cluster0['Hourly Totalization.Hourly Totals Trend ()']<minEnergy0) & (Cluster0['BTUs per Hour.Trend - Present Value ()'] + Cluster0['Hourly Totalization.Hourly Totals Trend ()']>minEnergy0-15000) & (Cluster0['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()']>minTemp0-5) & (Cluster0['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()']<minTemp0+5)
randClus0 = Cluster0.loc[mask]
randClus0.reset_index(drop=True,inplace=True)
randRow0 = randClus0.iloc[0]
randEnergy0 = randRow0['BTUs per Hour.Trend - Present Value ()']+randRow0['Hourly Totalization.Hourly Totals Trend ()']
randTemp0 = randRow0['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()']

Cluster1 = result.loc[result['ClassLabel'] == 1]
Cluster1.reset_index(drop=True,inplace=True)
len1 = Cluster1.shape[0]
minIdx1 = randint(0, len1)
#minIdx1 = Cluster1['Output'].idxmin()
minrow1 = Cluster1.iloc[minIdx1]
minEnergy1 = minrow1['BTUs per Hour.Trend - Present Value ()'] + minrow1['Hourly Totalization.Hourly Totals Trend ()']
minTemp1 = minrow1['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()']
#######
mask = (Cluster1['BTUs per Hour.Trend - Present Value ()'] + Cluster1['Hourly Totalization.Hourly Totals Trend ()']<minEnergy1) & (Cluster1['BTUs per Hour.Trend - Present Value ()'] + Cluster1['Hourly Totalization.Hourly Totals Trend ()']>minEnergy1-15000) & (Cluster1['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()']>minTemp1-5) & (Cluster1['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()']<minTemp1+5)
randClus1 = Cluster1.loc[mask]
randClus1.reset_index(drop=True,inplace=True)
randRow1 = randClus1.iloc[0]
randEnergy1 = randRow1['BTUs per Hour.Trend - Present Value ()']+randRow1['Hourly Totalization.Hourly Totals Trend ()']
randTemp1 = randRow1['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()']

Cluster2 = result.loc[result['ClassLabel'] == 2]
Cluster2.reset_index(drop=True,inplace=True)
len2 = Cluster2.shape[0]
minIdx2 = randint(0, len2)
#minIdx2 = Cluster2['Output'].idxmin()
minrow2 = Cluster2.iloc[minIdx2]
minEnergy2 = minrow2['BTUs per Hour.Trend - Present Value ()'] + minrow2['Hourly Totalization.Hourly Totals Trend ()']
minTemp2 = minrow2['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()']
#######
mask = (Cluster2['BTUs per Hour.Trend - Present Value ()'] + Cluster2['Hourly Totalization.Hourly Totals Trend ()']<minEnergy2) & (Cluster2['BTUs per Hour.Trend - Present Value ()'] + Cluster2['Hourly Totalization.Hourly Totals Trend ()']>minEnergy2-15000) & (Cluster2['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()']>minTemp2-5) & (Cluster2['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()']<minTemp2+5)
randClus2 = Cluster2.loc[mask]
randClus2.reset_index(drop=True,inplace=True)
randRow2 = randClus2.iloc[0]
randEnergy2 = randRow2['BTUs per Hour.Trend - Present Value ()']+randRow2['Hourly Totalization.Hourly Totals Trend ()']
randTemp2 = randRow2['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()']

Cluster3 = result.loc[result['ClassLabel'] == 3]
Cluster3.reset_index(drop=True,inplace=True)
len3 = Cluster3.shape[0]
minIdx3 = randint(0, len3)
#minIdx3 = Cluster3['Output'].idxmin()
minrow3 = Cluster3.iloc[minIdx3]
minEnergy3 = minrow3['BTUs per Hour.Trend - Present Value ()'] + minrow3['Hourly Totalization.Hourly Totals Trend ()']
minTemp3 = minrow3['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()']
#######
mask = (Cluster3['BTUs per Hour.Trend - Present Value ()'] + Cluster3['Hourly Totalization.Hourly Totals Trend ()']<minEnergy3) & (Cluster3['BTUs per Hour.Trend - Present Value ()'] + Cluster3['Hourly Totalization.Hourly Totals Trend ()']>minEnergy3-15000) & (Cluster3['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()']>minTemp3-5) & (Cluster3['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()']<minTemp3+5)
randClus3 = Cluster3.loc[mask]
randClus3.reset_index(drop=True,inplace=True)
randRow3 = randClus3.iloc[0]
randEnergy3 = randRow3['BTUs per Hour.Trend - Present Value ()']+randRow3['Hourly Totalization.Hourly Totals Trend ()']
randTemp3 = randRow3['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()']

Cluster4 = result.loc[result['ClassLabel'] == 4]
Cluster4.reset_index(drop=True,inplace=True)
len4 = Cluster4.shape[0]
minIdx4 = randint(0, len4)
#minIdx4 = Cluster4['Output'].idxmin()
minrow4 = Cluster4.iloc[minIdx4]
minEnergy4 = minrow4['BTUs per Hour.Trend - Present Value ()'] + minrow4['Hourly Totalization.Hourly Totals Trend ()']
minTemp4 = minrow4['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()']
#######
mask = (Cluster4['BTUs per Hour.Trend - Present Value ()'] + Cluster4['Hourly Totalization.Hourly Totals Trend ()']<minEnergy4) & (Cluster4['BTUs per Hour.Trend - Present Value ()'] + Cluster4['Hourly Totalization.Hourly Totals Trend ()']>minEnergy4-15000) & (Cluster4['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()']>minTemp4-5) & (Cluster4['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()']<minTemp4+5)
randClus4 = Cluster4.loc[mask]
randClus4.reset_index(drop=True,inplace=True)
randRow4 = randClus4.iloc[0]
randEnergy4 = randRow4['BTUs per Hour.Trend - Present Value ()']+randRow4['Hourly Totalization.Hourly Totals Trend ()']
randTemp4 = randRow4['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()']


t01=minrow0['BTUs per Hour.Trend - Present Value ()']
t02=minrow0['Hourly Totalization.Hourly Totals Trend ()']
t03=randRow0['BTUs per Hour.Trend - Present Value ()']
t04=randRow0['Hourly Totalization.Hourly Totals Trend ()']

t11=minrow1['BTUs per Hour.Trend - Present Value ()']
t12=minrow1['Hourly Totalization.Hourly Totals Trend ()']
t13=randRow1['BTUs per Hour.Trend - Present Value ()']
t14=randRow1['Hourly Totalization.Hourly Totals Trend ()']

t21=minrow2['BTUs per Hour.Trend - Present Value ()']
t22=minrow2['Hourly Totalization.Hourly Totals Trend ()']
t23=randRow2['BTUs per Hour.Trend - Present Value ()']
t24=randRow2['Hourly Totalization.Hourly Totals Trend ()']

t31=minrow3['BTUs per Hour.Trend - Present Value ()']
t32=minrow3['Hourly Totalization.Hourly Totals Trend ()']
t33=randRow3['BTUs per Hour.Trend - Present Value ()']
t34=randRow3['Hourly Totalization.Hourly Totals Trend ()']

t41=minrow4['BTUs per Hour.Trend - Present Value ()']
t42=minrow4['Hourly Totalization.Hourly Totals Trend ()']
t43=randRow4['BTUs per Hour.Trend - Present Value ()']
t44=randRow4['Hourly Totalization.Hourly Totals Trend ()']

N = 5
ind = np.arange(N)
width = 0.15 
fig, ax = plt.subplots()
rects1 = ax.bar(ind, (t01,t11,t21,t31,t41), width, color='c')
rects2 = ax.bar(ind+width, (t03,t13,t23,t33,t43), width, color='k')
rects3 = ax.bar(ind+2*width, (t02,t12,t22,t32,t42), width, color='r')
rects4 = ax.bar(ind+3*width, (t04,t14,t24,t34,t44), width, color='g')



# add some text for labels, title and axes ticks
ax.set_ylabel('EnergyConsumption in BTUs')
ax.set_title('Comparison of Energy Consumtption with adjusted temperature ')
ax.set_xticks(ind + 2*width)
ax.set_xticklabels(([round(minTemp0,2),round(randTemp0,2)], [round(minTemp1,2),round(randTemp1,2)], [round(minTemp2,2),round(randTemp2,2)], [round(minTemp3,2),round(randTemp3,2)],[round(minTemp4,2),round(randTemp4,2)]))
ax.legend((rects1[0], rects2[0],rects3[0], rects4[0] ), ('CWS BTUs at Original DischargeTeprerature', 'CWS BTUs at New DischargeTeprerature','Steam BTUs at Original DischargeTeprerature', 'Steam BTUs at New DischargeTeprerature'))

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
plt.show()

#import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Cluster0['Outside Air Temperature.Outside Air Temperature.Trend - Present Value ()'].values.tolist(),
Cluster0['Outside Air Humidity.Outside Air Humidity.Trend - Present Value ()'].values.tolist(),
Cluster0['DHI'].values.tolist(), c='r', marker='o')

ax.scatter(Cluster1['Outside Air Temperature.Outside Air Temperature.Trend - Present Value ()'].values.tolist(),
Cluster1['Outside Air Humidity.Outside Air Humidity.Trend - Present Value ()'].values.tolist(),
Cluster1['DHI'].values.tolist(), c='b', marker='*')

ax.scatter(Cluster2['Outside Air Temperature.Outside Air Temperature.Trend - Present Value ()'].values.tolist(),
Cluster2['Outside Air Humidity.Outside Air Humidity.Trend - Present Value ()'].values.tolist(),
Cluster2['DHI'].values.tolist(), c='c', marker='2')

ax.scatter(Cluster3['Outside Air Temperature.Outside Air Temperature.Trend - Present Value ()'].values.tolist(),
Cluster3['Outside Air Humidity.Outside Air Humidity.Trend - Present Value ()'].values.tolist(),
Cluster3['DHI'].values.tolist(), c='g', marker='3')

ax.scatter(Cluster4['Outside Air Temperature.Outside Air Temperature.Trend - Present Value ()'].values.tolist(),
Cluster4['Outside Air Humidity.Outside Air Humidity.Trend - Present Value ()'].values.tolist(),
Cluster4['DHI'].values.tolist(), c='y', marker='d')

ax.set_xlabel('Outside Air Temperature')
ax.set_ylabel('Outside Air Humidity')
ax.set_zlabel('DHI')
plt.show()