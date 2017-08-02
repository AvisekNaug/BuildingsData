import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
import itertools
import math

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
print type(labels)
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

Cluster0 = result.loc[result['ClassLabel'] == 0]#Change
Cluster0.reset_index(drop=True,inplace=True)
lst = Cluster0['Outside Air Temperature.Outside Air Temperature.Trend - Present Value ()'].values.tolist()
Tempcenter = clusterCenters[0][0]#Change
sqrs_OAT = [(x - Tempcenter)**2 for x in lst]
STD_OAT = math.sqrt(sum(sqrs_OAT)/len(lst))
lst = Cluster0['Outside Air Humidity.Outside Air Humidity.Trend - Present Value ()'].values.tolist()
Humiditycenter = clusterCenters[0][1]#Change
sqrs_OAH = [(x - Humiditycenter)**2 for x in lst]
STD_OAH = math.sqrt(sum(sqrs_OAH)/len(lst))
lst = Cluster0['DHI'].values.tolist()
DHIcenter = clusterCenters[0][2]#Change
sqrs_DHI = [(x - DHIcenter)**2 for x in lst]
STD_DHI = math.sqrt(sum(sqrs_DHI)/len(lst))
print ("Cluster0")#Change
print [Tempcenter,STD_OAT]
print [Humiditycenter, STD_OAH]
print [DHIcenter, STD_DHI]

Cluster0 = result.loc[result['ClassLabel'] == 1]#Change
Cluster0.reset_index(drop=True,inplace=True)
lst = Cluster0['Outside Air Temperature.Outside Air Temperature.Trend - Present Value ()'].values.tolist()
Tempcenter = clusterCenters[1][0]#Change
sqrs_OAT = [(x - Tempcenter)**2 for x in lst]
STD_OAT = math.sqrt(sum(sqrs_OAT)/len(lst))
lst = Cluster0['Outside Air Humidity.Outside Air Humidity.Trend - Present Value ()'].values.tolist()
Humiditycenter = clusterCenters[1][1]#Change
sqrs_OAH = [(x - Humiditycenter)**2 for x in lst]
STD_OAH = math.sqrt(sum(sqrs_OAH)/len(lst))
lst = Cluster0['DHI'].values.tolist()
DHIcenter = clusterCenters[1][2]#Change
sqrs_DHI = [(x - DHIcenter)**2 for x in lst]
STD_DHI = math.sqrt(sum(sqrs_DHI)/len(lst))
print ("Cluster1s")#Change
print [Tempcenter,STD_OAT]
print [Humiditycenter, STD_OAH]
print [DHIcenter, STD_DHI]

Cluster0 = result.loc[result['ClassLabel'] == 2]#Change
Cluster0.reset_index(drop=True,inplace=True)
lst = Cluster0['Outside Air Temperature.Outside Air Temperature.Trend - Present Value ()'].values.tolist()
Tempcenter = clusterCenters[2][0]#Change
sqrs_OAT = [(x - Tempcenter)**2 for x in lst]
STD_OAT = math.sqrt(sum(sqrs_OAT)/len(lst))
lst = Cluster0['Outside Air Humidity.Outside Air Humidity.Trend - Present Value ()'].values.tolist()
Humiditycenter = clusterCenters[2][1]#Change
sqrs_OAH = [(x - Humiditycenter)**2 for x in lst]
STD_OAH = math.sqrt(sum(sqrs_OAH)/len(lst))
lst = Cluster0['DHI'].values.tolist()
DHIcenter = clusterCenters[2][2]#Change
sqrs_DHI = [(x - DHIcenter)**2 for x in lst]
STD_DHI = math.sqrt(sum(sqrs_DHI)/len(lst))
print ("Cluster2")#Change
print [Tempcenter,STD_OAT]
print [Humiditycenter, STD_OAH]
print [DHIcenter, STD_DHI]

Cluster0 = result.loc[result['ClassLabel'] == 3]#Change
Cluster0.reset_index(drop=True,inplace=True)
lst = Cluster0['Outside Air Temperature.Outside Air Temperature.Trend - Present Value ()'].values.tolist()
Tempcenter = clusterCenters[3][0]#Change
sqrs_OAT = [(x - Tempcenter)**2 for x in lst]
STD_OAT = math.sqrt(sum(sqrs_OAT)/len(lst))
lst = Cluster0['Outside Air Humidity.Outside Air Humidity.Trend - Present Value ()'].values.tolist()
Humiditycenter = clusterCenters[3][1]#Change
sqrs_OAH = [(x - Humiditycenter)**2 for x in lst]
STD_OAH = math.sqrt(sum(sqrs_OAH)/len(lst))
lst = Cluster0['DHI'].values.tolist()
DHIcenter = clusterCenters[3][2]#Change
sqrs_DHI = [(x - DHIcenter)**2 for x in lst]
STD_DHI = math.sqrt(sum(sqrs_DHI)/len(lst))
print ("Cluster3")#Change
print [Tempcenter,STD_OAT]
print [Humiditycenter, STD_OAH]
print [DHIcenter, STD_DHI]

Cluster0 = result.loc[result['ClassLabel'] == 4]#Change
Cluster0.reset_index(drop=True,inplace=True)
lst = Cluster0['Outside Air Temperature.Outside Air Temperature.Trend - Present Value ()'].values.tolist()
Tempcenter = clusterCenters[4][0]#Change
sqrs_OAT = [(x - Tempcenter)**2 for x in lst]
STD_OAT = math.sqrt(sum(sqrs_OAT)/len(lst))
lst = Cluster0['Outside Air Humidity.Outside Air Humidity.Trend - Present Value ()'].values.tolist()
Humiditycenter = clusterCenters[4][1]#Change
sqrs_OAH = [(x - Humiditycenter)**2 for x in lst]
STD_OAH = math.sqrt(sum(sqrs_OAH)/len(lst))
lst = Cluster0['DHI'].values.tolist()
DHIcenter = clusterCenters[4][2]#Change
sqrs_DHI = [(x - DHIcenter)**2 for x in lst]
STD_DHI = math.sqrt(sum(sqrs_DHI)/len(lst))
print ("Cluster4")#Change
print [Tempcenter,STD_OAT]
print [Humiditycenter, STD_OAH]
print [DHIcenter, STD_DHI]

