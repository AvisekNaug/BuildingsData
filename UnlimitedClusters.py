import pandas as pd
import os
import numpy as np
#from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import itertools
import math
from random import randint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

#fancy_dendrogram(
#    Z,
#    truncate_mode='lastp',
#    p=12,
#    leaf_rotation=90.,
#    leaf_font_size=12.,
#    show_contracted=True,
#    annotate_above=10,
#    max_d=max_d,  # plot a horizontal cut-off line
#)
#plt.show()

# calculate full dendrogram
#plt.figure(figsize=(25, 10))
#plt.title('Hierarchical Clustering Dendrogram')
#plt.xlabel('sample index')
#plt.ylabel('distance')
#dendrogram(
#    Z,
#    leaf_rotation=90.,  # rotates the x axis labels
#    leaf_font_size=8.,  # font size for the x axis label
#)
#plt.show()

class envCluster:
    def __init__(self,dataframe,index):
        self.cluster = dataframe
        self.clusterIndex = index
        
    def findMin(self):
        self.cluster.reset_index(drop=True,inplace=True)
        lenOfCluster = self.cluster.shape[0]
        randIdx = randint(0, lenOfCluster)
        randRow = self.cluster.iloc[randIdx]
        randEnergy = randRow['BTUs per Hour.Trend - Present Value ()'] + randRow['Hourly Totalization.Hourly Totals Trend ()']
        self.randCWSBTU = randRow['BTUs per Hour.Trend - Present Value ()']
        self.randSteamBTU = randRow['Hourly Totalization.Hourly Totals Trend ()']
        self.randTemp = randRow['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()']
        mask = (self.cluster['BTUs per Hour.Trend - Present Value ()'] + self.cluster['Hourly Totalization.Hourly Totals Trend ()']<randEnergy) & (self.cluster['BTUs per Hour.Trend - Present Value ()'] + self.cluster['Hourly Totalization.Hourly Totals Trend ()']>randEnergy-15000) & (self.cluster['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()']>self.randTemp-5) & (self.cluster['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()']<self.randTemp+5)
        minCluster = self.cluster.loc[mask]
        minCluster.reset_index(drop=True,inplace=True)
        minRow = minCluster.iloc[0]
        self.minCWSBTU = minRow['BTUs per Hour.Trend - Present Value ()']
        self.minSteamBTU = minRow['Hourly Totalization.Hourly Totals Trend ()']
        self.minTemp = minRow['Discharge Air Temperature.Discharge Air Temperature.Trend - Present Value ()']

    def findClusterCenters(self,i):#Assuming euclidean metric for Hierarchical clustering
        OATlist = self.cluster['Outside Air Temperature.Outside Air Temperature.Trend - Present Value ()'].values.tolist()
        self.OATCenter = sum(OATlist)/len(OATlist)
        self.OATrange = max(OATlist) - min(OATlist)
        sqrs_OAT = [(x - self.OATCenter)**2 for x in OATlist]
        self.OATstdDev = math.sqrt(sum(sqrs_OAT)/len(OATlist))
        
        OAHlist = self.cluster['Outside Air Humidity.Outside Air Humidity.Trend - Present Value ()'].values.tolist()
        self.OAHCenter = sum(OAHlist)/len(OAHlist)
        self.OAHrange = max(OAHlist) - min(OAHlist)
        sqrs_OAH = [(x - self.OAHCenter)**2 for x in OAHlist]
        self.OAHstdDev = math.sqrt(sum(sqrs_OAH)/len(OAHlist))
        
        DHIlist = self.cluster['DHI'].values.tolist()
        self.DHICenter = sum(DHIlist)/len(DHIlist)
        self.DHIrange = max(DHIlist) - min(DHIlist)
        sqrs_DHI = [(x - self.DHICenter)**2 for x in DHIlist]
        self.DHIstdDev = math.sqrt(sum(sqrs_DHI)/len(DHIlist))
        
        print ("Cluster"+str(i+1)+": "+"Mean "+str([self.OATCenter,self.OAHCenter,self.DHICenter])+"StdDeviation: "+str([self.OATstdDev,self.OAHstdDev,self.DHIstdDev]))
        print ("")
        
def showClusterModel(clusterList,clusterSize):
    fig = plt.figure()
    cc = ['r','b','c','g','y','k','m','r','b']#Provision upto 9 clusters
    mm = ['o','*','2','3','d','x','v','>','<']#Provision upto 9 clusters
    ax = fig.add_subplot(111, projection='3d')
    for i in range(clusterSize):
        ax.scatter(clusterList[i].cluster['Outside Air Temperature.Outside Air Temperature.Trend - Present Value ()'].values.tolist(),clusterList[i].cluster['Outside Air Humidity.Outside Air Humidity.Trend - Present Value ()'].values.tolist(),clusterList[i].cluster['DHI'].values.tolist(), c=cc[i], marker=mm[i])
    ax.set_xlabel('Outside Air Temperature')
    ax.set_ylabel('Outside Air Humidity')
    ax.set_zlabel('DHI')
    plt.show()
    
def showEnergySavings(clusterList,clusterSize):
    N= clusterSize
    ind = np.arange(N)
    width = 0.15 
    fig, ax = plt.subplots()
    cc = ['r','b','c','g','y','k']#supports upto 6 types of bars 
    bloc = []
    t = ()
    for j in range(N):
        t = t + (clusterList[j].randCWSBTU,)
    bloc.append(ax.bar(ind, t, width, color=cc[0]))
    t = ()
    for j in range(N):
        t = t + (clusterList[j].minCWSBTU,)
    bloc.append(ax.bar(ind+width, t, width, color=cc[1]))
    t = ()
    for j in range(N):
        t = t + (clusterList[j].randSteamBTU,)
    bloc.append(ax.bar(ind+2*width, t, width, color=cc[2]))
    t = ()
    for j in range(N):
        t = t + (clusterList[j].minSteamBTU,)
    bloc.append(ax.bar(ind+3*width, t, width, color=cc[3]))
    
    # add some text for labels, title and axes ticks
    ax.set_ylabel('EnergyConsumption in BTUs')
    ax.set_title('Comparison of Energy Consumtption with adjusted temperature ')
    ax.set_xticks(ind + 2*width)
    tickLabels = []
    for i in range(N):
        tickLabels.append([round(clusterList[i].randTemp,2),round(clusterList[i].minTemp,2)])
    ax.set_xticklabels((tickLabels))
    ax.legend((bloc[0][0], bloc[1][0],bloc[2][0], bloc[3][0] ), ('CWS BTUs at Original DischargeTeprerature', 'CWS BTUs at New DischargeTeprerature','Steam BTUs at Original DischargeTeprerature', 'Steam BTUs at New DischargeTeprerature'))

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')
    
    autolabel(bloc[0])
    autolabel(bloc[1])
    autolabel(bloc[2])
    autolabel(bloc[3])
    plt.show()
       

#Getting the data        
os.chdir('C:\Users\\nauga\Google Drive\BuildingPrognostics\ForecastingTrainingData')#\nauga
df = pd.ExcelFile('FinalData.xlsx').parse('Sheet1')

#Scaling the data
OAT = df['Outside Air Temperature.Outside Air Temperature.Trend - Present Value ()'].values.tolist()
ORH = df['Outside Air Humidity.Outside Air Humidity.Trend - Present Value ()'].values.tolist()
DHI = df['DHI'].values.tolist()
OAT = Scaling(OAT)
scaledOAT = OAT[0]
ORH = Scaling(ORH)
scaledORH = ORH[0]
DHI = Scaling(DHI)
scaledDHI = DHI[0]

#Arranging the 3 tuple data for Hierarchical Clustering
mat = [[i,j,k] for i,j,k in itertools.izip(scaledOAT,scaledORH,scaledDHI)]
matrix = np.matrix(mat)

#Performing hierarchical Clustering
Z = linkage(matrix, 'ward')
max_d = 55# Not used right now
clusterSize=7
labels = fcluster(Z, clusterSize, criterion='maxclust')
if False:
    # calculate full dendrogram
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis label
    )
    plt.show()

#Adding Cluster Labels to the original dataframe
results = pd.DataFrame([labels]).T
results.columns = ["ClassLabel"]
result = pd.concat([df,results], axis=1)

#Finding Energy Optimization for each Cluster
#Finding cluster centers for each Cluster
clusterList = []
for i in range(clusterSize):
    clusterList.append(envCluster(result.loc[result['ClassLabel'] == (i+1)],i+1))
    if True:
        clusterList[i].findMin()
    if True:
        clusterList[i].findClusterCenters(i)

#Plotting the Cluster
if False:
    showClusterModel(clusterList,clusterSize) #works for upto 9 clusters

#Plotting the energy Savings
if True:
    showEnergySavings(clusterList,clusterSize)
