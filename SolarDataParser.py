import pandas as pd
import os
os.chdir('C:\Users\\nauga\Google Drive\BuildingPrognostics\ForecastingTrainingData\SolarData')
#print os.getcwd()
#print os.listdir('C:\\Users\\nauga\\Downloads\\SolarData')
df = pd.ExcelFile('reqData.xlsx').parse('Sheet1')
DHI = df['DHI'].values.tolist()
print len(DHI)