import pandas as pd
from pandas import ExcelWriter
import os
#import itertools
import numpy as np
os.chdir('C:\Users\\nauga\Google Drive\BuildingPrognostics\ForecastingTrainingData')
df = pd.ExcelFile('JunJulyCleanedData2.xlsx').parse('Sheet1')
df.replace('', np.nan, inplace=True)
#newpd = pd.DataFrame(df.iloc[0::6, 0::])###
#print newpd

newpd = df[np.isfinite(df['Hourly Totalization.Hourly Totals Trend ()'])]
writer = ExcelWriter('JunJulyFinalData.xlsx')
newpd.to_excel(writer,'Sheet1')
writer.save()

#newpd = df[np.isfinite(df['Hourly Totalization.Hourly Totals Trend ()'])]

#df = pd.ExcelFile('reqData.xlsx').parse('Sheet3')
#df.replace('', np.nan, inplace=True)
#print df