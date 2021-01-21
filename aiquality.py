import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_excel(r'C:\Users\ADMIN\Desktop\data_sci\regressions\AirQualityUCI.xlsx')

data.info()
data.describe()


data.replace(to_replace= -200, value=np.NaN, inplace=True)

sns.heatmap(data.isnull(),yticklabels=False, cbar=False, cmap='viridis')


data_corr=data.corr()
print(data_corr)

data.fillna(data.mean(), inplace=True)



data['Date']=data['Date'].astype(str)
data['Time']=data['Time'].astype(str)
data['DateTime']=data['Date']+' '+data['Time']
data.dtypes

#data['datetime_n'] = datetime.strptime('DateTime', '%Y/%m/%d %H:%M:%S')
#data['DateTime']=datetime[(data.date).hours]
#data['DateTime']=data['DateTime'].astype()
data['DateTime']=pd.to_datetime(data.DateTime, format='%Y/%m/%d')
data.info()

data['month'] = pd.DatetimeIndex(data['DateTime']).month
data['Year']=pd.DatetimeIndex(data['DateTime']).year
data.info()

data['YearMonth'] = data['DateTime'].dt.strftime('%Y-%m')


data=data.drop(['Date','Time'],axis=1)


data_corr=data.corr()
print(data_corr)

plt.figure(figsize=(12,9))
sns.heatmap(data.corr(), annot=True, cmap="viridis")

#data.groupby(['month'])['T'].mean()

#########Plotting###########

data.info()

plt.figure(figsize=(10,10))
sns.violinplot(x='YearMonth',y='T',data=data)

plt.figure(figsize=(10,10))
sns.scatterplot(x='YearMonth',y='T' ,data=data)



plt.figure(figsize=(10,7))
sns.boxplot(data['YearMonth'],data['T'], data = data)



plt.figure(figsize=(10,7))
sns.boxplot(data['YearMonth'],data['PT08.S5(O3)'], data = data)

################################Regression

X=data[['CO(GT)','PT08.S1(CO)','NMHC(GT)','C6H6(GT)','PT08.S2(NMHC)','NOx(GT)','PT08.S3(NOx)','NO2(GT)','PT08.S4(NO2)','PT08.S5(O3)','RH','AH']].values

y=data[["T"]].values

X_train ,X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

regressor=LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
