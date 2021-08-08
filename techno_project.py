# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

!pip install -q xlrd

df = pd.read_excel('/content/drive/MyDrive/Spectrum Database_March 2021.xlsx')
df

a = df['awardName'].unique()
a

auctions = df[df['awardName'] == 'Saudi 700 MHz, 800 MHz and 1800 MHz auction']
auctions

b = df.groupby(['countryName'])['awardName'].unique()
b

c = df['awardClassDescription'].unique()
c

temp = df[df['winner'].isna()]
temp1 = temp['awardName'].unique()
temp1

# for i in range(len(z)):
#   print(z[i])

df['total_freq'] = df['paired'] + df['unpaired']

df['per_MHz'] = df['reservePriceLocal']/df['total_freq']
df['per_MHz'] = df['per_MHz']/df['popCovered']

df['sold_per_MHz'] = df['headlinePriceLocal']/df['total_freq']
df['sold_per_MHz'] = df['sold_per_MHz']/df['popCovered']

awardvalues = ['Auction - Largely standard SMRA','Auction - Largely standard SMRA with augmented switching']

data = df.loc[ (df['freqBand']=='700MHz')]
data1 = data.loc[data['awardClassDescription'] == 'Auction - Largely standard SMRA'] 
data2 = data.loc[data['awardClassDescription'] == 'Auction - Largely standard SMRA with augmented switching']
data =  pd.concat([data1,data2],axis = 0)
data['awardClassDescription'].unique()

df['licenceUse'].unique()

data1 = data.loc[data['licenceUse'] == 'mobile']
data2 = data.loc[data['licenceUse'] == 'neutral']
data3 = data.loc[data['licenceUse'] == 'mobile and FWA']

data =  pd.concat([data1,data2,data3],axis = 0)
data['licenceUse'].unique()

reservePrice_data = data[['alpha3code','per_MHz','sold_per_MHz','nBidders','date','total_freq','licenceDuration	']]
reservePrice_data.groupby(by = ['alpha3code','date']).mean()

# payment_data = pd.read_excel('/content/drive/MyDrive/Spectrum Payments_March 2021.xlsx')

# brazil = reservePrice_data[reservePrice_data['alpha3code'] == 'BRA']
# canada = reservePrice_data[reservePrice_data['alpha3code'] == 'CAN']
germany = reservePrice_data[reservePrice_data['alpha3code'] == 'DEU']
finland = reservePrice_data[reservePrice_data['alpha3code'] == 'FIN']
fiji = reservePrice_data[reservePrice_data['alpha3code'] == 'FJI']
# france = reservePrice_data[reservePrice_data['alpha3code'] == 'FRA']
# britian = reservePrice_data[reservePrice_data['alpha3code'] == 'GBR']
# newzeland = reservePrice_data[reservePrice_data['alpha3code'] == 'NZL']
taiwan = reservePrice_data[reservePrice_data['alpha3code'] == 'TWN']
usa = reservePrice_data[reservePrice_data['alpha3code'] == 'USA']

# brazil['converted_reserv'] = brazil['per_MHz']*0.4151
# brazil['converted_headline'] = brazil['sold_per_MHz']*0.4151


# canada['converted_reserv'] = canada['per_MHz']*0.9109
# canada['converted_headline'] = canada['sold_per_MHz']*0.9109


germany['converted_reserv'] = germany['per_MHz']*1.1113
germany['converted_headline'] = germany['sold_per_MHz']*1.1113



finland['converted_reserv'] = finland['per_MHz']*1.055
finland['converted_headline'] = finland['sold_per_MHz']*1.055


fiji['converted_reserv'] = fiji['per_MHz']*1.8821
fiji['converted_headline'] = fiji['sold_per_MHz']*1.8821


# france['converted_reserv'] = france['per_MHz']*1.0646
# france['converted_headline'] = france['sold_per_MHz']*1.0646


# britian['converted_reserv'] = britian['per_MHz']*1.4628
# britian['converted_headline'] = britian['sold_per_MHz']*1.4628

# newzeland['converted_reserv'] = newzeland['per_MHz']*0.8668
# newzeland['converted_headline'] = newzeland['sold_per_MHz']*0.8668

taiwan['converted_reserv'] = taiwan['per_MHz']*0.0337
taiwan['converted_headline'] = taiwan['sold_per_MHz']*0.0337


usa['converted_reserv'] = usa['per_MHz']
usa['converted_headline'] = usa['sold_per_MHz']

new_data_set = pd.concat([germany,finland,fiji,taiwan,usa],axis=0)

new_data_set

new_data_set.to_csv('converted_data.csv')

# final_data = date_data.groupby(by = ['alpha3code','date']).mean()
# final_data.to_csv('final_data.csv')

date_data = pd.read_csv('/content/converted_data.csv',
                                parse_dates=['date'],
                                index_col=['date'] )

date_data

date_data = date_data.sort_index()
date_data

date_data.groupby(by = ['alpha3code','date']).mean()



v = reserv_pr, n biddres ,paired+unpaired, liscense duration
y = winning bid price

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

# date_data.insert(0,'Yr1',range(0,len(date_data)))
date_data.plot(y='converted_reserv',x='Yr1')

d = np.polyfit(date_data['Yr1'],date_data['converted_reserv'],1)
f = np.poly1d(d)

date_data.insert(10,'reg1',f(date_data['Yr1']))

ax = date_data.plot.scatter(x = 'Yr1',y='converted_reserv')
date_data.plot(x='Yr1', y='reg1',color='Red',legend = False,ax=ax)


# fig, ax = plt.subplots(figsize=(10, 10))

# # Add x-axis and y-axis
# ax.scatter(date_data.index.values,
#         date_data['converted_reserv'],
#         color='purple')

# # Set title and labels for axes
# ax.set(xlabel="Date",
#        ylabel="reserve prices",
#        title="Date vs reserve prices")

# # Rotate tick marks on x-axis
# plt.setp(ax.get_xticklabels(), rotation=45)
055
# plt.show()

date_data

date_data.plot(y='converted_headline',x='Yr1')

d = np.polyfit(date_data['Yr1'],date_data['converted_headline'],2)
f = np.polyval(d,2)

date_data.insert(9,'reg',f(date_data['Yr1']))

ax = date_data.plot.scatter(x = 'Yr1',y='converted_headline')
date_data.plot(x='Yr1', y='reg3',color='Red',legend = False,ax=ax)

f(16)

"""## New Model


"""

import pandas as pd
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

!pip install -q xlrd

df = pd.read_excel('/content/drive/MyDrive/Spectrum Database_March 2021.xlsx')
# data_set

df['total_alloted_freq'] = df['paired'] + df['unpaired']
df['total_available_freq'] = df['availableSpectrumPaired'] + df['availableSpectrumUnpaired']

df['per_MHz'] = df['reservePriceLocal']/df['total_alloted_freq']
df['per_MHz'] = df['per_MHz']/df['popCovered']

df['sold_per_MHz'] = df['headlinePriceLocal']/df['total_alloted_freq']
df['sold_per_MHz'] = df['sold_per_MHz']/df['popCovered']

data = df.loc[ (df['freqBand']=='700MHz')]

data['awardClassDescription'].unique()

data['licenceUse'].unique()

data1 = data.loc[data['awardClassDescription'] == 'Auction - Largely standard SMRA'] 
data2 = data.loc[data['awardClassDescription'] == 'Auction - Largely standard SMRA with augmented switching']
data =  pd.concat([data1,data2],axis = 0)
data['awardClassDescription'].unique()

data1 = data.loc[data['licenceUse'] == 'mobile']
data2 = data.loc[data['licenceUse'] == 'neutral']
data3 = data.loc[data['licenceUse'] == 'mobile and FWA']

data =  pd.concat([data1,data2,data3],axis = 0)
data['licenceUse'].unique()

data

reservePrice_data = data[['alpha3code','availableSpectrumPaired','availableSpectrumUnpaired','paired','unpaired','popCovered','reservePriceLocal','headlinePriceLocal','nBidders','date','total_alloted_freq','total_available_freq','licenceDuration','nationalLicence','per_MHz','sold_per_MHz']]
reservePrice_data.groupby(by = ['alpha3code','date']).mean()

reservePrice_data['licenceDuration'].unique()

# brazil = reservePrice_data[reservePrice_data['alpha3code'] == 'BRA']
# canada = reservePrice_data[reservePrice_data['alpha3code'] == 'CAN']
germany = reservePrice_data[reservePrice_data['alpha3code'] == 'DEU']
finland = reservePrice_data[reservePrice_data['alpha3code'] == 'FIN']
fiji = reservePrice_data[reservePrice_data['alpha3code'] == 'FJI']
# france = reservePrice_data[reservePrice_data['alpha3code'] == 'FRA']
# britian = reservePrice_data[reservePrice_data['alpha3code'] == 'GBR']
# newzeland = reservePrice_data[reservePrice_data['alpha3code'] == 'NZL']
taiwan = reservePrice_data[reservePrice_data['alpha3code'] == 'TWN']
usa = reservePrice_data[reservePrice_data['alpha3code'] == 'USA']

# germany['converted_reserve'] = (germany['per_MHz']*1.1113)/0.778122
# germany['converted_headline'] = (germany['sold_per_MHz']*1.1113)/0.778122

# finland['converted_reserve'] = (finland['per_MHz']*1.055)/0.880895
# finland['converted_headline'] = (finland['sold_per_MHz']*1.055)/0.880895

# fiji['converted_reserve'] = (fiji['per_MHz']*1.8821)/0.934711635112762
# fiji['converted_headline'] = (fiji['sold_per_MHz']*1.8821)/0.934711635112762

# taiwan['converted_reserve'] = (taiwan['per_MHz']*0.0337)/14.87
# taiwan['converted_headline'] = (taiwan['sold_per_MHz']*0.0337)/14.87

# usa['converted_reserve'] = (usa['per_MHz'])
# usa['converted_headline'] = (usa['sold_per_MHz'])

germany['converted_reserve'] = germany['per_MHz']/0.778122
germany['converted_headline'] = germany['sold_per_MHz']/0.778122

finland['converted_reserve'] = finland['per_MHz']/0.880895
finland['converted_headline'] = finland['sold_per_MHz']/0.880895

fiji['converted_reserve'] = fiji['per_MHz']/0.934711635112762
fiji['converted_headline'] = fiji['sold_per_MHz']/0.934711635112762

taiwan['converted_reserve'] = taiwan['per_MHz']/14.87
taiwan['converted_headline'] = taiwan['sold_per_MHz']/14.87

usa['converted_reserve'] = (usa['per_MHz'])
usa['converted_headline'] = (usa['sold_per_MHz'])

new_data = pd.concat([germany,finland,fiji,taiwan,usa],axis=0)
new_data.to_csv('final_data.csv')

date_data = pd.read_csv('/content/final_data.csv',
                                parse_dates=['date'],
                                index_col=['date'] )
date_data = date_data.sort_index()
date_data

date_data = date_data.drop(['Unnamed: 0'],axis=1)

date_data.groupby(by = ['alpha3code','date']).mean()

temp.to_csv('mean_country_data.csv')

date_data = date_data.dropna()

date_data.to_csv('international_prices.csv')

print("mean of the reserve data: ",date_data['converted_reserve'].mean())
print("mode of the reserve data: ",date_data['converted_reserve'].mode())

print("mean of the headline data: ",date_data['converted_headline'].mean())
print("mode of the headline data: ",date_data['converted_headline'].mode())

sorted_reserve = date_data.sort_values('converted_reserve')
sorted_headline = date_data.sort_values('converted_headline')

print("median of the reserve data: ",sorted_reserve['converted_reserve'].median())
print("median of the headline data: ",sorted_reserve['converted_headline'].median())

date = new_data_set
date = date.sort_values('date')
date

x_plot = plt.scatter(date['date'], date['converted_headline'], c='b')

plt.xlabel('time')
plt.ylabel('headline price')
# plt.hlines(y=0, xmin= 0, xmax=6)
# plt.plot(y_test,y_test,c = 'r',label = 'Ideal Model Prediction')
plt.legend(loc="upper left")
plt.title('Time vs headline price')

usa_plot = date.loc[ (date['alpha3code']=='USA')]

x_plot = plt.scatter(usa_plot['date'], usa_plot['converted_headline'], c='b')

plt.xlabel('time')
plt.ylabel('headline price')
# plt.hlines(y=0, xmin= 0, xmax=6)
# plt.plot(y_test,y_test,c = 'r',label = 'Ideal Model Prediction')
plt.legend(loc="upper left")
plt.title('Time vs headline price for only usa')

usa_data = {
  "date": ['2002-09-18','2003-06-13' ,'2005-07-26','2008-03-18' ,'2011-07-25'],
  "converted_headline": [0.000069,0.000140,0.027533,0.000847,0.033862 ]
}

#load data into a DataFrame object:
df = pd.DataFrame(usa_data)


x_plot = plt.scatter(df['date'], df['converted_headline'], c='b')

plt.xlabel('time')
plt.ylabel('headline price')
# plt.hlines(y=0, xmin= 0, xmax=6)
# plt.plot(y_test,y_test,c = 'r',label = 'Ideal Model Prediction')
plt.legend(loc="upper left")
plt.title('Time vs headline price for only usa')

mean_data = {
    "date": ['2002-09-18','2003-06-13' ,'2005-07-26','2008-03-18' ,'2011-07-25','2013-07-24','2013-10-30','2015-06-19','2016-11-24'],
  "converted_headline": [0.050903,0.035663,0.137664,0.887126,0.541789,0.055185,0.975021,0.264562,0.228251],
  "Country": ['USA','USA','USA','USA','USA','FJI','TWN','DEU','FIN'],
  "converted_reserve": [0.024990	,0.009981,0.016931,0.051121,0.048251,0.055117,0.663844,0.119000,0.227116],
  "mean": [0.54383123878024,0.54383123878024,0.54383123878024,0.54383123878024,0.54383123878024,0.54383123878024,0.54383123878024,0.54383123878024,0.54383123878024]
  
}
df = pd.DataFrame(mean_data)

import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data=df, x="date", y="converted_headline", hue="Country")
plt.title("Adjusted prices using PPP for previous 700MHz")
plt.xticks(rotation=60)

mean_data = {
    "date": ['2002-09-18','2003-06-13' ,'2005-07-26','2008-03-18' ,'2011-07-25','2013-07-24','2013-10-30','2015-06-19','2016-11-24','2016-10-1','2021-03-01'],
  "converted_headline": [0.050903,0.035663,0.137664,0.887126,0.541789,0.055185,0.975021,0.264562,0.228251,0,0],
  "Country": ['USA','USA','USA','USA','USA','FJI','TWN','DEU','FIN','IND','IND'],
  "converted_reserve": [0.024990	,0.009981,0.016931,0.051121,0.048251,0.055117,0.663844,0.119000,0.227116,3.517400299,1.891962344],
  "mean": [0.54383123878024,0.54383123878024,0.54383123878024,0.54383123878024,0.54383123878024,0.54383123878024,0.54383123878024,0.54383123878024,0.54383123878024,0.54383123878024,0.54383123878024]
  
}

df = pd.DataFrame(mean_data)

df

import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data=df, x="date", y="converted_reserve", hue="Country")
plt.title("Adjusted prices using PPP for previous 700MHz")
plt.xticks(rotation=60)

"""## X = reserve_price, n biddres ,paired+unpaired, liscense duration
## Y = winning bid price
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error ,mean_squared_error
from math import sqrt

X = date_data[['converted_reserve','licenceDuration','total_alloted_freq','nBidders','nationalLicence']]
y = date_data['converted_headline']

# X = X.dropna()
# y = y.dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42586)

"""## Linear Regression"""

reg = LinearRegression().fit(X_train, y_train)
print("regression's score for the trained data: ",reg.score(X_train,y_train))

print("regression's Coefficient: ",reg.coef_)
print("regression's Intercept: ",reg.intercept_) 

y_predicted = reg.predict(X_test)

accuracy = mean_absolute_error(y_test, y_predicted)
print("Accuracy of the model using mean_absolute_error: " , accuracy)

accuracy = mean_squared_error(y_test, y_predicted)
print("Accuracy of the model using mean_squared_error: " , accuracy)

accuracy = np.sqrt(mean_squared_error(y_test, y_predicted))
print("Accuracy of the model using root_mean_square_error: " , accuracy)

stats.coef_pval(reg, X_train, y_train)

stats.summary(reg, X_train, y_train)

importance = reg.coef_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

"""## Ridge Model"""

weights = [5,4,3,2,1]

clf = Ridge(alpha=1.0)
clf.fit(X_train, y_train)

y_predicted = clf.predict(X_test)

accuracy = mean_absolute_error(y_test, y_predicted)
print("Accuracy of the model using mean_absolute_error: " , accuracy)

accuracy = mean_squared_error(y_test, y_predicted)
print("Accuracy of the model using mean_squared_error: " , accuracy)

accuracy = np.sqrt(mean_squared_error(y_test, y_predicted))
print("Accuracy of the model using root_mean_square_error: " , accuracy)

print("Coefficients of the model: " ,clf.coef_)
print("Intercept of the model: ",clf.intercept_)

# !pip3 install regressors
from regressors import stats    
stats.coef_pval(clf, X_train, y_train)

stats.summary(clf, X_train, y_train)

importance = clf.coef_
# summarize feature importance
for i,v in enumerate(importance):
  if(i==0):
	  print('Feature: converted_reserve  Score: %.5f' % (v))
  if(i==1):
	  print('Feature: licenceDuration    Score: %.5f' % (v))
  if(i==2):
	  print('Feature: total_alloted_freq Score: %.5f' % (v))
  if(i==3):
	  print('Feature: nBidders           Score: %.5f' % (v))
  if(i==4):
	  print('Feature: nationalLicence    Score: %.5f' % (v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

"""## Residual plot of predictions and ground truth values."""

import matplotlib.pyplot as plt 
x_plot = plt.scatter(y_test, y_predicted, c='b',label = 'Model prediction')

plt.xlabel('Ground Truth')
plt.ylabel('Predicted final bid')
# plt.hlines(y=0, xmin= 0, xmax=6)
plt.plot(y_test,y_test,c = 'r',label = 'Ideal Model Prediction')
plt.legend(loc="upper left")
plt.title('Ground Truth vs predicted values')

"""## Correlation values between Independent variables and dependent variables."""

# date_data['r'] = date_data['converted_reserve']*date_data['total_alloted_freq']
# date_data['h'] = date_data['converted_headline']*date_data['total_alloted_freq']

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 6))

heatmap = sns.heatmap(date_data[['converted_reserve','licenceDuration','total_alloted_freq','total_available_freq','nBidders','converted_headline']].corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':20}, pad=20);

"""## Polynomial Regression"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

Input=[('polynomial',PolynomialFeatures(degree=3)),('modal',Ridge(alpha=1.0))]

pipe=Pipeline(Input)
pipe.fit(X_train,y_train)

predictions = pipe.predict(X_test)

accuracy = mean_absolute_error(y_test, predictions)
print("Accuracy of the model using mean_absolute_error: " , accuracy)

accuracy = mean_squared_error(y_test, predictions)
print("Accuracy of the model using mean_squared_error: " , accuracy)

print('RMSE for Polynomial Regression=>',np.sqrt(mean_squared_error(y_test,predictions)))

"""
## Obtaining Minima by randomizing """

r_mean_ = []
r_squared_ = []
for i in range(100000):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

  # reg = LinearRegression().fit(X_train, y_train)
  reg = Ridge(alpha=1.0).fit(X_train,y_train)

  y_predicted = reg.predict(X_test)
  mean_accuracy = mean_absolute_error(y_test, y_predicted)
  # print("Accuracy of the model using mean_absolute_error: " , accuracy)
  
  squared_accuracy = mean_squared_error(y_test, y_predicted)
  # print("Accuracy of the model using mean_squared_error: " , accuracy)

  r_mean_.append(mean_accuracy)
  r_squared_.append(squared_accuracy)

# print("regression's score for the trained data: ",reg.score(X_train,y_train))

# print("regression's Coefficient: ",reg.coef_)
# print("regression's Intercept: ",reg.intercept_)

min_val = min(r_squared_)
for i in range(100000):
  if(r_squared_[i] == min_val):
    state  = i
    break
state

