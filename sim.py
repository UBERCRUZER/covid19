import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from datetime import datetime as dt
from datetime import timedelta

from dateutil.parser import parse

import random

#%%



# confirmed_path = os.path.join(os.getcwd(), 'csse_covid_19_data', 'csse_covid_19_time_series', 'time_series_covid19_confirmed_US.csv')
confirmed_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
confDF = pd.read_csv(confirmed_URL)

#%%
byStateDF = confDF.groupby('Province_State').sum()

states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas',
       'California', 'Colorado', 'Connecticut', 'Delaware',
       'District of Columbia', 'Florida', 'Georgia',
       'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
       'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan',
       'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
       'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
       'North Carolina', 'North Dakota', 'Ohio',
       'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island',
       'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah',
       'Vermont', 'Virginia', 'Washington', 'West Virginia',
       'Wisconsin', 'Wyoming']

byStateDF = byStateDF.loc[states]

# %%

start = -10

x = pd.to_datetime(byStateDF[byStateDF.index=='California'].iloc[:,start:].transpose().index.tolist())
x = [date.toordinal() for date in x]

y = np.log(byStateDF[byStateDF.index=='California'].iloc[:,start:].transpose())

plt.figure(figsize=(22, 10))
plt.plot(x, y)
plt.xticks(rotation='vertical')
plt.show()


# %%


regressorConf = LinearRegression()  
regressorConf.fit(np.array(x).reshape(-1, 1), y)

lastDay = byStateDF[byStateDF.index=='California'].columns[-1]
lastDay = pd.to_datetime(lastDay)

predRange = pd.to_datetime(pd.date_range(lastDay, periods=10).tolist())
predRange = [dt.toordinal(date) for date in predRange]
predRange = np.array(predRange)

yPred = regressorConf.predict(predRange.reshape(-1, 1))

x = [dt.fromordinal(date) for date in x]
predRange = [dt.fromordinal(date) for date in predRange]

plt.figure(figsize=(22, 10))
plt.plot(x, np.exp(y))
plt.plot(predRange, np.exp(yPred))
plt.xticks(rotation='vertical')
plt.show()



# %%


regressorConf.coef_[0,0]



#%%

nyCountyPopulation = 1664727
nyDF = confDF[(confDF['Province_State']=='New York') & (confDF['Admin2']=='New York')]

x = pd.to_datetime(nyDF.iloc[:,15:].transpose().index.tolist())
x = [dt.toordinal(date) for date in x]
x = np.array(x)


y = nyDF.iloc[:,15:].transpose()
y = np.array(y).reshape(-1)


# %%


def initialInfect(city, initialCount, dateToday, recoveryTime):

    infectedList = np.arange(len(city)).tolist()
    infected = random.sample(infectedList, initialCount)
    city.loc[city.index.isin(infected),'infected'] = True
    city.loc[city.index.isin(infected),'recovery_day'] = dateToday + recoveryTime

    return (city)


def contactSim(city, recoveryTime, date, contactRate=2, probInfection=.1):

    newInfections = 0 
    contagiousList = city[(city['infected']==True) & (city['recovered']==False) & 
                        (city['quarantined']==False)].index.tolist()
    suceptableList = city[(city['infected']==False) & (city['recovered']==False)].index.tolist()

    for i in contagiousList:
        if (len(suceptableList) < contactRate):
            contactRate = len(suceptableList)
        potentialList = random.sample(suceptableList, contactRate)
        for j in potentialList:
            rand = random.random()
            if (probInfection > rand):
                city.loc[j, 'infected'] = True
                city.loc[j, 'recovery_day'] = date + recoveryTime
                # print(j, rand, date)
                
                newInfections = newInfections + 1
    # print(newInfections)
    return city, newInfections

def convertRecovered(city, date):
    toRecover = city[city['recovery_day'] == date].index.tolist()
    city.loc[toRecover, 'recovered'] = True

    return (city)




#%%


dayStart = x[-1]

simDuration = 40
POPULATION = 1000 # nyCountyPopulation
SPREAD_FACTOR = 1
DAYS_TO_RECOVER = 10
INITIALLY_AFFECTED = 1

day0summary = {'new':[INITIALLY_AFFECTED], 'total':[INITIALLY_AFFECTED]}
summary = pd.DataFrame(day0summary, index = [dayStart])

city = pd.DataFrame(data={'id': np.arange(POPULATION), 
                    'infected': False, 'recovery_day': None, 'recovered': False, 'quarantined': False, 'detected': False})
city = city.set_index('id')

new = 0

city = initialInfect(city, INITIALLY_AFFECTED, x[-1], DAYS_TO_RECOVER)

for today in range(1+dayStart, simDuration+dayStart):
    city, newInfections = contactSim(city, DAYS_TO_RECOVER, today)
    city = convertRecovered(city, today)
    total = len(city[city['infected']==True])

    daySummary = {'new':newInfections, 'total':total}
    newDay = pd.DataFrame(daySummary, index = [today])
    summary = summary.append(newDay)
    # print(today)




#%%


dateList = [dt.fromordinal(date) for date in summary.index.tolist()]
dateRename = dict(zip(summary.index.tolist(), dateList))

summary = summary.rename(index=dateRename)

summary.total.plot()

#%%




# %%
