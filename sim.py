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


regressor = LinearRegression()  
regressor.fit(np.array(x).reshape(-1, 1), y)

lastDay = byStateDF[byStateDF.index=='California'].columns[-1]
lastDay = pd.to_datetime(lastDay)

predRange = pd.to_datetime(pd.date_range(lastDay, periods=10).tolist())
predRange = [date.toordinal() for date in predRange]
predRange = np.array(predRange)

yPred = regressor.predict(predRange.reshape(-1, 1))

x = [dt.fromordinal(date) for date in x]
predRange = [dt.fromordinal(date) for date in predRange]

plt.figure(figsize=(22, 10))
plt.plot(x, np.exp(y))
plt.plot(predRange, np.exp(yPred))
plt.xticks(rotation='vertical')
plt.show()



# %%


regressor.coef_[0,0]



#%%

nyCountyPopulation = 1664727
nyDF = confDF[(confDF['Province_State']=='New York') & (confDF['Admin2']=='New York')]

x = pd.to_datetime(nyDF.iloc[:,15:].transpose().index.tolist())
x = [date.toordinal() for date in x]
x = np.array(x)


y = nyDF.iloc[:,15:].transpose()
y = np.array(y).reshape(-1)



# %%

# y[-1]

x[-1]

# %%


def initialInfect(city, initialCount, dateToday, recoveryTime):

    infectedList = np.arange(len(city)).tolist()
    infected = random.sample(infectedList, initialCount)
    city.loc[city.index.isin(infected),'infected'] = True
    city.loc[city.index.isin(infected),'recovery_day'] = dateToday + recoveryTime



    return (city)

date = 
city = initialInfect(city, INITIALLY_AFFECTED, x[-1], DAYS_TO_RECOVER)

#%%


# asdf = np.arange(nyCountyPopulation).tolist()
# asdf = random.sample(asdf, 2)
# city.loc[asdf,'infected']

len(city[city['infected']==True])

# city = pd.DataFrame(data={'id': np.arange(10), 
#                     'infected': False, 'infection_day': None, 'recovered': False, 'quarantined': False})
# city = city.set_index('id')

# samp = random.sample(np.arange(len(city)).tolist(), 3)

# city.loc[city.index.isin(samp),'infected'] = True

# city

#%%



DAYS = 180
POPULATION = 10 # nyCountyPopulation
Rnought = 2
DAYS_TO_RECOVER = 10
INITIALLY_AFFECTED = 3

city = pd.DataFrame(data={'id': np.arange(POPULATION), 
                    'infected': False, 'recovery_day': None, 'recovered': False, 'quarantined': False})
city = city.set_index('id')








firstCases = city.sample(INITIALLY_AFFECTED, replace=False)
city.loc[firstCases.index, 'infected'] = True
city.loc[firstCases.index, 'recovery_day'] = DAYS_TO_RECOVER

stat_active_cases = [INITIALLY_AFFECTED]
stat_recovered = [0]

for today in range(1, DAYS):
    # Mark recovered people
    city.loc[city['recovery_day'] == today, 'recovered'] = True
    city.loc[city['recovery_day'] == today, 'infected'] = False

    spreadingPeople = city[ (city['infected'] == True)]
    totalCasesToday = round(len(spreadingPeople) * SPREAD_FACTOR)
    casesToday = city.sample(totalCasesToday, replace=True)
    # Ignore already infected or recovered people
    casesToday = casesToday[ (casesToday['infected'] == False) & (casesToday['recovered'] == False) ]
    # Mark the new cases as infected
    city.loc[casesToday.index, 'infected'] = True
    city.loc[casesToday.index, 'recovery_day'] = today + DAYS_TO_RECOVER

    stat_active_cases.append(len(city[city['infected'] == True]))
    stat_recovered.append(len(city[city['recovered'] == True]))
    # if today >= 5:
    #     SPREAD_FACTOR = 1
    # if today >= 10:
    #     SPREAD_FACTOR = 0.1


fig = plt.figure(figsize=(16, 8))

plt.bar(x=np.arange(DAYS), height=stat_active_cases, color="red")
plt.text(145, 90000, f"SPREAD_FACTOR = {SPREAD_FACTOR}", fontsize=14)
plt.show()


# %%


city

# %%
