import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from datetime import datetime as dt
from datetime import timedelta

from dateutil.parser import parse

import random
import time



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

    suceptableList = city.index.tolist()


    for i in contagiousList:
        suceptableList.remove(i)
        if (len(suceptableList) < contactRate):
            contactRate = len(suceptableList)
        potentialList = random.sample(suceptableList, contactRate)
        for j in potentialList:
            rand = random.random()
            if ((probInfection > rand) & (city.loc[j, 'recovered'] == False)):
                city.loc[j, 'infected'] = True
                city.loc[j, 'recovery_day'] = date + recoveryTime
                # print(j, rand, date)
                city.loc[i,'numInfected'] = city.loc[i,'numInfected'] + 1
                newInfections = newInfections + 1
    # print(newInfections)
        suceptableList.append(i)
    return city, newInfections

def convertRecovered(city, date):
    toRecover = city[city['recovery_day'] == date].index.tolist()
    city.loc[toRecover, 'recovered'] = True

    return (city)



start = time.clock()
segmentStart = time.clock()

dayStart = 737521

simDuration = 180
POPULATION = 10000 # nyCountyPopulation
SPREAD_FACTOR = 1
DAYS_TO_RECOVER = 10
INITIALLY_AFFECTED = 1

day0summary = {'new':[INITIALLY_AFFECTED], 'total':[INITIALLY_AFFECTED]}
summary = pd.DataFrame(day0summary, index = [dayStart])

city = pd.DataFrame(data={'id': np.arange(POPULATION), 'infected': False, 'recovery_day': None, 
                          'recovered': False, 'quarantined': False, 'detected': False, 'numInfected': 0})
city = city.set_index('id')

new = 0

city = initialInfect(city, INITIALLY_AFFECTED, dayStart, DAYS_TO_RECOVER)

for today in range(1+dayStart, simDuration+dayStart):
    city, newInfections = contactSim(city, DAYS_TO_RECOVER, today)
    city = convertRecovered(city, today)
    total = len(city[city['infected']==True])

    daySummary = {'new':newInfections, 'total':total}
    newDay = pd.DataFrame(daySummary, index = [today])
    summary = summary.append(newDay)

    if today % 10 == 0:
        segmentTime = time.clock() - segmentStart
        print("Simulating day", today-dayStart+1)
        print("Segment Time:", round(segmentTime, 2), "seconds")
        print("")
        segmentStart = time.clock()
    
totalTime = time.clock() - start
print("Total Elapsed Time:", round(totalTime, 2), "seconds")
print("")
print("Rnought:" , round(city.numInfected.mean(), 2))



dateList = [dt.fromordinal(date) for date in summary.index.tolist()]
dateRename = dict(zip(summary.index.tolist(), dateList))

summary = summary.rename(index=dateRename)


plt.figure(figsize=(22, 10))
plt.plot(summary.new)
plt.title('New Infections per Day')
plt.xticks(rotation='vertical')
plt.show()

plt.figure(figsize=(22, 10))
plt.plot(summary.total)
plt.title('Total Infections per Day')
plt.xticks(rotation='vertical')
plt.show()

plt.figure(figsize=(22, 10))
plt.hist(city.numInfected)
plt.title('Rnought')
plt.xticks(rotation='vertical')
plt.show()

#%%

city.numInfected.mean()

# %%
