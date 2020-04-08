import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime as dt
from datetime import timedelta

import random
import time

import os



# %%


def initialInfect(city, initialCount, dateToday, recoveryTime):

    infectedList = np.arange(len(city)).tolist()
    infected = random.sample(infectedList, initialCount)
    city.loc[city.index.isin(infected),'infected'] = True
    city.loc[city.index.isin(infected),'recovery_day'] = dateToday + recoveryTime

    return (city)


def contactSim(city, recoveryTime, date, lambdaPoiss=5, probInfection=.05):

    newInfections = 0 
    contagiousList = city[(city['infected']==True) & (city['recovered']==False) & 
                        (city['quarantined']==False)].index.tolist()

    # contactRate = int(np.random.poisson(lambdaPoiss,1))
    suceptableList = city.index.tolist()

    for i in contagiousList:
        contactRate = int(np.random.poisson(lambdaPoiss,1))
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

    return city, len(toRecover)


#-----------------------------------------MODEL VARIABLES--------------------------------------

# ordinal number of April 6th, 2020
dayStart = 737521

# number of simulations days
simDuration = 180

# population of city
population = 20000 

# how long does the disease last?
infectionDuration = 15

# how many initial infections
initialInfections = 1

# number of infections before quarantine starts
quarantineInfectionStart = 1500

# days quarantine should last
quarantineDuration = 60

#----------------------------------------------------------------------------------------------

start = time.clock()
segmentStart = time.clock()

susceptible = population - initialInfections

day0summary = {'new':[initialInfections], 'susceptible':[susceptible], 
               'infected':[initialInfections], 'recovered':[0]}
summary = pd.DataFrame(day0summary, index = [dayStart])

city = pd.DataFrame(data={'id': np.arange(population), 'infected': False, 'recovery_day': None, 
                          'recovered': False, 'quarantined': False, 'detected': False, 'numInfected': 0})
city = city.set_index('id')

new = 0

city = initialInfect(city, initialInfections, dayStart, infectionDuration)

quarantineDayStart = 0
quarantineDayEnd = 0
totalRecovered = 0
totalInfected = initialInfections
totalSusceptible = population - initialInfections

for today in range(1+dayStart, simDuration+dayStart):

    if (len(city[city.infected==True]) < quarantineInfectionStart):
        # print('base')
        # pre quarantine
        city, newInfections = contactSim(city, infectionDuration, today)
        quarantineDayStart = today
        quarantineDayEnd = today

    elif ((len(city[city.infected==True]) > quarantineInfectionStart) & 
          (quarantineDayStart + quarantineDuration > today)):
        # print('in quarantine')
        # in quarantine 
        city, newInfections = contactSim(city, infectionDuration, today, lambdaPoiss=1.3)
        quarantineDayEnd = today
    
    elif ((len(city[city.infected==True]) > quarantineInfectionStart) & 
          (quarantineDayStart + quarantineDuration < today)):
        # print('end quarantine')
        # after quarantine is over
        city, newInfections = contactSim(city, infectionDuration, today, lambdaPoiss=2.8)


    city, newRecovered = convertRecovered(city, today)
    totalRecovered = len(city[city['recovered']==True])
    totalInfected = len(city[(city['infected']==True) & (city['recovered']==False)])
    totalSusceptible = population - totalRecovered - totalInfected

    daySummary = {'new':newInfections, 'susceptible':[totalSusceptible], 
                  'infected':[totalInfected], 'recovered':[totalRecovered]}
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
print("Rnought:" , round(city[city['infected']==True].numInfected.mean(), 2))



dateList = [dt.fromordinal(date) for date in summary.index.tolist()]
dateRename = dict(zip(summary.index.tolist(), dateList))

summary = summary.rename(index=dateRename)

SIR_path = os.path.join(os.getcwd(), 'simPlots', 'SIR.png')

plt.figure(figsize=(22, 10))
plt.plot(summary.susceptible.rolling(window=7).mean(), label='Susceptible', color='blue')
plt.plot(summary.infected.rolling(window=7).mean(), label='Infected', color='red')
plt.plot(summary.recovered.rolling(window=7).mean(), label='Recovered', color='green')
plt.title('SIR Simulation')
plt.axvline(dt.fromordinal(quarantineDayStart), color='k', label='Quarantine Start')
plt.axvline(dt.fromordinal(quarantineDayEnd), color='k', label='Quarantine End')
plt.legend(loc='left')
plt.ylim(top=population+700)
plt.xticks(rotation='vertical')
plt.savefig(SIR_path)
plt.show()


new_path = os.path.join(os.getcwd(), 'simPlots', 'new.png')

plt.figure(figsize=(22, 10))
plt.plot(summary.new.rolling(window=7).mean())
plt.title('New Infections per Day')
plt.axvline(dt.fromordinal(quarantineDayStart), color='k', label='Quarantine Start')
plt.axvline(dt.fromordinal(quarantineDayEnd), color='k', label='Quarantine End')
plt.xticks(rotation='vertical')
plt.legend(loc='left')
plt.savefig(new_path)
plt.show()


infectHist_path = os.path.join(os.getcwd(), 'simPlots', 'infectionHist.png')

plt.figure(figsize=(22, 10))
plt.hist(city[city['infected']==True].numInfected, bins=city.numInfected.max())
plt.title('Num Infected Distribution')
plt.xticks(rotation='vertical')
plt.savefig(infectHist_path)
plt.show()

#%%
