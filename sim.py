import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import datetime
from dateutil.parser import parse

#%%

# PLAN: going to augment code from https://colab.research.google.com/drive/1WGgVAFRnW0qvP_KpehYJs892MlWXwV86 
# need to calculate spread factor based on previous track record, then calculate NEW spread factor from "social 
# distancing"

DAYS = 180
POPULATION = 7000000
SPREAD_FACTOR = 1
DAYS_TO_RECOVER = 10
INITIALLY_AFFECTED = 4

city = pd.DataFrame(data={'id': np.arange(POPULATION), 'infected': False, 'recovery_day': None, 'recovered': False})
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


#%%

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16, 8))

plt.bar(x=np.arange(DAYS), height=stat_active_cases, color="red")
plt.text(145, 90000, f"SPREAD_FACTOR = {SPREAD_FACTOR}", fontsize=14)
plt.show()

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

y = np.log(byStateDF[byStateDF.index=='California'].iloc[:,start:].transpose().tolist())

plt.figure(figsize=(22, 10))
plt.plot(x, y)
plt.xticks(rotation='vertical')
plt.show()


# %%


regressor = LinearRegression()  
regressor.fit(np.array(x).reshape(-1, 1), y)


predRange = pd.to_datetime(pd.date_range(x[-1]+ datetime.timedelta(days=1), periods=10).tolist())
predRange = np.array(predRange).reshape(-1, 1)

yPred = regressor.predict(predRange)



# %%


#%%

byStateDF[byStateDF.index=='California'].iloc[:,start:].columns


# %%
