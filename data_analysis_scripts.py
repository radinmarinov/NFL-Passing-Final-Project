# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:22:44 2020

@author: Radin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import data
plays = pd.read_csv("plays.csv")

###############
# Data cleaning
###############

# Remove plays with penalties and plays with no offense formation
plays = plays[(pd.isna(plays.penaltyCodes)) & \
              (plays.playType != 'play_type_unknown') & \
              (~pd.isna(plays.offenseFormation))]
#check_nulls = [{col, sum(pd.isna(plays[col]))} for col in plays.columns]

# 5 entries missing partial data (typeDropback, scores, game clock, abs yardline)
plays = plays[~pd.isna(plays.typeDropback)]

#####################
# Feature Engineering
#####################

# long (>=7 yards on a down)
plays['longYardsToGo'] = plays.yardsToGo >= 7

# Redzone (absoluteYardlineNumber <= 20)
plays['redzone'] = plays.absoluteYardlineNumber <= 20

# Incomplete pass
plays['incompletePass'] = (plays.playDescription.str.contains("incomplete")) & (plays.playResult == 0)

# First down
plays['firstDown'] = plays.playResult >= plays.yardsToGo

# Touchdown
plays['touchDown'] = (plays.playResult >= plays.absoluteYardlineNumber)

# Add new column for number of players per each position for offense, defense
# Offense
accum = []

for row in plays['personnelO']:
    rowLst = row.split(',')
    vals = {}

    for el in rowLst:
        elLst = el.split(' ')

        if len(elLst) == 2:
            vals["num" + elLst[1] + "offense"] = int(elLst[0])
        else:
            vals["num" + elLst[2] + "offense"] = int(elLst[1])

    accum.append(vals)

tempDF = pd.DataFrame(accum)
tempDF = tempDF.fillna(0).astype(int)
plays = pd.concat([plays.reset_index(drop=True), tempDF.reset_index(drop=True)], axis=1)

# Defense
accum2 = []

for row in plays['personnelD']:
    rowLst = row.split(',')
    vals = {}

    for el in rowLst:
        elLst = el.split(' ')

        if len(elLst) == 2:
            vals["num" + elLst[1] + "defense"] = int(elLst[0])
        else:
            vals["num" + elLst[2] + "defense"] = int(elLst[1])

    accum2.append(vals)

tempDF2 = pd.DataFrame(accum2)
tempDF2 = tempDF2.fillna(0).astype(int)
plays = pd.concat([plays.reset_index(drop=True), tempDF2.reset_index(drop=True)], axis=1)

# cleaning clock data
time_accum_quarter = []
time_accum_overall = []

temp = plays['gameClock'].replace(np.nan, "00:00:00", regex=True)

for idx, t in enumerate(temp):
    m, s, ms = str(t).split(":")
    secs = int(m) * 60 + int(s)
    time_accum_quarter.append(secs)
    time_accum_overall.append(secs + (15*60* max(4-plays['quarter'].iloc[idx], 0)))

plays['gameClockSecsQuarter'] = time_accum_quarter
plays['gameClockSecsOverall'] = time_accum_overall

# Last two minutes of either half
plays['lastTwoMinutes'] = (plays['gameClockSecsQuarter'] <= 120) & (plays['quarter'] % 2 == 0)

# Exporting updated df
plays.to_csv('updated_plays.csv')

##########
# Analysis
##########
combos = ['offenseFormation'
          ,'personnelO'
          ,'personnelD'
          ,'defendersInTheBox'
          , ['offenseFormation'
             , 'personnelD'
             ]
          , ['offenseFormation'
             , 'defendersInTheBox'
             ]
          , ['offenseFormation'
             ,'personnelD'
             , 'defendersInTheBox'
             ]
          , ['offenseFormation'
             ,'personnelO'
             ,'personnelD'
             ,'defendersInTheBox'
             ]
          ]
resultVars = ['playResult'
              ,'firstDown']
analysis = {}
minCount = 10
for combo in combos:
    for resultVar in resultVars:
        agg = plays.groupby(combo)[resultVar].agg(['mean', 'count'])
        analysis[resultVar + "".join(combo)] = agg[agg['count'] >= minCount]

##########
# Graphing
##########

playResultoffenseFormation = analysis['playResultoffenseFormation']
plt.figure(1)
playResultoffenseFormation['mean'].plot(kind='bar')
plt.xlabel('Offense Formation')
plt.ylabel('Mean of Play Result')
plt.title('Offense Formation v. Mean of Play Result')
plt.savefig('graphs/OF_PR_M')

plt.figure(2)
playResultoffenseFormation['count'].plot(kind='bar')
plt.xlabel('Offense Formation')
plt.ylabel('Count of Play Result')
plt.title('Offense Formation v. Count of Play Result')
plt.savefig('graphs/OF_PR_C')

firstDownoffenseFormation = analysis['firstDownoffenseFormation']
plt.figure(3)
firstDownoffenseFormation['mean'].plot(kind='bar')
plt.xlabel('Offense Formation')
plt.ylabel('Mean of First Down')
plt.title('Offense Formation v. Mean of First Down')
plt.savefig('graphs/OF_FD_M')

plt.figure(4)
firstDownoffenseFormation['count'].plot(kind='bar')
plt.xlabel('Offense Formation')
plt.ylabel('Count of First Down')
plt.title('Offense Formation v. Count of First Down')
plt.savefig('graphs/OF_FD_C')



