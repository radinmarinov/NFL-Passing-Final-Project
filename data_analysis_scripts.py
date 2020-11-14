# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:22:44 2020

@author: Radin
"""

import pandas as pd

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
#plays[pd.isna(plays.typeDropback)]

#####################
# Feature Engineering
#####################

# long (>=7 yards on a down)
plays['longYardsToGo'] = plays.yardsToGo >= 7

# Redzone (absoluteYardlineNumber <= 20)
plays['redzone'] = plays.absoluteYardlineNumber <= 20

# Incomplete pass
plays['incompletePass'] = (plays.playDescription.str.contains("incomplete")) & (plays.playResult == 0)
