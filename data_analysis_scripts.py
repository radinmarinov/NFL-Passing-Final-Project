# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:22:44 2020

@author: Radin
"""

import pandas as pd
plays = pd.read_csv("plays.csv")

# Data cleaning
nulls = [{col, sum(pd.isnull(plays[col]))} for col in plays.columns]
# null yardlineSide means LOS is at midfield
# null offenseFormation should be removed from the data
# null personnelO means offsetting penalties, remove
# null defendersInTheBox means special teams pass or penalty
# null numberOfPassRushers means penalty
