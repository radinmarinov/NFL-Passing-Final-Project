import pandas as pd

updated_plays = pd.read_csv("updated_plays.csv")
cleaned_plays = updated_plays
cleaned_plays.drop(['gameId', 'playId', 'playDescription', 'personnelO', 'personnelD', 'penaltyCodes',
                    'penaltyJerseyNumbers', 'isDefensivePI', 'gameClock', 'possessionTeam',
                    'yardlineSide', 'touchDown'], axis=1, inplace=True)

feature_matrix = cleaned_plays.drop(['playType', 'passResult', 'offensePlayResult',
                                     'playResult', 'epa', 'incompletePass', 'incompletePass', 'firstDown'],
                                    axis=1, inplace=False)



