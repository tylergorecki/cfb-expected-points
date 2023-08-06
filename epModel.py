# Expected points model
# Goal is to create more efficient version of R code

import pandas as pd
import numpy as np

fb19 = pd.read_csv("/Users/tylergorecki/Desktop/Past UVA Semesters/2022 FALL/STAT 4800/2019 PFF All Plays.csv")

def expoints(field_pos, down, togo, poss):
    init_field_pos = field_pos
    gained = yardsGained(field_pos, down, togo)
    field_pos = init_field_pos+gained

    first = gained >= togo

    if (down < 4):
        if (field_pos >= 100):
            touchdown(poss)
        elif (field_pos < 0):
            safety(poss)
        else:
            if (first): # first down
                down = 1
                yardsToGoaline = 100-field_pos
                togo = np.where(yardsToGoaline < 10, yardsToGoaline, 10)
            else:
                down = down+1
                togo = togo-gained
            expoints(field_pos, down, togo, poss)

    else:
        print('need to add more')

def yardsGained(field_pos, down, togo):
    return(0)

def touchdown(poss):
    return(np.where(poss == 1, 7, -7))

def safety(poss):
    return(np.where(poss == 1, -2, 2))

def fieldgoal(poss, init_field_pos):
    return(np.where(poss == 1, 3, -3))

def punt(poss, init_field_pos):
    return(0)