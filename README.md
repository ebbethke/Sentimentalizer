# Sentimentalizer
Python pipeline for applying VADER (nltk) to student group free response items to understand sentiment within and between groups

## Dependencies:
- NLTK
- Pandas
- Numpy
- Matplotlib
- SciPy

## How it works
The script will perform the following tasks, in order:
1. Read in the CSV file with the Google Forms data in it. 
1. Split the responses up by cutoff dates (to divide into cohorts)
1. Edit the response text to realign any questions asked with negative connotation
1. Compute VADER sentiment scores and break out 'pos', 'neu', 'neg', 'compound' for desired questions
1. Correct for the tendency of VADER to assign overly positive scores
1. Perform cohort analysis, checking Welch's t-test and Mann-Whitney U for each pair of cohorts' corrected VADER score
1. Plot corrected VADER scores as Team Average (X) vs Individual (Y) by cohort, each cohort gets a subplot.
1. Saves out summary tables and the composite figure to file

Key lines where the user may want to make stylistic or functional changes are marked with: *#!#*
