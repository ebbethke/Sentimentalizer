import nltk
import pandas as pd
import numpy as np
from itertools import combinations as combos
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import ttest_ind, mannwhitneyu

# Fetch files for sentiment analysis; checks if needed.
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()


def split_by_date(df, dates=None, col_to_split="Timestamp", names=None):
	"""Function: split_by_date(df (Type: pandas.DataFrame), dates (Type: list or None)
	
	Parameters:
		df (pd.DataFrame): 		DataFrame containing survey responses with 
								columns containing questions, rows containing
								individual responses.
			
		dates (list or None): 	List of datetime.datetime objects with dates
								splitting up the responses, e.g. from semester
								to semester, month to month, or year to year.
								If 'None', use first day of current month.
		
		col_to_split (string):	String matching the column name of the dataframe 
								containing datetime data.
		
		names (list or None):	List of cohort names for the split DataFrames. 
								Must be of length len(dates)+1. If 'None', 
								str integers (0 indexed) will be used as keys
		
								
	Returns:
		dict of pd.Dataframes split at dates, where key=name, 
	
	"""
	if dates is None:
		dates = [datetime.today().replace(day=1).strftime('%m/%d/%Y %H:%M:%S')]
	else:
		dates.sort()
	
	if names is None:
		names = [str(k) for k in range(len(dates)+1)]
	else:
		try:
			argsmatch = len(names) == len(dates)+1
		
		except TypeError: 
			raise TypeError("Both args 'names' and 'dates' must be list or None.")
		
		else:
			if argsmatch is False:
				raise ValueError(f"length of cohorts must be length of dates+1: cohorts = {len(names)}, dates = {len(dates)}")
	
	split_dfs = {}
	# Get splits for dates less than each date in dates
	for ind,cutoff in enumerate(dates):
		split = df[df[col_to_split] < cutoff]
		split_dfs[names[ind]] = split.fillna('')
		df = df[df[col_to_split] >= cutoff]
	# Add final split for dates after final date cutoff
	final = df[df[col_to_split] >= dates[-1]]
	split_dfs[names[-1]] = final.fillna('')
	
	return split_dfs

# Pre-process the data to improve results. Questions were negatively positioned (What do you want to improve)
# So simple answers of No, None, Nothing should be interpreted as positive, not negative.
def realign_negatives(series):
	"""Function realign_negatives(pd.Series)
	
	Parameters:
		series (pandas.Series): 	A pandas series representing a question from 
									the survey. 
		
	Returns:
		pd.Series object with short or negative-leading response edited to reflect
			more positively in VADER analysis
			
	Example Use Cases:
		Question: "Do you feel your team is struggling?"
		Answers: Series("No, things are great!", 
						"Yes, everything is very bad",
						"Nope",
						"I don't know, I thinks we're fine")
		
		Because of the question wording, answers 1,3, and 4 should
		reasonably be interpreted as positive or neutral, but will 
		score negative. Instead of simply inverting the score 
		(negatives go positive), for longer answers it is better
		to adjust the context of the response to allow the rest of 
		the response to stand on its own.
	"""
	# Remove trailing or leading whitespace (including newlines)
	series = series.str.strip()
	
	# Take care of short answers like No, no, nope, N/A, n/a.
	# Because there is not enough context, code these as pure neutral ("Neutral" scores 0.0)
	series = series.apply(lambda s: "Neutral" if len(s) < 7 else s)
	# Handle "Nothing at this..." comments
	series = series.apply(lambda s: "Neutral" if s[:7].lower() == "nothing" else s)
	# Handle "Not at this..." comments
	series = series.apply(lambda s: "Neutral" if s[:3].lower() == "not" else s)
	# Handle "Nope, ..." comments
	series = series.apply(lambda s: "Neutral" if s[:4].lower() == "nope" else s)
	# Handle longer comments starting with "No, ..."
	series = series.apply(lambda s: s[3:] if s[:2].lower() == "no" else s)
	
	return series

def correct_partial_negative_scores(df, cutoff=0, prefix=""):
	"""Function correct_partial_negative_scores(pd.DataFrame, float)
	
	Parameters:
		df (pandas.DataFrame): 		A pandas DataFrame containing the following
									columns:
										- 'neg': negative VADER component
										- 'pos': positive VADER component
										- 'neu': neutral  VADER component
										- 'compound': VADER compound score 
		
		cutoff (float [0.-1.]):		Cutoff value for the negative component. 
									responses with a negative component (0-1)
									greater than the cutoff will be flipped
									negative if their compound is positive.
									
									Default = 0.0
		
		prefix (str):				A string prefix used to index the question 
									to correct, prepended pefore each (neg,
									neu, pos, compound).
									
									Default = ""
		
	Returns:
		pd.DataFrame object with mild negative compound results to be rescaled 
		as negative	if the compound score was overwhelmed by neutral or positive.
			
	Example Use Cases:
		Especially for longer answers, students tend not to rant purely
		negative words/phrases, but will qualify and cage a bit. We find
		that detecting the negative phrasing at all often warrants scoring
		the sentiment lower than what VADER produces, so this function
		accounts for this by simply flipping the sign of compound scores
		where the negative component was significantly present.
	"""
	if cutoff < 0 or cutoff > 1:
		raise ValueError(f"Arg: 'cutoff' must be in range [0. - 1.]. Current value: {cutoff}")
	
	df.loc[:,prefix+'compound'] = df.apply(lambda x: -x[prefix+'compound'] if (x[prefix+'neg'] > cutoff) and (np.sign(x[prefix+'compound'])>0) else x[prefix+'compound'], axis=1)
	return df
	
	
## Pre-Process the survey data!

# Provide simplified names for columns reading in from the survey, from left-to-right.
# If using Google forms, include the timestamp and email (if emails collected) or columns will not be aligned!
col_names = ["Timestamp", "Name", "Team #", "Dynamic Likert", "Contract Likert", "Improvements", "Other Comments"] #!#
with open('C:/Users/Eliot/Documents/UIUC/Fall2023/Bioe400/ASEE/TeamContracts/TeamContractSentimentSpring24.csv', 'r', encoding="UTF-8") as inf: #!#
    reviews = pd.read_csv(inf, names=col_names, skiprows=1)
# Convert timestamp strings (default from Goggle Forms) to datetime objects for easier manipulation/calculation
reviews['Timestamp'] = pd.to_datetime(reviews['Timestamp'], format="%m/%d/%Y %H:%M:%S")


# Define cohort cutoffs as dates, and split the responses into their corresponding cohorts.
# Fa 22 	| 	Sp 23
cutoff1 = datetime(2023, 1, 1, 0,0,0) #!#
# Sp 23 	| 	Fa 23
cutoff2 = datetime(2023, 9, 1, 0,0,0) #!#
# Fa 23		|	Sp 24
cutoff3 = datetime(2024, 1, 1, 0,0,0) #!#
cohorts = ["Fa22", "Sp23", "Fa23", "Sp24"] #!#
dfs = split_by_date(reviews, names=cohorts, dates=[cutoff1, cutoff2, cutoff3], col_to_split="Timestamp") #!#


# Process comments to realign free-response questions that had a negative inclination
questions_to_realign = ["Improvements"] #!#
for question in questions_to_realign:
	for name,df in dfs.items():
		print(f"Realigning {name} for question: {question}")
		df[question] = realign_negatives(df[question])

# Perform sentiment analysis on each processed DataFrame/cohort
questions_to_sentiment = ["Improvements"] #!#
for name,df in dfs.items():
	print(f"Now processing sentiment for {name}...")
	for question in questions_to_sentiment:
		# VADER-fy
		df[question+':sentiment'] = df[question].apply(sid.polarity_scores)
		# Split VADER results into component elements
		df[[question+':neg', question+':neu', question+':pos', question+':compound']] = df[question+':sentiment'].apply(pd.Series)
		# Correct for overly positive tendency in longer questions
		df = correct_partial_negative_scores(df, cutoff=0, prefix=question+':') #!#


## Statistics!
# Test for significant differences between cohorts
questions_to_analyze = ["Improvements:compound"] #!#
for question in questions_to_analyze:
	# Get unique combinations 
	pairs = combos(cohorts, 2)
	for pair in pairs:
		# Perform Welch's t-test (No assumption of equal variance)
		# Null Hypothesis: Both cohorts have the same average score
		ttest_mean_diff = ttest_ind(dfs[pair[0]][question], dfs[pair[1]][question], equal_var=False)
		print(f"\nFor cohorts {pair[0]} and {pair[1]}:\n\t Welch's t-test p-value = {ttest_mean_diff.pvalue}, "
				f"so {'difference is significant at p=0.10!' if ttest_mean_diff.pvalue < 0.10 else  'difference is NOT significant at p=0.10!'}")
		
		# Perform Mann-Whitney U nonparametric rank test of distribution
		# Null Hypothesis: both samples were drawn from the same distribution 
		MWU_distro_diff = mannwhitneyu(dfs[pair[0]][question], dfs[pair[1]][question])
		print(f"\n\t Mann-Whitney U p-value = {MWU_distro_diff.pvalue}, "
				f"so {'distribution is significantly different at p=0.10!' if MWU_distro_diff.pvalue < 0.10 else  'distribution is NOT significantly different at p=0.10!'}")


## Plot!

# list of the sentiment analysis questions we want to plot.
# Name needs to match column name from above.
questions_to_plot = ["Improvements:compound"] #!#

# Create separate graph for each question
for question in questions_to_plot:
	fig, axs = plt.subplots(len(cohorts), 1, sharey=True, figsize=(15,10*len(cohorts)))

	for ind,ax in enumerate(axs):
		# Split out the data from the cohorts to plot
		# All survey replies from the cohort
		reviews = dfs[cohorts[ind]]
		# Sort all replies into groups for later processing
		group = reviews.sort_values(['Team #', question])
		# split out just the data needed: 
		scores = reviews[['Team #', question]]
		
		# Get the team/group average score for each team
		group_means = group.groupby('Team #')[question].mean()
		# Apply the group average score to each individual student's response
		# The group average will be the 'X' value, and the individual question score will be the 'Y' value.
		group['xval'] = [group_means[x] for x in group['Team #']]
		# get colormap ranged fit the number of teams/groups
		cmap = plt.cm.get_cmap('jet', max(group['Team #'])) #!#
		
		
		# Mark the '0' axes with dotted lines to avoid 
		# clutter with standard axes.
		ax.axhline(0, c='gainsboro', ls='--') #!#
		ax.axvline(0, c='gainsboro', ls='--') #!#
		# Draw the line of agreement (where group avg = individual)
		ax.plot([-1,1],[-1,1], color='#393939') #!#
		
		# plot the data as a scatter plot
		sc = ax.scatter(group['xval'], group[question], c=group['Team #'], cmap=cmap, alpha=0.6, s=280, edgecolors='k', linewidths=.5) #!#
		# Snap the limits of the plot
		ax.set_xlim(-1,1)
		ax.set_ylim(-1,1)
		# Increase label font size
		ax.tick_params(axis='both', which='major', labelsize=20)
		# Ensure 1:1 aspect ratio (plots will be square)
		ax.set_aspect('equal', adjustable='box')
		# Add the label of which cohort this was to "Y" axis
		ax.set_ylabel(cohorts[ind], fontsize=26) #!#
		
		# print out some statistics
		groupstats = reviews.groupby('Team #').describe(percentiles=[]).fillna(0)
		#!#
		# OPTIONAL: save CSV summary table of each cohort
		with open(cohorts[ind]+"_VADER_summary_by_group.csv", "w") as summary:
			summary.write(groupstats.to_csv())
			
	
	# Label the figure 
	fig.text(0.5, 0.08, "Team Average VADER score", ha='center', fontsize=30) #!#
	fig.text(0.135, 0.5, 'VADER Sentiment Score', va='center', rotation='vertical', fontsize=30) #!#
	# Get rid of deadspace
	plt.subplots_adjust(hspace=0.1)
	# Save out to file!
	plt.savefig(question.replace(":","-")+"_VADER_plot.png", dpi=400)
