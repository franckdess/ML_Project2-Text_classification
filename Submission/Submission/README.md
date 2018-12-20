Machine Learning
Project 2 - Text classification


The goal of this project is to predict whether the tweets present in the tweet_test.txt file reflect a positive or a negative sentiment. More precisely, a smiley ':)' ':(' has been removed from each tweet and the goal is to predict it with the remaining tweet.

Getting Started

The .zip file contains everything needed to run the .py and .ipynb files directly. This unzipped folder contains:

- data folder: it contains:
	- train_pos.txt:			train set of positive tweets - 100000 tweets
	- train_neg.txt: 			train set of negative tweets - 100000 tweets
	- train_pos_full.txt:		bigger train set of positive tweets - 1250000 tweets
	- train_neg_full.txt:		bigger train set of negative tweets - 1250000 tweets
	- test_data.txt:			test set of tweets for which we want to predict the sentiment - 10000 tweets
	- twitter-stopwords.txt:	list of stop words related to social media context
- functions.py:	implementation of all functions used in the .py and .ipynb files.
- run.py: python script to run in order to obtain the final sentiment predictions of the tweets in test_data.txt
- run.ipynb: jupyter notebook to run in order to obtain the final sentiment predictions of the tweets in test_data.txt
- pred_final_submission.csv: the sentiment predictions obtained by running the run.py script.

Prerequisites

Before running the script described above, make sure that the following libraries are correctly installed:
- sklearn
- pandas
- nltk
- numpy

If this is not the case, run the following command:
	$pip install library_name
where library_name is the name of the missing library.

Understanding the code

In both functions.py and run.py files, the implementation of every function as well as every step are precisely described. Please refer to the corresponding file to have the complete explanation.