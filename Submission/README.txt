This 'Submission' folder contains the following items :

- functions.py:
Contains helper functions needed for the project : 'tokenize', 'import_data', 'clean_data', 'zero_to_neg' and 'build_submission'.

- run.py:
The script to be executed in order to get the classifications, without documentation.

- run.ipynb:
The Jupiter Notebook to be executed in order to get the classifications. The content is the same as run.py, but with all steps justified and documented.

- submission.csv:
This .csv file contains the classifications obtained by running the run.py script. This is the file submitted to CrowdAI and gave the final score.

- Data Folder:
Contains .txt files : 
   - 4 train sets :
      - train_pos.txt : small training set containing only positive tweets
      - train_neg.txt : small training set containing only negative tweets
      - train_pos_full.txt : large training set containing only positive tweets
      - train_neg_full.txt : large training set containing only negative tweets
   - test_data.txt : test set
   - twitter-stopwords-final.txt : twitter-aware stop words