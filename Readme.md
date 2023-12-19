# CMPT 353 Final Project

To run code, make sure you have Python, Pip and make installed
First thing to do is run "make setup"

After installation of libraries:

cd into py folder and make according to file names you wish
read report before making to understand file flow

To clear output, run:		make clear

See Makefile for more commands

Results of commands for /py/*.py will show up in /output/*/ respective folder

## Data Cleaning

For Data Cleaning, run the clean.ipynb, inside the ipynb/ directory.

**NOTE**: heart_cleaned.csv is already present inside the db/ directory.

## Exploratory Data Analysis

For visualizations, open the distribution.ipynb, inside the ipynb/ directory.

For stats on individual variables, open the analysis.ipynb, inside the ipynb/ directory.

## Model Building

For Model Building, run the code inside model_building_py/ directory

It contains four python scripts. To run:

1) python3 linear_regression.py
2) python3 random_forest.py
3) python3 knn.py
4) python3 xgboost_r.py
