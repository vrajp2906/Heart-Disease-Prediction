.PHONY: setup

setup:
	@echo "Installing required Python packages..."
	pip3 install pandas jupyter numpy seaborn matplotlib scikit-learn colorama mlxtend plotly statsmodels xgboost lightgbm catboost
	@echo "Setup complete."

clear:
	rm db/heart_data_cleaned.csv

all:
	cd ipynb
	python3 clean.py
	python3 analysis.py

clean:
	python3 clean.py

analysis:
	python3 analysis.py

distribution:
	python3 distribution.py

predict:
	python3 predict.py