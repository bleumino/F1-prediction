# F1-prediction

Description: Predicting which place each F1 driver will finish in a race. Each model will predict each ranking based on historical data from 2018 onwards, using only the top 10 rankings shown

Data:
Source: Formula 1 race data on Kaggle (jtrotman/formula-1-race-data), downloaded automatically with kagglehub.
Tables used: results, races, drivers and constructors, merged on their ID columns.
Seasons: 2018 onward (about 3,800 driver-race rows).
Missing values are written as \N in this dataset, so they are read in as NaN with na_values="\\N".

Models:A linear regression, KNN, decision tree, random forest, and neural network were trained to predict the racer's finishing rank. Only KNN and decision tree were trained using GridSearchCV. All 5 models were evaluated using the R^2 score, which tells how well the model predicted the results based on the variation in the data, with 0 being the worst and 1 being the best. If a model somehow manages to score below 0, it would mean that they preformed worst than just predicting the average 
