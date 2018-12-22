## Data Description
In the Forest Cover Type Prediction challenge we are asked to predict the type of tree found on 30x30m squares of the Roosevelt National Forest in northern 
Colorado. The features we are given include the altitude at which the land is found,its aspect (direction it faces), various distances to features like roads, 
rivers and fire ignition points, soil types and so forth. We are provided with a training set of around 15,000 entries where the tree types are given (Aspen, 
Cottonwood,Douglas Fir and so forth) for each 30x30m square, and a test set for which we are to predict the tree type given the “features”.
This test set runs to around 500,000 entries. This is a straightforward supervised machine learning “classification” problem.

Source (Kaggle) : https://www.kaggle.com/c/forest-cover-type-prediction/data

## Repo Description

This repo is all about prediction of Forest Cover Type and contains:

1.train.csv					- Training dataset
2.test.csv					- Test dataset
3.Forest_cover_pred.py		- contains EDA and RandomForest model in python
4.api.py					- API file that can handle the POST requests using Postman's ADE
5.model.kpl					- Our trained model converted into a pickle file
6.model_columns.pkl			- Pickle file containing the column names that are necessary for prediciton
7.sample_input.txt			- A sample input form that should be given when POST request is done in Postman
8.plots (folder)			- Contains output plots from Forest_cover_pred.py
and of course the README.md file
