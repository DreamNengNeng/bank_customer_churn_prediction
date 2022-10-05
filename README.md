# bank_customer_churn_prediction

This project addresses a classification problem -> to predict whether an existing bank customer is likely to close their bank account or not.

The EDA notebook shows explorative data analysis of the dataset. The dataset has both categorinal features and numerical features, and the target to be predicted is a binary label (churn or not churn). Correlations between the features and the target as well as among features themselves are analyzed in the EDA.

The pipeline is constructed with the following components:

Step0. Ask the right question (actionable) 

Step1. Data Collection + Data Inspection + Summary Statistics + Data Visualization 

Step2. Train test split: split the dataset into training and test datasets (8:2). The test dataset serves as a holdout set and will be used to evaluate model performance

Step3. Data preprocessing: Since the dataset has both categorical and numerical features. Before feeding them to a model, i first one-hot encode the categorical features and standardize the numerical features to keep them on the same scale. The OneHotEncoder and StandardScaler are fit to the training dataset, and then saved so i can use them to transform the test dataset later on.

Step4. Model training: I built 3 models on the preprocessed training data - Logistic Regression, K Nearest Neighbors, and Random Forests. Then perform a cross validation with each model and use Grid Search to find the optimal hyperparameters for the model. 

Step5. Make predictions on the test dataset: I use one of the three models that gives the best performance to predict the test dataset, and output evaluation metrics. But before making the predictions, don't forget to first preprocess the test dataset using the previously saved OneHotEncoder and StandardScaler.
