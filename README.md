# Credit-Defult-Risk-Prediction

Description: 

This project is to predict whether the person can repay the loan on time or not.
The data is provided by Home Credit from Kaggle competition. And the goal of the project is to ensure that clients capable of repayment are not rejected and that clients with potential loan problem are identified. There are four parts of this project: 1) Exploratory Data Analysis 2) Data Preprocessing 3) Modeling 4) Modeling turning 5) Esembling 6)Conclusion

Step 1: Exploratory Data Analysis

Exploratory Data Analysis is a process where we calculate statistics and make figures to find trends, anomalies, patterns or relationships within the data.

Step 2: Data Preprocessing

This step is to clean the dataset. It includes deleting the missing values, data encoding, Impute the missing values, Over/Under sampling, removing collinear features.

Step 3: Modeling

After preprocessing, I get three different datasets for modeling: Undersampled dataset, Smote oversampled dataset, and not sampled dataset.
I use six measurement for the performance our model: Testing accuracy, Cross Validation, Recall, Precision, and F1 score. And among these six measurements, we will focus on more on recall, precison, and F1 score. Recall means that how many people with bad loan are selected, precision means that how relevant is the data selected. And F1 score is the combination of recall and precision. Since we want to find the bad loan person and don’t want to refuse people who can pay the loan on time, we would focus on F1 score most.
I implement five different models: SVM, KNN, Logistic Regression, Random Forest, Neural Network.

Step 4: Model Tuning

I select SVM, Random Forest, and Logistic Regression to tune the parameters because of the good performance. I tuned three hyper-parameters for each model to make sure it has the best performance.

Step 5:Ensembling

After I get the best hyper-parameters for these three models, we use majority vote as our ensembling method to improve our model performance. We can see that SVM has a really good performance in recall, which means it does well in selecting the bad loan people from the entire people. But Random Forest and Esemble have the best precison, which means the person they selecte are more likely to be a bad loan person. Random Forest and Esemble also have the best F1 score.

Conclusion:

1) When we want to predict minority data, we will focus on more on the precision, recall and F1 score instead of the accuracy
2) If we want to recognize most person who have repay problems, we should SVM
3) If we want to recognize the person who have repay problems as accurate as possible, we should use Random Forest
4) If we want to consider both situations, we also would consider Random Forest



