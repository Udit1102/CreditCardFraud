This project aims to predict the credit card fraud using a given set of data.

#Dataset info
1. Step
2. Type: type of transaction
3. Amount
4. Name : name of origin as well as destination
5. Old and New balances for both origin and destination
6. IsPayment: it is 1 for debit or payment, else 0, csv file has already been modified
7. IsMovement: it is 1 for transfer or cash_out, else 0, it is also modified in the csv
8. Accountdiff: difference in old account balance of both origin and destination
9. Isfraud: it is the outcome

#Dependencies
sklearn, LogisticRegression, StandardScaler, Train_test_split

# Coding flow
1. Loading and analysis of data
2. Understanding of possible features and then formatting the data
3. Understanding the outcome variable
4. Standardization of features
5. Splitting the data and training the model
6. Fitting the model and checking the accuracy and then prediction of data
7. Using sample data for predictions