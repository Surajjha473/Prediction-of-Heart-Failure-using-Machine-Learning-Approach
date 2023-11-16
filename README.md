# Prediction-of-Heart-Failure-using-Machine-Learning-Approach

Loading Dataset:
 
Original Dataset:
 ![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/71736436-a863-4953-9691-fa4b3792c308)


 
After adding dead column:

 
Exploratory Data Analysis:
 
 
 
 
 
Calculating Correlation: (Excluding 14th variable ‘death’)
 
(Correlation values Rounded to 2 decimal digits)
Making heatmap to visualize correlation.
 
 Checking Highest Correlation variables:
 
 
Now,


















Training Dataset for K nearest algorithm then testing the dataset
Normalization function
 
Selecting 5 variables with highest correlation: time, serum_creatinine, ejection_fraction, age and serum_sodiumon
& Normalizing function used.
This code defines a normalization function that takes a vector x and normalizes its values to be between 0 and 1.
 
This code selects a subset of columns from the heartDisease dataset, containing the columns named DEATH_EVENT, time, serum_creatinine, ejection_fraction, and age. The resulting subset is assigned to a new variable heartDiseaseReduced. Then applies the normalize function to the subset of columns in heartDiseaseReduced that contain numerical data (i.e., columns 2 to 5). The resulting normalized values are assigned to a new data frame heartDiseaseNormalized.

 
The set.seed()  set a specific seed value for generating random numbers. in the example code set.seed(3214), the seed value is set to 3214. This means that any random number generated in the subsequent code that depends on a random number generator will be the same every time the code is run with this seed value. Setting a seed value is useful for reproducibility and testing purposes. If the same seed value is used, the results will be the same every time, which can help with debugging and testing code.
65-30 data partitioning
 

This code randomly samples from two values (1 and 2) with a probability of 0.8 for 1 and 0.2 for 2, creating a vector ind that assigns each row in heartDiseaseNormalized to either a training or testing set.
Then subsets the heartDiseaseNormalized dataset into two sets based on the ind vector created in the previous step: heartDiseaseTraining contains the rows where ind equals 1, and heartDiseaseTesting contains the rows where ind equals 2.
 
This code extracts the DEATH_EVENT column from heartDiseaseReduced and assigns the values corresponding to the training and testing sets to heartTrainLabel and heartTestLabel, respectively. The as.vector() function and t() transpose the matrix and convert it to a vector for compatibility with the knn() function.
 
This code performs a k-NN classification on the training and testing sets using the knn() function. The train argument specifies the training set, the test argument specifies the testing set, the cl argument specifies the class labels for the training set, and the k argument specifies the number of nearest neighbors to consider.
 
This code creates a data frame heartResult containing the predicted class labels and the actual class labels for the testing set. The predicted class labels are stored in heartPred, and the actual class labels are stored in `heart
Confusion Matrix AFTER TESTING & Training.
 
 
Model Accuracy 80 % =(67+23)/111
Now using Caret Package for KNN 
 
First, the code sets a random seed and creates a training and testing dataset using the createDataPartition function from the caret package. The heartDiseaseReduced dataset is assumed to contain the features of the patients, including age, sex, blood pressure, and other medical measurements, as well as the target variable DEATH_EVENT, which indicates whether the patient died during the follow-up period. The p parameter in createDataPartition specifies the proportion of data to be used for training.
Next, the code normalizes the training and testing datasets using the heartDiseaseNormalized dataset. The code then converts the DEATH_EVENT variable in the training and testing datasets to a factor variable using the as.factor function.
The code then fits a k-NN model to the training data using the train function from the caret package. The method parameter is set to 'knn', indicating that a k-NN algorithm should be used. The training features are specified using heartDiseaseTraining[, 1:4], which selects the first four columns of the heartDiseaseTraining dataset.
The code then generates predictions for the testing data using the predict function and the trained k-NN model. The predictions are stored in the results variable. The code then creates a data frame resultsDF that combines the predictions with the true DEATH_EVENT values from the testing data.
Finally, the code calculates the accuracy of the predictions by counting the number of correct predictions and dividing by the total number of predictions. The accuracy is stored in the Accuracy variable and printed to the console.
 
Accuracy for model is increased to 87.83%
















Now Decision Tree
 

The first two lines of code load the required packages and set a seed for reproducibility.
The third line of code removes the DEATH_EVENT variable since it is not needed for constructing a classification tree.
The fourth line constructs the classification tree using the rpart() function with the formula Dead~. indicating that we want to predict the Dead variable based on all the other variables in the heartDisease dataset. The control argument specifies the complexity parameter cp to be 0.00001. This value controls the amount of pruning to be done on the tree to avoid overfitting.
The fifth line prints the classification tree to the console.
The sixth line of code prints a table of the complexity parameter (CP) values and corresponding cross-validation error rates at each tree size, allowing us to determine the optimal value of CP for our tree.
The seventh line creates a confusion matrix to check the accuracy of the classification tree. The confusion matrix compares the actual outcomes of the Dead variable with the predicted outcomes using the predict() function with type="class".
The eighth and ninth lines of code create a colorful visualization of the classification tree using the rpart.plot() function. The boxcols variable specifies the colors for the terminal nodes, while the extra argument specifies the length of the horizontal lines at the bottom of the tree.
The final four lines of code add a legend to the visualization and calculate the accuracy of the classification tree. The accuracy is calculated by summing the number of correctly predicted outcomes and dividing by the total number of outcomes in the dataset.

 

 
 

