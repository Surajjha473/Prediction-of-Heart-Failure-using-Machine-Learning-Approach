# Prediction-of-Heart-Failure-using-Machine-Learning-Approach
Loading Dataset:
![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/d9a3fbb1-4eaa-4f6c-bf74-cb92827631ce)


Original Dataset:
 
![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/6ab06671-e18e-4393-a1db-867de667a441)


![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/7f371bd3-ce76-42db-a334-d95eed448b15)

 
After adding dead column:

![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/b6cf512a-4f6e-4249-a26b-a4f41e1ba975)

Exploratory Data Analysis:
 ![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/1b74f69e-cc67-4a17-8090-db7508bbd4e9)
 ![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/0a8163a5-907d-435a-8692-0221a3afccf0)

![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/d1e93b72-da38-4693-879c-ef78a13864ef)
![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/0fc5436f-839e-4a04-aa3e-fe6ad54fd0d9)

 ![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/8a0da8d6-12d9-461e-b8fc-c4b40b645463)

 
 
 
Calculating Correlation: (Excluding 14th variable ‘death’)
 ![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/b1882e12-f6c5-4da0-ac2a-776ea84ddb54)

(Correlation values Rounded to 2 decimal digits)
Making heatmap to visualize correlation.
 ![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/a0df27a6-fc52-46bd-b87e-a0bcbcb8d2ad)
![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/f4cdfadd-f939-458e-97c4-cfedaf8b0212)

 Checking Highest Correlation variables:
 ![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/414d6377-2a15-4c2f-8b80-94b144cf28a2)

 ![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/66e8af13-db91-46a9-9f8e-d6faf4988aaf)

Now,

Training Dataset for K nearest algorithm then testing the dataset
Normalization function
 ![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/18010580-e812-4abe-b430-5f4a80e2acb5)

Selecting 5 variables with highest correlation: time, serum_creatinine, ejection_fraction, age and serum_sodiumon
& Normalizing function used.
This code defines a normalization function that takes a vector x and normalizes its values to be between 0 and 1.
 ![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/2308be67-f751-463e-a20d-69d3e446dfab)

This code selects a subset of columns from the heartDisease dataset, containing the columns named DEATH_EVENT, time, serum_creatinine, ejection_fraction, and age. The resulting subset is assigned to a new variable heartDiseaseReduced. Then applies the normalize function to the subset of columns in heartDiseaseReduced that contain numerical data (i.e., columns 2 to 5). The resulting normalized values are assigned to a new data frame heartDiseaseNormalized.
![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/ecbcaaca-583c-483c-abfd-70144ec3d483)

 
The set.seed()  set a specific seed value for generating random numbers. in the example code set.seed(3214), the seed value is set to 3214. This means that any random number generated in the subsequent code that depends on a random number generator will be the same every time the code is run with this seed value. Setting a seed value is useful for reproducibility and testing purposes. If the same seed value is used, the results will be the same every time, which can help with debugging and testing code.
65-30 data partitioning
 ![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/5bae2eae-0a83-4e95-91e0-30b63c338f23)


This code randomly samples from two values (1 and 2) with a probability of 0.8 for 1 and 0.2 for 2, creating a vector ind that assigns each row in heartDiseaseNormalized to either a training or testing set.
Then subsets the heartDiseaseNormalized dataset into two sets based on the ind vector created in the previous step: heartDiseaseTraining contains the rows where ind equals 1, and heartDiseaseTesting contains the rows where ind equals 2.
 ![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/84dc7c79-c5ce-4f97-ae5f-9aab4011ee17)

This code extracts the DEATH_EVENT column from heartDiseaseReduced and assigns the values corresponding to the training and testing sets to heartTrainLabel and heartTestLabel, respectively. The as.vector() function and t() transpose the matrix and convert it to a vector for compatibility with the knn() function.
 ![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/97f7d14d-15d6-43c4-8aad-2ba5fe6c07c5)

This code performs a k-NN classification on the training and testing sets using the knn() function. The train argument specifies the training set, the test argument specifies the testing set, the cl argument specifies the class labels for the training set, and the k argument specifies the number of nearest neighbors to consider.
 ![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/e6d6b940-108b-4dc9-8ac2-7873a583c309)

This code creates a data frame heartResult containing the predicted class labels and the actual class labels for the testing set. The predicted class labels are stored in heartPred, and the actual class labels are stored in `heart
Confusion Matrix AFTER TESTING & Training.
 ![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/5c8cc54e-b79a-47b6-bc18-b6b50af66a84)
![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/8f175260-9269-46b6-b087-a43cb4c6ab20)

 
Model Accuracy 80 % =(67+23)/111
Now using Caret Package for KNN 
 ![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/b44144e5-208c-4798-a420-db399fd617b4)

First, the code sets a random seed and creates a training and testing dataset using the createDataPartition function from the caret package. The heartDiseaseReduced dataset is assumed to contain the features of the patients, including age, sex, blood pressure, and other medical measurements, as well as the target variable DEATH_EVENT, which indicates whether the patient died during the follow-up period. The p parameter in createDataPartition specifies the proportion of data to be used for training.
Next, the code normalizes the training and testing datasets using the heartDiseaseNormalized dataset. The code then converts the DEATH_EVENT variable in the training and testing datasets to a factor variable using the as.factor function.
The code then fits a k-NN model to the training data using the train function from the caret package. The method parameter is set to 'knn', indicating that a k-NN algorithm should be used. The training features are specified using heartDiseaseTraining[, 1:4], which selects the first four columns of the heartDiseaseTraining dataset.
The code then generates predictions for the testing data using the predict function and the trained k-NN model. The predictions are stored in the results variable. The code then creates a data frame resultsDF that combines the predictions with the true DEATH_EVENT values from the testing data.
Finally, the code calculates the accuracy of the predictions by counting the number of correct predictions and dividing by the total number of predictions. The accuracy is stored in the Accuracy variable and printed to the console.
 ![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/1c64e63c-1e0a-41fb-ae1b-d540299941e1)

Accuracy for model is increased to 87.83%


Now Decision Tree
 ![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/75f5ba1c-f243-4b53-a066-061cbf3318b4)


The first two lines of code load the required packages and set a seed for reproducibility.
The third line of code removes the DEATH_EVENT variable since it is not needed for constructing a classification tree.
The fourth line constructs the classification tree using the rpart() function with the formula Dead~. indicating that we want to predict the Dead variable based on all the other variables in the heartDisease dataset. The control argument specifies the complexity parameter cp to be 0.00001. This value controls the amount of pruning to be done on the tree to avoid overfitting.
The fifth line prints the classification tree to the console.
The sixth line of code prints a table of the complexity parameter (CP) values and corresponding cross-validation error rates at each tree size, allowing us to determine the optimal value of CP for our tree.
The seventh line creates a confusion matrix to check the accuracy of the classification tree. The confusion matrix compares the actual outcomes of the Dead variable with the predicted outcomes using the predict() function with type="class".
The eighth and ninth lines of code create a colorful visualization of the classification tree using the rpart.plot() function. The boxcols variable specifies the colors for the terminal nodes, while the extra argument specifies the length of the horizontal lines at the bottom of the tree.
The final four lines of code add a legend to the visualization and calculate the accuracy of the classification tree. The accuracy is calculated by summing the number of correctly predicted outcomes and dividing by the total number of outcomes in the dataset.

 ![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/69a048da-438e-497c-96cb-e970fa7afe0b)

![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/b40227c2-c809-41c6-ae06-ea84c5d014ce)

 

---------------------------------------------------------------------REPORT-------------------------------------------------------------------------------

Prediction of Patient’s Heart Failure Using a Machine Learning Approach


Introduction
	Congestive Heart Failure (CHF) is a chronic cardiovascular disease that is historically the leading cause of death among adults in the United States. The CDC estimates that roughly 6.2 million U.S. adults have CHF, and in 2018 was responsible for 380,000 deaths, which is roughly 13.5% of total deaths that year.2 Globally, heart failure is responsible for approximately 31% of total deaths annually.6 Congestive Heart Failure is characterized by the inadequate supply of blood and oxygen from the heart to other regions of the body. There are several factors that put individuals at high risk for developing CHF including diabetes, smoking status, hypertension, obesity, high cholesterol, and old age. 
Due to the degenerative nature of heart failure, early detection and treatment is critical to patient outcomes and reducing healthcare costs. In 2012 alone, roughly $30.7 billion was spent on medical services and medication for the treatment of heart failure.1 In recent years, studies have shown an increase in the use of machine learning algorithms as diagnostic support tools for the early detection of heart failure, among other conditions. With the high volume of data generated by electronic medical record systems (EMRs), historical data can be leveraged for the prediction of heart failure in new patients. This report aims to explore the application of machine learning algorithms to identify patterns within risk factors that may be overlooked using traditional methods of care. 
Previous studies have indicated that the early prediction of congestive heart failure using machine learning can be accomplished with an accuracy of up to 95.9%.5 Through analyzing 13 features of 299 records of patient health information, the team aims to identify the highest risk features for developing heart disease and use these key features to predict death events in patients with CHF using K-nearest neighbor, and decision tree models. Using this methodology, the team developed a K-nearest neighbor algorithm, that predicted the instance of CHF death events with 87.83% accuracy using the 5 features with the highest correlation to the target variable:  time, serum_creatinine, ejection_fraction, age, and serum_sodiumon.

Problem Description
The dataset contains 299 instances with 13 clinical features including clinical, body and lifestyle information. Of all the 13 features, the response is the death event. Upon further examination of the dataset, it is very reassuring that no missing data is presented. A brief description of the features is summarized in Table 1. A snapshot of the dataset is shown in Figure 1.

	Feature	Explanation
x1	age	Age of patients in years, continuous
x2	anemia	Decrease of red blood cells or hemoglobin, 0: no, 1: yes
x3	high blood pressure	If the patient has hypertension, 0: no, 1: yes
x4	creatinine phosphokinase (CPK)	Level of the CPK enzyme in the blood (mcg/L)
x5	diabetes	If the patient has diabetes, 0: no, 1: yes
x6	ejection fraction	Percentage of blood leaving the heart at each contraction
x7	platelets	Platelets in the blood (kilo platelets/mL)
x8	sex	Woman or man, 0: woman, 1: man
x9	serum creatinine	Level of serum creatinine in the blood (mg/dL)
x10	serum sodium	Level of serum sodium in the blood (mEq/L)
x11	smoking	If the patient smokes, 0: no, 1: yes
x12	time	Follow-up period (days)
y	death event	If the patient deceased during the follow-up period,
0: no, 1: yes
Table 1. Dataset description.

 ![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/2fca906c-eef0-4ca6-aba3-ba1f1e146639)

Figure 1. The first six records of the Heart Failure dataset. 
		
		The target variable for this analysis is the instance of a death event, “y”. X1 through X12 are a combination of categorical data and continuous data. For example, X2 indicates if a patient is anemic or not. For the purposes of analysis, categorical data is converted into a binary dummy variable with 0 indicating “no” and 1 indicating “yes”. Since the target variable is binary, logistic regression, and supervised learning algorithms will be used. Predictor variables will be reduced based on collinearity and low correlation with the response variable. The instance of a death event is the target variable, and a response of 1 indicates that a patient has died since their previous appointment. Through creating this model, a new patient record could be tested to predict if they are at risk for a death event. A positive prediction therefore would be associated with the recommendation for a patient to seek intensive care for their condition to avoid a death event and improve patient outcomes. 

Literature Review
	To provide appropriate context and motivations for this study, a brief literature review was conducted to analyze relevant studies and their findings. Using the databases provided through Binghamton University, the team aimed to find peer reviewed studies that included the keywords “machine learning” and “heart failure”.  Six studies were selected for their relevance to our project objectives and dataset features. 
	A 2023 publishing by the CDC listed congestive heart failure as the leading cause of death in America, accounting for 13.4% of US deaths annually with 6.2 million Americans currently having the condition.2  In total, there was $30.7 billion in healthcare costs in the U.S. associated with the treatment of heart failure each year.1  The global impact of heart failure is even more significant, accounting for roughly 31% of all global deaths each year.6 The above literature displays the global prevalence of heart failure as motivation for improvement efforts in treating and diagnosing heart failure. Due to the nature of heart disease’s high mortality rate and designation as a progressive and chronic condition, early detection is critical for the improvement of patient health outcomes.
	For the early diagnosis and treatment of heart disease, early detection measures must be applied. Machine learning has, in recent years, become a significant area of study due to the ability of algorithms to detect patterns in patient health data that could indicate early signs of heart disease. There is however, a degree of uncertainty when using machine learning models for diagnosis. To address the issue of medical uncertainties when using machine learning models, research indicates that they are best used in conjunction with traditional diagnosis and treatment methods. Serving as a clinical decision support tool to aid physicians, rather than relying solely on the model for an accurate diagnosis, has been shown to have the highest level of efficacy.3 Previous studies using patient health information for the development of machine learning models have indicated that two features, serum creatinine and ejection fraction, were highly correlated with the instance of heart failure related death events.4 
This report is focusing on the use of K-nearest neighbor algorithms, logistic regression, and decision trees to train data. Studies using other algorithms for predicting death events associated with heart failure have been quite successful. A 2022 study from the University of Mysore, India, compared the use of  a XGBoost-based machine learning algorithm, and a Naive Bayes-based algorithm to achieve a final model with 95.9% accuracy and 97.1% precision in predicting a death event in a patient with congestive heart failure.5 In conclusion, there is ample motivation for the reduction of heart failure related death events through the use of predictive diagnosis tools that leverage supervised learning algorithms. 

Methodology 
1.	Descriptive statistics.
          Before applying the machine learning approaches, we first visualize the correlation of all 13 features. From the correlation heatmap in figure 2, we notice that the dark yellow or dark purple denote that the two features correspond to each other strongly in a positive or negative direction. In our case, the death event and time and ejection and death event are negatively correlated. From figure 3, we confirm our observations from the heatmap, indicating that the death event is positively correlated with the level of serum creatinine in the blood, time, and ejection fraction.

 ![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/2c639eff-b51e-4b8d-9b16-ddb5a6ef3def)

Figure 2. Correlation heatmap for all 13 features

 ![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/3e08b229-ade1-4778-9116-cdf1856d2e21)

Figure 3. Correlation between 13 features

We also looked at some descriptive plots for the features including sex, smoking and serum creatinine. As we can see below, figure 4 demonstrates that death count by gender, indicating that male have a higher death count due to CHF than females do. Part (c) shows the serum level and serum creatinine between the dead and alive patients' relationship. It seems that slightly higher serum creatinine level is detected in dead patients and the cluster is more spread out than the alive patients’ cases.

    
 ![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/9f1ab1bd-c66d-4abe-9157-9ed8510be7d5)
![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/7b7e1b03-f7bf-411b-966a-39ade6b042f2)
![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/0ed0cca9-0e07-4de4-85ed-374d0dbf2620)

Figure 4. (a) histogram of death by sex; (b) density plot of death by sex and age; (c) scatter plot of serum creatinine by death.

2.	K- Nearest Neighbors
The k-nearest neighbors (k-NN) algorithm is a supervised machine learning algorithm used for both classification and regression tasks. It is a non-parametric method that makes predictions based on the similarity of data points in the feature space. It trains on data with predefined k nearest neighbors. It then makes predictions once the class label or target value is determined, and it is assigned to the unlabeled point as the predicted output. 
After selecting 5 variables with highest correlation: time, serum_creatinine, ejection_fraction, age and serum_sodiumon, we standardize the data. Next, a random 80%-20% split of data into training and test datasets is carried out. Finally, we calculate the accuracy of the predictions by counting the number of correct predictions and dividing by the total number of predictions. The accuracy is stored in the Accuracy variable and printed to the console.

3.	Decision Tree
The decision tree algorithm is a supervised learning algorithm that builds a tree-like model by recursively partitioning the data based on features. The most commonly used decision tree algorithm is called CART (Classification and Regression Trees). To make predictions for new data, traversing the decision tree from the root node to a leaf node will start the algorithm and based on the feature values of the new data, the label or value associated with that leaf node is returned. The decision tree algorithm is also the fundamental component of random forest which is a powerful tool in the machine learning world even at today’s standards.
We construct the algorithm to predict the Dead based on all the other predictors in the heart disease dataset. The control argument specifies the complexity parameter cp to be 0.00001. This value controls the amount of pruning to be done on the tree to avoid overfitting. We then calculate the accuracy of the classification tree by summing the number of correctly predicted outcomes and dividing by the total number of outcomes in the dataset. The algorithm can be visualized in figure 5 below, where at each node, we implement conditions to make judgements in each of the 5 predictors we select previously.

![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/f90c95f9-1ac9-4b81-a0ba-512e34828da9)
 
Figure 5. Decision Tree Algorithm Plotted

Results & Conclusions
This section presents the findings and outcomes for the study of CHF. It provides a clear and concise summary of the data analysis and evaluation of the algorithm's performance from KNN and Decision Tree algorithms. 
First, from the heatmap and preliminary data visualization techniques such as correlation plot, we can conclude that the top three features that have high positive correlation with Death, the response variable, are serum creatinine in the blood, time, and ejection fraction. We then run a correlation test to eliminate redundant features/predictors of the model. Finally, comparing the performance of the KNN algorithm to Decision Tree algorithms in figure 6 and 7, the accuracy from Decision Tree gives us 87.63% whereas the KNN algorithm yields 87.83% based on k=2. The results do not vary much using the two algorithms, which we are confident to suggest that both models based on the 5 predictors, time, serum creatinine, ejection fraction, age and serum sodium we have built the model that can yield about 87.6% accuracy to predict patients’ death.
	At last, we need to consider the limitations of the analysis. From figure 3, correlation test illustrates that the strongest predictor can only have 0.29 with the response variables besides predictor “Time”, the follow up period. The other three predictors have even less correlation values than 0.29. It is always beneficial to garner more data to support our analysis especially when we only have 299 records to study with. 

 ![image](https://github.com/Surajjha473/Prediction-of-Heart-Failure-using-Machine-Learning-Approach/assets/149039336/2f9ff4b0-a52c-440e-8a0f-320dec7cf4de)

Figure 6. KNN Algorithm Confusion Matrix


	             Predict: No	        Predict: Yes
Actual:No	     194	                   9
Actual: Yes	    28	                   68

Figure 7. Decision Tree Algorithm Confusion Matrix









 

