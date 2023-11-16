# tidyverse packages
library(tidyverse) 
#Now reading data set
heartDisease<-read.csv("D:/Binghamton university/Spring 2023/Health care data/Assignment & Solutions/term project/heart_failure_clinical_records_dataset.csv")
#adding one more column stating yes or no for death
# Male/Female from 0,1 for plotting
#Non-smokers & Smokers from 0,1 for plotting
heartDisease$Dead<-ifelse(heartDisease$DEATH_EVENT!=1,"No","Yes")
sex.labs<-c('Female','Male')
names(sex.labs)<-c(0,1)
smoking.labs<-c('Non-smokers','Smokers')
names(smoking.labs)<-c(0,1)
#Exploratory data analysis
ggplot(data=heartDisease,aes(x=Dead))+geom_bar(aes(fill=Dead))+facet_wrap(~sex,labeller=labeller(sex=sex.labs))
ggplot(data=heartDisease,aes(x=age,color=Dead,fill=Dead))+geom_density(alpha=.3,size=0.7)+facet_wrap(~sex,labeller=labeller(sex=sex.labs))
ggplot(data=heartDisease,aes(x=Dead))+geom_bar(aes(fill=Dead))+facet_wrap(~smoking,labeller=labeller(smoking=smoking.labs))+ggtitle("Dead chart grouped by smoking and non-smoking people")
ggplot(data=heartDisease,aes(x=serum_creatinine,y=serum_sodium))+geom_point(aes(col=Dead))
#Correlation Checking
library(reshape2)
cormap<-round(cor(heartDisease[,c(-14)]),2)
cormap_melted<-melt(cormap)
#creating heatmap
ggplot(data = cormap_melted, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()+scale_fill_gradient2(low="blue",high="yellow",mid="white")+theme(axis.text.x = element_text(angle = 90))
#Finding Highest Correlation
cormap_melted_Death<-cormap_melted[cormap_melted$Var2=='DEATH_EVENT',]
cormap_melted_Death$value<-abs(cormap_melted_Death$value)
cormap_melted_Death <-cormap_melted_Death %>%arrange(desc(value))
cormap_melted_Death
#Training Dataset for Knearest algorithm then testing the dataset
library(class)
normalize<-function(x){
  norm<-x-min(x)
  denom<-max(x)-min(x)
  return (norm/denom)
  }
heartDiseaseReduced<-heartDisease[,c('DEATH_EVENT','time','serum_creatinine','ejection_fraction','age')]
heartDiseaseNormalized<-as.data.frame(lapply(heartDiseaseReduced[,2:5],normalize))
set.seed(3214)
ind<-sample(2,nrow(heartDiseaseNormalized),replace=TRUE,prob=c(0.65,0.35))
heartDiseaseTraining<-heartDiseaseNormalized[ind==1,]
heartDiseaseTesting<-heartDiseaseNormalized[ind==2,]
heartTrainLabel<-heartDiseaseReduced[ind==1,1]
heartTestLabel<-heartDiseaseReduced[ind==2,1]
heartTrainLabel<-as.vector(t(heartTrainLabel))
heartPred<-knn(train=heartDiseaseTraining,test=heartDiseaseTesting,cl=heartTrainLabel,k=2)
heartResult<-data.frame(heartPred,heartTestLabel)
#Confusion Matrix after Training & Testing
library(gmodels)
heartTestLabel<-as.vector(t(heartTestLabel))
CrossTable(x=heartTestLabel,y=heartPred,prop.chisq=FALSE)
#Using Caret KNN for increasing the accuracy of the model
library(caret)
set.seed(54321)
ind <- createDataPartition(heartDiseaseReduced$DEATH_EVENT, p = 0.75, list = FALSE)
heartDiseaseTraining <- heartDiseaseNormalized[ind,]
heartDiseaseTesting <- heartDiseaseNormalized[-ind,]
heartTrainingLabel <- as.factor(heartDiseaseReduced[ind, 1])
heartTestingLabel <- as.factor(heartDiseaseReduced[-ind, 1])
heartPred <- train(heartDiseaseTraining[, 1:4], heartTrainingLabel, method = 'knn')
results <- predict(object = heartPred, heartDiseaseTesting[, 1:4])
resultsDF <- data.frame(results, heartTestingLabel)
names(resultsDF) <- c('Prediction', 'TrueValue')
resultsDF$isGood <- ifelse(resultsDF$Prediction == resultsDF$TrueValue, 'right', 'wrong')
resultsDF$isGood <- as.factor(resultsDF$isGood)
Accuracy <- nrow(resultsDF[resultsDF$isGood == 'right', ]) / nrow(resultsDF)
Accuracy
# Now drawing Decision Tree
library(rpart)
library(rpart.plot)
set.seed(5423)
#removing This variable since it is not needed for constructing a classification tree
heartDisease$DEATH_EVENT<-NULL
#creating tree, using a small critical point
heartTree<-rpart(Dead~.,data=heartDisease,control=rpart.control(cp=0.00001))
heartTree
printcp(heartTree)
#Creating a matrix to check the accuracy of decision tree
conf.matrix <- table(heartDisease$Dead, predict(heartTree,type="class"))
rownames(conf.matrix) <- paste("Actual", rownames(conf.matrix), sep = ":")
colnames(conf.matrix) <- paste("Pred", colnames(conf.matrix), sep = ":")
print(conf.matrix)
boxcols <- c( "red","blue")[heartTree$frame$yval]
par(xpd=TRUE)
prp(heartTree, faclen = 0, cex = 0.8, box.col = boxcols,extra=2)
legend("bottomleft", legend = c("Dead","Alive"), fill = c("red", "blue"),
       title = "Group")
Accuracy<-(conf.matrix[1,1]+conf.matrix[2,2])/sum(conf.matrix)*100
Accuracy
