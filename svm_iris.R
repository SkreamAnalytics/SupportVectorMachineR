#======================================================================
# Support vector machine code for the Fisher's Iris data set
# date : 27/02/2018

# requires kernlab and mlr packages

#======================================================================


#Visualisation of the data
#*************************
n=nrow(iris) #get the number of rows in the dataset

#Create a colour vector for each class
irisCol=rep('green',n)
irisCol[which(iris[,5]=="setosa")]<-'red'
irisCol[which(iris[,5]=="versicolor")]<-'blue'

#Create a symbol vector for each class
irisChar=rep(0,n)
irisChar[which(iris[,5]=="setosa")]<-20
irisChar[which(iris[,5]=="versicolor")]<-4

#Plot the scatter graphs for each combination of the variables
par(mfrow=c(2,3)) #set the layout
for( i in 1:3){
  for(j in 2:4){
    if(i<j){
      plot(iris[,i], iris[,j], pch=irisChar, col=irisCol, 
           xlab=colnames(iris)[i],ylab=colnames(iris)[j])
    }
  }
}


#Classification using SVM
#*************************

#Build test and train sub-sets (sampling evenly from each type)
srows=c(sample(1:50,30), sample(51:100,30),sample(101:150,30))
irisTrain=iris[srows,];irisTest=iris[-srows,]

# Load the kernlab package
library(kernlab)

#Train the classifier
filter <- ksvm(Species~., data=irisTrain, kernel='rbfdot',C=5, cross=3)

# Attributes that you can access
attributes(filter) # list of all attributes
SVindex(filter) # the indices of all of the support vectors
alphaindex(filter)[[1]] # the support vector indices for the first of the three boundaries. 

#Predict the values for the test subset
pred<-predict(filter, irisTest[,-5])

#Display the confusion matrix
table(pred,irisTest[,5])
# Calculate accuracy
sum(pred==irisTest[,5])/length(pred)*100


#Binary classification with 2 features
#*************************************

# First group the versicolor and virignica examples together.
irisMod<-iris
levels(irisMod$Species)[levels(irisMod$Species)=="virginica"] <- "vers_virg"
levels(irisMod$Species)[levels(irisMod$Species)=="versicolor"] <- "vers_virg"

#Create the train and test datasets
srows=c(sample(1:50,30),sample(51:150,30))
irisTrain=irisMod[srows,] ; irisTest=irisMod[-srows,]

#Train the classifier
filter<- ksvm(Species~Petal.Length + Petal.Width, data=irisTrain, kernel='rbfdot', kpar=list(sigma=0.05),C=5, cross=3)

par(mfrow=c(1,1))# undoes the layout change for the intial graphs
plot(filter, data=irisTrain)
irisTrain[SVindex(filter),-c(1,2)] #get the support vectors

#Check accuracy
pred=predict(filter, irisTest[,-5])
table(pred,irisTest[,5])
sum(pred==irisTest[,5])/length(pred)*100


#Tuning
#*************************

# Define possible parameter values
c <- 2^seq(-8,8) ; nc <- length(c)
sigma <- 2^seq(-5,5) ; nsigma <- length(sigma)

#Initialise the matrix of errors
error <- matrix(0,nrow=nc,ncol=nsigma)

#Calculate the error for each combination
for (i in seq(nc)) {
  for (j in seq(nsigma)) {
    filter <- ksvm(Species~Petal.Length + Petal.Width,data=irisTrain,kernel='rbfdot',kpar=list(sigma=sigma[j]),C=c[i],cross=3)
    error[i,j] <- cross(filter) #returns the cross-validation error
  }
}

# Graph the result
library(lattice) #load the package lattice
dimnames(error) <- list(C=round(c,3),sigma=round(sigma,3)) # Add labels 
jet.colors <-  colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000")) # Customise the color ramp
levelplot(error,scales=list(x=list(rot=90)),xlab="C",ylab='sigma',main="Cross-validation Error Rate",col.regions = jet.colors)# Plot the error rate

# Auto-tuning
library(mlr) #load the package

# Create a discrete parameter set 
ps = makeParamSet(
  makeDiscreteParam("C", values = c),
  makeDiscreteParam("sigma", values = sigma)
)

# Create the search grid using default values
ctrl = makeTuneControlGrid()

#Specifiy random sampling with max of 10 iterations (optional for this grid size)
ctrl = makeTuneControlRandom(maxit = 10L)

# Create the task
classif.task = makeClassifTask(id = "irisMod", data = irisMod, target = "Species")

# Specify the resampling description : 3-fold cross-validation as above
rdesc = makeResampleDesc("CV", iters = 3L) 

# Tune the model
res = tuneParams("classif.ksvm", task = classif.task, resampling = rdesc,par.set = ps, control = ctrl)
res # to get the resulting parameters

