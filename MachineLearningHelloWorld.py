#import sys
#print('python: {}'.format(sys.version))
#import scipy
#print('scipy: {}'.format(scipy.__version__))
#import numpy
#print('numpy: {}'.format(numpy.__version__))
#import matplotlib
#print('matplotlib: {}'.format(matplotlib.__version__))
#import pandas
#print('pandas: {}'.format(pandas.__version__))
#import sklearn
#print('sklearn: {}'.format(sklearn.__version__))
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)


#Printing out the columns the data contains with the shape property 
#print(dataset.shape)

#actually displays the data
#print(dataset.head(20))

#gives summary of each attribute 
#print(dataset.describe())

#displays number of instances that belong to each class
#print(dataset.groupby('class').size())

#Createes box and whisker plots of each 
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()

#this displays histograms to get an idea of distributions
#dataset.hist()
#plt.show()

#Scatter plot matrix. Can be helpful to spot structured relationships between input variables
#scatter_matrix(dataset)
#plt.show()

#this next section creates some models of data and estimates
#
#their accuracy on unseen data
#
#Must make sure that any model we create is any good.
#
#We need to know that the model we created is any good.
#
#Later, we will use statistical methods to estimate the accuracy of the models that we create on unseen data. 
#We also want a more concrete estimate of the accuracy of the best model on unseen data by evaluating it on actual unseen data.
#
#That is, we are going to hold back some data that the algorithms will not get to see and ...
#we will use this data to get a second and independent idea of how accurate the best model might actually be.
#We will split the loaded dataset into two, 80% of which we will use to train our models and 20% that we will hold back as a validation dataset.

#split-out validate dataset
#this makes training data in x_train and y_train pro preparing models
#and a x_validation and y_validation sets that will be used later
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#tests options and evaluation metric
#this splits dataset into 10 parts, trains on 9 and test on 1 and repeat for all combinations of train-test splits
seed = 7
scoring = 'accuracy'

#this next section creates a mixture of linear and nonliliar algorithms
#the random number seend is regenerated before each run the enure that the evaluation
#of each algorithm is performed using exact same data splits. 
#it entures the results are directly comparable

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	#print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
#plt.show()


# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))