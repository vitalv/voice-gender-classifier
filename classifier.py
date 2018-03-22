#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
sb.set_style("whitegrid", {'axes.grid' : False})
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier 

mydata = pd.read_csv('myData_filtered.csv')

#Preview voice dataset
mydata.head()

categorical = ['age_range', 'pronunciation', 'gender']

#Basic barplots on number of male/female; pronunciation types; age_ranges
sb.countplot(x='gender', data=mydata, palette='Greens_d')
sb.countplot(x='pronunciation', data=mydata, palette='Blues_d')
sb.countplot(x='age_range', data=mydata, palette='Reds_d')









#split population male/female using patterns
female = mydata.loc[mydata.gender=='Female']
male = mydata.loc[mydata.gender=='Male']
#take a sample of the male population of equal size of the female (otherwise the frequencies will be much lower for female)
male_ = male.sample(len(female))

features = ['mean', 'skew', 'kurtosis', 'median', 'mode', 'std', 'low', 'peak', 'q25', 'q75', 'iqr']

#Plot the histograms
fig, axes = plt.subplots(6, 2, figsize=(10,20))

ax = axes.flatten() #ravel()

for i in range(len(features)):
    ax[i].hist(male_.ix[:,features[i]], bins=20, color='blue', alpha=.5)
    ax[i].hist(female.ix[:, features[i]], bins=20, color='red', alpha=.5)
    ax[i].set_title(features[i])
    ax[i].set_yticks(())
    
ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["male", "female"], loc="best")
fig.tight_layout()

plt.show()









#Prepare data for modeling
mydata = mydata[mydata.gender != 'not_specified']
mydata.loc[:,'gender'][mydata['gender']=="Male"] = 0
mydata.loc[:,'gender'][mydata['gender']=="Female"] = 1

#split mydata into train and test
mydata_train, mydata_test = train_test_split(mydata, random_state=0, test_size=.2)

#Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
scaler.fit(mydata_train.ix[:,features])  
X_train = scaler.transform(mydata_train.ix[:,features])
X_test = scaler.transform(mydata_test.ix[:,features])
y_train = list(mydata_train['gender'].values)
y_test = list(mydata_test['gender'].values)

#Train decision tree model
tree = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
print("Decision Tree")
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

#Train random forest model
forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(X_train, y_train)
print("Random Forests")
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))

#Train gradient boosting model
gbrt = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)
print("Gradient Boosting")
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

#Train support vector machine model
svm = SVC().fit(X_train, y_train)
print("Support Vector Machine")
print("Accuracy on training set: {:.3f}".format(svm.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(svm.score(X_test, y_test)))

#Train neural network model
mlp = MLPClassifier(random_state=0).fit(X_train, y_train)
print("Multilayer Perceptron")
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test, y_test)))



#Plot the variable importance
def plot_feature_importances_mydata(model, c):
    n_features = len(features)
    plt.figure(1,figsize=(9,13))
    plt.bar(range(n_features), model.feature_importances_, align='center', color=c)
    plt.xticks(np.arange(n_features), features)
    plt.ylabel("Variable importance")
    plt.xlabel("Independent Variable")
    plt.title(model.__class__.__name__)
    plt.show()

plot_feature_importances_mydata(tree,  '#45935B')
plot_feature_importances_mydata(forest, '#45789D')
plot_feature_importances_mydata(gbrt, '#AD413C')

#Plot the heatmap on first layer weights for neural network
plt.figure(figsize=(100, len(features)))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(len(features)), list(mydata))
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar().set_label('Importance')

plt.show()