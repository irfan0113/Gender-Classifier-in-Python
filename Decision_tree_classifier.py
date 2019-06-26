import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# X variable containts the data of a person ['Height','Weight','Shoe Size']
# Y variable containts the corresponding gender

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38],
     [154, 54, 37],[166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42],
     [181, 85, 43], [168, 75, 41], [168, 77, 41]]

Y = ['male', 'male', 'female', 'female', 'male', 'male',
    'female','female','female', 'male', 'male',
     'female', 'female']

# Now we give the test data and test labels

test_data = [[190, 70, 43],[154, 75, 38],[181,65,40]]
test_labels = ['male','female','male']
print("Labeled data : {}\n".format(test_labels))


#using DecisionTreeClassifier

dtc_clf = tree.DecisionTreeClassifier()
dtc_clf.fit(X,Y)
dtc_prediction = dtc_clf.predict(test_data)

print("The predicited genders by DTC : {}".format(dtc_prediction))

#using RandomForestClassifier

rfc_clf = RandomForestClassifier()
rfc_clf.fit(X,Y)
rfc_prediction = rfc_clf.predict(test_data)

print("The predicted genders by RTC :{}".format(rfc_prediction))


#using LogisticRegression

lr_clf = LogisticRegression()
lr_clf.fit(X,Y)
lr_prediction = lr_clf.predict(test_data)

print("The predicted genders by LR :{}".format(lr_prediction))

#Using SupportVectorClassifier

svc_clf = SVC()
svc_clf.fit(X,Y)
svc_prediction = svc_clf.predict(test_data)

print("The predicted genders by SVC :{}".format(svc_prediction))


#Lets calculate the accuracy for all four classifiers
classifiers = ['Decision Tree','Random Forest','Logistic Regression','Support Vector Classifier']

dtc_accuracy = accuracy_score(dtc_prediction,test_labels)
rfc_accuracy = accuracy_score(rfc_prediction,test_labels)
lr_accuracy = accuracy_score(lr_prediction,test_labels)
svc_prediction = accuracy_score(svc_prediction,test_labels)

accuracy = np.array([dtc_accuracy,rfc_accuracy,lr_accuracy,svc_prediction])
max_accuracy = np.argmax(accuracy)

#Lets see which classifier is best for our test_data

print(classifiers[max_accuracy] + ' is the best classifier for the given data')
