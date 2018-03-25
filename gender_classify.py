from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn import tree
import numpy as np

'''
We will use an in memory and small dataset here to classify genders using various classification techniques
'''
# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

prediction_data = [[190, 70, 43]]

# more trees better result you can try to change n_estimators parameter and check by yourself
clf = RandomForestClassifier(n_estimators=5)
# train data
clf = clf.fit(X, Y)
rf_g = clf.predict(prediction_data)
pred_clf = clf.predict(X)
clf_accuracy = accuracy_score(Y, pred_clf) * 100
print("============== Random Forest ================")
print("prediction : {}".format(rf_g))
print("accuracy score : {}%".format(clf_accuracy))

nb = GaussianNB()
nb = nb.fit(X, Y)
nb_g = nb.predict(prediction_data)
pred_nb = nb.predict(X)
nb_accuracy = accuracy_score(Y, pred_nb) * 100
print()
print("============== Naive base ================")
print("prediction : {}".format(nb_g))
print("accuracy score : {}%".format(nb_accuracy))

lr = LogisticRegression()
lr = lr.fit(X, Y)
lr_g = lr.predict(prediction_data)
pred_lr = nb.predict(X)
lr_accuracy = accuracy_score(Y, pred_lr) * 100
print()
print("============== Linear regression ================")
print("prediction : {}".format(lr_g))
print("accuracy score : {}%".format(lr_accuracy))

per = Perceptron()
per = per.fit(X, Y)
per_g = per.predict(prediction_data)
pred_per = nb.predict(X)
per_accuracy = accuracy_score(Y, pred_lr) * 100
print()
print("============== Linear model perceptron ================")
print("prediction : {}".format(per_g))
print("accuracy score : {}%".format(per_accuracy))

knn = KNeighborsClassifier(n_neighbors=5)
knn = knn.fit(X, Y)
knn_g = knn.predict(prediction_data)
pred_knn = nb.predict(X)
knn_accuracy = accuracy_score(Y, pred_knn) * 100
print()
print("============== KNN ================")
print("prediction : {}".format(per_g))
print("accuracy score : {}%".format(per_accuracy))

tre = tree.DecisionTreeClassifier()
tre = tre.fit(X, Y)
tre_g = tre.predict(prediction_data)
pred_tre = nb.predict(X)
tre_accuracy = accuracy_score(Y, pred_tre) * 100
print()
print("============== Decision tree ================")
print("prediction : {}".format(tre_g))
print("accuracy score : {}%".format(tre_accuracy))


# finding best classifier
index = np.argmax([clf_accuracy, nb_accuracy, lr_accuracy, per_accuracy, knn_accuracy, tre_accuracy])
classifiers = {0: 'Random forest', 1: 'Naive base', 2: 'Linear Regression', 4: 'Perceptron', 5: 'KNN', 6: 'Decision tree'}
print('==============================================')
print('Best gender classifier is : {}'.format(classifiers[index]))
