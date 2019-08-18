import numpy as np

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

x = []
file = 'EEG data.csv'
with open(file) as f:
	x = f.readlines()

train = []
test = []
traininput = []
trainoutput = []
testinput = []
testoutput = []

for i, a in enumerate(x):
	if i < len(x) - 102:
		train.append(list(int(b) for b in a.split(',')))
	else:
		test.append(list(int(b) for b in a.split(',')))

for i,a in enumerate(train):
    traininput.append(a[2:13])
    trainoutput.append(a[13])

for i,a in enumerate(test):
    testinput.append(a[2:13])
    testoutput.append(a[13])

X = np.array(traininput)
y = np.array(trainoutput)


#SVM Classification Training
svm = SVC(C =0.5)
svm.fit(X, y)

#Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X, y)

#Artificial Neural Network
ann = MLPClassifier(learning_rate = 'adaptive')
ann.fit(X, y)

#K-Nearest Neighbor Classifier
knn = KNeighborsClassifier()
knn.fit(X, y)

correct = [0, 0, 0, 0]
incorrect = [0, 0, 0, 0]
svms = []
gnbs = []
anns = []
knns = []

for i, a in enumerate(testinput):
	if svm.predict([a])[0] == testoutput[i]:
		correct[0] += 1
		svms.append(0)
	else:
		incorrect[0] += 1
		svms.append(1)
	if gnb.predict([a])[0] == testoutput[i]:
		correct[1] += 1
		gnbs.append(0)
	else:
		incorrect[1] += 1
		gnbs.append(1)
	if ann.predict([a])[0] == testoutput[i]:
		correct[2] += 1
		anns.append(0)
	else:
		incorrect[2] += 1
		anns.append(1)
	if knn.predict([a])[0] == testoutput[i]:
		correct[3] += 1
		knns.append(0)
	else:
		incorrect[3] += 1
		knns.append(1)
print(svm.predict([a])[0],testoutput[i])

sums = [0, 0, 0, 0, 0]
tests = [0, 0]

print(correct)
print(incorrect)
print(sums)

from sklearn import joblib
gnb = joblib.dump(gnb, 'Classification of student Mental State.pkl')

#Only about 80% Accurate
