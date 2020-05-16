from tensorflow.keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# d1 = 8
# d2 = 0

# idx = (y_train == d1) | (y_train == d2)
# my_x_train = x_train[idx]
# my_y_train = y_train[idx] 

# idx = (y_test == d1) | (y_test == d2)
# my_x_test = x_test[idx]
# my_y_test = y_test[idx]

# second commit 
my_x_test = x_test
my_x_train = x_train
my_y_train = y_train
my_y_test = y_test

my_x_test = my_x_test.reshape((my_x_test.shape[0],784))
my_x_train = my_x_train.reshape((my_x_train.shape[0],784))

clf = LogisticRegression(random_state=0)
clf.fit(my_x_train[:2000], my_y_train[:2000])

clf.score(my_x_test,my_y_test)

y_pred = clf.predict(my_x_test)
arr = confusion_matrix(my_y_test, y_pred,labels = [0,1,2,3,4,5,6,7,8,9])