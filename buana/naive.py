import pandas as pd
import numpy as np

iris = pd.read_csv("soybean-cleaned.csv")
iris.head()

#  variabel bebas
x = iris.drop(["Class"], axis = 1)
x.head()

#variabel tidak bebas
y = iris["Class"]
y.head()


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)

from sklearn.naive_bayes import GaussianNB

iris_model = GaussianNB()

NB_train = iris_model.fit(x_train, y_train)
    # Next step: Prediction the x_test to the model built and save to the y_pred variable 
    # show the result of prediction 
y_pred = NB_train.predict(x_test)

np.array(y_pred) 

# show the y_test based on separation dataset
np.array(y_test)


NB_train.predict_proba(x_test)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)

#evaluate performance from the confusion matrix 
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

