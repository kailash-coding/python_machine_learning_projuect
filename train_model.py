from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# dataset load
iris = load_iris()
X = iris.data
y = iris.target

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model create
model = RandomForestClassifier()

# train model
model.fit(X_train, y_train)

# save model
pickle.dump(model, open("model_iris.pkl", "wb"))

print("Model saved as model_iris.pkl")
