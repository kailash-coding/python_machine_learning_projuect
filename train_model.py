import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X = iris.data
y = iris.target

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open("model_iris.pkl", "wb"))

print("Model saved successfully")
