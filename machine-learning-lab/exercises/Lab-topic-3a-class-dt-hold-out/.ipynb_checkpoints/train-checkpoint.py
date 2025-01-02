import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

train_size = 0.67

random_state = 42 # any positive integer is acceptable, it is necessary for **reproducibility


names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv('../Lab-topic-2-Data-Exploration/iris.csv', header=None, names=names)

# df.head()

# prepare the data, X is the input data and y is the label (or target) data

X = df.drop(columns=['class'])

y = df.drop(columns=names[:len(names)-1])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)

# train the model

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
