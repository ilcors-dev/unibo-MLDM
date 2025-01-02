from joblib import load

model = load('iris.joblib')

def query(sepal_length, sepal_width, petal_length, petal_width):
    data = [[sepal_length, sepal_width, petal_length, petal_width]]
    return model.predict(data)[0]

# keep asking for input from the user in a single line, each attribute separated by a space
while True:
    print("Enter the attributes of the iris flower separated by a space (sepal_length sepal_width petal_length petal_width):")

    try:
        print(query(*map(float, input().split())))
    except EOFError:
        break
