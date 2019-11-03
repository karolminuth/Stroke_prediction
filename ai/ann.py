# Artificial Neural Network
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Importing the dataset
data = pd.read_csv('../data/Preprocessed_stroke_data.csv', sep=',')

X = data.iloc[:, :-1].values    # matrix of features(independent variables)
y = data.iloc[:, -1].values     # vector of dependent variables

# encoding values
labelencoder_X = LabelEncoder()
for i in range(5):
    X[:, i] = labelencoder_X.fit_transform(X[:, i])

# Encoding categorical data
onehotencoder = OneHotEncoder(categorical_features=[0, 1])
X = onehotencoder.fit_transform(X).toarray()
# dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 - Now let's make the ANN!

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=15))

# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
# classifier.fit(X_train, y_train, batch_size=10, epochs=100)
classifier.fit(X_train, y_train, batch_size=10, epochs=20)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
