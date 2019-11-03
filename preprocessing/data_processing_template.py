import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

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
