import pandas as pd
import numpy as np

initial_data = pd.read_csv('../data/Initial_stroke_data.csv', sep=',',
                           names=['ID', 'Gender', 'Age_In_Days', 'Hypertension', 'Heart_Disease', 'Ever_Married',
                                  'Type_Of_Work', 'Residence', 'Avg_Glucose', 'BMI', 'Smoking_Status', 'Stroke'])

initial_data = initial_data.drop(index=0).reset_index()
initial_data = initial_data.drop(columns=['index', 'ID'])

initial_amount_data = len(initial_data)
print('Initial amount of data -> ', initial_amount_data)

# grouping and counting values by output value
print(initial_data.groupby('Stroke').count())
print(initial_data.groupby('Gender').count())

initial_data_not_stroke = len([i for i in initial_data['Stroke'] if int(i) == 0])
initial_data_stroke = len([i for i in initial_data['Stroke'] if int(i) == 1])

data = initial_data.copy()

for idx, row in enumerate(data.values):
    # Removing rows with no information about smoking status
    if isinstance(data['Smoking_Status'].loc[idx], float):
        stroke_value = data['Stroke'].loc[idx]
        data.drop(idx, inplace=True)

# grouping and counting values by output value
print(data.groupby('Stroke').count())

data_without_gaps = len(data)
print('Amount of data after delete gaps for smoking status -> ', data_without_gaps)

data = data.reset_index()

for idx, row in enumerate(data.values):
    # Removing rows with no information about binary gender - male or female
    if data['Gender'].loc[idx] == 'Other':
        data.drop(idx, inplace=True)

data = data.reset_index()

data['Age_In_Years'] = [round((float(data['Age_In_Days'].loc[idx])/365)) for idx, row in data.iterrows()]

data = data.drop(columns=['Age_In_Days'])

print('Before removing too large amount of years', data.groupby('Age_In_Years').count())

for idx, row in data.iterrows():
    # Removing values with people who has more tak 120 years, because we'are assuming that is impossible
    if row['Age_In_Years'] > 120:
        data.drop(idx, inplace=True)

print('After removing too large amount of years', data.groupby('Age_In_Years').count())

data_not_stroke = len([i for i in data['Stroke'] if int(i) == 0])
data_stroke = len([i for i in data['Stroke'] if int(i) == 1])

print('\n\nStatistic informations after processing of data...\n')
print('Saved percent of whole data ->', round((data_without_gaps/initial_amount_data)*100, 2))
print('Saved percent of stroke output ->', round((data_stroke/initial_data_stroke)*100, 2))
print('Saved percent of not stroke output ->', round((data_not_stroke/initial_data_not_stroke)*100, 2))
print('There is {0} gaps in whole dataset. All of them will be replaced by mean value of column'
      .format(data.isna().sum()['BMI']))

data = data.astype({'BMI': np.float})
data[data == np.inf] = np.nan
data.fillna(data['BMI'].mean(), inplace=True)

# mixed order of columns for future simpler ways to processing
data_in_proper_order = data[['Type_Of_Work', 'Smoking_Status', 'Gender', 'Ever_Married', 'Residence', 'BMI',
                             'Age_In_Years', 'Hypertension', 'Heart_Disease', 'Avg_Glucose', 'Stroke']]

name_save = '../data/Preprocessed_stroke_data.csv'
data_in_proper_order.to_csv(name_save, index=False)

print('\n\nData saved in location -> ', name_save)

# Task for future:
# 1 - exchange days for years (DONE)
# 2 - bmi -> categorical ?
# 3 - average glucose -> categorical ?
