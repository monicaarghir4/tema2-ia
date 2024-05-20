import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# Load the data
data = pd.read_csv('tema2_AVC/AVC_full.csv')

numerical_columns = ['mean_blood_sugar_level', 'body_mass_indicator', 'years_old', 'analysis_results', 'biological_age_index']
categorical_columns = ['cardiovascular_issues', 'job_category', 'sex', 'tobacco_usage', 'high_blood_pressure', 'married', 'living_area', 'chaotic_sleep', 'cerebrovascular_accident']

# split the columns into the numerical ones and the other
numerical = data[numerical_columns]
categorical = data[categorical_columns]

stats_numerical = numerical.describe().T
stats_numerical['count_no_missing'] = numerical.notnull().sum()
stats_numerical.rename(columns={'50%': '50% (Median)'}, inplace=True)

stats_numerical = stats_numerical[['count_no_missing', 'mean', 'std', 'min', '25%', '50% (Median)', '75%', 'max']]
stats_numerical.columns = [
    'Nr. de exemple fără valori lipsă', 'Valoarea medie', 'Deviația standard',
    'Valoarea minimă', 'Percentila de 25%', 'Percentila de 50%',
    'Percentila de 75%', 'Valoarea maximă'
]

stats_numerical = stats_numerical.round(2)

# Create a PrettyTable object
table_numerical = PrettyTable()
table_numerical.field_names = ["Atribut"] + list(stats_numerical.columns)

for row in stats_numerical.itertuples():
    table_numerical.add_row([row.Index] + list(row[1:]))

# Print the nicely formatted table
print(table_numerical)

for column in numerical.columns:
    plt.figure(figsize=(8, 6))
    numerical.boxplot(column=column, grid=False)
    plt.title(f'Boxplot pentru {column}')
    plt.show()

# Categorical data
stats_categorical = pd.DataFrame({
    'Număr de exemple fără valori lipsă': categorical.notnull().sum(),
    'Număr de valori unice': categorical.nunique()
})

# Create a PrettyTable object
table_categorical = PrettyTable()
table_categorical.field_names = ["Atribut"] + list(stats_categorical.columns)

for row in stats_categorical.itertuples():
    table_categorical.add_row([row.Index] + list(row[1:]))

# Print the nicely formatted table
print(table_categorical)

# Plot the histograms
for column in categorical.columns:
    plt.figure(figsize=(8, 6))
    categorical[column].hist(grid=False, bins=30, figsize=(10,6))
    plt.title(f'Histogramă pentru {column}')
    plt.xlabel(column)
    plt.ylabel('Frecvență')
    plt.xticks(rotation=90)
    plt.show()

