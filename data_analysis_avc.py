import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Incarcarea datelor 
data = pd.read_csv('tema2_AVC/AVC_full.csv')

# Separarea coloanelor in cele numerice si cele categorice
numerical_cont_columns = ['mean_blood_sugar_level', 'body_mass_indicator',
                     'analysis_results', 'biological_age_index']
numerical_discrete_columns = ['years_old']
categorical_columns = ['cardiovascular_issues', 'job_category', 'sex', 'tobacco_usage',
                       'high_blood_pressure', 'married', 'living_area', 'chaotic_sleep',
                       'cerebrovascular_accident']

for attribute in numerical_discrete_columns:
    categorical_columns.append(attribute)

numerical_cont = data[numerical_cont_columns]
categorical = data[categorical_columns]

# Statistica datelor numerice continue
stats_numerical = numerical_cont.describe().T
stats_numerical['count_no_missing'] = numerical_cont.notnull().sum()

# Redenumirea coloanelor
stats_numerical = stats_numerical[['count_no_missing', 'mean', 'std', 'min', '25%', '50%',
                                   '75%', 'max']]
stats_numerical.columns = [
    'Nr. de exemple fara valori lipsa', 'Valoarea medie', 'Deviatia standard',
    'Valoarea minima', 'Percentila de 25%', 'Percentila de 50%',
    'Percentila de 75%', 'Valoarea maxima'
]

# Rotunjirea datelor la 2 zecimale
stats_numerical = stats_numerical.round(2)

# Aranjare tabel in format pretty
table_numerical = PrettyTable()
table_numerical.field_names = ["Atribut"] + list(stats_numerical.columns)

for row in stats_numerical.itertuples():
    table_numerical.add_row([row.Index] + list(row[1:]))

print(table_numerical)

# Creare grafice boxplot pentru fiecare atribut numeric continuu
for column in numerical_cont.columns:
    plt.figure(figsize=(6, 6))
    numerical_cont.boxplot(column=column, grid=False)
    plt.title(f'Boxplot pentru {column}')
    plt.ylabel('Valoare')
    plt.show()

# Statistica pentru datele categorice/discrete.ordinale
stats_categorical = pd.DataFrame({
    'Numar de exemple fara valori lipsa': categorical.notnull().sum(),
    'Numar de valori unice': categorical.nunique()
})

# Aranjare tabel in format pretty
table_categorical = PrettyTable()
table_categorical.field_names = ["Atribut"] + list(stats_categorical.columns)

for row in stats_categorical.itertuples():
    table_categorical.add_row([row.Index] + list(row[1:]))

print(table_categorical)

# Creare histogramelor pentru atribute categorice/ordinale
categorical_columns_to_plot = categorical.columns[:-1]

for column in categorical_columns_to_plot:
    plt.figure(figsize=(10, 10))
    categorical[column].hist(grid=False, bins=30)
    plt.title(f'Histograma pentru {column}')
    plt.xlabel(column)
    plt.ylabel('Frecventa')
    plt.xticks(rotation=90)
    plt.show()

##################################################################################################

# # Incarcarea datelor de antrenare si de testare
# data_train = pd.read_csv('tema2_AVC/AVC_train.csv')
# data_test = pd.read_csv('tema2_AVC/AVC_test.csv')

# # Extragerea datelor categorice de antrenare si testare
# train_categorical = data_train[categorical_columns]
# test_categorical = data_test[categorical_columns]

# # Crearea graficelor de tip barplot pentru fiecare atribut categoric
# # din setul de antrenare
# for column in train_categorical.columns:
#     plt.figure(figsize=(10, 10))

#     train_counts = train_categorical[column].value_counts().reset_index()
#     train_counts.columns = [column, 'Frecventa']

#     sns.barplot(x=column, y='Frecventa', data=train_counts)

#     plt.title(f'Barplot pentru antrenare {column}')
#     plt.xlabel(column)
#     plt.ylabel('Frecventa')
#     plt.xticks(rotation=90)
#     plt.show()

# # Crearea graficelor de tip barplot pentru fiecare atribut categoric
# # din setul de testare
# for column in test_categorical.columns:
#     plt.figure(figsize=(10, 10))

#     test_counts = test_categorical[column].value_counts().reset_index()
#     test_counts.columns = [column, 'Frecventa']

#     sns.barplot(x=column, y='Frecventa', data=test_counts)

#     plt.title(f'Barplot pentru testare {column}')
#     plt.xlabel(column)
#     plt.ylabel('Frecventa')
#     plt.xticks(rotation=90)
#     plt.show()

##################################################################################################

# Matrice de corelatie pentru atributele numerice continue
corr_matrix = numerical_cont.corr(method='pearson')
plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matrice de corelatie pentru atributele numerice continue din datasetul AVC')
plt.yticks(rotation=0)
plt.xticks(rotation=0)
plt.show()

# Testul chi2 pentru atributele categorice
def chi2_test(data_frame, col1, col2):
    contingency_table = pd.crosstab(data_frame[col1], data_frame[col2])
    result = chi2_contingency(contingency_table)
    return result[1]

# Matrice de p-values pentru atributele categorice
p_values = pd.DataFrame(index=categorical_columns_to_plot, columns=categorical_columns_to_plot)

# Completarea matricei cu 1 pe diagonala principala si cu valoarea testului chi2 in rest
for col1 in categorical_columns_to_plot:
    for col2 in categorical_columns_to_plot:
        if col1 != col2:
            p_values.loc[col1, col2] = chi2_test(categorical, col1, col2)
        else:
            p_values.loc[col1, col2] = 1.0

# Afisare unei figuri pentru matricea obtinuta
plt.figure(figsize=(15,15))
sns.heatmap(p_values.astype(float), annot=True, cmap='coolwarm', vmin=0, vmax=1)
plt.title('Matrice de p-values pentru atributele categorice din datasetul AVC')
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.show()

##################################################################################################


# Separarea coloanelor in cele numerice si cele categorice
numerical_data_col = ['mean_blood_sugar_level', 'body_mass_indicator', 'years_old',
                     'analysis_results', 'biological_age_index']
categorical_data_col = ['cardiovascular_issues', 'job_category', 'sex', 'tobacco_usage',
                       'high_blood_pressure', 'married', 'living_area', 'chaotic_sleep',
                       'cerebrovascular_accident']

numerical_data = data[numerical_data_col]
categorical_data = data[categorical_data_col]

# Determinarea valorilor extreme pentru atributele numerice
for column in numerical_data_col:
    quantile_1 = numerical_data[column].quantile(0.25)
    quantile_3 = numerical_data[column].quantile(0.75)
    interquantile_range = quantile_3 - quantile_1

    threshold = 1.5

    lower_bound = quantile_1 - threshold * interquantile_range
    upper_bound = quantile_3 + threshold * interquantile_range

    # Inlocuirea valorilor extreme cu NaN
    numerical_data.loc[(numerical_data[column] < lower_bound) 
                       | (numerical_data[column] > upper_bound), column] = np.nan

# Verifica, daca s-au redus valorile extreme
for column in numerical_data_col:
    plt.figure(figsize=(6, 6))
    numerical_data.boxplot(column=column, grid=False)
    plt.title(f'Boxplot pentru {column}')
    plt.ylabel('Valoare')
    plt.show()

##################################################################################################

# Obtinerea valorii datelor numerice lipsa
missing_numerical_data = numerical_data.isnull().sum()
print("Date numerice lipsa inainte de imputare:\n",missing_numerical_data)

# Imputarea datelor lipsa
imputer_numerical = SimpleImputer(strategy='mean')
numerical_data.loc[:, numerical_data_col] = imputer_numerical.fit_transform(numerical_data.loc[:, numerical_data_col])

# Verificare daca datele au fost imputate
missing_numerical_data_after = numerical_data.isnull().sum()
print("Date numerice lipsa dupa imputare:\n",missing_numerical_data_after)

# Obtinerea valorii datelor categorice lipsa
missing_categorical_data = categorical_data.isnull().sum()
print("Date categorice lipsa inainte de imputare:\n",missing_categorical_data)

# Imputarea datelor lipsa pentru atributele categorice
imputer_categorical = SimpleImputer(strategy='most_frequent')
categorical_data.loc[:, categorical_data_col] = imputer_categorical.fit_transform(categorical_data.loc[:, categorical_data_col])

# Verificare daca datele au fost imputate
missing_categorical_data_after = categorical_data.isnull().sum()
print("Date categorice lipsa dupa imputare:\n",missing_categorical_data_after)

# Salvarea datelor imputate
data.loc[:, numerical_data_col] = numerical_data
data.loc[:, categorical_data_col] = categorical_data

##################################################################################################

# Normalizarea datelor numerice
min_max_scaler = MinMaxScaler()

for column in numerical_data_col:
    data[column]= min_max_scaler.fit_transform(data[numerical_data_col])

# Verificare cu ajutorul graficelor Boxplot
for column in numerical_data_col:
    plt.figure(figsize=(6, 6))
    data.boxplot(column=column, grid=False)
    plt.title(f'Boxplot pentru {column}')
    plt.ylabel('Valoare')
    plt.show()
