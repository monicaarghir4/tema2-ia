import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import seaborn as sns
from scipy.stats import chi2_contingency

# Incarcarea datelor 
data = pd.read_csv('tema2_AVC/AVC_full.csv')

# Separarea coloanelor in cele numerice si cele categorice
numerical_columns = ['mean_blood_sugar_level', 'body_mass_indicator', 'years_old',
                     'analysis_results', 'biological_age_index']
categorical_columns = ['cardiovascular_issues', 'job_category', 'sex', 'tobacco_usage',
                       'high_blood_pressure', 'married', 'living_area', 'chaotic_sleep',
                       'cerebrovascular_accident']

numerical = data[numerical_columns]
categorical = data[categorical_columns]

# Statistica datelor numerice
stats_numerical = numerical.describe().T
stats_numerical['count_no_missing'] = numerical.notnull().sum()

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

# Creare grafice boxplot pentru fiecare atribut
for column in numerical.columns:
    plt.figure(figsize=(6, 6))
    numerical.boxplot(column=column, grid=False)
    plt.title(f'Boxplot pentru {column}')
    plt.ylabel('Valoare')
    plt.show()

# Statistica pentru datele categorice
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

# Creare histogramelor pentru atribute
for column in categorical.columns:
    plt.figure(figsize=(10, 10))
    categorical[column].hist(grid=False, bins=30)
    plt.title(f'Histograma pentru {column}')
    plt.xlabel(column)
    plt.ylabel('Frecventa')
    plt.xticks(rotation=90)
    plt.show()

##################################################################################################

# Incarcarea datelor de antrenare si de testare
data_train = pd.read_csv('tema2_AVC/AVC_train.csv')
data_test = pd.read_csv('tema2_AVC/AVC_test.csv')

# Extragerea datelor categorice de antrenare si testare
train_categorical = data_train[categorical_columns]
test_categorical = data_test[categorical_columns]

# Crearea graficelor de tip barplot pentru fiecare atribut categoric
# din setul de antrenare
for column in train_categorical.columns:
    plt.figure(figsize=(10, 10))

    train_counts = train_categorical[column].value_counts().reset_index()
    train_counts.columns = [column, 'Frecventa']

    sns.barplot(x=column, y='Frecventa', data=train_counts)

    plt.title(f'Barplot pentru antrenare {column}')
    plt.xlabel(column)
    plt.ylabel('Frecventa')
    plt.xticks(rotation=90)
    plt.show()

# Crearea graficelor de tip barplot pentru fiecare atribut categoric
# din setul de testare
for column in test_categorical.columns:
    plt.figure(figsize=(10, 10))

    test_counts = test_categorical[column].value_counts().reset_index()
    test_counts.columns = [column, 'Frecventa']

    sns.barplot(x=column, y='Frecventa', data=test_counts)

    plt.title(f'Barplot pentru testare {column}')
    plt.xlabel(column)
    plt.ylabel('Frecventa')
    plt.xticks(rotation=90)
    plt.show()

##################################################################################################

# Matrice de corelatie pentru atributele numerice
corr_matrix = numerical.corr(method='pearson')
plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matrice de corelatie pentru atributele numerice din datasetul AVC')
plt.yticks(rotation=0)
plt.xticks(rotation=0)
plt.show()

# Testul chi2 pentru atributele categorice
def chi2_test(data_frame, col1, col2):
    contingency_table = pd.crosstab(data_frame[col1], data_frame[col2])
    result = chi2_contingency(contingency_table)
    return result[1]

# Matrice de p-values pentru atributele categorice
p_values = pd.DataFrame(index=categorical_columns, columns=categorical_columns)

# Completarea matricei cu 1 pe diagonala principala si cu valoarea testului chi2 in rest
for col1 in categorical_columns:
    for col2 in categorical_columns:
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
