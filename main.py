#%%
import pandas as pd
import os

#%%
def get_period(hour):
    if 6 <= hour < 12:
        return 1
    elif 12 <= hour < 18:
        return 2
    else:
        return 0

#%%
col = ["EID", "AbsT", "RelT", "NID", "Temp", 
       "RelH", "L1", "L2", "Occ", "Act", "Door", "Win"]

#%%
#Caminho, quantidades de janelas e portas respectivamente...
paths = [
    ('Room-Climate-Datasets/datasets-location_A/','A', 1, 1),
    ('Room-Climate-Datasets/datasets-location_B/','B', 2, 1),
    ('Room-Climate-Datasets/datasets-location_C/','C', 2, 1)
]

#%%
dfs = []

for path, dataset, windows, doors in paths:
    arquivos = [f for f in os.listdir(path) if f.endswith('.csv')]
    for arquivo in arquivos:
        df_temp = pd.read_csv(f'{path}{arquivo}', names=col)
        dfs.append(df_temp)
        df_temp['Dataset'] = dataset
        df_temp['Qtd_Windows'] = windows
        df_temp['Qtd_Door'] = doors

df_all = pd.concat(dfs, ignore_index=True)
df_all.to_parquet('all_locations.parquet', index=False)

#%%
df = pd.read_parquet('all_locations.parquet')

df['DateTime'] = pd.to_datetime(df["AbsT"], unit="ms")
df['Hour'] = df['DateTime'].dt.hour
df['TimeOfDay'] = df["DateTime"].dt.hour.apply(get_period)

# %%
df.columns

#%%
import matplotlib.pyplot as plt

# Agrupar por hora do dia e calcular a média da temperatura
temp_by_hour = df.groupby("Hour")["Temp"].mean()

plt.figure(figsize=(10,5))
plt.plot(temp_by_hour.index, temp_by_hour.values, marker="o", color="blue")

plt.title("Temperatura média por hora do dia")
plt.xlabel("Hora do dia")
plt.ylabel("Temperatura média (°C)")
plt.grid(True)
plt.xticks(range(0,24))  # garante que mostra de 0 a 23 horas
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

# Loop para cada Dataset
for dataset_name, group in df.groupby("Dataset"):
    temp_by_hour = group.groupby("Hour")["Temp"].mean()
    plt.plot(temp_by_hour.index, temp_by_hour.values, marker="o", label=f"Dataset {dataset_name}")

plt.title("Temperatura média por hora do dia por Dataset")
plt.xlabel("Hora do dia")
plt.ylabel("Temperatura média (°C)")
plt.grid(True)
plt.xticks(range(0,24))  # 0 até 23h
plt.legend()
plt.tight_layout()
plt.show()

#%%
df.head()
# %%
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

#%%
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

#%%
# Verificar a distribuição das classes (é balanceado?)
plt.figure(figsize=(8, 5))
sns.countplot(x='Occ', data=df)
plt.title('Distribuição do Número de Ocupantes (Occ)')
plt.show()

#%%
# Boxplots para ver como a temperatura varia com o número de ocupantes
plt.figure(figsize=(12, 6))
sns.boxplot(x='Occ', y='Temp', data=df)
plt.title('Temperatura por Número de Ocupantes')
plt.show()

#%%
# Boxplots para a umidade relativa
plt.figure(figsize=(12, 6))
sns.boxplot(x='Occ', y='RelH', data=df)
plt.title('Umidade Relativa por Número de Ocupantes')
plt.show()

#%%
# Matriz de correlação (analisar relações numéricas)
numeric_features = ['Temp', 'RelH', 'L1', 'L2', 'Occ']
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_features].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Matriz de Correlação')
plt.show()