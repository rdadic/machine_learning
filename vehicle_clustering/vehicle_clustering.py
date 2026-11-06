import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

import os 
from dotenv import load_dotenv
from pathlib import Path 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

CLUSTERING_DIR = Path(os.environ["CLUSTERING_DIR"])
CLUSTERING_DATASET = Path(os.environ["CLUSTERING_DATASET"])



outliers_path = CLUSTERING_DIR / "outliers.xlsx"
describe_path = CLUSTERING_DIR / "describe.xlsx"

df = pd.read_csv(CLUSTERING_DATASET)
print(df.shape)

df_clean = df.copy()

mask = (
    ((df_clean['oil_pressure_bar'] == 0)) |
    ((df_clean['fuel_rate_l_h'] == 0) &
     (df_clean['speed_kmh'] > 10) &
     (df_clean['engine_rpm'] > 800)) |
    ((df_clean['engine_temp_c'] > 130)) |
    ((df_clean['coolant_temp_delta_c'] > 30)) |
    ((df_clean['vibration_mm_s'] > 9))    
)

df_suspicious = df_clean[mask].copy()
df_suspicious.to_excel(os.path.join(CLUSTERING_DIR, "suspicious_values.xlsx"), index = False)

# After examinig the suspicious values 

df_clean.loc[
    df_clean['fuel_rate_l_h'] == 0, 'fuel_rate_l_h'
] = np.nan

df_clean.loc[
    df_clean['oil_pressure_bar'] == 0, 'oil_pressure_bar'
] = np.nan

mask = (
    ((df_clean['engine_temp_c'] > 130)) |
    ((df_clean['coolant_temp_delta_c'] > 30)) |
    ((df_clean['vibration_mm_s'] > 9))   
)

engine_vibration_coolant = df_clean[mask].copy()
engine_vibration_coolant.to_excel(os.path.join(CLUSTERING_DIR, "engine_vibration_coolant.xlsx"), index = False)


print('Missing values (nan):')
missing_count = df_clean.isna().sum()
missing_percent = (missing_count / len(df_clean) * 100).round(2)
missing_df = pd.DataFrame({
    'missing_values': missing_count,
    'missing_percent':missing_percent
})
print(missing_df)

print('Null values (0):')
print((df_clean == 0).sum())

# Coolant temeprature difference can be 0 - the temperature is eqault to the 88 °C reference value
# vibration = 0, a few rows have zero vibration values; leave for now; handle them later using
# median imputation

print(df_clean[df_clean['vibration_mm_s'] == 0])
df_nullvalues = df_clean[(df_clean['vibration_mm_s'] == 0) | (df_clean['coolant_temp_delta_c'] == 0)]
df_nullvalues.to_excel(os.path.join(CLUSTERING_DIR, "nullValues.xlsx"), index = False)


# outliers and describe()

columns = df.columns.tolist()
columns.remove('record_id')
columns.remove('vehicle_id')


with pd.ExcelWriter(outliers_path, engine="openpyxl") as writer:
    for column in columns:

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[
            (df[column] < lower) |
            (df[column] > upper)
        ]

        outliers.to_excel(
            writer,
            sheet_name = f"{column}_outliers",
            index = False
        )

with pd.ExcelWriter(outliers_path, engine="openpyxl") as writer:
    for column in columns:

        describe_df = df[column].describe()

        describe_df.to_excel(
            writer,
            sheet_name = f"{column}_describe",
            index = False
        )


# Graphs 

df_clean['engine_load_pct'].hist(bins=50)
plt.xlabel('Engine load in %')
plt.ylabel('Frequency')
plt.title('Distribution of engine load')
plt.show()

df_clean['engine_temp_c'].hist(bins = 50)
plt.xlabel('Engine temeprature in °C')
plt.ylabel('Frequency')
plt.title('Distribution of engine temperature')
plt.show()

df_clean['oil_pressure_bar'].hist(bins = 50)
plt.xlabel('Oil pressure in bar')
plt.ylabel('Frequency')
plt.title('Distribution of oil pressure')
plt.show()

df_clean['fuel_rate_l_h'].hist(bins = 50)
plt.xlabel('Fuel rate in l/h')
plt.ylabel('Frequency')
plt.title('Distribution of fuel rate')
plt.show()

df_clean['coolant_temp_delta_c'].hist(bins = 50)
plt.xlabel('Coolant - delta temperature in °C')
plt.ylabel('Frequency')
plt.title('Distributio of coolant delta temperature')
plt.show()

print('Correlations: \n')

corr = df_clean.corr(numeric_only = True)

plt.figure(figsize=(10,6))
plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(corr.columns)),corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.tight_layout()
plt.show()

# Machine Learning - Linear Regression

df_ml = df_clean.dropna(subset =['engine_temp_c']).copy()

y = df_ml['engine_temp_c']
X = df_ml.drop(columns = ['record_id', 'vehicle_id', 'engine_temp_c'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy = 'median')

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

y_predicted = model_lr.predict(X_test)

print('Linear Regression Metrics:')
print('Mean absolute error - MAE: ', mean_absolute_error(y_test, y_predicted))
print('Root mean squared error - RMSE: ', mean_squared_error(y_test, y_predicted) ** 0.5)
print('Coefficient of determination R2: ', r2_score(y_test, y_predicted))

# Machine Learning - Clustering

X = df_clean.drop(columns=['record_id', 'vehicle_id'])

imputer = SimpleImputer(strategy = 'median')
X = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters = 3, random_state = 42, n_init = 10)
clusters = kmeans.fit_predict(X_scaled)

df_clusters = df_clean.copy()
df_clusters['cluster'] = clusters

kmeans.cluster_centers_

plt.scatter(X_scaled[:,0], X_scaled[:,1], c=clusters, cmap='winter')
plt.scatter(
    kmeans.cluster_centers_[:,0], 
    kmeans.cluster_centers_[:,1],
    marker = '*',
    s = 200,
    label = 'Centroids'
)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.show()

# Elbow method - finding the optimal number of clusters

inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters = k, random_state = 42, n_init = 10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')

plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow method')
plt.show()
 
# Silhouette score 

score = silhouette_score(X_scaled, clusters)
print('Silhouette score: ', score)

df_clusters['cluster'] = clusters

cluster_analysis = df_clusters.groupby('cluster').mean(numeric_only = True)

cluster_analysis.to_excel(os.path.join(CLUSTERING_DIR, "cluster-analysis.xlsx"), index = True)
cluster_analysis.to_csv(os.path.join(CLUSTERING_DIR, "cluster-analysis.csv"), index = True)
print('Cluster analysis: ', cluster_analysis)

print(df_clusters['cluster'].value_counts().sort_index())


cols = ['engine_temp_c', 'oil_pressure_bar', 'vibration_mm_s', 'coolant_temp_delta_c']

fig, axes = plt.subplots(2, 2, figsize=(12,8))

for ax, col in zip(axes.flatten(), cols): 
    sns.boxplot(data = df_clusters, x='cluster', y=col, ax=ax)
    ax.set_title(f'{col} by cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel(col)

plt.tight_layout()
plt.show()

# Principal component analysis (PCA)

pca = PCA(n_components = 2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(9,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c = clusters,
    cmap = 'winter',
    alpha = 0.6
)

plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.title('K-Means clustering with PCA')
plt.colorbar(label = 'Cluster')
plt.show()

print('PCA ratio and ratio sum')
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())

loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index = df_clusters.drop(columns=['record_id', 'vehicle_id', 'cluster']).columns

)

print('Loadings: ')
print(loadings)

pca_all = PCA()
pca_all.fit(X_scaled)

print('pca all')
print(pca_all.explained_variance_ratio_)

cumulative_variance = np.cumsum(pca_all.explained_variance_ratio_)
print(cumulative_variance)

plt.figure(figsize = (9,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters,
    cmap='winter',
    alpha=0.5
)

plt.xlabel('PC1 42.77%')
plt.ylabel('PC2 35.90%')
plt.title('K-Means clustering - PCA')
plt.colorbar(label='Cluster')
plt.show()






















