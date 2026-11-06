import pandas as pd 
import numpy as np 
import os 
from dotenv import load_dotenv
from pathlib import Path 

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

REGRESSION_DIR = Path(os.environ["REGRESSION_DIR"])
VEHICLE_DIAGNOSTICS_DATASET = Path(os.environ['VEHICLE_DIAGNOSTICS_DATASET'])
outliers_path = REGRESSION_DIR / "outliers.xlsx"
describe_path = REGRESSION_DIR / "describe.xlsx"



df = pd.read_csv(VEHICLE_DIAGNOSTICS_DATASET)
print(df.shape)
print(df.info())
print(df.columns)

print('Missing values (nan):')
missing_count = df.isna().sum()
missing_percent = (missing_count / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'missing_values': missing_count,
    'missing_percent': missing_percent
})

print(missing_df)

print('Null values (0): ')
print((df == 0).sum())
print(df['record_id'].nunique())
print(df['vehicle_type'].nunique())
print(df['vehicle_type'].value_counts())
print(df['ac_on'].value_counts()) 
print(df['cylinders'].value_counts())

null_values = df[
    (df['tire_pressure_bar']== 0) |
    (df['engine_temp_c'] == 0) |
    (df['road_grade_pct'] == 0)
]
null_values.to_excel(os.path.join(REGRESSION_DIR, "null_values.xlsx"), index = False)

df_missing_outliers = df.copy()
df_missing_outliers = df_missing_outliers.drop(columns = ['ac_on'])
df_missing_outliers['missing_count'] = (df_missing_outliers.isna() | (df_missing_outliers == 0)).sum(axis = 1)
df_missing_outliers = df_missing_outliers.sort_values('missing_count', ascending = False)

cols = ['engine_temp_c', 'tire_pressure_bar', 'road_grade_pct', 'fuel_consumption_l_100km']
Q1 = df_missing_outliers[cols].quantile(0.25)
Q3 = df_missing_outliers[cols].quantile(0.75)
IQR = Q3 - Q1 
lower = Q1 - 1.5 * IQR 
upper = Q3 + 1.5 * IQR 
outliers_upper = df_missing_outliers[cols].gt(upper)
outliers_lower = df_missing_outliers[cols].lt(lower)

df_missing_outliers['outliers_count'] = outliers_upper.sum(axis = 1) + outliers_lower.sum(axis=1)
df_missing_outliers.to_excel(os.path.join(REGRESSION_DIR, "missing_outliers.xlsx"), index = False)

# After inspecting the Excel file, drop the 5 rows containing 3 missing values each
records =  df_missing_outliers[df_missing_outliers['missing_count'] == 3]
print('Records with 3 missing values:')
print(records)

print('Dataframe shape before removing the rows:')
print(df.shape)

for id in records['record_id']: 
    df = df[df['record_id'] != id]

print('Dataframe shape after removing the rows:')
print(df.shape)

# Outliers

columns = df.columns.tolist()

for col in ['record_id', 'vehicle_type', 'cylinders', 'ac_on']:
    columns.remove(col)

print(columns)

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

 # describe()

with pd.ExcelWriter(outliers_path, engine="openpyxl") as writer:
    for column in columns:

        describe_df = df[column].describe()

        describe_df.to_excel(
            writer,
            sheet_name = f"{column}_describe",
            index = False
        )


print('Setting tire pressure values to NaN where they are 0 or greater than 3.5 bar')
print(df['tire_pressure_bar'].isna().sum())
df['tire_pressure_bar'] = df['tire_pressure_bar'].replace(0, np.nan)
print(df['tire_pressure_bar'].isna().sum())

print(df[df['tire_pressure_bar'] > 3.5])
df.loc[df['tire_pressure_bar'] > 3.5, 'tire_pressure_bar']  = np.nan
print(df['tire_pressure_bar'].isna().sum())

df.to_csv(os.path.join(REGRESSION_DIR / 'vehicles_cleaned.csv'), index = False)
df.to_excel(os.path.join(REGRESSION_DIR / 'vehicles_cleaned.xlsx'), index = False)







