"""
Configurations to be used in shiny app.
"""
from pathlib import Path
import pandas as pd

str_PathToResourceDataFolder = Path(__file__).parent / 'resources'
str_PathToIcon = 'logo.png'
str_PathToCSS = str_PathToResourceDataFolder / 'style.css'


#df = dataset.dict_intermediate_data['80072NED_custom_y']
#df_filtered = df[df['BedrijfskenmerkenSBI2008'].str.match(r"^[A-Z].*") & (df['BedrijfskenmerkenSBI2008'] != "A-U Alle economische activiteiten")]
#df['BedrijfskenmerkenSBI2008'].unique().tolist()

#Styles to use in mock table
summary_table_styles = [
    {
        "cols": [0, 1,2],  # Age Group and Gender
        "class": "text_cols"
    },
    {
        "cols": [3],  # Average CB Premium
        "class": "numeric_cols"
    }
]

table_styles = [
    {
        "cols": [2, 3, 4],
        "class": "text_cols"
    },
    {
        "cols": [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        "class": "numeric_cols"
    }
]

# Data for model comparison
df_model_comparison = pd.read_csv(str_PathToResourceDataFolder / 'y_predictions_all_models.csv', sep=';', decimal = ",")
df_model_comparison['TargetDate'] = pd.to_datetime(df_model_comparison['TargetDate'], errors='coerce', format="%d/%m/%Y %H:%M")
df_model_comparison["Bias_AutoARIMA"] = df_model_comparison["AutoARIMA - all features"] - df_model_comparison["Actual"]
df_model_comparison["Bias_WA"] = df_model_comparison["WA"] - df_model_comparison["Actual"]
df_model_comparison["Bias_RF"] = df_model_comparison["Predicted_RF"] - df_model_comparison["Actual"]
#print(df_model_comparison[df_model_comparison["Sector"] == "K Financiële dienstverlening"].isna().sum())

#calculate average bias per sector
bias_by_sector = df_model_comparison.groupby("Sector")[["Bias_WA", "Bias_RF", "Bias_AutoARIMA"]].mean().reset_index()
bias_by_sector.columns = ["Sector", "Avg_Bias_WA", "Avg_Bias_RF", "Avg_Bias_AutoARIMA"]
#print(bias_by_sector.head(20))

# List of columns to ensure are numeric
numeric_cols = ['Actual', 'Predicted_RF', 'Auto Arima', 'WA']

# Convert them to numeric, coercing errors to NaN
for col in numeric_cols:
    df_model_comparison[col] = pd.to_numeric(df_model_comparison[col], errors='coerce')

# Mock data to use in table of premiums
df_mock = pd.read_csv(str_PathToResourceDataFolder / 'mock_premiums.csv')
df_mock['Year'] = df_mock['Period (Q)'].str[:4]

# Variable which shows the number of years to show in the graph
intNumberYearsToShowInGraph = 5

df_predictions = pd.read_csv(str_PathToResourceDataFolder / 'y_predictions.csv')
df_predictions['Year'] = pd.to_numeric(df_predictions['Period (Q)'].str[:4])
df_predictions['Error'] = df_predictions['y_hat_90_final'] - df_predictions['y_final']

# Extracting SBIs from the mock data table
df_SBI = df_mock[df_mock['SBI'].str.match(r"^[A-Z].*") & (df_mock['SBI'] != "A-U Alle economische activiteiten")]
SBI_Categories = sorted(df_SBI['SBI'].unique().tolist())

# Mock data on predictions
df_mock_y = pd.read_csv(str_PathToResourceDataFolder / 'y_predictions.csv')
df_mock_y["Period_Date"] = pd.to_datetime(df_mock_y["Period (Q)"].str[:4] + "Q" + df_mock_y["Period (Q)"].str[-1])
df_mock_y['Year'] = df_mock_y['Period (Q)'].str[:4]

# Data for the business case
df_business_case = pd.read_csv(str_PathToResourceDataFolder / 'df_forecast_SARIMAX_VThomas_V2.csv', delimiter=';')
df_business_case['Premie prijs_CB'] = pd.to_numeric(
    df_business_case['Premie prijs_CB']
        .str.replace('€', '', regex=False)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
        .str.strip(),
    errors='coerce'  # turn bad values into NaN
)

# import datatable for company size
df_company_size = pd.read_csv(str_PathToResourceDataFolder / 'Sector_Size.csv', delimiter=';')

# Calculate the bias cost per sector
bias_wage_table = pd.merge(bias_by_sector, df_company_size, left_on="Sector", right_on="SBI", how="left")
bias_wage_table["Financial risk RF"] = (bias_wage_table["Avg_Bias_RF"]/100) * bias_wage_table["Total Wage"]
bias_wage_table["Financial risk AutoARIMA"] = (bias_wage_table["Avg_Bias_AutoARIMA"]/100) * bias_wage_table["Total Wage"]
bias_wage_table["Financial risk WA"] = (bias_wage_table["Avg_Bias_WA"]/100) * bias_wage_table["Total Wage"]

# Now safely compute totals
total_risk_wa = bias_wage_table["Financial risk WA"].sum()
total_risk_autoarima = bias_wage_table["Financial risk AutoARIMA"].sum()
risk_reduction = total_risk_wa - total_risk_autoarima

total_risks = {
    "WA": total_risk_wa,
    "AutoARIMA": total_risk_autoarima,
    "Reduction": risk_reduction
}

bias_wage_table["Financial risk AutoARIMA"] = pd.to_numeric(bias_wage_table["Financial risk AutoARIMA"], errors='coerce')
bias_wage_table["Financial risk WA"] = pd.to_numeric(bias_wage_table["Financial risk WA"], errors='coerce')
#print(bias_wage_table[["Financial risk WA", "Financial risk AutoARIMA"]].dtypes)



# Dashboard
dashboard_ylabel_graph = 'EUR in millions'
dashboard_xlabel_graph = 'Year'
dashboard_graph_title = 'Insurance payout vs. collected premiums'

df_sickleave_vs_premium = pd.read_csv(str_PathToResourceDataFolder / 'premium_vs_sickleave.csv', sep=';')
df_sickleave_vs_premium['Sickleave'] = pd.to_numeric(df_sickleave_vs_premium['Sickleave'])
df_sickleave_vs_premium['Premium'] = pd.to_numeric(df_sickleave_vs_premium['Premium'])

df_premium_diff_man_woman = pd.read_csv(str_PathToResourceDataFolder / 'Premium_diff_man_woman.csv', sep=';')
df_premium_diff_man_woman['Man'] = pd.to_numeric(df_premium_diff_man_woman['Man'])
df_premium_diff_man_woman['Woman'] = pd.to_numeric(df_premium_diff_man_woman['Woman'])

