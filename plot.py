# %%
"""
Module containg the data for plots
"""
import config
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np

# %%
def create_plot(categories: list, totalincluded: bool, frequency: str, period: tuple):
    categories = [x for x in categories]
    if totalincluded:
        categories.append('Totaal gemiddelde')

    startyear, endyear = period

    temp_df = df[(  (df['Year'] >= startyear) & \
                    (df['Year'] <= endyear) & \
                    (df['BedrijfskenmerkenSBI2008'].isin(categories)) & \
                    (df['Frequency'] == frequency))]


    categorylist = temp_df['BedrijfskenmerkenSBI2008'].unique().tolist()    

    # Dynamically calculate figure height
    base_height = 8
    additional_height_per_legend_row = 0.6
    legend_columns = 2
    legend_rows = -(-len(categorylist) // legend_columns)  # Round up to calculate number of rows
    figure_height = base_height + legend_rows * additional_height_per_legend_row

    fig = plt.figure(figsize=(10, figure_height))
    gs = GridSpec(2, 1, height_ratios=[1, 1], figure=fig)  # 2 rows: 3/4 for plot, 1/4 for legend

    maxvalue = getmaxvalue()

    ax = fig.add_subplot(gs[0])
    ax.set_title('Sick leave % per industry')
    ax.set_xlabel('Period')
    ax.set_ylabel(r'Sick leave in % of total working days')
    ax.set_yticks(range(0, int(maxvalue + 1), 1))
    ax.set_ylim(top=maxvalue)
    ax.grid(visible=True, which='both', axis='y')

    for cat in categorylist:
        temp_df2 = temp_df[temp_df['BedrijfskenmerkenSBI2008'] == cat]
        x_data = temp_df2['Perioden']
        y_data = temp_df2['Ziekteverzuimpercentage_1']

        if cat == 'Totaal gemiddelde':
            linestyle = '-'
            linewidth = 2
        else:
            linestyle = '--'
            linewidth = 1
        ax.plot(x_data, y_data, label = cat, linestyle=linestyle, linewidth=linewidth, marker='o')

    ax_legend = fig.add_subplot(gs[1])
    ax_legend.axis('off')

    # Add the legend manually
    handles, labels = ax.get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc='upper left', ncol=legend_columns)

    fig.tight_layout()
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=6)  # Change fontsize to 12


    #ax.legend(labels=categorylist, fontsize='small', loc='upper center', ncol=3)

    return fig


# %%
def getcategories() -> list:
    categories = df['BedrijfskenmerkenSBI2008'].unique().tolist()
    categories = [x for x in categories if x != 'Totaal gemiddelde']
    return categories

def getmaxvalue() -> tuple:
    return df['Ziekteverzuimpercentage_1'].max().round(-1)

def getperiods() -> tuple:
    lowestyear = int(df['Year'].min())
    highestyear = int(df['Year'].max())
    return (lowestyear, highestyear)

# %%
df = pd.read_pickle(config.str_PathToResourceDataFolder / '80072NED.pkl')

df = df.drop(labels=['ID'], axis=1)

dict_replace = {' 1e kwartaal': 'Q1',
                ' 2e kwartaal': 'Q2',
                ' 3e kwartaal': 'Q3',
                ' 4e kwartaal': 'Q4'}

df['Perioden'] = df['Perioden'].replace(dict_replace, regex=True)

df['Frequency'] = df['Perioden'].apply(lambda x: 'Quarterly' if 'Q' in x else 'Annually')
df['Year'] = df['Perioden'].str[:4]
df['Year'] = df['Year'].astype(int)
df['Perioden'] = df['Perioden'].apply(lambda x: x if len(x) == 4 else x[2:])

df['Ziekteverzuimpercentage_1'] = df['Ziekteverzuimpercentage_1'].astype(np.float64)
df['BedrijfskenmerkenSBI2008'] = df['BedrijfskenmerkenSBI2008'].astype(str)
df['BedrijfskenmerkenSBI2008'] = df['BedrijfskenmerkenSBI2008'].str.replace('A-U Alle economische activiteiten', 'Totaal gemiddelde')
