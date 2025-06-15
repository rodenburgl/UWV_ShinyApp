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

# Formatter function
def to_percent(y, _):
    return f'{y * 100:.0f}%'

def sick_leave_vs_premiums():
    df_temp = config.df_sickleave_vs_premium
    df_temp['Period'] = [str(x) for x in df_temp['Period']]

    legend_columns = 2

    #fig = plt.figure(figsize=(10, figure_height))
    fig = plt.figure()

    gs = GridSpec(2, 1, height_ratios=[1, 1], figure=fig)  # 2 rows: 3/4 for plot, 1/4 for legend

    maxvalue = getmaxvalue()

    ax = fig.add_subplot(gs[0])
    ax.set_title('Actual sickleave % vs. premium sickleave %')
    ax.set_xlabel('Period')
    ax.set_ylabel('Sick\nleave\n% of\ntotal\nworking\ndays',
                  rotation=0,
                  labelpad=25,
                  ha='center',
                  va='center')
    ax.set_yticks(range(0, int(maxvalue + 1), 1))
    ax.set_yticklabels([str(x) + '%' for x in range(0, int(maxvalue + 1), 1)])
    ax.grid(visible=True, which='both', axis='y')

    categories = df_temp['Category'].unique()
    colors = ['red', 'blue']
    
    for i in range(len(categories)):
    # Draw small category
        temp_df2 = df_temp[df_temp['Category'] == categories[i]]
        x_data = temp_df2['Period']
        y_data1 = temp_df2['Sickleave']
        y_data2 = temp_df2['Premium']

        ax.vlines(x=x_data, ymin=y_data1, ymax=y_data2, color=colors[i], linestyles=':')
        ax.scatter(x=x_data, y=y_data2, color=colors[i], s=50, label=categories[i])
        ax.scatter(x=x_data, y=y_data1, facecolors='none', edgecolors=colors[i], s=25)

        #ax.plot(x_data, y_data, label = cat, marker='o', markersize=3)

    ax_legend = fig.add_subplot(gs[1])
    ax_legend.axis('off')

    # Add the legend manually
    handles, labels = ax.get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc='upper left', ncol=legend_columns)

    fig.tight_layout()
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=6)  # Change fontsize to 12
    return fig

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
    legend_columns = 2

    #fig = plt.figure(figsize=(10, figure_height))
    fig = plt.figure()

    gs = GridSpec(2, 1, height_ratios=[1, 1], figure=fig)  # 2 rows: 3/4 for plot, 1/4 for legend

    maxvalue = getmaxvalue()

    ax = fig.add_subplot(gs[0])
    ax.set_title('Sick leave % per business size over time')
    ax.set_xlabel('Period')
    ax.set_ylabel('Sick\nleave\n% of\ntotal\nworking\ndays',
                  rotation=0,
                  labelpad=25,
                  ha='center',
                  va='center')
    ax.set_yticks(range(0, int(maxvalue + 1), 1))
    ax.set_yticklabels([str(x) + '%' for x in range(0, int(maxvalue + 1), 1)])
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
        ax.plot(x_data, y_data, label = cat, linestyle=linestyle, linewidth=linewidth, marker='o', markersize=3)

    ax_legend = fig.add_subplot(gs[1])
    ax_legend.axis('off')

    # Add the legend manually
    handles, labels = ax.get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc='upper left', ncol=legend_columns)

    fig.tight_layout()
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=6)  # Change fontsize to 12
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

df['Perioden'] = df['Perioden'].str.strip()

df['Perioden'] = df['Perioden'].replace(dict_replace, regex=True)

df['Frequency'] = df['Perioden'].apply(lambda x: 'Quarterly' if 'Q' in x else 'Annually')
df['Year'] = df['Perioden'].str[:4]
df['Year'] = df['Year'].astype(int)
df['Perioden'] = df['Perioden'].apply(lambda x: x if len(x) == 4 else x[2:])

df['Ziekteverzuimpercentage_1'] = df['Ziekteverzuimpercentage_1'].astype(np.float64)
df['BedrijfskenmerkenSBI2008'] = df['BedrijfskenmerkenSBI2008'].astype(str)
df['BedrijfskenmerkenSBI2008'] = df['BedrijfskenmerkenSBI2008'].str.replace('A-U Alle economische activiteiten', 'Totaal gemiddelde')
