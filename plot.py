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

    fig, ax = plt.subplots()

    ax.set_title('Actual sickleave % vs. premium sickleave %')
    ax.set_xlabel('Sick leave % of total working days')
    ax.grid(visible=True, which='both', axis='x')

    categories = df_temp['Category'].unique()
    colors = ['red', 'blue']

    for i in range(len(categories)):
    # Draw small category
        temp_df2 = df_temp[df_temp['Category'] == categories[i]]
        x_data1 = temp_df2['Sickleave']
        x_data2 = temp_df2['Premium']
        y_data = temp_df2['Category']

        ax.hlines(y=y_data, xmin=x_data1, xmax=x_data2, color=colors[i], linestyles=':')

        ax.scatter(x=x_data1, y=y_data, facecolors='none', edgecolors=colors[i], s=25)
        ax.scatter(x=x_data2, y=y_data, color=colors[i], s=50, label=categories[i])

    ax.set_xticks(ax.get_xticks(), [str(x) + '%' for x in ax.get_xticks()])
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=6)

    ax.set_axisbelow(True)
    ax.set_ylim(-0.5, max(ax.get_yticks()) + 0.5)

    return fig

def premium_diff_man_woman():
    df_temp = config.df_premium_diff_man_woman

    fig, ax = plt.subplots()

    ax.set_title('Premium spread man vs. woman')
    ax.set_xlabel('Sick leave % of total working days')
    ax.grid(visible=True, which='both', axis='x')

    categories = df_temp['Category'].unique()
    colors = ['red', 'blue', 'green']
    linewidth = 3

    for i in range(len(categories)):
    # Draw small category
        temp_df2 = df_temp[df_temp['Category'] == categories[i]]
        x_data1 = temp_df2['Woman']
        x_data2 = temp_df2['Man']
        x_data3 = round((x_data1 + x_data2) / 2, 1)
        y_data = temp_df2['Category']

        ax.hlines(y=y_data, xmin=x_data3, xmax=x_data1, color=colors[0], linestyles=':', linewidth=linewidth)
        ax.hlines(y=y_data, xmin=x_data2, xmax=x_data3, color=colors[2], linestyles=':', linewidth=linewidth)

        legend1 = 'Woman only premium' if i == 0 else None
        legend2 = 'Man only premium' if i == 0 else None
        legend3 = 'Mixed premium' if i == 0 else None

        ax.scatter(x=x_data3, y=y_data, facecolors='none', edgecolors=colors[1], s=75, label=legend3)
        ax.scatter(x=x_data1, y=y_data, color=colors[0], s=100, label=legend1)
        ax.scatter(x=x_data2, y=y_data, color=colors[2], s=100, label=legend2)

    ax.legend(loc='lower right')


    ax.set_xticks(ax.get_xticks(), [str(x) + '%' for x in ax.get_xticks()])
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=6)

    ax.set_axisbelow(True)
    ax.set_ylim(-0.5, max(ax.get_yticks()) + 0.5)

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
