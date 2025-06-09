"""
Shiny app using shiny core
"""
from resources.icons import icons
import config
from pathlib import Path
from shiny import App, reactive, render, ui
import matplotlib.pyplot as plt
from plot import create_plot
import pandas as pd
import datetime
import seaborn as sns


app_ui = ui.page_navbar(
        #ui.tags.script(src="script.js"),
        # Screen 0 - Welcome
        ui.nav_panel("Welcome",
                     ui.p("Welcome to the UWV project!"),
                     ui.p("This is a demo application to calculate the premiums for sick leave."),
                     ui.p("First, we will show the problem this data science project addresses."),
                        ui.card(
                        ui.card_header("Sick Leave by Industry Over Time"),
                        ui.output_plot("sickleave_over_years")  # Connects to `@output sickleave_over_years`
                        ),
                        ui.card(
                        ui.card_header("Business case"),
                        ui.div(ui.output_plot('cb_premium_percentage_bar_chart'), style = "text-align: center; padding: 20px;")  # Connects to `@output cb_premium_percentage_bar_chart`
                        )    
                    ),

        # Screen 1 - Research
        ui.nav_panel("Research",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_checkbox_group(
                        "selected_models",
                        "Select models to display:",
                        choices=["Predicted_RF", "Auto Arima", "WA"]
                    )),
                ui.card(
                    ui.card_header('Graph of models vs. actual'),
                    ui.output_plot('model_comparison'))
                )),

        # Screen 2 - Calculate premium
        ui.nav_panel("Calculate premium",
                        ui.tags.head(
        ui.tags.script(src="script.js")
    ),
                    ui.layout_columns(
                        ui.card(
                            ui.div(ui.card_header('Your company information'), class_="cardheader"),
                                ui.layout_columns(
                                        ui.div(
                                            ui.div(
                                                ui.input_text('KvK_number', label='KvK number'),
                                                ui.div(
                                                    ui.input_action_button('search_button', label='', icon="🔎", class_='search_button'),
                                                    style="margin: 0px 0px 16px 0px; align-content: end"),
                                                class_='d-flex'),
                                            ui.input_text('Client_number', label='Client number'),
                                            ui.input_text('Company_name', label='Company name')
                                            ),
                                        ui.div(
                                            ui.input_text('Company_address', label='Company address'),
                                            ui.input_text('Company_city', label='City'),
                                            ui.input_text('Company_postal', label='Postal code')
                                            ),
                                        ui.div(
                                            ui.input_selectize('SBI_select', label='SBI category', choices=config.SBI_Categories, selected="A Landbouw, bosbouw en visserij")
                                            )
                        )),
                        ui.card(
                            ui.div(ui.card_header('Company size'), class_="cardheader"),
                            ui.layout_columns(
                                ui.input_numeric('Total_wage', label='Total wage', value=100000, min=0),
                                ui.input_numeric('Number_employees', label='Number of FTE (or employees)', value=1, min=1, max=249),
                            ),
                            ui.value_box(ui.div("Company size"), ui.div(ui.output_text("Company_size"), class_='cardvalue'))
                        )
                    ),
                    ui.card(
                        ui.div(ui.card_header('Calculated premium'), class_="cardheader"),
                        # Premium box positioned next to inputs (right side)
                            ui.value_box(ui.div("Annual premium"),ui.div(ui.output_text("annual_premium_output"), class_='cardvalue'), showcase=icons['currency']),
                            ui.value_box(ui.div("Monthly premium"),ui.div(ui.output_text("monthly_premium_output"), class_='cardvalue'), showcase=icons['currency'])
                        ),
                            ui.tags.div(
                                ui.input_task_button(id="Confirm_premium_Button", label="Confirm premium"),
                                style="text-align: right; padding-right: 20px;"
                            )),
        # Screen 3 - Premium details
        ui.nav_panel("Premium details",
                        ui.card(
                            ui.div(ui.card_header('Premium requests'), class_="cardheader"),
                            ui.div(ui.output_data_frame('data_store_output'), class_='tablestyle')),
                        ui.div(
                            ui.card(
                                ui.card_header('Last 2 years sickleave'),
                                ui.output_plot('Sickleave_P2Y')))
                    ),

        # Screen 4 - Management dashboard
        ui.nav_panel("Dashboard",
                ui.layout_columns(
                    ui.value_box(ui.div("Total numbers of insurances", class_='cardheader'), ui.div(f'{config.df_mock['Client'].unique().size:,}', class_="cardvalue"), showcase=icons['hashtag']),
                    ui.value_box(ui.div("Total incoming premiums (annual)", class_='cardheader'), ui.div(f'{config.df_mock['Premium'].sum().round(0):,.0f}', class_='cardvalue'), showcase=icons['currency']),
                    ui.value_box(ui.div("Estimated covered risk (annual)", class_='cardheader'), ui.div(f'{config.df_mock['Quarterly wage'].sum().round(0):,.0f}', class_='cardvalue'), showcase=icons['currency'])
                    ),
                ui.layout_columns(
                    ui.card(
                        ui.div(ui.card_header('Premiums'), class_="cardheader"),
                        ui.div(ui.output_data_frame('mock_dataframe'), class_='tablestyle')),
                    ui.card(
                        ui.card_header('Predictions vs. actuals'),
                        ui.output_plot('linegraph'))
                    ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header('Blank')),
                    ui.card(
                        ui.card_header('Scatter plot'),
                        ui.output_plot('scatter'))
                    )
            ),

        # Screen 5 - Settings
        ui.nav_panel("Settings", 
            ui.p("This is the settings pane, where you can change important parameters for the calculation of the premiums provided to customers."),
            ui.card(
                ui.input_radio_buttons('Confidence_interval', label='Confidence interval (%)', choices = [90, 95, 99], selected = 90),
                ui.input_numeric('Fixed_costs', label = 'Fixed costs', value = 315, min = 315, max = 600),
                ui.input_slider('Variable_costs', label='Variable cost (%)', min=5, max=50, step=1, value = 15)
            )),
        
        # Top bar with icon
        title=ui.div(
        ui.img(src=config.str_PathToIcon, width="50px", height="50px"),
        ui.div("UWV project", class_="title"),
        fillable=False,
        fluid=True)
     )

# Add the custom CSS globally using ui.tags.head
app_ui = ui.tags.html(
    ui.tags.head(
        ui.include_css(config.str_PathToCSS)),
    app_ui  # Embed the page_sidebar layout
)


def server(input, output, session):

    data_store = reactive.Value(pd.DataFrame(columns=["Client_number", "SBI", "Size", "Date", "Company_name", "Company_address", "Company_city", "Company_postal", "Total_wage", "Number_employees","Premium_sickleave","Premium_variable_costs","Premium_fixed_costs","Premium_total", "Confidence_interval_%"]))

    @render.data_frame
    def mock_dataframe():
        return render.DataGrid(config.df_mock, styles=config.table_styles)
        #return render.DataGrid(config.df_mock)

    @render.plot
    def cb_premium_percentage_bar_chart():
        # Prepare data
        df = config.df_business_case.copy()

        # Group and average
        df_summary = (
            df.groupby(['Sectorgrootte', 'Age dominance', 'Gender dominance'])['Premie prijs_CB']
            .mean()
            .reset_index()
            .rename(columns={
                'Sectorgrootte': 'Company size',
                'Premie prijs_CB': 'Average CB Premium (€)'
            })
        )

        # Create a label for hue
        df_summary["Age-Gender"] = df_summary["Age dominance"] + " / " + df_summary["Gender dominance"]

        # Calculate relative premiums
        df_summary["Relative Premium (%)"] = df_summary.groupby("Company size")["Average CB Premium (€)"].transform(
            lambda x: (x / x.min() * 100).round(0).astype(int)
        )

        # Sort the dataframe by premium
        df_summary = df_summary.sort_values(by=["Company size", "Relative Premium (%)"], ascending=[True, True])

        # Set up the plot
        g = sns.catplot(
            data=df_summary,
            kind="bar",
            y="Age-Gender",
            x="Relative Premium (%)",
            col="Company size",
            col_wrap=1,  # one column per subplot
            height=20,
            aspect=1.8,
            palette="Set2",
            sharex=False
        )

        # Add labels to bars
        for ax in g.axes.flatten():
            for p in ax.patches:
                width = p.get_width()
                ax.text(
                    width - 5,  # shift left slightly
                    p.get_y() + p.get_height() / 2,
                    f'{int(width)}%',
                    ha='right',
                    va='center',
                    color='black',
                    fontsize=10
                )

        g.set_titles("Company Size: {col_name}")
        g.set_axis_labels("Relative Premium (%) versus lowest per company size")
        g.fig.subplots_adjust(hspace = 0.4, top=0.9)
        g.fig.suptitle("Relative CB Premiums by Age and Gender (per Company Size)", fontsize=14)
        for ax in g.axes.flatten():
            ax.set_xlim(0, 300)
            # Optional: set specific x-axis ticks
            ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350])
        
        return g.fig
        #plt.tight_layout()


    @render.plot()
    def sickleave_over_years():
    # Example: select "Totaal gemiddelde", include it, frequency "Quarterly", full period
        categories = ["Totaal gemiddeld"]  # Show only average
        include_total = True
        frequency = "Annually"
        period = (2010, 2024)  # Adjust to your desired range or dynamically fetch

        return create_plot(categories, include_total, frequency, period)

    @render.plot
    def model_comparison():
        df = config.df_model_comparison.copy()
        df['TargetDate'] = pd.to_datetime(df['TargetDate'])

        # Group by TargetDate and aggregate with mean
        df = df.groupby('TargetDate').agg({
            'Actual': 'mean',
            'Predicted_RF': 'mean',
            'Auto Arima': 'mean',
            'WA': 'mean'
        }).reset_index()

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['TargetDate'], df['Actual'], label='Actual', color='black', linewidth=2)

        # Plot selected models
        for model_name in input.selected_models():
            if model_name in df.columns:
                ax.plot(df['TargetDate'], df[model_name], label=model_name, linestyle='--')

        ax.set_title("Model Comparison")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sick Leave %")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        return fig



    @output
    @render.text
    def Company_size():
        return company_size_calc()

    def company_size_calc():
        n = input.Number_employees() or 1
        if n < 10:
            return "Small Enterprise"
        elif n < 100:
            return "Medium Enterprise"
        else:
            return "Large Enterprise"

    @reactive.Effect
    def handle_large_enterprise():
        if company_size_calc() == "Large Enterprise":
            ui.modal_show(ui.modal(
                "Please note this insurance is only aimed at SME's. As such, we do not issue insurances for businesses with more than 150 employees. Please fire a few to continue.",
                title="Too many employees!",
                size="s"
            ))

    def get_predicted_sickleave_rate():
        def get_company_size_label():
            number_employees = input.Number_employees() or 1
            if number_employees < 10:
                return "1 tot 10 werkzame personen"
            elif number_employees < 100:
                return "10 tot 100 werkzame personen"
            else:
                return "100 of meer werkzame personen"

        Latest_Period = "2024Q3"
        SBI_chosen = input.SBI_select()
        def get_confidence_interval():
            if float(input.Confidence_interval()) == 90:
                return 80
            elif float(input.Confidence_interval()) == 95:
                return 90
            elif float(input.Confidence_interval()) == 99:
                return 98

        df_filtered = config.df_mock_y[
            (config.df_mock_y["Period (Q)"] == Latest_Period) &
            (config.df_mock_y["SBI"] == SBI_chosen) &
            (config.df_mock_y["Size"] == get_company_size_label())]

        column_name = f"y_hat_{get_confidence_interval()}_final"

        if not df_filtered.empty:
            return df_filtered[column_name].values[0]
        else:
            return "No data available"

    @output
    @render.text
    def annual_premium_output():
        total_wage = input.Total_wage() or 0
        Premium_fixed_costs = input.Fixed_costs() or 315
        variable_costs_percentage = input.Variable_costs() or 15
        Sick_leave_predicted = get_predicted_sickleave_rate()

        Premium_Sickleave = total_wage * (Sick_leave_predicted / 100)
        Premium_variable_costs = Premium_Sickleave * (variable_costs_percentage / 100)
        Premium_total = Premium_Sickleave + Premium_variable_costs + Premium_fixed_costs

        return f"{Premium_total:,.2f}"

    @output
    @render.text
    def monthly_premium_output():
        total_wage = input.Total_wage() or 0
        Premium_fixed_costs = input.Fixed_costs() or 315
        variable_costs_percentage = input.Variable_costs() or 15
        Sick_leave_predicted = get_predicted_sickleave_rate()

        Premium_Sickleave = total_wage * (Sick_leave_predicted / 100)
        Premium_variable_costs = Premium_Sickleave * (variable_costs_percentage / 100)
        Premium_total = Premium_Sickleave + Premium_variable_costs + Premium_fixed_costs
        premium_monthly = Premium_total / 12

        return f"{premium_monthly:,.2f}"

    @render.plot()
    def scatter():

        fig, ax = plt.subplots()

        temp_df = config.df_predictions.groupby(by=['Year', 'Size'])['Error'].sum().reset_index()

        sizes = temp_df['Size'].unique()

        for size in sizes:
            y = temp_df[temp_df['Size'] == size]['Error']
            x = temp_df[temp_df['Size'] == size]['Year']
            ax.scatter(x=x, y=y)

        ax.set_xlabel('Sick leave %')
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_ylabel('Error')
        ax.set_title('Historical error based on actual sickleave %')
        ax.axhline('0', color='black', linewidth=0.5)

        return fig

    @render.plot()
    def linegraph():
        fig, ax = plt.subplots()
        temp_df = config.df_mock.groupby(by=['Year'])[['Payout', 'Premium']].sum().reset_index()
        temp_df = temp_df[-config.intNumberYearsToShowInGraph:]

        temp_df[['Payout', 'Premium']] = (temp_df[['Payout', 'Premium']] / 1000000).round(1)

        ax.plot(temp_df['Year'], temp_df['Payout'], linestyle='-', linewidth=1, label='payout')
        ax.plot(temp_df['Year'], temp_df['Premium'], linestyle='-', linewidth=1, label='premium')

        ax.set_ylabel(config.dashboard_ylabel_graph)
        ax.set_xlabel(config.dashboard_xlabel_graph)
        ax.set_title(config.dashboard_graph_title)
        ax.legend()

        return fig

    @render.plot()
    def modelgraph():
        fig, ax = plt.subplots()

        # TODO graphs o.b.v. input checkboxen

        current_year = datetime.datetime.now().year
        from_year = current_year - config.intNumberYearsToShowInGraph

        temp_df = config.df_predictions.groupby(by=['SBI', 'Period (Q)', 'Year'])[['y_final', 'y_hat_80_final']].mean().reset_index()
        temp_df = temp_df[temp_df['Year'] >= from_year]
        temp_df['Year'] = temp_df['Year'].astype('string')

        ax.plot(temp_df['Year'], temp_df['y_final'], linestyle='-', linewidth=1)
        ax.plot(temp_df['Year'], temp_df['y_hat_80_final'], linestyle='-', linewidth=1)

        return fig

    @render.plot()
    def Sickleave_P2Y():
        fig, ax = plt.subplots()
        df = config.df_mock_y.copy()

        current_year = df["Period_Date"].dt.year.max()
        last_two_years = current_year - 2

        df_filtered = df[df["Period_Date"].dt.year >= last_two_years]  # Apply the filter

         # Group by 'Period (Q)' and calculate the mean for 'y' and 'y_final'
        df_grouped = df_filtered.groupby("Period (Q)")[['y', 'y_final']].mean().reset_index()

        ax.plot(df_grouped["Period (Q)"], df_grouped["y"], linestyle='-', linewidth=1, label="y")
        ax.plot(df_grouped["Period (Q)"], df_grouped["y_final"], linestyle='--', linewidth=1, label="y_final")

        ax.set_xlabel("Year")
        ax.set_ylabel("Values")
        ax.set_title("Sick Leave Trends (Last 2 Years)")
        ax.legend()

        return fig

    def today_date():
        return str(datetime.date.today())

    @reactive.effect
    @reactive.event(input.Confirm_premium_Button)
    def update_data_request():
        total_wage = input.Total_wage() or 0
        Sick_leave = get_predicted_sickleave_rate()
        Premium_Sickleave = round(total_wage * (float(Sick_leave) / 100), 2)
        Premium_variable_costs = round(Premium_Sickleave * (input.Variable_costs() / 100), 2)
        Premium_total = round(Premium_Sickleave + Premium_variable_costs + input.Fixed_costs(), 2)

        new_entry = pd.DataFrame({
            "Client_number": [input.Client_number()],
            "SBI": [input.SBI_select()],
            "Size": [input.Number_employees()],
            "Date": [today_date()],
            "Company_name": [input.Company_name()],
            "Company_address": [input.Company_address()],
            "Company_city": [input.Company_city()],
            "Company_postal": [input.Company_postal()],
            "Total_wage": [total_wage],
            "Number_employees": [input.Number_employees()],
            "Premium_sickleave": [Premium_Sickleave],
            "Premium_variable_costs": [Premium_variable_costs],
            "Premium_fixed_costs": [input.Fixed_costs()],
            "Premium_total": [Premium_total],
            "Confidence_interval_%": [input.Confidence_interval()]
            })

        data_store.set(pd.concat([data_store.get(), new_entry], ignore_index=True))

        ui.modal_show(ui.modal('Your insurance request has been received. \n\
                                You will receive a response within 24 hours.', size='s'))

    @reactive.effect
    @reactive.event(input.search_button)
    def updateinput():
        text = input.KvK_number()

        df = pd.read_csv(config.str_PathToResourceDataFolder / 'kvk.csv', dtype=str)
        company = df[df['kvk'] == text]

        if len(company) == 1:
            number = company['number'][0]
            name = company['name'][0]
            address = company['address'][0]
            SBI = company['SBI'][0]
            city = company['city'][0]
            postal = company['postal'][0]

            session.send_input_message("Client_number", {"value": number})
            session.send_input_message("Company_name", {"value": name})
            session.send_input_message("Company_address", {"value": address})
            session.send_input_message("Company_city", {"value": city})
            session.send_input_message("Company_postal", {"value": postal})
            session.send_input_message("SBI_select", {"value": SBI})
        else:
            ui.modal_show(ui.modal('The provided number was not found in the \
                                    chamber of commerce. Please make sure the number \
                                    is accurate and try again.', 
                                   title='Number not found',
                                   size='s'))


    @output
    @render.data_frame
    def data_store_output():
        return render.DataGrid(data_store.get(), styles=config.table_styles)

app = App(app_ui, server, static_assets=config.str_PathToResourceDataFolder)
