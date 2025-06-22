"""
Shiny app using shiny core
"""
from resources.icons import icons
import config
from pathlib import Path
from shiny import App, reactive, render, ui
import matplotlib.pyplot as plt
from plot import create_plot, sick_leave_vs_premiums, premium_diff_man_woman
import pandas as pd
import datetime
import seaborn as sns
from matplotlib.ticker import FuncFormatter


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
                        ui.card_header("Sick Leave premium vs. sick leave %"),
                        ui.output_plot("sickleave_vs_premiums")
                        ),
                        ui.card(
                        ui.card_header("Premium spread"),
                        ui.output_plot('plot_premium_diff_man_woman')
                        )    
                    ),

        # Screen 1 - Research
        ui.nav_panel("Research",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_checkbox_group(
                        "selected_baseline_models",
                        "Select models to display:",
                        choices=["Predicted_RF", "AutoARIMA", "WA"]
                    ),
                    ui.panel_conditional(
                    "input.selected_baseline_models.includes('AutoARIMA')",
                    ui.input_checkbox_group(
                        "selected_sarimax_variants",
                        "Select SARIMAX variants with features:",
                        choices=[
                            "AutoARIMA - no features",
                            "AutoARIMA - all features"
                        ],
                        selected=[]
                    )
                )
                    ),
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
        # # Screen 3 - Premium details
        # ui.nav_panel("Premium details",
        #                 ui.card(
        #                     ui.div(ui.card_header('Premium requests'), class_="cardheader"),
        #                     ui.div(ui.output_data_frame('data_store_output'), class_='tablestyle')),
        #                 ui.div(
        #                     ui.card(
        #                         ui.card_header('Last 2 years sickleave'),
        #                         ui.output_plot('Sickleave_P2Y')))
        #             ),

        # Screen 4 - Management dashboard
        ui.nav_panel("Dashboard",
                ui.layout_columns(
                        ui.value_box(
                            ui.div("Total risk WA (annual)", class_='cardheader'),
                            ui.div(f"€{config.total_risks['WA'] / 1_000_000:,.1f} mln", class_='cardvalue'),
                            showcase=icons['currency']
                        ),

                        ui.value_box(
                            ui.div("Total risk AutoARIMA (annual)", class_='cardheader'),
                            ui.div(f"€{config.total_risks['AutoARIMA'] / 1_000_000:,.1f} mln", class_='cardvalue'),
                            showcase=icons['currency']
                        ),

                        ui.value_box(
                            ui.div("Risk reduction (annual)", class_='cardheader'),
                            ui.div(f"€{config.total_risks['Reduction'] / 1_000_000:,.1f} mln", class_='cardvalue'),
                            showcase=icons['currency']
                        )
                    ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header('Financial risk per sector'),
                        ui.output_plot('horizontal_bar_bias_comparison', height = "600px"))
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
    def plot_premium_diff_man_woman():
        return premium_diff_man_woman()


    @render.plot()
    def sickleave_over_years():
    # Example: select "Totaal gemiddelde", include it, frequency "Quarterly", full period
        categories = ['1 tot 10 werkzame personen',
                      '10 tot 100 werkzame personen',
                      '100 of meer werkzame personen']  # Show only average
        include_total = True
        frequency = "Quarterly"
        period = (2018, 2024)  # Adjust to your desired range or dynamically fetch

        return create_plot(categories, include_total, frequency, period)

    @render.plot()
    def sickleave_vs_premiums():
        return sick_leave_vs_premiums()

    @render.plot
    def model_comparison():
        df = config.df_model_comparison.copy()
        df['TargetDate'] = pd.to_datetime(df['TargetDate'])

        # Combine selected models and SARIMAX variants
        selected_models = list(input.selected_baseline_models())
        if "AutoARIMA" in selected_models:
            selected_models.remove("AutoARIMA")
            selected_models += list(input.selected_sarimax_variants())

        # Filter only existing columns
        existing_models = [m for m in selected_models if m in df.columns]
        cols_to_plot = ['Actual'] + existing_models

        # Group and aggregate
        df = df.groupby('TargetDate')[cols_to_plot].mean().reset_index()

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['TargetDate'], df['Actual'], label='Actual', color='black', linewidth=2)

        # Define colors
        colors = {
            "Predicted_RF": "blue",
            "WA": "green",
            "AutoARIMA - No feature": "orange",
            "AutoARIMA - COVID19": "red",
            "AutoARIMA - Overweight": "purple",
            "AutoARIMA.- Happiness": "brown",
            "AutoARIMA - Financial difficulty": "pink"
        }

        # Add model lines
        for model in existing_models:
            ax.plot(df['TargetDate'], df[model], label=model, linestyle='--', color=colors.get(model, None))

        ax.set_title("Model Comparison")
        ax.set_xlabel("Target Date")
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

    @render.plot
    def horizontal_bar_bias_comparison():
        import matplotlib.pyplot as plt

        # Copy and sort data
        df = config.bias_wage_table.copy()
        df = df.sort_values("Sector")  # alphabetical sort

        sectors = df["Sector"]
        wa_risk = pd.to_numeric(df["Financial risk WA"], errors="coerce")
        autoARIMA_risk = pd.to_numeric(df["Financial risk AutoARIMA"], errors="coerce")
        y_pos = range(len(sectors))

        fig, ax = plt.subplots(figsize=(15, len(sectors) * 0.5))

        # Grey background bars (WA bias)
        ax.barh(y_pos, wa_risk, color='lightgrey', edgecolor='black', height=0.6, label="WA")

        # Colored overlay bars (AutoARIMA bias)
        ax.barh(y_pos, autoARIMA_risk, color='steelblue', height=0.4, label="AutoARIMA", alpha=0.9)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sectors)
        ax.set_title("Financial risk per Sector: AutoARIMA vs WA")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1_000_000:.1f}'))
        ax.set_xlabel("Financial Risk (in millions of EUR)")
        ax.legend()
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        # Remove spines for cleaner look
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)

        fig.tight_layout()
        return fig


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
