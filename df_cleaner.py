# python libraries
import os, statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotter import * #plotter functions

# numeri dati mancanti
def get_number_of_null_values(df):
    print(df.isna().sum())


# numeri dati duplicati
def get_number_of_duplicated_values(df):
    print(df.duplicate().sum())


# percentuali di variabili mancanti
def get_percentage_missing_values(df):
    # calculate % missing values for each column
    n_rows = len(df)
    missing = df.isna().sum()
    percentage_missing = missing / n_rows

    # insert data into DataFrame (df) to display
    percentage_missing_df = pd.DataFrame({
        "Missing": percentage_missing
    })
    percentage_missing_df.sort_values("Missing", ascending=False, inplace=True)

    print(percentage_missing_df)

    return percentage_missing_df


# seleziona dati con percentuale mancante < threshold
def select_columns_by_missing_threshold(original_df, percentage_df, threshold):
    columns_to_drop = list(percentage_df[percentage_df['Missing'] > threshold].index)
    print(columns_to_drop)

    original_df.drop(columns=columns_to_drop, axis=1, inplace=True)

    return original_df


# data diventa anno, month, day nel dataframe
def convert_date_into_day_month_year(df):
    df["Event.Date"] = pd.to_datetime(df["Event.Date"])
    df["Year"] = df["Event.Date"].dt.year
    df["Month"] = df["Event.Date"].dt.month
    df["Day"] = df["Event.Date"].dt.day

    return df


# weekend è 1 o 0
def add_flag_weekend(df):
    df.loc[(df['Event.Date'].dt.day_name().str[:3] == 'Sat') | (
                df['Event.Date'].dt.day_name().str[:3] == 'Sun'), 'Weekend'] = 0
    df.loc[(df['Event.Date'].dt.day_name().str[:3] != 'Sat') & (
                df['Event.Date'].dt.day_name().str[:3] != 'Sun'), 'Weekend'] = 1

    return df


# fa il merge di aeroporti con private in PRIVATE, none in NONE e poi printa (ma non taglia) i 10 più frequenti
def merge_same_airports(df):
    df['Airport.Name'].replace(to_replace='(?i)^.*private.*$', value='PRIVATE', inplace=True, regex=True)
    df['Airport.Name'].replace(to_replace='(?i)none', value='NONE', inplace=True, regex=True)
    df['Airport.Code'].replace(to_replace='(?i)none', value='NONE', inplace=True, regex=True)

    # print(df["Airport.Code"].value_counts().nlargest(10))
    # print(df["Airport.Name"].value_counts().nlargest(10))

    return df


def merge_engine_type(df):
    df['Engine.Type'].fillna('Unknown', inplace=True)
    # df['Engine.Type'].replace(to_replace=None, value='NONE', inplace=True, regex=False)
    return df


# analogo di aeroporti e poi in output ci sono i 10 frequenti
def merge_same_registrations(df):
    df["Registration.Number"].replace(to_replace='(?i)none', value='NONE', inplace=True, regex=True)
    df["Registration.Number"].replace(to_replace=['unknown', 'UNK'], value="UNKNOWN", inplace=True, regex=False)

    # print(df["Registration.Number"].value_counts().nlargest(10))

    return df


def merge_aircraftdamage(df):
    df['Aircraft.damage'].fillna('Unknown', inplace=True)
    return df


def merge_purposeofflight(df):
    df['Purpose.of.flight'].fillna('Unknown', inplace=True)
    return df


def merge_enginenumbers(df):
    nengines = df["Number.of.Engines"].values.tolist()
    moda = statistics.mode(nengines)

    nbins = int(max(nengines) - min(nengines) + 1)
    plt.figure()
    plt.hist(nengines, range=(min(nengines) - 0.5, max(nengines) + 0.5), bins=nbins)
    plt.yscale("log")
    plt.title("Number of Engines")
    plt.xlabel("Number of Engines")
    plt.ylabel("Events")
    plt.axvline(moda, color="red")
    plt.savefig("plots/nengines.png")

    df['Number.of.Engines'].fillna(moda, inplace=True)  # commentare per non sostituire vuoti

    return df


def yearclass(y, years):
    out = False  # y not in years (empty bin)
    for iy in range(10):
        if (y + iy in years):
            out = True
            break
    return out


def merge_year(df):
    years = df["Year"].values.tolist()

    yearmin = int(min(years) / 10) * 10
    yearmax = int(max(years) / 10) * 10 + 10

    yearbins = [y for y in range(yearmin, yearmax + 1, 10) if yearclass(y, years)]
    yearbins.append(yearmax)
    yearlabels = [("%is" % y) for iy, y in enumerate(yearbins) if iy != (len(yearbins) - 1)]

    # df['Year'] = pd.cut(x = df['Year'], bins = yearbins, labels = yearlabels, include_lowest = True)

    d = dict(enumerate(yearlabels, 1))
    df['Year'] = np.vectorize(d.get)(np.digitize(df['Year'], yearbins))

    return df


# fisso variabili a valori più standard
def fix_values(df):
    df["Make"] = df["Make"].str.title()  # mette lettere grandi della casa costruttrice

    # sostituisce yes e no con 1 e 0
    df["Amateur.Built"].replace(to_replace=['Yes', 'Y'], value=1, inplace=True, regex=False)
    df["Amateur.Built"].replace(to_replace=['No', 'N'], value=0, inplace=True, regex=False)

    # fatal(0) diventa fatal
    df["Injury.Severity"] = df["Injury.Severity"].str.split('(').str[0]  # seleziona

    print("\n- Counting Class Frequency")
    plot_frequency(df, "Injury.Severity")
    print(df["Injury.Severity"].value_counts())

    # mappa unk e UNK in unknown
    df["Weather.Condition"].replace(to_replace=['Unk', 'UNK'], value='UNKNOWN', inplace=True, regex=False)

    print("\n- Eliminating Unknown from Weather Condition...")
    print(df["Weather.Condition"].value_counts())
    plot_frequency(df, "Weather.Condition")

    print("\n- Eliminating Unknown from Broad Phase of Flight...")
    print(df["Broad.phase.of.flight"].value_counts())
    plot_frequency(df, "Broad.phase.of.flight")

    # toglie dati sconosciuti
    df = df[df['Weather.Condition'] != 'UNKNOWN']
    df = df[df["Injury.Severity"] != "Unavailable"]
    df = df[df["Injury.Severity"] != "Serious"]
    df = df[df["Injury.Severity"] != "Minor"]
    df = df[df["Broad.phase.of.flight"] != "Unknown"]

    # df = df[df["Year"] >= 1982]

    return df


# aggiunge city e state al dataframe
def split_city_state(df):
    df["City"] = df["Location"].str.split(",").str[0]
    df["State"] = df["Location"].str.split(",").str[1]

    return df


def cleaning_nan(df):  # same as transform_data_into_value without conversion (to save csv)

    data = df.copy()

    # binary classification
    data = data[data["Injury.Severity"] != "Unavailable"]
    data = data[data["Injury.Severity"] != "Serious"]
    data = data[data["Injury.Severity"] != "Minor"]
    data = data[data["Injury.Severity"] != "Incident"]

    # replace all infinite values with nan
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # remove all nan values
    data.dropna(inplace=True)

    return data