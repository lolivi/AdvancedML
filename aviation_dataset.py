import os
import pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score

#restituisce il dataframe di pandas
def read_file_w_pandas(filename):
    assert os.path.exists(filename), 'file %s does not exist' % filename
    df = pandas.read_csv(filename, encoding = "ISO-8859-1")
    return df

#restituisce a schermo le colonne
def show_all_columns(df):
    print(df.columns.values.tolist())

#restituisce a schermo le info sul dataframe
def get_info(df):
    print(df.info())

#Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution
def describe_data(df):
    print(df.describe(include=object))

#numeri dati mancanti
def get_number_of_null_values(df):
    print(df.isna().sum())

#numeri dati duplicati
def get_number_of_duplicated_values(df):
    print(df.duplicate().sum())


def get_percentage_missing_values(df):
    # calculate % missing values for each column
    n_rows = len(df)
    missing = df.isna().sum()
    percentage_missing = missing / n_rows

    # insert data into DataFrame (df) to display
    percentage_missing_df = pandas.DataFrame({
        "Missing": percentage_missing
    })
    percentage_missing_df.sort_values("Missing", ascending=False, inplace=True)

    print(percentage_missing_df)

    return percentage_missing_df

#rimuove colonne con dati mancanti spora la soglia
def select_columns_by_missing_threshold(original_df, percentage_df, threshold):

    columns_to_drop = list(percentage_df[percentage_df['Missing'] > threshold].index)
    print(columns_to_drop)

    original_df.drop(columns=columns_to_drop, axis = 1, inplace = True)

    return original_df


def convert_date_into_day_month_year(df):

    df["Event.Date"] = pandas.to_datetime(df["Event.Date"])
    df["Year"] = df["Event.Date"].dt.year
    df["Month"] = df["Event.Date"].dt.month
    df["Day"] = df["Event.Date"].dt.day

    return df


def add_flag_weekend(df):

    df.loc[(df['Event.Date'].dt.day_name().str[:3] == 'Sat') | (df['Event.Date'].dt.day_name().str[:3] == 'Sun'), 'Weekend'] = 0
    df.loc[(df['Event.Date'].dt.day_name().str[:3] != 'Sat') & (df['Event.Date'].dt.day_name().str[:3] != 'Sun'), 'Weekend'] = 1

    return df


def merge_same_airports(df):

    df['Airport.Name'].replace(to_replace='(?i)^.*private.*$', value='PRIVATE', inplace=True, regex=True)
    df['Airport.Name'].replace(to_replace='(?i)none', value='NONE', inplace=True, regex=True)
    df['Airport.Code'].replace(to_replace='(?i)none', value='NONE', inplace=True, regex=True)

    print(df["Airport.Code"].value_counts().nlargest(10))
    print(df["Airport.Name"].value_counts().nlargest(10))

    return df


def merge_same_registrations(df):

    df["Registration.Number"].replace(to_replace='(?i)none', value='NONE', inplace=True, regex=True)
    df["Registration.Number"].replace(to_replace=['unknown', 'UNK'], value="UNKNOWN", inplace=True, regex=False)

    print(df["Registration.Number"].value_counts().nlargest(10))

    return df


def fix_values(df):

    df["Make"] = df["Make"].str.title()

    df["Amateur.Built"].replace(to_replace=['Yes', 'Y'], value=1, inplace=True, regex=False)
    df["Amateur.Built"].replace(to_replace=['No', 'N'], value=0, inplace=True, regex=False)

    df["Injury.Severity"] = df["Injury.Severity"].str.split('(').str[0]

    df["Weather.Condition"].replace(to_replace=['Unk', 'UNK'], value='UNKNOWN', inplace=True, regex=False)

    df = df[df['Weather.Condition'] != 'UNKNOWN']
    df = df[df["Injury.Severity"] != "Unavailable"]
    df= df[df["Injury.Severity"] != "Serious"]
    df = df[df["Injury.Severity"] != "Minor"]
    df = df[df["Broad.phase.of.flight"] != "Unknown"]

    df = df[df["Year"] >= 1982]

    return df


def split_city_state(df):

    df["City"] = df["Location"].str.split(",").str[0]
    df["State"] = df["Location"].str.split(",").str[1]

    return df


def plot_accidents_per_year(df):

    accidents_per_year = df.groupby(['Year'], as_index=False)['Event.Id'].count()
    sns.lineplot(x='Year', y='Event.Id', data=accidents_per_year, color='#2990EA')
    plt.show()


def plot_accidents_based_on_weather(df):

    accidents_according_to_weather = df.groupby("Weather.Condition")["Event.Id"].count()
    accidents_according_to_weather.plot.bar(stacked = True, color = ['#003366','#2990EA'])

    plt.xticks(rotation=0)
    plt.xlabel('')
    plt.show()


def plot_accidents_based_on_injuriy(df):

    weather_accident = df.groupby("Weather.Condition")["Injury.Severity"]\
        .value_counts(normalize=True)\
        .unstack("Injury.Severity")

    colors = list(sns.color_palette("Set2"))[:3]
    weather_accident.plot.bar(stacked=True, color = colors)

    plt.xticks(rotation=0)
    plt.title('Fatality rate during IMC or VMC')
    plt.xlabel('')
    plt.legend(title="Injury Severity", loc='upper right')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.show()


def plot_correlation_matrix(df):

    f = plt.figure(figsize=(6, 6))
    sns.heatmap(df.corr(), vmin=-1, vmax=1, cmap="mako")
    plt.tight_layout()
    plt.show()


def plot_phase_of_flight(df):

    phase_of_flight = df.groupby("Broad.phase.of.flight")["Injury.Severity"]\
        .value_counts(normalize=True).unstack("Injury.Severity")

    phase_of_flight.plot.bar(stacked=True)

    plt.xlabel('')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    plt.show()


def plot_investigation_type(df):

    sns.catplot(x='Investigation.Type', y='Total.Fatal.Injuries', data=df)
    plt.tight_layout()
    plt.show()


def plot_number_of_engines(df):

    sns.catplot(x='Engine.Type', y='Total.Fatal.Injuries', data=df)

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_injuries(df):

    sns.catplot(x='Total.Minor.Injuries', y='Total.Fatal.Injuries', data=df)

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def transform_data_into_value(df):

    data = df.copy()
    data = data[data["Injury.Severity"] != "Unavailable"]
    data = data[data["Injury.Severity"] != "Serious"]
    data = data[data["Injury.Severity"] != "Minor"]
    data = data[data["Injury.Severity"] != "Incident"]

    data['Investigation.Type'] = data['Investigation.Type'].astype('category').cat.codes
    data['Aircraft.damage'] = data['Aircraft.damage'].astype('category').cat.codes
    data['Engine.Type'] = data['Engine.Type'].astype('category').cat.codes
    data['Purpose.of.flight'] = data['Purpose.of.flight'].astype('category').cat.codes
    data['Weather.Condition'] = data['Weather.Condition'].astype('category').cat.codes
    data['Broad.phase.of.flight'] = data['Broad.phase.of.flight'].astype('category').cat.codes

    data["Injury.Severity"] = data["Injury.Severity"].astype('category').cat.codes

    # replace all infinite values with nan
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # remove all nan values
    data.dropna(inplace=True)

    return data


def prepare_train_test(df):

    df = transform_data_into_value(df)

    X = df[['Investigation.Type', "Aircraft.damage",
            'Number.of.Engines','Engine.Type','Purpose.of.flight',
            'Year','Month','Day']]

    y = df["Injury.Severity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)

    return X_train, X_test, y_train, y_test


def train_tree_classifier(df):

    print("Training...")
    X_train, X_test, y_train, y_test = prepare_train_test(df)
    n_features = X_train.shape[1]
    tree = RandomForestClassifier()

    param_grid = {
        'min_samples_split': [2, 5, 7],
        'max_depth': [5, 10, 15, 20],
        'max_features': ['sqrt', 'log2', n_features],
        'min_samples_leaf': [2, 3, 4],
        'n_estimators': [100, 500, 1000, 1500],
        'ccp_alpha': [0., 1e-5, 5e-5], #0 no pruning
        "random_state": [42]
    }

    search = GridSearchCV(tree, param_grid=param_grid, n_jobs=-1, cv=3)
    search.fit(X_train, y_train)

    print(search.best_params_)

    print('Train Set')
    evaluate(search.predict(X_train), y_train)
    print('Test Set')
    evaluate(search.predict(X_test), y_test)


def evaluate(y_pred, y_gold):

    for avg in ['micro', 'macro']:
        print("precision score {}: {:3f}".format(avg, precision_score(y_gold, y_pred, average=avg)))
        print("recall score {}: {:3f}".format(avg, recall_score(y_gold, y_pred, average=avg)))
        print("F1 score {}: {:3f}".format(avg, f1_score(y_gold, y_pred, average=avg)))


def main():

    df = read_file_w_pandas(aviation_dataset)

    #Injury.Severity è il nostro output -> "Incident" possiamo utilizzarlo o no, ma in caso va riscalato a 2000
    #Fatal1, Fatal2, ... mettilo in un'unica classe Fatal

    #vedi come prima cosa correlazione fra questo e numero di feriti
    #anno e numero di feriti

    show_all_columns(df)
    get_number_of_null_values(df)

    percentage_missing_df = get_percentage_missing_values(df)
    dropped_df = select_columns_by_missing_threshold(df, percentage_missing_df, 0.5)
    extended_date_df = convert_date_into_day_month_year(dropped_df)
    extended_date_df = add_flag_weekend(extended_date_df)
    merged_airport_df = merge_same_airports(extended_date_df)
    merged_registration_df = merge_same_registrations(merged_airport_df)
    fixed_df = fix_values(merged_registration_df)
    city_state_df = split_city_state(fixed_df)

    #plot_accidents_per_year(city_state_df)
    #plot_accidents_based_on_weather(city_state_df)
    #plot_accidents_based_on_injuriy(city_state_df)
    #plot_correlation_matrix(city_state_df)
    #plot_phase_of_flight(city_state_df)
    #plot_investigation_type(city_state_df)
    #plot_number_of_engines(city_state_df)
    plot_injuries(city_state_df)

    #train_tree_classifier(city_state_df)


if __name__ == '__main__':

    aviation_dataset = './AviationData.csv'

    main()

