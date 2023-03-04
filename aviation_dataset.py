import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotter import * #plotter functions
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import preprocessing
import statistics, joblib

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

#variabili da tenere con verifiche
cleanvars = ["Injury.Severity","Investigation.Type","Country","Aircraft.damage","Amateur.Built","Number.of.Engines","Engine.Type","Purpose.of.flight","Total.Fatal.Injuries","Total.Serious.Injuries","Total.Minor.Injuries","Total.Uninjured","Weather.Condition","Broad.phase.of.flight","Year","Month"]

#restituisce il dataframe di pandas
def read_file_w_pandas(filename):
    assert os.path.exists(filename), 'file %s does not exist' % filename
    df = pd.read_csv(filename, encoding = "ISO-8859-1")
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

#percentuali di variabili mancanti
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

#seleziona dati con percentuale mancante < threshold
def select_columns_by_missing_threshold(original_df, percentage_df, threshold):

    columns_to_drop = list(percentage_df[percentage_df['Missing'] > threshold].index)
    print(columns_to_drop)

    original_df.drop(columns=columns_to_drop, axis = 1, inplace = True)

    return original_df

#data diventa anno, month, day nel dataframe
def convert_date_into_day_month_year(df):

    df["Event.Date"] = pd.to_datetime(df["Event.Date"])
    df["Year"] = df["Event.Date"].dt.year
    df["Month"] = df["Event.Date"].dt.month
    df["Day"] = df["Event.Date"].dt.day

    return df

#weekend è 1 o 0
def add_flag_weekend(df):

    df.loc[(df['Event.Date'].dt.day_name().str[:3] == 'Sat') | (df['Event.Date'].dt.day_name().str[:3] == 'Sun'), 'Weekend'] = 0
    df.loc[(df['Event.Date'].dt.day_name().str[:3] != 'Sat') & (df['Event.Date'].dt.day_name().str[:3] != 'Sun'), 'Weekend'] = 1

    return df

#fa il merge di aeroporti con private in PRIVATE, none in NONE e poi printa (ma non taglia) i 10 più frequenti
def merge_same_airports(df):

    df['Airport.Name'].replace(to_replace='(?i)^.*private.*$', value='PRIVATE', inplace=True, regex=True)
    df['Airport.Name'].replace(to_replace='(?i)none', value='NONE', inplace=True, regex=True)
    df['Airport.Code'].replace(to_replace='(?i)none', value='NONE', inplace=True, regex=True)

    #print(df["Airport.Code"].value_counts().nlargest(10))
    #print(df["Airport.Name"].value_counts().nlargest(10))

    return df
    
def merge_engine_type(df):

    df['Engine.Type'].fillna('Unknown', inplace = True)
    #df['Engine.Type'].replace(to_replace=None, value='NONE', inplace=True, regex=False)
    return df
   
#analogo di aeroporti e poi in output ci sono i 10 frequenti
def merge_same_registrations(df):

    df["Registration.Number"].replace(to_replace='(?i)none', value='NONE', inplace=True, regex=True)
    df["Registration.Number"].replace(to_replace=['unknown', 'UNK'], value="UNKNOWN", inplace=True, regex=False)

    #print(df["Registration.Number"].value_counts().nlargest(10))

    return df

def merge_aircraftdamage(df):

    df['Aircraft.damage'].fillna('Unknown', inplace = True)
    return df

def merge_purposeofflight(df):

    df['Purpose.of.flight'].fillna('Unknown', inplace = True)
    return df

def merge_enginenumbers(df):

    nengines = df["Number.of.Engines"].values.tolist()
    moda = statistics.mode(nengines)

    nbins = int(max(nengines)-min(nengines)+1)
    plt.figure()
    plt.hist(nengines, range = (min(nengines)-0.5,max(nengines)+0.5), bins=nbins)
    plt.yscale("log")
    plt.axvline(moda, color = "red")
    plt.savefig("plots/nengines.png")

    #df['Number.of.Engines'].fillna(moda, inplace=True) #commentare per non sostituire vuoti

    return df


#fisso variabili a valori più standard
def fix_values(df):

    df["Make"] = df["Make"].str.title() #mette lettere grandi della casa costruttrice

    #sostituisce yes e no con 1 e 0
    df["Amateur.Built"].replace(to_replace=['Yes', 'Y'], value=1, inplace=True, regex=False)
    df["Amateur.Built"].replace(to_replace=['No', 'N'], value=0, inplace=True, regex=False)

    #fatal(0) diventa fatal
    df["Injury.Severity"] = df["Injury.Severity"].str.split('(').str[0] #seleziona

    #mappa unk e UNK in unknown
    df["Weather.Condition"].replace(to_replace=['Unk', 'UNK'], value='UNKNOWN', inplace=True, regex=False)

    #toglie dati sconosciuti
    df = df[df['Weather.Condition'] != 'UNKNOWN']
    df = df[df["Injury.Severity"] != "Unavailable"]
    df = df[df["Injury.Severity"] != "Serious"]
    df = df[df["Injury.Severity"] != "Minor"]
    df = df[df["Broad.phase.of.flight"] != "Unknown"]
    df = df[df["Year"] >= 1982]

    return df

#aggiunge city e state al dataframe
def split_city_state(df):

    df["City"] = df["Location"].str.split(",").str[0]
    df["State"] = df["Location"].str.split(",").str[1]

    return df

def transform_data_into_value(df):

    print("\n- Binary Classification e Conversione in Categorie")

    #binary classification
    data = df.copy()

    data = data[data["Injury.Severity"] != "Unavailable"]
    data = data[data["Injury.Severity"] != "Serious"]
    data = data[data["Injury.Severity"] != "Minor"]
    data = data[data["Injury.Severity"] != "Incident"]

    #convering strings to codes
    data['Investigation.Type'] = data['Investigation.Type'].astype('category').cat.codes
    data['Country'] = data['Country'].astype('category').cat.codes
    data['Aircraft.damage'] = data['Aircraft.damage'].astype('category').cat.codes
    data['Engine.Type'] = data['Engine.Type'].astype('category').cat.codes
    data['Purpose.of.flight'] = data['Purpose.of.flight'].astype('category').cat.codes
    data['Weather.Condition'] = data['Weather.Condition'].astype('category').cat.codes
    data['Broad.phase.of.flight'] = data['Broad.phase.of.flight'].astype('category').cat.codes

    data["Injury.Severity"] = data["Injury.Severity"].astype('category').cat.codes
    data["Injury.Severity"].replace([-1], np.nan, inplace=True)

    # replace all infinite values with nan
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # remove all nan values
    data.dropna(inplace=True)

    return data

def feature_selection(df,variables): #variables da eliminare

    cleandf = df[variables]
    plot_correlation_matrix(cleandf)

    #month perché non è correlato a nulla tranne che a meteo, e quindi è ridondante
    #country è correlato a meteo ed è ridonante
    #broad phase of flight ha troppi vuoti
    #excludevars = ["Month","Country","Broad.phase.of.flight"]

    #escludiamo anche i total injuries perché basterebbe fatal injuries per il 100%, forniamo al classificatore un informazione ridonndante con l'output
    excludevars = ["Month","Country","Broad.phase.of.flight","Total.Fatal.Injuries","Total.Serious.Injuries","Total.Minor.Injuries","Total.Uninjured"]

    newvars = [var for var in variables if var not in excludevars]
    finaldf = df[newvars]

    return finaldf

def preprocessing_data(X_train,X_test):

    #scaler with mean and dev standard
    scaler = preprocessing.StandardScaler().fit(X_train)
    mu_train = scaler.mean_
    std_train = scaler.scale_

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train,X_test

def data_augmentation(X,y,method = SMOTETomek(random_state=42)):

    print("\n- Data Augmentation with %s" % method)

    print("Pre-Augmentation:")
    print("Class 0: %i" % y.value_counts()[0])
    print("Class 1: %i" % y.value_counts()[1])

    X, y = method.fit_resample(X, y)

    print("Post-Augmentation:")
    print("Class 0: %i" % y.value_counts()[0])
    print("Class 1: %i" % y.value_counts()[1])

    return X,y

def prepare_train_test(df):

    #df = transform_data_into_value(df)
    columns = df.columns.values.tolist()
    features = [col for col in columns if col != "Injury.Severity"]

    X = df[features]
    #X = df[["Total.Fatal.Injuries"]] #99% with only this

    y = df["Injury.Severity"]

    return X,y

def split_train_test(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle = True, random_state = 42)

    print("\n- Split Train/Test")
    print("Train: %i" % len(X_train))
    print("Test: %i" % len(X_test))

    return X_train, X_test, y_train, y_test

def cross_validation(clf,param_grid,ncv,X_train,y_train,X_test,y_test,name_output,dir_output):

    print("\n- Training...")
    search = GridSearchCV(clf, param_grid = param_grid, n_jobs=-1, cv = ncv, scoring="f1_macro")
    search.fit(X_train, y_train)

    print('\n- Performance:')
    print("- Train Set:")
    evaluate(search.predict(X_train), y_train)
    print("- Test Set:")
    evaluate(search.predict(X_test), y_test)

    #saving model
    if (not os.path.exists(dir_output)): os.makedirs(dir_output)
    joblib.dump(clf,dir_output+name_output)

    #plotting learning curve
    train_sizes = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    plot_learningcurve(clf,X_train,y_train,train_sizes,ncv,dir_output)
    
    resultfile=open('svm/result.txt','w')
    
    resultfile.write('\n SVM best parameters: \n')
    for item in search.best_params_:
        resultfile.write(search.best_params_[item])
    
    resultfile.close()
    
    
def evaluate(y_pred, y_gold):

    #for avg in ['micro', 'macro']:
    for avg in ['macro']:
        print("precision score {}: {:3f}".format(avg, precision_score(y_gold, y_pred, average=avg)))
        print("recall score {}: {:3f}".format(avg, recall_score(y_gold, y_pred, average=avg)))
        print("F1 score {}: {:3f}".format(avg, f1_score(y_gold, y_pred, average=avg)))

def plot_learningcurve(clf,X_train,y_train,train_sizes,n_folds,directory):

    print('\n- Plotting Learning Curve:')

    train_sizes, train_scores, validation_scores = learning_curve(estimator=clf, X=X_train, y=y_train, train_sizes = train_sizes,cv=n_folds, scoring="f1_macro", shuffle=True, random_state=42)
    train_scores = train_scores * 100.
    validation_scores = validation_scores * 100.

    train_scores_mean = train_scores.mean(axis=1)
    validation_scores_mean = validation_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, label="Training F1", c="b", marker="o", linestyle="dashed")
    plt.plot(train_sizes, validation_scores_mean, label=("Validation F1 (CV = %i)" % n_folds), c="g", marker="o",linestyle="dashed")
    plt.ylabel("F1 [%s]" % ("%"))
    plt.xlabel("Training Set Size")
    plt.title("Learning Curves")
    plt.legend(loc = "best")
    # plt.ylim(90,100)
    plt.savefig(directory+"learningcurve.png")

def main():

    df = read_file_w_pandas(aviation_dataset)

    #Injury.Severity è il nostro output -> "Incident" possiamo utilizzarlo o no, ma in caso va riscalato a 2000
    #Fatal1, Fatal2, ... mettilo in un'unica classe Fatal

    #vedi come prima cosa correlazione fra questo e numero di feriti
    #anno e numero di feriti

    print("---------------------------")
    print("----AVIATION DATASET-------")

    print("\n- Colonne Dataframe: ")
    show_all_columns(df)

    print("\n- Numero Valori Mancanti")
    get_number_of_null_values(df)

    print("\n- Percentuale Valori Mancanti")
    percentage_missing_df = get_percentage_missing_values(df)

    print("\n- Escludo Colonne con Percentuale > 0.5")
    dropped_df = select_columns_by_missing_threshold(df, percentage_missing_df, 0.5)

    extended_date_df = convert_date_into_day_month_year(dropped_df)
    extended_date_df = add_flag_weekend(extended_date_df)

    merged_airport_df = merge_same_airports(extended_date_df)
    merged_registration_df = merge_same_registrations(merged_airport_df)
    merged_engine_df = merge_engine_type(merged_registration_df)
    merged_damage_df = merge_aircraftdamage(merged_engine_df)
    merged_pof_df = merge_purposeofflight(merged_damage_df)
    merged_nengines_df = merge_enginenumbers(merged_pof_df)

    fixed_df = fix_values(merged_nengines_df)
    city_state_df = split_city_state(fixed_df)

    # plot_accidents_per_year(city_state_df)
    # plot_accidents_based_on_weather(city_state_df)
    # plot_accidents_based_on_injuriy(city_state_df)
    # plot_correlation_matrix(city_state_df)
    # plot_phase_of_flight(city_state_df)
    # plot_investigation_type(city_state_df)
    # plot_number_of_engines(city_state_df)
    # plot_injuries(city_state_df)
    # plot_amateur_engines(city_state_df)
    # plot_flight_purpose(city_state_df)
    # plot_engine_type(city_state_df)

    # city_state_df.to_csv("clean_dataset.csv")

    #feature selection
    df_categorical = transform_data_into_value(city_state_df)
    clean_df = feature_selection(df_categorical,cleanvars)
    X, y = prepare_train_test(clean_df)

    # data augmentation
    #method = RandomUnderSampler(random_state=42)
    method = SMOTETomek(random_state=42)
    X, y = data_augmentation(X, y, method)

    # preprocessing
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    X_train, X_test = preprocessing_data(X_train, X_test)
    n_features = X_train.shape[1]

    #choosing classifier
    '''
    clf = RandomForestClassifier(random_state=42)
    nameclf = "rndmforest.joblib"
    dirclf = "rndmforest/"

    #choosing param grid
    param_grid = {
        'min_samples_split': [2, 5, 7],
        'max_depth': [5, 10, 15, 20],
        'max_features': ['sqrt', 'log2', n_features],
        'min_samples_leaf': [2, 3, 4],
        'n_estimators': [100, 500, 1000, 1500],
        'ccp_alpha': [0., 1e-5, 5e-5],  # 0 no pruning
        "random_state": [42]
    }

    '''
    clf = SVC()
    nameclf = "svm.joblib"
    dirclf = "svm/"
    
    param_grid = {
        "kernel": ["linear", "rbf", "poly"],
        "C": [1.0, 10.0, 100.0],
        "degree": [2, 3],
        "random_state": [42]
    }

    param_grid = {
        "kernel": ["linear"]
    }
    

    '''
    clf = LogisticRegression()
    nameclf = "logreg.joblib"
    dirclf = "logreg/"

    param_grid = {
        "penalty": ["l2"]
    }
    '''


    cross_validation(clf,param_grid,5,X_train,y_train,X_test,y_test,nameclf,dirclf)

    '''
    clf.fit(X_train,y_train)
    print('Train Set')
    evaluate(clf.predict(X_train), y_train)
    print('Test Set')
    evaluate(clf.predict(X_test), y_test)
    '''

if __name__ == '__main__':
    aviation_dataset = './AviationData.csv'
    main()

