#python libraries
import os, statistics, joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#plotter.py
from plotter import * #plotter functions
if (not os.path.exists("plots")): os.makedirs("plots")

#sklearn models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, CategoricalNB

#sklearn processing functions
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
from sklearn import preprocessing

#sklearn metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

#augmentation libraries
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

#variabili da tenere con verifiche
cleanvars = ["Injury.Severity","Investigation.Type","Country","Aircraft.damage","Amateur.Built","Number.of.Engines","Engine.Type","Purpose.of.flight","Weather.Condition","Broad.phase.of.flight","Year","Month"]
plotopt = False #whether to plot figs or not

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
    plt.title("Number of Engines")
    plt.xlabel("Number of Engines")
    plt.ylabel("Events")
    plt.axvline(moda, color = "red")
    plt.savefig("plots/nengines.png")

    df['Number.of.Engines'].fillna(moda, inplace=True) #commentare per non sostituire vuoti

    return df

def yearclass(y,years):

    out = False #y not in years (empty bin)
    for iy in range(10):
        if (y+iy in years):
            out = True
            break
    return out

def merge_year(df):

    years = df["Year"].values.tolist()

    yearmin = int(min(years)/10)*10
    yearmax = int(max(years)/10)*10 + 10

    yearbins = [y for y in range(yearmin,yearmax+1,10) if yearclass(y,years)]
    yearbins.append(yearmax)
    yearlabels = [("%is" % y) for iy,y in enumerate(yearbins) if iy != (len(yearbins)-1)]

    #df['Year'] = pd.cut(x = df['Year'], bins = yearbins, labels = yearlabels, include_lowest = True)

    d = dict(enumerate(yearlabels, 1))
    df['Year'] = np.vectorize(d.get)(np.digitize(df['Year'], yearbins))

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

    #df = df[df["Year"] >= 1982]

    return df

#aggiunge city e state al dataframe
def split_city_state(df):

    df["City"] = df["Location"].str.split(",").str[0]
    df["State"] = df["Location"].str.split(",").str[1]

    return df

def cleaning_nan(df): #same as transform_data_into_value without conversion (to save csv)

    data = df.copy()

    #binary classification
    data = data[data["Injury.Severity"] != "Unavailable"]
    data = data[data["Injury.Severity"] != "Serious"]
    data = data[data["Injury.Severity"] != "Minor"]
    data = data[data["Injury.Severity"] != "Incident"]

    # replace all infinite values with nan
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # remove all nan values
    data.dropna(inplace=True)

    return data


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
    data['Year'] = data['Year'].astype('category').cat.codes

    data["Injury.Severity"] = data["Injury.Severity"].astype('category').cat.codes
    #data["Injury.Severity"].replace([-1], np.nan, inplace=True)
    data.replace([-1], np.nan, inplace=True)

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
    #escludiamo anche i total injuries perché basterebbe fatal injuries per il 100%, forniamo al classificatore un informazione ridonndante con l'output
    excludevars = ["Month","Country","Investigation.Type","Total.Fatal.Injuries","Total.Serious.Injuries","Total.Minor.Injuries","Total.Uninjured"]

    newvars = [var for var in variables if var not in excludevars]
    finaldf = df[newvars]

    return finaldf

def preprocessing_data(X_train,X_test):

    #scaler with mean and dev standard
    scaler = preprocessing.StandardScaler().fit(X_train)
    mu_train = scaler.mean_
    std_train = scaler.scale_

    print("\n- Preprocessing:")
    print("- Media: ",mu_train)
    print("- Dev Std: ",std_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train,X_test

def data_augmentation(X,y,method= SMOTETomek(random_state=42)):
     #= SMOTETomek(random_state=42)
    X, y = method.fit_resample(X, y)
    return X,y

def data_reduction(nfeatures,X_train,y_train,X_test):

    pca = PCA(n_components = nfeatures, random_state=42)
    pca.fit(X_train,y_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    return X_train,X_test

def prepare_train_test(df):

    #df = transform_data_into_value(df)
    columns = df.columns.values.tolist()
    features = [col for col in columns if col != "Injury.Severity"]

    X = df[features]
    #X = df[["Total.Fatal.Injuries"]] #99% with only this
    y = df["Injury.Severity"]

    X = X.values.tolist()
    y = y.values.tolist()
    
    return X,y

def split_train_test(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle = True, random_state = 42)
    return X_train, X_test, y_train, y_test

def cross_validation(clf,param_grid,ncv,X_train,y_train,X_test,y_test,name_output,dir_output):

    print("\n- Training...")
    search = GridSearchCV(clf, param_grid = param_grid, n_jobs=-1, cv = ncv, scoring="f1",verbose=10)
    search.fit(X_train, y_train)

    print("\n- Best Parameters: ")
    print(search.best_params_)

    # saving model
    if (not os.path.exists(dir_output)): os.makedirs(dir_output)

    #writing to file
    resultfile = open(dir_output + "result.txt", "w")
    resultfile.write("Best Parameters: \n")
    resultfile.write(str(search.best_params_))

    resultfile.write("\n\nBest Validation Score: \n")
    resultfile.write(str(search.best_score_))

    print('\n- Performance:')
    print("- Train Set:")
    #writing to file
    resultfile.write("\n\nScore Train Set:")
    #resultfile.close()
    evaluate(search.predict(X_train), y_train,resultfile)

    print("- Test Set:")
    resultfile.write("\n\nScore Test Set:")
    evaluate(search.predict(X_test), y_test,resultfile)
    resultfile.close()

    #saving model
    joblib.dump(clf,dir_output+name_output)

    #plotting learning curve
    train_sizes = [0.01,0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    plot_learningcurve(clf,X_train,y_train,train_sizes,ncv,dir_output) 

def evaluate(y_pred, y_gold, resultfile):

    acc = accuracy_score(y_gold, y_pred)
    prec = precision_score(y_gold, y_pred)
    rec = recall_score(y_gold,y_pred)
    f1 = f1_score(y_gold,y_pred)

    #resultfile = open(file_out, 'a')
    resultfile.write("\nAccuracy score: {:3f}".format(acc))
    resultfile.write("\nPrecision score: {:3f}".format(prec))
    resultfile.write("\nRecall score: {:3f}".format(rec))
    resultfile.write("\nF1 score: {:3f}".format(f1))

    print("Accuracy score: {:3f}".format(acc))
    print("Precision score: {:3f}".format(prec))
    print("Recall score: {:3f}".format(rec))
    print("F1 score: {:3f}".format(f1))
        
def plot_learningcurve(clf,X_train,y_train,train_sizes,n_folds,directory):

    print('\n- Plotting Learning Curve:')

    train_sizes, train_scores, validation_scores = learning_curve(estimator=clf, X=X_train, y=y_train, train_sizes = train_sizes,cv=n_folds, scoring="f1", shuffle=True, random_state=42)
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
    merged_year_df = merge_year(merged_nengines_df)

    fixed_df = fix_values(merged_year_df)
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

    #feature selection
    df_categorical = transform_data_into_value(city_state_df)
    clean_df = feature_selection(df_categorical,cleanvars)
    X, y = prepare_train_test(clean_df)

    # saving csv
    df_training = cleaning_nan(city_state_df)
    columns = clean_df.columns.values.tolist()
    features = [col for col in columns if col != "Injury.Severity"]
    df_training[columns].to_csv("clean_dataset.csv")

    if (plotopt):

        print("\n- Plotting...")

        #plot with injuries and fatal-nonfatal
        plot_injuries(df_training)

        #plotting also previously excluded features
        expanded_features = [col for col in cleanvars if (col != "Injury.Severity")]

        #plot correlations with output
        for f in expanded_features:
            plot_output_feature(df_training, f, "Injury.Severity")
            plot_output_feature(df_training,"Injury.Severity",f)

        #autocorrelation
        for f1 in expanded_features:
            for f2 in expanded_features:
                if (f1 == f2): continue
                plot_output_feature(df_training, f1, f2)

    #indici variabili categoriche e non
    #i_cat = [ifeat for ifeat,f in enumerate(features) if (f != "Number.of.Engines" and f != "Year")]
    #i_cat = [ifeat for ifeat, f in enumerate(features) if (f != "Number.of.Engines")]
    i_cat = [ifeat for ifeat,f in enumerate(features)]
    i_num = [ifeat for ifeat,f in enumerate(features) if (ifeat not in i_cat)]

    #splitting
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    print("\n- Split Train/Test")
    print("Train: %i" % len(X_train))
    print("Test: %i" % len(X_test))

    # data augmentation
    method = RandomOverSampler(random_state=42)
    #method = RandomUnderSampler(random_state=42)
    #method = SMOTE(random_state=42)
    #method = SMOTETomek(random_state=42)

    print("\n- Data Augmentation with %s" % method)
    print("Pre-Augmentation:")
    print("Train:")
    print("Class 0: %i" % y_train.count(0))
    print("Class 1: %i" % y_train.count(1))
    print("Test:")
    print("Class 0: %i" % y_test.count(0))
    print("Class 1: %i" % y_test.count(1))
    X_train, y_train = data_augmentation(X_train, y_train, method)
    print("Post-Augmentation:")
    print("Train:")
    print("Class 0: %i" % y_train.count(0))
    print("Class 1: %i" % y_train.count(1))
    print("Test:")
    print("Class 0: %i" % y_test.count(0))
    print("Class 1: %i" % y_test.count(1))

    #dividing in categorical and numerical data for naive bayes
    X_train_cat, X_test_cat = [],[]
    X_train_num, X_test_num = [],[]
    for x in X_train:
        X_train_cat.append(np.array(x)[i_cat])
        X_train_num.append(np.array(x)[i_num])
    for x in X_test:
        X_test_cat.append(np.array(x)[i_cat])
        X_test_num.append(np.array(x)[i_num])

    #preprocessing
    X_train, X_test = preprocessing_data(X_train, X_test)

    baseline = max(y_test.count(0), y_test.count(1))/len(y_test)
    w_0 = 1/y_test.count(0)
    w_1 = 1/y_test.count(1)
    print("\n- Baseline Test: %.2f" % baseline)
    
    # dimensionality reduction via pca
    n_features = X_train.shape[1]
    n_components = 8
    n_components = min(n_components,n_features)
    #X_train, X_test = data_reduction(n_components,X_train,y_train,X_test)

    n_features = X_train.shape[1]
    print("\n- Training con %i Features: %s" % (n_features, features))

    #choosing classifier
    clf = RandomForestClassifier()
    nameclf = "rndmforest.joblib"
    dirclf = "rndmforest/"

    #choosing param grid
    param_grid = {
        'min_samples_split': [2, 5, 7],
        'max_depth': [10, 50, 100, None],
        'max_features': ['sqrt', 'log2', n_features],
        'min_samples_leaf': [2, 3, 4],
        'n_estimators': [100, 500, 1000, 1500],
        'ccp_alpha': [0., 1e-5, 5e-5],  # 0 no pruning
        "random_state": [42]
    }

    #cross_validation(clf, param_grid, 5, X_train, y_train, X_test, y_test, nameclf, dirclf)

    clf = SVC()
    nameclf = "svm_rbf.joblib"
    dirclf = "svm_rbf/"

    param_grid = {
        "kernel": ["rbf"],
        "C": [1.0, 10.0, 50.0],
        "gamma": ["scale", "auto"],
        "random_state": [42]
    }

    cross_validation(clf, param_grid, 5, X_train, y_train, X_test, y_test, nameclf, dirclf)

    clf = SVC()
    nameclf = "svm_lin.joblib"
    dirclf = "svm_lin/"

    param_grid = {
        "kernel": ["linear"],
        "C": [1.0, 30.0, 50.0],
        "random_state": [42]
    }

    #cross_validation(clf, param_grid, 5, X_train, y_train, X_test, y_test, nameclf, dirclf)

    clf = SVC()
    nameclf = "svm_poly.joblib"
    dirclf = "svm_poly/"

    param_grid = {
        "kernel": ["poly"],
        "C": [1.0, 10.0, 50.0],
        "degree": [2, 3],
        "gamma": ["scale", "auto"],
        "random_state": [42]
    }

    cross_validation(clf, param_grid, 5, X_train, y_train, X_test, y_test, nameclf, dirclf)

    clf = SVC()
    nameclf = "svm_sigmoid.joblib"
    dirclf = "svm_sigmoid/"

    param_grid = {
        "kernel": ["sigmoid"],
        "C": [1.0, 10.0, 50.0],
        "gamma": ["scale", "auto"],
        "random_state": [42]
    }

    cross_validation(clf, param_grid, 5, X_train, y_train, X_test, y_test, nameclf, dirclf)

    #naive bayes
    #si basa su ipotesi distribuzione feature
    # gaussian -> serve una distribuzione con media e varianza
    #          -> Quindi o normalizziamo con scale
    #          -> Se non normalizziamo, utilizzamo solo feature non categoriche

    clf = GaussianNB()
    nameclf = "gnb.joblib"
    dirclf = "gaussiannb/"

    param_grid = {
        "var_smoothing": [1e-10,1e-9,1e-8]
    }

    #cross_validation(clf, param_grid, 5, X_train, y_train, X_test, y_test, nameclf, dirclf)
    #X_train_num, X_test_num = preprocessing_data(X_train_num, X_test_num)
    #cross_validation(clf, param_grid, 5, X_train_num, y_train, X_test_num, y_test, nameclf, dirclf)

    # multinomial si aspetta frequenze (per esempio in un testo vuole la frequenza di una parola)
    # bernoulli si aspetta binari (per esempio in un testo se una parola c'è o no)

    # categorical naive bayes -> Si basa su feature categoriche
    # La utilizziamo con feature solo categoriche senza normalizzazione

    clf = CategoricalNB()
    nameclf = "cnb.joblib"
    dirclf = "categoricalnb/"

    param_grid = {
        "alpha": [0.1, 0.5, 1, 1.5],
        "fit_prior": [True, False]
    }

    #cross_validation(clf,param_grid,5,X_train_cat,y_train,X_test_cat,y_test,nameclf,dirclf)

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

