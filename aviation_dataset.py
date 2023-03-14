#python libraries
import os, statistics, joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#plotter.py
from plotter import * #plotter functions
from df_cleaner import * #functions to merge and clean dataframe
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

#variabili qualitativamente scelte (verranno verificate con correlazione e
cleanvars = ["Injury.Severity","Investigation.Type","Country",
             "Aircraft.damage","Amateur.Built","Number.of.Engines",
             "Engine.Type","Purpose.of.flight","Weather.Condition",
             "Broad.phase.of.flight","Year","Month"]
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
    #escludiamo anche i total injuries perché basterebbe fatal injuries per il 100%, forniamo al classificatore un informazione ridondante con l'output
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

def data_augmentation(X,y,method = SMOTETomek(random_state=42)):

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

def cross_validation(clf,param_grid,ncv,X_train,y_train,X_test,y_test,name_output,dir_output, n_components = 8):

    if (n_components != 8):
        dir_output = dir_output.replace("/","_%i/" % (n_components))
        X_train, X_test = data_reduction(n_components,X_train,y_train,X_test) #pca components

    print("\n- Training con %i Componenti" % X_train.shape[1])

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

#explained variance and most important features
def component_analysis(X_train,feature_names):

    print("\n- Principal Component Analysis")

    n_features = X_train.shape[1]
    pca = PCA()
    model = pca.fit(X_train)
    X_pca = model.transform(X_train)

    print("Explained Variance Ratio: ",pca.explained_variance_ratio_)

    var_arr = np.zeros((n_features,n_features))
    for i in range(n_features): var_arr[i][i] = pca.explained_variance_ratio_[i]
    var_arr[var_arr == 0] = None

    f = plt.figure()
    sns.heatmap(pd.DataFrame(abs(pca.components_),columns = feature_names), vmin=0, vmax=1, annot=True)
    plt.title("Principal Component Analysis - Components")
    plt.ylabel("Principal Component")
    plt.xlabel("Feature Name")
    plt.savefig("plots/pca_components.png", bbox_inches='tight')

    n_pcs = model.components_.shape[0]
    most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]

    # get the names
    most_important_names = [feature_names[most_important[i]] for i in range(n_pcs)]

    dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}
    print("Most Important Features...",dic)

    f = plt.figure()
    sns.heatmap(pd.DataFrame(var_arr.tolist(), columns=most_important_names), vmin=0, vmax=1, annot=True)
    plt.title("Principal Component Analysis - Explained Variance")
    plt.ylabel("Principal Component")
    plt.xlabel("Most Important Feature")
    plt.savefig("plots/pca_variance.png", bbox_inches='tight')

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
    #method = None #no augmentation
    #method = RandomOverSampler(random_state=42)
    #method = RandomUnderSampler(random_state=42)
    #method = SMOTE(random_state=42)
    method = SMOTETomek(random_state=42)
    if (not method): weights = True
    else: weights = False

    print("\n- Data Augmentation with %s" % method)
    print("Pre-Augmentation:")
    print("Train:")
    print("Class 0: %i" % y_train.count(0))
    print("Class 1: %i" % y_train.count(1))
    print("Test:")
    print("Class 0: %i" % y_test.count(0))
    print("Class 1: %i" % y_test.count(1))
    if (method): X_train, y_train = data_augmentation(X_train, y_train, method)
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
    print("\n- Baseline Test: %.2f" % baseline)

    # dimensionality reduction via pca
    n_components = [2,4,6] #8 for no pca
    #n_components = [8]
    component_analysis(X_train,features)
    n_features = X_train.shape[1]

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

    if (weights):

        param_grid = {
            'min_samples_split': [2, 5, 7],
            'max_depth': [10, 50, 100, None],
            'max_features': ['sqrt', 'log2', n_features],
            'min_samples_leaf': [2, 3, 4],
            'n_estimators': [100, 500, 1000, 1500],
            'ccp_alpha': [0., 1e-5, 5e-5],  # 0 no pruning
            "random_state": [42],
            "class_weight": ["balanced"]
        }


    for comp in n_components: cross_validation(clf, param_grid, 5, X_train, y_train, X_test, y_test, nameclf, dirclf, comp)

    clf = SVC()
    nameclf = "svm_rbf.joblib"
    dirclf = "svm_rbf/"

    param_grid = {
        "kernel": ["rbf"],
        "C": [1.0, 10.0, 50.],
        "gamma": ["scale", "auto"],
        "random_state": [42]
    }

    if (weights):

        param_grid = {
            "kernel": ["rbf"],
            "C": [1.0, 10.0, 50.],
            "gamma": ["scale", "auto"],
            "random_state": [42],
            "class_weight": ['balanced']
        }

    #for comp in n_components: cross_validation(clf, param_grid, 5, X_train, y_train, X_test, y_test, nameclf, dirclf, comp)

    clf = SVC()
    nameclf = "svm_lin.joblib"
    dirclf = "svm_lin/"

    param_grid = {
        "kernel": ["linear"],
        "C": [1.],
        "random_state": [42]
    }

    if (weights):

        param_grid = {
            "kernel": ["linear"],
            "C": [1.],
            "random_state": [42],
            "class_weight": ['balanced']
        }


    for comp in n_components: cross_validation(clf, param_grid, 5, X_train, y_train, X_test, y_test, nameclf, dirclf, comp)

    clf = SVC()
    nameclf = "svm_poly.joblib"
    dirclf = "svm_poly/"

    param_grid = {
        "kernel": ["poly"],
        "C": [1.0, 10.0, 50.],
        "degree": [2, 3],
        "gamma": ["scale", "auto"],
        "random_state": [42]
    }

    if (weights):

        param_grid = {
            "kernel": ["poly"],
            "C": [1.0, 10.0, 50.],
            "degree": [2, 3],
            "gamma": ["scale", "auto"],
            "random_state": [42],
            "class_weight": ["balanced"]
        }

    #for comp in n_components: cross_validation(clf, param_grid, 5, X_train, y_train, X_test, y_test, nameclf, dirclf, comp)

    clf = SVC()
    nameclf = "svm_sigmoid.joblib"
    dirclf = "svm_sigmoid/"

    param_grid = {
        "kernel": ["sigmoid"],
        "C": [1.0, 10.0, 50.0],
        "gamma": ["scale", "auto"],
        "random_state": [42]
    }

    if (weights):

        param_grid = {
            "kernel": ["sigmoid"],
            "C": [1.0, 10.0, 50.0],
            "gamma": ["scale", "auto"],
            "random_state": [42],
            "class_weight": ["balanced"]
        }

    #for comp in n_components: cross_validation(clf, param_grid, 5, X_train, y_train, X_test, y_test, nameclf, dirclf, comp)

    #naive bayes
    #si basa su ipotesi distribuzione feature
    #gaussian -> serve una distribuzione con media e varianza

    clf = GaussianNB()
    nameclf = "gnb.joblib"
    dirclf = "gaussiannb/"

    param_grid = {
        "var_smoothing": [1e-10,1e-9,1e-8]
    }

    if (weights):

        w_0 = y_train.count(0) / len(y_train)
        w_1 = y_train.count(1) / len(y_train)
        print("Prior Probabilities: %.2f Class 0, %.2f Class 1" % (w_0,w_1))

        param_grid = {
            "var_smoothing": [1e-10, 1e-9, 1e-8],
            "priors": [[w_0,w_1]]
        }


    for comp in n_components: cross_validation(clf, param_grid, 5, X_train, y_train, X_test, y_test, nameclf, dirclf, comp)
    
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

    if (weights):

        w_0 = y_train.count(0) / len(y_train)
        w_1 = y_train.count(1) / len(y_train)
        print("Prior Probabilities: %.2f Class 0, %.2f Class 1" % (w_0, w_1))

        param_grid = {
            "alpha": [0.1, 0.5, 1, 1.5],
            "fit_prior": [True,False],
            "class_prior": [[w_0,w_1]]
        }

    #no pca because it would need scaled variables
    X_train_cat, y_train, X_test_cat, y_test = np.array(X_train_cat), np.array(y_train), np.array(X_test_cat), np.array(y_test)
    cross_validation(clf, param_grid, 5, X_train_cat, y_train, X_test_cat, y_test, nameclf, dirclf)

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

