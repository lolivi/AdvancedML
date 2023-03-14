import os.path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

def plot_accidents_per_year(df):
    accidents_per_year = df.groupby(['Year'], as_index=False)['Event.Id'].count()
    sns.lineplot(x='Year', y='Event.Id', data=accidents_per_year, color='#2990EA')
    plt.show()


def plot_accidents_based_on_weather(df):
    accidents_according_to_weather = df.groupby("Weather.Condition")["Event.Id"].count()
    accidents_according_to_weather.plot.bar(stacked=True, color=['#003366', '#2990EA'])

    plt.xticks(rotation=0)
    plt.xlabel('')
    plt.show()


def plot_accidents_based_on_injuriy(df):
    weather_accident = df.groupby("Weather.Condition")["Injury.Severity"] \
        .value_counts(normalize=True) \
        .unstack("Injury.Severity")

    colors = list(sns.color_palette("Set2"))[:3]
    weather_accident.plot.bar(stacked=True, color=colors)

    plt.xticks(rotation=0)
    plt.title('Fatality rate during IMC or VMC')
    plt.xlabel('')
    plt.legend(title="Injury Severity", loc='upper right')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.show()


def plot_correlation_matrix(df):
    
    f = plt.figure(figsize = (10,10))
    #sns.heatmap(df.corr(), vmin=-1, vmax=1, cmap="mako")
    sns.heatmap(df.corr(), vmin=-1, vmax=1, annot = True)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("plots/corrmatrix.png", bbox_inches='tight', dpi = 200)


def plot_phase_of_flight(df):
    phase_of_flight = df.groupby("Broad.phase.of.flight")["Injury.Severity"] \
        .value_counts(normalize=True).unstack("Injury.Severity")

    phase_of_flight.plot.bar(stacked=True)

    plt.xlabel('')

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
    #    accidents_according_to_weather = df.groupby("Weather.Condition")["Event.Id"].count()
    #    accidents_according_to_weather.plot.bar(stacked = True, color = ['#003366','#2990EA'])

    injuries = ["Total.Fatal.Injuries", "Total.Minor.Injuries", "Total.Serious.Injuries", "Total.Uninjured"]
    datainj = df.groupby("Injury.Severity")[injuries].sum()
    norm = datainj.sum(axis="columns")
    for inj in injuries:
        datainj[inj] = datainj[inj] / norm

    datainj.plot.bar(stacked=True)
    plt.xlabel('Injury Severity')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.legend(bbox_to_anchor=(1.05,1.02), loc="upper left")
    #plt.tight_layout()
    plt.savefig("plots/TotalInjuries.png",dpi = 200, bbox_inches='tight')


def plot_amateur_engines(df):
    dataeng = df.groupby("Amateur.Built")["Number.of.Engines"].value_counts(normalize=True).unstack()
    dataeng.plot.bar(stacked=True)
    plt.xlabel('Amateur Built?')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def plot_flight_purpose(df):
    datapurp = df["Purpose.of.flight"].value_counts(normalize=True)
    datapurp.plot.bar(stacked=True)
    plt.xlabel('Purpose of flight')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def plot_engine_type(df):
    datatype = df["Engine.Type"].value_counts(normalize=True)
    datatype.plot.bar(stacked=True)
    plt.xlabel('Engine Type')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_output_feature(df,out,feat):

    plt.figure(figsize=(9.6,5.4), clear=True)

    dataplot = df.groupby(feat)[out].value_counts(normalize=True).unstack()
    dataplot.plot.bar(stacked=True)
    plt.xlabel(feat)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.ylabel(out)
    plt.xticks(rotation=90)
    plt.legend(bbox_to_anchor=(1.05,1.02), loc="upper left")
    #plt.tight_layout()
    plt.savefig("plots/%s_%s.png" % (out,feat), dpi = 200, bbox_inches='tight')
    plt.close("all")

def plot_frequency(df,feat):

    plt.figure(figsize=(9.6, 5.4), clear=True)

    dataplot = df[feat].value_counts()
    dataplot.plot.bar()
    plt.xlabel(feat)
    plt.ylabel("Events")
    plt.xticks(rotation=90)
    plt.savefig("plots/1d_freq_%s.png" % (feat), bbox_inches='tight')
    plt.close("all")