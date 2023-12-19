import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

sys.stdout = open('../output/clean/clean.txt', 'w')
df1 = pd.read_csv('../db/heart_data.csv')
print("Shape of Dataframe : ", df1.shape)

df1 = df1[df1.columns].replace({'Yes':1, 'No':0, 'Male':1,'Female':0,'No, borderline diabetes':'0','Yes (during pregnancy)':'1' })
df1['Diabetic'] = df1['Diabetic'].astype(int)
print("\nDataframe is as follows\n", df1.head())
original_length = len(df1)
temp = "\nColumns with values True/False need not to be cleaned. We will focus on remaining columns to check if any cleaning is required. Following columns are True/False:\n[HeartDisease, Smoking, AlcoholDrinking, Stroke, DiffWalking, Sex, Diabetic, PhysicalActivity, Asthma, KidneyDisease, SkinCancer]\n"
print(temp)

def bmi_clean():
    ######BMI
    global df1
    print("\nBMI Cleaning")
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 4))
    sns.boxplot(df1["BMI"], palette="Set2").set(title="BMI before Cleaning")
    plt.savefig('../output/clean/bmi_boxplot.png')
    plt.close()
    print("\tMean of BMI: ", df1["BMI"].mean())
    print("\tRange of BMI: [", df1["BMI"].min(), ",", df1["BMI"].max(), "]")
    print("\tWe will remove BMI values which are more than 80, as in real life they are above normal body range")
    # Cleaning BMI
    Q1 = df1["BMI"].quantile(0.10)

    lower_bound = Q1
    upper_bound = 80
    # Remove outliers
    df1 = df1[(df1["BMI"] >= lower_bound) & (df1["BMI"] <= upper_bound)]
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 4))
    sns.boxplot(x=df1["BMI"], palette="Set2").set(title="BMI after Cleaning")
    plt.savefig('../output/clean/bmi_cleaned_boxplot.png')
    plt.close()
    print("\tSome data that was above reasonable human capacity i.e. more than 80 was removed.")

def phHealth_clean():
    ##########Physical Health
    global df1
    print("\nPhysicalHealth Cleaning")
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 4))
    sns.boxplot(x=df1["PhysicalHealth"], palette="Set2").set(title="Physical Health")
    plt.savefig('../output/clean/physical_health_boxplot.png')
    plt.close()
    print("\tMean of PhysicalHealth: ", df1["PhysicalHealth"].mean())
    print("\tRange of PhysicalHealth: [", df1["PhysicalHealth"].min(), ",", df1["PhysicalHealth"].max(), "]")
    print("\tData is reasonably distributed, no need to clean out.")

def mentalHealth_clean():
    ######Mental Health
    global df1
    print("\ntMentalHealth Cleaning")
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 4))
    sns.boxplot(x=df1["MentalHealth"], palette="Set2").set(title="Mental Health")
    plt.savefig('../output/clean/mental_health_boxplot.png')
    plt.close()
    print("\tMean of MentalHealth: ", df1["MentalHealth"].mean())
    print("\tRange of MentalHealth: [", df1["MentalHealth"].min(), ",", df1["MentalHealth"].max(), "]")
    print("\tData is reasonably distributed, no need to clean out.")

def age_clean():
    ########AgeCategory
    global df1
    print("\nAgeCategory Cleaning")
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 4))
    sns.histplot(x=df1["AgeCategory"]).set(title="Age")
    plt.savefig('../output/clean/age_histplot.png')
    plt.close()
    print("\tData is reasonably distributed, no need to clean out.")

def race_clean():
    ########Race
    global df1
    print("\nRace Cleaning")
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 4))
    sns.histplot(x=df1["Race"]).set(title="Race")
    plt.savefig('../output/clean/race_histplot.png')
    plt.close()
    print("\tTotal people: ",  len(df1))
    print(df1["Race"].value_counts())
    print("\tData is reasonably distributed, no need to clean out.")

def genHealth_clean():
    #########GenHealth
    global df1
    print("\nGeneral Health Cleaning")
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 4))
    sns.histplot(x=df1["GenHealth"]).set(title="General Health")
    plt.savefig('../output/clean/gen_health_histplot.png')
    plt.close()
    print("\tTotal people: ",  len(df1))
    print("\t", df1["GenHealth"].value_counts())
    print("\tData is reasonably distributed, no need to clean out.")

def sleep_clean():
    ###########SleepTime
    global df1
    print("\nSleep Time Cleaning")
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 4))
    sns.histplot(df1["SleepTime"], binwidth=1).set(title="Sleep Time before cleaning")
    plt.savefig('../output/clean/sleep_histplot.png')
    plt.close()
    print("\tTotal people: ",  len(df1))
    print("\tNumber of sleeptime values: ", df1["SleepTime"].value_counts())
    print("\tMean of SleepTime: ", df1["SleepTime"].mean())
    print("\tRange of SleepTime: [", df1["SleepTime"].min(), ",", df1["SleepTime"].max(), "]")
    print("\tData could be cleaned by removing people with sleep time of more than 12 hrs.")
    print("\tAs we can see below, people having more than 12 hours of sleep are very less")
    print("\tIn real life, that also lies above normal sleeping time for normal human.")
    
    #Clean SleepTime
    df1 = df1[(df1["SleepTime"] <= 12.00) & (df1["SleepTime"] >= 3.00)]
    print("\tSleep Time Cleaning")
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 4))
    sns.histplot(df1["SleepTime"], binwidth=1).set(title="Sleep Time after Cleaning")
    plt.savefig('../output/clean/sleep_clean_histplot.png')
    plt.close()
    print("\tTotal people:",  len(df1))
    print("\t", df1["SleepTime"].value_counts())
    print("\tMean of SleepTime: ", df1["SleepTime"].mean())
    print("\tRange of SleepTime: [", df1["SleepTime"].min(), ",", df1["SleepTime"].max(), "]")

def end():
    global df1
    final_length = len(df1)
    print("\n\nCleaning Summary")
    print("\tOriginal Entries: ",original_length)
    print("\tFinal Entries: ",final_length)
    print("\tNumber of entries removed: ", original_length - final_length)
    df1.to_csv('../db/heart_data_cleaned.csv', index=False)
    print("\tA new file named \"heart_data_cleaned\".csv has been generated in folder db")
    
def main():
    bmi_clean()
    phHealth_clean()
    mentalHealth_clean()
    age_clean()
    race_clean()
    genHealth_clean()
    sleep_clean()
    end()

if __name__ == "__main__":
    main()

sys.stdout.close()