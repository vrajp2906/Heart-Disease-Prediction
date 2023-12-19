import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.stdout = open('../output/distribution/distribution.txt', 'w')
df1 = pd.read_csv('../db/heart_data_cleaned.csv')
df1.head()

def bmi_dist():
    #BMI
    fig, ax = plt.subplots(figsize = (13,5))
    sns.kdeplot(df1[df1["HeartDisease"]==1]["BMI"], alpha=0.5,fill = True, color="red", label="HeartDisease", ax = ax)
    sns.kdeplot(df1[df1["HeartDisease"]==0]["BMI"], alpha=0.5,fill = True, color="green", label="Normal", ax = ax)
    plt.title('Distribution of Body Mass Index', fontsize = 18)
    
    ax.set_xlabel("BodyMass")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.savefig('../output/distribution/bmi_plot.png')
    plt.close()
    # plt.show()

def smoke_dist():
    #smoking
    grouped_data = df1.groupby(['Smoking', 'HeartDisease']).size().unstack()
    grouped_data.plot(kind='bar', color=['green', 'red'], alpha=0.7)

    plt.xlabel('Smoking Status')
    plt.ylabel('Count')
    plt.title('Distribution of Heart Disease based on Smoking Status')
    plt.legend(title='Heart Disease', loc='upper right', labels=['No', 'Yes'])
    plt.xticks(rotation=0)
    plt.savefig('../output/distribution/smoke_plot.png')
    plt.close()
    # plt.show()

def alcohol_dist():
    #alcohol
    grouped_data = df1.groupby(['AlcoholDrinking', 'HeartDisease']).size().unstack()
    grouped_data.plot(kind='bar', color=['green', 'red'], alpha=0.7)

    plt.xlabel('Alcohol Drinking Status')
    plt.ylabel('Count')
    plt.title('Distribution of Heart Disease based on Alcohol Drinking Status')
    plt.legend(title='Heart Disease', loc='upper right', labels=['No', 'Yes'])
    plt.xticks(rotation=0)
    plt.savefig('../output/distribution/alcohol_plot.png')
    plt.close()

    # plt.show()

def stroke_dist():
    #stroke
    df1["Stroke"]

    grouped_data = df1.groupby(['Stroke', 'HeartDisease']).size().unstack()
    grouped_data.plot(kind='bar', color=['green', 'red'], alpha=0.7)

    plt.xlabel('Stroke Status')
    plt.ylabel('Count')
    plt.title('Distribution of Heart Disease based on Stroke Status')
    plt.legend(title='Heart Disease', loc='upper right', labels=['No', 'Yes'])
    plt.xticks(rotation=0)
    plt.savefig('../output/distribution/stroke_plot.png')
    plt.close()
    # plt.show()

def physical_dist():
    #physical health
    fig, ax = plt.subplots(figsize = (13,5))
    sns.kdeplot(df1[df1["HeartDisease"]==1]["PhysicalHealth"], alpha=0.5,fill = True, color="red", label="HeartDisease", ax = ax)
    sns.kdeplot(df1[df1["HeartDisease"]==0]["PhysicalHealth"], alpha=0.5,fill = True, color="green", label="Normal", ax = ax)
    plt.title('Distribution of Physical Health', fontsize = 18)
    ax.set_xlabel("PhysicalHealth")
    ax.set_ylabel("Frequency")
    ax.legend(title='Heart Disease', loc='upper right', labels=['Yes', 'No'])
    plt.savefig('../output/distribution/physical_health_plot.png')
    plt.close()
    # plt.show()

def mental_dist():
    #mental health
    fig, ax = plt.subplots(figsize = (13,5))
    sns.kdeplot(df1[df1["HeartDisease"]==1]["MentalHealth"], alpha=0.5,fill = True, color="red", label="HeartDisease", ax = ax)
    sns.kdeplot(df1[df1["HeartDisease"]==0]["MentalHealth"], alpha=0.5,fill = True, color="green", label="Normal", ax = ax)
    plt.title('Distribution of MentalHealth', fontsize = 18)
    ax.set_xlabel("MentalHealth")
    ax.set_ylabel("Frequency")
    plt.legend(title='Heart Disease', loc='upper right', labels=['Yes', 'No'])
    plt.savefig('../output/distribution/mental_health_plot.png')
    plt.close()
    # plt.show()

def diff_dist():
    #difficulty walking
    grouped_data = df1.groupby(['DiffWalking', 'HeartDisease']).size().unstack()
    grouped_data.plot(kind='bar', color=['green', 'red'], alpha=0.7)

    plt.xlabel('DiffWalking Status')
    plt.ylabel('Count')
    plt.title('Distribution of Heart Disease based on Difficulty in Walking')
    plt.legend(title='Heart Disease', loc='upper right', labels=['No', 'Yes'])
    plt.xticks(rotation=0)
    plt.savefig('../output/distribution/difficulty_walking_plot.png')
    plt.close()

    # plt.show()

def sex_dist():
    #sex
    grouped_data = df1.groupby(['Sex', 'HeartDisease']).size().unstack()
    grouped_data.plot(kind='bar', color=['green', 'red'], alpha=0.7)
    sns.set(style="whitegrid", palette="deep")
    plt.xlabel('Sex')
    plt.ylabel('Count')
    plt.title('Distribution of Heart Disease among Males and Females')
    plt.legend(title='Heart Disease', loc='upper right', labels=['No', 'Yes'])
    plt.xticks(rotation=0)
    plt.savefig('../output/distribution/gender_plot.png')
    plt.close()

    # plt.show()

def age_dist():
    #age
    plt.figure(figsize = (13,6))
    sns.countplot(x = df1['AgeCategory'], hue = 'HeartDisease', data = df1, palette = 'Set2')
    plt.xlabel('AgeCategory')
    plt.legend(title='Heart Disease', loc='upper right', labels=['No', 'Yes'])
    plt.ylabel('Frequency')
    plt.savefig('../output/distribution/age_plot.png')
    plt.close()
    # plt.show()

def race_dist():
    #race
    plt.figure(figsize = (13,6))
    sns.countplot( x= df1['Race'], hue = 'HeartDisease', data = df1, palette = 'Set2')
    plt.xlabel('Race')
    plt.legend(title='Heart Disease', loc='upper right', labels=['No', 'Yes'])
    plt.ylabel('Frequency')
    plt.savefig('../output/distribution/race_plot.png')
    plt.close()
    # plt.show()

def diabetes_dist():
    #diabetes
    grouped_data = df1.groupby(['Diabetic', 'HeartDisease']).size().unstack()
    grouped_data.plot(kind='bar', color=['green', 'red'], alpha=0.7)

    plt.xlabel('Diabetic Status')
    plt.ylabel('Count')
    plt.title('Distribution of Heart Disease based on Diabetic')
    plt.legend(title='Heart Disease', loc='upper right', labels=['No', 'Yes'])
    plt.xticks(rotation=0)
    plt.savefig('../output/distribution/diabetes_plot.png')
    plt.close()

    # plt.show()

def phyicalhealth_dist():
    #physicalhealth
    fig, ax = plt.subplots(figsize = (13,5))
    sns.kdeplot(df1[df1["HeartDisease"]==1]["PhysicalHealth"], alpha=0.5,fill = True, color="red", label="HeartDisease", ax = ax)
    sns.kdeplot(df1[df1["HeartDisease"]==0]["PhysicalHealth"], alpha=0.5,fill = True, color="green", label="Normal", ax = ax)
    plt.title('Distribution of PhysicalHealth state for the last 30 days', fontsize = 18)
    ax.set_xlabel("PhysicalHealth")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.savefig('../output/distribution/physical_health_plot.png')
    plt.close()
    # plt.show()

def genhealth_dist():
    #genhealth
    plt.figure(figsize = (13,6))
    sns.countplot( x= df1['GenHealth'], hue = 'HeartDisease', data = df1, palette = 'Set2')
    plt.xlabel('Race')
    plt.legend(title='Heart Disease', loc='upper right', labels=['No', 'Yes'])
    plt.ylabel('Frequency')
    plt.savefig('../output/distribution/general_health_plot.png')
    plt.close()
    # plt.show()

def sleep_dist():
    #sleeptime
    fig, ax = plt.subplots(figsize = (13,5))
    sns.kdeplot(df1[df1["HeartDisease"]==1]["SleepTime"], alpha=0.5,fill = True, color="red", label="HeartDisease", ax = ax)
    sns.kdeplot(df1[df1["HeartDisease"]==0]["SleepTime"], alpha=0.5,fill = True, color="green", label="Normal", ax = ax)
    plt.title('Distribution of SleepTime values', fontsize = 18)
    ax.set_xlabel("SleepTime")
    ax.set_ylabel("Frequency")
    plt.legend(title='Heart Disease', loc='upper right', labels=['Yes', 'No'])
    plt.savefig('../output/distribution/sleep_plot.png')
    plt.close()
    # plt.show()

def smoke_dist():
    #smoking
    grouped_data = df1.groupby(['Smoking', 'HeartDisease']).size().unstack()
    grouped_data.plot(kind='bar', color=['green', 'red'], alpha=0.7)

    plt.xlabel('Smoking Status')
    plt.ylabel('Count')
    plt.title('Distribution of Heart Disease based on Smoking Status')
    plt.legend(title='Heart Disease', loc='upper right', labels=['No', 'Yes'])
    plt.xticks(rotation=0)
    plt.savefig('../output/distribution/smoking_plot.png')
    plt.close()
    # plt.show()

def asthma_dist():
    #Asthma
    grouped_data = df1.groupby(['Asthma', 'HeartDisease']).size().unstack()
    grouped_data.plot(kind='bar', color=['green', 'red'], alpha=0.7)

    plt.xlabel('Asthma Status')
    plt.ylabel('Count')
    plt.title('Distribution of Heart Disease based on Asthma')
    plt.legend(title='Heart Disease', loc='upper right', labels=['No', 'Yes'])
    plt.xticks(rotation=0)
    plt.savefig('../output/distribution/asthma_plot.png')
    plt.close()
    # plt.show()

def kidney_dist():
    #kidney
    grouped_data = df1.groupby(['KidneyDisease', 'HeartDisease']).size().unstack()
    grouped_data.plot(kind='bar', color=['green', 'red'], alpha=0.7)

    plt.xlabel('Kidney Disease')
    plt.ylabel('Count')
    plt.title('Distribution of Heart Disease based on kindey disease')
    plt.legend(title='Heart Disease', loc='upper right', labels=['No', 'Yes'])
    plt.xticks(rotation=0)
    plt.savefig('../output/distribution/kidney_disease_plot.png')
    plt.close()
    # plt.show()

def skincancer_dist():
    #kidney
    grouped_data = df1.groupby(['SkinCancer', 'HeartDisease']).size().unstack()
    grouped_data.plot(kind='bar', color=['green', 'red'], alpha=0.7)

    plt.xlabel('Kidney Disease')
    plt.ylabel('Count')
    plt.title('Distribution of Heart Disease based on skin cancer')
    plt.legend(title='Heart Disease', loc='upper right', labels=['No', 'Yes'])
    plt.xticks(rotation=0)
    plt.savefig('../output/distribution/skin_cancer_plot.png')
    plt.close()
    # plt.show()

def complete_dist():
    # Complete setup
    data_renamed = df1.rename(columns={
        'BMI': 'BMI',
        'Smoking': 'Smoking',
        'AlcoholDrinking': 'Alcohol Drinking',
        'Stroke': 'Stroke',
        'PhysicalHealth': 'Physical Health',
        'MentalHealth': 'Mental Health',
        'DiffWalking': 'Difficulty in Walking',
        'Sex': 'Gender',
        'AgeCategory': 'Age',
        'Race': 'Race',
        'Diabetic': 'Diabetes',
        'PhysicalActivity': 'Physical Activity',
        'GenHealth': 'General Health',
        'SleepTime': 'Sleep Time',
        'Asthma': 'Asthma',
        'KidneyDisease': 'Kidney Disease',
        'SkinCancer': 'Skin Cancer'
        
    })

    data_renamed['HeartDisease'] = data_renamed['HeartDisease'].map({
        'presence of heart disease': 1, 
        'absence of heart disease': 0
    })

    cmap = sns.diverging_palette(0, 230, as_cmap=False)

    correlation_matrix_custom = data_renamed.drop(columns=[]).corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix_custom, vmax=1, center=0, cmap=cmap, square=True, linewidths=1, cbar_kws={"shrink": .5})
    plt.savefig('../output/distribution/complete_plot.png')
    plt.close()
    # plt.show()

def main():
    bmi_dist()
    smoke_dist()
    alcohol_dist()
    stroke_dist()
    physical_dist()
    mental_dist()
    diff_dist()
    sex_dist()
    age_dist()
    race_dist()
    diabetes_dist()
    phyicalhealth_dist()
    genhealth_dist()
    sleep_dist()
    smoke_dist()
    asthma_dist()
    kidney_dist()
    skincancer_dist()
    complete_dist()

if __name__ == "__main__":
    main()

sys.stdout.close()