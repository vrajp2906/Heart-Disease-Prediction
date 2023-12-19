import sys
import pandas as pd 
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.api as sm

sys.stdout = open('../output/analysis/analysis.txt', 'w')
df_temp = pd.read_csv('../db/heart_data_cleaned.csv')

df1_temp =  df_temp[df_temp.columns].replace({'Yes':1, 'No':0, 'Male':1,'Female':0,'No, borderline diabetes':'0','Yes (during pregnancy)':'1' })
df1_temp.head()

def skinCancer():
    #SkinCancer
    filtered_df = df1_temp[(df1_temp['SkinCancer'] == 0) | (df1_temp['SkinCancer'] == 1)]
    count_heart_disease = filtered_df.groupby('SkinCancer')['HeartDisease'].sum()
    contingency_table = pd.crosstab([df1_temp['SkinCancer']], [df1_temp['HeartDisease']])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    count_heart_disease = contingency_table.iloc[1, :]
    print("\nAnalysis of Skin Cancer\n")
    print(f"\tCount of individuals with heart disease who did not had SkinCancer: {count_heart_disease[0]}")
    print(f"\tCount of individuals with heart disease who did had SkinCancer: {count_heart_disease[1]}")
    print(f"\tChi-square statistic: {chi2}")
    print(f"\tP-value: {p:.10f}")

    alpha = 0.05
    if p < alpha:
        print("\tThere is a significant relationship between SkinCancer and heart disease.")
    else:
        print("\tThere is no significant relationship between SkinCancer and heart disease.")

def kidneyDisease():
    #KidneyDisease
    filtered_df = df1_temp[(df1_temp['KidneyDisease'] == 0) | (df1_temp['KidneyDisease'] == 1)]
    count_heart_disease = filtered_df.groupby('KidneyDisease')['HeartDisease'].sum()
    contingency_table = pd.crosstab([df1_temp['KidneyDisease']], [df1_temp['HeartDisease']])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    count_heart_disease = contingency_table.iloc[1, :]
    print("\nAnalysis of Kidney Disease\n")
    print(f"\tCount of individuals with heart disease who did not had KidneyDisease: {count_heart_disease[0]}")
    print(f"\tCount of individuals with heart disease who did had KidneyDisease: {count_heart_disease[1]}")
    print(f"\tChi-square statistic: {chi2}")
    print(f"\tP-value: {p:.10f}")

    alpha = 0.05
    if p < alpha:
        print("\tThere is a significant relationship between KidneyDisease and heart disease.")
    else:
        print("\tThere is no significant relationship between KidneyDisease and heart disease.")

def asthma():
#Asthma
    filtered_df = df1_temp[(df1_temp['Asthma'] == 0) | (df1_temp['Asthma'] == 1)]
    count_heart_disease = filtered_df.groupby('Asthma')['HeartDisease'].sum()
    contingency_table = pd.crosstab([df1_temp['Asthma']], [df1_temp['HeartDisease']])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    count_heart_disease = contingency_table.iloc[1, :]
    print("\nAnalysis of Asthma\n")
    print(f"\tCount of individuals with heart disease who did not had Asthma: {count_heart_disease[0]}")
    print(f"\tCount of individuals with heart disease who did had Asthma: {count_heart_disease[1]}")
    print(f"\tChi-square statistic: {chi2}")
    print(f"\tP-value: {p:.10f}")

    alpha = 0.05
    if p < alpha:
        print("\tThere is a significant relationship between Asthma and heart disease.")
    else:
        print("\tThere is no significant relationship between Asthma and heart disease.")

def sleepTime():
    #sleeptime
    print("\nAnalysis of Sleep Time\n")
    df_heart = df1_temp[['SleepTime', 'HeartDisease']].copy()

    X_bmi_heart = sm.add_constant(df_heart['SleepTime'])
    y_bmi_heart = df_heart['HeartDisease']

    model_bmi_heart = sm.Logit(y_bmi_heart, X_bmi_heart)
    result_bmi_heart = model_bmi_heart.fit()

    contingency_table = pd.crosstab(df1_temp['HeartDisease'], df1_temp['SleepTime'])

    # Perform chi-squared test
    chi2, p, _, _ = chi2_contingency(contingency_table)

    print(result_bmi_heart.summary())
    print(f"\tChi-squared value: {chi2}")
    print(f"\tP-value: {p}")

    if (p < 0.5):
        print("\tThere is significant relation between Sleep Time and Heart Disease")
    else:
        print("\tThere is no significant relation between Sleep Time and Heart Disease")

def genHealth():
    #GenHealth
    gen_health_categories = ['Poor', 'Fair', 'Good', 'Very good', 'Excellent']
    df1_temp['GenHealth'] = pd.Categorical(df1_temp['GenHealth'], categories=gen_health_categories, ordered=True)

    contingency_table = pd.crosstab(df1_temp["GenHealth"], df1_temp["HeartDisease"])
    chi2, p_chi2, _, _ = chi2_contingency(contingency_table)

    print("\nAnalysis of General Health\n")
    print("\tChi-squared test statistic:", chi2)
    print("\tP-value:", p_chi2)
    alpha = 0.05
    if p_chi2 < alpha:
        print("\tThere is evidence of a significant association between General Health and Heart Disease.")
    else:
        print("\tThere is no significant evidence of an association between General Health and Heart Disease.")

def physicalActivity():
    #PhysicalActivity
    physicalActivity_filtered_df = df1_temp[(df1_temp['PhysicalActivity'] == 0) | (df1_temp['PhysicalActivity'] == 1)]
    physical_count_heart_disease = physicalActivity_filtered_df.groupby('PhysicalActivity')['HeartDisease'].sum()
    contingency_table = pd.crosstab([df1_temp['PhysicalActivity']], [df1_temp['HeartDisease']])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    physical_count_heart_disease = contingency_table.iloc[1, :]
    
    print("\nAnalysis of Physical Activity\n")
    print(f"\tCount of individuals with heart disease who did not had PhysicalActivity: {physical_count_heart_disease[0]}")
    print(f"\tCount of individuals with heart disease who did had PhysicalActivity: {physical_count_heart_disease[1]}")
    print(f"\tChi-square statistic: {chi2}")
    print(f"\tP-value: {p:.10f}")

    alpha = 0.05
    if p < alpha:
        print("\tThere is a significant relationship between PhysicalActivity and heart disease.")
    else:
        print("\tThere is no significant relationship between PhysicalActivity and heart disease.")

def race():
    #Race
    contingency_table = pd.crosstab(df1_temp["Race"], df1_temp["HeartDisease"])
    chi2, p_chi2, _, _ = chi2_contingency(contingency_table)

    print("\nAnalysis of Race\n")
    print("\tChi-squared test statistic:", chi2)
    print("\tP-value:", p_chi2)

    if p_chi2 < 0.05:
        print("\tThere is evidence of a significant association between Race and Heart Disease.")
    else:
        print("\tThere is no significant evidence of an association between Race and Heart Disease.")

def age():
    #Age
    contingency_table = pd.crosstab(df1_temp["AgeCategory"], df1_temp["HeartDisease"])

    categories = df1_temp["AgeCategory"].unique()
    print("\nAnalysis of Age")
    for category in categories:
        count_heart_disease = contingency_table.loc[category, 1]
        count_total = contingency_table.loc[category, :].sum()

        count = np.array([count_heart_disease, count_total - count_heart_disease])
        nobs = np.array([count_total, count_total])

        stat, p_ztest = proportions_ztest(count, nobs, alternative='two-sided')

        print(f"\n\tZ-test for Proportions in Age Category {category}:")
        print("\tTest Statistic:", stat)
        print("\tP-value:", p_ztest)

        if p_ztest < 0.05:
            print(f"\tAge Category {category}. There is evidence of a significant difference in the proportions of heart disease.")
        else:
            print(f"\tAge Category {category}. There is no significant evidence of a difference in the proportions of heart disease.")

def gender():
    #Sex
    filtered_df = df1_temp[(df1_temp['Sex'] == 0) | (df1_temp['Sex'] == 1)]
    print("\nAnalysis of Gender\n")
    count_heart_disease = filtered_df.groupby('Sex')['HeartDisease'].sum()
    total_count = count_heart_disease.sum()

    print(f"\t{count_heart_disease[0] / total_count * 100:.2f}% were females having heart disease\n\t{count_heart_disease[1] / total_count * 100:.2f}% were males having heart disease")

    contingency_table = pd.crosstab(df1_temp["Sex"], df1_temp["HeartDisease"])
    count_male_heart_disease = contingency_table.loc[1, 1]
    count_female_heart_disease = contingency_table.loc[0, 1]
    count_male_total = contingency_table.loc[1, :].sum()
    count_female_total = contingency_table.loc[0, :].sum()

    count = np.array([count_male_heart_disease, count_female_heart_disease])
    nobs = np.array([count_male_total, count_female_total])

    stat, p_ztest = proportions_ztest(count, nobs, alternative='two-sided')

    print("\tZ-test for Proportions:")
    print("\tTest Statistic:", stat)
    print("\tP-value:", p_ztest)

    if p_ztest < 0.05:
        print("\tThere is evidence of a significant difference in the proportions of heart disease between genders.")
    else:
        print("\tThere is no significant evidence of a difference in the proportions of heart disease between genders.")

def diffWalking():
    #DiffWalking
    filtered_df = df1_temp[(df1_temp['DiffWalking'] == 0) | (df1_temp['DiffWalking'] == 1)]
    count_heart_disease = filtered_df.groupby('DiffWalking')['HeartDisease'].sum()
    contingency_table = pd.crosstab([df1_temp['DiffWalking']], [df1_temp['HeartDisease']])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    count_heart_disease = contingency_table.iloc[1, :]
    print("\nAnalysis of Difficulty in Walking\n")
    print(f"\tCount of individuals with heart disease who did not had DiffWalking: {count_heart_disease[0]}")
    print(f"\tCount of individuals with heart disease who did had DiffWalking: {count_heart_disease[1]}")
    print(f"\tChi-square statistic: {chi2}")
    print(f"\tP-value: {p:.10f}")

    alpha = 0.05
    if p < alpha:
        print("\tThere is a significant relationship between DiffWalking and heart disease.")
    else:
        print("\tThere is no significant relationship between DiffWalking and heart disease.")

def mentalHealth():
    #MentalHealth
    print("\nAnalysis of Mental Health\n")
    df_heart = df1_temp[['MentalHealth', 'HeartDisease']].copy()

    X_bmi_heart = sm.add_constant(df_heart['MentalHealth'])
    y_bmi_heart = df_heart['HeartDisease']

    model_bmi_heart = sm.Logit(y_bmi_heart, X_bmi_heart)
    result_bmi_heart = model_bmi_heart.fit()

    print(result_bmi_heart.summary())
    contingency_table = pd.crosstab(df1_temp['HeartDisease'], df1_temp['MentalHealth'])

    # Perform chi-squared test
    chi2, p, _, _ = chi2_contingency(contingency_table)

    print(f"\tChi-squared value: {chi2}")
    print(f"\tP-value: {p}")

    if (p < 0.5):
        print("\tThere is significant relation between Mental Health and Heart Disease")
    else:
        print("\tThere is no significant relation between Mental Health and Heart Disease")

def physicalHealth():
    #PhysicalHealth
    print("\nAnalysis of Physical Health\n")
    df_heart = df1_temp[['PhysicalHealth', 'HeartDisease']].copy()

    X_bmi_heart = sm.add_constant(df_heart['PhysicalHealth'])
    y_bmi_heart = df_heart['HeartDisease']

    model_bmi_heart = sm.Logit(y_bmi_heart, X_bmi_heart)
    result_bmi_heart = model_bmi_heart.fit()

    print(result_bmi_heart.summary())
    contingency_table = pd.crosstab(df1_temp['HeartDisease'], df1_temp['PhysicalHealth'])

    # Perform chi-squared test
    chi2, p, _, _ = chi2_contingency(contingency_table)

    print(f"\tChi-squared value: {chi2}")
    print(f"\tP-value: {p}")

    if (p < 0.5):
        print("\tThere is significant relation between Physical Health and Heart Disease")
    else:
        print("\tThere is no significant relation between Physical Health and Heart Disease")

def stroke():
    #Stroke
    filtered_df = df1_temp[(df1_temp['Stroke'] == 0) | (df1_temp['Stroke'] == 1)]
    count_heart_disease = filtered_df.groupby('Stroke')['HeartDisease'].sum()
    contingency_table = pd.crosstab([df1_temp['Stroke']], [df1_temp['HeartDisease']])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    count_heart_disease = contingency_table.iloc[1, :]
    print("\nAnalysis of Stroke\n")
    print(f"\tCount of individuals with heart disease who did not had stroke: {count_heart_disease[0]}")
    print(f"\tCount of individuals with heart disease who did had stroke: {count_heart_disease[1]}")
    print(f"\tChi-square statistic: {chi2}")
    print(f"\tP-value: {p:.10f}")

    alpha = 0.05
    if p < alpha:
        print("\tThere is a significant relationship between Stroke and heart disease.")
    else:
        print("\tThere is no significant relationship between Stroke and heart disease.")

def alcoholDrinking():
    #AlcoholDrinking
    filtered_df = df1_temp[(df1_temp['AlcoholDrinking'] == 0) | (df1_temp['AlcoholDrinking'] == 1)]
    count_heart_disease = filtered_df.groupby('AlcoholDrinking')['HeartDisease'].sum()
    contingency_table = pd.crosstab([df1_temp['AlcoholDrinking']], [df1_temp['HeartDisease']])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    count_heart_disease = contingency_table.iloc[1, :]
    print("\nAnalysis of Alcohol Drinking\n")
    print(f"\tCount of individuals with heart disease who do not drink: {count_heart_disease[0]}")
    print(f"\tCount of individuals with heart disease who drink: {count_heart_disease[1]}")
    print(f"\tChi-square statistic: {chi2}")
    print(f"\tP-value: {p:.10f}")

    alpha = 0.05
    if p < alpha:
        print("\tThere is a significant relationship between Alcohol Drinking and heart disease.")
    else:
        print("\tThere is no significant relationship between Alcohol Drinking and heart disease.")

def diabetes():
    #diabetes
    filtered_df = df1_temp[(df1_temp['Diabetic'] == 0) | (df1_temp['Diabetic'] == 1)]
    count_heart_disease = filtered_df.groupby('Diabetic')['HeartDisease'].sum()
    contingency_table = pd.crosstab([df1_temp['Diabetic']], [df1_temp['HeartDisease']])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    count_heart_disease = contingency_table.iloc[1, :]
    print("\nAnalysis of Diabetes\n")
    print(f"\tCount of individuals with heart disease without diabetes: {count_heart_disease[0]}")
    print(f"\tCount of individuals with heart disease with diabetes: {count_heart_disease[1]}")
    print(f"\tChi-square statistic: {chi2}")
    print(f"\tP-value: {p:.10f}")

    alpha = 0.05
    if p < alpha:
        print("\tThere is a significant relationship between diabetes and heart disease.")
    else:
        print("\tThere is no significant relationship between diabetes and heart disease.")

def smoke():
    #smoke
    filtered_df = df1_temp[(df1_temp['Smoking'] == 0) | (df1_temp['Smoking'] == 1)]
    count_heart_disease = filtered_df.groupby('Smoking')['HeartDisease'].sum()
    contingency_table = pd.crosstab([df1_temp['Smoking']],[ df1_temp['HeartDisease']])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    count_heart_disease = contingency_table.iloc[1, :]
    print("\tAnalysis of Smoking\n")
    print(f"\tCount of individuals with heart disease who doesn't smoke: {count_heart_disease[0]}")
    print(f"\tCount of individuals with heart disease who smoke: {count_heart_disease[1]}")
    print(f"\tChi-square statistic: {chi2}")
    print(f"\tP-value: {p:.5f}")


    alpha = 0.05
    if p < alpha:
        print("\tThere is a significant relationship between Smoking and heart disease.")
    else:
        print("\tThere is no significant relationship between Smoking and heart disease.")

def bmi():
    #BMI
    print("Analysis of BMI\n")
    df_bmi_heart = df1_temp[['BMI', 'HeartDisease']].copy()

    X_bmi_heart = sm.add_constant(df_bmi_heart['BMI'])
    y_bmi_heart = df_bmi_heart['HeartDisease']

    model_bmi_heart = sm.Logit(y_bmi_heart, X_bmi_heart)
    result_bmi_heart = model_bmi_heart.fit()

    print(result_bmi_heart.summary())
    contingency_table = pd.crosstab(df1_temp['HeartDisease'], df1_temp['BMI'])

    # Perform chi-squared test
    chi2, p, _, _ = chi2_contingency(contingency_table)

    print(f"\tChi-squared value: {chi2}")
    print(f"\tP-value: {p}")

    if (p < 0.5):
        print("\tThere is significant relation between BMI and Heart Disease")
    else:
        print("\tThere is no significant relation between BMI and Heart Disease")

def main():
    skinCancer()
    kidneyDisease()
    asthma()
    sleepTime()
    genHealth()
    physicalActivity()
    race()
    age()
    gender()
    diffWalking()
    mentalHealth()
    physicalHealth()
    stroke()
    alcoholDrinking()
    diabetes()
    smoke()
    bmi()

if __name__ == "__main__":
    main()

sys.stdout.close()