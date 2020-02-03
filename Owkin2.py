#!/usr/bin/env python
# coding: utf-8

# # Predicting lung cancer survival time 
# ## Owkin Challenge
# 
# The goal of this study if to predict a patient's survival time to lung cancer at a given time (given several features extracted from CT-scans). Due to the lack of time I will use directly the features extracted from the CT-scans and therefore not use the actual images. 
# 
# To do this we will create and train a model on a dataset composed of two sub-sets. Our training set is composed of 300 patients, for each we have a set of 53 radiomics features (extracted from the CT-scans) and 6 clinical features (TNM staging, histology, age and source dataset). The target data (which we want to predict) is the time to event variable that corresponds to the survival time if the patient died during the study (event = 1) or the amount of time after which the patient escaped the study (event = 0). When the patient escaped the study we called the data censored because it creates a lack in information: we only have the information that the patient was still alive up to this time but not how long he survived afterwards. It can therefore not be considered to carry the exact same relation to the parameters as a uncensored data.
# 
# I first opened the csv files to see there format and saw that the headers of the radiomics files were comprised of one blank line that had to remove (copy/paste 'PatientID' in the header line and remove the extra line) in order to get proper header when importing the data.

# In[905]:


import pandas as pd
pd.options.display.max_rows = 20
pd.options.display.max_columns = 55
pd.options.display.float_format = '{:.1f}'.format

radiomics_train_ini = pd.read_csv('D:/Owkin_challenge/data_xtrain/features/radiomics.csv', index_col=0, header=1)
clinical_train_ini = pd.read_csv('D:/Owkin_challenge/data_xtrain/features/clinical_data.csv', index_col=0)
output_train_ini = pd.read_csv('D:/Owkin_challenge/output.csv', index_col=0)


# Before getting started on the actual model I'm used to exploring a bit the data, to get familiar with it and start extracting some insight on our problem which can be useful when selecting features and models.
# 
# ## Exploring the data
# 
# First let's check all the features available and the target data.

# In[11]:


import matplotlib.pyplot as plt

print(output_train_ini.keys())
print(clinical_train_ini.keys())
print(radiomics_train_ini.keys())


# The radiomics data consist of 53 features on different aspects of the CT-scan of the tumor. I will start by working on this table. I will check that there are no missing data and if there are I will fill them with the median value of the corresponding feature.

# In[504]:


radiomics_train_ini.isnull().any().any()


# There are no missing data in this table so we can use it as such.
# 
# Next I'll have a look on the data and see if normalization of the data would be needed (normalization can be necessary if different features have very different ranges because they would introduce inner differences in weight over the output).

# In[26]:


radiomics_train_ini.describe()


# The range of the values differs quite a lot (e.g. 'original_firstorder_Energy' as a mean of 2725406582.2 where 'original_shape_Compactness2' has a mean of 0.3). Normalizing the data will then help me have a set of features that are spread over even ranges without interfering with the distribution of values for each features. I need to keep the rescaling values in memory because I will need to apply the exact same rescaling method on the testing set before applying my model otherwise the results will be false.
# I will apply a standardization so that each features are spread with a mean of 0 and a standard deviation of 1.

# In[505]:


from sklearn import preprocessing

scaler_radiomics = preprocessing.StandardScaler().fit(radiomics_train_ini)
radiomics_train_norm = radiomics_train_ini.copy()
radiomics_train_norm[radiomics_train_norm.columns] = scaler_radiomics.transform(
    radiomics_train_ini[radiomics_train_ini.columns])

radiomics_train_norm.describe()


# My goal is now to select a sample of the radiomics features that hold information for the prediction of survival time. This is important because non-informative variables put noise in the model and can cause overfitting and decrease the prediction precision.
# To do so I'll use filter methods to determine the correlation between a feature and our targeted survival time. 
# 
# Some of our data are censored, which mean that the time-to-event value that we get from them doesn't correspond to a time of death but to a time where the subject escaped the study. 

# In[38]:


pd.value_counts(output_train_ini['Event']).plot.bar(title = "Repartition of death and escape from study")


# Our dataset presents a huge part of censored data, almost half of the subjects escaped the study. For our correlation analysis to select relevant features we will drop the observation corresponding to censored data because the survival time doesn't have the same meaning. 

# In[62]:


target_uncensored = output_train_ini['SurvivalTime'][output_train_ini['Event'] == 1]
radiomics_uncensored = radiomics_train_norm[output_train_ini['Event'] == 1]


# I will use the Pearson correlation coefficient on each of the feature to mesurate the correlation between this feature and the survival time.

# In[191]:


pearson_table = pd.concat([radiomics_uncensored, target_uncensored], axis=1, sort=False).corr(method='pearson')
pearson_table


# The different features have different correlation values with the survival time, we want to select only the most important for the prediction.

# In[196]:


idx_corr = pearson_table[np.abs(pearson_table.SurvivalTime) < 0.29].index
pearson_table_select = pearson_table.copy()
pearson_table_select = pearson_table_select.drop(index = idx_corr, columns = idx_corr).drop(columns='SurvivalTime', axis=1)
pearson_table_select


# By selecting only the features that have the strongest correlation (above 0.29 in absolute) with the survival time I would keep 14 of the 53 features for my model. Next I want to verify that the selected features don't bring redundant information. The features that we would keep differ from the features kept by the baseline model. 
# 
# I looked at the correlation table obtained by keeping the censored data (subjects that escaped the study) and some of the features of the baseline model appeared as the ones strongly correlated to the survival time. For my study of this problem I will rather keep the most correlated features when calculing it without the censored data because the correlation between the time one escapes the study doesn't have to be related to any radiomics parameter.
# 
# By looking at the correlation table above I see that some features are strongly correlated among themselves. Keeping all those features can maybe bring biais on the future model. I will then get rid of some features that seem strongly correlated with a lot of others. As a threshold I take that if a feature as a correlation very close to 1 or -1 with more than 4 features (not counting itself) I remove it from the set.

# In[506]:


mask = np.sum(np.abs(pearson_table_select[np.abs(pearson_table_select)>0.95]))>=4.8
pearson_table_select = pearson_table_select.drop(index = mask[mask==True].index, columns = mask[mask==True].index)
pearson_table_select


# In[323]:


import seaborn as sns

sns.heatmap(pearson_table_select, annot=True, cmap=plt.cm.Greens)
plt.show()


# That leaves me with a set of 9 radiomics features that I will try to work on.
# 
# After dealing with the radiomics features I will look at the clinical data. This table has 6 features: histology, NTM stages, source dataset and age. I will proceed the same way I did with the radiomics data, first checling for missing values.

# In[507]:


radiomics_train = radiomics_train_norm[pearson_table_select.columns]


# In[508]:


clinical_train_ini.isnull().any()


# There are some missing data on the histology and age features. 
# 
# The age is a numerical feature. I will then replace the missing values by the median value of the set and standardize it.
# I will plot the bar graph of the histology categorical values count to check the categories repartition as well as the categories names.

# In[901]:


clinical_train_norm = clinical_train_ini.copy()
clinical_train_norm['age'] = clinical_train_norm.age.replace(
    np.NaN, np.median(clinical_train_ini['age'][~clinical_train_ini['age'].isnull()].values))
                      
scaler_age = preprocessing.StandardScaler().fit(clinical_train_norm['age'].values.reshape(-1, 1))

clinical_train_norm['age'] = scaler_age.transform(clinical_train_norm['age'].values.reshape(-1, 1))

pd.value_counts(clinical_train_norm['Histology']).plot.bar(title = "Reparition of histology feature")                 


# Apart from the missing values there are also problems in the naming of the categories that are redundant. I will rename the categories to four labels: AC (adenocarcinoma), LC (large cell), SCC (squamous cell carcinoma) and nos (not otherwise specified). I will put the missing data in the last category.

# In[907]:


clinical_train_norm['Histology'] = clinical_train_ini['Histology'].replace(
    ['Adenocarcinoma','adenocarcinoma', 'NSCLC NOS (not otherwise specified)', 'Squamous cell carcinoma', 
     'squamous cell carcinoma', 'large cell', np.NaN], ['AC', 'AC', 'nos', 'SCC', 'SCC', 'LCC', 'nos'])
pd.value_counts(clinical_train_norm['Histology']).plot.bar(title = "Reparition of histology feature")


# Now that the values are all well categorized and the missing data dealt with I can start selecting the feature I would keep in the prediction model. Let's extract uncensored data to test the features on.

# In[ ]:


clinical_uncensored = clinical_train_norm.copy()
clinical_uncensored = clinical_train_norm[output_train_ini['Event'] == 1]


# In[888]:


sns.boxplot(x='Histology', y='SurvivalTime', 
            data = pd.concat([clinical_uncensored["Histology"],target_uncensored], axis=1, sort=False))


# The histology of the tumor doesn't seem to give much information on the survival time seeing the repartition of values on the above graph : the median and first quartile are very similar on all four possibilities, only the third quartile of the histology adenocarcinoma is higher that the rest. This feature will probably not produce much insight on the survival time.

# In[889]:


sns.boxplot(x='SourceDataset', y='SurvivalTime', 
            data = pd.concat([clinical_uncensored["SourceDataset"],target_uncensored], axis=1))


# The dataset labelled l1 seems more correlated with early deaths than the dataset labelled l2 seeing the repartition of values: survival time of patients in the second datasets are spread over the whole time range. This feature can thus be of importance in the prediction model.

# In[894]:


sns.boxplot(x='Nstage', y='SurvivalTime',
            data = pd.concat([clinical_uncensored["Mstage"],clinical_uncensored["Tstage"],
                              clinical_uncensored["Nstage"],target_uncensored], axis=1))


# For the feature Mstage, the majority of data is in the category 0, there is not enough variability among the data to use it as a predictor.
# 
# For the feature Tstage, the median value is similar for the categories 2 to 4, being lower than the median time for the category 1. Moreover the variance is greater in the first category than the others. This feature could thus be used as a predictor.
# 
# For the feature Nstage, the median seems similar between all categories though getting smaller towards category 3. The variance of the data are also decreasing with a higher category. It can then also be used as a interesting feature.
# 
# When plotting the survival time depending on both Nstage and Tstage it doesn't appear that the two stages are much related together because dots representing one Tstage are found within all Nstage values. 
# 
# Keeping the features Tstage and Nstage in the model could thus be useful to predict survival time. That seems quite a valid assumption, T and N stages being respectively the size of the original tumor and the nearby lymph nodes implicated, we can assume that the bigger the initial tumor is and the more lymph nodes there are, the fewer survival time the patient has.

# In[350]:


plt.scatter(clinical_train_ini['age'][output_train_ini['Event']==1], target_uncensored)


# The age can be an insightful feature to add to the model because it seems that the variability of survival time is greater when the subject is older. This has though to be a value to be careful on because of the repartition of ages among the subjects. Indeed there are more older patients and therefore the variability can be higher that the youngest without the age being a real cause.
# 
# From the clinical data 4 features are then kept. To create the model I will use a one-hot encoding on the categorical data to be sure that each values of the category have the same weight in the function.

# In[356]:


clinical_train = clinical_train_norm[['SourceDataset', 'Nstage', 'Tstage', 'age']]
clinical_train = pd.get_dummies(clinical_train,prefix=['SourceDataset', 'Nstage', 'Tstage'], 
                                columns = ['SourceDataset', 'Nstage', 'Tstage'], drop_first=True)
features_train = pd.concat([radiomics_train, clinical_train], axis=1, sort=False)
features_train.head()


# Since all the preprocessing of the data I've done will need to be done again on the training input set I'll create a function now that I can use later.

# In[873]:


def preprocessing_data(input_radiomics, input_clinical, radiomics_scaler, age_scaler, idx_col, 
                       idx_col_clini, separate_dataset=False):
    i_rad = input_radiomics.copy()
    if input_radiomics.isnull().any().any():
        for col in i_rad.columns:
            i_rad[col] = i_rad[col].replace(
                np.NaN, np.median(i_rad[col][~i_rad[col].isnull()].values))
    r_norm = i_rad.copy()
    r_norm[r_norm.columns] = radiomics_scaler.transform(i_rad[i_rad.columns])
    r = r_norm[idx_col]
    
    i_cli = input_clinical.copy()
    i_cli['age'] = i_cli.age.replace(
        np.NaN, np.median(i_cli['age'][~i_cli['age'].isnull()].values))
    i_cli['age'] = age_scaler.transform(i_cli['age'].values.reshape(-1, 1))
    i_cli['Histology'] = i_cli['Histology'].replace(
        ['Adenocarcinoma','adenocarcinoma', 'NSCLC NOS (not otherwise specified)', 'Squamous cell carcinoma', 
         'squamous cell carcinoma', 'large cell', np.NaN], ['AC', 'AC', 'nos', 'SCC', 'SCC', 'LCC', 'nos'])
    i = i_cli[idx_col_clini]
    if 'age' in idx_col_clini:
        col_dum = idx_col_clini.drop('age')
    else:
        col_dum = idx_col_clini
    if separate_dataset :
        col_dum = col_dum.drop('SourceDataset')
        i1 = i[i['SourceDataset']=="l1"].drop('SourceDataset', axis=1)
        i2 = i[i['SourceDataset']=="l2"].drop('SourceDataset', axis=1)
        i1 = pd.get_dummies(i1,prefix=col_dum.values, 
                                columns = col_dum.values, drop_first=True)
        i2 = pd.get_dummies(i2,prefix=col_dum.values, 
                                columns = col_dum.values, drop_first=True)
        return pd.concat([r[i['SourceDataset']=="l1"], i1], axis=1, sort=False), pd.concat(
            [r[i['SourceDataset']=="l2"], i2], axis=1, sort=False)
    else:
        i = pd.get_dummies(i,prefix=col_dum.values, 
                                columns = col_dum.values, drop_first=True)
        return pd.concat([r, i], axis=1, sort=False)
    


# ## Creating the model
# 
# The model that I will try first is a CoxPH model that expresses the hazard ration depending on the features and associated parameters that I want to evaluate.
# 
# ### CoxPH model with features selection
# 
# Since I need to test my model I will subdivise my dataset into smaller folds and perform cross-validation with 3 folds : the model will be fitted on 2 out of the 3 folds, tested on the tenth and the score (concordance index used in CoxPH models) will be calculated. This process is repeated 3 times and comparing the scores will give us an insight on the best model.

# In[536]:


from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn import model_selection

output_train = output_train_ini[['Event', 'SurvivalTime']]
output_train['Event'] = output_train['Event'].astype('bool')

estimator = CoxPHSurvivalAnalysis(alpha=0.1) 

def create_apply_model(est, feature, target, nb_cv):
    #adding a little noise on the data to be sure that they are independant and do not raise linalg error
    n = np.size(feature, 0)
    p = np.size(feature, 1)
    cv_results = model_selection.cross_validate(est, feature.values+0.00001*np.random.rand(n,p), 
                                            target.to_records(index=False), cv=nb_cv, return_estimator=True)
    
    best_estimator = cv_results['estimator'][np.where(cv_results['test_score']==np.max(cv_results['test_score']))[0][0]]
    print(cv_results['test_score'])
    return best_estimator

model_etimator = create_apply_model(estimator, features_train, output_train, 3)
pd.Series(model_etimator.coef_, index=features.columns)


# The resulting best model is the third one cause it has the best performance score (0.691). Next I will apply this model over the whole set (combining all 3 folds) to get the whole model parameters.

# In[544]:


from sksurv.metrics import concordance_index_censored

predict_train = model_etimator.predict(features_train)
result = concordance_index_censored(output_train["Event"], output_train["SurvivalTime"], predict_train)
result[0]


# The goal of this project is to predict the survival time of each subject. The CoxPH model gives us the hazard function and the global risk of death. By computing the hazard function I can access the median time when a estimated survival probability reaches 0.5. To do so I fit an exponential function on my survival probabilities step function and I access the time when the probability is 0.5.
# 
# This might not be appropriate for every subject because when the probability doesn't decrease a lot the estimated median time will grow.

# In[602]:


from scipy.optimize import curve_fit

predict_train_function = model_etimator.predict_survival_function(features_train)

#extrapolates linear function between steps above and below probability=0.5
def get_predict_time(predict, feature, fit=True):
    median_survival_time = np.zeros(np.size(predict))
    for i in range(np.size(predict)):
        if fit:
            sol = curve_fit(lambda t,a,b: a*np.log(b*t),  predict[i].y,  predict[i].x, p0 = (-1000,1))
            median_survival_time[i] = np.log(0.5*sol[0][1])*sol[0][0]
        else:
            j=0
            while j<np.size(predict[i].x) and predict[i].y[j]>0.5:
                j += 1
            if j < np.size(predict[i].x):
                x = predict[i].x[j-1]-predict[i].x[j]
                y = predict[i].y[j-1]-predict[i].y[j]
                median_survival_time[i] = predict[i].x[j-1]+(0.5-predict[i].y[j-1])*x/y
            else:
                median_survival_time[i] = predict[i].x[j-1]
    pred = pd.Series(median_survival_time,name='Predicted', index=feature.index)
    return pred
predicted_survival_time = get_predict_time(predict_train_function, features_train)
predicted_survival_time


# In[572]:


df = pd.concat([predicted_survival_time, output_train], axis=1, sort=False)
sns.scatterplot(x='SurvivalTime', y='Predicted', hue='Event', data=df)
print("n° of escape (Event = 0) : " + np.str(np.size(output_train[output_train["Event"]==0],0)) +
         " n° of deaths (Event = 1) : " + np.str(np.size(output_train[output_train["Event"]==1],0)))
sns.lineplot(x=predict_train_function[0].x, y=predict_train_function[0].x)


# I know have one model that can work to predict survival times.
# 
# Because the source of the dataset can affect the output greatly as we have seen (the variability of times is greater for the second dataset) I will try to cut my dataset in 2 and to consider two distinct models depending on whether the patient is from dataset l1 or l2 and see if they achieve better predictions by themselves.

# In[588]:


features_train_set1, features_train_set2 = preprocessing_data(
    radiomics_train_ini, clinical_train_ini, scaler_radiomics, scaler_age, pearson_table_select.columns, True)


# To ease the calcul, retrieval and plotting of predictions I create the following function.

# In[800]:


def get_predictions(feature, target, model, fit=True):
    pred = model.predict(feature)
    ci = concordance_index_censored(target["Event"], target['SurvivalTime'], pred)
    print("CI = " + np.str(ci[0]))

    pred_fct = model.predict_survival_function(feature)
    pred_median = get_predict_time(pred_fct, feature, fit)
    df = pd.concat([pred_median, target], axis=1, sort=False)
    sns.scatterplot(x='SurvivalTime', y='Predicted', hue='Event', data=df)
    print("n° of escape (Event = 0) : " + np.str(np.size(target[target["Event"]==0],0)) +
         " n° of deaths (Event = 1) : " + np.str(np.size(target[target["Event"]==1],0)))
    sns.lineplot(x=pred_fct[0].x, y=pred_fct[0].x)
    return pred, pred_fct


# In[575]:


model_etimator_set1 = create_apply_model(estimator, features_train_set1, 
                                         output_train[clinical_train_ini['SourceDataset']=='l1'], 3)
pd.Series(model_etimator_set1.coef_, index=features_train_set1.columns)

predict_train_set1, predict_fct_set1 = get_predictions(
    features_train_set1, output_train[clinical_train_ini['SourceDataset']=='l1'], model_etimator_set1)


# The graph shows the predicted survival times plotted against the actual survival time. The line means perfect equality between the two. The estimation of variance is large and the center of plotted data is located to the left of the line, so our predicted time might be biaised.
# 
# The correlation index is also smaller than the precedent model

# In[631]:


model_etimator_set2 = create_apply_model(estimator, features_train_set2, 
                                         output_train[clinical_train_ini['SourceDataset']=='l2'], 3)
pd.Series(model_etimator_set2.coef_, index=features_train_set2.columns)

predict_train_set2, predict_fct_set2 = get_predictions(
    features_train_set2, output_train[clinical_train_ini['SourceDataset']=='l2'], model_etimator_set2, False)


# To get the median time for this dataset I didn't used the exponential approximation because the rate of decreasing of the function is often very small (because this set comprises of a lot of uncensored data) and therefore the calculated times end up very high.
# 
# From the second dataset a lot of the predicted time reached the time limit, meaning that the predicted probability of survival was always above 0.5. It is interesting to see that this happened only for subjects among the second dataset. One of the main reason I see to this is the proportion of censored data in each subset : in the second dataset two third of the subjects escaped the study wheres in the first dataset only one third escaped it.
# 
# The correlation index is better than the last model. The reason I see are those maximum times: since the CI is calculated on the right ordering of valid pairs if two subjects have the same predicted time of 3500 days their order can be assessed as valid even if the information in it is false.
# 
# ### CoxPH models on two separate subsets
# 
# One of the problems I'm seeing is the way I separated my dataset in two, I'm creating two different models but I use the standardization accross the complete table and I keep the same features from both. Since I want to consider two different models for the two subsets, I need to cut my data from the beginning and do the whole scaling, features selection and model fitting process. To make it easier for next tests I will create a pipeline of this process that can be used with variations.

# In[705]:


def pipeline(radio, clini, out, idx_col, th, selectF = True):
    #normalize data
    scaler_rad = preprocessing.StandardScaler().fit(radio)
    radio_norm = radio.copy()
    radio_norm[radio_norm.columns] = scaler_rad.transform(radio[radio.columns])
    
    clini_norm = clini.copy()
    clini_norm['age'] = clini_norm.age.replace(
        np.NaN, np.median(clini['age'][~clini['age'].isnull()].values))
    scaler_a = preprocessing.StandardScaler().fit(clini_norm['age'].values.reshape(-1, 1))
    clini_norm['age'] = scaler_a.transform(clini_norm['age'].values.reshape(-1, 1))
  
    #select features  
    def select_features(feat, target, th):
        ps = pd.concat([feat, target], axis=1, sort=False).corr(method='pearson')
        i = ps[np.abs(ps.SurvivalTime) < th-0.01].index
        pss = ps.copy()
        pss = pss.drop(index = i, columns = i).drop(columns='SurvivalTime', axis=1)
        mask = np.sum(np.abs(pss[np.abs(pss)>0.95]))>=4.8
        return pss.drop(index = mask[mask==True].index, columns = mask[mask==True].index)
    if selectF:
        targ_un = out['SurvivalTime'][out['Event'] == 1]
        radio_un = radio[out['Event'] == 1]
        feat_col = select_features(radio_un, targ_un, th)
        feat = preprocessing_data(radio, clini, scaler_rad, scaler_a, feat_col.columns, idx_col)
    else:
        feat = preprocessing_data(radio, clini, scaler_rad, scaler_a, radio.columns, idx_col)
    
    return scaler_rad, scaler_a, feat


# In[684]:


id_col_clini = (clinical_train_ini.columns == 'age') + (clinical_train_ini.columns == 'Nstage') + (clinical_train_ini.columns == 'Tstage') 

idx_col1 = clinical_train_ini['SourceDataset']=='l1'
rad1 = radiomics_train_ini[idx_col1]
cli1 = clinical_train_ini[idx_col1]
scaler_rad1, scaler_a1, features1 = pipeline(rad1, cli1, output_train[idx_col1], clinical_train_ini.columns[id_col_clini], 0.3)

idx_col2 = clinical_train_ini['SourceDataset']=='l2'
rad2 = radiomics_train_ini[idx_col2]
cli2 = clinical_train_ini[idx_col2]
scaler_rad2, scaler_a2, features2 = pipeline(rad2, cli2, output_train[idx_col2], clinical_train_ini.columns[id_col_clini], 0.25)


# In[687]:


model_1 = create_apply_model(estimator, features1, 
                                         output_train[clinical_train_ini['SourceDataset']=='l1'], 3)
pd.Series(model_1.coef_, index=features1.columns)

predict_1, predict_fct_1 = get_predictions(
    features1, output_train[clinical_train_ini['SourceDataset']=='l1'], model_1)


# In[689]:


model_2 = create_apply_model(estimator, features2, 
                                         output_train[clinical_train_ini['SourceDataset']=='l2'], 3)
pd.Series(model_2.coef_, index=features2.columns)

predict_2, predict_fct_2 = get_predictions(
    features2, output_train[clinical_train_ini['SourceDataset']=='l2'], model_2, False)


# The prediction still don't seem the best. But again the large amount (almost half) of censored data if the set leads to a lot of errors in prediction.
# 
# ### Test of the model on uncensored data
# 
# In order to see how the censored data affect the model I'll try to apply the process (without subdividing into two datasets) to  only the uncensored data.

# In[691]:


col_id = (clinical_train_ini.columns == 'age') + (clinical_train_ini.columns == 'Nstage') + (
    clinical_train_ini.columns == 'Tstage') + (clinical_train_ini.columns == 'SourceDataset')  


scaler_rad_uncensored, scaler_age_uncensored, features_uncensored = pipeline(radiomics_train_ini[output_train_ini['Event']==1],
                                             clinical_train_ini[output_train_ini['Event']==1],
                                             output_train[output_train_ini['Event']==1], 
                                             clinical_train_ini.columns[col_id], 0.3)

model_uncensored = create_apply_model(estimator, features_uncensored, 
                                         output_train[output_train_ini['Event']==1], 3)
pd.Series(model_uncensored.coef_, index=features_uncensored.columns)

predict_uncensored, predict_fct_uncensored = get_predictions(
    features_uncensored, output_train[output_train_ini['Event']==1], model_uncensored, False)


# With this scenario the variance of the predicted times are greatly diminished but can maybe match better the reality. When dealing with survival times a small variation to the mean is preferred. However the CoxPH model is made to work on sets with censored data so the prediction should take that into account. The large fraction of censored data in this set might be the reason it is not working as well. 
# 
# ### CoxPH model with the features used in the baseline model
# In the project description, a basaline model is presented, which is the CoxPH model with 8 features of interest. I will try to run my pipeline with this characteristics.

# In[707]:


col_id = (clinical_train_ini.columns == 'Nstage')+ (clinical_train_ini.columns == 'SourceDataset')  

scaler_rad_baseline, scaler_age_baseline, features_baseline = pipeline(radiomics_train_ini, clinical_train_ini,
                                             output_train, clinical_train_ini.columns[col_id], 0.3, False) 
x = radiomics_train_ini.columns.drop(["original_shape_Sphericity",
    "original_shape_SurfaceVolumeRatio",
    "original_shape_Maximum3DDiameter",
    "original_firstorder_Entropy",
    "original_glcm_Id",
    "original_glcm_Idm"])
features_baseline = features_baseline.drop(x, axis=1)

model_baseline = create_apply_model(estimator, features_baseline, output_train, 3)
pd.Series(model_baseline.coef_, index=features_baseline.columns)

predict_baseline, predict_fct_baseline = get_predictions(features_baseline, output_train, model_baseline, False)


# The results are quite similar to those of my other models, with still a huge error brought by the censored data.
# The concordance index is a bit smaller than my first model tested.

# ## Validation on the test set and prediction
# 
# I will now retrieve the test set and fit my model on it to get the predicted times.

# In[877]:


radiomics_test_ini = pd.read_csv('D:/Owkin_challenge/data_xtest/features/radiomics.csv', index_col=0, header=1)
clinical_test_ini = pd.read_csv('D:/Owkin_challenge/data_xtest/features/clinical_data.csv', index_col=0)

features_test_set1, features_test_set2 = preprocessing_data(
    radiomics_test_ini, clinical_test_ini, scaler_radiomics, scaler_age, pearson_table_select.columns, 
    clinical_test_ini[['age', 'Nstage', 'Tstage', 'SourceDataset']].columns, True)

def add_column_missing(ref, test, value):
    for col in ref.columns:
        if (col not in test.columns):
            test[col] = ref[col]
            test[col] = value
    testcop = test.copy()
    for col in test.columns:
        if (col not in ref.columns):
            testcop = testcop.drop(col, axis=1)
    return testcop

features_test_set1 = add_column_missing(features_train_set1, features_test_set1, 0)
features_test_set2 = add_column_missing(features_train_set2, features_test_set2, 0)    


# In[878]:


predict_test_fct1 = model_etimator_set1.predict_survival_function(features_test_set1)
predict_test_median1 = get_predict_time(predict_test_fct1, features_test_set1)
predict_test_median1.describe()


# In[603]:


predict_test_fct2 = model_etimator_set2.predict_survival_function(features_test_set2)
predict_test_median2 = get_predict_time(predict_test_fct2, features_test_set2, False)
predict_test_median2.describe()


# With the second datasets, the predicted survival times don't seem accurate, this is due to the fact that our model was constructed on mostly uncensored data from which it is hard to concluded things. It might therefore be better to work on the entire dataset to spread the unknown fraction carried by the censored data.

# In[738]:


features_test = preprocessing_data(
    radiomics_test_ini, clinical_test_ini, scaler_radiomics, scaler_age, pearson_table_select.columns, 
    clinical_test_ini.columns.drop('Mstage', 'Histology'))
features_test = add_column_missing(features_train, features_test[features_train.columns], 0)

predict_test_fct = model_etimator.predict_survival_function(features_test)
predict_test_median = get_predict_time(predict_test_fct, features_test)
predict_test_median[predict_test_median>3500] = 3500 
predict_test_median = predict_test_median.rename(columns={'Predicted':'SurvivalTime'})

nan = np.empty(np.size(predict_test_median))
nan[:] = 'nan'
output_test = pd.DataFrame({'SurvivalTime':predict_test_median, 'Event':nan}, index=predict_test_median.index)
output_test


# In[739]:


output_test.to_csv(r'D:/Owkin_challenge/output_test.csv', na_rep='nan')


# ## Another try at a model, random forest
# 
# I managed to predict survival time with a performance of 0.6808 which is below the baseline. I want thus to create a new model whose predictions would be better. I thought of random forest.
# 
# For this I will use the features used in the baseline model.

# In[831]:


from sklearn.model_selection import train_test_split
from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest

radiomics_RTF = radiomics_train_ini[["original_shape_Sphericity",
    "original_shape_SurfaceVolumeRatio",
    "original_shape_Maximum3DDiameter",
    "original_firstorder_Entropy",
    "original_glcm_Id",
    "original_glcm_Idm"]]

#scale_RTF = preprocessing.StandardScaler().fit(radiomics_RTF)
radiomics_RTF_norm = radiomics_RTF.copy()
#radiomics_RTF_norm[radiomics_RTF_norm.columns] = scale_RTF.transform(
#    radiomics_RTF[radiomics_RTF.columns])

clinical_columns = ['SourceDataset', 'Nstage']
clinical_RTF = pd.get_dummies(clinical_train_ini[['SourceDataset', 'Nstage']], prefix=clinical_columns,
                              columns=clinical_columns, drop_first=True)
features_RTF = pd.concat([radiomics_RTF_norm, clinical_RTF], axis=1, sort=False).astype(np.float64)

output_train['SurvivalTime'] = output_train['SurvivalTime'].astype(np.float64)

random_state=20
X_train, X_test, y_train, y_test = train_test_split(features_RTF, output_train.to_records(index=False), 
                                                    test_size=0.25, random_state=random_state)


# In[832]:


rsf = RandomSurvivalForest(n_estimators=100,
                           min_samples_split=10,
                           min_samples_leaf=15,
                           max_features="sqrt",
                           n_jobs=-1,
                           random_state=random_state)
rsf.fit(X_train, y_train)


# In[833]:


rsf.score(X_test, y_test)


# In[813]:


def get_predict_time_RSF(predict, feature, model, fit=True):
    median_survival_time = np.zeros(np.size(predict,0))
    times = model.event_times_
    for i in range(np.size(predict,0)):
        if fit:
            sol = curve_fit(lambda t,a,b: a*np.log(b*t),  predict[i,:],  times, p0 = (-1000,1))
            median_survival_time[i] = np.log(0.5*sol[0][1])*sol[0][0]
        else:
            j=0
            while j<np.size(times) and predict[i,j]>0.5:
                j += 1
            if j < np.size(times):
                x = times[j-1]-times[j]
                y = predict[i,j-1]-predict[i,j]
                median_survival_time[i] = times[j-1]+(0.5-predict[i,j-1])*x/y
            else:
                median_survival_time[i] = times[j-1]
    pred = pd.Series(median_survival_time,name='Predicted', index=feature.index)
    return pred

def get_predictions_RSF(feature, target, model, fit=True):
    pred = model.predict(feature)
    ci = concordance_index_censored(target["Event"], target['SurvivalTime'], pred)
    print("CI = " + np.str(ci[0]))

    pred_fct = model.predict_survival_function(feature)
    pred_median = get_predict_time_RSF(pred_fct, feature, model, fit)
    df = pd.concat([pred_median, target], axis=1, sort=False)
    sns.scatterplot(x='SurvivalTime', y='Predicted', hue='Event', data=df)
    print("n° of escape (Event = 0) : " + np.str(np.size(target[target["Event"]==0],0)) +
         " n° of deaths (Event = 1) : " + np.str(np.size(target[target["Event"]==1],0)))
    sns.lineplot(x=model.event_times_, y=model.event_times_)
    return pred, pred_fct


# In[816]:


pred_RSF, pred_fct_RSF = get_predictions_RSF(features_RTF, output_train, rsf, True)


# The concordance index seems quite good, but the predicted times are still quite higher than the actual ones. 
# I will use this model to predict on my test set.

# In[880]:


radiomics_test_RTF = radiomics_test_ini[["original_shape_Sphericity",
    "original_shape_SurfaceVolumeRatio",
    "original_shape_Maximum3DDiameter",
    "original_firstorder_Entropy",
    "original_glcm_Id",
    "original_glcm_Idm"]]

clinical_columns = ['SourceDataset', 'Nstage']
clinical_test_RTF = pd.get_dummies(clinical_test_ini[['SourceDataset', 'Nstage']], prefix=clinical_columns,
                              columns=clinical_columns, drop_first=True)
features_test_RTF = pd.concat([radiomics_test_RTF, clinical_test_RTF], axis=1, sort=False).astype(np.float64)


features_test_RTF = add_column_missing(features_RTF, features_test_RTF, 0)

predict_test_fct_RTF = rsf.predict_survival_function(features_test_RTF)
predict_test_median_RTF = get_predict_time_RSF(predict_test_fct_RTF, features_test_RTF, rsf)
predict_test_median_RTF[predict_test_median_RTF>3500] = 3500 
predict_test_median_RTF = predict_test_median_RTF.rename(columns={'Predicted':'SurvivalTime'})

nan = np.empty(np.size(predict_test_median_RTF))
nan[:] = 'nan'
output_test_RTF = pd.DataFrame({'SurvivalTime':predict_test_median_RTF, 'Event':nan}, index=predict_test_median_RTF.index)
output_test_RTF.describe()


# In[882]:


output_test_RTF.to_csv(r'D:/Owkin_challenge/output_test_RSF.csv', na_rep='nan')


# ## Future perspectives
# 
# The CoxPH model gave a 0.68 concordance index and the Random Survival Foest a 0.67 concordance index on the test set which is slighty inferior to the baseline. With more time I would go over all choices and asumptions I made to improve my model, starting by pursuing the following ideas that I had for this project:
# 
# - find a way to deal with the uncensored data problem, maybe dig deeper into the CoxPH model to see how those data are dealt with. From what I saw in the litterature the used datasets are usually bigger and have a smaller proportion of censored data (here we have almost half..)
# 
# - when I predicted the survival times on the test sets I choose not to cut the dataset depending on the source to try and spread the censored data more but it might be more relevant to keep the datasets separated and to create a new model for the second dataset that wouldn't be fitted on that much censored data (2/3 of the source dataset l2, might be better to keep only 1/3 of censored data). But that would mean diminishing the pool of subjects to a little amount which can bring biais in the prediction with lack of variance between the features.
# 
# - explore a bit more the random survival forest. I only had the time to implement a quick method to test this type of model but maybe going more into it could be interesting for the predictions.
# 
# - when getting documented on the subject of survival time prediction through litterature I read about accelarated failure time models which are based on the survival curve rather than the hazard function. It links the log-transformed survival time with the explanatory covariates. This can be an idea of a new model to test.
# 
# - when I selected the features to use for my model I checked the correlation between them and the targetted value but I believe the survival probability curve could also be traced for different categories in the features: for example plotting the survival probability curve for each of our four histology category could provide insight on the speed of the evolution of the risk within the dataset and may lead to changes in the so-thought main important features.
# 
# This has been my first real confrontation with a machine learning problem, my knowledge of the field was only academical up to this point (classes followed) with only small exercices on basic datasets I therefore took some time getting used to the models and wasn't able to do all of the study and exploration that I thought of. However it was very interesting and formative to dive into this medical machine learning problem which lead me to understand all the problems and difficulties that can arise from a dataset and how to find ways to work past them to make the best of it.
# 
# Working on this project gave me the will to continue to work on gathering knowledge and pratice over machine learning techniques to get more efficient and be able to extract insightful informations from data. 
