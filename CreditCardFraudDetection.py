#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import math

import datetime as dt
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

#Feature Selection

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier


#Hyperparameter tuning

from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn import metrics
import scipy.stats as stats
from scipy.stats import skew


# In[ ]:





# In[2]:


trainData = pd.read_csv("fraudTrain.csv", index_col=0)
pd.options.display.float_format = '{:,.2f}'.format


# In[3]:


trainData.head(5)


# In[4]:


trainData.shape


# In[5]:


msno.matrix(trainData)


# In[6]:


trainData.isna().sum()


# In[ ]:





# In[7]:


trainData.describe()


# In[8]:


donut = trainData["is_fraud"].value_counts().reset_index()

labels = ["No", "Yes"]
explode = (0, 0)

fig, ax = plt.subplots(dpi=120, figsize=(8, 4))
plt.pie(donut["is_fraud"],
        labels=donut["is_fraud"],
        autopct="%1.1f%%",
        pctdistance=0.8,
        explode=explode)

centre_circle = plt.Circle((0.0, 0.0), 0.5, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title("Fraud proportion in Transactions")
plt.legend(labels, loc="center", frameon=False)
plt.show();


# In[9]:


sns.kdeplot(trainData["amt"], fill=True);


# In[10]:


p99 = trainData["amt"].quantile(0.99)
sns.kdeplot(x="amt", data=trainData[trainData["amt"] <= p99], fill=True);


# In[10]:


p99 = trainData["amt"].quantile(0.99)
sns.histplot(x="amt", hue="is_fraud", bins=30,
             stat="probability", data=trainData[trainData["amt"] <= p99],
             common_norm=False);


# In[11]:


categories = trainData['category'].unique()

num_plots = len(categories)
num_rows = math.isqrt(num_plots)
num_cols = math.ceil(num_plots / num_rows)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(
    5*num_cols, 5*num_rows), sharex=True)

for i, category in enumerate(categories):

    row = i // num_cols
    col = i % num_cols

    data_category = trainData[trainData['category'] == category]

    if num_rows == 1 and num_cols == 1:
        ax = axes
    elif num_rows == 1 or num_cols == 1:
        ax = axes[i]
    else:
        ax = axes[row, col]

    sns.histplot(x='amt', data=data_category[data_category['amt'] <= p99],
                 hue='is_fraud', stat='probability',
                 common_norm=False, bins=30, ax=ax)

    ax.set_ylabel('Percentage in Each Type')
    ax.set_xlabel('Transaction Amount in USD')
    ax.set_title(f'{category}')
    ax.legend(title='Type', labels=['Fraud', 'Not Fraud'])

plt.tight_layout()

plt.show();


# In[13]:


non_fraud = trainData[trainData['is_fraud'] == 0]['category'].value_counts(
    normalize=True).to_frame().reset_index()
non_fraud.columns = ['category', 'not_fraud_percentual_vs_total']

# fraud
fraud = trainData[trainData['is_fraud'] == 1]['category'].value_counts(
    normalize=True).to_frame().reset_index()
fraud.columns = ['category', 'fraud_percentage_vs_total']

# merging two dataframes and calculating "fraud level"
non_fraud_vs_fraud = non_fraud.merge(fraud, on='category')
non_fraud_vs_fraud['fraud_level'] = non_fraud_vs_fraud['fraud_percentage_vs_total'] - \
    non_fraud_vs_fraud['not_fraud_percentual_vs_total']

non_fraud_vs_fraud


# In[14]:


custom_palette = sns.color_palette("flare")
ax = sns.barplot(y='category', x='fraud_level',
                 data=non_fraud_vs_fraud.sort_values('fraud_level', ascending=False), palette=custom_palette)
ax.set_xlabel('Percentage Difference')
ax.set_ylabel('Transaction Category')
plt.title('Fraud Level');


# In[15]:


trainData['age'] = dt.date.today().year-pd.to_datetime(trainData['dob']).dt.year
ax = sns.kdeplot(x='age', data=trainData, hue='is_fraud', common_norm=False)
ax.set_xlabel('Credit Card Holder Age')
ax.set_ylabel('Density')
plt.xticks(np.arange(0, 110, 10))
plt.title('Age Distribution')
plt.legend(title='Type', labels=['Fraud', 'Not Fraud']);


# In[16]:


trainData['hour'] = pd.to_datetime(trainData['trans_date_trans_time']).dt.hour
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
ax1 = sns.histplot(x='hour', data=trainData[trainData["is_fraud"] == 0],
                   stat="density", bins=24, ax=ax1)
ax2 = sns.histplot(x='hour', data=trainData[trainData["is_fraud"] == 1],
                   stat="density", bins=24, ax=ax2, color="blue")
ax1.set_title("Normal")
ax2.set_title("Fraud")
ax1.set_xticks(np.arange(1, 24))
ax2.set_xticks(np.arange(1, 24));


# In[17]:


trainData['month'] = pd.to_datetime(trainData['trans_date_trans_time']).dt.month
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
ax1 = sns.histplot(x='month', data=trainData[trainData["is_fraud"] == 0],
                   stat="density", bins=12, ax=ax1)
ax2 = sns.histplot(x='month', data=trainData[trainData["is_fraud"] == 1],
                   stat="density", bins=12, ax=ax2, color="blue")
ax1.set_title("Normal")
ax2.set_title("Fraud")
ax1.set_xticks(np.arange(1, 13))
ax2.set_xticks(np.arange(1, 13));


# In[18]:


count = trainData.city.unique().size
print("Unique values count : "+ str(count))


# In[19]:


non_fraud_city = trainData[trainData['is_fraud'] == 0]['city'].value_counts(
    normalize=True).to_frame().reset_index()
non_fraud_city.columns = ['city', 'not_fraud_percentual_vs_total']

# fraud
fraud_city = trainData[trainData['is_fraud'] == 1]['city'].value_counts(
    normalize=True).to_frame().reset_index()
fraud_city.columns = ['city', 'fraud_percentage_vs_total']

# merging two dataframes and calculating "fraud level"
non_fraud_vs_fraud_city = non_fraud_city.merge(fraud_city, on='city')
non_fraud_vs_fraud_city['fraud_level'] = non_fraud_vs_fraud_city['fraud_percentage_vs_total'] - \
    non_fraud_vs_fraud_city['not_fraud_percentual_vs_total']

non_fraud_vs_fraud_city


# In[20]:


trainData.drop(columns=[ "first", "last", "street",
           "unix_time", "trans_num"], inplace=True)


# In[21]:


trainData.head(2)


# In[22]:


trainData["amt_log"] = np.log1p(trainData["amt"])
sns.kdeplot(trainData["amt_log"], fill=True);


# In[23]:


def check_normality(feature):
    plt.figure(figsize=(8, 8))
    ax1 = plt.subplot(1, 1, 1)
    stats.probplot(trainData[feature], dist=stats.norm, plot=ax1)
    ax1.set_title(f'{feature} Q-Q plot', fontsize=20)
    sns.despine()

    mean = trainData[feature].mean()
    std = trainData[feature].std()
    skew = trainData[feature].skew()
    print(f'{feature} : mean: {mean:.2f}, std: {std:.2f}, skew: {skew:.2f}')


# In[24]:


check_normality("amt");


# In[25]:


check_normality("amt_log")


# In[27]:


def apply_woe(trainData, columns, target_col):
    woe = ce.WOEEncoder()

    for col in columns:
        X = trainData[col]
        y = trainData[target_col]

        new_col_name = f"{col}_WOE"
        trainData[new_col_name] = woe.fit_transform(X, y)

    return trainData


columns_to_encode = ["category", "state", "city", "job"]
target_column = "is_fraud"

trainData = apply_woe(trainData, columns_to_encode, target_column)


# In[28]:


gender_mapping = {"F": 0, "M": 1}

trainData["gender_binary"] = trainData["gender"].map(gender_mapping)


# In[29]:


freq_enc = (trainData.groupby("cc_num").size())
freq_enc.sort_values(ascending=True)
trainData["cc_num_frequency"] = trainData["cc_num"].apply(lambda x: freq_enc[x])


# In[30]:


sns.histplot(trainData["cc_num_frequency"], bins=6);


# In[31]:


intervals = [600, 1200, 1800, 2400, 3000, 3600]


def classify_frequency(freq):
    for i, c in enumerate(intervals):
        if freq <= c:
            return i


trainData["cc_num_frequency_classification"] = trainData["cc_num_frequency"].apply(
    classify_frequency)


# In[32]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
ax1 = sns.histplot(x='cc_num_frequency_classification', data=trainData[trainData["is_fraud"] == 0],
                   stat="density", bins=6, ax=ax1)
ax2 = sns.histplot(x='cc_num_frequency_classification', data=trainData[trainData["is_fraud"] == 1],
                   stat="density", bins=6, ax=ax2, color="blue")
ax1.set_title("Normal")
ax2.set_title("Fraud");


# In[61]:


X_train = trainData.drop(columns=["is_fraud"])
y_train= trainData["is_fraud"]


# In[62]:


X_train.drop(columns=["gender_binary", "zip", "long", "lat",
                      "city_pop", "month", "cc_num_frequency_classification",
                       "merch_long"], inplace=True)


# In[63]:


X_train.head()


# In[65]:


X_train = X_train.drop(columns=["trans_date_trans_time" , "merchant", "amt" ,
                                       "city", "state", "category", "gender", "dob", "job", "cc_num"])


# In[90]:


testData = pd.read_csv("fraudTest.csv", index_col=0)

testData['age'] = dt.date.today().year-pd.to_datetime(testData['dob']).dt.year
testData['hour'] = pd.to_datetime(testData['trans_date_trans_time']).dt.hour
testData['month'] = pd.to_datetime(testData['trans_date_trans_time']).dt.month

testData.drop(columns=["merchant", "first", "last", "street",
                   "unix_time", "trans_num"], inplace=True)


# In[67]:


testData.head(2)


# In[91]:


testData["amt_log"] = np.log1p(testData["amt"])


testData["gender_binary"] = testData["gender"].map(gender_mapping)
testData = apply_woe(testData, columns_to_encode, target_column)
freq_enc_test = (testData.groupby("cc_num").size())
freq_enc_test.sort_values(ascending=True)
testData["cc_num_frequency"] = testData["cc_num"].apply(lambda x: freq_enc_test[x])
testData["cc_num_frequency_classification"] = testData["cc_num_frequency"].apply(
    classify_frequency)


# In[92]:





# In[93]:


X_test = testData.drop(columns=["trans_date_trans_time", "city", "state", "category", "gender","amt", "dob", "job", "cc_num", "is_fraud"])


# In[94]:


y_test = testData["is_fraud"]
X_test.head(2)


# In[96]:


X_test.drop(columns=["gender_binary", "zip", "long", "lat",
                     "city_pop", "month", "cc_num_frequency_classification", "merch_long"], inplace=True)


# In[41]:


def evaluate_model(target, predicted, y_score, normalize_matrix= None):
    accuracy = metrics.accuracy_score(target, predicted)
    precision = metrics.precision_score(target, predicted)
    recall = metrics.recall_score(target, predicted)
    f1 = f1_score(target, predicted)
    auc = metrics.roc_auc_score(target, y_score)

    confusion_matrix = metrics.confusion_matrix(
        target, predicted, normalize=normalize_matrix)
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=[False, True])
    cm_display.plot()
    plt.grid(False)
    plt.show()

    fpr, tpr, threshold = roc_curve(target, y_score)
    plt.plot(fpr, tpr, label="Model", c="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", c="yellow")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
    
    print("Accuracy", accuracy.round(2))
    print("Precision:", precision.round(2))
    print("Recall:", recall.round(2))
    print("F1 Score", f1.round(2))
    print("AUC:", auc)
    return None


# In[42]:


rf = RandomForestClassifier(random_state=23)
knn = KNeighborsClassifier()
gboost = GradientBoostingClassifier(random_state=23)
#lgbm = LGBMClassifier(random_state=23)


# In[97]:


X_train.head(2)


# In[98]:


X_test.head(2)


# In[99]:


X_test = X_test[["merch_lat","age","hour","amt_log","category_WOE","state_WOE","city_WOE","job_WOE","cc_num_frequency"]]
X_test.head(2)


# In[100]:


rf.fit(X_train, y_train)

y_pred_train = rf.predict(X_train)
y_score_train = rf.predict_proba(X_train)[:,1]

y_pred_test = rf.predict(X_test)
y_score_test = rf.predict_proba(X_test)[:,1]


# In[101]:


#training metrics
evaluate_model(y_train, y_pred_train, y_score_train)


# In[102]:


evaluate_model(y_test, y_pred_test, y_score_test)


# In[103]:




