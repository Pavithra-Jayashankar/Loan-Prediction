import pandas as pd
import matplotlib.pyplot as plt

import os
os.chdir("D:\Imarticus\Assignments\Python Project - Bank Lending")
os.getcwd()

data = pd.read_csv("Banking.txt", encoding = 'utf-8', sep = '\t', low_memory=False)

pd.set_option("display.max_columns", None)
data.describe()

# checking for missing values

(data.isnull().sum(axis=0)*100/len(data)).sort_values(ascending=False)

data['default_ind'].value_counts()

#deleting columns more than 50% missing vale

half_count=len(data)/2
Data1 = data.dropna(thresh = half_count,axis=1)
print(Data1.isnull().sum())

#missing value imputation 

for col in Data1.columns:
    if(Data1[col].dtype == 'object'):
        temp_mode = Data1.loc[:,col].mode()[0]
        Data1[col] = Data1[col].fillna(temp_mode)
    else:
        temp_median = Data1.loc[:,col].median()
        Data1[col].fillna(temp_median, inplace = True)
        
        
Data1['pymnt_plan'].value_counts()

#We are removing pymnt_plan because it has only 5 yes and 855964 no
Data1.drop("pymnt_plan", axis=1, inplace= True)

## strip months from 'term' and make it an int

Data1['term'] = Data1['term'].str.split(' ').str[1]
Data1['term'].unique()

# extract numbers from emp_length and fill missing values with the median

Data1['emp_length'] = Data1['emp_length'].str.extract('(\d+)').astype(float)
Data1['emp_length'].unique()

#Univariate EDA

Data1['home_ownership'].value_counts()

import seaborn as sns
sns.distplot(Data1['installment'])
sns.distplot(Data1['last_pymnt_amnt'])
sns.distplot(Data1['loan_amnt'])
sns.distplot(Data1['out_prncp'])
sns.distplot(Data1['total_pymnt'])
sns.distplot(Data1['total_rec_int'])
sns.distplot(Data1['total_rec_prncp'])

#Bivariate EDA

#'loan_amnt' vs 'last_pymnt_amnt'

sns.set_style('whitegrid')  
sns.FacetGrid(Data1, size = 5).map(plt.scatter, 'loan_amnt', 'last_pymnt_amnt').add_legend()
            
#'total_rec_int', 'total_rec_prncp'

sns.FacetGrid(Data1, size = 5).map(plt.scatter, 'total_rec_int', 'total_rec_prncp').add_legend()

#'loan_amnt', 'last_pymnt_amnt'

sns.FacetGrid(Data1, hue = 'home_ownership', size = 5).map(plt.scatter, 'loan_amnt', 'last_pymnt_amnt').add_legend()

#'total_rec_int', 'total_rec_prncp'

sns.FacetGrid(Data1, hue = 'home_ownership', size = 5).map(plt.scatter, 'total_rec_int', 'total_rec_prncp').add_legend()

sns.countplot(x='default_ind',data=Data1, palette='hls')
plt.show()

#Categorical Data

#Grade

grade=Data1['grade'].value_counts()
plt.figure(figsize=(12,5))
plt.title('Grades of Loans')
plt.ylabel('Borrowers')
plt.xlabel('grade')
sns.barplot(x=grade.index, y=grade.values)

#home_ownership

home=Data1['home_ownership'].value_counts()
plt.figure(figsize=(12,5))
plt.title('Types of Home')
plt.xlabel('home_ownership')
plt.ylabel('Borrowers')
sns.barplot(x=home.index, y=home.values)

#purpose

purpose=Data1['purpose'].value_counts()
plt.figure(figsize=(12,5))
plt.title('Purpose of The Loan')
plt.ylabel('Borrowers')
plt.xlabel('purpose')
sns.barplot(x=purpose.index, y=purpose.values)

#verification_status

verifi_status= Data1['verification_status'].value_counts()
plt.figure(figsize=(12,5))
plt.title('Loan Verification Status')
plt.ylabel('Borrowers')
plt.xlabel('verification_status')
sns.barplot(x=verifi_status.index , y=verifi_status.values)

#default_ind

default=Data1['default_ind'].value_counts()
plt.figure(figsize=(12,5))
plt.title('Loan Status')
plt.ylabel('Borrowers')
plt.xlabel('default_ind')
sns.barplot(x=default.index, y= default.values)

#releationship status b/w 'Purpose' vs 'Default_ind'

sns.countplot(x='default_ind',data=Data1, palette='hls')
pd.crosstab(Data1.purpose,Data1.default_ind).plot(kind='bar')
plt.title('default for purpose Title')
plt.xlabel('purpose')
plt.ylabel('default_ind')
plt.savefig('purpose')

#'Emp_length' vs 'Default_ind'

pd.crosstab(Data1.emp_length,Data1.default_ind).plot(kind='bar')
plt.title('default for emp_length Title')
plt.xlabel('emp_length')
plt.ylabel('default_ind')
plt.savefig('emp_length')

#'Grade' vs 'Default_ind'

pd.crosstab(Data1.grade,Data1.default_ind).plot(kind='bar')
plt.title('default for grade Title')
plt.xlabel('grade')
plt.ylabel('default_ind')
plt.savefig('grade')

#purpose vs default_ind

pd.crosstab(Data1.purpose,Data1.default_ind).plot(kind='bar')
plt.title('Default for Purpose Title')
plt.xlabel('purpose')
plt.ylabel('default_ind')

#emp_length vs Default_ind

pd.crosstab(Data1.emp_length,Data1.default_ind).plot(kind='bar')
plt.title('Default for emp_length Title')
plt.xlabel('emp_length')
plt.ylabel('default_ind')

#int_rate 

plt.title('Distribution of Interest Rate')
sns.distplot(Data1['int_rate'], color='g')

#loan amount

plt.title('Distribution of Loan Amount')
sns.distplot(Data1['loan_amnt'], color='b')

#installment

plt.title('Distribution of Installment')
sns.distplot(Data1['installment'], color='r')

#Feature Engineering and Variable Selection

Data1['amt_difference'] = 'eq'
Data1.loc[ ( Data1['funded_amnt'] - Data1['funded_amnt_inv']) > 0, 'amt_difference' ] = 'less'

# Make categorical

Data1[ 'delinq_2yrs_cat' ] = 'no'
Data1.loc[ Data1 [ 'delinq_2yrs' ] > 0, 'delinq_2yrs_cat' ] = 'yes'

Data1[ 'inq_last_6mths_cat' ] = 'no'
Data1.loc[ Data1['inq_last_6mths' ] > 0, 'inq_last_6mths_cat' ] = 'yes'

Data1[ 'pub_rec_cat' ] = 'no'
Data1.loc[ Data1['pub_rec'] > 0,'pub_rec_cat' ] = 'yes'

# Create new metric
Data1['acc_ratio'] = Data1.open_acc / Data1.total_acc

# Getting all factor variables
factor_x = [col for col in Data1.columns.values if Data1[col].dtype == 'object']
    
# Getting all character variables
data_cat = Data1[factor_x]
data_cat.head()

# Getting all numeric variables
data_num = Data1.drop(factor_x,axis=1)
data_num.head()

# Correlation plot

correlation = data_num.corr()
plt.figure(figsize=(17,13))
sns.heatmap(correlation,xticklabels=correlation.columns,yticklabels=correlation.columns,annot=True, annot_kws={"size": 8})

features = ['loan_amnt', 'amt_difference', 'term','installment', 'grade','emp_length','home_ownership',\
            'annual_inc','verification_status','purpose', 'dti', 'delinq_2yrs_cat', 'inq_last_6mths_cat', \
            'open_acc', 'pub_rec', 'pub_rec_cat', 'acc_ratio', 'initial_list_status','issue_d','default_ind']

Data1 = Data1[features]

# Drop any residual missing values
Data1.dropna( axis=0, how = 'any', inplace = True )

Testset = ['Jun-2015', 'Jul-2015', 'Aug-2015', 'Sep-2015', 'Oct-2015', 'Nov-2015', 'Dec-2015']

trainset = Data1.loc [ -data.issue_d.isin(Testset) ]
Testset = Data1.loc [ data.issue_d.isin(Testset) ]

categorical_features = ['term','amt_difference', 'grade', 'home_ownership', 'verification_status',\
                'purpose', 'delinq_2yrs_cat', 'inq_last_6mths_cat', 'initial_list_status', 'pub_rec_cat']

#creating dummies

Train_x = pd.get_dummies(trainset[trainset.columns[:-2]], columns=categorical_features).astype(float)
Train_y = trainset['default_ind']

Test_x = pd.get_dummies(Testset[Testset.columns[:-2]], columns=categorical_features).astype(float)
Test_y = Testset['default_ind']

print(Train_x.shape, Test_x.shape)

Testset_var = ['home_ownership_OWN','home_ownership_RENT','purpose_educational']
Train_x.drop( Testset_var , axis = 1, inplace = True )
print(Train_x.shape)

#Preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X_scaled_model_train = preprocessing.scale(Train_x)
X_scaled_oot_test = preprocessing.scale(Test_x)

print(X_scaled_model_train.shape, X_scaled_oot_test.shape)

def run_models(X_train, y_train, X_test, y_test, model_type = 'Unbalanced'):
    
    clfs = {'LogisticRegression' : LogisticRegression(),
            'RandomForestClassifier': RandomForestClassifier(n_estimators=10)
            }
    cols = ['model', 'roc_auc_score', 'precision_score', 'recall_score','f1_score']

    models_report = pd.DataFrame(columns = cols)
    conf_matrix = dict()
    accuracy_score = dict()
    precision_score = dict()
    f1_score = dict()
    recall_score=dict()
    
    
    for clf, clf_name in zip(clfs.values(), clfs.keys()):

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_score = clf.predict_proba(X_test)[:,1]

        print('computing {} - {} '.format(clf_name, model_type))

        tmp = pd.Series({'model_type': model_type,
                         'model': clf_name,
                         'roc_auc_score' : metrics.roc_auc_score(y_test, y_score),
                         'precision_score': metrics.precision_score(y_test, y_pred),
                         'recall_score': metrics.recall_score(y_test, y_pred),
                         'f1_score': metrics.f1_score(y_test, y_pred)})

        models_report = models_report.append(tmp, ignore_index = True)
        conf_matrix[clf_name] = pd.crosstab(y_test, y_pred, rownames=['True'],
                   colnames= ['Predicted'], margins=False)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score, drop_intermediate = False, pos_label = 1)
        accuracy_score[clf_name] = metrics.accuracy_score(y_test, y_pred)
        precision_score[clf_name] = metrics.precision_score(y_test, y_pred)
        f1_score[clf_name] = metrics.f1_score(y_test, y_pred)
        recall_score[clf_name] = metrics.recall_score(y_test, y_pred)

        plt.figure(1, figsize=(6,6))
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title('ROC curve - {}'.format(model_type))
        plt.plot(fpr, tpr, label = clf_name )
        plt.legend(loc=2, prop={'size':11})
    plt.plot([0,1],[0,1], color = 'black')
    
    return models_report, conf_matrix, accuracy_score, precision_score, f1_score, recall_score

X_train, X_test, y_train, y_test = train_test_split(X_scaled_model_train, Train_y.values, test_size=0.4, random_state=0)

models_report, conf_matrix, accuracy_score, precision_score, f1_score, recall_score = run_models(X_train, y_train, X_test, y_test,model_type = 'Unbalanced')

models_report

conf_matrix['LogisticRegression']
conf_matrix['RandomForestClassifier']

accuracy_score['LogisticRegression']
accuracy_score['RandomForestClassifier']

precision_score['RandomForestClassifier']
precision_score['LogisticRegression']

f1_score['LogisticRegression']
f1_score['RandomForestClassifier']

recall_score['RandomForestClassifier']
recall_score['LogisticRegression']

from imblearn.over_sampling import SMOTE
index_split = int(len(X_scaled_model_train)/2)
X_train, y_train = SMOTE().fit_sample(X_scaled_model_train[0:index_split, :], Train_y[0:index_split])
X_test, y_test = X_scaled_model_train[index_split:], Train_y[index_split:]

models_report_bal, conf_matrix_bal, accuracy_score_bal, f1_score_bal, precision_score_bal, recall_score_bal  = run_models(X_train, y_train, X_test, y_test,\
                                                                     model_type = 'Balanced')

conf_matrix_bal['LogisticRegression']
conf_matrix_bal['RandomForestClassifier']

accuracy_score_bal['LogisticRegression']
accuracy_score_bal['RandomForestClassifier']

precision_score_bal['RandomForestClassifier']
precision_score_bal['LogisticRegression']

f1_score_bal['LogisticRegression']
f1_score_bal['RandomForestClassifier']

recall_score_bal['RandomForestClassifier']
recall_score_bal['LogisticRegression']
