import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

#read data 
df = pd.read_csv('winequality-red.csv')

#get info about dataset 
df.info()

#first 5 row from dataset
df.head(5)

plt.bar(df['quality'], df['alcohol'])
plt.title('Relationship between alcohol and quality')
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.legend()
plt.show()

sclr = MinMaxScaler(feature_range=(0,1))
normal_df = sclr.fit_transform(df)
normal_df = pd.DataFrame(normal_df, columns=df.columns)
normal_df.head()

df["good wine"] = ["yes" if i >=7 else "no" for i in df ['quality']]

X = normal_df.drop(['quality'], axis = 1)
Y = df["good wine"]

Y.value_counts()

#visualize the count 
sns.countplot(Y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=42, stratify = Y)

dummy_classifier = DummyClassifier(strategy='most_frequent',random_state=42)
dummy_classifier.fit(X_train,y_train)
acc_baseline = dummy_classifier.score(X_test,y_test)
print("Baseline Accuracy = ", acc_baseline)

svc = SVC(random_state=42)
svc.fit(X_train,y_train)

#predict the outcomes for the test set and print its accuracy score.
predic = svc.predict(X_test)

random_grid = {"C": [0.001,0.01,0.1,1,10,100,1000]}
svc_random = RandomizedSearchCV(svc,random_grid,cv=5,random_state=42)
svc_random.fit(X_train, y_train)
print(svc_random.best_params_)

param_dist = {'C': [0.8,0.9,1,1.1,1.2,1.3,1.4],
              'kernel':['linear', 'rbf','poly']}
svc_cv = GridSearchCV(svc, param_dist, cv=10)
svc_cv.fit(X_train,y_train)
print(svc_cv.best_params_)

svc_new = SVC(C = 1.2, kernel = "rbf", random_state=42)
svc_new.fit(X_train, y_train)
y_pred_new = svc_new.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred_new))