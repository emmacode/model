
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv(r"/Users/moyin/Documents/school/project/Data/NJN/train.csv")

print(df.describe().T)  
print(df.isnull().sum())


#Rename Dataset
df = df.rename(columns={'class':'Label'})
print(df.dtypes)

#Understand the data 
sns.countplot(x="Label", data=df)


####### Replace categorical values with numbers########
df['Label'].value_counts()

#Define the dependent variable that needs to be predicted (labels)
y = df["Label"].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(y) 
###########################################################################
#Define x and normalize values
X = df.drop(labels = ["Label", "ID"], axis=1) 


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

#Split data_ 80% for training and 20% for testing 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#=========================================================================
########################=== KNN ===########################################
###########################################################################

#model = KNeighborsClassifier()

#=========================================================================
####################=== RANDOM FOREST ===##################################
###########################################################################
#from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(n_estimators = 10, random_state = 1)


#===========================================================================
###################=== XGBOOST ===########################################
##########################################################################
import xgboost as xgb
model = xgb.XGBClassifier()

#========================================================================

## Train the model on training data
model.fit(X_train, y_train)


# Predicting the Test set results
y_pred = model.predict(X_test)


#Evaluate the classifier on test data
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

accuracy = accuracy_score(y_test, y_pred)


print("Accuracy = ", (accuracy * 100.0), "%")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize = (10,7),dpi=300)
sns.heatmap(cm, annot=True,annot_kws={"size": 17},fmt=".0f",cmap='Blues_r') 
#cmap='RdGy' 
#cmap='Blues'
#cmap = "Greens"


