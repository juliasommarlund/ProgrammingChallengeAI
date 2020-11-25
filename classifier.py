import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, plot_confusion_matrix

#Load data
df = pd.read_csv('TrainOnMe.csv')
df = pd.DataFrame(df)

df_test = pd.read_csv('EvaluateOnMe.csv')
df_test= pd.DataFrame(df_test)

#Make data numerical for test and training set
df.loc[df['x5'] == 'False', 'x5'] = 0
df.loc[df['x5'] == 'True', 'x5'] = 1
df.loc[df['x6'] == 'F', 'x6'] = 0
df.loc[df['x6'] == 'Fx', 'x6'] = 1
df.loc[df['x6'] == 'E', 'x6'] = 2
df.loc[df['x6'] == 'D', 'x6'] = 3
df.loc[df['x6'] == 'C', 'x6'] = 4
df.loc[df['x6'] == 'B', 'x6'] = 5
df.loc[df['x6'] == 'A', 'x6'] = 6

df_test.loc[df_test['x5'] == 'False', 'x5'] = 0
df_test.loc[df_test['x5'] == 'True', 'x5'] = 1
df_test.loc[df_test['x6'] == 'F', 'x6'] = 0
df_test.loc[df_test['x6'] == 'Fx', 'x6'] = 1
df_test.loc[df_test['x6'] == 'E', 'x6'] = 2
df_test.loc[df_test['x6'] == 'D', 'x6'] = 3
df_test.loc[df_test['x6'] == 'C', 'x6'] = 4
df_test.loc[df_test['x6'] == 'B', 'x6'] = 5
df_test.loc[df_test['x6'] == 'A', 'x6'] = 6

#Drop bad values in training data
X = df.drop(columns=['y','id'])
X = X.apply(pd.to_numeric, errors='coerce')
y = df['y']
df = pd.concat([y,X], axis =1, sort = False)
df = df.dropna()

#Drop first column of test data (index)
test = df_test.drop(df_test.columns[0], axis=1)

#Shuffle and finalize training data
df = shuffle(df)
X = df.drop(columns = ['y'])
y = df['y']

#Split training and validation data (when testing the classifier)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=101)

#Create classifier, parameters found by GridSearchCV (see below)
clf = GradientBoostingClassifier(n_estimators=48, learning_rate=0.15, max_depth=3)

#GridsearchCV
#paramgrid = {
#    'max_depth' : [1, 2, 3],
#    'learning_rate' : [0.05, 0.1, 0.13, 0.15, 0.17, 0.18, 0.2, 0.25],
#    'n_estimators' : [8, 16, 32, 48]
#}
#
#grid_search = GridSearchCV(estimator=clf, param_grid= paramgrid)
#grid_search.fit(X, y)
#clf.fit(X, y)
#print(grid_search.best_params_)
#

clf.fit(X, y)
preds = clf.predict(test)
print(preds)

file = open("52508.txt","w")
for pred in preds:
   file.write(pred + '\n')
file.close()

#Used to check accuracy while tuning the model
# print(classification_report(y_test, preds))
# titles_options = [("Confusion matrix, without normalization", None),
#               ("Normalized confusion matrix", 'true')]
# for title, normalize in titles_options:
#     disp = plot_confusion_matrix(clf, X_test, y_test,
#                                 cmap=plt.cm.Blues,
#                                 normalize=normalize)
#     disp.ax_.set_title(title)
# print(title)
# print(disp.confusion_matrix)
# matthews_corrcoef(y_test, preds)









