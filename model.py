import pandas as pd
import re
import joblib
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

## Import data ##
data_features = pd.read_csv('bag_of_words.csv')
data_tags = pd.read_csv('data_tags_clean.csv')
data0 = pd.concat((data_features, data_tags.rename(
columns={'1st_tag': 'tag'})['tag']), axis=1)

## Keeping only relevant tags ##
data = data0.copy()

data.loc[data.tag == 'python-3.x', 'tag'] = 'python'
data.loc[data.tag == 'mysql', 'tag'] = 'sql'

main_tags = data[['tag']].value_counts()[
data[['tag']].value_counts()>1200].index

main_tags_clean = []
for i in range(len(main_tags)):
    main_tags_clean.append(re.sub(r'[(),"\']',"", str(main_tags[i])))

data_main = data[data['tag'].isin(main_tags_clean)].reset_index().drop(
columns = ['level_0'])


## Preprocessing the data ##
scaler = preprocessing.StandardScaler()
label_encoder = preprocessing.LabelEncoder()

X = data_main.drop(columns = ['tag'])

#X = scaler.fit_transform(data_wo_tags)
y = label_encoder.fit_transform(data_main['tag'])

## Learning ##
#clf = GradientBoostingClassifier(n_estimators = 300)
gb = GradientBoostingClassifier(n_estimators = 300)
clf = Pipeline([('scaler', scaler), ('gradboost', gb)])
clf.fit(X, y)

## Saving the model ##
joblib.dump(label_encoder, "le.pkl")
joblib.dump(clf, "clf.pkl")
