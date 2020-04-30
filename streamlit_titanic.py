# %%
import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

st.title("Prévoit la survie d'une personne")

st.subheader(" pendant le naufrage de Titanic selon des critères")


def age_missing_replace(means, dframe, title_list):
    for title in title_list:
        temp = dframe['Title'] == title 
        dframe.loc[temp, 'Age'] = dframe.loc[temp, 'Age'].fillna(means[title]) 

# @st.cache
def assignDeckValue(CabinCode):
    if pd.isnull(CabinCode):
        category = 'Unknow'
    else:
        category = CabinCode[0]
    return category

DATA_URL = "train.csv"

@st.cache
def load_data():
    data = pd.read_csv(DATA_URL)
    data.drop(['Ticket', 'PassengerId'], 1 ,inplace=True )
    DeckInitial = data['Cabin'].unique()
    Deck = np.array([assignDeckValue(cabin) for cabin in data.Cabin.values])
    data = data.assign(Deck = Deck)
    data['Title'] = data.Name.str.extract('([A-Za-z]+)\.', expand=False)
    data['Title'] = data['Title'].replace(['Dr', 'Rev', 'Col','Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')

    common ='S' 
    data['Embarked'] = data['Embarked'].fillna('S')

    means = data.groupby('Title')['Age'].mean()

    title_list = ['Master','Miss','Mr','Mrs','Others']

    age_missing_replace(means, data, title_list)

    data['Embarked'] = data['Embarked'].map({'C':0, 'Q':1, 'S':2})
    data['Sex'] = data['Sex'].map({'male':0, 'female':1})
    data['Title'] = data['Title'].map({'Master':0,'Miss':1,'Mr':2,'Mrs':3,'Others':4})

    le = preprocessing.LabelEncoder()
    data['Deck'] = le.fit_transform(data['Deck'])
    return data

# @st.cache
# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     # lowercase = lambda x: str(x).lower()
#     # data.rename(lowercase, axis='columns', inplace=True)
#     # data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data

data_load_state = st.text('Loading data...')
training_set = load_data()
data_load_state.text('Loading data...done!')


sex = st.sidebar.radio("Quelle est le sexe de la personne", ("Homme", "Femme"))

min_classe = min(training_set['Pclass'])

max_classe = max(training_set['Pclass'])

classe  = st.sidebar.slider("La classe dans le Titanic", round(min_classe), round(max_classe) )

min_age = min(training_set['Age'])

max_age = max(training_set['Age'])

age = st.sidebar.slider("L'âge de la personne", round(min_age), round(max_age) )

min_montant_billet = min(training_set['Fare'])

max_montant_billet = max(training_set['Fare'])

montant_billet = st.sidebar.slider("Le montant de billet", round(min_montant_billet), round(max_montant_billet) )

min_nb_frere_soeur = min(training_set['SibSp'])

max_nb_frere_soeur = max(training_set['SibSp'])

montant_nb_frere_soeur = st.sidebar.slider("Le nombre de frères et soeurs", round(min_nb_frere_soeur), round(max_nb_frere_soeur) )

min_nb_enfant = min(training_set['Parch'])

max_nb_enfant = max(training_set['Parch'])

nb_enfant = st.sidebar.slider("Le nombre d'enfants", round(min_nb_enfant), round(max_nb_enfant) )

porte_embarquement = st.sidebar.selectbox("La porte que la personne a embarqué ",['C','Q','S'])

dicto = {'C':0, 'Q':1, 'S':2}

rf = RandomForestClassifier()

X = training_set[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

Y = training_set['Survived']


st.warning("Le modèle d'apprentissage utilisé est Random Tree Forest")
if sex == 'Homme':
	st.write("La personne est un homme.")
	sex_encoder = 0
else:
	st.write("La personne est une femme.")
	sex_encoder = 1

st.write("La classe de la personne est la class {}.".format(classe))

st.write("La personne a {} ans.".format(age))

st.write("La personne a payé {}￡ pour son billet.".format(montant_billet))

st.write("La personne a  {} frères et soeurs.".format(montant_nb_frere_soeur))

st.write("La personne a  {} enfant.".format(nb_enfant))

st.write("La porte que la personne a embarqué est la porte {}.".format(porte_embarquement))

personne = [[classe, sex_encoder, age, montant_nb_frere_soeur, nb_enfant,montant_billet,dicto[porte_embarquement]]]

rf.fit(X,Y)

survived = rf.predict(personne)

proba = rf.predict_proba(personne)

if survived[0] == 1:
	st.success("La personne  a {}% d'être survit.".format(round(np.mean(proba)*100,2)))
else:
	st.error("La personne a {}% d'être morte.".format(round(np.mean(proba)*100,2)))