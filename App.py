import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder 


st.write("""
# Application pour MaMaison.sn - Prédiction des prix de vente des maisons
Cette application prédit le prix des maisons à partir les caractéristiques entrés !!!""")

st.sidebar.header("Les Entrées des détails de la maison ci-dessous pour prédire son prix de vente.")

# saisie des données d'entrée
def user_input():
    Id = st.sidebar.slider("Id", 1, 1460, 100),
    MSSubClass = st.sidebar.slider("MSSubClass", 20, 190, 60),
    MSZoning = st.sidebar.slider('Zone', 0, 4, 2),
    LotFrontage = st.sidebar.slider('LotFrontage', 21.0, 313.0, 70.0),
    LotShape = st.sidebar.slider('LotShape', 0, 3, 2),
    LotArea = st.sidebar.slider('la superficie totale du terrain en pieds carrés', 1300, 215245, 10000),
    LotConfig = st.sidebar.slider('LotConfig', 0, 4, 2),
    Neighborhood = st.sidebar.slider('le quartier où se trouve la propriété.', 0, 24, 12),
    Condition1 = st.sidebar.slider('Condition1', 0, 8, 2),
    Condition2 = st.sidebar.slider('Condition2', 0, 7, 2),
    HouseStyle = st.sidebar.slider('HouseStyle', 0, 5, 2),
    OverallQual = st.sidebar.slider('Qualite globale.', 1, 10, 8),
    OverallCond = st.sidebar.slider("OverallCond", 1, 9, 5),
    YearBuilt = st.sidebar.slider('Année de construction', 1872, 2010, 2000),
    YearRemodAdd = st.sidebar.slider('Année de la dernière rénovation majeure de la maison.', 1950, 2010, 1999),
    RoofStyle = st.sidebar.slider('RoofStyle', 0, 7, 2),
    RoofMatl = st.sidebar.slider('RoofMatl', 0, 7, 2),
    Exterior1st = st.sidebar.slider("Exterior1st", 0, 14, 5),
    Exterior2nd = st.sidebar.slider("Exterior2nd", 0, 14, 5),
    MasVnrType = st.sidebar.slider('MasVnrType', 0, 3, 2),
    ExterQual = st.sidebar.slider("ExterQual", 0, 3, 2),
    ExterCond = st.sidebar.slider("ExterCond", 0, 4, 2),
    Foundation = st.sidebar.slider("Foundation", 0, 5, 2),
    BsmtQual = st.sidebar.slider("BsmtQual", 0, 3, 2),
    BsmtCond = st.sidebar.slider("BsmtCond", 0, 3, 2),
    BsmtExposure = st.sidebar.slider("BsmtExposure", 0, 3, 2),
    BsmtFinType1 = st.sidebar.slider("BsmtFinType1", 0, 5, 2),
    BsmtFinSF1 = st.sidebar.slider("BsmtFinSF1", 0, 5644, 1000),
    BsmtFinType2 = st.sidebar.slider("BsmtFinType2", 0, 5, 2),
    BsmtUnfSF = st.sidebar.slider("BsmtUnfSF", 0, 2336, 1000),
    TotalBsmtSF = st.sidebar.slider('la superficie totale du sous-sol en pieds carrés.', 0, 6110, 567),
    Heating = st.sidebar.slider("Heating", 0, 5, 2),
    CentralAir= st.sidebar.slider("CentralAir", 0, 1, 0),
    Electrical = st.sidebar.slider('Electrical.', 0, 4, 2),
    FirstFlrSF = st.sidebar.slider("1stFlrSF", 334, 4692, 1000),
    GrLivArea = st.sidebar.slider("GrLivArea", 334, 5642, 1500),
    FullBath = st.sidebar.slider('le nombre de salles de bains complètes.', 0, 4, 1),
    BedroomAbvGr = st.sidebar.slider('le nombre de chambres à coucher situées au-dessus du niveau du sol.', 0, 8, 3),
    KitchenAbvGr = st.sidebar.slider("KitchenAbvGr", 0, 3, 1),
    KitchenQual = st.sidebar.slider("KitchenQual", 0, 3, 1),
    TotRmsAbvGrd = st.sidebar.slider("TotRmsAbvGrd", 2, 15, 6),
    Functional = st.sidebar.slider("Functional", 0, 6, 3),
    Fireplaces = st.sidebar.slider("Fireplaces", 0, 3, 1),
    GarageType = st.sidebar.slider('GarageType', 0, 5, 2),
    GarageYrBlt = st.sidebar.slider('GarageYrBlt', 1900.0, 2010.0, 2000.0),
    GarageFinish = st.sidebar.slider('GarageFinish', 0, 2, 2),
    GarageCars = st.sidebar.slider('le nombre de voitures que le garage peut contenir.', 0, 4, 2),
    GarageArea = st.sidebar.slider('la superficie du garage en pieds carrés.', 0, 1488, 500),
    OpenPorchSF = st.sidebar.slider("OpenPorchSF", 0, 547, 100, 20),
    MoSold = st.sidebar.slider('MoVendu.', 1, 12, 8),
    SaleCondition = st.sidebar.slider('SaleCondition.', 0, 5, 3),
    YrSold = st.sidebar.slider('YrSold.', 2006, 2010, 2009),


    # Demande à l'utilisateur de saisir les caractéristiques de la maison sous forme de slidebar de streamlit avec la valeur min, max et par defaut 
    donne = { 
        "Id" : Id,
        "MSSubClass" : MSSubClass,
        "MSZoning" : MSZoning,
        "LotFrontage" : LotFrontage,
        "LotArea" : LotArea,
        "LotShape" : LotShape,
        "LotConfig" : LotConfig,
        "Neighborhood" : Neighborhood,
        "Condition1" : Condition1,
        "Condition2" : Condition2,
        "HouseStyle" : HouseStyle,
        "OverallQual" : OverallQual,
        "OverallCond" :OverallCond,
        "YearBuilt" : YearBuilt,
        "YearRemodAdd" : YearRemodAdd,
        "RoofStyle" : RoofStyle,
        "RoofMatl" : RoofMatl,
        "Exterior1st" : Exterior1st,
        "Exterior2nd" : Exterior2nd,
        "MasVnrType" : MasVnrType,
        "ExterQual" : ExterQual,
        "ExterCond" : ExterCond,
        "Foundation" : Foundation,
        "BsmtQual" : BsmtQual,
        "BsmtCond" : BsmtCond,
        "BsmtExposure" : BsmtExposure,
        "BsmtFinType1" : BsmtFinType1,
        "BsmtFinSF1" :BsmtFinSF1,
        "BsmtFinType2" : BsmtFinType2,
        "BsmtUnfSF" : BsmtUnfSF,
        "TotalBsmtSF" : TotalBsmtSF,
        "Heating" : Heating,
        "CentralAir" : CentralAir,
        "Electrical" : Electrical,
        "FirstFlrSF" : FirstFlrSF,
        "GrLivArea" : GrLivArea,
        "FullBath" : FullBath,
        "BedroomAbvGr" : BedroomAbvGr,
        "KitchenAbvGr" : KitchenAbvGr,
        "KitchenQual" : KitchenQual,
        "TotRmsAbvGrd" : TotRmsAbvGrd,
        "Functional" : Functional,
        "Fireplaces" : Fireplaces,
        "GarageType" : GarageType,
        "GarageYrBlt" : GarageYrBlt,
        "GarageFinish" : GarageFinish,
        "GarageCars" : GarageCars,
        "GarageArea" : GarageArea,      
        "OpenPorchSF" : OpenPorchSF,
        "MoSold" : MoSold,
        "YrSold" : YrSold,
        "SaleCondition" : SaleCondition,
    }
    parametre_maison = pd.DataFrame(donne, index=[0])
    return parametre_maison

para = user_input()
st.subheader("On veut trouver le prix de la maison")
st.write(para)

def predict_price (input):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import LabelEncoder 

    sample_data = pd.read_csv("C:\\Users\\pc\\OneDrive\\Bureau\\Projet\\sample_submission.csv")
    test_data = pd.read_csv("C:\\Users\\pc\\OneDrive\\Bureau\\Projet\\test.csv")
    train_data = pd.read_csv("C:\\Users\\pc\\OneDrive\\Bureau\\Projet\\train.csv")

    train_data.info()
    train_data = train_data.rename(columns={'1stFlrSF': 'FirstFlrSF'})
    test_data = test_data.rename(columns={'1stFlrSF': 'FirstFlrSF'})

    # Calculer le pourcentage de valeurs manquantes par colonne
    percent_missing = train_data.isnull().sum() * 100 / len(train_data)
    # Sélectionner les noms des colonnes avec un pourcentage de valeurs manquantes supérieur à 50%
    missing_features = percent_missing[percent_missing > 30].index
    # Supprimer ces colonnes du dataframe
    train_data.drop(missing_features, axis=1, inplace=True)
    train_data.drop(['SaleType','Street','LandContour','GarageQual','GarageCond','PavedDrive','SaleType'], axis=1, inplace=True)
    test_data.drop(['SaleType','Street','LandContour','GarageQual','GarageCond','PavedDrive','SaleType'], axis=1, inplace=True)
  

    # Remplace les valeurs manquantes par la moyenne
    train_data.fillna(train_data.mean(), inplace=True)
    test_data.fillna(test_data.mean(), inplace=True)

    # Remplacer les valeurs manquantes pour les colonnes de type "object"
    object_cols = train_data.select_dtypes(include='object').columns.tolist()
    train_data[object_cols] = train_data[object_cols].fillna(train_data[object_cols].mode().iloc[0])

    object_cols = test_data.select_dtypes(include='object').columns.tolist()
    test_data[object_cols] = test_data[object_cols].fillna(test_data[object_cols].mode().iloc[0])

     #Encodage des valeurs catégorielles
    label_encoder = LabelEncoder()
    for col in train_data.select_dtypes(include=['object']).columns:
        train_data[col] = label_encoder.fit_transform(train_data[col].astype(str))

    # Calculer le nombre de zéros dans chaque colonne
    zeros = train_data.eq(0).sum()

    # Sélectionner les colonnes ayant moins de 50% de zéros
    threshold = 0.5 * len(train_data)
    cols_to_keep = zeros[zeros < threshold].index

    # Supprimer les colonnes ayant plus de 50% de zéros
    train_data = train_data[cols_to_keep]


    # Définition des variables indépendantes et de la variable dépendante
    X = train_data.drop(['SalePrice'], axis=1)
    y = train_data['SalePrice']

    # Séparation des données en ensemble d'entraînement et ensemble de validation
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    # Création et entraînement du modèle de régression linéaire
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Prédiction sur l'ensemble de validation
    y_pred = linear_model.predict(X_test)


    # Utilisation du modèle pour prédire le prix de vente d'une maison
    prix_predit = linear_model.predict(para)
    print("Prix prédit:", prix_predit)


    # Calcul des métriques d'évaluation
    mea = metrics.mean_absolute_error(y_pred,y_test)
    mse = metrics.mean_squared_error(y_pred,y_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_pred, y_test)

    print('mae = ',mea)
    print('mse = ',mse)
    print('rmse = ',rmse)
    print("R2 Score : ", r2)

    return prix_predit


prix_predit = predict_price (para)
st.subheader(" Le prix de la maison est :")
st.write(prix_predit)

st.write("""  

         """)
train_data = pd.read_csv("C:\\Users\\hp\\OneDrive\\Documents\\Mes Cours\\Machine Learning L3\\Examen\\files\\train.csv")

#Afficher la distribution des valeurs dans la colonne 'SalePrice' du train_data
st.write(" #### Un histogramme de la fréquence par rapport au Prix des Maisons de MaMaison.sn ")

fig, ax = plt.subplots()
ax.hist(train_data['SalePrice'], bins=25)
ax.set_xlabel('SalePrice')
ax.set_ylabel('Frequency')
st.pyplot(fig)

st.write(" Graphique des Prix de Maisons ")

st.line_chart(train_data['SalePrice'])




