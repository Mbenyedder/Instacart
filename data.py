import pandas as pd
import numpy as np


#Produit
produit = pd.read_csv("C:/Users/hp/PycharmProjects/Python/ML_1/Kaggle/Instacart/Data/products.csv")
df_produit = produit[["department_id", "product_id"]]


#Lecture des données

df_commande = pd.read_csv("C:/Users/hp/PycharmProjects/Python/ML_1/Kaggle/Instacart/Data/orders.csv")


df_commande = df_commande [["order_id", "user_id"]]

N=100  #On commence à tester pour 100 clients

def commande(n_clients):
   return df_commande[df_commande['user_id'].isin(range(N+1))]

commande_df = commande(N)


#Order products prior
#Lecture des données + on garde les données qui nous interesse

commande_produit = pd.read_csv("C:/Users/hp/PycharmProjects/Python/ML_1/Kaggle/Instacart/Data/order_products__prior.csv")

#print(commande_produit .columns)

df_commande_produit =commande_produit [['order_id', 'product_id']]

#print(df_commande_produit)


