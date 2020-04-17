import pandas as pd
import numpy as np


#Produit
produit = pd.read_csv("C:/Users/hp/PycharmProjects/Python/ML_1/Kaggle/Instacart/Data/products.csv")
df_produit = produit[["department_id", "product_id"]]



df_commande = pd.read_csv("C:/Users/hp/PycharmProjects/Python/ML_1/Kaggle/Instacart/Data/orders.csv")


df_commande = df_commande [["order_id", "user_id"]]

N=1000

def commande(n_clients):
   return df_commande[df_commande['user_id'].isin(range(N+1))]

commande_df = commande(N)



commande_produit = pd.read_csv("C:/Users/hp/PycharmProjects/Python/ML_1/Kaggle/Instacart/Data/order_products__prior.csv")


df_commande_produit =commande_produit [['order_id', 'product_id']]

#print(df_commande_produit)


