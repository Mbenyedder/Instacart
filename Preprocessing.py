from data import *
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

jointure1 = pd.merge (commande_df, df_commande_produit, on="order_id")
jointure2 = pd.merge (jointure1, df_produit, on="product_id")

Data = jointure2

Data.to_csv("C:/Users/hp/PycharmProjects/Python/ML_1/Kaggle/Instacart/Data/data.csv")

def Data_final ():

    L=[]

    for user in Data.groupby ("user_id") :
        user_id = user [0]
        user_data = user [1]
        seq_user=[]
        for order in user_data.groupby("order_id"):
            order_id = order[0]
            order_data = order[1]
            departement_ids = list(order_data["department_id"])
            seq_user.append(departement_ids)
        L.append(seq_user)

    return L

L= Data_final(Data)
X1=np.zeros((100,3,5))

for i in range(len(L)):
  panier =pad_sequences(L[i], maxlen=5, dtype="int32", padding="post", truncating='pre', value=0.0)
  panier = panier [:3]
  X1[i]=panier

X = X1[:,:-1,:]
Y = X1[:,-1,:]

x_train, x_test, y_train, y_test = train_test_split(X, Y)

for order in Data.groupby("user_id"):
    print(order)


for user in Data.groupby("user_id"):
    user_id = user[0]
    user_data = user[1]
    for order in user_data.groupby("order_id"):
        order_id = order[0]
        order_data = order[1]
        departement_ids = list(order_data["department_id"])
        print(user_id,order_id,departement_ids)