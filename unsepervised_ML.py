import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,silhouette_score
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
from sklearn.mixture import GaussianMixture


#Load Dataset

url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
#column names
columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
        "BMI", "DiabetesPedigree", "Age", "Outcome"]
#read the dataset
df=pd.read_csv(url, names=columns)
print(df.head())

#extract features and labels
X=df.drop(columns=['Outcome'],axis=1)
Y=df["Outcome"]

# Standardize the data
scaler=StandardScaler()
x_scaled = scaler.fit_transform(X)
# print(x_scaled[:5])

# PCA components

pca=PCA(n_components=0.95)
x_pca= pca.fit_transform(x_scaled)
print(f"Reduced features after PCA: {x_pca.shape[1]}")

# Split the data into training and testing sets

X_train,X_test,Y_train,Y_test=train_test_split(x_pca,Y,test_size=0.2,random_state=42)

# fit GMM with the different covariance types

best_gmm = None
best_score= -1
best_covariance = None
best_labels_train= None

for covariance_typ in ['spherical','diag','tied','full']:
    gmm=GaussianMixture(n_components=2, covariance_type=covariance_typ, random_state=42)
    labels_train= gmm.fit_predict(X_train)
    score=silhouette_score(X_train,labels_train)
    
    if score>best_score:
        best_score=score
        best_gmm=gmm
        best_covariance=covariance_typ
        best_labels_train=labels_train
#final prediction
final_labels=best_gmm.predict(X_test)

# print(f"X_test shape: {X_test.shape}")  # Should be (154, n_components)
# print(f"y_test shape: {Y_test.shape}")  # Should be (154,)
# print(f"final_labels shape: {final_labels.shape}")  # Should be (154,)

# Compute accuracy
def map_labels(predicted, actual):
    mapped_labels=np.zeros_like(predicted)
    actual=np.array(actual)
    print(actual)
    print(actual.shape)
    for i in range(2):
        mask = ((predicted==i)&(actual==0))
        print(f"Cluster {i}: {mask}")
        mapped_labels[mask]=1
        mask = (predicted==i)
        mapped_labels[mask]=0
        return mapped_labels
    
mapped_labels = map_labels(final_labels,Y_test)
accuracy=accuracy_score(Y_test, mapped_labels)
print(print(f"Improved Accuracy with GMM: {accuracy * 100:.2f}%"))

# Plot PCA with best GMM clustering
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:154,0],x_pca[:154,1],c=final_labels,cmap='viridis',alpha=0.6)
plt.xlabel(' PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA with GMM Clustering')
# plt.show()
plt.savefig("cluster_plot.png",dpi=300,bbox_inches='tight')

print(f'Best Covariance Type: {best_covariance}')
print(f'Best Silhouette Score: {best_score:.4f}')
print(f'Model Accuracy: {accuracy:.4f}')