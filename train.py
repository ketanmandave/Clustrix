
# Clustrix - Customer Segmentation
# Train Model




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA



# Load Dataset

df = pd.read_csv("dataset/smartcart_customers.csv")

print("=" * 50)
print("Dataset Loaded Successfully")
print("=" * 50)

print("\nDataset Shape :", df.shape)
print(df.head())


# Check Missing Values

print("\nMissing Values")
print(df.isnull().sum())


# DATA CLEANING
# Fill missing Income values with median

df["Income"] = df["Income"].fillna(df["Income"].median())

# FEATURE ENGINEERING

# Age
df["Age"] = 2026 - df["Year_Birth"]



# Customer Tenure
df["Dt_Customer"] = pd.to_datetime(
    df["Dt_Customer"],
    dayfirst=True
)

reference_date = df["Dt_Customer"].max()

df["Customer_Tenure_Days"] = (
    reference_date - df["Dt_Customer"]
).dt.days


# Total Spending
df["Total_Spending"] = (
    df["MntWines"]
    + df["MntFruits"]
    + df["MntMeatProducts"]
    + df["MntFishProducts"]
    + df["MntSweetProducts"]
    + df["MntGoldProds"]
)

# Total Children
df["Total_Children"] = (
    df["Kidhome"]
    + df["Teenhome"]
)

# Education
df["Education"] = df["Education"].replace({
    "Basic": "Undergraduate",
    "2n Cycle": "Undergraduate",
    "Graduation": "Graduate",
    "Master": "Postgraduate",
    "PhD": "Postgraduate"
})

# Living Status
df["Living_With"] = df["Marital_Status"].replace({
    "Married": "Partner",
    "Together": "Partner",
    "Single": "Alone",
    "Divorced": "Alone",
    "Widow": "Alone",
    "Absurd": "Alone",
    "YOLO": "Alone"
})

# Remove Unnecessary Columns
columns_to_drop = [
    "ID",
    "Year_Birth",
    "Marital_Status",
    "Kidhome",
    "Teenhome",
    "Dt_Customer",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]

df_cleaned = df.drop(columns=columns_to_drop)

print("\nDataset Shape After Cleaning :", df_cleaned.shape)

# OUTLIER REMOVAL

print("\nDataset Size Before Removing Outliers :", len(df_cleaned))

df_cleaned = df_cleaned[df_cleaned["Age"] < 90]
df_cleaned = df_cleaned[df_cleaned["Income"] < 600000]

print("Dataset Size After Removing Outliers :", len(df_cleaned))


# EXPLORATORY DATA ANALYSIS
# Pair Plot

eda_columns = [
    "Income",
    "Recency",
    "Response",
    "Age",
    "Total_Spending",
    "Total_Children"
]

sns.pairplot(df_cleaned[eda_columns])
plt.show()

# Correlation Heatmap

corr = df_cleaned.corr(numeric_only=True)

plt.figure(figsize=(10, 6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()

# ONE-HOT ENCODING

categorical_columns = [
    "Education",
    "Living_With"
]

encoder = OneHotEncoder()

encoded_features = encoder.fit_transform(
    df_cleaned[categorical_columns]
)

encoded_df = pd.DataFrame(
    encoded_features.toarray(),
    columns=encoder.get_feature_names_out(categorical_columns),
    index=df_cleaned.index
)

df_cleaned = df_cleaned.drop(columns=categorical_columns)

df_encoded = pd.concat(
    [df_cleaned, encoded_df],
    axis=1
)

print("\nEncoded Dataset Shape :", df_encoded.shape)

# FEATURE SCALING

X = df_encoded.copy()

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeature Scaling Completed")



# PCA


pca = PCA(n_components=3)

X_pca = pca.fit_transform(X_scaled)

print("\nExplained Variance Ratio")
print(pca.explained_variance_ratio_)


# PCA Visualization


fig = plt.figure(figsize=(10, 7))

ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    X_pca[:, 2]
)

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")

ax.set_title("3D PCA Projection")

plt.show()

# ELBOW METHOD


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import joblib
import os

print("\nFinding Optimal Number of Clusters...")

wcss = []

for k in range(1, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_pca)
    wcss.append(model.inertia())

# Find Elbow Point
knee = KneeLocator(
    range(1, 11),
    wcss,
    curve="convex",
    direction="decreasing"
)

optimal_k = knee.elbow

print(f"\nOptimal K : {optimal_k}")

# Plot Elbow Curve

plt.figure(figsize=(8,5))

plt.plot(
    range(1,11),
    wcss,
    marker="o"
)

if optimal_k is not None:
    plt.axvline(
        optimal_k,
        color="red",
        linestyle="--",
        label=f"Optimal K = {optimal_k}"
    )

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.legend()
plt.show()



# SILHOUETTE SCORE

scores = []

for k in range(2,11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_pca)

    score = silhouette_score(
        X_pca,
        labels
    )

    scores.append(score)

best_score = max(scores)
best_k = scores.index(best_score) + 2

print(f"Best Silhouette Score : {best_score:.4f}")
print(f"Best K (Silhouette) : {best_k}")

plt.figure(figsize=(8,5))

plt.plot(
    range(2,11),
    scores,
    marker="o",
    color="green"
)

plt.title("Silhouette Score")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()


# K-MEANS TRAINING


print("\nTraining Final KMeans Model...")

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

labels_kmeans = kmeans.fit_predict(X_pca)

print("Model Training Completed.")



# CLUSTER VISUALIZATION


fig = plt.figure(figsize=(10,7))

ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    X_pca[:,0],
    X_pca[:,1],
    X_pca[:,2],
    c=labels_kmeans,
    cmap="viridis",
    s=40
)

ax.set_title("Customer Segments")

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.show()

# CLUSTER SUMMARY


X["Cluster"] = labels_kmeans

print("\nCluster Count")

print(X["Cluster"].value_counts())

print("\nCluster Summary")

cluster_summary = X.groupby("Cluster").mean(numeric_only=True)

print(cluster_summary)


# Income vs Spending

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=X,
    x="Total_Spending",
    y="Income",
    hue="Cluster",
    palette="viridis"
)

plt.title("Income vs Spending by Cluster")

plt.show()



# SAVE MODELS

os.makedirs("model", exist_ok=True)

joblib.dump(
    encoder,
    "model/encoder.pkl"
)

joblib.dump(
    scaler,
    "model/scaler.pkl"
)

joblib.dump(
    pca,
    "model/pca.pkl"
)

joblib.dump(
    kmeans,
    "model/kmeans_model.pkl"
)

joblib.dump(
    X.columns.tolist(),
    "model/feature_columns.pkl"
)

print("\nModels Saved Successfully.")


# EXPORT CLUSTERED DATASET

df["Cluster"] = labels_kmeans

df.to_csv(
    "segmented_customers.csv",
    index=False
)

print("\nSegmented dataset exported successfully.")

