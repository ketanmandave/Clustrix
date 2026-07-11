import os
import pandas as pd
import numpy as np
import joblib
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, 'models')

ohe = joblib.load(os.path.join(MODEL_DIR, 'encoder.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
pca = joblib.load(os.path.join(MODEL_DIR, 'pca.pkl'))
model = joblib.load(os.path.join(MODEL_DIR, 'kmeans_model.pkl'))

# ==============================
# Preprocessing Function (SAME AS TRAINING)
# ==============================

# Remove columns not used in training
drop_extra = ["cluster", "Cluster", "Segment"]

for col in drop_extra:
    if col in df.columns:
        df = df.drop(columns=col)

def preprocess_data(df, ohe):
    
    df = df.copy()

    # Missing values
    df['Income'] = df['Income'].fillna(df['Income'].median())

    # Feature Engineering
    df["Age"] = 2026 - df["Year_Birth"]
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)

    reference_date = df["Dt_Customer"].max()
    df["Customer_Tenure_Days"] = (reference_date - df["Dt_Customer"]).dt.days

    df["Total_Spending"] = df[
        ["MntWines","MntFruits","MntMeatProducts",
         "MntFishProducts","MntSweetProducts","MntGoldProds"]
    ].sum(axis=1)

    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

    # Category cleaning
    df["Education"] = df["Education"].replace({
        "Basic": "Undergraduate",
        "2n Cycle": "Undergraduate",
        "Graduation": "Graduate",
        "Master": "Postgraduate",
        "PhD": "Postgraduate"
    })

    df["Living_With"] = df["Marital_Status"].replace({
        "Married": "Partner",
        "Together": "Partner",
        "Single": "Alone",
        "Divorced": "Alone",
        "Widow": "Alone",
        "Absurd": "Alone",
        "YOLO": "Alone"
    })

    # Drop columns
    drop_cols = ["ID","Year_Birth","Marital_Status","Kidhome","Teenhome","Dt_Customer",
                 "MntWines","MntFruits","MntMeatProducts","MntFishProducts",
                 "MntSweetProducts","MntGoldProds"]

    df = df.drop(columns=drop_cols)

    # Outlier filtering (optional in prediction, but keeping consistent)
    df = df[df["Age"] < 90]
    df = df[df["Income"] < 600000]

    # Encoding
    cat_cols = ["Education", "Living_With"]
    encoded = ohe.transform(df[cat_cols])

    # Convert to dense if sparse
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    
    # Create DataFrame WITHOUT forcing columns first
    enc_df = pd.DataFrame(encoded, index=df.index)
    
    # Assign correct column names AFTER creation
    enc_df.columns = ohe.get_feature_names_out()
    
    # Ensure same structure as training
    expected_cols = ohe.get_feature_names_out()
    enc_df = enc_df.reindex(columns=expected_cols, fill_value=0)
    df = df.drop(columns=cat_cols)
    df = pd.concat([df, enc_df], axis=1)
    
    return df

# ==============================
# Segment Labeling
# ==============================

def label_cluster(cluster):
    if cluster == 0:
        return "High Value"
    elif cluster == 1:
        return "Budget"
    elif cluster == 2:
        return "Frequent"
    else:
        return "Inactive"

# ==============================
# MAIN FUNCTION
# ==============================

def main(input_file, output_file):

    # Load new data
    df = pd.read_csv(input_file)

    # Keep original for output
    original_df = df.copy()

    # Preprocess
    X = preprocess_data(df, ohe)

    # Scale
    X_scaled = scaler.transform(X)

    # PCA
    X_pca = pca.transform(X_scaled)

    # Predict clusters
    clusters = model.predict(X_pca)

    # Add results
    original_df = original_df.loc[X.index]  # align after outlier removal
    original_df['Cluster'] = clusters
    original_df['Segment'] = original_df['Cluster'].apply(label_cluster)

    # Save output
    original_df.to_csv(output_file, index=False)

    print(f"✅ Segmentation complete. Saved to {output_file}")


# ==============================
# RUN FROM TERMINAL
# ==============================

if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("Usage: python predict.py input.csv output.csv")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        main(input_file, output_file)