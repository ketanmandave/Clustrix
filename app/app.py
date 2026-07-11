import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import zipfile

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🧠",
    layout="wide"
)

# ==============================
# PROFESSIONAL STYLING
# ==============================

st.markdown("""
<style>

/* Main App */
.stApp {
    background-color: #0f172a;
    color: white;
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* Headings */
h1 {
    color: white !important;
    font-size: 42px !important;
    font-weight: 700 !important;
}

h2, h3 {
    color: white !important;
    font-weight: 600 !important;
}

/* Paragraph text */
p {
    color: #cbd5e1;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1f2937;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

/* Metric label */
[data-testid="stMetricLabel"] {
    color: #94a3b8;
    font-size: 15px;
}

/* Metric value */
[data-testid="stMetricValue"] {
    color: white;
    font-size: 32px;
    font-weight: bold;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 15px;
}

/* Dataframe */
.stDataFrame {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #1f2937;
}

/* Download buttons */
.stDownloadButton button {
    width: 100%;
    background: #111827;
    color: white;
    border-radius: 12px;
    border: 1px solid #374151;
    padding: 12px;
    font-weight: 600;
    transition: 0.3s;
}

.stDownloadButton button:hover {
    background: #1e293b;
    border: 1px solid #2563eb;
}

/* Insight cards */
.insight-box {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 18px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Charts */
canvas {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# CONFIG
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ==============================
# LOAD MODELS
# ==============================

ohe = joblib.load(os.path.join(MODEL_DIR, "encoder.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
model = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))

# ==============================
# PREPROCESSING
# ==============================

def preprocess_data(df, ohe):

    df = df.copy()

    # Remove unwanted columns if present
    for col in ["cluster", "Cluster", "Segment"]:
        if col in df.columns:
            df = df.drop(columns=col)

    # Handle missing values
    df["Income"] = df["Income"].fillna(df["Income"].median())

    # Feature Engineering
    df["Age"] = 2026 - df["Year_Birth"]

    df["Dt_Customer"] = pd.to_datetime(
        df["Dt_Customer"],
        dayfirst=True
    )

    reference_date = df["Dt_Customer"].max()

    df["Customer_Tenure_Days"] = (
        reference_date - df["Dt_Customer"]
    ).dt.days

    df["Total_Spending"] = df[
        [
            "MntWines",
            "MntFruits",
            "MntMeatProducts",
            "MntFishProducts",
            "MntSweetProducts",
            "MntGoldProds"
        ]
    ].sum(axis=1)

    df["Total_Children"] = (
        df["Kidhome"] + df["Teenhome"]
    )

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
    drop_cols = [
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

    df = df.drop(columns=drop_cols)

    # Encoding
    cat_cols = ["Education", "Living_With"]

    encoded = ohe.transform(df[cat_cols])

    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()

    enc_df = pd.DataFrame(
        encoded,
        index=df.index
    )

    enc_df.columns = ohe.get_feature_names_out()

    enc_df = enc_df.reindex(
        columns=ohe.get_feature_names_out(),
        fill_value=0
    )

    df = df.drop(columns=cat_cols)

    df = pd.concat([df, enc_df], axis=1)

    return df

# ==============================
# LABELING
# ==============================

def label_cluster(c):

    if c == 0:
        return "High Value"

    elif c == 1:
        return "Budget"

    elif c == 2:
        return "Frequent"

    else:
        return "Inactive"

# ==============================
# INSIGHTS ENGINE
# ==============================

def get_segment_insight(segment):

    insights = {

        "High Value": {
            "description": "High income and premium spending customers",
            "action": "Target with loyalty rewards and premium offers"
        },

        "Budget": {
            "description": "Price-sensitive customers with lower spending",
            "action": "Provide discounts and bundle offers"
        },

        "Frequent": {
            "description": "Customers with consistent purchasing behavior",
            "action": "Upsell and cross-sell products"
        },

        "Inactive": {
            "description": "Low engagement customers",
            "action": "Run re-engagement campaigns"
        }
    }

    return insights.get(
        segment,
        {
            "description": "Unknown",
            "action": "Analyze manually"
        }
    )

# ==============================
# HEADER
# ==============================

st.markdown("""
<h1>🧠 Customer Segmentation System</h1>

<p style='font-size:18px; color:#94a3b8;'>
Upload customer data to segment users and generate business insights
</p>
""", unsafe_allow_html=True)

# ==============================
# FILE UPLOADER
# ==============================

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# ==============================
# MAIN APP
# ==============================

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Raw Data")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    try:

        # Process Data
        X = preprocess_data(df, ohe)

        X_scaled = scaler.transform(X)

        X_pca = pca.transform(X_scaled)

        clusters = model.predict(X_pca)

        result_df = df.loc[X.index].copy()

        result_df["Cluster"] = clusters

        result_df["Segment"] = result_df["Cluster"].apply(
            label_cluster
        )

        # ==============================
        # METRICS
        # ==============================

        st.subheader("📈 Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Customers",
            len(result_df)
        )

        col2.metric(
            "Segments",
            result_df["Segment"].nunique()
        )

        col3.metric(
            "Top Segment",
            result_df["Segment"].value_counts().idxmax()
        )

        # ==============================
        # SEGMENTED DATA
        # ==============================

        st.subheader("📊 Segmented Data")

        st.dataframe(
            result_df.head(),
            use_container_width=True
        )

        # ==============================
        # DISTRIBUTION CHART
        # ==============================

        st.subheader("📊 Segment Distribution")

        st.bar_chart(
            result_df["Segment"].value_counts()
        )

        # ==============================
        # BUSINESS INSIGHTS
        # ==============================

        st.subheader("🧠 Business Insights")

        for seg, count in result_df["Segment"].value_counts().items():

            info = get_segment_insight(seg)

            st.markdown(f"""
            <div class="insight-box">

            <h3>🔹 {seg} Customers ({count})</h3>

            <p><b>Description:</b> {info['description']}</p>

            <p><b>Recommended Action:</b> {info['action']}</p>

            </div>
            """, unsafe_allow_html=True)

        # ==============================
        # DOWNLOADS
        # ==============================

        st.subheader("⬇ Download Data")

        # Full CSV Download
        full_csv = result_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Full Data",
            full_csv,
            "segmented_customers.csv"
        )

        # Segment-wise Downloads
        for seg in result_df["Segment"].unique():

            seg_df = result_df[
                result_df["Segment"] == seg
            ]

            csv = seg_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                f"Download {seg}",
                csv,
                f"{seg.replace(' ', '_')}.csv"
            )

        # ZIP Download
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w") as z:

            for seg in result_df["Segment"].unique():

                seg_df = result_df[
                    result_df["Segment"] == seg
                ]

                z.writestr(
                    f"{seg}.csv",
                    seg_df.to_csv(index=False)
                )

        st.download_button(
            "Download All Segments (ZIP)",
            zip_buffer.getvalue(),
            "segments.zip"
        )

    except Exception as e:

        st.error(f"Error: {e}")