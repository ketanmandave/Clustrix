Clustrix – Customer Segmentation using Machine Learning
Overview

Clustrix is a machine learning project that segments customers into different groups based on their purchasing behavior and demographic information. The goal is to help businesses better understand their customers and make data-driven marketing decisions.

Instead of treating every customer the same, this project identifies customers with similar characteristics using K-Means Clustering. It also reduces data dimensionality using Principal Component Analysis (PCA) to improve visualization and clustering performance.

Features
Customer data preprocessing and cleaning
Feature engineering
One-Hot Encoding for categorical features
Feature scaling using StandardScaler
Dimensionality reduction using PCA
Optimal cluster selection using the Elbow Method and Silhouette Score
Customer segmentation using K-Means Clustering
Interactive Streamlit application
Download segmented customer data
Project Structure
CUSTOMER-SEGMENTATION/

├── app/          # Streamlit application
├── data/         # Dataset
├── models/       # Saved ML models
├── outputs/      # Generated plots and outputs
├── src/          # Training and preprocessing scripts
├── README.md
└── requirements.txt
Tech Stack
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Streamlit
Joblib
Machine Learning Workflow
Load customer dataset
Clean missing values
Perform feature engineering
Remove outliers
Encode categorical variables
Scale numerical features
Apply PCA for dimensionality reduction
Find the optimal number of clusters
Train the K-Means model
Visualize customer segments
Save trained models
Predict customer segments through the Streamlit application
Results

The project successfully groups customers into meaningful clusters based on spending habits and demographic features.

The trained model can be used to:

Identify high-value customers
Discover customer purchasing patterns
Support targeted marketing campaigns
Improve customer relationship strategies
Streamlit Application

The project includes an interactive Streamlit application where users can:

Upload a customer dataset
Predict customer segments
View cluster distribution
Download the segmented dataset
How to Run
Clone the repository
git clone https://github.com/your-username/customer-segmentation.git
Move into the project directory
cd customer-segmentation
Install dependencies
pip install -r requirements.txt
Train the model
python src/train.py
Launch the Streamlit application
streamlit run app/app.py
Future Improvements
Compare additional clustering algorithms such as DBSCAN and Gaussian Mixture Models
Deploy the application online
Add interactive visualizations using Plotly
Automate model evaluation
Support larger datasets
Author

Ketan Mandave

B.Tech Artificial Intelligence & Machine Learning

Passionate about Machine Learning, Software Development, and solving real-world problems through technology.

If you have suggestions or feedback, feel free to connect or open an issue on this repository.