from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow CORS for all routes

# Load dataset
try:
    df = pd.read_csv('diabetes.csv')
except FileNotFoundError:
    print("Dataset not found. Ensure 'diabetes.csv' is in the correct directory.")
    exit()  # Exit if the dataset is not found

X = df.drop('Outcome', axis=1)  # Features
y = df['Outcome']  # Target variable

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

# Function to determine optimal number of K-means clusters
def optimal_kmeans_clusters(X_train):
    silhouette_scores = []
    cluster_range = range(2, 10)  # Testing different cluster sizes
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_train)
        silhouette_avg = silhouette_score(X_train, labels)
        silhouette_scores.append(silhouette_avg)

    # Select the cluster count with the highest silhouette score
    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    return optimal_clusters, silhouette_scores[np.argmax(silhouette_scores)]

# Determine the optimal number of clusters
optimal_n_clusters, best_silhouette_score = optimal_kmeans_clusters(X_train)

# Train K-means with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
kmeans.fit(X_train)

# Assign clusters to the entire dataset
df['Cluster'] = kmeans.predict(X)

# Make predictions on test data for evaluation
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

# Predict clusters for the test data
kmeans_labels = kmeans.predict(X_test)
kmeans_silhouette = silhouette_score(X_test, kmeans_labels)

@app.route('/')
def index():
    return "Welcome to the Diabetes Prediction API!"

# Function to extract samples based on the cluster label
def get_cluster_samples(cluster_label):
    return df[df['Cluster'] == cluster_label].head(2).to_dict(orient='records')

# Predict route that includes sample patients from each cluster
@app.route('/predict', methods=['POST'])
def predict():
    print("Received request at /predict")
    data = request.json

    # Print received data for debugging
    print("Received data:", data)

    # Ensure the request contains the expected data
    required_fields = ['pregnancies', 'glucose', 'bloodPressure', 'skinThickness', 'insulin', 'bmi', 'dpf', 'age']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400

    # Create DataFrame for prediction
    try:
        features = pd.DataFrame([list(data.values())], columns=X.columns)
        print("Features for prediction:", features)

        # Random Forest Prediction
        rf_prediction = rf_model.predict(features)[0]
        print("Random Forest Prediction:", rf_prediction)

        # K-means Clustering
        cluster = kmeans.predict(features)[0]
        print("K-means cluster:", cluster)

        # Get two sample patients from each cluster
        cluster_0_patients = get_cluster_samples(0)
        cluster_1_patients = get_cluster_samples(1)

        result = {
            'random_forest_prediction': 'Positive' if rf_prediction == 1 else 'Negative',
            'cluster': int(cluster),
            'random_forest_accuracy': f"{rf_accuracy * 100:.2f}%",  # Return Random Forest accuracy
            'kmeans_silhouette_score': f"{kmeans_silhouette:.2f}",   # Return K-Means Silhouette score
            'optimal_kmeans_clusters': optimal_n_clusters,           # Optimal number of K-means clusters
            'best_silhouette_score': f"{best_silhouette_score:.2f}", # Best Silhouette score for optimal clusters
            'cluster_0_samples': cluster_0_patients,                 # Sample patients from Cluster 0
            'cluster_1_samples': cluster_1_patients                  # Sample patients from Cluster 1
        }

        return jsonify(result)
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': 'Prediction failed', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
