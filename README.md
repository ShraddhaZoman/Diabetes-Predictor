# Diabetes Prediction Application

This is a **full-stack diabetes prediction application** using **Flask** for the backend and **React** for the frontend. The application predicts diabetes using a **Random Forest model** and clusters patients using **K-Means clustering**.

## Features
- Predicts diabetes based on user input
- Uses **Random Forest** for classification
- Utilizes **K-Means clustering** to group patients
- Displays **model accuracy and cluster information**
- Secure API endpoints with **Flask & CORS**
- Frontend built with **React.js**

## Tech Stack
- **Frontend**: React.js
- **Backend**: Flask
- **Machine Learning**: Random Forest & K-Means (Scikit-learn)
- **Database**: CSV-based dataset
- **CORS Handling**: Flask-CORS

---

## Getting Started

### 1. Clone the Repository
```sh
git clone https://github.com/yourusername/diabetes-predictor.git
cd diabetes-predictor
```

### 2. Backend Setup (Flask)
#### Install Dependencies
```sh
cd backend
pip install -r requirements.txt
```
#### Run the Backend Server
```sh
python app.py
```

### 3. Frontend Setup (React)
#### Install Dependencies
```sh
cd frontend
npm install
```
#### Run the Frontend Server
```sh
npm start
```

---

## API Endpoints
- **Prediction**
  - `POST /predict` - Predict diabetes using user input
- **Home Route**
  - `GET /` - Welcome message for the API

### **Request Body Example (POST /predict)**
```json
{
  "pregnancies": 2,
  "glucose": 120,
  "bloodPressure": 70,
  "skinThickness": 20,
  "insulin": 85,
  "bmi": 25.5,
  "dpf": 0.5,
  "age": 30
}
```

### **Response Example**
```json
{
  "random_forest_prediction": "Positive",
  "cluster": 1,
  "random_forest_accuracy": "85.00%",
  "kmeans_silhouette_score": "0.65",
  "optimal_kmeans_clusters": 3,
  "best_silhouette_score": "0.72",
  "cluster_0_samples": [{"glucose": 110, "bmi": 23.4}],
  "cluster_1_samples": [{"glucose": 140, "bmi": 30.1}]
}
```

---

## Project Structure
```
diabetes-predictor/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── diabetes.csv
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   ├── package.json
│
└── README.md
```

## License
This project is open-source and available under the MIT License.

---

## Contributing
Feel free to fork this repository and make improvements. Pull requests are welcome!

---

### Contact
If you have any issues or questions, feel free to reach out!

