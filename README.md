# 📊 Student Performance Prediction System

A machine learning–powered application that predicts student academic performance using a **Logistic Regression** model. The system analyzes academic, demographic, and behavioral data to identify students at risk of underperforming, enabling educators to make proactive, data-driven decisions.

---

## ✅ Key Features

- **Logistic Regression–based prediction** for student performance
- **Interactive web interface** (Streamlit or similar)
- **CSV dataset upload** for batch prediction
- **Clean data preprocessing pipeline**
- **Feature engineering** for improved model accuracy
- **Visual analytics** including charts, class distribution & performance trends
- **Model evaluation metrics**: accuracy, precision, recall, F1-score & confusion matrix
- **Explainable outputs** to understand what influences student performance

---

## 🧠 Machine Learning Model

This system uses **Logistic Regression**, chosen for its:

- Interpretability  
- Fast training time  
- Strong performance with structured educational data  
- Ability to output probability-based predictions  

Training steps include:

- Data cleaning  
- Handling missing values  
- Label encoding  
- Normalization/standardization  
- Feature selection  
- Model training & evaluation  

---

## 🧰 Tech Stack

- **Python 3.9+**
- **Logistic Regression (scikit-learn)**
- **pandas, NumPy** for data processing
- **matplotlib / seaborn** for visual analytics
- **Streamlit** (if used for frontend UI)
- **Jupyter Notebook** for data exploration (optional)

---

# 🛠 Installation & Setup Instructions

Follow the steps below to run the system locally:

---

## 1️⃣ Clone the Repository
- git clone https://github.com/your-username/student-performance-prediction-system.git
- cd student-performance-prediction-system

## 2️⃣ Create and Activate a Virtual Environment
- python -m venv venv
- venv\Scripts\activate

## 3️⃣ Install Dependencies
- pip install -r requirements.txt

## 4️⃣ Run the Application
If using Streamlit:
- streamlit run app.py

Then open in your browser:
- http://localhost:8501

### 📊 Usage Instruction
- Launch the application
- Upload your student dataset (CSV)

The system preprocesses the data automatically
View:
- Performance predictions
- Probability scores
- Visual insights and distribution chart
