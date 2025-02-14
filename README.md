# **Disease Outbreak Prediction using Machine Learning**

This project focuses on predicting disease outbreaks using **machine learning** techniques. It leverages **historical patient data** to assess the risk of diseases such as **Heart Disease, Diabetes, and Parkinson's Disease**. The trained models provide quick and accurate health risk assessments, aiding in **early detection and preventive healthcare.**

---

## **Features**

- **Data Preprocessing**: Handles missing values, normalizes data, and applies feature scaling.
- **Machine Learning Models**: Uses **Random Forest, SVM, Neural Networks, and Gradient Boosting**.
- **Model Training & Evaluation**: Optimized models stored in `.sav` format for easy reuse.
- **User-Friendly Interface**: Jupyter notebooks for model exploration and an `app.py` for integration.
- **Scalability**: Can be expanded to support more diseases and real-time data integration.

---

## **Dataset**

The project utilizes medical datasets from the **AICTE Internship Repository**:

- **Heart Disease Dataset**: `heart.csv`
- **Diabetes Dataset**: `diabetes.csv`
- **Parkinson’s Disease Dataset**: `parkinsons.csv`

**Source:** [AICTE-Internship-files GitHub Repo](https://github.com/JayRathod341997/AICTE-Internship-files.git)

---

## **Installation**

### **Prerequisites**

Ensure you have **Python 3.8+** and the required dependencies installed:

```bash
pip install -r Requirements.txt
```

### **Project Setup**

1. **Clone the repository**:

   ```bash
   git clone https://github.com/MVENKATESH786/Disease-outbreak-prediction-using-Machine-Learning.git
   cd disease-outbreak-prediction
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   ```

3. **Run the application**:

   ```bash
   streamlit run app.py

   ```

---

## **Usage**

### **1. Load & Explore the Dataset**

Open Jupyter notebooks to analyze data and train models:

```bash
jupyter notebook Diabetes.ipynb
jupyter notebook Heart.ipynb
jupyter notebook Parkinsons.ipynb
```

### **2. Train the Models**

The pre-trained models are available in the `Saved_Models/` directory:

- `diabetes_model.sav`
- `heart_disease_model.sav`
- `parkinsons_model.sav`

### **3. Make Predictions**

You can use the trained models for predictions:

```python
import pickle
import pandas as pd

# Load the trained model
model_path = "Saved_Models/heart_disease_model.sav"  # Change for different diseases
model = pickle.load(open(model_path, "rb"))

# Load the scaler
scaler_path = "Saved_Models/scaler_heart.sav"  # Change accordingly
scaler = pickle.load(open(scaler_path, "rb"))

# Prepare new data for prediction
new_data = pd.read_csv("path/to/new_data.csv")  # Replace with actual file
new_data_scaled = scaler.transform(new_data)

# Make predictions
predictions = model.predict(new_data_scaled)
print(predictions)
```

### **4. Run the Application**

To integrate the models into a user-friendly interface:

```bash
streamlit run app.py

```

---

## **Evaluation Metrics**

The models are evaluated using the following metrics:

- **Accuracy**
- **Precision & Recall**
- **F1-Score**
- **AUC-ROC**

---

## **Future Enhancements**

- **Real-time data integration** from IoT devices & healthcare databases.
- **Mobile app deployment** for instant health risk assessments.
- **Blockchain-based secure data storage** for privacy and security.
- **Model generalization** to work across different populations and demographics.

---

## **Contributing**

Contributions are welcome! Follow these steps:

1. Fork the repository.
2. Create a new branch:

   ```bash
   git checkout -b feature-name
   ```

3. Make changes and commit:

   ```bash
   git commit -m "Description of changes"
   ```

4. Push to your branch:

   ```bash
   git push origin feature-name
   ```

5. Create a Pull Request.

---

## **Contact Information**

For any queries, feedback, or contributions, feel free to reach out:

- **Email**: <mvenkatesh0786@gmail.com>
