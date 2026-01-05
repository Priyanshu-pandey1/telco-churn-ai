# 📡 Telco Customer Churn Prediction AI

This project is a professional-grade Machine Learning application designed to predict customer attrition (churn) for a telecommunications provider. It features an optimized **XGBoost** backend and a user-friendly **Streamlit** web interface.

---

### 🚀 [Live Demo: Click Here to View the App]([https://telco-churn-ai.streamlit.app/](https://telco-churn-ai-gsxfbclqlfdn7k8lxctpm6.streamlit.app/)) 

---

### 📊 Model Performance & Results
This model was built with a focus on **Generalization** (working well on new data) rather than just memorizing the training set.

* **ROC-AUC Score:** **0.8419** (Indicates high ability to distinguish between churners and loyalists)
* **Test Accuracy:** **73.35%**
* **Generalization Gap:** **1.3%** (Extremely stable; reduced from an initial 16% gap)
* **Primary Churn Driver:** Identified as "Month-to-Month" contract types through Exploratory Data Analysis (EDA).

---

### 🛠️ Technical Implementation
* **Algorithm:** XGBoost (Extreme Gradient Boosting).
* **Optimization:** Implemented `scale_pos_weight=3` to handle class imbalance, ensuring the model focuses on detecting the minority churn class.
* **Overfitting Control:** Used **Early Stopping** and **Max Depth** constraints to ensure the model performs reliably on unseen data.
* **Deployment:** Model serialized via `Pickle` and deployed as a web app using `Streamlit`.

---

### 📂 Repository Structure
* `app.py`: The Python script for the Streamlit web interface.
* `churn_model.pkl`: The saved, trained XGBoost model "Brain."
* `encoder.pkl`: The LabelEncoder used to translate user input into model-readable data.
* `requirements.txt`: List of dependencies for cloud deployment.

---

### 💻 How to Run Locally
1. **Clone the repository:** `git clone https://github.com/Priyanshu-pandey1/telco-churn-ai.git`
2. **Install requirements:** `pip install -r requirements.txt`
3. **Run the application:** `streamlit run app.py`
