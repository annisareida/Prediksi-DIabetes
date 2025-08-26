# Diabetes Detection using Learning Vector Quantization (LVQ)

This project is a **Diabetes Classification System** using the **Learning Vector Quantization (LVQ)** algorithm.  
The dataset used is obtained from Kaggle: [Diabetes Dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset).  

The model classifies patients into **two classes**:  
- `1` → Diabetes  
- `0` → Not Diabetes  

---

## 📊 Dataset Description  

The dataset consists of several medical predictor variables and one target variable (`Outcome`).  

| Variable                   | Description                                          |
|----------------------------|------------------------------------------------------|
| **Pregnancies**            | Number of pregnancies                               |
| **Glucose**                | Glucose level in blood                              |
| **BloodPressure**          | Blood pressure measurement                          |
| **SkinThickness**          | Thickness of the skin                               |
| **Insulin**                | Insulin level in blood                              |
| **BMI**                    | Body mass index                                     |
| **DiabetesPedigreeFunction** | Diabetes percentage based on family history       |
| **Age**                    | Age of the individual                               |
| **Outcome**                | Final result (`1 = Diabetes`, `0 = Not Diabetes`)   |

---

## 🔄 Project Workflow  
1. Load the dataset from Kaggle  
2. Preprocess the data  
3. Train the **Learning Vector Quantization (LVQ)** model  
4. Classify whether a person has diabetes or not  

---

## 🚀 Streamlit App  

You can access the deployed project here:  
👉 [Diabetes Prediction App](https://prediksi-diabetes-universitas-sriwijaya.streamlit.app/)  

---

## 🛠️ How to Use  
1. Open the Streamlit app from the link above  
2. Input the required variables:  
   - Pregnancies  
   - Glucose  
   - BloodPressure  
   - SkinThickness  
   - Insulin  
   - BMI  
   - DiabetesPedigreeFunction  
   - Age  
3. Click **Predict**  
4. The program will classify the result as:  
   - **Diabetes** (`1`)  
   - **Not Diabetes** (`0`)  

---

## 👨‍🎓 Project Info  

This project was developed by students of the **Computer Engineering Department, Universitas Sriwijaya**,  
as part of a final project for the **Jaringan Syaraf Tiruan (Artificial Neural Networks)** course.  

---
