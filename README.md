# 🌍 Carbon Emission Forecasting & Trade Impact Analysis

> A capstone project that uses Machine Learning to forecast global CO₂ emissions and assess the impact of international trade on carbon output. Built using Python and Streamlit with real-world datasets.

---

## 📌 Project Overview

This project aims to:
- Forecast CO₂ emissions using ML models (Random Forest, ARIMA, SHAP-based interpretable models)
- Analyze trade-emission relationships using Input-Output models
- Perform sector-wise carbon contribution analysis using EDGAR dataset
- Simulate policies and visualize scenarios through an interactive dashboard

---

## 🧠 Key Features

- 📈 **Forecasting Models**: Trained and validated models for predicting future emissions
- 🌐 **Top Emitters Dashboard**: Year-wise and per capita comparison of CO₂ emitters
- 🔮 **Policy Simulation**: Predict impact of climate regulations and policy scenarios
- 📊 **Trade Emission Analysis**: Visualize how import/export activity affects emissions
- 🏭 **Sector-Wise Breakdown**: Based on EDGAR dataset (Power, Transport, Agriculture, etc.)
- 🎛 **Streamlit Dashboard**: Multi-tab interactive web app

---

## 📁 Project Structure
carbon-emission-analysis/
├── app.py # Streamlit dashboard application
├── filtered_emissions.csv # Cleaned emissions dataset (1930–2023)
├── sd01.csv # Trade statistics dataset
├── edgar.csv # Sector-wise CO₂ emissions dataset (EDGAR)
├── README.md # This file
├── requirements.txt # Required Python libraries


---

## 🔧 How to Run

### 1. Clone the Repository
git clone https://github.com/devsp2501/carbon-emission-analysis.git
cd carbon-emission-analysis

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Run the Dashboard
streamlit run app.py

---

## 🚀 Live Demo

Try the live interactive dashboard hosted on Streamlit Cloud or any cloud platform:

🔗 [Click here to view the live demo](https://carbon-emission-analysis-2ntcfq9gpbq3fquunrgdor.streamlit.app/)  
_(If not live, run locally using the steps above)_

---


