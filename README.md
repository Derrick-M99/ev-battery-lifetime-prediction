# Predicting the Lifetime of EV Lithium-Ion Batteries

This repository contains a machine learning project focused on predicting the capacity degradation and State of Health (SoH) of 18650 lithium-ion batteries, a critical component in modern Electric Vehicles (EVs). By analyzing time-series data from battery charge/discharge cycles, this project builds and compares several regression models to forecast battery lifetime.

## Project Workflow & Analysis

The project followed a structured data science workflow, from initial data exploration and pre-processing to final model evaluation. A full breakdown of the exploratory data analysis can be found in the `pre-processing` directory.

### Model Implementation

Several machine learning algorithms were implemented to predict the remaining battery capacity. The `models/` directory contains the individual scripts for each approach:

* **Random Forest**
* **XGBoost**
* **Ridge Regression**
* **Gaussian Naive Bayes**

## Dataset

**Note:** The dataset for this project is too large to be hosted directly on GitHub. You can download the required CSV files from the following link:

**[Download the Dataset Here](https://drive.google.com/drive/folders/13dQq7gDTUWzhzIm9KSEOtTJOWWURaaJH?usp=drive_link)**

After downloading, please place the CSV files inside a `data/` folder in the root of the project directory.

## Tech Stack

* **Language**: Python
* **Core Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Derrick-M99/ev-battery-lifetime-prediction.git
    ```

2.  **Download the data:**
    Use the link in the "Dataset" section above to download the data files. Create a folder named `data` in the project's root directory and place the CSV files inside it.

3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

The repository contains individual scripts for each model. To run a specific model (e.g., Random Forest):

```bash
python models/random_forest.py
```

## 📄 License

This project is licensed under the **MIT License**.
