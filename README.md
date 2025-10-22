# Customer Churn Prediction ML Project

## Overview

This project implements a machine learning solution to predict customer churn for a telecommunications company. Customer churn prediction helps businesses identify customers who are likely to discontinue their services, enabling proactive retention strategies and reducing revenue loss.

Using the Telco Customer Churn dataset, this project explores various machine learning algorithms to build accurate predictive models. The solution includes comprehensive exploratory data analysis (EDA), data preprocessing, model training, evaluation, and a user-friendly Streamlit web application for real-time predictions.

## Contents

The repository is organized as follows:

- **EDA.ipynb**: Jupyter notebook containing exploratory data analysis with visualizations and statistical insights into customer behavior patterns and churn indicators
- **Telco-Customer-Churn.csv**: The primary dataset containing customer demographics, account information, and service usage details
- **preprocessing.py**: Python module with data preprocessing functions including feature engineering, encoding, scaling, and data cleaning utilities
- **streamlit_app.py**: Interactive web application built with Streamlit for making real-time churn predictions with a user-friendly interface
- **models/**: Directory containing trained machine learning models saved in pickle format for deployment and inference

## Usage Instructions

### Prerequisites

Ensure you have Python 3.7+ installed along with the following packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit jupyter
```

### Running the Exploratory Data Analysis

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open `EDA.ipynb` and run the cells sequentially to explore the dataset and understand churn patterns

### Training Models

1. Run the preprocessing script to prepare the data:
   ```bash
   python preprocessing.py
   ```

2. Train models using the notebook or your custom training script

3. Trained models will be saved in the `models/` directory

### Running the Streamlit Application

Launch the interactive web application for making predictions:

```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser where you can:
- Input customer information through an intuitive form
- Get instant churn predictions
- View probability scores and model confidence
- Analyze feature importance for individual predictions

## Dataset Description

The **Telco Customer Churn** dataset contains information about:

- **Customer Demographics**: Gender, age, partners, dependents
- **Account Information**: Contract type, payment method, tenure, monthly charges, total charges
- **Service Usage**: Phone service, internet service, online security, tech support, streaming services
- **Churn Label**: Whether the customer left within the last month (target variable)

The dataset includes 7,043 customer records with 21 features, providing a comprehensive view of customer profiles and their relationship with service retention.

### Key Features

- **Tenure**: Number of months the customer has stayed with the company
- **MonthlyCharges**: The amount charged to the customer monthly
- **TotalCharges**: The total amount charged to the customer
- **Contract**: Contract term (Month-to-month, One year, Two year)
- **InternetService**: Type of internet service (DSL, Fiber optic, No)
- **Churn**: Target variable indicating if customer left (Yes/No)

## Contributing

Contributions are welcome! If you'd like to improve this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

Please ensure your code follows best practices and includes appropriate documentation and tests.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this software as per the terms of the license.

---

**Note**: This project is for educational and research purposes. When deploying to production, ensure compliance with data privacy regulations and implement appropriate security measures.
