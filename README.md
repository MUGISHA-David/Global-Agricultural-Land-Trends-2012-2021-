 Global Agricultural Land Trends Analysis - Capstone Project

## Project Overview
**Course:** INSY 8413 | Introduction to Big Data Analytics  
**Academic Year:** 2024-2025, Semester III  
**Instructor:** Eric Maniraguha  
**Student:** MUGISHA David  
**Date:** July 26, 2025

## 📊 Project Summary
This project analyzes global agricultural land trends from 2012-2021 using OECD agricultural environmental data. The analysis focuses on identifying patterns in agricultural land usage across different countries, implementing predictive modeling using linear regression, and creating interactive visualizations to communicate insights about global food security and land use sustainability.

---

## 🎯 Problem Definition & Planning

### Sector Selection
**Selected Sector:** ✅ **Agriculture**

### Problem Statement
**Research Question:** "How have global agricultural land areas changed over the past decade (2012-2021), and can we predict future agricultural land requirements for food security planning?"

This analysis addresses critical questions about:
- Global food security through land use trends
- Sustainable agricultural practices
- Country-specific agricultural land management
- Future agricultural planning and resource allocation

### Dataset Information
- **Dataset Title:** OECD Agricultural Environmental Indicators - Agricultural Land Area
- **Source:** OECD.TAD.ARP,DSD_AGRI_ENV@DF_AGLAND,1.0+.A.TOTAGR_LAND
- **Original File:** `OECD.TAD.ARP,DSD_AGRI_ENV@DF_AGLAND,1.0+.A.TOTAGR_LAND.....csv`
- **Cleaned File:** `cleaned_agri_land_data.csv`
- **Data Structure:** ✅ **Structured (CSV)**
- **Data Status:** ✅ **Requires Preprocessing**
- **Time Period:** 2012-2021
- **Geographic Coverage:** Multiple OECD countries

---

## 🐍 Python Analytics Implementation

### Data Cleaning & Preprocessing
- **Column Selection:** Extracted relevant columns: REF_AREA, Reference area, TIME_PERIOD, OBS_VALUE
- **Column Renaming:** Standardized to: Country_Code, Country, Year, Agri_Land_Area_Thousands_Ha
- **Missing Values:** Complete removal of rows with null values using `dropna()`
- **Data Type Conversion:** 
  - Year: Converted to integer
  - Agricultural land area: Converted to float
- **Data Validation:** Ensured data integrity through type checking and summary statistics

### Exploratory Data Analysis (EDA)
- **Descriptive Statistics:** Generated comprehensive summary statistics for agricultural land areas
- **Time Series Analysis:** Analyzed trends for key countries (Australia, Canada, Chile, Czech Republic)
- **Distribution Analysis:** Created box plots to understand land area distribution by year
- **Trend Visualization:** Line plots showing agricultural land changes over time

**Key Visualizations Created:**
- Multi-country time series line plots
- Box plots for distribution analysis by year
- Linear regression fit visualizations

### Machine Learning Model
- **Model Type:** ✅ **Regression** (Linear Regression)
- **Algorithm Used:** Scikit-learn LinearRegression
- **Features Selected:** Year (temporal predictor)
- **Target Variable:** Agricultural Land Area (Thousands of Hectares)
- **Training Process:** 80/20 train-test split with random_state=42

### Model Evaluation
**Evaluation Metrics Used:**
- ✅ **RMSE** (Root Mean Square Error)
- ✅ **R-squared** (Coefficient of Determination)

**Model Performance Results:**
- Successfully implemented country-specific prediction models
- Custom prediction function for future year forecasting
- Visual model validation through scatter plots with regression lines

### Innovation & Custom Features
- **Custom Prediction Function:** `predict_land_area(country_code, year_to_predict)`
  - Enables prediction for any country and future year
  - Automatic model training on country-specific data
  - Error handling for countries with insufficient data
- **Robust Data Processing:** Advanced error handling with try-catch blocks
- **Modular Code Structure:** Reusable functions for predictions
- **Multiple Visualization Techniques:** Combined matplotlib and seaborn for comprehensive analysis

---

## 📈 Power BI Dashboard

### Dashboard Components
1. **Problem Context & Insights**
   - Global agricultural land trends overview (2012-2021)
   - Key findings on land use sustainability
   - Country-specific agricultural patterns

2. **Interactive Features**
   - Country selection slicers for detailed analysis
   - Year range filters for temporal analysis
   - Drill-down capabilities by region and country

3. **Visualization Types Used**
   - World map showing agricultural land distribution
   - Line charts for trend analysis
   - Bar charts for country comparisons
   - KPI cards for summary statistics

4. **Design Elements**
   - Earth-tone color scheme reflecting agricultural theme
   - Clear navigation and user-friendly layout
   - Consistent labeling and formatting

5. **Advanced Features**
   - ✅ **DAX formulas** for calculated measures
   - ✅ **Custom tooltips** with detailed country information
   - ✅ **Bookmarks** for different view perspectives

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Power BI Desktop
- Required Python packages (see requirements.txt)

### Required Python Libraries
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```
## 📊 Key Findings & Results

### Main Insights

1. **Global Agricultural Stability:** Most OECD countries show relatively stable agricultural land areas from 2012-2021, indicating mature agricultural systems.

2. **Country-Specific Variations:** While overall trends are stable, individual countries show unique patterns:
   - Some countries demonstrate slight increases in agricultural land
   - Others show minor decreases, possibly due to urbanization or land use changes

3. **Predictive Accuracy:** Linear regression models show good performance for countries with consistent trends, enabling reliable short-term predictions.

4. **Data Quality:** High-quality OECD data enables robust analysis with minimal data cleaning requirements.

### Model Performance Summary
- **Best Performing Model:** Linear Regression for trend-consistent countries
- **Key Metrics:** RMSE and R-squared values vary by country based on trend consistency
- **Business Impact:** Enables agricultural planning and food security assessments

### Technical Achievements
- Successfully processed and cleaned large agricultural dataset
- Implemented scalable prediction system for multiple countries
- Created reusable analysis framework for agricultural data

---

## 🔮 Future Work & Recommendations

### Recommendations
1. **Policy Implications:** Use trend analysis for agricultural policy planning
2. **Food Security Planning:** Implement predictions in national food security strategies
3. **Sustainable Agriculture:** Monitor countries with declining agricultural land for intervention
4. **Resource Allocation:** Guide agricultural investment based on predicted land needs

### Future Enhancements
- **Advanced Modeling:** Implement time series forecasting (ARIMA, Prophet)
- **Additional Variables:** Include climate data, population growth, economic indicators
- **Machine Learning:** Explore ensemble methods and deep learning approaches
- **Real-time Analysis:** Integrate with live agricultural monitoring systems
- **Geographic Analysis:** Add spatial analysis and GIS integration

---

## 💻 Code Implementation

### Main Python Analysis Script

Below is the complete code implementation for the agricultural land analysis:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- Step 2a: Data Preprocessing ---
# Load the dataset. We use header=0 to correctly read the column names from the first row.
try:
    df = pd.read_csv('OECD.TAD.ARP,DSD_AGRI_ENV@DF_AGLAND,1.0+.A.TOTAGR_LAND.....csv', header=0)
    
    print("--- Original DataFrame Columns ---")
    print(df.columns)

    # Select only the relevant columns for analysis and create a new DataFrame.
    # This approach is more robust as it doesn't rely on dropping a long list of columns.
    # The columns from the first row are `REF_AREA`, `Reference area`, `TIME_PERIOD`, and `OBS_VALUE`.
    df_cleaned = df[['REF_AREA', 'Reference area', 'TIME_PERIOD', 'OBS_VALUE']].copy()

    # Rename columns for clarity and easier use in the code.
    df_cleaned = df_cleaned.rename(columns={
        'REF_AREA': 'Country_Code',
        'Reference area': 'Country',
        'TIME_PERIOD': 'Year',
        'OBS_VALUE': 'Agri_Land_Area_Thousands_Ha'
    })
    
    # Drop any rows that have missing values to ensure data integrity for the analysis.
    df_cleaned.dropna(inplace=True)
    
    # Convert 'Year' to integer and 'Agri_Land_Area_Thousands_Ha' to float for proper analysis.
    df_cleaned['Year'] = df_cleaned['Year'].astype(int)
    df_cleaned['Agri_Land_Area_Thousands_Ha'] = df_cleaned['Agri_Land_Area_Thousands_Ha'].astype(float)
    
    print("\n--- Data Preprocessing Complete ---")
    print("Cleaned DataFrame Info:")
    print(df_cleaned.info())
    print("\nFirst 5 rows of cleaned data:")
    print(df_cleaned.head())
    print("\n--- Summary Statistics ---")
    print(df_cleaned['Agri_Land_Area_Thousands_Ha'].describe())

    # --- Save the cleaned data to a new CSV file ---
    cleaned_file_name = 'cleaned_agri_land_data.csv'
    df_cleaned.to_csv(cleaned_file_name, index=False)
    print(f"\nCleaned data saved to '{cleaned_file_name}'")

    # --- Step 2b: Exploratory Data Analysis (EDA) ---
    print("\n--- Performing EDA: Time-series analysis ---")
    
    # Plotting agricultural land area trends for a few countries
    countries_to_plot = ['AUS', 'CAN', 'CHL', 'CZE'] 
    
    plt.figure(figsize=(12, 8))
    for country_code in countries_to_plot:
        # We now use 'Country_Code' to filter the data
        country_data = df_cleaned[df_cleaned['Country_Code'] == country_code]
        if not country_data.empty:
            plt.plot(country_data['Year'], country_data['Agri_Land_Area_Thousands_Ha'], label=f'{country_data["Country"].iloc[0]} ({country_code})')
        
    plt.title('Agricultural Land Area Trends (2012-2021)')
    plt.xlabel('Year')
    plt.ylabel('Agricultural Land Area (Thousands of Hectares)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Advanced Visualization with Seaborn: Box Plot ---
    # This visualization helps to understand the distribution of agricultural land area
    # across countries for each year.
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Year', y='Agri_Land_Area_Thousands_Ha', data=df_cleaned)
    plt.title('Distribution of Agricultural Land Area by Year')
    plt.xlabel('Year')
    plt.ylabel('Agricultural Land Area (Thousands of Hectares)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


    # --- Step 2c: Basic Modeling (Linear Regression) and Prediction ---
    print("\n--- Performing Basic Modeling and Prediction ---")
    
    # We will model the trend for Australia as an example
    country_for_model = 'AUS'
    # We now use 'Country_Code' to filter the data
    df_model = df_cleaned[df_cleaned['Country_Code'] == country_for_model].copy()
    
    # Define features (X) and target (y)
    X = df_model[['Year']]
    y = df_model['Agri_Land_Area_Thousands_Ha']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model for {country_for_model}:")
    print(f"  RMSE: {rmse:.2f} (Thousands of Hectares)")
    print(f"  R-squared: {r2:.2f}")

    # Plot the model's predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual Data')
    plt.plot(X, model.predict(X), color='red', label='Linear Regression Fit')
    plt.title(f'Linear Regression Model for {country_for_model}')
    plt.xlabel('Year')
    plt.ylabel('Agricultural Land Area (Thousands of Hectares)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Create a function to make a prediction for a specific country and year
    def predict_land_area(country_code, year_to_predict):
        """
        Trains a linear regression model on a country's data and predicts
        the agricultural land area for a given year.
        
        Args:
            country_code (str): The country code (e.g., 'AUS').
            year_to_predict (int): The year for which to make a prediction.

        Returns:
            float: The predicted agricultural land area in thousands of hectares, or None if no data is found.
        """
        # We now use 'Country_Code' to filter the data
        df_model = df_cleaned[df_cleaned['Country_Code'] == country_code].copy()
        
        if df_model.empty:
            print(f"No data available for country code '{country_code}'.")
            return None
            
        X = df_model[['Year']]
        y = df_model['Agri_Land_Area_Thousands_Ha']
        
        # We'll use the entire dataset for this country to train the prediction model
        model = LinearRegression()
        model.fit(X, y)
        
        prediction = model.predict([[year_to_predict]])
        
        print(f"\nPredicted agricultural land area for {df_model['Country'].iloc[0]} in {year_to_predict}: {prediction[0]:.2f} thousand hectares.")
        return prediction[0]

    # Example usage of the prediction function
    predict_land_area('AUS', 2025)
    predict_land_area('CAN', 2025)

except FileNotFoundError:
    print("Error: The CSV file was not found. Please ensure the file is in the correct directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
```

### Key Code Features

#### Data Preprocessing Functions
- **Robust column selection and renaming**
- **Data type conversion and validation**
- **Missing value handling with `dropna()`**
- **Error handling with try-catch blocks**

#### Exploratory Data Analysis
- **Multi-country time series visualization**
- **Statistical distribution analysis using box plots**
- **Comprehensive summary statistics generation**

#### Machine Learning Implementation
- **Linear regression modeling with scikit-learn**
- **Train-test split for model validation**
- **Performance evaluation using RMSE and R-squared**
- **Visual model validation with scatter plots**

#### Innovation Features
- **Custom prediction function for any country/year combination**
- **Automated model training per country**
- **Scalable architecture for multiple countries**
- **User-friendly prediction interface**

---

## 📸 Screenshots

### Python Analysis Results

#### Data Preprocessing Output

![Data proccessing](https://github.com/user-attachments/assets/5633a5b9-b97b-4370-a79f-b8e0174a40b7)



#### Exploratory Data Analysis Visualizations
<img width="1031" height="701" alt="agriculture land area trends" src="https://github.com/user-attachments/assets/34a750f8-190b-4531-8c59-c7f5af66e185" />
<img width="1186" height="722" alt="distribution agriculture land area" src="https://github.com/user-attachments/assets/2592cc4f-6431-4745-a002-1f8d3d2f6209" />
<img width="876" height="547" alt="linear regression model" src="https://github.com/user-attachments/assets/5f191006-ab9c-4345-b1cb-c9ea5a6fcd1f" />




### Power BI Dashboard Views

####  Main Dashboard Overview

![power BI Dashboard](https://github.com/user-attachments/assets/f6091439-c515-41a3-806c-d8a47af4e037)

## 📚 References & Data Sources

### Dataset Sources
- **Primary Dataset:** OECD Agricultural Environmental Indicators
  - Source: OECD.Stat Database
  - URL: [OECD Agricultural Data Portal]
  - Coverage: Agricultural land area data for OECD countries (2012-2021)

### Tools & Libraries Used
- **Python Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn
- **Data Analysis:** Jupyter Notebook, Python 3.8+
- **Visualization:** Power BI Desktop, matplotlib, seaborn
- **Development Environment:** Jupyter Notebook
- **Version Control:** Git/GitHub
- **Statistical Modeling:** scikit-learn LinearRegression

### Additional Resources
- OECD Agricultural Environmental Indicators Documentation
- Python Data Science Handbook
- Power BI Best Practices Guide

**Final Note:** This analysis contributes to understanding global food security challenges and supports evidence-based agricultural policy decisions through data-driven insights.
