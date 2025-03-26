# HousingPriceAnalysis: A Complete ML Pipeline

This project covers the full process of working with housing data—from data exploration and preparation to building and evaluating machine learning models for predicting house prices.

## Main Steps

1. **Data Exploration**  
   - Reviewing the structure of the dataset  
   - Identifying missing values and anomalies  
   - Creating initial visualizations of key features  

2. **Exploratory Data Analysis (EDA)**  
   - Examining correlations between features  
   - Analyzing the geographic distribution of properties  
   - Creating new features to improve model accuracy  

3. **Data Preparation**  
   - Encoding categorical features  
   - Normalizing numerical data  
   - Automating preprocessing with a `Pipeline`  

4. **Model Training and Evaluation**  
   - Building baseline models: linear regression and decision trees  
   - Using RMSE to measure model performance  
   - Evaluating overfitting and planning for prevention  

5. **Optimization and Conclusions**  
   - Applying cross-validation to improve model stability  
   - Tuning hyperparameters with `GridSearchCV`  
   - Suggestions for further improvements in predictions  

## Setting Up the Environment

To run this project, install the following libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

To launch Jupyter Notebook, run:

```bash
jupyter notebook HousingAnalysis.ipynb
```

The `HousingAnalysis.ipynb` file contains all the code and step-by-step instructions for working with the data.

## Key Points of the Analysis

### Initial Data Exploration

When analyzing the `housing.csv` data, we found some interesting points:

- The `median_income` feature has a strong impact on house prices and is an important factor in the model.  
- The categorical feature `ocean_proximity` can improve the predictions.  
- There are missing values in the `total_bedrooms` column that need to be handled appropriately.

We also created an `income_cat` feature for stratified splitting of the dataset into training and test sets.

### Data Visualization

- The distribution of house prices shows some outliers, so we apply logarithmic transformation.  
- Geographic analysis using `longitude` and `latitude` helps to identify areas with different price levels.  
- A scatter plot visually highlights regions with expensive housing.

### Data Preparation

We use a `Pipeline` for preprocessing that includes:

- Filling missing values with the median  
- Scaling numerical features  
- Applying One-Hot Encoding for categorical variables

### Model Training

1. **Linear Regression** – Used as a baseline model for comparison.  
2. **Decision Trees** – Show signs of overfitting on the training data.

The models are evaluated using the RMSE metric to measure prediction errors.

## Results and Future Improvements

- The linear regression model achieved an RMSE of approximately **68,000**, which is a reasonable start.  
- The decision tree model shows overfitting, suggesting that its depth needs to be controlled.  
- To improve accuracy, consider using models like Random Forest or Gradient Boosting.

Possible improvements include:  
✅ Using `GridSearchCV` to fine-tune model parameters  
✅ Removing features that do not contribute significantly  
✅ Applying ensemble methods (such as RandomForest or XGBoost)

## Conclusion

This project demonstrates a full cycle of data analysis and predictive modeling. Its flexible structure makes it easy to adapt and expand. The next steps include testing more advanced models (like XGBoost or CatBoost) and further optimizing the feature set.
