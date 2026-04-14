# Final Project

**CIS 5450: Big Data Analytics**

## All Group Members’ Names & Duties

- **Dania Hasan:** Data cleaning, preprocessing, and dataset integration
- **Isabel Portner:** EDA, visualizations, and results analysis
- **Toma Yasuda:** Feature engineering and data transformation
- **Ruichi Zhang:** Modeling, evaluation, and performance analysis

---

## Data Sources

### Hotels Dataset

List of hotels from different countries and regions, containing information on ratings & nearby attractions.

- **Dataset Link:** https://www.kaggle.com/datasets/raj713335/tbo-hotels-dataset
- **# of rows:** 10,000,000+
- **# of columns:** 16

**Relevant Features Used**

- `countyName` / `countyCode` → country-level categorical feature
- `cityName` → city-level categorical feature
- `hotelRating` → target variable (hotel quality)
- `attractions` → transformed into `attractions_count` (numeric feature)
- `Description` → optional derived feature such as description length (non-NLP)

### World Cities Dataset

List of cities in the world.

- **Dataset Link:** https://www.kaggle.com/datasets/max-mind/world-cities-database?resource=download&select=worldcitiespop.csv
- **# of rows:** 2,300,000+
- **# of columns:** 7

We will join this dataset with the Hotels dataset using city and country to extract:

- **Region** → categorical feature (geographic grouping)
- **Population** → numeric feature (city size indicator)
- **Latitude** → numeric feature
- **Longitude** → numeric feature

### Crime Index Dataset

Dataset contains safety-related metrics for cities.

- **Dataset Link:** https://www.kaggle.com/datasets/ahmadjalalmasood123/world-crime-index
- **# of rows:** 453
- **# of columns:** 4

**Relevant Features Used**

- `crime_index` → numeric feature
- `safety_index` → numeric feature

Due to the smaller size of this dataset, it will be used as an optional feature where city matches are available.

---

## Objective and Value Proposition

Explain what you intend to study with your project. What is the ultimate objective? Why is this project interesting?

The objective of this project is to analyze how different external factors influence hotel ratings.

We will combine hotel-level data with city-level features to study relationships between:

- geographic factors (region, latitude/longitude)
- demographic factors (population)
- hotel characteristics (attractions)

We will build a regression model to predict hotel ratings based on these features and evaluate which variables have the strongest impact.

**Key questions we aim to answer include:**

- How does city size (population) affect hotel ratings?
- Do hotels in certain regions or countries tend to have higher ratings?
- Do nearby attractions contribute to better hotel ratings?

**This project is interesting because:**

- It uses large-scale data (millions of rows) to uncover real-world patterns
- It integrates multiple datasets to create richer features
- It focuses on interpretable modeling, allowing us to understand which factors matter most

**The results of this project can help:**

- Identify key drivers of hotel quality
- Support data-driven decision-making in travel and hospitality
- Provide insights for building recommendation systems

---

## Modeling Plan

We plan to model this problem as a **regression task**, where the goal is to predict hotel ratings based on structured features such as location, safety, and hotel characteristics.

The **target variable** will be the hotel rating from the Hotels dataset.

We will integrate multiple datasets to construct meaningful features, including:

- City-level features (population, region, geographic location)
- Safety features (crime index and safety index, where available)
- Hotel-level features (facilities, nearby attractions)

We will begin with baseline models such as **Linear Regression**, and then explore more advanced models including **Random Forest Regressor** and **Gradient Boosting** models (e.g., XGBoost).

We will evaluate model performance using **RMSE** and **R²**, and analyze **feature importance** to identify the most significant factors influencing hotel ratings.

---

## Hypothesis Testing

Are you considering doing any hypothesis tests? If so, what are your null hypothesis and your test statistic? How do you plan on using topics covered in class (i.e. simulations) to test your hypotheses?

Yes, we are considering doing hypothesis tests to evaluate whether various external factors are significantly associated with hotel ratings.

### Test 1: Population and hotel ratings

**Null hypothesis:** The city-level population has no relationship with hotel ratings.

**Alternative hypothesis:** The city-level population is associated with hotel ratings.

To test this approach, we can use the **coefficient of population in a linear regression** as the main test statistic. If the estimated coefficient is significantly different from 0, it suggests that the population has a meaningful effect on hotel ratings. In cases where the null hypothesis is that the population has no relationship with hotel ratings, we repeatedly **shuffle hotel ratings** across observations, refit the regression model, and record the simulated population coefficients, thus generating an **empirical null distribution**.

### Test 2: Attractions and hotel ratings

**Null hypothesis:** The number of nearby attractions does not affect hotel ratings.

**Alternative hypothesis:** Hotels with more nearby attractions tend to have different ratings.

For this, we can again use the **regression coefficient for `attractions_count`** as the test statistic. We can then test the null hypothesis by **randomly permuting hotel ratings** several times, then refit the regression model after each permutation, and record the resulting coefficients. This **simulation-based null distribution** will allow us to determine whether the observed coefficient is significantly different from what would be expected by chance.

---

## Anticipated Obstacles & Challenges

One of the challenges is **data integration** across multiple datasets. Since we plan to join datasets using city and country, inconsistencies such as different naming conventions may lead to incorrect joins or data loss. Also, there might be **multiple cities with the same name** within the same countries. We will need to perform careful data cleaning and standardization to address this issue.

Another challenge is **imbalance**, as many hotels may cluster between 3–5 stars, and hotel ratings are often **subjective**. This subjectivity introduces noise and inconsistency in the target variable, making it harder for the model to learn reliable patterns, which could affect the model's performance.
