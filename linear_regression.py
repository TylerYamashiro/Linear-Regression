#!/usr/bin/env python
# coding: utf-8

# ### Linear Regression
# 
# Other Sources:
# - Here is a <a href="https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-guide-regression-analysis-plot-interpretations/tutorial/">nice introductory article</a>.  It gives some additional attention to examining and evaluating the regression model. The code is written in R rather than Python, but it shouldn't be hard to interpret the few lines of R code. 
# - Text section 18.6 cover the material theoretically and quickly.  Don't worry about the material on gradient descent.
# - Chapter 3 of <a href="https://www-bcf.usc.edu/~gareth/ISL/ISLR%20First%20Printing.pdf">An Introduction to Statistical Learning</a> is your best bet for a deeper treatment, even though R is the language of choice.  In fact, that should be your go-to book for more machine learning material.
# 
# Objectives:
# - Define data modeling and simple linear regression
# - Show various steps in examining a data set and preparing it for input to a learning library
# - Build a linear regression model using Python libraries
# - Understand how to evaluate the quality of a model and compare it alternative models

# ##### The Bikeshare Data Set

# We'll be working with a data set from Capital Bikeshare that was used in a [Kaggle competition](https://www.kaggle.com/c/bike-sharing-demand/data).
# 
# The goal is to predict total ridership of Capital Bikeshare in any given hour.

# ### Capital Bikeshare Data Dictionary

# | Variable| Description |
# |---------|----------------|
# |datetime| hourly date + timestamp  |
# |season|  1=winter, 2=spring, 3=summer, 4=fall |
# |holiday| whether the day is considered a holiday|
# |workingday| whether the day is neither a weekend nor holiday|
# |weather| 1 -> Clear or partly cloudy;  2 -> Clouds and mist; 3 -> Light rain or snow;  4 -> Heavy rain or snow|
# |temp| temperature in Celsius|
# |atemp| "feels like" temperature in Celsius|
# |humidity| relative humidity|
# |windspeed| wind speed|
# |casual| number of non-registered user rentals initiated|
# |registered| number of registered user rentals initiated|
# |count| number of total rentals|
# 

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 14


# <a id="read-in-the--capital-bikeshare-data"></a>
# ### Read In the Capital Bikeshare Data

# In[3]:


file_location = 'C:/Users/Tyler/Documents/AI/Notebook files/bikeshare.csv'
bikes = pd.read_csv(file_location, parse_dates=['datetime'])


# In[4]:


bikes.info()


# In[1]:


bikes.head(30)


# In[ ]:


bikes.describe()


# In[ ]:


bikes.columns


# <span style="color:blue;font-size:120%">What does each row in the dataframe represent?</span>

# <span style="color:blue;font-size:120%">What is the response variable (the quantity we are trying to predict)?</span>

# <span style="color:blue;font-size:120%">How many features are there that could be predictors in a regression?  What are other potential features?</span>

# In[ ]:





# #### Rename some variables
# 
# "count" is a method in Pandas (and a very non-specific name), so it's best to name that column something else
# 
# In general, rename columns immediately if they are reserved words, contain invalid characters, or it is unclear what the column represents.  
# 
# If you have a variable naming system you are comfortable with, rename your variables now -- it will pay off later.
# 
# Inplace means the dataframe is changed directly rather than making a copy, so be careful that you have a saved copy of the data if you need one.

# In[11]:


bikes.rename(columns={"count": "total_rentals", 
                      "season": "season_num", 
                      "holiday": "is_holiday",
                      "workingday": "is_working_day"}, inplace=True);


# <a id="visualizing-the-data"></a>
# ### Visualizing the Data

# It is important to have a general feeling for what the data looks like before building a model. 
# 
# We are generally interested in several things in examining the data set
# * Making sure the data values correspond to what we think they are representing (why temp and atemp for example)
# * Look for outliers
# * Look for variables that correlate with the response variable
# * Look for variables that correlate with each other (temp and atemp for example)

# In[ ]:


# Explore correlation between other variables and total_rentals.  Secondarily, predictor variables with each other


# In[5]:


sns.pairplot(bikes);


# We are going to focus on the single variate ```temp``` (temperature) first

# In[6]:


bikes.temp.plot(kind="box")
bikes.temp.describe()


# Is the difference between temp and atemp significant?

# In[ ]:


(bikes.temp / bikes.atemp).describe()


# In[7]:


(bikes.temp / bikes.atemp).plot.box();


# In[8]:


bikes.temp.hist(bins=100);


# In[12]:


# Pandas scatterplot
bikes.plot(kind='scatter', y='total_rentals', x='temp', alpha=0.2);


# In[13]:


# Seaborn scatterplot with regression line
#    aspect => plot is twice as long as it is high
#    alpha => transparancy on scale of 0 (transparent) to 1 (opaque)

sns.lmplot(x='temp', y='total_rentals', data=bikes, aspect = 2.0, scatter_kws={'alpha':0.2});


# ### Linear Regression Basics
# ---

# ### Form of Linear Regression
# 
# Think of our data frame as a set of observations of the form:  
# $$(x_{i1}, x_{i2}, \ldots, x_{ik}, y_i)$$ 
# 
# where the $x$ are the "independent variables" and $y$ is the "dependent variable".  We are trying to find a function of the $x$ variables that predict the value of the $y$ variable.   We have $n$ observations and $k$ independent variables (features).
# 
# *  *Regression*  the $y$ variable is numeric and ordered -- for example temperature or number of rentals
# *  *Classification* the $y$ variable is taken from a set -- for example true/false or {red, green, blue}
# 
# *Linear* -- Start with the formula for a line:  $$y = \alpha + \beta x\\y = \beta_0 + \beta_1 x$$ 
# (The second notation gives a consistent name to the coefficients.)
# 
# * First statement of the linear regression problem:  find the values of $\beta_0$ and $\beta_1$ that best predict the $y$ values for all of our observations
# * Second statement of the problem:  since every model contains some amount of noise -- "random irreducible error" $\epsilon$,  instead we are actually working optimizing $$y = \beta_0 + \beta_1  + \epsilon$$
# where $\epsilon$ is small and uncorrelated with $x$ and $y$.
# 
# ---
# 
# Here, we will generalize this to $n$ independent variables as follows:
# 
# $$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon\quad =\quad \beta_0 + (\sum_{i=0}^{k} \beta_ix_i) + \epsilon$$
# 
# - $y$ is the response.
# - $\beta_0$ is the intercept.
# - $\beta_1$ is the coefficient for $x_1$ (the first feature).
# - $\beta_k$ is the coefficient for $x_k$ (the kth feature).
# - $\epsilon$ is the _error_ term which is independent of x, y, and $k$
# 

# A practical example of this applied to our data might be:
# 
# $${\tt total\_rides} = 20 + -2 \cdot {\tt temp} + -3 \cdot {\tt windspeed}\ +\ ...\ +\ 0.1 \cdot {\tt registered}$$
# 
# This equation is still called **linear** because the highest degree of the independent variables (e.g. $x_i$) is 1. Note that because the $\beta$ values are constants, they will not be independent variables in the final model, as seen above.

# <span style="color:blue; font-size:120%"> What are the limits of this linearity assumption?  What kind of relationships can't we capture?  Can you think of real examples where linearity is violated?</span>

# ---
# 
# In the regression equation
# $y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon$
# 
# the $\beta$ values are called the **model coefficients**:
# 
# - These values are estimated (or "learned") during the model fitting process using the **least squares criterion**.
# - Specifically, we are trying to find the line (mathematically) that minimizes the **sum of squared residuals** (or "sum of squared errors").
# - Once we've learned these coefficients, we can use the model to predict the response.
# 
# ![Estimating coefficients](notebook-files/estimating_coefficients.png)
# 
# Earlier in the quarter we defined mean-squared error as follows:
# $$MSE = \frac{1} {n} \| \hat{y}(\mathbf{X}) - \vec{y} \|$$
# 
# and said the algorithm chooses coefficients that minimize this quantity.
# 
# In the diagram above:
# 
# - The black dots are the **observed values** of x and y.
# - The blue line is our **least squares line**.
# - The red lines are the **residuals**, which are the vertical distances between the observed values and the least squares line.
# 
# If there are two predictors, the model fits a plane, and residuals are the distance between the observed value and the least squares plane.
# 
# ![Regression with Two Variates](notebook-files/multiple_regression_plane.png)
# 
# If there are more than two predictors, it's hard to visualize.

# ### Running a Regression using scikit-learn
# 
# To fit the skitkit-learn data we have to make our observation data fit its model
# 
# 1. Features and response should be separate objects
# 2. Features and response should be entirely numeric
# 3. Features and response should be NumPy arrays (or easily converted to NumPy arrays)
# 4. Features and response should have specific shapes (outlined below)

# <a id="building-a-linear-regression-model-in-sklearn"></a>
# ### Building a (Single) Linear Regression Model in sklearn

# #### Create a feature matrix called X that holds a `DataFrame` with only the temp variable and a `Series` called y that has the "total_rentals" column.

# In[14]:


# Create X and y  try to predict total rentals from temperature (only)
feature_cols = ['temp']
X = bikes[feature_cols]
y = bikes.total_rentals


# <span style="color:blue; font-size:120%">Why do you think X is capitalized but y is not?</span>

# In[15]:


print(type(X))
print(type(y))
print(X.shape)
print(y.shape)


# 
# ### scikit-learn's Four-Step Modeling Pattern -- Same for Many Prediction Algorithms

# **Step 1:** Import the class you plan to use.

# In[16]:


from sklearn.linear_model import LinearRegression


# In[17]:


# Make an instance of a LinearRegression object.
lr = LinearRegression()
type(lr)


# - Created an object that "knows" how to do linear regression, and is just waiting for data.
# - There are some algorithm-specific parameters to control the learner;  all have default values
# 
# To view the possible parameters, either use the `help` built-in function or evaluate the newly instantiated model, as follows:

# In[18]:


#help(lr)
lr


# **Step 3:** Fit the model with data (aka "model training").
# 
# - Model is "learning" the relationship between X and y in our "training data."
# - Process through which learning occurs varies by model.
# - Occurs in-place.

# In[19]:


lr.fit(X, y)


# - Once a model has been fit with data, it's called a "fitted model."
# - Next steps
#   - We can predict y values from an X vector
#   - We need to assess how good the model is as a predictor

# In[20]:


# To get a prediction for one observation and one variable, we need a (1,1) array
#    to predict two observations we need a (2,1) array, etc.
print(lr.predict(np.array([11]).reshape(-1,1)))
print(lr.predict(np.array([11, 5]).reshape(-1,1)))


# In[21]:


# If we predict at x=0 we will get the intercept
lr.predict(np.array([0]).reshape(-1,1))


# In[ ]:


lr.predict(np.array([0]).reshape(1,-1))


# In[ ]:


#  Just for convenience -- take a number (x) gets its predicted y
def lrpred(xval):
    return lr.predict(np.array([xval]).reshape(1,-1))[0]


# In[ ]:


# f(n+1) - f(n) should give us the slope always
print(lrpred(4) - lrpred(3))
print(lrpred(21) - lrpred(20))


# In[ ]:


lr.coef_


# In[ ]:


lr.intercept_


# In[ ]:


lr.predict([[0], [10], [100]])


# What we just predicted using our model is, "If the temperature is 0 degrees, the total number of bike rentals will be ~6.046, and if the temperature is 10 degrees the total number of bike rentals will ~97.751."

# <span style="color:blue;font-size=120%">How would you informally confirm that these are reasonable predictions?</span>

# Interpreting the intercept ($\beta_0$):
# 
# - It is the value of $y$ when all independent variables are 0.
# - Here, it is the estimated number of rentals when the temperature is 0 degrees Celsius.
# - <span style="color:blue">**Note:** It does not always make sense to interpret the intercept. (Why?)</span>
# 
# Interpreting the "temp" coefficient ($\beta_1$):
# 
# - **Interpretation:** An increase of 1 degree Celcius is _associated with_ increasing the number of total rentals by $\beta_1$.
# - Here, a temperature increase of 1 degree Celsius is _associated with_ a rental increase of 9.17 bikes.
# - This is not a statement of causation.
# - $\beta_1$ would be **negative** if an increase in temperature was associated with a **decrease** in total rentals.
# - $\beta_1$ would be **zero** if temperature is not associated with total rentals.

# <a id="visualizing-the-data-part-"></a>
# ### Visualizing the Data (Part 2)

# #### Explore more features.

# In[22]:


# Create feature column variables
feature_cols = ['temp', 'season_num', 'weather', 'humidity']


# #### Create a subset of scatterplot matrix using Seaborn.
# We can use pairplot with the y_vars argument to only show relationships with the `total_rentals` variable

# In[23]:


bikes.columns


# In[24]:


# multiple scatterplots in Seaborn
sns.pairplot(bikes, x_vars=feature_cols, y_vars='total_rentals', kind='reg');


# #### Recreate the same functionality using Pandas.

# In[25]:


# Multiple scatterplots in Pandas -- put plots on a 1x4 grid, and force the same Y axis
fig, axs = plt.subplots(1, len(feature_cols), sharey=True)
for index, feature in enumerate(feature_cols):
    bikes.plot(kind='scatter', x=feature, y='total_rentals', ax=axs[index], figsize=(16, 3))


# You generally don't want to see variables with a small number of integer indexes -- these are *categorical* variables, but the regression treats them as real-valued.  So it thinks that season = 2 is "greater than" season = 1.  Since season gets a single coefficient in the regression, and "increase" in season always has the same effect on the dependent variable.  Be sure that's what you want!   Is it true of weather for example?

# #### Look at rentals over time.

# In[26]:


# Line plot of rentals
bikes.total_rentals.plot();


# #### What does this tell us?
# 
# There are more rentals in the winter than the spring, but only because the system is experiencing overall growth and the winter months happen to come after the spring months.
# 
# So does overall growth completely explain the model?

# In[27]:


df = pd.DataFrame(bikes['total_rentals'])
df['hours_since_open'] = range(0, bikes.shape[0])
sns.lmplot(x='hours_since_open', y='total_rentals', data=df, aspect=1.5, scatter_kws={'alpha':0.2});


# Question for later: how much better -- if at all -- is any model we build from the column featues than this model?

# #### Look at the correlation matrix for the bikes `DataFrame`.

# In[28]:


# Correlation matrix (ranges from 1 to -1)
bikes.corr()


# #### Use a heat map to make it easier to read the correlation matrix.

# In[29]:


# Visualize correlation matrix in Seaborn using a heat map.
sns.heatmap(bikes.corr())


# <span style="color:blue;font-size=120%">What relationships do you notice and which are important for our model building?</span>

# ### Adding More Features to the Model

# In the previous example, one variable explained the variance of another; however, more often than not, we will need multiple variables. 
# 
# - For example, a house's price may be best measured by square feet, but a lot of other variables play a vital role: bedrooms, bathrooms, location, appliances, etc. 
# 
# - For a linear regression, we want these variables to be largely independent of one another, but all of them should help explain the y variable.
# 
# We'll work with bikeshare data to showcase what this means and to explain a concept called *multicollinearity*.

# #### Create another `LinearRegression` instance that is fit using temp, season, weather, and humidity.

# In[30]:


# Create a list of features.
feature_cols = ['temp', 'season_num', 'weather', 'humidity']


# In[31]:


# Create X and y.
X = bikes[feature_cols]
y = bikes.total_rentals

# Instantiate and fit.
linreg = LinearRegression()
linreg.fit(X, y)

# Print the coefficients.
print(linreg.intercept_)
print(linreg.coef_)


# #### Display the linear regression coefficient along with the feature names.

# In[ ]:


# Pair the feature names with the coefficients.
list(zip(feature_cols, linreg.coef_))


# Interpreting the coefficients:
# 
# - Holding all other features fixed, a 1-unit increase in temperature is associated with a rental increase of 7.86 bikes.
# - Holding all other features fixed, a 1-unit increase in season is associated with a rental increase of 22.5 bikes.
# - Holding all other features fixed, a 1-unit increase in weather is associated with a rental increase of 6.67 bikes.
# - Holding all other features fixed, a 1-unit increase in humidity is associated with a rental decrease of 3.12 bikes.
# 
# Does anything look incorrect and does not reflect reality?

# 
# ### What Is Multicollinearity?
# ---
# 
# Multicollinearity happens when two or more features are highly correlated with each other. The problem is that due to the high correlation, it's hard to disambiguate which feature has what kind of effect on the outcome. In other words, the features mask each other. 
# 
# There is a second related issue called variance inflation where including correlated features increases the variability of our model and p-values by widening the standard errors. This can be measured with the variance inflation factor, which we will not cover here.

# #### With the bikeshare data, let's compare three data points: actual temperature, "feel" temperature, and guest ridership.

# In[ ]:


cmap = sns.diverging_palette(220, 10, as_cmap=True)
correlations = bikes[['temp', 'atemp', 'casual']].corr()
print(correlations)
print(sns.heatmap(correlations, cmap=cmap))


# It is common for a data set to have sets of highly correlated variables.  Generally a good policy is to choose one from each set.   
# 
# <span style="color:blue;font-size=120%">Best guess as to a good model to predict total_rentals?</span>
# 
# Let's build it then figure out how to measure "goodness," then see how good it is.

# In[ ]:


feature_cols = ['YOUR COLUMNS HERE']
X = bikes[feature_cols]
y = bikes.total_rentals

# Instantiate and fit.
bestModel = LinearRegression()
bestModel.fit(X, y)


# ### Evaluating Models / Comparing Models
# ---
# 
# We can make linear models now, but how do we select the "best" model to use for our applications? 
# 
# These are three different questions:
# 1.  How well does the model fit the observed data (training set)
# 2.  How well will the model predict future observations
# 3.  What is the best model for the business scenario (i.e. taking into account the reward for predicting correctly and the penalty for making an error).
# 
# Accuracy on the training set may differ from accuracy when the model goes into production for two reasons
# 1.  The training set was too small -- important features in the population didn't show up enough to be noticed
# 2.  The model found spurious signal in the training set "total sales is higher on days that begin with 'T'." (Overfitting)
# 
# Looking at the first:
# * The $R^2$ statistic -- the amount the model reduces variability
# * Significance tests on the slope coefficient(s) -- if a variable's coefficient is 0, then it is not adding predictive power
# 

# ### Statistics from the Residuals
# 
# **Residual sum of squares**
# 
# $$
# \begin{aligned}
# \| \vec{actual} - \vec{predicted} \|  & =\sqrt{(actual_1 - predicted_1)^2 + (actual_2 - predicted_2)^2 + \ldots + (actual_n - predicted_n)^2}\\
# &= \sqrt{\sum_{i=1}^{n} (actual_i - predicted_i)^2}\\
# &= \sqrt{\sum_{i=1}^{n} (y_i - \hat{y})^2}
# \end{aligned}
# $$
# $$
# \begin{aligned}
# RSS &= \sum_{i=1}^{n} (y_i - \hat{y})^2\\
# MSE &= \frac{\sum_{i=1}^{n} (y_i - \hat{y})^2}{n}
# \end{aligned}
# $$
# 
# Remember, the regression chooses coefficients and intercept to minimize this value **MSE**
# 
# 
# There are several reasons why a linear model of $f(x)$, $\hat{f}(x)$ will differ from the true $f(x)$
# 1. The real $f(\overrightarrow{x})$ is not linear, so $\hat{f}$ cannot provide a good approximation
# 2. The variables used in training $\hat{f}$ are not the same inputs as to the true $f$ -- most often $f$ depends on some unobserved factors as well as $\overrightarrow{x}$ 
# 3. The noise term $\epsilon$ makes it hard for any model to predict $f$ well
# 
# When we are comparing linear models with different sets of predictive variables, we can only measure the second.
# 
# **Total sum of squares**
# $$TSS = \sum_{i=1}^n(y_i - \overline{y})^2$$
# Standard definition of deviance of observations from the mean -- model independent.
# 
# **R-squared is proportion of the total variance accounted for by the model**
# $$R^2 \quad=\quad \frac{(TSS - RSS)}{TSS} \quad = \quad 1 - \frac{RSS}{TSS}$$
# If RSS = 0 then $R^2$ = 1 which means RSS accounts for 100% of the variance (it perfectly predicts the observed).  If $RSS = TSS$ then the regression does not reduce any variance (worst case model), same as guessing at the mean.
# 
# $R^2$ is a ratio (unit-free) so can be compared across models, but not clear what a "good enough" value is.  At least partially depends on what we know about the underlying system -- contributors to low values might be
# * Underlying system is not linear -- so linear model will no do well
# * Underlying system is linear, but we are missing key variates
# 
# **Adjusted R-squared adjusts for number of variables in the model**
# $R^2$ increases as more predictors are added to the model, regardless of whether they improve prediction.
# The adjusted version corrects for this
# $$ {R}^{2}_{adj \quad}= \quad 1-\left[\frac{(1-R^{2})(n-1)}{n-p-1}\right]$$
# where $n$ is the number of observations and $p$ is the number of variables in the model

# In[ ]:


def tss(actual):
    return ((actual - actual.mean())**2).sum()

def rss(actual, predicted):
    return ((actual - predicted)**2).sum()

def rsquared(actual, predicted):
    return (tss(actual) - rss(actual, predicted))/tss(actual)

def adjust_rsquared(rquared, n, p):
    return 1 - ((1-rsquared) * (n - 1) /(n - p - 1))


# In[ ]:


#  Examine adjusted rsqured from (a) predict at the mean, (b) our first model, (c) our best model


# ### Standard Regression Summary
# 
# Most software gives you a table summarizing the regression and giving some "quality" statistics.  
# A different python package does -- here is an example
#   * The add_constant step adds a column of 1's to the X matrix -- we get the intercept from that
#   * I added a random uncorrelated X variable for illustration

# In[ ]:


import statsmodels.api as sm
bikes['rand'] = 50*np.random.rand(bikes.shape[0])
X = sm.add_constant(bikes[['temp', 'atemp', 'windspeed', 'season_num', 'rand']])
y = bikes['total_rentals']
results = sm.OLS(y, X).fit()
print(results.summary())


# **Highlights of the Regression Results Table**
# * The F-statistic tests the hypothesis "all of the coefficients are 0" -- it's a ratio, and a value of 1 indicates no relationship.  The F-statistic strongly indicates that the model is better at predicting total rentals than using the mean value
# * For the intercept and coefficients, the $p$ value tests the hypothesis that the parameter is 0.  Values close 0 mean the parameter is unlikely to be 0.  Notice the confidence interval and $p$ value for *rand* -- its coefficient is very likely to be 0. 

# **Try some column combinations to get a feel for what variable selection does to R-squared**

# In[ ]:


#  A way to get some statistics like adjusted R-squared for a set of columns
def statsmodel_summary(dataset, columns):
    X = sm.add_constant(dataset[columns])
    Y = dataset['total_rentals']
    results = sm.OLS(y, X).fit()
    print(results.summary())


# In[ ]:


# For example, this took away the random column -- notice R-squared stayed the same
# but adjusted R-squared went up a little.

statsmodel_summary(bikes, ['atemp', 'windspeed', 'season'])


# <a id="handling-categorical-features"></a>
# ### Handling Categorical Features
# 
# scikit-learn expects all features to be numeric. So how do we include a categorical feature in our model?
# 
# - **Ordered categories:** Transform them to sensible numeric values (example: small=1, medium=2, large=3)
# - **Unordered categories:** Use dummy encoding (0/1). Here, each possible category would become a separate feature.
# 
# What are the categorical features in our data set?
# 
# - **Ordered categories:** `weather` (already encoded with sensible numeric values)
# - **Unordered categories:** `season` (needs dummy encoding), `holiday` (already dummy encoded), `workingday` (already dummy encoded)
# 
# For season, we can't simply leave the encoding as 1 = spring, 2 = summer, 3 = fall, and 4 = winter, because that would imply an ordered relationship. Instead, we create multiple dummy variables.

# #### Create dummy variables using `get_dummies` from Pandas.

# In[ ]:


season_dummies = pd.get_dummies(bikes.season, prefix='season')


# #### Inspect the `DataFrame` of `dummies`.

# In[ ]:


# Print five random rows.
season_dummies.sample(n=5, random_state=1)


# However, we actually only need three dummy variables (not four), and thus we'll drop the first dummy variable.
# 
# Why? Because three dummies captures all of the "information" about the season feature, and implicitly defines spring (season 1) as the baseline level.
# 
# This circles back to the concept multicollinearity, except instead of one feature being highly correlated to another, the information gained from three features is directly correlated to the fourth.

# #### Drop the first column.

# In[ ]:


season_dummies.drop(season_dummies.columns[0], axis=1, inplace=True)


# #### Reinspect the `DataFrame` of `dummies`.

# In[ ]:


# Print five random rows.
season_dummies.sample(n=5, random_state=1)


# In general, if you have a categorical feature with k possible values, you create k-1 dummy variables.
# 
# If that's confusing, think about why we only need one dummy variable for `holiday`, not two dummy variables (`holiday_yes` and `holiday_no`).

# #### We now need to concatenate the two `DataFrames` together.

# In[ ]:


# Concatenate the original DataFrame and the dummy DataFrame (axis=0 means rows, axis=1 means columns).
bikes_dummies = pd.concat([bikes, season_dummies], axis=1)

# Print 5 random rows.
bikes_dummies.sample(n=5, random_state=1)


# #### Rerun the linear regression with dummy variables included.

# In[3]:


# Include dummy variables for season in the model.
feature_cols = ['atemp', 'windspeed', 'season_2', 'season_3', 'season_4']
X = bikes_dummies[feature_cols]
y = bikes_dummies.total_rentals

linreg = LinearRegression()
linreg.fit(X, y)

list(zip(feature_cols, linreg.coef_))


# In[ ]:


#  Same model with dummy variables -- R-squared and adjusted R-squared both improve
statsmodel_summary(bikes_dummies, ['atemp', 'windspeed', 'season_2', 'season_3', 'season_4'])


# How do we interpret the season coefficients? They are measured against the baseline (spring):
# 
# - Holding all other features fixed, summer is associated with a rental decrease of 3.39 bikes compared to the spring.
# - Holding all other features fixed, fall is associated with a rental decrease of 41.7 bikes compared to the spring.
# - Holding all other features fixed, winter is associated with a rental increase of 64.4 bikes compared to the spring.
# 
# Would it matter if we changed which season was defined as the baseline?
# 
# - No, it would simply change our interpretation of the coefficients.
# 
# In most situations, it is best to have the dummy that is your baseline be the category that has the largest representation.
# 
# **Important:** Dummy encoding is relevant for all machine learning models, not just linear regression models.

# Compare adjusted R-squared for various models

# In[4]:


#  This is worst case scenario -- rand should have no predictive value
statsmodel_summary(bikes, ['rand'])


# In[5]:


# This is the full model -- we see an improvement -- only question is whether we could remove certain 
#  variables and improve adjusted R-squared.  Not likely since the "penalty" for adding a variable
#  seems to be low
full_model_columns = ['weather', 'season_2', 'season_3', 'season_4', 'holiday', 'workingday', 'windspeed', 'temp', 'humidity']
statsmodel_summary(bikes_dummies, full_model_columns)


# #### Diagnosing Problems with the Residual Plot
# 
# In a perfect model the residuals represent the noise term $\epsilon$ so residual values should be small, and evenly distributed across values of $y$.   Verifying this is an important part of post-checking your model.
# 
# This is a great overview and set of examples:
# 
# http://docs.statwing.com/interpreting-residual-plots-to-improve-your-regression/#hetero-header

# ---
# <a id="comparing-linear-regression-with-other-models"></a>
# ## Summary:  Comparing Linear Regression With Other Models
# 
# Advantages of linear regression:
# 
# - Simple to explain.
# - Highly interpretable.
# - Model training and prediction are fast.
# - No tuning is required (excluding regularization).
# - Features don't need scaling.
# - Can perform well with a small number of observations.
# - Well understood.
# 
# Disadvantages of linear regression:
# 
# - Presumes a linear relationship between the features and the response.
# - Performance is (generally) not competitive with the best supervised learning methods due to high bias.
# - Can't automatically learn feature interactions.

# In[ ]:




