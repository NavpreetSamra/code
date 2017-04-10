==============
HOUSING PRICES
==============

Introduction
============

Since property often represents a significant portion of an individual’s net worth and, due to changing home values, can be a significant driver of wealth growth the goal of this analysis is to build a model to predict Zillow's current valuations with the attached attributes of ~3000 single family residences in Denver.

The following details:
    
    #. The inspection and exploration process of the available data
    #. Feature engineering and reduction
    #. Modeling 
    #. Evaluation

For information on how to run the model in production please see README

Package Dependies

    * :py:mod:`pandas` version 0.19.2
    * :py:mod:`sklearn` version 0.18.1
    * :py:mod:`statsmodels` version 0.6.1


Conceptually, to build a predictive model, I look to combine two concepts:

    #. Evaluate the unadjusted value of the house (in a vaccuum what is the raw value of the house) 
    #. Modulate that value based on contextual attributes, such as location / neighborhood.


Inspection and Exploration
==========================

After a 20% train test split for model evaulation, the remaining 80% will discussed in this section. The subset of 2700+ houses yields a dataset which has numeric fields described by 


=====  =========  ================  ==========  ===========  ==============  ===========  ============  ==================  =================
Param    zipcode    square_footage    lot_size    num_rooms    num_bedrooms    num_baths    year_built    last_sale_amount    estimated_value
=====  =========  ================  ==========  ===========  ==============  ===========  ============  ==================  =================
count    2733             2733         2733      2733           2733          2733            2733             2733            2733
mean    80091.8           1531.57      7607.18      5.97585        2.82693       2.24277      1946.78        285902          498460
std      2142.13           779.588    19367.2       1.95356        0.859266      1.22701       110.519       977339          353287
min     37134                0            0         0              0             0               0                0           35951
25%     80210              984         5250         5              2             1            1927            87300          278234
50%     80219             1294         6256         6              3             2            1953           192500          396440
75%     80229             1865         7760         7              3             3            1971           341100          579737
max     80249             6098       858571        24              8            15            2016                4.56e+07        3.88669e+06
=====  =========  ================  ==========  ===========  ==============  ===========  ============  ==================  =================

Noting that zipcode is a categorical variable and is not uniformly distributed. Given the small sample size relative to number of homes in Denver this is an obstacle which will begin to be combatted in feature engineering with the note that the training set size is limiting 


While there are no missing fields there are several with a relatively small (up to < 10%) of zeroed out information. Notably square_footage has 7, and last_sale_amount has 202. Also note it's possible to have zero bedrooms (a studio), which will factor into feature selection discussed next.

================ ===============
Field             Missing Values 
================ ===============
address                0
---------------- ---------------
city                   0
---------------- ---------------
state                  0
---------------- ---------------
zipcode                0
---------------- ---------------
property_type          0
---------------- ---------------
square_footage         7
---------------- ---------------
lot_size               3
---------------- ---------------
num_rooms             51
---------------- ---------------
num_bedrooms           4
---------------- ---------------
num_baths              5
---------------- ---------------
year_built             8
---------------- ---------------
last_sale_amount    202
---------------- ---------------
last_sale_date         0
---------------- ---------------
estimated_value        0
================ ===============


Feature Engineering
===================

Because all houses are single-family-residences from Denver, Co neither city, state, nor property_type  are capable of adding information. 
Given the current time allocation and the level of granularity, investigating the address field is tabled for the future.

Inflation Note
    The last_sale_amount data is in nominal dollars. This data was inflated
    with the percent change in CPI via 
    https://fred.stlouisfed.org/series/CPIHOSSL
    attributed to the month of the last sale.

To first get an overview of the 1:1 relationships between features and the target a correlation heatmap is used, presented here including direction, and as an absolute value to highlight magnitude

.. figure:: ./images/housing/corr.png
    :align: center

`Raw Feature Correlation Heatmap`

.. figure:: ./images/housing/corr_abs.png
    :align: center

`Raw Feature Correlation Magnitude Heatmap`

As to be expected, larger houses (sq foot) have more rooms (absolute, bed, bath). Given the stronger pearson and spearman correlations with the estimated value of sq footage than number of rooms (any type) and the strong multi colinearity between the features I move forward with square footage as the representative feature for this component of the model. (Reducing these features by factorization (PCA/SVD/NMF) is another step to explore to improve this attribute, qualified by the fact that distilling missing room information vs a studio also supports using square footage here. 

As visible in the scatter matrix below, the relationship between square footage and estimated value is slightly non linear. To improve this feature gradient descent of the square footage exponentiated correlation with estimated values yields an improved relationship for parametric modeling (will be discussed in the next section). From here square footage will refer to the original value to the 1.81 power and then divided by 1000. (for intepretability, though during modeling a scalar will be applied rendering the divisor moot)

.. figure:: ./images/housing/scatter_matrix.png
    :align: center

Other features explored during modeling but ultimately dropped included seasonality of the last sale (month/quarter/{belongs to quarter to 2|3 vs 1|4}). With more data this could be combined to create a factor to modulate the the last sale value.

While trends are present, they were found to be not strong enough in the training data to contribute in modeling. For example:

.. figure:: ./images/housing/month_vs_value.png
    :align: center
    :scale: 70%

`Last Sale Amount vs Last Sale Month`

As an additional technique to distill signal from noise for primary features statsmodels OLS summary report provides statistics confirming that lot_size and year_built on there own are not directly signifigant at a 95% level. However we expect they may contribute in subsequent analysis.

To attempt to decompose the vaccum from the location based value of a house, the inflated last sale amounts (where non zero) are averaged by zipcode (last sale amount is used and **not** estimated value as this would produce target leakage) and the difference between it's inflated last sale value and it's zipcode last sale averaged value  are the two features leveraged to support square footage.

Features: next steps
--------------------

Other data sources and more housing data will substantially augment the ability to predict housing values. Topic modeling or apply a profiling/ generative clustering technique is critical to improving predictions to teach the model the values associated with location.
Housing value is substantially impacted by location, and incorporating data about the local public schools (teacher/student ratio, school rank) as well other ameneties,  the accesibility, neighborhood, and desirability in addition to the  the rates of changes of these attributes is paramount to building a high quality house price estimator.


Modeling
========

The models discussed are from scikit learn unless otherswise noted

Modeling proceeds with following design matrix

================  =======================     ================
  square footage  inflated value zip diff               zipAvg
================  =======================     ================
         210.929        -208742               325892
         798.039        -221575               367540
         177.286         -88799.2             167156
        2490.91           36811                    1.01184e+06
         360.387         -53460.2             167156
================  =======================     ================


Since less then 10 percent of rows contain missing information from the training set those rows are dropped
as are rows determined outliers by thresholding the magnitude of studentized residual at 2.0  of the  OLS fit (statsmodels) 

Summary of fit for residuals:

X:= [x1=square footage, x2= last sale diff value (diff from zip avg), x3 = last sale zip avg]

 ===== ================ =========== =========== ========= =====================
 X                coef    std err          t     P>|t|      [95.0% Conf. Int.]
 ===== ================ =========== =========== ========= =====================
 x1          2.355e+05   1.19e+04     19.841      0.000      2.12e+05  2.59e+05
 ----- ---------------- ----------- ----------- --------- ---------------------
 x2          4.276e+04   1.09e+04      3.935      0.000      2.15e+04  6.41e+04
 ----- ---------------- ----------- ----------- --------- ---------------------
 x3          1.059e+05   1.14e+04      9.254      0.000      8.34e+04  1.28e+05
 ===== ================ =========== =========== ========= =====================


.. figure:: ./images/housing/residuals_sql_ft_pwr.png
   :align: center
   :scale: 70%

`Outliers(red) detected in OLS fit. Estimated value as function of sq footage**1.8 defines y(x) position. Note red points embedded in blue have a 0 attribute value``

Pipeline
--------

Parametric and non parametric models are hyperparameter tuned via 5 fold cross validated grid search wth coarse to fine grid evolution. 
To prevent target leakage all fitting is performed inside a pipeline including a standard scaler (mean/variance) and imputer to handle NaN values in testing (such as from a new zipcodes) and scored via r**2- note Imputer is good practice here but is effectively as 0 NaN fill post Scaling. To compare models with different features spaces, adjusted r**2 and/or an f-statistic is required to account for bias variance trade off but with a fixed space, moving forward rmse / r**2 make for valid comparators


Parametric
----------

L1 and L2 regularized linear regression produced similar results with a best estimator R**2 of ~.64 during cross validated grid search tuning the regularization weight.


Non Parametric
--------------

Random Forest Regressor and Gradient Boosted Regressor produced similar results tuning max_depth and min_sample_split with 100 estimators (and a learning rate for boosting) producing R**2 scores during cross val of the training data of ~.87 (noting that the number of estimators was not varied as part of cross val and certainly can be, with the caveat that adding estimators will not lead to a Random Forest overfitting but will eventually for Boosting holding all other parameters fixed).

Without a signifigant increase in performace from boosting I chose a Random Forest which benefits from parallizability for improved training times at scale in addition to it's resiliency with respect to number of estimators

Avoided
-------

Support Vectors are better suited to higher dimensional spaces, Neural Networks require substantially larger training sets then available and are also better equipped for higher dimensional spaces. 

Evaluation
==========

Final Model
-----------

======================= =========================
Random Forest Regressor Modification from default
======================= =========================
n_estimators            100
----------------------- -------------------------
max_depth               18
----------------------- -------------------------
min_samples_split       12 
======================= =========================

Best R**2 scored during cross validation .87 or ~ $137000 RMSE (noting a mean housing value of $471000 in the training set)

R**2 score is consistent (.87) for best cross val estimator applied to test set.


Note
    Production model available is trained an entire dataset with hyperparameters specified from grid searching
