********
MODELING
********

=======
PREFACE
=======

Packages:
    * :py:mod:`pandas` version 0.19.2
    * :py:mod:`sklearn` version 0.18.1
    * :py:mod:`statsmodels` version 0.6.1

Process: after splitting off 20% of the data with stratified sampling for testing data
the remaining 80% is used for exploration and analysis.
The cleanliless (existence + validity) of the data is explored followed by feature engineering.
High level correlation analysis is preformed to understand relationships in the data followed by Logistic Regression via statsmodels because it has a cleaner summary interface than scikit-learn.
With design specified Logistic Regression and a Random Forest Classifier from scikit-learn are grid searched for hyperparameter tuning to implement the final model. Scaling and imputing is preformed inside a Pipeline to prevent target leakage. Variance Inflation Factor is evaluated for the design matrix before training at any stage.

========
CLEANING
========

EDA
===

Exploration begins with evaluating the phase space of the availabe data. The data follows the taxonomy closely although there are a few missing categories, which could be due to the train test split or the abscence of existence in this slice of data.
While the abscence is not particularly important to the rest of this analysis it serves as a reminder to build an extensible framework that is easy to adapt as more features and responds to  more categories becoming present  over time.


The percent of succesful first trips is evaluated as the percent of non NaN members of the first_completed_date field. Uniqueness of the ids was confirmed. Yielding **11.2%** succesful conversion.  


As expected a conversion funnel is present. 

Only 60% of prospective drivers complete background checks and only 24% have vehicle information sent in.

Note that 12% of signup_os identifiers are missing and it is unclear why (there is not a concentration in sign up date associated which likely rules out a widespread logging issue)


Vehicle Make
------------

It's unclear if this data is strictly for Uber auto, which if it is would likely prevent *bicycle*  *bike* vehicle makes from participating. Given that at at least 1 bike / bicycle had a succesful first trip it is either likely an incorrect vehicle make entry or potentially part of the Uber Eats or another non auto program. 

As well, there is also *walker* and *autobot* vehicle make.

It's possible there is a new or small auto manufacturer, but a quick google search did not find any.
With more information or a picture, which could help differentiate a new auto make, from a walking aid from an AT-AT walker or AT-ST walker that escaped from  LucasFilms/ Lucas Ranch, as well  for the autobots, but for this analysis these values will be treated as NaNs.

Unfortunately with limited time I was not able to taxonomize the vehicle make and model, but moving forward it could be a very valuable feature.
To best use it likely requires at least profiling and preferably topic modeling makes and models  of cars- sedans vs suv (XL eligible) and / or luxury vehicle (eligible for Black etc).

Vehicle Year
------------

There is at least one abnormal vehicle year (0) which is converted to NaN in cleaning via floor thresholding (all values below the floor are converted to NaN here 1990 is used)


FEATURE DEVELOPMENT
-------------------

Because we are modeling a conversion funnel we expect the features that represent the transitions between stages in the funnel to be primary features(vehicle information, background check information), and then hope to gain insigt into user experience through representative attributes (location, signup process).

The first two primary features developed are the difference between the signup date and the background (*bgc_delta*) and a boolean for the existence of vehicle year. Data points without a background check date are penalized and replaced with 90 days for *bgc_delta* (max seen in the training set is 69, a max imputation is reasonable here, but 3 months and the extra time as a penatly for members who do not have a background check yet is reasonable. However, this does have implications on when the model can be run best (it can only include signups more then 90 days old).

In addition to vehicle year boolean, the vehicle year existing feature will have NaNs imputed during modeling, it appears a relationship exists though it's unclear yet if it's present enough to add weight when compared to the boolean field given the large number of imputations that will be present .  

.. figure:: ./images/vehicle_year_cont_success.png
   :align: center

`For vehicles that have a model year recorded, the relative number of successful conversions in the training set`

ANALYSIS
--------

To begin looking for relationships with the data pandas built in :py:func:`pandas.tools.plotting.scatter_matrix` is leveraged as is seaborns heatmap to provide magnitude and direction + magnitude of the correlation between the available features and with the target. Unfortunately the total scatter matrix is not easily presentable as an image in this pdf but it is included in the supplemental materials and referenced in the html version of this document linked at the top. The axis for the heatmaps from top to bottom and left to right are as follows:

======================= =
FEATURES
======================= =
bgc_delta
----------------------- -
vehicle_year
----------------------- -
city_name_Strark
----------------------- -
city_name_Wrouver
----------------------- -
city_name_Berton
----------------------- -
signup_os_ios web
----------------------- -
signup_os_windows
----------------------- -
signup_os_android web
----------------------- -
signup_os_nan
----------------------- -
signup_os_mac
----------------------- -
signup_os_other
----------------------- -
signup_channel_Paid
----------------------- -
signup_channel_Organic
----------------------- -
signup_channel_Referral
----------------------- -
signup_channel_other
----------------------- -
first_completed_date
======================= =

Note categories are dummified without dropping members yet.

.. figure:: images/corr_heatmap.png
    :align: center

`Correlation heatmap`

.. figure:: images/corr_heatmap_abs.png
    :align: center

`Absolute value correlation heatmap`


Given the weak relationship confirmed by Logistic Regression the location information is dropped. The signup os is dropped as well. The signup os is more informative then the location and can provide insight into evaluating the success of web vs mobile (and within mobile) but are not as relevant as the primary features. The signup channels appear to provide a stronger signal and will be maintained for now. 

=======================  ======================
Feature                  Target Correlation 
=======================  ======================
bgc_delta                           -0.342966
vehicle_year_bool                    0.593979
city_name_Strark                    -0.00909867
city_name_Wrouver                   -0.0202788
city_name_Berton                     0.0215312
signup_os_ios web                    0.0408464
signup_os_windows                    0.0241797
signup_os_android web               -0.0290179
signup_os_nan                       -0.108742
signup_os_mac                        0.0552828
signup_os_other                      0.0205659
signup_channel_Paid                 -0.140678
signup_channel_Organic              -0.0399715
signup_channel_Referral              0.187017
first_completed_date                 1
=======================  ======================


=====
MODEL
=====

Two scikit-learn models are evaluated 1 parametric - Logistic Regression - and 1 non-parametric - Random Forest.

Logistic Regression's coeffeient interpretability makes it a great tool to effeciently derive useful insights about the impact features have on the target. Random Forests are able to convey relative importance about the feature space by ensembling the information gain across the splits in the trees, scale, well and have seen widespread success in a variety of applications. 

THINGS TO AVOID 
    Neural Nets are far better suited for high dimensional spaces and lack interpetability. SVMs are not particularly well suited for class imbalances and would require over/under sampling at the very least to get started.

Boosting our trees is reasonable and an XGBoost would be the next model to test however, the parallelizability of an RF when adding more estimators is lost when we turn to boosting. 

Rather than turning to another model, the likely best next step would be resample our data, at least SMOTE, if not SMOTE + undersampling to combat the 1:9 class imbalance present in the data. Here we poorly fake this through *class_weight* which penalizes a model for incorrectly classifying the underclass at the inverse of it's presence in the data in the Logistic Regression.


Hyperparameters are tuned via 5 fold cross validation. An 'l2' regularization penalty is applied to Logistic Regression with log spacing (NOTE in sklearn, the penalty term is the inversely proportional to the weight) 

To tune hyperparemeters we use f1 score which averages precision and recall as another nod to class imbalance and our focus of predicting the underclass.

To evaluate our model success we turn to a receiver operator characteristic (ROC) curve. The location of the threshold depends again on business objectives and applications but should be near after the elbow. As a metric for evaluation we use area under the mean curve of cross validation. 

FINAL MODEL
===========

There are plently of knobs to turn in hyperparameter tuning. To get a sense of what trees are likely to be good
estimators to get started with we change the following default arguments according to the best hyperparameters found during cross validation.

Final Design Matrix
-------------------

===========  ===================  ========================  =====================
  bgc_delta    vehicle_year_bool    signup_channel_Organic    signup_channel_Paid
===========  ===================  ========================  =====================
         90                    0                         0                      1
         90                    0                         0                      1
          0                    0                         1                      0
          5                    1                         0                      0
         15                    1                         0                      0
===========  ===================  ========================  =====================


======================== =======
RANDOM FOREST CLASSIFIER
======================== =======
n_estimators             60
------------------------ -------
max_depth                3
------------------------ -------
min_samples_split        40
======================== =======

Best cross validated f1 score during hyperparameter tuning on training set: .70

ROC AUC of Best Grid Search CV estimator on test set: .92

==================
BUESINESS INSIGHTS
==================

Like any funnel analysis increasing width through the funnel will always drive conversion and conversion rate. While the available data indicates some strong correlations that can help understand driver conversion, ultimately Uber's ability to impact the causality is relegated to secondary impact: **activation energy**. We cannot make peope want to drive, but we can reduce the energy to traverse the funnel which will hopefully convert some borderline potential drivers as well as make Uber the most attractive option for potential drivers to join the ride sharing economy. Given our model we expect to increase conversion and conversion rate by making it as easy and simple as possible for potential drivers to sign up, complete background checks, and get his or her vehicle inspected. We can encourage drivers to complete the steps as quickly as possible, however again this is likely a correlated not a causal effect. 
