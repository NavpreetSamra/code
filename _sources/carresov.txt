****************
CAR RESERVATIONS
****************

INTRODUCTION
============

The goal of this analysis to understand what factors drive the total number of reservations made for a vehicle and specifically the impact of keyless smart phone access enabled *technology*.
There are two components to this process

    1. Exploration & Engineering
    2. Analysis & Modeling

To complete these tasks we use :py:class:`pandas.DataFrame` for cleaning and aggregation to build a design matrix in addition to cursory high level analysis.
To further understand the impact features have on reservations we use `statsmodels <http://statsmodels.sourceforge.net/stable/>`_  to perform multiple regression and use elements of models such as relative slopes of hyperplanes and T-tests  to assess the effects on reservations and a final model built with `sklearn <http://scikit-learn.org/stable/documentation.html>`_. 
We conclude with a discussion of what our model could use to improve moving forward. Throughout this discussion we look to better understand how accessibility both with respect to the vehicles themseleves and the postings/quality of the advertisement impact a succesful vehicle.

A quick note on total reservations: 
    This analysis focuses on what drives the total number of reservations. 
    However, this statistic can be misleading and lead to some confounding factors in the analysis because there is not any time based data. 
    As a result, for example, a vehicle that has been available for reservation for 20 weeks and has 10 days of reservation will *appear* stronger than a vehicle that has been available for two weeks and has 5 days of reservations. 
    Rate of reservation and %time reserved out of time available will likely be better metrics moving forward for maximizing the number of reservations. 
    Since we do not have this information we exclude it from the rest of this analysis and assume all vehicles are available for the endpoints in time (it's also worth noting that this is baked into the different reservation types as a car can have multiple hourly reservations over the course of a single day)


DATA EXPLORATION
================

Data is stored in two tables: the first table contains a record of the attributes of each vehicle keyed off of **vehicle_id** and the second table contains a record of each reservation type (hourly / daily/ weekly) for each **vehicle_id**. 
To create a target and assemble a single array of data we create aggregate statistics from the vehicles table and merge it onto the reservations table.
To paint a more comprehensive picture aggregates based on reservation type as well as a boolean field for whether or not a vehicle has been reserved are also created.
While there is logic to leaving reservation type as a continous variable, since there are three distinct classes which monotonically increase in time value, until we find a consistent and monotonic correlation  between reservation type and number of reservations, we convert it to a categorical variable via dummification.
We will not treat this as a multivariate analysis but instead look independently at these targets to better understand the data. 

The first place we look to leverage our existing data is to create the total and percent difference in suggested vs actual price. 
There are two interwoven thought processes to explore here.
Typically, consumers want the best deal possible, as a result if we treat the recommended price as an expected value of the rental then the larger the percentage delta( note this is monotonic, a large in magnitude negative number is *small*) the less likely a car will be selected amongst its *peers* (cars with similar attributes).
This is not the only factor in consumer selection, in many cases consumers want the cheapest possible vehicle that meets her/his needs, which while similar to minimizing *percentage delta* is likely to emphasize less expensive vehicles. 

Yielding a dataset in the form:

============  ==============  ===================  ============  ===============  =============  ===================  =================  ====================  ====================  ====================  ==============
  technology    actual_price    recommended_price    num_images    street_parked    description    totalReservations    hasReservations    reservation_type_1    reservation_type_2    reservation_type_3    deltaPercent
============  ==============  ===================  ============  ===============  =============  ===================  =================  ====================  ====================  ====================  ==============
           1           67.85                   59             5                0              7                    1                  1                     1                     0                     0        0.130435
           0          100.7                    53             5                0            224                    7                  1                     4                     3                     0        0.473684
           0           74                      74             4                1             21                   17                  1                     1                     9                     7        0
           0          135                      75             1                0            184                    2                  1                     1                     0                     1        0.444444
           0           59.36                   53             2                1             31                    2                  1                     0                     1                     1        0.107143
============  ==============  ===================  ============  ===============  =============  ===================  =================  ====================  ====================  ====================  ==============

With a description of:

=====  ============  ==============  ===================  ============  ===============  =============  ===================  =================  ====================  ====================  ====================  ==============
aggr     technology    actual_price    recommended_price    num_images    street_parked    description    totalReservations    hasReservations    reservation_type_1    reservation_type_2    reservation_type_3    deltaPercent
=====  ============  ==============  ===================  ============  ===============  =============  ===================  =================  ====================  ====================  ====================  ==============
count   1000              1000                 1000         1000            1000             1000                 1000             1000                   1000                  1000                  1000           1000
mean       0.17             87.9407              62.206        3.008           0.511           90.792                6.376            0.911                  2.339                 2.057                 1.98           0.259162
std        0.375821         29.7246              16.0825       1.34898         0.500129        76.9486               4.8613           0.284886               2.26345               1.93585               1.85392        0.164317
min        0                32.76                35            1               0                1                    0                0                      0                     0                     0             -0.25
25%        0                64.9425              49            2               0               25                    3                1                      1                     1                     1              0.145299
50%        0                83.93                62            3               1               57.5                  5                1                      2                     2                     2              0.280576
75%        0               107.01                76            4               1              158                    9                1                      4                     3                     3              0.394852
max        1               174.44                90            5               1              250                   25                1                     14                    12                    12              0.5
=====  ============  ==============  ===================  ============  ===============  =============  ===================  =================  ====================  ====================  ====================  ==============

Before applying any statistical or modeling techiques, we split off a third of our data for validation and continue to operate on a subset of the total data. We note that for true predictive modeling we apply the split before inspecting any of the data as any type of scaling or centering (which we have not done) that includes data from the validation set would leak into our training set if it is an aggregate of the superset. 
Here we also look correlations scatter plots between features. The versions of those plots with only the selected features are presented in the next sections, the subselection is for legibility purposes.

ANALYSIS & MODELING 
====================

We begin by inspecting the correlation between design features and the target.  
To combat multicolinearity we drop the actual price of the vehicle as that information is embedded in the *deltaPercentage* field. First we perform an ordinary least squares fit of our predictors on **totalReservations**. 

Statsmodels OLS summary 

================== ========= ========== ========== ========== =======================
Feature                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
================== ========= ========== ========== ========== =======================
technology            2.1788      0.433      5.035      0.000         1.329     3.028
recommended_price     0.0683      0.007      9.841      0.000         0.055     0.082
num_images            1.0955      0.113      9.727      0.000         0.874     1.317
street_parked         0.0604      0.328      0.184      0.854        -0.584     0.705
description           0.0026      0.002      1.211      0.226        -0.002     0.007
deltaPercent         -7.9813      0.951     -8.391      0.000        -9.849    -6.114
================== ========= ========== ========== ========== =======================


The results of the multiple regression demonstrate that street parking and description length are not strong predictors of number of reservations so are dropped moving forward.
This yields a final design based on the total cost(**deltaPercent** & **recommended_price**), perceived value(**deltaPercent**), accessibility of the advertisement (**num_images**, which also stresses that customer care more about pictures then words), and accessibility of the car (**technology**) which all help to understand what a succesful car and advertisement on the platform can be modeled after. 
To get a feel for the data we can look at representations

    * The first few lines of that array
    * A heatmap of the correlation between features in the array (including the target)
    * A scatter matrix for each combination (of choose 2) features

Design :py:class:`pandas.DataFrame` head

============  ===================  ============  ==============
  technology    recommended_price    num_images    deltaPercent
============  ===================  ============  ==============
           0                   59             3        0.375
           1                   40             1        0.342105
           0                   75             5        0.462366
           0                   37             1        0.375
           1                   60             3       -0.149425
============  ===================  ============  ==============


.. figure:: ./images/cars/heatmap.png
    :align: center

`Correlation Heat Map`

.. figure:: ./images/cars/scatter_matrix.png 
    :align: center

`Scatter Matrix`

While technology has an impact on on the total number of reservations which are roughly evenly distributed amongst the three types of reservations, splitting the reservations by type it is clear that technology has a much more pronounced impact on hourly reservations then daily or weekly. It follows logically that both for the car owner and the renter lowering the activation energy required to rent the car and streamlining the process has a pronounced effect when time spent accessing and returning the vehicle compared to total time with the vehicle is a signifigant span of the trip. 
We can see this effect statistically through the correlation of technology with reservation types, ~.3 for hourly reservations and near 0 for daily and weekly reservation types. This is also evident in creating an OLS fit for each of the reservation types, where we can see an inverse relationship between the impact of technology and the length of the reservation.

Hourly Reservation Fit 

================== ========= ========== ========== ========== =======================
Feature                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
================== ========= ========== ========== ========== =======================
technology            2.1532      0.202     10.642      0.000         1.756     2.550
recommended_price     0.0248      0.003      9.027      0.000         0.019     0.030
num_images            0.4148      0.049      8.476      0.000         0.319     0.511
deltaPercent         -3.3617      0.426     -7.896      0.000        -4.197    -2.526
================== ========= ========== ========== ========== =======================


Daily Reservation Fit

================== ========= ========== ========== ========== =======================
Feature                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
================== ========= ========== ========== ========== =======================
technology            0.4902      0.184      2.658      0.008         0.128     0.852
recommended_price     0.0255      0.003     10.201      0.000         0.021     0.030
num_images            0.3266      0.045      7.320      0.000         0.239     0.414
deltaPercent         -2.5996      0.388     -6.698      0.000        -3.362    -1.838
================== ========= ========== ========== ========== =======================

Weekly Reservation Fit

================== ========= ========== ========== ========== =======================
Feature                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
================== ========= ========== ========== ========== =======================
technology            0.2880      0.175      1.649      0.100        -0.055     0.631
recommended_price     0.0224      0.002      9.456      0.000         0.018     0.027
num_images            0.3614      0.042      8.551      0.000         0.278     0.444
deltaPercent         -2.4453      0.368     -6.651      0.000        -3.167    -1.724
================== ========= ========== ========== ========== =======================

To assess the strength of this analysis we look to predict the number of reservations for the vehicle ids in our validation set split off at the beginning of our analysis.
To this end we use sklearn's Random Forest Regressor which, using 40 estimators, we grid serach the minimum number of samples per leaf max splits and number of features to consider.
We train two models one for total number of reservations and one specifically for hourly rentals.
To evaluate feature impacts the RFR uses an ensemble of the *information gain* associated with that feature over all the trees in the model. 


Hourly Reservations Feature Importances 

============  ===================  ============  ==============
  technology    recommended_price    num_images    deltaPercent
============  ===================  ============  ==============
    0.205169             0.141421      0.112395        0.541015
============  ===================  ============  ==============


Total Reservations Feature Importances 

============  ===================  ============  ==============
  technology    recommended_price    num_images    deltaPercent
============  ===================  ============  ==============
   0.0336299             0.122832       0.16406        0.679479
============  ===================  ============  ==============

Similar to the OLS fit the RFR finds the **technology** switch valuable when predicting hourly rentals however it does not find the field useful for total number of reservations like the OLS did.
There are a varietry of reasons this could be due ranging from leverage to the constraints of a hyperplane 

The Random Forest model with best parameters yields an R**2 score of .22, which correpsond to an RMSE~3.6 reservations per vehicle where a vehicle on average has ~6.4 reservations booked.


CONCLUSION
==========

As expected more descriptive ads (assuming a picture is worth ~1000 words) and better perceived value drive decision making across the board.
Vehicle accesibility impact correlates with the percent amount of time that could add to a trip making it more valuable for shorter trips and this manifests directly in the effects **technology** has on driving reservations.
Moving forward the most important piece of data to add is time, as it will provide a critical dimension to the existing data as well as help design key metrics moving forward.
