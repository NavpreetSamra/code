PREDICTING SHOPPER TIME
=======================

INTRODUCTION
------------

The goal of this analysis is to better understand the expected time a shopper will take inside a grocery store as a function of trip and order attributes. These two types of attributes are each organized in their own table, the first containing information about the trip including the shopper id, store id, fulfillment model, and start and end shopping times spanning a 10 week window. The second contains information about each order linked back to the trip id via a foreign key including the item id, the quantity associated with that order, and the department name of that item at that store.  To achieve our goal we hope to integrate two expected phenomena to create our model. The first is based on the size of the order, where we expect larger orders, orders with a larger number of different products, and/or with more difficult to find or complex items will require more time in the grocery store. The second is based on attributes of the trip including time of day, day of the week, location, and shopper. We hope these features combine to describe the different axis on which trip time varies. 


Trip Table 

===========  ============  ===================  ==========  =====================  ===================
    trip_id    shopper_id  fulfillment_model      store_id  shopping_started_at    shopping_ended_at
===========  ============  ===================  ==========  =====================  ===================
3.11952e+06         48539  model_1                       6  2015-09-01 07:03:56    2015-09-01 07:30:56
3.11951e+06          3775  model_1                       1  2015-09-01 07:04:33    2015-09-01 07:40:33
3.11952e+06          4362  model_1                       1  2015-09-01 07:23:21    2015-09-01 07:41:21
3.11979e+06         47659  model_1                       1  2015-09-01 07:29:52    2015-09-01 08:55:52
3.11992e+06         11475  model_1                       1  2015-09-01 07:32:21    2015-09-01 09:01:21
===========  ============  ===================  ==========  =====================  ===================

Orders Table

===========  =========  =================  ==========
    trip_id    item_id  department_name      quantity
===========  =========  =================  ==========
3.11951e+06     368671  Produce                    10
3.12046e+06     368671  Produce                    10
3.12047e+06     368671  Produce                    10
3.12191e+06     368671  Produce                     6
3.12233e+06     368671  Produce                    10
===========  =========  =================  ==========

CLEANING
--------

The first step in our data cleaning is to create a target. We convert the start and end timestamps into a difference of seconds. Moving forward, we need to be very careful if/when using time data in the design matrix to combat target leakage. In addition to processing the categorical and string variables from our trip table we need to transform our order table because our target is keyed off trip id. 

From the trip table there are 4 pieces of information we garner. The shopper, store, day of week, and time of day. By dummying the store id and hour of day and creating an on/off switch for weekend vs weekday we look to understand what stores and times are busier and therefore likely to slow down our shoppers. From the orders table we aggregate the number of items (*total**) and number of types of names by department name (**counts**) by trip id to join with our trip data. In the future we can look to analyze stores and department codes to both better understand which items in general and store specific cause more time to collect. We also note that the distribution of orders per department type has a long tail, which is likely due to varying point of sales systems across stores. Mapping stores internal POS system names to standardized forms could help our model and is an interesting area to explore in the future. 

.. figure:: ./images/shoppers/value_counts.png
   :align: center

`Number of instances of each department name in orders table`


The initial design matrix includes information about the shopper, fulfillment model, time of day, day of week, order size and distribution

===========  ============  =======  ========  ===========================  ============  ============  ============  =============  =============  =============  =============  =============  ==============  ==============  ==============  ==============  ==============  =======  =======  =======  ========  ========  ========  ========  ========  ========  ========  ========  ========  ========  ========  ========  ========  ========
         ..    shopper_id    total    counts    fulfillment_model_model_2    store_id_3    store_id_5    store_id_6    store_id_29    store_id_31    store_id_54    store_id_78    store_id_90    store_id_105    store_id_115    store_id_123    store_id_126    store_id_148    dow_1    hod_8    hod_9    hod_10    hod_11    hod_12    hod_13    hod_14    hod_15    hod_16    hod_17    hod_18    hod_19    hod_20    hod_21    hod_22    hod_23
===========  ============  =======  ========  ===========================  ============  ============  ============  =============  =============  =============  =============  =============  ==============  ==============  ==============  ==============  ==============  =======  =======  =======  ========  ========  ========  ========  ========  ========  ========  ========  ========  ========  ========  ========  ========  ========
3.19403e+06         10214    69           58                            1             1             0             0              0              0              0              0              0               0               0               0               0               0        1        0        0         0         0         0         1         0         0         0         0         0         0         0         0         0         0
4.2264e+06          52156     7            7                            0             0             0             0              0              0              0              0              0               0               0               0               0               0        1        0        0         0         0         0         0         1         0         0         0         0         0         0         0         0         0
4.05705e+06         61890    15            9                            0             1             0             0              0              0              0              0              0               0               0               0               0               0        1        0        0         0         0         1         0         0         0         0         0         0         0         0         0         0         0
3.60157e+06         46414     3            2                            1             0             1             0              0              0              0              0              0               0               0               0               0               0        1        0        0         0         0         0         1         0         0         0         0         0         0         0         0         0         0
3.15143e+06         17972    49.75        36                            1             1             0             0              0              0              0              0              0               0               0               0               0               0        1        0        0         0         0         0         0         0         1         0         0         0         0         0         0         0         0
===========  ============  =======  ========  ===========================  ============  ============  ============  =============  =============  =============  =============  =============  ==============  ==============  ==============  ==============  ==============  =======  =======  =======  ========  ========  ========  ========  ========  ========  ========  ========  ========  ========  ========  ========  ========  ========


.. figure:: ./images/shoppers/heatmap.png
   :align: center

`Correlation Heatmp, features share header with previous table. Target is appended as last field in heatmap`

Here we note that there is a strong relationship between **counts** and **total** which is not suprising and exhibited by a Variance Inflation factor over 6.

MODELING
--------

In our model evaluation process we begin by splitting off a validation set so that we can look to quantify the expected performance. The estimator we utilized scikit-learn’s Random Forest Regressor with mean square error cost. With more time we could explore other regressors, but with a grid search and minimal time to engineer features we expect the random forest to outperform a Linear Regression (feature engineering restricted by time) or a SVM (limited due to run time). This model uses 5 fold cross validated grid search varying *max_depth* *min_samples_split* and *max_features* from coarse to fine. n_estimators of 40 was selected as a coarse initial analysis began to show diminishing returns with more trees (likely close to the “elbow”, this is another area that can be investigated more moving forward)

Feature Importances

==========  ==============  ==============  ===============  ===============  =========================  ===============  ================  ================  ===============  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  =================  =================  =================  =================  =================
feature     counts          shopper_id      total            store_id_3       fulfillment_model_model_2  store_id_29      store_id_5        hod_18            hod_11           hod_17            hod_10            hod_12            hod_16            hod_15            hod_14            hod_9             hod_13            dow_1             hod_8             dow               hod_19            store_id_31       store_id_105      hod_20            store_id_90       store_id_54       store_id_6        store_id_126      store_id_115      store_id_123      hod_21             hod_22             store_id_78        store_id_148       hod_23
==========  ==============  ==============  ===============  ===============  =========================  ===============  ================  ================  ===============  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  =================  =================  =================  =================  =================
importance  0.583085973354  0.168116970532  0.0871047770267  0.0231925758538  0.016009218889             0.0103891225025  0.00918909750058  0.00643181576237  0.0064068927097  0.00637174237289  0.00618338305612  0.00603053471784  0.00600951461267  0.00582783405928  0.00573174335489  0.00524512930841  0.00523016631657  0.00497156611087  0.00479333186756  0.00429676440816  0.00420139433796  0.00397937369493  0.00389730224653  0.00295957800954  0.00280070930962  0.00240492820493  0.00217471206897  0.00204697658013  0.00182832893063  0.00155537625524  0.000911880839141  0.000252784072627  0.000232256268709  0.000134717018234  1.52784674318e-06
==========  ==============  ==============  ===============  ===============  =========================  ===============  ================  ================  ===============  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  ================  =================  =================  =================  =================  =================

DISCUSSION
----------

Design Data Frame

===========  ============  =======  ========
    trip_id    shopper_id    total    counts
===========  ============  =======  ========
4.22546e+06         62434     67.5        50
3.90618e+06         59715     21          19
3.77706e+06          3746      6           6
3.80646e+06         44234     49          39
4.24162e+06         50326      8           6
===========  ============  =======  ========

`Final Design`

Analyzing the feature importances across our models shows consistently an order of magnitude more information gain associated with the shopper, the total count of items in the cart features, and the number of unique departments in the cart compared to the rest of the features. As a result  day of week, time of day, and store locations are excluded. The number and types of items impact on shopper time is as expected. The shopper id impact is more interesting, one potential explanation here is that if shopper ids are assigned monotonically increasing in time resulting in more experienced shoppers who know the stores better or have had success and continue shopping contribute to a being more efficient and therefore faster, especially compared to the scenario of a first time shopper in a new store. A feature to investigate here (noting that the 10 week window may limit the information in this data set) is the number of trips a shopper makes and how that evolution through time looks. It's worth noting here that the variation in shopping hours over the course of the day is normal over the course of the day. While it’s possible that time of day and location do not influence the shop time, this is an area which could be explored more moving forward and likely to the benefit of the model. 

.. figure:: ./images/shoppers/hours.png
   :align: center

`Histogram of trips made per hour of day`


CONCLUSION
----------

Our parameters from grid search yield using 40 estimators in the Random Forest Regressor

Parameter Grid:
    * 'max_depth': 50,  (Note in multiple training sesssions the best estimator holding other estimators optimal xceeded a depth of 50)
    * 'max_features': 'auto', 
    * 'min_samples_split': 50
    
Yielding an **R**2** score of **.36**  when applied on the validation set and an **Out of Bag Error** from the fully fit model of **.37** or ~19 minutes mean error per trip where an average trip in the training data is ~41 minutes. The model is rebuilt with the validation set and fit to generate the test datat predictions.
