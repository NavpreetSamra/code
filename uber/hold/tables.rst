                            Logit Regression Results
================================================================================
Dep. Variable:     first_completed_date   No. Observations:                54681
Model:                            Logit   Df Residuals:                    54669
Method:                             MLE   Df Model:                           11
Date:                  Sun, 02 Apr 2017   Pseudo R-squ.:                  0.5610
Time:                          07:06:12   Log-Likelihood:                -8429.8
converged:                        False   LL-Null:                       -19202.
                                          LLR p-value:                     0.000
===========================================================================================
                              coef    std err          z      P>|z|      [95.0% Conf. Int.]
-------------------------------------------------------------------------------------------
bgc_delta                  -0.1591      0.004    -43.313      0.000        -0.166    -0.152
vehicle_year                4.0271      0.066     61.089      0.000         3.898     4.156
city_name_Strark           -1.7659      2e+06  -8.81e-07      1.000     -3.93e+06  3.93e+06
city_name_Wrouver          -1.7453      2e+06  -8.71e-07      1.000     -3.93e+06  3.93e+06
city_name_Berton           -1.6442      2e+06   -8.2e-07      1.000     -3.93e+06  3.93e+06
signup_os_ios web           0.0783      0.065      1.197      0.231        -0.050     0.206
signup_os_windows           0.3283      0.077      4.278      0.000         0.178     0.479
signup_os_android web      -0.0158      0.068     -0.230      0.818        -0.150     0.118
signup_os_mac               0.4757      0.077      6.199      0.000         0.325     0.626
signup_channel_Paid        -1.9319      2e+06  -9.64e-07      1.000     -3.93e+06  3.93e+06
signup_channel_Organic     -1.8553      2e+06  -9.25e-07      1.000     -3.93e+06  3.93e+06
signup_channel_Referral    -1.3682      2e+06  -6.82e-07      1.000     -3.93e+06  3.93e+06
===========================================================================================
