*************************
SQL IN THE SEVEN KINGDOMS
*************************

QUESTION 1
==========

.. literalinclude:: ../../src/queries.sql
    :language: sql
    :lines: 1-13



QUESTION 2
==========


ASSUMPTIONS
-----------

1. city_id in events is also foreign keyed to cities.city_id
2. The first trip must occur after a succesful signup, given that a person cannot sign up for uber with same contact information twice and even in that fraudulent case, they should receive a new rider_id.
3. This querry filters by signup location noting that there is ambiguity in the problem statement, if a rider signs up in Winterfell but has a trip in Meereen within the first week of the year that would not be included in this results.
   However if a rider signs up in Qarth and has a trip in King's Landing only within a week they will be included as a success.
   Given the phrasing of the problem statement this seemed like the correct interpretation but it's not 100% clear.


.. literalinclude:: ../../src/queries.sql
    :language: sql
    :lines: 15-39


