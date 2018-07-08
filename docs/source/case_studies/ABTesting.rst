===========
A/B Testing
===========

Introduction
============

The following analysis pertains to the funnel of a control {c} and two test variants A {a} and B {b} prescribed by:


====     ==========  ================  ================  ============
flow       visitors    createdAccount    addedEmployees    ranPayroll
====     ==========  ================  ================  ============
c            5005              3654              2856          2055
a            5012              3838              3073          2255
b            5025              3987              2974          2070
====     ==========  ================  ================  ============

It assumes
visitors were randomly assigned to each flow and that the distribution of members within flows are similar (i.e. as an extreme 
example of one potential aspect: the control has companies > 100 people, a has companies 50-100, and b has companies < 50)

To analyze the end to end impact {visitors_ranPayroll} as well as the stages in between, I constructed a table leveraging a Bayesian approach
to compare the variants, as well as each variant to the control for each combination of stages in the funnel.

The values of of the table are the results of leveraging the beta distribution to establish the percent likelihood (divided by 100) that i outperforms j for the step X_Y in the table (rounded to 4 decimal places).
In summary, at the extremes and mid point:

    * A value near 1 implies strongly {i} outperforms flow {j} between steps {X} to {Y}
    * A value near 0 implies strongly {j} outperforms  flow {i} between steps {X} to {Y}
    * A value near .5 implies {i} and {j} performed similarly between steps {X} to {Y}

=================  =========================  =========================  =====================  ===============================  ===========================  ===========================
Comparison (i, j)  visitors_createdAccount    visitors_addedEmployees    visitors_ranPayroll    createdAccount_addedEmployees    createdAccount_ranPayroll    addedEmployees_ranPayroll
=================  =========================  =========================  =====================  ===============================  ===========================  ===========================
('a', 'b')                                0                     0.9864                 1                                1                            1                            0.9994
('a', 'c')                                1                     1                      1                                0.979                        0.986                        0.8866
('b', 'c')                                1                     0.984                  0.5472                           0.0002                       0.0002                       0.0202
=================  =========================  =========================  =====================  ===============================  ===========================  ===========================

code docs at :py:func:`gusto.src.ab_testing` (source available in src folder and viewable through green hyperlinks in docs)


The goal of improving conversion rate in a funnel should always look to widening the funnel at all possible points, with an additional eye to 
catching any potential choke points or steep drop offs between levels. There are no severe chokes in the funnel so moving forward I will focus on the impact of the variants 
of the flows rather than stages of the funnel. 

With the exception of moving from Visit to Creating an Accounts, {a} outperforms both {b} and the control {c} at all other stages of the funnel and most importantly
the end to end test. As a result, I would recommend if a flow were being pushed back to full production without any change possible **{a}** should be pushed.
However, depending on the overhead and engineering requirements (and the setups of the flows)  I would also recommend trying to incorporate features from flow
B for between Visit and Creating Account to try and optimize each stage of the funnel. 

Different types of companies, based on attributes-, including but not limited to size, location, industry- may respond differently to different flows. Analyzing
the success of different flows (and stages of flows) for different types of companies to optimize the success rate could enable custom flows based on customer attributes (again
depending on engineering requirements and scalibilty trade offs). 

In addition, tracking the results of this test to monitor for any differences in stickiness
between the groups is critical. If {a} performs the best during onboarding but has a higher churn rate it could be a sub optimal flow for the goal of growing
Gusto. While it is unlikely that the onboarding flow creates a negative self selection of high churn rate clients, it's always important to be mindful of these effects, because no two companies are truly *identical*, let alone a pool of companies.
