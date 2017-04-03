***********
A/B TESTING
***********

=======
PREFACE
=======

One of the most powerful assets in a new A / B test for a company that has a wealth of A / B test results is leveraging those tests.
Without that or an analysis of the available driver data, the goal of implementing an A / B test is somewhat hamstrung but we will proceed cognizant of the fact we are pretending to be in vacuum whenever possible.

On a statistical note: testing multiple changes simultaneously reduces the ability to accurately evaluate each of the new feature's impact. However, because it can add substantial complexity to engineering it may not be feasible, tractable or valuable to require this rigor. For the purposes of evaluation we proceed from a metric perspective of an A control and a single treatment B, with the caveat that we can try to isolate the effects individual components of the new app in the data that is collected.

*BAYESIAN* VS *FREQUENTIST* 
==========================

In A / B testing, when designed and implemented correctly and equivalently they will generate the same result.
The decision to use one over the other should come from the ease of implementation and evaluation. 
For completeness both will be discussed in the following. The :ref:`test_plan` section will touch on the differences and restrictions of setup and execution between the two.

On a personal note, my favorite quote on the topic:  
    "In many cases this debate is the same as arguing the style of the screen door on a submarine. It’s a fun argument that will change how things look, but the very act of having it means that you are drowning." - Andrew Anderson

=======
METRICS
=======



    =========== ==================================
    Definitions
    =========== ==================================
    Driving     time spent driving a passenger
    ----------- ----------------------------------
    Available   time spent waiting for a passenger
    ----------- ----------------------------------
    Active      available + driving
    =========== ==================================

The app redesign has two goals

    1. Increase the supply of active driving hours / (driver * unit time)
    2. Increase the effeciency of drivers :=  driving / active

While the Uber ecosystem is complex, as are the business impacts the Partner App has, this discussion will focus on effects the Uber Partner App redesign has on the supply of driving time as it's primary goal and the effeciency of drivers as the secondary, the order can easily be switched and should be a function of where business objective sits on the spectrum of growth - effeciency.

GROWTH
======

There are multiple useful evaluation metrics but the primary one proposed here is the difference in **lift** in active time the two groups experience.
Comparing the relative rates of change for the drivers helps to control bias for external factors such as weather, holidays and events, as well as factors internal to the driver distributions such as comparing drivers who typically drive for different amounts of time. 

EFFECIENCY
==========

The first supporting metric is the lift in in driving time / active time. Improving communication and by making surge areas and other features available in the app, the redesign hopes to more effeciently put drivers in the correct place or push them towards surging areas. 
A profit based metric is also reasonable here if goals are more profit then growth focused.

USAGE
=====

The second support metric is the lift in app usage rate / active time.
This metric aims to help understand the more immediate reactions drivers have to the app and address whether that encourages or discourages driving, given that one of the goals is to encourage drivers  by giving immediate feedback. As well as provide better information through communication such as when a surge is happening (or potentially/eventually when a surge is predicted to happen). NB: since there is more to explore in the new app, this metric should require a larger margin to be considered succesful.

USER EXPERIENCE
===============

The third secondary metric is the rating from surveying the drivers in the test group on whether they prefer the new app (and by how much based on relative rating). This metric gives a direct channel to the user's experiences.

.. _test_plan:

==========
EVALUATION
==========

FRAMEWORKS
==========

BAYESIAN
--------

Without previous data and studies a uniform prior is a reasonable place to start. The probability that B is superior to A is evaluated via the beta distribution where success is the boolean of meeting the lift criteria. This can be a simple greater than, a jump/shock- greater then some amount- or vary among topics /personas.

Whereas the frequentist approach requires a predefined experiment scope, we can cut off the Bayesian experiment when the derivative of the variance in the normalized difference between A and B stabilizes and or credibility intervals settle. 

Bayesian results tend to be interpreted more correctly by diverse audiences. 
Bayesian models benefit from not requiring predefined experiment time frames. This is more helpful in a vacuum but loses some of its value as the number of similar of A / B tests available grow and an understanding what type of gain over a period of time and sample size should be expeceted in the ecosystem.

The evaluation (iterations to decision) is impacted by the initial prior so a poor understanding of the initial distribution can elongate the amount of time a Bayesian test requires and is one of the reasons that a local rather than absolute extrema can be reached.

Conversely a good initial prior can achieve a result quickly.

FREQUENTIST
-----------

The goal of a Frequentist study is prove the difference or lack there of between a parameter generated from two different sets of data, which lends itself to more definitive language in reporting at the sacrifice of flexibility in evaluating the test.
If  a real time / in the loop model and prediction is required, a Bayesian framework may not be possible due to computational expense. 


Frequentist evaluation of A/B Testing requires affirming 2 components
    
    * Signifigance(1 - `alpha`): Is A (> | ~= | <) B given a required confidence level ?
    * Power (`Pi`): Is the study indicative of a signifant difference in the distributions?

The best first step here is to likely leverage the previous Uber A \ B tests and compare the predicted results of the test with the acutal results when B is rolled out.
In a vacuum a 95% Signifigance and an 80% Power are fairly common and typically reasonable thresholds for A/B testing in web/mobile frameworks. 

MARGINS
=======

Bounding / Margin Interval: How different do A and B  need to be for it to be worthwhile to transition from A to B factoring in all external costs including opportunity.
Margins require more information about the surrouding enviroment.
For example if the margin is small, or even insignifigant and A~=B, but introducing the design enables a better/smoother transition to the next development and/or collecting data such as touch point and button clicks is valuable then it may be worthwhile to roll out the new app.
On the opposite end of the spectrum, if there's a large opportunity cost (on servers/ data stores/ maintenance) then the margin needs to be wider for it to be worthwhile even if A > B and the study proves powerful. 

VACUUM
======

Given a lack of data or experience predicting driver response I would employ a Bayesian framework and closely monitor the the credibility intervals, as well as crash reports of the app in addition to the defined metrics. I expect crash reports to be infrequent enough that they may need a longer time scale to be properly differentiated unless the underlying distributions are massivley different (but this is pure conjecture). I would set a lower bound of two weeks as the minimum reasonable test time for the initial kernel test population - as discussed below- given that I expect a roughly weekly cadence, less than two cylces seems overly risky. 

========
ROLL OUT
========

To balance statistical rigor with minimizing regret (the cost of not 100% utilizing the optimal app) we look to a Bandit based approach. There are multiple valid approaches but a modified discretized epsilon greedy(adaptive based on value differences) growth enables the test group to dynamically respond. 


SAMPLING
========

Since we will begin with a small (B << A) sample of drivers and the supply of drivers is large if we random sample from the population we risk creating a test group that does not remotely represent the true distribution of drivers.
We assume driver persona profiles exist (based on various attributes, location, car type, avg number of hours driven / week etc) and we stratify / partition along those attributes and randomly sample within to pull candidates for testing such that we create a test population that resembles the entire population.
If they do not exist, personas can be generated from topic modeling as well as additional clustering and profiling techniques.

SCALE
=====

We define a minimum required percent increase (*C*) and maximum percent increase (*C* + `epsilon`) of drivers to be added in each of the discretized blocks (discretizing by time, likely weekly, bimonthly, or monthly depending on the periodicity of driving hours) number proportional to the success metric (C + `epsilon` * % positive `delta` (Test Population) * size(Test Population). This enables the growth of users to be guided by a predefined slope path but modulate by the success the new app is showing to have or not to have. (Note also C and `epsilon`  can be defined piecewise to break the study into multiple phases)



CUT OFFS
========

The scaling profile described is the path from beginning to end as an exclusive interval. We begin with an initial kernel population and allow our growth model to develop the test group up to the point where the study ends per the above criteria and we roll back or push the redesign  to full production. The size of the initial kernel is dependent on what stage the redesigned app is at. Has it had an alpha phase (assuming this is a beta) how extensively has it been tested, what level of unhapiness or frustration is acceptable in users if acceptable / how much risk are we willing take on to expedite this process and how do we want the penatly as a function of risk response to look like. A single digit percent is a reasonable place to start but it is defensible to be much smaller or a marginally larger depending on confidence, past experience, risk, and business issues. 
