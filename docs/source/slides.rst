ROUTE VISION
============

**REDUCING THE NUMBER OF EXPLORATIONS REQUIRED TO FIND THE OPTIMAL PATH
FOR POINT TO POINT SEARCHES IN VELOCITY GRAPHS**

MOTIVATION
----------

* A* costs grow geometrically
* Hierarchal methods cannot dynamically update edge weights

.. figure:: ..//images/uber.png
   :align: center
   :scale: 40 %

`Uber blog link <https://eng.uber.com/engineering-an-efficient-route/>`_

FORMULATION & CONSTRAINTS
-------------------------

* Standard A* heurisitc fails to leverage available local latent data to aid graph traversal
* Heuristic can only utilize O(1) operations 

**Goal**: Constrain search growth by improving prioritization

* We embed computationally inexpensive & local data from our graph in the heuristic to **communicate** information.
* This gives A* the potential to reduce explorations without recomputing the entire graph by giving it **vision**

.. figure:: ..//images/Astar_progress_animation.gif
   :align: center
   :scale: 95 %
   
   Geometric growth of A*

SCALAR FIELD
------------

Scalar representation of the local velocity field as a perturbation in heuristic

.. math::
  :nowrap:

   \begin{eqnarray}
      f(n) & = &  g(n) + {\epsilon}h(n) \\
      g(n) & = & cost_{time}(n) \\
      h_{A^*}(n_{i+1}) & = & min_{target}[cost_{time}(x_{ni+1})] \\
      h_{Av^*}(n_{i+1}) & = &  min_{target}[cost_{time}(x_{ni+1},v_{ni+1})]
   \end{eqnarray}


|

.. image:: ../images/map_traffic.png
   :align: left
   :scale: 40 %

OBSTACLE NAVIGATION
-------------------

**Slow Zone**
    * Uniform field with embedded homogenous zone(10% field velocity) and linear deceleration on boundary

.. figure:: ..//images/obstacle_graph.png
   :align: center
   :scale: 50 %

TRAVERSAL
---------
Finding best route with optimal heuristic weighting around obstacle requires double the explorations without embedded scalar field!

.. figure:: ..//images/obstacle_comp.gif
   :align: center
   :scale: 60 %

========== ============ =================== ==============================
METHOD     A* 46 CYCLES                     A* WITH SCALAR FIELD 23 CYCLES
========== ============ =================== ==============================

OBSTACLE APPROACH
-----------------
Long approach of **Slow Zone**

.. figure:: ..//images/long.png
   :align: center
   :scale: 50 %


TRAVERSAL
---------

Note the corner

.. figure:: ..//images/comp_long.gif
   :align: center
   :scale: 65 %

====== == ======================  =============================
METHOD A*                         A* WITH EMBEDDED SCALAR FIELD
====== == ======================  =============================

VARIABLE FIELD
--------------

Harmonic Profile

* Variable velocity field (min at 5% Vmax) 

.. figure:: ..//images/harmonic.png
   :align: center
   :scale: 50 %

TRAVERSAL
---------

.. figure:: ..//images/comp_harm.gif
   :align: center
   :scale: 50 %

====== ================== ====== ===============================
METHOD A* 247 CYCLES             A* WITH SCALAR FIELD 103 CYCLES
====== ================== ====== ===============================

ANALYSIS
--------

Tighter profile and demonstated phobia of slower velocity zones

.. figure:: ..//images/harmonic_comb_end.png
   :align: center
   :scale: 75 %

====== ================== ====== ===============================
METHOD A* 247 CYCLES             A* WITH SCALAR FIELD 103 CYCLES
====== ================== ====== ===============================
    
ANALYSIS
--------

* 2 Nodes directly North of source are never explored!

.. figure:: ..//images/two_nodes.png
   :align: center
   :scale: 75 %

====== ================== ============ ===============================
METHOD A* 247 CYCLES                   A* WITH SCALAR FIELD 103 CYCLES
====== ================== ============ ===============================

CURRENT WORK \: VECTOR FIELD 
----------------------------

.. math::
  :nowrap:

   \begin{eqnarray}
      k & = & kernel \\
      \vec{v_{ni+1}} & = & w(v_{ni+1}, \vec{x_{ni}}, \vec{x_{ni+1}}, k_{r}) \\
      h_{Av^*}(n_{i+1}) & = &  min_{target}[cost_{time}(x_{ni+1},\vec{v_{ni+1}})]
   \end{eqnarray}

A field can be defined where each dimension corresponds to a direction
    * Method is memory intensive, but memory is cheap and does not violate our constraints
    * Enables propogation of information via a kernel


.. figure:: ../images/layers.png
   :align: center
   :scale: 60 %


FUTURE WORK \: COMPLEXITY
-------------------------
This framework spawns two machine learning problems
    * Prediction of state
        - Weight of an edge when we expect to traverse
        - historical & shock propogated
    * Kernel update
        - size & shape

.. figure:: ../images/Stationary_velocity_field.png
   :align: center
   :scale: 80 %

SCALAR STABILITY & ROBUSTNESS
-----------------------------

* Cost bounds
* Cycle behavior

Testing framework utilizes harmonic profile with 4 points near mean field velocity
on boundary and 4 internal points distributed across two minimums and 2
maximums. The resulting 56 permutations yield a substantial phase space for
analysis and insight into what issues need to be addressed.

For example, ending in a local minimum led to an increase in the number of cycles
required under certain conditions, especially if the path followed close to the gradient of the 
velocity surface (as detailed in the next figure).

In response a smoothing of the perturbation was implemented, a linear smoothing
was originally implemented which improved the response, a smoothing in the form
of a parabolic pde solution is currently being implemented which has the potential
to remove any negative impact in these circumstances.


COUNT
-----
.. figure:: ..//images/count.png
   :scale: 75 %
   :align: center

   Difference in number of cycles required to find path (NB. weight > 1 not admissable)

COST
----
.. figure:: ..//images/cost.png
   :scale: 75 %
   :align: center

   Difference in cost in time of final path (NB. weight > 1 not admissable)

TABLES
------

TODO
