BATCH MANAGER
=============

INTRODUCTION
------------

High latency functions are increasingly incorporated into workflows. Whether
the latency stems from servers on the other side of the world, or 
IO, reducing the cost of running these functions via batch becomes neccesary.
This package is designed to reduce run times by enabling information sharing
between different routines that utilize the same functions and in conjunction
reducing the number of calls to any one function throughout all of the processes.
Available on GitHub: `code <https://github.com/marksweissma/code/tree/master/batch_manager>`_

README
------

This package is for handling batch functions and evaluating function compostions with 
string representations linked to functions via a FunctionRegistry map- i.e. ['f(g(h(1,2),1),4)', 'f(4,h(g(1,2)))']


While designed for batch applications, a serial option is available for functions
that cannot utilize a batch apply:

    1. **serial** where every function is applied on its arguments
       every time 
    2. **batch** where the input computations are parsed and then instead
       of being evaluated a tree is built, which is pruned between function calls
       with the goal of *minimizing* the number of batch function calls during 
       computation.

This package is currently maintained in two containers:

    1. **src** folder containing the source code utilized for computation
    2. **tests** folder containing current tests. 

The docs are integrated with the rest of available packages but can be split off if needed 
   
   N.B. The **Makefile** in the root directory is for running the current test suite 


DESIGN
------

**serial** evaluation is designed to be simple and lightweight. **serial** evaluation
recurses through each computation until the local function can operate on the arguments.
Upon each exit in the stack the level above is now able to evaluate through computation.

**batch** evaluation is designed to minimize the number of function calls during computation.
**batch** parses the list of computations into a collection of trees, collects the leaves 
(leaf := node -function argument pairs- which can be explicity evaluated but have not been) from all trees and decides which function
to apply. This algorithim is greedy and not optimal, because it only considers the set of leaves
rather than evaluating the optimal course for reducing all of the trees before beginning. To reduce
the number of function calls a dictionary for memoizing the known results is used to prune
the tree in a post-order traversal. As a result the computation 'f(f(...f(0)...))' where f(0) = 0 will only call 'f'
once and the traversal + memoization will evaluate the rest of the tree.


USAGE
-----

The module  function_conjunction.py contains the :class:`.function_conjunction.Compute` class which has
the :meth:`.function_conjunction.Compute.compute` @classmethod for evaluating computations  given a function registry
map. By default compute operates in serial but as the option to operate in batch
via the evalType keyword.

For more information please see :mod:`.function_conjunction` documentation

.. code-block:: python
   :caption: serial_example

   from src.function_conjunction import Compute
   from src.helpers import s_sum, s_max

   computations = ['f(g(h(2,3),5),g(g(3),h(4)),10)']
   functionRegistrySerial = {'f': s_sum, 'g': s_sum, 'h': s_max}
   evalType = 'serial'

   output = Compute.compute(computations, functionRegistrySerial, evalType)
   print output


.. code-block:: python
   :caption: batch_example

   from src.function_conjunction import Compute
   from src.helpers import batch_sum, batch_max

   computations = ['f(f(f(f(0))))', 'f(f(0))']
   functionRegistryBatch= {'f': batch_sum, 'g': batch_sum, 'h': batch_max}
   evalType = 'batch'

   output = Compute.compute(computations, functionRegistryBatch, evalType)

   print output
   print Compute.batchCalls


   computations = ['f(g(h(2,3),5),g(g(3),h(4)),10)']
   functionRegistryBatch= {'f': batch_sum, 'g': batch_sum, 'h': batch_max}
   evalType = 'batch'

   output = Compute.compute(computations, functionRegistryBatch, evalType)

   print output
   print Compute.batchCalls


TEST
----

Testing is built on python unittest framework. The Makefile in the packages root
can be used to run the test suite from the root project directory  with ``make test``.  pep-8 style
checkers for both function_conjunction.py and helpers.py are built in.
