perr
=========================================
Calculate the error of analytical large-scale structure models by comparing them against simulations at the field level.


The code combines a number of source fields to get the best proxy of a target field. This is done by constructing transfer functions that minimize the mean-square model error in each k bin using linear regression. The code then computes the difference between the best-fit proxy and the target field, which can be used to quantify the error power spectrum of the halo bias model or other LSS models. For details, see https://arxiv.org/pdf/1811.10640.

Example applications
--------------------

- Combine delta_m, delta_m^2, G_2[delta_m] to get proxy of target=delta_halo. This tests the halo bias expansion.

- Combine different mass-weighted delta_h fields to get proxy of target=delta_m. This can be used to test how well mass-weighted halos can recover the dark matter field.


Example usage
-------------

- On single core:

.. code-block:: bash

    $ run.sh python main_calc_Perr.py

- On 8 cores:

.. code-block:: bash

    $ run.sh mpiexec -n 8 python main_calc_Perr.py


Modifying options
-----------------

- In the code: The main code is in file main_calc_Perr.py and options can be modified there. To run different models, modify opts['trf_specs']. This is a list, where each entry specifies the source fields and target field to be matched. The code evaluates all entries in this list and saves the results in a pickle file.

- On the command line: You can also change options on the command line by supplying a single string argument which contains a dictionary with options that overwrite the default options in the file. For example:
.. code-block:: bash
    $ run.sh python main_calc_Perr.py "{'sim_seed': 300}"
Use this with caution because unwanted behavior can result when some options depend on others and they are modified before getting overwritten by command line arguments.


Installation
------------
The code requires `lsstools <https://github.com/mschmittfull/lsstools>`_ -- see installation guidelines there.
