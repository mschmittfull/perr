perr
=========================================
Calculate analytical 3D forward models for galaxy surveys and compare with simulations.


The code combines a number of source fields to get the best proxy of a target field. This is done by constructing transfer functions that minimize the mean-square model error in each k bin using linear regression. The code then computes the difference between the best-fit proxy and the target field, which can be used to quantify the error power spectrum of the halo bias model or other LSS models. For details, see https://arxiv.org/abs/1811.10640.

Example applications
--------------------

- Real space halo bias expansion at the field level: Combine delta_m, delta_m^2, G_2[delta_m] and other fields to get proxy of target=delta_halo. Use main_calc_Perr.py for this. To submit multiple jobs to compute this, run main_calc_Perr.job.helios.spawn.py. The results will be saved in dill files that can be loaded as shown in the notebooks.

- The fields in the bias expansion can also be shifted fields as defined in arxiv:1811.10640. In that case, large-scale (infrared) displacements are included. To precompute these shifted fields, run main_shift_catalog_by_Psi_grid.py. An example job script is in main_shift_catalog_by_Psi_grid.job.helios. 

- The same can be done in redshift space, using main_calc_RSD_Perr.py.

- Combine different mass-weighted delta_h fields to get proxy of target=delta_m. This can be used to test how well mass-weighted halos can recover the dark matter field. For this, modify main_calc_Perr.py accordingly (exchanging source and target fields).


Example usage
-------------

- On single core:

  .. code-block:: bash

    $ python main_calc_Perr.py

- On 8 cores:

  .. code-block:: bash

    $ mpiexec -n 8 python main_calc_Perr.py

- If `lsstools <https://github.com/mschmittfull/lsstools>`_ is installed in an anaconda environment, activate it first. For example:

  .. code-block:: bash

    $ source activate nbodykit-env

  To automatically activate and deactivate the environment when running the code, modify and use run.sh. For example:

  .. code-block:: bash

    $ run.sh python main_calc_Perr.py

- To generate shifted densities (i.e. the IR-resummed shifted operators from arxiv:1811.10640), run

  .. code-block:: bash

    $ python main_shift_catalog_by_Psi_grid.py

  This saves the shifted densities to disk so they can be loaded with main_calc_Perr.py and used as operators in the bias expansion.



Modifying options
-----------------

- In the code: The main code is in file main_calc_Perr.py. This contains examples for different simulations and halo bias models, and can be modified for new applications (perhaps make a copy and modify that). For a more minimal run script, see main_calc_Perr_test.py. 

- On the command line: Options can also be changed on the command line by supplying a single string argument which contains a dictionary with options that overwrite the default options in the file. For example:

  .. code-block:: bash

    $ python main_calc_Perr.py "{'sim_seed': 300}"

  This should be used with caution because unwanted behavior can result when some options depend on others and they are modified before getting overwritten by command line arguments.

- To test different bias models, modify opts['trf_specs']. This is a list, where each entry specifies the source fields and target field to be matched. The code evaluates all entries in this list and saves the results in a pickle file.


Output
------
If opts['keep_pickle'] is set True, the output is saved in a pickle file. This can be loaded and plotted using plot/main_plot_Perr.py. It is recommended to log the pickle filenames in plot/logbook_main_calc_Perr_pickles.py so they can be easily loaded later without having to run the code again.


Installation
------------
The code requires `lsstools <https://github.com/mschmittfull/lsstools>`_ -- see installation guidelines there.

To test if the installation was successful, run

.. code-block:: bash

  $ python main_calc_Perr_test.py

and look for 'TEST Perr: OK' at the end.


Contributing
------------
To contribute, please create a fork on github, make changes and commits, and submit a pull request on github.
