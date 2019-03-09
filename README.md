# perr
Calculate the error of analytical large-scale structure models by comparing against simulations at the field level.


The code combines a number of source fields to get the best proxy of a target field. It then computes the difference between the best-fit proxy and the target field, which can be used to quantify the error power spectrum of the halo bias model or other LSS models. 

Example applications:

- Combine delta_m, delta_m^2, G_2[delta_m] to get proxy of target=delta_halo. This tests the halo bias expansion.

- Combine different mass-weighted delta_h fields to get proxy of target=delta_m. This can be used to test how well mass-weighted halos can recover the dark matter field.


Example usage:

- On single core:

  ./run.sh python main_calc_Perr.py

- On multiple cores:

  ./run.sh mpiexec -n 2 python main_calc_Perr.py


 Installation:

