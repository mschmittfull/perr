from __future__ import print_function, division
from argparse import ArgumentParser
from collections import namedtuple, OrderedDict
import cPickle
import numpy as np
import os
import sys

from lsstools import combine_fields_to_match_target as combine_fields
from lsstools import parameters
from lsstools import parameters_ms_gadget
from lsstools.cosmo_model import CosmoModel
from lsstools.gen_cosmo_fcns import generate_calc_Da
from lsstools.model_spec import *
from lsstools.pickle_utils.io import Pickler
from nbodykit import CurrentMPIComm, logging, setup_logging
import path_utils
import utils

def main():
    """
    Combine source fields to get proxy of a target field. This is stage 0 of 
    reconstruction, and can be used to quantify Perror of a bias model.

    For example:

      - Combine delta_m, delta_m^2, s^2 to get proxy of target=delta_halo.

      - Combine different mass-weighted delta_h fields to get proxy of
        target=delta_m.

    Usage examples:
      ./run.sh python main_calc_Perr.py
    or 
      ./run.sh mpiexec -n 4 python main_calc_Perr.py --SimSeed 403
    """

    #####################################
    # PARSE COMMAND LINE ARGS
    #####################################
    ap = ArgumentParser()

    ap.add_argument('--SimSeed',
                    type=int,
                    default=403,
                    help='Simulation seed to load.')

    ap.add_argument('--HaloMassString',
                    default='13.8_15.1',
                    help="Halo mass string, for example '13.8_15.1'.")

    cmd_args = ap.parse_args()

    #####################################
    # OPTIONS
    #####################################
    opts = OrderedDict()

    # Bump this when changing code without changing options. Otherwise pickle
    # loading might wrongly read old pickles.
    opts['main_calc_Perr_version'] = '0.1'

    # Simulation options. Will be used by path_utils to get input path, and
    # to compute deltalin at the right redshift.
    seed = cmd_args.SimSeed
    opts['sim_opts'] = parameters_ms_gadget.MSGadgetSimOpts.load_default_opts(
        sim_name='ms_gadget',
        sim_seed=seed,
        ssseed=40000+seed,
        halo_mass_string=cmd_args.HaloMassString)

    # Grid options.
    Ngrid = 64
    opts['grid_opts'] = parameters.GridOpts(
        Ngrid=Ngrid,
        kmax=2.0*np.pi/opts['sim_opts'].boxsize * float(Ngrid)/2.0,
        grid_ptcle2grid_deconvolution=None
        )

    # Options for measuring power spectrum. Use defaults.
    opts['power_opts'] = parameters.PowerOpts()

    # Transfer function options. See lsstools.parameters.py for details.
    opts['trf_fcn_opts'] = parameters.TrfFcnOpts(
        Rsmooth_for_quadratic_sources=0.1,
        Rsmooth_for_quadratic_sources2=0.1,
        N_ortho_iter=1,
        orth_method='CholeskyDecomp',
        interp_kind='manual_Pk_k_bins'
        )

    # External grids to load: deltalin, delta_m, shifted grids
    opts['ext_grids_to_load'] = opts['sim_opts'].get_default_ext_grids_to_load(
        Ngrid=opts['grid_opts'].Ngrid)

    # Catalogs to read
    opts['cats'] = opts['sim_opts'].get_default_catalogs()

    # Specify bias expansions to test
    opts['trf_specs'] = []

    # Quadratic Lagrangian bias: delta_Z + b1 deltalin(q+Psi) + b2 
    # [deltalin^2-<deltalin^2>](q+Psi) + bG2 [G2](q+Psi)
    opts['trf_specs'].append(
        TrfSpec(linear_sources=[
            'deltalin_SHIFTEDBY_deltalin',
            'deltalin_growth-mean_SHIFTEDBY_deltalin',
            'deltalin_G2_SHIFTEDBY_deltalin'
        ],
                fixed_linear_sources=['1_SHIFTEDBY_deltalin'],
                field_to_smoothen_and_square=None,
                quadratic_sources=[],
                target_field='delta_h',
                save_bestfit_field=
                'hat_delta_h_from_1_Tdeltalin2G2_SHIFTEDBY_PsiZ'))

    # Save results
    opts['keep_pickle'] = True
    opts['pickle_file_format'] = 'dill'
    opts['pickle_path'] = '$SCRATCH/perr/pickle/'

    # Save some additional power spectra that are useful for plotting later
    opts['Pkmeas_helper_columns'] = [
        'delta_h', 'delta_m', '1_SHIFTEDBY_deltalin', 'deltalin'
    ]

    # Save grids for 2d slice plots and histograms
    opts['save_grids4plots'] = False
    opts['grids4plots_base_path'] = '$SCRATCH/perr/grids4plots/'
    opts['grids4plots_R'] = 0.0  # Gaussian smoothing applied to grids4plots

    # Cache path
    opts['cache_base_path'] = '$SCRATCH/perr/cache/'

    # Run the program given the above opts.
    outdict = calc_Perr(opts)


def calc_Perr(opts):
    """
    TODO: move to some other module!
    """

    #####################################
    # Initialize program.
    #####################################

    # make sure we keep the pickle if it is a big run and do not plot
    if opts['grid_opts'].Ngrid > 256:
        opts['keep_pickle'] = True

    ### derived options (do not move above b/c command line args might
    ### overwrite some options!)
    opts['in_path'] = path_utils.get_in_path(opts)
    # for output densities
    opts['out_rho_path'] = os.path.join(
        opts['in_path'],
        'out_rho_Ng%d' % opts['grid_opts'].Ngrid
    )

    # expand environment names in paths
    paths = {}
    for key in [
            'in_path', 'in_fname', 'in_fname_PTsim_psi_calibration',
            'in_fname_halos_to_displace_by_mchi', 'pickle_path',
            'cache_base_path', 'grids4plots_base_path', 'out_rho_path'
    ]:
        if opts.has_key(key):
            if opts[key] is None:
                paths[key] = None
            else:
                paths[key] = os.path.expandvars(opts[key])

    setup_logging()
    comm = CurrentMPIComm.get()
    logger = logging.getLogger('PerrCalc')

    check_trf_specs_consistency(opts['trf_specs'])

  
    # Init Pickler instance to save pickle later (this will init pickle fname)
    pickler = None
    if comm.rank == 0:
        pickler = Pickler(path=paths['pickle_path'],
                          base_fname='main_calc_Perr',
                          file_format=opts['pickle_file_format'],
                          rand_sleep=(opts['grid_opts'].Ngrid > 128))
        print("Pickler: ", pickler.full_fname)
    pickler = comm.bcast(pickler, root=0)

    # where to save grids for slice and scatter plots
    if opts['save_grids4plots']:
        paths['grids4plots_path'] = os.path.join(
            paths['grids4plots_base_path'],
            os.path.basename(pickler.full_fname))
        if not os.path.exists(paths['grids4plots_path']):
            os.makedirs(paths['grids4plots_path'])
        print("grids4plots_path:", paths['grids4plots_path'])

    paths['cache_path'] = utils.make_cache_path(paths['cache_base_path'], comm)

    # Get list of all densities actually needed for trf fcns.
    densities_needed_for_trf_fcns = utils.get_densities_needed_for_trf_fcns(
        opts['trf_specs'])
    opts['densities_needed_for_trf_fcns'] = densities_needed_for_trf_fcns


    # ##########################################################################
    # Run program.
    # ##########################################################################

    pickle_dict = combine_fields.paint_combine_and_calc_power(
        trf_specs=opts['trf_specs'],
        paths=paths,
        catalogs=opts['cats'], 
        needed_densities=opts['densities_needed_for_trf_fcns'],
        ext_grids_to_load=opts['ext_grids_to_load'],
        trf_fcn_opts=opts['trf_fcn_opts'],
        grid_opts=opts['grid_opts'],
        sim_opts=opts['sim_opts'],
        power_opts=opts['power_opts'],
        save_grids4plots=opts['save_grids4plots'],
        grids4plots_R=opts['grids4plots_R'],
        Pkmeas_helper_columns=opts['Pkmeas_helper_columns']
        )

    # copy over opts so they are saved
    assert not pickle_dict.has_key('opts')
    pickle_dict['opts'] = opts.copy()

    # save all resutls to pickle
    if comm.rank == 0:
        pickler.write_pickle(pickle_dict)

    # print path with grids for slice and scatter plotting
    if opts['save_grids4plots']:
        print("grids4plots_path: %s" % paths['grids4plots_path'])

    # delete pickle if not wanted any more
    if comm.rank == 0:
        if opts['keep_pickle']:
            print("Pickle: %s" % pickler.full_fname)
        else:
            pickler.delete_pickle_file()

        # remove cache dir
        from shutil import rmtree
        rmtree(paths['cache_path'])

    return pickle_dict

if __name__ == '__main__':
    main()
