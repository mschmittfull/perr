from __future__ import print_function, division
import os

from lsstools import combine_fields_to_match_target as combine_fields
from lsstools.cosmo_model import CosmoModel
from lsstools.gen_cosmo_fcns import calc_f_log_growth_rate, generate_calc_Da
from lsstools import model_spec
from lsstools.pickle_utils.io import Pickler
from nbodykit import CurrentMPIComm, logging, setup_logging
import path_utils
import utils


def calculate_model_error(opts):
    """
    Calculate the model error for all models specified in opts['trf_specs'].

    TODO: use kwargs instead of large opts dictionary
    """

    #####################################
    # Initialize
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

    model_spec.check_trf_specs_consistency(opts['trf_specs'])

  
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

    if opts.get('RSDstrings', ['']) != ['']:
        # calculate D and f
        cosmo = CosmoModel(**opts['sim_opts'].cosmo_params)
        calc_Da = generate_calc_Da(cosmo=cosmo)
        f_log_growth = calc_f_log_growth_rate(
            a=opts['sim_opts'].sim_scale_factor,
            calc_Da=calc_Da,
            cosmo=cosmo,
            do_test=True
            )
        # save in opts so we can easily access it throughout code (although strictly
        # speaking it is not a free option but derived from cosmo_params)
        opts['f_log_growth'] = f_log_growth
    else:
        opts['f_log_growth'] = None

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
