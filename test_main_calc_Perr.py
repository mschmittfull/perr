from __future__ import print_function,division
from collections import OrderedDict, namedtuple, Counter
import cPickle
import glob
import numpy as np
import os
import random
import sys

# MS packages
from lsstools import combine_source_fields_to_match_target as combine_source_fields
from lsstools.cosmo_model import CosmoModel
from lsstools.gen_cosmo_fcns import generate_calc_Da
from lsstools.pickle_utils.io import Pickler
from lsstools.model_spec import TrfSpec, TargetSpec
import path_utils


def main(argv):
    """
    Combine source fields to get proxy of a target field. This is stage 0 of reconstruction,
    and can be used to quantify Perror of a bias model.

    For example:

      - Combine delta_m, delta_m^2, s^2 to get proxy of target=delta_halo.

      - Combine different mass-weighted delta_h fields to get proxy of target=delta_m.

    Run using 
      ./run.sh python main_calc_Perr.py
    or 
      ./run.sh mpiexec -n 2 python main_calc_Perr.py
    """

    
    #####################################
    # PARSE COMMAND LINE ARGS
    #####################################
    # parse args
    opts_update_dict = {}
    if len(argv) == 1:
        pass
    elif len(argv) == 2:
        # Update options given as 1st arg
        # e.g. python test_main_calc_Perr.py "{'Rsmooth_for_quadratic_sources': 10.0}"
        import ast
        opts_update_dict = ast.literal_eval(argv[1])
        print("UPDATE OPTS:", opts_update_dict)
        if 'sim_scale_factor' in opts_update_dict.keys():
            raise Exception(
                "sim_scale_factor must not be changed via argument b/c screws up dependent options")
    else:
        raise Exception("May use only 1 argument")

    
    #####################################
    # OPTIONS
    #####################################

    opts = OrderedDict()
 
    opts['test_main_calc_Perr_version'] = '1.2'
    
    ## ANALYSIS
    opts['Ngrid'] = 64

    # k bin width for power spectra, in units of k_f=2pi/L. Must be >=1.0. Choose 1.,2.,3. usually.
    opts['k_bin_width'] = 1.0
    
    ## Path of input catalog. All we need is in_path, which will be obtained using
    ## path_utils.py. The other options are
    ## just for convenience when comparing measured power spectra later.

    if True:
        # L=500 ms_gadget sims produced with MP-Gadget, 1536^3 particles, 64^3 test data
        opts['sim_name'] = 'ms_gadget_test_data'
        opts['sim_irun'] = 4
        # use value from cmd line b/c later options depend on this
        opts['sim_seed'] = opts_update_dict.get('sim_seed', 403)
        opts['ssseed'] = opts_update_dict.get('ssseed', 40000+opts['sim_seed'])      # seed used to draw subsample
        opts['sim_Ntimesteps'] = None  # Nbody, so used thousands of time steps
        opts['sim_Nptcles'] = 1536
        opts['sim_boxsize'] = 500.0
        opts['sim_wig_now_string'] = 'wig'
        # scale factor of simulation snapshot (only used to rescale deltalin -- do not change via arg!)
        opts['sim_scale_factor'] = 0.6250
        # halo mass
        opts['halo_mass_string'] = opts_update_dict.get('halo_mass_string', '13.8_15.1')

        # linear density (ICs of the sims)
        opts['ext_grids_to_load'] = OrderedDict()
        if True:
            # deltalin from mesh (created on mesh, no particles involved)
            opts['ext_grids_to_load']['deltalin'] = {
                'dir': 'IC_LinearMesh_z0_Ng%d' % opts['Ngrid'], 'file_format': 'nbkit_BigFileGrid',
                'dataset_name': 'Field',
                'scale_factor': 1.0, 'nbkit_normalize': True, 'nbkit_setMean': 0.0}
        if False:
            # deltalin from ptcles (created from particle snapshot so includes CIC artifacts)
            # on 64^3, noise curves looked the same as with linearMesh
            opts['ext_grids_to_load']['deltalin_PtcleDens'] = {
                'dir': 'IC_PtcleDensity_Ng%d' % opts['Ngrid'], 'file_format': 'nbkit_BigFileGrid',
                'dataset_name': 'Field',
                'scale_factor': 1.0/(1.0+99.0), # ICs were generated at z=99
                'nbkit_normalize': True, 'nbkit_setMean': 0.0}            
            
        if True:
            # delta_ZA, created by moving 1536^3 ptcles with NGenic (includes CIC artifacts, small shot noise)
            opts['ext_grids_to_load']['deltaZA'] = {
                'dir': 'ZA_%.4f_PtcleDensity_Ng%d' % (opts['sim_scale_factor'], opts['Ngrid']), 
                'file_format': 'nbkit_BigFileGrid',
                'dataset_name': 'Field',
                'scale_factor': opts['sim_scale_factor'], 'nbkit_normalize': True, 'nbkit_setMean': 0.0}
        if True:
            # deltanonl painted from all 1536^3 DM particles (includes CIC artifacts, small shot noise)
            opts['ext_grids_to_load']['delta_m'] = {
                'dir': 'snap_%.4f_PtcleDensity_Ng%d' % (opts['sim_scale_factor'], opts['Ngrid']),
                'file_format': 'nbkit_BigFileGrid',
                'dataset_name': 'Field',
                'scale_factor': opts['sim_scale_factor'], 'nbkit_normalize': True, 'nbkit_setMean': 0.0}

        ## shifted field options
        opts['shifted_fields_RPsi'] = 0.23  # Psi smoothing used in shifting code
        opts['shifted_fields_Np'] = 1536 # 1536     # Nptcles_per_dim used in shifting code; 768,1536
        opts['shifted_fields_Nmesh'] = 1536 #1536 # internal Nmesh used in shifting code

        #for psi_type_str in ['','Psi2LPT_']:
        for psi_type_str in ['']:
            if True:
                # 1 shifted by deltalin_Zeldovich displacement (using nbkit0.3; same as delta_ZA)
                opts['ext_grids_to_load']['1_SHIFTEDBY_%sdeltalin'%psi_type_str] = {
                    'dir': '1_intR0.00_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum' % (
                        psi_type_str,
                        opts['shifted_fields_RPsi'],opts['sim_scale_factor'],opts['shifted_fields_Np'], 
                        opts['shifted_fields_Nmesh'],opts['Ngrid']),
                    'file_format': 'nbkit_BigFileGrid',
                    'dataset_name': 'Field',
                    'scale_factor': opts['sim_scale_factor'], 'nbkit_normalize': True, 'nbkit_setMean': 0.0}

            if True:
                # deltalin shifted by deltalin_Zeldovich displacement (using nbkit0.3)
                opts['ext_grids_to_load']['deltalin_SHIFTEDBY_%sdeltalin'%psi_type_str] = {
                    'dir': 'IC_LinearMesh_intR0.00_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum' % (
                        psi_type_str,
                        opts['shifted_fields_RPsi'],opts['sim_scale_factor'],opts['shifted_fields_Np'],
                        opts['shifted_fields_Nmesh'],opts['Ngrid']),
                    'file_format': 'nbkit_BigFileGrid',
                    'dataset_name': 'Field',
                    'scale_factor': opts['sim_scale_factor'], 'nbkit_normalize': True, 'nbkit_setMean': 0.0}

                # deltalin^2 shifted by deltalin_Zeldovich displacement (using nbkit0.3)
                opts['ext_grids_to_load']['deltalin_growth-mean_SHIFTEDBY_%sdeltalin'%psi_type_str] = {
                    'dir': 'IC_LinearMesh_growth-mean_intR0.00_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum'%(
                        psi_type_str,
                        opts['shifted_fields_RPsi'],opts['sim_scale_factor'],opts['shifted_fields_Np'], 
                        opts['shifted_fields_Nmesh'],opts['Ngrid']),
                    'file_format': 'nbkit_BigFileGrid',
                    'dataset_name': 'Field',
                    'scale_factor': opts['sim_scale_factor'], 'nbkit_normalize': True, 'nbkit_setMean': 0.0}

                # G2[deltalin] shifted by deltalin_Zeldovich displacement (using nbkit0.3)
                opts['ext_grids_to_load']['deltalin_G2_SHIFTEDBY_%sdeltalin'%psi_type_str] = {
                    'dir': 'IC_LinearMesh_tidal_G2_intR0.00_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum' % (
                        psi_type_str,
                        opts['shifted_fields_RPsi'],opts['sim_scale_factor'],opts['shifted_fields_Np'], 
                        opts['shifted_fields_Nmesh'],opts['Ngrid']),
                    'file_format': 'nbkit_BigFileGrid',
                    'dataset_name': 'Field',
                    'scale_factor': opts['sim_scale_factor'], 'nbkit_normalize': True, 'nbkit_setMean': 0.0}

                if True:
                    # deltalin^3 shifted by deltalin_Zeldovich displacement (using nbkit0.3)
                    opts['ext_grids_to_load']['deltalin_cube-mean_SHIFTEDBY_%sdeltalin'%psi_type_str] = {
                        'dir': 'IC_LinearMesh_cube-mean_intR0.00_0.50_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum'%(
                            psi_type_str,
                            opts['shifted_fields_RPsi'],opts['sim_scale_factor'],opts['shifted_fields_Np'], 
                            opts['shifted_fields_Nmesh'],opts['Ngrid']),
                        'file_format': 'nbkit_BigFileGrid',
                        'dataset_name': 'Field',
                        'scale_factor': opts['sim_scale_factor'], 'nbkit_normalize': True, 'nbkit_setMean': 0.0}
            
   

    # ######################################################################
    # Catalogs to read
    # ######################################################################
    opts['cats'] = OrderedDict()

    if opts['sim_name'] == 'ms_gadget_test_data':    

        tmp_halo_dir = 'nbkit_fof_%.4f/ll_0.200_nmin25' % opts['sim_scale_factor']
        ## nonuniform catalogs without ptcle masses
        if True:
            # halos without mass weight, narrow mass cuts: 10.8..11.8..12.8..13.8..15.1
            opts['cats']['delta_h'] = {
                'in_fname': "%s/fof_nbkfmt.hdf5_BOUNDS_log10M_%s.hdf5" % (
                    tmp_halo_dir, opts['halo_mass_string']),
                'weight_ptcles_by': None}
        if False:
            # halos not weighted by mass but including mass info in file, broad mass cut
            # TODO: looks like nbodykit 0.3 does not read this properly b/c of hdf5
            opts['cats']['delta_h'] = {
                'in_fname': "%s/fof_nbkfmt.hdf5_WithMassCols.hdf5_BOUNDS_log10M_%s.hdf5" % (
                    tmp_halo_dir, opts['halo_mass_string']),
                'weight_ptcles_by': None}
        if False:
            # halos in narrow mass bins, no mass weights
            opts['cats']['delta_h_M10.8-11.8'] = {
                'in_fname': "%s/fof_nbkfmt.hdf5_WithMassCols_BOUNDS_log10M_10.8_11.8.hdf5" % tmp_halo_dir,
                'weight_ptcles_by': None}
            opts['cats']['delta_h_M11.8-12.8'] = {
                'in_fname': "%s/fof_nbkfmt.hdf5_WithMassCols_BOUNDS_log10M_11.8_12.8.hdf5" % tmp_halo_dir,
                'weight_ptcles_by': None}
            opts['cats']['delta_h_M12.8-13.8'] = {
                'in_fname': "%s/fof_nbkfmt.hdf5_WithMassCols_BOUNDS_log10M_12.8_13.8.hdf5" % tmp_halo_dir,
                'weight_ptcles_by': None}
            opts['cats']['delta_h_M13.8-15.1'] = {
                'in_fname': "%s/fof_nbkfmt.hdf5_WithMassCols_BOUNDS_log10M_13.8_15.1.hdf5" % tmp_halo_dir,
                'weight_ptcles_by': None}

    else:
        raise Exception("Invalid sim_name %s" % opts['sim_name'])
        



    # ######################################################################
    # Specify sources and targets for bias expansions and transfer functions
    # ######################################################################

    # Allowed quadratic_sources: 'growth','tidal_s2', 'tidal_G2'
    # Allowed cubic_sources: 'cube'
    opts['trf_specs'] = []

           
    ### HALO NUMBER DENSITY FROM DM
    #if True:
    #for target in ['delta_h_WEIGHT_M1']:
    #for target in ['hat_delta_m_from_b1_delta_h_M0_M1']: # test bias expansion of \hat\delta_m obtained from mass-weighted halos
    #for target in ['hat_deltaZA_from_b1_delta_h_M0_M1']: # test bias expansion of \hat\delta_Z obtained from mass-weighted halos
    for target in ['delta_h']:
    

        if True:    
            #if False:
            # Quadratic Lagrangian bias: delta_Z + b1 deltalin(q+Psi) + b2 [deltalin^2-<deltalin^2>](q+Psi) + bG2 [G2](q+Psi)
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltalin_SHIFTEDBY_deltalin',
                                                             'deltalin_growth-mean_SHIFTEDBY_deltalin',
                                                             'deltalin_G2_SHIFTEDBY_deltalin'],
                                             fixed_linear_sources=['1_SHIFTEDBY_deltalin'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field=target,
                                             save_bestfit_field='hat_delta_h_from_1_Tdeltalin2G2_SHIFTEDBY_PsiZ'))


    ## Smoothing for quadratic fields in Mpc/h
    #Rsmooth_lst = [0.1,2.5,5.0,10.0,20.0]
    Rsmooth_lst = [0.1]
    #Rsmooth_lst = [0.1,2.5,5.0,20.0]
    #Rsmooth_lst = [0.1,5.0,10.0,20.0,40.0,80.0]
    #Rsmooth_lst = [0.1,1.0,5.0,20.0,80.0]
    #Rsmooth_lst = [0.1,5.0,15.0,40.0]

    # smoothing scale for 2nd set of quadratic fields
    opts['Rsmooth_for_quadratic_sources2'] = 0.1

    ## Number of iterations to orthogonalize fields when computing trf fcns
    # (does not seem to improve results, so best to use 0)
    # 0: no orthogonalization, 1: orthogonalize once, 2: orthogonalize twice, etc
    opts['N_ortho_iter_for_trf_fcns'] = 1

    ## Orthogonalization method used for fields when computing trf fcns.
    # Only used if N_ortho_iter_for_trf_fcns>=1. This affects trf fcns but not final noise curves.
    # - 'EigenDecomp':    Use eigenvalue decomposition of S matrix; each orthogonal field
    #                     is a combination of all original fields, so not great.
    # - 'CholeskyDecomp': Use Cholesky decomposition of S matrix so that span of first
    #                     k orthogonal fields is the same as span of first k original fields.
    opts['orth_method_for_trf_fcns'] = 'CholeskyDecomp'

    # Interpolation used for trf fcns: 'nearest', 'linear', 'manual_Pk_k_bins'.
    # Used 'linear' until 6 Jun 2018.
    opts['interp_kind_for_trf_fcns'] = 'manual_Pk_k_bins'

    # save some additional power spectra that are useful for plotting later
    opts['Pkmeas_helper_columns'] = ['delta_h','delta_m','1_SHIFTEDBY_deltalin','deltalin']

   
    # copy boxsize        
    opts['boxsize'] = opts['sim_boxsize']
    
   


    
    #### Output pickles and cache. 
    opts['pickle_path'] = '$SCRATCH/perr/pickle/'
    opts['cache_base_path'] = '$SCRATCH/perr/cache/'
    opts['grids4plots_base_path'] = '$SCRATCH/perr/grids4plots/'
    opts['grids4plots_R'] = 0.0  # Gaussian smoothing applied to grids4plots


    ## ANTI-ALIASING OPTIONS (do not change unless you know what you do)
    # Kmax above which to 0-pad. Should use kmax<=2pi/L*N/2 to avoid
    # unwanted Dirac delta images/foldings when multipling fields.
    opts['kmax'] = 2.0*np.pi/opts['boxsize'] * float(opts['Ngrid'])/2.0
    # CIC deconvolution of grid: None or 'grid_non_isotropic'
    opts['grid_ptcle2grid_deconvolution'] = None # 'grid_non_isotropic' #'grid_non_isotropic'
    # CIC deconvolution of power: None or 'power_isotropic_and_aliasing' 
    opts['Pk_ptcle2grid_deconvolution'] = None

    
    
    ## what do to plot/save    
    opts['keep_pickle'] = False
    opts['pickle_file_format'] = 'pickle'
    # plot using plotting code for single realization; difft from code plotting avg 
    do_plot = False
    # save grids for slice plots and scatter plots
    opts['save_grids4plots'] = False
    

    ## COSMOLOGY OPTIONS
    if opts['sim_name'] in ['jerryou_baoshift','ms_gadget', 'ms_gadget_test_data']:
        # This is only used to scale deltalin to the redshift of the simulation snapshot.
        # Fastpm runs by jerry and marcel for baoshift
        #omega_m = 0.307494
        #omega_bh2 = 0.022300
        #omega_ch2 = 0.118800
        #h       = math.sqrt((omega_bh2 + omega_ch2) / omega_m) = 0.6774
        opts['cosmo_params'] = dict(Om_m=0.307494, Om_L=1.0-0.307494, Om_K=0.0,
                                    Om_r=0.0, h0=0.6774)

    elif opts['sim_name'] == 'RunPB':
        # This is only used to scale deltalin to the redshift of the simulation snapshot.
        # RunPB by Martin White; read cosmology from Martin email
        #omega_bh2=0.022
        #omega_m=0.292
        #h=0.69
        # Martin rescaled sigma8 from camb from 0.84 to 0.82 to set up ICs. But no need to include
        # here b/c we work with that rescaled deltalin directly (checked that linear power 
        # agrees with nonlinear one if rescaled deltalin rescaled by D(z)
        opts['cosmo_params'] = dict(Om_m=0.292, Om_L=1.0-0.292, Om_K=0.0,
                                    Om_r=0.0, h0=0.69)

        

    #########################################
    # UPDATE OPTS GIVEN ON COMMAND LINE
    #########################################
    
    # update options given on command line
    opts.update(opts_update_dict)
    print("opts:", opts)



    
    #####################################
    # START PROGRAM
    #####################################
    
    # make sure we keep the pickle if it is a big run and do not plot
    if opts['Ngrid'] > 256:
        opts['keep_pickle'] = True
        do_plot = False


    ### derived options (do not move above b/c command line args might
    ### overwrite some options!)
    opts['in_path'] = path_utils.get_in_path(opts)
    # for output densities
    opts['out_rho_path'] = os.path.join(
        opts['in_path'], 
        'out_rho_Ng%d' % opts['Ngrid'])
    

    
    # expand environment names in paths
    paths = {}
    for key in ['in_path', 'in_fname', 'in_fname_PTsim_psi_calibration',
                'in_fname_halos_to_displace_by_mchi',
                'pickle_path', 'cache_base_path', 'grids4plots_base_path',
                'out_rho_path']:
        if opts.has_key(key):
            if opts[key] is None:
                paths[key] = None
            else:
                paths[key] = os.path.expandvars(opts[key])

    from nbodykit import setup_logging, logging, CurrentMPIComm
    setup_logging()
    comm = CurrentMPIComm.get()
    logger = logging.getLogger('PerrCalc')
    

    # check for duplicate save_bestfit_field entries
    save_bestfit_fields = [tf.save_bestfit_field for tf in opts['trf_specs']]
    if len(Counter(save_bestfit_fields)) != len(opts['trf_specs']):
        raise Exception("Found duplicate save_bestfit_field: %str" % str(Counter(save_bestfit_fields)))
    
    # Init Pickler instance to save pickle later (this will init pickle fname)
    pickler = None
    if comm.rank == 0:
        pickler = Pickler(path=paths['pickle_path'], base_fname='main_calc_Perr',
                                  file_format=opts['pickle_file_format'],
                                  rand_sleep=(opts['Ngrid']>128))
        print("Pickler: ", pickler.full_fname)
    pickler = comm.bcast(pickler, root=0)

    # where to save grids for slice and scatter plots
    if opts['save_grids4plots']:
        paths['grids4plots_path'] = os.path.join(paths['grids4plots_base_path'],
                                                 os.path.basename(pickler.full_fname))
        if not os.path.exists(paths['grids4plots_path']):
            os.makedirs(paths['grids4plots_path'])
        print("grids4plots_path:", paths['grids4plots_path'])
    
    # unique id for cached files so we can run multiple instances at the same time
    # rank0 gets the cache id and then broadcasts it to other ranks

    cacheid = None
    if comm.rank == 0:
        file_exists = True
        while file_exists:
            cacheid = ('CACHE%06x' % random.randrange(16**6)).upper()
            paths['cache_path'] = os.path.join(paths['cache_base_path'], cacheid)
            file_exists = (len(glob.glob(paths['cache_path']))>0)
        # create cache path
        if not os.path.exists(paths['cache_path']):
            #os.system('mkdir -p %s' % paths['cache_path'])
            os.makedirs(paths['cache_path'])
        logger.info('cacheid: %s' % cacheid)

    # broadcast to all ranks
    cacheid = comm.bcast(cacheid, root=0)
    paths['cache_path'] = os.path.join(paths['cache_base_path'], cacheid)
        
    # Check some params
    if ((opts['grid_ptcle2grid_deconvolution'] is not None) and
        (opts['Pk_ptcle2grid_deconvolution'] is not None)):
        raise Exception("Must not simultaneously apply ptcle2grid deconvolution to grid and Pk.")

    
    # list of all densities actually needed for trf fcns
    densities_needed_for_trf_fcns = []
    for trf_spec in opts['trf_specs']:
        for linsource in trf_spec.linear_sources:
            if linsource not in densities_needed_for_trf_fcns:
                densities_needed_for_trf_fcns.append(linsource)
        #densities_needed_for_trf_fcns += trf_spec.linear_sources
        for fixedlinsource in getattr(trf_spec, 'fixed_linear_sources', []):
            if fixedlinsource not in densities_needed_for_trf_fcns:
                densities_needed_for_trf_fcns.append(fixedlinsource)
            
        if trf_spec.field_to_smoothen_and_square is not None:
            if trf_spec.field_to_smoothen_and_square not in densities_needed_for_trf_fcns:
                densities_needed_for_trf_fcns.append(trf_spec.field_to_smoothen_and_square)
        if trf_spec.field_to_smoothen_and_square2 is not None:
            if trf_spec.field_to_smoothen_and_square2 not in densities_needed_for_trf_fcns:
                densities_needed_for_trf_fcns.append(trf_spec.field_to_smoothen_and_square2)
        if trf_spec.target_field not in densities_needed_for_trf_fcns:
            densities_needed_for_trf_fcns.append(trf_spec.target_field)
        if hasattr(trf_spec, 'target_spec'):
            for tc in getattr(trf_spec.target_spec, 'linear_target_contris', []):
                if tc not in densities_needed_for_trf_fcns:
                    densities_needed_for_trf_fcns.append(tc)
    opts['densities_needed_for_trf_fcns'] = densities_needed_for_trf_fcns
    print("densities_needed_for_trf_fcns:", densities_needed_for_trf_fcns)

    
    # #################################################################################
    # loop over smoothing scales 
    # #################################################################################

    pickle_dict_at_R = OrderedDict()
    pickle_dict_at_R['opts'] = opts.copy()
    for R in Rsmooth_lst:
        # actually calculate power spectra
        print("\n\nRun with R=", R)
        tmp_opts = opts.copy()
        tmp_opts['Rsmooth_for_quadratic_sources'] = R
        this_pickle_dict = combine_source_fields.actually_calc_Pks(tmp_opts, paths)
        pickle_dict_at_R[(R,)] = this_pickle_dict
    
        residual_key = '[hat_delta_h_from_1_Tdeltalin2G2_SHIFTEDBY_PsiZ]_MINUS_[delta_h]'
        Perr = this_pickle_dict['Pkmeas'][(residual_key,residual_key)].P
        Perr_expected = np.array([
            9965.6,17175.8,22744.4,19472.3,19081.2,19503.4,19564.9,18582.9,19200.1,16911.3,
            16587.4,16931.9,15051.0,13835.1,13683.8,13109.9,12353.5,11900.2,11085.1,11018.4,
            10154.0,9840.7,8960.6,8484.1,7942.2,7426.7,6987.8,6578.1,6269.1,5810.7,5511.7])

        if comm.rank == 0:
            Perr_lst = ['%.1f' % a for a in list(Perr)]
            Perr_expected_lst = ['%.1f' % a for a in list(Perr)]
            logger.info('Perr:\n%s' % str(','.join(Perr_lst)))
            logger.info('Expected Perr:\n%s' % str(','.join(Perr_expected_lst)))
            if np.allclose(Perr, Perr_expected, rtol=1e-3):
                logger.info('TEST Perr: OK')
            else:
                logger.info('TEST Perr: FAILED')
                raise Exception('Test failed')

        # save all resutls to pickle
        if comm.rank == 0:
            pickler.write_pickle(pickle_dict_at_R)


    # #################################################################################
    # Plot power spectra and correlations (TODO: move to new script that loads pickle)
    # #################################################################################

    if do_plot:
        if comm.rank == 0:
            #try:
            if True:
                from main_quick_plot_Pk import plot_pickle_at_R
                # just plot last smoothing scale
                plot_pickle_at_R(pickler.full_fname, Rsmooth=Rsmooth_lst[0])
            #except:
            #print("Could not plot")


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


        
if __name__ == '__main__':
    sys.exit(main(sys.argv))


