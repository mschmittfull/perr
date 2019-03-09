#!/usr/bin/env python
#
# Marcel Schmittfull 2019 (mschmittfull@gmail.com)
#
# Python script for calculation of Perror.




from __future__ import print_function,division

import cPickle
import numpy as np
import os
from collections import OrderedDict, namedtuple, Counter
from scipy import interpolate as interp
#from scipy.interpolate import RectBivariateSpline
#import time
#import re
#import h5py
import random
import glob
import sys
import json


# MS packages
#import constants
from lsstools import Pickler
from lsstools.cosmo_model import CosmoModel
from lsstools.gen_cosmo_fcns import generate_calc_Da
import path_utils
#import transfer_functions
#import iterdisp_filters
#import PTsim_utils
from lsstools import combine_source_fields_to_match_target as combine_source_fields
from lsstools.TrfSpec import TrfSpec, TargetSpec


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
        # e.g. python main_quick_calc_Pk.py "{'Rsmooth_for_quadratic_sources': 10.0}"
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
    opts['use_mpi'] = False

    # 14 Feb 2018: v0.2: Save trf fcns results more systematically.
    # 15 Feb 2018: v0.3: Change normalization of basis vectors after Cholesky orthogonalization.
    #                    Also use 'nearest' interpolation of rotation matrix (much better than 'linear').   
    # 19 Feb 2018: v0.4: Add option to include delta_lin in sources. Generalize transfer functions specs
    #                    using TrfSpecs. Also save results with keys given by str(TrfSpecs).
    # 21 Feb 2018: v0.5: Save more power spectra.
    # 27 Feb 2018: v0.6: Add option to read linear density from RunPB.
    # 27 Feb 2018: v0.7: Add option to read ZA density from RunPB, and introduce ext_grids_to_load option.
    # 28 Feb 2018: v0.8: Reduce memory by computing fewer power spectra of end fields.
    # 28 Feb 2018: v0.9: Do not delete ABSK b/c need for orthogonalization.
    # 5 Mar 2018: v0.10: Calc deltalin+F2 correctly.
    # 6 Mar 2018: v0.11: Add mass weights to get DM from halos.
    # 8 Mar 2018: v0.12: Add scatter to halo mass
    # 8 Mar 2018: v0.13: Improve memory usage by caching grids immediately after computing them.
    # 8 Mar 2018: v0.14: Change order of options.
    # 19 Mar 2018: v0.15: Refactor code so that combination of fields is in extra code file.
    # 26 Mar 2018: v0.16: Add options for ms_gadget sims (produced with PM-Gadget)
    # 29 Mar 2018: v0.17: Compute power of (target-bestfit_field) directly from field difference.
    # 8 April 2018: v0.18: Fix bug in shifted linear density used for bias expansion.
    # 12 APril 2018: v0.19: Implement fixed linear sources that do not get trf fcns.
    # 13 April 2018: v0.20: Fix sign mistake in Psi^[2].
    # 8 May 2018: v0.21: Implement more bias schemes; bump version to avoid pickle issues (disk quota exceeded).
    # 14 May 2018: v0.22: Made an inconsistent change in TrfSpec.to_dict(); bump version after fixing it.
    # 21 May 2018: v0.23: Add option to change k binning width for power spectra (to get more robust error bars)
    # 23 May 2018: v0.24: Fix k vector bug introduced when changing k binning.
    # 28 May 2018: v0.25: Save grids4plots.
    # 29 May 2018: v0.26: Bump version to avoid pickle conflicts.
    # 30 May 2018: v0.27: Add option to change normalization of target when doing mass-weighting.
    # 30 May 2018: v0.28: Fix small bug introduced in v0.27 (make target weight positive at low k).
    # 30 May 2018: v0.29: Add option to minimize |target0+T*other_targets-T*model_sources|^2.
    # 31 May 2018: v0.30: Add option to normalize power to observed halo power.
    # 1 Jun 2018: v0.32: Fix bug when minimizing (d-m)^2 with multiple data contributions (introduced in v0.29);
    #                    Had wrong sign for alpha coeff.
    # 6 Jun 2018: v0.33: Improve orthogonalization of fields by using interpolation matched to P(k) k-binning.
    #                    Resulting fields are now orthogonal at 10^{-5} level instead of 10^{-2}. Hope this
    #                    resolves issues related to ordering of fields in data vector orthogonalization.
    # 6 Jun 2018: v0.34: Also use better orthogonalization for data contris. 
    # 6 Jun 2018: v0.35,0.36: Testing and debugging mass-weighting and (d-m)^2/d^2 minimization.
    # 7 Jun 2018: v0.37: Add option to normalize target when minimizing |target0+T*other_targets-T*sources|^2.
    # 8 Jun 2018: v0.38: Skip interp test because crashes at high k where P(k) is probably nan.
    # Upgraded internal functions to use nbodykit 0.3 instead of 0.1.    
    # 19 Feb 2019: v1.0: First attempt to run everything with nbodykit 0.3.
    # 19 Feb 2019: v1.1: New code running with nbkit 0.3 agrees with old code using nbkit 0.1. For 1,2,4 ranks.    
    opts['main_quick_calc_Pk_version'] = '1.0'
    
    ## ANALYSIS
    opts['Ngrid'] = 64

    # k bin width for power spectra, in units of k_f=2pi/L. Must be >=1.0. Choose 1.,2.,3. usually.
    opts['k_bin_width'] = 1.0
    
    ## Path of input catalog. All we need is in_path, which will be obtained using
    ## path_utils.py. The other options are
    ## just for convenience when comparing measured power spectra later.
    if False:
        # 120-step L=500 jerryou baoshift FastPM sims run by Marcel (used in 1704 paper)
        opts['sim_name'] = 'jerryou_baoshift'
        opts['sim_irun'] = 5
        opts['sim_seed'] = 505
        opts['sim_Ntimesteps'] = 120
        opts['sim_Nptcles'] = 2048
        opts['sim_boxsize'] = 500.0
        opts['sim_wig_now_string'] = 'wig'
        # scale factor of simulation snapshot (only used to rescale deltalin -- do not change via arg!)
        opts['sim_scale_factor'] = 0.6250
        #opts['sim_scale_factor'] = 1.0000

        # linear density (ICs of the sims)
        # Scale factor of linear density file will be rescaled to sim_scale_factor. 
        opts['ext_grids_to_load'] = OrderedDict()
        if True:
            opts['ext_grids_to_load']['deltalin'] = {
                'dir': 'IC', 'file_format': 'nbkit_BigFileGrid',
                'dataset_name': 'LinearDensityLowRes_Ng%d' % opts['Ngrid'],
                'scale_factor': 1.0, 'nbkit_normalize': True, 'nbkit_setMean': 0.0}

    if False:
        # Ten 40-step L=1380 jerryou baoshift FastPM sims run by Jerry (Ding et al. 2017), 2048^3 particles
        opts['sim_name'] = 'jerryou_baoshift'
        opts['sim_irun'] = 3
        # use value from cmd line b/c later options depend on this
        opts['sim_seed'] = opts_update_dict.get('sim_seed', 300)
        opts['ssseed'] = opts_update_dict.get('ssseed', 1000+opts['sim_seed'])      # seed used to draw subsample
        opts['sim_Ntimesteps'] = 40
        opts['sim_Nptcles'] = 2048
        opts['sim_boxsize'] = 1380.0
        opts['sim_wig_now_string'] = 'wig'
        # scale factor of simulation snapshot (only used to rescale deltalin -- do not change via arg!)
        opts['sim_scale_factor'] = 0.6250
        #opts['sim_scale_factor'] = 1.0000

        # linear density (ICs of the sims)
        opts['ext_grids_to_load'] = OrderedDict()
        if True:
            opts['ext_grids_to_load']['deltalin'] = {
                'dir': 'IC', 'file_format': 'nbkit_BigFileGrid',
                'dataset_name': 'LinearDensityLowRes_Ng%d' % opts['Ngrid'],
                'scale_factor': 1.0, 'nbkit_normalize': True, 'nbkit_setMean': 0.0}
        
    if False:
        # Ten RunPB TreePM L=1380 sims by Martin White, 2048^3 particles
        opts['sim_name'] = 'RunPB'
        # use value from cmd line b/c later options depend on this
        opts['sim_seed'] = opts_update_dict.get('sim_seed', 0)
        opts['ssseed'] = opts_update_dict.get('ssseed', 10000+opts['sim_seed'])      # seed used to draw subsample
        opts['sim_Nptcles'] = 2048
        opts['sim_boxsize'] = 1380.0
        opts['sim_wig_now_string'] = 'wig'
        # scale factor of simulation snapshot (only used to rescale deltalin -- do not change via arg!)
        opts['sim_scale_factor'] = 0.6452

        # Grid files to load from disk
        opts['ext_grids_to_load'] = OrderedDict()
        if True:
            # deltalin.
            # Scale factor of linear density file will be rescaled to sim_scale_factor. 
            opts['ext_grids_to_load']['deltalin'] = {
                'dir': './', 'file_format': 'rhox_ms_binary', 
                'fname': 'tpmsph_ic.bin._rhoparams_PMGRID%d_MASS0.000to20.000_real4.rhogrid_x' % opts['Ngrid'],
                'scale_factor': 1.0/(1.0+75.0) #For RunPB, a_init=1/(1+75)
                }
        if True:
            # deltaZA.
            opts['ext_grids_to_load']['deltaZA'] = {
                'dir': './', 'file_format': 'rhox_ms_binary', 
                'fname': 'tpmsph_%.4f_za.bin._rhoparams_PMGRID%d_MASS0.000to20.000_real4.rhogrid_x' % (
                    opts['sim_scale_factor'],opts['Ngrid']),
                'scale_factor': opts['sim_scale_factor']
                }

    if True:
        # L=500 ms_gadget sims produced with MP-Gadget, 1536^3 particles
        opts['sim_name'] = 'ms_gadget'
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
            
            
        if False:
            # shift grids manually here in code (not implemented)
            opts['shifted_ext_grids'] = OrderedDict()
            opts['shifted_ext_grids']['deltalin_SHIFTEDBY_deltalin'] = {
                'grid_to_shift': 'deltalin',
                'grid_to_shift_smoothing': None,
                'displacement_source': 'deltalin', # compute zeldovich of this
                'displacement_source_smoothing': {'mode': 'Gaussian', 'R':10.0}}

        
            

    # ######################################################################
    # Catalogs to read
    # ######################################################################
    opts['cats'] = OrderedDict()

    if opts['sim_name'] == 'jerryou_baoshift':
    
        if True:
            ## nonuniform catalogs without ptcle masses
            if False:
                # halos without mass weight, narrow mass cut
                opts['cats']['delta_h'] = {
                    'in_fname': "fof_%.4f.hdf5_BOUNDS_log10M_12.8_13.8.hdf5"%opts['sim_scale_factor'],
                    'weight_ptcles_by': None}
            if True:
                # halos not weighted by mass but including mass info in file, broad mass cut
                opts['cats']['delta_h'] = {
                    'in_fname': "fof_%.4f.hdf5_WithMassCols.hdf5_BOUNDS_log10M_11.8_16.0.hdf5"%opts['sim_scale_factor'],
                    'weight_ptcles_by': None}
            if True:
                # halos in narrow mass bins, no mass weights
                opts['cats']['delta_h_M11.8-12.8'] = {
                    'in_fname': "fof_%.4f.hdf5_BOUNDS_log10M_11.8_12.8.hdf5"%opts['sim_scale_factor'],
                    'weight_ptcles_by': None}
                opts['cats']['delta_h_M12.8-13.8'] = {
                    'in_fname': "fof_%.4f.hdf5_BOUNDS_log10M_12.8_13.8.hdf5"%opts['sim_scale_factor'],
                    'weight_ptcles_by': None}
                opts['cats']['delta_h_M13.8-14.8'] = {
                    'in_fname': "fof_%.4f.hdf5_BOUNDS_log10M_13.8_14.8.hdf5"%opts['sim_scale_factor'],
                    'weight_ptcles_by': None}
                opts['cats']['delta_h_M14.8-16.0'] = {
                    'in_fname': "fof_%.4f.hdf5_BOUNDS_log10M_14.8_16.0.hdf5"%opts['sim_scale_factor'],
                    'weight_ptcles_by': None}
                

            # nonuniform NL DM catalog without ptcle masses (e.g. subsamples 101, 10007)
            if opts['Ngrid'] >= 256:
                opts['cats']['delta_m'] = {
                'in_fname': "sub_%.4f_0.01_ssseedX.hdf5"%opts['sim_scale_factor'],
                'weight_ptcles_by': None}
            else:
                # *** TMP 19 Feb 2018: use smaller subsamples for speedup ****
                opts['cats']['delta_m'] = {
                'in_fname': "sub_%.4f_0.01_ssseedX.hdf5_bw_sr0.171.hdf5"%opts['sim_scale_factor'],
                'weight_ptcles_by': None}

            if False:
                # DM subsample matched to halos.
                # L=500 sims: M=11-12: 0.000328, M=12-13: 0.000079, M=13-14: 0.000020
                # L=1380 sims: M=11.6-12.5: sr0.216, M=12.5-13.5: sr0.0716, M=13.5-14.5: sr0.0203
                #              M=11.8-12.8: sr0.171, M=12.8-13.8: sr0.05, M=13.8-14.8: sr0.0109
                opts['cats']['delta_m_matched_subsample'] = {
                    'in_fname': "sub_%.4f_0.01_ssseedX.hdf5_bw_sr0.171.hdf5"%opts['sim_scale_factor'],
                    'weight_ptcles_by': None}

        else:
            ## uniform catalogs with ptcle masses
            # halos
            raise Exception("change fnames to better ssseed subsamples")
            opts['cats']['delta_h'] = {
                #'in_fname': 'fof_%.4f.hdf5_BOUNDS_log10M_11.0_12.0.hdf5_unifdhby_DM_CALIBRATED_Ng%d.hdf5' % (opts['sim_scale_factor'],opts['Ngrid']),
                'in_fname': 'fof_%.4f.hdf5_BOUNDS_log10M_11.0_12.0.hdf5_UNICAT_baseb1-1.0-R0.0_baseS-DHBb1_MS-DHU-1.000000_Ng%d.hdf5'%(opts['sim_scale_factor'],opts['Ngrid']),
                'weight_ptcles_by': 'Mass'}

            # nonuniform NL DM catalog without ptcle masses (e.g. subsamples 101, 10007)
            opts['cats']['delta_m'] = {
                'in_fname': "sub_%.4f_idmod_101_0.hdf5_BOUNDS_None.hdf5_UNICAT_baseb1-1.0-R0.0_baseS-DHBb1_MS-DHU-1.000000_Ng%d.hdf5" % (opts['sim_scale_factor'], opts['Ngrid']),
                'weight_ptcles_by':'Mass'}

            # DM subsample matched to halos; this has 0.000328*2048^3 DM particles
            opts['cats']['delta_m_matched_subsample'] = {
            #'in_fname': "sub_%.4f_idmod_101_0.hdf5_sub0.000020.hdf5_BOUNDS_None.hdf5_unifdhby1.00_Ng%d.hdf5" % (opts['sim_scale_factor'], opts['Ngrid']),
            'in_fname': "sub_%.4f_idmod_101_0.hdf5_sub0.000328.hdf5_BOUNDS_None.hdf5_UNICAT_baseb1-1.0-R0.0_baseS-DHBb1_MS-DHU-1.000000_Ng%d.hdf5" % (opts['sim_scale_factor'], opts['Ngrid']),
                'weight_ptcles_by': 'Mass'}

            
    elif opts['sim_name'] == 'RunPB':
        
        if True:
            ## nonuniform catalogs without ptcle masses
            if False:
                # halos without mass weight, narrow mass cut
                opts['cats']['delta_h'] = {
                    'in_fname': "tpmsph_%.4f.fofp.ascii.ms_nbkfmt.hdf5_BOUNDS_log10M_13.8_14.8.hdf5"%opts['sim_scale_factor'],
                    'weight_ptcles_by': None}
            if True:
                # halos not weighted by mass but including mass info in file, broad mass cut
                opts['cats']['delta_h'] = {
                    'in_fname': "tpmsph_%.4f.fofp.ascii.ms_nbkfmt.hdf5_WithMassCols.hdf5_BOUNDS_log10M_11.8_16.0.hdf5"%opts['sim_scale_factor'],
                    'weight_ptcles_by': None}
                
            # nonuniform NL DM catalog without ptcle masses
            if opts['Ngrid'] < 256:
                # use smaller subsample for speedup tests
                opts['cats']['delta_m'] = {'in_fname': "tpmsph_%.4f.ms_subsample0.01_ssseedX_nbkfmt.hdf5_bw_sub0.000158.hdf5" % (opts['sim_scale_factor']), 
                                           'weight_ptcles_by': None}
            else:
                # use larger subsample for 512^3 production runs
                opts['cats']['delta_m'] = {'in_fname': "tpmsph_%.4f.ms_subsample0.01_ssseedX_nbkfmt.hdf5"%(opts['sim_scale_factor']),
                                           'weight_ptcles_by': None}

            # DM subsample matched to halos; M=11.5-12.5: 0.003290, 12.5-13.5: 0.000975, 13.5-14.5: 0.000158 
            #                                M=11.8-12.8: sr0.143, M=12.8-13.8: sr0.0428, M=13.8-14.8: sr0.0082
            opts['cats']['delta_m_matched_subsample'] = {
                'in_fname': "tpmsph_%.4f.ms_subsample0.01_ssseedX_nbkfmt.hdf5_bw_sr0.0082.hdf5" % (
                    opts['sim_scale_factor']),
                'weight_ptcles_by': None}
        else:
            ## uniform catalogs with ptcle masses
            # halos
            raise Exception("todo")


    elif opts['sim_name'] == 'ms_gadget':    

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


        # nonuniform NL DM catalog without ptcle masses (e.g. subsamples 101, 10007)
        # Note: load delta_m from full DM snapshot for ms_gadget using ext_grid above;
        # here we only load subsample.
        if False:
            if opts['Ngrid'] >= 256:
                opts['cats']['delta_m_subsample'] = {
                'in_fname': "snap_%.4f_sub_bw_sr0.025_ssseedX.hdf5"%opts['sim_scale_factor'],
                'weight_ptcles_by': None}
            else:
                # *** TMP: use smaller subsamples for speedup ****
                opts['cats']['delta_m_subsample'] = {
                'in_fname': "snap_%.4f_sub_bw_sr0.00025_ssseedX.hdf5"%opts['sim_scale_factor'],
                'weight_ptcles_by': None}

        if False:
            pass
            # DM subsample matched to halos.
            # L=500 sims: M=11-12: 0.000328, M=12-13: 0.000079, M=13-14: 0.000020
            # L=1380 sims: M=11.6-12.5: sr0.216, M=12.5-13.5: sr0.0716, M=13.5-14.5: sr0.0203
            #              M=11.8-12.8: sr0.171, M=12.8-13.8: sr0.05, M=13.8-14.8: sr0.0109
            #opts['cats']['delta_m_matched_subsample'] = {
            #    'in_fname': "sub_%.4f_0.01_ssseedX.hdf5_bw_sr0.171.hdf5"%opts['sim_scale_factor'],
            #    'weight_ptcles_by': None}


    else:
        raise Exception("Invalid sim_name %s" % opts['sim_name'])
        

        

    if False:
        # halos weighted by exact mass
        opts['cats']['delta_h_WEIGHT_M1'] = {
            'in_fname': opts['cats']['delta_h']['in_fname'], 
            'weight_ptcles_by': 'Mass[1e10Msun/h]'}
        # weighted by exact mass^2
        opts['cats']['delta_h_WEIGHT_M2'] = {
            'in_fname': opts['cats']['delta_h']['in_fname'], 
            'weight_ptcles_by': 'Mass[1e10Msun/h]^2'}
    if False:
        # halos weighted by noisy mass
        #for myscatter in ['0.04dex','0.1dex','0.3dex','0.6dex']:
        for myscatter in ['0.1dex','0.2dex','0.4dex']:
            opts['cats']['delta_h_WEIGHT_M%s'%myscatter] = {
                'in_fname': opts['cats']['delta_h']['in_fname'], 
                'weight_ptcles_by': 'MassWith%sScatter[1e10Msun/h]' % myscatter}



    # ######################################################################
    # Specify sources and targets for bias expansions and transfer functions
    # ######################################################################

    # Allowed quadratic_sources: 'growth','tidal_s2', 'tidal_G2'
    # Allowed cubic_sources: 'cube'
    opts['trf_specs'] = []

    ### DM FROM DM
    if False:

        if False:
            # hat_delta_m_from_deltalin = b1 deltalin
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltalin'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field='delta_m', ## match to nonlinear DM
                                             save_bestfit_field='hat_delta_m_from_deltalin'))

        if False:
            # hat_delta_m_from_deltalin_shifted = b1 deltalin_shifted
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltalin_SHIFTEDBY_deltalin'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field='delta_m', ## match to nonlinear DM
                                             save_bestfit_field='hat_delta_m_from_deltalin_shifted'))

        
        if False:
            # hat_delta_m_one_loop_deltalin = b1 deltalin + b2 F2 deltalin^2
            # **This checks trf functions of 1st and 2nd order PT contri to delta_nonlinear**
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltalin'],
                                             field_to_smoothen_and_square='deltalin',
                                             quadratic_sources=['F2'],
                                             target_field='delta_m', ## match to nonlinear DM
                                             save_bestfit_field='hat_delta_m_one_loop_deltalin'))

        if False:
            # hat_delta_m_one_loop_deltalin_shifted = b1 deltalin_shifted + b2 F2 deltalin_shifted^2
            # **This checks trf functions of 1st and 2nd order shifted PT contri to delta_nonlinear**
            # TODOOO: modify shift term in F2 (multiply Psi by 1-W_R)
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltalin_SHIFTEDBY_deltalin'],
                                             field_to_smoothen_and_square='deltalin_SHIFTEDBY_deltalin',
                                             quadratic_sources=['F2'],
                                             target_field='delta_m', ## match to nonlinear DM
                                             save_bestfit_field='hat_delta_m_one_loop_deltalin_shifted'))

        if False:
            # t1 deltalin + t2 deltalin_shifted
            # **This checks if shifted field matches delta_NL better than unshifted field does**
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltalin','deltalin_SHIFTEDBY_deltalin'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field='delta_m', ## match to nonlinear DM
                                             save_bestfit_field='hat_delta_m_deltalin_deltalin_shifted'))

        if False:
            # t1 deltaZ + t2 G2_shifted + t3 delta^3_shifted
            # **This checks if Marko's 13 calculation for DM is ok**
            opts['trf_specs'].append(TrfSpec(linear_sources=['1_SHIFTEDBY_deltalin',
                                                             'deltalin_G2_SHIFTEDBY_deltalin',
                                                             'deltalin_cube-mean_SHIFTEDBY_deltalin'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field='delta_m', ## match to nonlinear DM
                                             save_bestfit_field='hat_delta_m_T_1G23_SHIFTEDBY_deltalin'))

        if False:
            # t0 deltaZ + t1 delta_shifted + t2 delta^2_shifted + t3 G2_shifted + t4 delta^3_shifted
            # **This checks if Marko's and Val's xcorrel calc is ok**
            opts['trf_specs'].append(TrfSpec(linear_sources=['1_SHIFTEDBY_deltalin',
                                                             'deltalin_SHIFTEDBY_deltalin',
                                                             'deltalin_growth-mean_SHIFTEDBY_deltalin',
                                                             'deltalin_G2_SHIFTEDBY_deltalin',
                                                             'deltalin_cube-mean_SHIFTEDBY_deltalin'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field='delta_m', ## match to nonlinear DM
                                             save_bestfit_field='hat_delta_m_T_1delta12G23_SHIFTEDBY_deltalin'))

        if False:
            # Same but without deltaZ
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltalin_SHIFTEDBY_deltalin',
                                                             'deltalin_growth-mean_SHIFTEDBY_deltalin',
                                                             'deltalin_G2_SHIFTEDBY_deltalin',
                                                             'deltalin_cube-mean_SHIFTEDBY_deltalin'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field='delta_m', ## match to nonlinear DM
                                             save_bestfit_field='hat_delta_m_T_delta12G23_SHIFTEDBY_deltalin'))

        if False:
            # t1 deltaZ + t2 delta0_shifted
            # **This checks correl between deltaZ and delta0_shifted
            opts['trf_specs'].append(TrfSpec(linear_sources=['1_SHIFTEDBY_deltalin',
                                                             'deltalin_SHIFTEDBY_deltalin'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field='delta_m', ## match to nonlinear DM
                                             save_bestfit_field='hat_delta_m_T_1deltaG23_SHIFTEDBY_deltalin'))

        if False:
            # deltaZ = t1 delta0_shifted + t2 G2_shifted
            # **This checks for Marko if deltaZ can be expanded easily in shifted density**
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltalin_SHIFTEDBY_deltalin',
                                                             'deltalin_G2_SHIFTEDBY_deltalin'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field='1_SHIFTEDBY_deltalin', ## match to deltaZ
                                             save_bestfit_field='hat_deltaZ_T_delta1G2_SHIFTEDBY_deltalin'))

        if False:
            # Check how well shifted bias expansion can represent deltaZ. This shows if we can drop deltaZ.
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltalin_SHIFTEDBY_deltalin',
                                                             'deltalin_growth-mean_SHIFTEDBY_deltalin',
                                                             'deltalin_G2_SHIFTEDBY_deltalin'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field='1_SHIFTEDBY_deltalin', ## match to deltaZ
                                             save_bestfit_field='hat_deltaZ_T_delta12G2_SHIFTEDBY_deltalin'))

        if False:
            # Same but include delta^3_shifted
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltalin_SHIFTEDBY_deltalin',
                                                             'deltalin_growth-mean_SHIFTEDBY_deltalin',
                                                             'deltalin_G2_SHIFTEDBY_deltalin',
                                                             'deltalin_cube-mean_SHIFTEDBY_deltalin'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field='1_SHIFTEDBY_deltalin', ## match to deltaZ
                                             save_bestfit_field='hat_deltaZ_T_delta12G23_SHIFTEDBY_deltalin'))

        if True:
            # Check how well cubic Lagrangian bias can describe nonlinear delta_m. This gives lower limit on 1-r^2.
            # hat_delta_m = delta_Z + b1 deltalin(q+Psi) + b2 [deltalin^2-<deltalin^2>](q+Psi) + bG2 [G2](q+Psi)
            # + b3 [delta^3](q+Psi)
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltalin_SHIFTEDBY_deltalin',
                                                             'deltalin_growth-mean_SHIFTEDBY_deltalin',
                                                             'deltalin_G2_SHIFTEDBY_deltalin',
                                                             'deltalin_cube-mean_SHIFTEDBY_deltalin'
                                                             ],
                                             fixed_linear_sources=['1_SHIFTEDBY_deltalin'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field='delta_m',
                                             save_bestfit_field='hat_delta_m_from_1_Tdeltalin2G23_SHIFTEDBY_PsiZ'))

    if False:
        ### DM FROM HALOS  or  DM Zeldovich FROM HALOS
        #for target in ['delta_m','deltaZA']:
        #for target in ['deltaZA']:
        for target in ['delta_m']:
            if False:
                # hat delta_m = t0 delta_h + t1 delta_h^2 + t2 G2[delta_h]
                opts['trf_specs'].append(TrfSpec(linear_sources=['delta_h'], 
                                                 field_to_smoothen_and_square='delta_h', 
                                                 quadratic_sources=['growth','tidal_G2'],
                                                 target_field=target,
                                                 save_bestfit_field='hat_%s_from_b1b2bG2_delta_h'%target))

            if False:
                # hat_delta_m_from_b1_delta_h_M0 = t0 delta_h^{M^0}
                # delta_h^{M^n} is mass-weighted halo density weighted by M^n.
                opts['trf_specs'].append(TrfSpec(linear_sources=['delta_h'], 
                                                 field_to_smoothen_and_square=None,
                                                 quadratic_sources=[],
                                                 target_field=target,
                                                 save_bestfit_field='hat_%s_from_b1_delta_h_M0'%target))

            if False:
                # hat_delta_m_from_b1_delta_h_M1 = t1 delta_h^{M^1}.
                # delta_h^{M^n} is mass-weighted halo density weighted by M^n, using exact mass.
                opts['trf_specs'].append(TrfSpec(linear_sources=['delta_h_WEIGHT_M1'], 
                                                 field_to_smoothen_and_square=None,
                                                 quadratic_sources=[],
                                                 target_field=target,
                                                 save_bestfit_field='hat_%s_from_b1_delta_h_M1'%target))

            if True:
                # hat_delta_m_from_b1_delta_h_M0M1 = t0 delta_h^{M^0} + t1 delta_h^{M^1}.
                # delta_h^{M^n} is mass-weighted halo density weighted by M^n, using exact mass.
                opts['trf_specs'].append(TrfSpec(linear_sources=['delta_h','delta_h_WEIGHT_M1'], 
                                                 field_to_smoothen_and_square=None,
                                                 quadratic_sources=[],
                                                 target_field=target,
                                                 save_bestfit_field='hat_%s_from_b1_delta_h_M0_M1'%target))

            if False:
                # hat_delta_m_from_b1_delta_h_M0M1M2 = t0 delta_h^{M^0} + t1 delta_h^{M^1} + t2 delta_h^{M^2}.
                # delta_h^{M^n} is mass-weighted halo density weighted by M^n.
                opts['trf_specs'].append(TrfSpec(linear_sources=['delta_h','delta_h_WEIGHT_M1','delta_h_WEIGHT_M2'], 
                                                 field_to_smoothen_and_square=None,
                                                 quadratic_sources=[],
                                                 target_field=target,
                                                 export_bestfit_field=True,
                                                 save_bestfit_field='hat_%s_from_b1_delta_h_M0_M1_M2'%target))

            if False:
                # hat_delta_m_from_b1_delta_h_M0_M0.3dex = t0 delta_h^{M^0} + t1 delta_h^{M^1_noisy}.
                # delta_h^{M^1_noisy} is mass-weighted halo density, using noisy M.
                #for myscatter in ['0.04dex','0.1dex','0.3dex','0.6dex']:
                for myscatter in ['0.1dex','0.2dex','0.4dex']:
                    opts['trf_specs'].append(TrfSpec(linear_sources=['delta_h','delta_h_WEIGHT_M%s'%myscatter], 
                                                     field_to_smoothen_and_square=None,
                                                     quadratic_sources=[],
                                                     target_field=target,
                                                     save_bestfit_field='hat_%s_from_b1_delta_h_M0_M%s'%(target,myscatter)))
                   
            if False:
                # Optimally combine a few halo mass bins (corresponds to knowing mass very crudely):
                # hat_delta_m_from_b1_delta_h_Mbins11.8_12.8_13.8_14.8_16.0 = t0 delta_h^{M=11.8-12.8} + t1 delta_h^{M=12.8-13.8} + ...
                opts['trf_specs'].append(TrfSpec(
                    linear_sources=['delta_h_M11.8-12.8','delta_h_M12.8-13.8','delta_h_M13.8-14.8','delta_h_M14.8-16.0',], 
                    field_to_smoothen_and_square=None,
                    quadratic_sources=[],
                    target_field=target,
                    save_bestfit_field='hat_%s_from_b1_delta_h_Mbins11.8_12.8_13.8_14.8_16.0'%target))


            
    ### HALO NUMBER DENSITY FROM DM
    #if True:
    #for target in ['delta_h_WEIGHT_M1']:
    #for target in ['hat_delta_m_from_b1_delta_h_M0_M1']: # test bias expansion of \hat\delta_m obtained from mass-weighted halos
    #for target in ['hat_deltaZA_from_b1_delta_h_M0_M1']: # test bias expansion of \hat\delta_Z obtained from mass-weighted halos
    for target in ['delta_h']:
    
        if False:
            # hat_delta_h_from_b1_deltalin = b1 deltalin
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltalin'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field=target,
                                             save_bestfit_field='hat_delta_h_from_b1_deltalin'))

        if False:
            # hat_delta_h_from_b1_deltaZA = b1 deltaZA
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltaZA'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field=target,
                                             save_bestfit_field='hat_delta_h_from_b1_deltaZA'))

        if False:
            #if True:
            # hat_delta_h_from_b1_delta_m = b1 delta_m (linear Eulerian bias)
            opts['trf_specs'].append(TrfSpec(linear_sources=['delta_m'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field=target,
                                             save_bestfit_field='hat_delta_h_from_b1_delta_m'))

        if False:
            # Expand in nonlinear density, hat_delta_h = b1 delta_m_NL + b2 delta_m_NL^2 + b_G2 G2[delta_m_NL].
            # Bad b/c squaring delta_m_NL is very large
            opts['trf_specs'].append(TrfSpec(linear_sources=['delta_m'],
                                             field_to_smoothen_and_square='delta_m',
                                             quadratic_sources=['growth','tidal_G2'],
                                             target_field=target,
                                             save_bestfit_field='hat_delta_h_from_deltanonl_12G2'))
            
        if False:
            # hat_delta_h_one_loop_deltalin = b1 deltalin + b1' F2 deltalin^2 + b2 deltalin^2 + b_G2 G2[deltalin].
            # **This includes free coefficient for F2**
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltalin'],
                                             field_to_smoothen_and_square='deltalin',
                                             quadratic_sources=['F2','growth','tidal_G2'],
                                             target_field=target,
                                             save_bestfit_field='hat_delta_h_one_loop_deltalin'))
        if False:
            # hat_delta_h_one_loop_deltalin = b1 ( deltalin + F2 deltalin^2 ) + b2 deltalin^2 + b_G2 G2[deltalin].
            # **This fixes coefficient for F2** (seems more consistent)
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltalin'],
                                             field_to_smoothen_and_square='deltalin',
                                             quadratic_sources=['F2','growth','tidal_G2'],
                                             sources_for_trf_fcn=['deltalin+deltalin_F2','deltalin_growth','deltalin_tidal_G2'],
                                             target_field=target,
                                             save_bestfit_field='hat_delta_h_one_loop_deltaSPT'))

        if False: #True:
            # hat_delta_h_one_loop_deltaZA = b1 deltaZA + b2 deltaZA^2 + b_G2 G2[deltaZA]. 
            # Do not add b1' kappa2 deltaZA^2 b/c kappa2=-G2 so it's already in G2.
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltaZA'],
                                             field_to_smoothen_and_square='deltaZA',
                                             quadratic_sources=['growth','tidal_G2'],
                                             target_field=target,
                                             save_bestfit_field='hat_delta_h_one_loop_deltaZA'))


        if False:
            # Hybrid expansion in b1 delta_NL and b2 delta_lin^2: b1 delta_NL + b2 delta_lin^2 + b_G2 G2[delta_lin]
            opts['trf_specs'].append(TrfSpec(linear_sources=['delta_m'],
                                             field_to_smoothen_and_square='deltalin',
                                             quadratic_sources=['growth','tidal_G2'],
                                             target_field=target,
                                             save_bestfit_field='hat_delta_h_from_delta_m_deltalin2_deltalinG2'))
            
        if False: #True:
            # Hybrid expansion in deltaZ and delta_lin^2: hat_delta_h = b1 delta_Z + b2 delta_lin^2 + b_G2 G2[delta_lin]
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltaZA'],
                                             field_to_smoothen_and_square='deltalin',
                                             quadratic_sources=['growth','tidal_G2'],
                                             target_field=target,
                                             save_bestfit_field='hat_delta_h_from_deltaZ_deltalin2_deltalinG2'))
        if False: #if True:
            # Full hybrid expansion in deltaZ^2 and delta_lin^2: hat_delta_h = ...
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltalin','deltaZA'],
                                             field_to_smoothen_and_square='deltalin',
                                             quadratic_sources=['growth','tidal_G2'],
                                             field_to_smoothen_and_square2='deltaZA',
                                             quadratic_sources2=['growth','tidal_G2'],
                                             target_field=target,
                                             save_bestfit_field='hat_delta_h_from_deltalin_deltaZ_deltalin2G2_deltaZ2G2'))
        if False:
            # Other hybrid expansion in deltaZ^2 and delta_lin^2: hat_delta_h = b1 delta_Z + b1' deltalin + F2 + b2 delta_lin^2 + b_G2 G2[delta_lin]
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltalin','deltaZA'],
                                             field_to_smoothen_and_square='deltalin',
                                             quadratic_sources=['growth','F2','tidal_G2'],
                                             target_field=target,
                                             save_bestfit_field='hat_delta_h_from_deltalin_deltaZ_deltalin2F2G2'))


        if False:
            # Hybrid expansion in deltaZ^2 with 2 smoothing scales: b1 deltaZ + b2 deltaZ_R1^2 + b2' deltaZ_R2^2 + ...
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltaZA'],
                                             field_to_smoothen_and_square='deltaZA',
                                             quadratic_sources=['growth','tidal_G2'],
                                             field_to_smoothen_and_square2='deltaZA',
                                             quadratic_sources2=['growth','tidal_G2'],
                                             target_field=target,
                                             save_bestfit_field='hat_delta_h_from_deltaZ_deltaZ2G2_deltaZ2G2'))

        if False:
            # hat_delta_h_from_b1_deltalin_shifted = b1 deltalin(x+Psi_R)
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltalin_SHIFTEDBY_deltalin'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field=target,
                                             save_bestfit_field='hat_delta_h_from_b1_deltalin_shifted'))

        if False:
            # hat_delta_h_from_b1_deltalin_shifted = b1 deltalin(x+Psi_R) + b2  deltalin^2(x+Psi_R) + G2
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltalin_SHIFTEDBY_deltalin'],
                                             field_to_smoothen_and_square='deltalin_SHIFTEDBY_deltalin',
                                             quadratic_sources=['growth','tidal_G2'],
                                             target_field=target,
                                             save_bestfit_field='hat_delta_h_from_b1_b2_bG2_deltalin_shifted'))


        if False:
            #if False:
            # Linear Lagrangian bias: delta_Z + b1 deltalin(q+Psi)
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltalin_SHIFTEDBY_deltalin'],
                                             fixed_linear_sources=['1_SHIFTEDBY_deltalin'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field=target,
                                             save_bestfit_field='hat_delta_h_from_1_Tdeltalin_SHIFTEDBY_PsiZ'))

            #if True:
        if False:
            # Linear Lagrangian bias without deltaZ: b1 deltalin(q+Psi)
            # Bad b/c cannot absorb 2nd terms of deltaZ.
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltalin_SHIFTEDBY_deltalin'],
                                             fixed_linear_sources=[],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field=target,
                                             save_bestfit_field='hat_delta_h_from_Tdeltalin_SHIFTEDBY_PsiZ'))

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
        if False:
            #if True:
            # Quadratic Lagrangian bias without deltaZ: b1 deltalin(q+Psi) + b2 [deltalin^2-<deltalin^2>](q+Psi) + bG2 [G2](q+Psi)
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltalin_SHIFTEDBY_deltalin',
                                                             'deltalin_growth-mean_SHIFTEDBY_deltalin',
                                                             'deltalin_G2_SHIFTEDBY_deltalin'],
                                             fixed_linear_sources=[],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field=target,
                                             save_bestfit_field='hat_delta_h_from_Tdeltalin2G2_SHIFTEDBY_PsiZ'))

            
        if False:
            # Lagrangian bias with trf fcn on 1: b0 delta_Z + b1 deltalin(q+Psi) b2 [deltalin^2-<deltalin^2>](q+Psi) 
            # + bG2 [G2](q+Psi)
            opts['trf_specs'].append(TrfSpec(linear_sources=['1_SHIFTEDBY_deltalin',
                                                             'deltalin_SHIFTEDBY_deltalin',
                                                             'deltalin_growth-mean_SHIFTEDBY_deltalin',
                                                             'deltalin_G2_SHIFTEDBY_deltalin'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field=target,
                                             save_bestfit_field='hat_delta_h_from_T1_Tdeltalin2G2_SHIFTEDBY_PsiZ'))


        if True:
            #if True:
            # Cubic Lagrangian bias with delta^3: delta_Z + b1 deltalin(q+Psi) + b2 [deltalin^2-<deltalin^2>](q+Psi) + bG2 [G2](q+Psi)
            # + b3 [delta^3](q+Psi)
            # BEST MODEL WITHOUT MASS WEIGHTING
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltalin_SHIFTEDBY_deltalin',
                                                             'deltalin_growth-mean_SHIFTEDBY_deltalin',
                                                             'deltalin_G2_SHIFTEDBY_deltalin',
                                                             'deltalin_cube-mean_SHIFTEDBY_deltalin'
                                                             ],
                                             fixed_linear_sources=['1_SHIFTEDBY_deltalin'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field=target,
                                             save_bestfit_field='hat_delta_h_from_1_Tdeltalin2G23_SHIFTEDBY_PsiZ'))

        if False:
            #if False:
            # Cubic Lagrangian bias without deltaZ: b1 deltalin(q+Psi) + b2 [deltalin^2-<deltalin^2>](q+Psi) + bG2 [G2](q+Psi)
            # + b3 [delta^3](q+Psi)
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltalin_SHIFTEDBY_deltalin',
                                                             'deltalin_growth-mean_SHIFTEDBY_deltalin',
                                                             'deltalin_G2_SHIFTEDBY_deltalin',
                                                             'deltalin_cube-mean_SHIFTEDBY_deltalin'
                                                             ],
                                             fixed_linear_sources=[],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field=target,
                                             save_bestfit_field='hat_delta_h_from_Tdeltalin2G23_SHIFTEDBY_PsiZ'))

            
        if False:
            # Lagrangian bias with 2LPT displacement: 1[q+Psi2] + b1 deltalin(q+Psi2) + b2 [deltalin^2-<deltalin^2>](q+Psi2)
            #  + bG2 [G2](q+Psi)
            opts['trf_specs'].append(TrfSpec(linear_sources=['deltalin_SHIFTEDBY_Psi2LPT_deltalin',
                                                             'deltalin_growth-mean_SHIFTEDBY_Psi2LPT_deltalin',
                                                             'deltalin_G2_SHIFTEDBY_Psi2LPT_deltalin'],
                                             fixed_linear_sources=['1_SHIFTEDBY_Psi2LPT_deltalin'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field=target,
                                             save_bestfit_field='hat_delta_h_from_1_Tdeltalin2G2_SHIFTEDBY_Psi2LPT'))

        if False:
            # Lagrangian bias with 2LPT displacement and trf fcn on 1: b0 [q+Psi2] + b1 deltalin(q+Psi2) 
            # + b2 [deltalin^2-<deltalin^2>](q+Psi2) + bG2 [G2](q+Psi)
            opts['trf_specs'].append(TrfSpec(linear_sources=['1_SHIFTEDBY_Psi2LPT_deltalin',
                                                             'deltalin_SHIFTEDBY_Psi2LPT_deltalin',
                                                             'deltalin_growth-mean_SHIFTEDBY_Psi2LPT_deltalin',
                                                             'deltalin_G2_SHIFTEDBY_Psi2LPT_deltalin'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field=target,
                                             save_bestfit_field='hat_delta_h_from_T1_Tdeltalin2G2_SHIFTEDBY_Psi2LPT'))

            
        if False:
            # Test 2 codes that generate delta_Z (main_ms_gadget_shift.. vs Genic: got rcc=1 and same P(k) so fine)
            opts['trf_specs'].append(TrfSpec(linear_sources=['1_SHIFTEDBY_deltalin', 'deltaZA'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field=target,
                                             save_bestfit_field='hat_delta_h_from_deltaZ_1shifted'))

        if False:
            # Test 2LPT by comparing vs delta_m
            opts['trf_specs'].append(TrfSpec(linear_sources=['1_SHIFTEDBY_deltalin'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field='delta_m',
                                             save_bestfit_field='hat_delta_m_from_1_SHIFTEDBY_PsiZ'))
            opts['trf_specs'].append(TrfSpec(linear_sources=['1_SHIFTEDBY_Psi2LPT_deltalin'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field='delta_m',
                                             save_bestfit_field='hat_delta_m_from_1_SHIFTEDBY_Psi2LPT'))
            opts['trf_specs'].append(TrfSpec(linear_sources=['1_SHIFTEDBY_-Psi2LPT_deltalin'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field='delta_m',
                                             save_bestfit_field='hat_delta_m_from_1_SHIFTEDBY_-Psi2LPT'))



    if False:
        ### MASS-WEIGHTED HALOS from DM
        if False:
            # Directly minimize the full residual between mass-weighted halos and PT,
            # |alpha delta_n + beta delta_M - (1_shifted + b2 deltalin^2_shifted + bG2 G2lin_shifted + b3 deltalin^3_shifted)|^2.
            # Set b1^L=0 to avoid degeneracy. 
            opts['trf_specs'].append(TrfSpec(linear_sources=['delta_h', 'delta_h_WEIGHT_M1',
                                                             #'deltalin_SHIFTEDBY_deltalin', # set b1^L=0
                                                             'deltalin_growth-mean_SHIFTEDBY_deltalin',
                                                             'deltalin_G2_SHIFTEDBY_deltalin',
                                                             'deltalin_cube-mean_SHIFTEDBY_deltalin'],
                                             field_to_smoothen_and_square=None,
                                             quadratic_sources=[],
                                             target_field='deltaZA',
                                             save_bestfit_field='hat_deltaZA_from_T_delta_h_M0_M1_Tdeltalin2G23_nob1_SHIFTEDBY_PsiZ'))

        if False:
            # Same but allow free b1^L. Looks good, but not sure what we get b/c can just set data to 0 to get low noise.
            opts['trf_specs'].append(TrfSpec(
                linear_sources=['delta_h', 'delta_h_WEIGHT_M1',
                                'deltalin_SHIFTEDBY_deltalin', # free b1^L
                                'deltalin_growth-mean_SHIFTEDBY_deltalin',
                                'deltalin_G2_SHIFTEDBY_deltalin',
                                'deltalin_cube-mean_SHIFTEDBY_deltalin'],
                field_to_smoothen_and_square=None,
                quadratic_sources=[],
                target_field='deltaZA',
                save_bestfit_field='hat_deltaZA_from_T_delta_h_M0_M1_Tdeltalin2G23_withb1_SHIFTEDBY_PsiZ'))

        if False:
            # Minimize difference between mass-weighted halos (=data) with fixed overall normalization and bias model, 
            # i.e. minimize
            # |delta_n + beta (delta_M-delta_n) - (1_shifted + b1^L delta0_shifted + b2^L delta0^2_shifted 
            # + bG2^L G2_shifted + b3^L delta0^3_shifted|^2.
            # Note that this should be the same as alpha delta_n + beta delta_M with imposing alpha+beta=1 (at k=0 this 
            # enforces correct avg density; at k>0 it is not so clear why we want this).
            # PROBLEMS: - Plots show target=delta_n and not so easy to do noise curves relative to delta_n+delta_M.
            #           - Minimizes (data-model)^2 rather than (data-model)^2/model^2 which seems more relevant
            #           - The model involves data, so it's not showing how well we can actually model the observations.
            #             (If we can represent the observation perfectly in terms of some other observation we still haven't learned anything).
            opts['trf_specs'].append(TrfSpec(
                linear_sources=['delta_h', 'delta_h_WEIGHT_M1', 
                                'deltalin_SHIFTEDBY_deltalin', 
                                'deltalin_growth-mean_SHIFTEDBY_deltalin', 
                                'deltalin_G2_SHIFTEDBY_deltalin', 
                                'deltalin_cube-mean_SHIFTEDBY_deltalin'], # all possible linear sources to load
                non_orth_linear_sources=['[delta_h_WEIGHT_M1]_MINUS_[delta_h]'], # sources with trf fcns that are not included in orthogonalization
                sources_for_trf_fcn=['deltalin_SHIFTEDBY_deltalin',
                                     'deltalin_growth-mean_SHIFTEDBY_deltalin',
                                     'deltalin_G2_SHIFTEDBY_deltalin', 
                                     'deltalin_cube-mean_SHIFTEDBY_deltalin',
                                     '[delta_h_WEIGHT_M1]_MINUS_[delta_h]'
                                     ],   # sources actually used for trf fcns (nonorth must be last)
                fixed_linear_sources=['1_SHIFTEDBY_deltalin'],
                field_to_smoothen_and_square=None,
                quadratic_sources=[],
                target_field='delta_h',
                save_bestfit_field='hat_delta_h_from_T_delta_M1_MINUS_delta_h_1_T_orth_deltalin12G23_SHIFTEDBY_PsiZ'))

        if False:
            # Same but without deltaZ
            opts['trf_specs'].append(TrfSpec(
                linear_sources=['delta_h', 'delta_h_WEIGHT_M1', 
                                'deltalin_SHIFTEDBY_deltalin', 
                                'deltalin_growth-mean_SHIFTEDBY_deltalin', 
                                'deltalin_G2_SHIFTEDBY_deltalin', 
                                'deltalin_cube-mean_SHIFTEDBY_deltalin'], # all possible linear sources to load
                non_orth_linear_sources=['[delta_h_WEIGHT_M1]_MINUS_[delta_h]'], # sources with trf fcns that are not included in orthogonalization
                sources_for_trf_fcn=['deltalin_SHIFTEDBY_deltalin',
                                     'deltalin_growth-mean_SHIFTEDBY_deltalin',
                                     'deltalin_G2_SHIFTEDBY_deltalin', 
                                     'deltalin_cube-mean_SHIFTEDBY_deltalin',
                                     '[delta_h_WEIGHT_M1]_MINUS_[delta_h]'
                                     ],   # sources actually used for trf fcns (nonorth must be last)
                fixed_linear_sources=[],
                field_to_smoothen_and_square=None,
                quadratic_sources=[],
                target_field='delta_h',
                save_bestfit_field='hat_delta_h_from_T_delta_M1_MINUS_delta_h_T_orth_deltalin12G23_SHIFTEDBY_PsiZ'))

            #if True:
        if False:
            # Change mass weighting optimization to avoid some of the issues above:
            # Target=delta_M+alpha1*delta_n^orth, where delta_n^orth is orthogonal to delta_M so that 
            # it can only add power but not decrease power of the data.
            # Model: cubic Lagr. bias. 
            # Result: 1-r^2 looks good. But power spectrum has bump at high k b/c adding power from delta_n^orth.
            # Fix this below.
            opts['trf_specs'].append(TrfSpec(
                linear_sources=['deltalin_SHIFTEDBY_deltalin', 
                                'deltalin_growth-mean_SHIFTEDBY_deltalin', 
                                'deltalin_G2_SHIFTEDBY_deltalin',
                                'deltalin_cube-mean_SHIFTEDBY_deltalin'],
                fixed_linear_sources=['1_SHIFTEDBY_deltalin'], # if target_spec!=None, using this gives slightly sub-optimal target weights
                field_to_smoothen_and_square=None,
                quadratic_sources=[],
                target_field='BEST_MSE_delta_h_M1M0_for_1_deltalin12G23_SHIFTEDBY_PsiZ', 
                target_spec=TargetSpec(
                    linear_target_contris=['delta_h_WEIGHT_M1','delta_h'],
                    minimization_objective='(target0+T*other_targets-T*sources)^2',
                    target_norm={'type': 'alpha0=1'},
                    save_bestfit_target_field='BEST_MSE_delta_h_M1M0_for_1_deltalin12G23_SHIFTEDBY_PsiZ' 
                       # where to save bestfit combi of targets
                    ),
                save_bestfit_field='hat_delta_h_M1M0_MSE_noZ_from_1_T_deltalin12G23_SHIFTEDBY_PsiZ' # where to save bestfit combi of sources
                ))
            
        if False:
            # same but without deltaZ in model
            # Result: very similar to case with deltaZ.
            opts['trf_specs'].append(TrfSpec(
                linear_sources=['deltalin_SHIFTEDBY_deltalin', 
                                'deltalin_growth-mean_SHIFTEDBY_deltalin', 
                                'deltalin_G2_SHIFTEDBY_deltalin',
                                'deltalin_cube-mean_SHIFTEDBY_deltalin'],
                fixed_linear_sources=[], # if target_spec!=None, using this gives slightly sub-optimal target weights
                field_to_smoothen_and_square=None,
                quadratic_sources=[],
                target_field='BEST_MSE_delta_h_M1M0_for_deltalin12G23_SHIFTEDBY_PsiZ', 
                target_spec=TargetSpec(
                    linear_target_contris=['delta_h_WEIGHT_M1','delta_h'],
                    minimization_objective='(target0+T*other_targets-T*sources)^2',
                    target_norm={'type': 'alpha0=1'},
                    save_bestfit_target_field='BEST_MSE_delta_h_M1M0_for_deltalin12G23_SHIFTEDBY_PsiZ' 
                       # where to save bestfit combi of targets
                    ),
                save_bestfit_field='hat_delta_h_M1M0_MSE_noZ_from_T_deltalin12G23_SHIFTEDBY_PsiZ' # where to save bestfit combi of sources
                ))

        if False:
            # Same but include deltaZ in model and change ordering of target contris, so data=target=delta_n+alpha1*delta_M^orth.
            # This checks for bugs: 1-r^2 should not depend on ordering of target contris.
            # Result: ok, 1-r^2 very similar to data contris with reversed ordering
            opts['trf_specs'].append(TrfSpec(
                linear_sources=['deltalin_SHIFTEDBY_deltalin', 
                                'deltalin_growth-mean_SHIFTEDBY_deltalin', 
                                'deltalin_G2_SHIFTEDBY_deltalin',
                                'deltalin_cube-mean_SHIFTEDBY_deltalin'],
                fixed_linear_sources=['1_SHIFTEDBY_deltalin'], # if target_spec!=None, using this gives slightly sub-optimal target weights
                field_to_smoothen_and_square=None,
                quadratic_sources=[],
                target_field='BEST_MSE_delta_h_M0M1_for_1_deltalin12G23_SHIFTEDBY_PsiZ', 
                target_spec=TargetSpec(
                    linear_target_contris=['delta_h','delta_h_WEIGHT_M1'],
                    minimization_objective='(target0+T*other_targets-T*sources)^2',
                    target_norm={'type': 'alpha0=1'},
                    save_bestfit_target_field='BEST_MSE_delta_h_M0M1_for_1_deltalin12G23_SHIFTEDBY_PsiZ' 
                       # where to save bestfit combi of targets
                    ),
                save_bestfit_field='hat_delta_h_M0M1_MSE_noZ_from_1_T_deltalin12G23_SHIFTEDBY_PsiZ' # where to save bestfit combi of sources
                ))
            
        if False:
            # Same but without deltaZ in model
            opts['trf_specs'].append(TrfSpec(
                linear_sources=['deltalin_SHIFTEDBY_deltalin', 
                                'deltalin_growth-mean_SHIFTEDBY_deltalin', 
                                'deltalin_G2_SHIFTEDBY_deltalin',
                                'deltalin_cube-mean_SHIFTEDBY_deltalin'],
                fixed_linear_sources=[], # if target_spec!=None, using this gives slightly sub-optimal target weights
                field_to_smoothen_and_square=None,
                quadratic_sources=[],
                target_field='BEST_MSE_delta_h_M0M1_for_deltalin12G23_SHIFTEDBY_PsiZ', 
                target_spec=TargetSpec(
                    linear_target_contris=['delta_h','delta_h_WEIGHT_M1'],
                    minimization_objective='(target0+T*other_targets-T*sources)^2',
                    target_norm={'type': 'alpha0=1'},
                    save_bestfit_target_field='BEST_MSE_delta_h_M0M1_for_deltalin12G23_SHIFTEDBY_PsiZ' 
                       # where to save bestfit combi of targets
                    ),
                save_bestfit_field='hat_delta_h_M0M1_MSE_noZ_from_T_deltalin12G23_SHIFTEDBY_PsiZ' # where to save bestfit combi of sources
                ))

        if True:
            # Same but normalize <d^2>=P_nn (power spectrum of halo number density).
            # This allows to look at noise curves in comparison to curves without mass weighting, not 
            # just 1-r^2. Include mass scatter
            # USE THIS AS BEST MASS-WEIGHTING SCHEME?
            #for myscatter in ['0.04dex','0.1dex','0.3dex','0.6dex']:
            for myscatter in ['0.4dex','0.2dex','0.1dex']:
                opts['trf_specs'].append(TrfSpec(
                    linear_sources=['deltalin_SHIFTEDBY_deltalin', 
                                    'deltalin_growth-mean_SHIFTEDBY_deltalin', 
                                    'deltalin_G2_SHIFTEDBY_deltalin',
                                    'deltalin_cube-mean_SHIFTEDBY_deltalin'],
                    fixed_linear_sources=[], # if target_spec!=None, using this gives slightly sub-optimal target weights
                    field_to_smoothen_and_square=None,
                    quadratic_sources=[],
                    target_field='BEST_MSE_delta_h_M0M%s_MPnn_for_deltalin12G23_SHIFTEDBY_PsiZ'%myscatter,
                    target_spec=TargetSpec(
                        linear_target_contris=['delta_h','delta_h_WEIGHT_M%s'%myscatter],
                        minimization_objective='(target0+T*other_targets-T*sources)^2',
                        target_norm={'type': 'MatchPower', 
                                     'Pk_to_match_id1': 'delta_h',
                                     'Pk_to_match_id2': 'delta_h'},
                        save_bestfit_target_field='BEST_MSE_delta_h_M0M%s_MPnn_for_deltalin12G23_SHIFTEDBY_PsiZ'%myscatter
                           # where to save bestfit combi of targets
                        ),
                    save_bestfit_field='hat_delta_h_M0M%s_MSE_noZ_MPnn_from_T_deltalin12G23_SHIFTEDBY_PsiZ'%myscatter
                # where to save bestfit combi of sources
                    ))
        
        if True:
            # Same but without mass scatter, and don't include deltaZ in model.
            opts['trf_specs'].append(TrfSpec(
                linear_sources=['deltalin_SHIFTEDBY_deltalin', 
                                'deltalin_growth-mean_SHIFTEDBY_deltalin', 
                                'deltalin_G2_SHIFTEDBY_deltalin',
                                'deltalin_cube-mean_SHIFTEDBY_deltalin'],
                fixed_linear_sources=[], # if target_spec!=None, using this gives slightly sub-optimal target weights
                field_to_smoothen_and_square=None,
                quadratic_sources=[],
                target_field='BEST_MSE_delta_h_M0M1_MPnn_for_deltalin12G23_SHIFTEDBY_PsiZ', 
                target_spec=TargetSpec(
                    linear_target_contris=['delta_h','delta_h_WEIGHT_M1'],
                    minimization_objective='(target0+T*other_targets-T*sources)^2',
                    target_norm={'type': 'MatchPower', 
                                 'Pk_to_match_id1': 'delta_h',
                                 'Pk_to_match_id2': 'delta_h'},
                    save_bestfit_target_field='BEST_MSE_delta_h_M0M1_MPnn_for_deltalin12G23_SHIFTEDBY_PsiZ' 
                       # where to save bestfit combi of targets
                    ),
                save_bestfit_field='hat_delta_h_M0M1_MSE_noZ_MPnn_from_T_deltalin12G23_SHIFTEDBY_PsiZ' # where to save bestfit combi of sources
                ))
            
        if False:
            # Same but no mass scatter, and change ordering of data vector to d=delta_M+delta_n^orth
            opts['trf_specs'].append(TrfSpec(
                linear_sources=['deltalin_SHIFTEDBY_deltalin', 
                                'deltalin_growth-mean_SHIFTEDBY_deltalin', 
                                'deltalin_G2_SHIFTEDBY_deltalin',
                                'deltalin_cube-mean_SHIFTEDBY_deltalin'],
                fixed_linear_sources=[], # if target_spec!=None, using this gives slightly sub-optimal target weights
                field_to_smoothen_and_square=None,
                quadratic_sources=[],
                target_field='BEST_MSE_delta_h_M1M0_MPnn_for_deltalin12G23_SHIFTEDBY_PsiZ', 
                target_spec=TargetSpec(
                    linear_target_contris=['delta_h_WEIGHT_M1','delta_h'],
                    minimization_objective='(target0+T*other_targets-T*sources)^2',
                    target_norm={'type': 'MatchPower', 
                                 'Pk_to_match_id1': 'delta_h',
                                 'Pk_to_match_id2': 'delta_h'},
                    save_bestfit_target_field='BEST_MSE_delta_h_M1M0_MPnn_for_deltalin12G23_SHIFTEDBY_PsiZ' 
                       # where to save bestfit combi of targets
                    ),
                save_bestfit_field='hat_delta_h_M1M0_MSE_noZ_MPnn_from_T_deltalin12G23_SHIFTEDBY_PsiZ' # where to save bestfit combi of sources
                ))


            

        ### CHANGE MINIMIZATION_OBJECTIVE to (data-model)^2/data^2
        
        if False:
            # Minimize (data-model)^2/data^2, where data = alpha_0 delta_n + alpha_1 delta_M^orth and 
            # model = beta_0 deltaZ + beta1 delta0_shifted_orth + beta2 delta0^2_shifted_orth + beta3 G2_shifted_orth
            # +beta4 delta0^3_shifted
            # Minimization for alpha is taylored to this problem; minimization for model-beta's is same as before.
            opts['trf_specs'].append(TrfSpec(
                linear_sources=['1_SHIFTEDBY_deltalin',
                                'deltalin_SHIFTEDBY_deltalin', 
                                'deltalin_growth-mean_SHIFTEDBY_deltalin', 
                                'deltalin_G2_SHIFTEDBY_deltalin',
                                'deltalin_cube-mean_SHIFTEDBY_deltalin'],
                fixed_linear_sources=[], # must be [] if minimization_objective='(T*data-T*model)^2/(T*data)^2'
                field_to_smoothen_and_square=None,
                quadratic_sources=[],
                target_field='BEST_delta_h_M0M1_for_1deltalin12G23_SHIFTEDBY_PsiZ', 
                target_spec=TargetSpec(
                    linear_target_contris=['delta_h','delta_h_WEIGHT_M1'],
                    minimization_objective='(T*target-T*sources)^2/(T*target)^2',
                    target_norm={'type': 'alpha0=1'},
                    save_bestfit_target_field='BEST_delta_h_M0M1_for_1deltalin12G23_SHIFTEDBY_PsiZ' 
                       # where to save bestfit combi of targets
                    ),
                save_bestfit_field='hat_delta_h_M0M1_from_T_1deltalin12G23_SHIFTEDBY_PsiZ' # where to save bestfit combi of sources
                ))
            
        if False:
            # Same but without deltaZ
            opts['trf_specs'].append(TrfSpec(
                linear_sources=['deltalin_SHIFTEDBY_deltalin', 
                                'deltalin_growth-mean_SHIFTEDBY_deltalin', 
                                'deltalin_G2_SHIFTEDBY_deltalin',
                                'deltalin_cube-mean_SHIFTEDBY_deltalin'],
                fixed_linear_sources=[], # must be [] if minimization_objective='(T*data-T*model)^2/(T*data)^2'
                field_to_smoothen_and_square=None,
                quadratic_sources=[],
                target_field='BEST_delta_h_M0M1_for_deltalin12G23_SHIFTEDBY_PsiZ', 
                target_spec=TargetSpec(
                    linear_target_contris=['delta_h','delta_h_WEIGHT_M1'],
                    minimization_objective='(T*target-T*sources)^2/(T*target)^2',
                    target_norm={'type': 'alpha0=1'},
                    save_bestfit_target_field='BEST_delta_h_M0M1_for_deltalin12G23_SHIFTEDBY_PsiZ' 
                       # where to save bestfit combi of targets
                    ),
                save_bestfit_field='hat_delta_h_M0M1_from_T_deltalin12G23_SHIFTEDBY_PsiZ' # where to save bestfit combi of sources
                ))

        ## MASS-WEIGHTING to optimize (data-model)^2/data^2
        # First get alpha for optimal data combination assuming model has no deltaZ as fixed source.
        # Given that data combination, find optimal model coefficients.
        # This gives slightly sub-optimal alpha's b/c they assume no deltaZ, but shouldn't matter 
        # if model has enough terms to absorb deltaZ.
        # This is best mass-weighting scheme so far.
        # Issues: - Target is normalized by imposing alpha0=1 which gives bad high-k power.

        if False:
            # Use linear Eul. bias for model.
            opts['trf_specs'].append(TrfSpec(
                linear_sources=['delta_m'],
                field_to_smoothen_and_square=None,
                quadratic_sources=[],
                target_field='BEST_delta_h_M0M1_for_b1_delta_m', 
                target_spec=TargetSpec(
                    linear_target_contris=['delta_h','delta_h_WEIGHT_M1'],
                    minimization_objective='(T*target-T*sources)^2/(T*target)^2',
                    target_norm={'type': 'alpha0=1'},
                    save_bestfit_target_field='BEST_delta_h_M0M1_for_b1_delta_m' 
                       # where to save bestfit combi of targets
                    ),
                save_bestfit_field='hat_delta_h_M0M1_noZ_from_b1_delta_m' # where to save bestfit combi of sources
                ))
                   
        if False:
            # Use quadratic Eul. bias for model.
            opts['trf_specs'].append(TrfSpec(
                linear_sources=['delta_m'],
                field_to_smoothen_and_square='delta_m',
                quadratic_sources=['growth','tidal_G2'],
                target_field='BEST_delta_h_M0M1_for_b12G2_delta_m',
                target_spec=TargetSpec(
                    linear_target_contris=['delta_h','delta_h_WEIGHT_M1'],
                    minimization_objective='(T*target-T*sources)^2/(T*target)^2',
                    target_norm={'type': 'alpha0=1'},
                    save_bestfit_target_field='BEST_delta_h_M0M1_for_b12G2_delta_m' 
                       # where to save bestfit combi of targets
                    ),
                save_bestfit_field='hat_delta_h_M0M1_noZ_from_b12G2_delta_m' # where to save bestfit combi of sources
                ))
                   
        if False:
            # Use linear Lagr. bias for model.
            opts['trf_specs'].append(TrfSpec(
                linear_sources=['deltalin_SHIFTEDBY_deltalin'],
                fixed_linear_sources=['1_SHIFTEDBY_deltalin'], # if minimization_objective='(T*data-T*model)^2/(T*data)^2', using this gives slightly sub-optimal target weights
                field_to_smoothen_and_square=None,
                quadratic_sources=[],
                target_field='BEST_delta_h_M0M1_for_deltalin1_SHIFTEDBY_PsiZ', 
                target_spec=TargetSpec(
                    linear_target_contris=['delta_h','delta_h_WEIGHT_M1'],
                    minimization_objective='(T*target-T*sources)^2/(T*target)^2',
                    target_norm={'type': 'alpha0=1'},
                    save_bestfit_target_field='BEST_delta_h_M0M1_for_deltalin1_SHIFTEDBY_PsiZ' 
                       # where to save bestfit combi of targets
                    ),
                save_bestfit_field='hat_delta_h_M0M1_noZ_from_T_1deltalin1_SHIFTEDBY_PsiZ' # where to save bestfit combi of sources
                ))
            
        if False:
            # Same but with cubic Lagr. bias.
            opts['trf_specs'].append(TrfSpec(
                linear_sources=['deltalin_SHIFTEDBY_deltalin', 
                                'deltalin_growth-mean_SHIFTEDBY_deltalin', 
                                'deltalin_G2_SHIFTEDBY_deltalin',
                                'deltalin_cube-mean_SHIFTEDBY_deltalin'],
                fixed_linear_sources=['1_SHIFTEDBY_deltalin'], # if minimization_objective='(T*data-T*model)^2/(T*data)^2', using this gives slightly sub-optimal target weights
                field_to_smoothen_and_square=None,
                quadratic_sources=[],
                target_field='BEST_delta_h_M0M1_for_deltalin12G23_SHIFTEDBY_PsiZ', 
                target_spec=TargetSpec(
                    linear_target_contris=['delta_h','delta_h_WEIGHT_M1'],
                    minimization_objective='(T*target-T*sources)^2/(T*target)^2',
                    target_norm={'type': 'alpha0=1'},
                    save_bestfit_target_field='BEST_delta_h_M0M1_for_deltalin12G23_SHIFTEDBY_PsiZ' 
                       # where to save bestfit combi of targets
                    ),
                save_bestfit_field='hat_delta_h_M0M1_noZ_from_T_1deltalin12G23_SHIFTEDBY_PsiZ' # where to save bestfit combi of sources
                ))


        ## below, change normalization of target in mass weighting scheme
        if False:
            # cubic Lagr. bias, with target normalized such that P_target=P_zeldovich
            opts['trf_specs'].append(TrfSpec(
                linear_sources=['deltalin_SHIFTEDBY_deltalin', 
                                'deltalin_growth-mean_SHIFTEDBY_deltalin', 
                                'deltalin_G2_SHIFTEDBY_deltalin',
                                'deltalin_cube-mean_SHIFTEDBY_deltalin'],
                fixed_linear_sources=['1_SHIFTEDBY_deltalin'], # if minimization_objective='(T*data-T*model)^2/(T*data)^2', using this gives slightly sub-optimal target weights
                field_to_smoothen_and_square=None,
                quadratic_sources=[],
                # 'MPZZ' stands for match power to Zeldovich-Zeldovich power spectrum.
                target_field='BEST_delta_h_M0M1_MPZZ_for_deltalin12G23_SHIFTEDBY_PsiZ',
                target_spec=TargetSpec(
                    linear_target_contris=['delta_h','delta_h_WEIGHT_M1'],
                    minimization_objective='(T*target-T*sources)^2/(T*target)^2',
                    target_norm={'type': 'MatchPower', 
                                  'Pk_to_match_id1': '1_SHIFTEDBY_deltalin',
                                  'Pk_to_match_id2': '1_SHIFTEDBY_deltalin'},
                    save_bestfit_target_field='BEST_delta_h_M0M1_MPZZ_for_deltalin12G23_SHIFTEDBY_PsiZ' 
                       # where to save bestfit combi of targets
                    ),
                save_bestfit_field='hat_delta_h_M0M1_noZ_MPZZ_from_T_1deltalin12G23_SHIFTEDBY_PsiZ' # where to save bestfit combi of sources
                ))

            
        if False:
            # Same but change ordering of target fields, so that data=alpha0*delta_M+alpha1*delta_n^orth,
            # still matching d^2 to Zeldovich power spectrum.
            # Result: 1-r^2 looks better than before (maybe b/c orthogonalization of target contris at low k not good)
            opts['trf_specs'].append(TrfSpec(
                linear_sources=['deltalin_SHIFTEDBY_deltalin', 
                                'deltalin_growth-mean_SHIFTEDBY_deltalin', 
                                'deltalin_G2_SHIFTEDBY_deltalin',
                                'deltalin_cube-mean_SHIFTEDBY_deltalin'],
                fixed_linear_sources=['1_SHIFTEDBY_deltalin'], # if minimization_objective='(T*data-T*model)^2/(T*data)^2', using this gives slightly sub-optimal target weights
                field_to_smoothen_and_square=None,
                quadratic_sources=[],
                # 'MPZZ' stands for match power to Zeldovich-Zeldovich power spectrum.
                target_field='BEST_delta_h_M1M0_MPZZ_for_deltalin12G23_SHIFTEDBY_PsiZ',
                target_spec=TargetSpec(
                    linear_target_contris=['delta_h_WEIGHT_M1','delta_h'],
                    minimization_objective='(T*target-T*sources)^2/(T*target)^2',
                    target_norm={'type': 'MatchPower', 
                                  'Pk_to_match_id1': '1_SHIFTEDBY_deltalin',
                                  'Pk_to_match_id2': '1_SHIFTEDBY_deltalin'},
                    save_bestfit_target_field='BEST_delta_h_M1M0_MPZZ_for_deltalin12G23_SHIFTEDBY_PsiZ' 
                       # where to save bestfit combi of targets
                    ),
                save_bestfit_field='hat_delta_h_M1M0_noZ_MPZZ_from_1_T_deltalin12G23_SHIFTEDBY_PsiZ' # where to save bestfit combi of sources
                ))

        if False:
            # Same but don't include deltaZ in model 
            opts['trf_specs'].append(TrfSpec(
                linear_sources=['deltalin_SHIFTEDBY_deltalin', 
                                'deltalin_growth-mean_SHIFTEDBY_deltalin', 
                                'deltalin_G2_SHIFTEDBY_deltalin',
                                'deltalin_cube-mean_SHIFTEDBY_deltalin'],
                fixed_linear_sources=[], # if minimization_objective='(T*data-T*model)^2/(T*data)^2', using this gives slightly sub-optimal target weights
                field_to_smoothen_and_square=None,
                quadratic_sources=[],
                # 'MPZZ' stands for match power to Zeldovich-Zeldovich power spectrum.
                target_field='BEST_delta_h_M1M0_MPZZ_for_deltalin12G23_SHIFTEDBY_PsiZ',
                target_spec=TargetSpec(
                    linear_target_contris=['delta_h_WEIGHT_M1','delta_h'],
                    minimization_objective='(T*target-T*sources)^2/(T*target)^2',
                    target_norm={'type': 'MatchPower', 
                                  'Pk_to_match_id1': '1_SHIFTEDBY_deltalin',
                                  'Pk_to_match_id2': '1_SHIFTEDBY_deltalin'},
                    save_bestfit_target_field='BEST_delta_h_M1M0_MPZZ_for_deltalin12G23_SHIFTEDBY_PsiZ' 
                       # where to save bestfit combi of targets
                    ),
                save_bestfit_field='hat_delta_h_M1M0_noZ_MPZZ_from_T_deltalin12G23_SHIFTEDBY_PsiZ' # where to save bestfit combi of sources
                ))

        if False:
            # Same but include deltaZ in model again, and match d^2 to observed delta_h power.
            # Result: Looks very good. Problem: mass scatter not good.
            opts['trf_specs'].append(TrfSpec(
                linear_sources=['deltalin_SHIFTEDBY_deltalin', 
                                'deltalin_growth-mean_SHIFTEDBY_deltalin', 
                                'deltalin_G2_SHIFTEDBY_deltalin',
                                'deltalin_cube-mean_SHIFTEDBY_deltalin'],
                fixed_linear_sources=['1_SHIFTEDBY_deltalin'], 
                   # if minimization_objective='(T*data-T*model)^2/(T*data)^2', using this gives slightly sub-optimal target weights
                field_to_smoothen_and_square=None,
                quadratic_sources=[],
                # 'MPZZ' stands for match power to Zeldovich-Zeldovich power spectrum.
                target_field='BEST_delta_h_M1M0_MPnn_for_deltalin12G23_SHIFTEDBY_PsiZ',
                target_spec=TargetSpec(
                    linear_target_contris=['delta_h_WEIGHT_M1','delta_h'],
                    minimization_objective='(T*target-T*sources)^2/(T*target)^2',
                    target_norm={'type': 'MatchPower', 
                                  'Pk_to_match_id1': 'delta_h',
                                  'Pk_to_match_id2': 'delta_h'},
                    save_bestfit_target_field='BEST_delta_h_M1M0_MPnn_for_deltalin12G23_SHIFTEDBY_PsiZ' 
                       # where to save bestfit combi of targets
                    ),
                save_bestfit_field='hat_delta_h_M1M0_noZ_MPnn_from_1_T_deltalin12G23_SHIFTEDBY_PsiZ' # where to save bestfit combi of sources
                ))


        if False:
            # Same but include mass scatter
            # Result: on 64^3, 0.6dex is worse than not including any mass weighting. May be related to inaccurate
            # orthogonalization of data vector contributions. Improve orthogonalization in v0.33, but 0.6dex mass
            # scaetter still gives worse 1-r^2 than using no mass-weighted field at all.
            #for myscatter in ['0.04dex','0.1dex','0.3dex','0.6dex']:
            for myscatter in ['0.4dex','0.2dex','0.1dex']:
                opts['trf_specs'].append(TrfSpec(
                    linear_sources=['deltalin_SHIFTEDBY_deltalin', 
                                    'deltalin_growth-mean_SHIFTEDBY_deltalin', 
                                    'deltalin_G2_SHIFTEDBY_deltalin',
                                    'deltalin_cube-mean_SHIFTEDBY_deltalin'],
                    fixed_linear_sources=['1_SHIFTEDBY_deltalin'], 
                       # if minimization_objective='(T*data-T*model)^2/(T*data)^2', using this gives slightly sub-optimal target weights
                    field_to_smoothen_and_square=None,
                    quadratic_sources=[],
                    # 'MPZZ' stands for match power to Zeldovich-Zeldovich power spectrum.
                    target_field='BEST_delta_h_M%sM0_MPnn_for_deltalin12G23_SHIFTEDBY_PsiZ'%myscatter,
                    target_spec=TargetSpec(
                        linear_target_contris=['delta_h_WEIGHT_M%s'%myscatter,'delta_h'],
                        minimization_objective='(T*target-T*sources)^2/(T*target)^2',
                        target_norm={'type': 'MatchPower', 
                                      'Pk_to_match_id1': 'delta_h',
                                      'Pk_to_match_id2': 'delta_h'},
                        save_bestfit_target_field='BEST_delta_h_M%sM0_MPnn_for_deltalin12G23_SHIFTEDBY_PsiZ'%myscatter
                           # where to save bestfit combi of targets
                        ),
                    save_bestfit_field='hat_delta_h_M%sM0_noZ_MPnn_from_1_T_deltalin12G23_SHIFTEDBY_PsiZ'%myscatter # where to save bestfit combi of sources
                    ))

        if False:
            # Same but change ordering of data contris: d=delta_n+delta_M^orth
            # Result: 1-r^2 bad at low k. Maybe related to orthogonalization.
            #for myscatter in ['0.04dex','0.1dex','0.3dex','0.6dex']:
            for myscatter in ['0.4dex','0.2dex','0.1dex']:
                opts['trf_specs'].append(TrfSpec(
                    linear_sources=['deltalin_SHIFTEDBY_deltalin',
                                    'deltalin_growth-mean_SHIFTEDBY_deltalin', 
                                    'deltalin_G2_SHIFTEDBY_deltalin',
                                    'deltalin_cube-mean_SHIFTEDBY_deltalin'],
                    fixed_linear_sources=['1_SHIFTEDBY_deltalin'], 
                       # if minimization_objective='(T*data-T*model)^2/(T*data)^2', using this gives slightly sub-optimal target weights
                    field_to_smoothen_and_square=None,
                    quadratic_sources=[],
                    # 'MPZZ' stands for match power to Zeldovich-Zeldovich power spectrum.
                    target_field='BEST_delta_h_M0M%s_MPnn_for_deltalin12G23_SHIFTEDBY_PsiZ'%myscatter,
                    target_spec=TargetSpec(
                        linear_target_contris=['delta_h','delta_h_WEIGHT_M%s'%myscatter],
                        minimization_objective='(T*target-T*sources)^2/(T*target)^2',
                        target_norm={'type': 'MatchPower', 
                                      'Pk_to_match_id1': 'delta_h',
                                      'Pk_to_match_id2': 'delta_h'},
                        save_bestfit_target_field='BEST_delta_h_M0M%s_MPnn_for_deltalin12G23_SHIFTEDBY_PsiZ'%myscatter
                           # where to save bestfit combi of targets
                        ),
                    save_bestfit_field='hat_delta_h_M0M%s_noZ_MPnn_from_1_T_deltalin12G23_SHIFTEDBY_PsiZ'%myscatter # where to save bestfit combi of sources
                    ))


        if False:
            # Same d=delta_n+delta_M^orth, but without mass scatter, and without deltaZ in model
            # Result: 
            opts['trf_specs'].append(TrfSpec(
                linear_sources=['deltalin_SHIFTEDBY_deltalin', 
                                'deltalin_growth-mean_SHIFTEDBY_deltalin', 
                                'deltalin_G2_SHIFTEDBY_deltalin',
                                'deltalin_cube-mean_SHIFTEDBY_deltalin'],
                fixed_linear_sources=[], 
                   # if minimization_objective='(T*data-T*model)^2/(T*data)^2', using this gives slightly sub-optimal target weights
                field_to_smoothen_and_square=None,
                quadratic_sources=[],
                # 'MPZZ' stands for match power to Zeldovich-Zeldovich power spectrum.
                target_field='BEST_delta_h_M0M1_MPnn_for_deltalin12G23_SHIFTEDBY_PsiZ',
                target_spec=TargetSpec(
                    linear_target_contris=['delta_h','delta_h_WEIGHT_M1'],
                    minimization_objective='(T*target-T*sources)^2/(T*target)^2',
                    target_norm={'type': 'MatchPower', 
                                  'Pk_to_match_id1': 'delta_h',
                                  'Pk_to_match_id2': 'delta_h'},
                    save_bestfit_target_field='BEST_delta_h_M0M1_MPnn_for_deltalin12G23_SHIFTEDBY_PsiZ' 
                       # where to save bestfit combi of targets
                    ),
                save_bestfit_field='hat_delta_h_M0M1_noZ_MPnn_from_T_deltalin12G23_SHIFTEDBY_PsiZ' # where to save bestfit combi of sources
                ))

        if False:
            # Same but d=delta_M+delta_n^orth
            # Result: 
            opts['trf_specs'].append(TrfSpec(
                linear_sources=['deltalin_SHIFTEDBY_deltalin', 
                                'deltalin_growth-mean_SHIFTEDBY_deltalin', 
                                'deltalin_G2_SHIFTEDBY_deltalin',
                                'deltalin_cube-mean_SHIFTEDBY_deltalin'],
                fixed_linear_sources=[], 
                   # if minimization_objective='(T*data-T*model)^2/(T*data)^2', using this gives slightly sub-optimal target weights
                field_to_smoothen_and_square=None,
                quadratic_sources=[],
                # 'MPZZ' stands for match power to Zeldovich-Zeldovich power spectrum.
                target_field='BEST_delta_h_M1M0_MPnn_for_deltalin12G23_SHIFTEDBY_PsiZ',
                target_spec=TargetSpec(
                    linear_target_contris=['delta_h_WEIGHT_M1','delta_h'],
                    minimization_objective='(T*target-T*sources)^2/(T*target)^2',
                    target_norm={'type': 'MatchPower', 
                                  'Pk_to_match_id1': 'delta_h',
                                  'Pk_to_match_id2': 'delta_h'},
                    save_bestfit_target_field='BEST_delta_h_M1M0_MPnn_for_deltalin12G23_SHIFTEDBY_PsiZ' 
                       # where to save bestfit combi of targets
                    ),
                save_bestfit_field='hat_delta_h_M1M0_noZ_MPnn_from_T_deltalin12G23_SHIFTEDBY_PsiZ' # where to save bestfit combi of sources
                ))



            
    
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
    opts['pickle_path'] = '$SCRATCH/lssbisp2013/psiRec/pickle/'
    opts['cache_base_path'] = '$SCRATCH/lssbisp2013/psiRec/cache/'
    opts['grids4plots_base_path'] = '$SCRATCH/lssbisp2013/psiRec/grids4plots/'
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
    opts['keep_pickle'] = True
    opts['pickle_file_format'] = 'pickle'
    # plot using plotting code for single realization; difft from code plotting avg 
    do_plot = False
    # save grids for slice plots and scatter plots
    opts['save_grids4plots'] = False
    

    ## COSMOLOGY OPTIONS
    if opts['sim_name'] in ['jerryou_baoshift','ms_gadget']:
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
        
    # init mpi    
    if opts['use_mpi']:
        from mpi4py import MPI

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
        pickler = Pickler.Pickler(path=paths['pickle_path'], base_fname='main_quick_calc_Pk',
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


