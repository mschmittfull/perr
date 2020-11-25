from __future__ import print_function, division
from argparse import ArgumentParser
from collections import OrderedDict
import numpy as np

from lsstools import parameters
from lsstools import parameters_ms_gadget
from lsstools.model_spec import *
from perr import model_error


def main():
    """
    Calculate Perror for velocity divergence in real space.
    """

    #####################################
    # PARSE COMMAND LINE ARGS
    #####################################
    ap = ArgumentParser()

    ap.add_argument('--SimSeed',
                    type=int,
                    default=400,
                    help='Simulation seed to load.')

    ap.add_argument('--HaloMassString',
                    default='13.8_15.1',
                    help="Halo mass string, for example '13.8_15.1'.")

    ap.add_argument('--ShiftedFieldsNp',
                    default=1536,
                    type=int,
                    help='Number of particles used to shift fields.')

    ap.add_argument('--ShiftedFieldsNmesh',
                    default=1536,
                    type=int,
                    help='Nmesh used to shift fields.')

    ap.add_argument('--Ngrid',
                    default=512,
                    type=int,
                    help='Number of grid points used to compute power spectra.')

    cmd_args = ap.parse_args()

    #####################################
    # OPTIONS
    #####################################
    opts = OrderedDict()

    # Bump this when changing code without changing options. Otherwise pickle
    # loading might wrongly read old pickles.
    # 17 Apr 2019: v0.1: Start implementation for RSD Perr.
    # 23 Apr 2019: v0.2: Get reasonable power spectra, orthogonalization looks good.
    # 0.3: Run yapf to format code.
    # 15 May 2019: v0.4: Change options organization.
    # 20 May 2019: v0.5: Refactored code, bump version.
    # 25 Feb 2020: v0.6: Add cubic operators.
    # 19 Mar 2020: v0.7: Add shifted velocity.
    opts['code_version_for_pickles'] = '0.7'

    # Simulation options. Will be used by path_utils to get input path, and
    # to compute deltalin at the right redshift.
    seed = cmd_args.SimSeed
    opts['sim_opts'] = parameters_ms_gadget.MSGadgetSimOpts.load_default_opts(
        sim_name='ms_gadget',
        sim_seed=seed,
        halo_mass_string=cmd_args.HaloMassString)

    # Grid options.
    Ngrid = cmd_args.Ngrid
    opts['shifted_fields_Np'] = cmd_args.ShiftedFieldsNp
    opts['shifted_fields_Nmesh'] = cmd_args.ShiftedFieldsNmesh

    opts['grid_opts'] = parameters.GridOpts(
        Ngrid=Ngrid,
        kmax=2.0*np.pi/opts['sim_opts'].boxsize * float(Ngrid)/2.0,
        grid_ptcle2grid_deconvolution=None
        )

    # Include RSD or not
    include_RSD = False

    # Options for measuring power spectrum. 
    if include_RSD:
        opts['power_opts'] = parameters.PowerOpts(
            k_bin_width=1.0,
            Pk_1d_2d_mode='2d',
            RSD_poles=[],
            RSD_Nmu=5,
            RSD_los=[0, 0, 1]
            )
    else:
        opts['power_opts'] = parameters.PowerOpts(
            k_bin_width=1.0,
            Pk_1d_2d_mode='1d'
            )

    # Transfer function options. See lsstools.parameters.py for details.
    if include_RSD:
        opts['trf_fcn_opts'] = parameters.TrfFcnOpts(
            Rsmooth_for_quadratic_sources=0.1,
            Rsmooth_for_quadratic_sources2=0.1,
            N_ortho_iter=0,
            orth_method='CholeskyDecomp',
            interp_kind='manual_Pk_k_mu_bins' # 2d binning in k,mu
            )
    else:
        opts['trf_fcn_opts'] = parameters.TrfFcnOpts(
            Rsmooth_for_quadratic_sources=0.1,
            Rsmooth_for_quadratic_sources2=0.1,
            N_ortho_iter=0,
            orth_method='CholeskyDecomp',
            interp_kind='manual_Pk_k_bins'
            )


    # External grids to load: deltalin, delta_m, shifted grids
    if include_RSD:
        RSDstring = '_RSD%d%d%d' % (opts['power_opts'].RSD_los[0],
                                    opts['power_opts'].RSD_los[1],
                                    opts['power_opts'].RSD_los[2]
                                   )
        opts['RSDstrings'] = [('',''), (RSDstring,RSDstring)]
    else:
        opts['RSDstrings'] = [('','')]

    opts['ext_grids_to_load'] = opts['sim_opts'].get_default_ext_grids_to_load(
        Ngrid=opts['grid_opts'].Ngrid,
        RSDstrings=opts['RSDstrings'],
        shifted_fields_Np=opts['shifted_fields_Np'],
        shifted_fields_Nmesh=opts['shifted_fields_Nmesh'],
        include_shifted_fields=True,
        include_2LPT_shifted_fields=False,
        include_3LPT_shifted_fields=False,
        include_minus_3LPT_shifted_fields=False,
        include_div_shifted_PsiDot1=True,
        include_div_shifted_PsiDot2=False,
        )

    # Catalogs to read
    opts['cats'] = opts['sim_opts'].get_default_catalogs(
        RSDstrings=opts['RSDstrings'])

    # Specify bias expansions to test
    opts['trf_specs'] = []


    # ######################################################################
    # Specify sources and targets for bias expansions and transfer functions
    # ######################################################################

    # Allowed quadratic_sources: 'growth','tidal_s2', 'tidal_G2'
    # Allowed cubic_sources: 'cube'
    opts['trf_specs'] = []

    ### HALO NUMBER DENSITY FROM DM  (see main_calc_Perr.py for old bias models
    # and doing mass weighting, getting DM from DM, DM from halos etc).
    for target_RSDstring, RSDstring in opts['RSDstrings']:
        for psi_type_str, psi_name in [
            ('', 'PsiZ'),
            #('Psi2LPT_', 'Psi2LPT'),
            #('Psi-3LPT_', 'Psi-3LPT'),
            #('Psi3LPT_', 'Psi3LPT')
            ]:

            # halo divergence
            for target in ['thetav_h']:
                target += target_RSDstring

                if True:
                    # b1 deltalin
                    opts['trf_specs'].append(
                        TrfSpec(linear_sources=['deltalin'],
                                field_to_smoothen_and_square=None,
                                quadratic_sources=[],
                                target_field=target,
                                save_bestfit_field='hat_%s_from_b1_deltalin' % target))

                if True:
                    # b1 deltaZA = b1 1_shifted. Includes RSD in displacement.
                    opts['trf_specs'].append(
                        TrfSpec(linear_sources=['1_SHIFTEDBY_%sdeltalin%s' % (
                            psi_type_str, RSDstring)],
                                field_to_smoothen_and_square=None,
                                quadratic_sources=[],
                                target_field=target,
                                save_bestfit_field='hat_%s_from_T1_SHIFTEDBY_%s%s' % (
                                target, psi_name, RSDstring)))

                if True:
                    # b1 delta_m (linear Eulerian bias)
                    opts['trf_specs'].append(
                        TrfSpec(linear_sources=['delta_m%s' % RSDstring],
                                field_to_smoothen_and_square=None,
                                quadratic_sources=[],
                                target_field=target,
                                save_bestfit_field='hat_%s_from_b1_delta_m%s' % (
                                    target, RSDstring)))


                if True:
                    # b1 deltalin(q+Psi) + bG2 [G2](q+Psi)
                    opts['trf_specs'].append(
                        TrfSpec(
                            linear_sources=[
                                'deltalin_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring),
                                'deltalin_growth-mean_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring),
                                'deltalin_G2_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring),
                                'deltalin_cube-mean_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring)
                            ],
                            fixed_linear_sources=[],
                            field_to_smoothen_and_square=None,
                            quadratic_sources=[],
                            target_field=target,
                            save_bestfit_field=
                            'hat_%s_from_TdeltalinG2_SHIFTEDBY_%s%s' % (
                                target, psi_name, RSDstring)))

                if True:
                    # deltaZ + b1 deltalin(q+Psi) + bG2 [G2](q+Psi)
                    opts['trf_specs'].append(
                        TrfSpec(
                            linear_sources=[
                                'deltalin_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring),
                                'deltalin_growth-mean_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring),
                                'deltalin_G2_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring),
                                'deltalin_cube-mean_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring)
                            ],
                            fixed_linear_sources=['1_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring)],
                            field_to_smoothen_and_square=None,
                            quadratic_sources=[],
                            target_field=target,
                            save_bestfit_field=
                            'hat_%s_from_1_TdeltalinG2_SHIFTEDBY_%s%s' % (
                                target, psi_name, RSDstring)))

                if True:
                    # Cubic Lagrangian bias with delta^3: delta_Z + b1 deltalin(q+Psi) + b2 [deltalin^2-<deltalin^2>](q+Psi) + bG2 [G2](q+Psi)
                    # + b3 [delta^3](q+Psi)
                    # With RSD, cannot orthogonalize deltalin^3 and deltalin at low k b/c too correlated.
                    opts['trf_specs'].append(
                        TrfSpec(
                            linear_sources=[
                                'deltalin_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring),
                                'deltalin_growth-mean_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring),
                                'deltalin_G2_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring),
                                'deltalin_cube-mean_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring)
                            ],
                            fixed_linear_sources=['1_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring)],
                            field_to_smoothen_and_square=None,
                            quadratic_sources=[],
                            target_field=target,
                            save_bestfit_field=
                            'hat_%s_from_1_Tdeltalin2G23_SHIFTEDBY_%s%s' % (
                                target, psi_name, RSDstring)))

                if True:
                    # Model velocity divergence using divergence of shifted 
                    # PsiDot_1storder. Allow also a bias prefactor.
                    opts['trf_specs'].append(
                        TrfSpec(linear_sources=['div_PsiDot1_SHIFTEDBY_%sdeltalin%s' % (
                            psi_type_str, RSDstring)],
                                field_to_smoothen_and_square=None,
                                quadratic_sources=[],
                                target_field=target,
                                save_bestfit_field='hat_%s_from_TPsiDot1_SHIFTEDBY_%s%s' % (
                                target, psi_name, RSDstring)))

                if False:
                    # Model velocity divergence using divergence of shifted 
                    # PsiDot_2ndorder. Allow also a bias prefactor.
                    opts['trf_specs'].append(
                        TrfSpec(linear_sources=['div_PsiDot2_SHIFTEDBY_%sdeltalin%s' % (
                            psi_type_str, RSDstring)],
                                field_to_smoothen_and_square=None,
                                quadratic_sources=[],
                                target_field=target,
                                save_bestfit_field='hat_%s_from_TPsiDot2_SHIFTEDBY_%s%s' % (
                                target, psi_name, RSDstring)))


            # halo overdensity
            #target = 'delta_h%s' % RSDstring
            # HOD galaxies
            #for target in ['delta_g', 'delta_gc', 'delta_gp', 'delta_gs']:
            #for target in ['delta_h']:
            #for target in ['delta_h']:
            for target in []:
                target += target_RSDstring


                


                if False:
                    #if False:
                    # Cubic Lagrangian bias without deltaZ: b1 deltalin(q+Psi) + b2 [deltalin^2-<deltalin^2>](q+Psi) + bG2 [G2](q+Psi)
                    # + b3 [delta^3](q+Psi)
                    opts['trf_specs'].append(
                        TrfSpec(
                            linear_sources=[
                                'deltalin_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring),
                                'deltalin_growth-mean_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring),
                                'deltalin_G2_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring),
                                'deltalin_cube-mean_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring)
                            ],
                            fixed_linear_sources=[],
                            field_to_smoothen_and_square=None,
                            quadratic_sources=[],
                            target_field=target,
                            save_bestfit_field='hat_%s_from_Tdeltalin2G23_SHIFTEDBY_%s%s'
                            % (target, psi_name, RSDstring)))

                if False:
                    # Cubic Lagrangian bias with delta^3 and *PsiNablaDelta* shift term: delta_Z + b1 deltalin(q+Psi)
                    # + b2 [deltalin^2-<deltalin^2>](q+Psi) + bG2 [G2](q+Psi) + b3 [delta^3](q+Psi) + [Psi.Nabla Delta](q+Psi)
                    # BEST MODEL WITHOUT MASS WEIGHTING
                    opts['trf_specs'].append(
                        TrfSpec(
                            linear_sources=[
                                'deltalin_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring),
                                'deltalin_growth-mean_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring),
                                'deltalin_G2_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring),
                                'deltalin_cube-mean_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring),
                                'PsiNablaDelta_SHIFTEDBY_d%seltalin%s' % (psi_type_str, RSDstring)
                            ],
                            fixed_linear_sources=['1_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring)],
                            field_to_smoothen_and_square=None,
                            quadratic_sources=[],
                            target_field=target,
                            save_bestfit_field=
                            'hat_%s_from_1_Tdeltalin2G23S_SHIFTEDBY_%s%s' % (
                               target, psi_name, RSDstring)))


                if False:
                    #if True:
                    # Cubic Lagrangian bias with delta^3, G2*delta, G3 and Gamma3: 
                    # delta_Z + b1 deltalin(q+Psi) + b2 [deltalin^2-<deltalin^2>](q+Psi) + bG2 [G2](q+Psi)
                    # + b3 [delta^3](q+Psi) + ...
                    # With RSD, cannot orthogonalize deltalin^3 and deltalin at low k b/c too correlated?
                    opts['trf_specs'].append(
                        TrfSpec(
                            linear_sources=[
                                'deltalin_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring),
                                'deltalin_growth-mean_SHIFTEDBY_d%seltalin%s' % (psi_type_str, RSDstring),
                                'deltalin_G2_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring),
                                'deltalin_cube-mean_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring),
                                'deltalin_G2_delta_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring),
                                'deltalin_G3_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring),
                                'deltalin_Gamma3_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring)
                            ],
                            fixed_linear_sources=['1_SHIFTEDBY_%sdeltalin%s' % (psi_type_str, RSDstring)],
                            field_to_smoothen_and_square=None,
                            quadratic_sources=[],
                            target_field=target,
                            save_bestfit_field=
                            'hat_%s_from_1_Tdeltalin2G23dG2G3Gamma3_SHIFTEDBY_%s%s' % (
                                target, psi_name, RSDstring)))

    # Save results
    opts['keep_pickle'] = True
    opts['pickle_file_format'] = 'dill'
    opts['pickle_path'] = '$SCRATCH/perr/pickle/'

    # Save some additional power spectra that are useful for plotting later
    opts['Pkmeas_helper_columns'] = [
        #'delta_h',
        #'delta_g', 'delta_gc', 'delta_gp',
        'deltalin',
        #'delta_m', '1_SHIFTEDBY_deltalin', 'deltalin'
        # 'deltalin_SHIFTEDBY_deltalin',
        # 'deltalin_G2_SHIFTEDBY_deltalin%s' % RSDstring,
        # 'deltalin_cube-mean_SHIFTEDBY_deltalin%s' % RSDstring,
        # 'deltalin_G2_delta_SHIFTEDBY_deltalin%s' % RSDstring,
        # 'deltalin_G3_SHIFTEDBY_deltalin%s' % RSDstring,
        # 'deltalin_Gamma3_SHIFTEDBY_deltalin%s' % RSDstring
    ]

    opts['Pkmeas_helper_columns_calc_crosses'] = False

    # Save grids for 2d slice plots and histograms
    opts['save_grids4plots'] = False
    opts['grids4plots_base_path'] = '$SCRATCH/perr/grids4plots/'
    opts['grids4plots_R'] = 0.0  # Gaussian smoothing applied to grids4plots

    # Cache path
    opts['cache_base_path'] = '$SCRATCH/perr/cache/'


    # what to return
    opts['return_fields'] = ['residual']

    # Run the program given the above opts, save results in pickle file.
    residual_fields, outdict = model_error.calculate_model_error(**opts)


if __name__ == '__main__':
    main()
