from __future__ import print_function, division

from collections import OrderedDict
import cPickle
import glob
from matplotlib import rc
from matplotlib import rcParams
from matplotlib import spines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, MaxNLocator
import numpy as np
import os
import random
import re
from scipy import interpolate as interp
from scipy.io import savemat
import sys
import ujson

import constants
import logbook_main_calc_Perr_pickles
from lsstools.pickle_utils.io import Pickler
from lsstools.pickle_utils import retrieve_pickles
from perr.path_utils import get_in_path


def main():
    """
    Plot power halo density and model for different mass bins.
    Average over realizations.

    Use new pickles produced by main_calc_Perr.py version>=0.2.
    """

    # save plots as pdf or show them
    save_plots = True

    # path of pickles
    pickle_path = os.path.expandvars('$SCRATCH/perr/pickle/')

    ## Which pickle to load. Edit logbook....py to select a pickle.
    loginfo = logbook_main_calc_Perr_pickles.get_pickle_fname()
    fname = loginfo['fname']
    plot_opts = loginfo['plot_opts']

    #### MORE OPTIONS

    sim_seeds = [400, 403]
    #sim_seeds = range(400,405)

    # switch to show title or not
    plot_title = True

    # show legend or annotate curves
    show_legend = False
    show_annotation = False

    # mass-weighting: change ylim and labels
    mass_weighted = False

    # include label of data
    include_data_label = False

    # plot results of fit_trf_fcns internally
    plot_fit_trf_fcns = False

    # fitting of trf fcns
    trf_fcn_fitspecs = []
    if False:
        trf_fcn_fitspecs.append({
            'name': 'const + const k^2 low kmax',
            'type': 'constrained_polyfit',
            'settings': {
                'kmin': 0.01,
                'kmax': 0.2,
                'exponents': [0, 2, 4]
            }
        })
    if False:
        trf_fcn_fitspecs.append({
            'name': 'const + const k^2 + const k^4 high kmax',
            'type': 'constrained_polyfit',
            'settings': {
                'kmin': 0.01,
                'kmax': 0.8,
                'exponents': [0, 2, 4]
            }
        })

    if False:
        # early jun 2018: marko/val model with no theoretical errors (good at high k but not great at low k); model includes deltaZ
        trf_fcn_fitspecs.append({
            'name': 'ValMarkoModel2',
            'type': 'fixed_fcn',
            'settings': {
                'tf_fname': 'from_marko_and_val/fit_b1_16may2018/TF.txt',
                'kmin': 0.01,
                'kmax': 2.0,
                'color': 'grey',
                'bestfit_params': {
                    ('10.8', '11.8'): {
                        'b1': -0.120745,
                        'cs': 0.0697826,
                        'bGamma3': 0.282946,
                        'b2': -0.183017,
                        'bG2': -0.211043,
                        'b3': 0.00233065
                    },
                    ('11.8', '12.8'): {
                        'b1': 0.0513072,
                        'cs': 0.277575,
                        'bGamma3': 0.236465,
                        'b2': -0.323,
                        'bG2': -0.183022,
                        'b3': -0.0844928
                    },
                    ('12.8', '13.8'): {
                        'b1': 0.677678,
                        'cs': 1.36496,
                        'bGamma3': -0.0132997,
                        'b2': -0.215479,
                        'bG2': -0.0301784,
                        'b3': -0.386499
                    },
                    ('13.8', '15.1'): {
                        'b1': 2.49556,
                        'cs': 3.45788,
                        'bGamma3': 1.0264,
                        'b2': 1.89056,
                        'bG2': -1.20618,
                        'b3': 0.649821
                    }
                }
            }
        })

    if False:
        # early jun 2018: marko/val model with theoretical errors (morally correct); model includes deltaZ
        trf_fcn_fitspecs.append({
            'name': 'ValMarkoModel1',
            'type': 'fixed_fcn',
            'settings': {
                'tf_fname': 'from_marko_and_val/fit_b1_16may2018/TF.txt',
                'kmin': 0.01,
                'kmax': 2.0,
                'color': 'k',
                'bestfit_params': {
                    ('10.8', '11.8'): {
                        'b1': -0.120705,
                        'cs': -0.0378647,
                        'bGamma3': 0.274557,
                        'b2': -0.18441,
                        'bG2': -0.209507,
                        'b3': 0.00233065
                    },
                    ('11.8', '12.8'): {
                        'b1': 0.0540231,
                        'cs': 0.378935,
                        'bGamma3': 0.243066,
                        'b2': -0.328223,
                        'bG2': -0.157525,
                        'b3': -0.0844928
                    },
                    ('12.8', '13.8'): {
                        'b1': 0.705413,
                        'cs': 3.79773,
                        'bGamma3': 0.288405,
                        'b2': -0.250961,
                        'bG2': 0.0964298,
                        'b3': -0.386499
                    },
                    ('13.8', '15.1'): {
                        'b1': 2.45389,
                        'cs': 0.429736,
                        'bGamma3': 0.10803,
                        'b2': 1.82867,
                        'bG2': -1.02188,
                        'b3': 0.649821
                    }
                }
            }
        })

    if False:
        # Marko/val trf fcns from 26/6/2018, **with theoretical errors**.
        # These are the parameters with theoretical errors. t_1 has two-loop theoretical error and t_2, t_G2
        # and t_3 have one-loop theoretical error. kmax for t_1 is 0.5 and kmax for the rest is 0.2.
        if False:
            # Old transfer functions with \delta_Z fitted to tryL32
            add_yoffset_to_bias_plots = True
            trf_fcn_fitspecs.append({
                'name': 'ValMarkoModel1',
                'type': 'fixed_fcn',
                'settings': {
                    'tf_fname': 'from_marko_and_val/fit_b1_16may2018/TF.txt',
                    'kmin': 0.01,
                    'kmax': 2.0,
                    'color': 'k',
                    'bestfit_params': {
                        ('10.8', '11.8'):
                        dict(b1=-0.121,
                             cs=-0.165,
                             bGamma3=0.263,
                             b2=-0.181,
                             bG2=-0.212,
                             b3=0.0023),
                        ('11.8', '12.8'):
                        dict(b1=0.054,
                             cs=0.482,
                             bGamma3=0.255,
                             b2=-0.326,
                             bG2=-0.162,
                             b3=-0.0845),
                        ('12.8', '13.8'):
                        dict(b1=0.701,
                             cs=3.491,
                             bGamma3=0.230,
                             b2=-0.237,
                             bG2=0.069,
                             b3=-0.386),
                        ('13.8', '15.1'):
                        dict(b1=2.456,
                             cs=-0.341,
                             bGamma3=0.0546,
                             b2=1.841,
                             bG2=-0.961,
                             b3=0.650)
                    }
                }
            })
        if False:
            # New transfer functions without \delta_Z fitted to tryL32cNoZ
            add_yoffset_to_bias_plots = True
            trf_fcn_fitspecs.append({
                'name': 'ValMarkoModel1',
                'type': 'fixed_fcn',
                'settings': {
                    'tf_fname': 'from_marko_and_val/fit_b1_16may2018/TF.txt',
                    'kmin': 0.01,
                    'kmax': 2.0,
                    'color': 'k',
                    'bestfit_params': {
                        ('10.8', '11.8'):
                        dict(b1=0.882,
                             cs=0.0115,
                             bGamma3=0.595,
                             b2=-0.226,
                             bG2=0.291,
                             b3=0.0029),
                        ('11.8', '12.8'):
                        dict(b1=1.057,
                             cs=0.648,
                             bGamma3=0.571,
                             b2=-0.384,
                             bG2=0.340,
                             b3=-0.084),
                        ('12.8', '13.8'):
                        dict(b1=1.703,
                             cs=3.678,
                             bGamma3=0.532,
                             b2=-0.307,
                             bG2=0.567,
                             b3=-0.388),
                        ('13.8', '15.1'):
                        dict(b1=3.457,
                             cs=-0.143,
                             bGamma3=0.321,
                             b2=1.758,
                             bG2=-0.455,
                             b3=0.656),
                    }
                }
            })

        if False:
            # Fit of the mass-weighted fields fitted to tryO13
            add_yoffset_to_bias_plots = False
            trf_fcn_fitspecs.append({
                'name': 'ValMarkoModel1',
                'type': 'fixed_fcn',
                'settings': {
                    'tf_fname': 'from_marko_and_val/fit_b1_16may2018/TF.txt',
                    'kmin': 0.01,
                    'kmax': 2.0,
                    'color': 'k',
                    'bestfit_params': {
                        ('10.8', '16.0'):
                        dict(b1=0.924,
                             cs=-1.354,
                             bGamma3=0.619,
                             b2=-0.00616,
                             bG2=0.139,
                             b3=-0.00164),
                        ('11.8', '16.0'):
                        dict(b1=1.149,
                             cs=-1.347,
                             bGamma3=0.618,
                             b2=0.062,
                             bG2=0.124,
                             b3=-0.00736),
                        ('12.8', '16.0'):
                        dict(b1=1.817,
                             cs=0.981,
                             bGamma3=0.978,
                             b2=0.411,
                             bG2=0.0392,
                             b3=0.0473),
                        ('13.8', '16.0'):
                        dict(b1=3.485,
                             cs=-1.495,
                             bGamma3=1.307,
                             b2=2.322,
                             bG2=-1.427,
                             b3=1.329),
                    }
                }
            })

    if False:
        # Marko/val trf fcns from 26/6/2018, **with no theoretical errors**.
        # These are the parameters without theoretical errors. This means that t_1 has no theoretical error and
        # t_2, t_G2 and t_3 have one-loop theoretical error. kmax for t_1 is 0.8 and kmax for the rest is 0.2.
        if False:
            # Old transfer functions with \delta_Z fitted to tryL32
            add_yoffset_to_bias_plots = True
            trf_fcn_fitspecs.append({
                'name': 'ValMarkoModelNoThErr',
                'type': 'fixed_fcn',
                'settings': {
                    'tf_fname': 'from_marko_and_val/fit_b1_16may2018/TF.txt',
                    'kmin': 0.01,
                    'kmax': 2.0,
                    'color': 'grey',
                    'bestfit_params': {
                        ('10.8', '11.8'):
                        dict(b1=-0.12,
                             cs=0.0587,
                             bGamma3=0.302,
                             b2=-0.178,
                             bG2=-0.221,
                             b3=0.0023),
                        ('11.8', '12.8'):
                        dict(b1=0.0508,
                             cs=0.23,
                             bGamma3=0.26,
                             b2=-0.314,
                             bG2=-0.205,
                             b3=-0.0845),
                        ('12.8', '13.8'):
                        dict(b1=0.674,
                             cs=1.205,
                             bGamma3=0.061,
                             b2=-0.19,
                             bG2=-0.112,
                             b3=-0.386),
                        ('13.8', '15.1'):
                        dict(b1=2.472,
                             cs=3.017,
                             bGamma3=1.165,
                             b2=1.982,
                             bG2=-1.407,
                             b3=0.650),
                    }
                }
            })
        if True:
            # New transfer functions without \delta_Z fitted to tryL32cNoZ
            add_yoffset_to_bias_plots = True
            trf_fcn_fitspecs.append({
                'name': 'ValMarkoModelNoThErr',
                'type': 'fixed_fcn',
                'settings': {
                    'tf_fname': 'from_marko_and_val/fit_b1_16may2018/TF.txt',
                    'kmin': 0.01,
                    'kmax': 2.0,
                    'color': 'grey',
                    'bestfit_params': {
                        ('10.8', '11.8'):
                        dict(b1=0.878,
                             cs=0.0778,
                             bGamma3=0.59,
                             b2=-0.222,
                             bG2=0.274,
                             b3=0.0029),
                        ('11.8', '12.8'):
                        dict(b1=1.05,
                             cs=0.249,
                             bGamma3=0.568,
                             b2=-0.366,
                             bG2=0.269,
                             b3=-0.084),
                        ('12.8', '13.8'):
                        dict(b1=1.672,
                             cs=1.223,
                             bGamma3=0.345,
                             b2=-0.254,
                             bG2=0.36,
                             b3=-0.388),
                        ('13.8', '15.1'):
                        dict(b1=3.468,
                             cs=3.035,
                             bGamma3=1.401,
                             b2=1.905,
                             bG2=-0.922,
                             b3=0.656),
                    }
                }
            })

        if False:
            # Fit of the mass-weighted fields fitted to tryO13
            add_yoffset_to_bias_plots = False
            trf_fcn_fitspecs.append({
                'name': 'ValMarkoModelNoThErr',
                'type': 'fixed_fcn',
                'settings': {
                    'tf_fname': 'from_marko_and_val/fit_b1_16may2018/TF.txt',
                    'kmin': 0.01,
                    'kmax': 2.0,
                    'color': 'grey',
                    'bestfit_params': {
                        ('10.8', '16.0'):
                        dict(b1=0.943,
                             cs=-0.0798,
                             bGamma3=0.826,
                             b2=-0.0177,
                             bG2=0.162,
                             b3=-0.00164),
                        ('11.8', '16.0'):
                        dict(b1=1.162,
                             cs=0.136,
                             bGamma3=0.852,
                             b2=0.063,
                             bG2=0.123,
                             b3=-0.00736),
                        ('12.8', '16.0'):
                        dict(b1=1.811,
                             cs=1.056,
                             bGamma3=1.175,
                             b2=0.441,
                             bG2=-0.137,
                             b3=0.0473),  # (Marko: a very bad fit)
                        ('13.8', '16.0'):
                        dict(b1=3.499,
                             cs=2.125,
                             bGamma3=2.342,
                             b2=2.426,
                             bG2=-1.847,
                             b3=1.329),
                    }
                }
            })

    ## Init some stuff
    # plot options from logbook
    ymax_plot = plot_opts.get('ymax_plot', None)
    ymax_plot2 = plot_opts.get('ymax_plot2', None)
    ymin_Tk = plot_opts.get('ymin_Tk', None)
    ymax_Tk = plot_opts.get('ymax_Tk', None)
    legloc_Tk = plot_opts.get('legloc_Tk', 'upper center')
    ymin_1mr2 = plot_opts.get('ymin_1mr2', None)
    legloc_1mr2 = plot_opts.get('legloc_1mr2', 'lower right')
    legloc_full = plot_opts.get('legloc_fullbb', 'lower right')
    ymin_full = plot_opts.get('ymin_fullbb', None)
    ymax_full = plot_opts.get('ymax_fullbb', 1e8)
    tryid = plot_opts.get('tryid', None)

    ### manual plot styles for certain cases
    manual_plot_kwargs = {}
    for target_id in ['delta_m', 'deltaZA']:
        manual_plot_kwargs.update({
            'hat_delta_h_from_b1_delta_m': {
                'color': constants.purples4[-1],
                'lw': 2
            },
            'hat_delta_h_from_1_Tdeltalin_SHIFTEDBY_PsiZ': {
                'color': constants.oranges[-2],
                'lw': 2
            },
            'hat_delta_h_from_1_Tdeltalin2G2_SHIFTEDBY_PsiZ': {
                'color': constants.oranges[-3]
            },
            'hat_delta_h_from_1_Tdeltalin2G23_SHIFTEDBY_PsiZ': {
                'color': constants.oranges[-5]
            },
            'hat_delta_h_from_Tdeltalin_SHIFTEDBY_PsiZ': {
                'color': constants.oranges[-2],
                'lw': 2
            },
            'hat_delta_h_from_Tdeltalin2G2_SHIFTEDBY_PsiZ': {
                'color': constants.oranges[-3],
                'lw': 3
            },
            'hat_delta_h_from_Tdeltalin2G23_SHIFTEDBY_PsiZ': {
                'color': constants.oranges[-5],
                'lw': 2
            },
            'hat_%s_from_b1_delta_h_M0' % target_id: {
                'ls': '--',
                'color': 'grey',
                'lw': 1
            },
            'hat_%s_from_b1_delta_h_M1' % target_id: {
                'ls': ':',
                'color': 'grey',
                'lw': 1
            },
            'hat_%s_from_b1_delta_h_M0_M1' % target_id: {
                'ls': '-',
                'color': 'k'
            },
            'hat_%s_from_b1_delta_h_M0_M0.04dex' % target_id: {
                'ls': '-',
                'color': constants.oranges[-1 - 0]
            },
            'hat_%s_from_b1_delta_h_M0_M0.1dex' % target_id: {
                'ls': '-',
                'color': constants.oranges[-1 - 1]
            },
            'hat_%s_from_b1_delta_h_M0_M0.3dex' % target_id: {
                'ls': '-',
                'color': constants.oranges[-1 - 2]
            },
            'hat_%s_from_b1_delta_h_M0_M0.6dex' % target_id: {
                'ls': '-',
                'color': constants.oranges[-1 - 3]
            },
            'hat_delta_h_from_deltanonl_12G2': {
                'color': constants.purples4[-2]
            },
            'hat_delta_h_one_loop_deltaSPT': {
                'color': constants.oranges[-1]
            },
            'hat_delta_h_from_delta_m_deltalin2_deltalinG2': {
                'color': constants.oranges[-2]
            }
        })
    # mass-weighted halos with scatter
    if True:
        if mass_weighted:
            manual_plot_kwargs[
                'hat_delta_h_from_Tdeltalin2G23_SHIFTEDBY_PsiZ'] = {
                    'ls': '-',
                    'lw': 3,
                    'color': 'darkgreen',
                    'label': 'No mass weights'
                }
            #'ls': '-', 'color': 'dimgrey', 'label': 'No mass weighting'}
        for i, myscatter in enumerate(['0.4dex', '0.2dex', '0.1dex']):
            manual_plot_kwargs[
                'hat_delta_h_M0M%s_MSE_noZ_MPnn_from_T_deltalin12G23_SHIFTEDBY_PsiZ'
                % myscatter] = {
                    'ls': '-',
                    'color': constants.greens4[-1 - i],
                    'label': '$\\sigma_M=$%s' % myscatter,
                    'lw': 3
                }
        manual_plot_kwargs[
            'hat_delta_h_M0M1_MSE_noZ_MPnn_from_T_deltalin12G23_SHIFTEDBY_PsiZ'] = {
                'ls': '-',
                'color': 'lightgrey',
                'label': '$\\sigma_M=0$',
                'lw': 3
            }
        manual_plot_kwargs[
            'hat_delta_m_from_1_Tdeltalin2G23_SHIFTEDBY_PsiZ'] = {
                'color': 'grey',
                'ls': (0, [9, 2, 2, 2]),
                'lw': 1,
                'label': 'Theory limit',
                'zorder': 0,
                'alpha': 0.9
            }
        #'ls': (0, [6,3]),

    # #################################################################################
    # START PROGRAM
    # #################################################################################

    # #################################################################################
    # vary params and stack on realiaztions
    # #################################################################################

    ## default args for vary base params and stack
    default_stack_and_vary_kwargs = dict(
        pickle_path=pickle_path,
        base_fname=fname,
        comp_key='opts',  #'flat_opts',
        fname_pattern=r'^main_calc_Perr.*.dill$',
        ignore_pickle_keys=['pickle_fname', 'out_rho_path', 'in_path'],
        # ignore_pickle_keys=[
        #     'pickle_fname', 'out_rho_path', 'in_path', 'sim_irun', 'ssseed',
        #     'cats'
        # ],
        skip_keys_when_stacking=['opts', 'flat_opts', 'cat_infos'],
        return_base_vp_pickles=True,
        return_base_vp_pickle_opts=True,
        get_in_path=get_in_path)

    if True:
        # #################################################################################
        # PLOT Perr/P_poisson for different halo mass bins
        # #################################################################################

        # '2x2' or '4x1'
        layout = '2x2'

        ## params to vary for different curves: difft mass bins
        varyparams = OrderedDict()
        #halo_masses = [(10.8, 11.8), (11.8, 12.8), (12.8, 13.8), (13.8, 15.1)]
        halo_masses = [(12.8,13.8)]
        halo_mass_strings = ['%.1f_%.1f' % (hm[0], hm[1]) for hm in halo_masses]
        if True:
            for halo_mass_string in halo_mass_strings:
                varyparams[((('sim_opts', 'halo_mass_string',),
                             halo_mass_string),)] = dict()
        halo_mass_string = None

        # get stacked pickles for each entry of varyparams
        stacked_vp_pickles, base_vp_pickles, base_vp_pickles_flat_opts = (
            retrieve_pickles.get_stacked_pickles_for_varying_base_opts(
                stack_key=('sim_opts', 'sim_seed'),
                stack_values=sim_seeds,
                base_param_names_and_values=varyparams.keys(),
                **default_stack_and_vary_kwargs))

        ## PLOT
        #fig, axarr = plt.subplots(4,1,figsize=(8,14), sharex=True, sharey=False); axorder = [0,1,2,3]
        if layout == '4x1':
            fig, axarr = plt.subplots(4,
                                      1,
                                      figsize=(8, 12),
                                      sharex=True,
                                      sharey=False)
            axorder = [0, 1, 2, 3]
        elif layout == '2x2':
            fig, axarr = plt.subplots(2,
                                      2,
                                      figsize=(12, 10),
                                      sharex=False,
                                      sharey=False)
            axorder = [(0, 0), (0, 1), (1, 0), (1, 1)]
        # loop over mass bins
        for iM in range(len(halo_masses)):
            log10M_min, log10M_max = halo_masses[iM]
            halo_mass_string = halo_mass_strings[iM]
            vpkey = ((('sim_opts', 'halo_mass_string',), halo_mass_string),)

            print("stacked_vp_pickles keys:", stacked_vp_pickles.keys())
            stacked_pickles = stacked_vp_pickles[vpkey]
            base_pickle_opts = base_vp_pickles[vpkey]['opts']

            # get some basic info
            boxsize = int(base_pickle_opts['sim_boxsize'])
            redshift = 1.0 / base_pickle_opts['sim_scale_factor'] - 1.0
            kmin_plot = 1e-2
            kmax_plot = np.min(np.array([1.0, base_pickle_opts['kmax']]))

            # switch legend vs annotation
            if True or tryid in ['L32c', 'L32cNoZ']:
                #show_legend_arr = [False, False, False, False]
                show_legend_arr = [False, False, False, True]
                #show_annotation_arr = [True, True, True, False]
                show_annotation_arr = [False, False, False, False]
                show_legend = show_legend_arr[iM]
                show_annotation = show_annotation_arr[iM]
                ymax_plot2_arr = [9, 5, 1.8, 1.8]
                #ymax_plot2_arr = [1.1, 1.1, 1.1, 1.1]
                #ymax_plot2_arr = [2, 2, 1.8, 1.8]
                ymax_plot2 = ymax_plot2_arr[iM]
                ytitle_arr = [0.825, 0.825, 0.825, 0.825]

            # set up axis
            ax = axarr[axorder[iM]]
            plt.sca(ax)
            #plt.title(halo_mass_string)

            Rsmooth_lst = [t[0] for t in stacked_vp_pickles[vpkey].keys()]
            Rsmooth_lst = sorted(Rsmooth_lst)
            Rsmooth_lst = Rsmooth_lst[::-1]
            base_pickle_first_R = base_vp_pickles[vpkey][(Rsmooth_lst[0],)]
            print("tmp:", base_pickle_first_R['opts'].keys())
            trf_specs = base_pickle_first_R['opts']['trf_specs']
            trf_specs_save_bestfit_fields = [
                t.save_bestfit_field for t in trf_specs
            ]
            trf_specs_targets = [t.target_field for t in trf_specs]
            print("Trf specs:", trf_specs)
            print("\ntrf_specs_save_bestfit_fields:\n",
                  "\n".join(trf_specs_save_bestfit_fields))
            #raise Exception("dbg")

            # attempt to extract number density:
            nbar_of_cat = OrderedDict()
            cat_infos = base_pickle_first_R['cat_infos']
            for catid, cat_info in cat_infos.items():
                if cat_info.has_key('simple'):
                    if cat_info['simple'].has_key('nbar_nonuni_cat'):
                        nbar_of_cat[catid] = cat_info['simple'][
                            'nbar_nonuni_cat']
                    elif cat_info['simple'].has_key('nbar'):
                        nbar_of_cat[catid] = cat_info['simple']['nbar']
            print("nbar_of_cat:", nbar_of_cat)
            #raise Exception("dbg")

            Rsmooth = Rsmooth_lst[-1]
            tmp_key = stacked_pickles[(Rsmooth,)]['Pkmeas'].keys()[0]
            Nsims = stacked_pickles[(Rsmooth,)]['Pkmeas'][tmp_key]['k'].shape[0]
            print("Nsims:", Nsims)
            assert Nsims == len(sim_seeds)

            N_ortho_iter = base_pickle_opts['N_ortho_iter_for_trf_fcns']

            for tfcounter, tf in enumerate(trf_specs[:1]):
                for icounter, Rsmooth in enumerate(Rsmooth_lst[:1]):
                    tmp_key = stacked_pickles[(Rsmooth,)]['Pkmeas'].keys()[0]
                    kvec = np.mean(
                        stacked_pickles[(Rsmooth,)]['Pkmeas'][tmp_key]['k'],
                        axis=0)

            # CIC window (eq. 21 from jing et al https://arxiv.org/pdf/astro-ph/0409240.pdf)
            Delta_x = base_pickle_opts['sim_boxsize'] / float(
                base_pickle_opts['Ngrid'])
            k_nyq = np.pi / Delta_x
            nbar_kvec = np.zeros(kvec.shape[0] + 1)
            nbar_kvec[0] = kmin_plot
            nbar_kvec[1:] = kvec
            Pk_CIC = 1.0 - 2. / 3. * (np.sin(np.pi * nbar_kvec /
                                             (2.0 * k_nyq)))**2
            eval_Pk_CIC_at_kvec = lambda mykvec: 1.0 - 2. / 3. * (np.sin(
                np.pi * mykvec / (2.0 * k_nyq)))**2

            # #################################################################################
            # Fit transfer functions
            # #################################################################################
            if trf_fcn_fitspecs != []:
                fitted_stacked_pickles = fit_trf_fcns.do_fit_trf_fcns(
                    stacked_pickles=stacked_pickles,
                    base_pickle_first_R=base_pickle_first_R,
                    base_pickle_opts=base_pickle_opts,
                    trf_specs=trf_specs,
                    trf_fcn_fitspecs=trf_fcn_fitspecs,
                    Rsmooth_lst=Rsmooth_lst,
                    plot_results=plot_fit_trf_fcns)

            # try to plot 1/nbar_h
            plotted_shotnoise_ids = []
            # plot nbar of delta_h
            if nbar_of_cat.has_key('delta_h'):
                my_one_over_nbar = 1.0 / nbar_of_cat['delta_h']
                label = 'Poisson prediction'
                ax.plot(nbar_kvec,
                        my_one_over_nbar * Pk_CIC / my_one_over_nbar,
                        color='k',
                        ls='-',
                        lw=3,
                        zorder=0,
                        alpha=0.9,
                        label=label)
                plotted_shotnoise_ids.append('delta_h')
            else:
                raise Exception("Could not figure out nbar")

            ymax = None

            if ymax_plot2 is not None:
                ax.set_ylim((0, ymax_plot2))

            color_counter = 0
            for tfcounter, tf in enumerate(trf_specs):
                # skip cubic bias for first few mass bins b/c makes no difference
                if tf.save_bestfit_field in [
                        'hat_delta_h_from_1_Tdeltalin2G23_SHIFTEDBY_PsiZ',
                        'hat_delta_h_from_Tdeltalin2G23_SHIFTEDBY_PsiZ'
                ]:
                    if iM < 2:
                        continue
                # plot
                for icounter_R, Rsmooth in enumerate(Rsmooth_lst):
                    tmp_key = stacked_pickles[(Rsmooth,)]['Pkmeas'].keys()[0]
                    kvec = np.mean(
                        stacked_pickles[(Rsmooth,)]['Pkmeas'][tmp_key]['k'],
                        axis=0)
                    Pks = stacked_pickles[(Rsmooth,)]['Pkmeas']
                    id1 = tf.save_bestfit_field
                    id2 = tf.target_field
                    label = '%s' % get_texlabel(
                        tf.save_bestfit_field,
                        include_data_label=include_data_label,
                        explicit=plot_opts.get('AllEulerianBiases', False))

                    #col = constants.oranges4[-1-color_counter]
                    col = constants.oranges[-2 - color_counter]
                    # ls = linestyles[tfcounter]
                    ls = ''
                    if tf.save_bestfit_field in manual_plot_kwargs:
                        plt_kwargs = manual_plot_kwargs[tf.save_bestfit_field]
                        if 'color' in plt_kwargs:
                            col = plt_kwargs['color']
                        else:
                            color_counter += 1
                        if 'label' in plt_kwargs:
                            label = plt_kwargs['label']
                    else:
                        plt_kwargs = {'color': col, 'ls': ls, 'label': label}
                        color_counter += 1

                    # enforce ls
                    linestyles = [
                        '-',
                        (0, [12, 6]),
                        (10, [14, 4, 2, 4]),
                        (0, [3, 3]),
                        (0, [1, 1]),
                        (0, [8, 4]),
                    ]
                    ls = linestyles[icounter_R]
                    plt_kwargs['ls'] = ls

                    if True:
                        # plot power of (target-best_fit_field) computed on grid
                        residual_key = '[%s]_MINUS_[%s]' % (id1, id2)
                        if Pks.has_key((residual_key, residual_key)):
                            ymat = Pks[(residual_key,
                                        residual_key)]['P'] / my_one_over_nbar
                            #print("residual power:", np.mean(ymat,axis=0))
                            #ax.semilogx(kvec, np.mean(ymat,axis=0), #lw=5,
                            #            color=col, ls=linestyles[tfcounter], label=label,alpha=0.5)
                            if True:
                                # plot line
                                if tfcounter == 0:
                                    label = 'R=%s' % Rsmooth
                                else:
                                    label = '_nolabel_'
                                ax.semilogx(kvec,
                                            np.mean(ymat, axis=0),
                                            ls=plt_kwargs['ls'],
                                            color='k',
                                            lw=1,
                                            alpha=0.8,
                                            label=label)
                            if False:
                                # plot errorbar
                                plt_kwargs['marker'] = '.'
                                ax.errorbar(kvec,
                                            np.mean(ymat, axis=0),
                                            yerr=np.std(ymat, axis=0) /
                                            np.sqrt(float(Nsims - 1)),
                                            **plt_kwargs)
                            if True:
                                # plot errors using fill_between
                                yerr = np.std(ymat, axis=0) / np.sqrt(
                                    float(Nsims - 1))
                                if show_legend:
                                    mylabel = label
                                else:
                                    # annotate lines later
                                    mylabel = '_nolabel_'
                                # enforce minimal line width
                                # if ymax_plot2 is None:
                                #     if my_one_over_nbar is not None:
                                #         myones = np.ones(yerr.shape) * 0.01 # divide by my_one_over_nbar later
                                #     else:
                                #         myones = np.ones(yerr.shape)
                                # else:
                                #     myones = np.ones(yerr.shape) * 0.005*ymax_plot2/my_one_over_nbar
                                # figure height in pixels
                                tmp_height = get_ax_size(fig, ax)[1]
                                #    myones = np.ones(yerr.shape) * tmp_height * 3e-5
                                if ymax_plot2 is not None:
                                    myones = np.ones(
                                        yerr.shape
                                    ) * ymax_plot2 / tmp_height * 2
                                else:
                                    myones = np.ones(yerr.shape) * 0.02
                                myerr = np.maximum(yerr, myones)
                                ax.fill_between(
                                    kvec,
                                    (np.mean(ymat, axis=0) - myerr),
                                    (np.mean(ymat, axis=0) + myerr),
                                    facecolor=col,
                                    lw=0,
                                    alpha=0.9,  #alpha=0.6,
                                    label=mylabel)
                                if (not show_legend) and False:
                                    # draw invisible lines and annotate them later
                                    ax.plot(
                                        kvec,
                                        (np.mean(ymat, axis=0) + yerr + 0.3),
                                        alpha=0,
                                        label=label)

                    if ymax is None:
                        tmpmean = np.mean(ymat, axis=0)
                        ymax2 = np.max(tmpmean[~np.isnan(tmpmean)])
                        if ymax2 > ymax:
                            ymax = ymax2

                    # show noise if trf fcns are used for best-fit field
                    for ifit, fitspec in enumerate(trf_fcn_fitspecs):
                        # plot using fitted_stacked_pickles
                        fitted_Pks = fitted_stacked_pickles[fitspec['name']][(
                            Rsmooth,)]['Pkmeas']
                        ww = np.where((kvec >= fitspec['settings']['kmin']) &
                                      (kvec <= fitspec['settings']['kmax']))[0]
                        ymat = (fitted_Pks[(id1, id1)].P -
                                2.0 * fitted_Pks[(id1, id2)].P +
                                fitted_Pks[(id2, id2)].P) / my_one_over_nbar
                        if True:
                            # no errorbar
                            ax.loglog(
                                kvec[ww],
                                np.mean(ymat, axis=0)[ww],
                                color='k',  #col, 
                                lw=2,
                                alpha=1,  #alpha=0.6, #+ifit*0.2,
                                ls='--',
                                label='_nolabel_')
                            #label='Fitted %s' % label)
                            if False:
                                # print
                                if len(
                                        np.where(~np.isnan(
                                            np.mean(ymat, axis=0)[ww]))[0]) > 0:
                                    print("plotted k:", kvec[ww])
                                    print("plotted noise:", np.mean(ymat,
                                                                    axis=0)[ww])
                                    raise Exception("hello")
                        else:
                            # with errorbar
                            ax.errorbar(kvec[ww],
                                        np.mean(ymat, axis=0)[ww],
                                        yerr=np.std(ymat, axis=0) /
                                        np.sqrt(float(Nsims - 1)),
                                        color=col,
                                        lw=1,
                                        ls=linestyles[tfcounter],
                                        label='Fitted %s' % label)
                    if iM == 0 and tryid == 'L32cNoZ':
                        mykvec, myyvec = get_manual_noise_over_poisson_curve_low_Mbin_with_Z(
                            ax)
                        ax.plot(mykvec, myyvec, '--', color='grey')

            if ymax_plot2 is not None:
                ymax = ymax_plot2

            # plot cosmetics
            ax.set_xscale('log')
            ax.set_yscale('linear')
            #ax.legend(loc='best',ncol=1,fontsize=constants.xylabelfs-2,frameon=False)
            if layout == '4x1':
                if iM == len(halo_masses) - 1:
                    ax.set_xlabel('$k\;[h\\mathrm{Mpc}^{-1}]$',
                                  fontsize=constants.xylabelfs)
            elif layout == '2x2':
                if axorder[iM][0] == 1:
                    ax.set_xlabel('$k\;[h\\mathrm{Mpc}^{-1}]$',
                                  fontsize=constants.xylabelfs)
                if iM in [0, 1]:
                    plt.xticks([0.01, 0.1, 1], [])
                elif iM == 2:
                    plt.xticks([0.01, 0.1], ['0.01', '0.1'])
                elif iM == 3:
                    plt.xticks([0.01, 0.1, 1], ['0.01', '0.1', '1'])
            # if trf_specs_targets.count('delta_h') == len(trf_specs_targets):
            #     # all targets are delta_h
            #     #ax.set_ylabel('Noise power $\\;P_{\\,\\delta_h^\\mathrm{model}-\\delta_h^\\mathrm{truth}}$',fontsize=constants.xylabelfs)
            #     #ax.set_ylabel('Squared model error $\\;|\\delta_h^\\mathrm{model}-\\delta_h^\\mathrm{truth}|^2$',
            #     #              fontsize=constants.xylabelfs)
            #     ax.set_ylabel('$($Model error$)^2\\;\\;[1/\\bar n]$',
            #                   fontsize=constants.xylabelfs)
            # elif trf_specs_targets.count('delta_h_WEIGHT_M1') == len(trf_specs_targets):
            #     ax.set_ylabel('Noise power $\\;|\\delta_{M_h}^\\mathrm{model}-\\delta_{M_h}^\\mathrm{truth}|^2$',
            #                   fontsize=constants.xylabelfs)
            #ax.set_ylabel('$P_{\\hat\\delta-\\mathrm{target}}$')
            if False:
                ax.set_ylabel('$($Model error$)^2$/Poisson prediction',
                              fontsize=constants.xylabelfs)
            # make space for ylabel
            if iM == 2:
                ax.set_ylabel('tmp',
                              color='w',
                              fontsize=constants.xylabelfs,
                              labelpad=20)
            #ax.set_ylabel('$($Model error$)^2/\\bar n^{-1}$', fontsize=constants.xylabelfs)
            ax.set_xlim((kmin_plot, kmax_plot))
            ymin, tmp_ymax = ax.get_ylim()

            if True:
                # also show curve with P_target,target of last trf_spec (e.g. to plot P_hh).
                # do this such that ylim is not changed
                for icounter, Rsmooth in enumerate(Rsmooth_lst):
                    tmp_key = stacked_pickles[(Rsmooth,)]['Pkmeas'].keys()[0]
                    kvec = np.mean(
                        stacked_pickles[(Rsmooth,)]['Pkmeas'][tmp_key]['k'],
                        axis=0)
                    Pks = stacked_pickles[(Rsmooth,)]['Pkmeas']
                    ymat = Pks[(tf.target_field, tf.target_field)]['P']
                    if tf.target_field == 'delta_h':
                        label = '_nolabel_'  # '$P_{hh}$'
                    elif tf.target_field == 'delta_h_WEIGHT_M1':
                        label = '$P_{\\delta_{M_h}}$'
                    else:
                        label = 'P_{%s%s}' % (get_texlabel(
                            tf.target_field), get_texlabel(tf.target_field))
                    if False:
                        # include errorbar
                        ax.errorbar(kvec,
                                    np.mean(ymat, axis=0) / my_one_over_nbar,
                                    yerr=np.std(ymat, axis=0) /
                                    np.sqrt(float(Nsims - 1)) /
                                    my_one_over_nbar,
                                    lw=2,
                                    color='grey',
                                    ls=':',
                                    label=label)
                    else:
                        # no errorbar
                        ax.plot(kvec,
                                np.mean(ymat, axis=0) / my_one_over_nbar,
                                lw=2,
                                color='grey',
                                ls=':',
                                label=label)

            if False:
                # also show power of (deltaZA-deltalin) for comparison
                if ('deltaZA', 'deltalin') in Pks.keys():
                    ymat = Pks[('deltaZA', 'deltaZA')]['P'] - 2.0 * Pks[
                        ('deltaZA', 'deltalin')]['P'] + Pks[('deltalin',
                                                             'deltalin')]['P']
                    ax.errorbar(
                        kvec,
                        1.2 + 0.05 * np.mean(ymat, axis=0) / my_one_over_nbar,
                        yerr=np.std(ymat, axis=0) / np.sqrt(float(Nsims - 1)) /
                        my_one_over_nbar,
                        color='grey',
                        ls='-',
                        lw=1,
                        label='$\\propto$ Power of $(\\delta_Z-\\delta_1)$')

            if plot_opts.get('AllEulerianBiases', False):
                # also show P22 for comparison (high-k bump)
                P22 = np.genfromtxt('from_marko_and_val/P22_19jun2018/p22.txt')
                interp_P22 = interp.interp1d(P22[:, 0],
                                             P22[:, 1],
                                             kind='linear',
                                             bounds_error=False,
                                             fill_value=(P22[0, 1], P22[-1, 1]))

                # M=10.8-11.8, bumpino
                ax.plot(kvec,
                        (1.2 + 0.025 * interp_P22(kvec) / my_one_over_nbar) *
                        eval_Pk_CIC_at_kvec(kvec),
                        'k--',
                        lw=1.5,
                        alpha=0.9,
                        label='_nolabel_')  #)$\\propto P_{22}$')
                # M=10.8-11.8, bump
                #ax.plot(kvec, (1.2+0.1*interp_P22(kvec)/my_one_over_nbar)*eval_Pk_CIC_at_kvec(kvec), 'k--', label='$\\propto P_{22}$')

                # M=11.8-12.8, bumpino
                #ax.plot(kvec, (1.1+0.1*interp_P22(kvec)/my_one_over_nbar)*eval_Pk_CIC_at_kvec(kvec), 'k--', label='$\\propto P_{22}$')
                # M=11.8-12.8, bump
                #ax.plot(kvec, (1.1+0.3*interp_P22(kvec)/my_one_over_nbar)*eval_Pk_CIC_at_kvec(kvec), 'k--', label='$\\propto P_{22}$')

                if False:
                    # enforce that error power can never be larger than target power
                    Pks = stacked_pickles[(Rsmooth,)]['Pkmeas']
                    ymat = Pks[(tf.target_field, tf.target_field)]['P']
                    Ptarget = np.mean(ymat, axis=0)
                    ax.plot(kvec, ((np.minimum(
                        1.1 + 0.1 * interp_P22(kvec) / my_one_over_nbar,
                        Ptarget / my_one_over_nbar)) *
                                   eval_Pk_CIC_at_kvec(kvec)),
                            'k--',
                            label='$\\propto P_{22}$')

            ax.set_ylim((0, ymax))

            if tryid in ['L32c', 'L32cNoZ']:
                ax.set_ylim((0, ymax_plot2))
                #ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='upper'))
                #ax.yaxis.set_minor_locator(MultipleLocator(0.25))

            if True:
                # yticks
                tmp_ymin, tmp_ymax = ax.get_ylim()
                if tmp_ymax is None:
                    pass
                elif tmp_ymax < 2.:
                    minorLocator = MultipleLocator(0.1)
                    ax.yaxis.set_minor_locator(minorLocator)
                    plt.yticks([0, 0.5, 1, 1.5], ['0', '0.5', '1', '1.5'])
                    #majorLocator = MultipleLocator(1)
                    #ax.yaxis.set_major_locator(majorLocator)
                elif tmp_ymax < 2.:
                    minorLocator = MultipleLocator(0.25)
                    ax.yaxis.set_minor_locator(minorLocator)
                    majorLocator = MultipleLocator(1.0)
                    ax.yaxis.set_major_locator(majorLocator)
                elif tmp_ymax < 10.:
                    minorLocator = MultipleLocator(0.5)
                    ax.yaxis.set_minor_locator(minorLocator)
                    majorLocator = MultipleLocator(2.0)
                    ax.yaxis.set_major_locator(majorLocator)

                if layout == '2x2':
                    if axorder[iM][1] == 1:
                        ax.yaxis.tick_right()
                    else:
                        ax.yaxis.tick_left()

                        # ax.set_yticklabels([])
                        # right_ax = ax.twinx()
                        # right_ax.set_ylabel('bla')

            # title
            if True:
                make_title(plot_title,
                           log10M_min,
                           log10M_max,
                           my_one_over_nbar,
                           redshift,
                           ytitle=ytitle_arr[iM])
            if iM == 0:
                plt.title('tmp', color='w',
                          fontsize=constants.xylabelfs - 4)  # just to pad space
                #plt.title('FOF halos at $\\,z=$%.1f, no mass weighting' % redshift, y=1.01, fontsize=constants.xylabelfs-4)
            # legend
            if show_legend:
                if len(Rsmooth_lst) > 1:
                    ax.legend(loc='best',
                              ncol=2,
                              fontsize=8,
                              frameon=False,
                              handlelength=4)
                else:
                    if not mass_weighted:
                        # no mass weighting
                        if plot_opts.get('AllEulerianBiases', False):
                            # special for tryO17:
                            legend = ax.legend(loc='best',
                                               ncol=1,
                                               fontsize=constants.xylabelfs - 6,
                                               frameon=True,
                                               handlelength=1.5,
                                               labelspacing=0.3,
                                               framealpha=0.7,
                                               fancybox=True)
                            legend.get_frame().set_linewidth(1)
                        else:
                            ax.legend(loc='best',
                                      ncol=2,
                                      fontsize=constants.xylabelfs - 6,
                                      frameon=False,
                                      handlelength=1.5)
                    else:
                        # with mass weighting
                        ax.legend(loc='upper left',
                                  ncol=2,
                                  fontsize=constants.xylabelfs - 4,
                                  frameon=False,
                                  handlelength=1.5,
                                  fancybox=False)
            if show_annotation:
                print("log10M:", log10M_min, log10M_max)
                if layout == '4x1':
                    if ((not (log10M_min is None or log10M_max is None))
                            and np.isclose(float(log10M_min), 10.8)
                            and np.isclose(float(log10M_max), 11.8)):
                        # annotate lines by hand
                        #ax.text(0.0175, 165, 'Linear Eulerian bias', fontsize=constants.xylabelfs-2)
                        ax.text(0.13,
                                5.25,
                                'Linear Eul. bias',
                                fontsize=constants.xylabelfs - 2,
                                rotation=-18)
                        ax.text(0.1,
                                3.5,
                                'Linear Lagr. bias',
                                fontsize=constants.xylabelfs - 2,
                                rotation=-12)
                        ax.text(0.02,
                                1.75,
                                'Quadratic Lagr. bias',
                                fontsize=constants.xylabelfs - 2,
                                rotation=0)
                        #ax.text(0.02, 0.25, 'Poisson prediction', fontsize=constants.xylabelfs-2, rotation=0)
                        #ax.text(0.6, 3., '$P_{hh}$', fontsize=constants.xylabelfs-2, color='grey', rotation=0)
                    elif ((not (log10M_min is None or log10M_max is None))
                          and np.isclose(float(log10M_min), 11.8)
                          and np.isclose(float(log10M_max), 12.8)):
                        # annotate lines by hand
                        ax.text(0.025,
                                0.45,
                                'Poisson prediction',
                                fontsize=constants.xylabelfs - 2,
                                rotation=0)
                        #ax.text(0.65, 2., '$P_{hh}$', fontsize=constants.xylabelfs-2, color='grey', rotation=0)
                        ax.text(0.55,
                                2.5,
                                '$P_{hh}$',
                                fontsize=constants.xylabelfs - 2,
                                color='grey',
                                rotation=0)
                    elif ((not (log10M_min is None or log10M_max is None))
                          and np.isclose(float(log10M_min), 12.8)
                          and np.isclose(float(log10M_max), 13.8)):
                        # annotate lines by hand
                        ax.text(0.04,
                                1.2,
                                'Poisson prediction',
                                fontsize=constants.xylabelfs - 2,
                                rotation=0)
                        ax.text(0.025,
                                0.4,
                                'Partial cubic Lagr. bias',
                                fontsize=constants.xylabelfs - 2,
                                rotation=0)
                elif layout == '2x2':
                    if ((not (log10M_min is None or log10M_max is None))
                            and np.isclose(float(log10M_min), 10.8)
                            and np.isclose(float(log10M_max), 11.8)):
                        # annotate lines by hand
                        #ax.text(0.0175, 165, 'Linear Eulerian bias', fontsize=constants.xylabelfs-2)
                        ax.text(0.13,
                                5.25,
                                'Linear Std. Eul. bias',
                                fontsize=constants.xylabelfs - 2,
                                rotation=-34)
                        ax.text(0.09,
                                3.6,
                                'Linear bias',
                                fontsize=constants.xylabelfs - 2,
                                rotation=-25)
                        ax.text(0.015,
                                2.1,
                                'Quadr. bias',
                                fontsize=constants.xylabelfs - 2,
                                rotation=0)
                        #ax.text(0.02, 0.25, 'Poisson prediction', fontsize=constants.xylabelfs-2, rotation=0)
                        #ax.text(0.6, 3., '$P_{hh}$', fontsize=constants.xylabelfs-2, color='grey', rotation=0)
                    elif ((not (log10M_min is None or log10M_max is None))
                          and np.isclose(float(log10M_min), 11.8)
                          and np.isclose(float(log10M_max), 12.8)):
                        # annotate lines by hand
                        #ax.text(0.025, 0.45, 'Poisson prediction', fontsize=constants.xylabelfs-2, rotation=0)
                        #ax.text(0.65, 2., '$P_{hh}$', fontsize=constants.xylabelfs-2, color='grey', rotation=0)
                        ax.text(0.55,
                                2.5,
                                '$P_{hh}$',
                                fontsize=constants.xylabelfs - 2,
                                color='grey',
                                rotation=0)
                    elif ((not (log10M_min is None or log10M_max is None))
                          and np.isclose(float(log10M_min), 12.8)
                          and np.isclose(float(log10M_max), 13.8)):
                        # annotate lines by hand
                        ax.text(0.06,
                                1.06,
                                'Poisson prediction',
                                fontsize=constants.xylabelfs - 2,
                                rotation=0)
                        ax.text(0.025,
                                0.45,
                                'Cubic bias',
                                fontsize=constants.xylabelfs - 2,
                                rotation=0)

                else:
                    # annotate lines automatically
                    from label_lines import labelLines
                    #print("children:", plt.gca().get_children())
                    #raise Exception("tmp plot")
                    labelLines(
                        plt.gca().get_lines(),
                        bgalpha=0,
                        xvals=0.08,  # [10,5,5,5,5,18, 300, 100]
                        backgroundcolor='white',
                        align=False,
                        clip_on=False,
                        zorder=2.5,
                        fontsize=18,
                        va='bottom',
                        ha='left',
                        color='k')

                    #fig.text(0.06, 0.5, '$($Model error$)^2$/Poisson prediction $\\;[=\\,\\bar n P_{\\!\\mathregular{err}}]$',
                    #fig.text(0.06, 0.5, '$($Model error$)^2$/Poisson prediction',
                    #fig.text(0.06, 0.5, '$P_\\mathrm{err}/(1/\\bar n)$ = $($Model error$)^2$/Poisson prediction',
        fig.text(0.06,
                 0.5,
                 '$P_\\mathrm{err}\\;/$ Poisson prediction',
                 fontsize=constants.xylabelfs,
                 ha='center',
                 va='center',
                 rotation='vertical')
        fig.text(0.5,
                 0.96,
                 'FOF halos at $\\,z=$%.1f, no mass weighting' % redshift,
                 fontsize=constants.xylabelfs - 4,
                 ha='center',
                 va='center')

        if layout == '4x1':
            plt.tight_layout(w_pad=0, h_pad=-0.5)
        elif layout == '2x2':
            plt.tight_layout(w_pad=-0.5, h_pad=0)

        if save_plots:
            plot_fname = 'Perr_over_poisson_multipanel.pdf'
            plt.savefig(plot_fname)
            print("Made %s" % plot_fname)
        else:
            plt.show()

    if False:
        # #################################################################################
        # PLOT r(bestfit_field,target) and 1-r^2
        # #################################################################################


        ## PLOT
        fig, axarr = plt.subplots(
            5,
            2,
            figsize=(12, 13),
            sharex=False,
            sharey=False,
            gridspec_kw={'height_ratios': [3, 1, 0.3, 3, 1]})
        axorder_1mr2 = [(0, 0), (0, 1), (3, 0), (3, 1)]
        axorder_r = [(1, 0), (1, 1), (4, 0), (4, 1)]

        DMerror_stacked_pickles = None
        twinx = False
        # if twinx:
        #     fig, axlst = plt.subplots(2,1,sharex=True,figsize=(8,8),gridspec_kw={'height_ratios':[3, 1]})
        # else:
        #     fig, axlst = plt.subplots(2,1,sharex=True,figsize=(8,8),gridspec_kw={'height_ratios':[3, 1]})
        show_fit = False
        #linestyles = ['-','--',':',(2, [10, 2, 4, 2]), '-.',(2, [10, 2, 10, 2]),
        #              (2, [10, 5, 2, 1]),(2, [10, 4, 10, 4]),(2,[10,8,10,8]),(2,[5,5,5,5])]
        # rec:  (0, [3,3]), (0, [20,2,2,2]), (0, [6,6])
        linestyles = [(10, [14, 4, 2, 4]), (0, [12, 6]), (0, [8, 4]),
                      (0, [3, 3]), (0, [1, 1])]
        # loop over panels
        for iM in range(len(halo_masses)):
            log10M_min, log10M_max = halo_masses[iM]
            halo_mass_string = halo_mass_strings[iM]
            vpkey = ((('halo_mass_string',), halo_mass_string),)

            print("stacked_vp_pickles keys:", stacked_vp_pickles.keys())
            stacked_pickles = stacked_vp_pickles[vpkey]
            base_pickle_opts = base_vp_pickles[vpkey]['opts']

            # get some basic info
            boxsize = int(base_pickle_opts['sim_boxsize'])
            redshift = 1.0 / base_pickle_opts['sim_scale_factor'] - 1.0
            kmin_plot = 1e-2
            kmax_plot = np.min(np.array([1.0, base_pickle_opts['kmax']]))

            for plt_type in ['1mr2', 'r']:
                # set axis
                if plt_type == '1mr2':
                    axrow, axcol = axorder_1mr2[iM]
                    # legend, ymin, etc
                    if True or tryid in ['L32c', 'L32cNoZ']:
                        show_legend_arr = [False, True, False, False]
                        #show_legend_arr = [True, False, False, False]
                        show_legend = show_legend_arr[iM]
                        show_legend2_arr = [False, False, False, True]
                        show_legend2 = show_legend2_arr[iM]
                        #ymin_1mr2_arr = [0.00005, 0.0005, 0.00101, 0.00101]
                        ymin_1mr2_arr = [0.00101, 0.00101, 0.00101, 0.00101]
                        ymin_1mr2 = ymin_1mr2_arr[iM]
                        ytitle_arr = [0.86, 0.86, 0.14, 0.14]
                        ytitle = ytitle_arr[iM]
                elif plt_type == 'r':
                    axrow, axcol = axorder_r[iM]
                ax = axarr[(axrow, axcol)]
                plt.sca(ax)

                Rsmooth_lst = [t[0] for t in stacked_vp_pickles[vpkey].keys()]
                Rsmooth_lst = sorted(Rsmooth_lst)
                Rsmooth_lst = Rsmooth_lst[::-1]
                base_pickle_first_R = base_vp_pickles[vpkey][(Rsmooth_lst[0],)]
                print("tmp:", base_pickle_first_R['opts'].keys())
                trf_specs = base_pickle_first_R['opts']['trf_specs']
                trf_specs_save_bestfit_fields = [
                    t.save_bestfit_field for t in trf_specs
                ]
                trf_specs_targets = [t.target_field for t in trf_specs]
                print("Trf specs:", trf_specs)
                print("\ntrf_specs_save_bestfit_fields:\n",
                      "\n".join(trf_specs_save_bestfit_fields))
                #raise Exception("dbg")

                # attempt to extract number density:
                nbar_of_cat = OrderedDict()
                cat_infos = base_pickle_first_R['cat_infos']
                for catid, cat_info in cat_infos.items():
                    if cat_info.has_key('simple'):
                        if cat_info['simple'].has_key('nbar_nonuni_cat'):
                            nbar_of_cat[catid] = cat_info['simple'][
                                'nbar_nonuni_cat']
                        elif cat_info['simple'].has_key('nbar'):
                            nbar_of_cat[catid] = cat_info['simple']['nbar']
                print("nbar_of_cat:", nbar_of_cat)
                #raise Exception("dbg")

                Rsmooth = Rsmooth_lst[-1]
                tmp_key = stacked_pickles[(Rsmooth,)]['Pkmeas'].keys()[0]
                Nsims = stacked_pickles[(
                    Rsmooth,)]['Pkmeas'][tmp_key]['k'].shape[0]
                print("Nsims:", Nsims)
                assert Nsims == len(sim_seeds)

                N_ortho_iter = base_pickle_opts['N_ortho_iter_for_trf_fcns']

                for tfcounter, tf in enumerate(trf_specs[:1]):
                    for icounter, Rsmooth in enumerate(Rsmooth_lst[:1]):
                        tmp_key = stacked_pickles[(
                            Rsmooth,)]['Pkmeas'].keys()[0]
                        kvec = np.mean(
                            stacked_pickles[(Rsmooth,)]['Pkmeas'][tmp_key]['k'],
                            axis=0)

                # CIC window (eq. 21 from jing et al https://arxiv.org/pdf/astro-ph/0409240.pdf)
                Delta_x = base_pickle_opts['sim_boxsize'] / float(
                    base_pickle_opts['Ngrid'])
                k_nyq = np.pi / Delta_x
                nbar_kvec = np.zeros(kvec.shape[0] + 1)
                nbar_kvec[0] = kmin_plot
                nbar_kvec[1:] = kvec
                Pk_CIC = 1.0 - 2. / 3. * (np.sin(np.pi * nbar_kvec /
                                                 (2.0 * k_nyq)))**2
                eval_Pk_CIC_at_kvec = lambda mykvec: 1.0 - 2. / 3. * (np.sin(
                    np.pi * mykvec / (2.0 * k_nyq)))**2

                for tfcounter, tf in enumerate(trf_specs):
                    for icounter, Rsmooth in enumerate(Rsmooth_lst):

                        # skip cubic bias for first few mass bins b/c makes no difference
                        if tf.save_bestfit_field in [
                                'hat_delta_h_from_1_Tdeltalin2G23_SHIFTEDBY_PsiZ',
                                'hat_delta_h_from_Tdeltalin2G23_SHIFTEDBY_PsiZ'
                        ]:
                            if iM in [0, 1]:
                                continue

                        tmp_key = stacked_pickles[(
                            Rsmooth,)]['Pkmeas'].keys()[0]
                        kvec = np.mean(
                            stacked_pickles[(Rsmooth,)]['Pkmeas'][tmp_key]['k'],
                            axis=0)
                        Pks = stacked_pickles[(Rsmooth,)]['Pkmeas']
                        id1 = tf.save_bestfit_field
                        id2 = tf.target_field
                        lw = 2
                        print("1mr2 id1,id2:", id1, id2)
                        if plt_type == 'r':
                            ymat = Pks[(id1, id2)]['P'] / np.sqrt(
                                Pks[(id1, id1)]['P'] * Pks[(id2, id2)]['P'])
                        elif plt_type == '1mr2':
                            ymat = 1.0 - Pks[(id1, id2)]['P']**2 / (Pks[
                                (id1, id1)]['P'] * Pks[(id2, id2)]['P'])
                        else:
                            raise Exception("Invalid plt_type %s" %
                                            str(plt_type))
                        if len(Rsmooth_lst) > 1:
                            label = '%s, $R=$%g' % (get_texlabel(
                                tf.save_bestfit_field,
                                include_data_label=include_data_label), Rsmooth)
                            col = constants.oranges[-2 - icounter]
                        else:
                            label = '%s' % get_texlabel(
                                tf.save_bestfit_field,
                                include_data_label=include_data_label)
                            col = 'k'
                        if tf.save_bestfit_field in manual_plot_kwargs:
                            plt_kwargs = manual_plot_kwargs[
                                tf.save_bestfit_field]
                            if 'color' in plt_kwargs:
                                col = plt_kwargs['color']
                            else:
                                color_counter += 1
                            if 'label' in plt_kwargs:
                                label = plt_kwargs['label']
                            if 'lw' in plt_kwargs:
                                lw = plt_kwargs['lw']

                        if show_legend2:
                            if 'cubic' in label or 'Cubic' in label or 'Linear' in label:
                                # leave label as is
                                pass
                            else:
                                label = '_nolabel_'
                        elif not show_legend:
                            label = '_nolabel_'
                        if show_legend and 'Linear' in label:
                            label = '_nolabel_'

                        # force to black
                        if not mass_weighted:
                            if lw > 2:
                                col = 'k'
                            else:
                                col = 'grey'

                        if True:
                            # no errorbar
                            if mass_weighted:
                                plt_kwargs.update({
                                    'color': col,
                                    'alpha': 1,
                                    'marker': '',
                                    'zorder': 1 + tfcounter,
                                    'label': label
                                })
                            else:

                                plt_kwargs = {
                                    'ls': linestyles[tfcounter],
                                    'label': label,
                                    'lw': lw,
                                    'alpha': 0.9,
                                    'color': col,
                                    'marker': '',
                                    'zorder': 1 + tfcounter
                                }
                            ax.loglog(kvec, np.mean(ymat, axis=0), **plt_kwargs)

                            # ax.loglog(kvec, np.mean(ymat,axis=0),
                            #           color=col, ls=ls, marker='', alpha=alpha, zorder=1+tfcounter,
                            #           label=label, lw=lw)
                        if False:
                            # with errorbar
                            ax.errorbar(kvec,
                                        np.mean(ymat, axis=0),
                                        yerr=np.std(ymat, axis=0) /
                                        np.sqrt(float(Nsims - 1)),
                                        color=col,
                                        ls=linestyles[tfcounter],
                                        label=label)
                        if False:
                            # plot errors using fill_between
                            yerr = np.std(ymat, axis=0) / np.sqrt(
                                float(Nsims - 1))
                            if show_legend:
                                mylabel = label
                            else:
                                # annotate lines later
                                mylabel = '_nolabel_'
                            # do as fraction of fig height
                            tmp_height = get_ax_size(fig, ax)[1]
                            myones = np.ones(yerr.shape) * tmp_height * 3e-5
                            myerr = np.maximum(yerr, myones)
                            myerr = yerr
                            ax.fill_between(
                                kvec,
                                (np.mean(ymat, axis=0) - myerr),
                                (np.mean(ymat, axis=0) + myerr),
                                facecolor=col,
                                lw=0,
                                alpha=0.9,  #alpha=0.6,
                                label=mylabel)

                        # show noise if trf fcns are used for best-fit field
                        if False:
                            for fitspec in trf_fcn_fitspecs:
                                # plot using fitted_stacked_pickles
                                fitted_Pks = fitted_stacked_pickles[
                                    fitspec['name']][(Rsmooth,)]['Pkmeas']
                                ymat = (1.0 - fitted_Pks[(id1, id2)].P**2 /
                                        (fitted_Pks[(id1, id1)].P * fitted_Pks[
                                            (id2, id2)].P))
                                ax.loglog(kvec,
                                          np.mean(ymat, axis=0),
                                          color=col,
                                          ls=linestyles[tfcounter],
                                          label='Fitted %s' % label)

                if False:
                    # also show r(deltaZA,deltalin) for comparison
                    if ('deltaZA', 'deltalin') in Pks.keys():
                        ymat = Pks[('deltaZA', 'deltalin')]['P'] / np.sqrt(
                            Pks[('deltaZA', 'deltaZA')]['P'] *
                            Pks[('deltalin', 'deltalin')]['P'])
                        ax.errorbar(kvec,
                                    np.mean(ymat, axis=0),
                                    yerr=np.std(ymat, axis=0) /
                                    np.sqrt(float(Nsims - 1)),
                                    color='grey',
                                    ls='-',
                                    lw=1,
                                    label='$1-r^2(\\delta_Z,\\delta_1)$')

                ## plot Poisson prediction 1-r^2 = 1/nbar_h / P_hh, and corresponding r
                plotted_shotnoise_ids = []
                #my_one_over_nbar = None
                for tfcounter, tf in enumerate([trf_specs[-1]]):

                    #if nbar_of_cat.has_key(tf.target_field):
                    if True:
                        if tf.target_field not in plotted_shotnoise_ids:
                            #if tf.target_field in ['delta_h','delta_h_WEIGHT_M1']:
                            my_one_over_nbar = 1.0 / nbar_of_cat['delta_h']
                            Ptarget = np.mean(
                                stacked_pickles[(Rsmooth_lst[0],)]['Pkmeas'][(
                                    tf.target_field, tf.target_field)]['P'],
                                axis=0)
                            if plt_type == '1mr2':
                                # 1-r^2 = Perr/Ptarget = 1/nbar_h / P_hh
                                yvec = my_one_over_nbar * eval_Pk_CIC_at_kvec(
                                    kvec) / Ptarget
                            elif plt_type == 'r':
                                # |r| = sqrt(1-(1-r^2)) = sqrt(1-1/nbar_h / P_hh)
                                yvec = np.sqrt(1.0 - my_one_over_nbar *
                                               eval_Pk_CIC_at_kvec(kvec) /
                                               Ptarget)
                                print("r prediction:", yvec)
                            if show_legend:
                                label = 'Poisson prediction'
                            else:
                                label = '_nolabel_'
                            ax.plot(kvec,
                                    yvec,
                                    color='k',
                                    ls='-',
                                    lw=3,
                                    zorder=0,
                                    alpha=1,
                                    label=label)

                ## plot error of DM model
                DMerror_stacked_pickles = plot_DM_model_error(
                    ax,
                    plt_type,
                    plot_opts,
                    pickle_path,
                    DMerror_stacked_pickles,
                    show_legend2,
                    sim_seeds=[400, 401, 402],
                )

                ## plot cosmetics
                ax.set_xscale('log')
                if plt_type == 'r':
                    ax.set_yscale('linear')
                elif plt_type == '1mr2':
                    ax.set_yscale('log')
                legend = None
                if plt_type == '1mr2':
                    if mass_weighted:
                        if False and np.isclose(float(log10M_min), 10.8):
                            legend = ax.legend(loc=legloc_1mr2,
                                               ncol=1,
                                               fontsize=constants.xylabelfs - 4,
                                               frameon=False,
                                               handlelength=1.5,
                                               handletextpad=0.1,
                                               columnspacing=0.4)
                        else:
                            if np.isclose(float(log10M_min), 10.8):
                                alpha = 0.9
                            elif np.isclose(float(log10M_min), 11.8):
                                alpha = 0.9
                            elif np.isclose(float(log10M_min), 12.8):
                                alpha = 0.96
                            else:
                                alpha = 1
                            legend = ax.legend(loc=legloc_1mr2,
                                               ncol=2,
                                               fontsize=constants.xylabelfs - 4,
                                               frameon=True,
                                               handlelength=1.5,
                                               handletextpad=0.1,
                                               labelspacing=0.3,
                                               columnspacing=0.4,
                                               fancybox=True,
                                               framealpha=alpha)
                            legend.get_frame().set_linewidth(1)
                    else:
                        if show_legend:
                            legloc_1mr2 = 'lower right'
                        elif show_legend2:
                            legloc_1mr2 = 'center right'
                        legend = ax.legend(
                            loc=legloc_1mr2,
                            ncol=1,
                            fontsize=constants.xylabelfs - 4,
                            handlelength=2.3,
                            handletextpad=0.2,
                            frameon=True,
                            framealpha=0.9,
                            fancybox=True,
                        )
                        if legend is not None:
                            legend.get_frame().set_linewidth(1)
                    # ax.legend(loc='best',ncol=2,fontsize=constants.xylabelfs-6,frameon=False,handlelength=3,
                    #           handletextpad=0.1, columnspacing=0.2)
                if legend is not None:
                    legend.set_zorder(20)  # put the legend on top
                if ax in [axarr[(4, 0)], axarr[(4, 1)]]:
                    ax.set_xlabel('$k\;[h\\mathrm{Mpc}^{-1}]$',
                                  fontsize=constants.xylabelfs)
                ax.set_xlim((kmin_plot, kmax_plot))
                if False:
                    if axcol == 1:
                        ax.yaxis.tick_right()
                    else:
                        ax.yaxis.tick_left()

                if ax == axarr[(4, 0)]:
                    plt.xticks([0.01, 0.1], ['0.01', '0.1'])
                elif ax == axarr[(4, 1)]:
                    plt.xticks([0.01, 0.1, 1], ['0.01', '0.1', '1'])
                else:
                    plt.xticks([0.01, 0.1, 1], [])

                if twinx:
                    ax.xaxis.set_tick_params(pad=15)
                if plt_type == 'r':
                    ax.set_ylim((0, 1.1))
                    ax.set_yticks([0, 0.5, 1.])
                    ax.set_yticklabels(['0', '0.5', '1'])
                    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
                    if axcol == 0:
                        #ax.set_ylabel('$r_{\\!cc}$(model, truth)', fontsize=constants.xylabelfs)
                        #ax.set_ylabel('$r_{\\mathregular{model,truth}}$', fontsize=constants.xylabelfs)
                        ax.set_ylabel('$r_{\\!cc}$',
                                      fontsize=constants.xylabelfs)
                        #ax.set_ylabel('$r_{\\mathregular{cc}}(\\delta_\\mathregular{model},\\delta_\\mathregular{truth})$',
                        #              fontsize=constants.xylabelfs)

                elif plt_type == '1mr2':
                    if axcol == 0:
                        #ax.set_ylabel('$($Model error$)^2$/truth$^2\\,$ [%]', fontsize=constants.xylabelfs)
                        #ax.set_ylabel('$($Model error$)^2/P_\\mathregular{truth}\\,$ [%]', fontsize=constants.xylabelfs)
                        ax.set_ylabel(
                            '$P_\\mathregular{err}/P_\\mathregular{truth}\\,$ [%]',
                            fontsize=constants.xylabelfs)
                    ax.set_yscale('log')

                    # ylim
                    ymax = 1.0
                    if ymin_1mr2 is None:
                        if mass_weighted:
                            ymin_1mr2 = 5e-5
                        else:
                            ymin_1mr2 = 5e-4
                    ax.set_ylim((ymin_1mr2, ymax))

                    # yticks and labels
                    if ymin_1mr2 < 1e-4:
                        yticks = [1.e-4, 1.e-3, 1.e-2, 0.1, 1.]
                        ax.set_yticks(yticks)
                        ax.set_yticklabels(['0.01', '0.1', '1', '10', '100'])
                    elif ymin_1mr2 < 1e-3:
                        yticks = [1.e-3, 1.e-2, 0.1, 1.]
                        ax.set_yticks(yticks)
                        ax.set_yticklabels(['0.1', '1', '10', '100'])
                    elif ymin_1mr2 < 1e-2:
                        yticks = [1.e-2, 0.1, 1.]
                        ax.set_yticks(yticks)
                        ax.set_yticklabels(['1', '10', '100'])
                    if twinx:
                        # set same ticks on right axis
                        plt.sca(ax_right)
                        plt.yticks(yticks)

                if ax == axarr[(0, 0)]:
                    #ax.text(0.15,0.004,
                    ax.text(0.015,
                            0.1,
                            'Equals $1-r_{\\!cc}^2$',
                            fontsize=constants.xylabelfs - 4,
                            verticalalignment='center')
                if ax == axarr[(1, 0)]:
                    ax.text(
                        0.015,
                        0.5,
                        'Equals $\\sqrt{P_{\\mathregular{model}}/P_{\\mathregular{truth}}}$',
                        fontsize=constants.xylabelfs - 4,
                        verticalalignment='center')

                if axcol == 1:
                    ax.axes.set_yticklabels([])

                # title
                if axrow in [0, 3]:
                    make_title_1mr2(plot_title,
                                    log10M_min,
                                    log10M_max,
                                    my_one_over_nbar,
                                    redshift,
                                    ax=ax,
                                    ytitle=ytitle)

        # blank axes to make space
        for ax in [axarr[(2, 0)], axarr[(2, 1)]]:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.patch.set_alpha(0)
            for child in ax.get_children():
                if isinstance(child, spines.Spine):
                    child.set_color('k')
                    child.set_alpha(0)

        # just to pad space for title
        plt.sca(axarr[(0, 0)])
        plt.title('tmp', color='w', fontsize=constants.xylabelfs - 4, y=1.01)

        fig.text(0.5,
                 0.97,
                 'FOF halos at $\\,z=$%.1f, no mass weighting' % redshift,
                 fontsize=constants.xylabelfs - 4,
                 ha='center',
                 va='center')

        plt.tight_layout(w_pad=-0.2, h_pad=-0.55)
        if save_plots:
            plt.savefig('r_1mr2_model_target_multipanel.pdf')
            print("Made r_1mr2_model_target_multipanel.pdf")
            plt.close(fig)
        else:
            plt.show()

            #raise Exception("mydbg r,1mr2 plot")


def get_manual_noise_over_poisson_curve_low_Mbin_with_Z(ax):
    # manually plot a best fit curve
    mykvec = np.array([
        0.01603648, 0.0280331, 0.03938502, 0.05102672, 0.06405809, 0.07693044,
        0.08886613, 0.10084619, 0.11381908, 0.12683675, 0.13899979, 0.15116243,
        0.16392559, 0.17660451, 0.18870457, 0.20111994, 0.21402617, 0.22685364,
        0.23921812, 0.25155979, 0.26429492, 0.27690971, 0.28930044, 0.30170608,
        0.31434393, 0.32707798, 0.33965057, 0.35220519, 0.36479989, 0.37730417,
        0.38981238, 0.40228549, 0.4147912, 0.42748588, 0.44006759, 0.45258608,
        0.46519542, 0.47776324, 0.49023706, 0.5027464, 0.51540071, 0.52803844,
        0.54057419, 0.55311686, 0.56572038, 0.57829171, 0.59070671, 0.60321969,
        0.61587656, 0.62856972, 0.64107859, 0.65361935, 0.66626143, 0.67882073,
        0.69128084, 0.70375228, 0.71643144, 0.72900069, 0.7415539, 0.7541005,
        0.7667442, 0.77930748, 0.79173762, 0.80422294, 0.81690204, 0.82956392,
        0.84210765, 0.8546384, 0.86721361, 0.87984884, 0.89229763, 0.9048022,
        0.91737491, 0.9300763, 0.94265044, 0.95520389, 0.96769828, 0.98029709,
        0.99283713, 1.00525081, 1.01804209, 1.03047383, 1.04310346, 1.05574453,
        1.06810057, 1.0809797, 1.09331846, 1.10586488, 1.11863148, 1.13088524,
        1.14382756, 1.15626192, 1.16868067, 1.18154037, 1.19360888, 1.20637083,
        1.21896255, 1.2316047, 1.24450791, 1.25659776, 1.26910591, 1.28177989,
        1.29457033, 1.30729127, 1.31943142, 1.33186817, 1.34444809, 1.3575176,
        1.37021661, 1.38240564, 1.39460373, 1.40706825, 1.42047548, 1.4330368,
        1.44520783, 1.45736229, 1.4696995, 1.48354566, 1.49588251, 1.50766611,
        1.52064288, 1.53244114, 1.54602838, 1.55924273, 1.56966817, 1.58440316,
        1.59512615, 1.60855031, 1.62241805, 1.63174021, 1.64810562, 1.65772605,
        1.67139399, 1.68544447, 1.69380116, 1.71184182, 1.7203182, 1.73430276,
        1.74832225, 1.75578475, 1.77553332, 1.78286004, 1.79711568, 1.81114793,
        1.81821001, 1.8388567, 1.84535956, 1.86019111, 1.87398303, 1.88101327,
        1.90094149, 1.90888941, 1.92416418, 1.9367559, 1.94344497, 1.96105647,
        1.97338712, 1.9883858, 1.99944174
    ])
    myyvec = np.array([
        1.76993141, 1.43805014, 1.41427444, 1.43880007, 1.34132438, 1.44560413,
        1.52518134, 1.48158892, 1.47821744, 1.53406319, 1.50605874, 1.48823982,
        1.52161471, 1.52164154, 1.52883657, 1.51841186, 1.50917734, 1.50242994,
        1.490242, 1.47784414, 1.47575955, 1.47329756, 1.48349674, 1.48526912,
        1.49278612, 1.47464662, 1.45990383, 1.449458, 1.45105281, 1.44864512,
        1.43930565, 1.42525864, 1.43506593, 1.412487, 1.41624347, 1.42317175,
        1.42902248, 1.41126099, 1.40898905, 1.4088685, 1.40757426, 1.39635286,
        1.40036514, 1.38713575, 1.37971232, 1.37706346, 1.37645052, 1.37463325,
        1.36295665, 1.36644193, 1.36211895, 1.34670175, 1.35296789, 1.34631351,
        1.33289808, 1.33025625, 1.33361289, 1.32244521, 1.31218434, 1.31691982,
        1.3066499, 1.30809652, 1.30401075, 1.29962011, 1.2871055, 1.28962498,
        1.28189584, 1.27968503, 1.27275513, 1.2708638, 1.26648443, 1.25351459,
        1.25226354, 1.250364, 1.24226615, 1.23595111, 1.23126725, 1.21805123,
        1.2217053, 1.21679938, 1.20615917, 1.20133164, 1.19657283, 1.18959237,
        1.17961006, 1.17806841, 1.17143275, 1.16432722, 1.16593943, 1.15617638,
        1.15004312, 1.1452753, 1.13673778, 1.13375656, 1.12273096, 1.12026694,
        1.11032183, 1.10572807, 1.1021038, 1.09326199, 1.09142093, 1.08094018,
        1.07402844, 1.0672428, 1.06323352, 1.05648763, 1.04974494, 1.04154631,
        1.03501814, 1.02868121, 1.02353058, 1.01703073, 1.01082138, 1.00595764,
        0.99655588, 0.98794621, 0.98261563, 0.97565181, 0.96978709, 0.96377941,
        0.95636901, 0.94863632, 0.94272858, 0.93659044, 0.92709141, 0.9231982,
        0.91668623, 0.9086559, 0.90538987, 0.895354, 0.88848352, 0.8822764,
        0.87739518, 0.87067007, 0.86089132, 0.85632247, 0.84902952, 0.84144269,
        0.83479691, 0.82967533, 0.82296237, 0.81577067, 0.80867148, 0.80274127,
        0.79515079, 0.78826413, 0.78121597, 0.77528243, 0.7701505, 0.76358114,
        0.75613061, 0.75039824, 0.7429135, 0.73680632, 0.72989709, 0.72451528,
        0.71745971, 0.71235132, 0.70633354
    ])
    return mykvec, myyvec


def make_title(plot_title,
               log10M_min,
               log10M_max,
               my_one_over_nbar,
               redshift,
               ytitle=0.85,
               ax=None):
    title_str = None
    if not (log10M_min is None or log10M_max is None):
        if plot_title:
            if my_one_over_nbar is not None:
                if np.isclose(float(log10M_max), 16.0):
                    title_str = '$%s\\geq $%s$\\,%s$, $\\,\\bar n=$%.1e' % (
                        constants.log10M_string2, log10M_min,
                        constants.hMsun_string, 1.0 / my_one_over_nbar)
                else:
                    title_str = '$%s=$%s$-$%s\n$\\bar n=$%.1e' % (
                        constants.log10M_string2, log10M_min, log10M_max,
                        1.0 / my_one_over_nbar)
    # write title as text field
    if ax is None:
        ax = plt.gca()
    bbox_props = dict(boxstyle='round',
                      facecolor='lightgrey',
                      alpha=0.5,
                      linewidth=0)
    mypatch = ax.text(0.95,
                      ytitle,
                      title_str,
                      fontsize=constants.xylabelfs - 4,
                      horizontalalignment='right',
                      transform=ax.transAxes,
                      color='black',
                      bbox=bbox_props)
    #mypatch.get_frame().set_linewidth(1)


def make_title_1mr2(plot_title,
                    log10M_min,
                    log10M_max,
                    my_one_over_nbar,
                    redshift,
                    ytitle=0.85,
                    ax=None):
    title_str = None
    if not (log10M_min is None or log10M_max is None):
        if plot_title:
            if my_one_over_nbar is not None:
                if np.isclose(float(log10M_max), 16.0):
                    title_str = '$%s\\geq $%s$\\,%s$, $\\,\\bar n=$%.1e' % (
                        constants.log10M_string2, log10M_min,
                        constants.hMsun_string, 1.0 / my_one_over_nbar)
                else:
                    title_str = '$\\log M=$%s$-$%s\n$\\bar n=$%.1e' % (
                        log10M_min, log10M_max, 1.0 / my_one_over_nbar)
    # write title as text field
    if ax is None:
        ax = plt.gca()
    bbox_props = dict(boxstyle='round',
                      facecolor='lightgrey',
                      alpha=0.5,
                      linewidth=0)
    if ytitle > 0.5:
        mypatch = ax.text(0.05,
                          ytitle,
                          title_str,
                          fontsize=constants.xylabelfs - 4,
                          horizontalalignment='left',
                          va='center',
                          transform=ax.transAxes,
                          color='black',
                          bbox=bbox_props)
    else:
        mypatch = ax.text(0.05,
                          ytitle,
                          title_str,
                          fontsize=constants.xylabelfs - 4,
                          horizontalalignment='left',
                          va='center',
                          transform=ax.transAxes,
                          color='black',
                          bbox=bbox_props)
    #mypatch.get_frame().set_linewidth(1)


def plot_DM_model_error(ax, plt_type, plot_opts, pickle_path,
                        DMerror_stacked_pickles, show_legend2, sim_seeds):
    """
    Load another pickle where we included DM model error and plot its 1-r^2 and r
    """
    pickle_fname = plot_opts.get('pickle_fname_DM_model_error', None)
    if pickle_fname is None:
        return

    if DMerror_stacked_pickles is None:
        # read pickle
        # default stack options for rec pickle
        default_stack_kwargs = dict(
            pickle_path=pickle_path,
            base_fname=pickle_fname,
            comp_key='flat_opts',
            outer_key_name=None,
            outer_key_values=None,
            call_wig_now_example=False,
            fname_pattern=r'^main_do_stage012_rec.*.pickle$',
            ignore_pickle_keys=[
                'pickle_fname', 'opts', "('stage0', 'out_rho_path')",
                "('stage0', 'in_path')", "('stage0', 'sim_irun')",
                "('stage0', 'ssseed')", "('stage12', 'sim_seed')",
                "('stage12', 'ssseed')"
            ],
            skip_keys_when_stacking=['opts', 'flat_opts', 'cat_infos'],
            return_base_pickle=True,
            return_base_pickle_opts=True)

        # get stacked pickles
        DMerror_stacked_pickles, base_pickle, base_pickle_flat_opts = retrieve_pickles.get_stacked_pickles(
            vary_key="('stage0', 'sim_seed')",
            vary_values=sim_seeds,
            **default_stack_kwargs)

    # plot from pickle
    id1, id2 = 'hat_delta_m_from_1_TdeltalinG2_SHIFTEDBY_PsiZ', 'delta_m'
    Pkmeas0 = DMerror_stacked_pickles['stage0']['Pkmeas']
    kvec = np.mean(Pkmeas0[(id1, id2)]['k'], axis=0)
    rmat = Pkmeas0[(id1, id2)]['P'] / np.sqrt(
        Pkmeas0[(id1, id1)]['P'] * Pkmeas0[(id2, id2)]['P'])
    plt_kwargs = dict(ls='-', color='grey', lw=1)
    if show_legend2:
        label = 'Error of DM model'
    else:
        label = '_nolabel_'
    if plt_type == '1mr2':
        ax.loglog(kvec,
                  np.mean(1.0 - rmat**2, axis=0),
                  label=label,
                  **plt_kwargs)
    elif plt_type == 'r':
        ax.semilogx(kvec, np.mean(rmat, axis=0), **plt_kwargs)

        #raise Exception("continue here")

    return DMerror_stacked_pickles


def get_texlabel(label, include_data_label=False, explicit=False):
    # support bold phi
    #rc('text', usetex=True)
    #rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    #\\boldsymbol{\psi}_Z

    texlabels = {
        'delta_m': '$\\delta_m$',
        'delta_m_growth': '$\\delta_{m}^2$',
        'delta_m_tidal_G2': '$\\mathcal{G}_{2}[\\delta_m]$',
        'hat_delta_h_from_b1_deltalin': '$t_1\delta_1$',
        'hat_delta_h_from_b1_deltaZA': '$t_1\delta_Z$',
        'hat_delta_h_one_loop_deltalin':
        '$t_1\\delta_1+t_2F_2[\\delta_1]+t_3\\delta_1^2+t_4\\mathcal{G}_2[\\delta_1]$',
        'hat_delta_h_one_loop_deltaSPT':
        '$t_1(\\delta_1+F_2[\\delta_1])+t_2\\delta_1^2+t_{\\mathcal{G}_2}\\mathcal{G}_2[\\delta_1]$',
        'hat_delta_h_one_loop_deltaZA':
        '$t_1\\delta_Z+t_2\\delta_Z^2+t_3\\mathcal{G}_2[\\delta_Z]$',
        'hat_delta_h_from_deltanonl':
        '$t_1\\delta_\\mathrm{NL}+t_2\\delta_\\mathrm{NL}^2+t_3\\mathcal{G}_2[\\delta_\\mathrm{NL}]$',
        'hat_delta_h_from_deltaZ_deltalin2_deltalinG2':
        '$t_1\\delta_Z+t_2\\delta_1^2+t_3\\mathcal{G}_2[\\delta_1]$',
        'hat_delta_h_from_deltalin_deltaZ_deltalin2G2_deltaZ2G2':
        '$t_1\\delta_1+\\tilde t_1\\delta_Z+t_2\\delta_1^2+\\tilde t_2\\delta_Z^2 + t_3\\mathcal{G}_2[\\delta_1]+\\tilde t_3\\mathcal{G}_2[\\delta_Z]$',
        'delta_h': '$\\delta_h$',
        'deltalin': '$\\delta_1$',
        'deltalin_growth': '$\\delta_1^2$',
        'deltalin_tidal_G2': '$\\mathcal{G}_2[\\delta_1]$',
        'deltalin_F2': '$F_2[\\delta_1]$',
        'hat_delta_h_from_1_Tdeltalin2G2_SHIFTEDBY_PsiZ':
        'Quadr. bias',  #'$[1+b_1^L(k)\\delta_0+b_2^L(k)(\\delta_0^2-\\langle\\delta_0^2\\rangle)+b^L_{\\mathcal{G}_2}\\mathcal{G}_2](\\mathbf{q}+\psi_Z)$',
        'hat_delta_h_from_1_Tdeltalin2G2_SHIFTEDBY_Psi2LPT':
        '$[1+b_1^L(k)\\delta_0+b_2^L(k)(\\delta_0^2-\\langle\\delta_0^2\\rangle)+b^L_{\\mathcal{G}_2}\\mathcal{G}_2](\\mathbf{q}+\psi_\\mathrm{2LPT})$',
        'hat_delta_h_from_Tdeltalin2G2_SHIFTEDBY_PsiZ':
        'Quadr. bias',  #'$[1+b_1^L(k)\\delta_0+b_2^L(k)(\\delta_0^2-\\langle\\delta_0^2\\rangle)+b^L_{\\mathcal{G}_2}\\mathcal{G}_2](\\mathbf{q}+\psi_Z)$',
        'hat_delta_h_from_Tdeltalin2G2_SHIFTEDBY_Psi2LPT':
        '$[b_1^L(k)\\delta_0+b_2^L(k)(\\delta_0^2-\\langle\\delta_0^2\\rangle)+b^L_{\\mathcal{G}_2}\\mathcal{G}_2](\\mathbf{q}+\psi_\\mathrm{2LPT})$',
        'hat_delta_h_from_b1_delta_m':
        'Linear Std. Eul. bias',  # $b_1(k)\\delta_\\mathrm{NL}$',
        'hat_delta_h_from_1_Tdeltalin_SHIFTEDBY_PsiZ':
        'Linear bias',  # '$[1+b_1^L(k)\\delta_0](\\mathbf{q}+\\psi_Z)$'
        'delta_h_WEIGHT_M1': '$\\delta_{M_h}$',
        'hat_delta_h_from_1_Tdeltalin2G23_SHIFTEDBY_PsiZ': 'Cubic bias',
        'hat_delta_h_from_Tdeltalin2G23_SHIFTEDBY_PsiZ': 'Cubic bias',
        'hat_delta_h_M0M1_noZ_from_T_1deltalin12G23_SHIFTEDBY_PsiZ':
        'Cubic bias',
        'hat_delta_h_M1M0_noZ_MPnn_from_1_T_deltalin12G23_SHIFTEDBY_PsiZ':
        'Cubic bias',
        'hat_delta_h_from_deltanonl_12G2':
        '$t_1\\delta_\\mathrm{NL}+t_2\\delta_\\mathrm{NL}^2+t_{\\mathcal{G}_2}\\mathcal{G}_2[\\delta_\\mathrm{NL}]$',
        'hat_delta_h_from_delta_m_deltalin2_deltalinG2':
        '$t_1\\delta_\\mathrm{NL}+t_2\\delta_1^2+t_{\\mathcal{G}_2}\\mathcal{G}_2[\\delta_1]$',
        'hat_delta_h_from_Tdeltalin_SHIFTEDBY_PsiZ': '$t_1\\tilde\\delta_1$'
    }

    if explicit:
        texlabels['hat_delta_h_from_b1_delta_m'] = '$t_1\\delta_\\mathrm{NL}$'
        texlabels[
            'hat_delta_h_from_Tdeltalin2G2_SHIFTEDBY_PsiZ'] = '$\\beta_1\\tilde\\delta_1+\\beta_2\\tilde\\delta_2+\\beta_{\\mathcal{G}_2}\\tilde{\\mathcal{G}}_2$'

    if label in texlabels:
        if not include_data_label:
            return texlabels[label]
        else:
            if label.startswith('hat_delta_h_from'):
                outlabel = 'data$=\\delta_n$, model = '
            elif label.startswith('hat_delta_h_M0M1_noZ_from'):
                outlabel = 'data$=\\alpha_0\\delta_n+\\alpha_1\\delta_M$, model = '
            elif label.startswith('hat_delta_h_M1M0_noZ_MPnn_from'):
                outlabel = 'data$=\\alpha_0\\delta_M+\\alpha_1\\delta_n^\\perp$, model = '
            else:
                outlabel = 'data=?, model = '
            return outlabel + texlabels[label]
    else:
        return label


def get_ax_size(fig, ax):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height


if __name__ == '__main__':
    sys.exit(main())
