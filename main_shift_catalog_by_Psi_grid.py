from __future__ import print_function, division
from argparse import ArgumentParser
from collections import OrderedDict
from nbodykit.lab import *
import numpy as np
import os

from shift import weigh_and_shift_uni_cats

def main():
    """
    Do the following: 
    - create uniform catalog
    - load deltalin or other density delta on grid
    - weigh ptcles in uniform catalog by 1+delta
    - compute smoothed Psi_lin on grid (interpolating to ptcle positions)
    - shift (displace) particles by Psi
    - save shifted catalog

    Example:
    ./run.sh mpiexec -n 3 python main_shift_catalog_by_Psi_grid.py 64 64

    Should make sure that PYTHONPATH="".
    And load environment with nbodykit 0.3, e.g. using run.sh script.
    """

    # command line args
    ap = ArgumentParser()
    ap.add_argument('Nptcles_per_dim',
                    type=int,
                    default=0,
                    help='Number of particles per dimension used in internal'
                    'catalog')
    ap.add_argument('out_Ngrid',
                    type=int,
                    default=0,
                    help='Ngrid for output density')

    # optional arguments
    ap.add_argument('--Nmesh',
                    type=int,
                    default=0,
                    help='Internal Nmesh used to compute density to be shifted '
                    ' and displacement. Should be >= Nptcles_per_dim.')

    ap.add_argument('--SimSeed',
                    type=int,
                    default=400,
                    help='Simulation seed to load.')

    ap.add_argument('--PsiOrder',
                    type=int,
                    default=1,
                    help='Order of displacement field (1 for Zeldovich, 2 for '
                         '2LPT displacement).')
    ap.add_argument('-v',
                    '--verbose',
                    action='store_const',
                    dest='verbose',
                    const=1,
                    default=0,
                    help='run in verbose mode, with increased logging output')
    ap.add_argument('--RSD',
                    type=int,
                    default=0,
                    help='0: No RSD, 1: Include RSD by displacing by Psi(q)+f '
                         '(\e_LOS.\vPsi(q)) \e_LOS, where e_LOS is unit vector '
                         'in line of sight direction.')

    # copy args
    cmd_args = ap.parse_args()
    opts = OrderedDict()
    opts['Nptcles_per_dim'] = cmd_args.Nptcles_per_dim
    opts['out_Ngrid'] = cmd_args.out_Ngrid
    #opts['plot_slices'] = cmd_args.plot_slices
    sim_seed = cmd_args.SimSeed
    opts['PsiOrder'] = cmd_args.PsiOrder
    opts['RSD'] = bool(cmd_args.RSD)
    opts['RSD_line_of_sight'] = [0, 0, 1]

    opts['verbose'] = cmd_args.verbose
    if cmd_args.Nmesh == 0:
        opts['Nmesh'] = opts['Nptcles_per_dim']
    else:
        opts['Nmesh'] = cmd_args.Nmesh

    # ################################################################################
    # OPTIONS (not from command line)
    # ################################################################################

    # plot slices
    opts['plot_slices'] = False
    if opts['Nptcles_per_dim'] >= 512:
        opts['plot_slices'] = False

    if True:
        # ms_gadget L=500 sim
        sim_name= 'ms_gadget'
        opts['boxsize'] = 500.0
        opts['basepath'] = os.path.expandvars(
            '$SCRATCH/lss/ms_gadget/run4/00000%d-01536-%.1f-wig/' %
            (sim_seed, opts['boxsize']))

        # Get deltalin at internal_scale_factor_for_weights, shift it by Psi(out_scale_factor),
        # and then rescale the result from internal_scale_factor_for_weights to out_scale_factor.
        # (We must use Psi at out_scale_factor to get correct displacement, which is larger at lower z.)
        opts['out_scale_factor'] = 0.6250
        deltalin_file_name = os.path.join(opts['basepath'],
                         'IC_LinearMesh_z0_Ng%d/' % opts['Nmesh'])
        deltalin_file_scale_factor = 1.0


    if False:
        # IllustrisTNG L=205 sim
        sim_name = 'IllustrisTNG_L205n2500TNG'
        opts['boxsize'] = 205.0
        opts['basepath'] = os.path.expandvars(
            '$DATA/lss/IllustrisTNG/L%dn2500TNG/output/' % int(opts['boxsize']))
        #opts['out_scale_factor'] = 0.66531496 # snap 67
        opts['out_scale_factor'] = 0.33310814 # snap 33
        # deltalin
        deltalin_file_name = os.path.join(opts['basepath'],
                         'snap_ics.hdf5_PtcleDensity_z127_Ng%d/' % opts['Nmesh'])
        deltalin_file_scale_factor = 1.0/(1.0+127.0)


    # Internal scale factor used for 1+delta weights. Initially used high z for that, but it should
    # be same as out_scale_factor b/c otherwise weights will be too small to matter (would get
    # almost the same as when shifting from uniform grid without weights).
    opts['internal_scale_factor_for_weights'] = opts['out_scale_factor']

    # Mode for painting shifted particles (with masses given by density) to grid
    # - 'avg': average contributions (divide by number of CIC contributions)
    # - 'sum': sum up contributions (like Zeldovich; increases density if ptcles move to same location;
    #          accounts for volume factor along the flow/Jacobian enforcing mass conservation)
    opts['weighted_CIC_mode'] = 'sum'

    # kmax for internal grid (Nptcles_per_dim grid points per dim)
    #opts['kmax'] = 2.0*np.pi/opts['boxsize'] * float(opts['Nptcles_per_dim'])/2.0

    ## densities to shift
    opts['densities_to_shift'] = []

    if True:
        # shift delta_lin
        opts['densities_to_shift'].append({
            'id_for_out_fname':
            'IC_LinearMesh',
            'in_fname': deltalin_file_name,
            'file_scale_factor': deltalin_file_scale_factor,
            'external_smoothing':
            None
        })
    if True:
        # shift delta_lin^2-<delta_lin^2>
        opts['densities_to_shift'].append({
            'id_for_out_fname':
            'IC_LinearMesh_growth-mean',
            'in_fname': deltalin_file_name,
            'file_scale_factor': deltalin_file_scale_factor,
            'external_smoothing':
            None,  # external smoothingn of delta^2(x)
            'smoothing_quadratic_source': {
                'mode': 'Gaussian',
                'R': 0.0
            },
            'calc_quadratic_field':
            'growth-mean'
        })
    if True:
        # shift G2[delta_lin]
        opts['densities_to_shift'].append({
            'id_for_out_fname':
            'IC_LinearMesh_tidal_G2',
            'in_fname': deltalin_file_name,
            'file_scale_factor': deltalin_file_scale_factor,
            'external_smoothing':
            None,  # external smoothingn of delta^2(x)
            'smoothing_quadratic_source': {
                'mode': 'Gaussian',
                'R': 0.0
            },  # 'kmax': opts['kmax']},
            'calc_quadratic_field':
            'tidal_G2'
        })

    if True:
        # shift the shift term Psi.nabla delta
        opts['densities_to_shift'].append({
            'id_for_out_fname':
            'IC_LinearMesh_PsiNablaDelta',
            'in_fname': deltalin_file_name,
            'file_scale_factor': deltalin_file_scale_factor,
            'external_smoothing':
            None,  # external smoothingn of delta^2(x)
            'smoothing_quadratic_source': {
                'mode': 'Gaussian',
                'R': 0.0
            },  # 'kmax': opts['kmax']},
            'calc_quadratic_field':
            'PsiNablaDelta'
        })

    if True:
        # shift the field 1 (this gives delta_ZA)
        opts['densities_to_shift'].append({
            'id_for_out_fname':
            '1',
            'in_fname': deltalin_file_name,
            'file_scale_factor': deltalin_file_scale_factor,
            'external_smoothing':
            None,
            'calc_trf_of_field':
            '1'
        })
    if True:
        # shift deltalin^3
        opts['densities_to_shift'].append({
            'id_for_out_fname':
            'IC_LinearMesh_cube-mean',
            'in_fname': deltalin_file_name,
            'file_scale_factor': deltalin_file_scale_factor,
            'external_smoothing':
            None,  # external smoothingn of delta^2(x)
            #'smoothing_quadratic_source': {'mode': 'Gaussian', 'R': 0.0},
            'smoothing_quadratic_source': {
                'mode': 'Gaussian',
                'R': 0.0,
                'kmax': 0.5
            },
            'calc_quadratic_field':
            'cube-mean'
        })

    if False:
        # shift the field delta_short = (1-W_R)delta
        # todo: implement
        opts['densities_to_shift'].append({
            'id_for_out_fname':
            'IC_LinearMesh_short',
            'in_fname': deltalin_file_name,
            'file_scale_factor': deltalin_file_scale_factor,
            'external_smoothing': {
                'mode': '1-Gaussian',
                'R': 10.0
            },
        })


    if True:
        #### ONLY TEST CUBIC OPERATORS
        opts['densities_to_shift'] = []

        # TODO: maybe apply more aggressive smoothing to avoid unwanted Dirac
        # delta images.

        if True:
            # shift G2*delta
            opts['densities_to_shift'].append({
                'id_for_out_fname':
                'IC_LinearMesh_G2_delta',
                'in_fname': deltalin_file_name,
                'file_scale_factor': deltalin_file_scale_factor,
                'external_smoothing':
                None,  # external smoothingn of delta^2(x)
                #'smoothing_quadratic_source': {'mode': 'Gaussian', 'R': 0.0},
                'smoothing_quadratic_source': {
                    'mode': 'Gaussian',
                    'R': 0.0,
                    'kmax': 0.5
                },
                'calc_quadratic_field':
                'G2_delta'
            })


        if True:
            # shift G3
            opts['densities_to_shift'].append({
                'id_for_out_fname':
                'IC_LinearMesh_tidal_G3',
                'in_fname': deltalin_file_name,
                'file_scale_factor': deltalin_file_scale_factor,
                'external_smoothing':
                None,  # external smoothingn of delta^2(x)
                #'smoothing_quadratic_source': {'mode': 'Gaussian', 'R': 0.0},
                'smoothing_quadratic_source': {
                    'mode': 'Gaussian',
                    'R': 0.0,
                    'kmax': 0.5
                },
                'calc_quadratic_field':
                'tidal_G3'
            })

        if True:
            # shift Gamma3
            opts['densities_to_shift'].append({
                'id_for_out_fname':
                'IC_LinearMesh_Gamma3',
                'in_fname': deltalin_file_name,
                'file_scale_factor': deltalin_file_scale_factor,
                'external_smoothing':
                None,  # external smoothingn of delta^2(x)
                #'smoothing_quadratic_source': {'mode': 'Gaussian', 'R': 0.0},
                'smoothing_quadratic_source': {
                    'mode': 'Gaussian',
                    'R': 0.0,
                    'kmax': 0.5
                },
                'calc_quadratic_field':
                'Gamma3'
            })



    ## Displacement field
    # Note: MP-Gadget libgenic/zeldovich.c applies smoothing by a gaussian kernel of 1 mesh grid
    # (i.e. R=boxsize/Nmesh*sqrt(2) b/c they use Gaussian w/o 0.5 in exponent), and they might do read-out
    # of displacements to ptcle positions using nearest neighbor instead of CIC we use. For L=500,
    # Np=1536, get Nmesh=3072 and thus R=500/3072*1.42==0.23
    if opts['PsiOrder'] == 1:
        # displace by Psi_ZA[delta_lin] = ik/k^2 delta_lin
        opts['displacement_source'] = {
            'id_for_out_fname':
            'IC_LinearMesh',
            'Psi_type':
            'Zeldovich',
            'in_fname': deltalin_file_name,
            'file_scale_factor': deltalin_file_scale_factor,
            'smoothing': {
                'mode': 'Gaussian',
                'R': 0.23
            }
        }
    elif opts['PsiOrder'] == 2:
        # displace by Psi_2LPT[delta_lin] = ik/k^2 delta_lin(k) - 3/14 ik/k^2 
        #                                   G2[delta_lin](k)
        opts['displacement_source'] = {
            'id_for_out_fname':
            'Psi2LPT_IC_LinearMesh',
            'Psi_type':
            '2LPT',
            'in_fname': deltalin_file_name,
            'file_scale_factor': deltalin_file_scale_factor,
            'smoothing': {
                'mode': 'Gaussian',
                'R': 0.23
            }
        }
    # elif opts['PsiOrder'] == -2:
    #     # just to check sign of Psi^[2] is correct
    #     # displace by -Psi_2LPT[delta_lin] = ik/k^2 delta_lin(k) + 3/14 ik/k^2
    #                                          G2[delta_lin](k)
    #     opts['displacement_source'] = {
    #         'id_for_out_fname': '-Psi2LPT_IC_LinearMesh',
    #         'Psi_type': '-2LPT',
    #        'in_fname': deltalin_file_name,
    #        'file_scale_factor': deltalin_file_scale_factor,
    #         'smoothing': {'mode': 'Gaussian', 'R': 0.23}
    #         }
    else:
        raise Exception("Invalid PsiOrder %s" % str(opts['PsiOrder']))

    # COSMOLOGY OPTIONS
    if sim_name in ['jerryou_baoshift', 'ms_gadget']:
        opts['cosmo_params'] = dict(Om_m=0.307494,
                                    Om_L=1.0 - 0.307494,
                                    Om_K=0.0,
                                    Om_r=0.0,
                                    h0=0.6774)

    elif sim_name in ['IllustrisTNG_L205n2500TNG']:
        opts['cosmo_params'] = dict(Om_m=0.3089,
                                    Om_L=0.6911,
                                    Om_K=0.0,
                                    Om_r=0.0,
                                    h0=0.6774)

    # save result to bigfile on disk
    opts['save_result'] = True

    opts['Nmesh_orig'] = opts['Nmesh']
    del opts['Nmesh']

    # ##########################################################################
    # START PROGRAM
    # ##########################################################################

    outmesh = weigh_and_shift_uni_cats(**opts)


if __name__ == '__main__':
    main()
