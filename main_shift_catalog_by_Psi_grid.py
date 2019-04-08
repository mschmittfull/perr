#!/usr/bin/env python
# #!/home/mschmittfull/anaconda2/envs/nbodykit-0.3-env/bin/python

# NOTE: We load nbodykit-0.3 environment above, so can call with ./main_test_nbkit0.3py.
# Better: Use run.sh script.

# Should make sure that PYTHONPATH="".

# Run with
#   ./main_test_nbkit0.3.py
# or
#   ./run.sh main_test_nbkit0.3py
# but NOT with
#   python main_test_nbkit0.3.py


from __future__ import print_function,division

from nbodykit.lab import *
import numpy as np
import os
from shutil import rmtree
from argparse import ArgumentParser
from collections import OrderedDict
from pmesh.pm import ParticleMesh


# MS code
from lsstools import nbkit03_utils

def main():

    """
    5 April 2018: Not finished, b/c would need a parallel version of RegularGridInterpolator.
    Actually finished later?

    Do the following: 
    1) Read particle positions of a catalog
    2) Interpolate displacement Psi defined on a grid to particle positions
    3) Shift (displace) particles by Psi
    4) Save shifted catalog

    Example:
    ./run.sh mpiexec -n 3 python main_ms_gadget_shift_catalog_by_Psi_grid.py 64 64
    """
    # TODO (already done?): 
    # - create uniform catalog
    # - load deltalin on grid
    # - weigh ptcles in uniform catalog by 1+delta_lin
    # - compute smoothed Psi_lin on grid
    # - call shift_catalog to displace uniform catalog 

    opts = OrderedDict()


    # command line args
    ap = ArgumentParser()
    ap.add_argument('Nptcles_per_dim', type=int, default=0, help='Number of particles per dimension used in internal catalog')
    ap.add_argument('out_Ngrid', type=int, default=0, help='Ngrid for output density')
    #ap.add_argument('in_fname_density_to_shift', default='', help='Name of file containing input density to be shifted by Psi')
    #ap.add_argument('in_fname_Psi_source', default='', help='Name of file containing density from which we calculate Psi')
    #ap.add_argument('boxsize', type=float, default=0.0, help='Boxsize in Mpc/h')
    #ap.add_argument('plot_slices', type=boolean, default=False, help='Plot slices')

    # optional arguments
    ap.add_argument('--Nmesh', type=int, default=0, 
                    help='Internal Nmesh used to compute density to shifted and displacement potential. Should be >= Nptcles_per_dim.')
    ap.add_argument('--SimSeed', type=int, default=403, help='Simulation seed to load.')
    ap.add_argument('--PsiOrder', type=int, default=1,
                        help="Order of displacement field (1 for Zeldovich, 2 for 2LPT displacement).")
    ap.add_argument('-v', '--verbose', action="store_const", dest="verbose", 
                        const=1, default=0,
                        help="run in 'verbose' mode, with increased logging output")

    
    # copy args
    cmd_args = ap.parse_args()
    opts['Nptcles_per_dim'] = cmd_args.Nptcles_per_dim
    opts['out_Ngrid'] = cmd_args.out_Ngrid
    #opts['plot_slices'] = cmd_args.plot_slices
    opts['sim_seed'] = cmd_args.SimSeed
    opts['PsiOrder'] = cmd_args.PsiOrder
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
    
    # ms_gadget L=500 sim
    opts['sim_name'] = 'ms_gadget'
    opts['boxsize'] = 500.0
    opts['basepath'] = os.path.expandvars('$SCRATCH/lss/ms_gadget/run4/00000%d-01536-%.1f-wig/' % (
        opts['sim_seed'], opts['boxsize']))
    
    # Get deltalin at internal_scale_factor_for_weights, shift it by Psi(out_scale_factor),
    # and then rescale the result from internal_scale_factor_for_weights to out_scale_factor.
    # (We must use Psi at out_scale_factor to get correct displacement, which is larger at lower z.)
    opts['out_scale_factor'] = 0.6250 
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
    if False:
        # shift delta_lin
        opts['densities_to_shift'].append({
            'id_for_out_fname': 'IC_LinearMesh',
            'in_fname': os.path.join(
                opts['basepath'], 'IC_LinearMesh_z0_Ng%d/' % opts['Nmesh']),
            'file_scale_factor': 1.0,
            'external_smoothing': None
            })
    if False:
        # shift delta_lin^2-<delta_lin^2>
        opts['densities_to_shift'].append({
            'id_for_out_fname': 'IC_LinearMesh_growth-mean',
            'in_fname': os.path.join(
                opts['basepath'], 'IC_LinearMesh_z0_Ng%d/' % opts['Nmesh']),
            'file_scale_factor': 1.0,
            'external_smoothing': None, # external smoothingn of delta^2(x)
            'smoothing_quadratic_source': {'mode': 'Gaussian', 'R': 0.0},
            'calc_quadratic_field': 'growth-mean'
            })
    if False:
        # shift G2[delta_lin]
        opts['densities_to_shift'].append({
            'id_for_out_fname': 'IC_LinearMesh_tidal_G2',
            'in_fname': os.path.join(
                opts['basepath'], 'IC_LinearMesh_z0_Ng%d/' % opts['Nmesh']),
            'file_scale_factor': 1.0,
            'external_smoothing': None, # external smoothingn of delta^2(x)
            'smoothing_quadratic_source': {'mode': 'Gaussian', 'R': 0.0}, # 'kmax': opts['kmax']},
            'calc_quadratic_field': 'tidal_G2'
            })

    if True:
        # shift the shift term Psi.nabla delta
        opts['densities_to_shift'].append({
            'id_for_out_fname': 'IC_LinearMesh_PsiNablaDelta',
            'in_fname': os.path.join(
                opts['basepath'], 'IC_LinearMesh_z0_Ng%d/' % opts['Nmesh']),
            'file_scale_factor': 1.0,
            'external_smoothing': None, # external smoothingn of delta^2(x)
            'smoothing_quadratic_source': {'mode': 'Gaussian', 'R': 0.0}, # 'kmax': opts['kmax']},
            'calc_quadratic_field': 'PsiNablaDelta'
            })

    if False:
        # shift the field 1 (this gives delta_ZA)
        opts['densities_to_shift'].append({
            'id_for_out_fname': '1',
            'in_fname': os.path.join(
                opts['basepath'], 'IC_LinearMesh_z0_Ng%d/' % opts['Nmesh']),
            'file_scale_factor': 1.0,
            'external_smoothing': None,
            'calc_trf_of_field': '1'
            })
    if False:
        # shift deltalin^3
        opts['densities_to_shift'].append({
            'id_for_out_fname': 'IC_LinearMesh_cube-mean',
            'in_fname': os.path.join(
                opts['basepath'], 'IC_LinearMesh_z0_Ng%d/' % opts['Nmesh']),
            'file_scale_factor': 1.0,
            'external_smoothing': None, # external smoothingn of delta^2(x)
            #'smoothing_quadratic_source': {'mode': 'Gaussian', 'R': 0.0},
            'smoothing_quadratic_source': {'mode': 'Gaussian', 'R': 0.0, 'kmax': 0.5},
            'calc_quadratic_field': 'cube-mean'
            })

    if False:
        # shift the field delta_short = (1-W_R)delta
        # todo: implement
        opts['densities_to_shift'].append({
            'id_for_out_fname': 'IC_LinearMesh_short',
            'in_fname': os.path.join(
                opts['basepath'], 'IC_LinearMesh_z0_Ng%d/' % opts['Nmesh']),
            'file_scale_factor': 1.0,
            'external_smoothing': {'mode': '1-Gaussian', 'R': 10.0}, 
            })


    ## Displacement field
    # Note: MP-Gadget libgenic/zeldovich.c applies smoothing by a gaussian kernel of 1 mesh grid
    # (i.e. R=boxsize/Nmesh*sqrt(2) b/c they use Gaussian w/o 0.5 in exponent), and they might do read-out
    # of displacements to ptcle positions using nearest neighbor instead of CIC we use. For L=500,
    # Np=1536, get Nmesh=3072 and thus R=500/3072*1.42==0.23
    if opts['PsiOrder'] == 1:
        # displace by Psi_ZA[delta_lin] = ik/k^2 delta_lin
        opts['displacement_source'] = {
            'id_for_out_fname': 'IC_LinearMesh',
            'Psi_type': 'Zeldovich',
            'in_fname': os.path.join(
                opts['basepath'], 'IC_LinearMesh_z0_Ng%d/' % opts['Nmesh']),
            'file_scale_factor': 1.0,
            'smoothing': {'mode': 'Gaussian', 'R': 0.23}
            }
    elif opts['PsiOrder'] == 2:
        # displace by Psi_2LPT[delta_lin] = ik/k^2 delta_lin(k) - 3/14 ik/k^2 G2[delta_lin](k)
        opts['displacement_source'] = {
            'id_for_out_fname': 'Psi2LPT_IC_LinearMesh',
            'Psi_type': '2LPT',
            'in_fname': os.path.join(
                opts['basepath'], 'IC_LinearMesh_z0_Ng%d/' % opts['Nmesh']),
            'file_scale_factor': 1.0,
            'smoothing': {'mode': 'Gaussian', 'R': 0.23}
            }
    # elif opts['PsiOrder'] == -2:
    #     # just to check sign of Psi^[2] is correct
    #     # displace by -Psi_2LPT[delta_lin] = ik/k^2 delta_lin(k) + 3/14 ik/k^2 G2[delta_lin](k)
    #     opts['displacement_source'] = {
    #         'id_for_out_fname': '-Psi2LPT_IC_LinearMesh',
    #         'Psi_type': '-2LPT',
    #         'in_fname': os.path.join(
    #             opts['basepath'], 'IC_LinearMesh_z0_Ng%d/' % opts['Nmesh']),
    #         'file_scale_factor': 1.0,
    #         'smoothing': {'mode': 'Gaussian', 'R': 0.23}
    #         }
    else:
        raise Exception("Invalid PsiOrder %s" % str(opts['PsiOrder']))
        

    ## COSMOLOGY OPTIONS
    if opts['sim_name'] in ['jerryou_baoshift','ms_gadget']:
        opts['cosmo_params'] = dict(Om_m=0.307494, Om_L=1.0-0.307494, Om_K=0.0,
                                    Om_r=0.0, h0=0.6774)




    # ################################################################################
    # START PROGRAM
    # ################################################################################
    
    # init MPI
    from nbodykit import CurrentMPIComm
    comm = CurrentMPIComm.get()
    print("%d: Greetings from rank %d" % (comm.rank, comm.rank))
    


    # loop over all densities to be shifted
    for specs_of_density_to_shift in opts['densities_to_shift']:
        


        # ################################################################################
        # Generate uniform catalog with Nptcles_per_dim^3 particles on regular grid
        # ################################################################################


        if True:
            # use pmesh generate_uniform_particle_grid
            # http://rainwoodman.github.io/pmesh/pmesh.pm.html?highlight=readout#pmesh.pm.ParticleMesh.generate_uniform_particle_grid
            pmesh = ParticleMesh(BoxSize=opts['boxsize'],
                                 Nmesh=[opts['Nptcles_per_dim'], opts['Nptcles_per_dim'], opts['Nptcles_per_dim']])
            ptcles = pmesh.generate_uniform_particle_grid(shift=0.0, dtype='f8')
            #print("type ptcles", type(ptcles), ptcles.shape)
            #print("head ptcles:", ptcles[:5,:])


            dtype = np.dtype( [ ('Position',('f8',3))] )

            # number of rows is given by number of ptcles on this rank
            uni_cat_array = np.empty( (ptcles.shape[0],), dtype=dtype)
            uni_cat_array['Position'] = ptcles

            uni_cat = ArrayCatalog(uni_cat_array, comm=None, 
                                   BoxSize=opts['boxsize']*np.ones(3),
                                   Nmesh=[opts['Nptcles_per_dim'], opts['Nptcles_per_dim'], opts['Nptcles_per_dim']])

            print("%d: local Nptcles=%d, global Nptcles=%d" % (comm.rank, uni_cat.size, uni_cat.csize))

            del ptcles
            del uni_cat_array

        else:

            # MS: own silly serial code to generate regular grid
            if comm.rank == 0:
                # Code copied from do_rec_v1.py and adopted

                # Note that nbkit UniformCatalog is random catalog, but we want a catalog
                # where each ptcle sits at grid points of a regular grid.
                # This is what we call 'regular uniform' catalog.
                Np = opts['Nptcles_per_dim']
                dtype = np.dtype( [ ('Position',('f8',3))] )
                # Have Np**3 particles, and each particle has position x,y,z and weight 'Weight'
                uni_cat_array = np.empty( (Np**3,), dtype=dtype)


                # x components in units such that box ranges from 0 to 1. Note dx=1/Np.
                #x_components_1d = np.linspace(0.0, (Np-1)*(L/float(Np)), num=Np, endpoint=True)/L
                x_components_1d = np.linspace(0.0, (Np-1)/float(Np), num=Np, endpoint=True)
                ones_1d = np.ones(x_components_1d.shape)

                # Put particles on the regular grid
                print("%d: Fill regular uniform catalog" % comm.rank)
                uni_cat_array['Position'][:,0] = np.einsum('a,b,c->abc', x_components_1d, ones_1d, ones_1d).reshape((Np**3,))
                uni_cat_array['Position'][:,1] = np.einsum('a,b,c->abc', ones_1d, x_components_1d, ones_1d).reshape((Np**3,))
                uni_cat_array['Position'][:,2] = np.einsum('a,b,c->abc', ones_1d, ones_1d, x_components_1d).reshape((Np**3,))
                print("%d: Done filling regular uniform catalog" % comm.rank)

                # in nbkit0.3 units must be in Mpc/h
                uni_cat_array['Position'] *= opts['boxsize']

            else:
                uni_cat_array = None

            # Scatter across all ranks
            print("%d: Scatter array" % comm.rank)
            from nbodykit.utils import ScatterArray
            uni_cat_array = ScatterArray(uni_cat_array, comm, root=0, counts=None)
            print("%d: Scatter array done. Shape: %s" % (comm.rank,str(uni_cat_array.shape)))

            # Save in ArrayCatalog object
            uni_cat = ArrayCatalog(uni_cat_array)
            uni_cat.attrs['BoxSize']  = np.ones(3) * opts['boxsize']
            uni_cat.attrs['Nmesh']  = np.ones(3) * opts['Nptcles_per_dim']
            uni_cat.attrs['Nmesh_internal']  = np.ones(3) * opts['Nmesh']


        # ################################################################################
        # Load density to be shifted and weigh particles by delta
        # ################################################################################

        # todo: could create linearmesh on the fly rather than reading from disk. but also fine 
        # to load from disk for now.

        # get field from bigmesh file; rescale to delta at internal_scale_factor_for_weights.
        # note that we get delta, not 1+delta.
        rfield_density_to_shift = nbkit03_utils.get_rfield_from_bigfilemesh_file(
            specs_of_density_to_shift['in_fname'], 
            file_scale_factor=specs_of_density_to_shift['file_scale_factor'],
            desired_scale_factor=opts['internal_scale_factor_for_weights'],
            cosmo_params=opts['cosmo_params'])

        nbkit03_utils.rfield_print_info(rfield_density_to_shift, comm, 'rfield_density_to_shift: ')

        # compute quadratic field if desired
        if specs_of_density_to_shift.get('calc_quadratic_field',None) is not None:
            # get quadratic field
            rfield_density_to_shift = nbkit03_utils.calc_quadratic_field(
                base_field_mesh=FieldMesh(rfield_density_to_shift),
                quadfield=specs_of_density_to_shift['calc_quadratic_field'],
                smoothing_of_base_field=specs_of_density_to_shift.get('smoothing_quadratic_source',None),
                verbose=opts['verbose']).compute(mode='real')

            # print info
            nbkit03_utils.rfield_print_info(rfield_density_to_shift, comm, 
                                            '%s: '%specs_of_density_to_shift['calc_quadratic_field'])

        if specs_of_density_to_shift.get('calc_trf_of_field', None) is not None:
            # calculate some transformation of the field
            if specs_of_density_to_shift['calc_trf_of_field'] == '1':
                # replace field by 1
                rfield_density_to_shift = 0*rfield_density_to_shift + 1.0
            else:
                raise Exception("Invalid calc_trf_of_field %s" % 
                                str(specs_of_density_to_shift['calc_trf_of_field']))
            
        if specs_of_density_to_shift['external_smoothing'] is not None:
            raise Exception("todo3")

        # print("%d: rfield_density_to_shift: type=%s, shape=%s, rms=%g"% (
        #     comm.rank, str(type(rfield_density_to_shift)), str(rfield_density_to_shift.shape),
        #     np.sqrt(np.mean(rfield_density_to_shift**2))))

        # Set weight of particles in uni_cat to delta (interpolated to ptcle positions)
        nbkit03_utils.interpolate_pm_rfield_to_catalog(
            rfield_density_to_shift, uni_cat, catalog_column_to_save_to='Mass')
        #del rfield_density_to_shift
        print("%d: rms Mass: %g" % (comm.rank, np.sqrt(np.mean(np.array(uni_cat['Mass'])**2))))
        #uni_cat['Mass'] *= 0.0

        if opts['plot_slices']:
            # paint the original uni_cat to a grid and plot slice
            import matplotlib.pyplot as plt

            tmp_meshsource = uni_cat.to_mesh(Nmesh=opts['out_Ngrid'],
                                             value='Mass',
                                             window='cic', compensated=False, interlaced=False)
            # paint to get delta(a_internal)
            tmp_outfield = tmp_meshsource.paint(mode='real')
            # linear rescale factor from internal_scale_factor_for_weights to out_scale_factor
            rescalefac = nbkit03_utils.linear_rescale_fac(
                opts['internal_scale_factor_for_weights'], opts['out_scale_factor'],
                cosmo_params=opts['cosmo_params'])
            tmp_outfield = 1.0 + rescalefac * (tmp_outfield - 1.0)       
            tmp_mesh = FieldMesh(tmp_outfield)
            plt.imshow(tmp_mesh.preview(Nmesh=32, axes=(0,1)))
            if comm.rank==0:
                plt_fname = 'inmesh_Np%d_Nm%d_Ng%d.pdf' % (opts['Nptcles_per_dim'], opts['Nmesh'], opts['out_Ngrid'])
                plt.savefig(plt_fname)
                print("Made %s" % plt_fname)
            del tmp_meshsource, rescalefac, tmp_outfield, tmp_mesh


        # ##################################################################################
        # Compute displacement Psi on grid, interpolate to particle positions, and shift them.
        # ##################################################################################

        # get displacement source field from bigmesh file; scale to out_scale_factor
        # to make sure we use full displacement up to low z of output.
        rfield_displacement_source = nbkit03_utils.get_rfield_from_bigfilemesh_file(
            opts['displacement_source']['in_fname'], 
            file_scale_factor=opts['displacement_source']['file_scale_factor'],
            desired_scale_factor=opts['out_scale_factor'],
            cosmo_params=opts['cosmo_params'])

        Psi_rfields = [None, None, None]
        for direction in range(3):
            # Get Psi_i = k_i delta_smoothed/k^2 following http://rainwoodman.github.io/pmesh/intro.html
            if comm.rank==0:
                print("%d: Get Psi_%d: " %(comm.rank,direction))

            # TMP: MULTIPLY BY 0 to get 0 displacement
            Psi_rfields[direction] = 1.0 * nbkit03_utils.get_displacement_from_density_rfield(
                rfield_displacement_source, component=direction, 
                Psi_type=opts['displacement_source']['Psi_type'],
                smoothing=opts['displacement_source']['smoothing'])

        # Shift uniform catalog particles by Psi (changes uni_cat)
        nbkit03_utils.shift_catalog_by_psi_grid(
            cat=uni_cat,  in_displacement_rfields=Psi_rfields,
            pos_column='Position',
            pos_units='Mpc/h', displacement_units='Mpc/h', boxsize=opts['boxsize'],
            verbose=opts['verbose'])
        #del Psi_rfields



        # ################################################################################
        # paint shifted catalog to grid, using field_to_shift as weights
        # ################################################################################

        print("%d: paint shifted catalog to grid using mass weights" % comm.rank)
        out_delta, meshsource_attrs = nbkit03_utils.weighted_paint_cat_to_delta(
            uni_cat, weight='Mass', weighted_CIC_mode=opts['weighted_CIC_mode'],
            set_mean=0.0,
            Nmesh=opts['out_Ngrid'], verbose=opts['verbose'],
            to_mesh_kwargs={'window': 'cic', 'compensated': False, 'interlaced': False})


        # ################################################################################
        # rescale to output redshift
        # ################################################################################

        # linear rescale factor from internal_scale_factor_for_weights to out_scale_factor
        rescalefac = nbkit03_utils.linear_rescale_fac(
            opts['internal_scale_factor_for_weights'], opts['out_scale_factor'],
            cosmo_params=opts['cosmo_params'])

        out_delta *= rescalefac

        # print some info:
        if comm.rank == 0:
            print("%d: Linear rescalefac from a=%g to a=%g, rescalefac=%g" % (
                comm.rank, opts['internal_scale_factor_for_weights'], opts['out_scale_factor'],
                rescalefac))

        if opts['verbose']:
            print("%d: out_delta: min, mean, max, rms(x-1):"%comm.rank, 
                  np.min(out_delta), np.mean(out_delta), np.max(out_delta), 
                  np.mean((out_delta-1.)**2)**0.5)

        # get 1+deta mesh from field
        outmesh = FieldMesh(1+out_delta)


        # print some info: this makes code never finish (race condition maybe?)
        #nbkit03_utils.rfield_print_info(outfield, comm, 'outfield: ')

        # copy MeshSource attrs
        for k,v in meshsource_attrs.items():
            outmesh.attrs['MeshSource_%s'%k] = v
        if comm.rank==0:
            print("outmesh.attrs:\n", outmesh.attrs)

        # save to bigfile
        out_fname = os.path.join(opts['basepath'], '%s_int%s_ext%s_SHIFTEDBY_%s%s_a%.4f_Np%d_Nm%d_Ng%d_CIC%s' % (
            specs_of_density_to_shift['id_for_out_fname'],
            smoothing_str(specs_of_density_to_shift.get('smoothing_quadratic_source',None)),
            smoothing_str(specs_of_density_to_shift['external_smoothing']),
            opts['displacement_source']['id_for_out_fname'],
            smoothing_str(opts['displacement_source']['smoothing']),
            opts['out_scale_factor'],
            opts['Nptcles_per_dim'],
            opts['Nmesh'],
            opts['out_Ngrid'],
            opts['weighted_CIC_mode']
            ))

        if comm.rank==0:
            print("Writing to %s" % out_fname)
        outmesh.save(out_fname, mode='real')
        if comm.rank==0:
            print("Wrote %s" % out_fname)

        # must call cmean collectively
        out_cmean = out_delta.cmean()
        if comm.rank==0:
            print("%d: out_delta cmean:" % comm.rank, out_cmean)

        # try to plot slice
        if opts['plot_slices']:
            plt.imshow(outmesh.preview(Nmesh=32, axes=(0,1)))
            if comm.rank==0:
                plt_fname = 'outmesh_Np%d_Nm%d_Ng%d.pdf' % (
                    opts['Nptcles_per_dim'], opts['Nmesh'], opts['out_Ngrid'])
                plt.savefig(plt_fname)
                print("Made %s" % plt_fname)



def smoothing_str(smoothing_dict):
    if smoothing_dict is None:
        return 'R0.00'
    else:
        if smoothing_dict['mode'] == 'Gaussian':
            if smoothing_dict.get('kmax',0.0) == 0.0:
                return 'R%.2f'%smoothing_dict['R']
            else:
                return 'R%.2f_%.2f'%(smoothing_dict['R'],smoothing_dict['kmax'])
        else:
            raise Exception("invalid smoothing mode %s" % str(smoothing['mode']))


    
if __name__ == '__main__':
    main()
