from __future__ import print_function, division
import os
import re


def get_in_path(opts):
    """
    Paths of different simulations. Add new entries for other simulations.
    """
    if opts['sim_name'] == 'ms_gadget':
        return '$SCRATCH/lss/ms_gadget/run%d/%08d-%05d-%.1f-%s/' % (
            opts['sim_irun'], opts['sim_seed'], opts['sim_Nptcles'],
            opts['sim_boxsize'], opts['sim_wig_now_string'])

    elif opts['sim_name'] == 'ms_gadget_test_data':
        return 'test_data/ms_gadget/run%d/%08d-%05d-%.1f-%s/' % (
            opts['sim_irun'], opts['sim_seed'], opts['sim_Nptcles'],
            opts['sim_boxsize'], opts['sim_wig_now_string'])

    elif opts['sim_name'] == 'yurecsims':
        return os.path.join('$SCRATCH/lss/yurecsims/', opts['sim_dir'])

    elif opts['sim_name'] == 'RunPB':
        return os.path.join('$DATA/lss/mwhite/RunPB/PB%02d/' % opts['sim_seed'])

    elif opts['sim_name'] == 'jerryou_baoshift':
        # MS 31/01/2018: Cannot recall why 309 is excluded here, removed it
        #if (opts['sim_boxsize']==1380.) and (opts['sim_Nptcles']==2048) and (opts['sim_seed']!=309):
        if (opts['sim_boxsize'] == 1380.) and (opts['sim_Nptcles'] == 2048):
            return '$SCRATCH/lss/jerryou/baoshift/run%d/%08d-%d-%s/' % (
                opts['sim_irun'], opts['sim_seed'], opts['sim_Ntimesteps'],
                opts['sim_wig_now_string'])
        else:
            return '$SCRATCH/lss/jerryou/baoshift/run%d/%08d-%d-%05d-%.1f-%s/' % (
                opts['sim_irun'], opts['sim_seed'], opts['sim_Ntimesteps'],
                opts['sim_Nptcles'], opts['sim_boxsize'],
                opts['sim_wig_now_string'])

    else:
        raise Exception("Unknown sim_name: %s" % str(opts['sim_name']))


def get_subdir_link_of_ssseed_file(path, fname):
    """
    Return name of link for ssseed file (subsample created with seed ssseed).

    Example:
      path=/data/mschmittfull/lss/mwhite/RunPB/PB09/
      fname=tpmsph_0.6452.ms_subsample0.01_ssseed10009_nbkfmt.hdf5 

    will return

      /data/mschmittfull/lss/mwhite/RunPB/PB09/subsamples_SSSEED100009/tpmsph_0.6452.ms_subsample0.01_ssseedX_nbkfmt.hdf5 
    """
    print("path:", path)
    print("fname:", fname)
    match = re.search(r'ssseed(\d*)', fname)
    ssseed = match.group(1)
    subdir = os.path.join(path, 'SSSEED' + ssseed)
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    new_fname = re.sub(r'ssseed(\d*)', r'ssseedX', fname)
    full_new_fname = os.path.join(subdir, new_fname)
    return full_new_fname
