from __future__ import print_function, division


def get_densities_needed_for_trf_fcns(trf_specs):
    """Get list of all densities actually needed for trf fcns.

    Parameters
    ----------
    trf_specs : list of ``TrfSpec`` objects.
        Transfer functions specs for all models to be used.

    Returns
    -------
    densities_needed_for_trf_fcns : list of strings
        Names of fields needed for the transfer functions.
    """
    densities_needed_for_trf_fcns = []
    for trf_spec in trf_specs:
        for linsource in trf_spec.linear_sources:
            if linsource not in densities_needed_for_trf_fcns:
                densities_needed_for_trf_fcns.append(linsource)
        #densities_needed_for_trf_fcns += trf_spec.linear_sources
        for fixedlinsource in getattr(trf_spec, 'fixed_linear_sources', []):
            if fixedlinsource not in densities_needed_for_trf_fcns:
                densities_needed_for_trf_fcns.append(fixedlinsource)

        if trf_spec.field_to_smoothen_and_square is not None:
            if trf_spec.field_to_smoothen_and_square not in densities_needed_for_trf_fcns:
                densities_needed_for_trf_fcns.append(
                    trf_spec.field_to_smoothen_and_square)
        if trf_spec.field_to_smoothen_and_square2 is not None:
            if trf_spec.field_to_smoothen_and_square2 not in densities_needed_for_trf_fcns:
                densities_needed_for_trf_fcns.append(
                    trf_spec.field_to_smoothen_and_square2)
        if trf_spec.target_field not in densities_needed_for_trf_fcns:
            densities_needed_for_trf_fcns.append(trf_spec.target_field)
        if hasattr(trf_spec, 'target_spec'):
            for tc in getattr(trf_spec.target_spec, 'linear_target_contris',
                              []):
                if tc not in densities_needed_for_trf_fcns:
                    densities_needed_for_trf_fcns.append(tc)
    return densities_needed_for_trf_fcns

def make_cache_path(cache_base_path, comm):
    """
    Unique id for cached files so we can run multiple instances simultaneously.
    Rank 0 gets the cache id and then broadcasts it to other ranks
    """
    cacheid = None
    if comm.rank == 0:
        file_exists = True
        while file_exists:
            cacheid = ('CACHE%06x' % random.randrange(16**6)).upper()
            cache_path = os.path.join(cache_base_path, cacheid)
            file_exists = (len(glob.glob(cache_path)) > 0)
        # create cache path
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
    # broadcast cacheid to all ranks
    cacheid = comm.bcast(cacheid, root=0)
    # get cache path on all ranks
    cache_path = os.path.join(cache_base_path, cacheid)
    return cache_path
