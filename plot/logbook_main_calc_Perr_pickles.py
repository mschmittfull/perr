"""
Log book of pickles saved to disk
"""

from __future__ import print_function, division


def get_pickle_fname():

    plot_opts = {}


    # WHICH PICKLE TO LOAD

    # Run A1: Quadratic bias model
    # M=12.8-13.8, Ngrid=64, seeds 400,403
    fname = 'main_calc_Perr_2019_May_09_22:55:21_time1557442521.dill'

    # ENTER YOUR PICKLE FILES BELOW, specifying the fname and if needed plot_opts.
    # Comment out the line fname=... above. Always make sure that all but one line
    # with fname are commented out, to avoid confusion.

    # Some description
    # fname = 'my_pickle_file'; plot_opts.update({})

    loginfo = {'fname': fname, 'plot_opts': plot_opts}

    return loginfo
