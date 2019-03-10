#!/usr/bin/env python

"""
Log book of pickles saved to disk
"""


from __future__ import print_function,division

def get_pickle_fname():

    plot_opts = {}

    # id for mathematica file export
    math_id = None
    
    # WHICH PICKLE TO LOAD

    # Run A1: Drop deltaZ from quadratic and cubic Lagr. bias (can be absorbed by other sources)
    fname = 'main_calc_Perr_2019_Mar_10_20:49:30_time1552250970.pickle';  plot_opts.update({'ymin_1mr2': 0.00005, 'ymin_Tk': -1, 'ymax_Tk': 2, 'tryid': 'L32cNoZ', 'pickle_fname_DM_model_error': 'main_do_stage012_rec_2018_Jul_10_15:15:54_time1531235754.pickle'});  math_id='tryL32cNoZ_M10.8-11.8';  # M=10.8-11.8

    # ENTER YOUR PICKLE FILES BELOW, specifying the fname and if needed plot_opts. 
    # Comment out the line fname=... above. Always make sure that all but one line
    # with fname are commented out, to avoid confusion.

    # Some description  
    # fname = 'my_pickle_file'; plot_opts.update({})

    



    loginfo = {'fname': fname, 'plot_opts': plot_opts, 'math_id': math_id}
    
    return loginfo
