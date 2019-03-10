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

    # L32cNoZ: Drop deltaZ from quadratic and cubic Lagr. bias (can be absorbed by other sources)
    fname = 'main_quick_calc_Pk_2018_Jun_15_21:40:06_time1529098806.pickle';  plot_opts.update({'ymin_1mr2': 0.00005, 'ymin_Tk': -1, 'ymax_Tk': 2, 'tryid': 'L32cNoZ', 'pickle_fname_DM_model_error': 'main_do_stage012_rec_2018_Jul_10_15:15:54_time1531235754.pickle'});  math_id='tryL32cNoZ_M10.8-11.8';  # M=10.8-11.8
    #fname = 'main_quick_calc_Pk_2018_Jun_15_21:40:14_time1529098814.pickle';  plot_opts.update({'ymin_1mr2': 0.0005, 'ymin_Tk': -0.7, 'ymax_Tk': 2});  math_id='tryL32cNoZ_M11.8-12.8';  # M=11.8-12.8
    #fname = 'main_quick_calc_Pk_2018_Jun_15_21:40:38_time1529098838.pickle';  plot_opts.update({'ymin_1mr2': 0.00101, 'ymax_plot':2300, 'ymax_plot2':1.3, 'ymin_Tk': -1, 'ymax_Tk': 4.});  math_id='tryL32cNoZ_M12.8-13.8'; # M=12.8-13.8
    #fname = 'main_quick_calc_Pk_2018_Jun_15_21:40:21_time1529098821.pickle';  plot_opts.update({'ymin_1mr2': 0.00101, 'legloc_1mr2': 'lower right', 'ymin_Tk': -5, 'ymax_Tk': 15,'ymax_plot':44000, 'ymax_plot2':1.2});  math_id='tryL32cNoZ_M13.8-15.1';  # M=13.8-15.1

    # ENTER YOUR PICKLE FILES BELOW, specifying the fname and if needed plot_opts. 
    # Comment out the line fname=... above. Always make sure that all but one line
    # with fname are commented out, to avoid confusion.

    # Some description  
    # fname = 'my_pickle_file'; plot_opts.update({})

    



    loginfo = {'fname': fname, 'plot_opts': plot_opts, 'math_id': math_id}
    
    return loginfo
