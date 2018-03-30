import os, sys, subprocess, re
import tempfile, logging
import pickle
from datetime import *
import warnings

from math import *
import numpy as np
import numpy.linalg as LA
import matplotlib.pylab as pl
import scipy as sp
from tables import *
import pandas as pd

from .EDFOperator import EDFOperator
from .Operator import Operator
from .EyeSignalOperator import EyeSignalOperator, detect_saccade_from_data

from IPython import embed as shell 

class HDFEyeOperator(Operator):
    """
    HDFEyeOperator is typically used to deal with the data from 
    an entire session. In that case it is associated with a single
    hdf5 file that contains all eye data for the runs of that session,
    and which is given as input_object upon the creation of the
    HDFEyeOperator object 
    """
    
    def __init__(self, input_object, **kwargs):
        super(HDFEyeOperator, self).__init__(input_object = input_object, **kwargs)
        """input_object is the name of the hdf5 file that this operator will create"""
    
    def add_edf_file(self, edf_file_name):
        """
        add_edf_file is the first step in adding a run's edf data 
        to the sessions hdf5 file indicated by self.input_object
        
        add_edf_file creates an EDFOperator object using edf_file_name, and thereby
        immediately converts the edf file to two asc files.
        add_edf_file then uses the EDFOperator object to read all kinds
        of event information (trials, keys, etc) from the event asc file into
        internal variables of the EDFOperator object,
        and also reads the sample data per block (i.e. interval between startrecording
        and stoprecording) into a separate internal variable of the EDFOperator object.
        Putting these data into a hdf5 file is not done here, but in 
        self.edf_message_data_to_hdf and self.edf_gaze_data_to_hdf
        """
        
        self.edf_operator = EDFOperator(edf_file_name)
        # now create all messages
        self.edf_operator.read_all_messages()
        
        # set up blocks for the floating-point data
        self.edf_operator.take_gaze_data_for_blocks()
    
    def open_hdf_file(self, mode = "a"):
        """
        open_hdf_file opens the hdf file that was indicated when
        first creating this HDFEyeOperator object
        """
        self.h5f = open_file(self.input_object, mode = mode )
    
    def close_hdf_file(self):
        """
        close_hdf_file closes the hdf file that was indicated when
        first creating this HDFEyeOperator object
        """
        self.h5f.close()
    
    def add_table_to_hdf(self, run_group, type_dict, data, name = 'bla',filename = []):
        """
        add_table_to_hdf adds a data table to the hdf file
        """
        if filename == []:
            filename = self.edf_operator.input_file_name
            
        this_table = self.h5f.create_table(run_group, name, type_dict, '%s in file %s' % (name, self.edf_operator.input_file_name))
        
        row = this_table.row
        for r in data:
            for par in r.keys():
                row[par] = r[par]
            row.append()
        this_table.flush()
    
    def edf_message_data_to_hdf(self, alias = None, mode = 'a'):
        """
        edf_message_data_to_hdf writes the message data
        from the run's edf file to the session's hdf5 file
        indicated by self.input_object.
        
        The data have typically been taken from the edf
        file and associated with self.edf_operator during an
        earlier call to self.add_edf_file(), but if this is not the case,
        and there is no self.edf_operator,
        then self.add_edf_file() can be called right here.
        
        within the hdf5 file, data are stored under, and can later be retrieved
        at /[source_edf_file_name_without_extension]/[data_kind_name],
        with the latter being e.g. 'trials'
        """
        if not hasattr(self, 'edf_operator'):
            self.add_edf_file(edf_file_name = alias)
        
        if alias == None:
            alias = os.path.split(self.edf_operator.input_file_name)[-1]
        self.open_hdf_file( mode = mode )
        self.logger.info('Adding message data from %s to group  %s to %s' % (os.path.split(self.edf_operator.input_file_name)[-1], alias, self.input_object))
        thisRunGroup = self.h5f.create_group("/", alias, 'Run ' + alias +' imported from ' + self.edf_operator.input_file_name)
        
        #
        #    trials and trial phases
        #
        
        if hasattr(self.edf_operator, 'trials'):
            # create a table for the parameters of this run's trials
            self.add_table_to_hdf(thisRunGroup, self.edf_operator.trial_type_dictionary, self.edf_operator.trials, 'trials')
        
        if hasattr(self.edf_operator, 'trial_phases'):
            # create a table for the parameters of this run's trials
            self.add_table_to_hdf(thisRunGroup, self.edf_operator.trial_phase_type_dictionary, self.edf_operator.trial_phases, 'trial_phases')
        
        #
        #    supporting data types
        #
        
        if hasattr(self.edf_operator, 'parameters'):
            # create a table for the parameters of this run's trials
            self.add_table_to_hdf(thisRunGroup, self.edf_operator.parameter_type_dictionary, self.edf_operator.parameters, 'parameters')
        
        if hasattr(self.edf_operator, 'events'):
            # create a table for the events of this run's trials
            self.add_table_to_hdf(thisRunGroup, self.edf_operator.event_type_dictionary, self.edf_operator.events, 'events')
        
        if hasattr(self.edf_operator, 'sounds'):
            # create a table for the events of this run's trials
            self.add_table_to_hdf(thisRunGroup, self.edf_operator.sound_type_dictionary, self.edf_operator.sounds, 'sounds')
        
        #
        #    eyelink data types
        #
        
        if hasattr(self.edf_operator, 'saccades_from_message_file'):
            # create a table for the saccades from the eyelink of this run's trials
            self.add_table_to_hdf(thisRunGroup, self.edf_operator.saccade_type_dictionary, self.edf_operator.saccades_from_message_file, 'saccades_from_message_file')
            self.add_table_to_hdf(thisRunGroup, self.edf_operator.blink_type_dictionary, self.edf_operator.blinks_from_message_file, 'blinks_from_message_file')
            self.add_table_to_hdf(thisRunGroup, self.edf_operator.fixation_type_dictionary, self.edf_operator.fixations_from_message_file, 'fixations_from_message_file')
        
        # first close the hdf5 file to write to it with pandas
        self.close_hdf_file()
    
    def edf_gaze_data_to_hdf(self, 
            alias = None, 
            which_eye = 0, 
            pupil_hp = 0.01, 
            pupil_lp = 6,
            sample_rate = 1000.,
            normalization = 'psc',
            regress_blinks = True,
            regress_sacs = True,
            regress_xy = False, 
            use_standard_blinksac_kernels = True,
            minimal_frequency_filterbank = 0.0025, 
            maximal_frequency_filterbank = 0.1, 
            nr_freq_bins_filterbank = 9, 
            n_cycles_filterbank = 1, 
            cycle_buffer_filterbank = 3,
            tf_decomposition_filterbank ='lp_butterworth' 
            ):
        """
        edf_gaze_data_to_hdf takes the gaze data
        that is in the run's edf file, processes it,
        and writes the results, as well as the raw data,
        to the sessions hdf5 file that is indicated by
        self.input_object, and also produces some visual
        feedback in the form of figures.
        
        The data have typically been taken from the edf
        file and associated with self.edf_operator during an
        earlier call to self.add_edf_file(), but if this is not the case,
        and there is no self.edf_operator,
        then self.add_edf_file() can be called right here.
        
        within the hdf5 file, data are stored under, and can later be retrieved
        at /[source_edf_file_name_without_extension]/[block_name],
        with the latter being e.g. 'block_1'
        
        edf_gaze_data_to_hdf also produces plots of the raw pupil data and
        blink-interpolated data (stored in pdf files named 'blink_interpolation_1_'[...])
        and of something akin to the derivative of the pupil data along with
        markers indicating that variable's peaks (stored in pdf files named 'blink_interpolation_2_'[...])
        """
        
        # shell()
        
        if not hasattr(self, 'edf_operator'):
            self.add_edf_file(edf_file_name = alias)
        
        if alias == None:
            alias = os.path.split(self.edf_operator.input_file_name)[-1]
        self.logger.info('Adding gaze data from %s to group  %s to %s' % (os.path.split(self.edf_operator.input_file_name)[-1], alias, self.input_object))
        
        #
        #    gaze data in blocks
        #
        with pd.HDFStore(self.input_object) as h5_file:
            # shell()
            # recreate the non-gaze data for the block, that is, its sampling rate, eye of origin etc.
            blocks_data_frame = pd.DataFrame([dict([[i,self.edf_operator.blocks[j][i]] for i in self.edf_operator.blocks[0].keys() if i not in ('block_data', 'data_columns')]) for j in range(len(self.edf_operator.blocks))])
            h5_file.put("/%s/blocks"%alias, blocks_data_frame)
            
            # gaze data per block
            if not 'block_data' in self.edf_operator.blocks[0].keys():
                self.edf_operator.take_gaze_data_for_blocks()
            for i, block in enumerate(self.edf_operator.blocks):
                bdf = pd.DataFrame(block['block_data'], columns = block['data_columns'])
            
                #
                # preprocess pupil:
                #
                for eye in blocks_data_frame.eye_recorded[i]: # this is a string with one or two letters, 'L', 'R' or 'LR'
                # create dictionary of data per block:
                    gazeX = bdf[eye+'_gaze_x']
                    gazeY = bdf[eye+'_gaze_y']
                    pupil = bdf[eye+'_pupil']
                    eye_dict = {'timepoints':bdf.time, 'gaze_X':gazeX, 'gaze_Y':gazeY, 'pupil':pupil,}
                    
                    # create instance of class EyeSignalOperator, and include the blink data as detected by the Eyelink 1000:
                    if hasattr(self.edf_operator, 'blinks_from_message_file'):
                        blink_dict = self.read_session_data(alias, 'blinks_from_message_file')
                        blink_dict[blink_dict['eye'] == eye]
                        sac_dict = self.read_session_data(alias, 'saccades_from_message_file')
                        sac_dict[sac_dict['eye'] == eye]
                        eso = EyeSignalOperator(input_object=eye_dict, eyelink_blink_data=blink_dict, eyelink_sac_data=sac_dict, sample_rate=sample_rate,)
                    else:
                        eso = EyeSignalOperator(input_object=eye_dict, sample_rate=sample_rate)
    
                    # interpolate blinks:
                    eso.interpolate_blinks(method='linear')
                    eso.interpolate_blinks2()

                    # low-pass and band-pass pupil data:
                    eso.filter_pupil(hp=pupil_hp, lp=pupil_lp)
                    
                    # # now dt the resulting pupil data:
                    # eso.dt_pupil(dtype='lp_filt_pupil')
                    # eso.dt_pupil(dtype='bp_filt_pupil')
                    
                    # regress blink and saccade responses
                    # try:
                    eso.regress_blinks(regress_blinks=regress_blinks, regress_sacs=regress_sacs, regress_xy=regress_xy, use_standard_blinksac_kernels=use_standard_blinksac_kernels)
                    # except:
                    #     eso.lp_filt_pupil_clean = eso.lp_filt_pupil.copy()
                    #     eso.bp_filt_pupil_clean = eso.bp_filt_pupil.copy()
                    
                    # add to existing dataframe:
                    bdf[eye+'_interpolated_timepoints'] = eso.interpolated_time_points
                    bdf[eye+'_pupil_int'] = eso.interpolated_pupil
                    bdf[eye+'_pupil_hp'] = eso.hp_filt_pupil
                    bdf[eye+'_pupil_baseline'] = eso.baseline_filt_pupil
                    bdf[eye+'_gaze_x_int'] = eso.interpolated_x
                    bdf[eye+'_gaze_y_int'] = eso.interpolated_y
                    
                    bdf[eye+'_pupil_lp'] = eso.lp_filt_pupil
                    # bdf[eye+'_pupil_lp_dt'] = eso.lp_filt_pupil_dt
                    bdf[eye+'_pupil_lp_clean'] = eso.lp_filt_pupil_clean
                    
                    bdf[eye+'_pupil_bp'] = eso.bp_filt_pupil
                    # bdf[eye+'_pupil_bp_dt'] = eso.bp_filt_pupil_dt
                    bdf[eye+'_pupil_bp_clean'] = eso.bp_filt_pupil_clean
                    
                    # add normalized timeseries:
                    if normalization == 'zscore':
                        eso.zscore_pupil(dtype='lp_filt_pupil')
                        eso.zscore_pupil(dtype='lp_filt_pupil_clean')
                        eso.zscore_pupil(dtype='bp_filt_pupil')
                        eso.zscore_pupil(dtype='bp_filt_pupil_clean')
                        bdf[eye+'_pupil_lp_zscore'] = eso.lp_filt_pupil_zscore
                        bdf[eye+'_pupil_lp_clean_zscore'] = eso.lp_filt_pupil_clean_zscore
                        bdf[eye+'_pupil_bp_zscore'] = eso.bp_filt_pupil_zscore
                        bdf[eye+'_pupil_bp_clean_zscore'] = eso.bp_filt_pupil_clean_zscore
                    if normalization == 'psc':
                        eso.percent_signal_change_pupil(dtype='lp_filt_pupil')
                        eso.percent_signal_change_pupil(dtype='lp_filt_pupil_clean')
                        eso.percent_signal_change_pupil(dtype='bp_filt_pupil')
                        eso.percent_signal_change_pupil(dtype='bp_filt_pupil_clean')
                        bdf[eye+'_pupil_lp_psc'] = eso.lp_filt_pupil_psc
                        bdf[eye+'_pupil_lp_clean_psc'] = eso.lp_filt_pupil_clean_psc
                        bdf[eye+'_pupil_bp_psc'] = eso.bp_filt_pupil_psc
                        bdf[eye+'_pupil_bp_clean_psc'] = eso.bp_filt_pupil_clean_psc
                        
                    # save summary plot:
                    fig = eso.summary_plot()
                    fig.savefig(os.path.join(os.path.split(self.input_object)[0], 'preprocess_{}_{}_{}.pdf'.format(alias, i, eye)))
                    
                    # try time-frequency decomposition of the baseline signal
                    try:
                        eso.time_frequency_decomposition_pupil(
                                minimal_frequency = minimal_frequency_filterbank, 
                                maximal_frequency = maximal_frequency_filterbank, 
                                nr_freq_bins = nr_freq_bins_filterbank, 
                                n_cycles = n_cycles_filterbank, 
                                cycle_buffer = cycle_buffer_filterbank,
                                tf_decomposition=tf_decomposition_filterbank,
                                )
                        self.logger.info('Performed T-F analysis of type %s'%tf_decomposition_filterbank)
                        for freq in eso.band_pass_filter_bank_pupil.keys():
                            bdf[eye+'_pupil_filterbank_bp_%2.5f'%freq] = eso.band_pass_filter_bank_pupil[freq]
                            self.logger.info('Saved T-F analysis %2.5f'%freq)
                    except:
                        self.logger.error('Something went wrong with T-F analysis of type %s'%tf_decomposition_filterbank)
                        pass
                    
                # put in HDF5:
                h5_file.put("/%s/block_%i"%(alias, i), bdf)
    
    def data_frame_to_hdf(self, alias, name, data_frame):
        """docstring for data_frame_to_hdf"""
        with pd.HDFStore(self.input_object) as h5_file:
            h5_file.put("/%s/%s"%(alias, name), data_frame)
    
    #
    #    we also have to get the data from the hdf5 file. 
    #    first, based on simply a EL timestamp period
    #
    
    def sample_in_block(self, sample, block_table):
        """docstring for sample_in_block"""
        return np.arange(block_table['block_end_timestamp'].shape[0])[np.array(block_table['block_end_timestamp'] > float(sample), dtype=bool)][0]
    
    def data_from_time_period(self, time_period, alias, columns = None):
        """data_from_time_period delivers a set of data of type data_type for a given timeperiod"""
        # find the block in which the data resides, based on just the first time of time_period
        with pd.HDFStore(self.input_object) as h5_file:
            period_block_nr = self.sample_in_block(sample = time_period[0], block_table = h5_file['%s/blocks'%alias]) 
            table = h5_file['%s/block_%i'%(alias, period_block_nr)]
            if columns == None:
                columns = table.keys()
        if 'L_vel' in columns:
            columns = table.keys()
        return table[(table['time'] > float(time_period[0])) & (table['time'] < float(time_period[1]))][columns]
    
    def eye_during_period(self, time_period, alias):
        """eye_during_period returns the identity of the eye that was recorded during a given period"""
        with pd.HDFStore(self.input_object) as h5_file:
            period_block_nr = self.sample_in_block(sample = time_period[0], block_table = h5_file['%s/blocks'%alias])
            eye = h5_file['%s/blocks'%alias]['eye_recorded'][period_block_nr]
        return eye
    
    def eye_during_trial(self, trial_nr, alias):
        """docstring for signal_from_trial"""
        with pd.HDFStore(self.input_object) as h5_file:
            table = h5_file['%s/trials'%alias]
            time_period = np.array(table[table['trial_start_index'] == trial_nr][['trial_start_EL_timestamp', 'trial_end_EL_timestamp']])
        return self.eye_during_period(time_period[0], alias)

    def screen_dimensions_during_period(self, time_period, alias):
        """docstring for eye_during_period"""
        with pd.HDFStore(self.input_object) as h5_file:
            period_block_nr = self.sample_in_block(sample = time_period[0], block_table = h5_file['%s/blocks'%alias])
            return h5_file['%s/blocks'%alias]['sample_rate'][period_block_nr]
    
    def sample_rate_during_period(self, time_period, alias):

        block = self.block_for_time_period(alias, time_period)

        return self.block_properties(alias, block).sample_rate



    def sample_rate_during_trial(self, trial_nr, alias):
        """docstring for signal_from_trial"""
        with pd.HDFStore(self.input_object) as h5_file:
            table = h5_file['%s/trials'%alias]
            time_period = np.array(table[table['trial_start_index'] == trial_nr][['trial_start_EL_timestamp', 'trial_end_EL_timestamp']])
        return float(self.sample_rate_during_period(time_period[0], alias))

    def trial_properties(self, alias, trial=None):
        if trial is None:
            trial= slice(None)

        with pd.HDFStore(self.input_object) as h5_file:
            table = h5_file['%s/trials'%alias]

        return table.ix[trial]

    def block_properties(self, alias, block_nr=None):
        if block_nr is None:
            block_nr = slice(None)

        with pd.HDFStore(self.input_object) as h5_file:
            table = h5_file['%s/blocks'%alias]

        return table.ix[block_nr]

    def block_parameters(self, alias, block_nr=None):
        if block_nr is None:
            block_nr = slice(None)

        with pd.HDFStore(self.input_object) as h5_file:
            table = h5_file['%s/parameters'%alias]

        return table.ix[block_nr]

    
    def signal_during_period(self, time_period, alias, signal, requested_eye=None):
        """docstring for gaze_during_period"""

        recorded_eye = self.eye_during_period(time_period, alias)

        if requested_eye is None:
            requested_eye = recorded_eye

        if requested_eye == 'LR' and recorded_eye == 'LR':
            if np.any([signal == 'gaze', signal == 'vel']):
                columns = [s%signal for s in ['L_%s_x', 'L_%s_y', 'R_%s_x', 'R_%s_y']]
            elif signal == 'time':
                columns = [s%signal for s in ['%s']]        
            else:
                columns = [s%signal for s in ['L_%s', 'R_%s']]
        elif requested_eye in recorded_eye:
            if np.any([signal == 'gaze', signal == 'vel']):
                columns = [s%signal for s in [requested_eye + '_%s_x', requested_eye + '_%s_y']]
            elif signal == 'time':
                columns = [s%signal for s in ['%s']]
            else:
                columns = [s%signal for s in [requested_eye + '_%s']]
        else:
            with pd.HDFStore(self.input_object) as h5_file:
                self.logger.error('requested eye %s not found in block %i' % (requested_eye, self.sample_in_block(time_period[0], block_table = h5_file['%s/blocks'%alias])))
            return None    # assert something, dammit!
        return self.data_from_time_period(time_period, alias, columns)
    
    def detect_saccades_during_period(self, time_period, alias, requested_eye=None, l = 5):
        recorded_eye = self.eye_during_period(time_period, alias)

        if requested_eye is None:
            requested_eye = recorded_eye

        xy_data = self.signal_during_period(time_period = time_period, alias = alias, signal = 'gaze', requested_eye = requested_eye)
        vel_data = self.signal_during_period(time_period = time_period, alias = alias, signal = 'vel', requested_eye = requested_eye) 
        saccades = detect_saccade_from_data(xy_data = xy_data, vel_data = vel_data, l = l, sample_rate = self.sample_rate_during_period(time_period, alias))
        saccades = pd.DataFrame(saccades)

        saccades['start_timestamp'] = saccades['expanded_start_time']  + time_period[0]
        saccades['end_timestamp'] = saccades['expanded_end_time']  + time_period[0]

        return saccades

    def detect_saccades_during_trial(self, trial_nr, alias):
        time_period = self.get_time_period_for_trial(trial_nr, alias)
        return self.detect_saccades_during_period(time_period, alias) 

    def blinks_during_period(self, time_period, alias):
        with pd.HDFStore(self.input_object) as h5_file:
            table = h5_file['%s/blinks_from_message_file'%alias]
       
        return table[(table.start_timestamp > time_period[0]) & (table.start_timestamp < time_period[1])]

    def saccades_from_message_file_during_period(self, time_period, alias):
        table =  self.read_session_data(alias, 'saccades_from_message_file')
        return table[(table.start_timestamp > time_period[0]) & (table.start_timestamp < time_period[1])]

    def saccades_from_message_file_during_trial(self, trial_nr, alias):
        time_period = self.get_time_period_for_trial(trial_nr, alias)
        return self.saccades_from_message_file_during_period(time_period, alias) 


    def blinks_during_trial(self, trial_nr, alias):
        time_period = self.get_time_period_for_trial(trial_nr, alias)
        return self.blinks_during_period(time_period, alias)

    def get_time_period_for_trial(self, trial_nr, alias, time_extensions=[0,0]):
        """ Get the timestamps for the start and end of a given trial and alias """
    
        with pd.HDFStore(self.input_object) as h5_file:
            table = h5_file['%s/trials'%alias]

        table = table.set_index('trial_start_index')

        return table.ix[trial_nr].trial_start_EL_timestamp, table.ix[trial_nr].trial_end_EL_timestamp

    def signal_from_trial(self, trial_nr, alias, signal, requested_eye = None, time_extensions = [0,0]):
        """ Get the eye-related signal from a specific trial """

        time_period = self.get_time_period_for_trial(trial_nr, alias, time_extensions)
        return self.signal_during_period(time_period, alias, signal, requested_eye = requested_eye)

    def get_signal_keys(self, trial_nr, alias):
        """ Get keys of processed data signals present in the HDF5
        file """
        with pd.HDFStore(self.input_object) as h5_file:
            table = h5_file['%s/block_%s' % (alias, trial_nr)]
        return table.columns

    def events_from_trial(self, trial_nr, alias):
        """ Get all events for a given trial """

        time_period = self.get_time_period_for_trial(trial_nr, alias)
        return self.events_during_period(time_period, alias)

    def events_during_period(self, time_period, alias):
        with pd.HDFStore(self.input_object) as h5_file:
            table = h5_file['%s/events'%alias]
       
        return table[(table.EL_timestamp > time_period[0]) & (table.EL_timestamp < time_period[1])]
        
    def sounds_from_trial(self, trial_nr, alias):
        """ Get all sounds for a given trial """

        time_period = self.get_time_period_for_trial(trial_nr, alias)
        return self.sounds_during_period(time_period, alias)

    def sounds_during_period(self, time_period, alias):
        with pd.HDFStore(self.input_object) as h5_file:
            table = h5_file['%s/sounds'%alias]
       
        return table[(table.EL_timestamp > time_period[0]) & (table.EL_timestamp < time_period[1])]

    def time_period_for_trial_phases(self, trial_nr, trial_phases, alias ):
        """the time period corresponding to the trial phases requested.
        """
        with pd.HDFStore(self.input_object) as h5_file:
            phase_table = h5_file['%s/trial_phases'%alias]
            trial_table = h5_file['%s/trials'%alias]
            # check whether one of the trial phases is the end or the beginning of the trial.
            # if so, then supplant the time of that phase with its trial's end or start time.
            if trial_phases[0] == 0:
                start_time = trial_table[trial_table['trial_start_index'] == trial_nr]['trial_start_EL_timestamp']
            else:
                start_time = phase_table[((phase_table['trial_phase_index'] == trial_phases[0]) * (phase_table['trial_phase_trial'] == trial_nr))]['trial_phase_EL_timestamp']
            if trial_phases[-1] == -1:
                end_time = trial_table[trial_table['trial_start_index'] == trial_nr]['trial_end_EL_timestamp']
            else:
                end_time = phase_table[((phase_table['trial_phase_index'] == trial_phases[1]) * (phase_table['trial_phase_trial'] == trial_nr))]['trial_phase_EL_timestamp']
            time_period = np.array([np.array(start_time), np.array(end_time)]).squeeze()
        return time_period

    def signal_from_trial_phases(self, trial_nr, trial_phases, alias, signal, requested_eye = 'L', time_extensions = [0,0]):
        """docstring for signal_from_trial"""
        time_period = self.time_period_for_trial_phases(trial_nr = trial_nr, trial_phases = trial_phases, alias = alias)
        time_period = np.array([time_period[0] + time_extensions[0], time_period[1] + time_extensions[1]]).squeeze()
        return self.signal_during_period(time_period, alias, signal, requested_eye = requested_eye)
    
    def detect_saccades_from_trial_phases(self, trial_nr, trial_phases, alias, requested_eye = 'L', time_extensions = [0,0], l = 5):
        time_period = self.time_period_for_trial_phases(trial_nr = trial_nr, trial_phases = trial_phases, alias = alias)
        time_period = np.array([time_period[0] + time_extensions[0], time_period[1] + time_extensions[1]]).squeeze()
        return self.detect_saccades_during_period(time_period = time_period, alias = alias, requested_eye = requested_eye, time_extensions = time_extensions, l = l)

    
    def read_session_data(self, alias, name):
        """
        read_session_data reads data from the hdf5 file indicated by self.input_object.
        Specifically, it reads the data associated with alias/name, with
        'alias' and 'name' typically referring to a run (e.g. 'QQ241214') and
        a data kind (e.g. 'trials'), respectively.
        """
        
        with pd.HDFStore(self.input_object) as h5_file:
            session_data = h5_file['%s/%s'%(alias, name)]
        return session_data

    def trial_for_time_period(self, alias, time_period):
        """
        Returns a trial index for a given time period.
        Raises a warning (no Exception) when the time period spans
        multiple trials, and only returns index of the first of those
        trials.
        """

        trial_properties = self.trial_properties(alias)
        trial = trial_properties[(trial_properties.trial_start_EL_timestamp < time_period[0]) * (trial_properties.trial_end_EL_timestamp > time_period[0])]

        if len(trial) == 0:
            raise Exception('No trial found for this time_period')

        assert(len(trial == 1)), "Found more than one trial in which this time period starts. Something is wrong with the data"

        trial = trial.iloc[0]

        if time_period[1] > trial.trial_end_EL_timestamp:
            warnings.warn(' *** This time period spans multiple trials. Only index of first trial is returned. *** ')

        return int(trial.trial_start_index)

    def block_for_time_period(self, alias, time_period):
        block_properties = self.block_properties(alias)

        block = block_properties[(block_properties.block_start_timestamp < time_period[0]) * (block_properties.block_end_timestamp > time_period[0])]

        assert(len(block == 1)), "Found more than one block in which this time period starts. Something is wrong with the data"

        block = block.iloc[0]

        if time_period[1] > block.block_end_timestamp:
            warnings.warn(' *** This time period spans multiple blocks. Only index of first trial is returned. *** ')

        return int(block.name)
