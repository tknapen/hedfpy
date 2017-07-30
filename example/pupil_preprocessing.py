#!/usr/bin/env python
# encoding: utf-8

import os, glob
from IPython import embed as shell

import hedfpy

task = 'yesno'
raw_dir = 'raw/'
output_dir = os.path.expanduser('~/Downloads/pupil_prep/')
analysis_params = {
                'hp' : 6.0,
                'lp' : 0.01,
                'normalization' : 'psc',
                }
subjects = [
            'sub-01',
            # 'sub-02',
            ]

def preprocess_subjects(subjects, task, output_dir, analysis_params):
    """import_all_data loops across the aliases of the sessions and converts the respective edf files, adds them to the self.ho's hdf5 file. """
    
    for subject in subjects:
    
        # data:
        edfs = glob.glob(os.path.join(raw_dir, '{}*{}*.edf'.format(subject, task)))
        
        # folder hierarchy:
        output_dir = os.path.join(output_dir, task, subject)
        try:
            os.makedirs(os.path.join(output_dir, 'raw'))
        except OSError:
            pass
    
        # hdf5 filename:
        hdf5_filename = os.path.join(output_dir, '{}_{}.hdf5'.format(subject, task))
        try:
            os.remove(hdf5_filename)
        except OSError:
            pass
    
        # initialize hdf5 HDFEyeOperator:
        ho = hedfpy.HDFEyeOperator(hdf5_filename)
    
        # variables:
        session_nrs = [r.split('ses-')[1][0:2] for r in edfs]
        run_nrs = [r.split('run-')[1][0:1] for r in edfs]
        aliases = []
        for i in range(len(session_nrs)):
            aliases.append('{}_{}_{}'.format(task, session_nrs[i], run_nrs[i]))
    
        # preprocessing:
        for edf_file, alias in zip(edfs, aliases):
            os.system('cp "' + edf_file + '" "' + os.path.join(output_dir, 'raw', alias + '.edf"'))
            ho.add_edf_file(os.path.join(output_dir, 'raw', alias + '.edf'))
            ho.edf_message_data_to_hdf(alias=alias)
            ho.edf_gaze_data_to_hdf(alias=alias,
                                    pupil_hp=analysis_params['hp'],
                                    pupil_lp=analysis_params['lp'],
                                    normalization=analysis_params['normalization']
                                    )

def main():
    preprocess_subjects(subjects=subjects, task=task, output_dir=output_dir, analysis_params=analysis_params)

if __name__ == '__main__':
    main()