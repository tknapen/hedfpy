#!/usr/bin/env python
# encoding: utf-8

import os, glob, shutil
from IPython import embed as shell

import hedfpy

task = 'yesno'
raw_dir = 'raw'
output_dir = 'preprocessed'
analysis_params = {
                'sample_rate' : 1000.0,
                'lp' : 6.0,
                'hp' : 0.01,
                'normalization' : 'psc',
                'regress_blinks' : True,
                'regress_sacs' : True,
                'regress_xy' : False,
                'use_standard_blinksac_kernels' : False,
                }
subjects = [
            'sub-01',
            # 'sub-02',
            ]

def preprocess_subjects(subjects, task, output_dir, analysis_params):
    
    for subject in subjects:
        
        # data:
        edfs = glob.glob(os.path.join(raw_dir, '{}*{}*.edf'.format(subject, task)))
        
        # folder hierarchy:
        preprocess_dir = os.path.join(output_dir, task, subject)
        try:
            os.makedirs(os.path.join(preprocess_dir, 'raw'))
        except OSError:
            pass
        
        # hdf5 filename:
        hdf5_filename = os.path.join(preprocess_dir, '{}_{}.hdf5'.format(subject, task))
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
            shutil.copy(edf_file, os.path.join(preprocess_dir, 'raw', '{}.edf'.format(alias)))
            ho.add_edf_file(os.path.join(preprocess_dir, 'raw', '{}.edf'.format(alias)))
            ho.edf_message_data_to_hdf(alias=alias)
            ho.edf_gaze_data_to_hdf(alias=alias,
                                    sample_rate=analysis_params['sample_rate'],
                                    pupil_lp=analysis_params['lp'],
                                    pupil_hp=analysis_params['hp'],
                                    normalization=analysis_params['normalization'],
                                    regress_blinks=analysis_params['regress_blinks'],
                                    regress_sacs=analysis_params['regress_sacs'],
                                    use_standard_blinksac_kernels=analysis_params['use_standard_blinksac_kernels'],
                                    )
        
def main():
    preprocess_subjects(subjects=subjects, task=task, output_dir=output_dir, analysis_params=analysis_params)

if __name__ == '__main__':
    main()