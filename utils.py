import os
import json
import warnings
import numpy as np
import pandas as pd
import nibabel as nib

from glob import glob
from neuromaps import datasets
from nilearn import signal, interfaces, glm
warnings.filterwarnings("ignore") # Nilearn is too verbose with its future warnings. 


def clean_fmriprep_output(cfg, subject):
    """
    Concatenate all of the fMRIPrep-cleaned runs specified in config.json for a single subject.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary (paths, filtering params, thresholds).
    subject : str
        Subject ID.
    """

    output          = cfg['output']
    fmriprep_output = cfg['fmriprep_output']

    run_files       = cfg['run_files']
    event_files      = cfg['event_files']

    kernel_size     = cfg['kernel_size']
    fd_threshold    = cfg['fd_threshold']
    n_dummy         = cfg['dummy_scans']
    low_pass        = cfg['low_pass']
    high_pass       = cfg['high_pass']


    # Create output directories.
    cleaned_dir = f'{output}/sub-{subject}'
    tmp_dir     = f'{output}/sub-{subject}/tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(cleaned_dir, exist_ok=True)


    # Clean each run
    log = []
    cii_files = [f'{fmriprep_output}/sub-{subject}/func/sub-{subject}_{run_name}' for run_name in run_files]

    for cii_file, event_file  in zip(cii_files, event_files):

        print(f'Processing {cii_file}...')

        scan_info = json.load(open(cii_file.replace('.dtseries.nii', '.json')))
        repetition_time = scan_info['RepetitionTime']

        # Convert to GIFTI.
        tmp_gii = os.path.join(tmp_dir, 'tmp_gifti.dtseries.gii')
        os.system(f'wb_command -cifti-convert -to-gifti-ext {cii_file} {tmp_gii}')
        gii = nib.load(tmp_gii)

        # Ensure GIFTI meets minimumim filter requirements.
        if gii.darrays[0].data.shape[1] < 33: continue

        # Load confounds and high-motion mask for scrubbing.
        confounds, mask = interfaces.fmriprep.load_confounds_strategy(
            cii_file,
            denoise_strategy='scrubbing',
            fd_threshold=fd_threshold
        )

        # Add first 5 a_comp_cor components to confounds.
        comp_cor, _ = interfaces.fmriprep.load_confounds_strategy(cii_file, denoise_strategy='compcor')
        comp_cols = [c for c in comp_cor.columns if 'a_comp_cor_' in c][:5]
        confounds[comp_cols] = comp_cor[comp_cols]

        # Remove dummy scans.
        if mask is None:
            mask = np.arange(len(confounds))
        mask = mask[~np.isin(mask, np.arange(n_dummy))]

        # If run is task-based, estimate the HDR evoked by the stimuli, and add this to the list of confounds.
        if event_file is not None:

            events = pd.read_csv(event_file, sep='\t', header=None)
            frame_times = np.linspace(0, (len(confounds) - 1) * repetition_time, len(confounds))

            # Clean events dataframe as expeceted by 'make_first_level_design_matrix'.
            events = events.dropna().reset_index(drop=True)
            events = events.rename(columns={0:'onset',1:'duration'})
            events = events[['onset','duration']]

            # Create design matrix.
            design_mat = glm.first_level.make_first_level_design_matrix(frame_times, events=events)

            # Add estimate of stimuli-evoked HDR to list of confounds.
            confounds = confounds.join(design_mat)

        # Log scrubbing info
        log.append([f"{len(mask)} of {len(confounds)} volumes remain after scrubbing [{os.path.basename(cii_file)}]"])
        
        # Exclude run if >50% volumes removed.
        if len(mask) < 0.5*len(confounds):
            continue

        # Clean signal.
        signals = gii.darrays[0].data.T
        signals_clean = signal.clean(
            signals,
            confounds=confounds,
            sample_mask=mask,
            low_pass=low_pass, high_pass=high_pass,
            standardize=False, t_r=repetition_time
        )

        # Save cleaned GIFTI.
        cleaned_darray = nib.gifti.GiftiDataArray(
            data=signals_clean.T,
            intent=gii.darrays[0].intent,
            meta=gii.darrays[0].meta
        )
        gii.darrays[0] = cleaned_darray
        cleaned_gii_path = os.path.join(tmp_dir, 'cleaned_signals.func.gii')
        nib.save(gii, cleaned_gii_path)

        # Convert back to CIFTI.
        cleaned_cifti = os.path.join(cleaned_dir, os.path.basename(cii_file))
        os.system(f'wb_command -cifti-convert -from-gifti-ext {cleaned_gii_path} {cleaned_cifti} -reset-timepoints {repetition_time} 0')


    # Write log to .txt
    pd.DataFrame(log).to_csv(os.path.join(cleaned_dir,'scrubbing_log.txt'), header=None, index=False)

    # Clean up temporary files
    for f in glob(os.path.join(tmp_dir,'*')):
        os.remove(f)

    # Resample surfaces to fsLR-32k
    fslr = datasets.fetch_atlas(atlas='fslr', density='32k')
    for hemi_idx, hemi in enumerate(['L','R']):

        surface_in  = glob(f'{fmriprep_output}/sub-{subject}/anat/*hemi-{hemi}_midthickness.surf.gii')[0]
        curr_sphere = glob(f'{fmriprep_output}/sub-{subject}/anat/*{hemi}_space-fsLR_desc-reg_sphere.surf.gii')[0]

        new_sphere = fslr['sphere'][hemi_idx]
        surface_out = os.path.join(cleaned_dir, f'sub-{subject}_midthickness.{hemi}.surf.gii')
        os.system(f'wb_command -surface-resample {surface_in} {curr_sphere} {new_sphere} BARYCENTRIC {surface_out}')


    # Demean, merge, smooth.
    ciftis = sorted(glob(f'{cleaned_dir}/*_bold.dtseries.nii'))
    demeaned = []
    for cii in ciftis:
        data = nib.load(cii).get_fdata()
        data -= data.mean(axis=0, keepdims=True)
        outname = cii.replace('.dtseries','_demeaned.dtseries')
        nib.save(nib.Cifti2Image(data, nib.load(cii).header, nib.load(cii).nifti_header), outname)
        demeaned.append(outname)


    merged = os.path.join(cleaned_dir, f'sub-{subject}_task-rest_concatenated.dtseries.nii')
    os.system(f'wb_command -cifti-merge {merged} {" ".join(["-cifti "+f for f in demeaned])}')

    smoothed = merged.replace('.dtseries','_smoothed.dtseries')
    os.system(
        f'wb_command -cifti-smoothing {merged} {kernel_size} {kernel_size} COLUMN {smoothed} '
        f'-left-surface {cleaned_dir}/sub-{subject}_midthickness.L.surf.gii '
        f'-right-surface {cleaned_dir}/sub-{subject}_midthickness.R.surf.gii -merged-volume'
    )

    print('\nProcessing complete.')
    return


def run_precision_mapping(cfg, subject):

    output = cfg['output']
    cleaned_dir = f'{output}/sub-{subject}'

    cii = f'{cleaned_dir}/sub-{subject}_task-rest_concatenated_smoothed.dtseries.nii'
    os.system(
        f'wb_command -cifti-separate {cii} COLUMN \
        -metric CORTEX_LEFT {cleaned_dir}/sub-{subject}_task-rest_concatenated_smoothed.L.func.gii \
        -metric CORTEX_RIGHT {cleaned_dir}/sub-{subject}_task-rest_concatenated_smoothed.R.func.gii \
    ')

    for hemi in ['L','R']:
        os.system(
            f"cortex_mapping \
            --func {cleaned_dir}/sub-{subject}_task-rest_concatenated_smoothed.{hemi}.func.gii \
            --surf {cleaned_dir}/sub-{subject}_midthickness.{hemi}.surf.gii \
            --output {cleaned_dir}/ \
        ")

    return
