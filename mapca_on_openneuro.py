import json
import os
import csv
import subprocess

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import io as scipy_io
from tedana.workflows import tedana as tedana_cli

from mapca import MovingAveragePCA

repoid = "ds000258"
wdir = "/bcbl/home/home_g-m/llecca/Scripts"

repo = os.path.join(wdir, repoid)
gift = "/bcbl/home/home_g-m/llecca/Scripts/gift"

if os.path.exists(os.path.join(wdir,repoid)):
    print('Repo exists')
else:
    subprocess.run(
        f"cd {wdir} && datalad install https://github.com/OpenNeuroDatasets/{repoid}.git", shell=True
    )

subjects = []

# Create .csv to store results with the following columns:
# Subject, maPCA_AIC, GIFT_AIC, maPCA_KIC, GIFT_KIC, maPCA_MDL, GIFT_MDL
with open(os.path.join(wdir,"mapca_gift_openneuro.csv"),"a") as csv_file:
    writer = csv.writer(csv_file,delimiter="\t")
    writer.writerow(["Subject", "maPCA_AIC", "GIFT_AIC", "maPCA_KIC",
                     "GIFT_KIC", "maPCA_MDL","GIFT_MDL"])

    for sbj in os.listdir(repo):
        sbj_dir = os.path.join(repo, sbj)

        # Access subject directory
        if os.path.isdir(sbj_dir) and "sub-" in sbj_dir:
            echo_times = []
            func_files = []

            subjects.append(os.path.basename(sbj_dir))

            print("Downloading subject", sbj)

            subprocess.run(f"datalad get {sbj}/func", shell=True, cwd=repo)

            print("Searching for functional files and echo times")

            # Get functional filenames and echo times
            for func_file in os.listdir(os.path.join(repo, sbj, "func")):
                if func_file.endswith(".json"):
                    with open(os.path.join(repo, sbj, "func", func_file)) as f:
                        data = json.load(f)
                        echo_times.append(data["EchoTime"])
                elif func_file.endswith(".nii.gz"):
                    func_files.append(os.path.join(repo, sbj, "func", func_file))

            # Sort echo_times values from lowest to highest and multiply by 1000
            echo_times = np.array(sorted(echo_times)) * 1000

            # Sort func_files
            func_files = sorted(func_files)

            # Tedana output directory
            tedana_output_dir = os.path.join(sbj_dir, "tedana")

            print("Running tedana")

            # Run tedana
            try:
                tedana_cli.tedana_workflow(
                    data=func_files,
                    tes=echo_times,
                    out_dir=tedana_output_dir,
                    tedpca="mdl",
                )
            except:
                print("Something went wrong in "+sbj_dir+", check .nii.gz")
                continue

            # Find tedana optimally combined data and mask
            tedana_optcom = os.path.join(tedana_output_dir, "desc-optcom_bold.nii.gz")
            tedana_mask = os.path.join(tedana_output_dir, "desc-adaptiveGoodSignal_mask.nii.gz")

            # Read tedana optimally combined data and mask
            tedana_optcom_img = nib.load(tedana_optcom)
            tedana_mask_img = nib.load(tedana_mask)

            # Make mask binary
            mask_array = tedana_mask_img.get_fdata()
            mask_array[mask_array > 0] = 1
            tedana_mask_img = nib.Nifti1Image(mask_array, tedana_mask_img.affine)

            # Save tedana optimally combined data and mask into mat files
            tedana_optcom_mat = os.path.join(sbj_dir, "optcom_bold.mat")
            tedana_mask_mat = os.path.join(sbj_dir, "mask.mat")
            print("Saving tedana optimally combined data and mask into mat files")
            scipy_io.savemat(tedana_optcom_mat, {"data": tedana_optcom_img.get_fdata()})
            scipy_io.savemat(tedana_mask_mat, {"mask": tedana_mask_img.get_fdata()})

            # Run mapca
            print("Running mapca")
            pca = MovingAveragePCA(normalize=True)
            _ = pca.fit_transform(tedana_optcom_img, tedana_mask_img)

            # Get AIC, KIC and MDL values
            aic = pca.aic_
            kic = pca.kic_
            mdl = pca.mdl_

            # Remove tedana output directory and the anat and func directories
            subprocess.run(f"rm -rf {tedana_output_dir}", shell=True, cwd=repo)
            subprocess.run(f"datalad drop {sbj}/anat", shell=True, cwd=repo)
            subprocess.run(f"datalad drop {sbj}/func", shell=True, cwd=repo)

            # Here run matlab script with subprocess.run
            print("Running GIFT version of maPCA")

            cmd = f'matlab -nodesktop -nosplash -nojvm -logfile {sbj_dir}/giftoutput.txt -r "addpath(genpath(\'{gift}\'));[comp_est_AIC,comp_est_KIC,comp_est_MDL,mdl,aic,kic]=icatb_estimate_dimension(\'{tedana_optcom_mat}\',\'{tedana_mask_mat}\',\'double\',3);save(\'{sbj_dir}/gift.mat\',\'comp_est_AIC\',\'comp_est_KIC\',\'comp_est_MDL\');quit"'

            proc = subprocess.Popen(
                cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            output, err = proc.communicate()
            print(output.decode("utf-8"))

            giftmat = scipy_io.loadmat(os.path.join(sbj_dir, "gift.mat"))

            # Append AIC, KIC and MDL values to a pandas dataframe
            print("Appending AIC, KIC and MDL values to csv file")
            writer.writerow([sbj,
                             aic["n_components"], 
                             giftmat["comp_est_AIC"][0][0], 
                             kic["n_components"],
                             giftmat["comp_est_KIC"][0][0],
                             mdl["n_components"],
                             giftmat["comp_est_MDL"][0][0]])
            csv_file.flush()
            print("Subject", sbj, "done")

    csv_file.close()
