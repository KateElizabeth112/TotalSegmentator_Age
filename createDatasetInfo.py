# Create a pkl file that contains dataset information in a dictionary of arrays
# Include only images that have 4 organs of interest labelled
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle as pkl
import argparse

# argparse
parser = argparse.ArgumentParser(description="Just an example",  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-l", "--local", default=False, help="are we running locally or on hcp clusters")
args = vars(parser.parse_args())

# set up variables
local = False

if local:
    root_folder = "/Users/katecevora/Documents/PhD/data/TotalSegmentator"
else:
    root_folder = "/rds/general/user/kc2322/projects/cevora_phd/live/TotalSegmentator"

images_folder = os.path.join(root_folder, "nnUNet_raw/Dataset300_Full/imagesTr")

def main():

    meta = pd.read_csv(os.path.join(root_folder, "meta.csv"), sep=";")
    patients_all = meta["image_id"].values
    genders_all = meta["gender"].values
    age_all = meta["age"].values
    institute_all = meta["institute"].values
    study_type_all = meta["study_type"].values

    patients = []
    genders = []
    age = []
    institute = []
    study_type = []

    fnames = os.listdir(images_folder)
    for f in fnames:
        id = f[5:9]

        if not ("s" + id in patients_all):
            print("subject {} not found in metadata".format(id))

        age.append(age_all[patients_all == "s" + id])
        genders.append(genders_all[patients_all == "s" + id])
        age.append(age_all[patients_all == "s" + id])
        institute.append(institute_all[patients_all == "s" + id])
        study_type.append(study_type_all[patients_all == "s" + id])

    # Save lists
    info = {"patients": patients,
            "genders": genders,
            "age": age,
            "institute": institute,
            "study_type": study_type}

    f = open(os.path.join(root_folder, "info.pkl"), "wb")
    pkl.dump(info, f)
    f.close()


if __name__ == "__main__":
    main()