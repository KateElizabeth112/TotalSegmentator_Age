# Create a pkl file that contains dataset information in a dictionary of arrays
# Include only images that have 4 organs of interest labelled
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle as pkl
import argparse
import numpy as np


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

def create():

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

        patients.append(id)
        age.append(age_all[patients_all == "s" + id][0])
        genders.append(genders_all[patients_all == "s" + id][0])
        institute.append(institute_all[patients_all == "s" + id][0])
        study_type.append(study_type_all[patients_all == "s" + id][0])

    # Save lists
    info = {"id": np.array(patients),
            "sex": np.array(genders),
            "age": np.array(age),
            "institute": np.array(institute),
            "study_type": np.array(study_type)}

    f = open(os.path.join(root_folder, "info.pkl"), "wb")
    pkl.dump(info, f)
    f.close()


def explore():
    f = open(os.path.join(root_folder, "info.pkl"), "rb")
    info = pkl.load(f)
    f.close()

    age = info["age"]
    age = np.squeeze(np.array(age))

    plt.clf()
    plt.hist(age)
    plt.show()


def main():
    create()
    #explore()


if __name__ == "__main__":
    main()