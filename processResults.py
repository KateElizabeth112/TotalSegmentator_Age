import numpy as np
import nibabel as nib
import os
import pickle as pkl
import pandas as pd
from monai.metrics import compute_hausdorff_distance
import argparse

# argparse
parser = argparse.ArgumentParser(description="Just an example",  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--dataset", default="Dataset303_Set3", help="Task to evaluate")
args = vars(parser.parse_args())

# set up variables
task = args["dataset"]

local = False
if local:
    root_dir = "/Users/katecevora/Documents/PhD/data/KITS19/"
else:
    root_dir = '/rds/general/user/kc2322/projects/cevora_phd/live/kits19/'

fold = "all"

preds_dir = os.path.join(root_dir, "inference", task, fold)
gt_dir = os.path.join(root_dir, "nnUNet_raw", task, "labelsTs")
meta_data_path = os.path.join(root_dir, "metadata.pkl")

labels = {"background": 0,
          "kidney": 1,
          "tumor": 2}

n_channels = int(len(labels))


def getVolume(pred, gt):
    # Get the organ volumes given the ground truth mask and the validation mask
    vol_preds = []
    vol_gts = []
    for channel in range(n_channels):
        vol_preds.append(np.sum(pred[pred == channel]))
        vol_gts.append(np.sum(gt[gt == channel]))

    return np.array(vol_preds), np.array(vol_gts)


def oneHotEncode(array):
    array_dims = len(array.shape)
    array_max = n_channels
    one_hot = np.zeros((array_max + 1, array.shape[0], array.shape[1], array.shape[2]))

    for i in range(0, array_max + 1):
        one_hot[i, :, :, :][array==i] = 1

    return one_hot


def computeHDDIstance(pred, gt):
    # To use the MONAI function pred must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32].
    # The values should be binarized.
    # gt: ground truth to compute mean the distance. It must be one-hot format and first dim is batch.
    # The values should be binarized.

    # Convert to one hot
    # covert predictions to one hot encoding
    pred_one_hot = oneHotEncode(pred)
    gt_one_hot = oneHotEncode(gt)

    # expand the number of dimensions to include batch
    pred_one_hot = np.expand_dims(pred_one_hot, axis=0)
    gt_one_hot = np.expand_dims(gt_one_hot, axis=0)

    hd = compute_hausdorff_distance(pred_one_hot, gt_one_hot, include_background=False, distance_metric='euclidean', percentile=None,
                               directed=False, spacing=None)

    return hd


def multiChannelDice(pred, gt, channels):

    dice = []

    for channel in range(channels):
        a = np.zeros(pred.shape)
        a[pred == channel] = 1

        b = np.zeros(gt.shape)
        b[gt == channel] = 1

        dice.append(np.sum(a[b == 1])*2.0 / (np.sum(a) + np.sum(a)))

    return np.array(dice)


def calculateMetrics():
    # open the metadata
    f = open(meta_data_path, "rb")
    info = pkl.load(f)
    f.close()

    patients = np.array(info["id"])
    genders = np.array(info["gender"])       # male = 0, female = 1
    ages = np.array(info["age"])

    # containers to store results
    case_id = []
    sex = []
    dice = []
    age = []
    hausdorff = []
    vol_pred = []
    vol_gt = []

    cases = os.listdir(preds_dir)
    for case in cases:
        if case.endswith(".nii.gz"):
            id = case[5:9]
            print("Processing {}".format(id))

            pred = nib.load(os.path.join(preds_dir, case)).get_fdata()
            gt = nib.load(os.path.join(gt_dir, case)).get_fdata()

            if np.unique(gt).sum() == 0:
                print("Only background")

            # Get Dice and NSD and volumes
            dice = multiChannelDice(pred, gt, n_channels)
            hd = computeHDDIstance(pred, gt)
            vol_pred, vol_gt = getVolume(pred, gt)

            if id in patients:
                case_id.append(id)
                sex.append(genders[cases == id])
                age.append(ages[case == id])
                dice.append(dice)
                hausdorff.append(hd)
                vol_pred.append(vol_pred)
                vol_gt.append(vol_gt)
            else:
                print("Not in list")

    # convert to numpy arrays (except for HD distance)
    case_id = np.array(case_id)
    dice = np.array(dice)
    sex = np.array(sex)
    age = np.array(age)
    vol_pred = np.array(vol_pred)
    vol_gt = np.array(vol_gt)

    print("Number of men: {}".format(sex.shape[0] - np.sum(sex)))
    print("Number of women: {}".format(np.sum(sex)))

    f = open(os.path.join(preds_dir, "results.pkl"), "wb")
    pkl.dump({"case_id": case_id,
              "sex": sex,
              "age": age,
              "dice": dice,
              "hd": hausdorff,
              "vol_pred": vol_pred,
              "vol_gt": vol_gt,
              }, f)
    f.close()


def main():
    calculateMetrics()



if __name__ == "__main__":
    main()