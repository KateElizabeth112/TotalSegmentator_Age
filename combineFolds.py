# script for combining the results from multiple folds into a single file
import pickle as pkl
import numpy as np
import os

local = False
if local:
    root_dir = "/Users/katecevora/Documents/PhD/data/TotalSegmentator"
else:
    root_dir = '/rds/general/user/kc2322/projects/cevora_phd/live/TotalSegmentator/'

folds = [0, 1, 2, 3, 4]


def main():
    # iterate over 3 experiments
    for ex in range(0, 3):
        case_id_all = []
        sex_all = []
        age_all = []
        dice_all = []
        hd_all = []
        vol_pred_all = []
        vol_gt_all = []

        # iterate over folds and combine
        for fold in folds:
            ds = "Dataset{}0{}_Age{}".format(5 + fold, ex, fold)
            f = open(os.path.join(root_dir, "inference", ds, "all", "results.pkl"), "rb")
            results = pkl.load(f)
            f.close()

            print(results["dice"].shape)

            case_id_all.append(list(results["case_id"]))
            sex_all.append(list(results["sex"]))
            age_all.append(list(results["age"]))
            dice_all.append(list(results["dice"]))
            hd_all.append(results["hd"])
            vol_pred_all.append(list(results["vol_pred"]))
            vol_gt_all.append(list(results["vol_gt"]))

        print(np.array(dice_all).shape)

        f = open(os.path.join(root_dir, "inference", "results_age_{}.pkl".format(ex)), 'wb')
        pkl.dump({"case_id": np.array(case_id_all),
                  "sex": np.array(sex_all),
                  "age": np.array(age_all),
                  "dice": np.array(dice_all),
                  "hd": hd_all,
                  "vol_pred": np.array(vol_pred_all),
                  "vol_gt": np.array(vol_gt_all),
                  }, f)
        f.close()


if __name__ == "__main__":
    main()