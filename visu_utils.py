import matplotlib.pyplot as plt
import nrrd
import pandas as pd
import numpy as np


from tracET.core.diff import prepare_input
from tracET.core.skel import surface_skel, line_skel, point_skel
from tracET.metrics.dice2 import cs_dice, cl_dice, pt_dice
from tracET.core.lio import load_mrc


def compute_metrics(pred_skel, gt_skel, pred_tomo, gt_tomo):
    """
    Computes surface DICE metric (s-DICE) for a segmented tomogram and its ground truth
    :param pred_skel: input predicted tomogram (values >0 are considered foreground)
    :param gt_skel: input ground truth (values >0 are considered foreground)
    :param pred_tomo: input predicted tomogram
    :param gt_tomo):: input ground truth
    :return: returns a 3-tuple where the 1st value is cl-DICE, 2nd TP (Topology Precision), and TS
             (Topology Sensitivity)

    """
    # Computing the metric
    tp = (pred_skel * gt_tomo).sum() / pred_skel.sum()
    ts = (gt_skel * pred_tomo).sum() / gt_skel.sum()

    return (
        2 * (tp * ts) / (tp + ts),
        tp,
        ts,
    )


def visualize(pred, gt, pred_skeleton, gt_skeleton, z, name):
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    axs[0, 0].imshow(pred[:, :, z], cmap="gray")
    axs[0, 0].set_title("Prediction")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(gt[:, :, z], cmap="gray")
    axs[0, 1].set_title("Ground Truth")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(pred_skeleton[:, :, z], cmap="gray")
    axs[1, 0].set_title("Prediction Skeleton")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(gt_skeleton[:, :, z], cmap="gray")
    axs[1, 1].set_title("Ground Truth Skeleton")
    axs[1, 1].axis("off")

    plt.suptitle(name)

    plt.show()


def read_skeletons(df, tomo_idx, suffix):
    pred_path = df[df["tomo_idx"] == tomo_idx][f"pred_path_{suffix}"].values[0]
    gt_path = df[df["tomo_idx"] == tomo_idx][f"gt_path"].values[0]
    pred_skeleton_path = df[df["tomo_idx"] == tomo_idx][
        f"pred_skeleton_path_{suffix}"
    ].values[0]
    gt_skeleton_path = df[df["tomo_idx"] == tomo_idx][f"gt_skeleton_path"].values[0]

    print(
        "Reading skeletons for tomo_idx:",
        tomo_idx,
        "path to skeletons:",
        pred_skeleton_path,
    )

    if pred_path.suffix == ".nrrd":
        pred, _ = nrrd.read(pred_path)
    elif pred_path.suffix == ".mrc":
        pred = load_mrc(pred_path)
    gt, _ = nrrd.read(gt_path)
    gt_skeleton, _ = nrrd.read(gt_skeleton_path)
    pred_skeleton, _ = nrrd.read(pred_skeleton_path)

    return pred, gt, pred_skeleton, gt_skeleton


def create_df(pred_paths, gt_paths, pred_skeleton_paths, gt_skeleton_paths):

    data = []

    for pred_path, gt_path, pred_skeleton_path, gt_skeleton_path in zip(
        pred_paths, gt_paths, pred_skeleton_paths, gt_skeleton_paths
    ):
        print("Reading skeletons for:", pred_path)
        if pred_path.suffix == ".nrrd":
            pred = nrrd.read(pred_path)[0]
        elif pred_path.suffix == ".mrc":
            pred = load_mrc(pred_path)

        gt, _ = nrrd.read(gt_path)
        gt_skeleton, _ = nrrd.read(gt_skeleton_path)
        pred_skeleton, _ = nrrd.read(pred_skeleton_path)

        dice, tp, ts = compute_metrics(
            pred_skel=pred_skeleton, gt_skel=gt_skeleton, pred_tomo=pred, gt_tomo=gt
        )

        # extract idx from the pred_path name, for example 	tomo_0050.nrrd
        idx = pred_path.name.split("_")[1].split(".")[0]
        idx = int(idx)

        data.append(
            {
                "tomo_idx": idx,
                "dice": dice,
                "tp": tp,
                "ts": ts,
                "pred_path": pred_path,
                "gt_path": gt_path,
                "pred_skeleton_path": pred_skeleton_path,
                "gt_skeleton_path": gt_skeleton_path,
            }
        )

    df = pd.DataFrame(data)

    return df


def compute_skeleton(tomo, sigma, bin, imf):
    tomo_dsts = prepare_input(tomo, sigma=sigma, bin=bin, imf=imf).astype(np.float32)
    tomo_dsts = tomo_dsts * (tomo_dsts > 0)
    tomo_skel = surface_skel(tomo_dsts, f=0)
    tomo_skel = tomo_skel * tomo  # eliminte noise
    del tomo_dsts
    return tomo_skel
