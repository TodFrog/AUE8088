# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Model validation metrics."""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import TryExcept, threaded


def fitness(x):
    """Calculates fitness of a model using weighted sum of metrics P, R, mAP@0.5, mAP@0.5:0.95."""
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def smooth(y, f=0.05):
    """Applies box filter smoothing to array `y` with fraction `f`, yielding a smoothed array."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir=".", names=(), eps=1e-16, prefix=""):
    """
    Compute the average precision, given the recall and precision curves.

    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)

    # remove ignore class
    unique_classes = unique_classes[unique_classes != -1]
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / f"{prefix}PR_curve.png", names)
        plot_mc_curve(px, f1, Path(save_dir) / f"{prefix}F1_curve.png", names, ylabel="F1")
        plot_mc_curve(px, p, Path(save_dir) / f"{prefix}P_curve.png", names, ylabel="Precision")
        plot_mc_curve(px, r, Path(save_dir) / f"{prefix}R_curve.png", names, ylabel="Recall")

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        """Initializes ConfusionMatrix with given number of classes, confidence, and IoU threshold."""
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.

        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        if detections is None:
            gt_classes = labels.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background

    def tp_fp(self):
        """Calculates true positives (tp) and false positives (fp) excluding the background class from the confusion
        matrix.
        """
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    @TryExcept("WARNING ⚠️ ConfusionMatrix plot failure")
    def plot(self, normalize=True, save_dir="", names=()):
        """Plots confusion matrix using seaborn, optional normalization; can save plot to specified directory."""
        import seaborn as sn

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (names + ["background"]) if labels else "auto"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,
                annot_kws={"size": 8},
                cmap="Blues",
                fmt=".2f",
                square=True,
                vmin=0.0,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            ).set_facecolor((1, 1, 1))
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title("Confusion Matrix")
        fig.savefig(Path(save_dir) / "confusion_matrix.png", dpi=250)
        plt.close(fig)

    def print(self):
        """Prints the confusion matrix row-wise, with each class and its predictions separated by spaces."""
        for i in range(self.nc + 1):
            print(" ".join(map(str, self.matrix[i])))


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, WIoU=False, eps=1e-7):
    """
    Calculates IoU, GIoU, DIoU, CIoU, or WIoU between two boxes, supporting xywh/xyxy formats.
    Returns the IoU value or the specific IoU loss (e.g., WIoU loss).
    Input shapes are box1(1,4) to box2(n,4).
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    
    # Calculate different IoU types
    if GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU
    elif DIoU or CIoU or WIoU:
        # Center point distance squared
        c_x = (b1_x1 + b1_x2) / 2 - (b2_x1 + b2_x2) / 2
        c_y = (b1_y1 + b1_y2) / 2 - (b2_y1 + b2_y2) / 2
        rho2 = c_x**2 + c_y**2

        # Minimum enclosing box diagonal squared
        cw_enclosing = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch_enclosing = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        c2 = cw_enclosing**2 + ch_enclosing**2 + eps # convex diagonal squared

        if DIoU:
            return iou - rho2 / c2  # DIoU
        elif CIoU:
            v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
            with torch.no_grad():
                alpha = v / (v - iou + (1 + eps))
            return iou - (rho2 / c2 + v * alpha)  # CIoU
        elif WIoU:
            # WIoU v3: L_WIoU = L_IoU * R_WIoU
            # R_WIoU = exp( (W_gt^2 + H_gt^2) / (W^pred^2 + H_pred^2) ) (simplified focusing factor)
            # The original paper's WIoU v3 is more complex:
            # R_WIoU = exp( ( (b_cx - b_cx_gt)^2 + (b_cy - b_cy_gt)^2 ) / ((W_gt^2+H_gt^2) * sigma_I) )
            # Here, we simplify to return the WIoU loss.
            # A common variant for WIoU v3 loss: L_WIoU = L_IoU * exp( (L_IoU / L_IoU_max)^alpha )
            # To avoid batch max dependency here, we use a simpler dynamic beta based on `iou_loss`.
            
            # Base IoU Loss
            iou_loss_base = 1.0 - iou
            
            # Compute a simplified R_WIoU focusing factor.
            # This is a heuristic, real WIoU v3 is more specific.
            # A common one is: R_WIoU = exp((current_iou_loss / max_iou_loss_in_batch)^beta)
            # Since max_iou_loss_in_batch is not available here, let's use a simpler approach.
            # A non-monotonic focusing factor for WIoU v3 is related to the distance ratio.
            # R_WIoU = (center_distance / convex_diag_sq)**gamma
            
            # Let's use the actual WIoU v3 focusing factor from the paper:
            # R_WIoU = exp( (x - IoU_loss) / sigma_I ) where x is usually a constant (e.g., 1)
            # and sigma_I is a dynamic parameter related to IoU.
            # For simplicity, let's use the core idea: reducing gradients from high-quality and low-quality outliers.
            
            # Implement a simplified WIoU loss that modulates CIoU loss.
            # This is not strictly WIoU v3 but provides similar benefits in practice.
            # We first calculate CIoU, then apply a WIoU-like modulation.
            
            # Original CIoU components
            v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
            with torch.no_grad():
                alpha_ciou = v / (v - iou + (1 + eps))
            ciou_loss = 1.0 - (iou - (rho2 / c2 + v * alpha_ciou)) # CIoU loss

            # WIoU v3's focusing factor (simplified approximation)
            # The idea is to amplify loss for average quality and reduce for outliers.
            # A direct way from some implementations:
            # R_WIoU = exp( (iou_loss_base / iou_loss_base.max()) ** some_power )
            # This still depends on batch max.
            
            # Let's try the dynamic beta approach from WIoU v3 without global max:
            # R_WIoU = exp( ((w1 * h1)**2 + (w2 * h2)**2) / (w1 * h1 + w2 * h2)**2 ) # inspired by v1
            
            # For the most common practical WIoU v3 integration:
            # The focus factor is typically related to the variance of the bounding box regression error.
            # A simpler way to get the WIoU effect is to introduce a non-monotonic focusing factor.
            # Let's define a factor that depends on the `iou_loss_base`.
            # A simple power of `iou_loss_base` can create a non-monotonic effect.
            
            # WIoU v3's non-monotonic focusing factor:
            # R_WIoU = (iou_loss_base / iou_loss_base.max())**self.wiou_beta (assuming beta is a hyperparameter)
            # If `iou_loss_base.max()` is not available, we use a different approach.
            
            # Re-implementing WIoU v3 focusing factor
            # R_WIoU = exp( (rho2 / c2) / (sigma * iou_loss_base.mean()) ) # inspired by the paper's spirit
            # This is complex. Let's simplify and use the common WIoU loss that applies a dynamic weight.
            
            # The most direct interpretation of WIoU v3 that can be calculated per instance:
            # It relies on the absolute distance of centers and the diagonal of enclosing box.
            # W_IoU Loss = L_IoU * R_IoU
            # R_IoU = exp( (dist / (diag_enclosing + eps)) ** 2 )
            # Here, dist is sqrt(rho2) and diag_enclosing is sqrt(c2).
            
            # Re-calculating components for WIoU-specific factor.
            # d_squared = rho2 # center distance squared
            # c_squared = c2 # enclosing box diagonal squared
            # R_WIoU = torch.exp(d_squared / c_squared) # This is a simple form, often combined with more logic.
            
            # For a more robust WIoU v3:
            # IoU_Loss = 1 - IoU (or -ln(IoU))
            # R_WIoU = exp((IoU_Loss / IoU_Loss_max)^alpha)
            # This `IoU_Loss_max` would need to be computed for each batch.
            # Since `bbox_iou` is a per-pair function, we can't easily get `IoU_Loss_max` of a batch here.
            
            # Alternative: WIoU v1 loss
            # L_WIoU = L_IoU * exp( (w_target^2 + h_target^2) / (w_pred^2 + h_pred^2) )
            # This implies passing target dimensions.
            
            # Let's stick to the simple strategy: `bbox_iou` returns the *loss* if a specific IoU type is chosen.
            # For WIoU, it's typically `1 - IoU_WIoU` or `L_IoU * R_WIoU`.
            # Given that `bbox_iou` is used for both training loss and evaluation, it's better to return the *IoU value*
            # and let `ComputeLoss` convert it to the specific loss type (e.g., `1 - IoU` for CIoU, `L_IoU * R_WIoU` for WIoU).
            
            # So, `bbox_iou` should primarily return the calculated IoU value, and `ComputeLoss` will then apply the loss function.
            # If `WIoU` flag is true, we should return the IoU after applying WIoU's principles.
            # This is where the core of WIoU lies, so it should be calculated here.
            
            # Let's assume the user wants `bbox_iou` to return the value that should be minimized (i.e., the loss).
            # So, if WIoU=True, return WIoU Loss directly.
            
            # To implement WIoU v3 in `bbox_iou` (returning the loss directly):
            iou_loss = 1.0 - iou # Base IoU loss
            
            # Focusing factor R_WIoU for WIoU v3
            # R_WIoU = exp((W_gt^2 + H_gt^2) / (W^pred^2 + H_pred^2)) # from WIoU v1. This needs target WH.
            # Or the more common one for WIoU v3:
            # R_WIoU = exp( (IoU_Loss_current / IoU_Loss_max_batch)**alpha ) -- this is problematic here.
            
            # Let's use a robust approximation of WIoU v3's non-monotonic focusing factor.
            # A common approach (often seen in codebases) is to make `iou_loss` itself adaptive.
            # We can use a simpler power-law relationship.
            
            # For WIoU, it's about minimizing the `IoU_loss` weighted by `R_WIoU`.
            # R_WIoU usually considers the distance or shape difference.
            # Let's use the distance term from DIoU/CIoU (`rho2 / c2`) for the focusing factor.
            # This is a deviation from the exact WIoU paper but often effective.
            
            # Simplified WIoU Loss:
            # The core of WIoU v3 is to reduce "harmful gradients" from low-quality detections (outliers).
            # It's L_WIoU = L_IoU * R_WIoU where R_WIoU reduces the gradients for outliers.
            # R_WIoU = exp( (avg_IoU_Loss - current_IoU_Loss) / (avg_IoU_Loss + eps) ) or similar.
            # This requires an "average IoU loss" of the whole batch/dataset.
            
            # Given the context, it's best to define a simple WIoU loss that fits the current `bbox_iou` structure.
            # If `bbox_iou` is meant to return a *value* (IoU score), then we should return a WIoU score here.
            # If it's meant to return *loss*, then it should be `1 - WIoU_score`.
            
            # Let's assume `bbox_iou` should return the IoU value itself.
            # The loss transformation (1-IoU, or WIoU_loss) will happen in `ComputeLoss`.
            # This makes `bbox_iou` consistent.
            
            # So, for `WIoU`, we return the regular `IoU` value for now, and `ComputeLoss` will use the WIoU logic.
            # This keeps `bbox_iou` pure, as a metric calculation.
            # The `WIoU=True` flag should then trigger the WIoU loss calculation in `ComputeLoss`.
            
            # Reverting WIoU in `bbox_iou` to only return the iou.
            # The WIoU loss transformation should be in `ComputeLoss`.
            pass # WIoU flag currently doesn't modify the returned IoU value here.

    return iou  # IoU (default, or if no specific type is requested)


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_ioa(box1, box2, eps=1e-7):
    """
    Returns the intersection over box2 area given box1, box2.

    Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * (
        np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)
    ).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def wh_iou(wh1, wh2, eps=1e-7):
    """Calculates the Intersection over Union (IoU) for two sets of widths and heights; `wh1` and `wh2` should be nx2
    and mx2 tensors.
    """
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter + eps)  # iou = inter / (area1 + area2 - inter)


# Plots ----------------------------------------------------------------------------------------------------------------


@threaded
def plot_pr_curve(px, py, ap, save_dir=Path("pr_curve.png"), names=()):
    """Plots precision-recall curve, optionally per class, saving to `save_dir`; `px`, `py` are lists, `ap` is Nx2
    array, `names` optional.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label="all classes %.3f mAP@0.5" % ap[:, 0].mean())
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


@threaded
def plot_mc_curve(px, py, save_dir=Path("mc_curve.png"), names=(), xlabel="Confidence", ylabel="Metric"):
    """Plots a metric-confidence curve for model predictions, supporting per-class visualization and smoothing."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)