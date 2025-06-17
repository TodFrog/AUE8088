
#!/usr/bin/env python3
"""
pipeline_check.py · KAIST RGB‑T end‑to‑end coordinate sanity‑checker
-------------------------------------------------------------------
Created : 2025-06-13 06:42
Author  : ChatGPT debug assistant

The script visualises how ground‑truth bounding‑boxes change at each
preprocessing step inside **LoadRGBTImagesAndLabels**.

Stages dumped:
    0. original_rgb.jpg          – raw Visible image with GT
    1. letterbox_rgb.jpg         – after letterbox() + box shift
    2. randompersp_rgb.jpg       – after shared RandomPerspective (if --augment)
    3. final_tensor_rgb.jpg      – what actually enters the network

It also prints numeric arrays so you can diff them quickly.

Usage
-----
    python pipeline_check.py \
        --data-root   "D:/AUE8088/datasets/kaist-rgbt" \
        --split       "val" \
        --index       0 \
        --img-size    640 \
        --out-dir     "debug_vis" \
        --augment          # verify augmentation branch

The script must be called from the repo root so that `utils.*` imports
resolve.  It does **not** train or validate; it only walks through one
sample.

"""

import argparse, sys, random, cv2, numpy as np, torch
from pathlib import Path
import pprint, os

# ─────────────────────────────── helpers
def draw_boxes(img, boxes, colour=(0,255,0), thickness=2, txt=True):
    out = img.copy()
    for b in boxes:
        cls, x1, y1, x2, y2 = map(int, b[:5])
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)
        if txt:
            cv2.putText(out, str(cls), (x1, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return out

def save(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)

# ─────────────────────────────── main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True,
                        help='KAIST rgbt dataset root folder')
    parser.add_argument('--split', default='train',
                        help='sub folder name used in YAML (train/val/test)')
    parser.add_argument('--index', type=int, default=0,
                        help='image index to inspect')
    parser.add_argument('--img-size', type=int, default=640,
                        help='desired network img-size (square)')
    parser.add_argument('--out-dir', type=str, default='debug_vis')
    parser.add_argument('--augment', action='store_true',
                        help='run augmentation branch incl. RandomPerspective')
    args = parser.parse_args()

    # repo root assumed = cwd
    repo_root = Path('.').resolve()
    sys.path.append(str(repo_root))

    from utils.dataloaders import LoadRGBTImagesAndLabels
    from utils.augmentations import letterbox, random_perspective
    from utils.general import xyxy2xywhn, xywhn2xyxy

    # minimal hyp just for toggles we touch
    hyp = dict(mosaic=0.0, mixup=0.0, hsv_h=0, hsv_s=0, hsv_v=0,
               flipud=0, fliplr=0, degrees=0, translate=0, scale=0,
               shear=0, perspective=0)

    yaml_cfg = Path(args.data_root) / "kaist.yaml"
    if not yaml_cfg.exists():
        print(f"[WARN] didn't find kaist.yaml at {yaml_cfg}, will build path list directly.")

    # Create dataset instance (batch_size=1)
    ds = LoadRGBTImagesAndLabels(
            path=str(Path(args.data_root) / args.split),   # ★ train / val / test 만!
            img_size=args.img_size,
            batch_size=1,
            augment=args.augment,
            hyp=hyp,
            rect=False,
            stride=32,
            pad=0.0,
            prefix='',
            single_cls=False,
            cache_images=False)

    # fetch single sample
    imgs_tensor, labels_out, path, shapes, _ = ds[args.index]

    # paths
    save_root = Path(args.out_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    # ── Original image & boxes (Visible modality for clarity)
    vis_path = Path(path).as_posix().replace('/lwir/', '/visible/').replace('/lwir\\', '/visible/')
    orig = cv2.imread(vis_path)
    if orig is None:
        raise FileNotFoundError(vis_path)
    orig_h, orig_w = orig.shape[:2]

    # labels_out is after all transforms → we need original labels from ds.labels
    orig_labels_norm = ds.labels[ds.indices[args.index]][:, :5]  # (N,5)
    if len(orig_labels_norm):
        orig_labels_xyxy = xywhn2xyxy(orig_labels_norm[:,1:5].copy(),
                                      w=orig_w, h=orig_h)
        orig_boxes = np.concatenate([orig_labels_norm[:,0:1], orig_labels_xyxy], axis=1)
    else:
        orig_boxes = np.zeros((0,5))

    save(draw_boxes(orig, orig_boxes), save_root/'0_original.jpg')
    print("Original size :", orig.shape, " boxes:", orig_boxes[:3], "...")

    # ── After Dataset pipeline (final tensor, CHW, RGB)
    # tensors list -> choose Visible modality (index 1)
    final_img = imgs_tensor[1].numpy()[::-1]  # to BGR
    final_img = np.transpose(final_img, (1,2,0))
    final_h, final_w = final_img.shape[:2]

    # our labels_out is (N,6) [batch, cls, xc, yc, w, h] – no batch idx (batch==0)
    boxes_final_xyxy = xywhn2xyxy(labels_out[:,2:6].numpy(),
                                  w=final_w, h=final_h)
    boxes_final = np.concatenate([labels_out[:,1:2].numpy(), boxes_final_xyxy], 1)

    save(draw_boxes(final_img.copy(), boxes_final, (0,255,255)),
         save_root/'3_final_tensor.jpg')
    print("Final tensor size :", final_img.shape, " boxes_final:", boxes_final[:3])

    # letterbox stage (taken from shapes)
    (h0,w0), (ratio, pad) = shapes
    lb_vis  = cv2.resize(orig, (int(w0*ratio[0]), int(h0*ratio[1])))
    lb_vis  = cv2.copyMakeBorder(lb_vis, int(pad[1]), int(pad[1]),
                                           int(pad[0]), int(pad[0]),
                                           cv2.BORDER_CONSTANT, value=(114,114,114))
    save(draw_boxes(lb_vis, orig_boxes, (255,0,0)), save_root/'1_letterbox.jpg')

    print("\nSaved debug jpgs under", save_root.resolve())

if __name__ == '__main__':
    main()
