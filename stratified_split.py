import os
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

# 사용자 설정
random_seed = 42
train_ratio = 0.8
img_list_path = r"D:/AUE8088/datasets/kaist-rgbt/train-all-04.txt"
labels_dir = r"D:/AUE8088/datasets/kaist-rgbt/train/labels/visible"
train_txt_out = r"D:/AUE8088/datasets/kaist-rgbt/train-split.txt"
val_txt_out = r"D:/AUE8088/datasets/kaist-rgbt/val-split.txt"

random.seed(random_seed)

# 라벨 파일에서 가장 첫 번째 클래스 ID를 기준으로 stratify
def get_first_label_class(label_file):
    try:
        with open(label_file, "r") as f:
            for line in f:
                if line.strip():
                    return int(line.strip().split()[0])
        return -1  # 빈 파일
    except:
        return -1

# 이미지 리스트 로딩
with open(img_list_path, "r") as f:
    img_paths = [line.strip() for line in f if line.strip()]

# 클래스 라벨 매핑
label_class_map = []
valid_imgs = []

for img_path in img_paths:
    base = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(labels_dir, base + ".txt")

    if not os.path.exists(label_path):
        continue  # 라벨이 없는 경우 제외

    cls_id = get_first_label_class(label_path)
    if cls_id == -1:
        continue  # 빈 라벨은 제외 (또는 포함하고 싶다면 로직 변경)

    valid_imgs.append(img_path)
    label_class_map.append(cls_id)

# Stratified split
train_imgs, val_imgs = train_test_split(
    valid_imgs,
    train_size=train_ratio,
    stratify=label_class_map,
    random_state=random_seed,
)

# 저장
with open(train_txt_out, "w") as f:
    f.write("\n".join(train_imgs))

with open(val_txt_out, "w") as f:
    f.write("\n".join(val_imgs))

print(f"✅ Stratified split complete: {len(train_imgs)} train / {len(val_imgs)} val")
