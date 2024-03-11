# %%
import cv2
import os

base_dir = './MeatData'
sub_dirs = ['beef(304)', 'lamb(253)', 'pork(290)']

for sub_dir in sub_dirs:
    full_dir = os.path.join(base_dir, sub_dir)
    
    for img_name in os.listdir(full_dir):
        img_path = os.path.join(full_dir, img_name)
        
        img = cv2.imread(img_path)
        
        # 이미지가 로드되지 않으면 삭제
        if img is None:
            print(f"Deleting corrupted or unsupported image: {img_path}")
            os.remove(img_path)
