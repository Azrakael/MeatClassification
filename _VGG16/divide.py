#%%
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil

# CSV 파일 로드
labels_df = pd.read_csv('../processed_224/labels.csv')

# 새로운 폴더 경로 설정
dataset_dir = './dataset'
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'valid')
test_dir = os.path.join(dataset_dir, 'test')

# 필요한 폴더 생성
for folder in [train_dir, val_dir, test_dir]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 각 클래스별로 데이터를 분할하는 함수
def split_data(df, source_dir, destination_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    # 클래스별 이미지 카운트를 저장할 딕셔너리
    counts = {'train': {'beef': 0, 'lamb': 0, 'pork': 0},
                'valid': {'beef': 0, 'lamb': 0, 'pork': 0},
                'test': {'beef': 0, 'lamb': 0, 'pork': 0}}

    # train, val, test로 분할
    train_df, temp_df = train_test_split(df, train_size=train_ratio, random_state=42)
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)  # 전체 대비 val 세트의 비율 조정
    val_df, test_df = train_test_split(temp_df, train_size=val_ratio_adjusted, random_state=42)

    # 이미지 파일을 각 폴더로 이동하고 카운트 업데이트
    for df, subfolder in zip([train_df, val_df, test_df], ['train', 'valid', 'test']):
        subfolder_path = os.path.join(destination_dir, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
        for _, row in df.iterrows():
            img_name = row['Image Name']
            # 라벨을 가져옴
            label = row['Label']  
            src_path = os.path.join(source_dir, img_name)
            dst_path = os.path.join(subfolder_path, img_name)
            shutil.copy(src_path, dst_path)
            # 라벨에 해당하는 카운트를 업데이트
            counts[subfolder][label] += 1  

    # 결과 출력
    for folder, folder_counts in counts.items():
        print(f"{folder} folder:")
        for label, count in folder_counts.items():
            print(f"  {label}: {count} images")

# 원본 이미지 파일이 있는 폴더 경로 설정
source_dir = '../processed_224'

# 각 클래스명에 대해 split_data 함수 호출
split_data(labels_df, source_dir, dataset_dir)

# %%
