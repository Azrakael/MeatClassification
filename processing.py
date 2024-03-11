#%%
#전처리
import cv2
import os
import csv

# 경로 설정
processed_224_dir = './processed_224'
processed_299_dir = './processed_299'
labels_dirs = ['./processed_224', './processed_299']
base_dir = './MeatData'

sub_dirs = ['beef(304)', 'lamb(253)', 'pork(290)']

for directory in [processed_224_dir, processed_299_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# 라벨 파일 생성
for labels_dir, size in zip(labels_dirs, [224, 299]):
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    labels_file_path = os.path.join(labels_dir, 'labels.csv')
    with open(labels_file_path, 'w', newline='', encoding='utf-8') as labels_file:
        writer = csv.writer(labels_file)
        writer.writerow(['Image Name', 'Label'])

        for sub_dir in sub_dirs:
            label = sub_dir.split('(')[0]
            full_dir = os.path.join(base_dir, sub_dir)

            for img_name in os.listdir(full_dir):
                img_path = os.path.join(full_dir, img_name)
                
                # OpenCV를 사용하여 이미지 로드
                img = cv2.imread(img_path)
                
                # 이미지가 정상적으로 로드되었는지 확인
                if img is None:
                    print(f"Warning: '{img_path}' 하자있음. 이미지를 건너뜁니다.")
                    continue  # 이미지를 건너뛰고 다음으로 진행
                
                # 이미지 리사이징
                img_resized = cv2.resize(img, (size, size))

                # 이미지 저장 경로 설정
                if size == 224:
                    save_dir = processed_224_dir
                else:
                    save_dir = processed_299_dir
                
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                save_path = os.path.join(save_dir, img_name)
                
                # OpenCV를 사용하여 이미지 저장
                cv2.imwrite(save_path, img_resized)

                writer.writerow([img_name, label])
