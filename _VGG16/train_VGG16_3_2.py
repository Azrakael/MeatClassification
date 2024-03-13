#%%
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import pickle

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# 라벨 파일 로드
labels_df = pd.read_csv('../processed_224/labels.csv')

# 데이터셋 경로 설정
base_dir = './dataset'

models_dir = '../models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

# 이미지 데이터 생성기 생성
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 의미: 이미지를 1/255로 스케일링
    shear_range=0.2,  # 의미: 이미지를 0.2만큼
    zoom_range=[0.8, 1.2],  # 의미: 이미지를 0.8배에서 1.2배 크기로 확대
    horizontal_flip=True,  # 의미: 이미지를 수평으로 뒤집기
    rotation_range=30,  # 의미: 이미지를 30도 회전
    width_shift_range=0.1,  # 의미: 이미지를 0.1만큼 수평으로 이동
    height_shift_range=0.1, # 의미: 이미지를 0.1만큼 수직으로 이동
    brightness_range=[0.8, 1.2], # 의미: 이미지를 0.8배에서 1.2배 크기로 밝기 조절
    fill_mode='nearest' # 의미: 이미지를 회전하거나 이동할 때 생기는 공간을 가장 가까운 값으로 채우기
)

val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

# 데이터셋 생성기 생성
train_generator, val_generator, test_generator = [
    datagen.flow_from_dataframe(
        dataframe=labels_df,
        directory=dir_,
        x_col='Image Name',
        y_col='Label',
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical',
        seed=42
    ) for datagen, dir_ in zip([train_datagen, val_datagen, test_datagen], [train_dir, val_dir, test_dir])]

# 모델 생성
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-6]: # 4 >> 6 마지막 6개의 레이어만 학습
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(2048, activation='relu', kernel_regularizer=l2(0.001))(x)  
x = BatchNormalization()(x)
x = Dropout(0.7)(x) # 0.5 >> 0.7
predictions = Dense(3, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

# 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
model_checkpoint = ModelCheckpoint(
    filepath=os.path.join(models_dir, 'classification_VGG16_BM_3_2.keras'),
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1)

# 모델 훈련
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,     
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

model.save(os.path.join(models_dir, 'classification_VGG16_3_2.keras'))

with open(os.path.join(models_dir, 'classification_VGG16_3_2_history.pkl'), 'wb') as file:
    pickle.dump(history.history, file)

# 모델 평가
val_loss, val_accuracy = model.evaluate(val_generator)
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Validation loss: {val_loss}, Validation accuracy: {val_accuracy}')
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')
# %%
# 