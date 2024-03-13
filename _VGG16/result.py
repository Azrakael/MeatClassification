#%%
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.font_manager as fm
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

fontpath = 'C:\Windows\Fonts\malgun.ttf'  
font = fm.FontProperties(fname=fontpath).get_name()
plt.rcParams['font.family'] = font

# 저장된 모델 불러오기
model = tf.keras.models.load_model('../models/classification_VGG16_6.keras')

# history 데이터 로드
history = pd.read_pickle('../models/classification_VGG16_6_history.pkl')

# 학습 및 검증 손실 그래프
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 학습 및 검증 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# 테스트 데이터셋을 위한 이미지 데이터 생성기
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=pd.read_csv('../processed_224/labels.csv'),
    directory='./dataset/test', 
    x_col='Image Name',
    y_col='Label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

# 모델을 사용하여 예측
test_generator.reset()
predictions = model.predict(test_generator, steps=test_generator.n // test_generator.batch_size + 1)
predicted_classes = np.argmax(predictions, axis=1)

# 실제 레이블
true_classes = test_generator.classes

# 클래스 레이블
class_labels = list(test_generator.class_indices.keys())

# 혼동 행렬 생성
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# 혼동 행렬 시각화
plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 분류 보고서
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
# %%
