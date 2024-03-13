#%%
import tensorflow as tf

# TensorFlow 버전 확인
print("TensorFlow version:", tf.__version__)

# GPU 사용 가능 여부 확인
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print("GPU devices available:", gpu_devices)
    for device in gpu_devices:
        print("Device:", device.name, "Type:", device.device_type)
else:
    print("No GPU devices available.")

# %%
conda create -n tf pip python==3.9.12
# %%
