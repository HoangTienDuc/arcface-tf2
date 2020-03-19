import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

from absl import app, flags, logging
from absl.flags import FLAGS
# import cv2
from modules.evaluations import get_val_data, perform_val
from modules.models import ArcFaceModel
from modules.utils import set_memory_growth, load_yaml, l2_norm

cfg = load_yaml('./configs/arc_res50.yaml')

model = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         training=False)
ckpt_path = tf.train.latest_checkpoint('./checkpoints/')
print('ckpt_path: ', ckpt_path)
model.load_weights(ckpt_path)

export_path = './frozen'
tf.keras.models.save_model(
    model,
    export_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)
