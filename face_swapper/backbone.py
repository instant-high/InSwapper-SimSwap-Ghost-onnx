import os
import cv2
import onnxruntime
import numpy as np
from . utils import align_crop

class ArcFace:
    def __init__(self, model_file, *args, **kwargs):
        self.session = onnxruntime.InferenceSession(model_file, **kwargs)
        self.inputs = self.session.get_inputs()
        self.input_size = tuple(self.inputs[0].shape[2:4][::-1])

    def compute_similarity(self, feat1, feat2):
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
        return similarity

    def forward(self, img, kps):
        aimg, matrix = align_crop(img, kps, self.input_size[0], mode="arcface")
        blob = aimg.astype("float32") / 127.5 - 1
        blob = blob[:, :, ::-1].transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)
        out = self.session.run(None, {self.inputs[0].name: blob})[0]
        return out.ravel()