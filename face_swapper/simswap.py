import cv2
import onnxruntime
import numpy as np


class SimSwap:
    def __init__(self, model_file, *args, **kwargs):
        self.session = onnxruntime.InferenceSession(model_file, **kwargs)
        self.crop_size = self.session.get_inputs()[0].shape[-2:]
        self.align_crop_size = 224
        self.align_crop_mode = "arcface"
        if self.crop_size[0] == 512:
            self.align_crop_size = 512
            self.align_crop_mode = "ffhq"

    def forward(self, target, source_embedding):
        latent = source_embedding.reshape(1, -1)
        latent /= np.linalg.norm(latent)

        blob = cv2.resize(target, self.crop_size)
        blob = blob.astype("float32") / 255
        blob = blob[:, :, ::-1]
        blob = np.expand_dims(blob, axis=0).transpose(0, 3, 1, 2)

        blob = self.session.run(None, {"target": blob, "source_embedding": latent})[0]

        out = blob[0].transpose((1, 2, 0))
        out = (out * 255).clip(0, 255)
        out = out.astype("uint8")[:, :, ::-1]

        return out


class SimSwapUnofficial:
    def __init__(self, model_file, *args, **kwargs):
        self.session = onnxruntime.InferenceSession(model_file, **kwargs)
        self.crop_size = self.session.get_inputs()[0].shape[-2:]
        self.align_crop_size = 512
        self.align_crop_mode = "arcface"

    def forward(self, target, source_embedding):
        latent = source_embedding.reshape(1, -1)
        latent /= np.linalg.norm(latent)

        blob = cv2.resize(target, self.crop_size)
        blob = blob.astype("float32") / 127.5 - 1
        blob = blob[:, :, ::-1].transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)

        blob = self.session.run(None, {"target": blob, "source_embedding": latent})[0]

        out = blob[0].transpose(1, 2, 0)
        out = out * 127.5 + 127.5
        out = out[:, :, ::-1].clip(0, 255)
        out = out.astype(np.uint8)

        return out
