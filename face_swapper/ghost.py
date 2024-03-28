import cv2
import onnxruntime
import numpy as np

class Ghost:
    def __init__(self, model_file, *args, **kwargs):
        self.session = onnxruntime.InferenceSession(model_file, **kwargs)
        self.align_crop_size = 224
        self.align_crop_mode = "set2"

    def forward(self, target, source_embedding):
        latent = source_embedding.reshape(1, -1)

        blob = cv2.resize(target, (256, 256))
        blob = blob.astype("float32") / 127.5 - 1
        blob = blob[:, :, ::-1].transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)

        blob = self.session.run(None, {"target": blob, "source_embedding": latent})[0]

        out = blob[0].transpose(1, 2, 0)
        out = out * 127.5 + 127.5
        out = out[:, :, ::-1]
        out = out.astype(np.uint8)

        return out
