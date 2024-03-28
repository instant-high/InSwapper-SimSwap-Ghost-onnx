
import cv2
import numpy
import onnxruntime

#FACE_MASK_REGIONS = (1, 2, 3, 4, 5, 6, 10, 11, 12, 13)
#FACE_MASK_REGIONS = (1, 2, 3, 4, 5, 10, 11, 12, 13)
class FACE_PARSER:
   
    def __init__(self, model_path="face_parser.onnx", device='cpu'):
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        if device == 'cuda':
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),"CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)
        self.resolution = self.session.get_inputs()[0].shape[-2:]


    def create_region_mask(self,crop_frame, FACE_MASK_REGIONS):
        prepare_frame = cv2.flip(cv2.resize(crop_frame, (512, 512)), 1)
        prepare_frame = numpy.expand_dims(prepare_frame, axis = 0).astype(numpy.float32)[:, :, ::-1] / 127.5 - 1
        prepare_frame = prepare_frame.transpose(0, 3, 1, 2)
        region_mask = self.session.run(None,{self.session.get_inputs()[0].name: prepare_frame})[0][0]

        region_mask = numpy.isin(region_mask.argmax(0), FACE_MASK_REGIONS)
        
        region_mask = cv2.resize(region_mask.astype(numpy.float32), crop_frame.shape[:2][::-1])

        return region_mask
