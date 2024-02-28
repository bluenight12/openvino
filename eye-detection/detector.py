import cv2

class Detector:
    def __init__(self, compiled_model):
        self.model = compiled_model
        self.input_layer_ir = self.model.input(0)
        self.output_layer_ir = self.model.output(0)
        self.N, self.C, self.H, self.W = self.input_layer_ir.shape
        self.output_shape = self.output_layer_ir.shape

    def detect(self, frame):
        frame = cv2.resize(frame, (self.W, self.H))
        # height, width, channel -> channel, height, width
        frame = frame.transpose((2, 0, 1))
        # channel, height, width -> batch, channel, height, width
        frame = frame.reshape(
            (self.N, self.C, self.H, self.W)).astype('float32')

        return self.model(frame)[self.output_layer_ir]
