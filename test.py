import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from pycoral.adapters import common
from pycoral.utils import edgetpu
from pycoral.adapters import detect


CORAL_COLOR = (86, 104, 237)

model = "./examples/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"

interpreter = edgetpu.make_interpreter(model)
interpreter.allocate_tensors()


def get_objects(frame, threshold=0.01):

    height, width, _ = frame.shape
    _, scale = common.set_resized_input(interpreter, (width, height),
                                        lambda size: cv2.resize(frame, size, fx=0, fy=0, interpolation=cv2.INTER_CUBIC))
    interpreter.invoke()
    return detect.get_objects(interpreter, threshold, scale)


def draw_objects(frame, objs, labels=None, color=CORAL_COLOR, thickness=5):

    for obj in objs:
        bbox = obj.bbox
        cv2.rectangle(frame, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax),
                      color, thickness)
        if labels: 
            cv2.putText(frame, labels.get(obj.id), (bbox.xmin + thickness, bbox.ymax - thickness),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=CORAL_COLOR, thickness=2)

    # faces = detector.get_objects(frame, threshold=0.1)
    # vision.draw_objects(frame, faces)


frame = cv2.imread("people.jpg")
face = get_objects(frame)
draw_objects(frame, face)

cv2.imshow(frame)
