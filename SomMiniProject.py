import cv2
import tflite_runtime.interpreter as tflite
import numpy as np

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

DETECTION_MODEL_PATH = 'DetectionModel.tflite'
CLASSIFICATION_MODEL_PATH = 'ClassificationModel.tflite'

colors = [(47,255,173), (255,255,0), (71,99,255), (211,85,186), (180,105,255),
(0,215,255), (87,139,46), (255,105,65), (45,82,160), (205,250,255)]

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:,:] = image

def get_output_tensor(interpreter, index):
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor

def detect_soms(interpreter, image, thd=0.9):
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
        if scores[i] >= thd:
            results.append({'box': boxes[i]})
        else:
            break

    return results

def classify_som(interpreter, image, top_k=1):
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    if output_details['dtype']==np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale*(output-zero_point)

    ordered = np.argpartition(-output, top_k)

    return [(i, output[i]) for i in ordered[:top_k]]

def main():
    modelDetect = tflite.Interpreter(DETECTION_MODEL_PATH)
    modelDetect.allocate_tensors()

    modelClassify = tflite.Interpreter(CLASSIFICATION_MODEL_PATH)
    modelClassify.allocate_tensors()

    _, input_height_detect, input_width_detect,_ = modelDetect.get_input_details()[0]['shape']
    _, input_height_classify, input_width_classify,_ = modelClassify.get_input_details()[0]['shape']

    favor_txt = 'loading...'

    while 1:
        _, frame = cap.read()
        frameshow = frame.copy()

        #detection part
        imgrz = cv2.resize(frame.copy()[:,:,[2,1,0]], (input_height_detect, input_width_detect))

        bboxes = detect_soms(modelDetect, imgrz)

        if len(bboxes)==0:
            favor_txt = 'loading...'

        for i in range(len(bboxes)):
            bb = bboxes[i]['box']
            ymin, xmin, ymax, xmax = bb
            xmin = int(xmin*CAMERA_WIDTH)
            ymin = int(ymin*CAMERA_HEIGHT)
            xmax = int(xmax*CAMERA_WIDTH)
            ymax = int(ymax*CAMERA_HEIGHT)

            #image preparation part
            imgcrop = frame.copy()[ymin:ymax, xmin:xmax, [2,1,0]]
            imgcrop = cv2.resize(imgcrop, (input_height_classify, input_width_classify))

            #prediction part
            favor = classify_som(modelClassify, imgcrop)

            if favor[0][0]==0 and favor[0][1]>=0.8:
                favor_txt = 'sweet'
            elif favor[0][0]==1 and favor[0][1]>=0.8:
                favor_txt = 'sour'
#            else:
#                favor_txt = 'not sure'

            frameshow = cv2.rectangle(frameshow, (xmin, ymin), (xmax, ymax), colors[i%10], 2)
            frameshow = cv2.putText(frameshow, favor_txt, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, colors[i%10], 2)

        cv2.imshow('somsom', frameshow)

        if cv2.waitKey(1)==27:
            cv2.destroyAllWindows()
            break

if __name__=='__main__':
    main()
