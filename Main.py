import cv2
import time
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Set up the configuration
cfg = get_cfg()
cfg.merge_from_file(r"C:\Users\user\Project_Real_Time_OUHANDS_5000_0.001_Choice1 valid 400 test 1000 - Copy\output\custom_config.yaml")
cfg.MODEL.WEIGHTS = r"C:\Users\user\Project_Real_Time_OUHANDS_5000_0.001_Choice1 valid 400 test 1000 - Copy\output\model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
predictor = DefaultPredictor(cfg)

# Define class mapping (replace with your actual class names)
class_mapping = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "H", 7: "I", 8: "J", 9: "K"}

# Open a connection to the webcam (you can change the index if you have multiple cameras)
cap = cv2.VideoCapture(0)

frame_rate = 1

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error reading frame from the webcam.")
        break

    outputs = predictor(frame)
    class_labels = outputs["instances"].pred_classes.cpu().numpy()
    scores = outputs["instances"].scores.cpu().numpy()

    v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    instances = outputs["instances"].to("cpu")
    instances = instances[instances.scores > cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST]
    frame = v.draw_instance_predictions(instances).get_image()

    for i, (label, score) in enumerate(zip(class_labels, scores)):
        class_label = class_mapping[label]
        cv2.putText(frame, f"{class_label}: {score:.2f}", (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Webcam", frame[:, :, ::-1])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
