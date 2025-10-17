from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_name="yolov8n.pt", conf=0.25):
        self.model = YOLO(model_name)
        self.conf = conf

    def detect_person(self, frame_bgr):
        # Returns best person bbox (x1,y1,x2,y2) or None
        results = self.model.predict(source=frame_bgr, conf=self.conf, verbose=False)
        if not results or len(results) == 0:
            return None
        r = results[0]
        best = None
        best_area = 0
        for b in r.boxes:
            cls = int(b.cls[0].item())
            # COCO 'person' == 0
            if cls != 0:
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best = (int(x1), int(y1), int(x2), int(y2))
        return best
