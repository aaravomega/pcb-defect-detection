import argparse
from ultralytics import YOLO

def predict(image_path, weights_path, conf=0.25):
    model = YOLO(weights_path)
    results = model.predict(source=image_path, conf=conf, save=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = result.names[int(box.cls)]
            confidence = float(box.conf)
            print(f"Detected: {cls} (confidence: {confidence:.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--weights", default="weights/best.pt", help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    predict(args.image, args.weights, args.conf)
