import cv2
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
import argparse
import os

def detect_vehicles_in_image(image_path, output_dir="outputs"):
    print(f"\n--- Running YOLOv10x on image: {image_path} ---")
    model = YOLO("yolov10x.pt")

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)
    detections = results[0].to_df()

    vehicle_classes = ['car', 'truck', 'bus', 'motorbike']
    vehicles = detections[detections['name'].isin(vehicle_classes)].copy()
    
    if not vehicles.empty:
        vehicles[['xmin', 'ymin', 'xmax', 'ymax']] = vehicles['box'].apply(lambda box: pd.Series({
            'xmin': box['x1'], 'ymin': box['y1'], 'xmax': box['x2'], 'ymax': box['y2']
        }))
        vehicles.drop(columns=['box'], inplace=True)

        for _, row in vehicles.iterrows():
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            label = row['name']
            confidence = row['confidence']
            text = f"{label} {confidence:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"detected_{base_name}")
    cv2.imwrite(output_path, img) 
    print(f"Result saved to: {output_path}")

def detect_vehicles_in_video(video_path, output_dir="outputs"):
    print(f"\n--- Running YOLOv8n on video: {video_path} ---")
    model = YOLO("yolov8n.pt")
    vehicle_classes = ['car', 'truck', 'bus', 'motorbike']

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width, height = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    base_name = os.path.basename(video_path)
    output_path = os.path.join(output_dir, f"detected_{base_name}")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        for box in results.boxes.data:
            x1, y1, x2, y2, score, cls_id = box.cpu().numpy()
            class_name = model.names[int(cls_id)]
            if class_name in vehicle_classes:
                label = f"{class_name} {score:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Video processing complete. Result saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Vehicle Detection")
    parser.get_params = argparse.ArgumentParser(description="YOLO Vehicle Detection")
    parser.add_argument("--source", type=str, required=True, help="Path to image or video file")
    parser.add_argument("--type", type=str, choices=["image", "video"], default="image", help="Type of source (image or video)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save results")
    
    args = parser.parse_args()
    
    if args.type == "image":
        detect_vehicles_in_image(args.source, args.output_dir)
    else:
        detect_vehicles_in_video(args.source, args.output_dir)
