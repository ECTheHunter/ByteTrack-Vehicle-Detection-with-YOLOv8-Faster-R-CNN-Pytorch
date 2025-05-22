from ultralytics import YOLO
from collections import defaultdict
# Load the trained model

model = YOLO('runs/detect/train/weights/best.pt')
# Run tracking on the video
video_files = [
    '../videos/test1.mp4',
    '../videos/test2.mp4',
    # add more video paths here
]

for video_path in video_files:
    print(f"\nProcessing video: {video_path}")
    results = model.track(video_path, save=True, stream=True, tracker="bytetrack.yaml")
    
    id_tracker = defaultdict(set)
    
    for r in results:
        if r.boxes.id is not None:
            for cls, track_id in zip(r.boxes.cls, r.boxes.id):
                class_id = int(cls.item())
                object_id = int(track_id.item())
                id_tracker[class_id].add(object_id)
    
    print("=== Unique Object Counts ===")
    for class_id, ids in id_tracker.items():
        class_name = model.names[class_id]
        print(f"{class_name}: {len(ids)} unique objects")
