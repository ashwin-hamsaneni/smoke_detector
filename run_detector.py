import os
import sys
from collections import defaultdict
from ultralytics import YOLO

def download_weights():
    """Ensures the pre-trained weights are downloaded from Hugging Face."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        import subprocess
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download
    
    print("Verifying model weights...")
    return hf_hub_download(repo_id="kittendev/YOLOv8m-smoke-detection", filename="best.pt")

def run_industrial_monitor(source_video, output_dir):
    # 1. Load the model
    weights_path = download_weights()
    model = YOLO(weights_path)

    print(f"\n[SYSTEM] Starting continuous monitoring on: {source_video}")
    print("[SYSTEM] Aggressive tracking enabled (Conf: 0.05). Temporal filtering active.\n")

    # 2. Configure Tracking Parameters
    # ByteTrack is excellent for keeping track of low-confidence, faint objects
    results = model.track(
        source=source_video,
        conf=0.05,               # Ultra-low threshold for initial ignition phases
        iou=0.30,                # Tight overlap threshold
        imgsz=960,               # High resolution to catch pixel-level anomalies
        tracker="bytetrack.yaml",# Uses Ultralytics' built-in ByteTrack algorithm
        stream=True,             # Memory-safe generator for continuous video feeds
        show=False,              # No GUI window to prevent system crashes
        save=True,               # Automatically compiles an output video with bounding boxes
        project=output_dir,
        name="industrial_smoke_alerts",
        exist_ok=True
    )

    # 3. Initialize Temporal Filtering Variables
    # Dictionary to count how many frames a specific tracked object has existed
    track_history = defaultdict(int)
    
    # REQUIREMENT: An object must be detected in this many frames before triggering a real alarm.
    # Adjust this based on your camera's FPS (e.g., 10 frames = ~0.3 seconds of continuous smoke).
    FRAMES_REQUIRED_FOR_ALARM = 10 
    
    # Keep track of IDs we've already sent an alert for, so we don't spam the console
    alerted_ids = set()

    # 4. Process the stream frame-by-frame
    for frame_idx, result in enumerate(results):
        boxes = result.boxes
        
        # If there are no detections in this frame, continue
        if boxes is None or len(boxes) == 0:
            continue
            
        for box in boxes:
            # We must have a tracking ID to perform temporal filtering
            if box.id is None:
                continue
                
            track_id = int(box.id[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            
            # Check if the detected object is smoke
            if "smoke" in class_name.lower():
                # Increment the frame lifespan of this specific smoke plume
                track_history[track_id] += 1
                
                # Check if it has met our temporal threshold AND hasn't been alerted yet
                if track_history[track_id] >= FRAMES_REQUIRED_FOR_ALARM and track_id not in alerted_ids:
                    coords = box.xyxy[0].tolist()
                    coords_formatted = [round(c, 1) for c in coords]
                    
                    print(f"🚨 [CONFIRMED ALARM] Early Smoke verified!")
                    print(f"   ┣━ Plume ID: #{track_id}")
                    print(f"   ┣━ Initial Confidence: {confidence:.2f}")
                    print(f"   ┣━ Location: {coords_formatted}")
                    print(f"   ┗━ Verified over {FRAMES_REQUIRED_FOR_ALARM} frames. Triggering factory protocols...\n")
                    
                    # Mark this plume as alerted so we don't spam the logs
                    alerted_ids.add(track_id)
                    
                    # ---> INSERT YOUR ACTUAL HARDWARE/SOFTWARE ALERT LOGIC HERE <---
                    # e.g., requests.post("https://factory-api/alert", data={"zone": 1})

    print(f"\n[SYSTEM] Video processing complete.")
    print(f"[SYSTEM] Output saved to: {os.path.join(output_dir, 'industrial_smoke_alerts')}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Point this to your test video or RTSP IP Camera stream
    TEST_VIDEO_PATH = r"C:\Users\HP\Downloads\gettyimages-1064208156-640_adpp.mp4" 
    OUTPUT_DIRECTORY = r"C:\Users\HP\Downloads"
    
    run_industrial_monitor(TEST_VIDEO_PATH, OUTPUT_DIRECTORY)