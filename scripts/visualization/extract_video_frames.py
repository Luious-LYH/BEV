#!/usr/bin/env python3
"""
extract_video_frames.py — Extract high-resolution frame sequences from perception videos
for interactive frame-by-frame scrubber player in the textbook.
"""

import os
import cv2

VIDEOS = [
    {
        "src": "artifacts/learning_report/videos/occupancy_temporal_flow.mp4",
        "out_dir": "artifacts/learning_report/videos/frames/occ_flow",
        "prefix": "frame"
    },
    {
        "src": "artifacts/learning_report/videos/bev_tracking_scene0061.mp4",
        "out_dir": "artifacts/learning_report/videos/frames/bev_tracking",
        "prefix": "frame"
    },
    {
        "src": "artifacts/learning_report/videos/temporal_ego_alignment.mp4",
        "out_dir": "artifacts/learning_report/videos/frames/ego_alignment",
        "prefix": "frame"
    }
]

def extract():
    for item in VIDEOS:
        src = item["src"]
        out_dir = item["out_dir"]
        prefix = item["prefix"]
        os.makedirs(out_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"Failed to open {src}")
            continue
            
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out_file = os.path.join(out_dir, f"{prefix}_{count:03d}.jpg")
            # Save as high-quality compressed JPEG (quality=88)
            cv2.imwrite(out_file, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            count += 1
            
        cap.release()
        print(f"✓ Extracted {count} frames from {src} -> {out_dir}")

if __name__ == "__main__":
    extract()
