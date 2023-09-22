import cv2
import os

video_path = '/media/kevin/Ubuntu/Frames-video/[EMBER] Kimetsu no Yaiba - Mugen Ressha-hen - 07.mkv'

output_dir = '/media/kevin/Ubuntu/Frames-video/Frames'


os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Initialize a frame counter
frame_count = 0

# Read frames from the video and save them as image files
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Save the frame as an image file (e.g., frame_001.jpg, frame_002.jpg, ...)
    frame_filename = os.path.join(output_dir, f'frame_{frame_count:03d}.jpg')
    cv2.imwrite(frame_filename, frame)
    
    frame_count += 1

# Release the video capture object
cap.release()

print(f'{frame_count} frames extracted and saved to {output_dir}')
