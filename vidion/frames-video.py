import cv2
import os
import re

# Define the path to your frames and the output video file
frames_folder = '/media/kevin/Ubuntu/Frames-video/Frames'
output_video = 'output_video.mkv'

# Get the list of frame filenames
frame_files = [f for f in os.listdir(frames_folder) if f.endswith('.jpg')]  # Adjust the file extension if needed

# Sort the frame filenames based on their numeric part (e.g., frame001.jpg, frame002.jpg, ...)
frame_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

# Get the dimensions of the first frame to set the video size
first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
height, width, _ = first_frame.shape


fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Create a VideoWriter object
out = cv2.VideoWriter(output_video, fourcc, 30.0, (width, height))  # Adjust the frame rate (30.0) as needed

# Iterate through frames and write them to the video
for frame_file in frame_files:
   frame = cv2.imread(os.path.join(frames_folder, frame_file))
   out.write(frame)

# Release the VideoWriter and close any open windows
out.release()
cv2.destroyAllWindows()
