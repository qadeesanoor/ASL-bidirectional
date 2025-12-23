import cv2
import os
import sys
import numpy as np
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# Path to your ASL video dataset
video_dir = "F:\\asl\\dataset\\videos"

# Map letters to indices
label_map = {chr(i+97): i for i in range(26)}  # a=0, b=1, ..., z=25

# Fixed window size
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Crop region (adjust based on your video content)
CROP_X = 150
CROP_Y = 50
CROP_WIDTH = 600
CROP_HEIGHT = 600


class TextToASLVideo:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_files = {}
        self.load_video_paths()
    
    def load_video_paths(self):
        """Load video file paths for each letter"""
        print("Loading video paths...")
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        
        for letter in label_map.keys():
            for ext in video_extensions:
                # Check lowercase
                video_file = os.path.join(self.video_path, f"{letter.lower()}{ext}")
                if os.path.exists(video_file):
                    self.video_files[letter.upper()] = video_file
                    break
                # Check uppercase
                video_file = os.path.join(self.video_path, f"{letter.upper()}{ext}")
                if os.path.exists(video_file):
                    self.video_files[letter.upper()] = video_file
                    break
        
        print(f"Loaded {len(self.video_files)} video files")
        if self.video_files:
            print(f"Available letters: {', '.join(sorted(self.video_files.keys()))}")
    
    def crop_frame(self, frame):
        """Crop the frame to a fixed region and resize to window"""
        # Crop
        cropped = frame[CROP_Y:CROP_Y+CROP_HEIGHT, CROP_X:CROP_X+CROP_WIDTH]
        # Resize cropped frame to window
        resized = cv2.resize(cropped, (FRAME_WIDTH, FRAME_HEIGHT))
        return resized
    
    def play_video(self, video_path, letter):
        """Play a single video file with cropping"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30
        delay = int(1000 / fps)
        
        print(f"Playing: {letter}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Crop and resize frame
            frame = self.crop_frame(frame)
            cv2.imshow('ASL Video Player', frame)
            
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return False
        
        cap.release()
        return True
    
    def text_to_videos(self, text):
        """Convert text to sequence of ASL videos"""
        text = text.upper()
        valid_chars = []
        
        for char in text:
            if char in self.video_files:
                valid_chars.append(char)
            elif char == ' ':
                if 'SPACE' in self.video_files:
                    valid_chars.append('SPACE')
                else:
                    print("Note: SPACE video not found, skipping spaces")
            elif not char.isalnum():
                continue
        
        if not valid_chars:
            print("No valid characters to display!")
            print(f"Available letters: {', '.join(sorted(self.video_files.keys()))}")
            return
        
        print(f"\nConverting text: '{text}'")
        print(f"Will play {len(valid_chars)} videos")
        print("Press 'q' to stop at any time\n")
        
        cv2.namedWindow('ASL Video Player', cv2.WINDOW_NORMAL)
        
        for i, char in enumerate(valid_chars):
            if char in self.video_files:
                video_path = self.video_files[char]
                print(f"[{i+1}/{len(valid_chars)}] ", end='')
                continue_playing = self.play_video(video_path, char)
                if not continue_playing:
                    print("\nPlayback stopped by user")
                    break
            else:
                print(f"Video not available for: {char}")
        
        cv2.destroyAllWindows()
        print("\n✓ Conversion complete!")
    
    def list_available_letters(self):
        """Display all available letters"""
        if self.video_files:
            letters = sorted(self.video_files.keys())
            print(f"\nAvailable letters ({len(letters)}):")
            print(', '.join(letters))
        else:
            print("\nNo video files found!")
            print(f"Make sure videos are in: {self.video_path}")
            print("Expected format: a.mp4, b.mp4, c.mp4, etc.")


# ============= USAGE EXAMPLES =============

converter = TextToASLVideo(video_dir)
converter.list_available_letters()

user_input = input()
print("\n" + "="*50)
print("="*50)
converter.text_to_videos(user_input)
