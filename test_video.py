import os
import cv2
import numpy as np
import shutil
import subprocess

def main():
    print("=== EdgeVision Video Encoding Test ===")
    
    # Check for FFmpeg
    sys_ffmpeg = shutil.which("ffmpeg")
    if not sys_ffmpeg:
        print("❌ ERROR: FFmpeg is not installed on this system!")
        print("Please run: sudo apt install ffmpeg")
        return
        
    print(f"✅ Found FFmpeg at: {sys_ffmpeg}")
    
    # 1. Create temporary frames
    frames_dir = "temp_test_frames"
    os.makedirs(frames_dir, exist_ok=True)
    
    output_video = "test_playable_output.mp4"
    if os.path.exists(output_video):
        os.remove(output_video)
        
    print("-> Generating 30 test frames (1 second of video)...")
    try:
        for i in range(1, 31):
            # Create a green frame with frame number
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (0, 150, 0) # Green background
            cv2.putText(frame, f"TEST FRAME {i}", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            # Save frame
            cv2.imwrite(os.path.join(frames_dir, f"frame_{i:06d}.jpg"), frame)
            
        # 2. Run the exact FFmpeg command used in pipeline.py
        print("-> Stitching frames together using FFmpeg...")
        
        result = subprocess.run([
            sys_ffmpeg, "-y", "-framerate", "30",
            "-start_number", "1",
            "-i", os.path.join(frames_dir, "frame_%06d.jpg"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", output_video
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"\n✅ SUCCESS! A perfectly playable web-compatible video was created at: {output_video}")
            print("Download this file and test playing it. If it plays, your main pipeline will work 100%!")
        else:
            print(f"\n❌ ERROR: FFmpeg failed to encode the video!")
            print(f"FFmpeg Output:\n{result.stderr}")
            
    except Exception as e:
        print(f"❌ Script crashed: {e}")
    finally:
        # Cleanup temp frames
        shutil.rmtree(frames_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
