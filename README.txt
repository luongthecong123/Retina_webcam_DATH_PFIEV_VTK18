- Run face_count_single_image.py to Count faces on an image.
- Run face_count_webcam.py to Count faces with webcam:
	+ (IMPORTANT) Click on the window 'COUNTING...' and  press 'q' on your keyboard after you finished or your saved video will be corrupted.
	+ Final video and .txt file is stored in RESULT_VIDEO and RESULT_TEXT
	+ Final video: 1280x720, frame rate: 15 FPS; out = cv2.VideoWriter('./RESULT_VIDEO/result.mp4', fourcc, 15, (1280,720))
	+ Size of input from webcam: 1920x1080, frame rate: 15 FPS; video_capture.set(3, 1920) and video_capture.set(4, 1080), the higher the better result but can't exceed your webcam resolution.
	+ Frame rate from Final video and input from webcam should match.
	+ Date's format: dd/mm/yyyy