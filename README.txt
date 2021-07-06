					TURORIAL

1. Run directly (must have CUDA installed), full functionalities and better performance:
	1.1. Install the packages in requirements.txt located in the program's folder.
	Open anaconda prompt (miniconda) and change dir to Pytorch_Retinaface-master :
		conda create -n retina python = 3.7
		conda activate retina
		pip install -r requirements.txt
		
		
	1.2. Run face_count_single_image.py to Count faces on an image.
	1.3. Run face_count_webcam.py to Count faces with webcam:
	+ (IMPORTANT) Click on the window 'COUNTING...' and  press 'q' on your keyboard after you finished or your saved video will be corrupted.
	+ Final video and .txt file is stored in RESULT_VIDEO and RESULT_TEXT
	+ Size of input from webcam: 1920x1080, frame rate: 15 FPS; video_capture.set(3, 1920) and video_capture.set(4, 1080), the higher the better result but can't exceed your webcam resolution.
	+ Date's format in .txt file: dd/mm/yyyy




2. Using google colab (powerful computer from google will host this operation so installing packages isn't required):
	2.1. Method #1: 	
	https://colab.research.google.com/drive/1ITrhDPQkRSWSM_fBwobpFrjwpHNLdmMn?usp=sharing
	
	2.2. Mothod #2:
	https://colab.research.google.com/drive/1DBaA0kCpYQF6Px7fmED5oeFNoNfxMAcJ?usp=sharing


https://drive.google.com/drive/folders/10i9nQFdj_PCGublIe0F6TSw3Fh8JV23W?usp=sharing