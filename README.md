# Weapon Detection Model (Gun & Knife) using YOLOv8

This project detects weapons (**guns & knives**) from CCTV footage and images using **YOLOv8**.  
It is trained on a custom dataset from [Roboflow](https://universe.roboflow.com/shivam-1iztu/weapon-detection-cctv-v3-dataset-zthmt/dataset/1) and includes a **Streamlit web app** for real-time detection.

---

## Project Structure

Weapon-Detection-Model
1. Streamlit web app/ # Web app
   1. app.py # Streamlit detection app
   2. requirements.txt # Dependencies for the app

2. data/ 
   1. data.yaml # YOLO dataset config
   2. download_dataset.py # Script to fetch dataset from Roboflow

3. notebook/ # Training notebook
   1. Weapon detection training code.ipynb

4. result/ # Outputs
   1. Weapon Detection working video.mp4
   2. best (12).pt # Trained YOLOv8 model weights

5. README.md
6. gitignore


---

## Dataset
- **Source:** [Roboflow Weapon Detection CCTV Dataset](https://universe.roboflow.com/shivam-1iztu/weapon-detection-cctv-v3-dataset-zthmt/dataset/1)
- **Classes:**  
  - `Gun`
  - `Knife`

---

## Install Dependencies

pip install -r "Streamlit web app/requirements.txt"


python data/download_dataset.py


cd "Streamlit web app"
streamlit run app.py

1.Upload an Image: Get the detected image with bounding boxes.

2.Upload a Video: Process video frames and download the annotated output.



Future Improvements:-
 1. Deploy on Streamlit Cloud.

 2. Integrate DeepSORT for real-time tracking.

 3. Optimize for edge devices.
