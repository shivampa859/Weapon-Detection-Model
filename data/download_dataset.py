!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="596FbDkAxGkP3w9QvBcT")
project = rf.workspace("shivam-1iztu").project("weapon-detection-cctv-v3-dataset-zthmt")
version = project.version(1)
dataset = version.download("yolov8")
                
