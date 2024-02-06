# run in llm4 : 3.10.13 
from tqdm import tqdm
import torch
import os, re
from torchvision.transforms import ToTensor, Normalize, Compose
from DatasetUtils.DataParsing import leafDataset  # Ensure correct import path
from modelTraining.modelDefinition import get_instance_segmentation_model
from tqdm import notebook
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os,re
from tqdm.notebook import tqdm,tnrange
from torchvision.transforms import PILToTensor, ToPILImage

from DatasetUtils.PlotUtils import plotResultsWithMasks, plotPairsWithMasks, plotLeafTimeSeries,plotExampleWithMasks
from DatasetUtils.DataParsing import leafDataset, findPairs_for_evaluation
from tracking.timeSeries import createMaskTimeSeries
from modelTraining.modelDefinition import get_instance_segmentation_model
from PIL import ImageDraw
from torchvision.transforms import ToPILImage
from cv2 import findContours, boundingRect, RETR_TREE, CHAIN_APPROX_SIMPLE

from tracking.pairing import *
import time
# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = get_instance_segmentation_model(2)  # Adjust number of output classes as necessary
state_dict = torch.load("Models/Leaf_Segmentation_MaskedRCNN_7_7_2022_8h.h5", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Setup directories and paths
images_dir = "Datasets/Mask_rcnn_validation_dataset/images"
annotations_dir = "Datasets/Mask_rcnn_validation_dataset/Annotations"
images = [file for file in os.listdir(images_dir) if file.endswith(('.png', '.jpg'))]
test_images = [os.path.join(images_dir, image) for image in images]
test_json = [os.path.join(annotations_dir, re.sub("\.png|\.jpg", ".json", image)) for image in images]

# Create dataset instance
test_dataset = leafDataset(test_images, test_json)

# Define transforms
transforms = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Evaluate dataset
results = []
for img_data in tqdm(test_dataset):
    img = img_data[0]
    if not isinstance(img, torch.Tensor):
        img_tensor = transforms(img).unsqueeze(0).to(device)
    else:
        img_tensor = img.unsqueeze(0).to(device) if img.dim() == 3 else img.to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
    
    output_cpu = [{k: v.to('cpu') for k, v in result.items()} for result in output]
    results.append(output_cpu)
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()

# Process ground truth and results for comparison
for i in range(len(test_dataset)):
    test_dataset[i][1]["Category"] = re.sub(".*\\\\|_.*", "", test_dataset.ids_dict[i])

ground_truth = [e[1] for e in test_dataset]

filtered_results = []
for e in results:
    temp_results = {
        "boxes": [],
        "labels": [],
        "scores": [],
        "masks": []
    }
    if e[0]["scores"].nelement() != 0:
        for idx, val in enumerate(e[0]["scores"]):
            if val >= 0.9:
                temp_results["boxes"].append(e[0]["boxes"][idx])
                temp_results["labels"].append(e[0]["labels"][idx])
                temp_results["scores"].append(e[0]["scores"][idx])
                temp_results["masks"].append(e[0]["masks"][idx][0])
        if temp_results["boxes"]:
            temp_results["boxes"] = torch.stack(temp_results["boxes"])
            temp_results["labels"] = torch.stack(temp_results["labels"])
            temp_results["scores"] = torch.stack(temp_results["scores"])
            temp_results["masks"] = torch.stack(temp_results["masks"])
    filtered_results.append(temp_results)

from tqdm import tqdm

all_pairs = []
all_notDetected = []
all_missDetection = []

# Correct the variable name to match its definition
for i in tqdm(range(len(ground_truth)), desc="Processing"):
    all_data = findPairs_for_evaluation(ground_truth[i], filtered_results[i])
    all_pairs.append(all_data[0])
    all_notDetected.append(all_data[1])
    all_missDetection.append(all_data[2])

early = []
mid = []
late = []
for e in all_pairs:
    for j in e:
        if j["type"] == "Early":
            early.append(j["IoU"])
        elif j["type"] == "Late":
            late.append(j["IoU"])
        else:
            mid.append(j["IoU"])

df = pd.DataFrame([early,mid,late]).transpose()
df.columns = ["Early","Mid","Late"]
df.describe()

earlyNotDetectedCounter = 0
midNotDetectedCounter = 0
lateNotDetectedCounter = 0
for i in all_notDetected:
    if len(i)> 0:
        if i[0]["type"] == "Early":
            earlyNotDetectedCounter += len(i)
        elif i[0]["type"] == "Late":
            lateNotDetectedCounter += len(i)
        else:
            midNotDetectedCounter += len(i)
print("""
Leafs not detected:
Early stage: {}
Mid stage: {}
Late stage: {}""".format(earlyNotDetectedCounter,midNotDetectedCounter,lateNotDetectedCounter))
earlymissDetectedCounter = 0
midmissDetectedCounter = 0
latemissDetectedCounter = 0
for i in all_missDetection:
    if len(i) > 0:
        if i[0]["type"] == "Early":
            earlymissDetectedCounter += len(i)
        elif i[0]["type"] == "Late":
            latemissDetectedCounter += len(i)
        else:
            midmissDetectedCounter += len(i)
print("""
Leafs errounisly detected:
Early stage: {}
Mid stage: {}
Late stage: {}""".format(earlymissDetectedCounter, midmissDetectedCounter, latemissDetectedCounter))

earlyPairs = 0
midPairs = 0
latePairs = 0
for i in all_pairs:
    if len(i) > 0:
        if i[0]["type"] == "Early":
            earlyPairs += len(i)
        elif i[0]["type"] == "Late":
            latePairs += len(i)
        else:
            midPairs += len(i)
print("""
Leafs correctly detected:
Early stage: {}
Mid stage: {}
Late stage: {}""".format(earlyPairs, midPairs, latePairs))

f1df = pd.DataFrame([[earlyPairs,midPairs,latePairs],
                    [earlyNotDetectedCounter,midNotDetectedCounter,lateNotDetectedCounter],
                    [earlymissDetectedCounter,midmissDetectedCounter,latemissDetectedCounter]]).transpose()
f1df.columns = ["Correct","FalseNegative","FalsePositive"]
f1df.index = ["Early","Mid","Late"]
f1df["Precision"] = f1df["Correct"]/(f1df["Correct"]+f1df["FalsePositive"])
f1df["Recall"] = f1df["Correct"]/(f1df["Correct"]+f1df["FalseNegative"])
f1df["F-score"] = (f1df["Precision"]*f1df["Recall"])/(f1df["Precision"]+f1df["Recall"])*2
f1df.to_excel("PrecisionAndRecall.xlsx")

df.to_excel("IoU_raw_values.xlsx")
df.describe().to_excel("IoU_statistics.xlsx")