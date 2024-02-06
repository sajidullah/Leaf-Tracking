import torch
import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

# Your custom functions to load the model
from modelTraining.modelDefinition import get_instance_segmentation_model

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = get_instance_segmentation_model(2)  # Adjust the number of output classes as necessary
model_path = "Models/Leaf_Segmentation_MaskedRCNN_7_7_2022_8h.h5"
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Define image transforms
transforms = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def visualize_prediction(image, prediction, threshold=0.5):
    """
    Visualizes the prediction by overlaying masks on the original image.
    """
    image = Image.fromarray(image.mul(255).permute(1, 2, 0).byte().numpy())
    draw = ImageDraw.Draw(image)
    
    for element in range(len(prediction["masks"])):
        mask = prediction["masks"][element, 0]
        if prediction["scores"][element] > threshold:
            mask = mask.mul(255).byte().cpu().numpy()
            mask_image = Image.fromarray(mask)
            image.paste(mask_image, (0,0), mask_image)
    return image

# Setup directories
images_dir = r"E:\Sajid_PSI_Projects\output"  # Update this path
output_dir = r"E:\Sajid_PSI_Projects\results"  # Specify where to save output images

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process images and save predictions
for image_name in tqdm(os.listdir(images_dir)):
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue  # Skip non-image files
    
    image_path = os.path.join(images_dir, image_name)
    img = Image.open(image_path).convert("RGB")
    img_tensor = transforms(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(img_tensor)
    
    # Visualize the prediction
    vis_image = visualize_prediction(img_tensor.squeeze(0), prediction[0])
    
    # Save the visualized prediction to the output directory
    output_path = os.path.join(output_dir, f"processed_{image_name}")
    vis_image.save(output_path)
