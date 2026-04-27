import torch
#################################### For Image ####################################
from PIL import Image
import pandas as pd
import glob
import json 
import os
import cv2
import numpy as np
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score


# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)

# Load images
full_image_paths = glob.glob( "/scratch/project_2001382/data/shared/zebra/images/*.jpg")
#last component of the file path
#image_paths = os.path.basename(full_image_paths)
#image = cv2.imread(image_paths)
#image_paths =glob.glob( "/images/*.jpg")
#image = Image.open(image_paths)
image_paths = full_image_paths[:1600]

#inference_state = processor.set_image(image)
# Prompt the model with text
#output = processor.set_text_prompt(state=inference_state, prompt="animal")
# Get the masks, bounding boxes, and scores
#masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

#Ground_truth_annotations = glob.glob("/scratch/project_2001382/data/shared/zebra/annotations/GZCD_gt.json")

thresholds = 0.9
SAM_results = []
for image_path in image_paths:
    try:
        image = Image.open(image_path)
    except:
        print("Invalid image with 0.00 B:", image_path)
        continue    
    
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt="animal")
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
 
    detection=len(scores)
    if detection > 0:
        has_animal = "True"
    else:
        has_animal = "False" 
    
    image_path = os.path.basename(image_path)
    print("image_path:", image_path)
    print("has_animal_SAM:", has_animal)
    print("Detection:", len(scores))
    print("Localization:", boxes)
    print("Segmentation:", masks)

    SAM_results.append({"image_path": image_path, "Detection": len(scores), "has_animal_SAM": has_animal, "Localization": boxes, "Segmentation": masks})
    images_amount = len(SAM_results)
    print("Number of images:", images_amount)

#pandas DataFrame table
df_sam = pd.DataFrame(SAM_results)
df_sam =df_sam.drop_duplicates()
df_sam.to_csv("SAM_results.csv", index=False)
print("SAM_results.csv")

#Load the JSON data
with open('/scratch/project_2001382/data/shared/zebra/annotations/GZCD_gt.json', 'r') as f:
    data = json.load(f)

print("working1!")

#df_gt = pd.DataFrame('data'['images',{'image_path', 'has_animal'}])
df_gt = pd.DataFrame(data['images'])
print("Ground truth table:", df_gt)

#the needed columns
df_gt_cul = df_gt[['image_path', 'has_animal']]
images = cv2.imread(image_path)

df_sam_pred = pd.read_csv("SAM_results.csv", usecols=['image_path', 'Detection', 'has_animal_SAM'])
print("SAM results table:", df_sam_pred)
print(df_sam_pred.dtypes)
print("Ground truth needed columns:", df_gt_cul)
print(df_gt_cul.dtypes)

print("working2!") 

columns_merge = pd.merge(df_sam_pred, df_gt_cul, how="inner", on="image_path")
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None) #it's working, just uncomment it

print("\nComperation table", columns_merge)

print("working3!")

#print(columns_merge[['image_path', 'Detection', 'has_animal', 'has_animal_SAM']])
columns_merge[['image_path', 'Detection', 'has_animal', 'has_animal_SAM']]
print("\nComperation table", columns_merge)

print("working4!")

#=np.array([])
#actual = np.array(['has_animal']) #ground truth annotaions
#predicted = np.array(["has_animal_SAM"])
 
#Convert the DataFrame to a NumPy array.
actual = columns_merge['has_animal'].to_numpy() #ground truth annotaions
predicted = columns_merge ["has_animal_SAM"].to_numpy()

#actual = columns_merge['has_animal'] #ground truth annotaions
#predicted = columns_merge ["has_animal_SAM"] 

# Print the confusion matrix and accuracy (number of correct predictions/ total predictions)
cm = confusion_matrix(actual, predicted)
print("Confusion matrix: ", cm)
accuracy = accuracy_score(actual, predicted)
print("Accuracy: ", accuracy)

#creating confusio report from the results
print(classification_report(actual, predicted))


print("working5!")


