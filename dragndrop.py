import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
import os
import torch
import torchvision.models as models
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader
mean = torch.tensor([0.1152,0.1152,0.1152])
std = torch.tensor([0.1977,0.1977,0.1977])



class SingleImageDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return 1  # Only one image

    def __getitem__(self, idx):
        image = Image.open(self.image_path).convert('RGB')  # Convert to RGB
        if self.transform:
            image = self.transform(image)
        return image

class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, 1, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 12, 3, 1, 2)
        self.fc1 = nn.Linear(12 * 57 * 57, 32)
        self.fc2 = nn.Linear(32, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 12 * 57 * 57)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class StackedModel(nn.Module):
    def __init__(self):
        super(StackedModel, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.googlenet = models.googlenet(pretrained=True)
        self.baseline = BaselineModel()

        num_features_resnet50 = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features_resnet50, 5)

        num_features_googlenet = self.googlenet.fc.in_features
        self.googlenet.fc = nn.Linear(num_features_googlenet, 5)




    def forward(self, x):

        resnet_output = self.resnet50(x)
        googlenet_output = self.googlenet(x)
        baseline_output = self.baseline(x)

        averaged_output = (resnet_output + googlenet_output + baseline_output) / 3
        return averaged_output


model = StackedModel()
model_path = 'C:/Users/alanf/Desktop/aps360project/stacked_model_64_15_0.0005.pth'
model.load_state_dict(torch.load(model_path))


def evaluate(image_path):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std), 
    ])
    single_image_dataset = SingleImageDataset(image_path, transform=transform)


    single_image_loader = torch.utils.data.DataLoader(single_image_dataset, batch_size=1) 
    for img in single_image_loader:   
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        img.to(device)
        with torch.no_grad():
            prediction = model(img)
            predicted_label = torch.argmax(prediction, axis=1)
    return predicted_label.item()

def drop(event):
    file_path = event.data
    if os.path.isfile(file_path):
        label = evaluate(file_path)
        predicted_label.config(text=f"predicted label: {label}")

        

root = TkinterDnD.Tk()
root.title("BIRAD DETECTOR")

prompt_label = tk.Label(root, text="Drag and drop your image here", padx=50, pady=20)
prompt_label.pack()

predicted_label = tk.Label(root, text="File path will appear here", padx=50, pady=20)
predicted_label.pack()

prompt_label.drop_target_register(DND_FILES)
prompt_label.dnd_bind('<<Drop>>', drop)

root.mainloop()



