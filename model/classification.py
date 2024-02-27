import torch
import pandas as pd
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


def get_class_name(class_id):
    """
    Get class name by class id
    """

    df = pd.read_csv('data/sports.csv')
    return df.loc[df["class id"] == class_id].iloc[0]["labels"]


def single_image_predict(file):
    """
    Function to predict the label of a single image
    """

    num_labels = 100

    # define preprocess transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])  

    model = models.efficientnet_b1()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_labels, bias=True)
    best_model = torch.load('best_efficientnet.pth',  map_location=torch.device('cpu'))
    model.load_state_dict(best_model['model_state_dict'])
    loss_fn = nn.CrossEntropyLoss()

    image = Image.open(file)
    image = transform(image)
    image = torch.unsqueeze(image, 0)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
    output_label = torch.topk(outputs, 1)
    pred_class_id = int(output_label.indices)
    pred_class_name = get_class_name(pred_class_id)
    
    return  pred_class_name