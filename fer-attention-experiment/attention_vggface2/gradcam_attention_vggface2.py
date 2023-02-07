import argparse
import torch
from torchvision import transforms
from utility.checkpoint import load_model
import torch.nn.functional as F
from attention_vggface2.resnet50.vggface2_gradcam import VGGFace2GradCAM
from attention_vggface2.stn.vggface2_stn import VGGFace2STN
from attention_vggface2.se.vggface2_se_gradcam import VGGFace2SEGradCAM
from attention_vggface2.bam.vggface2_bam_gradcam import VGGFace2BAMGradCAM
from attention_vggface2.cbam.vggface2_cbam_gradcam import VGGFace2CBAMGradCAM
from dataloader.affectnet import AffectNet
from dataloader.fer2013 import FER2013
from dataloader.rafdb import RAFDB
from dataloader.ferplus import FERPlus
from dataloader.elfemplus import ElfemPlus
from tqdm import tqdm, notebook
import numpy as np
import cv2
import os


parser = argparse.ArgumentParser(description="Configuration test phase")
parser.add_argument("-a", "--attention", type=str, default="no", choices=["no", "stn", "ran", "se", "bam", "cbam"], help='Chose the attention module')
parser.add_argument("-m", "--model", type=str, required=False, help='Path of the model')
parser.add_argument("-d", "--dataset", type=str, default="fer2013", choices=["fer2013", "fer2013clean", "rafdb", "rafdbaligned", "affectnet", "ferplus", "ferplus8", "elfemplus"], help='Chose the dataset')
# remove the batch size, must be 1
parser.add_argument("-bs", "--batch_size", type=int, default=1, help='Batch size to use for training')
parser.add_argument("-s", "--stats", type=str, default="vggface2", choices=["vggface2", "imagenet"], help='Chose the mean and standard deviation')
args = parser.parse_args()

print("Starting test with the following configuration:")
print("Attention module: {}".format(args.attention))
print("Model path: {}".format(args.model))
print("Dataset: {}".format(args.dataset))
print("Batch size: {}".format(args.batch_size))
print("Stats: {}".format(args.stats))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.attention == "no":
    model = VGGFace2GradCAM(pretrained=True).to(device)
elif args.attention == "stn":
    model = VGGFace2STN().to(device)
elif args.attention == "se":
    model = VGGFace2SEGradCAM().to(device)
elif args.attention == "bam":
    model = VGGFace2BAMGradCAM().to(device)
elif args.attention == "cbam":
    model = VGGFace2CBAMGradCAM().to(device)
else:
    model = VGGFace2GradCAM(pretrained=True).to(device)

if args.stats == "imagenet":
    # imagenet
    data_mean = [0.485, 0.456, 0.406]
    data_std = [0.229, 0.224, 0.225]
else:
    # vggface2
    data_mean = [131.0912 / 255, 103.8827 / 255, 91.4953 / 255]
    data_std = [0.5, 0.5, 0.5]

test_preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=data_mean, std=data_std),
])

inv_norm = transforms.Normalize((-data_mean[0] / data_std[0], -data_mean[1] / data_std[1], -data_mean[2] / data_std[2]),
                                (1 / data_std[0], 1 / data_std[1], 1 / data_std[2]))

if args.dataset == "rafdb":
    test_data = RAFDB(split="test", dataset=args.dataset, transform=test_preprocess)
    label_mapping = {0: "Surprise",
                     1: "Fear",
                     2: "Disgust",
                     3: "Happiness",
                     4: "Sadness",
                     5: "Anger",
                     6: "Neutral"}
elif args.dataset == "rafdbaligned":
    test_data = RAFDB(split="test", dataset=args.dataset, transform=test_preprocess)
    label_mapping = {0: "Surprise",
                     1: "Fear",
                     2: "Disgust",
                     3: "Happiness",
                     4: "Sadness",
                     5: "Anger",
                     6: "Neutral"}
elif args.dataset == "affectnet":
    test_data = AffectNet(split="val", transform=test_preprocess)
    # cambiare con mapping di affectnet
    label_mapping = {0: "Angry",
                     1: "Disgust",
                     2: "Fear",
                     3: "Happy",
                     4: "Sad",
                     5: "Surprise",
                     6: "Neutral"}
elif args.dataset == "ferplus":
    test_data = FERPlus(split="test", dataset=args.dataset, transform=test_preprocess, as_rgb=True)
    label_mapping = {0: "Neutral",
                     1: "Happy",
                     2: "Surprise",
                     3: "Sad",
                     4: "Angry",
                     5: "Disgust",
                     6: "Fear"}
elif args.dataset == "ferplus8":
    test_data = FERPlus(split="test", dataset=args.dataset, transform=test_preprocess, as_rgb=True)
    label_mapping = {0: "Neutral",
                     1: "Happy",
                     2: "Surprise",
                     3: "Sad",
                     4: "Angry",
                     5: "Disgust",
                     6: "Fear",
                     7: "Contempt"}
elif args.dataset == "elfemplus":
    test_data = ElfemPlus(split = "test", transform = test_preprocess, as_rgb = True)
    label_mapping = {0: "Angry",
                     1: "Disgust",
                     2: "Fear",
                     3: "Happy",
                     4: "Sad",
                     5: "Surprise",
                     6: "Neutral"}
else:
    test_data = FER2013(split="test", dataset=args.dataset, transform=test_preprocess, as_rgb=True)
    label_mapping = {0: "Angry",
                     1: "Disgust",
                     2: "Fear",
                     3: "Happy",
                     4: "Sad",
                     5: "Surprise",
                     6: "Neutral"}

test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

model = load_model(args.model, model, device)
model.eval()

num_correct = 0
num_samples = 0

batch_bar = tqdm(total=len(test_loader), desc="Batch", position=0)

os.makedirs("../result/gradcam/correct", exist_ok = True)
os.makedirs("../result/gradcam/wrong", exist_ok = True)

# with torch.no_grad():
for batch_idx, batch in enumerate(test_loader):
    images, labels, names = batch["image"].to(device), batch["label"].to(device), batch["name"]
    outputs = model(images)
    # preds = F.softmax(outputs, dim=1)
    # classes = torch.argmax(preds, 1)
    classes = torch.argmax(outputs, 1)

    # preds[:, classes].backward()
    outputs[:, classes].backward()
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim = [0, 2, 3])
    activations = model.get_activations(images).detach()
    for i in range(activations.size(1)):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim = 1).squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)

    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image_orig = inv_norm(images[0])
    image_orig = image_orig.permute(1, 2, 0).numpy()
    image_orig = np.uint8(255 * image_orig)
    image_orig = cv2.cvtColor(image_orig, cv2.COLOR_RGB2BGR)

    superimposed_img = cv2.addWeighted(image_orig, 0.6, heatmap, 0.4, 0.0)

    images_stacked = np.concatenate((image_orig, superimposed_img), axis=1)

    images_stacked = cv2.copyMakeBorder(images_stacked, 0, 55, 0, 0, cv2.BORDER_CONSTANT)

    cv2.putText(images_stacked, 'True label: {}'.format(label_mapping[labels[0].item()]), (5, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(images_stacked, 'Predicted: {}'.format(label_mapping[classes[0].item()]), (5, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 1, cv2.LINE_AA)

    if classes[0] == labels[0]:
        cv2.imwrite("../result/gradcam/correct/{}".format(names[0]), images_stacked)
    else:
        cv2.imwrite("../result/gradcam/wrong/{}".format(names[0]), images_stacked)

    num_correct += (classes == labels).sum()
    num_samples += classes.size(0)
    batch_bar.update(1)

print("Number of samples: {}".format(num_samples))
print("Correctly classified: {}".format(num_correct.item()))
print('Accuracy of the network on the test images: {}%'.format((num_correct.item()/num_samples*100)))
