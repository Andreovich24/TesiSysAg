import argparse
import torch
from torchvision import transforms
from utility.checkpoint import load_model
import torch.nn.functional as F
from attention_vggface2.resnet50.vggface2 import VGGFace2
from attention_vggface2.stn.vggface2_stn import VGGFace2STN
from attention_vggface2.se.vggface2_se import VGGFace2SE
from attention_vggface2.bam.vggface2_bam import VGGFace2BAM
from attention_vggface2.cbam.vggface2_cbam import VGGFace2CBAM
from attention_vggface2.sa.vggface2_sa import VGGFace2SA
from dataloader.affectnet import AffectNet
from dataloader.fer2013 import FER2013
from dataloader.rafdb import RAFDB
from dataloader.ferplus import FERPlus
from dataloader.elfemplus import ElfemPlus
from dataloader.rafdb_fer2013 import RAFDBFER2013
from dataloader.viv import VIV
from dataloader.fer2013mask import FER2013Mask
from tqdm import tqdm, notebook
from utility.confusion_matrix import show_confusion_matrix, get_classification_report



parser = argparse.ArgumentParser(description="Configuration test phase")
parser.add_argument("-a", "--attention", type=str, default="no", choices=["no", "stn", "ran", "se", "bam", "cbam", "sa"], help='Chose the attention module')
parser.add_argument("-m", "--model", type=str, required=False, help='Path of the model')
parser.add_argument("-d", "--dataset", type=str, default="fer2013", choices=["fer2013", "fer2013clean", "rafdb", "rafdbaligned", "affectnet", "ferplus", "ferplus8", "elfemplus", "rafdbfer2013", "valrafdb-fer2013", "rafdb-valfer2013", "rafdb-testfer2013", "viv", "fer2013mask", "fer2013nomask"], help='Chose the dataset')
parser.add_argument("-bs", "--batch_size", type=int, default=64, help='Batch size to use for training')
parser.add_argument("-s", "--stats", type=str, default="vggface2", choices=["vggface2", "imagenet"], help='Chose the mean and standard deviation')
args = parser.parse_args()

print("Starting test with the following configuration:")
print("Attention module: {}".format(args.attention))
print("Model path: {}".format(args.model))
print("Dataset: {}".format(args.dataset))
print("Batch size: {}".format(args.batch_size))
print("Stats: {}".format(args.stats))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# if args.attention == "no":
#     model = VGGFace2(pretrained=True).to(device)
# elif args.attention == "stn":
#     model = VGGFace2STN().to(device)
# elif args.attention == "se":
#     model = VGGFace2SE().to(device)
# elif args.attention == "bam":
#     model = VGGFace2BAM().to(device)
# elif args.attention == "cbam":
#     model = VGGFace2CBAM().to(device)
# else:
#     model = VGGFace2(pretrained=True).to(device)

if args.stats == "imagenet":
    # imagenet
    data_mean = [0.485, 0.456, 0.406]
    data_std = [0.229, 0.224, 0.225]
else:
    # vggface2
    # data_mean = [131.0912 / 255, 103.8827 / 255, 91.4953 / 255]
    # data_std = [0.5, 0.5, 0.5]
    data_mean = [91.4953 / 255, 103.8827 / 255, 131.0912 / 255]
    data_std = [1., 1., 1.]

test_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=data_mean, std=data_std),
])

if args.dataset == "rafdb":
    test_data = RAFDB(split="test", dataset=args.dataset, transform=test_preprocess)
    label_mapping = {0: "Surprise",
                     1: "Fear",
                     2: "Disgust",
                     3: "Happiness",
                     4: "Sadness",
                     5: "Anger",
                     6: "Neutral"}
    classes = 7
elif args.dataset == "rafdbaligned":
    test_data = RAFDB(split="test", dataset=args.dataset, transform=test_preprocess)
    label_mapping = {0: "Surprise",
                     1: "Fear",
                     2: "Disgust",
                     3: "Happiness",
                     4: "Sadness",
                     5: "Anger",
                     6: "Neutral"}
    classes = 7
elif args.dataset == "affectnet":
    test_data = AffectNet(split="val", transform=test_preprocess)
    label_mapping = {0: "Angry",
                     1: "Disgust",
                     2: "Fear",
                     3: "Happy",
                     4: "Sad",
                     5: "Surprise",
                     6: "Neutral"}
    classes = 7
elif args.dataset == "ferplus":
    test_data = FERPlus(split = "test", dataset=args.dataset, transform = test_preprocess, as_rgb = True)
    label_mapping = {0: "Neutral",
                     1: "Happy",
                     2: "Surprise",
                     3: "Sad",
                     4: "Angry",
                     5: "Disgust",
                     6: "Fear"}
    classes = 7
elif args.dataset == "ferplus8":
    test_data = FERPlus(split = "test", dataset=args.dataset, transform = test_preprocess, as_rgb = True)
    label_mapping = {0: "Neutral",
                     1: "Happy",
                     2: "Surprise",
                     3: "Sad",
                     4: "Angry",
                     5: "Disgust",
                     6: "Fear",
                     7: "Contempt"}
    classes = 8
elif args.dataset == "elfemplus":
    test_data = ElfemPlus(split = "test", transform = test_preprocess, as_rgb = True)
    label_mapping = {0: "Angry",
                     1: "Disgust",
                     2: "Fear",
                     3: "Happy",
                     4: "Sad",
                     5: "Surprise",
                     6: "Neutral"}
    classes = 7
elif args.dataset == "rafdbfer2013":
    test_data = RAFDBFER2013(split = "val", transform = test_preprocess)
    label_mapping = {0: "Angry",
                     1: "Disgust",
                     2: "Fear",
                     3: "Happy",
                     4: "Sad",
                     5: "Surprise",
                     6: "Neutral"}
    classes = 7
elif args.dataset == "valrafdb-fer2013":
    test_data = RAFDBFER2013(split = "valrafdb", transform = test_preprocess)
    label_mapping = {0: "Angry",
                     1: "Disgust",
                     2: "Fear",
                     3: "Happy",
                     4: "Sad",
                     5: "Surprise",
                     6: "Neutral"}
    classes = 7
elif args.dataset == "rafdb-valfer2013":
    test_data = RAFDBFER2013(split = "valfer2013", transform = test_preprocess)
    label_mapping = {0: "Angry",
                     1: "Disgust",
                     2: "Fear",
                     3: "Happy",
                     4: "Sad",
                     5: "Surprise",
                     6: "Neutral"}
    classes = 7
elif args.dataset == "rafdb-testfer2013":
    test_data = RAFDBFER2013(split = "test", transform = test_preprocess)
    label_mapping = {0: "Angry",
                     1: "Disgust",
                     2: "Fear",
                     3: "Happy",
                     4: "Sad",
                     5: "Surprise",
                     6: "Neutral"}
    classes = 7
elif args.dataset == "viv":
    test_data = VIV(split = "test", transform = test_preprocess)
    label_mapping = {0: "Boredom",
                     1: "Confusion",
                     2: "Enthusiasm",
                     3: "Frustration",
                     4: "Interest",
                     5: "Neutral",
                     6: "Surprise"}
    classes = 7
elif args.dataset == "fer2013mask":
    test_data = FER2013Mask(split="test", masked=True, transform=test_preprocess, as_rgb=True)
    label_mapping = {0: "Angry",
                     1: "Disgust",
                     2: "Fear",
                     3: "Happy",
                     4: "Sad",
                     5: "Surprise",
                     6: "Neutral"}
    classes = 7
elif args.dataset == "fer2013nomask":
    test_data = FER2013Mask(split="test", masked=False, transform=test_preprocess, as_rgb=True)
    label_mapping = {0: "Angry",
                     1: "Disgust",
                     2: "Fear",
                     3: "Happy",
                     4: "Sad",
                     5: "Surprise",
                     6: "Neutral"}
    classes = 7
else:
    test_data = FER2013(split="test", dataset=args.dataset, transform=test_preprocess, as_rgb=True)
    label_mapping = {0: "Angry",
                     1: "Disgust",
                     2: "Fear",
                     3: "Happy",
                     4: "Sad",
                     5: "Surprise",
                     6: "Neutral"}
    classes = 7

# test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

if args.attention == "no":
    model = VGGFace2(pretrained=True, classes=classes).to(device)
elif args.attention == "stn":
    model = VGGFace2STN(classes=classes).to(device)
elif args.attention == "se":
    model = VGGFace2SE(classes=classes).to(device)
elif args.attention == "bam":
    model = VGGFace2BAM(classes=classes).to(device)
elif args.attention == "cbam":
    model = VGGFace2CBAM(classes=classes).to(device)
elif args.attention == "sa":
    model = VGGFace2SA(classes=classes).to(device)
else:
    model = VGGFace2(pretrained=True, classes=classes).to(device)

model = load_model(args.model, model, device)
model.eval()

num_correct = 0
num_samples = 0

y_true = []
y_pred = []

batch_bar = tqdm(total=len(test_loader), desc="Batch", position=0)

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        # images, labels = batch["image"].to(device), batch["label"].to(device)
        images, labels, image_name = batch["image"].to(device), batch["label"].to(device), batch["name"]
        outputs = model(images)
        preds = F.softmax(outputs, dim=1)
        classes = torch.argmax(preds, 1)
        num_correct += (classes == labels).sum()
        num_samples += classes.size(0)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(classes.detach().cpu().numpy().tolist())
        batch_bar.update(1)

print('Accuracy of the network on the test images: {}%'.format((num_correct.item()/num_samples*100)))

labels_list = []
for i in range(len(label_mapping)):
    labels_list.append(label_mapping[i])

show_confusion_matrix(y_true, y_pred, labels_list)
print(get_classification_report(y_true, y_pred, labels_list))

