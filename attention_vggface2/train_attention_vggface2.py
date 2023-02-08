import argparse
from multiprocessing import freeze_support

import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import itertools
from tqdm import tqdm, notebook
import numpy as np
# from attention_vggface2.resnet50.resnet50 import VGGFace2
# from utility.checkpoint import save_checkpoint
#from utility.checkpoint import *
# import attention_vggface2.resnet50.vggface2 as vggface2
from attention_vggface2.resnet50.vggface2 import VGGFace2
from attention_vggface2.stn.vggface2_stn import VGGFace2STN
from attention_vggface2.se.vggface2_se import VGGFace2SE
from attention_vggface2.bam.vggface2_bam import VGGFace2BAM
from attention_vggface2.cbam.vggface2_cbam import VGGFace2CBAM
from attention_vggface2.hybrid.vggface2_stnbam import VGGFace2STNBAM
from attention_vggface2.ab.vggface2_ab import VGGFace2AB
from attention_vggface2.bot.vggface2_bot import VGGFace2Bot, VGGFace2BotS1
from attention_vggface2.attention.vggface2_attention import VGGFace2AM
from attention_vggface2.sa.vggface2_sa import VGGFace2SA
from dataloader.affectnet import AffectNet
from dataloader.fer2013 import FER2013
from dataloader.rafdb import RAFDB
from dataloader.ferplus import FERPlus
from dataloader.elfemplus import ElfemPlus
from dataloader.rafdb_fer2013 import RAFDBFER2013
from dataloader.viv import VIV
from dataloader.fer2013mask import FER2013Mask
from dataloader.fer2013mouth import FER2013Mouth

if __name__ == '__main__':
    freeze_support()

    parser = argparse.ArgumentParser(description="Configuration train phase")
    parser.add_argument("-a", "--attention", type=str, default="bam", choices=["no", "stn", "ran", "se", "bam", "cbam", "stnbam", "ab", "bot", "bots1", "am", "sa"], help='Chose the attention module')
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help='Batch size to use for training')
    parser.add_argument("-o", "--optimizer", type=str, default="adam", choices=["adam", "sgd"], help='Chose the optimizer')
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-05, help='Learning rate to use for training')
    parser.add_argument("-e", "--epochs", type=int, default=2, help='Number of epochs')
    parser.add_argument("-p", "--patience", type=int, default=5, help='Number of epochs without improvements before reducing the learning rate')
    parser.add_argument("-wd", "--weight_decay", type=float, default=0, help='Value of weight decay')
    parser.add_argument("-nm", "--momentum", type=float, default=0, help='Value of momentum')
    parser.add_argument("-m", "--monitor", type=str, default="acc", choices=["acc", "loss"], help='Chose to monitor the validation accuracy or loss')
    parser.add_argument("-d", "--dataset", type=str, default="fer2013mouth", choices=["fer2013", "fer2013clean", "rafdb", "rafdbaligned", "affectnet", "ferplus", "ferplus8", "elfemplus", "rafdbfer2013", "viv", "fer2013mask", "fer2013nomask", "fer2013mouth"], help='Chose the dataset')
    parser.add_argument("-cw", "--class_weights", type=bool, default=False, help='Use the class weights in loss function')
    parser.add_argument("-s", "--stats", type=str, default="vggface2", choices=["vggface2", "imagenet"], help='Chose the mean and standard deviation')
    args = parser.parse_args()

    print("Starting training with the following configuration:")
    print("Attention module: {}".format(args.attention))
    print("Batch size: {}".format(args.batch_size))
    print("Optimizer: {}".format(args.optimizer))
    print("Learning rate: {}".format(args.learning_rate))
    print("Epochs: {}".format(args.epochs))
    print("Patience: {}".format(args.patience))
    print("Weight decay: {}".format(args.weight_decay))
    print("Momentum: {}".format(args.momentum))
    print("Class weights: {}".format(args.class_weights))
    print("Metric to monitor: {}".format(args.monitor))
    print("Dataset: {}".format(args.dataset))
    print("Stats: {}".format(args.stats))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

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

    train_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_mean, std=data_std),
    ])

    val_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_mean, std=data_std),
    ])

    if args.dataset == "rafdb":
        train_data = RAFDB(split="train", dataset=args.dataset, transform=train_preprocess)
        val_data = RAFDB(split="test", dataset=args.dataset, transform=val_preprocess)
        classes = 7
    elif args.dataset == "rafdbaligned":
        train_data = RAFDB(split="train", dataset=args.dataset, transform=train_preprocess)
        val_data = RAFDB(split="test", dataset=args.dataset, transform=val_preprocess)
        classes = 7
    elif args.dataset == "affectnet":
        train_data = AffectNet(split="train", transform=train_preprocess)
        val_data = AffectNet(split="val", transform=val_preprocess)
        classes = 8
    elif args.dataset == "ferplus":
        train_data = FERPlus(split="train", dataset=args.dataset, transform=train_preprocess, as_rgb=True)
        val_data = FERPlus(split="val", dataset=args.dataset, transform=val_preprocess, as_rgb=True)
        classes = 7
    elif args.dataset == "ferplus8":
        train_data = FERPlus(split="train", dataset=args.dataset, transform=train_preprocess, as_rgb=True)
        val_data = FERPlus(split="val", dataset=args.dataset, transform=val_preprocess, as_rgb=True)
        classes = 8
    elif args.dataset == "elfemplus":
        train_data = ElfemPlus(split = "train", transform = train_preprocess, as_rgb = True)
        val_data = ElfemPlus(split = "val", transform = val_preprocess, as_rgb = True)
        classes = 7
    elif args.dataset == "rafdbfer2013":
        train_data = RAFDBFER2013(split = "train", transform = train_preprocess)
        val_data = RAFDBFER2013(split = "val", transform = val_preprocess)
        classes = 7
    elif args.dataset == "viv":
        train_data = VIV(split = "train", transform = train_preprocess)
        val_data = VIV(split = "val", transform = val_preprocess)
        classes = 7
    elif args.dataset == "fer2013mask":
        train_data = FER2013Mask(split="train", masked=True, transform=train_preprocess, as_rgb=True)
        val_data = FER2013Mask(split="val", masked=True, transform=val_preprocess, as_rgb=True)
        classes = 7
    elif args.dataset == "fer2013nomask":
        train_data = FER2013Mask(split="train", masked=False, transform=train_preprocess, as_rgb=True)
        val_data = FER2013Mask(split="val", masked=False, transform=val_preprocess, as_rgb=True)
        classes = 7
    elif args.dataset == "fer2013mouth":
        train_data = FER2013Mouth(split="train", cropped_mouth=True, transform=train_preprocess, as_rgb=True)
        val_data = FER2013Mouth(split="val", cropped_mouth=True, transform=val_preprocess, as_rgb=True)
        classes = 7
    # if args.dataset == "fer2013" or args.dataset == "fer2013clean":
    else:
        train_data = FER2013(split="train", dataset=args.dataset, transform=train_preprocess, as_rgb=True)
        val_data = FER2013(split="val", dataset=args.dataset, transform=val_preprocess, as_rgb=True)
        classes = 7

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

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
    elif args.attention == "stnbam":
        model = VGGFace2STNBAM().to(device)
    elif args.attention == "ab":
        model = VGGFace2AB().to(device)
    elif args.attention == "bot":
        model = VGGFace2Bot(classes=classes).to(device)
    elif args.attention == "bots1":
        model = VGGFace2BotS1(classes=classes).to(device)
    elif args.attention == "am":
        model = VGGFace2AM(classes=classes).to(device)
    elif args.attention == "sa":
        model = VGGFace2SA(classes=classes).to(device)
    else:
        model = VGGFace2(pretrained=True, classes=classes).to(device)

    print("Model archticture: ", model)

    start_epoch = 0

    best_val_loss = 1000000
    best_val_acc = 0

    # aggiungere class_weights
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                              momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                               weight_decay=args.weight_decay)

    if args.monitor == "loss":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, verbose=True)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, mode="max", verbose=True)

    print("===================================Start Training===================================")

    for e in range(start_epoch, args.epochs):
        train_loss = 0
        validation_loss = 0
        train_correct = 0
        val_correct = 0
        is_best = False

        batch_bar = tqdm(total=len(train_loader), desc="Batch", position=0)

        # train the model
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()

            # without data augmentation
            outputs = model(images)
            loss = criterion(outputs, labels)

            # with data augmentation
            # bs, ncrops, c, h, w = images.size()
            # outputs = model(images.view(-1, c, h, w))
            # outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
            # loss = criterion(outputs_avg, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # without data augmentation
            _, preds = torch.max(outputs, 1)

            # with data augmentation
            # _, preds = torch.max(outputs_avg, 1)

            train_correct += torch.sum(preds == labels.data)
            batch_bar.update(1)

        # validate the model
        model.eval()
        for batch_idx, batch in enumerate(val_loader):
            images, labels = batch["image"].to(device), batch["label"].to(device)

            with torch.no_grad():
                # without data augmentation
                val_outputs = model(images)
                val_loss = criterion(val_outputs, labels)

                # with data augmentation
                # bs, ncrops, c, h, w = images.size()
                # val_outputs = model(images.view(-1, c, h, w))
                # val_outputs_avg = val_outputs.view(bs, ncrops, -1).mean(1)
                # val_loss = criterion(val_outputs_avg, labels)

                validation_loss += val_loss.item()

                _, val_preds = torch.max(val_outputs, 1)
                # _, val_preds = torch.max(val_outputs_avg, 1)

                val_correct += torch.sum(val_preds == labels.data)

        train_loss = train_loss/len(train_loader)
        train_acc = train_correct.double() / len(train_data)
        validation_loss = validation_loss / len(val_loader)
        val_acc = val_correct.double() / len(val_data)

        if args.monitor == "loss":
            scheduler.step(validation_loss)

            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                is_best = True
        else:
            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                is_best = True

        checkpoint = {
            'epoch': e + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        #save_checkpoint(checkpoint, is_best, "../result/{}/checkpoint".format(args.attention), "../result/{}".format(args.attention))

        f = open('../result/train_results.txt', 'a')

        if is_best:
            print(
                '\nEpoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Accuracy {:.3f}% \tValidation Accuracy {:.3f}% \t[saved]'
                .format(e + 1, train_loss, validation_loss, train_acc * 100, val_acc * 100))
            f.write('\nEpoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Accuracy {:.3f}% \tValidation Accuracy {:.3f}% \t[saved]'
                .format(e + 1, train_loss, validation_loss, train_acc * 100, val_acc * 100) + '\n')
        else:
            print(
                '\nEpoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Accuracy {:.3f}% \tValidation Accuracy {:.3f}%'
                .format(e + 1, train_loss, validation_loss, train_acc * 100, val_acc * 100))
            f.write('\nEpoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Accuracy {:.3f}% \tValidation Accuracy {:.3f}%'
                .format(e + 1, train_loss, validation_loss, train_acc * 100, val_acc * 100)+'\n')

        f.close()

    print("===================================Training Finished===================================")