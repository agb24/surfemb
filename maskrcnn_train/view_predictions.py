import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.utils import make_grid
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.io import read_image
import torchvision.transforms.functional as F

import torchviz_utils.utils as utils
import torchviz_utils.transforms as T
from torchviz_utils.coco_eval import CocoEvaluator
from torchviz_utils.coco_utils import get_coco_api_from_dataset
import rgbmaskdataset_v1 as MaskDataLoader

from datetime import datetime as dt
import time, os, cv2
import numpy as np
import matplotlib.pyplot as plt


dir_nums=[0]
num_classes=31
data_split_train=0.75
root_path = "D:/Akshay_Work/aks_git_repos/Akshays_project"

coco_format_path = os.path.join(root_path, "coco_apiformat_20_test.pt")
train_test_ind_file = os.path.join(root_path, "train_test_indices.pt")
trained_model_path = os.path.join(root_path, "trained_models", "run-5-2_fol-0_model-std_CONT_lr-red-1e-5_150_model.pt")


def get_model(number_classes=31):
    # Load model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT",
                                                                 trainable_backbone_layers=1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Set RoI predictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, number_classes)
    # Get num features for Mask Classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Set Mask Predictor 
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       number_classes)
    return model

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def eval_view(model, epoch, data_loader, device, dir_nums=[20], tb_writer=None):
    def show(imgs):
        plt.switch_backend("TkAgg")
        if not isinstance(imgs, list):
            imgs = [imgs]
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            plt.show()

    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    
    # Convert model to model.module if DDP
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    i = 0
    for images, targets in data_loader:
        t_loop = dt.now()
        images = list(img.to(device) for img in images)
        #targets = list(trg.to(device) for trg in targets)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()

        # Get evaluation metrics for IoU
        model.eval()
        outputs = model(images)

        # outputs[0] is a dict with keys: ['boxes', 'labels', 'scores', 'masks']
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        if i == len(data_loader)-2: print(f"Loop {i} took: ", dt.now() - t_loop)
        i += 1

        # Display images and masks
        # Scale image back to [0,255]
        img = images[0]*255
        img = img.to(torch.uint8)
        boxes = outputs[0]["boxes"]
        masks = outputs[0]["masks"]
        scores = outputs[0]["scores"] > 0.8
        
        valid_boxes = boxes.to(torch.int32, copy=True)   # boxes[scores.T].to(torch.uint8)
        valid_masks = masks.to(torch.float32, copy=True)
        valid_masks = valid_masks.squeeze()
        valid_masks = valid_masks >= 0.5
        #valid_masks = valid_masks*127

        final_preds = torchvision.ops.box_convert(valid_boxes, in_fmt="xywh", out_fmt="xyxy")
        '''final_preds = torch.clone(valid_boxes)
        final_preds[:,2] = torch.add(valid_boxes[:,0], valid_boxes[:,2])
        final_preds[:,3] = torch.add(valid_boxes[:,1], valid_boxes[:,3])'''

        result = draw_bounding_boxes(img, boxes=final_preds, width=5)
        show(result)

        res_seg = draw_segmentation_masks(img.to("cpu"), masks=valid_masks, alpha=0.5)
        show(res_seg)
        






if __name__ == "__main__":
    
    # Set fixed random number seed & device ID
    torch.manual_seed(42)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create dataset with transformations
    '''dataset = MaskDataLoader.RgbMasksDataset(dir_nums = dir_nums,
                                            transforms = get_transform(train=True))'''


    dataset_test = MaskDataLoader.RgbMasksDataset(dir_nums = dir_nums,
                                            transforms = get_transform(train=False))
    

    # Split dataset in to test/train sets
    '''indices = torch.randperm(len(dataset)).tolist()'''
    indices = torch.load(train_test_ind_file)

    train_size = int(data_split_train * len(dataset_test))
    '''dataset = torch.utils.data.Subset(dataset, indices[:train_size])'''
    dataset_test = torch.utils.data.Subset(dataset_test, indices[train_size:])

    # Create dataloaders for test & train data
    '''data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size_train, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)'''
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)
    print("TEST DATASET_SIZE::: -------- :::", len(data_loader_test)*1)
    '''for i,t in data_loader_test:
        im = cv2.cvtColor(i[0].numpy().reshape(540,720,3), cv2.COLOR_RGB2BGR)
        cv2.imshow("a", im)
        cv2.waitKey(10)
    '''
    # ----------LOAD PRETAINED MODEL
    checkpoint = torch.load(trained_model_path)
    # Initialize the Model
    '''model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=False,
                        num_classes=num_classes, pretrained_backbone=True,
                        trainable_backbone_layers=3)
    '''
    # get the model using our helper function
    model = get_model(num_classes)
    model.to(device)
    model.eval()


    # Initialize the optimizer & scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    #optimizer = torch.optim.SGD(params, lr=0.00005,
    #                            momentum=0.9, weight_decay=0.05)
    optimizer = torch.optim.Adam(params, lr=0.001, 
                                betas=(0.9,0.999), weight_decay=0.05)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    # ----------EVALUATE on the test dataset
    eval_start = dt.now()
    coco_evaluator, eval_metric_logger = eval_view(model=model,
                                                epoch=50, 
                                                data_loader=data_loader_test, 
                                                device=device,                                                
                                                dir_nums=dir_nums)
    print("@@@@@@@@@@@@@@ Eval Metrics: @@@@@@@@@@@@", eval_metric_logger)
    print("@@@@@@@@@@@@@@ Eval Took: @@@@@@@@@@@@", dt.now() - eval_start)