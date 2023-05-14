import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.utils import make_grid
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.io import read_image
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as F

import torchviz_utils.utils as utils
import torchviz_utils.transforms as T

from datetime import datetime as dt
import time, os, cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

dir_nums=[0]
num_classes= 31    # 6
root_path = "D:\\Akshay_Work\\aks_git_repos\\surfemb\\maskrcnn_train"
trained_model_path = os.path.join(root_path, "trained_models", "run-5-2_fol-0_model-std_CONT_lr-red-1e-5_150_model.pt")
        # "run_train_motor_no_3_3_cont_20_model.pt"
        # "run-5-2_fol-0_model-std_CONT_lr-red-1e-5_150_model.pt" 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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

    # Load the pre-trained model
    checkpoint = torch.load(trained_model_path)
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

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def show(imgs):
        #plt.switch_backend("TkAgg")
        if not isinstance(imgs, list):
            imgs = [imgs]
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            plt.show()


def get_raw_logits(orig_img, mask_lgts, index, boxes, labels):
    from torchvision.models.detection.roi_heads import paste_mask_in_image
    # Get probabilities from logits ----> This is 28x28
    prob_mask_lgts = mask_lgts.sigmoid()
    img_size = orig_img.shape

    list_mask_logits = []
    for ind, label in enumerate(labels):
        # Project Mask to desired BBoex shape; then place on original shape of the full image
        img_mask = paste_mask_in_image(mask = prob_mask_lgts[ind, label,:,:], 
                                    box = boxes[ind], 
                                    im_h = img_size[1], 
                                    im_w = img_size[2])
        full_size_mask_logits = img_mask.logit()
        list_mask_logits.append(full_size_mask_logits)
    
    return torch.stack(list_mask_logits, dim=0)


def predict_masks(images, mask_thresh=0.5, vis=True):
    # images is a list of PIL images
    # Convert this to tensor
    transf = torchvision.transforms.Compose([torchvision.transforms.PILToTensor(),
                                             torchvision.transforms.ConvertImageDtype(torch.float)
                                             ])
    images = [transf(i) for i in images]

    # Load the pre-trained model
    model = get_model(num_classes)
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")

    # Convert model to model.module if DDP
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    
    # Get the outputs for the given batch of images
    images = list(img.to(device) for img in images)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_time = time.time()

    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    # "images" is a list of image tensors, each with shape [C, H, W]
    model.eval()
    model.roi_heads.mask_predictor.register_forward_hook(get_features("mask_fcn_logits"))
    outputs = model(images)


    # outputs[0] is a dict with keys: ['boxes', 'labels', 'scores', 'masks']
    outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
    model_time = time.time() - model_time

    return_masks = []
    return_boxes = []
    return_scores = []
    return_labels = []
    return_raw_logits = []
    # Get the images and masks
    # Scale image back to [0,255]
    for i, image in enumerate(images):
        img = image*255
        img = img.to(torch.uint8)

        # Get the results of the Mask-RCNN model
        boxes = outputs[i]["boxes"]
        labels = [l.item() for l in outputs[i]["labels"]]
        masks = outputs[i]["masks"]
        scores = outputs[i]["scores"]

        # Filter any intersecting boxes by "NMS" using IoU::: (enter boxes & scores)
        # Get the indices
        valid_indices = filter_boxes_nms(boxes, scores)
        # --------------- Filter by the above indices
        # ----- Boxes
        valid_boxes = torch.index_select(boxes.to(torch.int32, copy=True),
                                         0,
                                         valid_indices)
        return_boxes.append(valid_boxes)
        # ----- Labels
        valid_labels = torch.index_select(torch.as_tensor(labels), 
                                          0, 
                                          valid_indices)
        return_labels.append(valid_labels)
        # ----- Masks
        valid_masks = masks.to(torch.float32, copy=True)
        valid_masks = valid_masks.squeeze()
        valid_masks = valid_masks >= mask_thresh
        valid_masks = torch.index_select(valid_masks, 0, valid_indices)
        return_masks.append(valid_masks)

        # ----- Scores
        valid_scores = torch.index_select(scores, 0, valid_indices)     #outputs[i]["scores"] > 0.8
        return_scores.append(valid_scores)


        # ----- RAW LOGITS FROM MASK-RCNN
        # Getting the raw logits of 28x28
        mask_lgts = features["mask_fcn_logits"]
        valid_raw_logits = get_raw_logits(img, mask_lgts, i, valid_boxes, valid_labels)
        return_raw_logits.append(valid_raw_logits)


        # Get Bounding Box in xywh format
        final_preds = torchvision.ops.box_convert(valid_boxes, in_fmt="xyxy", out_fmt="xywh")
        # If visualization is True, then show the valid boxes & masks
        if vis==True:
            # Init colors for viz
            colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255),
                      (255,0,255), (128,0,0), (255,140,0), (0,128,128), (188,143,143)]
            boxes_from_masks = masks_to_boxes(valid_masks)
            result = draw_bounding_boxes(img, boxes=valid_boxes, width=5, 
                                         colors=colors)
            show(result)
            res_seg = draw_segmentation_masks(img.to("cpu"), masks=valid_masks, alpha=0.5,
                                              colors=colors)
            show(res_seg)
            for obj_ct in range(len(final_preds)):
                # Display the cropped images in a sequence
                crop_img = torchvision.transforms.functional.crop(res_seg, 
                                                        top=final_preds[obj_ct][1].item(),
                                                        left=final_preds[obj_ct][0].item(),
                                                        height=final_preds[obj_ct][3].item(),
                                                        width=final_preds[obj_ct][2].item())
                cv2.imshow(f"crop {obj_ct}", crop_img.cpu().numpy().transpose(1, 2, 0) )
                cv2.waitKey(2000)
        
    return return_masks, return_boxes, return_scores, return_labels, return_raw_logits


def filter_boxes_nms(boxes, scores):
    indices = torchvision.ops.nms(boxes.to(torch.float32), scores, iou_threshold=0.15)
    print("\n\nOriginal boxes shape: ", boxes.shape, indices)
    filtered_box = torch.index_select(boxes, 0, indices)
    print("Filter boxes shape: ", filtered_box.shape)
    return indices


if __name__ == "__main__":
    image_paths = ["D:/Akshay_Work/datasets/motor_dataset_trials/test_img3.jpg"]
    # ["D:\\Akshay_Work\\datasets\\motor\\train_pbr\\000001\\rgb\\000246.jpg",]
    #"D:/Akshay_Work/datasets/motor_dataset_trials/test_img1.jpg",
    #"D:\\Akshay_Work\\datasets\\motor\\train_pbr\\000001\\rgb\\000246.jpg",

    #image_paths = ["D:\\Akshay_Work\\datasets\\tless\\test_primesense\\000001\\rgb\\000334.png"]
    images = [Image.open(i_p).convert("RGB") for i_p in image_paths]
    ret_masks, ret_boxes, ret_scores, labels, ret_raw_logits = predict_masks(images)
    print("Mask shapes, scores, labels:", [r_m.shape for r_m in ret_masks], 
          [r_s for r_s in ret_scores],
          [l for l in labels])
    
    filter_boxes_nms(ret_boxes[0], ret_scores[0])
