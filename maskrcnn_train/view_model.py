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
num_classes=31  #6
root_path = "D:\\Akshay_Work\\aks_git_repos\\surfemb\\maskrcnn_train"
trained_model_path = os.path.join(root_path, "trained_models", "run-5-2_fol-0_model-std_CONT_lr-red-1e-5_150_model.pt")
    # "run_train_motor_no_1_10_model.pt")
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

def run_model(images, mask_thresh=0.5, vis=True):
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
    model.roi_heads.mask_head[3].register_forward_hook(get_features("ReLU"))
    model.roi_heads.mask_predictor.register_forward_hook(get_features("mask_fcn_logits"))
    outputs = model(images)
    #print(features)

    #wts = model.roi_heads.mask_predictor.mask_fcn_logits.weight
    #print(model.roi_heads.mask_head)

    '''from torchvision.models.detection.roi_heads import maskrcnn_inference
    labels = [t["labels"] for t in outputs]
    mask_prob = maskrcnn_inference(features["mask_fcn_logits"], labels)
    print(mask_prob[0].shape)
    print()'''


    # Getting the raw logits of 28x28
    from torchvision.models.detection.roi_heads import paste_mask_in_image, maskrcnn_inference
    mask_lgts = features["mask_fcn_logits"]
    
    # SELECT INDEX, Get prob from lgts, get BBoxes, get Labels
    index = 0
    box = outputs[0]["boxes"][index]
    box = box.to(torch.int64)
    label = outputs[0]["labels"][index]
    img_size = images[0].shape


    # Get logits for 1 Mask Instance: Gives a [31, 28, 28] Tensor
    mask_lgt_single_inst = mask_lgts[index,:,:,:]
    # Get probabilities from logits
    prob_single_mask_lgts = mask_lgt_single_inst.sigmoid() #torch.nn.functional.softmax(mask_lgt_single_inst, dim=0)
    # Project Mask to desired BBoex shape; then place on original shape of the full image
    img_mask = paste_mask_in_image(mask = prob_single_mask_lgts[label,:,:], 
                                box = box, 
                                im_h = img_size[1], 
                                im_w = img_size[2])
    img_mask = img_mask.unsqueeze(dim=0)
    from torchvision.models.detection.transform import GeneralizedRCNNTransform 
    print(img_mask.shape)


    '''list_img_logits_label = []
    for l in range(0,31): #outputs[0]["labels"]:
        img_logits_label = paste_mask_in_image(mask = mask_lgts[index,l,:,:],   #mask_lgt_single_inst[i,:,:],
                                            box = box,
                                            im_h = img_size[1], 
                                            im_w = img_size[2])
        list_img_logits_label.append(img_logits_label)
    mask_lgt_single_inst = torch.stack(list_img_logits_label, dim=0)
    prob_single_mask_lgts = torch.nn.functional.softmax(mask_lgt_single_inst, dim=0)
    img_mask = prob_single_mask_lgts[label]
    img_mask = img_mask.unsqueeze(dim=0)'''


    diff = outputs[0]["masks"][index] - img_mask
    diff = diff.detach().cpu().numpy()*255
    diff = diff.astype("uint8").transpose((1,2,0))
    show_mask = img_mask.detach().cpu().numpy()*255
    show_mask = show_mask.astype("uint8").transpose((1,2,0))
    print(type(diff), diff.dtype, diff.shape, diff.max())
    div = np.divide(diff[:,:,0], show_mask[:,:,0])
    div = np.nan_to_num(div, nan=0, posinf=1)
    print(div)

    import cv2
    cv2.imshow("diff", diff)
    cv2.imshow("mask_prob", show_mask)
    cv2.waitKey(0)
    print()


if __name__ == "__main__":
    image_paths = ["D:/Akshay_Work/datasets/tless/train_pbr/000008/rgb/000066.jpg"]
    images = [Image.open(i_p).convert("RGB") for i_p in image_paths]
    ret_masks, ret_boxes, ret_scores, labels = run_model(images)