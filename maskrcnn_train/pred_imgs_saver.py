import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN, MaskRCNN
from torchvision.utils import make_grid
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.io import read_image
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as F

import torchviz_utils.utils as utils
import torchviz_utils.transforms as T

from datetime import datetime as dt
import time, os, cv2, math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

dir_nums=[0]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cpu_device = torch.device("cpu")

from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator


class DetectorMaskRCNN(MaskRCNN):
    def __init__(self, input_resize=(240, 320), n_classes=2,
                 backbone_str='resnet50-fpn',
                 anchor_sizes=((32, ), (64, ), (128, ), (256, ), (512, ))):

        assert backbone_str == 'resnet50-fpn'
        backbone = resnet_fpn_backbone('resnet50', pretrained=False)

        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        super().__init__(backbone=backbone, num_classes=n_classes,
                         rpn_anchor_generator=rpn_anchor_generator,
                         max_size=max(input_resize), min_size=min(input_resize))


def get_model_tless(trained_model_path, number_classes=31,):
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
    '''model = DetectorMaskRCNN(input_resize=(240, 320), n_classes=number_classes,
                 backbone_str='resnet50-fpn',
                 anchor_sizes=((32, ), (64, ), (128, ), (256, ), (512, )))'''

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
    #model.load_state_dict(checkpoint['state_dict'])

    return model


def get_model_motor(trained_model_path, number_classes=31,):
    # load a pre-trained model for classification
    backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
    # No of output channels in mobilenet_v2 baclkbonde
    backbone.out_channels = 1280

    # RPN generates 5 x 3 anchors per spatial location, 
    # with 5 different sizes, 3 different aspect ratios.
    anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))

    # Feature maps used for RoI Mapping, & the output size.
    # featmap_names is expected to be [0] if backbone returns Tensor. 
    # If backbone returns an OrderedDict[Tensor], choose 
    # featmap_names from this dict.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0','1','2','3'],
                                                    output_size=14,
                                                    sampling_ratio=2)

    model = MaskRCNN(backbone,
                    num_classes=number_classes,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler,)
    
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

def show(imgs, ret_boxes, labels):
    #plt.switch_backend("TkAgg")
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        timer = fig.canvas.new_timer(interval = 3000)
        timer.add_callback(plt.close)
        timer.start()
        # Add the Object Label text
        for j, labl in enumerate(labels):
            plt.text(ret_boxes[j][0], ret_boxes[j][1],
                     str(labl.item()))
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


def filter_boxes_nms(boxes, scores):
    indices = torchvision.ops.nms(boxes.to(torch.float32), scores, iou_threshold=0.7)
    print("\n\nOriginal boxes shape: ", boxes.shape, indices)
    filtered_box = torch.index_select(boxes, 0, indices)
    print("Filter boxes shape: ", filtered_box.shape)
    return indices


def predict_masks(model, images, img_id, mask_thresh=0.60, vis=True):
    # images is a list of PIL images
    # Convert this to tensor
    transf = torchvision.transforms.Compose([torchvision.transforms.PILToTensor(),
                                             torchvision.transforms.ConvertImageDtype(torch.float)
                                             ])
    images = [transf(i) for i in images]

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

        #cv2.imshow("img", img.permute(1, 2, 0).cpu().numpy())
        #cv2.waitKey(1000)

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
        valid_masks = valid_masks.squeeze(1)
        valid_masks = valid_masks >= mask_thresh
        valid_masks = torch.index_select(valid_masks, 0, valid_indices)
        return_masks.append(valid_masks)

        # ----- Scores
        valid_scores = torch.index_select(scores, 0, valid_indices)     #outputs[i]["scores"] > 0.8
        return_scores.append(valid_scores)


        # ----- RAW LOGITS FROM MASK-RCNN
        # Getting the raw logits -> Input 28x28, Output IMG_SIZE
        mask_lgts = features["mask_fcn_logits"]
        if valid_indices.shape[0] > 0:
            valid_raw_logits = get_raw_logits(img, mask_lgts, i, valid_boxes, valid_labels)
            return_raw_logits.append(valid_raw_logits)
        else:
            return_raw_logits.append([None])


        # Get Bounding Box in xywh format
        final_preds = torchvision.ops.box_convert(valid_boxes, in_fmt="xyxy", out_fmt="xywh")
        # If visualization is True, then show the valid boxes & masks
        if vis==True: #and img_id%100==0):
            # Init colors for viz
            colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255),
                      (255,0,255), (128,0,0), (255,140,0), (0,128,128), (188,143,143),
                      (128,0,128), (65,105,225), (139,69,19), (0,0,0), (188,143,143)]
            #boxes_from_masks = masks_to_boxes(valid_masks)
            result = draw_bounding_boxes(img, boxes=valid_boxes, width=5,) 
                                         #colors=colors)
            show(result, valid_boxes, valid_labels)
            res_seg = draw_segmentation_masks(img.to("cpu"), masks=valid_masks, alpha=0.5,)
                                              #colors=colors)
            #show(res_seg)


    return return_masks, return_boxes, return_scores, return_labels, return_raw_logits


def run_and_save(
        bop_test_path = "D:\\Akshay_Work\\datasets\\motor\\test_primesense",
        detec_save_path = "D:\\Akshay_Work\\aks_git_repos\\surfemb\\data\\detection_results\\motor",
        ):
    
    dataset = detec_save_path.split("/")[-1]
    if dataset == "tless":
        num_classes= 31    
        root_path = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/maskrcnn_train"
        trained_model_path = os.path.join(root_path, "trained_models", "run-5-2_fol-0_model-std_CONT_lr-red-1e-5_150_model.pt")
        #trained_model_path = "/home/ise.ros/akshay_work/NN_Implementations/cosypose/local_data/experiments/detector-bop-tless-synt+real--452847/checkpoint.pth.tar"
        model = get_model_tless(trained_model_path, num_classes)
    elif dataset == "motor":
        num_classes= 6    
        root_path = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/maskrcnn_train"
        trained_model_path = os.path.join(root_path, "trained_models", "run_train_motor_no_4_4_20_model.pt")
        # Load the pre-trained model
        model = get_model_motor(trained_model_path, num_classes)

    torch.set_num_threads(1)

    scene_ids = []
    view_ids = []
    bboxes = []
    obj_ids = []
    scores = [] 

    # def run():
    bop_test_path = bop_test_path
    chunk_dir = [a for a in os.listdir(bop_test_path) if "." not in a]

    for cdir in chunk_dir:
        # RGB Dir with pictures
        rgb_dir = os.path.join(bop_test_path, cdir, "rgb")
        # Paths to all images
        rgb_imgs = [os.path.join(rgb_dir, a) 
                    for a in os.listdir(rgb_dir)]

        img_paths = []
        run_images = []
        for i, img_path in enumerate(rgb_imgs):
            print(i)
            # Opened Images in PIL format
            img = Image.open(img_path).convert("RGB")
            img_paths.append(img_path)
            run_images.append(img)
            if i % 3 == 0:
                time.sleep(0.5)
                # Predicted masks
                ret_masks, ret_boxes, ret_scores, labels, ret_raw_logits = predict_masks(model, run_images, i)

                bboxes += [a for b in ret_boxes
                            for a in b.numpy()]
                obj_ids += [y.item() for x in labels
                            for y in x]
                scores += [q for s in ret_scores
                            for q in s.detach().numpy()]

                #file_names = [Path(a).stem for a in rgb_imgs]
                for i, img_lbls in enumerate(labels):
                    for j in range( len([x for x in img_lbls]) ):
                        scene_ids.append( int(cdir) )     #math.floor(int(sid) / 25) )
                        img_id = Path(img_paths[i]).stem
                        view_ids.append( int(img_id) )       #  % 25

                img_paths = []
                run_images = []
            
            '''if i == 50:
                break'''
    print()


    scene_ids = np.asarray(scene_ids)
    view_ids = np.asarray(view_ids)
    bboxes = np.asarray(bboxes)
    obj_ids = np.asarray(obj_ids)
    scores = np.asarray(scores)

    path = detec_save_path
    with open(os.path.join(path, "scene_ids.npy"), "wb") as f:
        np.save(f, scene_ids)
    with open(os.path.join(path, "view_ids.npy"), "wb") as f:
        np.save(f, view_ids)
    with open(os.path.join(path, "bboxes.npy"), "wb") as f:
        np.save(f, bboxes)
    with open(os.path.join(path, "obj_ids.npy"), "wb") as f:
        np.save(f, obj_ids)
    with open(os.path.join(path, "scores.npy"), "wb") as f:
        np.save(f, scores)

    return bboxes, obj_ids


if __name__ == "__main__":
    '''run_and_save(bop_test_path="/home/ise.ros/akshay_work/datasets/tless/test_primesense",
                 detec_save_path="/home/ise.ros/akshay_work/NN_Implementations/surfemb/data/detection_results/tless")
    '''
    run_and_save(bop_test_path="/home/ise.ros/akshay_work/NN_Implementations/surfemb/surfemb/scripts/dataset_img_detec/tless/test_primesense",
                 detec_save_path="/home/ise.ros/akshay_work/NN_Implementations/surfemb/surfemb/scripts/dataset_img_detec/detection_results/tless")
    
    print()