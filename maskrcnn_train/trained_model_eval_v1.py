import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import torchviz_utils.utils as utils
import torchviz_utils.transforms as T
from torchviz_utils.coco_eval import CocoEvaluator
from torchviz_utils.coco_utils import get_coco_api_from_dataset
import rgbmaskdataset_v1 as MaskDataLoader

import os, time, pstats, pickle
import cProfile as profile
from datetime import datetime as dt


root_path = "D:\\Akshay_Work\\aks_git_repos\\surfemb\\maskrcnn_train"
# Number of classes in the dataset
num_classes = 6
dir_nums=[i for i in range(0,15) if i not in [1,3]]
batch_size_train=4
data_split_train=0.75
num_epochs = 25


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model(number_classes=31):
    # load a pre-trained model for classification
    backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
    # No of output channels in mobilenet_v2 baclkbonde
    backbone.out_channels = 1280

    # RPN generates 5 x 3 anchors per spatial location, 
    # with 5 different sizes, 3 different aspect ratios.
    anchor_generator = AnchorGenerator(sizes=((8, 16, 32, 64, 128),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))

    # Feature maps used for RoI Mapping, & the output size.
    # featmap_names is expected to be [0] if backbone returns Tensor. 
    # If backbone returns an OrderedDict[Tensor], choose 
    # featmap_names from this dict.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    model = MaskRCNN(backbone,
                    num_classes=number_classes,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler,)
    return model

@torch.inference_mode()
def evaluate(model, epoch, data_loader, device, dir_nums=[20], tb_writer=None):
    def _get_iou_types(model):
        model_without_ddp = model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_without_ddp = model.module
        iou_types = ["bbox"]
        if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
            iou_types.append("segm")
        if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
            iou_types.append("keypoints")
        return iou_types

    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    t1 = dt.now()

    if epoch % 1 != 0:  # if epoch == 0:
        coco = get_coco_api_from_dataset(data_loader.dataset)
        torch.save(coco, coco_format_path)
    elif epoch % 1 == 0:   #elif epoch > 0:
        coco = torch.load(coco_format_path)
    print("get_coco_api_from_dataset: ", dt.now() - t1)
    print()

    t1 = dt.now()
    iou_types = _get_iou_types(model)
    print("_get_iou_types: ", dt.now() - t1)
    print()
    t1 = dt.now()
    coco_evaluator = CocoEvaluator(coco, iou_types)
    print("Init CocoEvaluator took: ", dt.now() - t1)
    
    # Convert model to model.module if DDP
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    i = 0
    for images, targets in metric_logger.log_every(data_loader, 20, header):
        t_loop = dt.now()
        images = list(img.to(device) for img in images)
        #targets = list(trg.to(device) for trg in targets)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()

        # Get evaluation metrics for IoU
        model.eval()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

        metric_logger.update(model_time=model_time, 
                            evaluator_time=evaluator_time,
                            )
        if i == len(data_loader)-2: print(f"Loop {i} took: ", dt.now() - t_loop)
        i += 1

    # gather the stats from all processes
    '''metric_logger.synchronize_between_processes()'''
    print("Averaged stats:", metric_logger)
    '''coco_evaluator.synchronize_between_processes()'''
    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    return coco_evaluator, metric_logger


if __name__ == "__main__":
    trained_model_path = os.path.join(root_path, "trained_models", "run_train_motor_no_3_3_cont_20_model.pt")
    coco_format_path = os.path.join(root_path, "motor_data_in_coco_format.pt")
    train_test_ind_file = os.path.join(root_path, "train_test_indices.pt")
    
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
    

    # ----------LOAD PRETAINED MODEL
    checkpoint = torch.load(trained_model_path)
    # Initialize the Model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=False,
                        num_classes=num_classes, pretrained_backbone=True,
                        trainable_backbone_layers=3)
    
    # get the model using our helper function
    '''model = get_model(num_classes)'''
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
    coco_evaluator, eval_metric_logger = evaluate(model=model,
                                                epoch=200, 
                                                data_loader=data_loader_test, 
                                                device=device,                                                
                                                dir_nums=dir_nums)
    print("@@@@@@@@@@@@@@ Eval Metrics: @@@@@@@@@@@@", eval_metric_logger)
    print("@@@@@@@@@@@@@@ Eval Took: @@@@@@@@@@@@", dt.now() - eval_start)
