import torch
import torchvision
from torchvision.models.detection import FasterRCNN, MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.tensorboard import SummaryWriter
import tqdm, math, sys
import matplotlib.pyplot as plt
import numpy as np
import os, time, pstats
import cProfile as profile
from datetime import datetime as dt
import pickle
from collections import defaultdict, deque

import rgbmaskdataset_v1 as MaskDataLoader
import torchviz_utils.utils as utils
import torchviz_utils.transforms as T
from torchviz_utils.coco_eval import CocoEvaluator
from torchviz_utils.coco_utils import get_coco_api_from_dataset

# Distributed Training Imports
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group



# ---------------------------INITIALIZE------------------------------
root_path = "D:/Akshay_Work/aks_git_repos/surfemb/maskrcnn_train"

run_name="run_train_motor_no_3_3_cont"
distrib={"state": True,
        "gpu_ids": [0, 1],
        }
continue_train=True
checkpt_path=os.path.join(root_path, "trained_models", "run_train_motor_no_3_2_wt_dec_5e-4_15_model.pt")


# Number of classes in the dataset
num_classes = 6
dir_nums=[i for i in range(0,15) if i not in [1,3]]        # [0,2,4,6] 
batch_size_train=8
data_split_train=0.75
num_epochs = 5
# ---------------------------INITIALIZE------------------------------
coco_format_path = os.path.join(root_path, "motor_data_in_coco_format.pt")
model_save_path = os.path.join(root_path, "trained_models")
tb_logdir = os.path.join(root_path, "runs", run_name)
train_test_ind_file = os.path.join(root_path, "train_test_indices.pt")



def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    #init_process_group(backend="nccl", rank=rank, world_size=world_size)
    init_process_group(backend="gloo", rank=rank, world_size=world_size)


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model_old(number_classes=31):
    # load a pre-trained model for classification
    backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
    # No of output channels in mobilenet_v2 baclkbonde
    backbone.out_channels = 1280

    # RPN generates 5 x 3 anchors per spatial location, 
    # with 5 different sizes, 3 different aspect ratios.
    anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256),),
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
    return model


def get_model(number_classes=31):
    # Load model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT",
                                                                 trainable_backbone_layers=1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Set RoI predictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, number_classes)


    # RPN generates 5 x 3 anchors per spatial location, 
    # with 5 different sizes, 3 different aspect ratios.
    anchor_generator = AnchorGenerator(sizes=((8, 16, 32, 64, 128, 256),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))
    # Set RPN Anchor Box Generator
    model.rpn_anchor_generator = anchor_generator


    # Get num features for Mask Classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Set Mask Predictor 
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       number_classes)
    return model


@torch.inference_mode()
def evaluate(model, epoch, data_loader, device, tb_writer=None):
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

    if epoch == 0:       # epoch % 1 != 0:   
        coco = get_coco_api_from_dataset(data_loader.dataset)
        torch.save(coco, coco_format_path)
    elif epoch > 0:      # epoch % 1 == 0:   
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


class Trainer():
    def __init__(self,
                model,
                data_loader,
                data_loader_test,
                optimizer,
                this_gpu_rank,
                trained_epochs,
                load_metrics):
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            
        self.data_loader = data_loader
        self.data_loader_test = data_loader_test
        self.optimizer = optimizer
        self.this_gpu_id = this_gpu_rank
        self.trained_epochs = trained_epochs
        self.load_metrics = load_metrics

        self.model = model.to(self.this_gpu_id)
        if distrib["state"] == True:
            self.model = DDP(model, 
                            device_ids=[self.this_gpu_id])

        self.lr_scheduler = None
        self.tb_writer = SummaryWriter(log_dir=tb_logdir,
                                       flush_secs=10,)
        
    
    def _run_batch(self, images, targets, scaler):
        #---------- Run the Batch - Forward pass ----------
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(loss_dict.keys()):
            names.append(k)
            values.append(loss_dict[k])
        values = torch.stack(values, dim=0)
        reduced_dict = {k: v for k, v in zip(names, values)}
        #loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced = reduced_dict
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        # Preventing infinite error
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        #---------- Backpropogation ----------
        self.optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(self.optimizer)
            scaler.update()
        else:
            losses.backward()
            self.optimizer.step()

        # update the learning rate
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        # Return loss in different forms for logging in Tensorboard
        return loss_dict, loss_dict_reduced, losses_reduced, loss_value

    def _run_epoch(self, 
                   epoch, 
                   metric_logger,
                   scaler=None, 
                   print_freq=10,
                   header=" "):
                   
        # Set sampler for training data, if training on multiple GPUs
        if distrib["state"] == True:
            self.data_loader.sampler.set_epoch(epoch)

        # Start training process ---> load batch by batch
        batch_losses = []
        batch = 1
        for images, targets in metric_logger.log_every(self.data_loader, 
                                                       print_freq, 
                                                       header):
            #images = list(image.to(self.device) for image in images)
            #targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            images = list(image.to(self.this_gpu_id) for image in images)
            images = torch.stack(images)       # TODO changed for single GPU training
            targets = [{k: v.to(self.this_gpu_id) for k, v in t.items()} for t in targets]

             #---------- Run the BATCH - Forward & Backward pass ----------
            loss_dict, loss_dict_reduced, losses_reduced, loss_value = self._run_batch(images,
                                                                                targets, 
                                                                                scaler=scaler)

            # Update metrics within program
            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            # Write metrics to TensorBoard summary
            self.tb_writer.add_scalars("training/batch/losses", loss_dict_reduced, batch)
            self.tb_writer.add_scalar("training/batch/losses/loss_summed", losses_reduced, batch)
            self.tb_writer.add_scalar("training/batch/learning_rate_lr", 
                                    self.optimizer.param_groups[0]["lr"], batch)
            batch_losses.append(loss_value)
            batch += 1

        return metric_logger, np.average(np.asarray(batch_losses))

    def train_test_model(self):
        for epoch in range(num_epochs):
            header = f"Epoch: [{epoch}]"
            # TODO REMOVED TO TRY AND FIX TENSOR IN-PLACE ERROR 
            self.model.train()

            # Load previous metrics
            # If no metrics are available, then-
            if self.load_metrics == {}:
                metric_logger = utils.MetricLogger(delimiter="  ")
                metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
            # If previous metrics are available
            else:
                metric_logger = self.load_metrics

            # Learning rate scheduler: Initialize
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(self.data_loader) - 1)
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )     
            
            #---------- Run the EPOCH ----------
            # train for one epoch, printing every 10 iterations
            start_epoch = dt.now()
            metric_logger, epoch_loss = self._run_epoch(epoch,
                                                        metric_logger, 
                                                        scaler=None, 
                                                        print_freq=10,
                                                        header=header)
            
            eval_start = dt.now()
            if (epoch > 100): #((epoch % 4 == 0) and (self.this_gpu_id == 0)):
                #---------- evaluate on the test dataset ----------
                # AKS MOD::: Changed this line to self.trained_epochs, 
                # helps load the Test Dataset in COCO Format quicker
                coco_evaluator, eval_metric_logger = evaluate(self.model,
                                                                self.trained_epochs, #epoch, 
                                                                self.data_loader_test, 
                                                                device=self.device,
                                                                tb_writer=self.tb_writer)
                print("@@@@@@@@@@@@@@ Eval Metrics: @@@@@@@@@@@@", eval_metric_logger)
                print("@@@@@@@@@@@@@@ Eval Took: @@@@@@@@@@@@", dt.now() - eval_start)
                # Write to TensorBoard summary
                bbox_ious = [a for a in coco_evaluator.coco_eval["bbox"].ious.values()
                            if a != []]
                segm_ious = [a for a in coco_evaluator.coco_eval["segm"].ious.values()
                            if a != []]
                if bbox_ious != []:
                    print(bbox_ious)
                '''self.tb_writer.add_scalars("testing/epoch/loss", 
                                            final_eval_loss, 
                                            self.trained_epochs + epoch)'''

            # Add Epoch loss to the torchwriter
            self.tb_writer.add_scalar("training/epoch/loss", 
                                    epoch_loss, 
                                    self.trained_epochs + epoch)
            print(f"@@@@@@@@@@@@@@ GPU No. {self.this_gpu_id}, Epoch no. {epoch}: Train + Eval Took: @@@@@@@@@@@@", dt.now() - start_epoch)
        
        #---------- SAVE THE MODEL AT THE END ----------
        if self.this_gpu_id == 0:
            self._save_checkpoint(metric_logger)
        # Clear the Tensorboard writer
        self.tb_writer.flush()
        self.tb_writer.close()
    
    def _save_checkpoint(self, metric_logger):
        if distrib["state"] == True:
            model_state_dict = self.model.module.state_dict()
        elif distrib["state"] == False:
            model_state_dict = self.model.state_dict()
        # Additional information
        EPOCH = self.trained_epochs + num_epochs
        PATH = os.path.join(model_save_path, f"{run_name}_{EPOCH}_model.pt")
        torch.save({
                    'epoch': EPOCH,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'ending_metrics': metric_logger.meters,
                    }, 
                    PATH)


def get_dataloader():
    # Create dataset with transformations
    dataset = MaskDataLoader.RgbMasksDataset(dir_nums = dir_nums,
                                            transforms = get_transform(train=True))
    dataset_test = MaskDataLoader.RgbMasksDataset(dir_nums = dir_nums,
                                                 transforms = get_transform(train=False))
    # Split dataset in to test/train sets
    '''indices = torch.randperm(len(dataset)).tolist()'''
    indices = torch.load(train_test_ind_file)
    train_size = int(data_split_train * len(dataset))
    dataset = torch.utils.data.Subset(dataset, indices[:train_size])
    '''dataset_test = torch.utils.data.Subset(dataset, indices[train_size:])'''
    dataset_test = torch.utils.data.Subset(dataset_test, indices[train_size:])

    # Create dataloaders for test & train data
    if distrib["state"] == False:
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size_train, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size_train, shuffle=False, num_workers=4,
            sampler=DistributedSampler(dataset),
            collate_fn=utils.collate_fn)
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)
    print("DATASET_SIZES::: -------- :::", len(data_loader)*batch_size_train, 
        len(data_loader_test)*1)
    
    return data_loader, data_loader_test


def main(rank, world_size, continue_train=False, checkpt_path=""):
    if distrib["state"] == True:
        ddp_setup(rank, world_size)

    # Set fixed random number seed & device ID
    torch.manual_seed(75)
    # Get the data loaders for train & test
    data_loader, data_loader_test = get_dataloader()
    # Initialize the Model
    '''model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT",
                                                                  num_classes=num_classes, 
                                                                  trainable_backbone_layers=3,
                                                                  )
    '''
    # get the model using our helper function
    model = get_model(num_classes)

    # Initialize the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.000005, 
                                betas=(0.9,0.999), weight_decay=0.0005)
    
    # Check if training is -from scratch-, or -continued- 
    if continue_train == False:
        ending_metrics = {}
        trained_epochs = 0
        pass
    elif continue_train == True and checkpt_path == "":
        print("Checkpoint path not given!!! Training the model from scratch......")
        ending_metrics = {}
        trained_epochs = 0
        pass
    elif continue_train == True and checkpt_path != "":
        checkpt = torch.load( checkpt_path, map_location=torch.device(f"cuda:{rank}") )
        model.load_state_dict(checkpt["model_state_dict"])
        '''optimizer.load_state_dict(checkpt["optimizer_state_dict"])'''
        trained_epochs = checkpt["epoch"]
        ending_metrics = utils.MetricLogger(delimiter="  ")
        ending_metrics.meters = defaultdict(None, checkpt['ending_metrics'].items())    #metric_logger.meters

    #---------- START TRAINING ----------
    model.train()
    trainer = Trainer(model,
                    data_loader,
                    data_loader_test,
                    optimizer,
                    rank,
                    trained_epochs=trained_epochs,
                    load_metrics=ending_metrics,
                    )
    trainer.train_test_model()

    if distrib["state"] == True:
        destroy_process_group()
    

if __name__ =="__main__":
    if distrib["state"] == True:
        world_size = torch.cuda.device_count()
        mp.spawn(main, 
                args=(world_size,
                    continue_train,
                    checkpt_path,), 
                nprocs=world_size
                )

    else:
        print("--------Single GPU training starting--------")
        main(rank=0, world_size=1, continue_train=False, checkpt_path="")    