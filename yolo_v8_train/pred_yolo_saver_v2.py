from datetime import datetime as dt
import time, os, cv2, math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import platform
import yaml

import torch
import torchvision
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms.functional as F

#from ultralytics.models.yolo.segment import SegmentationTrainer
#from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.models.yolo.detect import DetectionPredictor

from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops


# CHANGE OBJECT IDs TO CONTINUOUS INDICES, for eg., in range(0,20)
# IN CASE OBEJCTS ARE REMOVED
yaml_path = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/yolo_v8_train/tless_mod_train.yaml"
with open(yaml_path, "r") as file:
    read_dict = yaml.safe_load(file)
    yaml_dict = read_dict["names"]



class CustomPredictor(DetectionPredictor):
    def __init__(self, overrides, save_dir):
        super().__init__(overrides=overrides)
        self.save_dir = Path(save_dir)

    def postprocess(self, preds, img, orig_imgs):
        """Postprocesses predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                #pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                np_img = img.cpu().numpy()
                pred[:, :4] = ops.scale_boxes(np_img.shape[2:], pred[:, :4], orig_img.shape)
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results


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
        for j, labl in enumerate(labels[i]):
            plt.text(ret_boxes[i][j][0], ret_boxes[i][j][1],
                     str(labl.item()))
        plt.show()



def predict_yolo(model_path, images, viewer="y"):
    model = model_path

    final_boxes = []
    final_scores = []
    final_labels = []

    for index in range(len(images)):
        source = images[index]
        
        #source = cv2.imread(source)
        #source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        #source = np.stack((source,)*3, axis=-1)
        '''import albumentations as A
        T = [A.HueSaturationValue(hue_shift_limit=75, sat_shift_limit=75, 
                                val_shift_limit=75, p=1),
            A.ChannelShuffle(p=1)]
        aug = A.Compose(T)
        source = aug( image = np.array(source) )["image"]
        source = Image.fromarray(source)'''
        #source[np.where((source<=[60,60,60]).all(axis=(2)))] = [255,211,155]
        #source = cv2.bitwise_not(source)


        """
        YOLO PREDICTOR INITIALIZATION
        """
        args = dict(model=model, source=source,
                agnostic_nms = False,
                conf = 0.5,
                iou = 0.7,
                augment = True)
        save_dir = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/yolo_v8_train"
        predictor = CustomPredictor(overrides = args,
                                    save_dir = save_dir
                                   )
        #predictor.setup_model(model_path)
        predictor.predict_cli()
        """
        YOLO PREDICTOR INITIALIZATION
        """


        # Expand dims to include batch size=1
        image = np.expand_dims(source, axis=0)
        predictor.setup_source(image)
        image = predictor.preprocess(image)
        init_preds = predictor.model(image,
                                    augment=True)
        # List of predictions. Access by predictions[0].boxes.xyxy, .cls, .conf or .xywh
        predictions = predictor.postprocess(init_preds, image, np.asarray(source))
        #print()

        ret_boxes = [a.boxes.xyxy.cpu() for a in predictions]
        ret_scores = [a.boxes.conf.cpu() for a in predictions]
        labels = [a.boxes.cls.cpu() for a in predictions]

        final_boxes += ret_boxes
        final_scores += ret_scores
        final_labels += labels

        if viewer == "y":
            result = draw_bounding_boxes(pil_to_tensor(source), 
                                        boxes=ret_boxes[0], 
                                        width=5,) 
                                            #colors=colors)
            show(result, ret_boxes, labels)
        
    return final_boxes, final_scores, final_labels



def filter_boxes_nms(boxes, scores):
    indices = torchvision.ops.nms(boxes.to(torch.float32), 
                                  scores.to(torch.float32), 
                                  iou_threshold=0.85)
    print("\n\nOriginal boxes shape: ", boxes.shape, indices)
    filtered_box = torch.index_select(boxes, 0, indices)
    print("Filter boxes shape: ", filtered_box.shape)
    return indices



def run_and_save(
        bop_test_path = "/home/ise.ros/akshay_work/datasets/tless_mod/tless_mod/test_primesense",
        detec_save_path = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/data/detection_results/tlessmod",
        ):
    
    dataset = detec_save_path.split("/")[-1]
    if dataset == "tless":
        num_classes= 31    
        model_path = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/runs/detect/pbr_train_5_ep_1000/weights/best.pt"
    elif dataset in ["tless_mod", "tlessmod"]:
        num_classes= 16    
        model_path = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/runs/detect/pbr_blk_regen_tless_mod_1800/weights/best.pt"
    else:
        print("NOT SUPPORTED. ---------------------- EXITING...")
        quit()


    scene_ids = []
    view_ids = []
    bboxes = []
    obj_ids_yolo_labels = []
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
            if i % 1 == 0:
                time.sleep(0.5)
                # Predicted masks
                ret_boxes, ret_scores, labels = predict_yolo(model_path, 
                                                             run_images)

                bboxes += [a for b in ret_boxes
                            for a in b.cpu().numpy()]
                obj_ids_yolo_labels += [y.item() for x in labels
                            for y in x.cpu()]
                # REASSIGN CONTINUOUS INDEX TO ITS OBJECT ID BASED ON YAML
                obj_ids = [ yaml_dict[yolo_pred_id] 
                                for yolo_pred_id in obj_ids_yolo_labels
                          ]
                scores += [q for s in ret_scores
                            for q in s.cpu().detach().numpy()]

                #file_names = [Path(a).stem for a in rgb_imgs]
                for indx, img_lbls in enumerate(labels):
                    for j in range( len([x for x in img_lbls]) ):
                        scene_ids.append( int(cdir) )     #math.floor(int(sid) / 25) )
                        img_id = Path(img_paths[indx]).stem
                        view_ids.append( int(img_id) )       #  % 25

                img_paths = []
                run_images = []
            '''if i == 10:
                break'''
            


    #------------ Before filtering, for Multiple Images.
    '''scene_ids = np.asarray(scene_ids)
    view_ids = np.asarray(view_ids)
    bboxes = np.asarray(bboxes)
    obj_ids = np.asarray(obj_ids).astype(np.int16)
    scores = np.asarray(scores)'''

    # Create the DF for filtering by each UNIQUE Folder & Image
    filter_df = pd.DataFrame({"scene_ids": scene_ids, 
                              "view_ids": view_ids, 
                               "bboxes": bboxes, 
                               "obj_ids": obj_ids,
                               "scores": scores}, 
                            )
    unique_scenes = np.unique(filter_df['scene_ids'])
    unique_views = np.unique(filter_df['view_ids'])

    # Filter by Folder, followed by Images
    ct = 0
    for scn in unique_scenes:
        for view in unique_views:
            search_df = filter_df[(filter_df['scene_ids'] == scn)
                                      & (filter_df['view_ids'] == view)]
            
            srch_scene_ids = np.vstack(search_df["scene_ids"])
            srch_view_ids = np.vstack(search_df["view_ids"])
            srch_bboxes = np.vstack(search_df["bboxes"])
            srch_obj_ids = np.vstack(search_df["obj_ids"])
            srch_scores = np.vstack(search_df["scores"])

            if srch_scores.shape[0] > 1:
                ct += 1
                # NMS For each image
                final_ind = filter_boxes_nms(torch.from_numpy(srch_bboxes),
                                            torch.from_numpy(srch_scores).squeeze()
                                            )
                final_ind = final_ind.numpy()

                # After Filtering
                if ct == 1:
                    fil_scene_ids = srch_scene_ids[final_ind]
                    fil_view_ids = srch_view_ids[final_ind]
                    fil_bboxes = srch_bboxes[final_ind]
                    fil_obj_ids = srch_obj_ids[final_ind]
                    fil_scores = srch_scores[final_ind]
                else:
                    fil_scene_ids = np.vstack( (fil_scene_ids, srch_scene_ids[final_ind]) )
                    fil_view_ids = np.vstack( (fil_view_ids, srch_view_ids[final_ind]) )
                    fil_bboxes = np.vstack( (fil_bboxes, srch_bboxes[final_ind]) )
                    fil_obj_ids = np.vstack( (fil_obj_ids, srch_obj_ids[final_ind]) )
                    fil_scores = np.vstack( (fil_scores, srch_scores[final_ind]) )


    path = detec_save_path
    with open(os.path.join(path, "scene_ids.npy"), "wb") as f:
        np.save(f, fil_scene_ids.reshape(-1))
    with open(os.path.join(path, "view_ids.npy"), "wb") as f:
        np.save(f, fil_view_ids.reshape(-1))
    with open(os.path.join(path, "bboxes.npy"), "wb") as f:
        np.save(f, fil_bboxes)
    with open(os.path.join(path, "obj_ids.npy"), "wb") as f:
        np.save(f, fil_obj_ids.reshape(-1))
    with open(os.path.join(path, "scores.npy"), "wb") as f:
        np.save(f, fil_scores.reshape(-1))

    return bboxes, obj_ids


if __name__ == "__main__":
    '''run_and_save(bop_test_path="/home/ise.ros/akshay_work/datasets/tless/test_primesense",
                 detec_save_path="/home/ise.ros/akshay_work/NN_Implementations/surfemb/data/detection_results/tless")
    '''
    run_and_save()
    
    print()