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
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data import load_inference_source
from ultralytics.yolo.data.augment import LetterBox, classify_transforms
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from ultralytics.yolo.utils.checks import check_imgsz, check_imshow
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode

from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops

STREAM_WARNING = """
    WARNING ⚠️ stream/video/webcam/dir predict source will accumulate results in RAM unless `stream=True` is passed,
    causing potential out-of-memory errors for large sources or long-running streams/videos.

    Usage:
        results = model(source=..., stream=True)  # generator of Results objects
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs
            probs = r.probs  # Class probabilities for classification outputs
"""

# CHANGE OBJECT IDs TO CONTINUOUS INDICES, for eg., in range(0,20)
# IN CASE OBEJCTS ARE REMOVED
yaml_path = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/yolo_v8_train/tless_mod_train.yaml"
with open(yaml_path, "r") as file:
    read_dict = yaml.safe_load(file)
    yaml_dict = read_dict["names"]



class BasePredictor:
    """
    BasePredictor

    A base class for creating predictors.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_setup (bool): Whether the predictor has finished setup.
        model (nn.Module): Model used for prediction.
        data (dict): Data configuration.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        vid_path (str): Path to video file.
        vid_writer (cv2.VideoWriter): Video writer for saving video output.
        annotator (Annotator): Annotator used for prediction.
        data_path (str): Path to data.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None,
                save_dir = "."):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        
        #self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
        self.save_dir = Path(save_dir)

        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.plotted_img = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

    def preprocess(self, im):
        """Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.
        """
        if not isinstance(im, torch.Tensor):
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)
        # NOTE: assuming im with (b, 3, h, w) if it's a tensor
        img = im.to(self.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def pre_transform(self, im):
        """Pre-tranform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Return: A list of transformed imgs.
        """
        same_shapes = all(x.shape == im[0].shape for x in im)
        auto = same_shapes and self.model.pt
        return [LetterBox(self.imgsz, auto=auto, stride=self.model.stride)(image=x) for x in im]

    def write_results(self, idx, results, batch):
        """Write inference results to a file or directory."""
        p, im, _ = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        result = results[idx]
        log_string += result.verbose()

        if self.args.save or self.args.show:  # Add bbox to image
            plot_args = dict(line_width=self.args.line_width,
                             boxes=self.args.boxes,
                             conf=self.args.show_conf,
                             labels=self.args.show_labels)
            if not self.args.retina_masks:
                plot_args['im_gpu'] = im[idx]
            self.plotted_img = result.plot(**plot_args)
        # Write
        if self.args.save_txt:
            result.save_txt(f'{self.txt_path}.txt', save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / 'crops', file_name=self.data_path.stem)

        return log_string

    def postprocess(self, preds, img, orig_img):
        """Post-processes predictions for an image and returns them."""
        return preds

    def __call__(self, source=None, model=None, stream=False):
        """Performs inference on an image or stream."""
        self.stream = stream
        if stream:
            return self.stream_inference(source, model)
        else:
            return list(self.stream_inference(source, model))  # merge list of Result into one

    def predict_cli(self, source=None, model=None):
        """Method used for CLI prediction. It uses always generator as outputs as not required by CLI mode."""
        gen = self.stream_inference(source, model)
        for _ in gen:  # running CLI inference without accumulating any outputs (do not modify)
            pass

    def setup_source(self, source):
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = getattr(self.model.model, 'transforms', classify_transforms(
            self.imgsz[0])) if self.args.task == 'classify' else None
        self.dataset = load_inference_source(source=source, imgsz=self.imgsz, vid_stride=self.args.vid_stride)
        self.source_type = self.dataset.source_type
        if not getattr(self, 'stream', True) and (self.dataset.mode == 'stream' or  # streams
                                                  len(self.dataset) > 1000 or  # images
                                                  any(getattr(self.dataset, 'video_flag', [False]))):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_path, self.vid_writer = [None] * self.dataset.bs, [None] * self.dataset.bs

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None):
        """Streams real-time inference on camera feed and saves results to file."""
        if self.args.verbose:
            LOGGER.info('')

        # Setup model
        if not self.model:
            self.setup_model(model)
        # Setup source every time predict is called
        self.setup_source(source if source is not None else self.args.source)

        # Check if save_dir/ label file exists
        if self.args.save or self.args.save_txt:
            (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        # Warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True

        self.seen, self.windows, self.batch, profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile())
        self.run_callbacks('on_predict_start')
        for batch in self.dataset:
            self.run_callbacks('on_predict_batch_start')
            self.batch = batch
            path, im0s, vid_cap, s = batch
            visualize = increment_path(self.save_dir / Path(path[0]).stem,
                                       mkdir=True) if self.args.visualize and (not self.source_type.tensor) else False

            # Preprocess
            with profilers[0]:
                im = self.preprocess(im0s)

            # Inference
            with profilers[1]:
                preds = self.model(im, augment=self.args.augment, visualize=visualize)

            # Postprocess
            with profilers[2]:
                self.results = self.postprocess(preds, im, im0s)
            self.run_callbacks('on_predict_postprocess_end')

            # Visualize, save, write results
            n = len(im0s)
            for i in range(n):
                self.results[i].speed = {
                    'preprocess': profilers[0].dt * 1E3 / n,
                    'inference': profilers[1].dt * 1E3 / n,
                    'postprocess': profilers[2].dt * 1E3 / n}
                if self.source_type.tensor:  # skip write, show and plot operations if input is raw tensor
                    continue
                p, im0 = path[i], im0s[i].copy()
                p = Path(p)

                if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                    s += self.write_results(i, self.results, (p, im, im0))

                if self.args.show and self.plotted_img is not None:
                    self.show(p)

                if self.args.save and self.plotted_img is not None:
                    self.save_preds(vid_cap, i, str(self.save_dir / p.name))
            self.run_callbacks('on_predict_batch_end')
            yield from self.results

            # Print time (inference-only)
            if self.args.verbose:
                LOGGER.info(f'{s}{profilers[1].dt * 1E3:.1f}ms')

        # Release assets
        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
            self.vid_writer[-1].release()  # release final video writer

        # Print results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1E3 for x in profilers)  # speeds per image
            LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape '
                        f'{(1, 3, *self.imgsz)}' % t)
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob('labels/*.txt')))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

        self.run_callbacks('on_predict_end')

    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        device = select_device(self.args.device, verbose=verbose)
        model = model or self.args.model
        self.args.half &= device.type != 'cpu'  # half precision only supported on CUDA
        self.model = AutoBackend(model,
                                 device=device,
                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=verbose)
        self.device = device
        self.model.eval()

    def show(self, p):
        """Display an image in a window using OpenCV imshow()."""
        im0 = self.plotted_img
        if platform.system() == 'Linux' and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(500 if self.batch[3].startswith('image') else 1)  # 1 millisecond

    def save_preds(self, vid_cap, idx, save_path):
        """Save video predictions as mp4 at specified path."""
        im0 = self.plotted_img
        # Save imgs
        if self.dataset.mode == 'image':
            cv2.imwrite(save_path, im0)
        else:  # 'video' or 'stream'
            if self.vid_path[idx] != save_path:  # new video
                self.vid_path[idx] = save_path
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))  # integer required, floats produce error in MP4 codec
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                self.vid_writer[idx] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            self.vid_writer[idx].write(im0)

    def run_callbacks(self, event: str):
        """Runs all registered callbacks for a specific event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func):
        """
        Add callback
        """
        self.callbacks[event].append(func)



class DetectionPredictor(BasePredictor):

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



def predict_yolo(model_path, images):
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
                iou = 0.7)
        save_dir = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/yolo_v8_train"
        predictor = DetectionPredictor(overrides = args,
                                save_dir = save_dir)
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

        result = draw_bounding_boxes(pil_to_tensor(source), 
                                    boxes=ret_boxes[0], 
                                    width=5,) 
                                        #colors=colors)
        #show(result, ret_boxes, labels)
        
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
        model_path = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/runs/detect/pbr_blk_regen_tless_mod_1500/weights/best.pt"
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
            if i % 100 == 0:
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
        np.save(f, fil_scene_ids)
    with open(os.path.join(path, "view_ids.npy"), "wb") as f:
        np.save(f, fil_view_ids)
    with open(os.path.join(path, "bboxes.npy"), "wb") as f:
        np.save(f, fil_bboxes)
    with open(os.path.join(path, "obj_ids.npy"), "wb") as f:
        np.save(f, fil_obj_ids)
    with open(os.path.join(path, "scores.npy"), "wb") as f:
        np.save(f, fil_scores)

    return bboxes, obj_ids


if __name__ == "__main__":
    '''run_and_save(bop_test_path="/home/ise.ros/akshay_work/datasets/tless/test_primesense",
                 detec_save_path="/home/ise.ros/akshay_work/NN_Implementations/surfemb/data/detection_results/tless")
    '''
    run_and_save()
    
    print()