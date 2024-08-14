import os
import time
import torch
import numpy as np
import cv2
from pyk4a import PyK4A, Config
from ultralytics import FastSAM
from torchvision import transforms
import matplotlib.pyplot as plt
import logging
import _pickle as cPickle
import gorilla
import pyk4a
import matplotlib.patches as patches
from utils.data_utils import fill_missing
from agpose.model.Net import Net
import argparse
import matplotlib.colors as mcolors
# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# def get_parser():
#     parser = argparse.ArgumentParser(
#         description="Pose Estimation")

#     # pretrain
#     parser.add_argument("--gpus",
#                         type=str,
#                         default="0",
#                         help="gpu num")
#     parser.add_argument("--config",
#                         type=str,
#                         default = "config/ist_net_default.yaml",
#                         help="path to config file")
#     parser.add_argument("--test_epoch",
#                         type=int,
#                         default=10,
#                         help="test epoch")
#     parser.add_argument("--cat_id",
#                         type=int,
#                         default=-1,
#                         help="category id, -1 for mean aps")
#     parser.add_argument('--mask_label', action='store_true', default=False,
#                         help='whether having mask labels of real data')
#     parser.add_argument('--only_eval', action='store_true', default=False,
#                         help='whether directly evaluating the results')
#     args_cfg = parser.parse_args()
#     return args_cfg

# def init():
#     args = get_parser()
#     exp_name = args.config.split("/")[-1].split(".")[0]
#     log_dir = os.path.join("log", exp_name)

#     cfg = gorilla.Config.fromfile(args.config)
#     cfg.exp_name = exp_name
#     cfg.log_dir = log_dir
#     os.makedirs(log_dir, exist_ok=True)
#     cfg.ckpt_dir = os.path.join(cfg.log_dir, 'ckpt')
#     cfg.gpus = args.gpus
#     cfg.test_epoch = args.test_epoch
#     cfg.mask_label = args.mask_label
#     cfg.only_eval = args.only_eval
#     cfg.cat_id = args.cat_id

#     gorilla.utils.set_cuda_visible_devices(gpu_ids = cfg.gpus)

#     return logger, cfg

def draw_bboxes(image, bboxes, pred_RTs, pred_scales, intrinsics, axis_length_factor=0.6):
    cam_fx, cam_fy, cam_cx, cam_cy = intrinsics
    image_copy = image.copy()

    for i, bbox in enumerate(bboxes):
        # Get the 3D points from pred_RTs and pred_scales
        pred_RT = pred_RTs[i]
        pred_scale = pred_scales[i]

        # Define 3D bounding box corners
        bbox_3d = np.array([
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ]) * pred_scale

        bbox_3d_hom = np.hstack((bbox_3d, np.ones((8, 1))))
        bbox_3d_transformed = np.dot(pred_RT, bbox_3d_hom.T).T

        # Project 3D points to 2D
        bbox_2d = np.zeros((8, 2), dtype=np.int32)
        for j in range(8):
            X, Y, Z = bbox_3d_transformed[j, :3]
            x = int((X * cam_fx) / Z + cam_cx)
            y = int((Y * cam_fy) / Z + cam_cy)
            bbox_2d[j] = [x, y]

        # Draw 3D bounding box on 2D image
        for k in range(4):
            cv2.line(image_copy, tuple(bbox_2d[k]), tuple(bbox_2d[(k + 1) % 4]), (255, 0, 0), 4)
            cv2.line(image_copy, tuple(bbox_2d[k + 4]), tuple(bbox_2d[(k + 1) % 4 + 4]), (0, 255, 0), 4)
            cv2.line(image_copy, tuple(bbox_2d[k]), tuple(bbox_2d[k + 4]), (0, 0, 255), 4)

        # Draw coordinate axes at the center point
        # center_3d = np.dot(pred_RT, np.array([0, 0, 0, 1]))[:3]
        # center_2d = [int((center_3d[0] * cam_fx) / center_3d[2] + cam_cx), int((center_3d[1] * cam_fy) / center_3d[2] + cam_cy)]

        # # Axes lengths
        # axis_length = axis_length_factor * np.linalg.norm(pred_scale)

        # # X axis (red)
        # x_axis_3d = np.dot(pred_RT, np.array([axis_length, 0, 0, 1]))[:3]
        # x_axis_2d = [int((x_axis_3d[0] * cam_fx) / x_axis_3d[2] + cam_cx), int((x_axis_3d[1] * cam_fy) / x_axis_3d[2] + cam_cy)]
        # cv2.line(image_copy, tuple(center_2d), tuple(x_axis_2d), (0, 0, 255), 2)

        # # Y axis (green)
        # y_axis_3d = np.dot(pred_RT, np.array([0, axis_length, 0, 1]))[:3]
        # y_axis_2d = [int((y_axis_3d[0] * cam_fx) / y_axis_3d[2] + cam_cx), int((y_axis_3d[1] * cam_fy) / y_axis_3d[2] + cam_cy)]
        # cv2.line(image_copy, tuple(center_2d), tuple(y_axis_2d), (0, 255, 0), 2)

        # # Z axis (blue)
        # z_axis_3d = np.dot(pred_RT, np.array([0, 0, axis_length, 1]))[:3]
        # z_axis_2d = [int((z_axis_3d[0] * cam_fx) / z_axis_3d[2] + cam_cx), int((z_axis_3d[1] * cam_fy) / z_axis_3d[2] + cam_cy)]
        # cv2.line(image_copy, tuple(center_2d), tuple(z_axis_2d), (255, 0, 0), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
    plt.title('3D Bounding Boxes and Poses')
    plt.show()


def get_bbox(bbox):
    """ Compute square image crop window based on bbox. """
    x1, y1, x2, y2 = bbox
    img_width = 1080
    img_length = 1920
    window_size = (max(y2 - y1, x2 - x1) // 40 + 1) * 40
    window_size = min(window_size, 400)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

# Configuration
class config:
    num_category = 6
    gpus = "0"
    log_dir = "/media/midea2/787d7b0f-26f9-4255-a892-756df38d1c15/home/midea1/Desktop/IST-Net/log/ist_net_default"
    test_epoch = 200

cfg = config()

def capture_image():
    k4a = PyK4A(Config(color_resolution=pyk4a.ColorResolution.RES_1080P,
                        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
                        camera_fps=pyk4a.FPS.FPS_30,
                        synchronized_images_only=True))
    k4a.start()

    capture = k4a.get_capture()
    if capture.color is None or capture.transformed_depth is None:
        k4a.stop()
        raise Exception("Failed to capture images.")

    rgb_image = capture.color[:, :, :3]
    depth_image = capture.transformed_depth
    k4a.stop()
    return rgb_image, depth_image

def load_depth(img):
    """ Load depth image from img_path. """
    depth = img
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16==32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16


def preprocess_image(rgb_image, depth_image, intrinsics):
    model = FastSAM('/media/midea2/787d7b0f-26f9-4255-a892-756df38d1c15/home/midea1/Desktop/IST-Net/kinect/FastSAM-x.pt')
    points = np.array([[366, 854], [1113, 850], [1100, 400], [327, 404]])
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    cropped_rgb_image = rgb_image[y_min:y_max, x_min:x_max]

    results = model.predict(cropped_rgb_image, conf=0.95)
    pred_masks = results[0].masks.data.cpu().numpy()  # Multiple masks
    bboxes = results[0].boxes.xyxy.cpu().numpy()

    cam_fx, cam_fy, cam_cx, cam_cy = intrinsics
    depth_image = load_depth(depth_image)
    depth_cropped = depth_image[y_min:y_max, x_min:x_max]

    depth_filled = fill_missing(depth_cropped, 1000, 1)

    xmap = np.array([[i for i in range(x_min, x_max)] for j in range(y_min, y_max)])
    ymap = np.array([[j for i in range(x_min, x_max)] for j in range(y_min, y_max)])
    pts2 = depth_filled.copy() / 1000.0
    pts0 = (xmap - cam_cx) * pts2 / cam_fx
    pts1 = (ymap - cam_cy) * pts2 / cam_fy
    pts = np.transpose(np.stack([pts0, pts1, pts2]), (1, 2, 0)).astype(np.float32)

    instances_rgb, instances_pts, instances_choose = [], [], []
    all_choose_points = []

    for j, pred_mask in enumerate(pred_masks):
        inst_mask = 255 * pred_mask.astype('uint8')

        # Resize depth_filled to match the inst_mask dimensions
        depth_filled_resized = cv2.resize(depth_filled, (inst_mask.shape[1], inst_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        rmin, rmax, cmin, cmax = get_bbox(bboxes[j])

        rmin, rmax, cmin, cmax = int(rmin), int(rmax), int(cmin), int(cmax)

        mask = inst_mask > 0
        mask = np.logical_and(mask, depth_filled_resized > 0)
        mask_resized = cv2.resize(inst_mask, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_NEAREST)

        choose = mask_resized[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        all_choose_points.append((choose, rmin, cmin, rmax-rmin, cmax-cmin))

        instance_rgb, instance_pts, instance_choose = None, None, None
        num_sample = 1024

        if len(choose) > 16:
            if len(choose) <= num_sample:
                choose_idx = np.random.choice(len(choose), num_sample)
            else:
                choose_idx = np.random.choice(len(choose), num_sample, replace=False)
            choose = choose[choose_idx]

            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])
            instance_pts = pts[rmin:rmax, cmin:cmax, :].reshape((-1, 3))[choose, :]

            instance_rgb = cropped_rgb_image[rmin:rmax, cmin:cmax, :].copy()
            instance_rgb = cv2.resize(instance_rgb, (192, 192), interpolation=cv2.INTER_LINEAR)

            instance_rgb = transform(np.array(instance_rgb))
            crop_w = rmax - rmin
            ratio = 192/ crop_w
            col_idx = choose % crop_w
            row_idx = choose // crop_w
            choose = (np.floor(row_idx * ratio) * 192+ np.floor(col_idx * ratio)).astype(np.int64)

        if instance_rgb is None or instance_pts is None:
            instance_rgb = torch.zeros((3, 192, 192))
            instance_pts = np.zeros((num_sample, 3))
            choose = np.zeros(num_sample)

        instances_rgb.append(instance_rgb)
        instances_pts.append(instance_pts)
        instances_choose.append(choose)

    instances_rgb = torch.stack(instances_rgb)
    instances_pts = torch.FloatTensor(instances_pts)
    instances_choose = torch.LongTensor(instances_choose)

    visualize_choose(cropped_rgb_image, all_choose_points)
    visualize_pred_masks(cropped_rgb_image, pred_masks,gpt_indices=[1])

    return instances_rgb, instances_pts, pred_masks, bboxes, instances_choose


def visualize_choose(image, all_choose_points):
    vis_image = image.copy()

    for choose, rmin, cmin, height, width in all_choose_points:
        col_idx = choose % width
        row_idx = choose // width
        for i in range(len(choose)):
            cv2.circle(vis_image, (cmin + col_idx[i], rmin + row_idx[i]), 2, (0, 255, 0), -1)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.title('Choose Visualization')
    plt.show()


def visualize_pred_masks(cropped_rgb_image, pred_masks, gpt_indices):
    # Create a blank image with the same dimensions as cropped_rgb_image
    mask_overlay = np.zeros_like(cropped_rgb_image)

    # List of colors for different masks
    colors = list(mcolors.CSS4_COLORS.values())

    for i in gpt_indices:
        mask = pred_masks[i]
        
        # Resize the mask to match the dimensions of cropped_rgb_image
        mask_resized = cv2.resize(mask, (cropped_rgb_image.shape[1], cropped_rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        color = colors[i % len(colors)]  # Cycle through the list of colors
        color_rgb = mcolors.hex2color(color)  # Convert to RGB
        color_rgb = [int(c * 255) for c in color_rgb]  # Scale to 0-255

        # Apply the resized mask with the chosen color
        mask_overlay[mask_resized > 0] = color_rgb

    # Combine the original image with the mask overlay
    overlayed_image = cv2.addWeighted(cropped_rgb_image, 0.5, mask_overlay, 0.5, 0)

    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB))
    plt.title('Selected Predicted Masks Overlay')
    plt.axis('off')
    plt.show()

def visualize_result(rgb_image, result):
    points = np.array([[366, 854], [1113, 850], [1100, 400], [327, 404]])
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    cropped_rgb_image = rgb_image[y_min:y_max, x_min:x_max]
    plt.figure(figsize=(10, 10))
    plt.imshow(cropped_rgb_image)
    for bbox in result['pred_bboxes']:
        x1, y1, x2, y2 = bbox
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', facecolor='none', linewidth=2))
    plt.show()

def main():
    # Model Initialization
    # logger, cfg = init()
    logger.info("=> creating model ...")
    from model.Lightpose import Lightpose
    # model = Net(cfg.pose_net)
    # model = model.cuda()
    model = Lightpose(cfg.num_category)

    if len(cfg.gpus) > 1:
        model = torch.nn.DataParallel(model, range(len(cfg.gpus.split(","))))
    model = model.cuda()

    # Checkpoint loading
    checkpoint = os.path.join(cfg.log_dir, 'epoch_' + str(cfg.test_epoch) + '.pth')
    logger.info("=> loading checkpoint from path: {} ...".format(checkpoint))
    gorilla.solver.load_checkpoint(model=model, filename=checkpoint)

    model.eval()

    # Capture image from Kinect
    rgb_image, depth_image = capture_image()
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # RGB Visualization
    axes[0].imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('RGB Visualization')
    axes[0].axis('off')

    # Depth Visualization
    axes[1].imshow(cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Depth Visualization')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


    # Preprocess image
    intrinsics = [913.0894775390625, 912.6361694335938, 960.84765625, 551.1499633789062]
    instances_rgb, instances_pts, pred_masks, bboxes, instances_choose = preprocess_image(rgb_image, depth_image, intrinsics)

    results = []
    all_bboxes = []
    all_pred_RTs = []
    all_pred_scales = []

    for i in range(len(bboxes)):
        # Prepare input for the model
        inputs = {
            'rgb': instances_rgb[i].unsqueeze(0).cuda(),
            'pts': instances_pts[i].unsqueeze(0).cuda(),
            'choose': instances_choose[i].unsqueeze(0).cuda(),
            'category_label': torch.LongTensor([1]).cuda()  # Modify as per your requirement
        }
        
                # self.class_name_map = {
        #                        0: 'real',
        #                        1: 'bottle_',
        #                        2: 'bowl_',
        #                        3: 'camera_',
        #                        4: 'can_',
        #                        5: 'laptop_',
        #     

        # Inference
        # start_time = time.time()
        with torch.no_grad():
            end_points = model(inputs)
        # inference_time = time.time() - start_time
        # logger.info(f"Inference time: {inference_time:.4f} seconds")

        # Process the output
        pred_translation = end_points['pred_translation']
        pred_size = end_points['pred_size']
        pred_scale = torch.norm(pred_size, dim=1, keepdim=True)
        pred_size = pred_size / pred_scale
        pred_rotation = end_points['pred_rotation']

        num_instance = pred_rotation.size(0)
        pred_RTs = torch.eye(4).unsqueeze(0).repeat(num_instance, 1, 1).float().to(pred_rotation.device)
        pred_RTs[:, :3, 3] = pred_translation
        pred_RTs[:, :3, :3] = pred_rotation * pred_scale.unsqueeze(2)
        pred_scales = pred_size

        all_bboxes.append(bboxes[i])
        all_pred_RTs.append(pred_RTs.detach().cpu().numpy())
        all_pred_scales.append(pred_scales.detach().cpu().numpy())

        result = {
            'pred_bboxes': bboxes[i],
            'pred_RTs': pred_RTs.detach().cpu().numpy(),
            'pred_scales': pred_scales.detach().cpu().numpy()
        }
        results.append(result)

    # Visualize all results in one image
    draw_bboxes(rgb_image, all_bboxes, np.concatenate(all_pred_RTs), np.concatenate(all_pred_scales), intrinsics)

    # Save results
    save_path = './results'
    os.makedirs(save_path, exist_ok=True)
    save_filename = os.path.join(save_path, 'result.pkl')
    with open(save_filename, 'wb') as f:
        cPickle.dump(results, f)

if __name__ == "__main__":
    main()
