import os
import time
import torch
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import FastSAM
from torchvision import transforms
import matplotlib.pyplot as plt
import logging
import _pickle as cPickle
import gorilla

from utils.data_utils import fill_missing


# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            cv2.line(image_copy, tuple(bbox_2d[k]), tuple(bbox_2d[(k + 1) % 4]), (255, 0, 0), 2)
            cv2.line(image_copy, tuple(bbox_2d[k + 4]), tuple(bbox_2d[(k + 1) % 4 + 4]), (255, 0, 0), 2)
            cv2.line(image_copy, tuple(bbox_2d[k]), tuple(bbox_2d[k + 4]), (255, 0, 0), 2)

        # Draw coordinate axes at the center point
        center_3d = np.dot(pred_RT, np.array([0, 0, 0, 1]))[:3]
        center_2d = [int((center_3d[0] * cam_fx) / center_3d[2] + cam_cx), int((center_3d[1] * cam_fy) / center_3d[2] + cam_cy)]

        # Axes lengths
        axis_length = axis_length_factor * np.linalg.norm(pred_scale)

        # X axis (red)
        x_axis_3d = np.dot(pred_RT, np.array([axis_length, 0, 0, 1]))[:3]
        x_axis_2d = [int((x_axis_3d[0] * cam_fx) / x_axis_3d[2] + cam_cx), int((x_axis_3d[1] * cam_fy) / x_axis_3d[2] + cam_cy)]
        cv2.line(image_copy, tuple(center_2d), tuple(x_axis_2d), (0, 0, 255), 2)

        # Y axis (green)
        y_axis_3d = np.dot(pred_RT, np.array([0, axis_length, 0, 1]))[:3]
        y_axis_2d = [int((y_axis_3d[0] * cam_fx) / y_axis_3d[2] + cam_cx), int((y_axis_3d[1] * cam_fy) / y_axis_3d[2] + cam_cy)]
        cv2.line(image_copy, tuple(center_2d), tuple(y_axis_2d), (0, 255, 0), 2)

        # Z axis (blue)
        z_axis_3d = np.dot(pred_RT, np.array([0, 0, axis_length, 1]))[:3]
        z_axis_2d = [int((z_axis_3d[0] * cam_fx) / z_axis_3d[2] + cam_cx), int((z_axis_3d[1] * cam_fy) / z_axis_3d[2] + cam_cy)]
        cv2.line(image_copy, tuple(center_2d), tuple(z_axis_2d), (255, 0, 0), 2)

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
    freeze_world_enhancer = False
    gpus = "0"
    log_dir = "/home/midea1/Desktop/Efficient6d/log/efficient6d"
    test_epoch = 200

cfg = config()

def capture_image():
    
    pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream different resolutions of color and depth
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    rgb_image = np.asanyarray(color_frame.get_data())

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
    model = FastSAM('kinect/FastSAM-s.pt')

    results = model.predict(rgb_image, conf=0.9)
    pred_masks = results[0].masks.data.cpu().numpy()  # Multiple masks
    bboxes = results[0].boxes.xyxy.cpu().numpy()

    cam_fx, cam_fy, cam_cx, cam_cy = intrinsics
    depth_image = load_depth(depth_image)
    depth_cropped = depth_image

    depth_filled = fill_missing(depth_cropped, 1000, 1)

    xmap = np.array([[i for i in range(640)] for j in range(480)])
    ymap = np.array([[j for i in range(640)] for j in range(480)])
    pts2 = depth_filled.copy() / 1000.0
    pts0 = (xmap - cam_cx) * pts2 / cam_fx
    pts1 = (ymap - cam_cy) * pts2 / cam_fy
    pts = np.transpose(np.stack([pts0, pts1, pts2]), (1, 2, 0)).astype(np.float32)

    instances_rgb, instances_pts, instances_choose = [], [], []

    for j, pred_mask in enumerate(pred_masks):
        inst_mask = 255 * pred_mask.astype('uint8')

        # Resize depth_filled to match the inst_mask dimensions
        depth_filled_resized = cv2.resize(depth_filled, (inst_mask.shape[1], inst_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        rmin, rmax, cmin, cmax = get_bbox(bboxes[j])

        rmin, rmax, cmin, cmax = int(rmin), int(rmax), int(cmin), int(cmax)

        mask = inst_mask > 0
        mask = np.logical_and(mask, depth_filled_resized > 0)
        # mask_resized = cv2.resize(inst_mask, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_NEAREST)

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        # visualize_choose(cropped_rgb_image, choose, rmin, cmin, rmax-rmin, cmax-cmin)

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

            instance_rgb = rgb_image[rmin:rmax, cmin:cmax, :].copy()
            instance_rgb = cv2.resize(instance_rgb, (192, 192), interpolation=cv2.INTER_LINEAR)

            instance_rgb = transform(np.array(instance_rgb))
            crop_w = rmax - rmin
            ratio = 192 / crop_w
            col_idx = choose % crop_w
            row_idx = choose // crop_w
            choose = (np.floor(row_idx * ratio) * 192 + np.floor(col_idx * ratio)).astype(np.int64)

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

    return instances_rgb, instances_pts, pred_masks, bboxes, instances_choose


def visualize_choose(image, choose, rmin, cmin, height, width):
    # Create a blank image with the same size as the original image
    vis_image = image.copy()
    
    # Map choose indices to 2D coordinates
    col_idx = choose % width
    row_idx = choose // width
    
    # Plot each point in the choose
    for i in range(len(choose)):
        cv2.circle(vis_image, (cmin + col_idx[i], rmin + row_idx[i]), 2, (0, 255, 0), -1)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.title('Choose Visualization')
    plt.show()

def visualize_pts(pts):
    # Visualize 3D points using matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title('3D Points Visualization')
    plt.show()


def visualize_result(rgb_image, result):
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)
    for bbox in result['pred_bboxes']:
        x1, y1, x2, y2 = bbox
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', facecolor='none', linewidth=2))
    plt.show()

def main():
    # Model Initialization
    logger.info("=> creating model ...")
    from model.efficient6d import Efficient6d
    model = Efficient6d(cfg.num_category, cfg.freeze_world_enhancer)

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

    # Preprocess image
    intrinsics = [386.346, 385.842, 318.046, 239.748]
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
            'category_label': torch.LongTensor([4]).cuda()  # Modify as per your requirement
        }
        
        # self.class_name_map = {
        #                        0: 'real',
        #                        1: 'bottle_',
        #                        2: 'bowl_',
        #                        3: 'camera_',
        #                        4: 'can_',
        #                        5: 'laptop_',
        #                        6: 'mug_'}

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

