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
from utils.data_utils import fill_missing
import argparse
from matplotlib.patches import Circle
import base64 
from openai import OpenAI
import matplotlib.colors as mcolors
# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def draw_bboxes_axis(image, bboxes, pred_RTs, pred_scales, intrinsics, axis_length_factor=0.6):
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
        cv2.line(image_copy, tuple(center_2d), tuple(x_axis_2d), (0, 0, 255), 4)

        # Y axis (green)
        y_axis_3d = np.dot(pred_RT, np.array([0, axis_length, 0, 1]))[:3]
        y_axis_2d = [int((y_axis_3d[0] * cam_fx) / y_axis_3d[2] + cam_cx), int((y_axis_3d[1] * cam_fy) / y_axis_3d[2] + cam_cy)]
        cv2.line(image_copy, tuple(center_2d), tuple(y_axis_2d), (0, 255, 0), 4)

        # Z axis (blue)
        z_axis_3d = np.dot(pred_RT, np.array([0, 0, axis_length, 1]))[:3]
        z_axis_2d = [int((z_axis_3d[0] * cam_fx) / z_axis_3d[2] + cam_cx), int((z_axis_3d[1] * cam_fy) / z_axis_3d[2] + cam_cy)]
        cv2.line(image_copy, tuple(center_2d), tuple(z_axis_2d), (255, 0, 0), 4)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
    plt.title('3D Bounding Boxes and Poses')
    plt.show()

def draw_bboxes(image, bboxes, pred_RTs, pred_scales, intrinsics, crop_coords, axis_length_factor=0.6):
    """
    绘制3D边界框并根据给定的裁剪坐标对图像进行裁剪。
    
    :param image: 原始RGB图像
    :param bboxes: 2D边界框数组
    :param pred_RTs: 预测的RT矩阵
    :param pred_scales: 预测的尺度
    :param intrinsics: 相机内参
    :param crop_coords: 裁剪的坐标 (x_min, y_min, x_max, y_max)
    :param axis_length_factor: 坐标轴长度因子
    :return: 裁剪后的图像
    """
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
            cv2.line(image_copy, tuple(bbox_2d[k]), tuple(bbox_2d[(k + 1) % 4]), (0, 255, 0), 4)
            cv2.line(image_copy, tuple(bbox_2d[k + 4]), tuple(bbox_2d[(k + 1) % 4 + 4]), (0, 255, 0), 4)
            cv2.line(image_copy, tuple(bbox_2d[k]), tuple(bbox_2d[k + 4]), (0, 255, 0), 4)
                # Draw coordinate axes at the center point
        center_3d = np.dot(pred_RT, np.array([0, 0, 0, 1]))[:3]
        center_2d = [int((center_3d[0] * cam_fx) / center_3d[2] + cam_cx), int((center_3d[1] * cam_fy) / center_3d[2] + cam_cy)]

        # Axes lengths
        axis_length = axis_length_factor * np.linalg.norm(pred_scale)

        # X axis (red)
        x_axis_3d = np.dot(pred_RT, np.array([axis_length, 0, 0, 1]))[:3]
        x_axis_2d = [int((x_axis_3d[0] * cam_fx) / x_axis_3d[2] + cam_cx), int((x_axis_3d[1] * cam_fy) / x_axis_3d[2] + cam_cy)]
        cv2.line(image_copy, tuple(center_2d), tuple(x_axis_2d), (0, 0, 255), 4)

        # Y axis (green)
        y_axis_3d = np.dot(pred_RT, np.array([0, axis_length, 0, 1]))[:3]
        y_axis_2d = [int((y_axis_3d[0] * cam_fx) / y_axis_3d[2] + cam_cx), int((y_axis_3d[1] * cam_fy) / y_axis_3d[2] + cam_cy)]
        cv2.line(image_copy, tuple(center_2d), tuple(y_axis_2d), (0, 255, 0), 4)

        # Z axis (blue)
        z_axis_3d = np.dot(pred_RT, np.array([0, 0, axis_length, 1]))[:3]
        z_axis_2d = [int((z_axis_3d[0] * cam_fx) / z_axis_3d[2] + cam_cx), int((z_axis_3d[1] * cam_fy) / z_axis_3d[2] + cam_cy)]
        cv2.line(image_copy, tuple(center_2d), tuple(z_axis_2d), (255, 0, 0), 4)

    # 裁剪图像
    x_min, y_min, x_max, y_max = crop_coords
    cropped_image = image_copy[y_min:y_max, x_min:x_max]

    # 显示结果
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    save_filename = os.path.join('vis_result', 'cropped_image_with_bboxes.png')
    plt.savefig(save_filename, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

    return cropped_image


def process_results(results):
    centers = []
    masks = []

    if hasattr(results[0], 'masks'):
        masks = results[0].masks.data.cpu().numpy()
        bboxes = results[0].boxes.xyxy
        print("Bounding boxes:", bboxes)

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[:4]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append((center_x.item(), center_y.item()))
            print(f"Center of bbox (x, y): ({center_x.item()}, {center_y.item()})")

    return centers, masks

def draw_labeled_centers_on_image(image, centers, scale_factor=2.0):
    height, width = image.shape[:2]
    fig, ax = plt.subplots(figsize=(width * scale_factor / 100, height * scale_factor / 100), dpi=100)
    ax.imshow(image)
    ax.axis('off')  # 去除坐标轴
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除外层边距
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    for i, center in enumerate(centers):
        px, py = int(center[0]), int(center[1])
        circle = Circle((px, py), radius=8, color='black')
        ax.add_patch(circle)
        ax.text(px, py, str(i + 1), color='white', ha='center', va='center', size=16)
    fig.canvas.draw()

    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image_from_plot

def draw_labeled_centers_on_image_show(image, centers, scale_factor=2.0, dpi=600):
    height, width = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 调整 dpi 参数以提高分辨率
    fig, ax = plt.subplots(figsize=(width * scale_factor / 100, height * scale_factor / 100), dpi=dpi)
    ax.imshow(image_rgb)
    ax.axis('off')  # 去除坐标轴
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除外层边距
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    
    for i, center in enumerate(centers):
        px, py = int(center[0]), int(center[1])
        circle = Circle((px, py), radius=8, color='black')
        ax.add_patch(circle)
        ax.text(px, py, str(i + 1), color='white', ha='center', va='center', size=16)
    
    fig.canvas.draw()
    # save_filename = os.path.join('.','labeled_centers_image.png')
    # fig.savefig(save_filename, bbox_inches='tight', pad_inches=0)

    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return image_from_plot


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
def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode("utf-8")


def preprocess_image(rgb_image, depth_image, intrinsics, objects, model=None, gpt_client=None, MODEL=None):
    model = FastSAM('kinect/FastSAM-x.pt')
    if gpt_client is None:
        gpt_client = OpenAI(api_key='') #get you openai-api to run
    
    if MODEL is None:
        MODEL = "gpt-4o"
    points = np.array([[366, 854], [1113, 850], [1100, 400], [327, 404]])
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    cropped_rgb_image = rgb_image[y_min:y_max, x_min:x_max]
    plt.imshow(cv2.cvtColor(cropped_rgb_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    save_filename = os.path.join('vis_result', 'imput.png')
    plt.savefig(save_filename, bbox_inches='tight', pad_inches=0)
    plt.show()

    results = model.predict(cropped_rgb_image, conf=0.93)
    centers, masks = process_results(results)
    labeled_pic = draw_labeled_centers_on_image(cropped_rgb_image, centers)
    labeled_pic_show = draw_labeled_centers_on_image_show(cropped_rgb_image, centers)
    plt.imshow(labeled_pic_show)
    plt.axis('off')
    save_filename = os.path.join('vis_result', 'labeled.png')
    plt.savefig(save_filename, bbox_inches='tight', pad_inches=0)
    plt.show()
    base64_image = encode_image(labeled_pic)
    try:
        response = gpt_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是一个视觉检测专家"},
                {"role": "user", "content": [
                    {"type": "text", "text": f"识别图中的{objects}，并只输出其的序号，每个序号之间用豆号隔开，无需输出任何其他的信息"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpg;base64,{base64_image}"}
                    }
                ]}
            ],
            temperature=0.0,
        )
        gpt_response = response.choices[0].message.content.strip()
        print("GPT response:", gpt_response)
    except Exception as e:
        print(f"Error calling GPT model: {e}")
        return
    gpt_indices = [int(idx) for idx in gpt_response.split(",") if idx.strip().isdigit()]

    # Map GPT indices to original data indices
    mapped_indices = []
    for gpt_idx in gpt_indices:
        if 0 <= gpt_idx - 1 < len(centers):  # Ensure the index is within bounds
            mapped_indices.append(gpt_idx - 1)  # Subtract 1 because GPT likely uses 1-based indexing
    
    pred_masks = results[0].masks.data.cpu().numpy()  # Multiple masks
    bboxes = results[0].boxes.xyxy.cpu().numpy()

    cam_fx, cam_fy, cam_cx, cam_cy = intrinsics
    depth_image = load_depth(depth_image)
    depth_cropped = depth_image[y_min:y_max, x_min:x_max]

    depth_filled = fill_missing(depth_cropped, 1000, 1)
    plt.imshow(depth_cropped)
    plt.axis('off')
    save_filename = os.path.join('vis_result', 'depth.png')
    plt.savefig(save_filename, bbox_inches='tight', pad_inches=0)
    plt.show()

    xmap = np.array([[i for i in range(x_min, x_max)] for j in range(y_min, y_max)])
    ymap = np.array([[j for i in range(x_min, x_max)] for j in range(y_min, y_max)])
    pts2 = depth_filled.copy() / 1000.0
    pts0 = (xmap - cam_cx) * pts2 / cam_fx
    pts1 = (ymap - cam_cy) * pts2 / cam_fy
    pts = np.transpose(np.stack([pts0, pts1, pts2]), (1, 2, 0)).astype(np.float32)

    instances_rgb, instances_pts, instances_choose = [], [], []
    all_choose_points = []

    for j, pred_mask in enumerate(pred_masks):
        if j not in mapped_indices:
            continue
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
            # instance_rgb = np.array(instance_rgb)
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

    if instances_rgb:
        instances_rgb = torch.stack(instances_rgb)
        instances_pts = torch.FloatTensor(instances_pts)
        instances_choose = torch.LongTensor(instances_choose)
    else:
        print("No valid instances found.")
        return

    # visualize_choose(cropped_rgb_image, all_choose_points, mapped_indices)
    visualize_pred_masks(cropped_rgb_image, pred_masks, mapped_indices)

    return instances_rgb, instances_pts, pred_masks, bboxes, instances_choose


def visualize_pred_masks(cropped_rgb_image, pred_masks, gpt_indices):
    # Create a blank image with the same dimensions as cropped_rgb_image for all masks
    all_masks_overlay = np.zeros_like(cropped_rgb_image)

    # Create a blank image with the same dimensions as cropped_rgb_image for selected masks
    selected_masks_overlay = np.zeros_like(cropped_rgb_image)

    # List of colors for different masks
    colors = list(mcolors.CSS4_COLORS.values())

    # Overlay all masks
    for i, mask in enumerate(pred_masks):
        # Resize the mask to match the dimensions of cropped_rgb_image
        mask_resized = cv2.resize(mask, (cropped_rgb_image.shape[1], cropped_rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        color = colors[i % len(colors)]  # Cycle through the list of colors
        color_rgb = mcolors.hex2color(color)  # Convert to RGB
        color_rgb = [int(c * 255) for c in color_rgb]  # Scale to 0-255

        # Apply the resized mask with the chosen color
        all_masks_overlay[mask_resized > 0] = color_rgb

    # Display all masks overlay
    all_overlayed_image = cv2.addWeighted(cropped_rgb_image, 0.3, all_masks_overlay, 0.7, 20)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(all_overlayed_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    save_filename = os.path.join('vis_result', 'all sam.png')
    plt.savefig(save_filename, bbox_inches='tight', pad_inches=0)
    plt.show()

    # Overlay only the selected masks
    for i in gpt_indices:
        mask = pred_masks[i]
        
        # Resize the mask to match the dimensions of cropped_rgb_image
        mask_resized = cv2.resize(mask, (cropped_rgb_image.shape[1], cropped_rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        color = colors[i % len(colors)]  # Cycle through the list of colors
        color_rgb = mcolors.hex2color(color)  # Convert to RGB
        color_rgb = [int(c * 255) for c in color_rgb]  # Scale to 0-255

        # Apply the resized mask with the chosen color
        selected_masks_overlay[mask_resized > 0] = color_rgb

    # Combine the original image with the selected masks overlay
    overlayed_image = cv2.addWeighted(cropped_rgb_image, 0.3, selected_masks_overlay, 0.7, 20)

    # Display the selected masks overlay
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    save_filename = os.path.join('vis_result', 'gpt sam.png')
    plt.savefig(save_filename, bbox_inches='tight', pad_inches=0)
    plt.show()


def visualize_choose(image, all_choose_points, mapped_indices):
    vis_image = image.copy()

    for i in mapped_indices:
        if i < len(all_choose_points):  # Ensure index is within bounds
            choose, rmin, cmin, height, width = all_choose_points[i-1]
            col_idx = choose % width
            row_idx = choose // width
            for j in range(len(choose)):
                cv2.circle(vis_image, (cmin + col_idx[j], rmin + row_idx[j]), 2, (0, 255, 0), -1)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.title('Selected Choose Visualization')
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
    logger.info("=> creating model ...")
    from model.Lightpose import Lightpose
    model = Lightpose(cfg.num_category, cfg.freeze_world_enhancer)
    

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
    points = np.array([[366, 854], [1113, 850], [1100, 400], [327, 404]])
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    # Preprocess image
    intrinsics = [913.0894775390625, 912.6361694335938, 960.84765625, 551.1499633789062]
    
    # Assuming objects description is passed correctly
    instances_rgb, instances_pts, pred_masks, bboxes, instances_choose = preprocess_image(
        rgb_image, depth_image, intrinsics, objects="i need a Toy Mango "
    )
    
    # Initialize lists for storing results
    results = []
    all_bboxes = []
    all_pred_RTs = []
    all_pred_scales = []

    # Only process selected bboxes
    for i, idx in enumerate(instances_choose):
        # Prepare input for the model
        inputs = {
            'rgb': instances_rgb[i].unsqueeze(0).cuda(),
            'pts': instances_pts[i].unsqueeze(0).cuda(),
            'choose': instances_choose[i].unsqueeze(0).cuda(),
            'category_label': torch.LongTensor([1]).cuda()  # Modify as per your requirement
        }
        
        with torch.no_grad():
            end_points = model(inputs)

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
    crop_coords = (x_min, y_min, x_max, y_max)
    cropped_image_with_bboxes = draw_bboxes(
    rgb_image,
    all_bboxes,
    np.concatenate(all_pred_RTs),
    np.concatenate(all_pred_scales),
    intrinsics,
    crop_coords
    )

    # Save results
    save_path = './results'
    os.makedirs(save_path, exist_ok=True)
    save_filename = os.path.join(save_path, 'result.pkl')
    with open(save_filename, 'wb') as f:
        cPickle.dump(results, f)


if __name__ == "__main__":
    main()
