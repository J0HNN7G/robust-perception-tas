"""Visualize pedestrian detection data"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torchvision.transforms as T


# Constants
GT_CONFIG = {
    'color': 'b',
    'linewidth': 1
}

DT_CONFIG = {
    'color': 'y',
    'linewidth': 1
}

# for visualising detections
FIG_WIDTH = 20
FIG_NROWS = 3
FIG_NCOLS = 3
FIG_NUM_IMAGES = FIG_NROWS * FIG_NCOLS
FIG_DPI = 75


# plotting --------------------------------

def bb2plot(bb):
    """
    Convert a bounding box to a set of corner points.

    Parameters:
    - bb (list): Bounding box coordinates [xmin, ymin, xmax, ymax].

    Returns:
    - corners (numpy array): Array of corner points.
    """
    xmin, ymin, xmax, ymax = bb
    corners = np.array([[xmin, ymin], 
                        [xmin, ymax],
                        [xmax, ymax],
                        [xmax, ymin]])
    return corners

def plot_bb(axis, selected_corners, color, linewidth):
    """
    Plot a bounding box on a given axis.

    Parameters:
    - axis (matplotlib axis): The axis to plot on.
    - selected_corners (list): List of corner points.
    - color (str): Color of the bounding box.
    - linewidth (int): Line width for the bounding box.

    Returns:
    None
    """
    prev = selected_corners[-1]
    for corner in selected_corners:
        axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
        prev = corner

def plot_annotations(axis, im, scene_data_bb):
    """
    Plot ground truth annotations on an image.

    Parameters:
    - axis (matplotlib axis): The axis to plot on.
    - im (PIL image): The image to plot on.
    - scene_data_bb (list): List of ground truth bounding boxes.

    Returns:
    None
    """
    axis.imshow(im)
    axis.axis('off')
    axis.set_aspect('equal')
    for bb in scene_data_bb:
        corners = bb2plot(bb)
        plot_bb(axis, corners, **GT_CONFIG)

def plot_results(axis, im, scene_data_bb, scene_data_pred):
    """
    Plot ground truth annotations and predicted bounding boxes on an image.

    Parameters:
    - axis (matplotlib axis): The axis to plot on.
    - im (PIL image): The image to plot on.
    - scene_data_bb (list): List of ground truth bounding boxes.
    - scene_data_pred (list): List of predicted bounding boxes.

    Returns:
    None
    """
    axis.imshow(im)
    axis.axis('off')
    axis.set_aspect('equal')
    
    for bb in scene_data_bb:
        corners = bb2plot(bb)
        plot_bb(axis, corners, **GT_CONFIG)
        
    for bb in scene_data_pred:
        corners = bb2plot(bb)
        plot_bb(axis, corners, **DT_CONFIG)


# drawing (with patches) --------------------------------

def draw_bb(axis, bb, color, linewidth):
    """
    Draw a bounding box on a given axis using patches.

    Parameters:
    - axis (matplotlib axis): The axis to draw on.
    - bb (list): Bounding box coordinates [xmin, ymin, xmax, ymax].
    - color (str): Color of the bounding box.
    - linewidth (int): Line width for the bounding box.

    Returns:
    None
    """
    rect = mpatches.Rectangle((bb[0], bb[1]), width=bb[2]-bb[0], height=bb[3]-bb[1],
                      edgecolor=color, facecolor='none', linewidth=linewidth)
    axis.add_patch(rect)


def draw_predictions(axis, im, scene_data_bb):
    """
    Draw ground truth annotations on an image using patches.

    Parameters:
    - axis (matplotlib axis): The axis to draw on.
    - im (PIL image): The image to draw on.
    - scene_data_bb (list): List of ground truth bounding boxes.

    Returns:
    None
    """
    axis.imshow(im)
    axis.axis('off')
    axis.set_aspect('equal')
    for bb in scene_data_bb:
        draw_bb(axis, bb, **DT_CONFIG)


def draw_annotations(axis, im, scene_data_bb):
    """
    Draw ground truth annotations on an image using patches.

    Parameters:
    - axis (matplotlib axis): The axis to draw on.
    - im (PIL image): The image to draw on.
    - scene_data_bb (list): List of ground truth bounding boxes.

    Returns:
    None
    """
    axis.imshow(im)
    axis.axis('off')
    axis.set_aspect('equal')
    for bb in scene_data_bb:
        draw_bb(axis, bb, **GT_CONFIG)


def draw_results(axis, im, scene_data_bb, scene_data_pred):
    """
    Draw ground truth annotations and predicted bounding boxes on an image using patches.

    Parameters:
    - axis (matplotlib axis): The axis to draw on.
    - im (PIL image): The image to draw on.
    - scene_data_bb (list): List of ground truth bounding boxes.
    - scene_data_pred (list): List of predicted bounding boxes.

    Returns:
    None
    """
    axis.imshow(im)
    axis.axis('off')
    axis.set_aspect('equal')
    
    for bb in scene_data_bb:
        draw_bb(axis, bb, **GT_CONFIG)
        
    for bb in scene_data_pred:
        draw_bb(axis, bb, **DT_CONFIG)


# detection visualization 

def visualize_results(visual_fp, images, gt_bbs, dt_bbs):
    """
    Visualize pedestrian detection results and save them to a file.

    Parameters:
    - visual_fp (str): File path to save the visualization.
    - images (list): List of images to visualize.
    - gt_bbs (list): List of ground truth bounding boxes.
    - dt_bbs (list): List of predicted bounding boxes.

    Returns:
    None
    """
    img_width = images[0].size[0] 
    img_height = images[0].size[1] 
    img_ratio = img_height / img_width

    fig_height = FIG_WIDTH * img_ratio
    fig, _ = plt.subplots(FIG_NROWS, FIG_NCOLS, figsize=(FIG_WIDTH, fig_height))
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    num_visuals = min(len(images), len(fig.axes))
    for i in range(num_visuals):
        draw_results(fig.axes[i], images[i], gt_bbs[i], dt_bbs[i])
    # clear the rest
    for j in range(num_visuals, len(fig.axes)):
        fig.axes[j].axis('off')
        fig.axes[j].set_aspect('equal')
    
    # legend
    gt_patch = mpatches.Patch(color=GT_CONFIG['color'], label='GT')
    dt_patch = mpatches.Patch(color=DT_CONFIG['color'], label='DT')
    fig.axes[1].legend(handles = [gt_patch, dt_patch], loc='upper center', fancybox=False, framealpha=1, borderpad=1, edgecolor='black')

    plt.savefig(visual_fp, dpi=FIG_DPI, bbox_inches='tight')


def show_predictions(images, output):
    """
    Show pedestrian detection predictions.

    Parameters:
    - images (list): List of images to visualize.
    - output (list): List of model predictions.

    Returns:
    None
    """
    img_transform = T.ToPILImage()
    images = list(img_transform(image) for image in images)

    img_width = images[0].size[0] 
    img_height = images[0].size[1] 
    img_ratio = img_height / img_width

    fig_height = FIG_WIDTH * img_ratio
    fig, _ = plt.subplots(FIG_NROWS, FIG_NCOLS, figsize=(FIG_WIDTH, fig_height))
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    num_visuals = min(len(images), len(fig.axes))
    for i in range(num_visuals):
        draw_predictions(fig.axes[i], images[i], output[i]['boxes'].detach().numpy())
    # clear the rest
    for j in range(num_visuals, len(fig.axes)):
        fig.axes[j].axis('off')
        fig.axes[j].set_aspect('equal')
    
    plt.show()


def show_results(images, target, output):
    """
    Show pedestrian detection results

    Parameters:
    - images (list): List of images to show.
    - target (list): List of targets ground truth bounding boxes.
    - output (list): List of model predictions.

    Returns:
    None
    """
    img_transform = T.ToPILImage()
    images = list(img_transform(image) for image in images)

    img_width = images[0].size[0] 
    img_height = images[0].size[1] 
    img_ratio = img_height / img_width

    fig_height = FIG_WIDTH * img_ratio
    fig, _ = plt.subplots(FIG_NROWS, FIG_NCOLS, figsize=(FIG_WIDTH, fig_height))
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    num_visuals = min(len(images), len(fig.axes))
    for i in range(num_visuals):
        draw_results(fig.axes[i], images[i], target[i]['boxes'].detach().numpy(), output[i]['boxes'].detach().numpy())
    # clear the rest
    for j in range(num_visuals, len(fig.axes)):
        fig.axes[j].axis('off')
        fig.axes[j].set_aspect('equal')
    
    # legend
    gt_patch = mpatches.Patch(color=GT_CONFIG['color'], label='GT')
    dt_patch = mpatches.Patch(color=DT_CONFIG['color'], label='DT')
    fig.axes[1].legend(handles = [gt_patch, dt_patch], loc='upper center', fancybox=False, framealpha=1, borderpad=1, edgecolor='black')

    plt.show()

