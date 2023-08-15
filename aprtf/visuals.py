import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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
    xmin, ymin, xmax, ymax = bb
    corners = np.array([[xmin, ymin], 
                        [xmin, ymax],
                        [xmax, ymax],
                        [xmax, ymin]])
    return corners

def plot_bb(axis, selected_corners, color, linewidth):
    prev = selected_corners[-1]
    for corner in selected_corners:
        axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
        prev = corner

def plot_annotations(axis, im, scene_data_bb):
    axis.imshow(im)
    axis.axis('off')
    axis.set_aspect('equal')
    for bb in scene_data_bb:
        corners = bb2plot(bb)
        plot_bb(axis, corners, **GT_CONFIG)

def plot_results(axis, im, scene_data_bb, scene_data_pred):
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
    rect = mpatches.Rectangle((bb[0], bb[1]), width=bb[2]-bb[0], height=bb[3]-bb[1],
                      edgecolor=color, facecolor='none', linewidth=linewidth)
    axis.add_patch(rect)

def draw_annotations(axis, im, scene_data_bb):
    axis.imshow(im)
    axis.axis('off')
    axis.set_aspect('equal')
    for bb in scene_data_bb:
        draw_bb(axis, bb, **GT_CONFIG)

def draw_results(axis, im, scene_data_bb, scene_data_pred):
    axis.imshow(im)
    axis.axis('off')
    axis.set_aspect('equal')
    
    for bb in scene_data_bb:
        draw_bb(axis, bb, **GT_CONFIG)
        
    for bb in scene_data_pred:
        draw_bb(axis, bb, **DT_CONFIG)


# detection visualization 

def visualize_results(visual_fp, images, gt_bbs, dt_bbs):
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


