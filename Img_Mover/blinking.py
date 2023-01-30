import numpy as np
from PIL import Image
from rembg import remove
import PIL.Image
import numpy
from tha2.util import extract_pytorch_image_from_PIL_image, convert_output_image_from_torch_to_numpy
import tha2.poser.modes.mode_20
import time
import threading
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
import random


device = torch.device("cuda:0") if torch.has_cuda else torch.device("cpu")



poser = tha2.poser.modes.mode_20.create_poser(device)
pose_parameters = tha2.poser.modes.mode_20.get_pose_parameters()
pose_size = poser.get_num_parameters()
last_pose = torch.zeros(1, pose_size).to(device)


# Converts an image with a plain background to one
# with a clear background.
# Input:
#   PIL image
# Ouput:
#   PIL image of the image foreground
#   Numpy array of the image background
def remove_bg(img):
    global img2
    img2 = img
    fg = remove(img)
    bg = np.array(img.convert("RGBA"))-np.array(fg)
    return fg, bg


def get_pytorch_image(pytorch_image, output_widget=None, numpy_bg=None):
    output_image = pytorch_image.detach().cpu()
    numpy_image = numpy.uint8(numpy.rint(convert_output_image_from_torch_to_numpy(output_image) * 255.0))
    
    # If the background exists, reapply it so the image isn't so plain
    if numpy_bg is not None:
        numpy_image += numpy_bg
        
    pil_image = PIL.Image.fromarray(numpy_image, mode='RGBA')
    return pil_image


def change_pose(pose, torch_input_image, numpy_bg):
    # Now let's try to get the output image with no posing
    output_image = poser.pose(torch_input_image, pose)[0]
    
    # Get the image
    return get_pytorch_image(output_image, numpy_bg=numpy_bg)


def load_img(path):
    # Load in the image
    img = Image.open(path).convert("RGB")
    
    # Image must be 256x256
    img = img.resize((256, 256))
        
    # Remove the background from the image and save the background
    # for later use
    img, bg = remove_bg(img)

    # Save the numpy background so that it can
    # be reapplied when showing the image
    # Note: alpha values with 0 and 100% transparent and alpha values at 255
    # are 100 not transparent. So instead of keeping the trash artifacts
    # that may mess up the original image when adding this mask to it, we
    # can just remove anything we don't want. Since the BG is found really well,
    # this method mostly works and has a slight issue around the border, but
    # it's better than having annoying artifacts.
    numpy_bg = Image.fromarray(numpy.where(bg.transpose(2, 0, 1)[-1] > 10, bg.transpose(2, 0, 1), 0).transpose(1, 2, 0))
    
    # Make the image a torch tensor
    torch_input_image = extract_pytorch_image_from_PIL_image(img).to(device)
    
    return torch_input_image, numpy_bg


def main():
    # Load in the image
    torch_input_image, numpy_bg = load_img("data/illust/waifu_00.png")

    # Posing vector
    pose = torch.zeros((42)).to(device)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    img = change_pose(pose, torch_input_image, numpy_bg)
    im = ax.imshow(img, animated=True)

    eye_percent = [0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0, 0]
    dilate_percent = [0, 0.2, 0.4, 0.8, 0.4, 0.2, 0, 0, 0, 0]

    def update_image(i):
        # Get the percentages
        eye_per = eye_percent[i%len(eye_percent)]
        dilate_per = dilate_percent[i%len(dilate_percent)]

        # Make the eyes close
        pose[12] = eye_per
        pose[13] = eye_per

        # Make the eyes dilate
        pose[24] = dilate_per
        pose[25] = dilate_per
        
        # for item in range(0, len(pose)):
        #     pose[item] = random.random()
        # for item in range(37, 42):
        #     pose[item] = (random.random()*2)-1
        
        # Change the pose
        img = change_pose(pose, torch_input_image, numpy_bg)
        im.set_array(img)

        # Wait a little to blink again
        if i%len(eye_percent) == len(eye_percent)-1:
            time.sleep(5)

    ani = animation.FuncAnimation(fig, update_image, interval=0)
    plt.show()





# Posing vector info
# It looks like /pose keeps the pose of the image in a 42-dim vector
# in the following way:
#   eyebrow:
#      0: troubled left %
#      1: troubled right %
#      2: angry left %
#      3: angry right %
#      4: lowered left %
#      5: lowered right %
#      6: raised left %
#      7: riased right %
#      8: happy left %
#      9: happy right %
#     10: serious left %
#     11: serious right %
#   eye:
#     12: wink right %
#     13: wink left %
#     14: happy wink right %
#     15: happy wink left %
#     16: surprised left %
#     17: surprised right %
#     18: relaxed left %
#     19: relaxed right %
#     20: unimpressed left %
#     21: unimpressed right %
#     22: raised_lower_eyelid left %
#     23: raised_lower_eyelid right %
#   iris shrinkage:
#     24: left %
#     25: right %
#   mouth:
#     26: aaa %
#     27: iii %
#     28: uuu %
#     29: eee %
#     30: ooo %
#     31: delta %
#     32: lowered_corner left %
#     33: lowered_corner right %
#     34: raised_corner left %
#     35: raised_corner right %
#     36: smirk %
#   iris rotation:
#     37: left % (between -1 and 1)
#     38: right % (between -1 and 1)
#   head rotation:
#     39: x-axis % (between -1 and 1)
#     40: y-axis % (between -1 and 1)
#     41: z-axis % (between -1 and 1)






if __name__ == "__main__":
    main()