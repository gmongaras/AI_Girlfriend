import numpy as np
from PIL import Image
from rembg import remove
import PIL.Image
import numpy
from .tha2.util import extract_pytorch_image_from_PIL_image, convert_output_image_from_torch_to_numpy
from .tha2.poser.modes import mode_20
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
import random
import math



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




class Talking_Head():
    # device - Device to put the model on
    # blink_time - (optional) Time for each blink cycle
    # img_path - (Optional) Path to the image to load
    def __init__(self, device, blink_time=0.5, img_path=None):
        self.device = device

        # Load in the model
        self.poser = mode_20.create_poser(device)
        self.pose_parameters = mode_20.get_pose_parameters()
        self.pose_size = self.poser.get_num_parameters()

        # Load the new image in if any
        if img_path is not None:
            self.load_new_image(img_path)

        # Current position vector itinialized as zeros
        self.pose = torch.zeros((42)).to(self.device)

        # The configuration cycle for blinking
        # self.eye_percent = [0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0, 0]
        # self.dilate_percent = [0, 0.2, 0.4, 0.8, 0.4, 0.2, 0, 0, 0, 0]
        self.eye_cycle_num = 0               # Current eye cycle index
        self.cycle_end = False               # Has the blink cycle ended?
        self.num_frames = 0                  # Initial frame count to blink
        self.midpoint = 0
        self.total_blink_time_i = blink_time # Total time for a single blink cycle

        # Increase the total blink cycle time by 2 so that it has a chance to
        # actually work. If this isn't set to a high value at first, it
        # will not end up working for some reason. I think the GPU just
        # needs time to warm up
        self.total_blink_time = 2

        # configuration cycle for talking
        self.talking_percent = [0, 0.2, 0.4, 0.8, 0.4, 0.2, 0, 0, 0, 0]

        # EMA giving the expected value of the new image generation.
        # This statistic will be used as a correcting factor
        self.EMA_rate = 0.1
        self.EMA = 0


    # Used to update the EMA
    def update_EMA(self, val):
        self.EMA = (1-self.EMA_rate)*self.EMA + self.EMA_rate*val




    # Load a new image for moving around
    def load_new_image(self, img_path):
        # Load in the image and save the resulting tensor
        # and numpy array
        self.torch_input_image, self.numpy_bg = self.load_img(img_path)



    # Load in a new image given the path to that image
    def load_img(self, path):
        # Load in the image
        img = Image.open(path).convert("RGB")
        
        # Image must be 256x256
        img = img.resize((256, 256))
            
        # Remove the background from the image and save the background
        # for later use
        img, bg = self.remove_bg(img)

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
        torch_input_image = extract_pytorch_image_from_PIL_image(img).to(self.device)
        
        return torch_input_image, numpy_bg



    
    # Converts an image with a plain background to one
    # with a clear background.
    # Input:
    #   PIL image
    # Ouput:
    #   PIL image of the image foreground
    #   Numpy array of the image background
    def remove_bg(self, img):
        global img2
        img2 = img
        fg = remove(img)
        bg = np.array(img.convert("RGBA"))-np.array(fg)
        return fg, bg



    # Given a pytorch image and a background, return the
    # image to display
    # pytorch_image - Image as a pytorch tensor
    # numpy_bg - (Optional) numpy array with the original background
    #                       of the original image
    def get_pytorch_image(self, pytorch_image, numpy_bg=None):
        output_image = pytorch_image.detach().cpu()
        numpy_image = numpy.uint8(numpy.rint(convert_output_image_from_torch_to_numpy(output_image) * 255.0))
        
        # If the background exists, reapply it so the image isn't so plain
        if numpy_bg is not None:
            numpy_image += numpy_bg
            
        pil_image = PIL.Image.fromarray(numpy_image, mode='RGBA')
        return pil_image
    

    # Change the pose of the stored image using the
    # stored vector state
    def change_pose(self):
        # Start timing this function for the EMA
        start = time.time()

        # Pose the image
        output_image = self.poser.pose(self.torch_input_image, self.pose)[0]
        
        # Get the image
        img = self.get_pytorch_image(output_image, numpy_bg=self.numpy_bg)

        # End the timer
        val = time.time()-start

        # Update the EMA
        self.update_EMA(val)

        return img



    # Given a position value, return the updated vector
    # for the new face with the eyes moved to the next position
    def Move_eyes(self):
        blink_time = self.total_blink_time

        # Get the eye cycle. The cycle has `midpoint` number
        # of values to move the eye down and back up
        eye_cycle = [i/max(1, self.midpoint-1) for i in range(0, self.midpoint)]
        eye_cycle += eye_cycle[::-1][1:]

        # The dilation cycle is the same as the eye cycle, but
        # the cycle start later and ends earlier. This cycle should
        # be 0 during the beginning and last frames
        dilate_cycle = [0] + [i/(self.midpoint+1) for i in range(0, self.midpoint-1)]
        dilate_cycle += dilate_cycle[::-1][1:]

        # Get the value in the cycle. The value in the
        # cycle is the current cycle number. If the EMA
        # was too large, this may overshoot, so default
        # to a value of 0
        try:
            eye_per = eye_cycle[self.eye_cycle_num]
            dilate_per = dilate_cycle[self.eye_cycle_num]
        except IndexError:
            eye_per = 0
            dilate_per = 0

        # Has the last cycle been reached? If so,
        # signify the end and reset the cycle number
        if self.eye_cycle_num >= len(eye_cycle)-1:
            # The cycle has ended, set the flag and
            # set the cycle index to 0
            self.cycle_end = True
            self.eye_cycle_num = 0

            # Decrease the initial blink rate
            self.total_blink_time = max(self.total_blink_time-0.25, self.total_blink_time_i)

            # Calculate how many frames we want to blink for. Assuming
            # the EMA is correct, this will be the time to
            # blink divided by the expected value of generation
            self.num_frames = (blink_time//self.EMA)+1

            # Get the number of frams to reach the midpoint
            self.midpoint = max(1, round(math.ceil(self.num_frames/2)))

        # If not, update the eye cycle num
        else:
            self.eye_cycle_num += 1

        # Update the pose
        self.pose[12] = eye_per
        self.pose[13] = eye_per
        self.pose[24] = dilate_per
        self.pose[25] = dilate_per

        # for item in range(0, len(pose)):
        #     pose[item] = random.random()
        # for item in range(37, 42):
        #     pose[item] = (random.random()*2)-1

        # Return the new vector
        return self.pose
    

    # Given a position value, return the updated vector
    # for the new face with the mouth moved to the next position
    # pos - Current movement position. Cen be cumulative
    #       or in the range of possible values
    def Move_mouth(self, pos):
        # Get the new position
        talking_per = self.talking_percent[pos%len(self.talking_percent)]

        # Update the vector to make the image mover its mouth
        self.pose[26] = talking_per

        # Return the new vector
        return self.pose



def main():
    # Create the object
    obj = Talking_Head("cuda:0", 0.60, "Talking_Head/data/illust/img.png")

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    img = obj.change_pose()
    im = ax.imshow(img, animated=True)

    def update_image(i):
        # Update the vector
        obj.Move_eyes()
        
        # Change the pose
        img = obj.change_pose()
        im.set_array(img)

        # Wait a little to blink again
        if obj.cycle_end:
            plt.pause(np.clip(np.random.normal(5, 1, size=1)[0], 2, 7))
            obj.cycle_end = False

    ani = animation.FuncAnimation(fig, update_image, interval=0)
    plt.show()




if __name__=="__main__":
    # Move_eyes()
    main()