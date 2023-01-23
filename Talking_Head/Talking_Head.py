import numpy as np
from PIL import Image
from rembg import remove
import PIL.Image
import numpy
from tha2.util import extract_pytorch_image_from_PIL_image, convert_output_image_from_torch_to_numpy
import tha2.poser.modes.mode_20
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
import random





class Talking_Head():
    # device - Device to put the model on
    # img_path - (Optional) Path to the image to load
    def __init__(self, device, img_path=None):
        self.device = device

        # Load in the model
        self.poser = tha2.poser.modes.mode_20.create_poser(device)
        self.pose_parameters = tha2.poser.modes.mode_20.get_pose_parameters()
        self.pose_size = self.poser.get_num_parameters()

        # Load the new image in if any
        if img_path is not None:
            self.load_new_image(img_path)




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
    

    # Change the pose of the stored image given the
    # pose to change to
    def change_pose(self, pose):
        # Now let's try to get the output image with no posing
        output_image = self.poser.pose(self.torch_input_image, pose)[0]
        
        # Get the image
        return self.get_pytorch_image(output_image, numpy_bg=self.numpy_bg)





def main():
    # Create the object
    obj = Talking_Head("cuda:0", "data/illust/img2.png")

    # Posing vector
    pose = torch.zeros((42)).to(obj.device)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    img = obj.change_pose(pose)
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
        img = obj.change_pose(pose)
        im.set_array(img)

        # Wait a little to blink again
        if i%len(eye_percent) == len(eye_percent)-1:
            time.sleep(5)

    ani = animation.FuncAnimation(fig, update_image, interval=0)
    plt.show()


if __name__=="__main__":
    main()