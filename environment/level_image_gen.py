import os
import numpy as np
from PIL import Image, ImageOps, ImageEnhance


class LevelImageGen:
    """ Generates PIL Image files from environment. ascii levels.
    Initialize once and then use LevelImageGen.render() to generate images. """
    def __init__(self, sprite_path):
        """ sprite_path: path to the folder of sprite files, e.g. 'environment/sprites/' """

        # Load Graphics (assumes sprite_path points to "img" folder of Environment-AI-Framework or provided sprites folder
        dronesheet = Image.open(os.path.join(sprite_path, 'drone.png'))
        prizesheet = Image.open(os.path.join(sprite_path, 'prize.png'))

        #itemsheet = Image.open(os.path.join(sprite_path, 'itemsheet.png'))
        mapsheet = Image.open(os.path.join(sprite_path, 'mapsheet.png'))

        # Cut out the actual sprites:
        sprite_dict = dict()

        # Drone Img
        sprite_dict['D'] = mapsheet.crop((4*16, 0, 5*16, 1*16)) #dronesheet#.crop((4*16, 0, 5*16, 16))

        # Prize Img
        sprite_dict['X'] = mapsheet.crop((7*16, 1*16, 8*16, 2*16)) #prizesheet 

        # Wall Img
        sprite_dict['O'] = mapsheet.crop((2*16, 0, 3*16, 1*16))

        # Obstacle Img
        sprite_dict['W'] = mapsheet.crop((1*16, 0, 2*16, 1*16))

        # Ground Img
        sprite_dict['-'] = mapsheet.crop((2*16, 5*16, 3*16, 6*16))

        self.sprite_dict = sprite_dict

    def prepare_sprite_and_box(self, ascii_level, sprite_key, curr_x, curr_y):
        """ Helper to make correct sprites and sprite sizes to draw into the image.
         Some sprites are bigger than one tile and the renderer needs to adjust for them."""

        # Init default size
        new_left = curr_x * 16
        new_top = curr_y * 16
        new_right = (curr_x + 1) * 16
        new_bottom = (curr_y + 1) * 16

        # Handle sprites depending on their type:
        actual_sprite = self.sprite_dict[sprite_key]

        return actual_sprite, (new_left, new_top, new_right, new_bottom)

    def render(self, ascii_level):
        """ Renders the ascii level as a PIL Image. Assumes the Background is sky """
        len_level = len(ascii_level[-1])
        height_level = len(ascii_level)

        # Fill base image with sky tiles
        dst = Image.new('RGB', (len_level*16, height_level*16))
        for y in range(height_level):
            for x in range(len_level):
                dst.paste(self.sprite_dict['-'], (x*16, y*16, (x+1)*16, (y+1)*16))

        # Fill with actual tiles
        for y in range(height_level):
            for x in range(len_level):
                curr_sprite = ascii_level[y][x]
                sprite, box = self.prepare_sprite_and_box(ascii_level, curr_sprite, x, y)
                dst.paste(sprite, box, mask=sprite)

        # Fill with actual tiles
        # pos = []
        # nagents = 4
        # while(len(pos)!= nagents):
        #     rand1 = np.random.randint(1, 4)
        #     rand2 = np.random.randint(1, 4)
        #     if [rand1, rand2] not in pos:
        #         pos.append([rand1, rand2])
        #         dst.paste(self.sprite_dict['D'], (rand1*16, rand2*16, (rand1+1)*16, (rand2+1)*16))
        #     else:
        #         pass

        return dst
