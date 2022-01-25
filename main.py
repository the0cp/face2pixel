import cv2
import numpy as np
import os
import sys
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
import math
import pickle
from time import sleep
from glob import glob
sys.path.append(r'./faceDetect/')
import faceDetect

# number of colors per image
COLOR_DEPTH = 4
# tiles scales
RESIZING_SCALES = [0.5, 0.4, 0.3, 0.2, 0.1]
# number of pixels shifted to create each box (x,y)
PIXEL_SHIFT = (5, 5)
# multiprocessing pool size
POOL_SIZE = 8
# if tiles can overlap
OVERLAP_TILES = False

# reduces the number of colors in an image
def color_quantization(img, n_colors):
    return np.round(img / 255 * n_colors) / n_colors * 255


# returns an image given its path
def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img = color_quantization(img.astype('float'), 4)
    return img.astype('uint8')


# scales an image
def resize_image(img, ratio):
    img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))
    return img


# the most frequent color in an image and its relative frequency
def mode_color(img, ignore_alpha=False):
    counter = defaultdict(int)
    total = 0
    for y in img:
        for x in y:
            if len(x) < 4 or ignore_alpha or x[3] != 0:
                counter[tuple(x[:3])] += 1
            else:
                counter[(-1,-1,-1)] += 1
            total += 1

    if total > 0:
        mode_color = max(counter, key=counter.get)
        if mode_color == (-1,-1,-1):
            return None, None
        else:
            return mode_color, counter[mode_color] / total
    else:
        return None, None


# displays an image
def show_image(img, wait=True):
    cv2.imshow('img', img)
    if wait:
        cv2.waitKey(0)
    else:
        cv2.waitKey(1)


# load and process the tiles
def load_tiles(paths):
    print('Loading tiles')
    tiles = defaultdict(list)

    for path in paths:
        if os.path.isdir(path):
            for tile_name in tqdm(os.listdir(path)):
                tile = read_image(os.path.join(path, tile_name))
                mode, rel_freq = mode_color(tile, ignore_alpha=True)
                if mode is not None:
                    for scale in [0.5, 0.4, 0.3, 0.2, 0.1]:
                        t = resize_image(tile, scale)
                        res = tuple(t.shape[:2])
                        tiles[res].append({
                            'tile': t,
                            'mode': mode,
                            'rel_freq': rel_freq
                        })


            with open('tiles.pickle', 'wb') as f:
                pickle.dump(tiles, f)

        # load pickle with tiles (one file only)
        else:
            with open(path, 'rb') as f:
                tiles = pickle.load(f)

    return tiles


# returns the boxes (image and start pos) from an image, with 'res' resolution
def image_boxes(img, res):
    if not (5, 5):
        shift = np.flip(res)
    else:
        shift = (5, 5)

    boxes = []
    for y in range(0, img.shape[0], shift[1]):
        for x in range(0, img.shape[1], shift[0]):
            boxes.append({
                'img': img[y:y+res[0], x:x+res[1]],
                'pos': (x,y)
            })

    return boxes


# euclidean distance between two colors
def color_distance(c1, c2):
    c1_int = [int(x) for x in c1]
    c2_int = [int(x) for x in c2]
    return math.sqrt((c1_int[0] - c2_int[0])**2 + (c1_int[1] - c2_int[1])**2 + (c1_int[2] - c2_int[2])**2)


# returns the most similar tile to a box (in terms of color)
def most_similar_tile(box_mode_freq, tiles):
    if not box_mode_freq[0]:
        return (0, np.zeros(shape=tiles[0]['tile'].shape))
    else:
        min_distance = None
        min_tile_img = None
        for t in tiles:
            dist = (1 + color_distance(box_mode_freq[0], t['mode'])) / box_mode_freq[1]
            if min_distance is None or dist < min_distance:
                min_distance = dist
                min_tile_img = t['tile']
        return (min_distance, min_tile_img)


# builds the boxes and finds the best tile for each one
def get_processed_image_boxes(image_path, tiles):
    print('Getting and processing boxes')
    img = read_image(image_path)
    print('done!!!!!!!!!!!!!!!!!!!!!!!!!!')
    pool = Pool(8)
    all_boxes = []

    for res, ts in tqdm(sorted(tiles.items(), reverse=True)):
        boxes = image_boxes(img, res)
        modes = pool.map(mode_color, [x['img'] for x in boxes])
        most_similar_tiles = pool.starmap(most_similar_tile, zip(modes, [ts for x in range(len(modes))]))

        i = 0
        for min_dist, tile in most_similar_tiles:
            boxes[i]['min_dist'] = min_dist
            boxes[i]['tile'] = tile
            i += 1

        all_boxes += boxes

    return all_boxes, img.shape


# places a tile in the image
def place_tile(img, box):
    p1 = np.flip(box['pos'])
    p2 = p1 + box['img'].shape[:2]
    img_box = img[p1[0]:p2[0], p1[1]:p2[1]]
    mask = box['tile'][:, :, 3] != 0
    mask = mask[:img_box.shape[0], :img_box.shape[1]]
    if False or not np.any(img_box[mask]):
        img_box[mask] = box['tile'][:img_box.shape[0], :img_box.shape[1], :][mask]


# tiles the image
def create_tiled_image(boxes, res, render=False):
    print('Creating tiled image')
    img = np.zeros(shape=(res[0], res[1], 4), dtype=np.uint8)

    for box in tqdm(sorted(boxes, key=lambda x: x['min_dist'], reverse=False)):
        place_tile(img, box)
        if render:
            show_image(img, wait=False)
            sleep(0.025)

    return img

def main(faces_list):
    for facefilename in faces_list:
        tiles_path = ['./gen_pixel@50x50']
        tiles0 = load_tiles(tiles_path)
        boxes, original_res = get_processed_image_boxes(facefilename, tiles0)
        img = create_tiled_image(boxes, original_res, render=False)
        save_filename = '%s.png' % (os.path.basename(facefilename).split('.')[0])
        cv2.imwrite('output/' + save_filename, img)


if __name__ == "__main__":
    if os.path.exists('./faces') is False:
        os.makedirs('./faces')
        
    if not os.path.exists('faces/'):
        print('Image not found')
        exit(-1)
        
    if not os.path.exists('gen_pixel@50x50/'):
        print('Tiles folder not found')
        exit(-1)
        
    file_list = glob('./imgs/*.jpg')
    for filename in file_list:
        faceDetect.detect(filename)
        
    faces_list = glob('faces/*.png')
    
    main(faces_list)


