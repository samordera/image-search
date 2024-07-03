import os
import time
import torch
import torch.backends.cudnn as cudnn
import cv2
import numpy as np
import pytesseract
from utils import craft_utils, file_utils, imgproc
from nets.nn import CRAFT, RefineNet
import yaml
from collections import OrderedDict

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

with open(os.path.join('utils', 'config.yaml')) as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args['test_folder'])

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def read_coordinates(filename, x_padding=5, y_padding=2):
    with open(filename, 'r') as file:
        lines = file.readlines()
        coords = []
        for line in lines:
            line = line.strip()
            if line:  # Ensure the line is not empty
                # Convert the line into a list of floats
                coord_list = list(map(float, line.split(',')))
                if len(coord_list) % 2 == 0:
                    # Reshape the coordinates to add padding
                    x_coords = coord_list[0::2]
                    y_coords = coord_list[1::2]
                    min_x, max_x = min(x_coords) - x_padding, max(x_coords) + x_padding
                    min_y, max_y = min(y_coords) - y_padding, max(y_coords) + y_padding
                    padded_coords = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
                    coords.append(padded_coords)
    return coords

def overlap(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1[0], box1[1], box1[4], box1[5]
    x2_min, y2_min, x2_max, y2_max = box2[0], box2[1], box2[4], box2[5]
    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

def merge_boxes(box1, box2):
    x_min = min(box1[0], box2[0])
    y_min = min(box1[1], box2[1])
    x_max = max(box1[4], box2[4])
    y_max = max(box1[5], box2[5])
    return [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]

def merge_overlapping_boxes(coords):
    merged_coords = []
    while coords:
        box1 = coords.pop(0)
        temp_coords = []
        for box2 in coords:
            if overlap(box1, box2):
                box1 = merge_boxes(box1, box2)
            else:
                temp_coords.append(box2)
        merged_coords.append(box1)
        coords = temp_coords
    return merged_coords

def extract_text_from_image(image, coords):
    extracted_texts = []
    for box in coords:
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        cropped = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(cropped)
        extracted_texts.append(text)
    return extracted_texts

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args['canvas_size'],
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=args['mag_ratio'])
    ratio_h = ratio_w = 1 / target_ratio

    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = x.unsqueeze(0)  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    with torch.no_grad():
        y, feature = net(x)

    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args['show_time']: print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

def draw_bounding_boxes(image, coords):
    for box in coords:
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        x, y, w, h = cv2.boundingRect(pts)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

if __name__ == '__main__':
    net = CRAFT()

    print('Loading weights from checkpoint (' + args['trained_model'] + ')')
    if torch.cuda.is_available() and args['cuda']:
        net.load_state_dict(copyStateDict(torch.load(args['trained_model'])))
    else:
        net.load_state_dict(copyStateDict(torch.load(args['trained_model'], map_location=torch.device('cpu'))))

    if torch.cuda.is_available() and args['cuda']:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    refine_net = None
    if args['refine']:
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args['refiner_model'] + ')')
        if torch.cuda.is_available() and args['cuda']:
            refine_net.load_state_dict(copyStateDict(torch.load(args['refiner_model'])))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args['refiner_model'], map_location=torch.device('cpu'))))

        refine_net.eval()
        args['poly'] = True

    t = time.time()

    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, args['text_threshold'], args['link_threshold'],
                                             args['low_text'],
                                             torch.cuda.is_available() and args['cuda'], args['poly'], refine_net)

        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        coords_file = result_folder + "/res_" + filename + '.txt'

        with open(coords_file, 'w') as f:
            for box in polys:
                f.write(','.join(map(str, box.flatten())) + '\n')

        if os.path.exists(coords_file):
            coords = read_coordinates(coords_file)
        else:
            print(f"Coordinates file {coords_file} not found.")
            continue

        merged_coords = merge_overlapping_boxes(coords)

        padded_img = draw_bounding_boxes(image, merged_coords)
        result_img_file = result_folder + "/res_" + filename + '.jpg'
        cv2.imwrite(result_img_file, padded_img)

        # Use the original image for text extraction
        texts = extract_text_from_image(image, merged_coords)

        # Save the result image with bounding boxes
        file_utils.saveResult(image_path, padded_img[:, :, ::-1], merged_coords, dirname=result_folder)

        print(f'Detected texts in {image_path}:')
        print(' '.join(f'{text}'.replace('\n', ' ') for i, text in enumerate(texts)))

    print("elapsed time : {}s".format(time.time() - t))
