#!/usr/bin/env python3
"""
* Copyright (c) 2018 Intel Corporation.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
* LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
* OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
* WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import sys
import os
import glob
import numpy as np
import logging as log
from time import time
from argparse import ArgumentParser
import warnings
from inference import Network
import json
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

try:
    from PIL import ImageEnhance
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
    ImageEnhance = None


# CONSTANTS
CPU_EXTENSION = ""
CONFIG_FILE = '../resources/config.json'


def read_image(path):
    image_original = load_img(path, color_mode="rgb")
    img = resize_image(image_original, target_size=(224, 224))
    x = img_to_array(img, data_format='channels_first')
    return [x, image_original]


def build_argparser():

    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-d", "--device", help="Specify the target device to infer on; CPU, GPU "
                        "FPGA, HDDL or MYRIAD is acceptable. To run with multiple devices use "
                        "MULTI:<device1>,<device2>,etc. Application will look for a suitable plugin "
                        "for device specified (CPU by default)", default="CPU", type=str)
    parser.add_argument("-ni", "--number_iter", help="Number of inference iterations", default=20, type=int)
    parser.add_argument("-pc", "--perf_counts", help="Report performance counters", default=False, action="store_true")
    parser.add_argument("-o", "--output_dir", help="If set, it will write a video here instead of displaying it",
                        default="../output", type=str)

    return parser


if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS


def save_img(path,
             x,
             data_format='channels_last',
             file_format=None,
             scale=True,
             **kwargs):
    """Saves an image stored as a Numpy array to a path or file object.

    # Arguments
        path: Path or file object.
        x: Numpy array.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        file_format: Optional file format override. If omitted, the
            format to use is determined from the filename extension.
            If a file object was used instead of a filename, this
            parameter should always be used.
        scale: Whether to rescale image values to be within `[0, 255]`.
        **kwargs: Additional keyword arguments passed to `PIL.Image.save()`.
    """
    img = array_to_img(x, data_format=data_format, scale=scale)
    if img.mode == 'RGBA' and (file_format == 'jpg' or file_format == 'jpeg'):
        warnings.warn('The JPG format does not support '
                      'RGBA images, converting to RGB.')
        img = img.convert('RGB')
    img.save(path, format=file_format, **kwargs)


def load_img(path, grayscale=False, color_mode='rgb', target_size=None,
             interpolation='nearest'):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file.
        grayscale: DEPRECATED use `color_mode="grayscale"`.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if grayscale is True:
        warnings.warn('grayscale is deprecated. Please use '
                      'color_mode = "grayscale"')
        color_mode = 'grayscale'
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `load_img` requires PIL.')
    img = pil_image.open(path)
    if color_mode == 'grayscale':
        if img.mode != 'L':
            img = img.convert('L')
    elif color_mode == 'rgba':
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    elif color_mode == 'rgb':
        if img.mode != 'RGB':
            img = img.convert('RGB')
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img


def resize_image(img, target_size, interpolation='bilinear'):

    width_height_tuple = (target_size[1], target_size[0])
    if img.size != width_height_tuple:
        if interpolation not in _PIL_INTERPOLATION_METHODS:
            raise ValueError(
                'Invalid interpolation method {} specified. Supported '
                'methods are {}'.format(
                    interpolation,
                    ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
        resample = _PIL_INTERPOLATION_METHODS[interpolation]
        img = img.resize(width_height_tuple, resample)
    return img


def array_to_img(x, data_format='channels_last', scale=True, dtype='float32'):
    """Converts a 3D Numpy array to a PIL Image instance.

    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
            either "channels_first" or "channels_last".
        scale: Whether to rescale image values
            to be within `[0, 255]`.
        dtype: Dtype to use.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=dtype)
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape: %s' % (x.shape,))

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format: %s' % data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 4:
        # RGBA
        return pil_image.fromarray(x.astype('uint8'), 'RGBA')
    elif x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: %s' % (x.shape[2],))


def img_to_array(img, data_format='channels_last', dtype='float32'):
    """Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        dtype: Dtype to use for the returned array.

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: %s' % data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: %s' % (x.shape,))
    return x


def bname():
    global model_xml
    bs = BeautifulSoup(open(model_xml), 'xml')
    bnTag = bs.findAll(attrs={"id": "365"})
    bn = bnTag[0]['name']
    return bn


def main():
    global CONFIG_FILE, model_xml
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    device = args.device

    assert os.path.isfile(CONFIG_FILE), "{} file doesn't exist".format(CONFIG_FILE)
    config = json.loads(open(CONFIG_FILE).read())

    infer_network = Network()
    n, c, h, w = infer_network.load_model(model_xml, device, 1, 1, 0, CPU_EXTENSION)[1]
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    f = open(os.path.join(args.output_dir, 'result' + '.txt'), 'w')
    f1 = open(os.path.join(args.output_dir, 'stats' + '.txt'), 'w')
    time_images = []
    colormap = 'viridis'
    for item in config['inputs']:
        files = glob.glob(os.getcwd() + '/' + item['image'])
        for file in files:
            [image1, image] = read_image(file)
            t0 = time()
            for i in range(args.number_iter):
                infer_network.exec_net(image1)
            infer_time = (time() - t0) * 1000
            # log.info("Average running time of one iteration: {} ms".format(np.average(np.asarray(infer_time))))
            if args.perf_counts:
                perf_counts = infer_network.performance_counter(0)
                log.info("Performance counters:")
                print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type', 'exet_type', 'status',
                                                                  'real_time, us'))
                for layer, stats in perf_counts.items():
                    print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer, stats['layer_type'], stats['exec_type'],
                                                                      stats['status'], stats['real_time']))
            res = infer_network.get_output(0)
            probs = res[0][0]
            avg_time = round((infer_time / args.number_iter), 1)

            f.write("Pneumonia probability of " + str(file.split('/')[-1]) + ' : '
                    + str(probs) + "\n Inference performed in " + str(avg_time) + "ms \n")
            time_images.append(avg_time)

        if 'PNEUMONIA' in item['image']:
            bn = bname()
            infer_network.load_model_for_activation_map(bn, 0, device)
            fc = "predictions_1/MatMul"
            # iterate over the pneumonia cases
            for file in files:
                # read the image
                [image1, image] = read_image(file)
                # Start inference
                res = infer_network.exec_net(image1)

                # Class Activation Map
                cam = infer_network.visualize_class_activation_map_openvino(res, bn, fc)
                fig = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
                # Visualize the CAM heatmap
                cam /= np.max(cam)
                fig.add_subplot(1, 2, 1)
                # plt.imshow(cam, cmap=colormap)
                # plt.colorbar(fraction=0.046, pad=0.04)

                # Visualize the CAM overlaid over the X-ray image
                colormap_val = cm.get_cmap(colormap)
                imss = np.uint8(colormap_val(cam) * 255)
                im = Image.fromarray(imss)
                width, height = image.size
                cam1 = resize_image(im, (height, width))
                heatmap = np.asarray(cam1)
                img1 = heatmap[:, :, :3] * 0.3 + image
                fig.add_subplot(1, 2, 2)
                file_name = file.split('/')[-1]
                output_file = "{}/{}".format(args.output_dir, file_name)
                save_img(output_file, img1, file_format='jpeg')

    log.info("Success")
    f1.write("Total average Inference time : " + str(np.average(np.asarray(time_images))) + "ms \n")
    log.info("Total average Inference time : {} ms".format(np.average(np.asarray(time_images))))
    print("The Output X-ray images and results.txt file are stored in the {} directory".format(args.output_dir))


if __name__ == '__main__':
    sys.exit(main() or 0)
