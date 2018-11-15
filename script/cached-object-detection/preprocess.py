#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import os
import re
import json
import shutil
import numpy as np
import scipy.io
from scipy.ndimage import zoom


# Load list of images to be processed
def load_image_list(images_dir, images_count, skip_images):
  assert os.path.isdir(images_dir), 'Input dir does not exit'
  files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
  files = [f for f in files if re.search(r'\.jpg$', f, re.IGNORECASE)
                            or re.search(r'\.jpeg$', f, re.IGNORECASE)]
  assert len(files) > 0, 'Input dir does not contain image files'
  files = sorted(files)[skip_images:]
  assert len(files) > 0, 'Input dir does not contain more files'
  images = files[:images_count]
  # Repeat last image to make full last batch
  if len(images) < images_count:
    for _ in range(images_count-len(images)):
      images.append(images[-1])
  return images


# Zoom to target size
def resize_img(img, target_size):
  zoom_w = float(target_size)/float(img.shape[0])
  zoom_h = float(target_size)/float(img.shape[1])
  return zoom(img, [zoom_w, zoom_h, 1])


# Load and preprocess image
def load_image(image_path,            # Full path to processing image
               target_size            # Desired size of resulting image
              ):
  img = scipy.misc.imread(image_path)

  # check if grayscale and convert to RGB
  if len(img.shape) == 2:
      img = np.dstack((img,img,img))

  # drop alpha-channel if present
  if img.shape[2] > 3:
      img = img[:,:,:3]

  # scale to targes size
  img = resize_img(img, target_size)

  return img


def ck_preprocess(i):
  print('\n--------------------------------')
  def my_env(var): return i['env'].get(var)
  def dep_env(dep, var): return i['deps'][dep]['dict']['env'].get(var)
  def has_dep_env(dep, var): return var in i['deps'][dep]['dict']['env']

  # Init variables from environment

  IMAGE_COUNT = int(my_env('CK_BATCH_COUNT')) * int(my_env('CK_BATCH_SIZE'))
  SKIP_IMAGES = int(my_env('CK_SKIP_IMAGES'))
  IMAGE_DIR = dep_env('dataset', 'CK_ENV_DATASET_IMAGE_DIR')
  IMAGE_FILE = my_env('CK_IMAGE_FILE')
  DATASET_TYPE = dep_env('dataset', 'CK_ENV_DATASET_TYPE')
  RESULTS_DIR = 'detections'
  IMAGE_LIST_FILE = 'image_list.txt'
  IMAGE_SIZE = 300

  # Full path of dir for caching prepared images.
  # Store preprocessed images in sources directory, not in `tmp`, as
  # `tmp` directory can de cleaned between runs and caches will be lost.
  CACHE_DIR_ROOT = my_env('CK_IMG_CACHE_DIR')
  if not CACHE_DIR_ROOT:
    CACHE_DIR_ROOT = os.path.join('..', 'preprocessed')

  # Single file mode
  if IMAGE_FILE:
    image_dir, IMAGE_FILE = os.path.split(IMAGE_FILE)
    # If only filename is set, assume that file is in images package
    if not image_dir:
      image_dir = IMAGE_DIR
    else:
      IMAGE_DIR = image_dir
    assert os.path.isfile(os.path.join(IMAGE_DIR, IMAGE_FILE)), "Input file does not exist"
    IMAGES_COUNT = 1
    SKIP_IMAGES = 1
    RECREATE_CACHE = True
    CACHE_DIR = os.path.join(CACHE_DIR_ROOT, 'single-image')
    print('Single file mode')
    print('Input image file: {}'.format(IMAGE_FILE))
  else:
    RECREATE_CACHE = my_env("CK_RECREATE_CACHE") == "YES"
    CACHE_DIR = os.path.join(CACHE_DIR_ROOT, '{}'.format(DATASET_TYPE))

  print('Input images dir: {}'.format(IMAGE_DIR))
  print('Preprocessed images dir: {}'.format(CACHE_DIR))
  print('Results dir: {}'.format(RESULTS_DIR))
  print('Image count: {}'.format(IMAGE_COUNT))
  print('Skip images: {}'.format(SKIP_IMAGES))

  # Prepare cache dir
  if not os.path.isdir(CACHE_DIR_ROOT):
    os.mkdir(CACHE_DIR_ROOT)
  if RECREATE_CACHE:
    if os.path.isdir(CACHE_DIR):
      shutil.rmtree(CACHE_DIR) 
  if not os.path.isdir(CACHE_DIR):
    os.mkdir(CACHE_DIR)

  # Prepare results directory
  if os.path.isdir(RESULTS_DIR):
    shutil.rmtree(RESULTS_DIR)
  os.mkdir(RESULTS_DIR)

  # Load processing images filenames
  if IMAGE_FILE:
    image_list = [IMAGE_FILE]
  else:
    image_list = load_image_list(IMAGE_DIR, IMAGE_COUNT, SKIP_IMAGES)

  # Preprocess images which are not cached yet
  print('Preprocess images...')
  preprocessed_count = 0
  for image_file in image_list:
    cached_path = os.path.join(CACHE_DIR, image_file)
    if not os.path.isfile(cached_path):
      original_path = os.path.join(IMAGE_DIR, image_file)
      image_data = load_image(original_path, IMAGE_SIZE)
      image_data.tofile(cached_path)
      preprocessed_count += 1
      if preprocessed_count % 10 == 0:
        print('  Done {} of {}'.format(preprocessed_count, len(image_list)))
  print('  Done {} of {}'.format(len(image_list), len(image_list)))

  # Save list of images to be classified
  with open(IMAGE_LIST_FILE, 'w') as f:
    for image_file in image_list:
      f.write(image_file + '\n')

  # Setup parameters for program
  def to_flag(val):
    return 1 if val and (str(val).upper() == "YES" or int(val) == 1) else 0

  new_env = {
    'RUN_OPT_IMAGE_SIZE': IMAGE_SIZE,
    'RUN_OPT_IMAGE_LIST': IMAGE_LIST_FILE,
    'RUN_OPT_RESULT_DIR': RESULTS_DIR,
    'RUN_OPT_IMAGE_DIR': CACHE_DIR,
    'RUN_OPT_IMAGE_LIST_PATH': os.path.join(os.getcwd(), IMAGE_LIST_FILE),
    'RUN_OPT_BATCH_COUNT': my_env('CK_BATCH_COUNT'),
    'RUN_OPT_BATCH_SIZE': my_env('CK_BATCH_SIZE'),
    'RUN_OPT_SILENT_MODE': to_flag(my_env('CK_SILENT_MODE')),
    'RUN_OPT_NORMALIZE_DATA': 1,
    'RUN_OPT_SUBTRACT_MEAN': 0
  }
  tflite_file = dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_TFLITE_FILE')
  if tflite_file:
    new_env['RUN_OPT_GRAPH_FILE'] = tflite_file

  files_to_push = []
  files_to_pull = []

  # Some special preparation to run program on Android device
  if i.get('target_os_dict', {}).get('ck_name2', '') == 'android':
    # When files will being pushed to Android, current path will be sources path,
    # not `tmp` as during preprocessing. So we have to set `files_to_push` accordingly,
    if CACHE_DIR.startswith('..'):
      CACHE_DIR = CACHE_DIR[3:]

    for image_file in image_list:
      files_to_push.append(os.path.join(CACHE_DIR, image_file))
      files_to_pull.append(os.path.join(RESULTS_DIR, image_file) + '.txt')

    # Set list of additional files to be copied to Android device.
    # When we set these files via env variables with full paths
    # they will be copied into remote program dir without sub-paths.
    files_to_push.append('$<<RUN_OPT_IMAGE_LIST_PATH>>$')
    if tflite_file:
      files_to_push.append('$<<RUN_OPT_GRAPH_FILE>>$')

  print('Prepared env:')
  print(json.dumps(new_env, indent=2, sort_keys=True))

  print('--------------------------------\n')
  return {
    'return': 0,
    'new_env': new_env,
    'run_input_files': files_to_push,
    'run_output_files': files_to_pull,
  }

