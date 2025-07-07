import tqdm, random, pathlib, itertools, collections
import os
import cv2
import numpy as np
import remotezip as rz
import tensorflow as tf

# Some modules to displaty an animation using imageio.
import imageio
from IPython import display
from urllib import request
from tensorflow_docs.vis import embed

URL = 'https://storage.googleapis.com/thumos14_files/UCF101_videos.zip'

def list_files_from_zip_url(zip_url):
    """ List the files in each class of the dataset given a URL with the zip file.

    Args:
      zip_url: A URL from which the files can be extracted from.

    Returns:
      List of files in each of the classes.
    """
  
    files = []
    with rz.RemoteZip(zip_url) as zip:
         for zip_info in zip.infolist():
            files.append(zip_info.filename)
    return files

files = list_files_from_zip_url(URL)
files = [f for f in files if f.endswith('.avi')]
files[:10]  

def get_class(fname):
    """ Retrieve the name of the class given a filename.

    Args:
      fname: Name of the file in the UCF101 dataset.

    Returns:
      Class that the file belongs to.
  """
    return fname.split('_')[-3]

def get_files_per_class(files):
  """ Retrieve the files that belong to each class.

    Args:
      files: List of files in the dataset.

    Returns:
      Dictionary of class names (key) and files (values). 
  """
  files_for_class = collections.defaultdict(list)
  for fname in files:
      class_name = get_class(fname)
      files_for_class[class_name].append(fname)
  return files_for_class

NUM_CLASSES = 10
FILES_PER_CLASS = 50

files_for_class = get_files_per_class(files)
classes = list(files_for_class.keys())

print('Num classes:', len(classes))
print('Num videos for class[0]:', len(files_for_class[classes[0]]))

def select_subset_of_classes(files_for_class, classes, files_per_class):
  """ Create a dictionary with the class name and a subset of the files in that class.

    Args:
      files_for_class: Dictionary of class names (key) and files (values).
      classes: List of classes.
      files_per_class: Number of files per class of interest.

    Returns:
      Dictionary with class as key and list of specified number of video files in that class.
  """
  
  files_subset = dict()
  
  for class_name in classes:
      class_files = files_for_class[class_name]
      files_subset[class_name] = class_files[:files_per_class]
  
  return files_subset

files_subset = select_subset_of_classes(files_for_class, classes[:NUM_CLASSES], FILES_PER_CLASS)
list(files_subset.keys())

def download_from_zip(zip_url, to_dir, file_names):
  """ Download the contents of the zip file from the zip URL.

    Args:
      zip_url: A URL with a zip file containing data.
      to_dir: A directory to download data to.
      file_names: Names of files to download.
  """
  with rz.RemoteZip(zip_url) as zip:
      for fn in tqdm.tqdm(file_names):
          class_name = get_class(fn)
          zip.extract(fn, str(to_dir / class_name))
          unzipped_file = to_dir / class_name / fn
          
          fn = pathlib.Path(fn).parts[-1]
          output_file = to_dir / class_name / fn
          unzipped_file.rename(output_file)
          
def split_class_lists(files_for_class, count):
  """ Returns the list of files belonging to a subset of data as well as the remainder of
    files that need to be downloaded.

    Args:
      files_for_class: Files belonging to a particular class of data.
      count: Number of files to download.

    Returns:
      Files belonging to the subset of data and dictionary of the remainder of files that need to be downloaded.
  """
  split_files = []
  remainder = {}
  for cls in files_for_class:
      split_files.extend(files_for_class[cls][:count])
      remainder[cls] = files_for_class[cls][count:]
  return split_files, remainder

def download_ucf_101_subset(zip_url, num_classes, splits, download_dir):
  """ Download a subset of the UCF101 dataset and split them into various parts, such as
    training, validation, and test.

    Args:
      zip_url: A URL with a ZIP file with the data.
      num_classes: Number of labels.
      splits: Dictionary specifying the training, validation, test, etc. (key) division of data 
              (value is number of files per split).
      download_dir: Directory to download data to.

    Return:
      Mapping of the directories containing the subsections of data.
  """
  files = list_files_from_zip_url(zip_url)
  for f in files:
      path = os.path.normpath(f)
      tokens = path.split(os.sep)
      if len(tokens) <= 2:
          files.remove(f) # Remove that item from the list if it does not have a filename
          
  files_for_class = get_files_per_class(files)
  
  classes = list(files_for_class.keys())[:num_classes]
  
  for cls in classes:
      random.shuffle(files_for_class[cls])
      
  # Only use the number of classes you want in the dictionary
  files_for_class = {x: files_for_class[x] for x in classes}
  
  dirs = {}
  for split_name, split_count in splits.items():
      print(split_name, ":")
      split_dir = download_dir / split_name
      split_files, files_for_class = split_class_lists(files_for_class,split_count)
      download_from_zip(zip_url, split_dir, split_files)
      
  return dirs

download_dir = pathlib.Path('./UCF101_subset/')
subset_paths = download_ucf_101_subset(URL,
                                       num_classes = NUM_CLASSES,
                                       splits = {"train": 30, "val": 10, "test": 10},
                                       download_dir = download_dir)


video_count_train = len(list(download_dir.glob('train/*/*.avi')))
video_count_val = len(list(download_dir.glob('val/*/*.avi')))
video_count_test = len(list(download_dir.glob('test/*/*.avi')))
video_total = video_count_train + video_count_val + video_count_test
print(f"Total videos: {video_total}")

def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded. 
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame

