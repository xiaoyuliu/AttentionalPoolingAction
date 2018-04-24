from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import random
import sys
import scipy.io
import json
import operator
import numpy as np

import tensorflow as tf

# Set the following paths
_MPII_MAT_FILE = '../../datasets/mpii_human_pose_v1_u12_1.mat'
_IMG_DIR = '/home/xiaoyu/Documents/action/datasets/hmdb51/fall_floor-frame'
_LABEL_DIR = '/home/xiaoyu/Documents/action/openpose/output/fall_floor'

dataset_dir = '../../src/data/hmdb/hmdb_tfrecords/'

# Seed for repeatability.
_RANDOM_SEED = 42

# The number of shards per dataset split.
_NUM_SHARDS = 1

_NUM_JOINTS = 16  # for pose

OPENPOSE2APAR = [None, 8, 12, None, 10,
                 13, 15, 2, 1, 0, 3, 4, 5,
                 None, None, None, None,
                 None]


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def int64_feature(values):
    """Returns a TF-Feature of int64s.

    Args:
      values: A scalar or list of values.

    Returns:
      a TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.

    Args:
      values: A string.

    Returns:
      a TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, image_format, height, width,
                       pose,  # [x,y,is_vis,...]
                       action_label):
    assert (len(pose) % (_NUM_JOINTS * 3) == 0)
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/pose': int64_feature([int(el) for el in pose]),
        'image/class/action_label': int64_feature(action_label),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
    }))


def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'hmdb_%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, image_obj, dataset_dir):
    num_per_shard = int(math.ceil(len(image_obj) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(image_obj))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i + 1, len(image_obj), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        fname = os.path.join(image_obj[i][0])
                        poses = image_obj[i][1]
                        if poses[0][0] == -1:
                            action_label = -1
                        else:
                            action_label = 1
                        all_joints = []

                        for pose in poses:
                            final_pose = [0] * 16
                            final_pose[pose[0] - 1] = pose[1:]
                            for i in range(_NUM_JOINTS):
                                if final_pose[i] == 0:
                                    final_pose[i] = [-1, -1, 0]

                            final_pose = [item for sublist in final_pose for item in sublist]
                            all_joints += final_pose
                        assert (len(all_joints) % 16 == 0)
                        image_data = tf.gfile.FastGFile(fname, 'r').read()
                        height, width = image_reader.read_image_dims(sess, image_data)
                        example = image_to_tfexample(
                            image_data, 'jpg', height, width, all_joints, action_label)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def _get_action_class(cname, D, act_id):
    try:
        if cname not in D:
            D[cname] = (len(D.keys()), set([act_id]))  # act_id is the actual MPII action id
        else:
            D[cname][1].add(act_id)
            # It's pretty crazy that same action will have multiple action IDs
        return D[cname][0]
    except Exception, e:
        print('Invalid class name {}. setting -1. {}'.format(cname, e))
        return -1


def main():
    frame_folders = os.listdir(_IMG_DIR)
    all_imnames = []
    image_obj = []
    for frame_folder in frame_folders:
        image_dir = os.path.join(_IMG_DIR, frame_folder)
        label_dir = os.path.join(_LABEL_DIR, frame_folder)
        frames = os.listdir(image_dir)
        frame_num = len(frames)

        for fid in range(frame_num):
            imname = "%04d.jpg" % (fid + 1)
            all_imnames.append(os.path.join(image_dir, imname))
            points_fmted = []  # put all points one after the other
            label_file = os.path.join(label_dir, '%04d_keypoints.json' % (fid + 1))

            with open(label_file, 'r') as json_data:
                labels = json.load(json_data)

            if labels['people']:
                for people in labels['people']:
                    keypoints = people['pose_keypoints_2d']
                    for i in range(18):
                        joint_id = i + 1
                        apar_id = OPENPOSE2APAR[joint_id - 1]
                        if OPENPOSE2APAR[joint_id - 1]:
                            keypoint_x = (keypoints[3 * i])
                            keypoint_y = (keypoints[3 * i + 1])
                            points_fmted.append([apar_id, keypoint_x, keypoint_y, 1])
            else:
                points_fmted.append([-1, -1, -1, 0])

            # points_rect.append((point.id, point.x, point.y, is_visible))
            # points_fmted.append(points_rect)
            # [el.sort() for el in points_fmted]

            # the following assert is not true, so putting -1 when writing it out
            # assert(all([len(el) == 16 for el in points_fmted]))
            image_obj.append((os.path.join(image_dir, imname),
                         points_fmted))

    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    # Only randomize the train set
    random.seed(_RANDOM_SEED)
    # train_ids = range()
    # random.shuffle(train_ids)

    with open(os.path.join(dataset_dir, 'imnames.txt'), 'w') as fout:
        fout.write('\n'.join(all_imnames))

    spl_name = 'trainval'
    _convert_dataset(spl_name, image_obj,
                     dataset_dir)

    print('\nFinished converting the HMDB dataset!')


if __name__ == '__main__':
    main()
