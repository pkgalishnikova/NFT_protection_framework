import os
import sys

import tensorflow as tf
from six.moves import urllib

_URL = 'http://rail.eecs.berkeley.edu/models/lpips'


def _download(url, output_dir):
    """Downloads the `url` file into `output_dir`.

    Modified from https://github.com/tensorflow/models/blob/master/research/slim/datasets/dataset_utils.py
    """
    filename = url.split('/')[-1]
    filepath = os.path.join(output_dir, filename)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')


def lpips(input0, input1, model='net-lin', net='alex', version=0.1):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) metric.
    TF 2.x + compat.v1 version.
    """
    # flatten the leading dimensions
    batch_shape = tf.shape(input0)[:-3]
    input0 = tf.reshape(input0, tf.concat([[-1], tf.shape(input0)[-3:]], axis=0))
    input1 = tf.reshape(input1, tf.concat([[-1], tf.shape(input1)[-3:]], axis=0))
    # NHWC to NCHW
    input0 = tf.transpose(input0, [0, 3, 1, 2])
    input1 = tf.transpose(input1, [0, 3, 1, 2])

    # normalize to [-1, 1]
    input0 = input0 * 2.0 - 1.0
    input1 = input1 * 2.0 - 1.0

    # FIX: Resize to match LPIPS expected input size
    input0 = tf.transpose(input0, [0, 2, 3, 1])  # NCHW -> NHWC
    input1 = tf.transpose(input1, [0, 2, 3, 1])
    input0 = tf.image.resize(input0, [64, 64])
    input1 = tf.image.resize(input1, [64, 64])
    input0 = tf.transpose(input0, [0, 3, 1, 2])  # NHWC -> NCHW
    input1 = tf.transpose(input1, [0, 3, 1, 2])

    input0_name, input1_name = '0:0', '1:0'

    # 🔧 FIXED: tf.get_default_graph → tf.compat.v1.get_default_graph
    default_graph = tf.compat.v1.get_default_graph()
    producer_version = default_graph.graph_def_versions.producer

    cache_dir = os.path.expanduser('~/.lpips')
    os.makedirs(cache_dir, exist_ok=True)
    pb_fnames = [
        '%s_%s_v%s_%d.pb' % (model, net, version, producer_version),
        '%s_%s_v%s.pb' % (model, net, version),
    ]
    for pb_fname in pb_fnames:
        pb_path = os.path.join(cache_dir, pb_fname)
        if not os.path.isfile(pb_path):
            try:
                _download(os.path.join(_URL, pb_fname), cache_dir)
            except urllib.error.HTTPError:
                pass
        if os.path.isfile(pb_path):
            break
    else:
        raise FileNotFoundError("Could not download or find any LPIPS model file.")

    # 🔧 FIXED: tf.GraphDef → tf.compat.v1.GraphDef
    with open(pb_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    tf.compat.v1.import_graph_def(
        graph_def,
        input_map={input0_name: input0, input1_name: input1},
        name=''  # avoid name collisions
    )

    distance = default_graph.get_tensor_by_name('Reshape:0')
    distance = tf.reshape(distance, batch_shape)
    return distance