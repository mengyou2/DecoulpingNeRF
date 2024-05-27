# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core dataset methods and classes."""
import abc
import functools
import itertools
from typing import Any, Optional, Sequence, Union

from absl import logging
from flax import jax_utils
import immutabledict
import jax
import numpy as np
import tensorflow as tf

from hypernerf import camera as cam
from hypernerf import gpath
from hypernerf import image_utils
from hypernerf import tf_camera as tfcam
from hypernerf import utils
from tensorflow.python.data.util import nest


_TF_AUTOTUNE = tf.data.experimental.AUTOTUNE
_TF_CAMERA_PARAMS_SIGNATURE = {
    'orientation': tf.TensorSpec(shape=(3, 3), dtype=tf.float32),
    'position': tf.TensorSpec(shape=(3,), dtype=tf.float32),
    'focal_length': tf.TensorSpec(shape=(), dtype=tf.float32),
    'principal_point': tf.TensorSpec(shape=(2,), dtype=tf.float32),
    'skew': tf.TensorSpec(shape=(), dtype=tf.float32),
    'pixel_aspect_ratio': tf.TensorSpec(shape=(), dtype=tf.float32),
    'radial_distortion': tf.TensorSpec(shape=(3,), dtype=tf.float32),
    'tangential_distortion': tf.TensorSpec(shape=(2,), dtype=tf.float32),
    'image_size': tf.TensorSpec(shape=(2,), dtype=tf.float32),
}


def camera_to_rays(camera: cam.Camera):
  """Converts a vision sfm camera into rays.

  Args:
    camera: the camera to convert to rays.

  Returns:
    A dictionary of rays. Contains:
      `origins`: the origin of each ray.
      `directions`: unit vectors representing the direction of each ray.
      `pixels`: the pixel centers of each ray.
  """
  camera = camera.copy()

  image_shape = camera.image_shape

  img_rays_origin = np.tile(camera.position[None, None, :],
                            image_shape + (1,))
  img_rays_dir = camera.pixels_to_rays(camera.get_pixel_centers())
  img_rays_pixels = camera.get_pixel_centers()

  return {
      'origins': img_rays_origin.astype(np.float32),
      'directions': img_rays_dir.astype(np.float32),
      'pixels': img_rays_pixels.astype(np.float32),
  }


def load_camera(
    camera_path: gpath.GPath,
    scale_factor: float = 1.0,
    scene_center: Optional[np.ndarray] = None,
    scene_scale: Optional[Union[float, np.ndarray]] = None) -> cam.Camera:
  """Loads camera and rays defined by the center pixels of a camera.

  Args:
    camera_path: a path to an sfm_camera.Camera proto.
    scale_factor: a factor to scale the camera image by.
    scene_center: the center of the scene where the camera will be centered to.
    scene_scale: the scale of the scene by which the camera will also be scaled
      by.

  Returns:
    A Camera instance.
  """
  if camera_path.suffix == '.pb':
    camera = cam.Camera.from_proto(camera_path)
  elif camera_path.suffix == '.json':
    camera = cam.Camera.from_json(camera_path)
  else:
    raise ValueError('File must have extension .pb or .json.')

  if scale_factor != 1.0:
    camera = camera.scale(scale_factor)

  if scene_center is not None:
    camera.position = camera.position - scene_center
  if scene_scale is not None:
    camera.position = camera.position * scene_scale

  return camera


def prepare_data(xs):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # reshape (proc_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


def prepare_tf_data(xs, devices=None):
  """Convert a input batch from tf Tensors to numpy arrays."""
  if devices is None:
    devices = jax.local_devices()

  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    # reshape (proc_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    return x.reshape((len(devices), -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


def prepare_tf_data_unbatched(xs):
  """Prepare TF dataset into unbatched numpy arrays."""
  # Use _numpy() for zero-copy conversion between TF and NumPy.
  # pylint: disable=protected-access
  return jax.tree_map(lambda x: x._numpy(), xs)


def iterator_from_dataset(dataset: tf.data.Dataset,
                          batch_size: int,
                          repeat: bool = True,
                          prefetch_size: int = 0,
                          devices: Optional[Sequence[Any]] = None):
  """Create a data iterator that returns JAX arrays from a TF dataset.

  Args:
    dataset: the dataset to iterate over.
    batch_size: the batch sizes the iterator should return.
    repeat: whether the iterator should repeat the dataset.
    prefetch_size: the number of batches to prefetch to device.
    devices: the devices to prefetch to.

  Returns:
    An iterator that returns data batches.
  """
  if repeat:
    dataset = dataset.repeat()

  if batch_size > 0:
    dataset = dataset.batch(batch_size)
    it = map(functools.partial(prepare_tf_data, devices=devices), dataset)
  else:
    it = map(prepare_tf_data_unbatched, dataset)

  if prefetch_size > 0:
    it = jax_utils.prefetch_to_device(it, prefetch_size, devices)

  return it


def _camera_to_rays_fn(item, use_tf_camera=False):
  """Converts camera params to rays."""
  camera_params = item.pop('camera_params')
  if use_tf_camera:
    camera = tfcam.TFCamera(**camera_params)
    pixels = camera.get_pixel_centers()
    directions = camera.pixels_to_rays(pixels)
    origins = tf.broadcast_to(camera.position[None, None, :],
                              tf.shape(directions))
  else:
    camera = cam.Camera(**camera_params)
    pixels = camera.get_pixel_centers()
    directions = camera.pixels_to_rays(pixels)
    origins = np.broadcast_to(camera.position[None, None, :], directions.shape)
  item['origins'] = origins
  item['directions'] = directions
  item['pixels'] = pixels
  return item


def _tf_broadcast_metadata_fn(item):
  """Broadcasts metadata to the ray shape."""
  shape = tf.shape(item['rgb'])
  item['metadata'] = jax.tree_map(
      lambda x: tf.broadcast_to(x, (shape[0], shape[1], x.shape[-1])),
      item['metadata'])
  return item


class DataSource(abc.ABC):
  """An abstract class that defines a data source."""

  def __init__(self,
               train_ids,
               val_ids,
               use_appearance_id=False,
               use_camera_id=False,
               use_warp_id=False,
               use_depth=False,
               use_relative_depth=False,
               use_time=False,
               random_seed=0,
               train_stride=1,
               val_stride=1,
               preload=True,
               **_):
    self._train_ids = train_ids
    self._val_ids = val_ids
    self.train_stride = train_stride
    self.val_stride = val_stride
    self.use_appearance_id = use_appearance_id
    self.use_camera_id = use_camera_id
    self.use_warp_id = use_warp_id
    self.use_depth = use_depth
    self.use_time = use_time
    self.use_relative_depth = use_relative_depth
    self.rng = np.random.RandomState(random_seed)
    self.preload = preload
    logging.info(
        'Creating datasource of type %s with use_appearance_id=%s, '
        'use_camera_id=%s, use_warp_id=%s, use_depth=%s, use_time=%s, '
        'train_stride=%d, val_stride=%d', self.__class__.__name__,
        use_appearance_id, use_camera_id, use_warp_id, use_depth, use_time,
        train_stride, val_stride)
    self.metadata_dict = None

  @property
  def all_ids(self):
    return sorted(itertools.chain(self.train_ids, self.val_ids))

  @property
  def train_ids(self):
    return self._train_ids[::self.train_stride]

  @property
  def val_ids(self):
    return self._val_ids[::self.val_stride]

  @abc.abstractmethod
  def load_rgb(self, item_id):
    raise NotImplementedError()

  def load_depth(self, item_id):
    raise NotImplementedError()

  def load_relative_depth(self, item_id):
    raise NotImplementedError()

  @abc.abstractmethod
  def load_camera(self, item_id, scale_factor=1.0):
    raise NotImplementedError()

  def load_points(self, shuffle=False):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_appearance_id(self, item_id):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_camera_id(self, item_id):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_warp_id(self, item_id):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_time_id(self, item_id):
    raise NotImplementedError()

  def get_time(self, item_id):
    max_time = max(self.time_ids)
    return (self.get_time_id(item_id) / max_time) * 2.0 - 1.0

  @property
  @functools.lru_cache(maxsize=None)
  def appearance_ids(self) -> Sequence[int]:
    if not self.use_appearance_id:
      return tuple()
    return tuple(
        sorted(set([self.get_appearance_id(i) for i in self.train_ids])))

  @property
  @functools.lru_cache(maxsize=None)
  def camera_ids(self) -> Sequence[int]:
    if not self.use_camera_id:
      return tuple()
    return tuple(sorted(set([self.get_camera_id(i) for i in self.train_ids])))

  @property
  @functools.lru_cache(maxsize=None)
  def warp_ids(self) -> Sequence[int]:
    if not self.use_warp_id:
      return tuple()
    return tuple(sorted(set([self.get_warp_id(i) for i in self.train_ids])))

  @property
  def embeddings_dict(self):
    return immutabledict.immutabledict({
        'warp': self.warp_ids,
        'appearance': self.appearance_ids,
        'camera': self.camera_ids,
    })

  @property
  @functools.lru_cache(maxsize=None)
  def time_ids(self):
    if not self.use_time:
      return []
    return sorted(set([self.get_time_id(i) for i in self.train_ids]))

  @property
  def near(self) -> float:
    raise NotImplementedError()

  @property
  def far(self) -> float:
    raise NotImplementedError()

  @property
  def has_metadata(self):
    return self.use_appearance_id or self.use_warp_id or self.use_camera_id

  @abc.abstractmethod
  def load_test_cameras(self, count=None):
    raise NotImplementedError()

  def create_cameras_dataset(self,
                             cameras: Union[Sequence[cam.Camera],
                                            Sequence[gpath.GPath]],
                             flatten=False,
                             shuffle=False):
    """Creates a tf.data.Dataset from a list of cameras."""
    if isinstance(cameras[0], gpath.GPath) or isinstance(cameras[0], str):
      cameras = utils.parallel_map(self.load_camera, cameras)

    def _generator():
      for camera in cameras:
        yield {'camera_params': camera.get_parameters()}

    dataset = tf.data.Dataset.from_generator(
        _generator,
        output_signature={'camera_params': _TF_CAMERA_PARAMS_SIGNATURE})
    dataset = dataset.map(
        functools.partial(_camera_to_rays_fn, use_tf_camera=True), _TF_AUTOTUNE)

    if flatten:
      # Unbatch images to rows.
      dataset = dataset.unbatch()
      if shuffle:
        dataset = dataset.shuffle(20000)
      # Unbatch rows to rays.
      dataset = dataset.unbatch()
      if shuffle:
        dataset = dataset.shuffle(20000)

    return dataset

  def create_iterator(self,
                      item_ids,
                      batch_size: int,
                      repeat: bool = True,
                      flatten: bool = False,
                      shuffle: bool = False,
                      prefetch_size: int = 0,
                      shuffle_buffer_size: int = 1000000,
                      devices: Optional[Sequence[Any]] = None):
    """Creates a dataset iterator for JAX."""
    dataset = self.create_dataset(
        item_ids,
        flatten=flatten,
        shuffle=shuffle,
        row_shuffle_buffer_size=shuffle_buffer_size,
        pixel_shuffle_buffer_size=shuffle_buffer_size)
    return iterator_from_dataset(dataset=dataset,
                                 batch_size=batch_size,
                                 repeat=repeat,
                                 prefetch_size=prefetch_size,
                                 devices=devices)

  def create_dataset(self,
                     item_ids,
                     flatten=False,
                     shuffle=False,
                     row_shuffle_buffer_size=1000000,
                     pixel_shuffle_buffer_size=1000000):
    """Creates a tf.data.Dataset."""
    if self.preload:
      return self._create_preloaded_dataset(
          item_ids, flatten=flatten, shuffle=shuffle)
    else:
      return self._create_lazy_dataset(
          item_ids,
          flatten=flatten,
          shuffle=shuffle,
          row_shuffle_buffer_size=row_shuffle_buffer_size,
          pixel_shuffle_buffer_size=pixel_shuffle_buffer_size)

  def _create_preloaded_dataset(self, item_ids, flatten=False, shuffle=False):
    """Crates a dataset completely preloaded in memory.

    This creates a tf.data.Dataset which is constructed by load all data
    into memory and pre-shuffling (if applicable). This is much faster than
    having tf.data.Dataset handle individual items.

    Args:
      item_ids: the item IDs to construct the datset with.
      flatten: whether to flatten the image dimensions.
      shuffle: whether to shuffle the dataset.

    Returns:
      A tf.data.Dataset instance.
    """
    load_fn = functools.partial(self.get_item)
    data_list = utils.parallel_map(load_fn, item_ids)
    data_list = [_camera_to_rays_fn(item) for item in data_list]
    data_dict = utils.tree_collate(data_list)

    num_examples = data_dict['origins'].shape[0]
    heights = [x.shape[0] for x in data_dict['origins']]
    widths = [x.shape[1] for x in data_dict['origins']]

    # Broadcast appearance ID to match ray shapes.
    if 'metadata' in data_dict:
      for metadata_key, metadata in data_dict['metadata'].items():
        data_dict['metadata'][metadata_key] = np.stack([
            np.broadcast_to(x[None, None, :],
                            (heights[i], widths[i], x.shape[-1]))
            for i, x in enumerate(metadata)
        ])

    num_rays = int(sum([x * y for x, y in zip(heights, widths)]))
    shuffled_inds = self.rng.permutation(num_rays)

    logging.info('*** Loaded dataset items: num_rays=%d, num_examples=%d',
                 num_rays, num_examples)

    def _prepare_array(x):
      if not isinstance(x, np.ndarray):
        x = np.asarray(x)
      # Create last dimension if it doesn't exist.
      # The `and` part of the check ensures we're not touching ragged arrays.
      if x.ndim == 1 and x[0].ndim == 0:
        x = np.expand_dims(x, -1)
      if flatten:
        x = np.concatenate([x.reshape(-1, x.shape[-1]) for x in x], axis=0)
      if shuffle:
        x = x[shuffled_inds]
      return x

    out_dict = {}
    for key, value in data_dict.items():
      out_dict[key] = jax.tree_map(_prepare_array, value)

    return tf.data.Dataset.from_tensor_slices(out_dict)

  def _create_lazy_dataset(self,
                           item_ids,
                           flatten=False,
                           shuffle=False,
                           row_shuffle_buffer_size=1000000,
                           pixel_shuffle_buffer_size=1000000):
    """Crates a dataset that loads data in on the fly.

    This creates a tf.data.Dataset which lazily loads data as it is read.
    This allows for datasets that do not fit in memory, but this performs
    much worse. Only use if necessary.

    Args:
      item_ids: the item IDs to construct the datset with.
      flatten: whether to flatten the image dimensions.
      shuffle: whether to shuffle the dataset.
      row_shuffle_buffer_size: the shuffle buffer size for rows.
      pixel_shuffle_buffer_size: the shuffle buffer size for pixels.

    Returns:
      A tf.data.Dataset instance.
    """
    if shuffle:
      item_ids = self.rng.permutation(item_ids).tolist()

    dataset = tf.data.Dataset.from_tensor_slices(item_ids)
    dataset = dataset.map(
        self._get_item_py_function, num_parallel_calls=_TF_AUTOTUNE)
    dataset = dataset.map(_camera_to_rays_fn, num_parallel_calls=_TF_AUTOTUNE)
    if self.has_metadata:
      dataset = dataset.map(_tf_broadcast_metadata_fn)

    if flatten:
      # Unbatch images to rows.
      dataset = dataset.unbatch()
      if shuffle:
        dataset = dataset.shuffle(
            row_shuffle_buffer_size, reshuffle_each_iteration=True)
      # Unbatch rows to rays.
      dataset = dataset.unbatch()
      if shuffle:
        dataset = dataset.shuffle(
            pixel_shuffle_buffer_size, reshuffle_each_iteration=True)

    return dataset

  def _get_item_signature(self):
    """Returns the Tensorflow data signature for an item."""
    sig = {
        'rgb': tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
        'camera_params': _TF_CAMERA_PARAMS_SIGNATURE,
        'metadata': {},
    }
    if self.use_appearance_id:
      sig['metadata']['appearance'] = tf.TensorSpec(
          shape=[None], dtype=tf.uint32)
    if self.use_camera_id:
      sig['metadata']['camera'] = tf.TensorSpec(shape=[None], dtype=tf.uint32)
    if self.use_warp_id:
      sig['metadata']['warp'] = tf.TensorSpec(shape=[None], dtype=tf.uint32)
    if self.use_time:
      sig['metadata']['time'] = tf.TensorSpec(shape=[None], dtype=tf.float32)

    return sig

  def _get_item_py_function(self, item_id):
    """Wrap get_item in tf.numpy_function to allow eager execution.

    The only way to get Tensorflow to evaluate our simple `get_item` function
    in parallel seems to be to wrap it using tf.py_function or
    tf.numpy_function.

    Args:
      item_id: a single item ID encoded in a Tensor of type tf.string.

    Returns:
      A dictionary of tensors. See `self.get_item`.
    """
    output_signature = self._get_item_signature()
    # Unpack signature spec into types and shapes.
    output_types = nest.pack_sequence_as(
        output_signature, [x.dtype for x in nest.flatten(output_signature)])
    output_shapes = nest.pack_sequence_as(
        output_signature, [x.shape for x in nest.flatten(output_signature)])
    # tf.py_function and tf.numpy_function don't support a pytree of
    # TensorSpec yet so flatten them into a list.
    flattened_types = [
        tf.dtypes.as_dtype(dt) for dt in nest.flatten(output_types)
    ]
    flattened_shapes = nest.flatten(output_shapes)

    def _get_flat_item(i):
      values = self.get_item(i.decode())
      flattened_values = nest.flatten_up_to(output_types, values)
      # Cast types to the expected ones.
      flattened_values = [
          np.array(v, d.as_numpy_dtype)
          for v, d in zip(flattened_values, flattened_types)
      ]
      return flattened_values

    # Run `get_item` on the graph and get flat values.
    flat_values = tf.numpy_function(_get_flat_item, [item_id], flattened_types)

    # Fix the shapes of the output tensors.
    if output_shapes is not None:
      for ret_t, shape in zip(flat_values, flattened_shapes):
        ret_t.set_shape(shape)

    # Pack the values back into the dictionary structure.
    return nest.pack_sequence_as(output_types, flat_values)

  def parallel_get_items(self, item_ids, scale_factor=1.0):
    """Load data dictionaries indexed by indices in parallel."""
    load_fn = functools.partial(self.get_item, scale_factor=scale_factor)
    data_list = utils.parallel_map(load_fn, item_ids)
    data_dict = utils.tree_collate(data_list)
    return data_dict

  def get_item(self, item_id, scale_factor=1.0):
    """Load an example as a data dictionary.

    Args:
      item_id: the ID of the item to fetch.
      scale_factor: a scale factor to apply to the camera.

    Returns:
      A dictionary containing one of more of the following items:
        `rgb`: the RGB pixel values of each ray.
        `rays_dir`: the direction of each ray.
        `rays_origin`: the origin of each ray.
        `rays_pixels`: the pixel center of each ray.
        `metadata`: a dictionary of containing various metadata arrays. Each
          item is an array containing metadata IDs for each ray.
    """
    rgb = self.load_rgb(item_id)
    if scale_factor != 1.0:
      rgb = image_utils.rescale_image(rgb, scale_factor)

    camera = self.load_camera(item_id, scale_factor)
    data = {
        'camera_params': camera.get_parameters(),
        'rgb': rgb,
        'metadata': {},
    }

    if self.use_appearance_id:
      data['metadata']['appearance'] = np.atleast_1d(
          self.get_appearance_id(item_id))
    if self.use_camera_id:
      data['metadata']['camera'] = np.atleast_1d(self.get_camera_id(item_id))
    if self.use_warp_id:
      data['metadata']['warp'] = np.atleast_1d(self.get_warp_id(item_id))
    if self.use_time:
      data['metadata']['time'] = np.atleast_1d(self.get_time(item_id))

    if self.use_depth:
      depth = self.load_depth(item_id)
      if depth is not None:
        if scale_factor != 1.0:
          depth = image_utils.rescale_image(depth, scale_factor)
        data['depth'] = depth[..., np.newaxis]

    logging.info(
        '\tLoaded item %s: shape=%s, scale_factor=%f, metadata=%s',
        item_id,
        rgb.shape,
        scale_factor,
        str(data.get('metadata')))

    return data





# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Class for handling cameras."""
import copy
import json
from typing import Tuple, Union, Optional

import numpy as np

from hypernerf import gpath
from hypernerf import types


def _compute_residual_and_jacobian(
    x: np.ndarray,
    y: np.ndarray,
    xd: np.ndarray,
    yd: np.ndarray,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray]:
  """Auxiliary function of radial_and_tangential_undistort()."""
  # let r(x, y) = x^2 + y^2;
  #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3;
  r = x * x + y * y
  d = 1.0 + r * (k1 + r * (k2 + k3 * r))

  # The perfect projection is:
  # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
  # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
  #
  # Let's define
  #
  # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
  # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
  #
  # We are looking for a solution that satisfies
  # fx(x, y) = fy(x, y) = 0;
  fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
  fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

  # Compute derivative of d over [x, y]
  d_r = (k1 + r * (2.0 * k2 + 3.0 * k3 * r))
  d_x = 2.0 * x * d_r
  d_y = 2.0 * y * d_r

  # Compute derivative of fx over x and y.
  fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
  fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

  # Compute derivative of fy over x and y.
  fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
  fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

  return fx, fy, fx_x, fx_y, fy_x, fy_y


def _radial_and_tangential_undistort(
    xd: np.ndarray,
    yd: np.ndarray,
    k1: float = 0,
    k2: float = 0,
    k3: float = 0,
    p1: float = 0,
    p2: float = 0,
    eps: float = 1e-9,
    max_iterations=10) -> Tuple[np.ndarray, np.ndarray]:
  """Computes undistorted (x, y) from (xd, yd)."""
  # Initialize from the distorted point.
  x = xd.copy()
  y = yd.copy()

  for _ in range(max_iterations):
    fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
        x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2)
    denominator = fy_x * fx_y - fx_x * fy_y
    x_numerator = fx * fy_y - fy * fx_y
    y_numerator = fy * fx_x - fx * fy_x
    step_x = np.where(
        np.abs(denominator) > eps, x_numerator / denominator,
        np.zeros_like(denominator))
    step_y = np.where(
        np.abs(denominator) > eps, y_numerator / denominator,
        np.zeros_like(denominator))

    x = x + step_x
    y = y + step_y

  return x, y


class Camera:
  """Class to handle camera geometry."""

  def __init__(self,
               orientation: np.ndarray,
               position: np.ndarray,
               focal_length: Union[np.ndarray, float],
               principal_point: np.ndarray,
               image_size: np.ndarray,
               skew: Union[np.ndarray, float] = 0.0,
               pixel_aspect_ratio: Union[np.ndarray, float] = 1.0,
               radial_distortion: Optional[np.ndarray] = None,
               tangential_distortion: Optional[np.ndarray] = None,
               dtype=np.float32):
    """Constructor for camera class."""
    if radial_distortion is None:
      radial_distortion = np.array([0.0, 0.0, 0.0], dtype)
    if tangential_distortion is None:
      tangential_distortion = np.array([0.0, 0.0], dtype)

    self.orientation = np.array(orientation, dtype)
    self.position = np.array(position, dtype)
    self.focal_length = np.array(focal_length, dtype)
    self.principal_point = np.array(principal_point, dtype)
    self.skew = np.array(skew, dtype)
    self.pixel_aspect_ratio = np.array(pixel_aspect_ratio, dtype)
    self.radial_distortion = np.array(radial_distortion, dtype)
    self.tangential_distortion = np.array(tangential_distortion, dtype)
    self.image_size = np.array(image_size, np.uint32)
    self.dtype = dtype

  @classmethod
  def from_json(cls, path: types.PathType):
    """Loads a JSON camera into memory."""
    path = gpath.GPath(path)
    with path.open('r') as fp:
      camera_json = json.load(fp)

    # Fix old camera JSON.
    if 'tangential' in camera_json:
      camera_json['tangential_distortion'] = camera_json['tangential']

    return cls(
        orientation=np.asarray(camera_json['orientation']),
        position=np.asarray(camera_json['position']),
        focal_length=camera_json['focal_length'],
        principal_point=np.asarray(camera_json['principal_point']),
        skew=camera_json['skew'],
        pixel_aspect_ratio=camera_json['pixel_aspect_ratio'],
        radial_distortion=np.asarray(camera_json['radial_distortion']),
        tangential_distortion=np.asarray(camera_json['tangential_distortion']),
        image_size=np.asarray(camera_json['image_size']),
    )

  def to_json(self):
    return {
        k: (v.tolist() if hasattr(v, 'tolist') else v)
        for k, v in self.get_parameters().items()
    }

  def get_parameters(self):
    return {
        'orientation': self.orientation,
        'position': self.position,
        'focal_length': self.focal_length,
        'principal_point': self.principal_point,
        'skew': self.skew,
        'pixel_aspect_ratio': self.pixel_aspect_ratio,
        'radial_distortion': self.radial_distortion,
        'tangential_distortion': self.tangential_distortion,
        'image_size': self.image_size,
    }

  @property
  def scale_factor_x(self):
    return self.focal_length

  @property
  def scale_factor_y(self):
    return self.focal_length * self.pixel_aspect_ratio

  @property
  def principal_point_x(self):
    return self.principal_point[0]

  @property
  def principal_point_y(self):
    return self.principal_point[1]

  @property
  def has_tangential_distortion(self):
    return any(self.tangential_distortion != 0.0)

  @property
  def has_radial_distortion(self):
    return any(self.radial_distortion != 0.0)

  @property
  def image_size_y(self):
    return self.image_size[1]

  @property
  def image_size_x(self):
    return self.image_size[0]

  @property
  def image_shape(self):
    return self.image_size_y, self.image_size_x

  @property
  def optical_axis(self):
    return self.orientation[2, :]

  @property
  def translation(self):
    return -np.matmul(self.orientation, self.position)

  def pixel_to_local_rays(self, pixels: np.ndarray):
    """Returns the local ray directions for the provided pixels."""
    y = ((pixels[..., 1] - self.principal_point_y) / self.scale_factor_y)
    x = ((pixels[..., 0] - self.principal_point_x - y * self.skew) /
         self.scale_factor_x)

    if self.has_radial_distortion or self.has_tangential_distortion:
      x, y = _radial_and_tangential_undistort(
          x,
          y,
          k1=self.radial_distortion[0],
          k2=self.radial_distortion[1],
          k3=self.radial_distortion[2],
          p1=self.tangential_distortion[0],
          p2=self.tangential_distortion[1])

    dirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    return dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

  def pixels_to_rays(self, pixels: np.ndarray) -> np.ndarray:
    """Returns the rays for the provided pixels.

    Args:
      pixels: [A1, ..., An, 2] tensor or np.array containing 2d pixel positions.

    Returns:
        An array containing the normalized ray directions in world coordinates.
    """
    if pixels.shape[-1] != 2:
      raise ValueError('The last dimension of pixels must be 2.')
    if pixels.dtype != self.dtype:
      raise ValueError(f'pixels dtype ({pixels.dtype!r}) must match camera '
                       f'dtype ({self.dtype!r})')

    batch_shape = pixels.shape[:-1]
    pixels = np.reshape(pixels, (-1, 2))

    local_rays_dir = self.pixel_to_local_rays(pixels)
    rays_dir = np.matmul(self.orientation.T, local_rays_dir[..., np.newaxis])
    rays_dir = np.squeeze(rays_dir, axis=-1)

    # Normalize rays.
    rays_dir /= np.linalg.norm(rays_dir, axis=-1, keepdims=True)
    rays_dir = rays_dir.reshape((*batch_shape, 3))
    return rays_dir

  def pixels_to_points(self, pixels: np.ndarray, depth: np.ndarray):
    rays_through_pixels = self.pixels_to_rays(pixels)
    cosa = np.matmul(rays_through_pixels, self.optical_axis)
    points = (
        rays_through_pixels * depth[..., np.newaxis] / cosa[..., np.newaxis] +
        self.position)
    return points

  def points_to_local_points(self, points: np.ndarray):
    translated_points = points - self.position
    local_points = (np.matmul(self.orientation, translated_points.T)).T
    return local_points

  def project(self, points: np.ndarray):
    """Projects a 3D point (x,y,z) to a pixel position (x,y)."""
    batch_shape = points.shape[:-1]
    points = points.reshape((-1, 3))
    local_points = self.points_to_local_points(points)

    # Get normalized local pixel positions.
    x = local_points[..., 0] / local_points[..., 2]
    y = local_points[..., 1] / local_points[..., 2]
    r2 = x**2 + y**2

    # Apply radial distortion.
    distortion = 1.0 + r2 * (
        self.radial_distortion[0] + r2 *
        (self.radial_distortion[1] + self.radial_distortion[2] * r2))

    # Apply tangential distortion.
    x_times_y = x * y
    x = (
        x * distortion + 2.0 * self.tangential_distortion[0] * x_times_y +
        self.tangential_distortion[1] * (r2 + 2.0 * x**2))
    y = (
        y * distortion + 2.0 * self.tangential_distortion[1] * x_times_y +
        self.tangential_distortion[0] * (r2 + 2.0 * y**2))

    # Map the distorted ray to the image plane and return the depth.
    pixel_x = self.focal_length * x + self.skew * y + self.principal_point_x
    pixel_y = (self.focal_length * self.pixel_aspect_ratio * y
               + self.principal_point_y)

    pixels = np.stack([pixel_x, pixel_y], axis=-1)
    return pixels.reshape((*batch_shape, 2))

  def get_pixel_centers(self):
    """Returns the pixel centers."""
    xx, yy = np.meshgrid(np.arange(self.image_size_x, dtype=self.dtype),
                         np.arange(self.image_size_y, dtype=self.dtype))
    return np.stack([xx, yy], axis=-1) + 0.5

  def scale(self, scale: float):
    """Scales the camera."""
    if scale <= 0:
      raise ValueError('scale needs to be positive.')

    new_camera = Camera(
        orientation=self.orientation.copy(),
        position=self.position.copy(),
        focal_length=self.focal_length * scale,
        principal_point=self.principal_point.copy() * scale,
        skew=self.skew,
        pixel_aspect_ratio=self.pixel_aspect_ratio,
        radial_distortion=self.radial_distortion.copy(),
        tangential_distortion=self.tangential_distortion.copy(),
        image_size=np.array((int(round(self.image_size[0] * scale)),
                             int(round(self.image_size[1] * scale)))),
    )
    return new_camera

  def look_at(self, position, look_at, up, eps=1e-6):
    """Creates a copy of the camera which looks at a given point.

    Copies the provided vision_sfm camera and returns a new camera that is
    positioned at `camera_position` while looking at `look_at_position`.
    Camera intrinsics are copied by this method. A common value for the
    up_vector is (0, 1, 0).

    Args:
      position: A (3,) numpy array representing the position of the camera.
      look_at: A (3,) numpy array representing the location the camera
        looks at.
      up: A (3,) numpy array representing the up direction, whose
        projection is parallel to the y-axis of the image plane.
      eps: a small number to prevent divides by zero.

    Returns:
      A new camera that is copied from the original but is positioned and
        looks at the provided coordinates.

    Raises:
      ValueError: If the camera position and look at position are very close
        to each other or if the up-vector is parallel to the requested optical
        axis.
    """

    look_at_camera = self.copy()
    optical_axis = look_at - position
    norm = np.linalg.norm(optical_axis)
    if norm < eps:
      raise ValueError('The camera center and look at position are too close.')
    optical_axis /= norm

    right_vector = np.cross(optical_axis, up)
    norm = np.linalg.norm(right_vector)
    if norm < eps:
      raise ValueError('The up-vector is parallel to the optical axis.')
    right_vector /= norm

    # The three directions here are orthogonal to each other and form a right
    # handed coordinate system.
    camera_rotation = np.identity(3)
    camera_rotation[0, :] = right_vector
    camera_rotation[1, :] = np.cross(optical_axis, right_vector)
    camera_rotation[2, :] = optical_axis

    look_at_camera.position = position
    look_at_camera.orientation = camera_rotation
    return look_at_camera

  def crop_image_domain(
      self, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0):
    """Returns a copy of the camera with adjusted image bounds.

    Args:
      left: number of pixels by which to reduce (or augment, if negative) the
        image domain at the associated boundary.
      right: likewise.
      top: likewise.
      bottom: likewise.

    The crop parameters may not cause the camera image domain dimensions to
    become non-positive.

    Returns:
      A camera with adjusted image dimensions.  The focal length is unchanged,
      and the principal point is updated to preserve the original principal
      axis.
    """

    crop_left_top = np.array([left, top])
    crop_right_bottom = np.array([right, bottom])
    new_resolution = self.image_size - crop_left_top - crop_right_bottom
    new_principal_point = self.principal_point - crop_left_top
    if np.any(new_resolution <= 0):
      raise ValueError('Crop would result in non-positive image dimensions.')

    new_camera = self.copy()
    new_camera.image_size = np.array([int(new_resolution[0]),
                                      int(new_resolution[1])])
    new_camera.principal_point = np.array([new_principal_point[0],
                                           new_principal_point[1]])
    return new_camera

  def copy(self):
    return copy.deepcopy(self)