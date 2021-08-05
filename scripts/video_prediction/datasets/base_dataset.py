import glob
import os
import random
import re
import six
from collections import OrderedDict

import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# from tensorflow.contrib.training import HParams

PARAM_RE = re.compile(r"""
  (?P<name>[a-zA-Z][\w]*)      # variable name: "var" or "x"
  (\[\s*(?P<index>\d+)\s*\])?  # (optional) index: "1" or None
  \s*=\s*
  ((?P<val>[^,\[]*)            # single value: "a" or None
   |
   \[(?P<vals>[^\]]*)\])       # list of values: None or "1,2,3"
  ($|,\s*)""", re.VERBOSE)


def _parse_fail(name, var_type, value, values):
  """Helper function for raising a value error for bad assignment."""
  raise ValueError(
      'Could not parse hparam \'%s\' of type \'%s\' with value \'%s\' in %s' %
      (name, var_type.__name__, value, values))


def _reuse_fail(name, values):
  """Helper function for raising a value error for reuse of name."""
  raise ValueError('Multiple assignments to variable \'%s\' in %s' % (name,
                                                                      values))


def _process_scalar_value(name, parse_fn, var_type, m_dict, values,
                          results_dictionary):
  """Update results_dictionary with a scalar value.
  Used to update the results_dictionary to be returned by parse_values when
  encountering a clause with a scalar RHS (e.g.  "s=5" or "arr[0]=5".)
  Mutates results_dictionary.
  Args:
    name: Name of variable in assignment ("s" or "arr").
    parse_fn: Function for parsing the actual value.
    var_type: Type of named variable.
    m_dict: Dictionary constructed from regex parsing.
      m_dict['val']: RHS value (scalar)
      m_dict['index']: List index value (or None)
    values: Full expression being parsed
    results_dictionary: The dictionary being updated for return by the parsing
      function.
  Raises:
    ValueError: If the name has already been used.
  """
  try:
    parsed_value = parse_fn(m_dict['val'])
  except ValueError:
    _parse_fail(name, var_type, m_dict['val'], values)

  # If no index is provided
  if not m_dict['index']:
    if name in results_dictionary:
      _reuse_fail(name, values)
    results_dictionary[name] = parsed_value
  else:
    if name in results_dictionary:
      # The name has already been used as a scalar, then it
      # will be in this dictionary and map to a non-dictionary.
      if not isinstance(results_dictionary.get(name), dict):
        _reuse_fail(name, values)
    else:
      results_dictionary[name] = {}

    index = int(m_dict['index'])
    # Make sure the index position hasn't already been assigned a value.
    if index in results_dictionary[name]:
      _reuse_fail('{}[{}]'.format(name, index), values)
    results_dictionary[name][index] = parsed_value


def _process_list_value(name, parse_fn, var_type, m_dict, values,
                        results_dictionary):
  """Update results_dictionary from a list of values.
  Used to update results_dictionary to be returned by parse_values when
  encountering a clause with a list RHS (e.g.  "arr=[1,2,3]".)
  Mutates results_dictionary.
  Args:
    name: Name of variable in assignment ("arr").
    parse_fn: Function for parsing individual values.
    var_type: Type of named variable.
    m_dict: Dictionary constructed from regex parsing.
      m_dict['val']: RHS value (scalar)
    values: Full expression being parsed
    results_dictionary: The dictionary being updated for return by the parsing
      function.
  Raises:
    ValueError: If the name has an index or the values cannot be parsed.
  """
  if m_dict['index'] is not None:
    raise ValueError('Assignment of a list to a list index.')
  elements = filter(None, re.split('[ ,]', m_dict['vals']))
  # Make sure the name hasn't already been assigned a value
  if name in results_dictionary:
    raise _reuse_fail(name, values)
  try:
    results_dictionary[name] = [parse_fn(e) for e in elements]
  except ValueError:
    _parse_fail(name, var_type, m_dict['vals'], values)


def _cast_to_type_if_compatible(name, param_type, value):
  """Cast hparam to the provided type, if compatible.
  Args:
    name: Name of the hparam to be cast.
    param_type: The type of the hparam.
    value: The value to be cast, if compatible.
  Returns:
    The result of casting `value` to `param_type`.
  Raises:
    ValueError: If the type of `value` is not compatible with param_type.
      * If `param_type` is a string type, but `value` is not.
      * If `param_type` is a boolean, but `value` is not, or vice versa.
      * If `param_type` is an integer type, but `value` is not.
      * If `param_type` is a float type, but `value` is not a numeric type.
  """
  fail_msg = (
      "Could not cast hparam '%s' of type '%s' from value %r" %
      (name, param_type, value))

  # Some callers use None, for which we can't do any casting/checking. :(
  if issubclass(param_type, type(None)):
    return value

  # Avoid converting a non-string type to a string.
  if (issubclass(param_type, (six.string_types, six.binary_type)) and
      not isinstance(value, (six.string_types, six.binary_type))):
    raise ValueError(fail_msg)

  # Avoid converting a number or string type to a boolean or vice versa.
  if issubclass(param_type, bool) != isinstance(value, bool):
    raise ValueError(fail_msg)

  # Avoid converting float to an integer (the reverse is fine).
  if (issubclass(param_type, numbers.Integral) and
      not isinstance(value, numbers.Integral)):
    raise ValueError(fail_msg)

  # Avoid converting a non-numeric type to a numeric type.
  if (issubclass(param_type, numbers.Number) and
      not isinstance(value, numbers.Number)):
    raise ValueError(fail_msg)

  return param_type(value)


def parse_values(values, type_map):
  """Parses hyperparameter values from a string into a python map.
  `values` is a string containing comma-separated `name=value` pairs.
  For each pair, the value of the hyperparameter named `name` is set to
  `value`.
  If a hyperparameter name appears multiple times in `values`, a ValueError
  is raised (e.g. 'a=1,a=2', 'a[1]=1,a[1]=2').
  If a hyperparameter name in both an index assignment and scalar assignment,
  a ValueError is raised.  (e.g. 'a=[1,2,3],a[0] = 1').
  The `value` in `name=value` must follows the syntax according to the
  type of the parameter:
  *  Scalar integer: A Python-parsable integer point value.  E.g.: 1,
     100, -12.
  *  Scalar float: A Python-parsable floating point value.  E.g.: 1.0,
     -.54e89.
  *  Boolean: Either true or false.
  *  Scalar string: A non-empty sequence of characters, excluding comma,
     spaces, and square brackets.  E.g.: foo, bar_1.
  *  List: A comma separated list of scalar values of the parameter type
     enclosed in square brackets.  E.g.: [1,2,3], [1.0,1e-12], [high,low].
  When index assignment is used, the corresponding type_map key should be the
  list name.  E.g. for "arr[1]=0" the type_map must have the key "arr" (not
  "arr[1]").
  Args:
    values: String.  Comma separated list of `name=value` pairs where
      'value' must follow the syntax described above.
    type_map: A dictionary mapping hyperparameter names to types.  Note every
      parameter name in values must be a key in type_map.  The values must
      conform to the types indicated, where a value V is said to conform to a
      type T if either V has type T, or V is a list of elements of type T.
      Hence, for a multidimensional parameter 'x' taking float values,
      'x=[0.1,0.2]' will parse successfully if type_map['x'] = float.
  Returns:
    A python map mapping each name to either:
    * A scalar value.
    * A list of scalar values.
    * A dictionary mapping index numbers to scalar values.
    (e.g. "x=5,L=[1,2],arr[1]=3" results in {'x':5,'L':[1,2],'arr':{1:3}}")
  Raises:
    ValueError: If there is a problem with input.
    * If `values` cannot be parsed.
    * If a list is assigned to a list index (e.g. 'a[1] = [1,2,3]').
    * If the same rvalue is assigned two different values (e.g. 'a=1,a=2',
      'a[1]=1,a[1]=2', or 'a=1,a=[1]')
  """
  results_dictionary = {}
  pos = 0
  while pos < len(values):
    m = PARAM_RE.match(values, pos)
    if not m:
      raise ValueError('Malformed hyperparameter value: %s' % values[pos:])
    # Check that there is a comma between parameters and move past it.
    pos = m.end()
    # Parse the values.
    m_dict = m.groupdict()
    name = m_dict['name']
    if name not in type_map:
      raise ValueError('Unknown hyperparameter type for %s' % name)
    type_ = type_map[name]

    # Set up correct parsing function (depending on whether type_ is a bool)
    if type_ == bool:

      def parse_bool(value):
        if value in ['true', 'True']:
          return True
        elif value in ['false', 'False']:
          return False
        else:
          try:
            return bool(int(value))
          except ValueError:
            _parse_fail(name, type_, value, values)

      parse = parse_bool
    else:
      parse = type_

    # If a singe value is provided
    if m_dict['val'] is not None:
      _process_scalar_value(name, parse, type_, m_dict, values,
                            results_dictionary)

    # If the assigned value is a list:
    elif m_dict['vals'] is not None:
      _process_list_value(name, parse, type_, m_dict, values,
                          results_dictionary)

    else:  # Not assigned a list or value
      _parse_fail(name, type_, '', values)

  return results_dictionary


class BaseVideoDataset(object):
    def __init__(self, input_dir, mode='train', num_epochs=None, seed=None,
                 hparams_dict=None, hparams=None, hparam_def=None):
        """
        Args:
            input_dir: either a directory containing subdirectories train,
                val, test, etc, or a directory containing the tfrecords.
            mode: either train, val, or test
            num_epochs: if None, dataset is iterated indefinitely.
            seed: random seed for the op that samples subsequences.
            hparams_dict: a dict of `name=value` pairs, where `name` must be
                defined in `self.get_default_hparams()`.
            hparams: a string of comma separated list of `name=value` pairs,
                where `name` must be defined in `self.get_default_hparams()`.
                These values overrides any values in hparams_dict (if any).

        Note:
            self.input_dir is the directory containing the tfrecords.
        """
        self.input_dir = os.path.normpath(os.path.expanduser(input_dir))
        self.mode = mode
        self.num_epochs = num_epochs
        self.seed = seed
        self._hparam_types = {}
        if hparam_def:
          self._init_from_proto(hparam_def)
          if kwargs:
            raise ValueError('hparam_def and initialization values are '
                             'mutually exclusive')
        else:
          for name, value in six.iteritems(self.get_default_hparams_dict()):
            self.add_hparam(name, value)

        if self.mode not in ('train', 'val', 'test'):
            raise ValueError('Invalid mode %s' % self.mode)

        if not os.path.exists(self.input_dir):
            raise FileNotFoundError("input_dir %s does not exist" % self.input_dir)
        self.filenames = None
        # look for tfrecords in input_dir and input_dir/mode directories
        for input_dir in [self.input_dir, os.path.join(self.input_dir, self.mode)]:
            filenames = glob.glob(os.path.join(input_dir, '*.tfrecord*'))
            if filenames:
                self.input_dir = input_dir
                self.filenames = sorted(filenames)  # ensures order is the same across systems
                break
        if not self.filenames:
            raise FileNotFoundError('No tfrecords were found in %s.' % self.input_dir)
        self.dataset_name = os.path.basename(os.path.split(self.input_dir)[0])

        self.state_like_names_and_shapes = OrderedDict()
        self.action_like_names_and_shapes = OrderedDict()

        self.hparams = self.parse_hparams(hparams_dict, hparams)

    def get_default_hparams_dict(self):
        """
        Returns:
            A dict with the following hyperparameters.

            crop_size: crop image into a square with sides of this length.
            scale_size: resize image to this size after it has been cropped.
            context_frames: the number of ground-truth frames to pass in at
                start.
            sequence_length: the number of frames in the video sequence, so
                state-like sequences are of length sequence_length and
                action-like sequences are of length sequence_length - 1.
                This number includes the context frames.
            long_sequence_length: the number of frames for the long version.
                The default is the same as sequence_length.
            frame_skip: number of frames to skip in between outputted frames,
                so frame_skip=0 denotes no skipping.
            time_shift: shift in time by multiples of this, so time_shift=1
                denotes all possible shifts. time_shift=0 denotes no shifting.
                It is ignored (equiv. to time_shift=0) when mode != 'train'.
            force_time_shift: whether to do the shift in time regardless of
                mode.
            shuffle_on_val: whether to shuffle the samples regardless if mode
                is 'train' or 'val'. Shuffle never happens when mode is 'test'.
            use_state: whether to load and return state and actions.
        """
        hparams = dict(
            crop_size=0,
            scale_size=0,
            context_frames=1,
            sequence_length=0,
            long_sequence_length=0,
            frame_skip=0,
            time_shift=1,
            force_time_shift=False,
            shuffle_on_val=False,
            use_state=False,
        )
        return hparams

    # def get_default_hparams(self):
    #     return HParams(**self.get_default_hparams_dict())
    
    def add_hparam(self, name, value):
    # Keys in kwargs are unique, but 'name' could the name of a pre-existing
    # attribute of this object.  In that case we refuse to use it as a
    # hyperparameter name.
        if getattr(self, name, None) is not None:
          raise ValueError('Hyperparameter name is reserved: %s' % name)
        if isinstance(value, (list, tuple)):
          if not value:
            raise ValueError(
                'Multi-valued hyperparameters cannot be empty: %s' % name)
          self._hparam_types[name] = (type(value[0]), True)
        else:
          self._hparam_types[name] = (type(value), False)
        setattr(self, name, value)
    
    def set_hparam(self, name, value):
        """Set the value of an existing hyperparameter.
        This function verifies that the type of the value matches the type of the
        existing hyperparameter.
        Args:
          name: Name of the hyperparameter.
          value: New value of the hyperparameter.
        Raises:
          ValueError: If there is a type mismatch.
        """
        param_type, is_list = self._hparam_types[name]
        if isinstance(value, list):
          if not is_list:
            raise ValueError(
                'Must not pass a list for single-valued parameter: %s' % name)
        else:
          if is_list:
            raise ValueError(
                'Must pass a list for multi-valued parameter: %s.' % name)
    
    def parse(self, values):
        """Override hyperparameter values, parsing new values from a string.
        See parse_values for more detail on the allowed format for values.
        Args:
          values: String.  Comma separated list of `name=value` pairs where
            'value' must follow the syntax described above.
        Returns:
          The `HParams` instance.
        Raises:
          ValueError: If `values` cannot be parsed.
        """
        type_map = dict()
        for name, t in self._hparam_types.items():
          param_type, _ = t
          type_map[name] = param_type

        values_map = parse_values(values, type_map)
        return self.override_from_dict(values_map)
    

    def override_from_dict(self, values_dict):
        """Override hyperparameter values, parsing new values from a dictionary.
        Args:
           values_dict: Dictionary of name:value pairs.
        Returns:
          The `HParams` instance.
        Raises:
          ValueError: If `values_dict` cannot be parsed.
        """
        for name, value in values_dict.items():
          self.set_hparam(name, value)
    

    def parse_hparams(self, hparams_dict, hparams):
        # parsed_hparams = self.get_default_hparams().override_from_dict(hparams_dict or {})
        if hparams_dict:
          parsed_hparams = hparams_dict
        else:
          parsed_hparams = self.get_default_hparams_dict()
        # parsed_hparams = self.override_from_dict(hparams_dict or {})
        if hparams:
            if not isinstance(hparams, (list, tuple)):
                hparams = [hparams]
            for hparam in hparams:
                 parsed_hparams = self.parse(hparam)
        if not "long_sequence_length" in parsed_hparams:
            parsed_hparams["long_sequence_length" ] = parsed_hparams['sequence_length']
        return parsed_hparams

    @property
    def jpeg_encoding(self):
        raise NotImplementedError

    def set_sequence_length(self, sequence_length):
        self.hparams.sequence_length = sequence_length

    def filter(self, serialized_example):
        return tf.convert_to_tensor(True)

    def parser(self, serialized_example):
        """
        Parses a single tf.train.Example or tf.train.SequenceExample into
        images, states, actions, etc tensors.
        """
        raise NotImplementedError

    def make_dataset(self, batch_size):
        filenames = self.filenames
        shuffle = self.mode == 'train' or (self.mode == 'val' and self.hparams.shuffle_on_val)
        if shuffle:
            random.shuffle(filenames)

        dataset = tf.data.TFRecordDataset(filenames, buffer_size=8 * 1024 * 1024)
        dataset = dataset.filter(self.filter)
        if shuffle:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1024, count=self.num_epochs))
        else:
            dataset = dataset.repeat(self.num_epochs)

        def _parser(serialized_example):
            state_like_seqs, action_like_seqs = self.parser(serialized_example)
            seqs = OrderedDict(list(state_like_seqs.items()) + list(action_like_seqs.items()))
            return seqs

        num_parallel_calls = None if shuffle else 1  # for reproducibility (e.g. sampled subclips from the test set)
        dataset = dataset.apply(tf.data.experimental.map_and_batch(
            _parser, batch_size, drop_remainder=True, num_parallel_calls=num_parallel_calls))
        dataset = dataset.prefetch(batch_size)
        return dataset

    def make_batch(self, batch_size):
        dataset = self.make_dataset(batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def decode_and_preprocess_images(self, image_buffers, image_shape):
        def decode_and_preprocess_image(image_buffer):
            image_buffer = tf.reshape(image_buffer, [])
            if self.jpeg_encoding:
                image = tf.io.decode_jpeg(image_buffer)
            else:
                image = tf.io.decode_raw(image_buffer, tf.uint8)
            image = tf.reshape(image, image_shape)
            crop_size = self.hparams['crop_size']
            scale_size = self.hparams['scale_size']
            if crop_size or scale_size:
                if not crop_size:
                    crop_size = min(image_shape[0], image_shape[1])
                image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
                image = tf.reshape(image, [crop_size, crop_size, 3])
                if scale_size:
                    # upsample with bilinear interpolation but downsample with area interpolation
                    if crop_size < scale_size:
                        image = tf.image.resize_images(image, [scale_size, scale_size],
                                                       method=tf.image.ResizeMethod.BILINEAR)
                    elif crop_size > scale_size:
                        image = tf.image.resize_images(image, [scale_size, scale_size],
                                                       method=tf.image.ResizeMethod.AREA)
                    else:
                        # image remains unchanged
                        pass
            return image

        if not isinstance(image_buffers, (list, tuple)):
            image_buffers = tf.unstack(image_buffers)
        images = [decode_and_preprocess_image(image_buffer) for image_buffer in image_buffers]
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        return images

    def slice_sequences(self, state_like_seqs, action_like_seqs, example_sequence_length):
        """
        Slices sequences of length `example_sequence_length` into subsequences
        of length `sequence_length`. The dicts of sequences are updated
        in-place and the same dicts are returned.
        """
        # handle random shifting and frame skip
        sequence_length = self.hparams['sequence_length']  # desired sequence length
        frame_skip = self.hparams['frame_skip']
        time_shift = self.hparams['time_shift']
        if (time_shift and self.mode == 'train') or self.hparams.force_time_shift:
            assert time_shift > 0 and isinstance(time_shift, int)
            if isinstance(example_sequence_length, tf.Tensor):
                example_sequence_length = tf.cast(example_sequence_length, tf.int32)
            num_shifts = ((example_sequence_length - 1) - (sequence_length - 1) * (frame_skip + 1)) // time_shift
            assert_message = ('example_sequence_length has to be at least %d when '
                              'sequence_length=%d, frame_skip=%d.' %
                              ((sequence_length - 1) * (frame_skip + 1) + 1,
                               sequence_length, frame_skip))
            with tf.control_dependencies([tf.assert_greater_equal(num_shifts, 0,
                    data=[example_sequence_length, num_shifts], message=assert_message)]):
                t_start = tf.random_uniform([], 0, num_shifts + 1, dtype=tf.int32, seed=self.seed) * time_shift
        else:
            t_start = 0
        state_like_t_slice = slice(t_start, t_start + (sequence_length - 1) * (frame_skip + 1) + 1, frame_skip + 1)
        action_like_t_slice = slice(t_start, t_start + (sequence_length - 1) * (frame_skip + 1))

        for example_name, seq in state_like_seqs.items():
            seq = tf.convert_to_tensor(seq)[state_like_t_slice]
            seq.set_shape([sequence_length] + seq.shape.as_list()[1:])
            state_like_seqs[example_name] = seq
        for example_name, seq in action_like_seqs.items():
            seq = tf.convert_to_tensor(seq)[action_like_t_slice]
            seq.set_shape([(sequence_length - 1) * (frame_skip + 1)] + seq.shape.as_list()[1:])
            # concatenate actions of skipped frames into single macro actions
            seq = tf.reshape(seq, [sequence_length - 1, -1])
            action_like_seqs[example_name] = seq
        return state_like_seqs, action_like_seqs

    def num_examples_per_epoch(self):
        raise NotImplementedError


class VideoDataset(BaseVideoDataset):
    """
    This class supports reading tfrecords where a sequence is stored as
    multiple tf.train.Example and each of them is stored under a different
    feature name (which is indexed by the time step).
    """
    def __init__(self, *args, **kwargs):
        super(VideoDataset, self).__init__(*args, **kwargs)
        self._max_sequence_length = None
        self._dict_message = None

    def _check_or_infer_shapes(self):
        """
        Should be called after state_like_names_and_shapes and
        action_like_names_and_shapes have been finalized.
        """
        state_like_names_and_shapes = OrderedDict([(k, list(v)) for k, v in self.state_like_names_and_shapes.items()])
        action_like_names_and_shapes = OrderedDict([(k, list(v)) for k, v in self.action_like_names_and_shapes.items()])
        from google.protobuf.json_format import MessageToDict
        example = next(tf.compat.v1.python_io.tf_record_iterator(self.filenames[0]))
        self._dict_message = MessageToDict(tf.train.Example.FromString(example))
        for example_name, name_and_shape in (list(state_like_names_and_shapes.items()) +
                                             list(action_like_names_and_shapes.items())):
            name, shape = name_and_shape
            feature = self._dict_message['features']['feature']
            names = [name_ for name_ in feature.keys() if re.search(name.replace('%d', '\d+'), name_) is not None]
            if not names:
                raise ValueError('Could not found any feature with name pattern %s.' % name)
            if example_name in self.state_like_names_and_shapes:
                sequence_length = len(names)
            else:
                sequence_length = len(names) + 1
            if self._max_sequence_length is None:
                self._max_sequence_length = sequence_length
            else:
                self._max_sequence_length = min(sequence_length, self._max_sequence_length)
            name = names[0]
            feature = feature[name]
            list_type, = feature.keys()
            if list_type == 'floatList':
                inferred_shape = (len(feature[list_type]['value']),)
                if shape is None:
                    name_and_shape[1] = inferred_shape
                else:
                    if inferred_shape != shape:
                        raise ValueError('Inferred shape for feature %s is %r but instead got shape %r.' %
                                         (name, inferred_shape, shape))
            elif list_type == 'bytesList':
                image_str, = feature[list_type]['value']
                # try to infer image shape
                inferred_shape = None
                if not self.jpeg_encoding:
                    spatial_size = len(image_str) // 4
                    height = width = int(np.sqrt(spatial_size))  # assume square image
                    if len(image_str) == (height * width * 4):
                        inferred_shape = (height, width, 3)
                if shape is None:
                    if inferred_shape is not None:
                        name_and_shape[1] = inferred_shape
                    else:
                        raise ValueError('Unable to infer shape for feature %s of size %d.' % (name, len(image_str)))
                else:
                    if inferred_shape is not None and inferred_shape != shape:
                        raise ValueError('Inferred shape for feature %s is %r but instead got shape %r.' %
                                         (name, inferred_shape, shape))
            else:
                raise NotImplementedError
        self.state_like_names_and_shapes = OrderedDict([(k, tuple(v)) for k, v in state_like_names_and_shapes.items()])
        self.action_like_names_and_shapes = OrderedDict([(k, tuple(v)) for k, v in action_like_names_and_shapes.items()])

        # set sequence_length to the longest possible if it is not specified
        if not self.hparams.sequence_length:
            self.hparams.sequence_length = (self._max_sequence_length - 1) // (self.hparams.frame_skip + 1) + 1

    def set_sequence_length(self, sequence_length):
        if not sequence_length:
            sequence_length = (self._max_sequence_length - 1) // (self.hparams.frame_skip + 1) + 1
        self.hparams.sequence_length = sequence_length

    def parser(self, serialized_example):
        """
        Parses a single tf.train.Example into images, states, actions, etc tensors.
        """
        features = dict()
        for i in range(self._max_sequence_length):
            for example_name, (name, shape) in self.state_like_names_and_shapes.items():
                if example_name == 'images':  # special handling for image
                    features[name % i] = tf.io.FixedLenFeature([1], tf.string)
                else:
                    features[name % i] = tf.io.FixedLenFeature(shape, tf.float32)
        for i in range(self._max_sequence_length - 1):
            for example_name, (name, shape) in self.action_like_names_and_shapes.items():
                features[name % i] = tf.io.FixedLenFeature(shape, tf.float32)

        # check that the features are in the tfrecord
        for name in features.keys():
            if name not in self._dict_message['features']['feature']:
                raise ValueError('Feature with name %s not found in tfrecord. Possible feature names are:\n%s' %
                                 (name, '\n'.join(sorted(self._dict_message['features']['feature'].keys()))))

        # parse all the features of all time steps together
        features = tf.io.parse_single_example(serialized_example, features=features)

        state_like_seqs = OrderedDict([(example_name, []) for example_name in self.state_like_names_and_shapes])
        action_like_seqs = OrderedDict([(example_name, []) for example_name in self.action_like_names_and_shapes])
        for i in range(self._max_sequence_length):
            for example_name, (name, shape) in self.state_like_names_and_shapes.items():
                state_like_seqs[example_name].append(features[name % i])
        for i in range(self._max_sequence_length - 1):
            for example_name, (name, shape) in self.action_like_names_and_shapes.items():
                action_like_seqs[example_name].append(features[name % i])

        # for this class, it's much faster to decode and preprocess the entire sequence before sampling a slice
        _, image_shape = self.state_like_names_and_shapes['images']
        state_like_seqs['images'] = self.decode_and_preprocess_images(state_like_seqs['images'], image_shape)

        state_like_seqs, action_like_seqs = \
            self.slice_sequences(state_like_seqs, action_like_seqs, self._max_sequence_length)
        return state_like_seqs, action_like_seqs


class SequenceExampleVideoDataset(BaseVideoDataset):
    """
    This class supports reading tfrecords where an entire sequence is stored as
    a single tf.train.SequenceExample.
    """
    def parser(self, serialized_example):
        """
        Parses a single tf.train.SequenceExample into images, states, actions, etc tensors.
        """
        sequence_features = dict()
        for example_name, (name, shape) in self.state_like_names_and_shapes.items():
            if example_name == 'images':  # special handling for image
                sequence_features[name] = tf.io.FixedLenSequenceFeature([1], tf.string)
            else:
                sequence_features[name] = tf.io.FixedLenSequenceFeature(shape, tf.float32)
        for example_name, (name, shape) in self.action_like_names_and_shapes.items():
            sequence_features[name] = tf.io.FixedLenSequenceFeature(shape, tf.float32)

        _, sequence_features = tf.io.parse_single_sequence_example(
            serialized_example, sequence_features=sequence_features)

        state_like_seqs = OrderedDict()
        action_like_seqs = OrderedDict()
        for example_name, (name, shape) in self.state_like_names_and_shapes.items():
            state_like_seqs[example_name] = sequence_features[name]
        for example_name, (name, shape) in self.action_like_names_and_shapes.items():
            action_like_seqs[example_name] = sequence_features[name]

        # the sequence_length of this example is determined by the shortest sequence
        example_sequence_length = []
        for example_name, seq in state_like_seqs.items():
            example_sequence_length.append(tf.shape(seq)[0])
        for example_name, seq in action_like_seqs.items():
            example_sequence_length.append(tf.shape(seq)[0] + 1)
        example_sequence_length = tf.reduce_min(example_sequence_length)

        state_like_seqs, action_like_seqs = \
            self.slice_sequences(state_like_seqs, action_like_seqs, example_sequence_length)

        # decode and preprocess images on the sampled slice only
        _, image_shape = self.state_like_names_and_shapes['images']
        state_like_seqs['images'] = self.decode_and_preprocess_images(state_like_seqs['images'], image_shape)
        return state_like_seqs, action_like_seqs


class VarLenFeatureVideoDataset(BaseVideoDataset):
    """
    This class supports reading tfrecords where an entire sequence is stored as
    a single tf.train.Example.

    https://github.com/tensorflow/tensorflow/issues/15977
    """
    def filter(self, serialized_example):
        features = dict()
        features['sequence_length'] = tf.io.FixedLenFeature((), tf.int64)
        features = tf.io.parse_single_example(serialized_example, features=features)
        example_sequence_length = features['sequence_length']
        return tf.greater_equal(example_sequence_length, self.hparams.sequence_length)

    def parser(self, serialized_example):
        """
        Parses a single tf.train.SequenceExample into images, states, actions, etc tensors.
        """
        features = dict()
        features['sequence_length'] = tf.io.FixedLenFeature((), tf.int64)
        for example_name, (name, shape) in self.state_like_names_and_shapes.items():
            if example_name == 'images':
                features[name] = tf.io.VarLenFeature(tf.string)
            else:
                features[name] = tf.io.VarLenFeature(tf.float32)
        for example_name, (name, shape) in self.action_like_names_and_shapes.items():
            features[name] = tf.io.VarLenFeature(tf.float32)

        features = tf.io.parse_single_example(serialized_example, features=features)

        example_sequence_length = features['sequence_length']

        state_like_seqs = OrderedDict()
        action_like_seqs = OrderedDict()
        for example_name, (name, shape) in self.state_like_names_and_shapes.items():
            if example_name == 'images':
                seq = tf.io.parse_tensor_to_dense(features[name], '')
            else:
                seq = tf.io.sparse_tensor_to_dense(features[name])
                seq = tf.reshape(seq, [example_sequence_length] + list(shape))
            state_like_seqs[example_name] = seq
        for example_name, (name, shape) in self.action_like_names_and_shapes.items():
            seq = tf.io.sparse_tensor_to_dense(features[name])
            seq = tf.reshape(seq, [example_sequence_length - 1] + list(shape))
            action_like_seqs[example_name] = seq

        state_like_seqs, action_like_seqs = \
            self.slice_sequences(state_like_seqs, action_like_seqs, example_sequence_length)

        # decode and preprocess images on the sampled slice only
        _, image_shape = self.state_like_names_and_shapes['images']
        state_like_seqs['images'] = self.decode_and_preprocess_images(state_like_seqs['images'], image_shape)
        return state_like_seqs, action_like_seqs


if __name__ == '__main__':
    import cv2
    from video_prediction import datasets

    datasets = [
        datasets.SV2PVideoDataset('data/shape', mode='val'),
        datasets.SV2PVideoDataset('data/humans', mode='val'),
        datasets.SoftmotionVideoDataset('data/bair', mode='val'),
        datasets.KTHVideoDataset('data/kth', mode='val'),
        datasets.KTHVideoDataset('data/kth_128', mode='val'),
        datasets.UCF101VideoDataset('data/ucf101', mode='val'),
    ]
    batch_size = 4

    sess = tf.Session()

    for dataset in datasets:
        inputs = dataset.make_batch(batch_size)
        images = inputs['images']
        images = tf.reshape(images, [-1] + images.get_shape().as_list()[2:])
        images = sess.run(images)
        images = (images * 255).astype(np.uint8)
        for image in images:
            if image.shape[-1] == 1:
                image = np.tile(image, [1, 1, 3])
            else:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow(dataset.input_dir, image)
            cv2.waitKey(50)
