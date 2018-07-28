# encoding: utf-8

import apache_beam as beam
import argparse
import io
import os
import tempfile
import tensorflow as tf
import csv
from tensorflow.python.lib.io import file_io
from tensorflow.python.framework import errors
from apache_beam.options.pipeline_options import PipelineOptions


class ReadImage(beam.DoFn):

    def __init__(self, image_folder, mask_folder):
        super(ReadImage, self).__init__()
        self.image_folder = image_folder
        self.mask_folder = mask_folder

    def process(self, element):

        def _open_file_read_binary(uri):
            try:
                return file_io.FileIO(uri, mode='rb')
            except errors.InvalidArgumentError:
                return file_io.FileIO(uri, mode='r')

        image_path = os.path.join(self.image_folder,element)
        mask_path = os.path.join(self.mask_folder,element)    

        image_bytes = _open_file_read_binary(image_path).read()
        mask_bytes = _open_file_read_binary(mask_path).read()

        yield image_bytes, mask_bytes


class TFExampleFromImageDoFn(beam.DoFn):

    def __init__(self):
        super(TFExampleFromImageDoFn, self).__init__()

    def process(self, element):

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        image_bytes, mask_bytes = element

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_bytes': _bytes_feature([image_bytes]),
            'mask_bytes': _bytes_feature([mask_bytes])
        }))
        yield example


def run(_pipeline_args, _know_args):

    pipeline_options = PipelineOptions(pipeline_args)
    read_input_list = beam.io.ReadFromText(_know_args.image_list, strip_trailing_newlines=True)

    output = '%s/%s' % (_know_args.output_path, _know_args.phase)
    with beam.Pipeline(options=pipeline_options) as p:
        init = p | 'Start pipeline' >> beam.Create([''])
        (init |
            'read_image_list' >> read_input_list |
            'parse_list' >> beam.Map(lambda l: csv.reader([l]).next()[0]) |
            'read_images' >> beam.ParDo(ReadImage(_know_args.image_folder,_know_args.mask_folder)) |
            'build_tfexample' >> beam.ParDo(TFExampleFromImageDoFn()) |
            'serialize_tfexample' >> beam.Map(lambda x: x.SerializeToString()) |
            'write_tfrecord' >> beam.io.WriteToTFRecord(output, file_name_suffix='.tfrecord.gz'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-folder', dest='image_folder', required=True)
    parser.add_argument('--mask-folder', dest='mask_folder', required=True)
    parser.add_argument('--image-list', dest='image_list', required=True)
    parser.add_argument('--output-path', dest='output_path', required=True)
    parser.add_argument('--phase', dest='phase', required=True)
    know_args, pipeline_args = parser.parse_known_args()
    run(_pipeline_args=pipeline_args, _know_args=know_args)
