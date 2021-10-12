#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import sys
import os

from object_detector_retinanet.keras_retinanet import models

md='02'
def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for converting a training model to an inference model.')

    parser.add_argument('--model_in', help='The model to convert.',default='../data/eu/snapshots/resnet101_csv_'+md+'.h5')
    parser.add_argument('--model_out', help='Path to save the converted model to.',default='../data/eu/model/101_'+md+'.h5')
    parser.add_argument('--backbone', help='The backbone of the model to convert.', default='resnet101')
    parser.add_argument('--no-nms', help='Disables non maximum suppression.', dest='nms', action='store_false')
    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # load and convert model
    model = models.load_model(args.model_in, convert=True, backbone_name=args.backbone, nms=args.nms)

    # save model
    model.save(args.model_out)


if __name__ == '__main__':
    main()
