#!/usr/bin/env python

# Copyright 2017 Vertex.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import importlib
import json
import os
import platform
import sys

import cv2
import imageio
import numpy as np
import pygame
import scipy.misc

# for backwards compat with opencv 2.x
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4

CAP_WIDTH = 640
CAP_HEIGHT = 480

SUPPORTED_MODELS = {
    'inception_v3': {
        'shape': (299, 299, 3),
        'class': 'InceptionV3',
    },
    'mobilenet': {
        'shape': (224, 224, 3),
        'class': 'MobileNet',
    },
    'resnet50': {
        'shape': (224, 224, 3),
        'class': 'ResNet50',
    },
    'vgg16': {
        'shape': (224, 224, 3),
        'class': 'VGG16',
    },
    'vgg19': {
        'shape': (224, 224, 3),
        'class': 'VGG19',
    },
    'xception': {
        'shape': (299, 299, 3),
        'class': 'Xception',
    },
}


class Input:

    def __init__(self, path, stop):
        self.path = path
        self.count = 0
        self.stop = stop

    def open(self):
        if self.path:
            self.cap = cv2.VideoCapture(self.path)
        else:
            self.cap = cv2.VideoCapture(0)
        self.cap.set(CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
        self.cap.set(CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

    def poll(self):
        if self.stop and self.count == self.stop:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.count += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        return frame

    def close(self):
        self.cap.release()


class Compositor:

    def __init__(self):
        pygame.font.init()
        self._font = pygame.font.SysFont("monospace", 14, bold=True)
        self._tgt_size = (CAP_WIDTH, CAP_HEIGHT)
        self._tgt = pygame.Surface(self._tgt_size)

    def process(self, frame, predictions, clock):
        self._tgt.fill((0, 0, 0))
        # Convert the image into a surface object we can blit to the pygame window.
        surface = pygame.surfarray.make_surface(frame)
        # Fit the image proportionally into the window.
        (tgt_width, tgt_height) = self._tgt_size
        (img_width, img_height, _) = frame.shape
        result_size = (img_width, img_height)
        # If the image is larger than the window, in both dimensions, scale it
        # down proportionally so that it entirely fills the window.
        if img_width > tgt_width and img_height > tgt_height:
            hscale = float(tgt_width) / float(img_width)
            vscale = float(tgt_height) / float(img_height)
            if hscale > vscale:
                result_size = (tgt_width, int(img_height * hscale))
            else:
                result_size = (int(img_width * vscale), tgt_height)
        # Center the image in the tgt, cropping if necessary.
        hoff = (tgt_width - result_size[0]) / 2
        voff = (tgt_height - result_size[1]) / 2
        surface = pygame.transform.scale(surface, result_size)
        self._tgt.blit(surface, (hoff, voff))
        # Print some text explaining what we think the image contains, using some
        # contrasting colors for a little drop-shadow effect.
        captions = [self.make_caption(x) for x in predictions]
        for (i, caption) in enumerate(captions):
            self.blit_prediction(i, caption)
        # Print the FPS
        fps_text = 'FPS: {:3.1f}'.format(clock.get_fps())
        self.blit_text(fps_text, (8, self._tgt.get_height() - 24))
        return self._tgt

    def make_caption(self, prediction):
        (label_id, label_name, confidence) = prediction
        return label_name + " ({0:.0f}%)".format(confidence * 100.0)

    def blit_prediction(self, i, caption):
        self.blit_text(caption, (8, 18 * i))

    def blit_text(self, text, pos):
        self.blit_text_part(text, pos, -1, (110, 110, 240))
        self.blit_text_part(text, pos, 2, (0, 0, 100))
        self.blit_text_part(text, pos, 1, (100, 100, 255))
        self.blit_text_part(text, pos, 0, (240, 240, 110))

    def blit_text_part(self, caption, pos, offset, color):
        label = self._font.render(caption, 1, color)
        label_pos = (pos[0] + offset, pos[1] + offset)
        self._tgt.blit(label, label_pos)


class OutputScreen:

    def __init__(self):
        self._screen_size = (CAP_WIDTH, CAP_HEIGHT)
        pygame.display.init()
        self._screen = pygame.display.set_mode(self._screen_size)
        pygame.display.set_caption("Plaidvision")

    def close(self):
        pass

    def process(self, surface):
        surface = surface.convert(self._screen)
        self._screen.blit(surface, self._screen.get_rect())
        pygame.display.flip()


class OutputFile:

    def __init__(self, path):
        if platform.machine() != 'armv7l':
            imageio.plugins.ffmpeg.download()
        self.writer = imageio.get_writer(path, fps=30)

    def close(self):
        self.writer.close()

    def process(self, surface):
        surface = pygame.transform.flip(surface, True, False)
        surface = pygame.transform.rotate(surface, 90)
        frame = pygame.surfarray.array3d(surface)
        self.writer.append_data(frame)


class Model:

    def __init__(self, name, weights):
        info = SUPPORTED_MODELS.get(name)
        self.shape = info.get('shape')

        module = importlib.import_module('.'.join(['keras', 'applications', name]))
        ModelClass = getattr(module, info['class'])
        self.preprocess_input = getattr(module, 'preprocess_input')
        self.decode_predictions = getattr(module, 'decode_predictions')

        # Some models think they only work with TensorFlow, but the truth is that it
        # won't work with Theano or CNTK, and it doesn't know that PlaidML exists.
        # It'll work just fine with PlaidML as long as we pretend to be tensorflow
        # by monkeypatching the backend() function during model initialization.
        import keras.backend as K
        old_backend = K.backend
        K.backend = lambda: "tensorflow"
        self.model = ModelClass(weights=weights)
        K.backend = old_backend

    def classify(self, img, top_n=5):
        if img.shape != self.shape:
            img = scipy.misc.imresize(img, self.shape).astype(float)
        data = np.expand_dims(img, axis=0)
        data = self.preprocess_input(data)
        predictions = self.model.predict(data)
        return self.decode_predictions(predictions, top=top_n)[0]


def loop(headless):
    if headless:
        return True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
    return True


def has_plaid():
    try:
        import plaidml.keras
        return True
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser()
    backend_args = parser.add_mutually_exclusive_group()
    backend_args.add_argument('--plaid', action='store_true')
    backend_args.add_argument('--no-plaid', action='store_true')
    parser.add_argument('-v', '--verbose', type=int, nargs='?', const=3)
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--json')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--weights', default='imagenet')
    parser.add_argument('--frames', type=int)
    parser.add_argument('model', choices=list(SUPPORTED_MODELS))
    args = parser.parse_args()

    if args.plaid or (not args.no_plaid and has_plaid()):
        print("Using PlaidML backend.")
        import plaidml.keras
        if args.verbose:
            plaidml._internal_set_vlog(args.verbose)
        plaidml.keras.install_backend()

    clock = pygame.time.Clock()

    input = Input(args.input, args.frames)
    model = Model(args.model, args.weights)

    outputs = []
    if not args.headless:
        outputs.append(OutputScreen())
    if args.output:
        outputs.append(OutputFile(args.output))
    compositor = Compositor()
    json_output = dict(results=[])

    inference_clock = pygame.time.Clock()
    try:
        input.open()
        while loop(args.headless):
            clock.tick()
            frame = input.poll()
            if frame is None:
                break
            inference_clock.tick()
            predictions = model.classify(frame)
            inference_clock.tick()
            record = dict(
                elapsed=inference_clock.get_time(),
                predictions=[
                    dict(
                        label_id=x[0],
                        label_name=x[1],
                        confidence=float(x[2]),
                    ) for x in predictions
                ],
            )
            json_output['results'].append(record)
            surface = compositor.process(frame, predictions, clock)
            for output in outputs:
                output.process(surface)
    except KeyboardInterrupt:
        pass
    except Exception as ex:
        json_output['exception'] = ex
        raise
    finally:
        for output in outputs:
            output.close()
        input.close()
        if args.json:
            with open(args.json, 'w') as file_:
                json.dump(json_output, file_)


if __name__ == "__main__":
    main()
