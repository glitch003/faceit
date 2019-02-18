import os
import subprocess
from argparse import Namespace
import argparse
import youtube_dl
import cv2
import time
import tqdm
import numpy
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.video.fx.all import crop
from moviepy.editor import AudioFileClip, clips_array, TextClip, CompositeVideoClip
import shutil
from pathlib import Path
import sys
sys.path.append('faceswap')

import lib.cli as cli
from scripts.train import Train
from scripts.convert import Convert

class FaceIt:
    VIDEO_PATH = 'data/videos'
    PERSON_PATH = 'data/persons'
    PROCESSED_PATH = 'data/processed'
    OUTPUT_PATH = 'data/output'
    MODEL_PATH = 'models'
    MODELS = {}

    @classmethod
    def add_model(cls, model):
        FaceIt.MODELS[model._name] = model
    
    def __init__(self, name, person_a, person_b):
        def _create_person_data(person):
            return {
                'name' : person,
                'videos' : [],
                'faces' : os.path.join(FaceIt.PERSON_PATH, person + '.jpg'),
                'photos' : []
            }
        
        self._name = name

        self._people = {
            person_a : _create_person_data(person_a),
            person_b : _create_person_data(person_b),
        }
        self._person_a = person_a
        self._person_b = person_b
        
        self._faceswap = FaceSwapInterface()

        if not os.path.exists(os.path.join(FaceIt.VIDEO_PATH)):
            os.makedirs(FaceIt.VIDEO_PATH)            

    def add_photos(self, person, photo_dir):
        self._people[person]['photos'].append(photo_dir)
            
    def add_video(self, person, name, url=None, fps=20):
        self._people[person]['videos'].append({
            'name' : name,
            'url' : url,
            'fps' : fps
        })

    def fetch(self):
        self._process_media(self._fetch_video)

    def extract_frames(self):
        self._process_media(self._extract_frames)

    def extract_faces(self):        
        self._process_media(self._extract_faces)
        self._process_media(self._extract_faces_from_photos, 'photos')        

    def all_videos(self):
        return self._people[self._person_a]['videos'] + self._people[self._person_b]['videos']

    def _process_media(self, func, media_type = 'videos'):
        for person in self._people:
            for video in self._people[person][media_type]:
                func(person, video)

    def _video_path(self, video):
        return os.path.join(FaceIt.VIDEO_PATH, video['name'])        

    def _video_frames_path(self, video):
        return os.path.join(FaceIt.PROCESSED_PATH, video['name'] + '_frames')        

    def _video_faces_path(self, video):
        return os.path.join(FaceIt.PROCESSED_PATH, video['name'] + '_faces')

    def _model_path(self, use_gan = False):
        path = FaceIt.MODEL_PATH
        if use_gan:
            path += "_gan"
        return os.path.join(path, self._name)

    def _model_data_path(self):
        return os.path.join(FaceIt.PROCESSED_PATH, "model_data_" + self._name)
    
    def _model_person_data_path(self, person):
        return os.path.join(self._model_data_path(), person)

    def _fetch_video(self, person, video):
        options = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio',
            'outtmpl': os.path.join(FaceIt.VIDEO_PATH, video['name']),
            'merge_output_format' : 'mp4'
        }
        with youtube_dl.YoutubeDL(options) as ydl:
            x = ydl.download([video['url']])

    def _extract_frames(self, person, video):
        video_frames_dir = self._video_frames_path(video)
        video_clip = VideoFileClip(self._video_path(video))
        
        start_time = time.time()
        print('[extract-frames] about to extract_frames for {}, fps {}, length {}s'.format(video_frames_dir, video_clip.fps, video_clip.duration))
        
        if os.path.exists(video_frames_dir):
            print('[extract-frames] frames already exist, skipping extraction: {}'.format(video_frames_dir))
            return
        
        os.makedirs(video_frames_dir)
        frame_num = 0
        for frame in tqdm.tqdm(video_clip.iter_frames(fps=video['fps']), total = video_clip.fps * video_clip.duration):
            video_frame_file = os.path.join(video_frames_dir, 'frame_{:03d}.jpg'.format(frame_num))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Swap RGB to BGR to work with OpenCV
            cv2.imwrite(video_frame_file, frame)
            frame_num += 1

        print('[extract] finished extract_frames for {}, total frames {}, time taken {:.0f}s'.format(
            video_frames_dir, frame_num-1, time.time() - start_time))            

    def _extract_faces(self, person, video):
        video_faces_dir = self._video_faces_path(video)

        start_time = time.time()
        print('[extract-faces] about to extract faces for {}'.format(video_faces_dir))
        
        if os.path.exists(video_faces_dir):
            print('[extract-faces] faces already exist, skipping face extraction: {}'.format(video_faces_dir))
            return
        
        os.makedirs(video_faces_dir)
        self._faceswap.extract(self._video_frames_path(video), video_faces_dir, self._people[person]['faces'])

    def _extract_faces_from_photos(self, person, photo_dir):
        photo_faces_dir = self._video_faces_path({ 'name' : photo_dir })

        start_time = time.time()
        print('[extract-faces] about to extract faces for {}'.format(photo_faces_dir))
        
        if os.path.exists(photo_faces_dir):
            print('[extract-faces] faces already exist, skipping face extraction: {}'.format(photo_faces_dir))
            return
        
        os.makedirs(photo_faces_dir)
        self._faceswap.extract(self._video_path({ 'name' : photo_dir }), photo_faces_dir, self._people[person]['faces'])


    def preprocess(self):
        self.fetch()
        self.extract_frames()
        self.extract_faces()
    
    def _symlink_faces_for_model(self, person, video):
        if isinstance(video, str):
            video = { 'name' : video }
        for face_file in os.listdir(self._video_faces_path(video)):
            target_file = os.path.join(self._model_person_data_path(person), video['name'] + "_" + face_file)
            face_file_path = os.path.join(os.getcwd(), self._video_faces_path(video), face_file)
            os.symlink(face_file_path, target_file)

    def train(self, use_gan = False):
        # Setup directory structure for model, and create one director for person_a faces, and
        # another for person_b faces containing symlinks to all faces.
        if not os.path.exists(self._model_path(use_gan)):
            os.makedirs(self._model_path(use_gan))

        if os.path.exists(self._model_data_path()):
            shutil.rmtree(self._model_data_path())

        for person in self._people:
            os.makedirs(self._model_person_data_path(person))
        self._process_media(self._symlink_faces_for_model)

        self._faceswap.train(self._model_person_data_path(self._person_a), self._model_person_data_path(self._person_b), self._model_path(use_gan), use_gan)

    def convert(self, video_file, swap_model = False, duration = None, start_time = None, use_gan = False, face_filter = False, photos = True, crop_x = None, width = None, side_by_side = False):
        print('converting')
	

class FaceSwapInterface:
    def __init__(self):
        print("initing faceswapinterface")

    def bad_args(args):
        """ Print help on bad arguments """
        PARSER.print_help()
        exit(0)

    def extract(self, input_dir, output_dir, filter_path):
        args_str = "python3 faceswap/faceswap.py extract --input-dir {} --output-dir {} --detector mtcnn -mp -A fan --filter {}"
        args_str = args_str.format(input_dir, output_dir, filter_path)
        print("args str: {}".format(args_str))
        args_array = args_str.split(" ")
        print("args array: {}".format(args_array))
        for line in self._execute(args_array):
            print(line, end="")

    def convert(self, input_dir, output_dir, model_dir, filter_path):
        args_str = "convert -i {} -o {} -m {} -b 4 -c Masked -S -M facehullandrect -g 4 -e 2 -t OriginalHighRes"
        args_str = args_str.format(input_dir, output_dir, filter_path)
        print("args str: {}".format(args_str))
        args_array = args_str.split(" ")
        print("args array: {}".format(args_array))
        for line in self._execute(args_array):
            print(line, end="")


    def train(self, input_a_dir, input_b_dir, model_dir):
        args_str = "train --input-A {} --input-B {} --model-dir {} --trainer OriginalHighRes -g 4 --batch-size 512"
        args_str = args_str.format(input_a_dir, input_b_dir, model_dir)
        print("args str: {}".format(args_str))
        args_array = args_str.split(" ")
        print("args array: {}".format(args_array))
        for line in self._execute(args_array):
            print(line, end="")

    def _execute(self, cmd):
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
        for stdout_line in iter(popen.stdout.readline, ""):
            yield stdout_line 
        popen.stdout.close()
        return_code = popen.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, cmd)

    def _run_script(self, args_str):
        PARSER = cli.FullHelpArgumentParser()
        SUBPARSER = PARSER.add_subparsers()
        EXTRACT = cli.ExtractArgs(SUBPARSER,
                                  "extract",
                                  "Extract the faces from pictures")
        TRAIN = cli.TrainArgs(SUBPARSER,
                              "train",
                              "This command trains the model for the two faces A and B")
        CONVERT = cli.ConvertArgs(SUBPARSER,
                                  "convert",
                                  "Convert a source image to a new one with the face swapped")
        GUI = cli.GuiArgs(SUBPARSER,
                          "gui",
                          "Launch the Faceswap Graphical User Interface")
        PARSER.set_defaults(func=self.bad_args)
        ARGUMENTS = PARSER.parse_args(args_str.split(' '))
        ARGUMENTS.func(ARGUMENTS)



if __name__ == '__main__':

    faceit = FaceIt('trump_to_adler', 'trump', 'adler')

    faceit.add_video('trump', 'trump_vs_media.mp4', 'https://www.youtube.com/watch?v=jtl5XK7QP38')



    faceit.add_video('adler', 'adler_fc.mp4', 'https://www.youtube.com/watch?v=kos21rHIOgQ')

    FaceIt.add_model(faceit)

    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices = ['preprocess', 'train', 'convert'])
    parser.add_argument('model', choices = FaceIt.MODELS.keys())
    parser.add_argument('video', nargs = '?')
    parser.add_argument('--duration', type = int, default = None)
    parser.add_argument('--photos', action = 'store_true', default = False)    
    parser.add_argument('--swap-model', action = 'store_true', default = False)
    parser.add_argument('--face-filter', action = 'store_true', default = False)
    parser.add_argument('--start-time', type = int, default = 0)
    parser.add_argument('--crop-x', type = int, default = None)
    parser.add_argument('--width', type = int, default = None)
    parser.add_argument('--side-by-side', action = 'store_true', default = False)    
    args = parser.parse_args()

    faceit = FaceIt.MODELS[args.model]
    
    if args.task == 'preprocess':
        faceit.preprocess()
    elif args.task == 'train':
        faceit.train()
    elif args.task == 'convert':
        if not args.video:
            print('Need a video to convert. Some ideas: {}'.format(", ".join([video['name'] for video in faceit.all_videos()])))
        else:
            faceit.convert(args.video, duration = args.duration, swap_model = args.swap_model, face_filter = args.face_filter, start_time = args.start_time, photos = args.photos, crop_x = args.crop_x, width = args.width, side_by_side = args.side_by_side)


