import subprocess
import os
from utils import log
logger = log.get_logger(__name__)


def decode_jpg(src_path, dst_folder, quiet=True):
    if quiet:
        cmd = '/usr/bin/ffmpeg -loglevel quiet -i {} -q:v 0 -f image2  {}/%06d.jpg'.format(
            src_path, dst_folder)
    else:
        cmd = '/usr/bin/ffmpeg -i {} -q:v 0 -f image2  {}/%06d.jpg'.format(
            src_path, dst_folder)
    subprocess.run(cmd, shell=True, check=True)


def decode_bmp(src_path, dst_folder, quiet=True):
    if quiet:
        cmd = '/usr/bin/ffmpeg -loglevel quiet -i {} -q:v 0 {}/%06d.bmp'.format(
            src_path, dst_folder)
    else:
        cmd = '/usr/bin/ffmpeg -i {} -q:v 0 {}/%06d.bmp'.format(
            src_path, dst_folder)
    subprocess.run(cmd, shell=True, check=True)


def encode_jpg(src_folder, dst_path, quantitizer, framerate, startnumber):
    cmd = '''/usr/bin/ffmpeg -y -f image2  -loglevel quiet -start_number {}  
            -framerate {} -pattern_type glob -i "{}/*.jpg"
            -vcodec libx264 -crf {} {}'''.format(startnumber, framerate, src_folder, quantitizer, dst_path) \
        .replace('\n', '')
    subprocess.run(cmd, shell=True)


def encode_bmp(src_folder, dst_path, quantitizer, framerate, startnumber):
    cmd = '''/usr/bin/ffmpeg -y -loglevel quiet -start_number {}  
            -framerate {} -pattern_type glob -i "{}/*.bmp"
            -vcodec libx264 -crf {} {}'''.format(startnumber, framerate, src_folder, quantitizer, dst_path) \
        .replace('\n', '')
    subprocess.run(cmd, shell=True)


def encode_jpg_gpu(src_folder, dst_path, quantitizer, framerate, startnumber):
    cmd = '''~/Downloads/ffmpeg/ffmpeg -y -loglevel quiet -f image2 -start_number {}  
            -framerate {} -pattern_type glob -i "{}/*.jpg"
            -vcodec libx264 -crf {} {}'''.format(startnumber, framerate, src_folder, quantitizer, dst_path) \
        .replace('\n', '')
    subprocess.run(cmd, shell=True, check=True)


def get_size(path):
    # return bytes
    cmd = 'du -b {}'.format(path)
    return int(subprocess.check_output(cmd, shell=True, universal_newlines=True).split()[0])


if __name__ == "__main__":
    # decode_jpg('/home/shen/Downloads/darknet/video/AR\ proj\ train.mov',
    #           '/home/shen/Downloads/darknet/video/imgs/', False)
    print(get_size('/home/shen/Downloads/slice/000000.csv')/1e6 * 8)
