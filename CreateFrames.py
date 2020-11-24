import cv2
from pytube import YouTube
import os
import datetime
from PIL import Image
import math
from torchvision import transforms


# from https://towardsdatascience.com/the-easiest-way-to-download-youtube-videos-using-python-2640958318ab
# Convert video to images
class FrameExtractor:

    def __init__(self, video_path):
        self.video_path = video_path
        self.vid_cap = cv2.VideoCapture(video_path)
        self.n_frames = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))

    def get_video_duration(self):
        duration = self.n_frames / self.fps
        print(f'Duration: {datetime.timedelta(seconds=duration)}')

    def get_n_images(self, every_x_frame):
        n_images = math.floor(self.n_frames / every_x_frame) + 1
        print(f'Extracting every {every_x_frame} (nd/rd/th) frame would result in {n_images} images.')

    def extract_frames(self, every_x_frame, img_name, dest_path=None, img_ext='.jpg', transform=transforms.Pad(0)):
        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)

        if dest_path is None:
            dest_path = os.getcwd()
        else:
            if not os.path.isdir(dest_path):
                os.mkdir(dest_path)
                print(f'Created the following directory: {dest_path}')

        frame_cnt = 0
        img_cnt = 0

        print(f'Saving images to {dest_path}')

        while self.vid_cap.isOpened():

            success, image = self.vid_cap.read()

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if not success:
                break

            if frame_cnt % every_x_frame == 0:
                img_path = os.path.join(dest_path, ''.join([img_name, '_', str(img_cnt), img_ext]))
                transform(Image.fromarray(image)).save(img_path)
                # cv2.imwrite(img_path, image)
                img_cnt += 1

            if frame_cnt % (self.n_frames // 20) == 0:
                print(f'{frame_cnt / (self.n_frames / 20) * 5}% complete')

            frame_cnt += 1

        self.vid_cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # If downloading from youtube
    videoURL = None #"https://www.youtube.com/watch?v=lB0gmneMZxg"
    save_dir = './images/videos'
    downloaded_video_name = 'video1'
    video_file_type = "mp4"
    video_resolution = "720p"
    every_n_frames = 1

    # If using a video file
    videoFile = "./images/videos/video1.mp4"

    if videoFile is None and videoURL is None:
        raise Exception('Both videoFile and videoURL are None')

    # Download Youtube video
    if videoFile is None and videoURL is not None:
        video = YouTube(videoURL)
        print("--------Download Video---------")
        print(f'Title: {video.title}')
        print(f'Views: {video.views}')
        print(f'Rating: {video.rating}')
        print(f'Author: {video.author}')
        print(f'Length: {video.length}')
        print(f'Captions: {video.captions}')

        print('-----Streams-----')
        for s in video.streams.filter(file_extension=video_file_type, resolution=video_resolution):
            print(s)
        print("-----------------")
        itag = int(input('Enter the itag of the stream to download: '))

        print("Downloading video...")
        video.streams.get_by_itag(itag).download(save_dir, downloaded_video_name)
        print("Download complete!")
        videoFile = f"{save_dir}/{downloaded_video_name}.{video_file_type}"

    # Extract and write out frames
    fe = FrameExtractor(f'{save_dir}/{downloaded_video_name}.{video_file_type}' if videoURL is not None else videoFile)
    fe.get_n_images(every_n_frames)
    fe.extract_frames(every_n_frames, img_name=downloaded_video_name, dest_path=f"{save_dir}/{downloaded_video_name}", img_ext='.png')
    print("Done!")


# vidcap = cv2.VideoCapture(videoFile)
# def getFrame(sec):
#     vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
#     hasFrames, image = vidcap.read()
#     if hasFrames:
#         cv2.imwrite("./images/video_input/image"+str(count)+".jpg", image)     # save frame as JPG file
#     else:
#         print("End of video")
#     return hasFrames
#
# sec = 0
# frameRate = 1/30
# count=1
# success = getFrame(sec)
# while success:
#     count = count + 1
#     print(f"Wrote frame {count}")
#     sec = sec + frameRate
#     sec = round(sec, 2)
#     success = getFrame(sec)

