import ffmpeg
import torch
import cv2
from transformers import CLIPProcessor, CLIPModel
from transformers import BridgeTowerProcessor, BridgeTowerForImageAndTextRetrieval
import time
from pytube import YouTube
import os
from PIL import Image

COUNTER = 1 #counter for video results
DOWNLOAD_PATH = "./website/videos"
RESULT_PATH = "./website/static/results"
IS_PROCESSING = False
os.makedirs(RESULT_PATH,exist_ok=True)

def trim(input, output, start, end):
    # if end - start > 2:
    time = end-start
    ffmpeg.input(input, ss=start).output(output, t=time).run()
    global COUNTER
    COUNTER += 1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#BASE
# model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# processor_bt = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
# model_bt = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-base-itm-mlm").to(device)

#LARGE
model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

processor_bt = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-gaudi")
model_bt = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-gaudi").to(device)


def clip_process_frame(frame,prompt):

    with torch.no_grad():
        inputs = processor_clip(text=[prompt,"an object"], images=frame,
                           return_tensors="pt", padding=True).to(device)

        outputs = model_clip(**inputs)
        # this is the image-text similarity score
        logits_per_image = outputs.logits_per_image
        # we can take the softmax to get the label probabilities
        # probs = logits_per_image.numpy()
        probs = logits_per_image.softmax(dim=1).cpu().numpy()  # we can take the softmax to get the label probabilities
    return probs[0][0]

def bt_process_frame(frame,prompt):
    scores = dict()
    # prepare inputs
    encoding = processor_bt(frame, prompt, return_tensors="pt").to(device)
    outputs = model_bt(**encoding)
    scores[prompt] = outputs.logits[0, 1].item()
    # encoding = processor_bt(frame, "an object", return_tensors="pt").to(device)
    # outputs = model_bt(**encoding)
    return scores


# function that take as input a video, cuts the video into frames based on user input
def process_video(video,prompt,n):
    capture = cv2.VideoCapture(video)
    fps = capture.get(cv2.CAP_PROP_FPS) #how many fps is the video
    print(fps)
    frameNr = 1
    i = 0 #COUNTER for frame number
    timestamp_begin = 0
    timestamp_end = 0
    first_timestamp_found = False
    forgiving_frames = 0
    valid_frames = 0

    if n > fps:
        n = fps
    start_time = time.time()
    global IS_PROCESSING
    IS_PROCESSING = True
    while (IS_PROCESSING):
        success, frame = capture.read()
        if success:
            if (i % (fps//n) == 0): #to check if the current frame should be proccessed or not(according to the input n)
                # cv2.imwrite(f'tests/divideVideo/frames/frame_{frameNr}.jpg', frame)
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                score_clip = clip_process_frame(frame,prompt)
                print("clip:", score_clip)
                if score_clip >= 0.9 and first_timestamp_found == False:
                    score_bt = bt_process_frame(frame,prompt)
                    print("bt:", score_bt[prompt])
                    # cv2.imwrite(f'{RESULT_PATH}/frame_{score_bt[prompt]}.jpg', frame)
                    if score_bt[prompt] >= 2:
                        first_timestamp_found = True
                        timestamp_begin = i/fps
                        forgiving_frames = 0
                        valid_frames += 1
                elif score_clip >= 0.9 and first_timestamp_found == True:
                    score_bt = bt_process_frame(frame,prompt)
                    print("bt:", score_bt[prompt])
                    # cv2.imwrite(f'{RESULT_PATH}/frame_{score_bt[prompt]}.jpg', frame)
                    if score_bt[prompt] < 2:
                        if forgiving_frames <= n:
                            forgiving_frames += 1
                        else:
                            if valid_frames >= n:
                                timestamp_end = (i-1)/fps
                                first_timestamp_found = False
                                trim(
                                    video, f"{RESULT_PATH}/video{COUNTER}.mp4", timestamp_begin, timestamp_end)
                                forgiving_frames = 0
                                valid_frames = 0
                            else:
                                forgiving_frames = 0
                                valid_frames = 0
                                first_timestamp_found = False

                    else:
                        valid_frames += 1
                elif score_clip < 0.9 and first_timestamp_found == True:
                    if forgiving_frames <= n:
                        forgiving_frames += 1
                    else:
                        if valid_frames >= n:
                            timestamp_end = (i-1)/fps
                            first_timestamp_found = False
                            trim(
                                video, f"{RESULT_PATH}/video{COUNTER}.mp4", timestamp_begin, timestamp_end)
                            forgiving_frames = 0
                            valid_frames = 0
                        else:
                            forgiving_frames = 0
                            valid_frames = 0
                            first_timestamp_found = False
                frameNr = frameNr+1
        else:
            break

        i += 1
    if first_timestamp_found == True:
        timestamp_end = (i-1)/fps
        trim(video, f"{RESULT_PATH}/video{COUNTER}.mp4",
                timestamp_begin, timestamp_end)

    capture.release()
    elapsed_time = time.time() - start_time
    print(elapsed_time)


def downloader(links):
    video_names = []
    for link in links:
        try:
            # Object creation using YouTube
            # which was imported in the beginning
            yt = YouTube(link)
            if os.path.exists(os.path.join(DOWNLOAD_PATH,yt.video_id+".mp4")):
                print("Video already exist")
                video_names.append(yt.video_id+".mp4")
                continue
            print("video title:" + yt.title)
        except:
            # Handle exception
            print("Connection Error")

        # Get all streams and filter for mp4 files
        mp4_streams = yt.streams.filter(file_extension='mp4')

        # Get the video with the highest resolution
        d_video = mp4_streams.get_highest_resolution()
        try:
            # Download the video
            d_video.download(output_path=DOWNLOAD_PATH, filename=yt.video_id+".mp4")
            video_names.append(yt.video_id+".mp4")
            print('Video downloaded successfully!')
        except:
            print("Some Error!")

    print('Task Completed!')
    return video_names

def process_links(links,prompt,fps):
    for video in os.listdir(RESULT_PATH):
        os.remove(os.path.join(RESULT_PATH, video))
    names = downloader(links)
    print(names)
    for name in names:
        process_video(f"{DOWNLOAD_PATH}/{name}",prompt,fps)

def process_custom_videos(names,prompt,fps):
    for video in os.listdir(RESULT_PATH):
        os.remove(os.path.join(RESULT_PATH, video))
    print(names)
    for name in names:
        process_video(f"{DOWNLOAD_PATH}/{name}",prompt,fps)

def stop_processing():
    global IS_PROCESSING
    IS_PROCESSING = False

#"https://youtu.be/AJWpvoXP5d4?si=CdJXMBZKmrG6A3Yc"