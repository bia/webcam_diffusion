import cv2
import numpy as np
from PIL import Image, ImageFilter

import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from cvzone.SelfiSegmentationModule import SelfiSegmentation

# from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt

PROMPT = "red and orange tropical flowers bloom, photorealistic"  # ok
# PROMPT = "A peaceful Zen garden at dawn, with perfectly raked sand, blooming cherry blossoms, and a gently flowing stream."
# PROMPT = "large red and orange tropical flowers surrounded by smaller petals, photorealistic"  # ok
PROMPT = "Michael Jackson"

WIDTH, HEIGHT = 1280, 720
TORCH_DEVICE = "mps"  # "cuda" if you have a cuda GPU
TORCH_DTYPE = torch.float16

# creating segmentation instance for taking the foreground (the person).
segmentor = SelfiSegmentation()


model_loc = "models/LCM_Dreamshaper_v7"
pipe = StableDiffusionPipeline.from_pretrained(model_loc).to(
    device=torch.device(TORCH_DEVICE),
    dtype=TORCH_DTYPE,
)

stream = StreamDiffusion(
    pipe,
    t_index_list=[32, 45],
    torch_dtype=TORCH_DTYPE,
    do_add_noise=False,
)

# If the loaded model is not LCM, merge LCM
stream.load_lcm_lora()
stream.fuse_lora()

# Use Tiny VAE for further acceleration
stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
    device=pipe.device, dtype=pipe.dtype
)

# Enable acceleration (FOR THOSE WITH CUDA GPU)
# pipe.enable_xformers_memory_efficient_attention()

# Prepare the stream
stream.prepare(prompt=PROMPT, num_inference_steps=50, guidance_scale=0)

# optional
stream.enable_similar_image_filter(
    # similar_image_filter_threshold,
    # similar_image_filter_max_skip_frame
)

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

CAP_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 320
CAP_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 240

# loading and resizing the background image
background_image = cv2.imread("bianco.jpg")
if background_image is None:
    print("Wrong path:")


cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH / 2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT / 2)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Read a frame from the webcam (for warmup)
ret, image = cap.read()
center_x = (image.shape[1] - WIDTH) // 2
center_y = (image.shape[0] - HEIGHT) // 2
result_image, image_cutout = image, image

# Warmup >= len(t_index_list) x frame_buffer_size
for _ in range(4):
    stream(image_cutout)

print("here.")

# Run the stream infinitely
while True:

    # Read frame (image) from the webcam
    ret, frame = cap.read()

    # Break the loop if reading the frame fails
    if not ret:
        print("Error: Failed to capture frame.")
        break

    background_image = cv2.resize(
        background_image,
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )

    # segmenting the image
    segmentated_img = segmentor.removeBG(frame, background_image, 0.9)

    # get center
    center_x = (frame.shape[1] - WIDTH) // 2
    center_y = (frame.shape[0] - HEIGHT) // 2

    result_image, result_cutout = frame, segmentated_img
    result_cutout = Image.fromarray(cv2.cvtColor(result_cutout, cv2.COLOR_BGR2RGB))

    x_output = stream(result_cutout)
    rendered_image = postprocess_image(x_output, output_type="pil")[0]  # .show()

    # add cutout to video output
    # result_image[center_y : center_y + HEIGHT, center_x : center_x + WIDTH] = (
    #     cv2.cvtColor(np.array(rendered_image), cv2.COLOR_RGB2BGR)
    # )

    rendered_image = cv2.resize(np.array(rendered_image), (WIDTH, HEIGHT))

    result_image[0 : frame.shape[0], 0 : frame.shape[1]] = cv2.cvtColor(
        rendered_image, cv2.COLOR_RGB2BGR
    )

    # # overlay red rectangle
    # red_rectangle = cv2.rectangle(
    #     result_image, (0, 0), (720, 720), (0, 0, 255), cv2.FILLED
    # )  # (0, 0, 255) is red in BGR
    # red_rectangle = cv2.resize(result_image, (1280, 720))
    # alpha = 0.5
    # result_image_overlay = cv2.addWeighted(
    #     result_image, alpha, red_rectangle, 1 - alpha, 0
    # )
    # # Display output
    # cv2.imshow("output", result_image_overlay)

    # Display output
    cv2.imshow("output", result_image)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()


#################
#################
#################
