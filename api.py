from fastapi import FastAPI
from fastapi.responses import FileResponse
import base64
import uvicorn
import random
import cv2 as cv
import os
import torch
import datetime
import tarfile
import numpy as np
import time
import subprocess
from typing import List, Optional
from pydantic import BaseModel
from diffusers import (
    ControlNetModel, DiffusionPipeline, AutoPipelineForImage2Image
)
from latent_consistency_controlnet import LatentConsistencyModelPipeline_controlnet
# from cog import BasePredictor, Input, Path
from PIL import Image

MODEL_CACHE_URL = "https://weights.replicate.delivery/default/fofr-lcm/model_cache.tar"
MODEL_CACHE = "model_cache"
DATA_CACHE = "data_cache"

if not os.path.exists(DATA_CACHE):
    os.mkdir(DATA_CACHE)


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


def create_pipeline(
    pipeline_class,
    safety_checker: bool = True,
    controlnet: Optional[ControlNetModel] = None,
):
    kwargs = {
        "cache_dir": MODEL_CACHE,
        "local_files_only": True,
    }

    if not safety_checker:
        kwargs["safety_checker"] = None

    if controlnet:
        kwargs["controlnet"] = controlnet
        kwargs["scheduler"] = None

    pipe = pipeline_class.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", **kwargs)
    pipe.to(torch_device="cuda", torch_dtype=torch.float16)
    pipe.enable_xformers_memory_efficient_attention()
    return pipe


def do_control_image(image, canny_low_threshold, canny_high_threshold):
    image = np.array(image)
    canny = cv.Canny(image, canny_low_threshold, canny_high_threshold)
    return Image.fromarray(canny)


def get_allowed_dimensions(base=512, max_dim=1024):
    """
    Function to generate allowed dimensions optimized around a base up to a max
    """
    allowed_dimensions = []
    for i in range(base, max_dim + 1, 64):
        for j in range(base, max_dim + 1, 64):
            allowed_dimensions.append((i, j))
    return allowed_dimensions


def get_resized_dimensions(width, height):
    """
    Function adapted from Lucataco's implementation of SDXL-Controlnet for Replicate
    """
    allowed_dimensions = get_allowed_dimensions()
    aspect_ratio = width / height
    print(f"Aspect Ratio: {aspect_ratio:.2f}")
    # Find the closest allowed dimensions that maintain the aspect ratio
    # and are closest to the optimum dimension of 768
    optimum_dimension = 768
    closest_dimensions = min(
        allowed_dimensions,
        key=lambda dim: abs(dim[0] / dim[1] - aspect_ratio)
        + abs(dim[0] - optimum_dimension),
    )
    return closest_dimensions


def resize_images(images, width, height):
    return [
        img.resize((width, height))
        if img is not None else None
        for img in images
    ]


def open_image(image_path):
    return Image.open(str(image_path)) if image_path is not None else None


def apply_sizing_strategy(
    sizing_strategy, width, height, control_image=None, image=None
):
    image = open_image(image)
    control_image = open_image(control_image)

    if image and image.mode == "RGBA":
        image = image.convert("RGB")

    if control_image and control_image.mode == "RGBA":
        control_image = control_image.convert("RGB")

    if sizing_strategy == "input_image":
        print("Resizing based on input image")
        # width, height = get_dimensions(image)
        width, height, _ = image.shape
    elif sizing_strategy == "control_image":
        print("Resizing based on control image")
        # width, height = get_dimensions(control_image)
        width, height, _ = control_image.shape
    else:
        print("Using given dimensions")

    image, control_image = resize_images([image, control_image], width, height)
    return width, height, control_image, image


"""Load the model into memory to make running multiple predictions efficient"""

if not os.path.exists(MODEL_CACHE):
    download_weights(MODEL_CACHE_URL, MODEL_CACHE)

# txt2img_pipe = create_pipeline(DiffusionPipeline)
# txt2img_pipe_unsafe = create_pipeline(
#     DiffusionPipeline, safety_checker=False
# )

img2img_pipe = create_pipeline(AutoPipelineForImage2Image)
# img2img_pipe.to(torch_device=torch.device("cuda"), torch_dtype=torch.float16)
# img2img_pipe_unsafe = create_pipeline(
#     AutoPipelineForImage2Image, safety_checker=False
# )

# controlnet_canny = ControlNetModel.from_pretrained(
#     "lllyasviel/control_v11p_sd15_canny",
#     cache_dir="model_cache",
#     local_files_only=True,
#     torch_dtype=torch.float16,
# ).to("cuda")
#
# controlnet_pipe = create_pipeline(
#     LatentConsistencyModelPipeline_controlnet,
#     controlnet=controlnet_canny,
# )
# controlnet_pipe_unsafe = create_pipeline(
#     LatentConsistencyModelPipeline_controlnet,
#     safety_checker=False,
#     controlnet=controlnet_canny,
# )

# warm the pipes
# txt2img_pipe(prompt="warmup")
# txt2img_pipe_unsafe(prompt="warmup")
img2img_pipe(prompt="warmup", image=[Image.new("RGB", (768, 768))])
# img2img_pipe_unsafe(prompt="warmup", image=[Image.new("RGB", (768, 768))])
# controlnet_pipe(
#     prompt="warmup",
#     image=[Image.new("RGB", (768, 768))],
#     control_image=[Image.new("RGB", (768, 768))],
# )
# controlnet_pipe_unsafe(
#     prompt="warmup",
#     image=[Image.new("RGB", (768, 768))],
#     control_image=[Image.new("RGB", (768, 768))],
# )
app = FastAPI()


class Data(BaseModel):
    # update_sql: bool = False
    # school_list: List[int] = []
    # level_id_list: List[int] = []
    # course_type_id_list: List[int] = []
    # description = "For multiple prompts, enter each on a new line."
    prompt: str = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

    # description = "Width of output image. Lower if out of memory"
    width: int = 768

    # description = "Height of output image. Lower if out of memory",
    height: int = 768

    # description = "Decide how to resize images – use width/height, resize based on input image or control image",
    # choices = ["width/height", "input_image", "control_image"],
    sizing_strategy: str = "width/height"

    # description = "Input image for img2img",
    # need base64
    image_base64: str
    image_name: str

    # description = "Prompt strength when using img2img. 1.0 corresponds to full destruction of information in image",
    # 0-1.0
    prompt_strength: float = 0.8

    # description = "Number of images per prompt",
    # 1-50
    num_images: int = 1

    # description = "Number of denoising steps. Recommend 1 to 8 steps.",
    # 1-50
    num_inference_steps: int = 8

    # description = "Scale for classifier-free guidance"
    # ge = 1, le = 20, default = 8.0
    guidance_scale: float = 8.0

    # 1-50
    lcm_origin_steps: int = 50

    # description = "Random seed. Leave blank to randomize the seed", default = None
    seed: int = None

    # description = "Image for controlnet conditioning",
    control_image_base64: str = None
    control_image_type: str = None

    # description = "Controlnet conditioning scale",
    # 0.1 ~ 4.0
    controlnet_conditioning_scale: float = 2.0

    # description = "Controlnet start",
    # 0.0 ~ 1.0
    control_guidance_start: float = 0.0

    # description = "Controlnet end",
    # 0.0 ~ 1.0
    control_guidance_end: float = 1.0

    # description = "Canny low threshold",
    # 1-255
    canny_low_threshold: float = 100

    # description = "Canny high threshold",
    # 1-255
    canny_high_threshold: float = 200

    # description = "Option to archive the output images",
    # archive_outputs: bool = False

    # description = "Disable safety checker for generated images. This feature is only available through the API",
    # disable_safety_checker: bool = False


@app.post("/predict")
@torch.inference_mode()
def predict(
        data: Data,
):
    """Run a single prediction on the model"""
    prediction_start = time.time()

    if data.seed is None:
        seed = int.from_bytes(os.urandom(2), "big")
    else:
        seed = data.seed
    print(f"Using seed: {seed}")

    prompt = data.prompt.strip().splitlines()
    if len(prompt) == 1:
        print("Found 1 prompt:")
    else:
        print(f"Found {len(prompt)} prompts:")
    for p in prompt:
        print(f"- {p}")

    if len(prompt) * data.num_images == 1:
        print("Making 1 image")
    else:
        print(f"Making {len(prompt) * data.num_images} images")

    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    file_type = os.path.splitext(data.image_name)[-1]
    file_name = f"{datetime_str}_{random.randint(100000, 1000000)}{file_type}"
    image_path = os.path.join(DATA_CACHE, file_name)
    decoded_string = base64.b64decode(data.image_base64)
    with open(image_path, 'wb') as output_file:
        output_file.write(decoded_string)
    if data.control_image_base64:
        file_type = os.path.splitext(data.control_image_type)[-1]
        file_name = f"{datetime_str}_{random.randint(100000, 1000000)}{file_type}"
        control_image_path = os.path.join(DATA_CACHE, file_name)
        control_decoded_string = base64.b64decode(data.control_image_base64)
        with open(control_image_path, 'wb') as output_file:
            output_file.write(control_decoded_string)
    else:
        control_image_path = None
    # if data.image or data.control_image:
    (
        width,
        height,
        control_image,
        image,
    ) = apply_sizing_strategy(
        data.sizing_strategy,
        data.width,
        data.height,
        control_image_path,
        image_path
    )
    os.remove(image_path)
    kwargs = {}
    canny_image = None

    if image:
        kwargs["image"] = image
        kwargs["strength"] = data.prompt_strength

    if control_image:
        canny_image = do_control_image(
            control_image,
            data.canny_low_threshold,
            data.canny_high_threshold
        )
        kwargs["control_guidance_start"] = data.control_guidance_start
        kwargs["control_guidance_end"] = data.control_guidance_end
        kwargs["controlnet_conditioning_scale"] = data.controlnet_conditioning_scale

        # TODO: This is a hack to get controlnet working without an image input
        # The current pipeline doesn't seem to support not having an image, so
        # we pass one in but set strength to 1 to ignore it
        if not data.image:
            kwargs["image"] = Image.new(
                "RGB",
                (width, height),
                (128, 128, 128)
            )
            kwargs["strength"] = 1.0

        kwargs["control_image"] = canny_image

    mode = "controlnet" if control_image else "img2img" if image else "txt2img"
    print(f"{mode} mode")

    # pipe = getattr(
    #     self,
    #     f"{mode}_pipe" if not disable_safety_checker else f"{mode}_pipe_unsafe",
    # )
    pipe = img2img_pipe

    common_args = {
        "width": width,
        "height": height,
        "prompt": prompt,
        "guidance_scale": data.guidance_scale,
        "num_images_per_prompt": data.num_images,
        "num_inference_steps": data.num_inference_steps,
        "lcm_origin_steps": data.lcm_origin_steps,
        "output_type": "pil",
    }

    start = time.time()
    result = pipe(
        **common_args,
        **kwargs,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images
    print(f"Inference took: {time.time() - start:.2f}s")

    # if data.archive_outputs:
    #     start = time.time()
    #     archive_start_time = datetime.datetime.now()
    #     print(f"Archiving images started at {archive_start_time}")
    #     rand_seed = random.randint(1, 10000)
    #     tar_path = os.path.join(
    #         DATA_CACHE,
    #         f"{datetime_str}_output_{rand_seed}.tar"
    #     )
    #     with tarfile.open(tar_path, "w") as tar:
    #         rand_seed = random.randint(1, 10000)
    #         for i, sample in enumerate(result):
    #             output_name = f"{datetime_str}_output_{rand_seed}_{i}.jpg"
    #             output_path = os.path.join(DATA_CACHE, output_name)
    #             sample.save(output_path)
    #             tar.add(output_path, output_name)

    #     print(f"Archiving took: {time.time() - start:.2f}s")
    #     return [FileResponse(tar_path)]
    # If not archiving, or there is an error in archiving, return the paths of individual images.
    image_data_list = []
    rand_seed = random.randint(1, 10000)
    for i, sample in enumerate(result):
        output_name = f"{datetime_str}_output_{rand_seed}_{i}.jpg"
        output_path = os.path.join(DATA_CACHE, output_name)
        sample.save(output_path)
        with open(output_path, "rb") as f:
            image_data = f.read()
        os.remove(output_path)
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        image_data_list.append(image_base64)

    if canny_image:
        canny_image_name = f"{datetime_str}_canny_{rand_seed}.jpg"
        canny_image_path = os.path.join(DATA_CACHE, canny_image_name)
        canny_image.save(canny_image_path)
        with open(canny_image_path, "rb") as f:
            image_data = f.read()
        os.remove(canny_image_path)
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        image_data_list.append(image_base64)
    print(f"Prediction took: {time.time() - prediction_start:.2f}s")
    return {
        "status": "success",
        "data": image_data_list
    }


if __name__ == '__main__':
    uvicorn.run(
        app='api:app', host="0.0.0.0", port=5525, reload=False, workers=1
    )
