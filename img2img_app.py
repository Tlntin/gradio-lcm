import gradio as gr
import requests
import base64
import datetime
import random
import os

DATA_CACHE = "data_cache"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "512"))
if not os.path.exists(DATA_CACHE):
    os.mkdir(DATA_CACHE)


def infer(
        input_prompt,
        img_path,
        num_img,
        img_width,
        img_height,
        num_inference_step
):
    url = "http://localhost:5525/predict/"
    img_name = os.path.split(img_path)[-1]
    data = {
        "prompt": input_prompt,
        "width": img_width,
        "height": img_height,
        "image_base64": base64.b64encode(open(img_path, 'rb').read()).decode(),
        "image_name": img_name,
        "sizing_strategy": "width/height",
        "prompt_strength": 0.8,
        "num_images": num_img,
        "num_inference_steps": num_inference_step,
        "guidance_scale": 8,
        "lcm_origin_steps": 50,
        "seed": 0,
        "controlnet_conditioning_scale": 2,
        "control_guidance_start": 0,
        "control_guidance_end": 1,
        "canny_low_threshold": 100,
        "canny_high_threshold": 200,
        "archive_outputs": False,
        "disable_safety_checker": False
    }

    response = requests.post(url, json=data)
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    images = []
    if response.status_code == 200:
        output_data = response.json()
        image_data_list = output_data["data"]
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        for i, image_data_base64 in enumerate(image_data_list):
            image_data = base64.b64decode(image_data_base64)
            file_name = f"{datetime_str}_{random.randint(100000, 1000000)}-{i + 1}.jpg"
            file_path = os.path.join(DATA_CACHE, file_name)
            with open(file_path, "wb") as f:
                f.write(image_data)
            images.append(file_path)
    else:
        print(response)
    return images


def clear_input(p, g):
    return "", []


title = "# Img2Img for [Latent Consistency Model](https://github.com/luosiallen/latent-consistency-model)"
examples = [
    "cute and intelligent anime little boy.",
    "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
]
with gr.Blocks(css="style.css") as demo:
    gr.Markdown(value=title)
    with gr.Row():
        with gr.Column(scale=1):
            source_img = gr.Image(
                sources=["upload"],
                type="filepath",
                label="init_img | 512*512 px",
                height=324,
            )
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                lines=6,
                placeholder="Enter your prompt",
                container=False,
            )
        with gr.Column(scale=2):
            gallery = gr.Gallery(
                label="Generated images",
                show_label=False,
                elem_id="gallery",
                height=480,
            )
    with gr.Row():
        run_button = gr.Button("Run")
        clear_button = gr.Button("Clear")
    with gr.Accordion("Advanced options", open=False):
        num_images = gr.Slider(
            label="Number of images",
            minimum=1,
            maximum=8,
            step=1,
            value=4,
            visible=True,
        )
        width = gr.Slider(
            label="Width",
            minimum=256,
            maximum=MAX_IMAGE_SIZE,
            step=32,
            value=512,
        )
        height = gr.Slider(
            label="Height",
            minimum=256,
            maximum=MAX_IMAGE_SIZE,
            step=32,
            value=512,
        )
        num_inference_steps = gr.Slider(
            label="Number of inference step",
            minimum=4,
            maximum=40,
            step=4,
            value=8,
            visible=True,
        )
    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=gallery,
        fn=infer,
    )
    clear_button.click(
        fn=clear_input,
        inputs=[prompt, gallery],
        outputs=[prompt, gallery],
    )
    gr.on(
        triggers=[
            prompt.submit,
            run_button.click,
        ],
        fn=infer,
        inputs=[
            prompt, source_img, num_images, width, height, num_inference_steps
        ],
        outputs=gallery,
    )


if __name__ == "__main__":
    demo.queue(api_open=False)
    # demo.queue(max_size=20).launch()
    demo.launch()