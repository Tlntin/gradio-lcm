import requests
import base64
import os

url = "http://localhost:5525/predict/"

data = {
  "prompt": "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
  "width": 768,
  "height": 768,
  "image_base64": base64.b64encode(open("xxxx.jpg", 'rb').read()).decode(),
  "image_name": "xxxx.jpg",
  "sizing_strategy": "width/height",
  "prompt_strength": 0.8,
  "num_images": 1,
  "num_inference_steps": 8,
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
if response.status_code == 200:
    output_data = response.json()
    image_data_list = output_data["data"]
    for i, image_data_base64 in enumerate(image_data_list):
        image_data = base64.b64decode(image_data_base64)
        file_name = f"image{i + 1}.jpg"
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(image_data)
    print("OK")
else:
  print("Error", response)
