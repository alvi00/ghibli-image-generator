from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import io
from google.colab import files
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import time

# Load model function
def load_model():
    model_id = ""  # Using the existing Ghibli Diffusion model
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print("Loading model...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.enable_attention_slicing()  # Optimize memory usage
    print("Model loaded!")
    return pipe

# Function to generate Ghibli-style image
def generate_ghibli_image(image, pipe, strength):
    image = image.convert("RGB")
    width, height = image.size
    
    # Calculate new dimensions to maintain aspect ratio but fit within 512x512
    ratio = min(512/width, 512/height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    image = image.resize((new_width, new_height))
    
    prompt = ""
    print("Generating image...")
    start_time = time.time()
    result = pipe(prompt=prompt, image=image, strength=strength).images[0]
    print(f"Image generated in {time.time() - start_time:.2f} seconds!")
    return result

# Check for GPU
gpu_info = "GPU is available!" if torch.cuda.is_available() else "Warning: GPU not available. Processing will be slow."
print(gpu_info)

# Load the model
pipe = load_model()

# Upload image section
print("Please upload your image file:")
uploaded = files.upload()

if uploaded:
    # Get the first uploaded file
    file_name = list(uploaded.keys())[0]
    image = Image.open(io.BytesIO(uploaded[file_name]))
    
    # Display original image
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    plt.show()
    
    # Ask for strength input
    strength = float(input("Enter stylization strength (0.3-0.8, recommended 0.6): "))
    strength = max(0.3, min(0.8, strength))  # Clamp between 0.3 and 0.8
    
    # Generate and display the result
    result_img = generate_ghibli_image(image, pipe, strength)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(result_img)
    plt.title("Ghibli Portrait")
    plt.axis('off')
    plt.show()
    
    # Save the output image and offer download
    output_filename = f"ghibli_portrait_{file_name}"
    result_img.save(output_filename)
    files.download(output_filename)
    print(f"Image saved as {output_filename} and download initiated!")
else:
    print("No file was uploaded. Please run the cell again and upload an image.")