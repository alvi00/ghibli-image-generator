from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt
import time
import os

# Load model function
def load_model():
    model_id = "nitrosocke/Ghibli-Diffusion"  # Ghibli-style model from Hugging Face
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print("Loading model...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.enable_attention_slicing()  # Optimize memory usage
    print("Model loaded!")
    return pipe

# Preprocess image (resize + padding)
def preprocess_image(image):
    image = image.convert("RGB")
    target_size = 512
    width, height = image.size
    ratio = target_size / max(width, height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Add padding to make it exactly 512x512
    new_image = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    new_image.paste(image, ((target_size - new_width) // 2, (target_size - new_height) // 2))
    
    return new_image

# Function to generate Ghibli-style image
def generate_ghibli_image(image, pipe, strength):
    image = image.convert("RGB")

    # Resize while maintaining aspect ratio
    width, height = image.size
    new_size = min(768, width, height)
    image = image.resize((new_size, new_size))

    # Stronger Prompt
    prompt = (
        "A young woman with expressive anime-style eyes, wearing elegant traditional clothing, "
        "surrounded by a magical and dreamy environment, warm soft lighting, Studio Ghibli style, "
        "beautifully detailed, cinematic shading, soft pastel colors"
    )

    # Negative Prompt (if supported)
    negative_prompt = "blurry, distorted, low quality, extra limbs, creepy, deformed face"

    print("Generating image...")
    start_time = time.time()

    result = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        negative_prompt=negative_prompt  # Optional
    ).images[0]

    print(f"Image generated in {time.time() - start_time:.2f} seconds!")
    return result


# Check for GPU
gpu_info = "✅ GPU is available!" if torch.cuda.is_available() else "⚠️ Warning: GPU not available. Processing will be slow."
print(gpu_info)

# Load the model
pipe = load_model()

# Get image file path from user
image_path = input("Enter image path (default: D:/GhibliArt/mim.jpg): ").strip()
if not image_path:
    image_path = "D:/GhibliArt/mim.jpg"  # Default path

if os.path.exists(image_path):
    image = Image.open(image_path)

    # Display original image
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    plt.show()

    # Ask for strength input with validation
    while True:
        strength_input = input("Enter stylization strength (0.3-0.8, recommended 0.6): ").strip()
        if not strength_input:  
            strength = 0.6  # Default value
            break
        try:
            strength = float(strength_input)
            if 0.3 <= strength <= 0.8:
                break
            else:
                print("❌ Please enter a value between 0.3 and 0.8.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")

    # Generate and display the result
    result_img = generate_ghibli_image(image, pipe, strength)

    plt.figure(figsize=(5, 5))
    plt.imshow(result_img)
    plt.title("Ghibli Portrait")
    plt.axis("off")
    plt.show()

    # Save the output image
    output_filename = f"ghibli_portrait_{os.path.basename(image_path)}"
    result_img.save(output_filename)
    print(f"✅ Image saved as {output_filename} in the current directory!")
else:
    print(f"❌ File not found: {image_path}. Please check the path and try again.")
