# initiate
import random
import torch
torch.cuda.set_device(0)
import cv2
import numpy as np
from tools.run_infinity import *


# Add parent directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
infinity_dir = os.path.dirname(script_dir)  # Infinity/

# Import text mask client
sys.path.insert(0, os.path.dirname(infinity_dir))  # Add parent for text_mask_client
try:
    from text_mask_client import TextMaskClient
    TEXT_MASK_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: text_mask_client not found. Text region analysis will be skipped.")
    TEXT_MASK_AVAILABLE = False


model_path='./weights/infinity_2b_reg.pth'
# model_path='./weights/mm_2b.pth'
# model_path='/media/avlab/f09873b9-7c6a-4146-acdb-7db847b573201/VAR_ckpt/local_output/toy/ar-ckpt-giter001K-ep0-iter1248-last.pth'
vae_path='./weights/infinity_vae_d32reg.pth'
text_encoder_ckpt = "./weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001"

args=argparse.Namespace(
    pn='1M',
    model_path=model_path,
    cfg_insertion_layer=0,
    vae_type=32,
    vae_path=vae_path,
    add_lvl_embeding_only_first_block=1,
    use_bit_label=1,
    model_type='infinity_2b',
    rope2d_each_sa_layer=1,
    rope2d_normalized_by_hw=2,
    use_scale_schedule_embedding=0,
    sampling_per_bits=1,
    text_encoder_ckpt=text_encoder_ckpt,
    
    text_channels=2048,
    apply_spatial_patchify=0,
    h_div_w_template=1.000,
    use_flex_attn=0,
    cache_dir='/media/avlab/f09873b9-7c6a-4146-acdb-7db847b573201/VAR_ckpt/local_output/toy',
    checkpoint_type='torch',
    enable_model_cache=True, #modify
    seed=0,
    bf16=1,
    save_file='tmp.jpg',
    noise_apply_layers=1,
    noise_apply_requant=1,
    noise_apply_strength=0.01,
)

# load text encoder
text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
# load vae
vae = load_visual_tokenizer(args)
# load infinity
infinity = load_transformer(vae, args)

# generate image
prompt = """
Rendered text: \"ANNO DOMINI\",\"SANER\"
System prompt: 
Render the rendered text \"ANNO DOMINI\",\"SANER\" in the following image:
The image is taken from a slightly elevated angle, providing a clear view of the theater marquee. The marquee displays the text \"ANNO DOMINI\" in large, bold letters at the top, followed by \"SANER\" in smaller letters below. The background consists of a gray brick wall with some greenery peeking through at the top. The foreground features several hanging street lamps, adding to the urban setting. The overall composition suggests a daytime scene with natural lighting.
"""

prompt = """
System prompt: Render the text \"LAWSON\" in the following image:
A photograph of a Japanese convenience store, "LAWSON", set against the iconic Mount Fuji in the early morning. The store occupies the lower third of the frame, with its signature blue and white signboard featuring bold, capitalized lettering "LAWSON". In the background, the majestic Mount Fuji, covered in snow, dominates the upper portion of the frame, bathed in soft morning sunlight that casts a pinkish hue across its rugged slopes. The sky transitions from a gentle blue to a warm golden glow, enhancing the tranquil and serene atmosphere."""

# prompt = """
# Ultra‑realistic 35 mm photograph of an outdoor public swimming pool at midday:  In the foreground, fixed to a stainless‑steel safety fence, a slightly weather‑worn white sign with no lettering, edges peeling and casting a delicate shadow on the slick tiles. High dynamic range, authentic colors, subtle grain, cinematic realism.
# """

# prompt = """ A white sign with no words on it in the right bottom side of the photo. The background is a swimming pool."""

# prompt = '''
# Render the rendered text \"Angelo's\",\"Hamburgers\" and \"Beer TV Video Arcade\" in the following image:
# The realistic photo is taken from a street-level perspective, looking up at a sign for \"Angelo's\" with a car hop service logo below it. The sign advertises \"Hamburgers\" and mentions \"Beer TV Video Arcade.\" The background shows a clear sky and some trees, indicating a sunny day. The foreground includes part of a building with a \"Laundromat\" sign and another sign that says \"Welcome.\" The overall scene suggests a casual dining establishment with entertainment options.
# '''

# prompt = """
# Render the rendered text \"Marble Academy\" in the following image:
# The image is taken from a Google Street View perspective, showing a sign for \"Marble Academy\" placed on a stack of bricks. The sign has a white background with blue text. In the background, there is a building with large windows and some trees. The foreground shows part of a parking lot with white lines marking the spaces. The overall scene appears to be outdoors during daytime.
# """

# prompt = """
# Key words: Room
# Render the key word \"Room\" in the following image:
# The image is a digitally rendered image viewed from a side angle (not front!), showcasing a modern bedroom setup. The word \"Room\" is subtly rendered on the wall in the back of the room in a white, minimalist font. 
# """

# edit
prompt = """
System prompt: Render the text \"ANDERSON\" in the following image:
A photograph of a Japanese convenience store, "ANDERSON", set against the iconic Mount Fuji in the early morning. The store occupies the lower third of the frame, with its signature blue and white signboard featuring bold, capitalized lettering "ANDERSON". In the background, the majestic Mount Fuji, covered in snow, dominates the upper portion of the frame, bathed in soft morning sunlight that casts a pinkish hue across its rugged slopes. The sky transitions from a gentle blue to a warm golden glow, enhancing the tranquil and serene atmosphere.'
"""
# prompt = """
# A photograph of a Japanese convenience store set against the iconic Mount Fuji in the early morning. The store occupies the lower third of the frame, with its signature blue and white signboard featuring bold, capitalized lettering. In the background, the majestic Mount Fuji, covered in snow, dominates the upper portion of the frame, bathed in soft morning sunlight that casts a pinkish hue across its rugged slopes. The sky transitions from a gentle blue to a warm golden glow, enhancing the tranquil and serene atmosphere.'
# """

# ############################################# test prompts ##########################################
# 18_09
# prompt = """
# System prompt: Render the text \"PIKE AND WESTERN WINE SHOP\" in the following image:
# A realistic, eye-level street view of a wine shop with a maroon rectangular sign that reads "PIKE AND WESTERN WINE SHOP" in white capital letters. The storefront is glass-fronted with wine bottles and gift items displayed inside. Above the entrance is a large green metal awning, and on top of the awning are planters with dense leafy bushes or small trees. A sidewalk runs in front of the shop, and a man in a dark cap and blue shirt is walking by with a child. The reflection in the glass shows the opposite side of the street and the waterfront in the distance. Front-facing, centered composition, soft daylight, urban setting, sharp focus on storefront.
# """

# #17_19
# prompt = """
# System prompt: Render the text \"ANNO DOMINI / SANER\" in the following image:
# A realistic, eye-level street view of a minimalist gray brick building with a small marquee sign above the entrance. The marquee displays black and red block letters reading: "ANNO DOMINI / SANER." The sign is slightly weathered, framed in white with a rusted metal border. Above the sign, a row of four hanging dome-shaped blue lights is mounted on the building's blue-trimmed roofline. The entrance beneath the marquee is dark and recessed. The overall scene has an urban, industrial gallery feel. Centered composition, front-facing view, daytime with diffused lighting, sharp focus.
# """

#03_09
# prompt = """
# System prompt: Render the text \"COMMON THREADS\" , \"old & new\" in the following image:
# A realistic, eye-level street view of a small vintage-style boutique with a brown rectangular sign that reads "COMMON THREADS" in white capital letters, and the subtitle "old & new" below. The store has a simple white-painted brick exterior and a green sloped roof. In front of the entrance, clothing racks with assorted garments are displayed outdoors, along with a pair of planters and Adirondack chairs. The surrounding lot includes several parked cars on both sides, and a wooden utility pole with overhead wires is visible on the left. The background includes trees, rooftops, and power lines. Centered composition, front-facing, clear daylight, suburban setting, sharp focus.
# """

# # poster
prompt = """
A highly detailed and professionally designed poster for a fictional event titled “Voices of Tomorrow: A Global Youth Innovation Summit.” The poster should have a futuristic and visionary design, incorporating vibrant gradients (neon blue, electric purple, and soft coral pink) over a dark navy background with glowing constellation patterns subtly embedded into the sky. The top section features a stylized title in bold, sleek, sans-serif typography with a glowing outline effect: “VOICES OF TOMORROW” — and a subtitle beneath it in smaller but elegant serif font: “A Global Youth Innovation Summit.”
The central visual should depict a group of diverse, young individuals (multi-ethnic, gender-inclusive) standing confidently on a slightly elevated, illuminated platform, overlooking a stylized, abstract cityscape of the future — with sustainable buildings, vertical gardens, drones in the sky, and glowing data lines arching above like auroras. Their silhouettes are dramatically backlit with rim lighting to create a hopeful, heroic mood.
"""
# #####################################################################################################
# prompt = """
# Render the text \"Target\" in the following image:
# This image shows the exterior of a large retail store on a sunny day. The building has a modern, clean architectural design with neutral tones of white and gray. A prominent feature is the bright blue rectangular sign centered above the entrance, which displays the name \"Target\" in bold, white letters. Next to the text is a simple, iconic yellow spark-like symbol that accompanies the branding.
# Below the sign is a covered entranceway supported by steel rods and beams, providing shade and protection to customers entering or exiting the store. There are several sets of glass sliding doors, and shopping carts can be seen inside and just outside the entrance. A row of bollards stands in front of the store, ensuring pedestrian safety.
# To the left of the entrance, a few people are walking or pushing shopping carts, and the pavement includes clearly marked pedestrian pathways with blue lines. Trees with green leaves frame the top of the image, suggesting it's a pleasant, mild-weather day. In the background, additional buildings and greenery are visible, giving the impression that the store is located in a well-developed suburban or urban area.
# """
# prompt = """
# The image is taken from a Google Street View perspective, showing a sign for \"Marble Yard\" placed on a stack of bricks. The sign has a white background with blue text. The address \"900\" is displayed below the company name. In the background, there is a building with large windows and some trees. The foreground shows part of a parking lot with white lines marking the spaces. The overall scene appears to be outdoors during daytime.
# """

indoor_prompt= \
"""Render the text "foreground" and "burbank" in the following image:
The image is taken from a slightly elevated angle, showcasing a bathroom interior with a vertical camera orientation. The prominent features include a towel warmer on the wall to the left and a white door in the center. The floor is adorned with beige tiles, complementing the light-colored walls. Verbatim text snippets visible in the image are "foreground" and "burbank." The text appears to be in a light purple or pinkish color with a varying font style, likely casual, with "foreground" oriented vertically on the left side and "burbank" horizontally at the bottom right corner. The background displays a clean, minimalistic design, emphasizing functionality with essential fixtures such as a mirrored cabinet above the bathtub."""

prompt = \
'''Render the text "Wall Street" in the following image:
The image captures a close-up view of a street sign at the iconic intersection of Wall Street and Broad Street in New York City. The "WALL ST" sign is prominently displayed in the center of the image, mounted on a metal pole along with another sign below it labeled "BROAD ST". The Wall Street sign includes the range “11–21 →” and features a small grayscale image of George Washington's statue standing in front of the Federal Hall, symbolizing the financial significance of the area.
A yellow streetlight, mounted above the signs, casts a warm glow in the overcast lighting, giving the scene a moody urban feel. In the background, towering skyscrapers with a mix of classical and modern architecture dominate the view, emphasizing the dense and historic character of the Financial District in Manhattan. The image has a gritty, realistic texture, possibly taken during a light drizzle, as suggested by the subtle wet sheen on the pole and light specks in the air.
'''

prompt="""A cozy coffee shop chalkboard menu that says \"LATTE\" in handwritten letters."""
prompt="""a wathet bird sitting on a branch of yellow flowers"""
# prompt="""a young boy is playing with water in the street, laughing happily"""
cfg = 2.8
tau = 0.3 # my setting

cfg = 2.8 # text mask
tau = 0.53
h_div_w =1/1 # aspect ratio, height:width
seed = random.randint(0, 10000)
enable_positive_prompt=0

h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
# ----------------------------------------------------------------------------------
# modified
from infinity.models.bitwise_self_correction import BitwiseSelfCorrection
if not hasattr(args, 'debug_bsc'):
    args.debug_bsc = False
bitwise_self_correction = BitwiseSelfCorrection(vae, args)
infinity.setup_mask_processor(vae, scale_schedule, bitwise_self_correction)
mask_path = 'imgs/bird.jpg'

# # ----------------------------------------------------------------------------------
# # Convert image to numpy array
# image_np = generated_image.cpu().numpy()

# # Extract text mask (REQUIRED for F1 score calculation)
# text_mask_full = None
# if not TEXT_MASK_AVAILABLE:
#     raise RuntimeError("❌ text_mask_client module not available. Please ensure text_mask_client.py is accessible.")

# if not hasattr(args, 'text_mask_server'):
#     raise RuntimeError("❌ text_mask_server argument not provided. Please specify --text_mask_server URL.")

# print(f"\n🔍 Extracting text mask from generated image...")
# text_mask_client = TextMaskClient(args.text_mask_server)
# text_mask_full = text_mask_client.predict_from_array(image_np)

# # Convert to grayscale if needed
# if len(text_mask_full.shape) == 3:
#     text_mask_full = cv2.cvtColor(text_mask_full, cv2.COLOR_BGR2GRAY)
# ----------------------------------------------------------------------------------


infinity.set_mask(
    mask_path=mask_path,
    scale_idx=[0,1,2,3,4,5,6,7,8,9,10,11],
    method='weighted',
    alpha=0.3,
    strength=[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
)
cfg = 4
tau = 0.3
# ----------------------------------------------------------------------------------
generated_image = gen_one_img(
    infinity,
    vae,
    text_tokenizer,
    text_encoder,
    prompt,
    g_seed=seed,
    gt_leak=0,
    gt_ls_Bl=None,
    cfg_list=cfg,
    tau_list=tau,
    scale_schedule=scale_schedule,
    cfg_insertion_layer=[args.cfg_insertion_layer],
    vae_type=args.vae_type,
    sampling_per_bits=args.sampling_per_bits,
    enable_positive_prompt=enable_positive_prompt,
)
import os
os.makedirs("./compareTest", exist_ok=True)
args.save_file = './compareTest/train3.jpg'
os.makedirs(osp.dirname(osp.abspath(args.save_file)), exist_ok=True)
cv2.imwrite(args.save_file, generated_image.cpu().numpy())
print(f'Save to {osp.abspath(args.save_file)}')

"""USAGE:
# Generate an image with mask processor (source image injection).
python gen.py --model_path ./weights/infinity_2b_reg.pth \
              --vae_path ./weights/infinity_vae_d32reg.pth \
              --text_encoder_ckpt ./weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001 \
              --save_file ./output/generated_image.jpg
"""