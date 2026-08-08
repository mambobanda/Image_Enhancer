from super_image import EdsrModel, ImageLoader
from PIL import Image
from nafnetlib import DeblurProcessor, DenoiseProcessor


# ===== DEBLURRING EXAMPLE =====
# Initialize deblurring processor
deblur = DeblurProcessor(
    model_id="gopro_width64",          # Choose model
    model_dir="./models/",              # Where to store models
    device="cuda"                       # or "cpu"
)

# Load your blurry image
blurry_img = Image.open('E:/Comfy Output/MiniMax Tests/Res_Test2/output_01091_.png')

# Process it
deblurred = deblur.process(blurry_img)
deblurred.show()
deblurred.save('C:/test/scaled_unblurred.png')

# ===== DENOISING EXAMPLE =====
# Initialize denoising processor
denoise = DenoiseProcessor(
    model_id="sidd_width64",
    model_dir="./models/",
    device="cuda"
)

# Load noisy image
noisy_img = Image.open('C:/test/scaled_unblurred.png')

# Process it
denoised = denoise.process(noisy_img)
denoised.show()
denoised.save('C:/test/scaled_unblurred.png')


################################

model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
image = Image.open('C:/test/scaled_unblurred.png')
inputs = ImageLoader.load_image(image)
preds = model(inputs)
ImageLoader.save_image(preds, 'C:/test/scaled_2x.png')