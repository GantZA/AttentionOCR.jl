using AttentionOCR

# Load and save images for faster loading later
img_names = glob("*.png", "data/phase_1/images/")
images = store_images(img_names)
JLD.save("images.jld", "images", images)

# Load Labels
labels = get_labels("D:/Documents/Projects/AttentionOCR.jl/data/phase_1/labels")

# Load labels from .jld format
images_jld = JLD.load("D:/Documents/Projects/AttentionOCR.jl/myfile.jld")["images"]

# Batch data
batches, label_batches = batch(images_jld[1:100], labels[1:100], 10)



cnn_model = build_cnn_network() |> gpu
enc = Encoder(140) |> gpu
dec = Decoder(HIDDEN, 12) |> gpu

aocr_model(cnn_model, enc, dec)
