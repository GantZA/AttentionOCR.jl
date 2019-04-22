

max_image_width = 1300
max_image_height = 75
max_prediction_length = 19

max_resized_width = ceil(Int,max_image_width / max_image_height * 32)

encoder_size = ceil(Int,max_width/4)
decoder_size = max_prediction_length + 2

buckets = [encoder_size, decoder_size]

height = param()
height_float = param(32f0)
typeof(height)


function prepare_img(image)
    # Resize the image to the maximum width and height
    img = Float32.(image);
    dims = size(img)

    max_width = Int32(ceil(dims[2]/dims[1]) * 32)
    max_height = Int32(ceil(max_resized_width/max_width) * 32)

    if (max_resized_width >= max_width) & (dims[1] > 32)
        img = imresize(img, 32, max_width)
    else
        img = imresize(img, 32, max_resized_width)
    end
    return img
end

using Images
using FileIO
using Glob

img_names = glob("*.png", "data/phase_1/images/")
image = load(img_names[154]);
img = prepare_img(image)
Gray.(img)
