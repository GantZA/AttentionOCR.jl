using JLD
using Images
using FileIO
using Glob

max_image_width = 1300
max_image_height = 75
max_prediction_length = 19

max_resized_width = ceil(Int,max_image_width / max_image_height * 32)
encoder_size= ceil(Int,max_resized_width/4)
decoder_size = max_prediction_length + 1
buckets = [encoder_size, decoder_size]

function prepare_img(image::Matrix{Gray{Normed{UInt8,8}}})
    # Resize the image to the maximum width and height
    img = reshape(Float32.(image),(75,1300,1))
    dims = size(img)

    max_width = Int32(ceil(dims[2]/dims[1]) * 32)
    max_height = Int32(ceil(max_resized_width/max_width) * 32)

    if (max_resized_width >= max_width) & (dims[1] > 32)
        img = imresize(img[:,:,1], (32, max_width))
    else
        img = imresize(img[:,:,1], (32, max_resized_width))
    end
    return reshape(vec(img), size(img, 2), size(img, 1), 1, :)
end

function prepare_img(images::Vector{Array{Gray{Normed{UInt8,8}},2}})
    img_array = prepare_img.(images)
    img_array = reshape(collect(Iterators.flatten(img_array)),
        (size(img_array[1],1), size(img_array[1],2),1,size(img_array,1)))
    return img_array
end

function store_images(image_paths)
    N = size(image_paths,1)
    stored_imgs = Array{Array{Float32,4}}(undef, N)
    for i in 1:N
        stored_imgs[i] = prepare_img(load(image_paths[i]))
    end
    return stored_imgs
end
