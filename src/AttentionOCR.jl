module AttentionOCR
using JLD

include("model.jl")
include("save_images.jl")
include("get_labels.jl")
include("batching.jl")

export aocr_model, stored_imgs, prepare_img, get_labels, batch
export build_cnn_network, Encoder, Decoder

end # module
