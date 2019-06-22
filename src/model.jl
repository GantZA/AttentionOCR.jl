using Flux
using Flux: @epochs
using CuArrays

include("cnn.jl")
include("get_labels.jl")
include("batching.jl")
include("seq2seq_model.jl")

ATTN_NUM_HIDDEN = 128
ATTN_NUM_LAYERS = 2
TARGET_VOCAB_SIZE = 12
BATCH_SIZE = 10


function cnn(cnn_model, data)
    cnn_output = cnn_model.(gpu.(data))
    return [hcat([cnn_output[i][:,j] for i in 1:BATCH_SIZE]...) for j in 1:512]
end

function model(cnn_model, encoder, decoder, x, y)
    encoder_inputs = cnn(cnn_model, x)
    label_inputs = gpu.([hcat([y[i][:,j] for i in 1:BATCH_SIZE]...) for j in 1:20])
    total_loss = model_seq_2_seq(encoder, decoder, encoder_inputs, label_inputs)
    return total_loss
end

mutable struct aocr_model
    cnn
    encoder
    decoder
end # struct

function (a::aocr_model)(x, y, learning_rate, epochs)
    model(x,y) = model(a.cnn_model, a.encoder, a.decoder, x, y)
    opt = ADAM(learning_rate)
    @epochs epochs Flux.train(model, params(a), zip(x,y), opt)
end
