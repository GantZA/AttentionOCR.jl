using Flux

function nullify_grad!(p)
    if typeof(p) <: TrackedArray
        p.grad .= 0.0f0
    end
    return p
end

# Convert internal parameters for defined model so that they become Tracked cuArray
function zero_grad!(model)
    model = mapleaves(nullify_grad!, model)
end

function bce(yhat, y)
    neg_abs = -abs.(yhat)
    mean(relu.(yhat) .- yhat .* y .+ log.(1 .+ exp.(neg_abs)))
end

function build_cnn_network()

    input_layer = Chain(
        x -> 2x .- 1f0)

    conv_relu_layer_1 = Chain(
        Conv((3,3),1=>64, relu, pad=(1,1), stride=(1,1)),
        MaxPool((2,2),pad=(1,0), stride=(2,2)))

    conv_relu_layer_2 = Chain(
        Conv((3,3),64=>128, relu, pad=(1,1), stride=(1,1)),
        MaxPool((2,2),pad=(0,0), stride=(2,2)))

    conv_relu_BN_layer_1 = Chain(
        Conv((3,3),128=>256, pad=(1,1), stride=(1,1)),
        BatchNorm(256, relu))

    conv_relu_layer_3 = Chain(
        Conv((3,3),256=>256, relu, pad=(1,1), stride=(1,1)),
        MaxPool((1,2),pad=(0,0), stride=(1,2)))

    conv_relu_BN_layer_2 = Chain(
        Conv((3,3),256=>512, pad=(1,1), stride=(1,1)),
        BatchNorm(512, relu))

    conv_relu_layer_4 = Chain(
        Conv((3,3),512=>512, relu, pad=(1,1), stride=(1,1)),
        MaxPool((1,2),pad=(0,0), stride=(1,2)))

    conv_relu_BN_layer_3 = Chain(
        Conv((2,2),512=>512, pad=(1,1), stride=(1,1)),
        BatchNorm(512, relu),
        MaxPool((1,2),pad=(0,0), stride=(1,2)))

    drop_and_squeeze = Chain(
        Dropout(0.5),
        x -> dropdims(x, dims=Tuple(findall(y->(y==1),size(x)))))

    model = Chain(
        input_layer...,
        conv_relu_layer_1...,
        conv_relu_layer_2...,
        conv_relu_BN_layer_1...,
        conv_relu_layer_3...,
        conv_relu_BN_layer_2...,
        conv_relu_layer_4...,
        conv_relu_BN_layer_3...,
        drop_and_squeeze...)
    return model
end
