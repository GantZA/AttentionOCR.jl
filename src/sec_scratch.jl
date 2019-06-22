function get_seq(n_timesteps)
    X = rand(n_timesteps)

    limit = n_timesteps/4.0
    Y = [x < limit ? 0 : 1 for x in cumsum(X)]

    X = reshape(X, n_timesteps, 1)
    Y = reshape(Y, n_timesteps, 1)
    return X, Y
end

val_set = [cu(get_seq(10)) for i in 1:100]
dataset = [cu(get_seq(10)) for i in 1:100000]

lstm_model = Chain(LSTM(10, 10), Dense(10, 10, σ)) |> gpu

import Flux.binarycrossentropy
binarycrossentropy(ŷ, y; ϵ=eps(ŷ)) = -y*log(ŷ + ϵ) - (1 - y)*log(1 - ŷ + ϵ)

function loss(x, y)
    ŷ = lstm_model(x)
    l = sum([binarycrossentropy(x...) for x in zip(ŷ, y)])
    Flux.reset!(lstm_model)
    return l
end


ps = Flux.params(lstm_model)
opt = ADAM()
eval_cb = () -> @show(mean([loss(val...) for val in val_set]))

@epochs 10 Flux.train!(loss, ps, dataset, opt, cb=Flux.throttle(eval_cb, 30))


predict(x) = x <= 0.5 ? 0 : 1

X_test, Y_test = get_seq(10)

Y_hat=lstm_model(X_test) |> cpu

loss(X_test, Y_test)
[predict.(Y_hat) Y_test]


function padding(in_height, in_width, strides, filter_height, filter_width)
    out_height = ceil(in_height / strides[2])
    out_width = ceil(in_width / strides[1])
    pad_along_height = max((out_height - 1) * strides[2] + filter_height - in_height, 0)
    pad_along_width = max((out_width - 1) * strides[1] + filter_width - in_width, 0)
    pad_top = floor(pad_along_height / 2)
    pad_bottom = pad_along_height - pad_top
    pad_left = floor(pad_along_width / 2)
    pad_right = pad_along_width - pad_left
    return (pad_top, pad_bottom, pad_left, pad_right)
end


using Flux
using Flux: onehot, chunk, batchseq, throttle, crossentropy
using StatsBase: wsample
using Base.Iterators: partition

cd(@__DIR__)

isfile("input.txt") ||
  download("https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
           "input.txt")

text = collect(String(read("input.txt")))
alphabet = [unique(text)..., '_']
text = map(ch -> onehot(ch, alphabet), text)
stop = onehot('_', alphabet)

N = length(alphabet)
seqlen = 50
nbatch = 50

Xs = collect(partition(batchseq(chunk(text, nbatch), stop), seqlen))
Ys = collect(partition(batchseq(chunk(text[2:end], nbatch), stop), seqlen))

m = Chain(
  LSTM(N, 128),
  LSTM(128, 128),
  Dense(128, N),
  softmax)

m = gpu(m)
Xs[1][1]
m(Xs[1][1])
lstm_input[1]

lstm_fw_cell = LSTM(512,ATTN_NUM_HIDDEN*ATTN_NUM_LAYERS) |> gpu
lstm_bw_cell = LSTM(512,ATTN_NUM_HIDDEN*ATTN_NUM_LAYERS) |> gpu
output_layer = Dense(ATTN_NUM_HIDDEN*ATTN_NUM_LAYERS*2, 512) |> gpu
BLSTM(x) = vcat.(lstm_fw_cell.(x), Flux.reverse(lstm_bw_cell.(Flux.reverse(x, dims=1)), dims=1))
BLSTM(lstm_input)


decoder_inputs

model(x) = softmax.(output_layer.(BLSTM(x)))
a = model(lstm_input)
lstm_fw_cell(lstm_input[end])
