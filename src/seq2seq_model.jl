using Flux
using Flux: @epochs, binarycrossentropy, reverse
using Statistics


# some constants to be used for the model
HIDDEN = 128
LEARNING_RATE = 0.1
DROPOUT = 0.2
BATCH_SIZE = 32

struct Encoder
    embedding
    rnn
    out
end
Encoder(voc_size::Integer, h_size::Integer=HIDDEN) = Encoder(
    param(Flux.glorot_uniform(h_size, voc_size)),
    GRU(h_size, h_size),
    Dense(h_size, h_size))
function (e::Encoder)(x; dropout=0)
    x = map(y-> Dropout(dropout)(e.embedding*y), x)
    enc_outputs = e.rnn.(x)
    h = e.out(enc_outputs[end])
    return(enc_outputs, h)
end
Flux.@treelike Encoder

struct Decoder
    embedding
    attention
    rnn
    output
end
Decoder(h_size, voc_size) = Decoder(
    param(Flux.glorot_uniform(h_size, voc_size)),
    Attention(h_size),
    GRU(h_size*2, h_size),
    Dense(h_size, voc_size, relu))
function (d::Decoder)(x, enc_outputs;dropout=0)
    x = d.embedding * x
    x = Dropout(dropout)(x)
    decoder_state = d.rnn.state
    context = d.attention(enc_outputs, decoder_state)
    x = d.rnn([x; context])
    x = softmax(d.output(x))
    return(x)
end
Flux.@treelike Decoder

struct Attention
    W1
    W2
    v
end
Attention(h_size) = Attention(
    Dense(h_size, h_size),
    Dense(h_size, h_size),
    param(Flux.glorot_uniform(1, h_size)))
function (a::Attention)(enc_outputs, d)
    U = [a.v*tanh.(x) for x in a.W1.(enc_outputs) .+ [a.W2(d)]]
    A = softmax(vcat(U...))
    out = sum([gpu(collect(A[i,:]')) .* h for (i, h) in enumerate(enc_outputs)])
end
Flux.@treelike Attention


function model_seq_2_seq(encoder::Encoder, decoder::Decoder, x, y;
    teacher_forcing = 0.5, dropout=DROPOUT, voc_size=12)

    total_loss = 0
    max_length = size(y,1)
    batch_size = size(x[1],2)
    Flux.reset!.([encoder, decoder])
    enc_outputs, h = encoder(x; dropout=dropout)
    decoder_input = Flux.onehotbatch(ones(batch_size), [1:voc_size...])
    decoder.rnn.state = h
    for i in 1:max_length
        use_teacher_forcing = rand() < teacher_forcing
        decoder_output = decoder(decoder_input, enc_outputs; dropout=dropout)
        total_loss += loss(decoder_output, y[i])
        if use_teacher_forcing
            decoder_input = y[i]
        else
            decoder_input = Flux.onehotbatch(Flux.onecold(decoder_output.data), [1:voc_size...])
        end
    end
    return(total_loss)
end

lossmask = ones(12) |> gpu
lossmask[12] = 0f0
loss(logits, target) = Flux.crossentropy(logits, target; weight=lossmask)
