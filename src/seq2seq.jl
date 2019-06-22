using Flux

HIDDEN = 128

function extract_argmax_and_embed(embedding, output_projection)

        function loop_function(prev, _)
                if output_projection != nothing
                        W = param(output_projection[1])
                        b = param(output_projection[2])
                        layer(x) = W * x .+ b
                        prev = layer(prev)
                        prev_symbol = map(argmax, eachcol(prev))

                        emb_prev = embedding[prev_symbol]

                        return emb_prev
                end
        end
        return loop_function
end

function attention_decoder(decoder_inputs, initial_state, attention_states,
        cell, ouput_size=nothing, num_heads=1, loop_function=nothing,
        dtype=Float32, scope=nothing, initial_state_attention=false,
        attn_num_hidden=128)

        @assert num_heads == 1
        if ouput_size == nothing
                output_size = size(cell.cell.Wh,2)
        end

        batch_size = size(decoder_inputs[1],1)
        attn_length = size(attention_states,2)
        attn_size = size(attention_states,3)

        hidden = reshape(attention_states, (:, attn_length, 1, attn_size))
        hidden_features = []
        v = []
        attention_vec_size = attn_size
        for i in 1:num_heads
                append!(hidden_features,
                        Conv((), attn_size=>attention_vec_size),
                        stride=[1,1], pad=[1,1]
        end

end

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
    U = [a.v*tanh.(x) for x in a.W1.(enc_outputs).+[a.W2(d)]]
    A = softmax(vcat(U...))
    out = sum([gpu(collect(A[i, :]')) .* h for (i, h) in enumerate(enc_outputs)])
end
Flux.@treelike Attention

struct Encoder
    embedding
    rnn
    out
end
Encoder(voc_size::Integer; h_size::Integer=HIDDEN) = Encoder(
    param(Flux.glorot_uniform(h_size, voc_size)),
    GRU(h_size, h_size),
    Dense(h_size, h_size))
function (e::Encoder)(x; dropout=0)
    x = map(x->Dropout(dropout)(e.embedding*x), x)
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
function (d::Decoder)(x, enc_outputs; dropout=0)
    x = d.embedding * x
    x = Dropout(dropout)(x)
    decoder_state = d.rnn.state
    context = d.attention(enc_outputs, decoder_state)
    x = d.rnn([x; context])
    x = softmax(d.output(x))
    return(x)
end
Flux.@treelike Decoder

test_enc = Encoder(19)
test_dec = Decoder(HIDDEN, 19)


test_enc()

function basic_LSTM_seq2seq(encoder::Encoder, decoder::Decoder, x, y;
    teacher_forcing = 0.5, dropout=DROPOUT, voc_size=fr.n_words)
    total_loss = 0
    max_length = length(y)
    batch_size = size(x[1], 2)
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


using Flux
Flux.glorot_uniform()

dense_layer(x) = Dense(size(x))
test = rand(9)
map(argmax, eachcol(test))
glorot_uniform
@assert 1 == 2.0
if test == nothing
        print("hello")
end

RNN()
a = LSTM(4, 10)
size(a.cell.Wh,2)

test
reshape(test, (-1,10))
test = reshape(test,(3,3))
test = rand(2,2,2)
test = reshape(test,(2,4))

test = rand(3,2,3)
reshape(test, (2,:))
buckets
lstm_input = [cnn_output[i, :, :] for i = 1:buckets[1]]
cnn_output
139/21
cnn_output
buckets
function model_with_buckets(
    encoder_inputs_tensor, decoder_inputs, targets, weights,
    buckets, seq2seq, softmax_loss_function, per_example_loss)

    # The seq2seq argument is a function that defines a sequence-to-sequence model
    # e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(x, y, rnn_cell.GRUCell(24))

    bucket = buckets[1]
    encoder_inputs = [encoder_inputs_tensor[i,:,:] for i=1:bucket]
    encoder_inputs =
end
