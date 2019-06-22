mutable struct DataGen
    GO_ID
    EOS_ID
    PAD_ID
    CHARSMAP
end

function set_full_ascii_char_map()
    return [Char(i) for i in 32:126]
end

DataGen() = DataGen(:GO,:EOS, :PAD,set_full_ascii_char_map())

function embed_image_label(image_label; max_prediction_length=19)
    data_map = DataGen()
    embedding = vcat(data_map.GO_ID,
        [findfirst(isequal(i),data_map.CHARSMAP) for i in image_label],
        [data_map.PAD_ID for i in 1:max_prediction_length-length(image_label)],
        data_map.EOS_ID)
    return embedding
end
