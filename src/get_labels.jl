using Glob: glob
using Flux: onehotbatch

function get_labels(folder_path)
    embed_labels(x) = onehotbatch(x, [collect('0':'9')...,'α','β'])
    label_names = glob("*.json", folder_path)
    N_longest = 19
    labels = []
    for label_name in label_names
        file = open(label_name, "r")
        label = readlines(file)[1][2:end-1]
        label = label * 'α' * repeat('β', 19-length(label))
        push!(labels, gpu(embed_labels(label)))
        close(file)
    end
    return labels
end
