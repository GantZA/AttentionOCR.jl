using Base.Iterators: partition

function batch(data, labels, batch_size=10)
    N = size(data,1)
    indx = partition(1:size(data,1), batch_size)
    batches = [data[i] for i in indx]
    label_batches = [labels[i] for i in indx]
    return batches, label_batches
end
