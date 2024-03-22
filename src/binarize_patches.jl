#import ImageIO
import Pkg
Pkg.activate("/faststore/Data/Imagenet")
import ColorTypes
import Statistics, StatsBase
import DataStructures
import FileIO
import NPZ
import WAV
import VideoIO
import ThreadsX


dim_n_patch(imsz, psz, pstep) = 1 + floor((imsz - psz) ÷ pstep) |> Int
unsqueeze_last_dim(arr) = reshape(arr, (size(arr)..., 1))
read_color_image(path::String) = FileIO.load(path) |> convert_pix_arr_no_alpha
read_color_image_alpha(path::String) = FileIO.load(path) |> convert_pix_arr_alpha
read_gray_image(path::String) = FileIO.load(path) |> convert_pix_arr_gray
read_color_video(path::String) = permutedims(cat((VideoIO.load(path) .|> convert_pix_arr_no_alpha .|> unsqueeze_last_dim)..., dims=4), (1, 2, 4, 3))
read_wav(path::String) = WAV.wavread(path)[1]
read_np_color2gray_video(path::String) = NPZ.npzread(path) |> compress_to_Y
zip2(a...) = [[item[i] for item in a] for i in 1:min(length.(a)...)]


function compress_to_Y(arr::Array)
    size(arr)[end] == 3 || error("arr should have last dimension size 3.")
    inds = ((:) for _ in 1:length(size(arr))-1)
    0.299 .* arr[inds..., 1] + 0.587 .* arr[inds..., 2] + 0.114 .* arr[inds..., 3]
end


function get_size_from_np(fpath::String)
    open(fpath) do f
        line = readline(f)
        shape_start = findfirst("shape", line)[end]+1
        line = line[shape_start:end]
        obrace_i = findfirst("(", line)[end]+1
        cbrace_i = findfirst(")", line)[end]-1
        shape_str = line[obrace_i:cbrace_i]
        nums = split(shape_str, ", ")
        shape = Tuple(parse.(Int, nums))
    end
end


function downsample(im)
    im_size = size(im)
    im_size = ([sz-sz%2 for sz in im_size[1:end-1]]..., im_size[end]) # TODO check if this helps odd sizes and doesn't change even sizes
    ndim = length(im_size) - 1
    out = (im[((start:2:sz) for (start, sz) in zip(starts, im_size[1:end-1]))..., :] for starts in Iterators.product([1:2 for _ in 1:ndim]...)) |> sum
    out ./ 2^ndim
end


function convert_pix_arr_gray(image)
    gr = Float32.(ColorTypes.gray.(image))
    unsqueeze_last_dim(gr)
end


function convert_pix_arr_no_alpha(image)
    r = ColorTypes.red.(image)
    g = ColorTypes.green.(image)
    b = ColorTypes.blue.(image)
    im = cat(r, g, b, dims=3)
    Float32.(im)
end


function convert_pix_arr_alpha(image)
    a = Float32.(ColorTypes.alpha.(image))
    unsqueeze_last_dim(a)
end


function convert_pix_arr(image)
    rgb = convert_pix_arr_no_alpha(image)
    a = convert_pix_arr_alpha(image)
    im = cat(rgb, a, dims=3)
    Float32.(im)
end


function ternarize_value!(bin, val, a, i)
    if val > a
        bin[(i-1)*2+1] = 1
    elseif val < -a
        bin[(i-1)*2+2] = 1 
    end
end


function ternarize_patch(bin, patch, a=0.5)
    patch_dim = size(patch)
    normed = zeros(patch_dim)
    for i in 1:size(patch)[end] # remember that last dim is channel, always normalize channel-wise
        ind = ([(:) for _ in 1:length(patch_dim)-1]..., i)
        normed[ind...] = (patch[ind...] .- Statistics.mean(patch[ind...])) ./ Statistics.std(patch[ind...])
    end
    for (i, val) in enumerate(normed)
        ternarize_value!(bin, val, a, i)
    end
    return bin
end


function patch_size_inds(arr, patch_side::Int, step::Int, sample_prob::Float64)
    dim_sizes = size(arr)
    nchannels = dim_sizes[end]
    num_patch = count_num_patch(arr, patch_side, step)
    sample_indicator = rand(num_patch) .< sample_prob
    num_patch = sum(sample_indicator)
    patch_sz = patch_side ^ (length(dim_sizes) - 1) * nchannels  # TODO binarize first patch, get length of patch to get patch_sz
    # index into the product iterator with sample_indicator
    start_inds = collect(Iterators.product([1:step:d-patch_side+1 for d in dim_sizes[1:end-1]]...))[:][sample_indicator]
    return patch_sz, num_patch, start_inds
end


function extract_binary_patches(arr, patch_side::Int, step::Int, sample_prob::Float64, bin_func)
    patch_sz, num_patch, start_inds = patch_size_inds(arr, patch_side, step, sample_prob)
    #binaries = zeros(Int8, patch_sz * 2, num_patch)
    binaries = Vector{Int8}[]
    for (i, start_ind) in enumerate(start_inds)
        ind = (s:s+patch_side-1 for s in start_ind)
        push!(binaries, bin_func(zeros(patch_sz * 2), arr[ind..., :]))
        #binaries[:, i] = bin_func(binaries[:, i], arr[ind..., :])
    end
    return binaries
end


function extract_binary_patches_thread(arr, patch_side::Int, step::Int, sample_prob::Float64, bin_func)
    patch_sz, num_patch, start_inds = patch_size_inds(arr, patch_side, step, sample_prob)
    bin_th = [[] for _ in 1:Threads.nthreads()]
    # for large samples (video) to increase memory-efficiency
    println("Start extraction")
    Threads.@threads for start_ind in start_inds
        ind = (s:s+patch_side-1 for s in start_ind)
        push!(bin_th[Threads.threadid()], bin_func(zeros(patch_sz * 2), arr[ind..., :]))
    end
    collect(Iterators.flatten(bin_th))
    # TODO async put the result into an output channel
end


function patchcnt2arr(patch_counts)
    patch_vals = collect(keys(patch_counts))
    counts = collect(values(patch_counts))
    patch_vals = hcat(patch_vals...)'
    return patch_vals .|> Int8, counts .|> Int64 # try to compress a bit lol
end


function count_bin_patches(patch_binary)
    counts = DataStructures.DefaultDict{Vector{Int8}, Int64}(0)
    #for i in 1:size(patch_binary)[1]
    #    counts[patch_binary[i, :]] += 1
    for bin in patch_binary
        counts[bin] += 1
    end
    return counts
end


function extract_dir_patches(full_arr, patch_side::Int, step::Int, multichannel::Bool, sample_probs::Vector{Pair{Int, F}}, read_func, bin_func, down_func=identity, bigfile=false) where F<:AbstractFloat
    # collect counts for each sample probs, return
    per_ds_patch_counts = []
    for (numdown, sp) in sample_probs
        local downsample_func = identity
        for i in 1:numdown
            downsample_func = downsample_func ∘ down_func
        end
        arr = down_wrapper(full_arr, multichannel, downsample_func)  # TODO potentially can optimise this to only perform 1 downsample each loop
        if bigfile
            patch_binary = extract_binary_patches_thread(arr, patch_side, step, sp, bin_func)
            # TODO async take! item from the output channel to push into per_ds_patch_counts
        else
            patch_binary = extract_binary_patches(arr, patch_side, step, sp, bin_func)
        end
        push!(per_ds_patch_counts, patch_binary)
    end
    return per_ds_patch_counts
end


function create_downsampled_dset(base_src_dir_path::String, base_save_dir_path::String, class_dir::String, multichannel::Bool, patch_side::Int, step::Int, read_func, bin_func, down_func, bigfiles::Bool, sample_probs::Vector{Pair{Int, F}}=[0=>1.0]) where F<:AbstractFloat
    patch_side > 0 && step > 0 || error("Patch parameters must be positive.")
    # put downsample in inner loop to reduce number of times files are read out
    src_dir_path = joinpath(base_src_dir_path, class_dir)
    fnames = readdir(src_dir_path)
    if bigfiles
        local merged_patch_counts = Dict(nd=>DataStructures.DefaultDict{Vector{Int8}, Int64}(0) for (nd, _) in sample_probs)
        for i in 1:length(fnames)
            fpath = joinpath(src_dir_path, fnames[i])
            println(fpath)
            arr = read_func(fpath)
            patch_counts = extract_dir_patches(arr, patch_side, step, multichannel, sample_probs, read_func, bin_func, down_func, bigfiles)
            # start counting each scale here
            println("Start counting")
            for (j, numdown) in enumerate(first.(sample_probs))
                patches = ThreadsX.unique(patch_counts[j])
                for p in patches
                    merged_patch_counts[numdown][p] += ThreadsX.count(x->x==p, patch_counts[j])
                end
            end
        end
    else
        local scale_patch_counts = [[] for _ in 1:Threads.nthreads()]
        Threads.@threads for i in 1:length(fnames)
            fpath = joinpath(src_dir_path, fnames[i])
            arr = read_func(fpath)
            patch_counts = extract_dir_patches(arr, patch_side, step, multichannel, sample_probs, read_func, bin_func, down_func, bigfiles)
            push!(scale_patch_counts[Threads.threadid()], patch_counts)
        end
        # flatten
        scale_patch_counts = collect(Iterators.flatten(scale_patch_counts))
        # zip results (since they're per file, per downsample, we want per downsample, per file)
        scale_patch_counts = zip2(scale_patch_counts...)
        merged_patch_counts = Dict()
        for (i, numdown) in enumerate(first.(sample_probs))
            patch_count_per_scale = scale_patch_counts[i]
            mpc = DataStructures.DefaultDict(0)
            for patch_count in patch_count_per_scale
                pc = count_bin_patches(patch_count)
                for (patch, cnt) in pc
                    mpc[patch] += cnt
                end
            end
            merged_patch_counts[numdown] = mpc
        end
    end
    # merge results (loop over unique and sum for each file), then save to file
    for (numdown, patch_count_per_scale) in merged_patch_counts
        patch_vals, counts = patchcnt2arr(patch_count_per_scale)
        save_path = joinpath(base_save_dir_path, "$(class_dir)_$(patch_side)_$(numdown)xdown.npy")
        println(save_path, size(patch_vals), " ", size(counts))
        NPZ.npzwrite(save_path; patches=patch_vals, counts=counts)
    end
end


count_num_patch(arr_sz::Tuple, patch_side::Int, step::Int) = dim_n_patch.(arr_sz, patch_side, step) |> prod
arr_size_no_channel(arr::Array) = size(arr) |> arr_size_no_channel
arr_size_no_channel(arr_sz::Tuple) = arr_sz[1:end-1]


function count_num_patch(arr::Array, patch_side::Int, step::Int)
    arr_sz = arr_size_no_channel(arr)
    num_patch = count_num_patch(arr_sz, patch_side, step)
end


function unsqueeze_no_channel(arr::Array, multichannel::Bool)
    if !multichannel
        return unsqueeze_last_dim(arr)
    end
    return arr
end


function unsqueeze_no_channel(arr_sz::Tuple, multichannel::Bool)
    if !multichannel
        return (arr_sz..., 1)
    end
    return arr_sz
end


function down_wrapper(arr, multichannel::Bool, downsample_func)
    arr = unsqueeze_no_channel(arr, multichannel)
    downsample_func(arr)
end


function count_num_patch_down_by_ratio(fpath::String, ratios::Tuple, patch_side::Int, step::Int, get_size_func)
    arr_sz = get_size_func(fpath)
    arr_sz = arr_size_no_channel(arr_sz)
    down_sz = arr_sz ./ ratios .|> floor .|> Int
    count_num_patch(down_sz, patch_side, step)
end


function count_num_patch(src_dir_path::String, patch_side::Int, step::Int, multichannel::Bool, read_func, get_size_func, downsample_func=identity)
    fnames = readdir(src_dir_path)
    patch_counts = [[] for _ in 1:Threads.nthreads()]
    # take first one, downsample and see the factor, then determine the size of array based on factor and number of downsamples
    orig = unsqueeze_no_channel(read_func(joinpath(src_dir_path, fnames[1])), multichannel)
    trydown = down_wrapper(orig, true, downsample_func) # prev line already ensures channel dim is present
    ratios = (size(orig) ./ size(trydown))[1:end-1]
    println(ratios)
    Threads.@threads for i in 1:length(fnames)
        fpath = joinpath(src_dir_path, fnames[i])
        np = count_num_patch_down_by_ratio(fpath, ratios, patch_side, step, get_size_func)
        push!(patch_counts[Threads.threadid()], np)
    end
    collect(Iterators.flatten(patch_counts))
end


function num_patch_per_class(class_dir_paths::Vector{String}, patch_side::Int, step::Int, read_func, down_func, get_size_func, numdown::Int)
    # downsample each item numdown times, compute number of patches in total
    patch_side > 0 && step > 0 || error("Patch parameters must be positive.")
    local downf = identity
    for _ in 1:numdown
        downf = downf ∘ down_func
    end
    num_patch_per_cls_per_sample = []
    for src_dir_path in class_dir_paths
        num_patches = count_num_patch(src_dir_path, patch_side, step, multichannel, read_func, get_size_func, downf)
        push!(num_patch_per_cls_per_sample, num_patches)
    end
    num_patch_per_cls_per_sample
end


function prob_max_downsample(base_src_dir_path::String, patch_side::Int, step::Int, read_func, down_func, get_size_func, atleast::Int)
    numdown = 0
    class_dirs = readdir(base_src_dir_path)
    class_dir_paths = joinpath.(base_src_dir_path, class_dirs)
    num_patch_per_cls_scale = [num_patch_per_class(class_dir_paths, patch_side, step, read_func, down_func, get_size_func, numdown)]
    while min(sum.(num_patch_per_cls_scale[end])...) > atleast
        numdown += 1
        push!(num_patch_per_cls_scale, num_patch_per_class(class_dir_paths, patch_side, step, read_func, down_func, get_size_func, numdown))
    end
    pop!(num_patch_per_cls_scale)
    num_patch_per_cls = [sum.(per_cls) for per_cls in num_patch_per_cls_scale]
    target_num = min([min(per_cls...) for per_cls in num_patch_per_cls]...)
    probs = Vector{Float64}[target_num ./ per_cls for per_cls in num_patch_per_cls]
    return probs, class_dirs
end


function create_max_downsampled_dset(base_src_dir_path::String, base_save_dir_path::String, multichannel::Bool, patch_side::Int, step::Int, read_func, bin_func, down_func, bigfiles::Bool, atleast::Int)
    get_size_func = size ∘ read_func
    create_max_downsampled_dset(base_src_dir_path, base_save_dir_path, multichannel, patch_side, step, read_func, bin_func, down_func, get_size_func, bigfiles, atleast)
end


function create_max_downsampled_dset(base_src_dir_path::String, base_save_dir_path::String, multichannel::Bool, patch_side::Int, step::Int, read_func, bin_func, down_func, get_size_func, bigfiles::Bool, atleast::Int)
    probs, class_dirs = prob_max_downsample(base_src_dir_path, patch_side, step, read_func, down_func, get_size_func, atleast)
    cls_probs = [[i-1=>val for (i, val) in enumerate(p)] for p in collect(zip(probs...))]
    println("class probs ", cls_probs, " ", typeof(cls_probs))
    create_downsampled_dset_probs(base_src_dir_path, base_save_dir_path, class_dirs, multichannel, patch_side, step, read_func, bin_func, down_func, bigfiles, cls_probs)
end


function create_downsampled_dset_probs(base_src_dir_path::String, base_save_dir_path::String, class_dirs::Vector{String}, multichannel::Bool, patch_side::Int, step::Int, read_func, bin_func, down_func, bigfiles::Bool, probs::Vector{Vector{Pair{Int, F}}}) where F<:AbstractFloat
    @sync for (cls_dir, cls_probs) in zip(class_dirs, probs)
        @async create_downsampled_dset(base_src_dir_path, base_save_dir_path, cls_dir, multichannel, patch_side, step, read_func, bin_func, down_func, bigfiles, cls_probs)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    basedir = "ILSVRC/Data/CLS-LOC/train/"
    outdir = "2x2_patch_counts"
    # TODO specify read function to combine videos/images/audio into an np array
    patch_side = 2
    step = 1
    numdown = 0
    multichannel = true
    processed = split.(readdir(outdir), ".") .|> first
    patch_counts = DataStructures.DefaultDict(0)
    for dir in readdir(basedir)
        if dir in processed
            continue
        end
        create_downsampled_dset(joinpath(basedir, dir), outdir, multichannel, patch_side, step, read_color_image, ternarize_patch, downsample, numdown)
        #pc = extract_dir_patches(joinpath(basedir, dir), patch_side, step, multichannel, convert_pix_arr_no_alpha, ternarize_patch)

        #for (patch, count) in pc
        #    patch_counts[patch] += count
        #end
    end
    #patch_count_arr = patchcnt2arr(patch_counts)
    #NPZ.npzwrite("imagenet_2x2_rgb.npy", patch_count_arr)
    #println(patch_count_arr |> size)
end

