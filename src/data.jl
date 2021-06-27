using Scratch, NPZ

download_cache = ""

function __init__()
    global download_cache = @get_scratch!("downloaded_files")
end

function download_data(url)
    fname = joinpath(download_cache, basename(url))
    if !isfile(fname)
        download(url, fname)
    end
    return fname
end

function get_data()
    fname = download_data("https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz")
    npzread(fname)
end