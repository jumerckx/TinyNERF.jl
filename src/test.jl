using TinyNERF, Images, Flux, CUDA, EllipsisNotation, Statistics

dataset = get_data()
images = permutedims(dataset["images"], (4, 2, 3, 1)) # Images×Height×Width×Channels -> Channels×Height×Width×Images
poses = permutedims(dataset["poses"], (2, 3, 1))
focal = dataset["focal"]

const H, W = size(images)[[2, 3]]

testimg, testpose = images[:, :, :, 102], poses[:, :, 102]
colorview(RGB, testimg)

images, poses = images[:, :, :, 1:101], poses[:, :, 1:101]

function posenc(x, L_embed=6)
    out = similar(x, (3 + 3 * 2 * L_embed, size(x, 2)))
    out[1:3, ..] .= x
    out[4:end, ..] .= vcat((f.(2^i .* x) for i in 0:(L_embed - 1) for f in (sin, cos))...)

    return out
end
Flux.@nograd posenc

function init_model(W=256; L_embed=6)
    inputsize = 3 * (1 + 2 * L_embed)
    Chain(
        SkipConnection(
            Chain(
                Dense(inputsize, W, relu),
                Dense(W, W, relu),
                Dense(W, W, relu),
                Dense(W, W, relu)),
            (mx, x) -> vcat(mx, x)),
        Dense(W + inputsize, W, relu),
        Dense(W, W, relu),
        Dense(W, W, relu),
        Dense(W, 4, identity))
end

function get_rays(H, W, focal, cam2world)
    dirs = fill(eltype(cam2world)(-1), (3, H, W))
    dirs[1, :, :] .= (((0:(W - 1)) .- W / 2) ./ focal)'
    dirs[2, :, :] .= -((0:(H - 1)) .- H / 2) ./ focal

    dirs = cam2world[1:3, 1:3] * reshape(dirs, (3, H * W))
    dirs = reshape(dirs, (3, H, W))

    origin = cam2world[1:3, 4]

    return gpu(origin), gpu(dirs)
end
Flux.@nograd get_rays

# using GLMakie
# begin
#     s = Scene()
#     cam3d!(s)
#     for p in eachslice(poses, dims=3)
#         origin, dirs = get_rays(H, W, focal, p)
#         arrows!(s,
#             [Point3(origin) for _ in 1:H÷20 for _ in 1:W÷20],
#             [Vec3(dirs[:, i*20, j*20]) for i in 1:H÷20 for j in 1:W÷20],
#             linewidth=0.01,
#             arrowsize=0,
#             lengthscale=0.5)
#     end
#     display(s)
# end

function get_z_vals(origin, dirs, near, far, N_samples, randomized=false)
    H, W = size(dirs)[[2, 3]]
    z_vals = reshape(range(near, stop=far, length=N_samples), (1, 1, 1, N_samples))
    if randomized
        z_vals = rand(1, H, W, N_samples) .* ((far - near) / N_samples) .+ z_vals
    end

    return collect(z_vals)
end
Flux.@nograd get_z_vals

function render_rays(emb_fn, network_fn, origin, dirs, near, far, N_samples, randomized=false)
    z_vals = gpu(get_z_vals(origin, dirs, near, far, N_samples, randomized))
    pts = origin .+ (dirs .* z_vals)
    raw = network_fn(emb_fn(reshape(pts, (3, :))))

    σₐ = reshape(relu.(raw[end, :]), (H, W, N_samples))
    rgb = reshape(sigmoid.(raw[1:3, :]), (3, H, W, N_samples))

    e = size(z_vals)[end] # 'end', necessary because of bug in EllipsisNotation: https://github.com/ChrisRackauckas/EllipsisNotation.jl/issues/19

    δ = (z_vals[.., 2:e] .- z_vals[.., 1:e - 1])[1, ..]
    δ = cat(δ, fill(1e10, size(δ)[[1, 2]]), dims=3)

    α = 1 .- exp.(-σₐ .* δ)
    weights = α .* cumprod(cat(gpu(ones(H, W)), (1 .- α .+ 1e-10)[:, :, 1:end - 1], dims=3), dims=3) # exclusive cumprod
    weights = reshape(weights, 1, size(weights)...)
    
    rgb_map = sum(rgb .* weights, dims=4)[.., 1]
    depth_map = sum(z_vals .* weights, dims=4)[1, .., 1]
    acc_map = sum(weights, dims=4)[1, .., 1]

    return rgb_map, depth_map, acc_map
end

# rgb, depth, acc = render_rays(posenc, gpu(init_model()), get_rays(H, W, focal, testpose)..., 2, 6, 64, true)

function train!(emb_fn, network_fn, images, poses, opt, N_samples, N_iters; i_plot=25)
    PSNRs = []
    iternums = []
    
    local training_loss
    ps = params(model)
    for i in 1:N_iters
        println("iteration $i")
        img_i = rand(1:size(images, 4))
        target = gpu(images[:, :, :, img_i])
        pose = poses[:, :, img_i]
        
        rays_o, rays_d = get_rays(H, W, focal, pose)
        
        gs = gradient(ps) do
            rgb, depth, acc = render_rays(emb_fn, network_fn, rays_o, rays_d, 2, 5, N_samples, true)
            training_loss = mean((rgb .- target).^2)
            return training_loss
        end
        Flux.update!(opt, ps, gs)
        if i%i_plot==0
            rays_o, rays_d = get_rays(H, W, focal, testpose)
            rgb, depth, acc = render_rays(posenc, model, rays_o, rays_d, 2, 6, N_samples)
            rgb = cpu(rgb)
            display(colorview(RGB, rgb))
            
            testing_loss = mean((rgb .- testimg).^2)
            PSNR = -10 * log10(testing_loss)
            println("training loss: $training_loss\ttesting loss: $testing_loss\tPSNR: $PSNR")
            push!(PSNRs, PSNR)
            push!(iternums, i)
        end
    end
    return iternums, PSNRs
end

model = gpu(init_model())
optimizer = ADAM(5e-4)

N_samples = 10
N_iters = 1000

train!(posenc, model, images, poses, optimizer, N_samples, N_iters)
