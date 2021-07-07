using TinyNERF, Images, Flux, CUDA, EllipsisNotation, Statistics, StatsBase, JSON
using Flux: @nograd, Zygote.ignore, params, glorot_uniform

transforms = JSON.parsefile("local/transforms_train.json")

const W, H = 800, 800

const focal = W / (2*tan(transforms["camera_angle_x"]/2))

poses = []
images = []

for frame in transforms["frames"]
    push!(poses, hcat(frame["transform_matrix"]...)')
    push!(images, float32.(channelview(load("./local/$(frame["file_path"]).png"))[1:3, :, :]))
end

poses = cat(poses..., dims=3)
images = cat(images..., dims=4) # channels × H × W × images

testpose = poses[:, :, 100]
testimg = gpu(images[:, :, :, 100])

poses = poses[:, :, 1:99]
images = images[:, :, :, 1:99]

function posenc(x, L_embed=6)
    out = similar(x, (3 + 3 * 2 * L_embed, size(x)[2:end]...))
    out[1:3, ..] .= x
    out[4:end, ..] .= vcat((f.(2^i .* x) for i in 0:(L_embed - 1) for f in (sin, cos))...)

    return out
end
@nograd posenc

struct MLP{T}
    arch::T
end
Flux.@functor MLP

function MLP(W::Integer=256, L_embed=6)
    inputsize = 3 * (1 + 2 * L_embed)
    directionsize = inputsize

    arch = Chain(
        SkipConnection(
            Chain(
                Dense(inputsize, W, relu),
                Dense(W, W, relu),
                Dense(W, W, relu),
                Dense(W, W, relu), 
                Dense(W, W, relu)),
            (mx, x) -> vcat(mx, x)),
        Dense(W + inputsize, W, relu),
        Dense(W, W, relu),
        Dense(W, W, relu),
        Dense(W, W, identity),
        Dense(W-1+directionsize, W÷2, relu),
        Dense(W÷2, 3, sigmoid))
    return MLP(arch)
end

function (m::MLP)(𝐱, viewing_dir)
    x = reshape(𝐱, (size(𝐱, 1), :))    
    x = m.arch[1:5](x)
    σ = reshape(relu.(x[1, :]), (1, size(𝐱, 2), size(𝐱, 3)) )
    x = vcat(x[2:end, :], reshape(viewing_dir, (size(viewing_dir, 1), :) ))
    x = m.arch[6:7](x)
    rgb = reshape(x, (3, size(𝐱, 2), size(𝐱, 3)) )

    return σ, rgb
end

"""
    get_rays(pixels, H, W, focal, cam2world)

Get origin and directions of rays through `pixels`.
`pixels` should be a vector of CartesianIndices.

"""
function get_rays(pixels, H, W, focal, cam2world)
    origin = similar(cam2world, 3)
    directions = similar(cam2world, (3, length(pixels)))

    get_rays!(origin, directions, pixels, H, W, focal, cam2world)

    return origin, directions
end

function get_rays!(origin, directions, pixels, H, W, focal, cam2world)
    directions[1, :] .= (getindex.(pixels, 2) .- W/2 .- 1/2) ./ focal
    directions[2, :] .= -(getindex.(pixels, 1) .- H/2 .- 1/2) ./ focal
    directions[3, :] .= -1
    directions .= cam2world[1:3, 1:3] * directions

    origin .= cam2world[1:3, 4]

    return origin, directions
end

gaussian_2d(x,y, σx, σy) = 1/(2π*σx*σy)*exp(-(((x-W/2)/σx)^2+((y-H/2)/σy)^2) / 2)

# using GLMakie
# begin
#     s = Scene()
#     cam3d!(s)
#     for p in eachslice(poses, dims=3)
#         weights = ProbabilityWeights(map((x)->gaussian_2d(x..., W/5, H/5), Tuple.(CartesianIndices((H, W))))[:])
#         pixels = CartesianIndices((H, W))[sample(1:H*W, weights, 10, replace=false)]
#         origin, directions = get_rays(pixels, H, W, focal, p)
#         # origin, directions = get_rays(CartesianIndices((H, W))[:], H, W, focal, p)
#         # origin, dirs = get_rays(H, W, focal, p)
#         arrows!(s,
#             [Point3(origin) for _ in 1:size(directions, 2)],
#             [Vec3(directions[:, i]) for i in 1:size(directions, 2)],
#             linewidth=0.01,
#             arrowsize=0,
#             lengthscale=0.5)
#     end
#     display(s)
# end

function get_pts(origin, directions, near, far, n_samples; randomized=false)
    z_vals = reshape(range(near, stop=far, length=n_samples), 1, 1, n_samples)
    if randomized
        z_vals = rand(1, size(directions, 2), n_samples) .* ((far - near) / n_samples) .+ z_vals
    end
    pts = origin .+ (directions .* z_vals)

    view_dirs = similar(pts)
    view_dirs .= directions ./ sqrt.(sum(directions.^2, dims=1))

    return pts, view_dirs, collect(z_vals)
end

function render(emb_fn, network_fn, pts, directions, z_vals)
    σₐ, rgb = network_fn(emb_fn(pts), emb_fn(directions))

    δ = ignore() do 
        cat(z_vals[:, :, 2:end] .- z_vals[:, :, 1:(end-1)], fill(eltype(z_vals)(1e10), size(z_vals)[[1, 2]]), dims=3)
    end
    α = 1 .- exp.(-σₐ .* δ)
    weights = α .* cat(CUDA.ones(1, size(α, 2)), cumprod((1 .- α .+ eps(eltype(α)))[:, :, 1:end-1], dims=3), dims=3) # exclusive cumprod
    
    rgb_map = sum(rgb .* weights, dims=3)[:, :, 1]
end

function train!(mlp, opt, images, poses; n_samples=64, batch_size=3000, n_iters=1000, i_plot=25)
    @assert batch_size <= H*W

    PSNRs, iternums = [], []
    
    ps = params(mlp)
    local ℒ
    
    for i in 1:n_iters
        img_index = rand(1:size(images, 4))
        pose = poses[:, :, img_index]

        weights = ProbabilityWeights(map((x)->gaussian_2d(x..., W/6, H/6), Tuple.(CartesianIndices((H, W))))[:])

        pixels = CartesianIndices((H, W))[sample(1:H*W, weights, batch_size, replace=false)]

        target = gpu(images[:, :, :, img_index][:, pixels])

        origin, directions = get_rays(pixels, H, W, focal, pose)
        pts, directions, z_vals = gpu(get_pts(origin, directions, 2, 5, n_samples, randomized=true))
        
        gs = gradient(ps) do
            prediction = render(posenc, mlp, pts, directions, z_vals)
            ℒ = mean((prediction .- target).^2)
            return ℒ
        end

        Flux.update!(opt, ps, gs)
        println(i)
        if i%i_plot == 0
            pixels = CartesianIndices((H, W))[:]
            origin, directions = get_rays(pixels, H, W, focal, testpose)
            pts, directions, z_vals = gpu(get_pts(origin, directions, 2, 5, min(16, n_samples), randomized=true))

            prediction = hcat([render(posenc, mlp, pts[:, 10000*i+1:10000*(i+1), :], directions[:, 10000*i+1:10000*(i+1), :], z_vals[:, 10000*i+1:10000*(i+1), :]) for i in 0:63]...)
            rgb = reshape(prediction, (3, H, W))
            display(colorview(RGB, cpu(rgb)))
            
            testing_loss = mean((rgb .- testimg).^2)
            PSNR = -10 * log10(testing_loss)
            println("training loss: $ℒ\ttesting loss: $testing_loss\tPSNR: $PSNR")
            push!(PSNRs, PSNR)
            push!(iternums, i)
        end
    end
    return PSNRs, iternums
end

opt = ADAM(5e-3)
mlp = MLP()|>gpu

train!(mlp, opt, images, poses, n_samples=64, batch_size=2048, n_iters=1000, i_plot=100)


pixels = CartesianIndices((H, W))[:]
origin, directions = get_rays(pixels, H, W, focal, testpose)
pts, directions, z_vals = gpu(get_pts(origin, directions, 2, 5, 128, randomized=true))

prediction = hcat([render(posenc, mlp, pts[:, 5000*i+1:5000*(i+1), :], directions[:, 5000*i+1:5000*(i+1), :], z_vals[:, 5000*i+1:5000*(i+1), :]) for i in 0:64*2-1]...)
rgb = reshape(prediction, (3, H, W))
display(colorview(RGB, cpu(rgb)))

testing_loss = mean((rgb .- testimg).^2)
PSNR = -10 * log10(testing_loss)


pts, directions, z_vals = 0, 0, 0
prediction = 0