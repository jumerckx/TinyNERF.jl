using TinyNERF, Images, Flux, CUDA, EllipsisNotation, Statistics, StatsBase, JSON
using Flux: @nograd, Zygote.ignore, params, glorot_uniform
CUDA.allowscalar(false)

transforms = JSON.parsefile("local/transforms_train.json")

W, H = 800, 800

const focal = W / (2*tan(transforms["camera_angle_x"]/2))
ṙ = tan(transforms["camera_angle_x"]/2) * 2 / W * 2/√12

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
colorview(RGB, cpu(testimg))

# function multiscale(imagetensor, n_scales, downsample, sigma=1)
#     out = [gaussian_pyramid(colorview(RGB, img), n_scales, downsample, sigma) for img in eachslice(imagetensor, dims=4)]
#     [cat(channelview.(getindex.(out, i))..., dims=4) for i in 1:n_scales+1]
# end

# multi_images = multiscale(images, 3, 2)

function posenc(x, L_embed)
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

shifted_softplus(x) = softplus(x-1)

widened_sigmoid(x, ϵ) = (1+2ϵ)*sigmoid(x)-ϵ
widened_sigmoid(x) = widened_sigmoid(x, eltype(x)(0.001))

function MLP(W::Integer=256, size_pos=96, size_dir=3*(1+2*4))
    arch = Chain(
        SkipConnection(
            Chain(
                Dense(size_pos, W, relu),
                Dense(W, W, relu),
                Dense(W, W, relu),
                Dense(W, W, relu), 
                Dense(W, W, relu)),
            (mx, x) -> vcat(mx, x)),
        Dense(W + size_pos, W, relu),
        Dense(W, W, relu),
        Dense(W, W, relu),
        Dense(W, W, identity),
        Dense(W-1+size_dir, W÷2, relu),
        Dense(W÷2, 3, widened_sigmoid))
    return MLP(arch)
end

function (m::MLP)(𝐱, normed_directions)
    x = reshape(𝐱, (size(𝐱, 1), :))
    x = m.arch[1:5](x)
    σ = reshape(shifted_softplus.(x[1, :]), (1, size(𝐱, 2), size(𝐱, 3)) )
    x = vcat(x[2:end, :], reshape(normed_directions, (size(normed_directions, 1), :) ))
    x = m.arch[6:7](x)
    rgb = reshape(x, (3, size(𝐱, 2), size(𝐱, 3)) )

    return rgb, σ
end

using BenchmarkTools
mlp = MLP()|>gpu

a = CUDA.rand(96, 1024, 128)
b = CUDA.rand(27, 1024, 128)

CUDA.@time mlp(a, b)

CUDA.@time begin
    gradient(mlp) do mlp
        rgb, s = Flux.Zygote.checkpointed(mlp, a, b)
        rgb2, s2 = Flux.Zygote.checkpointed(mlp, 2 .* a, b)

        sum(rgb .+ rgb2)
    end;
end

a = CUDA.rand(96, 1024, 20)
b = CUDA.rand(27, 1024, 20)

gs1 = gradient(Flux.params(mlp)) do
    sum(sum.(mlp(a, b)))
end 

gs2 = gradient(Flux.params(mlp)) do
    sum(sum.(mlp(a, b)))
end

gs1 .+ 0.5 .* gs2

map!

@benchmark CUDA.@sync sum(sum.(mlp(a, b)))

@benchmark CUDA.@sync blocking=false gradient(mlp) do mlp
    sum(sum.(mlp(a, b)))
end

CUDA.@time gradient(mlp) do mlp
    sum(sum.(mlp(a, b)))
end

@btime CUDA.@sync blocking=false gradient(mlp) do mlp
    sum(sum.(mlp(a, b)))
end


"""
    get_rays(pixels, H, W, focal, cam2world)

Get origin and directions of rays through `pixels`.
`pixels` should be a vector of CartesianIndices.
"""
function get_rays(pixels, H, W, focal, cam2world)
    origin = similar(cam2world, 3)
    directions = similar(cam2world, (3, length(pixels)))

    directions[1, :] = (getindex.(pixels, 2) .- W/2 .- 1/2) ./ focal
    directions[2, :] = -(getindex.(pixels, 1) .- H/2 .- 1/2) ./ focal
    directions[3, :] .= -1
    directions = cam2world[1:3, 1:3] * directions

    origin = cam2world[1:3, 4]

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

function get_t(directions, near, far, n_samples; randomized=false)
    t = reshape(range(near, stop=far, length=n_samples+1), (1, 1, :))
    if randomized
        t = t .+ rand(1, size(directions, 2), n_samples+1) .* ((far - near)/(n_samples+1))
    end
    return t
end

function get_t(directions::CuArray{T}, near, far, n_samples; randomized=false) where T
    t = CuArray(reshape(range(T(near), stop=far, length=n_samples+1), (1, 1, :)))
    if randomized
        return t .+ (CUDA.rand(1, size(directions, 2), n_samples+1) .* T((far - near)/(n_samples+1)))
    end
    return t
end

function cast(origin, directions, ṙ, t)
    T = eltype(t)

    t₀, t₁ = t[:, :, 1:(end-1)], t[:, :, 2:end]
    midpoint, halfwidth = (t₀ .+ t₁)./2, (t₁ .- t₀)./2
    

    μₜ = @. midpoint + (2*midpoint * (halfwidth^2)) / (3*(midpoint^2) + halfwidth^2)

    # https://github.com/JuliaGPU/CUDA.jl/issues/1044
    # varₜ = @. (halfwidth^2)/3 -  T(4/15) * ((halfwidth^4) * (12*(midpoint^2) - halfwidth^2) / (3*(midpoint^2) + halfwidth^2)^2)
    # varᵣ = @. ṙ^2 * ((midpoint^2)/4 + T(5/12)*(halfwidth^2) - T(4/15)*(halfwidth^4) / (3*midpoint^2 + halfwidth^2))
    
    varₜ = (halfwidth.^2)./3 .-  T(4/15) .* ((halfwidth.^4) .* (12 .*(midpoint.^2) .- halfwidth.^2) ./ (3 .*(midpoint.^2) + halfwidth.^2).^2)
    varᵣ = T(ṙ) .^ 2 .* ((midpoint .^ 2)./4 .+ T(5/12).*(halfwidth .^ 2) .- T(4/15).*(halfwidth .^ 4) ./ (3 .* (midpoint .^ 2)  .+ halfwidth .^ 2))

    μ = @. origin + μₜ * directions
    Σ_diag = varₜ .* (directions .^ 2) .+ varᵣ .* ((1 .- (directions .^ 2)) ./ sum(directions .^ 2, dims=1))
    
    normed_directions = similar(μ)
    normed_directions .= directions ./ sqrt.(sum(directions .^ 2, dims=1))

    return μ, Σ_diag, normed_directions
end

function IPE(μ, Σ_diag, degree=16)
    elt = eltype(Σ_diag)

    μᵧ = reduce(vcat, [2^i .* μ for i in 0:degree-1])
    Σᵧ_diag = reduce(vcat, [exp.((elt(-1/2) * 4^i) .* Σ_diag) for i in 0:degree-1])

    return [sin.(μᵧ) .* Σᵧ_diag; cos.(μᵧ) .* Σᵧ_diag]
end
@nograd IPE

function render(rgb, σ, t, directions)
    δ, midpoint, ϵ = ignore() do 
        t₀, t₁ = t[:, :, 1:(end-1)], t[:, :, 2:end]
        midpoint, distance = (t₀ .+ t₁)./2, (t₁ .- t₀) # 1 × n_rays × (1 or n_samples)

        δ = distance .* sqrt.(sum(directions .^ 2, dims=1)) # 1 × n_rays × n_samples

        ϵ = eps(eltype(σ))
        return δ, midpoint, ϵ
    end

    α = 1 .- exp.(-σ .* δ) # 1 × n_rays × n_samples
    weights = α .* cat(gpu(ones(1, size(α, 2))), cumprod((1 .- α .+ ϵ)[:, :, 1:end-1], dims=3), dims=3) # 1 × n_rays × n_samples

    rgb = sum(weights .* rgb, dims=3) # 3 × n_rays × 1
    depth = sum(weights .* midpoint, dims=3) # 1 × n_rays × 1
    acc = sum(weights, dims=3) # 1 × n_rays × 1

    return rgb, depth, acc, weights
end

function train!(mlp, opt, images, poses; n_samples=128, batch_size=1024, n_iters=1000, i_plot=25)
    @assert batch_size <= H*W

    ps = params(mlp)
    local ℒ
    
    for i in 1:n_iters
        img_index = rand(1:size(images, 4))
        pose = gpu(poses[:, :, img_index])

        H, W = size(images)[[2, 3]]

        weights = ProbabilityWeights(map((x)->gaussian_2d(x..., W/6, H/6), Tuple.(CartesianIndices((H, W))))[:])

        pixels = CartesianIndices((H, W))[sample(1:H*W, weights, batch_size, replace=false)]

        # pixels = CartesianIndices((H, W))[rand(1:H*W, batch_size)]

        target = gpu(images[:, :, :, img_index][:, pixels])

        origin, directions = get_rays(pixels, H, W, focal, pose)
        t = get_t(directions, 2, 5, n_samples, randomized=true)
        μ, Σ_diag, normed_directions = cast(origin, directions, ṙ, t)
        
        gs = gradient(ps) do
            rgb, σ = mlp(IPE(μ, Σ_diag), posenc(normed_directions, 4))
            rgb, depth, acc, weights = render(rgb, σ, t, directions)
            ℒ = mean((rgb .- target).^2)
            return ℒ
        end

        Flux.update!(opt, ps, gs)

        if i%i_plot == 0
            rgbs = []
            for chunk in 0:ceil(Integer, (H*W)/batch_size)-1
                pixels = CartesianIndices((H, W))[chunk*batch_size+1:min(end, (chunk+1)*batch_size)]
                origin, directions = get_rays(pixels, H, W, focal, gpu(testpose))
                t = get_t(directions, 2, 5, n_samples, randomized=true)
                μ, Σ_diag, normed_directions = cast(origin, directions, ṙ, t)
            
                rgb, σ = mlp(IPE(μ, Σ_diag), posenc(normed_directions, 4))
                rgb, depth, acc, weights = render(rgb, σ, t, directions)
                push!(rgbs, rgb)
            end
            rgb = reshape(reduce(hcat, rgbs), (3, H, W))
            display(colorview(RGB, cpu(rgb)))
            testing_loss = mean((rgb .- testimg) .^ 2)
            PSNR = -10 * log10(testing_loss)
            println("training loss: $ℒ\ttesting loss: $testing_loss\tPSNR: $PSNR")
        end
    end
end

opt = ADAM(5e-3)
mlp = MLP()|>gpu

using BenchmarkTools

# multi_images[3]

# testimg = gpu(multiscale(reshape(cpu(testimg), (3, H, W, 1)), 3, 2)[1])

# ṙ = tan(transforms["camera_angle_x"]/2) * 2 / 200 * 2/√12

CUDA.@time train!(mlp, opt, images, poses, n_samples=128, batch_size=1024, n_iters=49, i_plot=50)
CUDA.@time train!(mlp, opt, images, poses; n_samples=64, batch_size=1024, n_iters=49, i_plot=50)



function test(mlp, opt, images, poses; n_samples=128, batch_size=1024, n_iters=1000)
    ps = params(mlp)
    for i in 1:n_iters
        img_index = rand(1:size(images, 4))
        pose = gpu(poses[:, :, img_index])

        H, W = size(images)[[2, 3]]

        weights = ProbabilityWeights(map((x)->gaussian_2d(x..., W/4, H/4), Tuple.(CartesianIndices((H, W))))[:])

        pixels = CartesianIndices((H, W))[sample(1:H*W, weights, batch_size, replace=false)]

        target = gpu(images[:, :, :, img_index][:, pixels])

        origin, directions = get_rays(pixels, H, W, focal, pose)
        t = get_t(directions, 2, 5, n_samples, randomized=true)
        μ, Σ_diag, normed_directions = cast(origin, directions, ṙ, t)
        
        gs = gradient(ps) do
            rgb, σ = mlp(IPE(μ, Σ_diag), posenc(normed_directions, 4))
            rgb, depth, acc, weights = render(rgb, σ, t, directions)
            ℒ = mean((rgb .- target).^2)
            return ℒ
        end

        ps_cpy, re = Flux.destructure(mlp)

        Flux.update!(opt, ps, gs)
        if any(cpu(isnan.(Flux.destructure(mlp)[1])))
            return re(ps_cpy), IPE(μ, Σ_diag), posenc(normed_directions, 4), t, directions, target
        end
        println(i)
    end
end

opt = ADAM(5e-4)
mlp = MLP()|>gpu

mlp, a, b, t, directions, target = test(mlp, opt, images, poses, n_samples=64, batch_size=2048, n_iters=1000)

any(cpu(isnan.(Flux.destructure(mlp)[1])))

target

gs = gradient(params(Flux.destructure(mlp)[1])) do 
    rgb, σ = mlp(a, b)
    rgb, depth, acc, weights = render(rgb, σ, t, directions)
    ℒ = mean((rgb .- target).^2)
    return ℒ
end

gs.params[1]

extrema(cpu(abs.(gs.params[1])))

opt.state

any(cpu(isnan.(gs.params[1])))

c = nothing

for p in gs.params
    if any(cpu(isnan.(p)))
        c = p
        break
    end
end 

collect(values(opt.state))[1][1]

c

findall(isnan, cpu(c))

extrema.(cpu.(gs.params))

[extrema(abs.(p)) for p in cpu.(gs.params)]

cpu.(gs.params)

params(mlp)

Flux.update!(opt, params(mlp), gs)


#/------------------------------------------------------------------------------------------------------\

img_index = rand(1:size(images, 4))
pose = gpu(poses[:, :, img_index])

weights = ProbabilityWeights(map((x)->gaussian_2d(x..., W/6, H/6), Tuple.(CartesianIndices((H, W))))[:])

pixels = CartesianIndices((H, W))[sample(1:H*W, weights, 1024, replace=false)]

target = gpu(images[:, :, :, img_index][:, pixels])

CUDA.@time origin, directions = get_rays(pixels, H, W, focal, pose)
CUDA.@time t = get_t(directions, 2, 5, 128, randomized=true)
CUDA.@time μ, Σ_diag, normed_directions = cast(origin, directions, ṙ, t)
@benchmark CUDA.@sync begin
    rgb, s = mlp(IPE(μ, Σ_diag), posenc(normed_directions, 4))
    # return sum(rgb)
    rgb, depth, acc, weights = render(rgb, s, t, directions)
    ℒ = mean((rgb .- target).^2)
end

a, b = IPE(μ, Σ_diag), posenc(normed_directions, 4)

CUDA.@time gs = Flux.gradient(mlp) do mlp
    rgb, s = mlp(IPE(μ, Σ_diag), posenc(normed_directions, 4))
    # return sum(rgb)
    rgb, depth, acc, weights = render(rgb, s, t, directions)
    ℒ = mean((rgb .- target).^2)
end
#\------------------------------------------------------------------------------------------------------/

using Printf
function lr_finder(mlp, opt, images, poses; start_lr=1e-6, end_lr=1, factor=2, n_samples=128, batch_size=1024)
    losses = []
    local ℒ
    ps = params(mlp)
    i = 1
    opt.eta = start_lr

    img_index = rand(1:size(images, 4))
    pose = gpu(poses[:, :, img_index])

    weights = ProbabilityWeights(map((x)->gaussian_2d(x..., W/4, H/4), Tuple.(CartesianIndices((H, W))))[:])

    pixels = CartesianIndices((H, W))[sample(1:H*W, weights, batch_size, replace=false)]

    target = gpu(images[:, :, :, img_index][:, pixels])

    origin, directions = get_rays(pixels, H, W, focal, pose)
    t = get_t(directions, 2, 5, n_samples, randomized=true)
    μ, Σ_diag, normed_directions = cast(origin, directions, ṙ, t)

    while opt.eta <= end_lr        
        gs = gradient(ps) do
            rgb, σ = mlp(IPE(μ, Σ_diag), posenc(normed_directions, 4))
            rgb, depth, acc, weights = render(rgb, σ, t, directions)
            ℒ = mean((rgb .- target).^2)
            return ℒ
        end
        @printf "iteration %d\tη: %.2e\tloss: %.3f\n" i opt.eta ℒ
        if isnan(ℒ); break; end
        push!(losses, (opt.eta, ℒ))
        Flux.update!(opt, ps, gs)
        i += 1
        opt.eta *= factor
    end
    return losses
end