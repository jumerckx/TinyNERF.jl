module TinyNERF

include("cumprod_adjoint.jl")
include("data.jl")

using Flux, CUDA, EllipsisNotation, Statistics, StatsBase, JSON, Images
using Flux: @nograd, Zygote.ignore, params, glorot_uniform

function posenc(x, L_embed)
    out = similar(x, (3 + 3 * 2 * L_embed, size(x)[2:end]...))
    out[1:3, ..] .= x
    out[4:end, ..] .= vcat((f.(2^i .* x) for i in 0:(L_embed - 1) for f in (sin, cos))...)

    return out
end
@nograd posenc

shifted_softplus(x) = softplus(x-1)

widened_sigmoid(x, ϵ) = (1+2ϵ)*sigmoid(x)-ϵ
widened_sigmoid(x) = widened_sigmoid(x, eltype(x)(0.001))

struct MLP{T}
    arch::T
end
Flux.@functor MLP

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

gaussian_2d(x, y, σx, σy, W, H) = 1/(2π*σx*σy)*exp(-(((x-W/2)/σx)^2+((y-H/2)/σy)^2) / 2)

function sample_t(directions, near, far, n_samples; randomized=true)
    t = reshape(range(near, stop=far, length=n_samples+2)[1:end-1], (1, 1, :))
    if randomized
        t = t .+ rand(1, size(directions, 2), n_samples+1) .* ((far - near)/(n_samples+1))
    end
    return t
end

function sample_t(directions::CuArray{T}, near, far, n_samples; randomized=true) where T
    t = CuArray(reshape(range(T(near), stop=far, length=n_samples+2)[1:end-1], (1, 1, :)))
    if randomized
        return t .+ (CUDA.rand(1, size(directions, 2), n_samples+1) .* T((far - near)/(n_samples+1)))
    end
    return t
end

function resample_t(t_vals, weights, n_samples; randomized=true, α=0.01)
    α = eltype(weights)(α)

    weights_pad = cat(weights[:, :, 1], weights, weights[:, :, end], dims=3) # 1 × n_rays × n_bins+2 (n_bins is the number of samples in the coarse pass.)
    weights_max = max.(weights_pad[:, :, 1:end-1], weights_pad[:, :, 2:end]) # 1 × n_rays × n_bins+1
    weights_blur = (weights_max[:, :, 1:end-1] + weights_max[:, :, 2:end]) ./ 2 # 1 × n_rays × n_bins
    weights = weights_blur .+ α

    ϵ = eltype(weights)(1e-5)
    weight_sum = sum(weights, dims=3)
    padding = max.(0, ϵ .- weight_sum)
    weights .+= padding ./ size(weights, 3)
    weight_sum += padding
    cdf = weights ./ weight_sum
    cdf = min.(1, cumsum(cdf[:, :, 1:end-1], dims=3)) # 1 × n_rays × n_bins-1
    cdf = cat(CUDA.zeros((1, size(cdf, 2))),
        cdf,
        CUDA.ones((1, size(cdf, 2))), dims=3) # 1 × n_rays × n_bins+1
    
    if randomized
        u = range(eltype(weights)(0), stop=1, length=n_samples+2)[1:end-1]
        u = reshape(u, (1, 1, 1, n_samples+1))
        u = u .+ CUDA.rand(1, size(weights, 2), 1, n_samples+1) .* (eltype(weights)(1/(n_samples+1)) - ϵ)
    else
        u = reshape(range(eltype(weights)(0), stop=1-ϵ, length=n_samples+1), (1, 1, 1, n_samples+1))
    end

    bin_indices = sum(u .>= cdf, dims=3)[:, :, 1, :] # 1 × n_rays × n_samples

    I = CartesianIndices((size(bin_indices)[1:2]..., 1)) .- CartesianIndex((0, 0, 1))

    cdf_interval = cdf[I .+ CartesianIndex.(0, 0, bin_indices)], cdf[I .+ CartesianIndex.(0, 0, bin_indices .+ 1)]
    bins_interval = t_vals[I .+ CartesianIndex.(0, 0, bin_indices)], t_vals[I .+ CartesianIndex.(0, 0, bin_indices .+ 1)]

    t = (dropdims(u, dims=3) .- cdf_interval[1]) ./ (cdf_interval[2] .- cdf_interval[1])
    return bins_interval[1] .+ t .* (bins_interval[2] .- bins_interval[1])
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

# function train!(mlp, opt, images, poses, focal, ṙ, testimg, testpose; n_samples=128, batch_size=1024, n_iters=1000, i_plot=25)
#     H, W = size(images)[[2, 3]]
#     ps = params(mlp)
#     local ℒ
    
#     for i in 1:n_iters
#         CUDA.@time begin
#             img_index = rand(1:size(images, 4))
#             pose = gpu(poses[:, :, img_index])

#             weights = ProbabilityWeights(map((x)->gaussian_2d(x..., W/6, H/6, W, H), Tuple.(CartesianIndices((H, W))))[:])

#             pixels = CartesianIndices((H, W))[sample(1:H*W, weights, batch_size, replace=false)]

#             # pixels = CartesianIndices((H, W))[rand(1:H*W, batch_size)]

#             target = gpu(images[:, :, :, img_index][:, pixels])

#             origin, directions = get_rays(pixels, H, W, focal, pose)
#             t = sample_t(directions, 2, 5, n_samples, randomized=true)
#             μ, Σ_diag, normed_directions = cast(origin, directions, ṙ, t)
            
#             gs = gradient(ps) do
#                 rgb, σ = mlp(IPE(μ, Σ_diag), posenc(normed_directions, 4))
#                 rgb, depth, acc, weights = render(rgb, σ, t, directions)
#                 ℒ = mean((rgb .- target).^2)
#                 return ℒ
#             end

#             Flux.update!(opt, ps, gs)
#         end
#         # println(i)
#         if i%i_plot == 0
#             rgbs = []
#             for chunk in 0:ceil(Integer, (H*W)/batch_size)-1
#                 pixels = CartesianIndices((H, W))[chunk*batch_size+1:min(end, (chunk+1)*batch_size)]
#                 origin, directions = get_rays(pixels, H, W, focal, gpu(testpose))
#                 t = sample_t(directions, 2, 5, n_samples, randomized=true)
#                 μ, Σ_diag, normed_directions = cast(origin, directions, ṙ, t)
            
#                 rgb, σ = mlp(IPE(μ, Σ_diag), posenc(normed_directions, 4))
#                 rgb, depth, acc, weights = render(rgb, σ, t, directions)
#                 push!(rgbs, rgb)
#             end
#             rgb = reshape(reduce(hcat, rgbs), (3, H, W))
#             display(colorview(RGB, cpu(rgb)))
#             testing_loss = mean((rgb .- testimg) .^ 2)
#             PSNR = -10 * log10(testing_loss)
#             println("training loss: $ℒ\ttesting loss: $testing_loss\tPSNR: $PSNR")
#         end
#     end
# end

function train!(mlp, opt, images, poses, focal, ṙ, testimg, testpose; n_samples=128, batch_size=1024, n_iters=1000, i_plot=25)
    H, W = size(images)[[2, 3]]
    ps = params(mlp)
    local ℒ
    
    for i in 1:n_iters
        CUDA.@time begin
            img_index = rand(1:size(images, 4))
            pose = gpu(poses[:, :, img_index])

            weights = ProbabilityWeights(map((x)->gaussian_2d(x..., W/6, H/6, W, H), Tuple.(CartesianIndices((H, W))))[:])

            pixels = CartesianIndices((H, W))[sample(1:H*W, weights, batch_size, replace=false)]

            # pixels = CartesianIndices((H, W))[rand(1:H*W, batch_size)]

            target = gpu(images[:, :, :, img_index][:, pixels])

            origin, directions = get_rays(pixels, H, W, focal, pose)
            t = sample_t(directions, 2, 5, n_samples, randomized=true)
            μ, Σ_diag, normed_directions = cast(origin, directions, ṙ, t)

            local sample_weights

            gs_coarse = gradient(ps) do
                rgb, σ = mlp(IPE(μ, Σ_diag), posenc(normed_directions, 4))
                rgb, _, _, sample_weights = render(rgb, σ, t, directions)
                ℒ = mean((rgb .- target).^2)
                return ℒ
            end
            
            t = resample_t(t, sample_weights, n_samples)
            μ, Σ_diag, normed_directions = cast(origin, directions, ṙ, t)

            gs_fine = gradient(ps) do
                rgb, σ = mlp(IPE(μ, Σ_diag), posenc(normed_directions, 4))
                rgb, _, _, _ = render(rgb, σ, t, directions)
                ℒ = mean((rgb .- target).^2)
                return ℒ
            end

            Flux.update!(opt, ps, (0.1 .* gs_coarse) .+ gs_fine)
        end
        # println(i)
        if i%i_plot == 0
            rgbs = []
            for chunk in 0:ceil(Integer, (H*W)/batch_size)-1
                pixels = CartesianIndices((H, W))[chunk*batch_size+1:min(end, (chunk+1)*batch_size)]
                origin, directions = get_rays(pixels, H, W, focal, gpu(testpose))
                t = sample_t(directions, 2, 5, n_samples, randomized=true)
                μ, Σ_diag, normed_directions = cast(origin, directions, ṙ, t)
            
                rgb, σ = mlp(IPE(μ, Σ_diag), posenc(normed_directions, 4))
                rgb, _, _, weights = render(rgb, σ, t, directions)

                t = resample_t(t, weights, n_samples)
                μ, Σ_diag, normed_directions = cast(origin, directions, ṙ, t)
            
                rgb, σ = mlp(IPE(μ, Σ_diag), posenc(normed_directions, 4))
                rgb, _, _, _ = render(rgb, σ, t, directions)

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


end
