# code from https://github.com/FluxML/Zygote.jl/pull/294

import Flux.Zygote

reversesumscan_(x::AbstractArray; dims::Integer) =
    reverse(cumsum(reverse(x, dims=dims), dims=dims), dims=dims)

Zygote.@adjoint function cumprod(A::AbstractArray; dims::Integer)
    return cumprod(A; dims=dims), function (∆)
        dim_size = size(A, dims)
        if dim_size == 1
            return (∆, nothing)
        end
        ndims = length(size(A))

        # Simple case with nonzero elements in the input
        if all(A .!= 0)
            output = cumprod(A, dims=dims)
            return (reversesumscan_(∆ .* output, dims=dims) ./ A, nothing)
        end

        grad_input = similar(A)

        pre_idx = Any[Colon() for _=1:dims-1]
        post_idx = [Colon() for _=dims+1:ndims]
        for k=1:dim_size

            if k == 1
                idx_kplus1 = vcat(pre_idx, [k+1:dim_size], post_idx)
                prods_from_k_plus_1 = cumprod(A[idx_kplus1...], dims=dims)

                # We add 1s to the omitted_products of the first element
                ones_size = [size(A)...]
                ones_size[dims] = 1
                ones_ = ones(eltype(A), ones_size...)

                omitted_products = cat(ones_, prods_from_k_plus_1, dims=dims)

            elseif k == dim_size
                idx_kminus1 = vcat(pre_idx, [1:k-1], post_idx)
                prods_until_k = prod(A[idx_kminus1...], dims=dims)
                omitted_products = prods_until_k

            else
                idx_kplus1 = vcat(pre_idx, [k+1:dim_size], post_idx)
                prods_from_k_plus_1 = cumprod(A[idx_kplus1...], dims=dims)

                idx_kminus1 = vcat(pre_idx, [1:k-1], post_idx)
                prods_until_k = prod(A[idx_kminus1...], dims=dims)

                omitted_products = prods_until_k .* prods_from_k_plus_1
                omitted_products = cat(prods_until_k, omitted_products, dims=dims)
            end

            @assert size(omitted_products, dims) == dim_size - k + 1 "$(size(omitted_products, dims)) != $(dim_size - k - 1)"

            idx_ktodimsize = vcat(pre_idx, [k:dim_size], post_idx)
            idx_k = vcat(pre_idx, [k:k], post_idx)

            grad_input[idx_k...] .= sum(∆[idx_ktodimsize...] .* omitted_products, dims=dims)
        end
        return (grad_input, nothing)
    end
end