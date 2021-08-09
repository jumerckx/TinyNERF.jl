using TinyNERF, Flux, CUDA, JSON, Images, StatsBase

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


mlp = TinyNERF.MLP()|>gpu
opt = ADAM(5e-4)

TinyNERF.train!(mlp, opt, images, poses, focal, ṙ, testimg, testpose; n_samples=64, batch_size=256, i_plot=500, n_iters=2000)