using Flux
using Flux: crossentropy, @epochs
using CuArrays

using DataLoaders

include("envnet.jl")
include("esc50.jl")

function mytrain!(loss, ps, data, opt)
  ps = Flux.Params(ps)
  for (x, y) in data
    x = x |> gpu
    y = y |> gpu
    gs = gradient(ps) do
      training_loss = loss(x, y)
      return training_loss
    end
    println("train loss: $(loss(x, y))")
    Flux.update!(opt, ps, gs)
  end
end

function main()
  model = Envnetv2(50) |> gpu

  datapath = downloadesc()
  traindataset = ESC50Dataset(datapath, [1,2,3])
  valdataset   = ESC50Dataset(datapath, [4])

  trainloader = DataLoader(traindataset, 16, ntasks=8)
  valloader   = DataLoader(valdataset,   16, ntasks=8, shuffle = false)

  optimizer = Nesterov(0.01, 0.9)

  function softmax_cross_entropy(x, y)
    ỹ = softmax(model(x))
    crossentropy(ỹ, y)
  end

  @epochs 2 mytrain!(softmax_cross_entropy, params(model), trainloader, optimizer)
end

main()