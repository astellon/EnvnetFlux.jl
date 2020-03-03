using Flux
using Flux: crossentropy, @epochs
using CuArrays

using DataLoaders

include("envnet.jl")
include("esc50.jl")

function main()
  model = Envnetv2(50)

  datapath = "/home/astellon/workspace/python/envnet-pytorch/ESC-50-master/audio"
  traindataset = ESC50Dataset(datapath, [1,2,3])
  valdataset   = ESC50Dataset(datapath, [4])

  trainloader = DataLoader(traindataset, 16, ntasks=8)
  valloader   = DataLoader(valdataset,   16, ntasks=8, shuffle = false)

  optimizer = Nesterov(0.01, 0.9)

  function softmax_cross_entropy(x, y)
    ỹ = softmax(model(x))
    crossentropy(ỹ, y)
  end

  function cb()
    println("training")
  end

  @epochs 2 Flux.train!(softmax_cross_entropy, params(model), trainloader, optimizer, cb = cb)
end

main()