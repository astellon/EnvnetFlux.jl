using Statistics
using Flux
using Flux: crossentropy, onecold
using CuArrays

using Printf

using DataLoaders

include("envnet.jl")
include("esc50.jl")

function lrschedule(epoch)
  if epoch < 5
    return 0.001
  elseif epoch < 300
    return 0.01
  elseif epoch < 600
    return 0.001
  else
    return 0.0001
  end
end

function mytrain!(loss, ps, data, opt)
  ps = Flux.Params(ps)
  nitr = length(data)
  for (i, (x, y)) in enumerate(data)
    x = x |> gpu
    y = y |> gpu
    gs = gradient(ps) do
      training_loss = loss(x, y)
      return training_loss
    end
    @printf "\t[%2d/%2d] train loss: %6.5f\n" i nitr loss(x, y)
    Flux.update!(opt, ps, gs)
  end
end

function validation(model, data)
  correct_all = 0
  all = 0
  for (i, (x, y)) in enumerate(data)
    x = x |> gpu
    a = onecold(cpu(model(x))) .== onecold(y)

    all += length(a)
    correct_all += sum(a)
  end

  @printf "\tvalidation accuracy: %7.4f\n" 100 * correct_all / all
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

  for epoch in 1:1000
    optimizer.eta = lrschedule(epoch)
    @printf "\nEpoch %2d/1000 @LR=%f\n" epoch optimizer.eta
    mytrain!(softmax_cross_entropy, params(model), trainloader, optimizer)
    validation(model, valloader)
  end
end

main()