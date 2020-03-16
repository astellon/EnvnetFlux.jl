using Statistics, Printf

using Flux,CuArrays
using Flux: crossentropy, onecold
using BSON: @save

using DataLoaders

include("envnet.jl")
include("esc50.jl")
include("scheduler.jl")

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

    l = loss(x, y)

    if isnan(l)
      @warn "loss is nan! end."
      exit(1)
    end

    @printf "\t[%2d/%2d] train loss: %6.5f\n" i nitr loss(x, y)
    Flux.update!(opt, ps, gs)
  end
end

function validation(model, data)
  correct_all = 0
  all = 0
  testmode!(model)
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

  trainloader = DataLoader(traindataset, 32, ntasks=8)
  valloader   = DataLoader(valdataset,   32, ntasks=8, shuffle = false)

  optimizer = Momentum(0.01, 0.99)

  sc = OneCycleLR(0.001, 0.1, 0.0001, 0.80, 0.99, 1000, 0.05)

  function softmax_cross_entropy(x, y)
    trainmode!(model)
    ỹ = model(x)
    crossentropy(ỹ, y)
  end

  for epoch in 1:1000
    optimizer.eta, optimizer.rho = sc(epoch)
    @printf "\nEpoch %2d/1000 @LR=%f\n" epoch optimizer.eta
    mytrain!(softmax_cross_entropy, params(model), trainloader, optimizer)
    validation(model, valloader)
    cpumodel = cpu(model)
    @save "model.bson" cpumodel
  end
end

main()