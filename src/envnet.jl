using Flux
using Flux: @functor

function ConvBNReLu(size, ch; stride=(1, 1), pad = (0, 0), bias=false)
  return Chain(
    Conv(size, ch, stride = stride, pad = pad),
    BatchNorm(ch[2], relu)
  )
end

struct Envnet
  layers
end

function Envnet(nclass)
  return Chain(
    # 1xTx1xN
    ConvBNReLu((1, 8),  1=>40),
    ConvBNReLu((1, 8), 40=>40),
    MaxPool((1, 160)),
    # 1xTxCxN ==> CxTx1xN
    x->permutedims(x, (3, 2, 1, 4)),
    ConvBNReLu((8, 13), 1=>50),
    MaxPool((3, 3)),
    ConvBNReLu((1, 5), 50=>50),
    MaxPool((1, 3)),
    # flatten
    x->reshape(x, (50*11*14, :)),
    Dense(50*11*14, 4096, relu),
    Dense(4096, 4096, relu),
    Dense(4096, nclass),
  )
end

function Envnetv2(nclass)
  return Chain(
    # 1xTx1xN
    ConvBNReLu((1, 64),  1=>32, stride = (1, 2)),
    ConvBNReLu((1, 16), 32=>64, stride = (1, 2)),
    MaxPool((1, 64)),
    # 1xTxCxN ==> CxTx1xN
    x->permutedims(x, (3, 2, 1, 4)),
    ConvBNReLu((8, 8),  1=>32),
    ConvBNReLu((8, 8), 32=>32),
    MaxPool((5, 3)),
    ConvBNReLu((1, 4), 32=>64),
    ConvBNReLu((1, 4), 64=>64),
    MaxPool((1, 2)),
    ConvBNReLu((1, 2),  64=>128),
    ConvBNReLu((1, 2), 128=>128),
    MaxPool((1, 2)),
    ConvBNReLu((1, 2), 128=>256),
    ConvBNReLu((1, 2), 256=>256),
    MaxPool((1, 2)),
    # flatten
    x->reshape(x, (256*10*8, :)),
    Dense(256*10*8, 4096, relu),
    Dense(4096, 4096, relu),
    Dense(4096, nclass),
  )
end

(e::Envnet)(x) = e.layers(x)

@functor Envnet

function testv1()
  envnet = Envnet(50)
  x = rand(24014, 16)
  y = envnet(reshape(x, (1, size(x, 1), 1, size(x, 2))))
  size(y) == (50, 16)
end

function testv2()
  envnet = Envnetv2(50)
  x = rand(66650, 16)
  y = envnet(reshape(x, (1, size(x, 1), 1, size(x, 2))))
  size(y) == (50, 16)
end