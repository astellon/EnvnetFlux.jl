using Flux, WAV

function nonzero(xs)
  s = findfirst(x->x!=0, xs)
  e = findlast(x->x!=0, xs)
  @view xs[s:e]
end

function pad(xs, L = 33325)
  z = zeros(eltype(xs), L*2 + length(xs))
  z[L+1:L+length(xs)] = xs
  z
end

function loadwav(path, mono = true)
  xs, fs = wavread(path)
  @views ifelse(mono, sum(xs, dims = 2)[:, 1], xs)
end

struct ESC50Dataset
  sounds
  labels
end

function ESC50Dataset(path::String, folds::Vector{Int}, mono::Bool = true)
  files = joinpath.(path, readdir(path))
  folds = string.(folds)  # for comparison

  allfolds = [ match(r"(\d)-\d+-[A-Z]-\d+.wav", basename(f))[1] for f in files ]
  indices  = findall(x->in(x, folds), allfolds)

  sounds = [ Float32.(pad(nonzero(loadwav(f, mono)))) for f in files[indices] ]
  labels = [ match(r"\d-\d+-[A-Z]-(\d+).wav", basename(f))[1] for f in files[indices] ]

  labels = [ Flux.onehot(parse(Int, l), 0:49) for l in labels]

  ESC50Dataset(sounds, labels)
end

Base.IndexStyle(::ESC50Dataset) = IndexLinear()

Base.size(esc::ESC50Dataset) = size(esc.labels)

Base.length(esc::ESC50Dataset) = length(esc.labels)

function Base.getindex(esc::ESC50Dataset, index::Int)
  s = esc.sounds[index]
  l = esc.labels[index]

  start = rand(1:(length(s) - 66650))

  s = s[start:(start+66650-1)]

  reshape(s, (1, length(s), 1)), l
end

function downloadesc()
  dist = joinpath(@__DIR__, "..", "data", "audio")

  # already installed
  ispath(dist) && return dist

  # install
  link = "https://github.com/karoldvl/ESC-50/archive/master.zip"
  temp = mktempdir()

  run(`curl -L $link -o $(joinpath(temp, "esc50.zip"))`)
  run(`unzip -q $(joinpath(temp, "esc50.zip")) -d $temp`)

  mkpath(joinpath(@__DIR__, "..", "data"))
  mv(joinpath(temp, "ESC-50-master", "audio"), joinpath(@__DIR__, "..", "data", "audio"))
  rm(temp, recursive=true)

  return dist
end
