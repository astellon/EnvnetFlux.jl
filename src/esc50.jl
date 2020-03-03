using Flux, WAV

function loadwav(path, mono = true)
  xs, fs = wavread(path)
  ifelse(mono, sum(xs, dims = 2), xs)
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

  sounds = [ loadwav(f, mono) for f in files[indices] ]
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