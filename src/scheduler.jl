function cosineannealing(t, tmax, η₀, η₁)
  η₀ + (η₁ - η₀) * (1 + cos(t/tmax*π))/2
end

struct OneCycleLR
  init_lr
  max_lr
  final_lr
  momentum_min
  momentum_max
  nepochs
  warmup
end

function (scheduler::OneCycleLR)(epoch)
  1 <= epoch <= scheduler.nepochs || error("given epoch is out of range")

  if epoch <= scheduler.nepochs * scheduler.warmup
    lr = cosineannealing(
        epoch, scheduler.nepochs * scheduler.warmup, scheduler.max_lr, scheduler.init_lr
      )
    momentum = cosineannealing(
      epoch, scheduler.nepochs * scheduler.warmup, scheduler.momentum_min, scheduler.momentum_max
    )
  else
    lr = cosineannealing(
        epoch - scheduler.nepochs * scheduler.warmup,
        scheduler.nepochs - scheduler.nepochs * scheduler.warmup,
        scheduler.final_lr, scheduler.max_lr
      )
    momentum = cosineannealing(
      epoch - scheduler.nepochs * scheduler.warmup,
      scheduler.nepochs - scheduler.nepochs * scheduler.warmup,
      scheduler.momentum_max, scheduler.momentum_min
    )
  end

  return lr, momentum
end
