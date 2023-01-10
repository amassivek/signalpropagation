def identity(input, func):
    h0, t0, context = input

    h1 = func(h0)
    t1 = t0

    return (h1, t1, context)

def fixed(input, func):
    h0, t0, context = input

    h0 = h0.detach()
    h1 = func(h0)

    if t0 is not None:
        t0 = t0.detach()
        t1 = func(t0)

        t1 = t1.detach()
    else:
        t1 = t0

    h1 = h1.detach()

    return (h1, t1, context)

def forward(input, func):
    h0, t0, context = input

    h1 = func(h0)

    if t0 is not None:
        t1 = func(t0)

    else:
        t1 = t0

    return (h1, t1, context)
