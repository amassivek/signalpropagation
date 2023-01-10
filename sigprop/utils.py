def shape_numel(shape):
    numel = 1
    for d in shape:
        numel *= d
    return numel
