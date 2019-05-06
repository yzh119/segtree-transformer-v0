def unpack_params(params):
    embed_params, other_params, wd_params = [], [], []
    for k, v in params:
        if 'embed' in k:
            embed_params.append(v)
        elif 'norm' in k or 'bias' in k:
            other_params.append(v)
        else: # applies weight decay
            wd_params.append(v)

    return embed_params, other_params, wd_params